//! TF_GLM53_MULTI: the GPU's time between a prompt being computed (prefill) and the live conversations' decode, while
//! both want it. The decode share (percent) is the operator's: TF_GLM53_MULTI_DECODE_SHARE at start, POST /v1/schedule
//! while the engine runs. 0: the prompts first (no round inside a prefill, the queue admitted back to back); 100: the
//! decode first (a prompt waits while a conversation decodes); between: the times measured on rank 0 kept at that
//! share. With one kind of work alone, it takes the whole GPU.
//! TF_GLM53_MULTI: prompt の計算 (prefill) と使用中の会話の decode が両方 GPU を求めている間の、GPU の時間の分け方。
//! decode の取り分 (%) は運用者が決める: 起動時は TF_GLM53_MULTI_DECODE_SHARE、動作中は POST /v1/schedule。0 は prompt を
//! 先に (prefill の中にラウンドを挟まず、待ち行列を続けて受け入れる)。100 は decode を先に (会話が decode している間、
//! prompt は待つ)。その間は、rank 0 で測った時間をその割合に保つ。片方の仕事しかないときは、その仕事が GPU 全部を使う。

const std = @import("std");

/// The counted times and the decisions from them: rank 0's scheduler thread alone (the share comes from the HTTP
/// threads as a number and is passed in). Times in nanoseconds of the awake clock.
/// 数えた時間とそれによる判断: rank 0 のスケジューラのスレッドだけが使う (取り分は HTTP のスレッドから数として来て、
/// 引数で渡す)。時間は awake の時計のナノ秒。
pub const Pace = struct {
    /// Prefill and decode time while both had work, the share they were counted for (another starts them again).
    /// 両方に仕事があった間の prefill と decode の時間と、それを数えた取り分 (別の取り分になったら数え直す)。
    prefill: i96 = 0,
    decode: i96 = 0,
    share: u8 = 0,
    /// The prefill segment running since (null: none counted), the layers it may run (0: to the chunk's end, so its
    /// time is no layer's measure), and one prompt layer's time (0: not measured yet).
    /// 実行中の prefill の区間の始まり (null は数えていない)、その区間が進められる層数 (0 はチャンクの終わりまで。その
    /// 時間は 1 層の目安にしない)、prompt の 1 層の時間 (0 は未計測)。
    seg: ?i96 = null,
    seg_layers: u8 = 0,
    layer_ns: i96 = 0,
    /// The chunk's rows the layer's time was measured for (a chunk of other rows measures it again).
    /// 1 層の時間を測ったチャンクの行数 (行数の違うチャンクでは測り直す)。
    rows: u32 = 0,
    /// For the HTTP threads: the decode's part of the time while both last had work (percent, 255: never yet), and
    /// whether both have work now.
    /// HTTP のスレッド向け: 最後に両方に仕事があった間の decode の割合 (%、255 はまだない) と、今両方に仕事があるか。
    shown: std.atomic.Value(u8) = .init(255),
    both: std.atomic.Value(bool) = .init(false),

    /// The counts are halved past this, so the share holds over the last seconds, not since the start.
    /// これを超えたら数を半分にする: 取り分は起動からではなく直近の数秒で保つ。
    pub const window: i96 = 10 * std.time.ns_per_s;
    /// Share 0 beside live conversations: a pause every this many layers with no round, so a change of the share
    /// reaches a running prefill within a few layers.
    /// 使用中の会話の横で取り分 0: この層数ごとにラウンドなしで止まる所を置き、取り分の変更を実行中の prefill へ数層で
    /// 届ける。
    pub const check_layers: u8 = 4;

    fn counted(pc: *Pace, share: u8) void {
        if (pc.share == share) return;
        pc.prefill = 0;
        pc.decode = 0;
        pc.share = share;
    }

    fn trim(pc: *Pace) void {
        while (pc.prefill + pc.decode > window) {
            pc.prefill = @divTrunc(pc.prefill, 2);
            pc.decode = @divTrunc(pc.decode, 2);
        }
        if (pc.measured()) |m| pc.shown.store(m, .monotonic);
        pc.both.store(true, .monotonic);
    }

    /// One kind of work alone: nothing is shared, the counts start again.
    /// 片方の仕事だけ: 分けるものがないので、数え直す。
    pub fn idle(pc: *Pace) void {
        pc.prefill = 0;
        pc.decode = 0;
        pc.seg = null;
        pc.both.store(false, .monotonic);
    }

    /// A round of the live conversations while a prompt waited for the GPU.
    /// prompt が GPU を待っている間の、使用中の会話のラウンド 1 回。
    pub fn addDecode(pc: *Pace, share: u8, ns: i96) void {
        pc.counted(share);
        pc.decode += @max(0, ns);
        pc.trim();
    }

    /// A prefill segment begins at `t` beside live conversations; `n`: the layers it may run before its pause (0: up to
    /// the chunk's end).
    /// 使用中の会話の横で prefill の区間が `t` に始まる。`n`: 止まる所までに進められる層数 (0 はチャンクの終わりまで)。
    pub fn begin(pc: *Pace, t: i96, n: u8) void {
        pc.seg = t;
        pc.seg_layers = n;
    }

    /// The segment ends at `t`: its time to the prefill and, when it ran a known count of layers, to one layer's time.
    /// 区間が `t` に終わる: その時間を prefill に足し、決まった層数を進めた区間なら 1 層の時間にも使う。
    pub fn end(pc: *Pace, share: u8, t: i96) void {
        const s = pc.seg orelse return;
        pc.seg = null;
        const ns = @max(0, t - s);
        pc.counted(share);
        pc.prefill += ns;
        if (pc.seg_layers > 0) {
            const one = @divTrunc(ns, pc.seg_layers);
            pc.layer_ns = if (pc.layer_ns == 0) one else @divTrunc(pc.layer_ns * 3 + one, 4);
        }
        pc.trim();
    }

    /// A chunk of `rows` rows begins: one layer's time is measured again when its rows differ.
    /// `rows` 行のチャンクが始まる: 行数が違えば 1 層の時間を測り直す。
    pub fn chunk(pc: *Pace, rows: u32) void {
        if (pc.rows == rows) return;
        pc.rows = rows;
        pc.layer_ns = 0;
    }

    /// The segment cut short at `t` (a chunk's or the prompt's end, before its layers ran out): its time counted, not
    /// as a layer's measure.
    /// 区間が `t` で途中で終わった (層を使い切る前のチャンクか prompt の終わり): 時間は数え、1 層の目安にはしない。
    pub fn cut(pc: *Pace, share: u8, t: i96) void {
        pc.seg_layers = 0;
        pc.end(share, t);
    }

    /// Whether the decode is behind its share (decode / (prefill + decode) < share / 100): its turn next. 100 always,
    /// 0 never.
    /// decode が取り分に届いていないか (decode / (prefill + decode) < share / 100): 次は decode の番。100 は常に、0 は決して。
    pub fn behind(pc: *const Pace, share: u8) bool {
        if (share >= 100) return true;
        if (share == 0) return false;
        const p, const d = pc.counts(share);
        return d * (100 - share) < p * share;
    }

    /// The layers a prefill runs before its next pause beside live conversations (`live` false: none, 0): enough to
    /// bring the decode down to its share again, at least 1.
    /// 使用中の会話の横で、prefill が次に止まるまでに進める層数 (`live` が false ならなし、0): decode を取り分まで下げる
    /// だけの層数。最低 1。
    pub fn layers(pc: *const Pace, share: u8, live: bool) u8 {
        if (!live) return 0;
        if (share == 0) return check_layers;
        if (share >= 100) return 1;
        if (pc.layer_ns <= 0) return 2;
        const p, const d = pc.counts(share);
        const more = @divTrunc(d * (100 - share), share) - p;
        if (more <= 0) return 1;
        return @intCast(std.math.clamp(@divTrunc(more + pc.layer_ns - 1, pc.layer_ns), 1, 255));
    }

    /// The decode's part of the counted time (percent), null when nothing is counted.
    /// 数えた時間のうち decode の割合 (%)。何も数えていなければ null。
    pub fn measured(pc: *const Pace) ?u8 {
        const all = pc.prefill + pc.decode;
        if (all <= 0) return null;
        const m: u8 = @intCast(@divTrunc(pc.decode * 100 + @divTrunc(all, 2), all));
        return m;
    }

    fn counts(pc: *const Pace, share: u8) struct { i96, i96 } {
        return if (pc.share == share) .{ pc.prefill, pc.decode } else .{ 0, 0 };
    }
};

const ms = std.time.ns_per_ms;

test "Pace: the ends" {
    var pc: Pace = .{};
    try std.testing.expect(!pc.behind(0));
    try std.testing.expect(pc.behind(100));
    try std.testing.expectEqual(@as(u8, 0), pc.layers(50, false));
    try std.testing.expectEqual(Pace.check_layers, pc.layers(0, true));
    try std.testing.expectEqual(@as(u8, 1), pc.layers(100, true));
}

test "Pace: a prefill beside rounds keeps the share" {
    // a layer of 190 ms, a round of 120 ms, share 50: the simulated time ends near half and half
    var pc: Pace = .{};
    var t: i96 = 0;
    var n = pc.layers(50, true);
    for (0..400) |_| {
        pc.begin(t, n);
        t += @as(i96, n) * 190 * ms;
        pc.end(50, t);
        while (pc.behind(50)) {
            pc.addDecode(50, 120 * ms);
            t += 120 * ms;
        }
        n = pc.layers(50, true);
    }
    const m = pc.measured().?;
    try std.testing.expect(m >= 45 and m <= 55);
}

test "Pace: share 20 and 80" {
    for ([_]u8{ 20, 80 }) |share| {
        var pc: Pace = .{};
        var t: i96 = 0;
        var n = pc.layers(share, true);
        for (0..400) |_| {
            pc.begin(t, n);
            t += @as(i96, n) * 60 * ms;
            pc.end(share, t);
            while (pc.behind(share)) {
                pc.addDecode(share, 300 * ms);
                t += 300 * ms;
            }
            n = pc.layers(share, true);
        }
        const m = pc.measured().?;
        try std.testing.expect(m + 6 >= share and m <= share + 6);
    }
}

test "Pace: a new share starts the counts again, idle forgets" {
    var pc: Pace = .{};
    pc.begin(0, 1);
    pc.end(30, 200 * ms);
    pc.addDecode(30, 100 * ms);
    try std.testing.expectEqual(@as(u8, 33), pc.measured().?);
    // another share: what was counted for 30 does not count for it (the prompt goes on first)
    try std.testing.expect(!pc.behind(70));
    pc.addDecode(70, 50 * ms);
    try std.testing.expectEqual(@as(u8, 100), pc.measured().?);
    pc.idle();
    try std.testing.expect(pc.measured() == null);
    try std.testing.expect(pc.seg == null);
    // what the HTTP threads see: the last share measured, both no longer busy
    try std.testing.expectEqual(@as(u8, 100), pc.shown.load(.monotonic));
    try std.testing.expect(!pc.both.load(.monotonic));
}

test "Pace: the window halves the counts" {
    var pc: Pace = .{};
    pc.addDecode(40, 8 * std.time.ns_per_s);
    pc.begin(0, 0);
    pc.end(40, 8 * std.time.ns_per_s);
    try std.testing.expect(pc.prefill + pc.decode <= Pace.window);
    try std.testing.expectEqual(@as(u8, 50), pc.measured().?);
    // a segment to the chunk's end does not measure a layer, nor one cut short
    try std.testing.expectEqual(@as(i96, 0), pc.layer_ns);
    pc.begin(0, 4);
    pc.cut(40, 100 * ms);
    try std.testing.expectEqual(@as(i96, 0), pc.layer_ns);
    pc.begin(0, 4);
    pc.end(40, 100 * ms);
    try std.testing.expectEqual(@as(i96, 25 * ms), pc.layer_ns);
    // a chunk of other rows measures it again, one of the same rows keeps it
    pc.chunk(16384);
    try std.testing.expectEqual(@as(i96, 0), pc.layer_ns);
    pc.begin(0, 2);
    pc.end(40, 100 * ms);
    pc.chunk(16384);
    try std.testing.expectEqual(@as(i96, 50 * ms), pc.layer_ns);
}
