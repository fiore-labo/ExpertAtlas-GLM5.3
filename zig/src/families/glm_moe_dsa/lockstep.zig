//! How the ranks agree on a request: rank 0's header, then its prompt ids, each shared by int32 all-gathers, length
//! first. At start-up the ranks also check that they loaded the same things.
//! ランクどうしが要求をそろえる手順: rank 0 のヘッダ、続いて prompt の id を、それぞれ int32 の all-gather で配る
//! (長さが先)。起動時には、全ランクが同じものを読み込んだかも確かめる。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const slots = @import("slots_serve.zig");
const pool_serve = @import("pool_serve.zig");
const kvdigest = @import("kvdigest.zig");
const poison = @import("poison.zig");
const pool_plan = @import("pool/plan.zig");

/// A request's settings as 17 non-negative words: max_tokens, stop_eos, k | mode << 8, seed (3), temperature (3),
/// top_k, top_p (3), min_p (3), r0 (the slot cache's resume point). max_tokens < 0 tells the followers to stop.
/// 要求の設定を負でない 17 語で表す: max_tokens、stop_eos、k | mode << 8、seed (3)、temperature (3)、top_k、
/// top_p (3)、min_p (3)、r0 (スロットキャッシュの再開位置)。max_tokens < 0 はフォロワーへの停止指示。
pub const Header = struct {
    max_tokens: i32,
    stop_eos: bool = true,
    k: u8 = 0,
    mode: u8 = 0,
    seed: u64 = 0,
    temperature: f64 = 0,
    top_k: i32 = 0,
    top_p: f64 = 1,
    min_p: f64 = 0,
    r0: i32 = 0,
    /// The slot cache's 21 words, present when the engine keeps slots: resume flag, metadata digest (8), plan digest
    /// (8), drafter depth, confidence (3).
    /// スロットキャッシュの 21 語。エンジンがスロットを持つときだけある: resume フラグ、メタデータのダイジェスト (8)、
    /// 計画のダイジェスト (8)、drafter の深さ、信頼度 (3)。
    extra: ?[21]i32 = null,

    pub const words = 17;

    /// The 17 words, then the slot cache's 21 when present.
    /// 17 語。スロットキャッシュの 21 語があれば続ける。
    pub fn encodeAll(h: Header, out: *[38]i32) []const i32 {
        out[0..17].* = h.encode();
        if (h.extra) |x| {
            out[17..38].* = x;
            return out[0..38];
        }
        return out[0..17];
    }

    pub fn encode(h: Header) [words]i32 {
        var w: [words]i32 = undefined;
        w[0] = h.max_tokens;
        w[1] = @intFromBool(h.stop_eos);
        w[2] = @as(i32, h.k) | (@as(i32, h.mode) << 8);
        w[3] = @intCast(h.seed & 0x7FFFFFFF);
        w[4] = @intCast((h.seed >> 31) & 0x7FFFFFFF);
        w[5] = @intCast(h.seed >> 62);
        f64Words(h.temperature, w[6..9]);
        w[9] = h.top_k;
        f64Words(h.top_p, w[10..13]);
        f64Words(h.min_p, w[13..16]);
        w[16] = h.r0;
        return w;
    }

    pub fn decode(w: []const i32) !Header {
        if (w.len < words) return error.ShortHeader;
        return .{
            .max_tokens = w[0],
            .stop_eos = w[1] != 0,
            .k = @intCast(w[2] & 0xFF),
            .mode = @intCast((w[2] >> 8) & 0xFF),
            .seed = @as(u64, @intCast(w[3])) | (@as(u64, @intCast(w[4])) << 31) | (@as(u64, @intCast(w[5])) << 62),
            .temperature = wordsF64(w[6..9]),
            .top_k = w[9],
            .top_p = wordsF64(w[10..13]),
            .min_p = wordsF64(w[13..16]),
            .r0 = w[16],
            .extra = if (w.len >= 38) w[17..38].* else null,
        };
    }
};

/// An f64's bits as three non-negative words: low 31, next 31, top 2.
/// f64 のビットを負でない 3 語に分ける: 下位 31、次の 31、上位 2 ビット。
pub fn f64Words(v: f64, out: []i32) void {
    const b: u64 = @bitCast(v);
    out[0] = @intCast(b & 0x7FFFFFFF);
    out[1] = @intCast((b >> 31) & 0x7FFFFFFF);
    out[2] = @intCast(b >> 62);
}

pub fn wordsF64(w: []const i32) f64 {
    const b = @as(u64, @intCast(w[0])) | (@as(u64, @intCast(w[1])) << 31) | (@as(u64, @intCast(w[2])) << 62);
    return @bitCast(b);
}

/// The all-gathers' device buffers (grown as needed) on the engine's stream.
/// all-gather のデバイスバッファ (必要に応じて拡張)。エンジンの stream で動く。
pub const Channel = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    group: *const cuda.tp.Group,
    stream: cuda.Stream,
    send: ?cuda.DeviceBuffer = null,
    recv: ?cuda.DeviceBuffer = null,
    cap: usize = 0, // int32 words per rank / ランクごとの int32 の語数

    pub fn init(gpa: std.mem.Allocator, e: *const engine.Engine) Channel {
        return .{ .gpa = gpa, .d = e.d, .group = e.group, .stream = e.stream };
    }

    fn reserve(ch: *Channel, n: usize) !void {
        if (n <= ch.cap) return;
        const want = @max(n, 1024);
        // the new pair, then the old one freed (each all-gather ends with a synchronize): a prompt longer than any
        // before grows the pair, and each conversation turn can be one
        // 新しい組を確保してから古い組を解放する (all-gather はどれも synchronize で終わる): それまでより長い prompt が
        // 来るたびに組は大きくなり、会話の 1 ターンごとにそうなりうる
        var send = try cuda.DeviceBuffer.alloc(ch.d, want * 4);
        errdefer send.free();
        const recv = try cuda.DeviceBuffer.alloc(ch.d, want * 4 * ch.group.world);
        ch.deinit();
        ch.send = send;
        ch.recv = recv;
        ch.cap = want;
    }

    /// Frees the device buffers; the channel grows them again when used.
    /// デバイスバッファを解放する。使えばチャネルはまた確保する。
    pub fn deinit(ch: *Channel) void {
        if (ch.send) |*b| b.free();
        if (ch.recv) |*b| b.free();
        ch.send = null;
        ch.recv = null;
        ch.cap = 0;
    }

    /// Every rank's `mine` (n words each) gathered in rank order into `out` (n * world words).
    /// 各ランクの `mine` (各 n 語) をランク順に `out` (n * world 語) へ集める。
    pub fn allGather(ch: *Channel, mine: []const i32, out: []i32) !void {
        const n = mine.len;
        if (out.len != n * ch.group.world) return error.BadLength;
        try ch.reserve(n);
        try ch.send.?.uploadAsync(0, std.mem.sliceAsBytes(mine), ch.stream.handle);
        const lib = ch.group.lib;
        try lib.check(lib.api.ncclAllGather(ch.send.?.ptr, ch.recv.?.ptr, n, .i32, ch.group.comm, ch.stream.handle), "ncclAllGather");
        try ch.recv.?.downloadAsync(0, std.mem.sliceAsBytes(out), ch.stream.handle);
        try ch.stream.synchronize();
    }

    /// Rank 0's `values` to every rank (others pass null), length first; the caller owns the result.
    /// rank 0 の `values` を全ランクへ配る (他のランクは null を渡す)。長さが先。結果は呼び出し側が持つ。
    pub fn share(ch: *Channel, values: ?[]const i32) ![]i32 {
        const W = ch.group.world;
        const lens = try ch.gpa.alloc(i32, W);
        defer ch.gpa.free(lens);
        const mine_len: i32 = if (values) |v| @intCast(v.len) else 0;
        try ch.allGather(&.{mine_len}, lens);
        const n: usize = @intCast(lens[0]);
        const out = try ch.gpa.alloc(i32, n);
        errdefer ch.gpa.free(out);
        if (n == 0) return out;
        const mine = try ch.gpa.alloc(i32, n);
        defer ch.gpa.free(mine);
        if (values) |v| @memcpy(mine, v) else @memset(mine, 0);
        const all = try ch.gpa.alloc(i32, n * W);
        defer ch.gpa.free(all);
        try ch.allGather(mine, all);
        @memcpy(out, all[0..n]);
        return out;
    }

    /// Whether every rank's flag is set.
    /// 全ランクのフラグが立っているか。
    pub fn allOk(ch: *Channel, ok: bool) !bool {
        var all: [64]i32 = undefined;
        try ch.allGather(&.{@intFromBool(ok)}, all[0..ch.group.world]);
        for (all[0..ch.group.world]) |v| if (v == 0) return false;
        return true;
    }
};

/// What a rank loaded (sha256 of config.json, of the weights index and of the build id), world and rank; the ranks
/// start only when the digests agree everywhere and the ranks are 0 .. world - 1 in order.
/// ランクが読み込んだもの (config.json・重みの index・ビルド ID の sha256) と world・rank。ダイジェストが全ランクで
/// 一致し、ランクが 0 .. world - 1 の順に並ぶときだけ起動する。
pub const Fingerprint = struct {
    words: [19]i32,

    pub fn of(io: std.Io, gpa: std.mem.Allocator, ckpt: []const u8, build_id: []const u8, world: u32, rank: u32) !Fingerprint {
        var f: Fingerprint = .{ .words = undefined };
        try fileDigest(io, gpa, ckpt, "config.json", f.words[0..4]);
        try fileDigest(io, gpa, ckpt, "model.safetensors.index.json", f.words[4..8]);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(build_id, &d, .{});
        digestWords(d, f.words[8..17]);
        f.words[17] = @intCast(world);
        f.words[18] = @intCast(rank);
        return f;
    }

    /// Every rank calls it; error.Mismatch on all of them when any differs.
    /// 全ランクが呼ぶ。どれかが違えば全ランクで error.Mismatch になる。
    pub fn agree(f: Fingerprint, ch: *Channel) !void {
        const W = ch.group.world;
        const all = try ch.gpa.alloc(i32, 19 * W);
        defer ch.gpa.free(all);
        try ch.allGather(&f.words, all);
        for (0..W) |r| {
            const row = all[r * 19 ..][0..19];
            if (!std.mem.eql(i32, row[0..18], all[0..18]) or row[18] != @as(i32, @intCast(r))) return error.Mismatch;
        }
    }
};

fn digestWords(d: [32]u8, out: []i32) void {
    for (out, 0..) |*w, i| w.* = @bitCast(std.mem.readInt(u32, d[(i * 4) % 32 ..][0..4], .little) & 0x7FFFFFFF);
}

fn fileDigest(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, name: []const u8, out: []i32) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    var d: [32]u8 = @splat(0);
    if (std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30))) |text| {
        defer gpa.free(text);
        std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    } else |err| if (err != error.FileNotFound) return err;
    digestWords(d, out);
}

/// The header mode that asks for DFlash2 drafts; 0 decodes serially.
/// DFlash2 の draft を求めるヘッダの mode。0 は逐次 decode。
pub const mode_dflash: u8 = 9;
/// The header mode that asks for drafts from the checkpoint's MTP layer (TF_GLM53_DRAFTER=mtp).
/// チェックポイントの MTP 層の draft を求めるヘッダの mode (TF_GLM53_DRAFTER=mtp)。
pub const mode_mtp: u8 = 4;
/// DFlash2's draft tokens a round, the same on every rank.
/// DFlash2 が 1 ラウンドに出す draft トークン数。全ランクで同じ。
pub const dflash_depth: usize = 7;

/// One request on this rank: the slot cache's agreement first when there is one, then the decode the header asks for.
/// このランクでの 1 要求: スロットキャッシュがあればその合意を先に行い、次にヘッダが求める decode を行う。
fn run(ch: *Channel, e: *engine.Engine, h: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer, plan: ?*slots.Plan, pplan: ?*pool_plan.Plan) !engine.Stats {
    if (e.pool != null) return pool_serve.serve(ch, e, h, prompt, sink, prop, pplan, runPlain);
    // TF_GLM53_POOL_POISON (tests): prefix off poisons every row before every request (poison.zig)
    // TF_GLM53_POOL_POISON (試験用): prefix off は要求ごとに全行を毒で埋める (poison.zig)
    if (e.slots == null) if (poison.byte()) |v| try poison.all(e, v);
    const st = if (e.slots != null) try slots.serve(ch, e, h, prompt, sink, prop, plan, runPlain) else try runPlain(e, h, prompt, sink, prop);
    if (kvdigest.on()) {
        // the rows at [0, V) contiguous, or through TF_GLM53_POOL_TEST's fixed table
        // 行は [0, V) に連続、または TF_GLM53_POOL_TEST の決まった表を通す
        const V = kvdigest.validOf(e, prompt.len, st);
        try kvdigest.report(e, V, @min(V, prompt.len), if (e.mtp) |*m| m.track.kvEnd(V) else V, try kvdigest.testTable(e));
    }
    return st;
}

fn runPlain(e: *engine.Engine, h: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) !engine.Stats {
    // temperature > 0 samples by keyed draws (the same tokens on every rank, drafted or not); else greedy
    // temperature > 0 ならキー付きの抽選でサンプルする (draft の有無によらず全ランクで同じトークン)。他は greedy
    e.samp = if (h.temperature > 0) .{ .seed = h.seed, .temperature = h.temperature, .top_k = @intCast(@max(0, h.top_k)), .top_p = h.top_p, .min_p = h.min_p } else null;
    defer e.samp = null;
    if (h.mode == mode_dflash) {
        const p = prop orelse e.drafterProposer() orelse return error.NoDrafter;
        return e.generateDrafted(prompt, @intCast(h.max_tokens), h.stop_eos, sink, p, dflash_depth);
    }
    if (h.mode == mode_mtp) {
        const p = prop orelse e.mtpProposer() orelse return error.NoMtp;
        return e.generateDrafted(prompt, @intCast(h.max_tokens), h.stop_eos, sink, p, e.mtp.?.k);
    }
    if (h.mode != 0) return error.UnportedMode;
    return e.generate(prompt, @intCast(h.max_tokens), h.stop_eos, sink);
}

/// Rank 0: shares one request (header, then prompt) with every rank, then runs its own part.
/// rank 0: 1 つの要求 (ヘッダ、続いて prompt) を全ランクと共有し、自分の分を実行する。
pub fn lead(ch: *Channel, e: *engine.Engine, h0: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) !engine.Stats {
    var h = h0;
    // the slot cache's plan comes before anything is shared, so a failure here touches no rank
    // スロットキャッシュの計画は共有の前に立てるので、ここで失敗してもどのランクにも影響しない
    var plan: ?slots.Plan = null;
    defer if (plan) |*p| p.deinit();
    var pplan: ?pool_plan.Plan = null;
    defer if (pplan) |*p| p.deinit(e.gpa);
    if (e.pool != null) pplan = pool_serve.lead(e, &h, prompt) catch |err| blk: {
        // go on cold: words that match no rank's metadata
        // cold で続行: どのランクのメタデータとも一致しない語を送る
        std.debug.print("[tensorfold] pool: rank 0 could not plan ({s}) - every rank goes cold\n", .{@errorName(err)});
        var x: [21]i32 = @splat(0);
        x[0] = 1;
        h.extra = x;
        h.r0 = 0;
        break :blk null;
    };
    if (e.slots != null) plan = slots.lead(e, &h, prompt) catch |err| blk: {
        // go on cold: slot words that match no rank's metadata
        // cold で続行: どのランクのメタデータとも一致しないスロットの語を送る
        std.debug.print("[tensorfold] slots: rank 0 could not plan ({s}) - every rank goes cold\n", .{@errorName(err)});
        var x: [21]i32 = @splat(0);
        x[0] = 1;
        h.extra = x;
        h.r0 = 0;
        break :blk null;
    };
    // refuse prompt chunk shapes not run at start-up, before the followers see the request
    // 起動時に試していない prompt チャンクの形は、フォロワーが要求を見る前に断る
    try e.admit(prompt.len, @intCast(@max(0, h.r0)));
    var buf: [38]i32 = undefined;
    ch.gpa.free(try ch.share(h.encodeAll(&buf)));
    const ids = try ch.gpa.alloc(i32, prompt.len);
    defer ch.gpa.free(ids);
    for (ids, prompt) |*w, t| w.* = @intCast(t);
    ch.gpa.free(try ch.share(ids));
    return run(ch, e, h, prompt, sink, prop, if (plan) |*p| p else null, if (pplan) |*p| p else null);
}

/// Rank 0: tells the followers to return from `follow`.
/// rank 0: フォロワーに `follow` から戻るよう伝える。
pub fn shutdown(ch: *Channel) !void {
    const hw = (Header{ .max_tokens = -1 }).encode();
    ch.gpa.free(try ch.share(&hw));
}

/// What a follower received, for tests that check every rank got the same request.
/// フォロワーが受け取ったもの。全ランクが同じ要求を受けたかを確かめるテスト用。
pub const Received = struct {
    header: Header,
    prompt_sha256: [32]u8,
    stats: engine.Stats,
};

/// Ranks above 0: run each request as rank 0 does, without output, until rank 0 says stop. `seen` (if any) gets each
/// request's header and the digest of the prompt received.
/// rank 1 以上: rank 0 が停止を告げるまで、各要求を rank 0 と同じく出力なしで実行する。`seen` があれば、各要求の
/// ヘッダと受け取った prompt のダイジェストを渡す。
pub fn follow(ch: *Channel, e: *engine.Engine, seen: ?*std.ArrayList(Received), prop: ?engine.Proposer) !void {
    while (true) {
        const hw = try ch.share(null);
        defer ch.gpa.free(hw);
        const h = try Header.decode(hw);
        if (h.max_tokens < 0) return;
        const words = try ch.share(null);
        defer ch.gpa.free(words);
        const prompt = try ch.gpa.alloc(i64, words.len);
        defer ch.gpa.free(prompt);
        for (prompt, words) |*t, w| t.* = w;
        const st = try run(ch, e, h, prompt, null, prop, null, null);
        if (seen) |list| {
            var d: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(prompt), &d, .{});
            try list.append(ch.gpa, .{ .header = h, .prompt_sha256 = d, .stats = st });
        }
    }
}

test "the header round-trips through its words" {
    const h: Header = .{ .max_tokens = 300, .stop_eos = false, .k = 2, .mode = 9, .seed = 0x7FFF_FFFF_1234_5678, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0.05, .r0 = 17 };
    const w = h.encode();
    for (w) |v| try std.testing.expect(v >= 0);
    const back = try Header.decode(&w);
    try std.testing.expectEqualDeep(h, back);
    // 1.0's bits split low 31 / next 31 / top 2
    // 1.0 のビットを下位 31・次の 31・上位 2 に分ける
    var one: [3]i32 = undefined;
    f64Words(1.0, &one);
    const b: u64 = @bitCast(@as(f64, 1.0));
    try std.testing.expectEqual(@as(i32, @intCast(b & 0x7FFFFFFF)), one[0]);
    try std.testing.expectEqual(@as(i32, @intCast((b >> 31) & 0x7FFFFFFF)), one[1]);
    try std.testing.expectEqual(@as(i32, @intCast(b >> 62)), one[2]);
}
