// Fiore-lab: the live expert usage for the control panel (TF_GLM53_EXPERT_LIVE), not in TensorFold 1.0.2.
//! The routed experts the target picked for each decoded input row, kept on rank 0: running totals by layer and
//! expert, and a ring of the most recent rows. The engine thread writes after a decode round's sync (the rows the
//! round kept: a drafter window's accepted rows, never padding); HTTP threads read under the mutex and format the
//! JSON outside it. Nothing here touches the GPU. The prompt's picks come in by prompt chunk (recordPrompt: counts
//! the engine's pick_hist kernel made on the device, copied out once the chunk ended): running totals, and a ring of
//! the recent chunks' counts.
//! 制御パネル用の expert の使用状況のライブ (TF_GLM53_EXPERT_LIVE)。decode した入力行ごとに target が選んだ routed
//! expert を rank 0 に置く: 層と expert ごとの累計と、直近の行のリング。エンジンのスレッドが decode ラウンドの同期の後
//! に書き (ラウンドが残した行: drafter のウィンドウなら受理した行。詰め物の行は書かない)、HTTP のスレッドは mutex の
//! 下で写して、JSON は mutex の外で作る。GPU には触れない。prompt の選択は prompt のチャンクごとに入る (recordPrompt:
//! エンジンの pick_hist カーネルがデバイスで数え、チャンクが終わってから写した数): 累計と、直近のチャンクの数のリング。
const std = @import("std");
const forward = @import("forward.zig");

/// Rows the ring keeps (about 600 bytes each with 75 MoE layers).
/// リングが持つ行の数 (MoE 層 75 で 1 行約 600 バイト)。
pub const ring_rows = 4096;

/// Prompt chunks the chunk ring keeps (a chunk's counts: layers x experts u32, 76.8 KB with 75 MoE layers).
/// チャンクのリングが持つ prompt チャンクの数 (1 チャンクの数は 層 x expert の u32。MoE 層 75 で 76.8 KB)。
pub const chunk_ring = 64;

pub const Live = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    /// The first MoE layer's index, the MoE layers, experts a layer.
    /// 最初の MoE 層の index、MoE 層の数、層あたりの expert 数。
    first: u32,
    layers: u32,
    experts: u32,
    /// Each expert's bits as loaded (layer-major), 0 unknown.
    /// 読み込んだ各 expert の bit (層が外側)。0 は不明。
    bits: []u8,
    /// Decoded rows' picks by layer and expert since the start (or the last reset), and the rows.
    /// 開始 (または最後の reset) からの decode 行の選択を層と expert ごとに数えたもの、と行の数。
    totals: []u64,
    rows: u64 = 0,
    /// The ring: row i's picks at ids[i * layers * 8 ..], newest at (head - 1); seq counts every row ever written.
    /// リング: 行 i の選択は ids[i * layers * 8 ..]、最新は (head - 1)。seq は書いたすべての行を数える。
    ids: []u8,
    meta: []Row,
    head: usize = 0,
    held: usize = 0,
    seq: u64 = 0,
    /// The prompt rows' picks by layer and expert since the start (or the last reset), and the rows; the chunk ring:
    /// chunk i's counts at pcounts[i * layers * experts ..], newest at (phead - 1); pseq counts every chunk ever.
    /// 開始 (または最後の reset) からの prompt の行の選択を層と expert ごとに数えたもの、と行の数。チャンクのリング:
    /// チャンク i の数は pcounts[i * layers * experts ..]、最新は (phead - 1)。pseq は書いたすべてのチャンクを数える。
    ptotals: []u64,
    prows: u64 = 0,
    pcounts: []u32,
    pmeta: []Chunk,
    phead: usize = 0,
    pheld: usize = 0,
    pseq: u64 = 0,
    /// The clock's zero for the rows' times (ms), set at init and at reset.
    /// 行の時刻 (ms) の起点。init と reset で決める。
    t0: std.Io.Timestamp,

    /// A decoded row: its input token (`tok`, the token at `pos`; maxInt unknown) for its text.
    /// decode した行: その入力 token (`tok`、`pos` の token。maxInt は不明)。文字のため。
    pub const Row = struct { seq: u64, t_ms: u64, conv: u16, pos: u32, tok: u32 = std.math.maxInt(u32) };

    /// A token's raw bytes for the rows' text, from the server's tokenizer (engine_api.TokenBytes, the same shape):
    /// null for a special token.
    /// 行の文字のための token の生のバイト。サーバーのトークナイザーから (engine_api.TokenBytes と同じ形)。特殊 token は null。
    pub const TokenBytes = struct {
        ctx: *anyopaque,
        bytes: *const fn (ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8,
    };

    /// Rows before the first one asked for that recentJson also copies, so a row's text can take the bytes of a
    /// character begun in its conversation's previous rows (up to 3 rows each, multi_max conversations interleaved).
    /// recentJson が、求められた最初の行より前にも写す行の数。行の文字が、同じ会話の前の行で始まった文字のバイトを
    /// 受け取れるように (会話ごとに最大 3 行、multi_max 個の会話が交互に並ぶ)。
    const text_context = 32;
    /// A prompt chunk: its end (t_ms, the live clock), its compute time, the conversation, its first position and rows,
    /// and whether it was the prompt's last.
    /// prompt のチャンク: 終わった時刻 (t_ms、ライブの時計)、計算時間、会話、最初の位置と行数、prompt の最後か。
    pub const Chunk = struct { seq: u64, t_ms: u64, ms: f32, conv: u16, pos: u32, rows: u32, done: bool };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, first: u32, layers: u32, experts: u32) !*Live {
        const l = try gpa.create(Live);
        errdefer gpa.destroy(l);
        const n = @as(usize, layers) * experts;
        l.* = .{
            .gpa = gpa, .io = io, .first = first, .layers = layers, .experts = experts,
            .bits = try gpa.alloc(u8, n), .totals = try gpa.alloc(u64, n),
            .ids = try gpa.alloc(u8, ring_rows * @as(usize, layers) * 8), .meta = try gpa.alloc(Row, ring_rows),
            .ptotals = try gpa.alloc(u64, n), .pcounts = try gpa.alloc(u32, chunk_ring * n), .pmeta = try gpa.alloc(Chunk, chunk_ring),
            .t0 = std.Io.Clock.awake.now(io),
        };
        @memset(l.bits, 0);
        @memset(l.totals, 0);
        @memset(l.ptotals, 0);
        return l;
    }

    pub fn deinit(l: *Live) void {
        const gpa = l.gpa;
        gpa.free(l.bits);
        gpa.free(l.totals);
        gpa.free(l.ids);
        gpa.free(l.meta);
        gpa.free(l.ptotals);
        gpa.free(l.pcounts);
        gpa.free(l.pmeta);
        gpa.destroy(l);
    }

    /// Forget everything counted so far (after the start-up warm-up, before requests).
    /// ここまでの数をすべて捨てる (起動時の暖機の後、要求の前に)。
    pub fn reset(l: *Live) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        @memset(l.totals, 0);
        l.rows = 0;
        l.head = 0;
        l.held = 0;
        @memset(l.ptotals, 0);
        l.prows = 0;
        l.phead = 0;
        l.pheld = 0;
        l.t0 = std.Io.Clock.awake.now(l.io);
    }

    /// The experts' bits after a change (expert_swap); the counts stay.
    /// bit を変えた後の各 expert の bit (expert_swap)。数はそのまま。
    pub fn setBits(l: *Live, bits: []const u8) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        @memcpy(l.bits, bits);
    }

    pub fn nowMs(l: *const Live) u64 {
        const ms = l.t0.durationTo(std.Io.Clock.awake.now(l.io)).toMilliseconds();
        return if (ms < 0) 0 else @intCast(ms);
    }

    /// Rows `row0 .. row0 + n` of the log (the host view of the decode windows' pick log: layer i, row r at
    /// (i * forward.log_rows + r) * 8), the input rows at positions pos0 .. of conversation `conv`, their input tokens
    /// `toks[0 .. n]` (fewer: unknown).
    /// ログ (decode ウィンドウの選択ログのホスト側: 層 i、行 r は (i * forward.log_rows + r) * 8) の行
    /// row0 .. row0 + n を、会話 `conv` の位置 pos0 .. の入力行として書く。入力 token は `toks[0 .. n]` (足りなければ不明)。
    pub fn record(l: *Live, log: []const u8, row0: usize, n: usize, conv: u16, pos0: usize, toks: []const i64) void {
        if (n == 0) return;
        const t = l.nowMs();
        const span = @as(usize, l.layers) * 8;
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        for (0..n) |k| {
            const r = row0 + k;
            if (r >= forward.log_rows) break;
            const dst = l.ids[l.head * span ..][0..span];
            for (0..l.layers) |i| {
                const src = log[((l.first + i) * forward.log_rows + r) * 8 ..][0..8];
                @memcpy(dst[i * 8 ..][0..8], src);
                for (src) |e| {
                    if (e < l.experts) l.totals[i * l.experts + e] += 1;
                }
            }
            const tok: u32 = if (k < toks.len and toks[k] >= 0 and toks[k] < std.math.maxInt(u32)) @intCast(toks[k]) else std.math.maxInt(u32);
            l.meta[l.head] = .{ .seq = l.seq, .t_ms = t, .conv = conv, .pos = @intCast(@min(pos0 + k, std.math.maxInt(u32))), .tok = tok };
            l.seq += 1;
            l.rows += 1;
            l.head = (l.head + 1) % ring_rows;
            if (l.held < ring_rows) l.held += 1;
        }
    }

    /// A prompt chunk's counts: `counts` in the engine's layout (u32, layer index i's expert e at i * experts + e, the
    /// MoE layers from `first`), `rows` rows of conversation `conv` from position `pos`.
    /// prompt チャンクの数: `counts` はエンジンの並び (u32、層 index i の expert e は i * experts + e。MoE 層は
    /// `first` から)。会話 `conv` の位置 `pos` からの `rows` 行。
    pub fn recordPrompt(l: *Live, counts: []const u32, rows: u32, conv: u16, pos: u32, done: bool, ms: f32, t_ms: u64) void {
        const n = @as(usize, l.layers) * l.experts;
        const src = counts[@as(usize, l.first) * l.experts ..][0..n];
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        @memcpy(l.pcounts[l.phead * n ..][0..n], src);
        for (l.ptotals, src) |*t, c| t.* += c;
        l.pmeta[l.phead] = .{ .seq = l.pseq, .t_ms = t_ms, .ms = ms, .conv = conv, .pos = pos, .rows = rows, .done = done };
        l.pseq += 1;
        l.prows += rows;
        l.phead = (l.phead + 1) % chunk_ring;
        if (l.pheld < chunk_ring) l.pheld += 1;
    }

    fn b64(a: std.mem.Allocator, bytes: []const u8) ![]u8 {
        const out = try a.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
        _ = std.base64.standard.Encoder.encode(out, bytes);
        return out;
    }

    /// GET /v1/experts/usage[?window=SECONDS]: totals (u64 little-endian, layer-major, base64), the last `window`
    /// seconds' counts from the ring (u32 little-endian, base64) and the loaded bits (u8, base64).
    /// GET /v1/experts/usage[?window=SECONDS]: 累計 (u64 リトルエンディアン、層が外側、base64)、リングから直近
    /// `window` 秒の数 (u32 リトルエンディアン、base64)、読み込んだ bit (u8、base64)。
    pub fn usageJson(l: *Live, a: std.mem.Allocator, window_s: u32) ![]u8 {
        const n = @as(usize, l.layers) * l.experts;
        const totals = try a.alloc(u64, n);
        const ptotals = try a.alloc(u64, n);
        var prows: u64 = 0;
        const win = try a.alloc(u32, n);
        @memset(win, 0);
        var rows: u64 = 0;
        var seq: u64 = 0;
        var win_rows: u64 = 0;
        var oldest_ms: u64 = 0;
        var truncated = false;
        const bits = try a.alloc(u8, n);
        var now: u64 = 0;
        {
            l.mutex.lockUncancelable(l.io);
            defer l.mutex.unlock(l.io);
            // the clock read under the mutex, so no row is newer than `now`; the bits copied with the counts,
            // so a swap's setBits never tears them
            // 時計は mutex の下で読む (`now` より新しい行は無い)。bit は数と一緒に写す (入れ替えの setBits で崩れない)
            now = l.nowMs();
            const from = if (now > @as(u64, window_s) * 1000) now - @as(u64, window_s) * 1000 else 0;
            @memcpy(bits, l.bits);
            @memcpy(totals, l.totals);
            rows = l.rows;
            @memcpy(ptotals, l.ptotals);
            prows = l.prows;
            seq = l.seq;
            const span = @as(usize, l.layers) * 8;
            var k: usize = 0;
            while (k < l.held) : (k += 1) {
                const i = (l.head + ring_rows - 1 - k) % ring_rows;
                const m = l.meta[i];
                if (m.t_ms < from) break;
                oldest_ms = m.t_ms;
                win_rows += 1;
                const ids = l.ids[i * span ..][0..span];
                for (0..l.layers) |li| for (ids[li * 8 ..][0..8]) |e| {
                    if (e < l.experts) win[li * l.experts + e] += 1;
                };
            }
            // the ring ran out before the window's start: the window is shorter than asked
            // ウィンドウの始まりより前にリングが尽きた: ウィンドウは求めたより短い
            truncated = k == l.held and l.held == ring_rows;
        }
        if (@import("builtin").cpu.arch.endian() != .little) return error.Unsupported;
        const tb = try b64(a, std.mem.sliceAsBytes(totals));
        const pb = try b64(a, std.mem.sliceAsBytes(ptotals));
        const wb = try b64(a, std.mem.sliceAsBytes(win));
        const bb = try b64(a, bits);
        const actual: f64 = if (win_rows == 0) 0 else @as(f64, @floatFromInt(now -| oldest_ms)) / 1000.0;
        return std.fmt.allocPrint(a,
            "{{\"layers\":[{d},{d}],\"experts\":{d},\"seq\":{d},\"rows\":{{\"decode\":{d},\"prompt\":{d}}},\"total\":{{\"decode\":\"{s}\",\"prompt\":\"{s}\"}}," ++
                "\"window\":{{\"seconds\":{d},\"rows\":{d},\"actual_seconds\":{d:.3},\"truncated\":{},\"decode\":\"{s}\"}},\"bits\":\"{s}\"}}",
            .{ l.first, l.first + l.layers - 1, l.experts, seq, rows, prows, tb, pb, window_s, win_rows, actual, truncated, wb, bb });
    }

    /// GET /v1/experts/recent?after=SEQ&max=N: the rows written after `after` (oldest first, at most `max`), each with
    /// its picks (layers x 8 u8, base64); `dropped` counts rows after `after` the ring no longer holds.
    /// GET /v1/experts/recent?after=SEQ&max=N: `after` より後に書いた行 (古い順、最大 `max`)。各行に選択 (層 x 8 の
    /// u8、base64)。`dropped` は `after` より後でリングにもう無い行の数。
    pub fn recentJson(l: *Live, a: std.mem.Allocator, after: ?u64, max: u32, text: ?TokenBytes) ![]u8 {
        const span = @as(usize, l.layers) * 8;
        const cap = @min(@as(usize, max), ring_rows);
        const ids = try a.alloc(u8, cap * span);
        // the rows' metas, after up to text_context rows before the first asked for (their text's context)
        // 行の meta。その前に、求められた最初の行より前の最大 text_context 行 (文字の前後関係のため)
        const metas = try a.alloc(Row, text_context + cap);
        var ctx_n: usize = 0;
        var got: usize = 0;
        var seq: u64 = 0;
        var dropped: u64 = 0;
        {
            l.mutex.lockUncancelable(l.io);
            defer l.mutex.unlock(l.io);
            seq = l.seq;
            const oldest = l.seq - l.held; // the seq of the oldest row held / 持っている最古の行の seq
            // `after` at or past the newest: nothing new (and no overflow for after = 2^64 - 1)
            // `after` が最新以降: 新しいものは無い (after = 2^64 - 1 でも溢れない)
            const start = if (after) |s| @min(s, l.seq) +| 1 else if (l.seq > cap) l.seq - cap else 0;
            const first = @max(start, oldest);
            dropped = first - start;
            if (text != null) {
                var s = first -| text_context;
                if (s < oldest) s = oldest;
                while (s < first) : (s += 1) {
                    metas[ctx_n] = l.meta[(l.head + ring_rows - (l.seq - s)) % ring_rows];
                    ctx_n += 1;
                }
            }
            var s = first;
            while (s < l.seq and got < cap) : (s += 1) {
                const i = (l.head + ring_rows - (l.seq - s)) % ring_rows;
                metas[ctx_n + got] = l.meta[i];
                @memcpy(ids[got * span ..][0..span], l.ids[i * span ..][0..span]);
                got += 1;
            }
        }
        // each row's token bytes (outside the mutex) / 各行の token のバイト (mutex の外)
        const bytes = try a.alloc(?[]const u8, ctx_n + got);
        if (text) |tx| {
            for (metas[0 .. ctx_n + got], bytes) |m, *b| b.* = if (m.tok == std.math.maxInt(u32)) null else tx.bytes(tx.ctx, a, m.tok);
        }
        var out: std.ArrayList(u8) = .empty;
        try out.print(a, "{{\"seq\":{d},\"layers\":[{d},{d}],\"dropped\":{d},\"records\":[", .{ seq, l.first, l.first + l.layers - 1, dropped });
        for (0..got) |k| {
            const m = metas[ctx_n + k];
            try out.print(a, "{s}{{\"seq\":{d},\"t\":{d},\"conv\":{d},\"pos\":{d},\"ids\":\"{s}\"", .{ if (k == 0) "" else ",", m.seq, m.t_ms, m.conv, m.pos, try b64(a, ids[k * span ..][0..span]) });
            if (text != null) try out.print(a, ",\"text\":{f}", .{std.json.fmt(try rowText(a, metas[0 .. ctx_n + got], bytes, ctx_n + k), .{})});
            try out.append(a, '}');
        }
        try out.appendSlice(a, "]}");
        return out.toOwnedSlice(a);
    }

    /// Row `at`'s text: the whole UTF-8 characters that end in its token's bytes, a character begun in the
    /// conversation's previous rows (the rows at pos - 1, pos - 2, pos - 3 before it in `metas`) taken with them; a
    /// character still open at the token's end goes to the next row; a special token or an unknown one is "".
    /// Invalid bytes read as U+FFFD. The same row always reads the same while its previous rows are in the ring.
    /// 行 `at` の文字: その token のバイトの中で終わる UTF-8 の文字全部。会話の前の行 (`metas` でその前にある
    /// pos - 1, pos - 2, pos - 3 の行) で始まった文字はいっしょに出す。token の終わりでまだ閉じない文字は次の行へ。
    /// 特殊 token や不明な token は ""。不正なバイトは U+FFFD。前の行がリングにある限り、同じ行はいつも同じ文字。
    fn rowText(a: std.mem.Allocator, metas: []const Row, bytes: []const ?[]const u8, at: usize) ![]const u8 {
        const own = bytes[at] orelse return "";
        const m = metas[at];
        // the previous rows' bytes of the same conversation at consecutive positions, oldest first
        // 同じ会話の連続した位置の前の行のバイト。古い順
        var prev: [3][]const u8 = undefined;
        var np: usize = 0;
        var want = m.pos;
        var i = at;
        while (i > 0 and np < 3 and want > 0) {
            i -= 1;
            const p = metas[i];
            if (p.conv != m.conv) continue;
            if (p.pos != want - 1) break;
            const b = bytes[i] orelse break; // a special token ends a character run / 特殊 token で文字は切れる
            prev[np] = b;
            np += 1;
            want -= 1;
        }
        var buf: std.ArrayList(u8) = .empty;
        var j = np;
        while (j > 0) {
            j -= 1;
            try buf.appendSlice(a, prev[j]);
        }
        const own_at = buf.items.len;
        try buf.appendSlice(a, own);
        const all = buf.items;
        var out: std.ArrayList(u8) = .empty;
        // start at the first lead byte (the bytes may begin inside a character whose start is not in view: dropped)
        // 最初の先頭バイトから (見えない所で始まった文字の途中から始まることがある: 捨てる)
        var k: usize = 0;
        while (k < all.len and isCont(all[k])) k += 1;
        while (k < all.len) {
            const n = std.unicode.utf8ByteSequenceLength(all[k]) catch {
                if (k >= own_at) try out.appendSlice(a, "\u{fffd}");
                k += 1;
                continue;
            };
            if (k + n > all.len) break; // still open at the token's end: the next row's / token の終わりで閉じない: 次の行へ
            const ok = for (all[k + 1 .. k + n]) |c| {
                if (!isCont(c)) break false;
            } else true;
            if (!ok) {
                if (k >= own_at) try out.appendSlice(a, "\u{fffd}");
                k += 1;
                continue;
            }
            if (k + n > own_at) try out.appendSlice(a, all[k .. k + n]); // ends in this row / この行で終わる
            k += n;
        }
        return out.toOwnedSlice(a);
    }

    fn isCont(c: u8) bool {
        return c & 0xC0 == 0x80;
    }

    /// GET /v1/experts/prefill?after=SEQ&max=N: the prompt chunks written after `after` (oldest first, at most `max`;
    /// without `after`, the newest `max`), each with its counts (layers x experts u32 little-endian, base64); `dropped`
    /// counts chunks after `after` the ring no longer holds.
    /// GET /v1/experts/prefill?after=SEQ&max=N: `after` より後に書いた prompt チャンク (古い順、最大 `max`。`after` が
    /// なければ最新の `max` 個)。各チャンクに数 (層 x expert の u32 リトルエンディアン、base64)。`dropped` は `after`
    /// より後でリングにもう無いチャンクの数。
    pub fn prefillJson(l: *Live, a: std.mem.Allocator, after: ?u64, max: u32) ![]u8 {
        const n = @as(usize, l.layers) * l.experts;
        const cap = @min(@as(usize, max), chunk_ring);
        const counts = try a.alloc(u32, cap * n);
        const metas = try a.alloc(Chunk, cap);
        var got: usize = 0;
        var seq: u64 = 0;
        var dropped: u64 = 0;
        {
            l.mutex.lockUncancelable(l.io);
            defer l.mutex.unlock(l.io);
            seq = l.pseq;
            const oldest = l.pseq - l.pheld;
            const start = if (after) |s| @min(s, l.pseq) +| 1 else if (l.pseq > cap) l.pseq - cap else 0;
            const first = @max(start, oldest);
            // without `after` the newest chunks are asked for: none counts as dropped
            // `after` がなければ最新のチャンクを求めている: 落ちたものは無い
            dropped = if (after != null) first - start else 0;
            var s = first;
            while (s < l.pseq and got < cap) : (s += 1) {
                const i = (l.phead + chunk_ring - (l.pseq - s)) % chunk_ring;
                metas[got] = l.pmeta[i];
                @memcpy(counts[got * n ..][0..n], l.pcounts[i * n ..][0..n]);
                got += 1;
            }
        }
        if (@import("builtin").cpu.arch.endian() != .little) return error.Unsupported;
        var out: std.ArrayList(u8) = .empty;
        try out.print(a, "{{\"seq\":{d},\"layers\":[{d},{d}],\"experts\":{d},\"dropped\":{d},\"records\":[", .{ seq, l.first, l.first + l.layers - 1, l.experts, dropped });
        for (0..got) |k| {
            const m = metas[k];
            try out.print(a, "{s}{{\"seq\":{d},\"t\":{d},\"ms\":{d:.1},\"conv\":{d},\"pos\":{d},\"rows\":{d},\"done\":{},\"counts\":\"{s}\"}}", .{ if (k == 0) "" else ",", m.seq, m.t_ms, m.ms, m.conv, m.pos, m.rows, m.done, try b64(a, std.mem.sliceAsBytes(counts[k * n ..][0..n])) });
        }
        try out.appendSlice(a, "]}");
        return out.toOwnedSlice(a);
    }
};
