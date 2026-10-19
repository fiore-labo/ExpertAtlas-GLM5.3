//! One rank's forward pass of GLM-5.3: the layer (attention, gather, MLP or experts) for prompt chunks and decode
//! windows, the head's argmax, and the sequence-parallel prompt chunk.
//! 1 ランク分の GLM-5.3 の forward: prompt チャンクと decode ウィンドウの層 (attention・gather・MLP または expert)、
//! head の argmax、sequence-parallel の prompt チャンク。
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const model = @import("model.zig");
const linear = @import("linear.zig");
const exl3 = @import("exl3.zig");
const ops_mod = @import("ops.zig");
const experts = @import("experts.zig");
const prompt_experts = @import("prompt_experts.zig");
const moe = @import("moe.zig");
const roce = @import("roce.zig");
const unpack_cache = @import("unpack_cache.zig");

const Buf = cuda.DeviceBuffer;
pub const max_rows = 2048;
pub const MAX_LIN_ROWS = 128; // the row-invariant linear's widest call / 行に依らない linear の最大行数
/// Rows up to this many take the decode class of kernels, prompt chunks as well as decode windows.
/// この行数までは prompt チャンクでも decode ウィンドウでも decode 用のカーネルを使う。
const small_rows = 16;
const sel_rows = 128; // the indexer scores rows in blocks of this many / indexer はこの行数ずつ点数を付ける
// Glue.topkWordsAny16 from this many slots a row (shorter rows: topkWords, measured faster at 20,000)
// Glue.topkWordsAny16 を使う行のスロット数の下限 (それより短い行は topkWords。20,000 ではそちらが速かった)
const any16_min_keys = 32768;
// Ctx.sel_tok: rows a logged decode window may hold (engine.vrows)
// Ctx.sel_tok: 記録する decode ウィンドウの最大行数 (engine.vrows)
pub const sel_log_rows = 8;
// topkValuesWide from this many keys a row (below it, the one-block kernel is faster)
// 1 行のキーがこの数以上なら topkValuesWide (未満なら 1 ブロックのカーネルの方が速い)
const wide_topk_min = 16384;
// blocks of this many rows take the radix select: the k-th best by the score word alone, ties in column order
// この行数以上のブロックは radix select: 点数の語だけで k 番目を決め、同点は列の順
const radix_min_rows = 32;
// the default widest key range scored (Buffers.keys)
// 点数を付けるキー範囲の既定の最大 (Buffers.keys)
pub const max_keys = 8192;

/// TF_ZIG_TOPK=sort keeps the bitonic full-row sort for the indexer's top-k keys (default: radix select).
/// TF_ZIG_TOPK=sort なら indexer の top-k キーに行全体の bitonic ソートを使う (既定は radix select)。
pub var topk_select: bool = true;

/// The plain-CUDA kernels of glue.cu.
/// glue.cu の素の CUDA カーネル。
pub const Glue = struct {
    module: cuda.Module,
    gather_rows16: cuda.Function,
    copy3_16: cuda.Function,
    copy3_32: cuda.Function,
    scatter3_16: cuda.Function,
    scatter3_32: cuda.Function,
    add_f32: cuda.Function,
    topk_keys: cuda.Function,
    row_argmax: cuda.Function,
    pick_fn: cuda.Function,
    set_i32: cuda.Function,
    topk_select: cuda.Function,
    draft_rope: cuda.Function,
    draft_invfreq: cuda.Function,
    draft_ring: cuda.Function,
    draft_advance: cuda.Function,
    rank_sum_bf16: cuda.Function,
    draft_topk: cuda.Function,
    topk_values: cuda.Function,
    topk_words: cuda.Function,
    topk_words_any: cuda.Function,
    dcp_select: cuda.Function,
    dcp_own: cuda.Function,
    round_bf16: cuda.Function,
    to_bf16: cuda.Function,
    add_to_bf16: cuda.Function,
    radd_to_bf16: cuda.Function,
    topk_words_any16: cuda.Function,
    from_bf16: cuda.Function,
    tk_hist: cuda.Function,
    tk_digit: cuda.Function,
    tk_split: cuda.Function,
    tk_finish: cuda.Function,
    slot_lanes: cuda.Function,
    slot_fold: cuda.Function,
    slot_copy: cuda.Function,
    slot_zero: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Glue {
        const m = try cuda.Module.load(d, cuda.kernels.glm53_glue);
        return .{ .module = m, .gather_rows16 = try m.function("tf_glm53_gather_rows16"), .copy3_16 = try m.function("tf_glm53_copy3_16"), .copy3_32 = try m.function("tf_glm53_copy3_32"), .scatter3_16 = try m.function("tf_glm53_scatter3_16"), .scatter3_32 = try m.function("tf_glm53_scatter3_32"), .add_f32 = try m.function("tf_glm53_add_f32"), .topk_keys = try m.function("tf_glm53_topk_keys"), .row_argmax = try m.function("tf_glm53_row_argmax"), .pick_fn = try m.function("tf_glm53_pick"), .set_i32 = try m.function("tf_glm53_set_i32"), .topk_select = try m.function("tf_glm53_topk_keys_select"), .draft_rope = try m.function("tf_glm53_draft_rope"), .draft_invfreq = try m.function("tf_glm53_draft_invfreq"), .draft_ring = try m.function("tf_glm53_draft_ring"), .draft_advance = try m.function("tf_glm53_draft_advance"), .rank_sum_bf16 = try m.function("tf_glm53_rank_sum_bf16"), .draft_topk = try m.function("tf_glm53_draft_topk"), .topk_values = try m.function("tf_glm53_topk_values"), .topk_words = try m.function("tf_glm53_topk_words"), .topk_words_any = try m.function("tf_glm53_topk_words_any"), .dcp_select = try m.function("tf_glm53_dcp_select"), .dcp_own = try m.function("tf_glm53_dcp_own"), .round_bf16 = try m.function("tf_glm53_round_bf16"), .to_bf16 = try m.function("tf_glm53_to_bf16"), .add_to_bf16 = try m.function("tf_glm53_add_to_bf16"), .radd_to_bf16 = try m.function("tf_glm53_radd_to_bf16"), .topk_words_any16 = try m.function("tf_glm53_topk_words_any16"), .from_bf16 = try m.function("tf_glm53_from_bf16"), .tk_hist = try m.function("tf_glm53_tk_hist"), .tk_digit = try m.function("tf_glm53_tk_digit"), .tk_split = try m.function("tf_glm53_tk_split"), .tk_finish = try m.function("tf_glm53_tk_finish"), .slot_lanes = try m.function("tf_glm53_slot_lanes"), .slot_fold = try m.function("tf_glm53_slot_fold"), .slot_copy = try m.function("tf_glm53_slot_copy"), .slot_zero = try m.function("tf_glm53_slot_zero") };
    }

    fn grid(n2: u32, rows: u32) cuda.Config {
        return .{ .grid = .{ .x = @min((n2 + 255) / 256, 64), .y = rows, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
    }

    /// The 3-d copies' grid: the two outer axes as the grid's y (n1) and z (n0), so no thread divides to find them.
    /// 3 次元コピーの grid: 外側の 2 軸を grid の y (n1) と z (n0) にするので、スレッドは割り算で求めずに済む。
    fn grid3(n2: u32, n0: u32, n1: u32) cuda.Config {
        return .{ .grid = .{ .x = @min((n2 + 255) / 256, 64), .y = n1, .z = n0 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
    }

    /// dst [n0, n1, n2] contiguous <- src view with element strides (si, sj, 1); `wide`: 4-byte elements.
    /// dst [n0, n1, n2] (連続) <- 要素ストライド (si, sj, 1) の src ビュー。`wide` なら 4 バイト要素。
    pub fn copy3(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, n0: u32, n1: u32, n2: u32, si: i64, sj: i64, wide: bool) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(c_int, @intCast(n1)));
        a.add(@as(c_int, @intCast(n2)));
        a.add(si);
        a.add(sj);
        try cuda.launch.launch(if (wide) g.copy3_32 else g.copy3_16, grid3(n2, n0, n1), s, &a);
    }

    /// dst view with element strides (di, dj, 1) <- src [n0, n1, n2] contiguous.
    /// 要素ストライド (di, dj, 1) の dst ビュー <- src [n0, n1, n2] (連続)。
    pub fn scatter3(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, n0: u32, n1: u32, n2: u32, di: i64, dj: i64, wide: bool) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(c_int, @intCast(n1)));
        a.add(@as(c_int, @intCast(n2)));
        a.add(di);
        a.add(dj);
        try cuda.launch.launch(if (wide) g.scatter3_32 else g.scatter3_16, grid3(n2, n0, n1), s, &a);
    }

    pub fn gatherRows(g: *const Glue, s: cuda.Stream, src: u64, ids: u64, dst: u64, rows: u32, width: u32) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(ids);
        a.add(dst);
        a.add(@as(c_int, @intCast(width)));
        try cuda.launch.launch(g.gather_rows16, grid(width, rows), s, &a);
    }

    /// Each row's k largest packed int64 scores as ascending int32 keys.
    /// 各行の packed int64 点数の上位 k 個を、昇順の int32 キーとして出す。
    pub fn topkKeys(g: *const Glue, s: cuda.Stream, sc: u64, out: u64, n: u32, t: u32, k: u32) !void {
        if (topk_select) {
            // radix select + compaction: the same keys without sorting the row (TF_ZIG_TOPK=sort: the bitonic sort)
            // radix select と詰め直し: 行をソートせずに同じキーを得る (TF_ZIG_TOPK=sort なら bitonic ソート)
            var a: cuda.Args = .{};
            a.add(sc);
            a.add(out);
            a.add(@as(c_int, @intCast(t)));
            a.add(@as(c_int, @intCast(k)));
            return cuda.launch.launch(g.topk_select, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = k * 4 }, s, &a);
        }
        const npow2 = std.math.ceilPowerOfTwo(u32, t) catch unreachable;
        const smem: u32 = npow2 * 8 + k * 4;
        if (smem > 48 * 1024) try g.topk_keys.allowDynamicShared(smem);
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(out);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(npow2)));
        try cuda.launch.launch(g.topk_keys, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = smem }, s, &a);
    }

    /// Each of n rows' first maximum over V logits -> amax [n, 4] = [value, float(id + off), 0, 0].
    /// n 行それぞれの V 個の logit で最初の最大値 -> amax [n, 4] = [値, float(id + off), 0, 0]。
    pub fn rowArgmax(g: *const Glue, s: cuda.Stream, logits: u64, amax: u64, n: u32, v: u32, off: u32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(amax);
        a.add(@as(c_int, @intCast(v)));
        a.add(@as(c_int, @intCast(off)));
        try cuda.launch.launch(g.row_argmax, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// The gathered rows [world, n, 4] -> ids int64 [n] and tok int32 [n]: the lowest rank among equal maxima.
    /// 集めた行 [world, n, 4] -> ids int64 [n] と tok int32 [n]。最大値が同じなら最も小さいランクを選ぶ。
    pub fn pick(g: *const Glue, s: cuda.Stream, all: u64, world: u32, n: u32, ids: u64, tok: u64) !void {
        var a: cuda.Args = .{};
        a.add(all);
        a.add(@as(c_int, @intCast(world)));
        a.add(@as(c_int, @intCast(n)));
        a.add(ids);
        a.add(tok);
        try cuda.launch.launch(g.pick_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn setI32(g: *const Glue, s: cuda.Stream, p: u64, v: u32) !void {
        var a: cuda.Args = .{};
        a.add(p);
        a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(g.set_i32, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// inv [half] fp32 = 1 / theta ** (2 j / hd): the rotary frequencies, once at load.
    /// inv [half] fp32 = 1 / theta ** (2 j / hd): 回転の周波数。読み込み時に 1 回だけ。
    pub fn draftInvfreq(g: *const Glue, s: cuda.Stream, inv: u64, theta: f32, half: u32, hd: u32) !void {
        var a: cuda.Args = .{};
        a.add(inv);
        a.add(theta);
        a.add(@as(c_int, @intCast(half)));
        a.add(@as(c_int, @intCast(hd)));
        try cuda.launch.launch(g.draft_invfreq, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 64, .y = 1, .z = 1 } }, s, &a);
    }

    /// The drafter's RoPE tables: cos / sin [n, half] fp32 of (pos + i) * inv_freq.
    /// drafter の RoPE 表: (pos + i) * inv_freq の cos / sin [n, half] fp32。
    pub fn draftRope(g: *const Glue, s: cuda.Stream, pos: u64, inv: u64, cos: u64, sin: u64, n: u32, half: u32) !void {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(inv);
        a.add(cos);
        a.add(sin);
        a.add(@as(c_int, @intCast(half)));
        try cuda.launch.launch(g.draft_rope, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 64, .y = 1, .z = 1 } }, s, &a);
    }

    /// src [kvh, n, hd] bf16 into the ring [kvh, ring, hd] at slots (pos + i) % ring.
    /// src [kvh, n, hd] bf16 をリング [kvh, ring, hd] のスロット (pos + i) % ring に書く。
    pub fn draftRing(g: *const Glue, s: cuda.Stream, src: u64, ring: u64, pos: u64, n: u32, hd: u32, kvh: u32, ring_rows: u32) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(ring);
        a.add(pos);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(hd)));
        a.add(@as(c_int, @intCast(ring_rows)));
        try cuda.launch.launch(g.draft_ring, .{ .grid = .{ .x = n, .y = kvh, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn draftAdvance(g: *const Glue, s: cuda.Stream, pos: u64, n: u32) !void {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(@as(c_int, @intCast(n)));
        try cuda.launch.launch(g.draft_advance, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// The ranks' fp32 partials [world, n] added in rank order -> bf16 [n].
    /// 各ランクの fp32 部分和 [world, n] をランク順に足す -> bf16 [n]。
    pub fn rankSumBf16(g: *const Glue, s: cuda.Stream, gath: u64, out: u64, n: u32, world: u32) !void {
        var a: cuda.Args = .{};
        a.add(gath);
        a.add(out);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(world)));
        try cuda.launch.launch(g.rank_sum_bf16, .{ .grid = .{ .x = @min((n + 255) / 256, 1024), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// Each row's top k (values descending, equal values by lower index) -> [rows, 2k] (values | ids as fp32 bits).
    /// 各行の上位 k 個 (値の降順、同じ値は index の小さい順) -> [rows, 2k] (値 | fp32 のビットとしての id)。
    pub fn draftTopk(g: *const Glue, s: cuda.Stream, logits: u64, out: u64, rows: u32, v: u32, k: u32, vocab_off: u32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        a.add(@as(c_int, @intCast(v)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(vocab_off)));
        try cuda.launch.launch(g.draft_topk, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// Each row's k largest packed keys of t (any order; past min(k, t): the -inf sentinel) -> out [n, k] int64.
    /// 各行の t 個の packed キーの上位 k 個 (順不同。min(k, t) より後は -inf の番兵) -> out [n, k] int64。
    pub fn topkValues(g: *const Glue, s: cuda.Stream, sc: u64, out: u64, n: u32, t: u32, k: u32) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(out);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(c_int, @intCast(k)));
        try cuda.launch.launch(g.topk_values, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// topkValues' bytes from the scores' int32 order words (unpacked scores) of rows r0.. at `pos`: 4 + 1 passes over
    /// 4 bytes a key instead of 8 + 1 over 8. `ordered` false: the same keys in any order from two passes, enough where
    /// the merge after them takes the packed keys' exact order (spSelect).
    /// topkValues と同じバイトを、`pos` の行 r0.. の点数の int32 順序語 (pack なし) から出す: 1 キー 8 バイトを
    /// 8 + 1 回読む代わりに 4 バイトを 4 + 1 回。`ordered` が false なら同じキーを順不同で 2 pass で出す。後のマージが
    /// packed キーの正確な順を取る所 (spSelect) ではそれで足りる。
    /// topkWords' any-order picks from the words' top 16 bits `p16` (Ops.indexScoresWords16) and the words `sc`: the
    /// same keys, the full words read only at the bucket and above.
    /// 語の上位 16 ビット `p16` (Ops.indexScoresWords16) と語 `sc` からの、topkWords の順不同の選択: 同じキーで、語
    /// 全体はバケット以上の所だけ読む。
    pub fn topkWordsAny16(g: *const Glue, s: cuda.Stream, sc: u64, p16: u64, out: u64, pos: u64, n: u32, t: u32, k: u32, r0: u32, dcp: u32, rank: u32) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(p16);
        a.add(out);
        a.add(pos);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(r0)));
        a.add(@as(c_int, @intCast(dcp)));
        a.add(@as(c_int, @intCast(rank)));
        try cuda.launch.launch(g.topk_words_any16, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn topkWords(g: *const Glue, s: cuda.Stream, sc: u64, out: u64, pos: u64, n: u32, t: u32, k: u32, r0: u32, dcp: u32, rank: u32, ordered: bool) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(out);
        a.add(pos);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(r0)));
        a.add(@as(c_int, @intCast(dcp)));
        a.add(@as(c_int, @intCast(rank)));
        try cuda.launch.launch(if (ordered) g.topk_words else g.topk_words_any, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// topkValues for decode windows over long rows (many blocks a row): the same set, any order.
    /// `state`: 16 x TkState (1040 bytes each, zeroed once), `cand`: [n, t] int64 scratch.
    /// 長い行に対する decode ウィンドウ用の topkValues (1 行に多数のブロック): 同じ集合を順不同で出す。
    /// `state`: TkState x 16 (各 1040 バイト、最初に 1 回ゼロにする)、`cand`: [n, t] int64 の作業領域。
    pub fn topkValuesWide(g: *const Glue, s: cuda.Stream, sc: u64, out: u64, n: u32, t: u32, k: u32, state: u64, cand: u64) !void {
        // short rows: one block a row is faster
        // 短い行は 1 行 1 ブロックの方が速い
        if (t < wide_topk_min) return g.topkValues(s, sc, out, n, t, k);
        const nb: u32 = @max(1, @min(256, t / 1024));
        for (0..2) |pass| {
            var a: cuda.Args = .{};
            a.add(sc);
            a.add(@as(c_int, @intCast(t)));
            a.add(state);
            a.add(@as(c_int, @intCast(pass)));
            try cuda.launch.launch(g.tk_hist, .{ .grid = .{ .x = nb, .y = n, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
            var d: cuda.Args = .{};
            d.add(state);
            d.add(@as(c_int, @intCast(t)));
            d.add(@as(c_int, @intCast(k)));
            d.add(@as(c_int, @intCast(pass)));
            try cuda.launch.launch(g.tk_digit, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &d);
        }
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(@as(c_int, @intCast(t)));
        a.add(state);
        a.add(out);
        a.add(cand);
        a.add(@as(c_int, @intCast(k)));
        try cuda.launch.launch(g.tk_split, .{ .grid = .{ .x = nb, .y = n, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        var f: cuda.Args = .{};
        f.add(cand);
        f.add(@as(c_int, @intCast(t)));
        f.add(state);
        f.add(out);
        f.add(@as(c_int, @intCast(k)));
        try cuda.launch.launch(g.tk_finish, .{ .grid = .{ .x = n, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &f);
    }

    /// The group's candidates [G, n, k] of each row -> the global top k: this member's local slots ascending (tok
    /// [n, k]) and their count (cnt [n]). `member` null (spSelect's owner step): every global pick ascending into tok,
    /// no count. Runs the first `rows` rows of the `n`-row candidate blocks.
    /// 各行のグループの候補 [G, n, k] -> 全体の上位 k 個: このメンバーのローカルスロット (昇順、tok [n, k]) とその数
    /// (cnt [n])。`member` が null (spSelect の owner の段) なら全体の選択をすべて昇順で tok に書き、数は出さない。
    /// `n` 行の候補ブロックのうち先頭の `rows` 行を処理する。
    pub fn dcpSelect(g: *const Glue, s: cuda.Stream, cand: u64, tok: u64, cnt: u64, n: u32, rows: u32, k: u32, groups: u32, member: ?u32, hiword: bool) !void {
        var a: cuda.Args = .{};
        a.add(cand);
        a.add(tok);
        a.add(cnt);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(groups)));
        a.add(@as(c_int, if (member) |m| @intCast(m) else -1));
        a.add(@as(c_int, @intFromBool(hiword)));
        const smem: u32 = groups * k * 8 + k * 4;
        if (smem > 48 * 1024) try g.dcp_select.allowDynamicShared(smem);
        try cuda.launch.launch(g.dcp_select, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = smem }, s, &a);
    }

    /// x[:n] = float(bf16(x[:n])) in place (fp32).
    /// x[:n] = float(bf16(x[:n])) をその場で (fp32)。
    pub fn roundBf16(g: *const Glue, s: cuda.Stream, x: u64, n: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(n)));
        try cuda.launch.launch(g.round_bf16, .{ .grid = .{ .x = @intCast(@min((n + 1023) / 1024, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// dst[:n] = bf16(src[:n]) (fp32 in, round to nearest even).
    /// dst[:n] = bf16(src[:n]) (fp32 入力、最近接偶数への丸め)。
    pub fn toBf16(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, n: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(n)));
        try cuda.launch.launch(g.to_bf16, .{ .grid = .{ .x = @intCast(@min((n + 1023) / 1024, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// dst[:n] = float(src[:n]) (bf16 in, exact).
    /// dst[:n] = float(src[:n]) (bf16 入力、誤差なし)。
    pub fn fromBf16(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, n: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(n)));
        try cuda.launch.launch(g.from_bf16, .{ .grid = .{ .x = @intCast(@min((n + 1023) / 1024, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// spSelect's last step: each of n rows' k global picks (ascending) -> this member's local slots in place
    /// (ascending) and their count.
    /// spSelect の最後の段: n 行それぞれの全体の選択 k 個 (昇順) -> このメンバーのローカルスロット (昇順、その場で)
    /// とその数。
    pub fn dcpOwn(g: *const Glue, s: cuda.Stream, tok: u64, cnt: u64, n: u32, k: u32, groups: u32, member: u32) !void {
        var a: cuda.Args = .{};
        a.add(tok);
        a.add(cnt);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, @intCast(groups)));
        a.add(@as(c_int, @intCast(member)));
        try cuda.launch.launch(g.dcp_own, .{ .grid = .{ .x = (n + 7) / 8, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// The lane hashes of rows [r0, r0 + n) of `src` (a device table of L lane pointers, `words` 8-byte words a row
    /// each) into lanes [L, n] u64; with `dst` the rows also go to its rows d0 ...
    /// `src` (L 個の lane ポインタのデバイス上の表、各行 `words` 個の 8 バイト語) の行 [r0, r0 + n) の lane ハッシュを
    /// lanes [L, n] u64 に書く。`dst` があれば行を dst の行 d0 ... にも写す。
    pub fn slotLanes(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, words: u64, L: u32, r0: i64, d0: i64, n: u32, lanes: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(words);
        a.add(@as(c_longlong, r0));
        a.add(@as(c_longlong, d0));
        a.add(@as(c_int, @intCast(n)));
        a.add(lanes);
        try cuda.launch.launch(g.slot_lanes, .{ .grid = .{ .x = (n + 127) / 128, .y = L, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// H of positions p0 + i * step from lanes [L, n] into out (exp == 0), or compared with exp, mismatches added to
    /// bad[0].
    /// lanes [L, n] から位置 p0 + i * step の H を出す: out に書く (exp == 0) か、exp と比べて不一致の数を bad[0] に
    /// 足す。
    pub fn slotFold(g: *const Glue, s: cuda.Stream, lanes: u64, L: u32, n: u32, p0: i64, step: i64, pid: u64, gen: u64, out: u64, exp: u64, bad: u64) !void {
        var a: cuda.Args = .{};
        a.add(lanes);
        a.add(@as(c_int, @intCast(L)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_longlong, p0));
        a.add(@as(c_longlong, step));
        a.add(pid);
        a.add(gen);
        a.add(out);
        a.add(exp);
        a.add(bad);
        try cuda.launch.launch(g.slot_fold, .{ .grid = .{ .x = (n + 127) / 128, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// Rows [r0, r0 + n) of every lane of `src` to rows [d0, d0 + n) of `dst`.
    /// `src` の全 lane の行 [r0, r0 + n) を `dst` の行 [d0, d0 + n) に写す。
    pub fn slotCopy(g: *const Glue, s: cuda.Stream, src: u64, dst: u64, words: u64, L: u32, max_words: u32, r0: i64, d0: i64, n: i64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(words);
        a.add(@as(c_longlong, r0));
        a.add(@as(c_longlong, d0));
        a.add(@as(c_longlong, n));
        const pieces: u64 = @as(u64, @intCast(n)) * max_words / 2;
        const blocks: u32 = @intCast(@max(1, @min(65535, (pieces + 2047) / 2048)));
        try cuda.launch.launch(g.slot_copy, .{ .grid = .{ .x = blocks, .y = L, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// p[0 .. n) = 0 (int64).
    /// p[0 .. n) (int64) をゼロにする。
    pub fn slotZero(g: *const Glue, s: cuda.Stream, p: u64, n: u64) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(p);
        a.add(@as(c_longlong, @intCast(n)));
        try cuda.launch.launch(g.slot_zero, .{ .grid = .{ .x = @intCast(@min(1024, (n + 255) / 256)), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// dst[:n] = bf16(a[:n] + x[:n]) (addF32 then toBf16, one launch).
    /// dst[:n] = bf16(a[:n] + x[:n]) (addF32 と toBf16 を 1 回の launch で)。
    pub fn addToBf16(g: *const Glue, s: cuda.Stream, a_ptr: u64, x: u64, dst: u64, n: u64) !void {
        var a: cuda.Args = .{};
        a.add(a_ptr);
        a.add(x);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(n)));
        try cuda.launch.launch(g.add_to_bf16, .{ .grid = .{ .x = @intCast(@min((n + 1023) / 1024, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    /// dst[:n] = bf16(r(a[:n]) + x[:n]), r through bf16 when `round` (roundBf16, addF32 and toBf16 in one pass).
    /// dst[:n] = bf16(r(a[:n]) + x[:n])。`round` なら r は bf16 を通す (roundBf16・addF32・toBf16 を 1 pass で)。
    pub fn raddToBf16(g: *const Glue, s: cuda.Stream, a_ptr: u64, x: u64, dst: u64, n: u64, round: bool) !void {
        return g.radd(s, a_ptr, x, dst, n, @intFromBool(round));
    }

    /// raddToBf16 with `a` fp16 [n] (prompt_experts' fp16 sums), widened exactly and rounded through bf16.
    /// `a` が fp16 [n] (prompt_experts の fp16 の和) の raddToBf16。誤差なしに広げて bf16 を通す。
    pub fn raddF16ToBf16(g: *const Glue, s: cuda.Stream, a_ptr: u64, x: u64, dst: u64, n: u64) !void {
        return g.radd(s, a_ptr, x, dst, n, 2);
    }

    fn radd(g: *const Glue, s: cuda.Stream, a_ptr: u64, x: u64, dst: u64, n: u64, mode: c_int) !void {
        var a: cuda.Args = .{};
        a.add(a_ptr);
        a.add(x);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(n)));
        a.add(mode);
        try cuda.launch.launch(g.radd_to_bf16, .{ .grid = .{ .x = @intCast(@min((n + 1023) / 1024, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn addF32(g: *const Glue, s: cuda.Stream, acc: u64, x: u64, n: u64) !void {
        var a: cuda.Args = .{};
        a.add(acc);
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(g.add_f32, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
};

/// Work buffers for windows up to max_rows rows (a prompt chunk or a decode window uses their first rows).
/// max_rows 行までのウィンドウ用の作業バッファ (prompt チャンクも decode ウィンドウも先頭の行を使う)。
pub const Buffers = struct {
    ids: Buf, x: Buf, normed: Buf, qa: Buf, qn: Buf, kva: Buf, q: Buf, qlat: Buf, qrot: Buf, ol: Buf, o: Buf,
    pm: Buf, pl: Buf, ik: Buf, iw: Buf, iq: Buf, iq8: Buf, iqs: Buf, tok: Buf, g: Buf, u: Buf, act: Buf, part: Buf, red: Buf, gath: Buf,
    send: Buf, recv: Buf, mine: Buf, tmp: Buf, rpart: Buf, xh: Buf, wq: Buf, had: Buf, inv: Buf,
    mlog: Buf, pick: Buf, wts: Buf, sy: Buf, sc: Buf, po: Buf, dummy: Buf, fnormed: Buf, lpart: Buf,
    amax: Buf, amax_all: Buf, tok_out: Buf,
    rows: u32,
    keys: usize, // the widest key range the indexer scores / indexer が点数を付ける最大のキー範囲
    scratch: linear.Scratch,
    /// DFlash2: the residual after each tap layer, [rows, n_taps * D] bf16; set by the engine.
    /// DFlash2: 各 tap 層の後の residual、[rows, n_taps * D] bf16。engine が設定する。
    taps: ?Buf = null,
    // DCP: the absorbed queries packed and gathered, the partials' exchange, the rows' own key counts, the group's
    // candidates
    // DCP: 吸収済みの query の pack と gather、部分結果の交換、各行の自分のキー数、グループの候補
    dcp: u32 = 1,
    qpack: ?Buf = null,
    qall: ?Buf = null,
    osend: ?Buf = null,
    lsend: ?Buf = null,
    orecv: ?Buf = null,
    lrecv: ?Buf = null,
    // decode windows: a member's partials and their log-sum-exps in one block (`Ctx.xBytes`), so that one collective
    // carries both; the group's blocks gathered in member order
    // decode ウィンドウ: メンバーの部分結果と log-sum-exp を 1 ブロックにまとめ (`Ctx.xBytes`)、1 回の collective で
    // 両方を送る。グループのブロックはメンバー順に集める
    xsend: ?Buf = null,
    xrecv: ?Buf = null,
    cnt: ?Buf = null,
    cand: ?Buf = null,
    // this rank's top index_topk packed keys of a block of rows
    // 行ブロックに対する、このランクの上位 index_topk 個の packed キー
    dmine: ?Buf = null,
    // topkValuesWide's per-row state (16 x 1040 bytes, zeroed) and candidate scratch [min(rows, 31), keys / dcp]
    // topkValuesWide の行ごとの状態 (16 x 1040 バイト、ゼロ) と候補の作業領域 [min(rows, 31), keys / dcp]
    tk_state: ?Buf = null,
    tk_cand: ?Buf = null,
    // sequence-parallel prompt parts (SpPart): this rank's own rows of the residual stream [ceil(rows / world), D]
    // bf16, and the part's device positions (int32: first row, first own row, first row of the DCP group's)
    // sequence-parallel の prompt の part (SpPart): このランクが持つ residual stream の行 [ceil(rows / world), D]
    // bf16 と、part のデバイス上の位置 (int32: 先頭行、自分の先頭行、DCP グループの先頭行)
    xo: ?Buf = null,
    ppos: ?Buf = null,
    // PREFILL_REDUCE=ring under SP: the partial rows as bf16 [rows, D] and this rank's reduce-scatter share [n, D]
    // SP で PREFILL_REDUCE=ring のとき: 部分和の行 (bf16 [rows, D]) と、このランクの reduce-scatter の取り分 [n, D]
    hpart: ?Buf = null,
    hred: ?Buf = null,
    // DECODE_BF16: a window's partial rows as bf16 and their bf16 sum
    // DECODE_BF16: ウィンドウの部分和の行 (bf16) と、その bf16 の和
    dpart: ?Buf = null,
    dred: ?Buf = null,

    /// `reduce_rows`: the rows the TP reduce scratch (red, gath, send, recv, mine) must hold - `rows`, or fewer where
    /// the wide windows reduce elsewhere (engine.zig: the SP parts under PREFILL_REDUCE=ring reduce through hpart).
    /// gath still holds `rows` rows of bf16 (spGatherX).
    /// `reduce_rows`: TP の reduce の作業域 (red、gath、send、recv、mine) が持つべき行数 - `rows`、または広いウィンドウが
    /// 別の所で reduce するならそれより少なく (engine.zig: PREFILL_REDUCE=ring の SP の part は hpart で reduce する)。
    pub fn init(d: *const cuda.Driver, cfg: Config, heads: u32, world: u32, rows: u32, vocab_share: usize, inv: []const u8, had: []const u8, keys: usize, dcp: u32, partials: bool, reduce_rows: u32) !Buffers {
        const R: usize = rows;
        const RR: usize = @min(rows, reduce_rows);
        const D = cfg.hidden;
        const lw = cfg.kv_lora;
        const width = @max(cfg.inter, cfg.moe_inter * @max(cfg.shared_experts, 1)) / world;
        const a = struct {
            fn z(dd: *const cuda.Driver, n: usize) !Buf {
                var b = try Buf.alloc(dd, @max(n, 16));
                try b.fill8(0, null);
                return b;
            }
        }.z;
        // po: chunk partials of the decode class (8 chunks for each of up to 16 rows) or one chunk a row
        // po: decode クラスのチャンクの部分結果 (16 行までの各行に 8 チャンク) か、1 行 1 チャンク
        const slots: usize = @max(cfg.index_topk / 256 * @min(R, 16), R);
        var out: Buffers = .{
            .ids = try a(d, R * 8),
            .x = try a(d, R * D * 2),
            .normed = try a(d, R * D * 2),
            .qa = try a(d, R * cfg.q_lora * 2),
            .qn = try a(d, R * cfg.q_lora * 2),
            .kva = try a(d, R * 640 * 2),
            .q = try a(d, R * heads * (cfg.qk_nope + cfg.qk_rope) * 2),
            .qlat = try a(d, R * heads * lw * 2),
            .qrot = try a(d, R * heads * cfg.qk_rope * 2),
            .ol = try a(d, R * heads * lw * 2),
            .o = try a(d, R * heads * cfg.v_head * 2),
            .pm = try a(d, slots * heads * 4 * dcp),
            .pl = try a(d, slots * heads * 4 * dcp),
            .ik = try a(d, R * cfg.index_dim * 4),
            .iw = try a(d, R * cfg.index_heads * 4),
            .iq = try a(d, R * cfg.index_heads * cfg.index_dim * 2),
            // TF_GLM53_INDEX_FP8: the index queries as e4m3 codes (fragment order) and their scales
            // TF_GLM53_INDEX_FP8: index の query の e4m3 コード (fragment の順) とその scale
            .iq8 = try a(d, R * cfg.index_heads * cfg.index_dim),
            .iqs = try a(d, R * cfg.index_heads * 4),
            .tok = try a(d, R * cfg.index_topk * 4),
            .g = try a(d, R * width * 2),
            .u = try a(d, R * width * 2),
            .act = try a(d, R * width * 2),
            .part = try a(d, R * D * 4),
            .red = try a(d, RR * D * 4),
            // also every row of the window as bf16: spGatherX gathers the ranks' own rows of an SP part through it
            // (the drafter's taps, the head's last part)
            // ウィンドウの全行の bf16 も入ること: spGatherX が SP の part の各ランクの自分の行をこれで集める (drafter の
            // tap、head の最後の part)
            .gath = try a(d, @max(world * RR * D * 4, R * D * 2)),
            .send = try a(d, RR * D * 4),
            .recv = try a(d, RR * D * 4),
            .mine = try a(d, RR * D * 4 / world),
            .tmp = try a(d, R * heads * lw * 2),
            // the router's K-slice partials [8, m, ne]: up to 1024 columns for every row (indexer, experts), the
            // vocabulary share for the head's rows (at most 16)
            // router の K スライスの部分和 [8, m, ne]: 全行に 1024 列まで (indexer、expert)、head の行 (16 まで) には
            // 語彙の取り分
            .rpart = try a(d, 8 * @max(R * 1024, @min(R, 16) * vocab_share) * 4),
            .xh = try a(d, R * D * 2),
            .wq = try a(d, 6144 * 3072 * 2),
            .mlog = try a(d, R * cfg.experts * 4),
            .pick = try a(d, R * cfg.top_k * 4),
            .wts = try a(d, R * cfg.top_k * 4),
            .sy = try a(d, R * D * 4),
            // DCP: each member scores its own slots
            // DCP: 各メンバーは自分のスロットだけに点数を付ける
            .sc = try a(d, @min(R, sel_rows) * ((keys + dcp - 1) / dcp) * 8),
            // the chunks' fp32 partials; an extra sequence-parallel part whose chunks the direct attention takes whole
            // never reads them (`partials` false)
            // チャンクの fp32 部分結果。direct attention がチャンクを丸ごと処理する追加の sequence-parallel の part は
            // 読まない (`partials` が false)
            .po = try a(d, if (partials) slots * heads * lw * 4 * dcp else 16),
            .dummy = try a(d, 16),
            .fnormed = try a(d, R * D * 2),
            .lpart = try a(d, @min(R, 16) * vocab_share * 4),
            .amax = try a(d, 16 * 4 * 4),
            .amax_all = try a(d, world * 16 * 4 * 4),
            .tok_out = try a(d, 16 * 4),
            .rows = rows,
            .keys = keys,
            .had = try Buf.fromHost(d, had),
            .inv = try Buf.fromHost(d, inv),
            .scratch = try linear.Scratch.init(d, MAX_LIN_ROWS, @intCast(D), lin_z_floats),
        };
        if (dcp > 1) {
            const G: usize = dcp;
            const H: usize = heads;
            // decode windows exchange by all-gather (G x the bytes)
            // decode ウィンドウは all-gather で交換する (G 倍のバイト)
            const fan: usize = if (R <= 16) G else 1;
            out.dcp = dcp;
            out.qpack = try a(d, R * H * 576 * 2);
            out.qall = try a(d, G * R * H * 576 * 2);
            out.osend = try a(d, G * R * H * lw * 2);
            out.lsend = try a(d, G * R * H * 4);
            out.orecv = try a(d, fan * G * R * H * lw * 2);
            out.lrecv = try a(d, fan * G * R * H * 4);
            const xb = G * @min(R, small_rows) * H * (lw * 2 + 4) + 16;
            out.xsend = try a(d, xb);
            out.xrecv = try a(d, G * xb);
            out.cnt = try a(d, R * 4);
            out.cand = try a(d, G * @min(R, sel_rows) * cfg.index_topk * 8);
            out.dmine = try a(d, @min(R, sel_rows) * cfg.index_topk * 8);
            out.tk_state = try a(d, 16 * 1040);
            out.tk_cand = try Buf.alloc(d, @min(R, radix_min_rows - 1) * ((keys + dcp - 1) / dcp) * 8);
        }
        return out;
    }
};

/// The committed caches: a 576-wide latent row a position and layer, index keys on full-indexer layers.
/// 確定したキャッシュ: 位置と層ごとに 576 幅の latent 行、full indexer の層には index キー。
/// Bytes of a latent cache row: FP4 432, FP8 656, else bf16.
/// latent キャッシュの 1 行のバイト数: FP4 は 432、FP8 は 656、それ以外は bf16。
pub fn latRow(cfg: Config, kv8: bool, kv4: bool) usize {
    return if (kv4) 432 else if (kv8) 656 else cfg.latentWidth() * 2;
}

pub const State = struct {
    kc: []Buf,
    ic: []?Buf,
    pos: Buf, // int32 [1]: the window's first position / int32 [1]: ウィンドウの先頭位置

    /// `kv8`: FP8 rows, latent 656 / index 144 bytes (e4m3 codes, a scale, the RoPE part in bf16).
    /// `kv8`: FP8 の行。latent 656 / index 144 バイト (e4m3 コード、scale、bf16 の RoPE 部分)。
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cfg: Config, layers: []const model.Layer, capacity: usize, kv8: bool, kv4: bool) !State {
        const kc = try gpa.alloc(Buf, layers.len);
        const ic = try gpa.alloc(?Buf, layers.len);
        const lat_row: usize = latRow(cfg, kv8, kv4);
        const idx_row: usize = if (kv8) 144 else cfg.index_dim * 2;
        for (layers, 0..) |L, i| {
            kc[i] = try Buf.alloc(d, (capacity + 1) * lat_row);
            try kc[i].fill8(0, null);
            ic[i] = null;
            if (L.indexer != null) {
                ic[i] = try Buf.alloc(d, (capacity + 1) * idx_row);
                try ic[i].?.fill8(0, null);
            }
        }
        var pos = try Buf.alloc(d, 16);
        try pos.fill8(0, null);
        return .{ .kc = kc, .ic = ic, .pos = pos };
    }
};

/// A rank's tensor-parallel context and the kernels the forward launches.
/// ランクの tensor-parallel の文脈と、forward が起動するカーネル。
pub const Ctx = struct {
    cfg: Config,
    heads: u32,
    world: u32,
    rank: u32,
    stream: cuda.Stream,
    ops: ops_mod.Ops,
    x3: *const exl3.Kernels,
    glue: *const Glue,
    blas: *const cuda.cublas.Library,
    handle: cuda.cublas.Handle,
    group: *const cuda.tp.Group,
    b: *Buffers,
    ex: *const experts.Kernels,
    // the routed experts' scratch for these buffers (prompt or decode)
    // これらのバッファ用の routed expert の作業領域 (prompt 用か decode 用)
    xs: *const moe.Scratch,
    /// Called after each layer step with (layer, "attn" / "ffn" / "x", rows, device pointer, bytes), for tests that
    /// hash each step.
    /// 層の各段の後に (layer, "attn" / "ffn" / "x", 行数, デバイスポインタ, バイト数) で呼ばれる。各段をハッシュする
    /// テスト用。
    probe: ?*const fn (ctx: *anyopaque, layer: usize, what: []const u8, rows: u32, ptr: u64, bytes: usize) anyerror!void = null,
    probe_ctx: *anyopaque = undefined,
    /// DFlash2: each layer's tap column block by layer index, -1 none; empty: no taps.
    /// DFlash2: 層 index ごとの tap の列ブロック。-1 は無し、空なら tap 無し。
    tap_slot: []const i8 = &.{},
    n_taps: u32 = 0,
    /// DCP: the group's communicator (members rank / dcp, ordered by rank % dcp); null: DCP 1.
    /// DCP: グループの communicator (メンバーは rank / dcp が同じランク、rank % dcp の順)。null なら DCP 1。
    dgroup: ?*const cuda.tp.Group = null,
    /// TF_EXL3_PROMPT_EXPERTS=1: windows wider than MAX_LIN_ROWS take prompt_experts on the layers it supports (pe_ok
    /// by layer index).
    /// TF_EXL3_PROMPT_EXPERTS=1: MAX_LIN_ROWS 行より広いウィンドウは、対応する層 (層 index ごとの pe_ok) で
    /// prompt_experts を使う。
    pek: ?*const prompt_experts.Kernels = null,
    pes: ?*const prompt_experts.Scratch = null,
    pe_ok: []const bool = &.{},
    /// The prompt GEMMs' unpacked weights, shared by the SP parts, and this context's stream index in it (0 main, 1 the
    /// SP comm stream).
    /// prompt の GEMM 用に展開した重み (SP の part で共有) と、その中でのこの文脈のストリーム番号 (0 は main、1 は
    /// SP の comm ストリーム)。
    ucache: ?*unpack_cache.Cache = null,
    sid: u1 = 0,
    /// TF_GLM53_ROCE: decode windows' reductions and gathers over the RoCE one-shot runtimes (world; the DCP group's
    /// for its gathers); DECODE_BF16: the window's partials reduced as bf16.
    /// TF_GLM53_ROCE: decode ウィンドウの reduce と gather を RoCE の one-shot ランタイムで行う (world 用と、DCP
    /// グループの gather 用)。DECODE_BF16: ウィンドウの部分和を bf16 で reduce する。
    fast: ?*const roce.Runtime = null,
    dfast: ?*const roce.Runtime = null,
    decode_bf16: bool = false,
    /// Tests: emit each DCP selection's tok / cnt (TF_ZIG_SELECT=1).
    /// テスト用: DCP の選択ごとに tok / cnt を出す (TF_ZIG_SELECT=1)。
    probe_select: bool = false,
    /// Tests (Engine.selLogEnable): the picks of every index layer of a decode window, copied as the window's graph
    /// runs (a probe cannot see inside a replayed graph). Call k of `select` writes b.tok (this DCP member's local
    /// slots, ascending, [R, index_topk] int32) at sel_tok + k * sel_log_rows * index_topk * 4 and their counts at
    /// sel_cnt + k * sel_log_rows * 4. sel_n counts the calls of a capture.
    /// テスト用 (Engine.selLogEnable): decode ウィンドウの各 index 層の選択を、ウィンドウの graph の実行中に写す
    /// (再生する graph の中は probe から見えない)。`select` の k 回目の呼び出しは b.tok (この DCP メンバーの
    /// ローカルスロット、昇順、[R, index_topk] int32) を sel_tok + k * sel_log_rows * index_topk * 4 に、その数を
    /// sel_cnt + k * sel_log_rows * 4 に書く。sel_n は 1 回の capture での呼び出し回数を数える。
    sel_tok: ?u64 = null,
    sel_cnt: u64 = 0,
    sel_n: ?*u32 = null,
    /// PREFILL_REDUCE=ring (the default): SP prompt partials reduce-scattered as bf16 by NCCL (not an exact sum).
    /// PREFILL_REDUCE=ring (既定): SP の prompt の部分和を NCCL で bf16 のまま reduce-scatter する (正確な和ではない)。
    ring: bool = false,
    /// TF_GLM53_TOPK_WORDS (default on): `candidates` over int32 order words where the native kernels run.
    /// TF_GLM53_TOPK_WORDS (既定 on): native カーネルがある所では `candidates` を int32 の順序語で行う。
    topk_words: bool = true,
    /// TF_GLM53_TOPK_ANY (default on): spSelect's candidates from two passes in any order (0: five, column order).
    /// TF_GLM53_TOPK_ANY (既定 on): spSelect の候補を 2 pass で順不同に出す (0 なら 5 pass で列の順)。
    topk_any: bool = true,
    /// TF_GLM53_PE_F16: the prompt experts' down adds the slots into fp16 sums (red.add.f16x2) in sequence-parallel
    /// parts, read back by raddF16ToBf16 (other numbers than the fp32 sums: one fp16 rounding a slot).
    /// TF_GLM53_PE_F16: sequence-parallel の part で、prompt expert の down がスロットを fp16 の和に足す
    /// (red.add.f16x2)。raddF16ToBf16 で読む (fp32 の和とは数値が違う: スロットごとに fp16 の丸め 1 回)。
    pe_f16: bool = false,
    /// TF_GLM53_TOPK16 (default on): spSelect's any-order candidates from the words' top 16 bits (Glue.topkWordsAny16).
    /// TF_GLM53_TOPK16 (既定 on): spSelect の順不同の候補を語の上位 16 ビットから (Glue.topkWordsAny16)。
    topk16: bool = true,
    /// This context runs a sequence-parallel part (SpPart.main / .side): only there does spRsResidual read the
    /// partial from b.hpart (`laneSp`). The first part's buffers are the engine's prompt buffers, which chunks of
    /// 129..257 rows also run on without sequence parallelism; their partial must stay in b.part.
    /// この文脈は sequence-parallel の part (SpPart.main / .side) を実行する: spRsResidual が部分和を b.hpart から
    /// 読むのはその時だけ (`laneSp`)。最初の part のバッファは engine の prompt バッファで、129..257 行のチャンクも
    /// sequence parallel なしでそこで動くので、その部分和は b.part に置かなければならない。
    sp: bool = false,
    /// TF_GLM53_FUSE (default on): steps folded into their neighbours where the native kernels run: the packed
    /// queries written in place, a decode window's partial kept bf16 from the linear to residual_add.
    /// TF_GLM53_FUSE (既定 on): native カーネルがある所で段を隣の段にまとめる。packed query をその場で書き、decode
    /// ウィンドウの部分和を linear から residual_add まで bf16 のまま保つ。
    fuse: bool = true,
    /// TF_GLM53_XGROUP (default on): dcpExchange's two all-to-all exchanges of a prompt chunk in one NCCL group.
    /// TF_GLM53_XGROUP (既定 on): prompt チャンクの dcpExchange の 2 つの all-to-all を 1 つの NCCL group で行う。
    xgroup: bool = true,
    /// The layers carry the DCP partner's q_b / wk (model.Layer.peer_q_b): `peerQ`.
    /// 層が DCP の相手の q_b / wk を持つ (model.Layer.peer_q_b): `peerQ`。
    peer_q: bool = false,
    /// TF_GLM53_INDEX_FP8 (default off; changes the picks): the indexer's scores on the FP8 tensor-core product, the
    /// queries quantized to e4m3 (`scores`, `iqPrep`). Needs the native kernels.
    /// TF_GLM53_INDEX_FP8 (既定 off、選択が変わる): indexer の点数を FP8 の tensor core 積で計算し、query を e4m3 に
    /// 量子化する (`scores`、`iqPrep`)。native カーネルが必要。
    index_fp8: bool = false,

    /// emit with a label of its own in place of "L<layer>" (layer = maxInt: the probe writes `tag`).
    /// "L<layer>" の代わりに独自のラベルで emit する (layer = maxInt のとき probe は `tag` を書く)。
    fn emitTag(c: *const Ctx, tag: []const u8, what: []const u8, rows: u32, ptr: u64, bytes: usize) !void {
        _ = tag;
        try c.emit(std.math.maxInt(usize), what, rows, ptr, bytes);
    }

    fn emit(c: *const Ctx, layer: usize, what: []const u8, rows: u32, ptr: u64, bytes: usize) !void {
        if (c.probe) |p| {
            try c.stream.synchronize();
            try p(c.probe_ctx, layer, what, rows, ptr, bytes);
        }
    }

    /// The EXL3 linear: the row-invariant kernel up to 128 rows, else rotate the input, unpack the weights and GEMM.
    /// EXL3 の linear: 128 行までは行に依らないカーネル、それより多ければ入力の回転、重みの展開、GEMM。
    fn lin(c: *const Ctx, l: *const linear.Linear, x: u64, m: u32, out: u64, out_bf16: bool) !void {
        if (m <= MAX_LIN_ROWS) {
            return linear.call(c.x3, c.stream, &c.b.scratch, &.{l}, x, .bf16, m, &.{out}, &.{if (out_bf16) exl3.DType.bf16 else exl3.DType.f32});
        }
        try c.x3.rotIn(c.stream, x, .bf16, &.{l.suh.ptr}, &.{c.b.xh.ptr}, m, l.k, false);
        if (c.ucache) |uc| {
            // the unpacked W_q kept for the next calls (the chunk's other parts)
            // 展開した W_q を次の呼び出し (チャンクの他の part) のために残す
            const slot = try uc.get(c.x3, c.stream, c.sid, l);
            try c.ops.gemm(c.b.xh.ptr, slot.buf.ptr, c.b.had.ptr, l.svh.ptr, out, out_bf16, m, l.n, l.k, l.n);
            return uc.done(slot, c.stream, c.sid);
        }
        const st = l.strides();
        try c.x3.unpackW(c.stream, l.words.ptr, c.b.wq.ptr, l.k, l.n, l.k2, l.cb, st.k, st.nb);
        try c.ops.gemm(c.b.xh.ptr, c.b.wq.ptr, c.b.had.ptr, l.svh.ptr, out, out_bf16, m, l.n, l.k, l.n);
    }

    /// A sequence-parallel part under PREFILL_REDUCE=ring: the partial goes to the reduce-scatter as bf16 (b.hpart).
    /// With the native GEMM, o_proj (and a dense MLP's down projection) writes it there itself, and one pass makes
    /// bf16(experts + shared expert) of a MoE layer (Glue.raddToBf16), instead of separate passes over [R, D].
    /// PREFILL_REDUCE=ring の sequence-parallel の part: 部分和を bf16 で reduce-scatter に渡す (b.hpart)。native の
    /// GEMM では o_proj (と dense MLP の down projection) が直接そこに書き、MoE 層の bf16(expert + shared expert) は
    /// 1 pass で作る (Glue.raddToBf16)。[R, D] を何度も読む pass がいらない。
    fn laneSp(c: *const Ctx, R: u32) bool {
        return c.sp and c.fuse and c.ring and c.b.hpart != null and R > MAX_LIN_ROWS and c.probe == null;
    }

    /// A sequence-parallel part's last linear straight into b.hpart as bf16 (`laneSp`).
    /// sequence-parallel の part の最後の linear を、bf16 で直接 b.hpart に書く (`laneSp`)。
    fn linHpart(c: *const Ctx, l: *const linear.Linear, x: u64, R: u32) !void {
        return c.lin(l, x, R, c.b.hpart.?.ptr, true);
    }

    /// Linears over one input: one launch for groupable ones up to 128 rows, else one lin each.
    /// 同じ入力に対する複数の linear: 128 行までで、まとめられるなら 1 回の launch、そうでなければ 1 つずつ lin。
    fn lins(c: *const Ctx, ls: []const *const linear.Linear, x: u64, m: u32, outs: []const u64, bf16s: []const bool) !void {
        if (m <= MAX_LIN_ROWS and linear.groupable(ls)) {
            var dts: [exl3.max_jobs]exl3.DType = undefined;
            for (bf16s, 0..) |b16, i| dts[i] = if (b16) .bf16 else .f32;
            return linear.call(c.x3, c.stream, &c.b.scratch, ls, x, .bf16, m, outs, dts[0..ls.len]);
        }
        for (ls, outs, bf16s) |l, o, b16| try c.lin(l, x, m, o, b16);
    }

    /// The partials' exact sum for prompt chunks of 64 rows or more: column shares to their owners, summed in rank
    /// order, gathered back -> b.red [1, R, D]. Returns (pointer, world dimension) for residual_add.
    /// 64 行以上の prompt チャンクの部分和の正確な和: 列の取り分を持ち主に送り、ランク順に足し、集め直す
    /// -> b.red [1, R, D]。residual_add 用に (ポインタ, world の次元) を返す。
    fn gatherRs(c: *const Ctx, R: u32) !struct { ptr: u64, world: u32 } {
        const D: u32 = @intCast(c.cfg.hidden);
        const W = c.world;
        const q = D / W;
        const b = c.b;
        // each rank's column share together: send [W, R, q] from part's strides (q, D, 1)
        // ランクごとの列の取り分をまとめる: part のストライド (q, D, 1) から send [W, R, q]
        try c.glue.copy3(c.stream, b.part.ptr, b.send.ptr, W, R, q, q, D, true);
        const api = c.group.lib.api;
        const n: usize = @as(usize, R) * q;
        try c.group.lib.check(api.ncclGroupStart(), "ncclGroupStart");
        for (0..W) |p| {
            try c.group.lib.check(api.ncclSend(try b.send.at(p * n * 4), n, .f32, @intCast(p), c.group.comm, c.stream.handle), "ncclSend");
            try c.group.lib.check(api.ncclRecv(try b.recv.at(p * n * 4), n, .f32, @intCast(p), c.group.comm, c.stream.handle), "ncclRecv");
        }
        try c.group.lib.check(api.ncclGroupEnd(), "ncclGroupEnd");
        // mine = recv[0], then += recv[r] for r = 1 .. W - 1 (rank order)
        // mine = recv[0] に recv[r] (r = 1 .. W - 1) をランク順に足す
        try b.mine.copyFrom(0, b.recv.ptr, n * 4, c.stream.handle);
        for (1..W) |r| try c.glue.addF32(c.stream, b.mine.ptr, try b.recv.at(r * n * 4), n);
        try c.group.lib.check(api.ncclAllGather(b.mine.ptr, b.gath.ptr, n, .f32, c.group.comm, c.stream.handle), "ncclAllGather");
        // back to rows: red [R, W, q] from gath's strides (q, R * q, 1)
        // 行の並びに戻す: gath のストライド (q, R * q, 1) から red [R, W, q]
        try c.glue.copy3(c.stream, b.gath.ptr, b.red.ptr, R, W, q, q, @as(i64, R) * q, true);
        return .{ .ptr = b.red.ptr, .world = 1 };
    }

    /// Below 64 rows without RoCE: every rank's fp32 partial [W, R, D], for residual_add's sum in rank order.
    /// 64 行未満で RoCE なし: 全ランクの fp32 部分和 [W, R, D] を集め、residual_add がランク順に足す。
    fn gatherAll(c: *const Ctx, R: u32) !struct { ptr: u64, world: u32 } {
        const n: usize = @as(usize, R) * c.cfg.hidden;
        try c.group.lib.check(c.group.lib.api.ncclAllGather(c.b.part.ptr, c.b.gath.ptr, n, .f32, c.group.comm, c.stream.handle), "ncclAllGather");
        return .{ .ptr = c.b.gath.ptr, .world = c.world };
    }

    /// A decode window's partial stays bf16 from its last linear (b.dpart) through the RoCE reduce (b.dred) into
    /// residual_add: the bytes of fp32 -> bf16 -> reduce -> fp32 -> residual_add, without the two conversions (and the
    /// expert sum's add). Off under a probe, which hashes the fp32 sum.
    /// decode ウィンドウの部分和を、最後の linear (b.dpart) から RoCE の reduce (b.dred) を経て residual_add まで
    /// bf16 のまま保つ: fp32 -> bf16 -> reduce -> fp32 -> residual_add と同じバイトを、2 回の変換 (と expert の和の
    /// 加算) なしで得る。probe があるときは使わない (probe は fp32 の和をハッシュする)。
    fn laneBf16(c: *const Ctx, R: u32) bool {
        return c.fuse and c.fast != null and R <= small_rows and c.decode_bf16 and c.b.dpart != null and c.probe == null;
    }

    const Gathered = struct { ptr: u64, world: u32, bf16: bool = false };

    fn gather(c: *const Ctx, R: u32) !Gathered {
        if (c.fast) |f| if (R <= small_rows) {
            // the RoCE one-shot: the rank-order sum [1, R, D] (DECODE_BF16: of the bf16 partials, stored bf16, then
            // copied into the fp32 red)
            // RoCE の one-shot: ランク順の和 [1, R, D] (DECODE_BF16 なら bf16 の部分和の和を bf16 で受け、fp32 の
            // red に写す)
            const b = c.b;
            const n: u64 = @as(u64, R) * c.cfg.hidden;
            if (c.laneBf16(R)) {
                try f.allReduce(c.stream, b.dpart.?.ptr, b.dred.?.ptr, @intCast(n * 2), true);
                return .{ .ptr = b.dred.?.ptr, .world = 1, .bf16 = true };
            }
            if (c.decode_bf16 and b.dpart != null) {
                try c.glue.toBf16(c.stream, b.part.ptr, b.dpart.?.ptr, n);
                try f.allReduce(c.stream, b.dpart.?.ptr, b.dred.?.ptr, @intCast(n * 2), true);
                try c.glue.fromBf16(c.stream, b.dred.?.ptr, b.red.ptr, n);
            } else try f.allReduce(c.stream, b.part.ptr, b.red.ptr, @intCast(n * 4), false);
            return .{ .ptr = b.red.ptr, .world = 1 };
        };
        if (R > 16 and R >= 64) {
            const g = try c.gatherRs(R);
            return .{ .ptr = g.ptr, .world = g.world };
        }
        // up to 63 rows (decode windows, short prompt chunks): an all-gather, summed in rank order by the consumer (no
        // RoCE one-shot here: TF_GLM53_ROCE=0)
        // 63 行まで (decode ウィンドウ、短い prompt チャンク): all-gather し、使う側がランク順に足す (ここでは RoCE の
        // one-shot なし: TF_GLM53_ROCE=0)
        const g = try c.gatherAll(R);
        return .{ .ptr = g.ptr, .world = g.world };
    }

    /// cuBLAS strided batched GEMM (column-major, bf16 in / out, fp32 compute, the default algo).
    /// cuBLAS の strided batched GEMM (列優先、bf16 入出力、fp32 で計算、既定の algo)。
    fn bmm(c: *const Ctx, transa: c_int, m: c_int, n: c_int, k: c_int, a: u64, lda: c_int, sa: i64, bb: u64, ldb: c_int, sb: i64, cc: u64, ldc: c_int, sc: i64, batch: c_int) !void {
        const bf16: c_int = 14; // CUDA_R_16BF: cuBLAS's bf16 type / cuBLAS の bf16 型
        try c.blas.gemmStridedBatched(c.handle, .{ .transa = transa, .transb = 0, .m = m, .n = n, .k = k, .a = a, .a_type = bf16, .lda = lda, .stride_a = sa, .b = bb, .b_type = bf16, .ldb = ldb, .stride_b = sb, .c = cc, .c_type = bf16, .ldc = ldc, .stride_c = sc, .batch = batch, .compute = 68, .algo = 99 });
    }

    /// Attention for R rows at `pos` (a prompt chunk or a decode window): the latent and index key rows written, the
    /// indexer's selection past index_topk, this rank's fp32 partial in b.part[:R]. Up to 16 rows: 8 chunks of 256
    /// keys, then a merge; wider: one pass of 2048 keys. Up to 128 rows the absorb / expand kernels, wider the batched
    /// GEMMs.
    /// `pos` からの R 行 (prompt チャンクか decode ウィンドウ) の attention: latent と index キーの行を書き、キーが
    /// index_topk を超えれば indexer で選び、このランクの fp32 部分和を b.part[:R] に置く。16 行までは 256 キーの
    /// チャンク 8 つの後にマージ、それより多ければ 2048 キーを 1 pass。128 行までは absorb / expand カーネル、それより
    /// 多ければ batched GEMM。
    fn attention(c: *const Ctx, L: *const model.Layer, R: u32, kc: u64, ic: ?u64, pos: u64, t: ?u32) !void {
        const cfg = c.cfg;
        const b = c.b;
        const H = c.heads;
        const D: u32 = @intCast(cfg.hidden);
        try c.lins(&.{ &L.q_a, &L.kv_a }, b.normed.ptr, R, &.{ b.qa.ptr, b.kva.ptr }, &.{ true, true });
        try c.ops.rmsnorm(b.qa.ptr, @intCast(cfg.q_lora), L.q_a_norm.ptr, b.qn.ptr, @intCast(cfg.q_lora), R, @intCast(cfg.q_lora), cfg.eps);
        try c.ops.kvWrite(b.kva.ptr, L.kv_a.n, L.kv_a_norm.ptr, kc, pos, b.inv.ptr, cfg.eps, R);
        var q_done = false;
        if (L.indexer) |ix| {
            try c.ops.router(b.normed.ptr, D, ix.wk.ptr, b.rpart.ptr, b.ik.ptr, R, D, @intCast(cfg.index_dim));
            try c.ops.ikWrite(b.ik.ptr, ix.k_norm_w.ptr, ix.k_norm_b.ptr, b.inv.ptr, ic.?, pos, R);
            if (t) |T| {
                try c.ops.router(b.normed.ptr, D, ix.weights_proj.ptr, b.rpart.ptr, b.iw.ptr, R, D, @intCast(cfg.index_heads));
                try c.lins(&.{ &ix.wq_b, &L.q_b }, b.qn.ptr, R, &.{ b.iq.ptr, b.q.ptr }, &.{ true, true });
                q_done = true;
                try c.ops.iqRope(b.iq.ptr, b.inv.ptr, pos, R);
                try c.iqPrep(0, R);
                try c.select(R, T, ic.?, pos);
            }
        }
        try c.coreQ(L, R, pos, q_done);
        if (c.dgroup) |dg| try c.attentionDcp(dg, R, kc, pos) else if (R <= small_rows) {
            // 256 keys a chunk, 32 a tile
            // 1 チャンク 256 キー、1 タイル 32 キー
            const nch: u32 = @intCast(cfg.index_topk / 256);
            try c.ops.attnChunks(b.qlat.ptr, b.qrot.ptr, kc, b.tok.ptr, pos, b.po.ptr, b.pm.ptr, b.pl.ptr, R, H, nch, 256, 32, false);
            try c.ops.merge(b.po.ptr, b.pm.ptr, b.pl.ptr, b.ol.ptr, b.dummy.ptr, R, H, nch);
        } else try c.ops.attnChunks(b.qlat.ptr, b.qrot.ptr, kc, b.tok.ptr, pos, b.ol.ptr, b.pm.ptr, b.pl.ptr, R, H, 1, 2048, 64, true);
        try c.coreOut(L, R);
    }

    /// The attention's query step: q_b (unless already grouped with wq_b), the absorbed latent queries and the RoPE
    /// part (b.qlat, b.qrot); DCP: packed [R, H, 576] into b.qpack.
    /// attention の query の段: q_b (wq_b とまとめて済んでいなければ)、吸収済みの latent query と RoPE 部分 (b.qlat、
    /// b.qrot)。DCP では [R, H, 576] に pack して b.qpack へ。
    fn coreQ(c: *const Ctx, L: *const model.Layer, R: u32, pos: u64, q_done: bool) !void {
        const cfg = c.cfg;
        const b = c.b;
        const H = c.heads;
        const lw: u32 = @intCast(cfg.kv_lora);
        const nope: u32 = @intCast(cfg.qk_nope);
        const qd: u32 = @intCast(cfg.qk_nope + cfg.qk_rope);
        if (!q_done) try c.lin(&L.q_b, b.qn.ptr, R, b.q.ptr, true);
        if (c.peerQ(R)) {
            // both members' packed queries straight into b.qall [2, R, H, 576]: this rank's and, from the partner's
            // q_b / wk over the same q_a rows, the partner's: the bytes it would have sent (no collective)
            // 両メンバーの packed query を直接 b.qall [2, R, H, 576] に書く: このランクの分と、同じ q_a の行に相手の
            // q_b / wk を掛けた相手の分。相手が送るはずのバイトと同じ (collective なし)
            const per: u64 = @as(u64, R) * H * 576 * 2;
            const me: u64 = c.dgroup.?.rank;
            try c.packQ(L.wk.ptr, b.qall.?.ptr + me * per, R, pos);
            try c.lin(&L.peer_q_b.?, b.qn.ptr, R, b.q.ptr, true);
            return c.packQ(L.peer_wk.?.ptr, b.qall.?.ptr + (1 - me) * per, R, pos);
        }
        if (c.fuse and c.dgroup != null) return c.packQ(L.wk.ptr, b.qpack.?.ptr, R, pos);
        if (R > MAX_LIN_ROWS) {
            // per head qlat [R, H, lw] = q[:, h, :nope] @ wk[h], through tmp [H, R, lw]
            // ヘッドごとに qlat [R, H, lw] = q[:, h, :nope] @ wk[h]。tmp [H, R, lw] を経由する
            try c.bmm(0, @intCast(lw), @intCast(R), @intCast(nope), L.wk.ptr, @intCast(lw), @as(i64, nope) * lw, b.q.ptr, @intCast(H * qd), qd, b.tmp.ptr, @intCast(lw), @as(i64, R) * lw, @intCast(H));
            try c.glue.scatter3(c.stream, b.tmp.ptr, b.qlat.ptr, H, R, lw, lw, @as(i64, H) * lw, false);
            try c.ops.qrope(b.q.ptr, b.inv.ptr, b.qrot.ptr, pos, R, H);
        } else try c.ops.absorb(b.q.ptr, L.wk.ptr, b.inv.ptr, b.qlat.ptr, b.qrot.ptr, pos, R, H);
        if (c.dgroup != null) {
            const qp = b.qpack.?.ptr;
            try c.glue.scatter3(c.stream, b.qlat.ptr, qp, 1, R * H, lw, 0, 576, false);
            try c.glue.scatter3(c.stream, b.qrot.ptr, qp + lw * 2, 1, R * H, 64, 0, 576, false);
        }
    }

    /// A prompt chunk under DCP 2 whose partner's queries are computed here (model.Layer.peer_q_b) instead of
    /// gathered: `coreQ` fills b.qall, `dcpGatherQ` has nothing to do. Decode windows keep the RoCE gather (cheaper
    /// than a second q_b there).
    /// DCP 2 の prompt チャンクで、相手の query を集めずにここで計算するか (model.Layer.peer_q_b): `coreQ` が b.qall
    /// を埋め、`dcpGatherQ` は何もしない。decode ウィンドウは RoCE の gather のまま (そこでは 2 回目の q_b より安い)。
    fn peerQ(c: *const Ctx, R: u32) bool {
        const dg = c.dgroup orelse return false;
        return c.fuse and c.peer_q and dg.world == 2 and R > small_rows;
    }

    /// The absorbed queries of b.q (one member's heads, by its `wk`) with their RoPE part, packed [R, H, 576] at `qp`.
    /// b.q の吸収済み query (1 メンバーのヘッド、その `wk` で) と RoPE 部分を、[R, H, 576] に pack して `qp` に書く。
    fn packQ(c: *const Ctx, wk: u64, qp: u64, R: u32, pos: u64) !void {
        const cfg = c.cfg;
        const b = c.b;
        const H = c.heads;
        const lw: u32 = @intCast(cfg.kv_lora);
        const nope: u32 = @intCast(cfg.qk_nope);
        const qd: u32 = @intCast(cfg.qk_nope + cfg.qk_rope);
        if (R > MAX_LIN_ROWS) {
            // straight into the packed [R, H, 576] (ldc H 576, stride_c 576): the same bits as through tmp [H, R, lw]
            // and a scatter (glm53/tools/native/cublas_layout.cu), without the scatter
            // pack した [R, H, 576] へ直接 (ldc H 576、stride_c 576): tmp [H, R, lw] と scatter を経るのと同じビットで、
            // scatter がない (glm53/tools/native/cublas_layout.cu)
            try c.bmm(0, @intCast(lw), @intCast(R), @intCast(nope), wk, @intCast(lw), @as(i64, nope) * lw, b.q.ptr, @intCast(H * qd), qd, qp, @intCast(H * 576), 576, @intCast(H));
            return c.ops.qropePacked(b.q.ptr, b.inv.ptr, qp, pos, R, H);
        }
        return c.ops.absorbPacked(b.q.ptr, wk, b.inv.ptr, qp, pos, R, H);
    }

    /// The attention's output step: the value expansion (b.ol -> b.o) and o_proj -> this rank's fp32 partial
    /// b.part[:R].
    /// attention の出力の段: value の展開 (b.ol -> b.o) と o_proj -> このランクの fp32 部分和 b.part[:R]。
    fn coreOut(c: *const Ctx, L: *const model.Layer, R: u32) !void {
        const cfg = c.cfg;
        const b = c.b;
        const H = c.heads;
        const lw: u32 = @intCast(cfg.kv_lora);
        const vd: u32 = @intCast(cfg.v_head);
        if (R > MAX_LIN_ROWS) {
            // per head o [R, H, v] = ol[:, h] @ wv[h]^T, straight into o (ldc H v, stride_c v; the same bits as through
            // tmp [H, R, v] and a scatter)
            // ヘッドごとに o [R, H, v] = ol[:, h] @ wv[h]^T。o へ直接書く (ldc H v、stride_c v。tmp [H, R, v] と scatter を
            // 経るのと同じビット)
            try c.bmm(1, @intCast(vd), @intCast(R), @intCast(lw), L.wv.ptr, @intCast(lw), @as(i64, vd) * lw, b.ol.ptr, @intCast(H * lw), lw, b.o.ptr, @intCast(H * vd), vd, @intCast(H));
        } else try c.ops.expand(b.ol.ptr, L.wv.ptr, b.o.ptr, R, H);
        if (c.laneBf16(R)) return c.lin(&L.o_proj, b.o.ptr, R, b.dpart.?.ptr, true);
        if (c.laneSp(R)) return c.linHpart(&L.o_proj, b.o.ptr, R);
        try c.lin(&L.o_proj, b.o.ptr, R, b.part.ptr, false);
    }

    /// A decode window's exchange block of one member: its normalized partials for every member's heads
    /// [G, R, H, 512] bf16, then their log-sum-exps [G, R, H] fp32 (padded to 16 bytes).
    /// decode ウィンドウでの 1 メンバーの交換ブロック: 全メンバーのヘッドに対する正規化済みの部分結果
    /// [G, R, H, 512] bf16、続いてその log-sum-exp [G, R, H] fp32 (16 バイトまで詰め物)。
    fn xBytes(c: *const Ctx, G: u32, R: u32) u32 {
        return G * R * c.heads * 1024 + (G * R * c.heads * 4 + 15) / 16 * 16;
    }

    /// Attention under DCP: this rank's absorbed queries (b.qpack) gathered over the group, every member's heads over
    /// this rank's keys, the chunk partials merged into each head's normalized output and log-sum-exp for its owner,
    /// exchanged (decode windows: all-gather, prompt chunks: all-to-all) and merged in rank order into b.ol. The steps
    /// between the collectives are dcpAttend / dcpCombine (SP runs the collectives on its comm stream).
    /// DCP の attention: このランクの吸収済み query (b.qpack) をグループで集め、全メンバーのヘッドをこのランクのキーで
    /// 計算し、チャンクの部分結果を各ヘッドの持ち主向けに正規化済みの出力と log-sum-exp にまとめ、交換し (decode
    /// ウィンドウは all-gather、prompt チャンクは all-to-all)、ランク順にマージして b.ol に置く。collective の間の段は
    /// dcpAttend / dcpCombine (SP では collective を comm ストリームで実行する)。
    fn attentionDcp(c: *const Ctx, dg: *const cuda.tp.Group, R: u32, kc: u64, pos: u64) !void {
        try c.dcpGatherQ(dg, R);
        try c.dcpAttend(dg, R, kc, pos);
        try c.dcpExchange(dg, R);
        try c.dcpCombine(dg, R);
    }

    fn dcpGatherQ(c: *const Ctx, dg: *const cuda.tp.Group, R: u32) !void {
        const b = c.b;
        if (c.peerQ(R)) return; // coreQ wrote both members' queries / coreQ が両メンバーの query を書いた
        if (c.dfast) |f| if (R <= small_rows) return f.allGather(c.stream, b.qpack.?.ptr, b.qall.?.ptr, R * c.heads * 576 * 2);
        try dg.lib.check(dg.lib.api.ncclAllGather(b.qpack.?.ptr, b.qall.?.ptr, @as(usize, R) * c.heads * 576, .bf16, dg.comm, c.stream.handle), "ncclAllGather");
    }

    fn dcpAttend(c: *const Ctx, dg: *const cuda.tp.Group, R: u32, kc: u64, pos: u64) !void {
        const b = c.b;
        const H = c.heads;
        const G: u32 = dg.world;
        const small = R <= small_rows;
        // keys a chunk and a tile: (256, 32) up to 16 rows, else (2048, 64), one chunk
        // 1 チャンクと 1 タイルのキー数: 16 行までは (256, 32)、それより多ければ (2048, 64) で 1 チャンク
        const chk: u32 = if (small) 256 else 2048;
        const kt: u32 = if (small) 32 else 64;
        const nch: u32 = @intCast(c.cfg.index_topk / chk);
        if (!small and nch == 1 and c.fuse and c.ops.canAttnDirect(G, H)) {
            // one chunk: the kernel normalizes its partial itself (no fp32 partials, no merge pass)
            // 1 チャンク: カーネルが自分で部分結果を正規化する (fp32 の部分結果もマージの pass もない)
            return c.ops.attnDcpDirect(b.qall.?.ptr, kc, b.tok.ptr, b.cnt.?.ptr, pos, b.po.ptr, b.pm.ptr, b.pl.ptr, b.osend.?.ptr, b.lsend.?.ptr, R, H, G);
        }
        try c.ops.attnDcpAll(b.qall.?.ptr, kc, b.tok.ptr, b.cnt.?.ptr, pos, b.po.ptr, b.pm.ptr, b.pl.ptr, R, H, G, nch, chk, kt);
        const os = if (small) b.xsend.?.ptr else b.osend.?.ptr;
        const ls = if (small) os + @as(u64, G) * R * H * 1024 else b.lsend.?.ptr;
        try c.ops.mergeLse(b.po.ptr, b.pm.ptr, b.pl.ptr, os, ls, R, G * H, H, nch);
    }

    fn dcpExchange(c: *const Ctx, dg: *const cuda.tp.Group, R: u32) !void {
        const b = c.b;
        const G: u32 = dg.world;
        const per_o: usize = @as(usize, R) * c.heads * 512;
        const per_l: usize = @as(usize, R) * c.heads;
        const lib = dg.lib;
        if (R <= small_rows) {
            const xb = c.xBytes(G, R);
            // RoCE's one-shot holds a block of up to roce.max_bytes: DCP 4's 16-row windows pass it, NCCL takes those
            // RoCE の one-shot が持てるのは roce.max_bytes まで: DCP 4 の 16 行のウィンドウはそれを超えるので NCCL で
            if (c.dfast) |f| if (xb <= roce.max_bytes) return f.allGather(c.stream, b.xsend.?.ptr, b.xrecv.?.ptr, xb);
            try lib.check(lib.api.ncclAllGather(b.xsend.?.ptr, b.xrecv.?.ptr, xb, .u8, dg.comm, c.stream.handle), "ncclAllGather");
        } else {
            // the outputs and their log-sum-exps to each owner: one NCCL group (TF_GLM53_XGROUP; else a group each)
            // 出力と log-sum-exp を各持ち主へ: 1 つの NCCL group で (TF_GLM53_XGROUP。無効ならそれぞれ別の group)
            if (c.xgroup) return dg.allToAll2(b.osend.?.ptr, b.orecv.?.ptr, per_o, .bf16, b.lsend.?.ptr, b.lrecv.?.ptr, per_l, .f32, c.stream.handle);
            try dg.allToAll(b.osend.?.ptr, b.orecv.?.ptr, per_o, .bf16, c.stream.handle);
            try dg.allToAll(b.lsend.?.ptr, b.lrecv.?.ptr, per_l, .f32, c.stream.handle);
        }
    }

    fn dcpCombine(c: *const Ctx, dg: *const cuda.tp.Group, R: u32) !void {
        const b = c.b;
        const H = c.heads;
        const G: u32 = dg.world;
        const per_o: usize = @as(usize, R) * H * 512;
        const per_l: usize = @as(usize, R) * H;
        if (R <= small_rows) {
            const me: u64 = dg.rank;
            const xb = c.xBytes(G, R);
            const xr = b.xrecv.?.ptr;
            try c.ops.dcpCombine(xr + me * per_o * 2, xr + @as(u64, G) * per_o * 2 + me * per_l * 4, b.ol.ptr, R, xb / 2, xb / 4, H, G);
        } else try c.ops.dcpCombine(b.orecv.?.ptr, b.lrecv.?.ptr, b.ol.ptr, R, @intCast(per_o), @intCast(per_l), H, G);
    }

    /// The indexer's scores of rows r0 .. r0 + n - 1 of b.iq / b.iw (TF_GLM53_INDEX_FP8: of their e4m3 codes in b.iq8).
    /// b.iq / b.iw の行 r0 .. r0 + n - 1 の indexer の点数 (TF_GLM53_INDEX_FP8 なら b.iq8 の e4m3 コードから)。
    fn scores(c: *const Ctx, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32, pack: bool) !void {
        const b = c.b;
        const nh: usize = c.cfg.index_heads;
        const idd: usize = c.cfg.index_dim;
        const w = try b.iw.at(@as(usize, r0) * nh * 4);
        if (c.index_fp8) return c.ops.indexScores8(try b.iq8.at(@as(usize, r0) * nh * idd), try b.iqs.at(@as(usize, r0) * nh * 4), w, ic, pos, out, n, t, r0, pack);
        return c.ops.indexScores(try b.iq.at(@as(usize, r0) * nh * idd * 2), w, ic, pos, out, n, t, r0, pack);
    }

    /// After iqRope of rows r0 .. r0 + n - 1 of b.iq: their e4m3 codes for `scores` (TF_GLM53_INDEX_FP8 only).
    /// b.iq の行 r0 .. r0 + n - 1 の iqRope の後: `scores` 用の e4m3 コード (TF_GLM53_INDEX_FP8 のときだけ)。
    fn iqPrep(c: *const Ctx, r0: u32, n: u32) !void {
        if (!c.index_fp8) return;
        const b = c.b;
        const nh: usize = c.cfg.index_heads;
        const idd: usize = c.cfg.index_dim;
        try c.ops.iqFp8(try b.iq.at(@as(usize, r0) * nh * idd * 2), try b.iq8.at(@as(usize, r0) * nh * idd), try b.iqs.at(@as(usize, r0) * nh * 4), n);
    }

    /// This rank's candidates under DCP for a block of prompt rows: each row's index_topk best packed keys of this
    /// rank's slots, in column order, into `out` [n, index_topk] int64. With the native kernels the scores are int32
    /// order words and tf_glm53_topk_words packs the picks (the same bytes; TF_GLM53_TOPK_WORDS=0: packed scores and
    /// tf_glm53_topk_values). `ordered` false (the caller's merge takes the keys' exact order): the same keys in any
    /// order, from two passes.
    /// DCP で prompt の行ブロックに対するこのランクの候補: 各行について、このランクのスロットの packed キーの上位
    /// index_topk 個を列の順で `out` [n, index_topk] int64 に書く。native カーネルでは点数を int32 の順序語にし、
    /// tf_glm53_topk_words が選んだものを pack する (同じバイト。TF_GLM53_TOPK_WORDS=0 なら packed の点数と
    /// tf_glm53_topk_values)。`ordered` が false (呼び出し側のマージがキーの正確な順を取る) なら、同じキーを順不同で
    /// 2 pass で出す。
    fn candidates(c: *const Ctx, ic: u64, pos: u64, out: u64, n: u32, Tl: u32, r0: u32, ordered: bool) !void {
        const b = c.b;
        const K: u32 = @intCast(c.cfg.index_topk);
        const words = c.topk_words and (c.index_fp8 or c.ops.wordScores());
        if (!words) {
            try c.scores(ic, pos, b.sc.ptr, n, Tl, r0, true);
            return c.glue.topkValues(c.stream, b.sc.ptr, out, n, Tl, K);
        }
        const ord = ordered or !c.topk_any;
        const iq = try b.iq.at(@as(usize, r0) * c.cfg.index_heads * c.cfg.index_dim * 2);
        const iw = try b.iw.at(@as(usize, r0) * c.cfg.index_heads * 4);
        // TF_GLM53_TOPK16, rows of any16_min_keys slots or more: the words' top 16 bits beside them, the any-order
        // picks reading those (the same keys, half the bytes or less)
        // TF_GLM53_TOPK16 で any16_min_keys スロット以上の行: 語の隣に上位 16 ビットも書き、順不同の選択はそれを読む
        // (同じキーで、読むバイトは半分以下)
        if (!c.index_fp8 and !ord and c.topk16 and Tl >= any16_min_keys) {
            try c.ops.indexScoresWords16(iq, iw, ic, pos, b.sc.ptr, n, Tl, r0);
            return c.glue.topkWordsAny16(c.stream, b.sc.ptr, b.sc.ptr + ops_mod.Ops.prefix16Offset(n, Tl), out, pos, n, Tl, K, r0, c.ops.dcp, c.ops.drank);
        }
        if (c.index_fp8) try c.scores(ic, pos, b.sc.ptr, n, Tl, r0, false) else try c.ops.indexScoresWords(iq, iw, ic, pos, b.sc.ptr, n, Tl, r0);
        try c.glue.topkWords(c.stream, b.sc.ptr, out, pos, n, Tl, K, r0, c.ops.dcp, c.ops.drank, ord);
    }

    /// The indexer's selection under DCP: each rank scores its own slots (T / G of them), keeps its top index_topk
    /// packed keys, the group's candidates gathered in member order, the global top index_topk of them, this rank's
    /// share as ascending local slots into b.tok and their count into b.cnt.
    /// DCP での indexer の選択: 各ランクは自分のスロット (T / G 個) に点数を付けて packed キーの上位 index_topk 個を
    /// 残し、グループの候補をメンバー順に集め、その中の全体の上位 index_topk 個のうちこのランクの分を、昇順の
    /// ローカルスロットとして b.tok に、その数を b.cnt に書く。
    fn selectDcp(c: *const Ctx, dg: *const cuda.tp.Group, R: u32, T: u32, ic: u64, pos: u64) !void {
        const b = c.b;
        const K: u32 = @intCast(c.cfg.index_topk);
        const G: u32 = dg.world;
        const Tl = (T + G - 1) / G;
        if (T > b.keys) return error.TooManyKeys;
        var r0: u32 = 0;
        while (r0 < R) : (r0 += sel_rows) {
            const n = @min(sel_rows, R - r0);
            // below radix_min_rows rows any order (the merge is exact); from there the radix select in column order,
            // which the merge's ties follow
            // radix_min_rows 行未満は順不同 (マージが正確な順を取る)。それ以上は列の順の radix select で、マージの
            // 同点はその順に従う
            if (n < radix_min_rows and b.tk_state != null) {
                try c.scores(ic, pos, b.sc.ptr, n, Tl, r0, true);
                try c.glue.topkValuesWide(c.stream, b.sc.ptr, b.dmine.?.ptr, n, Tl, K, b.tk_state.?.ptr, b.tk_cand.?.ptr);
            } else try c.candidates(ic, pos, b.dmine.?.ptr, n, Tl, r0, true);
            // the candidates gathered: decode windows over the DCP group's RoCE one-shot, prompt chunks over NCCL (the
            // same bytes in [G, n, K] order)
            // 候補を集める: decode ウィンドウは DCP グループの RoCE one-shot、prompt チャンクは NCCL (どちらも
            // [G, n, K] の順の同じバイト)
            const lib = dg.lib;
            const cb: usize = @as(usize, n) * K * 8;
            if (c.dfast != null and n <= small_rows and cb <= roce.max_bytes) {
                try c.dfast.?.allGather(c.stream, b.dmine.?.ptr, b.cand.?.ptr, @intCast(cb));
            } else try lib.check(lib.api.ncclAllGather(b.dmine.?.ptr, b.cand.?.ptr, @as(usize, n) * K, .i64, dg.comm, c.stream.handle), "ncclAllGather");
            try c.glue.dcpSelect(c.stream, b.cand.?.ptr, try b.tok.at(@as(usize, r0) * K * 4), try b.cnt.?.at(@as(usize, r0) * 4), n, n, K, G, dg.rank, n >= radix_min_rows);
        }
    }

    /// The indexer's selection (DCP 1): each row's top index_topk keys into b.tok (ascending), in blocks of 128 rows;
    /// the radix path (int32 order words + selectKeys) for blocks of 32+ rows over index_topk+ keys.
    /// indexer の選択 (DCP 1): 各行の上位 index_topk 個のキーを b.tok に昇順で、128 行ずつ。32 行以上でキーが
    /// index_topk 個以上のブロックは radix の経路 (int32 の順序語 + selectKeys)。
    fn select(c: *const Ctx, R: u32, T: u32, ic: u64, pos: u64) !void {
        if (c.dgroup) |dg| {
            if (c.probe_select) {
                const nh: usize = c.cfg.index_heads;
                try c.emitTag("S", "iq", R, c.b.iq.ptr, @as(usize, R) * nh * c.cfg.index_dim * 2);
                try c.emitTag("S", "iw", R, c.b.iw.ptr, @as(usize, R) * nh * 4);
                const Tl = (T + dg.world - 1) / dg.world;
                try c.emitTag("S", "ic", Tl, ic, @as(usize, Tl) * (if (c.ops.kv8) @as(usize, 144) else c.cfg.index_dim * 2));
            }
            try c.selectDcp(dg, R, T, ic, pos);
            if (c.probe_select) {
                try c.emitTag("S", "tok", R, c.b.tok.ptr, @as(usize, R) * c.cfg.index_topk * 4);
                try c.emitTag("S", "cnt", R, c.b.cnt.?.ptr, @as(usize, R) * 4);
            }
            if (c.sel_tok) |log| {
                if (R > sel_log_rows) return error.WindowTooWide;
                const k: usize = c.sel_n.?.*;
                c.sel_n.?.* += 1;
                const K: usize = c.cfg.index_topk;
                const d = c.stream.d;
                try d.check(d.api.cuMemcpyDtoDAsync_v2(log + k * sel_log_rows * K * 4, c.b.tok.ptr, @as(usize, R) * K * 4, c.stream.handle), "cuMemcpyDtoDAsync");
                try d.check(d.api.cuMemcpyDtoDAsync_v2(c.sel_cnt + k * sel_log_rows * 4, c.b.cnt.?.ptr, @as(usize, R) * 4, c.stream.handle), "cuMemcpyDtoDAsync");
            }
            return;
        }
        const b = c.b;
        const K: u32 = @intCast(c.cfg.index_topk);
        if (T > b.keys) return error.TooManyKeys;
        var r0: u32 = 0;
        while (r0 < R) : (r0 += sel_rows) {
            const n = @min(sel_rows, R - r0);
            const radix = n >= 32 and T >= K;
            if (!radix) {
                try c.scores(ic, pos, b.sc.ptr, n, T, r0, true);
                try c.glue.topkKeys(c.stream, b.sc.ptr, try b.tok.at(@as(usize, r0) * K * 4), n, T, K);
                continue;
            }
            try c.scores(ic, pos, b.sc.ptr, n, T, r0, false);
            try c.ops.selectKeys(b.sc.ptr, try b.tok.at(@as(usize, r0) * K * 4), n, T);
        }
    }

    /// One layer for a decode window: the same steps as a prompt chunk (the row count decides the kernels).
    /// decode ウィンドウの 1 層: prompt チャンクと同じ段 (行数でカーネルが決まる)。
    pub fn layerDecode(c: *const Ctx, L: *const model.Layer, R: u32, kc: u64, ic: ?u64, pos: u64, t: ?u32) !void {
        return c.layerPrompt(L, R, kc, ic, pos, t);
    }

    /// The head's first step for rows `row0 .. row0 + n - 1` of b.x: final RMSNorm and this rank's vocabulary share of
    /// the logits into b.lpart [n, V/W] (fp32). The pick across ranks is `pickArgmax`.
    /// head の最初の段 (b.x の行 `row0 .. row0 + n - 1`): 最後の RMSNorm と、このランクの語彙の取り分の logit を
    /// b.lpart [n, V/W] (fp32) に。ランクをまたいだ選択は `pickArgmax`。
    pub fn headLogits(c: *const Ctx, norm: u64, lm_head: u64, row0: u32, n: u32, vocab_share: u32) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        try c.ops.rmsnorm(try b.x.at(@as(usize, row0) * D * 2), D, norm, b.fnormed.ptr, D, n, D, c.cfg.eps);
        try c.ops.router(b.fnormed.ptr, D, lm_head, b.rpart.ptr, b.lpart.ptr, n, D, vocab_share);
    }

    /// The head's argmax for one row: this rank's first maximum over its share, the 16-byte rows all-gathered, the
    /// lowest rank among equal maxima (the full argmax's own choice) -> the token.
    /// 1 行に対する head の argmax: このランクの取り分で最初の最大値、16 バイトの行を all-gather し、最大値が同じなら
    /// 最も小さいランク (全体の argmax と同じ選び方) -> トークン。
    pub fn pickArgmax(c: *const Ctx, gpa: std.mem.Allocator, vocab_share: u32) !u32 {
        const host = try gpa.alloc(f32, vocab_share);
        defer gpa.free(host);
        try c.stream.synchronize();
        try c.b.lpart.download(0, std.mem.sliceAsBytes(host));
        var best: usize = 0;
        for (host, 0..) |v, i| {
            if (std.math.isNan(v)) {
                best = i;
                break;
            }
            if (v > host[best]) best = i;
        }
        const mine = [4]f32{ host[best], @floatFromInt(best + c.rank * vocab_share), 0, 0 };
        try c.b.mine.upload(0, std.mem.asBytes(&mine));
        try c.group.lib.check(c.group.lib.api.ncclAllGather(c.b.mine.ptr, c.b.gath.ptr, 4, .f32, c.group.comm, c.stream.handle), "ncclAllGather");
        try c.stream.synchronize();
        var all: [64]f32 = undefined;
        try c.b.gath.download(0, std.mem.sliceAsBytes(all[0 .. 4 * c.world]));
        var r_best: usize = 0;
        for (0..c.world) |r| if (all[4 * r] > all[4 * r_best]) {
            r_best = r;
        };
        return @intFromFloat(all[4 * r_best + 1]);
    }

    /// One decode round of one row with no host work (CUDA-graph capturable): `windowStep` with R = 1.
    /// 1 行の decode 1 回分、ホストの処理なし (CUDA graph に capture できる): R = 1 の `windowStep`。
    pub fn decodeStep(c: *const Ctx, embed_table: u64, layers: []const model.Layer, kc: []const Buf, ic: []const ?Buf, pos: u64, t: ?u32, norm: u64, lm_head: u64, vocab_share: u32) !void {
        return c.windowStep(1, embed_table, layers, kc, ic, pos, t, norm, lm_head, vocab_share);
    }

    /// One verify window of R rows with the argmax head and no host work (CUDA-graph capturable): embed b.ids[0..R],
    /// every layer at `pos` with key range `t`, the head on all R rows, each row's argmax across ranks into
    /// b.ids[0..R] and b.tok_out[0..R].
    /// R 行の verify ウィンドウ 1 つ (argmax の head、ホストの処理なし、CUDA graph に capture できる): b.ids[0..R] を
    /// 埋め込み、全層を `pos` からキー範囲 `t` で、R 行すべてに head、各行のランクをまたいだ argmax を b.ids[0..R] と
    /// b.tok_out[0..R] に書く。
    pub fn windowStep(c: *const Ctx, R: u32, embed_table: u64, layers: []const model.Layer, kc: []const Buf, ic: []const ?Buf, pos: u64, t: ?u32, norm: u64, lm_head: u64, vocab_share: u32) !void {
        const b = c.b;
        if (R > 16) return error.WindowTooWide;
        try c.embed(embed_table, R);
        for (layers, 0..) |*L, i| try c.layerDecode(L, R, kc[i].ptr, if (ic[i]) |x| x.ptr else null, pos, t);
        try c.headLogits(norm, lm_head, 0, R, vocab_share);
        try c.glue.rowArgmax(c.stream, b.lpart.ptr, b.amax.ptr, R, vocab_share, c.rank * vocab_share);
        if (c.fast) |f| {
            try f.allGather(c.stream, b.amax.ptr, b.amax_all.ptr, 16 * R);
        } else try c.group.lib.check(c.group.lib.api.ncclAllGather(b.amax.ptr, b.amax_all.ptr, 4 * @as(usize, R), .f32, c.group.comm, c.stream.handle), "ncclAllGather");
        try c.glue.pick(c.stream, b.amax_all.ptr, c.world, R, b.ids.ptr, b.tok_out.ptr);
    }

    /// The MLP: gate and up (EXL3), SwiGLU, down into `out` (fp32 unless `out_bf16`).
    /// MLP: gate と up (EXL3)、SwiGLU、down を `out` に (`out_bf16` でなければ fp32)。
    fn mlp(c: *const Ctx, m: *const model.Mlp, R: u32, out: u64, out_bf16: bool) !void {
        const b = c.b;
        const width = m.gate.n;
        try c.lins(&.{ &m.gate, &m.up }, b.normed.ptr, R, &.{ b.g.ptr, b.u.ptr }, &.{ true, true });
        try c.ops.swiglu2(b.g.ptr, b.u.ptr, b.act.ptr, R, width);
        try c.lin(&m.down, b.act.ptr, R, out, out_bf16);
    }

    /// The FFN: the dense MLP, or route + routed experts + the shared expert (b.part += b.sy) on MoE layers.
    /// FFN: dense MLP、MoE 層ではルーティング + routed expert + shared expert (b.part += b.sy)。
    fn ffnPart(c: *const Ctx, L: *const model.Layer, R: u32) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        if (L.experts != null) {
            // route: router logits (bf16 GEMM) and the sigmoid + bias top-8
            // ルーティング: router の logit (bf16 GEMM) と sigmoid + bias の top-8
            try c.ops.router(b.normed.ptr, D, L.router_w.?.ptr, b.rpart.ptr, b.mlog.ptr, R, D, @intCast(c.cfg.experts));
            try c.ops.topk(b.mlog.ptr, L.router_bias.?.ptr, b.pick.ptr, b.wts.ptr, R, c.cfg.routed_scaling);
        }
        return c.ffnMain(L, R);
    }

    /// The routed experts (picks in b.pick / b.wts) and the shared expert, or the dense MLP -> b.part[:R].
    /// routed expert (選択は b.pick / b.wts) と shared expert、または dense MLP -> b.part[:R]。
    fn ffnMain(c: *const Ctx, L: *const model.Layer, R: u32) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        const lane = c.laneBf16(R);
        const sp = c.laneSp(R);
        const ex = L.experts orelse {
            if (lane) return c.mlp(&L.mlp, R, b.dpart.?.ptr, true);
            if (sp) {
                const m = &L.mlp;
                try c.lins(&.{ &m.gate, &m.up }, b.normed.ptr, R, &.{ b.g.ptr, b.u.ptr }, &.{ true, true });
                try c.ops.swiglu2(b.g.ptr, b.u.ptr, b.act.ptr, R, m.gate.n);
                return c.linHpart(&m.down, b.act.ptr, R);
            }
            return c.mlp(&L.mlp, R, b.part.ptr, false);
        };
        // the routed experts' sum went through bf16 (prompt_experts), or would have
        // routed expert の和が bf16 を経た (prompt_experts)、または経るはずだった
        var rounded = false;
        // fp16 sums (TF_GLM53_PE_F16): sequence-parallel parts only, whose next step reads them (raddF16ToBf16)
        // fp16 の和 (TF_GLM53_PE_F16): sequence-parallel の part だけ (次の段が raddF16ToBf16 で読む)
        var half_sums = false;
        if (R > MAX_LIN_ROWS and c.pek != null and L.index < c.pe_ok.len and c.pe_ok[L.index]) {
            half_sums = sp and c.pe_f16;
            try prompt_experts.run(c.pek.?, c.stream, b.normed.ptr, D, b.pick.ptr, b.wts.ptr, &ex, c.pes.?, b.part.ptr, R, c.glue, !sp, half_sums);
            rounded = true;
        } else try moe.routed(c.ex, c.stream, b.normed.ptr, D, b.pick.ptr, b.wts.ptr, &ex, c.xs, b.part.ptr, R);
        try c.mlp(&L.mlp, R, b.sy.ptr, false);
        if (sp) {
            // bf16(experts (through bf16 where prompt_experts made them) + shared expert) into b.hpart, one pass
            // bf16(expert (prompt_experts が作った所は bf16 を経る) + shared expert) を 1 pass で b.hpart に
            const n: u64 = @as(u64, R) * D;
            if (half_sums) return c.glue.raddF16ToBf16(c.stream, b.part.ptr, b.sy.ptr, b.hpart.?.ptr, n);
            return c.glue.raddToBf16(c.stream, b.part.ptr, b.sy.ptr, b.hpart.?.ptr, n, rounded);
        }
        if (lane) return c.glue.addToBf16(c.stream, b.part.ptr, b.sy.ptr, b.dpart.?.ptr, @as(u64, R) * D);
        try c.glue.addF32(c.stream, b.part.ptr, b.sy.ptr, @as(u64, R) * D);
    }

    /// The rows' cache writes alone for a layer whose output no prompt row reads (the MTP layer's prompt rows, mtp.zig):
    /// the input norm of b.x, kv_a's latent and RoPE key, the indexer's key, written for the R rows at `pos`: what the
    /// attention's front leaves in the caches, without its queries, selection, attention or FFN.
    /// 出力をどの prompt 行も読まない層 (MTP 層の prompt 行、mtp.zig) のキャッシュ書き込みだけ: b.x の入力 norm、
    /// kv_a の latent と RoPE キー、indexer のキーを `pos` からの R 行分書く。attention の前段がキャッシュに残すものと
    /// 同じで、query・選択・attention・FFN はしない。
    pub fn cacheFill(c: *const Ctx, L: *const model.Layer, R: u32, kc: u64, ic: ?u64, pos: u64) !void {
        const cfg = c.cfg;
        const b = c.b;
        const D: u32 = @intCast(cfg.hidden);
        try c.ops.rmsnorm(b.x.ptr, D, L.input_norm.ptr, b.normed.ptr, D, R, D, cfg.eps);
        try c.lin(&L.kv_a, b.normed.ptr, R, b.kva.ptr, true);
        try c.ops.kvWrite(b.kva.ptr, L.kv_a.n, L.kv_a_norm.ptr, kc, pos, b.inv.ptr, cfg.eps, R);
        const ix = L.indexer orelse return;
        try c.ops.router(b.normed.ptr, D, ix.wk.ptr, b.rpart.ptr, b.ik.ptr, R, D, @intCast(cfg.index_dim));
        try c.ops.ikWrite(b.ik.ptr, ix.k_norm_w.ptr, ix.k_norm_b.ptr, b.inv.ptr, ic.?, pos, R);
    }

    /// One layer: attention, residual, FFN (dense or MoE), residual, for R rows at `pos`.
    /// 1 層: attention、residual、FFN (dense か MoE)、residual。`pos` からの R 行。
    pub fn layerPrompt(c: *const Ctx, L: *const model.Layer, R: u32, kc: u64, ic: ?u64, pos: u64, t: ?u32) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        try c.ops.rmsnorm(b.x.ptr, D, L.input_norm.ptr, b.normed.ptr, D, R, D, c.cfg.eps);
        try c.attention(L, R, kc, ic, pos, t);
        const a = try c.gather(R);
        try c.emit(L.index, "attn", R, a.ptr, @as(usize, R) * D * 4);
        if (a.bf16) try c.ops.residualAddBf16(b.x.ptr, b.x.ptr, a.ptr, R, D) else try c.ops.residualAdd(b.x.ptr, b.x.ptr, a.ptr, R, D, a.world);
        try c.ops.rmsnorm(b.x.ptr, D, L.post_attn_norm.ptr, b.normed.ptr, D, R, D, c.cfg.eps);
        try c.ffnPart(L, R);
        const f = try c.gather(R);
        try c.emit(L.index, "ffn", R, f.ptr, @as(usize, R) * D * 4);
        if (f.bf16) try c.ops.residualAddBf16(b.x.ptr, b.x.ptr, f.ptr, R, D) else try c.ops.residualAdd(b.x.ptr, b.x.ptr, f.ptr, R, D, f.world);
        try c.emit(L.index, "x", R, b.x.ptr, @as(usize, R) * D * 2);
        // after a tap layer: x into column block s of b.taps [R, n_taps * D]
        // tap 層の後: x を b.taps [R, n_taps * D] の列ブロック s に写す
        if (L.index < c.tap_slot.len and c.tap_slot[L.index] >= 0) {
            const slot: usize = @intCast(c.tap_slot[L.index]);
            try c.glue.scatter3(c.stream, b.x.ptr, try b.taps.?.at(slot * D * 2), 1, R, D, 0, @as(i64, c.n_taps) * D, false);
        }
    }

    /// The embedding: x[i] = table[ids[i]] for the R rows.
    /// 埋め込み: R 行について x[i] = table[ids[i]]。
    pub fn embed(c: *const Ctx, table: u64, R: u32) !void {
        try c.glue.gatherRows(c.stream, table, c.b.ids.ptr, c.b.x.ptr, R, @intCast(c.cfg.hidden));
    }
};

// ------------------------------------------------------ sequence-parallel prompt / sequence-parallel の prompt ---

/// A non-owning view of `len` bytes of `b` from `off`.
/// `b` の `off` からの `len` バイトを指す、所有しないビュー。
fn view(b: Buf, off: usize, len: usize) Buf {
    return .{ .d = b.d, .ptr = b.ptr + off, .len = len };
}

/// Parts of a sequence-parallel chunk: `want` (TF_GLM53_SP_PARTS) while every part stays wider than MAX_LIN_ROWS,
/// else 2.
/// sequence-parallel チャンクの part 数: どの part も MAX_LIN_ROWS 行より多いままなら `want` (TF_GLM53_SP_PARTS)、
/// そうでなければ 2。
pub fn spParts(R: u32, want: u32) u32 {
    return if (want > 2 and R >= want * (MAX_LIN_ROWS + 1)) want else 2;
}

/// Sequence parallelism (and its overlap) from 2 MAX_LIN_ROWS + 2 rows.
/// sequence parallel (とその overlap) は 2 MAX_LIN_ROWS + 2 行から。
pub const sp_min_rows = 2 * MAX_LIN_ROWS + 2;
/// The linears' split-K scratch a layer (fp32 [SK, M, N] at M up to MAX_LIN_ROWS): plan()'s tiles fit the default;
/// the engine raises it before the buffers are made when TF_GLM53_TUNE picked wider K splits.
/// linear の層ごとの split-K 作業領域 (fp32 [SK, M, N]、M は MAX_LIN_ROWS まで): plan() のタイルは既定値に収まる。
/// TF_GLM53_TUNE がより広い K 分割を選んだときは、バッファを作る前に engine が増やす。
pub var lin_z_floats: usize = MAX_LIN_ROWS * 8 * 6144;
pub const sp_max_parts = 4;

/// Part of a sequence-parallel prompt chunk: rows split into world blocks of n (padded to Rp = world n; padding rows
/// kept at zero); this rank owns rows [o, o + n), its residual stream rows in b.xo[:n]. `main` runs the every-row
/// work (attention over this rank's heads, the MLP / experts over its width share), `side` (the comm stream) the
/// own-row glue and every collective; `done` marks the end of the part's latest side chain.
/// sequence-parallel の prompt チャンクの part: 行を world 個の n 行ブロックに分け (Rp = world n まで詰め物、詰め物の
/// 行はゼロのまま)、このランクは行 [o, o + n) を持ち、その residual stream の行は b.xo[:n]。`main` は全行の処理
/// (このランクのヘッドの attention、幅の取り分の MLP / expert)、`side` (comm ストリーム) は自分の行のつなぎと全
/// collective を行う。`done` は part の最新の side の処理の終わりを示す。
pub const SpPart = struct {
    main: Ctx,
    side: Ctx,
    done: cuda.Event,
    R: u32 = 0,
    n: u32 = 0,
    Rp: u32 = 0,
    o: u32 = 0,
    nv: u32 = 0,

    fn pos(h: *const SpPart) u64 {
        return h.main.b.ppos.?.ptr;
    }
    // the positions 16 bytes apart, so each is 16-byte aligned
    // 位置は 16 バイト間隔なので、どれも 16 バイト境界にある
    fn spos(h: *const SpPart) u64 {
        return h.main.b.ppos.?.ptr + 16;
    }
    fn gpos(h: *const SpPart) u64 {
        return h.main.b.ppos.?.ptr + 32;
    }
};

/// Every rank's own rows of each buffer (rank r's at rows [r n, (r + 1) n), in place) -> rows [0, Rp), one NCCL
/// group. `bufs`: (pointer, row bytes).
/// 各バッファの全ランクの自分の行 (ランク r の分は行 [r n, (r + 1) n)、その場で) -> 行 [0, Rp)。1 つの NCCL group。
/// `bufs`: (ポインタ, 行のバイト数)。
fn spAgRows(c: *const Ctx, h: *const SpPart, bufs: []const [2]u64) !void {
    const lib = c.group.lib;
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    for (bufs) |bf| {
        const rb: usize = bf[1];
        try lib.check(lib.api.ncclAllGather(bf[0] + h.o * rb, bf[0], h.n * rb, .u8, c.group.comm, c.stream.handle), "ncclAllGather");
    }
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
}

/// The partial b.part[:R] (fp32, every row) -> the own rows' sum, added to the residual b.xo by residual_add: in
/// rank order under PREFILL_REDUCE=rs, an NCCL bf16 reduce-scatter under ring.
/// 部分和 b.part[:R] (fp32、全行) -> 自分の行の和を residual_add で residual b.xo に足す。PREFILL_REDUCE=rs では
/// ランク順、ring では NCCL の bf16 の reduce-scatter。
fn spRsResidual(c: *const Ctx, h: *const SpPart) !void {
    const b = c.b;
    const D: usize = c.cfg.hidden;
    if (c.laneSp(h.R)) {
        // the main stream wrote the partial as bf16 itself (coreOut / ffnMain): only the padding rows are left
        // main ストリームが部分和を bf16 で書いた (coreOut / ffnMain): 残るのは詰め物の行だけ
        if (h.Rp > h.R) try view(b.hpart.?, h.R * D * 2, (h.Rp - h.R) * D * 2).fill8(0, c.stream.handle);
        const lib = c.group.lib;
        try lib.check(lib.api.ncclReduceScatter(b.hpart.?.ptr, b.hred.?.ptr, h.n * D, .bf16, .sum, c.group.comm, c.stream.handle), "ncclReduceScatter");
        return c.ops.residualAddBf16(b.xo.?.ptr, b.xo.?.ptr, b.hred.?.ptr, h.n, @intCast(D));
    }
    if (h.Rp > h.R) try view(b.part, h.R * D * 4, (h.Rp - h.R) * D * 4).fill8(0, c.stream.handle);
    if (c.ring) {
        // b.hpart[:Rp] = bf16(b.part[:Rp]); NCCL reduce-scatter (bf16 sum) -> b.hred[:n]; residual_add of [1, n, D]
        // b.hpart[:Rp] = bf16(b.part[:Rp])。NCCL の reduce-scatter (bf16 の和) -> b.hred[:n]。[1, n, D] の residual_add
        const hp = b.hpart.?.ptr;
        const hr = b.hred.?.ptr;
        try c.glue.toBf16(c.stream, b.part.ptr, hp, @as(u64, h.Rp) * D);
        const lib = c.group.lib;
        try lib.check(lib.api.ncclReduceScatter(hp, hr, h.n * D, .bf16, .sum, c.group.comm, c.stream.handle), "ncclReduceScatter");
        return c.ops.residualAddBf16(b.xo.?.ptr, b.xo.?.ptr, hr, h.n, @intCast(D));
    }
    try c.group.allToAll(b.part.ptr, b.gath.ptr, h.n * D, .f32, c.stream.handle);
    try c.ops.residualAdd(b.xo.?.ptr, b.xo.?.ptr, b.gath.ptr, h.n, @intCast(D), c.world);
}

/// Every row's residual stream from the ranks' own rows -> out [R, D] (rows `stride` elements apart).
/// 各ランクの自分の行から全行の residual stream を集める -> out [R, D] (行の間隔は `stride` 要素)。
fn spGatherX(c: *const Ctx, h: *const SpPart, out: u64, stride: u32) !void {
    const b = c.b;
    const D: u32 = @intCast(c.cfg.hidden);
    const lib = c.group.lib;
    try lib.check(lib.api.ncclAllGather(b.xo.?.ptr, b.gath.ptr, @as(usize, h.n) * D, .bf16, c.group.comm, c.stream.handle), "ncclAllGather");
    try c.glue.scatter3(c.stream, b.gath.ptr, out, 1, h.R, D, 0, stride, false);
}

/// The attention's front under DCP: own rows' input norm, q_a / kv_a / the indexer's wk and weights_proj, gathered;
/// every row's latent and index key into this member's caches; spSelect.
/// DCP での attention の前段: 自分の行の入力 norm、q_a / kv_a / indexer の wk と weights_proj を計算して集め、全行の
/// latent と index キーをこのメンバーのキャッシュに書き、spSelect。
fn spFrontAttn(c: *const Ctx, h: *const SpPart, L: *const model.Layer, kc: u64, ic: ?u64, t: ?u32) !void {
    const cfg = c.cfg;
    const b = c.b;
    const D: u32 = @intCast(cfg.hidden);
    const ql: u32 = @intCast(cfg.q_lora);
    const o: usize = h.o;
    const n = h.n;
    const nx = b.normed.ptr + o * D * 2;
    try c.ops.rmsnorm(b.xo.?.ptr, D, L.input_norm.ptr, nx, D, n, D, cfg.eps);
    try c.lin(&L.q_a, nx, n, b.qa.ptr + o * ql * 2, true);
    try c.lin(&L.kv_a, nx, n, b.kva.ptr + o * L.kv_a.n * 2, true);
    try c.ops.rmsnorm(b.qa.ptr + o * ql * 2, ql, L.q_a_norm.ptr, b.qn.ptr + o * ql * 2, ql, n, ql, cfg.eps);
    var bufs: [4][2]u64 = undefined;
    var nb: usize = 0;
    bufs[nb] = .{ b.qn.ptr, ql * 2 };
    nb += 1;
    bufs[nb] = .{ b.kva.ptr, L.kv_a.n * 2 };
    nb += 1;
    const idim: u32 = @intCast(cfg.index_dim);
    const ih: u32 = @intCast(cfg.index_heads);
    if (L.indexer) |ix| {
        try c.ops.router(nx, D, ix.wk.ptr, b.rpart.ptr, b.ik.ptr + o * idim * 4, n, D, idim);
        bufs[nb] = .{ b.ik.ptr, idim * 4 };
        nb += 1;
        if (t != null) {
            try c.ops.router(nx, D, ix.weights_proj.ptr, b.rpart.ptr, b.iw.ptr + o * ih * 4, n, D, ih);
            bufs[nb] = .{ b.iw.ptr, ih * 4 };
            nb += 1;
        }
    }
    try spAgRows(c, h, bufs[0..nb]);
    try c.ops.kvWrite(b.kva.ptr, L.kv_a.n, L.kv_a_norm.ptr, kc, h.pos(), b.inv.ptr, cfg.eps, h.R);
    const ix = L.indexer orelse return;
    try c.ops.ikWrite(b.ik.ptr, ix.k_norm_w.ptr, ix.k_norm_b.ptr, b.inv.ptr, ic.?, h.pos(), h.R);
    if (t) |T| try spSelect(c, h, L, ic.?, T);
}

/// A layer whose caches each prompt chunk also fills (the MTP layer, mtp.zig): after the target's last layer,
/// `input` turns part k's own rows of the final residual (b.xo; chunk rows row0 + h.o ..) into the layer's input in
/// place, on the part's comm stream, and spCacheFill writes the layer's caches from them.
/// prompt チャンクごとにキャッシュも埋める層 (MTP 層、mtp.zig): target の最後の層の後、`input` が part k の最終
/// residual の自分の行 (b.xo、チャンクの行 row0 + h.o ..) をその場で層の入力に変え (part の comm ストリーム上)、
/// spCacheFill がそこから層のキャッシュを書く。
pub const SpExtra = struct {
    layer: *const model.Layer,
    kc: u64,
    ic: ?u64,
    ctx: *anyopaque,
    input: *const fn (ctx: *anyopaque, c: *const Ctx, h: *const SpPart, k: usize, row0: u32) anyerror!void,
};

/// spFrontAttn's cache writes alone: own rows' input norm, kv_a and the indexer's key, gathered; every row's latent
/// and index key into this member's caches. For a layer whose output no prompt row reads (SpExtra): its caches hold
/// what a full pass would leave, without its queries, selection, attention or FFN.
/// spFrontAttn のキャッシュ書き込みだけ: 自分の行の入力 norm、kv_a と indexer のキーを計算して集め、全行の latent と
/// index キーをこのメンバーのキャッシュに書く。出力をどの prompt 行も読まない層 (SpExtra) 用: キャッシュには全部を
/// 計算したときと同じものが入り、query・選択・attention・FFN はしない。
fn spCacheFill(c: *const Ctx, h: *const SpPart, L: *const model.Layer, kc: u64, ic: ?u64) !void {
    const cfg = c.cfg;
    const b = c.b;
    const D: u32 = @intCast(cfg.hidden);
    const o: usize = h.o;
    const n = h.n;
    const nx = b.normed.ptr + o * D * 2;
    try c.ops.rmsnorm(b.xo.?.ptr, D, L.input_norm.ptr, nx, D, n, D, cfg.eps);
    try c.lin(&L.kv_a, nx, n, b.kva.ptr + o * L.kv_a.n * 2, true);
    var bufs: [2][2]u64 = undefined;
    var nb: usize = 0;
    bufs[nb] = .{ b.kva.ptr, L.kv_a.n * 2 };
    nb += 1;
    const idim: u32 = @intCast(cfg.index_dim);
    if (L.indexer) |ix| {
        try c.ops.router(nx, D, ix.wk.ptr, b.rpart.ptr, b.ik.ptr + o * idim * 4, n, D, idim);
        bufs[nb] = .{ b.ik.ptr, idim * 4 };
        nb += 1;
    }
    try spAgRows(c, h, bufs[0..nb]);
    try c.ops.kvWrite(b.kva.ptr, L.kv_a.n, L.kv_a_norm.ptr, kc, h.pos(), b.inv.ptr, cfg.eps, h.R);
    const ix = L.indexer orelse return;
    try c.ops.ikWrite(b.ik.ptr, ix.k_norm_w.ptr, ix.k_norm_b.ptr, b.inv.ptr, ic.?, h.pos(), h.R);
}

/// The indexer's selection of a sequence-parallel part under DCP: the group's rows' index queries (wq_b + RoPE); per
/// block of each member's rows, the scores against this member's slots and their top index_topk, sent (all-to-all
/// over the group) to the rows' owners; an owner's global top index_topk of its own rows (global positions,
/// ascending), all-gathered over the world; this member keeps its own slots (ascending) and their count.
/// DCP での sequence-parallel の part の indexer の選択: グループの行の index query (wq_b + RoPE)。各メンバーの行の
/// ブロックごとに、このメンバーのスロットに対する点数とその上位 index_topk 個を行の持ち主に送る (グループ内の
/// all-to-all)。持ち主は自分の行の全体の上位 index_topk 個 (全体の位置、昇順) を出し、world で all-gather する。
/// このメンバーは自分のスロット (昇順) とその数を残す。
fn spSelect(c: *const Ctx, h: *const SpPart, L: *const model.Layer, ic: u64, T: u32) !void {
    const cfg = c.cfg;
    const b = c.b;
    const dg = c.dgroup.?;
    const G: u32 = dg.world;
    const nh: usize = cfg.index_heads;
    const idd: usize = cfg.index_dim;
    const K: u32 = @intCast(cfg.index_topk);
    const ql: usize = cfg.q_lora;
    const n = h.n;
    const g0: u32 = (c.rank / G) * G * n; // the group's rows: [g0, g0 + G n) / グループの行: [g0, g0 + G n)
    const gR = G * n;
    try c.lin(&L.indexer.?.wq_b, b.qn.ptr + g0 * ql * 2, gR, b.iq.ptr + g0 * nh * idd * 2, true);
    try c.ops.iqRope(b.iq.ptr + g0 * nh * idd * 2, b.inv.ptr, h.gpos(), gR);
    try c.iqPrep(g0, gR);
    const Tl = (T + G - 1) / G;
    if (T > b.keys) return error.TooManyKeys;
    const m: u32 = @max(1, sel_rows / G);
    const mine = b.cand.?.ptr;
    const recv = b.dmine.?.ptr;
    var j: u32 = 0;
    while (j < n) : (j += m) {
        const mr = @min(m, n - j);
        for (0..G) |k| {
            const r0: u32 = g0 + @as(u32, @intCast(k)) * n + j;
            try c.candidates(ic, h.pos(), mine + k * m * K * 8, mr, Tl, r0, false);
        }
        try dg.allToAll(mine, recv, @as(usize, m) * K, .i64, c.stream.handle);
        try c.glue.dcpSelect(c.stream, recv, b.tok.ptr + @as(u64, h.o + j) * K * 4, 0, m, mr, K, G, null, false);
    }
    try spAgRows(c, h, &.{.{ b.tok.ptr, K * 4 }});
    try c.glue.dcpOwn(c.stream, b.tok.ptr, b.cnt.?.ptr, h.R, K, G, dg.rank);
}

/// The FFN's front: own rows' post-attention norm and routing; the normed rows and picks gathered.
/// FFN の前段: 自分の行の post-attention norm とルーティング。norm 後の行と選択を集める。
fn spFrontFfn(c: *const Ctx, h: *const SpPart, L: *const model.Layer) !void {
    const cfg = c.cfg;
    const b = c.b;
    const D: u32 = @intCast(cfg.hidden);
    const o: usize = h.o;
    const n = h.n;
    const nx = b.normed.ptr + o * D * 2;
    try c.ops.rmsnorm(b.xo.?.ptr, D, L.post_attn_norm.ptr, nx, D, n, D, cfg.eps);
    if (L.experts == null) return spAgRows(c, h, &.{.{ b.normed.ptr, D * 2 }});
    const E: u32 = @intCast(cfg.experts);
    const tk: usize = cfg.top_k;
    try c.ops.router(nx, D, L.router_w.?.ptr, b.rpart.ptr, b.mlog.ptr + o * E * 4, n, D, E);
    try c.ops.topk(b.mlog.ptr + o * E * 4, L.router_bias.?.ptr, b.pick.ptr + o * tk * 4, b.wts.ptr + o * tk * 4, n, c.cfg.routed_scaling);
    try spAgRows(c, h, &.{ .{ b.normed.ptr, D * 2 }, .{ b.pick.ptr, tk * 4 }, .{ b.wts.ptr, tk * 4 } });
}

/// One sequence-parallel prompt chunk under DCP, its parts overlapped: rows `ids` at position `a` over `parts` on
/// the `main` stream and the parts' common comm stream (`ready`: the main stream's progress handed to it). Every
/// collective is issued on the comm stream in one order on every rank; the main stream only computes. Returns the
/// part holding the chunk's last row; with `head` (the prompt's last chunk) its b.x has the part's rows of the final
/// residual stream; nothing reads the other parts' or chunks'.
/// DCP での sequence-parallel の prompt チャンク 1 つ (part を重ねて実行): 位置 `a` からの行 `ids` を `parts` に
/// 分け、`main` ストリームと part 共通の comm ストリーム (`ready` で main の進み具合を渡す) で処理する。collective
/// はすべて comm ストリームで、全ランクで同じ順に発行する。main ストリームは計算だけ。チャンクの最後の行を持つ part
/// を返す。`head` (prompt の最後のチャンク) なら、その b.x に part の最終 residual stream の行が入る。他の part や
/// チャンクの分は誰も読まない。
pub fn promptSp(parts: []SpPart, want: u32, ready: cuda.Event, ids: []const i64, a: u32, embed_table: u64, layers: []const model.Layer, kc: []const Buf, ic: []const ?Buf, t: ?u32, taps: ?u64, tap_row: usize, head: bool, extra: ?SpExtra) !*SpPart {
    const R: u32 = @intCast(ids.len);
    const P = spParts(R, want);
    if (P > parts.len) return error.SpParts;
    const c0 = &parts[0].main;
    const main = c0.stream;
    const comm = parts[0].side.stream;
    const W = c0.world;
    const D: u32 = @intCast(c0.cfg.hidden);
    const dg = c0.dgroup orelse return error.SpNeedsDcp;
    var offs: [sp_max_parts + 1]u32 = undefined;
    if (P == 2) {
        offs[0] = 0;
        offs[1] = R / 2;
        offs[2] = R;
    } else for (0..P + 1) |k| {
        offs[k] = @intCast(k * R / P);
    }
    const hs = parts[0..P];
    for (hs, 0..) |*h, k| {
        h.R = offs[k + 1] - offs[k];
        h.n = (h.R + W - 1) / W;
        h.Rp = h.n * W;
        h.o = c0.rank * h.n;
        h.nv = if (h.R > h.o) @min(h.n, h.R - h.o) else 0;
        if (h.Rp > h.main.b.rows) return error.SpPartTooWide;
        if (taps) |tp| h.main.b.taps = .{ .d = h.main.b.x.d, .ptr = tp + offs[k] * tap_row, .len = h.R * tap_row };
    }
    const n_layers = layers.len;
    const side = struct {
        fn begin(rd: cuda.Event, m: cuda.Stream, cs: cuda.Stream) !void {
            try rd.record(m);
            try cs.wait(rd);
        }
    };
    // start: ids, positions, the own rows' embeddings, layer 0's front
    // 開始: ids、位置、自分の行の埋め込み、層 0 の前段
    for (hs, 0..) |*h, k| {
        const s = &h.side;
        const b = s.b;
        try side.begin(ready, main, comm);
        try b.ids.uploadAsync(0, std.mem.sliceAsBytes(ids[offs[k]..offs[k + 1]]), comm.handle);
        const p0 = a + offs[k];
        try view(b.ppos.?, 0, 4).fill32(p0, comm.handle);
        try view(b.ppos.?, 16, 4).fill32(p0 + h.o, comm.handle);
        try view(b.ppos.?, 32, 4).fill32(p0 + (c0.rank / dg.world) * dg.world * h.n, comm.handle);
        if (h.nv > 0) try s.glue.gatherRows(comm, embed_table, b.ids.ptr + @as(u64, h.o) * 8, b.xo.?.ptr, h.nv, D);
        if (h.n > h.nv) try view(b.xo.?, h.nv * D * 2, (h.n - h.nv) * D * 2).fill8(0, comm.handle);
        try spFrontAttn(s, h, &layers[0], kc[0].ptr, if (ic[0]) |x| x.ptr else null, t);
        try h.done.record(comm);
    }
    for (layers, 0..) |*L, i| {
        const kci = kc[i].ptr;
        // attention: the main steps between the group's collectives, the parts round robin
        // attention: グループの collective の間の main の段を、part を順に回して
        for (0..3) |stage| for (hs) |*h| {
            const m = &h.main;
            try main.wait(h.done);
            switch (stage) {
                0 => try m.coreQ(L, h.R, h.pos(), false),
                1 => try m.dcpAttend(dg, h.R, kci, h.pos()),
                else => {
                    try m.dcpCombine(dg, h.R);
                    try m.coreOut(L, h.R);
                },
            }
            try side.begin(ready, main, comm);
            switch (stage) {
                0 => try h.side.dcpGatherQ(dg, h.R),
                1 => try h.side.dcpExchange(dg, h.R),
                else => {
                    try spRsResidual(&h.side, h);
                    try spFrontFfn(&h.side, h, L);
                },
            }
            try h.done.record(comm);
        };
        for (hs, 0..) |*h, k| {
            try main.wait(h.done);
            try h.main.ffnMain(L, h.R);
            try side.begin(ready, main, comm);
            const s = &h.side;
            try spRsResidual(s, h);
            if (L.index < s.tap_slot.len and s.tap_slot[L.index] >= 0) {
                const slot: usize = @intCast(s.tap_slot[L.index]);
                try spGatherX(s, h, s.b.taps.?.ptr + slot * D * 2, s.n_taps * D);
            }
            if (i + 1 < n_layers) {
                try spFrontAttn(s, h, &layers[i + 1], kc[i + 1].ptr, if (ic[i + 1]) |x| x.ptr else null, t);
            } else {
                // the one row the head reads: the last part's last
                // head が読む 1 行: 最後の part の最後の行
                if (head and h == &hs[P - 1]) try spGatherX(s, h, s.b.x.ptr, D);
                if (extra) |x| {
                    try x.input(x.ctx, s, h, k, offs[k]);
                    try spCacheFill(s, h, x.layer, x.kc, x.ic);
                }
            }
            try h.done.record(comm);
        }
    }
    for (hs) |*h| try main.wait(h.done);
    return &hs[P - 1];
}
