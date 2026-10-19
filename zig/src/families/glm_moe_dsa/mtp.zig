//! GLM-5.3's own MTP layer as the drafted rounds' proposer (TF_GLM53_DRAFTER=mtp): the row at position q reads the
//! target's final-normed hidden at q and the token at q + 1, and predicts the token at q + 2. Prompt chunks only fill
//! its caches. Every rank computes the same drafts (the layer's collectives, the distributed argmax).
//! GLM-5.3 自身の MTP 層を draft ラウンドの提案元に使う (TF_GLM53_DRAFTER=mtp): 位置 q の行は、ターゲットの q での
//! final norm 後の hidden と q + 1 のトークンを読み、q + 2 のトークンを予測する。prompt チャンクではキャッシュを
//! 埋めるだけ。全ランクが同じ draft を計算する (層の collective 通信と分散 argmax)。
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const model = @import("model.zig");
const moe = @import("moe.zig");
const forward = @import("forward.zig");
const pool_plan = @import("pool/plan.zig");

const Buf = cuda.DeviceBuffer;

/// A round's backlog and chain stay within a verify window (engine.vrows).
/// 1 ラウンドのバックログと連鎖は verify ウィンドウ (engine.vrows) に収まる。
pub const max_rows = 8;
/// Prompt rows an MTP layer pass takes at once (the prompt buffers' width).
/// MTP 層が 1 回に処理する prompt の行数 (prompt バッファの幅)。
pub const piece_rows = forward.max_rows;
/// eh_proj rows a router call takes (its K-split partials fit Buffers.rpart).
/// router 呼び出し 1 回が処理する eh_proj の行数 (K 分割の部分和が Buffers.rpart に収まる)。
const eh_rows = 256;

fn view(b: Buf, off: usize, len: usize) Buf {
    return .{ .d = b.d, .ptr = b.ptr + off, .len = len };
}

pub const Mtp = struct {
    layer: model.Layer,
    enorm: Buf,
    hnorm: Buf,
    head_norm: Buf,
    eh_proj: Buf, // bf16 [D, 2D], every rank whole / bf16 [D, 2D]、全ランクが全体を持つ
    index: usize, // the layer's caches: State.kc / ic entry / 層のキャッシュ: State.kc / ic のエントリ
    k: usize, // drafts a round / 1 ラウンドの draft 数
    me: Buf, // bf16 [piece_rows, D]: embeddings / 埋め込み
    mcat: Buf, // bf16 [piece_rows, 2D]: [enorm(embedding), hnorm(hidden)] / [enorm(埋め込み), hnorm(hidden)]
    mx32: Buf, // fp32 [eh_rows, D]: eh_proj's output / eh_proj の出力
    // bf16 [chunk rows + 1, D]: the rows' target hiddens; rows 0 .. max_rows hold the decode backlog
    // bf16 [チャンク行数 + 1, D]: 各行のターゲットの hidden。decode では行 0 .. max_rows がバックログを持つ
    hin: Buf,
    ids: Buf, // int64 [chunk rows + 1] / int64 [チャンク行数 + 1]
    carry: Buf, // bf16 [D]: the prompt's last row's normed hidden (the head's) / prompt 最終行の norm 後の hidden
    dtok: Buf, // int32 [max_rows]: a round's drafts / 1 ラウンドの draft
    pos: Buf, // int32: the MTP window's first position / MTP ウィンドウの先頭位置
    m: usize = 0, // backlog rows: positions P - m .. P - 1 / バックログの行数: 位置 P - m .. P - 1
    // the request's first round: the backlog is (carry, the pending token)
    // 要求の最初のラウンド: バックログは (carry, 保留中のトークン)
    fresh: bool = true,
    // the prompt's last chunk left its last row's hidden in `carry`
    // prompt の最終チャンクが、最終行の hidden を `carry` に残した
    have_carry: bool = false,
    host_ids: []i64 = &.{},
    // the sequence-parallel chunk spBegin set up for spInput
    // spBegin が spInput のために用意した sequence-parallel チャンク
    sp_a: usize = 0,
    sp_rows: usize = 0,
    // TF_GLM53_PREFIX=pool: where this request's MTP rows are right, [0, F) (glm53/POOL.md 4.6.1); the engine starts it
    // at the resume point and advances it per prompt chunk, propose advances it per backlog rewrite
    // TF_GLM53_PREFIX=pool: この要求の MTP の行が正しい範囲 [0, F) (glm53/POOL.md 4.6.1)。エンジンが再開位置で始めて
    // prompt のチャンクごとに進め、propose がバックログを書き直すごとに進める
    track: pool_plan.MtpF = .{ .F = 0, .n = 0 },
    embed: u64 = 0,
    final_norm: u64 = 0,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, cfg: Config, heads: usize, index: usize, k: usize, chunk_rows: usize) !Mtp {
        if (cfg.mtp_layers < 1) return error.NoMtpLayer;
        const li = cfg.layers;
        var layer = try model.loadLayerAs(gpa, d, r, cfg, li, heads, true);
        layer.experts = try moe.load(gpa, d, r, cfg, li);
        var nb: [128]u8 = undefined;
        const D = cfg.hidden;
        const rows = chunk_rows + 1;
        return .{
            .layer = layer,
            .enorm = try bf(gpa, d, r, try std.fmt.bufPrint(&nb, "model.layers.{d}.enorm.weight", .{li})),
            .hnorm = try bf(gpa, d, r, try std.fmt.bufPrint(&nb, "model.layers.{d}.hnorm.weight", .{li})),
            .head_norm = try bf(gpa, d, r, try std.fmt.bufPrint(&nb, "model.layers.{d}.shared_head.norm.weight", .{li})),
            .eh_proj = try bf(gpa, d, r, try std.fmt.bufPrint(&nb, "model.layers.{d}.eh_proj.weight", .{li})),
            .index = index,
            .k = @min(k, max_rows - 1),
            .me = try Buf.alloc(d, piece_rows * D * 2),
            .mcat = try Buf.alloc(d, piece_rows * 2 * D * 2),
            .mx32 = try Buf.alloc(d, eh_rows * D * 4),
            .hin = try Buf.alloc(d, @max(rows, max_rows) * D * 2),
            .ids = try Buf.alloc(d, @max(rows, max_rows) * 8),
            .carry = try Buf.alloc(d, D * 2),
            .dtok = try Buf.alloc(d, max_rows * 4),
            .pos = try Buf.alloc(d, 16),
            .host_ids = try gpa.alloc(i64, @max(rows, max_rows)),
        };
    }

    fn bf(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, name: []const u8) !Buf {
        var s = try r.get(name);
        defer s.deinit(gpa);
        if (s.dtype != .bf16) return error.MtpTensorNotBf16;
        return Buf.fromHost(d, s.bytes);
    }

    /// A request starts: no carry, the first round seeds the backlog.
    /// 要求の開始: carry はなく、最初のラウンドがバックログを作る。
    pub fn begin(m: *Mtp) void {
        m.fresh = true;
        m.have_carry = false;
        m.m = 0;
    }

    /// The MTP layer's input for n rows into `out` (bf16 [n, D]): the embeddings of `ids` (int64) and the hiddens at
    /// `hin`, through mix. `zero_first`: row 0's embedding is zero (position 0).
    /// n 行分の MTP 層の入力を `out` (bf16 [n, D]) へ: `ids` (int64) の埋め込みと `hin` の hidden を mix に通す。
    /// `zero_first`: 行 0 の埋め込みをゼロにする (位置 0)。
    fn input(m: *Mtp, c: *const forward.Ctx, embed_table: u64, ids: u64, hin: u64, n: u32, zero_first: bool, out: u64) !void {
        const D: u32 = @intCast(c.cfg.hidden);
        try c.glue.gatherRows(c.stream, embed_table, ids, m.me.ptr, n, D);
        if (zero_first) try view(m.me, 0, D * 2).fill8(0, c.stream.handle);
        try m.mix(c, hin, n, out);
    }

    /// m.me's n embeddings and the hiddens at `hin`, each RMS-normed, concatenated, through eh_proj (fp32, then bf16)
    /// into `out`.
    /// m.me の n 個の埋め込みと `hin` の hidden をそれぞれ RMS norm して連結し、eh_proj を通して (fp32 から bf16)
    /// `out` へ書く。
    fn mix(m: *Mtp, c: *const forward.Ctx, hin: u64, n: u32, out: u64) !void {
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        try c.ops.rmsnorm(m.me.ptr, D, m.enorm.ptr, m.mcat.ptr, 2 * D, n, D, c.cfg.eps);
        try c.ops.rmsnorm(hin, D, m.hnorm.ptr, m.mcat.ptr + @as(u64, D) * 2, 2 * D, n, D, c.cfg.eps);
        if (c.b.rpart.len < 8 * @as(usize, @min(eh_rows, n)) * D * 4) return error.MtpRouterPartials;
        var r0: u32 = 0;
        while (r0 < n) : (r0 += eh_rows) {
            const rows = @min(eh_rows, n - r0);
            try c.ops.router(m.mcat.ptr + @as(u64, r0) * 2 * D * 2, 2 * D, m.eh_proj.ptr, c.b.rpart.ptr, m.mx32.ptr, rows, 2 * D, D);
            try c.glue.toBf16(s, m.mx32.ptr, out + @as(u64, r0) * D * 2, @as(u64, rows) * D);
        }
    }

    /// The prompt's token after each of rows a .. a + R - 1 into host_ids. The prompt's last row has none yet: a
    /// placeholder, whose cache rows the first round rewrites at the same position.
    /// 行 a .. a + R - 1 それぞれの次の prompt トークンを host_ids へ。prompt の最終行にはまだ次がないので仮の値を
    /// 置き、そのキャッシュ行は最初のラウンドが同じ位置で書き直す。
    fn nextIds(m: *Mtp, toks: []const i64, a: usize, R: usize) void {
        for (0..R) |i| m.host_ids[i] = if (a + 1 + i < toks.len) toks[a + 1 + i] else 0;
    }

    /// A prompt chunk's MTP rows after the target ran it (no sequence-parallel parts): `x` holds rows a .. a + R - 1
    /// as the last layer left them (bf16 [R, D]), `toks` the prompt. Only the cache writes run.
    /// ターゲットが実行したあとの prompt チャンクの MTP 行 (sequence-parallel の分割なし): `x` は最終層が残した行
    /// a .. a + R - 1 (bf16 [R, D])、`toks` は prompt。キャッシュへの書き込みだけを実行する。
    pub fn prompt(m: *Mtp, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, final_norm: u64, x: u64, toks: []const i64, a: usize, R: u32) !void {
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        try c.ops.rmsnorm(x, D, final_norm, m.hin.ptr, D, R, D, c.cfg.eps);
        m.nextIds(toks, a, R);
        var r0: u32 = 0;
        while (r0 < R) : (r0 += piece_rows) {
            const n = @min(piece_rows, R - r0);
            const q0 = a + r0;
            try c.b.ids.uploadAsync(0, std.mem.sliceAsBytes(m.host_ids[r0..][0..n]), s.handle);
            try m.input(c, embed_table, c.b.ids.ptr, m.hin.ptr + @as(u64, r0) * D * 2, n, q0 == 0, c.b.x.ptr);
            try view(m.pos, 0, 4).fill32(@intCast(q0), s.handle);
            try c.cacheFill(&m.layer, n, kc, ic, m.pos.ptr);
        }
    }

    /// A sequence-parallel chunk (rows a .. a + R - 1 of `toks`) is about to run: its next tokens for spInput.
    /// sequence-parallel チャンク (`toks` の行 a .. a + R - 1) の実行前: spInput 用に次のトークンを用意する。
    pub fn spBegin(m: *Mtp, toks: []const i64, a: usize, R: usize, embed_table: u64, final_norm: u64) void {
        m.nextIds(toks, a, R);
        m.sp_a = a;
        m.sp_rows = R;
        m.embed = embed_table;
        m.final_norm = final_norm;
    }

    /// The SpExtra of a sequence-parallel chunk: the MTP layer, its caches at kc / ic.
    /// sequence-parallel チャンクの SpExtra: MTP 層と、kc / ic にあるそのキャッシュ。
    pub fn spExtra(m: *Mtp, kc: u64, ic: ?u64) forward.SpExtra {
        return .{ .layer = &m.layer, .kc = kc, .ic = ic, .ctx = m, .input = spInput };
    }

    /// forward.SpExtra.input: part k's own rows of the final residual (b.xo, n rows from chunk row row0 + h.o) become
    /// the MTP layer's input in place, on the part's comm stream (every part's work runs on it in order).
    /// forward.SpExtra.input: 部分 k が持つ最終 residual の行 (b.xo の n 行、チャンク行 row0 + h.o から) を、その場で
    /// MTP 層の入力に変える。部分の comm stream 上で実行する (全部分の処理がその上に順に並ぶ)。
    fn spInput(ctx: *anyopaque, c: *const forward.Ctx, h: *const forward.SpPart, k: usize, row0: u32) anyerror!void {
        const m: *Mtp = @ptrCast(@alignCast(ctx));
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        const xo = c.b.xo.?.ptr;
        if (k == 0) try m.ids.uploadAsync(0, std.mem.sliceAsBytes(m.host_ids[0..m.sp_rows]), s.handle);
        try c.ops.rmsnorm(xo, D, m.final_norm, m.hin.ptr, D, h.n, D, c.cfg.eps);
        const r = row0 + h.o; // the own rows' first chunk row / 自分の行の先頭のチャンク行
        if (h.nv > 0) try c.glue.gatherRows(s, m.embed, m.ids.ptr + @as(u64, r) * 8, m.me.ptr, h.nv, D);
        if (h.n > h.nv) try view(m.me, h.nv * D * 2, (h.n - h.nv) * D * 2).fill8(0, s.handle);
        if (h.nv > 0 and m.sp_a + r == 0) try view(m.me, 0, D * 2).fill8(0, s.handle);
        try m.mix(c, m.hin.ptr, h.n, xo);
    }

    /// The prompt's last row's normed hidden (forward.Ctx.headLogits left it in fnormed row 0), kept for the first
    /// round's backlog.
    /// prompt 最終行の norm 後の hidden (forward.Ctx.headLogits が fnormed の行 0 に残したもの) を、最初のラウンドの
    /// バックログ用に保持する。
    pub fn setCarry(m: *Mtp, s: cuda.Stream, fnormed: u64, D: usize) !void {
        try m.carry.copyFrom(0, fnormed, D * 2, s.handle);
        m.have_carry = true;
    }

    /// One MTP window of n rows (ids / hin rows 0 .. n) at P0 in the decode buffers' context: its last row's draft
    /// into dtok[j] and ids[0], its normed output into hin[0] (the next step's input).
    /// decode バッファの context で、位置 P0 の n 行 (ids / hin の行 0 .. n) の MTP ウィンドウを 1 回実行する: 最終行の
    /// draft を dtok[j] と ids[0] へ、norm 後の出力を hin[0] へ (次のステップの入力)。
    fn step(m: *Mtp, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, n: u32, P0: usize, t: ?u32, j: usize) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        try m.input(c, embed_table, m.ids.ptr, m.hin.ptr, n, false, b.x.ptr);
        try view(m.pos, 0, 4).fill32(@intCast(P0), s.handle);
        try c.layerDecode(&m.layer, n, kc, ic, m.pos.ptr, t);
        try c.headLogits(m.head_norm.ptr, lm_head, n - 1, 1, vshare);
        try c.glue.rowArgmax(s, b.lpart.ptr, b.amax.ptr, 1, vshare, c.rank * vshare);
        if (c.fast) |f| {
            try f.allGather(s, b.amax.ptr, b.amax_all.ptr, 16);
        } else try c.group.lib.check(c.group.lib.api.ncclAllGather(b.amax.ptr, b.amax_all.ptr, 4, .f32, c.group.comm, s.handle), "ncclAllGather");
        try c.glue.pick(s, b.amax_all.ptr, c.world, 1, m.ids.ptr, m.dtok.ptr + @as(u64, j) * 4);
        try m.hin.copyFrom(0, b.fnormed.ptr, D * 2, s.handle);
    }

    /// One MTP round: the backlog (rows at P - m .. P - 1) drafts the token after the pending `tok` (at P), then single
    /// rows chain on, min(room, k) drafts in all. Drafts to `out`; returns how many.
    /// MTP の 1 ラウンド: バックログ (位置 P - m .. P - 1 の行) が保留中の `tok` (位置 P) の次を draft し、続けて
    /// 1 行ずつ連鎖する。draft は合計 min(room, k) 個。draft を `out` へ書き、その数を返す。
    pub fn propose(m: *Mtp, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, tok: u32, P: usize, room: usize, t: ?u32, out: []u32) !usize {
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        const n = @min(room, m.k);
        if (n == 0) return 0;
        if (m.fresh) {
            if (!m.have_carry) return error.MtpNoCarry;
            try m.hin.copyFrom(0, m.carry.ptr, D * 2, s.handle);
            m.host_ids[0] = tok;
            try m.ids.uploadAsync(0, std.mem.sliceAsBytes(m.host_ids[0..1]), s.handle);
            m.m = 1;
            m.fresh = false;
        }
        if (m.m == 0 or m.m > max_rows) return error.MtpBacklog;
        try m.step(c, kc, ic, embed_table, lm_head, vshare, @intCast(m.m), P - m.m, t, 0);
        m.track.propose(P, m.m, room);
        for (1..n) |j| try m.step(c, kc, ic, embed_table, lm_head, vshare, 1, P + j - 1, t, j);
        m.track.chain(P, n - 1);
        var got: [max_rows]i32 = undefined;
        try m.dtok.downloadAsync(0, std.mem.sliceAsBytes(got[0..n]), s.handle);
        try s.synchronize();
        for (got[0..n], 0..) |g, i| out[i] = @intCast(g);
        m.m = 0; // the backlog is in the MTP caches now / バックログは MTP のキャッシュに入った
        return n;
    }

    /// The window's kept rows (the pending token and the accepted drafts, n rows at P .. P + n - 1) become the next
    /// round's backlog: row i is (the target's normed hidden of row i, picks[i]). `fnormed`: the window's normed rows.
    /// ウィンドウで残った行 (保留中のトークンと採用された draft、位置 P .. P + n - 1 の n 行) を次のラウンドの
    /// バックログにする: 行 i は (行 i のターゲットの norm 後の hidden, picks[i])。
    /// `fnormed`: ウィンドウの norm 後の行。
    pub fn kept(m: *Mtp, s: cuda.Stream, fnormed: u64, D: usize, picks: []const i32) !void {
        const n = picks.len;
        if (n == 0 or n > max_rows) return error.MtpBacklog;
        try m.hin.copyFrom(0, fnormed, n * D * 2, s.handle);
        for (picks, 0..) |p, i| m.host_ids[i] = p;
        try m.ids.uploadAsync(0, std.mem.sliceAsBytes(m.host_ids[0..n]), s.handle);
        m.m = n;
        m.fresh = false;
    }
};
