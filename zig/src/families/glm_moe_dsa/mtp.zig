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

/// Ranks whose gathered argmax rows a step's probability reads at most (forward.Ctx.pickArgmax's bound).
/// ステップの確率が読む、集めた argmax の行のランク数の上限 (forward.Ctx.pickArgmax と同じ)。
const max_world = 16;

/// TF_GLM53_MTP_CONF: where a round's chain stops. `floor` 0 is off (every draft room and k allow, as before); else
/// the first draft always, then draft j while the product of the drafts' probabilities through j stays at `floor` or
/// above (DFlash2's rule, drafter.Drafter.propose); a chain already under it runs no more steps. `inv_t`: 1 / the
/// request's temperature (1 when greedy). Every rank cuts alike: the probabilities come from the gathered bytes.
/// TF_GLM53_MTP_CONF: ラウンドの連鎖を止める所。`floor` 0 は無効 (room と k の許すだけ draft する、従来どおり)。
/// それ以外は最初の draft は必ず残し、draft j は j までの確率の積が `floor` 以上である間だけ残す (DFlash2 の規則、
/// drafter.Drafter.propose)。すでに下回った連鎖はそれ以上の段を実行しない。`inv_t`: 1 / 要求の温度 (greedy は 1)。
/// 確率は集めたバイトから出すので、全ランクが同じ所で切る。
pub const Cut = struct {
    floor: f64 = 0,
    inv_t: f32 = 1,

    pub fn on(c: Cut) bool {
        return c.floor > 0;
    }
};

/// The pick's probability of row `row` from the gathered argmax rows `all` ([world, rows, 4]: value, id, the rank's
/// mass from rowLse, 0): 1 / sum_r mass_r exp((value_r - M) inv_t), M the largest value (its own term is 1, so the sum
/// is at least 1). 0 when a value is a NaN or +inf, or no rank has a finite one. fp64 in rank order.
/// 集めた argmax の行 `all` ([world, rows, 4]: 値、id、rowLse によるランクの質量、0) から、行 `row` で選んだトークンの
/// 確率: 1 / sum_r mass_r exp((value_r - M) inv_t)。M は最大の値 (その項自身が 1 なので和は 1 以上)。値に NaN か
/// +inf があるとき、または有限の値を持つランクがないときは 0。fp64 でランク順に。
pub fn pickProb(all: []const f32, world: usize, rows: usize, row: usize, inv_t: f32) f64 {
    const ninf = -std.math.inf(f32);
    var M: f32 = ninf;
    for (0..world) |r| {
        const v = all[(r * rows + row) * 4];
        if (std.math.isNan(v) or v == std.math.inf(f32)) return 0;
        if (v > M) M = v;
    }
    if (M == ninf) return 0;
    var z: f64 = 0;
    for (0..world) |r| {
        const b = (r * rows + row) * 4;
        if (all[b] == ninf) continue;
        z += @as(f64, all[b + 2]) * @exp((@as(f64, all[b]) - @as(f64, M)) * @as(f64, inv_t));
    }
    if (!(z >= 1)) return 0;
    return 1 / z;
}

/// TF_GLM53_MULTI: one conversation's decode state, the fields of Mtp a round reads and writes (the backlog rows, the
/// carry, the drafts, the position, the input scratch, the counters and the rows' range), in buffers of its own: a
/// round of the other conversations inside a prefill (MidHook) must leave Mtp's own (the prompt's next tokens in
/// host_ids, its track) as they are. Rows: up to max_rows.
/// TF_GLM53_MULTI: 1 つの会話の decode の状態。ラウンドが読み書きする Mtp のフィールド (バックログの行、carry、draft、
/// 位置、入力の作業域、数、行の範囲) を専用のバッファで持つ: prefill の途中のほかの会話のラウンド (MidHook) は、Mtp
/// 自身のもの (host_ids の prompt の次のトークン、track) をそのまま残さねばならない。行は最大 max_rows。
pub const Dec = struct {
    hin: Buf,
    ids: Buf,
    carry: Buf,
    dtok: Buf,
    pos: Buf,
    me: Buf,
    mcat: Buf,
    mx32: Buf,
    host_ids: []i64,
    m: usize = 0,
    fresh: bool = true,
    have_carry: bool = false,
    track: pool_plan.MtpF = .{ .F = 0, .n = 0 },

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, D: usize) !Dec {
        return .{
            .hin = try Buf.alloc(d, max_rows * D * 2), .ids = try Buf.alloc(d, max_rows * 8), .carry = try Buf.alloc(d, D * 2),
            .dtok = try Buf.alloc(d, max_rows * 4), .pos = try Buf.alloc(d, 16), .me = try Buf.alloc(d, max_rows * D * 2),
            .mcat = try Buf.alloc(d, max_rows * 2 * D * 2), .mx32 = try Buf.alloc(d, max_rows * D * 4),
            .host_ids = try gpa.alloc(i64, max_rows),
        };
    }

    /// The state the prefill left in `m` (its rows' range, the carry): the conversation's first round seeds its backlog
    /// from them. Stream-ordered after the prefill.
    /// prefill が `m` に残した状態 (行の範囲、carry) を受け取る: 会話の最初のラウンドがそこからバックログを作る。
    /// prefill の後に stream の順で。
    pub fn take(d: *Dec, m: *const Mtp, s: cuda.Stream, D: usize) !void {
        d.track = m.track;
        d.fresh = true;
        d.m = 0;
        d.have_carry = m.have_carry;
        if (m.have_carry) try d.carry.copyFrom(0, m.carry.ptr, D * 2, s.handle);
    }
};

/// TF_GLM53_MULTI with TF_GLM53_MULTI_MTP=batch: the rows of every drafting conversation in one step, gathered from their
/// Decs (backlog hiddens and tokens), the input scratch over them, and the step's picks (the next tokens, int64, and the
/// drafts, int32) before they go back to each conversation.
/// TF_GLM53_MULTI と TF_GLM53_MULTI_MTP=batch: 1 つの段の、draft する全会話の行。各 Dec から集めたもの (バックログの
/// hidden とトークン)、その入力の作業域、段の選択 (次のトークン int64 と draft int32。各会話へ戻す前)。
pub const Stage = struct {
    hin: Buf,
    ids: Buf,
    me: Buf,
    mcat: Buf,
    mx32: Buf,
    pids: Buf,
    ptok: Buf,

    pub fn init(d: *const cuda.Driver, D: usize, rows: usize) !Stage {
        return .{
            .hin = try Buf.alloc(d, rows * D * 2), .ids = try Buf.alloc(d, rows * 8), .me = try Buf.alloc(d, rows * D * 2),
            .mcat = try Buf.alloc(d, rows * 2 * D * 2), .mx32 = try Buf.alloc(d, rows * D * 4), .pids = try Buf.alloc(d, rows * 8),
            .ptok = try Buf.alloc(d, rows * 4),
        };
    }
};

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
    /// `st`: the scratch (me, mcat, mx32) and state used, Mtp's own or a conversation's Dec.
    /// `st`: 使う作業域 (me、mcat、mx32) と状態。Mtp 自身か会話の Dec。
    fn input(m: *const Mtp, st: anytype, c: *const forward.Ctx, embed_table: u64, ids: u64, hin: u64, n: u32, zero_first: bool, out: u64) !void {
        const D: u32 = @intCast(c.cfg.hidden);
        try c.glue.gatherRows(c.stream, embed_table, ids, st.me.ptr, n, D);
        if (zero_first) try view(st.me, 0, D * 2).fill8(0, c.stream.handle);
        try m.mix(st, c, hin, n, out);
    }

    /// The MTP layer's input for the n rows a Stage gathered (its ids and hiddens) into `out`.
    /// Stage が集めた n 行 (その id と hidden) の MTP 層の入力を `out` へ。
    pub fn inputStage(m: *const Mtp, st: *Stage, c: *const forward.Ctx, embed_table: u64, n: u32, out: u64) !void {
        return m.input(st, c, embed_table, st.ids.ptr, st.hin.ptr, n, false, out);
    }

    /// m.me's n embeddings and the hiddens at `hin`, each RMS-normed, concatenated, through eh_proj (fp32, then bf16)
    /// into `out`.
    /// m.me の n 個の埋め込みと `hin` の hidden をそれぞれ RMS norm して連結し、eh_proj を通して (fp32 から bf16)
    /// `out` へ書く。
    fn mix(m: *const Mtp, st: anytype, c: *const forward.Ctx, hin: u64, n: u32, out: u64) !void {
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        try c.ops.rmsnorm(st.me.ptr, D, m.enorm.ptr, st.mcat.ptr, 2 * D, n, D, c.cfg.eps);
        try c.ops.rmsnorm(hin, D, m.hnorm.ptr, st.mcat.ptr + @as(u64, D) * 2, 2 * D, n, D, c.cfg.eps);
        if (c.b.rpart.len < 8 * @as(usize, @min(eh_rows, n)) * D * 4) return error.MtpRouterPartials;
        var r0: u32 = 0;
        while (r0 < n) : (r0 += eh_rows) {
            const rows = @min(eh_rows, n - r0);
            try c.ops.router(st.mcat.ptr + @as(u64, r0) * 2 * D * 2, 2 * D, m.eh_proj.ptr, c.b.rpart.ptr, st.mx32.ptr, rows, 2 * D, D);
            try c.glue.toBf16(s, st.mx32.ptr, out + @as(u64, r0) * D * 2, @as(u64, rows) * D);
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
            try m.input(m, c, embed_table, c.b.ids.ptr, m.hin.ptr + @as(u64, r0) * D * 2, n, q0 == 0, c.b.x.ptr);
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
        try m.mix(m, c, m.hin.ptr, h.n, xo);
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
    /// into dtok[j] and ids[0], its normed output into hin[0] (the next step's input). With the cut on, the row's
    /// softmax mass goes beside its argmax before the gather (stepProb reads it).
    /// decode バッファの context で、位置 P0 の n 行 (ids / hin の行 0 .. n) の MTP ウィンドウを 1 回実行する: 最終行の
    /// draft を dtok[j] と ids[0] へ、norm 後の出力を hin[0] へ (次のステップの入力)。打ち切りが有効なら、行の softmax
    /// の質量を gather の前に argmax の横へ置く (stepProb が読む)。
    fn step(m: *const Mtp, st: anytype, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, n: u32, P0: usize, t: ?u32, j: usize, cut: Cut) !void {
        const b = c.b;
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        try m.input(st, c, embed_table, st.ids.ptr, st.hin.ptr, n, false, b.x.ptr);
        try view(st.pos, 0, 4).fill32(@intCast(P0), s.handle);
        try c.layerDecode(&m.layer, n, kc, ic, st.pos.ptr, t);
        try c.headLogits(m.head_norm.ptr, lm_head, n - 1, 1, vshare);
        try c.glue.rowArgmax(s, b.lpart.ptr, b.amax.ptr, 1, vshare, c.rank * vshare);
        if (cut.on()) try c.glue.rowLse(s, b.lpart.ptr, b.amax.ptr, 1, vshare, cut.inv_t);
        if (c.fast) |f| {
            try f.allGather(s, b.amax.ptr, b.amax_all.ptr, 16);
        } else try c.group.lib.check(c.group.lib.api.ncclAllGather(b.amax.ptr, b.amax_all.ptr, 4, .f32, c.group.comm, s.handle), "ncclAllGather");
        try c.glue.pick(s, b.amax_all.ptr, c.world, 1, st.ids.ptr, st.dtok.ptr + @as(u64, j) * 4);
        try st.hin.copyFrom(0, b.fnormed.ptr, D * 2, s.handle);
    }

    /// The probability of the last step's pick (its gathered row: one row), on the host once the stream is done. A
    /// RoCE runtime that failed fails here, before the ranks could part on a stale gather.
    /// 直前のステップで選んだトークンの確率 (集めた行。1 行)。stream が終わってからホストで。RoCE のランタイムが失敗して
    /// いればここで失敗する (古い gather のままランクが分かれる前に)。
    fn stepProb(c: *const forward.Ctx, cut: Cut) !f64 {
        const W: usize = c.world;
        if (W > max_world) return error.MtpConfWorld;
        var all: [4 * max_world]f32 = undefined;
        try c.b.amax_all.downloadAsync(0, std.mem.sliceAsBytes(all[0 .. 4 * W]), c.stream.handle);
        try c.stream.synchronize();
        if (c.fast) |f| if (!f.healthy()) return error.RoceFailed;
        if (c.dfast) |f| if (!f.healthy()) return error.RoceFailed;
        return pickProb(all[0 .. 4 * W], W, 1, 0, cut.inv_t);
    }

    /// One MTP round: the backlog (rows at P - m .. P - 1) drafts the token after the pending `tok` (at P), then single
    /// rows chain on, min(room, k) drafts in all, or fewer where `cut` stops the chain. Drafts to `out`; returns how
    /// many.
    /// MTP の 1 ラウンド: バックログ (位置 P - m .. P - 1 の行) が保留中の `tok` (位置 P) の次を draft し、続けて
    /// 1 行ずつ連鎖する。draft は合計 min(room, k) 個、`cut` が連鎖を止めればそれより少ない。draft を `out` へ書き、
    /// その数を返す。
    pub fn propose(m: *Mtp, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, tok: u32, P: usize, room: usize, t: ?u32, cut: Cut, out: []u32) !usize {
        return m.proposeOn(m, c, kc, ic, embed_table, lm_head, vshare, tok, P, room, t, cut, out);
    }

    /// `propose` on a conversation's state (TF_GLM53_MULTI; `c` with that conversation's PoolMap).
    /// 会話の状態での `propose` (TF_GLM53_MULTI。`c` はその会話の PoolMap を持つ)。
    pub fn proposeDec(m: *const Mtp, d: *Dec, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, tok: u32, P: usize, room: usize, t: ?u32, cut: Cut, out: []u32) !usize {
        return m.proposeOn(d, c, kc, ic, embed_table, lm_head, vshare, tok, P, room, t, cut, out);
    }

    fn proposeOn(m: *const Mtp, st: anytype, c: *const forward.Ctx, kc: u64, ic: ?u64, embed_table: u64, lm_head: u64, vshare: u32, tok: u32, P: usize, room: usize, t: ?u32, cut: Cut, out: []u32) !usize {
        const D: u32 = @intCast(c.cfg.hidden);
        const s = c.stream;
        const n = @min(room, m.k);
        if (n == 0) return 0;
        if (st.fresh) {
            if (!st.have_carry) return error.MtpNoCarry;
            try st.hin.copyFrom(0, st.carry.ptr, D * 2, s.handle);
            st.host_ids[0] = tok;
            try st.ids.uploadAsync(0, std.mem.sliceAsBytes(st.host_ids[0..1]), s.handle);
            st.m = 1;
            st.fresh = false;
        }
        if (st.m == 0 or st.m > max_rows) return error.MtpBacklog;
        try m.step(st, c, kc, ic, embed_table, lm_head, vshare, @intCast(st.m), P - st.m, t, 0, cut);
        st.track.propose(P, st.m, room);
        // the steps run and the drafts kept: all n, or with the cut up to where the chain falls under the floor (the
        // step whose draft falls under it has run and its draft is dropped; one sync a step to read the probability)
        // 実行した段と残す draft: n 個すべて、または打ち切りがあれば連鎖が floor を下回る所まで (下回った draft の段は
        // 実行済みで、その draft は捨てる。確率を読むために段ごとに 1 回同期する)
        var steps: usize = 1;
        var keep: usize = 1;
        if (cut.on() and n > 1) {
            var chain = try stepProb(c, cut);
            while (steps < n and chain >= cut.floor) {
                try m.step(st, c, kc, ic, embed_table, lm_head, vshare, 1, P + steps - 1, t, steps, cut);
                steps += 1;
                chain *= try stepProb(c, cut);
                if (chain < cut.floor) break;
                keep = steps;
            }
        } else {
            for (1..n) |j| try m.step(st, c, kc, ic, embed_table, lm_head, vshare, 1, P + j - 1, t, j, cut);
            steps = n;
            keep = n;
        }
        st.track.chain(P, steps - 1);
        var got: [max_rows]i32 = undefined;
        try st.dtok.downloadAsync(0, std.mem.sliceAsBytes(got[0..keep]), s.handle);
        try s.synchronize();
        for (got[0..keep], 0..) |g, i| out[i] = @intCast(g);
        st.m = 0; // the backlog is in the MTP caches now / バックログは MTP のキャッシュに入った
        return keep;
    }

    /// The window's kept rows (the pending token and the accepted drafts, n rows at P .. P + n - 1) become the next
    /// round's backlog: row i is (the target's normed hidden of row i, picks[i]). `fnormed`: the window's normed rows.
    /// ウィンドウで残った行 (保留中のトークンと採用された draft、位置 P .. P + n - 1 の n 行) を次のラウンドの
    /// バックログにする: 行 i は (行 i のターゲットの norm 後の hidden, picks[i])。
    /// `fnormed`: ウィンドウの norm 後の行。
    pub fn kept(m: *Mtp, s: cuda.Stream, fnormed: u64, D: usize, picks: []const i32) !void {
        return keptOn(m, s, fnormed, D, picks);
    }

    /// `kept` into a conversation's state (TF_GLM53_MULTI; `fnormed`: its rows of the batched window).
    /// 会話の状態への `kept` (TF_GLM53_MULTI。`fnormed`: まとめたウィンドウの中のその会話の行)。
    pub fn keptDec(d: *Dec, s: cuda.Stream, fnormed: u64, D: usize, picks: []const i32) !void {
        return keptOn(d, s, fnormed, D, picks);
    }

    fn keptOn(st: anytype, s: cuda.Stream, fnormed: u64, D: usize, picks: []const i32) !void {
        const n = picks.len;
        if (n == 0 or n > max_rows) return error.MtpBacklog;
        try st.hin.copyFrom(0, fnormed, n * D * 2, s.handle);
        for (picks, 0..) |p, i| st.host_ids[i] = p;
        try st.ids.uploadAsync(0, std.mem.sliceAsBytes(st.host_ids[0..n]), s.handle);
        st.m = n;
        st.fresh = false;
    }
};
