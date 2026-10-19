//! The DFlash2 drafter on one rank: layers fed by the target's tap rows, a 4096-slot KV ring, a block pass whose top-k
//! candidates from every rank are merged on the host, and the chain that picks the drafts. The target verifies every
//! draft, but all ranks must make the same ones, so host work reads only all-gathered or replicated values.
//! 1 ランク上の DFlash2 drafter: target の tap 行を入力にする層、4096 スロットの KV リング、全ランクの top-k 候補を
//! ホストでまとめる block pass、draft を選ぶ chain。draft は target がすべて検証するが、全ランクが同じ draft を
//! 作る必要があるので、ホスト側の処理は all-gather した値か複製された値だけを読む。
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const dw = @import("draft_weights.zig");
const roce = @import("roce.zig");
const qmm = @import("qmm.zig");
const forward = @import("forward.zig");
const ops_mod = @import("ops.zig");

// window (2047) + block (8) + a tap batch (64), rounded up
// window (2047) + block (8) + tap 1 バッチ (64) を切り上げた値
pub const ring_rows = 4096;
/// resumeAt's answer: where the window of a resumed prompt came from.
/// resumeAt の答え: 再開した prompt の window をどこから得たか。
pub const Resume = enum { none, live, saved, partial };

pub const tap_batch = 64; // context updates in pieces of this many rows / context の更新はこの行数ずつ
const edge_weight = 0.6; // the selector edge's weight in a score / score での selector edge の重み
// the weight of the target's keyed noise when sampling
// サンプリング時の target の keyed noise の重み
const noise_weight = 0.7;
const no_limit: f32 = 1e30; // no clamp in SwiGLU / SwiGLU で clamp しない

const Buf = cuda.DeviceBuffer;

const Layer = struct {
    in_norm: Buf,
    post_norm: Buf,
    a_base: Buf,
    m_base: Buf,
    q_norm: Buf,
    k_norm: Buf,
    a_kp: qmm.Matrix,
    m_kp: qmm.Matrix,
    qkv: qmm.Matrix,
    kv: qmm.Matrix,
    o: qmm.Matrix,
    gu: qmm.Matrix,
    down: qmm.Matrix,
    kc: Buf, // the K ring [kvh, ring_rows, hd] / K のリング
    vc: Buf,
};

/// What the drafter borrows from its rank of the target.
/// drafter が同じランクの target から借りるもの。
pub const Target = struct {
    stream: cuda.Stream,
    ops: ops_mod.Ops,
    glue: *const forward.Glue,
    group: *const cuda.tp.Group,
    blas: *const cuda.cublas.Library,
    handle: cuda.cublas.Handle,
    embed: u64, // the target's bf16 embedding [V, D] / target の bf16 embedding
    lm_head: u64, // this rank's bf16 lm_head share [V / W, D] / このランクの bf16 lm_head の分
    vshare: u32,
    rank: u32,
    world: u32,
    /// TF_GLM53_ROCE: the world's RoCE one-shot runtime for the drafter's gathers (data only: the same bits)
    /// TF_GLM53_ROCE: drafter の gather に使う RoCE one-shot ランタイム (データを運ぶだけなので bit は同じ)
    fast: ?*const roce.Runtime = null,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    cfg: dw.Config,
    t: Target,
    qk: qmm.Kernels,
    layers: []Layer,
    fc: qmm.Matrix,
    hidden_norm: Buf,
    norm: Buf,
    hproj: Buf,
    pred: []f32, // [V, rank] on the host / ホスト上
    succ: []f32,
    // device state
    // デバイス上の状態
    pos: Buf, // int64 [1]: the context's next position / context の次の位置
    ids: Buf, // int32 [block]: [pending, mask x (block - 1)]
    tap_in: Buf, // [tap_batch, n_taps * D] bf16
    // scratch (rows up to tap_batch)
    // scratch (行数は最大 tap_batch)
    x: Buf, x2: Buf, normed: Buf, xs: Buf, dyn: Buf, conv: Buf, qkv: Buf, q: Buf, k: Buf, v: Buf, attn: Buf, part: Buf, gath: Buf,
    rowout: Buf, gu: Buf, act: Buf, axs: Buf, ctx: Buf, big: Buf, cos: Buf, sin: Buf, inv: Buf, logits: Buf, rpart: Buf, packed_mine: Buf,
    packed_all: Buf, proj: Buf,
    context_end: usize = 0,
    // the slot cache: the rings (k, v a layer) copied after a prompt
    // スロットキャッシュ: prompt の後に写したリング (層ごとに k と v)
    snap: []Buf = &.{},
    // TF_GLM53_PREFIX=pool with TF_GLM53_POOL_DRAFT_SAVES: per-slot saves, each the live rings then the snapshot (k,
    // v a layer each: 4 x layers buffers), copied whole (pool_serve.zig decides which, glm53/POOL.md 4.6.2)
    // TF_GLM53_PREFIX=pool で TF_GLM53_POOL_DRAFT_SAVES: スロットごとの写し。それぞれ live のリング、続いて snapshot
    // (層ごとに k と v で 4 x 層数のバッファ)。丸ごとコピーする (どれを使うかは pool_serve.zig、glm53/POOL.md 4.6.2)
    saves: [][]Buf = &.{},
    // the block pass and the context updates of 1 .. block rows as CUDA graphs
    // block pass と 1 .. block 行の context 更新を CUDA graph で走らせる
    graphs: bool = true,
    block_exec: ?cuda.graph.Exec = null,
    tap_execs: [16]?cuda.graph.Exec = @splat(null),
    // host copies of a block pass
    // block pass の結果のホスト側コピー
    host_packed: []f32,
    host_proj: []u16,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, t: Target) !*Drafter {
        var w = try dw.load(gpa, io, dir, t.rank, t.world);
        defer w.close(io);
        const c = w.cfg;
        const D = c.hidden;
        const R = tap_batch;
        const e = try gpa.create(Drafter);
        const layers = try gpa.alloc(Layer, c.layers);
        for (layers, w.layers) |*L, src| {
            L.* = .{
                .in_norm = try up16(d, src.in_norm), .post_norm = try up16(d, src.post_norm),
                .a_base = try up16(d, src.a_base), .m_base = try up16(d, src.m_base),
                .q_norm = try up16(d, src.q_norm), .k_norm = try up16(d, src.k_norm),
                .a_kp = try qmm.Matrix.upload(d, src.a_kp), .m_kp = try qmm.Matrix.upload(d, src.m_kp),
                .qkv = try qmm.Matrix.upload(d, src.qkv), .kv = try qmm.Matrix.upload(d, src.kv),
                .o = try qmm.Matrix.upload(d, src.o), .gu = try qmm.Matrix.upload(d, src.gu), .down = try qmm.Matrix.upload(d, src.down),
                .kc = try zeros(d, c.kv_heads * ring_rows * c.head_dim * 2),
                .vc = try zeros(d, c.kv_heads * ring_rows * c.head_dim * 2),
            };
            var m = src;
            freePacked(gpa, &m);
        }
        const hd = c.head_dim;
        const qkv_n = (c.heads + 2 * c.kv_heads) * hd;
        e.* = .{
            .gpa = gpa, .cfg = c, .t = t, .qk = try qmm.Kernels.load(d), .layers = layers,
            .fc = try qmm.Matrix.upload(d, w.fc),
            .hidden_norm = try up16(d, w.hidden_norm), .norm = try up16(d, w.norm), .hproj = try up16(d, w.hproj),
            .pred = w.pred, .succ = w.succ,
            .pos = try zeros(d, 16), .ids = try zeros(d, c.block * 4),
            .tap_in = try zeros(d, R * c.n_taps * D * 2),
            .x = try zeros(d, R * D * 2), .x2 = try zeros(d, R * D * 2), .normed = try zeros(d, R * D * 2),
            .xs = try zeros(d, R * (D / 64) * 4 + R * (c.n_taps * D / 64) * 4),
            .dyn = try zeros(d, R * layers[0].a_kp.n * 2), .conv = try zeros(d, R * D * 2),
            .qkv = try zeros(d, R * qkv_n * 2), .q = try zeros(d, c.heads * R * hd * 2),
            .k = try zeros(d, c.kv_heads * R * hd * 2), .v = try zeros(d, c.kv_heads * R * hd * 2),
            .attn = try zeros(d, R * c.heads * hd * 2), .part = try zeros(d, R * D * 4),
            .gath = try zeros(d, t.world * R * D * 4), .rowout = try zeros(d, R * D * 2),
            .gu = try zeros(d, R * 2 * c.inter * 2), .act = try zeros(d, R * c.inter * 2), .axs = try zeros(d, R * (c.inter / 64) * 4),
            .ctx = try zeros(d, R * D * 2), .big = try zeros(d, R * D * 2),
            .cos = try zeros(d, R * (hd / 2) * 4), .sin = try zeros(d, R * (hd / 2) * 4), .inv = try zeros(d, @max(hd / 2, 64) * 4),
            .logits = try zeros(d, c.block * t.vshare * 4), .rpart = try zeros(d, 8 * c.block * t.vshare * 4),
            .packed_mine = try zeros(d, c.block * 2 * c.top_k * 4),
            .packed_all = try zeros(d, t.world * c.block * 2 * c.top_k * 4),
            .proj = try zeros(d, c.block * c.selector_rank * 2),
            .host_packed = try gpa.alloc(f32, t.world * c.block * 2 * c.top_k),
            .host_proj = try gpa.alloc(u16, c.block * c.selector_rank),
        };
        freePacked(gpa, &w.fc);
        var mask: [16]i32 = undefined;
        @memset(mask[0..c.block], @intCast(c.mask_id));
        try e.ids.upload(0, std.mem.sliceAsBytes(mask[0..c.block]));
        try t.glue.draftInvfreq(t.stream, e.inv.ptr, @floatCast(c.rope_theta), @intCast(hd / 2), @intCast(hd));
        return e;
    }

    /// A new request: the context starts empty.
    /// 新しい要求: context を空から始める。
    pub fn reset(e: *Drafter) !void {
        e.context_end = 0;
        try e.pos.fill8(0, e.t.stream.handle);
    }

    /// Copies the rings after a prompt's prefill, before decoding writes past it, and returns where the context ends;
    /// a later prompt that resumes inside it gets back the window rows decoding overwrote (resumeAt).
    /// prompt の prefill 後、decode が先へ書く前にリングを写し、context の終わりの位置を返す。後でその中から再開する
    /// prompt は、decode が上書きした window の行をここから取り戻す (resumeAt)。
    pub fn snapshot(e: *Drafter) !usize {
        const bytes = e.layers[0].kc.len;
        if (e.snap.len == 0) {
            e.snap = try e.gpa.alloc(Buf, 2 * e.layers.len);
            for (e.snap) |*b| b.* = try Buf.alloc(e.t.stream.d, bytes);
        }
        for (e.layers, 0..) |*L, i| {
            try copyD2D(e.snap[2 * i].ptr, L.kc.ptr, bytes, e.t.stream);
            try copyD2D(e.snap[2 * i + 1].ptr, L.vc.ptr, bytes, e.t.stream);
        }
        return e.context_end;
    }

    /// `n` per-slot saves (each four rings a layer, the ring's own size) and the snapshot buffers they fill.
    /// `n` 個のスロットごとの写し (それぞれ層ごとに 4 本のリング、リングと同じ大きさ) と、それが埋める snapshot の
    /// バッファ。
    pub fn initSaves(e: *Drafter, n: usize) !void {
        const bytes = e.layers[0].kc.len;
        if (e.snap.len == 0) {
            e.snap = try e.gpa.alloc(Buf, 2 * e.layers.len);
            for (e.snap) |*b| b.* = try Buf.alloc(e.t.stream.d, bytes);
        }
        e.saves = try e.gpa.alloc([]Buf, n);
        for (e.saves) |*sv| {
            sv.* = try e.gpa.alloc(Buf, 4 * e.layers.len);
            for (sv.*) |*b| b.* = try zeros(e.t.stream.d, bytes);
        }
    }

    /// Save k gets the live rings and the snapshot (stream-ordered after whatever wrote them).
    /// 写し k に live のリングと snapshot を写す (それを書いた処理の後に stream の順で)。
    pub fn saveTo(e: *Drafter, k: usize) !void {
        const bytes = e.layers[0].kc.len;
        const sv = e.saves[k];
        for (e.layers, 0..) |*L, i| {
            try copyD2D(sv[4 * i].ptr, L.kc.ptr, bytes, e.t.stream);
            try copyD2D(sv[4 * i + 1].ptr, L.vc.ptr, bytes, e.t.stream);
            try copyD2D(sv[4 * i + 2].ptr, e.snap[2 * i].ptr, bytes, e.t.stream);
            try copyD2D(sv[4 * i + 3].ptr, e.snap[2 * i + 1].ptr, bytes, e.t.stream);
        }
    }

    /// The live rings and the snapshot back from save k.
    /// 写し k から live のリングと snapshot を戻す。
    pub fn restoreFrom(e: *Drafter, k: usize) !void {
        const bytes = e.layers[0].kc.len;
        const sv = e.saves[k];
        for (e.layers, 0..) |*L, i| {
            try copyD2D(L.kc.ptr, sv[4 * i].ptr, bytes, e.t.stream);
            try copyD2D(L.vc.ptr, sv[4 * i + 1].ptr, bytes, e.t.stream);
            try copyD2D(e.snap[2 * i].ptr, sv[4 * i + 2].ptr, bytes, e.t.stream);
            try copyD2D(e.snap[2 * i + 1].ptr, sv[4 * i + 3].ptr, bytes, e.t.stream);
        }
    }

    /// The range-based resume (pool, glm53/POOL.md 4.6.2): the context continues at r; the snapshot's positions
    /// [copy[0], copy[1]) go into the live rings at the same ring slots. pool/plan.zig resumeAt decided the kind.
    /// 範囲付きの再開 (pool、glm53/POOL.md 4.6.2): context は r から続ける。snapshot の位置 [copy[0], copy[1]) を
    /// live のリングの同じスロットへ入れる。種類は pool/plan.zig の resumeAt が決めた。
    pub fn resumeRange(e: *Drafter, r: usize, copy: ?[2]usize) !void {
        e.context_end = r;
        const at = [2]i64{ @intCast(r), 0 };
        try e.pos.uploadAsync(0, std.mem.sliceAsBytes(&at), e.t.stream.handle);
        if (copy) |c| try e.copySnapRows(c[0], c[1]);
    }

    /// Positions [a, b) from the snapshot into the live rings (ring slot p % ring_rows: one or two runs a head).
    /// 位置 [a, b) を snapshot から live のリングへ (リングのスロット p % ring_rows: head ごとに 1 つか 2 つの区間)。
    fn copySnapRows(e: *Drafter, a: usize, b: usize) !void {
        const hd = e.cfg.head_dim;
        const head_bytes = ring_rows * hd * 2;
        var p = a;
        while (p < b) {
            const s0 = p % ring_rows;
            const n = @min(b - p, ring_rows - s0);
            for (e.layers, 0..) |*Ly, i| for (0..e.cfg.kv_heads) |h| {
                const off = h * head_bytes + s0 * hd * 2;
                try copyD2D(Ly.kc.ptr + off, e.snap[2 * i].ptr + off, n * hd * 2, e.t.stream);
                try copyD2D(Ly.vc.ptr + off, e.snap[2 * i + 1].ptr + off, n * hd * 2, e.t.stream);
            };
            p += n;
        }
    }

    /// Continues the context at position r (rows below r are the previous request's, same tokens) for a prompt of L
    /// tokens; the first draft block at L attends [L - window, L). Rows [r, L) come from this prompt's prefill, rows
    /// below r from the live ring (positions [live_end - ring, live_end), when the previous request ended cleanly) or
    /// else from the snapshot (positions [snap_end - ring, snap_end)). The live ring's lowest block positions are never
    /// trusted (draft passes write past its end). `below` (slot cache): both rings hold another conversation's rows
    /// from there on, never taken.
    /// 位置 r から context を続ける (r より下の行は前の要求のもので、トークンは同じ)。prompt は L トークンで、L の
    /// 最初の draft block は [L - window, L) を参照する。行 [r, L) はこの prompt の prefill から、r より下の行は live
    /// リング (位置 [live_end - ring, live_end)、前の要求が正常に終わったとき) から、なければ snapshot (位置
    /// [snap_end - ring, snap_end)) から取る。live リングの最も低い block 分の位置は信用しない (draft pass が終わりより
    /// 先に書くため)。`below` (スロットキャッシュ): そこから先は両リングとも別の会話の行なので使わない。
    pub fn resumeAt(e: *Drafter, r: usize, L: usize, live_end: ?usize, snap_end: ?usize, below: ?usize) !Resume {
        e.context_end = r;
        const at = [2]i64{ @intCast(r), 0 };
        try e.pos.uploadAsync(0, std.mem.sliceAsBytes(&at), e.t.stream.handle);
        const lo = if (L > e.cfg.window + 1) L - e.cfg.window - 1 else 0;
        if (lo >= r) return .none;
        // rows [top, r): another conversation's
        // 行 [top, r) は別の会話のもの
        const top = if (below) |b| @max(lo, @min(r, b)) else r;
        // the live ring's lowest block rows may hold a draft pass's speculative rows (it writes [end, end + block))
        // live リングの最も低い block 分の行は draft pass の投機的な行かもしれない ([end, end + block) に書くため)
        const live_lo: ?i64 = if (live_end) |le| (if (le >= top) @as(i64, @intCast(le)) - ring_rows + @as(i64, @intCast(e.cfg.block)) else null) else null;
        if (live_lo) |ll| if (ll <= @as(i64, @intCast(lo))) return if (top == r) .live else .partial;
        // rows the live ring lacks: [lo, hi)
        // live リングにない行: [lo, hi)
        const hi: usize = if (live_lo) |ll| @min(top, @as(usize, @intCast(@max(ll, @as(i64, @intCast(lo)))))) else top;
        const se = snap_end orelse return .partial;
        if (e.snap.len == 0) return .partial;
        const a = @max(lo, if (se > ring_rows) se - ring_rows else 0);
        const b = @min(hi, se);
        if (a < b) {
            const hd = e.cfg.head_dim;
            const head_bytes = ring_rows * hd * 2;
            // slots a % ring .. (b - 1) % ring: one or two runs a head
            // スロット a % ring .. (b - 1) % ring: head ごとに 1 つか 2 つの連続区間
            var p = a;
            while (p < b) {
                const s0 = p % ring_rows;
                const n = @min(b - p, ring_rows - s0);
                for (e.layers, 0..) |*Ly, i| for (0..e.cfg.kv_heads) |h| {
                    const off = h * head_bytes + s0 * hd * 2;
                    try copyD2D(Ly.kc.ptr + off, e.snap[2 * i].ptr + off, n * hd * 2, e.t.stream);
                    try copyD2D(Ly.vc.ptr + off, e.snap[2 * i + 1].ptr + off, n * hd * 2, e.t.stream);
                };
                p += n;
            }
        }
        return if (a <= lo and b >= hi and top == r) .saved else .partial;
    }

    /// out [m, w.n] = x [m, w.k] @ w^T on the 4-bit matmul; `xs` the input's 64-group sums (null: computed here).
    /// 4 bit の matmul で out [m, w.n] = x [m, w.k] @ w^T。`xs` は入力の 64 要素ごとの和 (null ならここで計算)。
    fn mm(e: *Drafter, x: u64, m: u32, w: *const qmm.Matrix, xs: ?u64, out: u64, wide: bool) !void {
        const sums = xs orelse blk: {
            try e.t.ops.groupSums(x, w.k, e.xs.ptr, m, w.k);
            break :blk e.xs.ptr;
        };
        try e.qk.matmul(e.t.stream, x, w.k, sums, w, out, m, wide);
    }

    /// A projection of rank-split input: the ranks' fp32 partials added in rank order, into bf16 rowout.
    /// ランクで分けた入力の projection: 各ランクの fp32 部分和をランク順に足し、bf16 の rowout に入れる。
    fn row(e: *Drafter, x: u64, m: u32, w: *const qmm.Matrix, xs: ?u64) !void {
        const D: u32 = @intCast(e.cfg.hidden);
        try e.mm(x, m, w, xs, e.part.ptr, true);
        const n = m * D;
        const lib = e.t.group.lib;
        if (e.t.fast != null and n * 4 <= roce.max_bytes) {
            try e.t.fast.?.allGather(e.t.stream, e.part.ptr, e.gath.ptr, n * 4);
        } else try lib.check(lib.api.ncclAllGather(e.part.ptr, e.gath.ptr, n, .f32, e.t.group.comm, e.t.stream.handle), "ncclAllGather");
        try e.t.glue.rankSumBf16(e.t.stream, e.gath.ptr, e.rowout.ptr, n, e.t.world);
    }

    /// RMSNorm into `out`, with the 64-group sums into e.xs.
    /// RMSNorm を `out` に、64 要素ごとの和を e.xs に出す。
    fn rms(e: *Drafter, x: u64, w: u64, out: u64, m: u32) !void {
        const D: u32 = @intCast(e.cfg.hidden);
        try e.t.ops.rmsnormSums(x, D, w, out, D, e.xs.ptr, m, D, e.cfg.eps);
    }

    /// One layer over `m` rows of e.x (block rows at positions pos .. pos + m - 1), back into e.x.
    /// e.x の `m` 行 (位置 pos .. pos + m - 1 の block 行) に 1 層をかけ、e.x に戻す。
    fn layer(e: *Drafter, L: *const Layer, m: u32) !void {
        const c = e.cfg;
        const D: u32 = @intCast(c.hidden);
        const gs: u32 = @intCast(c.conv_gs);
        const hd: u32 = @intCast(c.head_dim);
        const H: u32 = @intCast(c.heads);
        const KV: u32 = @intCast(c.kv_heads);
        const o = e.t.ops;
        try e.rms(e.x.ptr, L.in_norm.ptr, e.normed.ptr, m);
        try e.mm(e.normed.ptr, m, &L.a_kp, e.xs.ptr, e.dyn.ptr, false);
        try o.dconv(e.normed.ptr, e.dyn.ptr, L.a_base.ptr, null, e.conv.ptr, m, D, gs, 0);
        try e.mm(e.conv.ptr, m, &L.qkv, null, e.qkv.ptr, false);
        try o.prep(e.qkv.ptr, L.q_norm.ptr, L.k_norm.ptr, e.cos.ptr, e.sin.ptr, e.q.ptr, e.k.ptr, e.v.ptr, m, L.qkv.n, c.eps, H, KV, hd / 2);
        try e.t.glue.draftRing(e.t.stream, e.k.ptr, L.kc.ptr, e.pos.ptr, m, hd, KV, ring_rows);
        try e.t.glue.draftRing(e.t.stream, e.v.ptr, L.vc.ptr, e.pos.ptr, m, hd, KV, ring_rows);
        const scale: f32 = @bitCast(@as(u32, 0x3db504f3)); // 128 ** -0.5
        try o.dattnRing(e.q.ptr, L.kc.ptr, L.vc.ptr, e.attn.ptr, e.pos.ptr, @intCast(c.window), scale, m, H / KV, H, hd, KV, ring_rows, c.causal);
        try e.row(e.attn.ptr, m, &L.o, null);
        try o.dconv(e.rowout.ptr, e.dyn.ptr, L.a_base.ptr, e.x.ptr, e.x2.ptr, m, D, gs, 1);
        try e.rms(e.x2.ptr, L.post_norm.ptr, e.normed.ptr, m);
        try e.mm(e.normed.ptr, m, &L.m_kp, e.xs.ptr, e.dyn.ptr, false);
        try o.dconv(e.normed.ptr, e.dyn.ptr, L.m_base.ptr, null, e.conv.ptr, m, D, gs, 0);
        try e.mm(e.conv.ptr, m, &L.gu, null, e.gu.ptr, false);
        try o.swiglu(e.gu.ptr, e.act.ptr, e.axs.ptr, m, @intCast(c.inter), no_limit);
        try e.row(e.act.ptr, m, &L.down, e.axs.ptr);
        try o.dconv(e.rowout.ptr, e.dyn.ptr, L.m_base.ptr, e.x2.ptr, e.x.ptr, m, D, gs, 1);
    }

    /// The context rows of tap_in[:n]: fc, hidden_norm, every layer's k / v into the ring at positions
    /// pos .. pos + n - 1, then pos += n.
    /// tap_in[:n] の context 行: fc、hidden_norm、各層の k / v を位置 pos .. pos + n - 1 のリングに入れ、pos += n。
    fn tapsCompute(e: *Drafter, n: u32) !void {
        const c = e.cfg;
        const hd: u32 = @intCast(c.head_dim);
        const KV: u32 = @intCast(c.kv_heads);
        try e.mm(e.tap_in.ptr, n, &e.fc, null, e.big.ptr, false);
        // its 64-group sums of e.ctx serve every layer below
        // ここで出す e.ctx の 64 要素ごとの和を、下の全層で使う
        try e.rms(e.big.ptr, e.hidden_norm.ptr, e.ctx.ptr, n);
        try e.t.glue.draftRope(e.t.stream, e.pos.ptr, e.inv.ptr, e.cos.ptr, e.sin.ptr, n, hd / 2);
        for (e.layers) |*L| {
            // the same sums groupSums would make of e.ctx (both add a thread's 8 values in order, then 4 / 2 / 1)
            // groupSums が e.ctx から作る和と同じ (どちらもスレッドの 8 値を順に足し、次に 4 / 2 / 1)
            try e.mm(e.ctx.ptr, n, &L.kv, if (L.kv.k == c.hidden) e.xs.ptr else null, e.qkv.ptr, false);
            try e.t.ops.prep(e.qkv.ptr, L.q_norm.ptr, L.k_norm.ptr, e.cos.ptr, e.sin.ptr, e.qkv.ptr, e.k.ptr, e.v.ptr, n, L.kv.n, c.eps, 0, KV, hd / 2);
            try e.t.glue.draftRing(e.t.stream, e.k.ptr, L.kc.ptr, e.pos.ptr, n, hd, KV, ring_rows);
            try e.t.glue.draftRing(e.t.stream, e.v.ptr, L.vc.ptr, e.pos.ptr, n, hd, KV, ring_rows);
        }
        try e.t.glue.draftAdvance(e.t.stream, e.pos.ptr, n);
    }

    /// Committed rows' taps (`src` [n, n_taps * D] bf16 on the device) into the context at positions context_end ...
    /// 確定した行の tap (`src` [n, n_taps * D] bf16、デバイス上) を位置 context_end ... の context に入れる。
    pub fn addTaps(e: *Drafter, src: u64, n: usize) !void {
        const row_bytes = e.cfg.n_taps * e.cfg.hidden * 2;
        var at: usize = 0;
        while (at < n) {
            const k = @min(tap_batch, n - at);
            try copyD2D(e.tap_in.ptr, src + at * row_bytes, k * row_bytes, e.t.stream);
            if (e.graphs and k <= e.cfg.block) {
                if (e.tap_execs[k] == null) e.tap_execs[k] = try e.capture(.taps, @intCast(k));
                try e.tap_execs[k].?.launchOn(e.t.stream);
            } else try e.tapsCompute(@intCast(k));
            e.context_end += k;
            at += k;
        }
    }

    /// The block pass's device work (capturable): [ids] at the context end through the layers, each rank's top-k
    /// candidates all-gathered into packed_all, the selector projection into proj.
    /// block pass のデバイス側の処理 (capture できる): context の終わりの [ids] を全層に通し、各ランクの top-k 候補を
    /// packed_all に all-gather し、selector の projection を proj に出す。
    fn blockCompute(e: *Drafter) !void {
        const c = e.cfg;
        const D: u32 = @intCast(c.hidden);
        const hd: u32 = @intCast(c.head_dim);
        const n: u32 = @intCast(c.block);
        try e.t.ops.embedB16(e.ids.ptr, e.t.embed, e.x.ptr, n, D);
        try e.t.glue.draftRope(e.t.stream, e.pos.ptr, e.inv.ptr, e.cos.ptr, e.sin.ptr, n, hd / 2);
        for (e.layers) |*L| try e.layer(L, n);
        const m = n - 1;
        try e.rms(try e.x.at(D * 2), e.norm.ptr, e.normed.ptr, m);
        try e.t.ops.router(e.normed.ptr, D, e.t.lm_head, e.rpart.ptr, e.logits.ptr, m, D, e.t.vshare);
        try e.t.glue.draftTopk(e.t.stream, e.logits.ptr, e.packed_mine.ptr, m, e.t.vshare, @intCast(c.top_k), e.t.rank * e.t.vshare);
        const lib = e.t.group.lib;
        const words = m * 2 * @as(u32, @intCast(c.top_k));
        if (e.t.fast != null and words * 4 <= roce.max_bytes and (words * 4) % 16 == 0) {
            try e.t.fast.?.allGather(e.t.stream, e.packed_mine.ptr, e.packed_all.ptr, words * 4);
        } else try lib.check(lib.api.ncclAllGather(e.packed_mine.ptr, e.packed_all.ptr, words, .f32, e.t.group.comm, e.t.stream.handle), "ncclAllGather");
        // proj = h @ hproj^T (bf16): column-major C [rank, m] = hproj [rank, D] (transposed) x h [D, m]
        // proj = h @ hproj^T (bf16): 列優先の C [rank, m] = hproj [rank, D] (転置) x h [D, m]
        const sr: c_int = @intCast(c.selector_rank);
        try e.t.blas.gemmStridedBatched(e.t.handle, .{ .transa = 1, .transb = 0, .m = sr, .n = @intCast(m), .k = @intCast(D), .a = e.hproj.ptr, .a_type = 14, .lda = @intCast(D), .stride_a = 0, .b = e.normed.ptr, .b_type = 14, .ldb = @intCast(D), .stride_b = 0, .c = e.proj.ptr, .c_type = 14, .ldc = sr, .stride_c = 0, .batch = 1, .compute = 68, .algo = 99 });
    }

    const What = enum { block, taps };

    fn capture(e: *Drafter, what: What, n: u32) !cuda.graph.Exec {
        try cuda.graph.beginCapture(e.t.stream, .thread_local);
        const r = switch (what) {
            .block => e.blockCompute(),
            .taps => e.tapsCompute(n),
        };
        r catch |err| {
            if (cuda.graph.endCapture(e.t.stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(e.t.stream);
        defer g.deinit();
        return g.instantiate();
    }

    /// The pending token into ids[0], the block pass (graph or eager), its candidates and projection to the host.
    /// pending トークンを ids[0] に入れて block pass を走らせ (graph か eager)、候補と projection をホストに取る。
    fn blockPass(e: *Drafter, pending: u32) !void {
        const c = e.cfg;
        const one = [1]i32{@intCast(pending)};
        try e.ids.uploadAsync(0, std.mem.asBytes(&one), e.t.stream.handle);
        if (e.graphs) {
            if (e.block_exec == null) e.block_exec = try e.capture(.block, 0);
            try e.block_exec.?.launchOn(e.t.stream);
        } else try e.blockCompute();
        const m = c.block - 1;
        const words = m * 2 * c.top_k;
        try e.packed_all.downloadAsync(0, std.mem.sliceAsBytes(e.host_packed[0 .. e.t.world * words]), e.t.stream.handle);
        try e.proj.downloadAsync(0, std.mem.sliceAsBytes(e.host_proj[0 .. m * c.selector_rank]), e.t.stream.handle);
        try e.t.stream.synchronize();
    }

    /// Every graph up front: the block pass and the context updates of 1 .. block rows.
    /// 全 graph を先に作る: block pass と、1 .. block 行の context 更新。
    pub fn prewarm(e: *Drafter) !void {
        if (!e.graphs) return;
        if (e.block_exec == null) e.block_exec = try e.capture(.block, 0);
        var k: usize = 1;
        while (k <= e.cfg.block and k < e.tap_execs.len) : (k += 1) {
            if (e.tap_execs[k] == null) e.tap_execs[k] = try e.capture(.taps, @intCast(k));
        }
    }

    /// The graphs go before the communicator (NCCL holds a captured collective's plan until its graph is gone).
    /// graph は communicator より先に破棄する (NCCL は capture した collective の plan を graph が消えるまで持つ)。
    pub fn destroyGraphs(e: *Drafter) void {
        if (e.block_exec) |*x| x.deinit();
        e.block_exec = null;
        for (&e.tap_execs) |*t| if (t.*) |*x| {
            x.deinit();
            t.* = null;
        };
    }

    /// Up to `depth` greedy drafts for the positions after the pending token (at context_end). `samp` (a sampled
    /// request): scores over the temperature plus noise_weight x the target's Gumbel draw at that position, so the
    /// drafts lean to the tokens the target will draw.
    /// pending トークン (位置 context_end) の後の位置に、最大 `depth` 個の draft を greedy に出す。`samp` (サンプリング
    /// の要求) では score / temperature にその位置での target の Gumbel 乱数 x noise_weight を足し、target が引く
    /// トークンへ draft を寄せる。
    pub fn propose(e: *Drafter, pending: u32, depth_in: usize, confidence: f64, samp: ?lanes.Sampling, out: []u32) !usize {
        const c = e.cfg;
        const depth = @min(depth_in, c.block - 1, out.len);
        if (depth < 1 or e.context_end == 0) return 0;
        try e.blockPass(pending);
        const k = c.top_k;
        const W = e.t.world;
        const m = c.block - 1;
        const rk = c.selector_rank;
        var prev: usize = pending;
        var chain: f64 = 1.0;
        var nout: usize = 0;
        var vals: [8 * 32]f64 = undefined;
        var toks: [8 * 32]i64 = undefined;
        var order: [8 * 32]usize = undefined;
        var score: [32]f64 = undefined;
        var weighted: [512]f64 = undefined;
        if (W * k > vals.len or k > score.len or rk > weighted.len) return error.DrafterTooWide;
        for (0..depth) |d| {
            // candidates: every rank's k for this row, ordered by (-value, id), the first k kept
            // 候補: この行の全ランクの k 個を (-value, id) の順に並べ、先頭の k 個を残す
            const n = W * k;
            for (0..W) |r| {
                const base = (r * m + d) * 2 * k;
                for (0..k) |j| {
                    vals[r * k + j] = e.host_packed[base + j];
                    toks[r * k + j] = @as(i32, @bitCast(e.host_packed[base + k + j]));
                }
            }
            for (0..n) |i| order[i] = i;
            const Ctx = struct { v: []const f64, t: []const i64 };
            std.sort.pdq(usize, order[0..n], Ctx{ .v = vals[0..n], .t = toks[0..n] }, struct {
                fn lt(cx: Ctx, a: usize, b: usize) bool {
                    if (cx.v[a] != cx.v[b]) return cx.v[a] > cx.v[b];
                    return cx.t[a] < cx.t[b];
                }
            }.lt);
            // chain: score = value + edge_weight * succ[tok] . (pred[prev] * proj[d]) (fp64, a fixed order), argmax
            // chain: score = value + edge_weight * succ[tok] . (pred[prev] * proj[d]) (fp64、決まった順)、argmax
            for (0..rk) |i| weighted[i] = @as(f64, e.pred[prev * rk + i]) * bf16ToF64(e.host_proj[d * rk + i]);
            const sampled = samp != null and samp.?.temperature > 0;
            const temp: f64 = if (sampled) samp.?.temperature else 1.0;
            // argmax of the pick (score, plus the noise when sampled)
            // pick の argmax (score、サンプリング時は noise を足したもの)
            var best: usize = 0;
            var top: usize = 0; // argmax of the score / score の argmax
            var best_pick: f64 = -std.math.inf(f64);
            for (0..k) |j| {
                const tok: usize = @intCast(toks[order[j]]);
                var dot: f64 = 0;
                for (0..rk) |i| dot += @as(f64, e.succ[tok * rk + i]) * weighted[i];
                score[j] = (vals[order[j]] + edge_weight * dot) / temp;
                var pick = score[j];
                if (sampled) {
                    const u = lanes.sampling.uniform(samp.?.seed, @intCast(e.context_end + 1 + d), @intCast(tok));
                    pick += noise_weight * -@log(-@log(u));
                }
                if (j == 0 or pick > best_pick) {
                    best = j;
                    best_pick = pick;
                }
                if (score[j] > score[top]) top = j;
            }
            if (confidence > 0) {
                var sum: f64 = 0;
                for (score[0..k]) |s| sum += @exp(s - score[top]);
                chain *= @exp(score[best] - score[top]) / sum; // p[best]
                if (d > 0 and chain < confidence) break;
            }
            prev = @intCast(toks[order[best]]);
            out[nout] = @intCast(prev);
            nout += 1;
        }
        return nout;
    }
};

fn bf16ToF64(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn up16(d: *const cuda.Driver, x: []const u16) !Buf {
    return Buf.fromHost(d, std.mem.sliceAsBytes(x));
}

fn zeros(d: *const cuda.Driver, n: usize) !Buf {
    var b = try Buf.alloc(d, @max(n, 16));
    try b.fill8(0, null);
    return b;
}

fn freePacked(gpa: std.mem.Allocator, p: anytype) void {
    _ = gpa;
    // the host copies are freed with the process (load-time only)
    // ホスト側コピーはプロセス終了で解放する (読み込み時だけのもの)
    _ = p;
}

fn copyD2D(dst: u64, src: u64, bytes: usize, s: cuda.Stream) !void {
    try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
}
