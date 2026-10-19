//! The kernels of kernels/cuda/glm53/native_ops.cu, loaded once; `Ops` (ops.zig) calls them.
//! kernels/cuda/glm53/native_ops.cu のカーネルを 1 回だけ読み込む。`Ops` (ops.zig) がこれを呼ぶ。
const std = @import("std");
const cuda = @import("cuda");

const RouterSumArgs = extern struct { part: u64, out: u64, total: c_int, ks: c_int, vec: c_int };
const ResidualArgs = extern struct { x: u64, xout: u64, g: u64, rows: c_int, d: c_int, world: c_int, rs: c_int, g_bf16: c_int, vec: c_int };
const EmbedB16Args = extern struct { ids: u64, w: u64, out: u64, rows: c_int, d: c_int };
const Swiglu2Args = extern struct { g: u64, u: u64, out: u64, rows: c_int, width: c_int };
const QropeArgs = extern struct { q: u64, inv: u64, qr: u64, pos: u64, rows: c_int, h: c_int, qr_stride: c_int };
const IqRopeArgs = extern struct { q: u64, inv: u64, pos: u64, rows: c_int };
const KvWriteArgs = extern struct { kva: u64, nw: u64, lc: u64, pos: u64, inv: u64, rows: c_int, kva_stride: c_int, dcp: c_int, rank: c_int, kv8: c_int, eps: f32, pmap: u64 };
const IkWriteArgs = extern struct { ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: c_int, dcp: c_int, rank: c_int, kv8: c_int, eps: f32, pmap: u64 };
const MergeLseArgs = extern struct { po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, rows: c_int, ht: c_int, h: c_int, nch: c_int };
const DcpCombineArgs = extern struct { orecv: u64, lrecv: u64, out: u64, rows: c_int, ss: c_int, ssl: c_int, h: c_int, world: c_int };
const AbsorbArgs = extern struct { q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, rows: c_int, h: c_int, rbk: c_int, qa_stride: c_int, qr_stride: c_int };
const ExpandArgs = extern struct { ol: u64, wv: u64, out: u64, rows: c_int, h: c_int, rbk: c_int };
const RouterPartArgs = extern struct { x: u64, w: u64, part: u64, m: c_int, x_stride: c_int, d: c_int, ne: c_int, ks: c_int };
const IndexScoresArgs = extern struct { q: u64, w: u64, ic: u64, pos: u64, out: u64, n: c_int, t: c_int, r0: c_int, dcp: c_int, rank: c_int, pack: c_int, rb: c_int, wscale: f32, qscale: f32, pmap: u64 };
const IqFp8Args = extern struct { q: u64, q8: u64, qs: u64, rows: c_int };
const IndexScores8Args = extern struct { q8: u64, qs: u64, w: u64, ic: u64, pos: u64, out: u64, n: c_int, t: c_int, r0: c_int, dcp: c_int, rank: c_int, pack: c_int, rb: c_int, wscale: f32, qscale: f32, pmap: u64 };
const SelectKeysArgs = extern struct { s: u64, out: u64, rows: c_int, nc: c_int, k: c_int };
const MergeArgs = extern struct { po: u64, pm: u64, pl: u64, out: u64, rows: c_int, h: c_int, nch: c_int };
const AttnArgs = extern struct { qa: u64, qr: u64, lc: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, rows: c_int, h: c_int, g: c_int, chk: c_int, dcp: c_int, rank: c_int, direct: c_int, scale: f32, ht: c_int, pmap: u64 };
const GemmArgs = extern struct { x: u64, w: u64, h: u64, svh: u64, out: u64, m: c_int, o_stride: c_int, k: c_int, n: c_int, out_bf16: c_int, scale: f32 };
// GM_SMEM: GM_STAGES = 3 stages of GM_BK = 32 columns (49,152 bytes: two blocks an SM)
// GM_SMEM: GM_BK = 32 列の段を GM_STAGES = 3 段 (49,152 バイト: SM あたり 2 ブロック)
const gemm_smem: u32 = 3 * (128 * 64 + 32 * 256);
const GroupSumsArgs = extern struct { x: u64, xs: u64, rows: c_int, x_stride: c_int, k: c_int };
const SwigluArgs = extern struct { gu: u64, out: u64, xs: u64, rows: c_int, w: c_int, limit: f32 };
const DconvArgs = extern struct { x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: c_int, d: c_int, gs: c_int, branch: c_int, has_res: c_int, vec: c_int };
const PrepArgs = extern struct { qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: c_int, stride: c_int, h: c_int, hkv: c_int, eps: f32 };
const DattnArgs = extern struct { q: u64, k: u64, v: u64, out: u64, pos: u64, window: c_int, nh: c_int, scale: f32 };
const dattn_smem: u32 = 5 * 16384 + 8192 + 2 * 1024; // DR_SMEM
const TopkArgs = extern struct { logits: u64, bias: u64, pick: u64, wts: u64, rows: c_int, scale: f32 };
const RmsArgs = extern struct { x: u64, w: u64, out: u64, xs: u64, rows: c_int, d: c_int, x_stride: c_int, o_stride: c_int, vec: c_int, eps: f32, rd: f32 };

comptime {
    std.debug.assert(@sizeOf(RouterSumArgs) == 32 and @sizeOf(ResidualArgs) == 48 and @sizeOf(EmbedB16Args) == 32 and @sizeOf(Swiglu2Args) == 32);
    // the KV kernels' structs with the pool's map last (native_ops.cu: the same fields in the same order)
    // KV カーネルの構造体。pool の表が最後 (native_ops.cu と同じフィールドを同じ順に)
    std.debug.assert(@offsetOf(KvWriteArgs, "pmap") == 64 and @sizeOf(KvWriteArgs) == 72);
    std.debug.assert(@offsetOf(IkWriteArgs, "pmap") == 72 and @sizeOf(IkWriteArgs) == 80);
    std.debug.assert(@offsetOf(IndexScoresArgs, "pmap") == 80 and @sizeOf(IndexScoresArgs) == 88);
    std.debug.assert(@offsetOf(IndexScores8Args, "pmap") == 88 and @sizeOf(IndexScores8Args) == 96);
    std.debug.assert(@offsetOf(AttnArgs, "pmap") == 112 and @sizeOf(AttnArgs) == 120);
}

/// The pool map's header before its table (native_ops.cu PoolMap): write_lo, write_hi, err, z (int32 each).
/// pool の表の前のヘッダ (native_ops.cu の PoolMap): write_lo、write_hi、err、z (それぞれ int32)。
pub const pool_map_header: usize = 16;

pub const Kernels = struct {
    module: cuda.Module,
    f_router_sum: cuda.Function,
    f_residual_add: cuda.Function,
    f_embed_b16: cuda.Function,
    f_swiglu2: cuda.Function,
    f_qrope: cuda.Function,
    f_iq_rope: cuda.Function,
    // the 16-byte path of f_rmsnorm for one shape (no sums): straight-line code
    // f_rmsnorm の 16 バイト経路を 1 つの形に限ったもの (sums なし)。分岐のないコード
    f_rmsnorm_d6144: cuda.Function,
    f_rmsnorm_d2048: cuda.Function,
    f_merge_lse_c8: cuda.Function, // f_merge_lse for 8 chunks / 8 チャンク用の f_merge_lse
    f_rmsnorm: cuda.Function,
    f_kv_write: cuda.Function,
    f_ik_write: cuda.Function,
    f_merge_lse: cuda.Function,
    f_dcp_combine: cuda.Function,
    f_topk: cuda.Function,
    f_topk_one: cuda.Function,
    f_absorb: cuda.Function,
    f_expand: cuda.Function,
    f_router_part: cuda.Function,
    f_index_scores: cuda.Function,
    // TF_GLM53_INDEX_FP8: the scores on the FP8 tensor-core product (other bits than f_index_scores)
    // TF_GLM53_INDEX_FP8: FP8 の tensor core の積で求めた点数 (f_index_scores とはビットが違う)
    f_index_scores8: cuda.Function,
    f_iq_fp8: cuda.Function,
    f_select_keys: cuda.Function,
    f_merge: cuda.Function,
    f_attn: [4]cuda.Function, // [warps 4 | 8][heads 8 | 16] / [warp 数 4 | 8][head 数 8 | 16]
    f_gemm: cuda.Function,
    // 8 warps, 16 heads: the one-chunk DCP partial into the exchange layout
    // 8 warp、16 head: 1 チャンクの DCP 部分結果を交換用の配置で書く
    f_attn_dl: cuda.Function,
    f_attn4: [4]cuda.Function, // the same over FP4 caches (TF_GLM53_KV=fp4) / FP4 キャッシュ上の同じもの
    f_attn_dl4: cuda.Function,
    // 8 warps, FP4 caches, 32 heads (TP=4 prompt chunks under DCP 2): the two programs of 16 in one block
    // 8 warp、FP4 キャッシュ、32 head (TP=4 の DCP 2 の prompt チャンク): 16 head の program 2 つを 1 ブロックで
    f_attn4x2: cuda.Function,
    // the one-chunk DCP partial into the exchange layout for 32 heads (TP=4), FP8 / FP4 caches: two groups of 16 a block
    // 32 head (TP=4) の 1 チャンクの DCP 部分結果を交換用の配置で、FP8 / FP4 キャッシュ: 1 ブロックに 16 head の組 2 つ
    f_attn_dlx2: [2]cuda.Function,
    /// The build loads the attention's PV operand by transposed byte loads (tf_glm53_n_pv_b8, sm_XXXa / f targets):
    /// FP4's 32 heads then run as two programs, not one block.
    /// attention の PV の入力を転置の 1 バイト読みで読むビルド (tf_glm53_n_pv_b8、sm_XXXa / f の target): FP4 の
    /// 32 head も 1 ブロックでなく program 2 つで走らせる。
    pv_b8: bool,
    f_group_sums: cuda.Function,
    f_swiglu: cuda.Function,
    f_dconv: cuda.Function,
    f_prep: cuda.Function,
    f_dattn_ring: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_native_ops);
        errdefer m.unload();
        const f_attn = [4]cuda.Function{ try m.function("tf_glm53_n_attn_w4h8"), try m.function("tf_glm53_n_attn_w4h16"), try m.function("tf_glm53_n_attn_w8h8"), try m.function("tf_glm53_n_attn_w8h16") };
        for (f_attn, 0..) |f, i| try f.allowDynamicShared(attnSmem(if (i < 2) 4 else 8));
        const f_gemm = try m.function("tf_glm53_n_gemm");
        try f_gemm.allowDynamicShared(gemm_smem);
        const f_attn_dl = try m.function("tf_glm53_n_attn_w8h16_dl");
        try f_attn_dl.allowDynamicShared(attnSmem(8));
        const pv_b8 = m.hasFunction("tf_glm53_n_pv_b8");
        const f_attn4 = [4]cuda.Function{ try m.function("tf_glm53_n_attn_w4h8_k4"), try m.function("tf_glm53_n_attn_w4h16_k4"), try m.function("tf_glm53_n_attn_w8h8_k4"), try m.function("tf_glm53_n_attn_w8h16_k4") };
        for (f_attn4, 0..) |f, i| try f.allowDynamicShared(attnSmem4(if (i < 2) 4 else 8, pv_b8));
        const f_attn_dl4 = try m.function("tf_glm53_n_attn_w8h16_dl_k4");
        try f_attn_dl4.allowDynamicShared(attnSmem4(8, pv_b8));
        const f_attn4x2 = try m.function("tf_glm53_n_attn_w8h16x2_k4");
        try f_attn4x2.allowDynamicShared(attnSmem4x2(pv_b8));
        const f_attn_dlx2 = [2]cuda.Function{ try m.function("tf_glm53_n_attn_w8h16x2_dl"), try m.function("tf_glm53_n_attn_w8h16x2_dl_k4") };
        try f_attn_dlx2[0].allowDynamicShared(attnSmemX2);
        try f_attn_dlx2[1].allowDynamicShared(attnSmem4x2(pv_b8));
        const f_dattn = try m.function("tf_glm53_n_dattn_ring");
        try f_dattn.allowDynamicShared(dattn_smem);
        return .{
            .module = m,
            .f_gemm = f_gemm,
            .f_attn_dl = f_attn_dl,
            .f_attn4 = f_attn4,
            .f_attn_dl4 = f_attn_dl4,
            .f_attn4x2 = f_attn4x2,
            .f_attn_dlx2 = f_attn_dlx2,
            .pv_b8 = pv_b8,
            .f_dattn_ring = f_dattn,
            .f_group_sums = try m.function("tf_glm53_n_group_sums"),
            .f_swiglu = try m.function("tf_glm53_n_swiglu"),
            .f_dconv = try m.function("tf_glm53_n_dconv"),
            .f_prep = try m.function("tf_glm53_n_prep"),
            .f_router_sum = try m.function("tf_glm53_n_router_sum"),
            .f_residual_add = try m.function("tf_glm53_n_residual_add"),
            .f_embed_b16 = try m.function("tf_glm53_n_embed_b16"),
            .f_swiglu2 = try m.function("tf_glm53_n_swiglu2"),
            .f_qrope = try m.function("tf_glm53_n_qrope"),
            .f_iq_rope = try m.function("tf_glm53_n_iq_rope"),
            .f_rmsnorm_d6144 = try m.function("tf_glm53_n_rmsnorm_d6144"),
            .f_rmsnorm_d2048 = try m.function("tf_glm53_n_rmsnorm_d2048"),
            .f_merge_lse_c8 = try m.function("tf_glm53_n_merge_lse_c8"),
            .f_rmsnorm = try m.function("tf_glm53_n_rmsnorm"),
            .f_kv_write = try m.function("tf_glm53_n_kv_write"),
            .f_ik_write = try m.function("tf_glm53_n_ik_write"),
            .f_merge_lse = try m.function("tf_glm53_n_merge_lse"),
            .f_dcp_combine = try m.function("tf_glm53_n_dcp_combine"),
            .f_topk = try m.function("tf_glm53_n_topk"),
            .f_topk_one = try m.function("tf_glm53_n_topk_one"),
            .f_absorb = try m.function("tf_glm53_n_absorb"),
            .f_expand = try m.function("tf_glm53_n_expand"),
            .f_router_part = try m.function("tf_glm53_n_router_part"),
            .f_index_scores = try m.function("tf_glm53_n_index_scores"),
            .f_index_scores8 = try m.function("tf_glm53_n_index_scores8"),
            .f_iq_fp8 = try m.function("tf_glm53_n_iq_fp8"),
            .f_select_keys = try m.function("tf_glm53_n_select_keys"),
            .f_merge = try m.function("tf_glm53_n_merge"),
            .f_attn = f_attn,
        };
    }

    fn go(f: cuda.Function, s: cuda.Stream, gx: u32, gy: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        a.add(args);
        try cuda.launch.launch(f, .{ .grid = .{ .x = gx, .y = gy, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn routerSum(k: *const Kernels, s: cuda.Stream, part: u64, out: u64, total: u32, ks: u32) !void {
        try go(k.f_router_sum, s, (total + 1023) / 1024, 1, RouterSumArgs{ .part = part, .out = out, .total = @intCast(total), .ks = @intCast(ks), .vec = @intFromBool(part % 16 == 0 and out % 16 == 0 and total % 4 == 0) });
    }

    pub fn residualAdd(k: *const Kernels, s: cuda.Stream, x: u64, xout: u64, g: u64, rows: u32, d: u32, world: u32, g_bf16: bool) !void {
        try go(k.f_residual_add, s, (d + 1023) / 1024, rows, ResidualArgs{ .x = x, .xout = xout, .g = g, .rows = @intCast(rows), .d = @intCast(d), .world = @intCast(world), .rs = @intCast(rows * d), .g_bf16 = @intFromBool(g_bf16), .vec = @intFromBool(x % 8 == 0 and xout % 8 == 0 and g % 16 == 0 and d % 4 == 0) });
    }

    pub fn embedB16(k: *const Kernels, s: cuda.Stream, ids: u64, w: u64, out: u64, rows: u32, d: u32) !void {
        try go(k.f_embed_b16, s, (d + 1023) / 1024, rows, EmbedB16Args{ .ids = ids, .w = w, .out = out, .rows = @intCast(rows), .d = @intCast(d) });
    }

    /// grid (gx, gy) of one warp
    /// 1 warp のブロックで grid (gx, gy)
    fn goWarp(f: cuda.Function, s: cuda.Stream, gx: u32, gy: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        a.add(args);
        try cuda.launch.launch(f, .{ .grid = .{ .x = gx, .y = gy, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// `qr_stride`: elements between QR's (row, head) records (64; 576 writes into the packed queries).
    /// `qr_stride`: QR の (row, head) レコード間の要素数 (64。576 なら pack した query に書く)。
    pub fn qrope(k: *const Kernels, s: cuda.Stream, q: u64, inv: u64, qr: u64, pos: u64, rows: u32, h: u32, qr_stride: u32) !void {
        try goWarp(k.f_qrope, s, rows, h, QropeArgs{ .q = q, .inv = inv, .qr = qr, .pos = pos, .rows = @intCast(rows), .h = @intCast(h), .qr_stride = @intCast(qr_stride) });
    }

    pub fn iqRope(k: *const Kernels, s: cuda.Stream, q: u64, inv: u64, pos: u64, rows: u32) !void {
        try goWarp(k.f_iq_rope, s, rows, 32, IqRopeArgs{ .q = q, .inv = inv, .pos = pos, .rows = @intCast(rows) });
    }

    /// rows of d bf16 (x_stride / o_stride elements apart); a block is 128 threads for d up to 2048, else 256
    /// The rmsnorm kernel's 16-byte path: aligned rows of a multiple of 8 values, at most 4 repetitions a thread.
    /// 行は d 個の bf16 (x_stride / o_stride 要素おき)。ブロックは d が 2048 以下なら 128 スレッド、超えれば 256
    /// rmsnorm カーネルの 16 バイト経路: 整列した 8 の倍数個の値の行で、スレッドあたりの繰り返しは 4 回まで。
    fn rmsVec(x: u64, x_stride: u32, w: u64, out: u64, o_stride: u32, d: u32, threads: u32) bool {
        return x % 16 == 0 and w % 16 == 0 and out % 16 == 0 and x_stride % 8 == 0 and o_stride % 8 == 0 and d % 8 == 0 and d <= 32 * threads;
    }

    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: u32, w: u64, out: u64, o_stride: u32, rows: u32, d: u32, eps: f32) !void {
        const block = std.math.ceilPowerOfTwo(u32, d) catch unreachable;
        var a: cuda.Args = .{};
        const threads: u32 = if (block <= 2048) 128 else 256;
        const vec = rmsVec(x, x_stride, w, out, o_stride, d, threads);
        a.add(RmsArgs{ .x = x, .w = w, .out = out, .xs = 0, .rows = @intCast(rows), .d = @intCast(d), .x_stride = @intCast(x_stride), .o_stride = @intCast(o_stride), .vec = @intFromBool(vec), .eps = eps, .rd = 1.0 / @as(f32, @floatFromInt(d)) });
        // the model's two widths have an entry of their own (the same arithmetic as the 16-byte path, unrolled)
        // モデルの 2 つの幅には専用の入口がある (16 バイト経路と同じ計算を展開したもの)
        const f = if (vec and d == 6144) k.f_rmsnorm_d6144 else if (vec and d == 2048) k.f_rmsnorm_d2048 else k.f_rmsnorm;
        try cuda.launch.launch(f, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    /// `kv8`: 0 bf16 rows, 1 FP8, 2 FP4 (TF_GLM53_KV=fp4).
    /// `kv8`: 0 は bf16 の行、1 は FP8、2 は FP4 (TF_GLM53_KV=fp4)。
    pub fn kvWrite(k: *const Kernels, s: cuda.Stream, kva: u64, kva_stride: u32, nw: u64, lc: u64, pos: u64, inv: u64, eps: f32, rows: u32, dcp: u32, rank: u32, kv8: u32, pmap: u64) !void {
        var a: cuda.Args = .{};
        a.add(KvWriteArgs{ .kva = kva, .nw = nw, .lc = lc, .pos = pos, .inv = inv, .rows = @intCast(rows), .kva_stride = @intCast(kva_stride), .dcp = @intCast(dcp), .rank = @intCast(rank), .kv8 = @intCast(kv8), .eps = eps, .pmap = pmap });
        // DCP 2: the owner's rows alone (t_own_row)
        // DCP 2: このランクが持つ行だけ (t_own_row)
        try cuda.launch.launch(k.f_kv_write, .{ .grid = .{ .x = if (dcp == 2) (rows + 1) / 2 else rows, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn ikWrite(k: *const Kernels, s: cuda.Stream, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: u32, dcp: u32, rank: u32, kv8: bool, pmap: u64) !void {
        var a: cuda.Args = .{};
        a.add(IkWriteArgs{ .ik = ik, .nw = nw, .nb = nb, .inv = inv, .ic = ic, .pos = pos, .rows = @intCast(rows), .dcp = @intCast(dcp), .rank = @intCast(rank), .kv8 = @intFromBool(kv8), .eps = 1e-6, .pmap = pmap });
        // DCP 2: the owner's rows alone (t_own_row)
        // DCP 2: このランクが持つ行だけ (t_own_row)
        try cuda.launch.launch(k.f_ik_write, .{ .grid = .{ .x = if (dcp == 2) (rows + 1) / 2 else rows, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    fn go128(f: cuda.Function, s: cuda.Stream, gx: u32, gy: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        a.add(args);
        try cuda.launch.launch(f, .{ .grid = .{ .x = gx, .y = gy, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn mergeLse(k: *const Kernels, s: cuda.Stream, po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, r: u32, ht: u32, h: u32, nch: u32) !void {
        if (po % 16 != 0 or osend % 8 != 0) return error.UnalignedNativeOperand;
        try go128(if (nch == 8) k.f_merge_lse_c8 else k.f_merge_lse, s, r, ht, MergeLseArgs{ .po = po, .pm = pm, .pl = pl, .osend = osend, .lsend = lsend, .rows = @intCast(r), .ht = @intCast(ht), .h = @intCast(h), .nch = @intCast(nch) });
    }

    pub fn dcpCombine(k: *const Kernels, s: cuda.Stream, orecv: u64, lrecv: u64, out: u64, r: u32, ss: u32, ssl: u32, h: u32, world: u32) !void {
        if (orecv % 8 != 0 or out % 8 != 0 or ss % 4 != 0 or world > 8) return error.UnalignedNativeOperand;
        try go128(k.f_dcp_combine, s, r, h, DcpCombineArgs{ .orecv = orecv, .lrecv = lrecv, .out = out, .rows = @intCast(r), .ss = @intCast(ss), .ssl = @intCast(ssl), .h = @intCast(h), .world = @intCast(world) });
    }

    pub fn topk(k: *const Kernels, s: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, n: u32, scale: f32) !void {
        // from 128 rows the entry whose blocks do less work in all is the faster one; below, the one without a
        // hand-over inside the block (topk_body)
        // 128 行からはブロックの仕事の合計が少ない入口が速い。それより下では、ブロック内の受け渡しがない方が速い
        // (topk_body)
        try go(if (n >= 128) k.f_topk_one else k.f_topk, s, n, 1, TopkArgs{ .logits = logits, .bias = bias, .pick = pick, .wts = wts, .rows = @intCast(n), .scale = scale });
    }

    /// Row blocks of RBK = min(r, 16) rows; the kernels read the weights and the 8-wide row slices by 16-byte loads.
    /// 行ブロックは RBK = min(r, 16) 行。カーネルは重みと幅 8 の行の切片を 16 バイトずつ読む。
    fn goRows(f: cuda.Function, s: cuda.Stream, h: u32, blocks: u32, r: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        a.add(args);
        const rbk: u32 = if (r > 16) 16 else r;
        try cuda.launch.launch(f, .{ .grid = .{ .x = h, .y = blocks, .z = (r + rbk - 1) / rbk }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// `qa_stride` / `qr_stride`: elements between the (row, head) records of QA and QR (512 / 64; 576 / 576 with
    /// qr = qa + 512 elements writes the packed queries of the DCP exchange in place).
    /// `qa_stride` / `qr_stride`: QA と QR の (row, head) レコード間の要素数 (512 / 64。576 / 576 で qr = qa + 512
    /// 要素なら、DCP の交換用に pack した query へその場で書く)。
    pub fn absorb(k: *const Kernels, s: cuda.Stream, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, r: u32, h: u32, qa_stride: u32, qr_stride: u32) !void {
        if (wk % 16 != 0) return error.UnalignedNativeOperand;
        try goRows(k.f_absorb, s, h, 16, r, AbsorbArgs{ .q = q, .wk = wk, .inv = inv, .qa = qa, .qr = qr, .pos = pos, .rows = @intCast(r), .h = @intCast(h), .rbk = @intCast(@min(r, 16)), .qa_stride = @intCast(qa_stride), .qr_stride = @intCast(qr_stride) });
    }

    pub fn expand(k: *const Kernels, s: cuda.Stream, ol: u64, wv: u64, out: u64, r: u32, h: u32) !void {
        if (wv % 16 != 0 or ol % 16 != 0) return error.UnalignedNativeOperand;
        try goRows(k.f_expand, s, h, 16, r, ExpandArgs{ .ol = ol, .wv = wv, .out = out, .rows = @intCast(r), .h = @intCast(h), .rbk = @intCast(@min(r, 16)) });
    }

    /// part[s, m, ne] fp32 = x bf16 [m, d] (rows x_stride apart) @ w^T over K slice s of ks; d / ks a multiple of 64.
    /// part[s, m, ne] fp32 = x bf16 [m, d] (行は x_stride おき) @ w^T、K を ks 分割した切片 s で。d / ks は 64 の倍数。
    pub fn routerPart(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: u32, w: u64, part: u64, m: u32, d: u32, ne: u32, ks: u32) !void {
        if (x % 16 != 0 or w % 16 != 0 or (m > 1 and x_stride % 8 != 0) or d % (ks * 64) != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(RouterPartArgs{ .x = x, .w = w, .part = part, .m = @intCast(m), .x_stride = @intCast(x_stride), .d = @intCast(d), .ne = @intCast(ne), .ks = @intCast(ks) });
        try cuda.launch.launch(k.f_router_part, .{ .grid = .{ .x = (m + 15) / 16, .y = (ne + 31) / 32, .z = ks }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// FP8 index keys (144-byte rows) only; `out` int64 (pack) or int32 [n, t].
    /// FP8 の index key (144 バイトの行) のみ。`out` は int64 (pack) か int32 の [n, t]。
    /// `mode`: 0 int32 order words, 1 packed int64 scores, 2 the order words and their top 16 bits after them
    /// (IndexMode; topkWordsAny16 reads them).
    /// `mode`: 0 は int32 の順序語、1 は packed の int64 の点数、2 は順序語とその後ろに上位 16 ビット (IndexMode。
    /// topkWordsAny16 が読む)。
    pub fn indexScores(k: *const Kernels, s: cuda.Stream, q: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32, dcp: u32, rank: u32, mode: u32, wscale: f32, qscale: f32, pmap: u64) !void {
        if (q % 16 != 0 or ic % 16 != 0) return error.UnalignedNativeOperand;
        // rows a block: the block's 128 key rows are staged once for all of them. Only when there is more work than the
        // GPU holds at once (48 SMs x ~4 blocks): with few blocks (a decode window over a short context) rows side by
        // side finish sooner than rows in sequence
        // ブロックあたりの行数: ブロックの key 128 行を 1 回置けば全行で使える。ただし GPU が一度に抱えられる以上の仕事
        // (48 SM x 約 4 ブロック) があるときだけ。ブロックが少ないとき (短い文脈での decode window) は、行を並べて
        // 走らせる方が順に処理するより早く終わる
        const rb: u32 = if (n * ((t + 127) / 128) <= 256) 1 else @min(n, 16);
        try go128(k.f_index_scores, s, (n + rb - 1) / rb, (t + 127) / 128, IndexScoresArgs{ .q = q, .w = w, .ic = ic, .pos = pos, .out = out, .n = @intCast(n), .t = @intCast(t), .r0 = @intCast(r0), .dcp = @intCast(dcp), .rank = @intCast(rank), .pack = @intCast(mode), .rb = @intCast(rb), .wscale = wscale, .qscale = qscale, .pmap = pmap });
    }

    /// TF_GLM53_INDEX_FP8: the index queries q [rows, 32, 128] bf16 as e4m3 codes in fragment order (q8 [rows, 4096])
    /// with a scale a (row, head) (qs [rows, 32]).
    /// TF_GLM53_INDEX_FP8: index の query q [rows, 32, 128] bf16 を fragment 順の e4m3 コード (q8 [rows, 4096]) と
    /// (row, head) ごとのスケール (qs [rows, 32]) にする。
    pub fn iqFp8(k: *const Kernels, s: cuda.Stream, q: u64, q8: u64, qs: u64, rows: u32) !void {
        try go128(k.f_iq_fp8, s, rows, 32, IqFp8Args{ .q = q, .q8 = q8, .qs = qs, .rows = @intCast(rows) });
    }

    /// `indexScores` over `iqFp8`'s codes: the dot on the FP8 tensor-core product. The scores differ from
    /// `indexScores` by the query's rounding to e4m3.
    /// `iqFp8` のコードに対する `indexScores`: 内積は FP8 の tensor core の積で取る。点数は query を e4m3 に丸めた
    /// 分だけ `indexScores` と違う。
    pub fn indexScores8(k: *const Kernels, s: cuda.Stream, q8: u64, qs: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32, dcp: u32, rank: u32, pack: bool, wscale: f32, qscale: f32, pmap: u64) !void {
        if (q8 % 16 != 0 or ic % 16 != 0) return error.UnalignedNativeOperand;
        const rb: u32 = if (n * ((t + 127) / 128) <= 256) 1 else @min(n, 16);
        try go128(k.f_index_scores8, s, (n + rb - 1) / rb, (t + 127) / 128, IndexScores8Args{ .q8 = q8, .qs = qs, .w = w, .ic = ic, .pos = pos, .out = out, .n = @intCast(n), .t = @intCast(t), .r0 = @intCast(r0), .dcp = @intCast(dcp), .rank = @intCast(rank), .pack = @intFromBool(pack), .rb = @intCast(rb), .wscale = wscale, .qscale = qscale, .pmap = pmap });
    }

    /// out int32 [n, kk] = each row's kk best columns (ascending) of the int32 order words sc [n, t].
    /// out int32 [n, kk] = int32 の順序語 sc [n, t] の各行で上位 kk 列 (昇順)。
    pub fn selectKeys(k: *const Kernels, s: cuda.Stream, sc: u64, out: u64, n: u32, t: u32, kk: u32) !void {
        try go(k.f_select_keys, s, n, 1, SelectKeysArgs{ .s = sc, .out = out, .rows = @intCast(n), .nc = @intCast(t), .k = @intCast(kk) });
    }

    pub fn merge(k: *const Kernels, s: cuda.Stream, po: u64, pm: u64, pl: u64, out: u64, r: u32, h: u32, nch: u32) !void {
        if (po % 16 != 0 or out % 8 != 0) return error.UnalignedNativeOperand;
        try go128(k.f_merge, s, r, h, MergeArgs{ .po = po, .pm = pm, .pl = pl, .out = out, .rows = @intCast(r), .h = @intCast(h), .nch = @intCast(nch) });
    }

    /// AT_SMEM(W): the key tiles (two for W 4, one for W 8), bf16(p), the max / sum partials.
    /// AT_SMEM(W): key のタイル (W 4 で 2 枚、W 8 で 1 枚)、bf16(p)、max / sum の部分結果。
    fn attnSmem(w: u32) u32 {
        const tiles: u32 = if (w == 4) 2 else 1;
        return tiles * 8 * w * 656 + 16 * 8 * w * 2 + 2 * 16 * w * 4;
    }

    /// The FP4 entries' AT_SMEM_K(W, 1): with `p4` (TF_KV4P, the builds with pv_b8) tiles of the cache's 432-byte rows
    /// (W 8: 30,720 bytes, three blocks an SM); else attnSmem plus the tiles' block scales (32 bytes a key).
    /// FP4 の実体の AT_SMEM_K(W, 1): `p4` (TF_KV4P、pv_b8 のあるビルド) ならキャッシュの 432 バイトの行のタイル
    /// (W 8 で 30,720 バイト、1 SM に 3 ブロック)。それ以外は attnSmem にタイルのブロックのスケール (key あたり 32 バイト)。
    fn attnSmem4(w: u32, p4: bool) u32 {
        const tiles: u32 = if (w == 4) 2 else 1;
        if (p4) return tiles * 8 * w * 432 + 16 * 8 * w * 2 + 2 * 16 * w * 4;
        return attnSmem(w) + tiles * 8 * w * 32;
    }

    /// AT_SMEM_G(8, 2): two key tiles of 64 (the 512-thread entries stage tile ti while tile ti - 1 computes), two
    /// groups' bf16(p) and max / sum partials; the FP4 entries add the two tiles' block scales.
    /// AT_SMEM_G(8, 2): 64 key のタイル 2 枚 (512 スレッドの実体はタイル ti - 1 の計算中にタイル ti を読み込む)、
    /// 2 組分の bf16(p) と max / sum の部分結果。FP4 の実体は 2 枚分のブロックのスケールを足す。
    const attnSmemX2: u32 = 2 * 64 * 656 + 2 * (16 * 64 * 2 + 2 * 16 * 8 * 4);
    fn attnSmem4x2(p4: bool) u32 {
        return if (p4) 2 * 64 * 432 + 2 * (16 * 64 * 2 + 2 * 16 * 8 * 4) else attnSmemX2 + 2 * 64 * 32;
    }

    /// The shapes the native attention covers: FP8 cache, 8 or 16 heads a program, tiles of 32 or 64 keys.
    /// native attention が扱う形: FP8 キャッシュ、プログラムあたり 8 か 16 head、key 32 か 64 個のタイル。
    /// 8 heads, or a multiple of 16 up to 64 (programs of 16 heads along the grid's z: TP=4 with DCP=2 gathers 32).
    /// 8 head、または 16 の倍数で 64 まで (grid の z 方向に 16 head ずつの program。TP=4 で DCP=2 なら 32 を集める)。
    pub fn attnOk(nh: u32, kt: u32) bool {
        return (nh == 8 or (nh % 16 == 0 and nh <= 64)) and (kt == 32 or kt == 64);
    }

    /// Attention over key chunks: g = 0 takes qa / qr of `h` heads; else every gathered query head (g members of h
    /// heads: qa = QALL).
    /// key チャンク上の attention: g = 0 は `h` head の qa / qr、それ以外は集めた全 query head (h head の g メンバー:
    /// qa = QALL)。
    pub fn attn(k: *const Kernels, s: cuda.Stream, qa: u64, qr: u64, lc: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, r: u32, h: u32, g: u32, nch: u32, chk: u32, kt: u32, dcp: u32, rank: u32, direct: bool, scale: f32, kv4: bool, pmap: u64) !void {
        const nh = if (g == 0) h else g * h;
        if (!attnOk(nh, kt) or qa % 16 != 0 or (g == 0 and qr % 16 != 0) or lc % 16 != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(AttnArgs{ .qa = qa, .qr = qr, .lc = lc, .tok = tok, .cnt = cnt, .pos = pos, .po = po, .pm = pm, .pl = pl, .rows = @intCast(r), .h = @intCast(h), .g = @intCast(g), .chk = @intCast(chk), .dcp = @intCast(dcp), .rank = @intCast(rank), .direct = @intFromBool(direct), .scale = scale, .ht = @intCast(nh), .pmap = pmap });
        const per: u32 = if (nh == 8) 8 else 16;
        const fi: usize = (if (kt == 64) @as(usize, 2) else 0) + (if (per == 16) @as(usize, 1) else 0);
        // FP4, 32 heads, tiles of 64, without pv_b8: one block a row and chunk stages each key tile once for both
        // programs (the same bytes as two launches along z; FP8's staging is a plain copy and gains nothing). With
        // pv_b8 the two programs are faster (bench_pair attn4: 8,706 against 8,957 us)
        // FP4、32 head、64 のタイルで pv_b8 がないとき: 行とチャンクごとに 1 ブロックが key のタイルを両方の program の
        // ために 1 回だけ置く (z 方向の 2 つと同じバイト。FP8 のタイルは単なるコピーなので効果がない)。pv_b8 があれば
        // program 2 つの方が速い (bench_pair attn4: 8,706 対 8,957 us)
        if (kv4 and nh == 32 and kt == 64 and !k.pv_b8) {
            return cuda.launch.launch(k.f_attn4x2, .{ .grid = .{ .x = r, .y = nch, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 }, .shared = attnSmem4x2(k.pv_b8) }, s, &a);
        }
        const f = if (kv4) k.f_attn4[fi] else k.f_attn[fi];
        try cuda.launch.launch(f, .{ .grid = .{ .x = r, .y = nch, .z = nh / per }, .block = .{ .x = kt * 4, .y = 1, .z = 1 }, .shared = if (kv4) attnSmem4(kt / 8, k.pv_b8) else attnSmem(kt / 8) }, s, &a);
    }

    /// The gathered-query attention of one chunk (2,048 keys, tiles of 64) of g * h = 16 heads and the LSE merge in one
    /// launch: the normalized partials into osend [g, r, h, 512] bf16 and their log-sum-exps into lsend [g, r, h] fp32.
    /// 集めた query の 1 チャンク (key 2,048 個、64 のタイル、g * h = 16 head) の attention と LSE の merge を 1 回で:
    /// 正規化した部分結果を osend [g, r, h, 512] bf16 に、その log-sum-exp を lsend [g, r, h] fp32 に書く。
    pub fn attnDl(k: *const Kernels, s: cuda.Stream, qall: u64, lc: u64, tok: u64, cnt: u64, pos: u64, osend: u64, lsend: u64, r: u32, h: u32, g: u32, dcp: u32, rank: u32, scale: f32, kv4: bool, pmap: u64) !void {
        if ((g * h != 16 and g * h != 32) or qall % 16 != 0 or lc % 16 != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(AttnArgs{ .qa = qall, .qr = 0, .lc = lc, .tok = tok, .cnt = cnt, .pos = pos, .po = osend, .pm = 0, .pl = lsend, .rows = @intCast(r), .h = @intCast(h), .g = @intCast(g), .chk = 2048, .dcp = @intCast(dcp), .rank = @intCast(rank), .direct = 0, .scale = scale, .ht = @intCast(g * h), .pmap = pmap });
        // 32 heads (TP=4): FP4 without pv_b8 runs both groups of 16 in one block of 512 threads, each key tile staged
        // once; FP8, and FP4 with pv_b8, run them as two programs along z (FP8's staging is a plain copy: one block
        // measured 2 % slower; FP4 with pv_b8 8,507 against 8,704 us). All give the bytes of attnDcpAll + mergeLse
        // (bench_pair attn4 DLCHK)
        // 32 head (TP=4): pv_b8 のない FP4 は 16 head の組 2 つを 512 スレッドの 1 ブロックで (key のタイルは 1 回
        // だけ置く)。FP8 と pv_b8 のある FP4 は z 方向の program 2 つで (FP8 のタイルは単なるコピーで、1 ブロックに
        // すると 2% 遅かった。pv_b8 のある FP4 は 8,507 対 8,704 us)。どれも attnDcpAll + mergeLse と同じバイト
        // (bench_pair attn4 DLCHK)
        if (g * h == 32 and kv4 and !k.pv_b8) return cuda.launch.launch(k.f_attn_dlx2[1], .{ .grid = .{ .x = r, .y = 1, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 }, .shared = attnSmem4x2(k.pv_b8) }, s, &a);
        if (g * h == 32) return cuda.launch.launch(if (kv4) k.f_attn_dl4 else k.f_attn_dl, .{ .grid = .{ .x = r, .y = 1, .z = 2 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = if (kv4) attnSmem4(8, k.pv_b8) else attnSmem(8) }, s, &a);
        try cuda.launch.launch(if (kv4) k.f_attn_dl4 else k.f_attn_dl, .{ .grid = .{ .x = r, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = if (kv4) attnSmem4(8, k.pv_b8) else attnSmem(8) }, s, &a);
    }

    /// out [m, n] (fp32 or bf16, rows o_stride apart) = ((xh fp16 [m, kk] @ w fp16 [kk, n]) @ H) scale svh; kk a
    /// multiple of 32, n of 128.
    /// out [m, n] (fp32 か bf16、行は o_stride おき) = ((xh fp16 [m, kk] @ w fp16 [kk, n]) @ H) scale svh。kk は 32
    /// の倍数、n は 128 の倍数。
    pub fn gemm(k: *const Kernels, s: cuda.Stream, xh: u64, w: u64, had: u64, svh: u64, out: u64, out_bf16: bool, m: u32, o_stride: u32, kk: u32, n: u32, scale: f32) !void {
        if (xh % 16 != 0 or w % 16 != 0 or had % 16 != 0 or out % 16 != 0 or kk % 32 != 0 or n % 128 != 0 or o_stride % 8 != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(GemmArgs{ .x = xh, .w = w, .h = had, .svh = svh, .out = out, .m = @intCast(m), .o_stride = @intCast(o_stride), .k = @intCast(kk), .n = @intCast(n), .out_bf16 = @intFromBool(out_bf16), .scale = scale });
        try cuda.launch.launch(k.f_gemm, .{ .grid = .{ .x = ((m + 127) / 128) * (n / 128), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = gemm_smem }, s, &a);
    }

    // ---- the drafter's kernels / drafter のカーネル ----

    /// rmsnorm with the 64-value group sums of the output into xs fp32 [rows, d / 64]; d a multiple of 8 * the
    /// block's threads (every thread then runs the same number of repetitions).
    /// rmsnorm に加え、出力の 64 値ごとのグループ和を xs fp32 [rows, d / 64] に書く。d は 8 * ブロックのスレッド数の
    /// 倍数 (どのスレッドも同じ回数だけ繰り返す)。
    pub fn rmsnormSums(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: u32, w: u64, out: u64, o_stride: u32, xs: u64, rows: u32, d: u32, eps: f32) !void {
        const block = std.math.ceilPowerOfTwo(u32, d) catch unreachable;
        const threads: u32 = if (block <= 2048) 128 else 256;
        if (d % (8 * threads) != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(RmsArgs{ .x = x, .w = w, .out = out, .xs = xs, .rows = @intCast(rows), .d = @intCast(d), .x_stride = @intCast(x_stride), .o_stride = @intCast(o_stride), .vec = @intFromBool(rmsVec(x, x_stride, w, out, o_stride, d, threads)), .eps = eps, .rd = 1.0 / @as(f32, @floatFromInt(d)) });
        try cuda.launch.launch(k.f_rmsnorm, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    pub fn groupSums(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: u32, xs: u64, m: u32, kk: u32) !void {
        try go128(k.f_group_sums, s, m, (kk / 64 + 15) / 16, GroupSumsArgs{ .x = x, .xs = xs, .rows = @intCast(m), .x_stride = @intCast(x_stride), .k = @intCast(kk) });
    }

    /// w a multiple of 512.
    /// w は 512 の倍数。
    pub fn swiglu(k: *const Kernels, s: cuda.Stream, gu: u64, out: u64, xs: u64, rows: u32, w: u32, limit: f32) !void {
        try go128(k.f_swiglu, s, rows, w / 512, SwigluArgs{ .gu = gu, .out = out, .xs = xs, .rows = @intCast(rows), .w = @intCast(w), .limit = limit });
    }

    pub fn dconv(k: *const Kernels, s: cuda.Stream, x: u64, dyn: u64, base: u64, res: ?u64, out: u64, rows: u32, d: u32, gs: u32, branch: u32) !void {
        try go128(k.f_dconv, s, rows, (d + 1023) / 1024, DconvArgs{ .x = x, .dyn = dyn, .base = base, .res = res orelse 0, .out = out, .rows = @intCast(rows), .d = @intCast(d), .gs = @intCast(gs), .branch = @intCast(branch), .has_res = @intFromBool(res != null), .vec = @intFromBool(x % 16 == 0 and base % 16 == 0 and out % 16 == 0 and (res orelse 0) % 16 == 0 and d % 8 == 0 and gs % 8 == 0) });
    }

    /// head dims 128 (HALF 64) only.
    /// head の次元は 128 (HALF 64) のみ。
    pub fn prep(k: *const Kernels, s: cuda.Stream, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: u32, stride: u32, eps: f32, heads: u32, kvh: u32) !void {
        try goWarp(k.f_prep, s, rows, heads + 2 * kvh, PrepArgs{ .qkv = qkv, .qn = qn, .kn = kn, .cos = cos, .sin = sin, .qo = qo, .ko = ko, .vo = vo, .rows = @intCast(rows), .stride = @intCast(stride), .h = @intCast(heads), .hkv = @intCast(kvh), .eps = eps });
    }

    /// The shape the native drafter attention covers (the DFlash2 drafter's).
    /// native の drafter attention が扱う形 (DFlash2 drafter のもの)。
    pub fn dattnOk(n: u32, g: u32, hd: u32, ring: u32, causal: bool) bool {
        return n == 8 and g == 8 and hd == 128 and ring == 4096 and !causal;
    }

    pub fn dattnRing(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, out: u64, pos: u64, window: u32, scale: f32, nh: u32, kvh: u32) !void {
        if (q % 16 != 0 or kc % 16 != 0 or vc % 16 != 0) return error.UnalignedNativeOperand;
        var a: cuda.Args = .{};
        a.add(DattnArgs{ .q = q, .k = kc, .v = vc, .out = out, .pos = pos, .window = @intCast(window), .nh = @intCast(nh), .scale = scale });
        try cuda.launch.launch(k.f_dattn_ring, .{ .grid = .{ .x = kvh, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 }, .shared = dattn_smem }, s, &a);
    }

    pub fn swiglu2(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, out: u64, rows: u32, width: u32) !void {
        try go(k.f_swiglu2, s, (width + 1023) / 1024, rows, Swiglu2Args{ .g = g, .u = u, .out = out, .rows = @intCast(rows), .width = @intCast(width) });
    }
};
