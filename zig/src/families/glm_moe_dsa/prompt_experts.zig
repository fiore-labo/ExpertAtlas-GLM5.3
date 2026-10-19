//! The routed experts of prompt chunks wider than forward.max_rows, on prompt_experts.cu (TF_EXL3_PROMPT_EXPERTS=1).
//! The slots add with atomics in arrival order, so the bits may differ run to run (decode windows keep the
//! row-invariant experts kernel). `run` rounds the fp32 sums through bf16.
//! forward.max_rows より広い prompt チャンクの routed expert を prompt_experts.cu で (TF_EXL3_PROMPT_EXPERTS=1)。
//! スロットは atomic で到着順に足すので、実行ごとにビットが変わりうる (decode window は行に依存しない experts
//! カーネルのまま)。`run` は fp32 の和を bf16 に丸める。
const std = @import("std");
const cuda = @import("cuda");
const moe = @import("moe.zig");

const Buf = cuda.DeviceBuffer;

const ipm = 112; // TF_PE_GPM: routed pairs an item / item あたりの routed pair 数
const chunk_rows = 4096; // rows a pass at most / 1 パスの最大行数
const slab_bytes = 9 << 20; // the most bytes of out a down pass covers / down の 1 パスが受け持つ out の最大バイト数
const gu_threads = 512; // GU_NW (16 / GU_MT 1) warps / GU_NW (16 / GU_MT 1) 個の warp
// GU_STAGES x (rows + suh), more than the epilogue's 2 x 64 x 132 x 4
// GU_STAGES x (行 + suh)。epilogue の 2 x 64 x 132 x 4 より大きい
const gu_smem = 3 * (ipm * 128 * 2 + 2 * 128 * 2);
const dn_threads = 512;
// DN_STAGES x rows + DN_EROWS x DN_ES x 4
// DN_STAGES x 行 + DN_EROWS x DN_ES x 4
const dn_smem = 2 * ipm * 128 * 2 + 32 * 260 * 4;
const act_f32 = 1; // act_mode 1: SwiGLU in fp32 / act_mode 1: SwiGLU を fp32 で

comptime {
    std.debug.assert(gu_smem > 2 * 64 * 132 * 4);
}

const route_symbol = "_ZN8tf_exl3p12route_kernelEPKiiiPiS2_S2_ii";
const gateup_symbol = "_ZN8tf_exl3p20prompt_gateup_kernelILi1EEEvPK13__nv_bfloat16iPKiS5_S5_PKlS7_S5_PK6__halfSA_SA_SA_SA_PS8_iiiifiP6float4x";
const down_symbol = "_ZN8tf_exl3p18prompt_down_kernelILi1EEEvPK6__halfPKiS5_S5_PKlS5_S3_PKfPfiiiii";

/// Whether the kernels cover `ex`: gate and up of an expert at one width, every width 2 / 3 / 4 bits, I % 128, D % 256,
/// and the codebook these instantiations take (1 = mcg).
/// カーネルが `ex` を扱えるか: expert の gate と up が同じ幅、幅はどれも 2 / 3 / 4 ビット、I % 128、D % 256、
/// そしてこのインスタンスの codebook (1 = mcg)。
pub fn supported(gpa: std.mem.Allocator, ex: *const moe.Routed) !bool {
    if (ex.cb != 1 or ex.width % 128 != 0 or ex.dims % 256 != 0) return false;
    const E = ex.count;
    const k2 = try gpa.alloc(i32, 3 * E);
    defer gpa.free(k2);
    for (0..3) |p| try ex.k2[p].download(0, std.mem.sliceAsBytes(k2[p * E ..][0..E]));
    for (0..E) |e| {
        if (k2[e] != k2[E + e]) return false;
        for (0..3) |p| switch (k2[p * E + e]) {
            4, 6, 8 => {},
            else => return false,
        };
    }
    return true;
}

pub const Kernels = struct {
    module: cuda.Module,
    route: cuda.Function,
    gateup: cuda.Function,
    down: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_prompt_experts);
        var k: Kernels = .{ .module = m, .route = try m.function(route_symbol), .gateup = try m.function(gateup_symbol), .down = try m.function(down_symbol) };
        try k.gateup.allowDynamicShared(gu_smem);
        try k.down.allowDynamicShared(dn_smem);
        return k;
    }
};

/// Scratch for up to `rows` (<= chunk_rows) rows of `slots` slots.
/// `slots` スロットの行を最大 `rows` 行 (chunk_rows 以下) 扱うための作業領域。
pub const Scratch = struct {
    sorted: Buf,
    items: Buf,
    count: Buf,
    xd: Buf,
    rows: u32,
    slots: u32,
    max_items: u32,

    pub fn init(d: *const cuda.Driver, ex: *const moe.Routed, rows: u32, slots: u32) !Scratch {
        const r = @min(rows, chunk_rows);
        const n: usize = @as(usize, r) * slots;
        const mi: u32 = @intCast(n / ipm + @min(n, ex.count));
        var count = try Buf.alloc(d, 16);
        try count.fill8(0, null);
        return .{
            .sorted = try Buf.alloc(d, n * 4),
            .items = try Buf.alloc(d, 3 * @as(usize, mi) * 4),
            .count = count,
            .xd = try Buf.alloc(d, n * ex.width * 2),
            .rows = r,
            .slots = slots,
            .max_items = mi,
        };
    }
};

/// Column blocks (of 256) a down pass: the widest slab of out within slab_bytes.
/// down の 1 パスあたりの列ブロック数 (256 列単位): slab_bytes に収まる最も広い out の slab。
fn autoNcb(R: u32, D: u32) u32 {
    const blocks = D / 256;
    var best: u32 = 1;
    var dd: u32 = 1;
    while (dd <= blocks) : (dd += 1) {
        if (blocks % dd == 0 and @as(u64, R) * dd * 256 * 4 <= slab_bytes) best = dd;
    }
    return best;
}

/// out [R, D] fp32 = sum over slots of wts * expert(x), each row's sum rounded through bf16; x bf16 [R, D] with rows
/// `xstride` apart, pick int32 / wts fp32 [R, slots]. `round.roundBf16`: an in-place fp32 -> bf16 -> fp32 rounding of
/// n values on the stream. `do_round` false leaves the fp32 sums as they are (the caller folds the rounding into its
/// next step: Glue.raddToBf16).
/// out [R, D] fp32 = スロットについての wts * expert(x) の和。各行の和は bf16 に丸める。x は bf16 [R, D] で行は
/// `xstride` おき、pick int32 / wts fp32 [R, slots]。`round.roundBf16`: stream 上で n 個の値をその場で
/// fp32 -> bf16 -> fp32 に丸める。`do_round` が false なら fp32 の和のまま残す (呼び出し側が次の段で丸める:
/// Glue.raddToBf16)。
/// `half_sums`: out holds fp16 sums instead (red.add.f16x2 into [R, D] fp16 at `out`; do_round must be false).
/// `half_sums`: out は代わりに fp16 の和 (`out` の [R, D] fp16 に red.add.f16x2。do_round は false であること)。
pub fn run(ks: *const Kernels, s: cuda.Stream, x: u64, xstride: u32, pick: u64, wts: u64, ex: *const moe.Routed, sc: *const Scratch, out: u64, R: u32, round: anytype, do_round: bool, half_sums: bool) !void {
    if (half_sums and do_round) return error.Invalid;
    const ob: u64 = if (half_sums) 2 else 4; // bytes a value of out / out の値 1 つのバイト数
    const D = ex.dims;
    const I = ex.width;
    const S = sc.slots;
    var r0: u32 = 0;
    while (r0 < R) : (r0 += sc.rows) {
        const n = @min(sc.rows, R - r0);
        const pairs: u64 = @as(u64, n) * S;
        const items: u32 = @intCast(@min(sc.max_items, pairs / ipm + @min(pairs, ex.count)));
        const xs = x + @as(u64, r0) * xstride * 2;
        const pk = pick + @as(u64, r0) * S * 4;
        const wt = wts + @as(u64, r0) * S * 4;
        const o = out + @as(u64, r0) * D * ob;
        {
            var a: cuda.Args = .{};
            a.add(pk);
            a.add(@as(c_int, @intCast(pairs)));
            a.add(@as(c_int, @intCast(ex.count)));
            a.add(sc.sorted.ptr);
            a.add(sc.items.ptr);
            a.add(sc.count.ptr);
            a.add(@as(c_int, @intCast(items)));
            a.add(@as(c_int, 0)); // by_count off: items in expert order / by_count なし: item は expert 順
            try cuda.launch.launch(ks.route, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = 3 * ex.count * 4 }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(xs);
            a.add(@as(c_int, @intCast(xstride)));
            a.add(sc.sorted.ptr);
            a.add(sc.items.ptr);
            a.add(sc.count.ptr);
            a.add(ex.ptr[0].ptr);
            a.add(ex.ptr[1].ptr);
            a.add(ex.k2[0].ptr);
            a.add(ex.suh_g.ptr);
            a.add(ex.suh_u.ptr);
            a.add(ex.svh_g.ptr);
            a.add(ex.svh_u.ptr);
            a.add(ex.suh_d.ptr);
            a.add(sc.xd.ptr);
            a.add(@as(c_int, @intCast(D)));
            a.add(@as(c_int, @intCast(I)));
            // NS: the trellis row stride in tiles (gate and up stored apart)
            // NS: trellis の行の stride、タイル単位 (gate と up は別々に置く)
            a.add(@as(c_int, @intCast(I / 16)));
            a.add(@as(c_int, @intCast(S)));
            a.add(std.math.inf(f32));
            a.add(@as(c_int, act_f32));
            a.add(o);
            a.add(@as(c_longlong, @intCast(@as(u64, n) * D * ob / 16)));
            try cuda.launch.launch(ks.gateup, .{ .grid = .{ .x = I / 128, .y = items, .z = 1 }, .block = .{ .x = gu_threads, .y = 1, .z = 1 }, .shared = gu_smem }, s, &a);
        }
        {
            const ncb = autoNcb(n, D);
            var a: cuda.Args = .{};
            a.add(sc.xd.ptr);
            a.add(sc.sorted.ptr);
            a.add(sc.items.ptr);
            a.add(sc.count.ptr);
            a.add(ex.ptr[2].ptr);
            a.add(ex.k2[2].ptr);
            a.add(ex.svh_d.ptr);
            a.add(wt);
            a.add(o);
            a.add(@as(c_int, @intCast(I)));
            a.add(@as(c_int, @intCast(D)));
            a.add(@as(c_int, @intCast(S)));
            a.add(@as(c_int, @intCast(ncb)));
            a.add(@as(c_int, @intFromBool(half_sums))); // fp16 sums (TF_GLM53_PE_F16) / fp16 の和 (TF_GLM53_PE_F16)
            try cuda.launch.launch(ks.down, .{ .grid = .{ .x = items, .y = D / 256 / ncb, .z = 1 }, .block = .{ .x = dn_threads, .y = 1, .z = 1 }, .shared = dn_smem }, s, &a);
        }
        if (do_round) try round.roundBf16(s, o, @as(u64, n) * D);
    }
}
