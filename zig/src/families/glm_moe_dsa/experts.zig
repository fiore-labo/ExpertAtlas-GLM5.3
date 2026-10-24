//! EXL3 routed experts on CUDA (TF_GLM53_EXPERTS=tf): launches the kernels of zig/kernels/cuda/glm53/exl3_experts.cu.
//! CUDA 上の EXL3 routed expert (TF_GLM53_EXPERTS=tf): zig/kernels/cuda/glm53/exl3_experts.cu のカーネルを起動する。
const std = @import("std");
const cuda = @import("cuda");

/// GROUP_THREADS (one block groups every pick) and the group kernel's static shared bytes (warp_tot).
/// GROUP_THREADS (1 ブロックで全 pick をまとめる) と、group カーネルの静的共有メモリのバイト数 (warp_tot)。
const group_threads = 1024;
const group_static_smem = group_threads / 32 * 4;

/// The grouped_kernel<CB, NT, W, PF, LO, HI> instantiations exl3_experts.cu exports (codebook 1 = mcg, K2 in [2, 10]).
/// exl3_experts.cu が出す grouped_kernel<CB, NT, W, PF, LO, HI> の実体 (codebook 1 = mcg、K2 は [2, 10])。
const Grouped = struct { cb: u32, nt: u32, warps: u32, pf: u32, lo: u32, hi: u32, symbol: [:0]const u8 };
const grouped_variants = [_]Grouped{
    .{ .cb = 1, .nt = 8, .warps = 4, .pf = 1, .lo = 2, .hi = 10, .symbol = "_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi2ELi10EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiiii" },
    // a layer whose experts are all 4 bits (K2 8): the [8, 8] range
    // expert がすべて 4 bit (K2 8) の層: [8, 8] の範囲
    .{ .cb = 1, .nt = 8, .warps = 4, .pf = 1, .lo = 8, .hi = 8, .symbol = "_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiiii" },
};
const group_symbol = "_ZN8tf_exl3e12group_kernelEPKiPiS2_S2_iiii";
const rot_in_bf16_symbol = "_ZN8tf_exl3e13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
const gateup_symbol = "_ZN8tf_exl3e22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi";
const down_combine_symbol = "_ZN8tf_exl3e19down_combine_kernelEPKfPKiPK6__halfPfS1_S7_iiiii";

/// TF_GLM53_EXPERTS_GENERIC=1: every layer on the [2, 10] instance, none on the all-4-bit [8, 8] one. The two are the
/// same kernel with fewer widths compiled in: on GB10 they take the same time (1,410 / 1,411 us a TP4 gate/up
/// launch, 3 blocks a SM both); so a layer loaded all 4 bit may then take any width in a swap. Read once.
/// TF_GLM53_EXPERTS_GENERIC=1: どの層も [2, 10] のインスタンスで、全 4 bit 用の [8, 8] は使わない。両者は同じカーネルで
/// コンパイルする幅が少ないだけ: GB10 では同じ時間 (TP4 の gate/up 1 回 1,410 / 1,411 us、どちらも 1 SM に
/// 3 ブロック)。そのため全 4 bit で読み込んだ層も、入れ替えでどの幅でも受けられる。読むのは 1 回だけ。
pub fn genericOnly() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_GLM53_EXPERTS_GENERIC")) |e| std.mem.eql(u8, std.mem.span(e), "1") else false;
    return S.v.?;
}

/// experts_grouped.cuh's GroupedArgs (host side only, never passed to the device): pointers and the tile setting.
/// experts_grouped.cuh の GroupedArgs (ホスト側だけで使い、デバイスには渡さない): ポインタとタイル設定。
pub const GroupedArgs = struct {
    x0: u64,
    x1: u64,
    tp0: u64,
    tp1: u64,
    k2_0: u64,
    k2_1: u64,
    uids: u64,
    ucount: u64,
    members: u64,
    z: u64,
    K: u32,
    N: u32,
    P: u32,
    SK: u32,
    maxm: u32,
    slots: u32,
    nexp_max: u32,
    mats: u32,
    cb: u32,
    nt: u32,
    warps: u32,
    pf: u32,
    lo: u32,
    hi: u32,
    ns: u32 = 0,
};

fn ci(v: anytype) c_int {
    return @intCast(v);
}

pub const Kernels = struct {
    module: cuda.Module,
    group_fn: cuda.Function,
    rot_in_bf16: cuda.Function,
    gateup: cuda.Function,
    down_combine_fn: cuda.Function,
    grouped_fns: [grouped_variants.len]cuda.Function,
    optin: u32,

    /// `optin`: the device's sharedMemPerBlockOptin (the group kernel's dynamic shared memory past 48 KiB).
    /// `optin`: デバイスの sharedMemPerBlockOptin (group カーネルが 48 KiB を超えて動的共有メモリを使うため)。
    pub fn load(d: *const cuda.Driver, optin: u32) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_exl3_experts);
        errdefer m.unload();
        var k: Kernels = .{ .module = m, .group_fn = try m.function(group_symbol), .rot_in_bf16 = try m.function(rot_in_bf16_symbol), .gateup = try m.function(gateup_symbol), .down_combine_fn = try m.function(down_combine_symbol), .grouped_fns = undefined, .optin = optin };
        for (grouped_variants, 0..) |v, i| k.grouped_fns[i] = try m.function(v.symbol);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }

    /// uids / ucount / members [maxu, maxm] of R rows' picks [R, slots].
    /// R 行の pick [R, slots] から uids / ucount / members [maxu, maxm] を作る。
    pub fn group(k: *const Kernels, stream: cuda.Stream, pick: u64, uids: u64, ucount: u64, members: u64, rows: u32, slots: u32, experts: u32, maxm: u32) !void {
        if (experts > group_threads * 4) return error.TooManyExperts;
        if (slots > 32) return error.TooManySlots;
        const smem: u32 = rows * slots * 4;
        if (smem + group_static_smem > 48 * 1024) {
            const static: u32 = @intCast(try k.group_fn.attribute(.shared_size_bytes));
            const limit = k.optin - static;
            if (smem > limit) return error.GroupSharedMemory;
            if (smem > @as(u32, @intCast(try k.group_fn.attribute(.max_dynamic_shared_size_bytes)))) try k.group_fn.allowDynamicShared(limit);
        }
        var args: cuda.Args = .{};
        args.add(pick);
        args.add(uids);
        args.add(ucount);
        args.add(members);
        args.add(ci(rows));
        args.add(ci(slots));
        args.add(ci(experts));
        args.add(ci(maxm));
        try cuda.launch.launch(k.group_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = group_threads, .y = 1, .z = 1 }, .shared = smem }, stream, &args);
    }

    /// out0 / out1 [P, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)) of every routed slot's gate and up (x bf16).
    /// routed スロットごとの gate と up について、上の式で out0 / out1 を作る (x は bf16)。
    pub fn rotIn(k: *const Kernels, stream: cuda.Stream, x: u64, x_stride: u32, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: u32, kdim: u32, slots: u32, experts: u32) !void {
        if (kdim % 128 != 0) return error.BadRotIn;
        var args: cuda.Args = .{};
        args.add(x);
        args.add(ci(x_stride));
        args.add(pick);
        args.add(suh0);
        args.add(suh1);
        args.add(out0);
        args.add(out1);
        args.add(ci(kdim));
        args.add(ci(slots));
        args.add(ci(experts));
        try cuda.launch.launch(k.rot_in_bf16, .{ .grid = .{ .x = rows * slots, .y = kdim / 128, .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, stream, &args);
    }

    /// One grouped launch: Z [mats, SK, P, N] fp32 partials of the routed slots.
    /// grouped カーネル 1 回の起動: routed スロットの部分和 Z [mats, SK, P, N] fp32。
    pub fn grouped(k: *const Kernels, stream: cuda.Stream, a: GroupedArgs) !void {
        if (a.K % (16 * a.SK * a.warps) != 0 or a.N % (16 * a.nt) != 0) return error.UnevenSplit;
        if (a.ns != 0 and a.ns < a.N / 16) return error.BadRowStride;
        // the K2 ranges: [8, 8] (unless genericOnly), else [2, 10], else [2, 16]
        // K2 の範囲: [8, 8] (genericOnly でなければ)、だめなら [2, 10]、だめなら [2, 16]
        const lo: u32, const hi: u32 = if (a.lo == 8 and a.hi == 8 and !genericOnly()) .{ 8, 8 } else if (a.lo >= 2 and a.hi <= 10) .{ 2, 10 } else .{ 2, 16 };
        const idx = for (grouped_variants, 0..) |v, i| {
            if (v.cb == a.cb and v.nt == a.nt and v.warps == a.warps and v.pf == a.pf and v.lo == lo and v.hi == hi) break i;
        } else return error.UnportedGroupedVariant;
        const mt = (a.maxm + 15) / 16;
        var args: cuda.Args = .{};
        args.add(a.x0);
        args.add(a.x1);
        args.add(a.tp0);
        args.add(a.tp1);
        args.add(a.k2_0);
        args.add(a.k2_1);
        args.add(a.uids);
        args.add(a.ucount);
        args.add(a.members);
        args.add(a.z);
        args.add(ci(a.K));
        args.add(ci(a.N));
        args.add(ci(a.P));
        args.add(ci(a.SK));
        args.add(ci(a.maxm));
        args.add(ci(a.slots));
        args.add(ci(a.ns));
        try cuda.launch.launch(k.grouped_fns[idx], .{ .grid = .{ .x = a.nexp_max, .y = a.N / (16 * a.nt), .z = a.mats * a.SK * mt }, .block = .{ .x = a.warps * 32, .y = 1, .z = 1 } }, stream, &args);
    }

    /// xd [P, N] fp16: gate/up splits summed, rotated, * svh, SwiGLU (act_mode 0 bf16 roundings, 1 fp32),
    /// * suh_d, rotated.
    /// xd [P, N] fp16: gate/up の分割を足し、回転、* svh、SwiGLU (act_mode 0 は bf16 丸め、1 は fp32)、* suh_d、回転。
    pub fn gateupEpilogue(k: *const Kernels, stream: cuda.Stream, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: u32, p: u32, n: u32, sk: u32, slots: u32, experts: u32, limit: f32, act_mode: u32) !void {
        if (n % 128 != 0) return error.BadWidth;
        var args: cuda.Args = .{};
        args.add(z);
        args.add(pick);
        args.add(svh_g);
        args.add(svh_u);
        args.add(suh_d);
        args.add(xd);
        args.add(ci(p));
        args.add(ci(n));
        args.add(ci(sk));
        args.add(ci(experts));
        args.add(limit);
        args.add(ci(act_mode));
        try cuda.launch.launch(k.gateup, .{ .grid = .{ .x = rows * slots, .y = n / 128, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, stream, &args);
    }

    /// y [P, D] = down splits summed, rotated, * svh_d; out [R, D] fp32 = sum over slots in order of wts * y
    /// (one launch).
    /// y [P, D] = down の分割を足し、回転、* svh_d。out [R, D] fp32 = wts * y をスロット順に足した和 (1 回の起動)。
    pub fn downCombine(k: *const Kernels, stream: cuda.Stream, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, rows: u32, p: u32, d: u32, sk: u32, slots: u32, experts: u32) !void {
        if (slots > 32) return error.TooManySlots;
        if (d % 128 != 0) return error.BadWidth;
        var args: cuda.Args = .{};
        args.add(z);
        args.add(pick);
        args.add(svh_d);
        args.add(y);
        args.add(wts);
        args.add(out);
        args.add(ci(p));
        args.add(ci(d));
        args.add(ci(sk));
        args.add(ci(experts));
        args.add(ci(slots));
        try cuda.launch.launch(k.down_combine_fn, .{ .grid = .{ .x = rows, .y = d / 128, .z = 1 }, .block = .{ .x = 32 * slots, .y = 1, .z = 1 } }, stream, &args);
    }
};
