//! EXL3 linears on CUDA: picks and launches the kernels of zig/kernels/cuda/glm53/exl3_linear.cu.
//! CUDA 上の EXL3 linear: zig/kernels/cuda/glm53/exl3_linear.cu のカーネルを選んで起動する。
const std = @import("std");
const cuda = @import("cuda");

pub const max_jobs = 3;

pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

/// The kernels' Job / Jobs / RotJobs, byte for byte (passed by value).
/// カーネル側の Job / Jobs / RotJobs とバイト単位で同じ (値渡し)。
pub const Job = extern struct {
    xh: u64,
    T: u64,
    stride_k: i64,
    stride_nb: i64,
    svh: u64,
    bias: u64 = 0,
    y: u64,
    Z: u64 = 0,
    counters: u64,
    y_dtype: c_int,
    N: c_int,
    SK: c_int,
    start: c_int = 0,
};
pub const Jobs = extern struct { j: [max_jobs]Job = undefined, n: c_int = 0 };
pub const RotJobs = extern struct { suh: [max_jobs]u64 = .{ 0, 0, 0 }, xh: [max_jobs]u64 = .{ 0, 0, 0 } };

comptime {
    std.debug.assert(@sizeOf(Job) == 88 and @sizeOf(Jobs) == 272 and @sizeOf(RotJobs) == 48);
}

/// The kernels exl3_linear.cu exports: (K2, codebook) (10, 2) = 5 bits mul1, (12, 2) = 6 bits mul1,
/// (8, 1) = 4 bits mcg, (8, 2) = 4 bits mul1; V 1 (16-byte loads); WK 2 / 4 / 8 warps (tune.zig picks); G 1 / 2.
/// exl3_linear.cu が出すカーネル: (K2, codebook) が (10, 2) = 5 bit mul1、(12, 2) = 6 bit mul1、(8, 1) = 4 bit mcg、
/// (8, 2) = 4 bit mul1。V 1 (16 バイト単位の読み込み)、WK 2 / 4 / 8 warp (tune.zig がこの中から選ぶ)、G 1 / 2。
const Variant = struct { k2: u32, cb: u32, wk: u32, g: u32, symbol: [:0]const u8 };
const Width = struct { k2: u32, cb: u32 };
const widths = [_]Width{ .{ .k2 = 10, .cb = 2 }, .{ .k2 = 12, .cb = 2 }, .{ .k2 = 8, .cb = 1 }, .{ .k2 = 8, .cb = 2 } };
const variants = blk: {
    var v: [widths.len * 6]Variant = undefined;
    var i: usize = 0;
    for (widths) |w| for ([_]u32{ 2, 4, 8 }) |wk| for ([_]u32{ 1, 2 }) |g| {
        v[i] = .{ .k2 = w.k2, .cb = w.cb, .wk = wk, .g = g, .symbol = std.fmt.comptimePrint("_ZN8tf_exl3l13linear_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi1EEEvNS_4JobsEii", .{ w.k2, w.cb, wk, g }) };
        i += 1;
    };
    break :blk v;
};
const rot_in_symbol = "_ZN8tf_exl3l13rot_in_kernelEPKviNS_7RotJobsEi";
const unpack_symbols = blk: {
    var u: [widths.len][:0]const u8 = undefined;
    for (widths, 0..) |w, i| u[i] = std.fmt.comptimePrint("_ZN8tf_exl3l13unpack_kernelILi{d}ELi{d}EEEvPKjP6__halfill", .{ w.k2, w.cb });
    break :blk u;
};

pub const Kernels = struct {
    module: cuda.Module,
    rot_in: cuda.Function,
    unpack: [widths.len]cuda.Function,
    linear_fns: [variants.len]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_exl3_linear);
        errdefer m.unload();
        var k: Kernels = .{ .module = m, .rot_in = try m.function(rot_in_symbol), .unpack = undefined, .linear_fns = undefined };
        for (unpack_symbols, 0..) |u, i| k.unpack[i] = try m.function(u);
        for (variants, 0..) |v, i| k.linear_fns[i] = try m.function(v.symbol);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }

    /// xh_j [M, K] fp16 = fp16(((x * suh_j) @ H) / sqrt(128)) for 1 to 3 (suh_j, xh_j) pairs of one input x [M, K].
    /// 1 つの入力 x [M, K] から、1〜3 組の (suh_j, xh_j) それぞれについて上の式で xh_j を作る。
    pub fn rotIn(k: *const Kernels, stream: cuda.Stream, x: u64, x_dtype: DType, suh: []const u64, xh: []const u64, m: u32, kdim: u32, pdl: bool) !void {
        if (suh.len == 0 or suh.len > max_jobs or xh.len != suh.len or kdim % 128 != 0) return error.BadRotIn;
        var jobs: RotJobs = .{};
        for (suh, xh, 0..) |s, h, i| {
            jobs.suh[i] = s;
            jobs.xh[i] = h;
        }
        var args: cuda.Args = .{};
        args.add(x);
        args.add(@intFromEnum(x_dtype));
        args.add(jobs);
        args.add(@as(c_int, @intCast(kdim)));
        try cuda.launch.launch(k.rot_in, .{ .grid = .{ .x = (kdim / 128 + 3) / 4, .y = m, .z = @intCast(suh.len) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = pdl }, stream, &args);
    }

    /// One launch over `jobs` (start offsets filled here); G is 1 up to 8 rows, else 2; V from loads & 2.
    /// `jobs` を 1 回で起動する (開始オフセットはここで埋める)。G は 8 行まで 1、それ以上は 2。V は loads & 2 から。
    pub fn linear(k: *const Kernels, stream: cuda.Stream, jobs_in: Jobs, m: u32, kdim: u32, k2: u32, cb: u32, wk: u32, loads: u32) !void {
        // only the load walk with bits 1 and 2 set (the engine's 35) is built
        // bit 1 と 2 が立つ読み込み方 (エンジンが使う 35) だけを実装している
        if (loads & 1 == 0 or loads & 2 == 0) return error.UnportedLinearWalk;
        var jobs = jobs_in;
        var blocks: c_int = 0;
        for (jobs.j[0..@intCast(jobs.n)]) |*j| {
            if (@divTrunc(@as(c_int, @intCast(kdim / 16)), j.SK * @as(c_int, @intCast(wk))) * j.SK * @as(c_int, @intCast(wk)) != kdim / 16) return error.UnevenKSplit;
            j.start = blocks;
            blocks += @divTrunc(j.N, 128) * j.SK;
        }
        const g: u32 = if (m <= 8) 1 else 2;
        const idx = for (variants, 0..) |v, i| {
            if (v.k2 == k2 and v.cb == cb and v.wk == wk and v.g == g) break i;
        } else return error.UnportedLinearVariant;
        // step_words<K2>() words of 4 bytes
        // 4 バイトの語が step_words<K2>() 個
        const step_bytes: u32 = 8 * 4 * k2 * 4;
        const smem = @max(wk * @min(m, 8) * 128 * 4, wk * step_bytes);
        const f = k.linear_fns[idx];
        if (smem > 48 * 1024) try f.allowDynamicShared(smem);
        var args: cuda.Args = .{};
        args.add(jobs);
        args.add(@as(c_int, @intCast(m)));
        args.add(@as(c_int, @intCast(kdim)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = wk * 32, .y = 1, .z = 1 }, .shared = smem, .pdl = loads & 32 != 0 }, stream, &args);
    }

    /// W [K, N] fp16: the trellis tiles decoded (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb words).
    /// W [K, N] fp16: trellis のタイルを復号する (タイル (kt, nt) は kt * stride_k + (nt / 8) * stride_nb 語目にある)。
    pub fn unpackW(k: *const Kernels, stream: cuda.Stream, t: u64, w: u64, kdim: u32, n: u32, k2: u32, cb: u32, stride_k: i64, stride_nb: i64) !void {
        const ui = for (widths, 0..) |q, i| {
            if (q.k2 == k2 and q.cb == cb) break i;
        } else return error.UnportedLinearVariant;
        var args: cuda.Args = .{};
        args.add(t);
        args.add(w);
        args.add(@as(c_int, @intCast(n)));
        args.add(stride_k);
        args.add(stride_nb);
        try cuda.launch.launch(k.unpack[ui], .{ .grid = .{ .x = n / 16, .y = kdim / 16, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, stream, &args);
    }
};
