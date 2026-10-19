//! One EXL3 linear on the GPU (trellis words in the "strips" layout, fp16 scales, K splits and warps) and its calls:
//! alone, or 2-3 layers of one input in one launch.
//! GPU 上の EXL3 linear 1 つ (trellis の語は "strips" レイアウト、fp16 のスケール、K 分割数と warp 数) とその呼び出し:
//! 単独で、または同じ入力の 2〜3 層を 1 回の起動で。
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const exl3 = @import("exl3.zig");

pub const Linear = struct {
    words: cuda.DeviceBuffer, // int32 [N/128, K/16, 8, 8 * bits]
    suh: cuda.DeviceBuffer, // fp16 [K]
    svh: cuda.DeviceBuffer, // fp16 [N]
    counters: cuda.DeviceBuffer, // int32 [8 * N / 128], left zero by the kernel / カーネルが 0 に戻す
    k: u32,
    n: u32,
    k2: u32,
    cb: u32,
    sk: u32,
    wk: u32,

    pub fn strides(l: Linear) struct { k: i64, nb: i64 } {
        const tw: i64 = 4 * @as(i64, l.k2);
        return .{ .k = 8 * tw, .nb = @as(i64, l.k / 16) * 8 * tw };
    }

    pub fn deinit(l: *Linear) void {
        l.words.free();
        l.suh.free();
        l.svh.free();
        l.counters.free();
    }
};

/// The default (K splits, warps a program), from the shape alone.
/// 既定の (K 分割数, プログラムあたりの warp 数)。形だけで決まる。
pub fn plan(k: u32, n: u32) struct { sk: u32, wk: u32 } {
    const kt = k / 16;
    const nb = n / 128;
    var sk: u32 = 1;
    var wk: u32 = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < 192 and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < 8)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = sk, .wk = wk };
}

/// Codebook id from the tensors a prefix stores: 3inst 0, mcg 1, mul1 2.
/// prefix が持つテンソルから決める codebook の id: 3inst 0、mcg 1、mul1 2。
fn codebook(r: *weights.Reader, prefix: []const u8, buf: []u8) !u32 {
    if (r.has(try std.fmt.bufPrint(buf, "{s}.mul1", .{prefix}))) return 2;
    if (r.has(try std.fmt.bufPrint(buf, "{s}.mcg", .{prefix}))) return 1;
    return 0;
}

/// `prefix`'s share on this rank (trellis int16 [K/16, N/16, 16 * bits] -> strips).
/// `prefix` のうちこのランクの分 (trellis int16 [K/16, N/16, 16 * bits] を strips に並べ替える)。
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, prefix: []const u8) !Linear {
    var buf: [256]u8 = undefined;
    var t = try r.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}));
    defer t.deinit(gpa);
    if (t.dtype != .i16 or t.rank != 3) return error.BadTrellis;
    const kt = t.shape[0];
    const nt = t.shape[1];
    const w16 = t.shape[2]; // 16 * bits int16 = 8 * bits int32 words a tile / タイル 1 つの語数
    if (w16 % 16 != 0 or nt % 8 != 0) return error.BadTrellis;
    const tile_bytes = w16 * 2;
    // strips: [K/16, N/16, W] viewed [kt, nt/8, 8, W] -> permute(1, 0, 2, 3) -> [nt/8, kt, 8, W]
    // strips: [K/16, N/16, W] を [kt, nt/8, 8, W] と見て、軸を (1, 0, 2, 3) の順に入れ替え [nt/8, kt, 8, W] にする
    const strip = try gpa.alloc(u8, t.bytes.len);
    defer gpa.free(strip);
    for (0..nt / 8) |b| for (0..kt) |i| {
        const src = t.bytes[(i * nt + b * 8) * tile_bytes ..][0 .. 8 * tile_bytes];
        @memcpy(strip[((b * kt) + i) * 8 * tile_bytes ..][0 .. 8 * tile_bytes], src);
    };
    var suh = try r.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{prefix}));
    defer suh.deinit(gpa);
    var svh = try r.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{prefix}));
    defer svh.deinit(gpa);
    if (suh.dtype != .f16 or svh.dtype != .f16) return error.PackedScalesUnsupported;
    const k: u32 = @intCast(kt * 16);
    const n: u32 = @intCast(nt * 16);
    const p = plan(k, n);
    var counters = try cuda.DeviceBuffer.alloc(d, 8 * (n / 128) * 4);
    try counters.fill8(0, null);
    return .{
        .words = try cuda.DeviceBuffer.fromHost(d, strip),
        .suh = try cuda.DeviceBuffer.fromHost(d, suh.bytes),
        .svh = try cuda.DeviceBuffer.fromHost(d, svh.bytes),
        .counters = counters,
        .k = k,
        .n = n,
        .k2 = @intCast(2 * (w16 / 16)),
        .cb = try codebook(r, prefix, &buf),
        .sk = p.sk,
        .wk = p.wk,
    };
}

/// Whether the layers can share one launch: 1 to 3 layers, one K, width, codebook and warps (K splits may differ).
/// 1 回の起動にまとめられるか: 1〜3 層で、K・幅・codebook・warp 数が同じ (K 分割数は違ってよい)。
pub fn groupable(ls: []const *const Linear) bool {
    if (ls.len == 0 or ls.len > exl3.max_jobs) return false;
    const a = ls[0];
    for (ls) |l| if (l.k != a.k or l.k2 != a.k2 or l.cb != a.cb or l.wk != a.wk) return false;
    return true;
}

/// Scratch a call needs: xh fp16 [M, K] a layer, Z fp32 [SK, M, N] a layer with SK > 1.
/// 呼び出しに要る scratch: 層ごとの xh fp16 [M, K] と、SK > 1 の層ごとの Z fp32 [SK, M, N]。
pub const Scratch = struct {
    xh: [exl3.max_jobs]cuda.DeviceBuffer,
    z: [exl3.max_jobs]cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, max_rows: u32, max_k: u32, max_zn: usize) !Scratch {
        var s: Scratch = undefined;
        for (0..exl3.max_jobs) |i| {
            s.xh[i] = try cuda.DeviceBuffer.alloc(d, @as(usize, max_rows) * max_k * 2);
            s.z[i] = try cuda.DeviceBuffer.alloc(d, max_zn * 4);
        }
        return s;
    }
};

/// outs[j] (fp32 / bf16 [M, N_j], `out_dtypes`) = x [M, K] @ W_j for 1 to 3 groupable layers in one rot_in and one
/// linear launch, M up to 128 rows.
/// まとめられる 1〜3 層について outs[j] = x [M, K] @ W_j (fp32 / bf16 [M, N_j]、`out_dtypes`) を、rot_in 1 回と
/// linear 1 回の起動で計算する。M は最大 128 行。
pub fn call(k: *const exl3.Kernels, stream: cuda.Stream, s: *const Scratch, ls: []const *const Linear, x: u64, x_dtype: exl3.DType, m: u32, outs: []const u64, out_dtypes: []const exl3.DType) !void {
    if (!groupable(ls) or m < 1 or m > 128) return error.NotGroupable;
    var suh: [exl3.max_jobs]u64 = undefined;
    var xh: [exl3.max_jobs]u64 = undefined;
    for (ls, 0..) |l, i| {
        suh[i] = l.suh.ptr;
        xh[i] = s.xh[i].ptr;
    }
    // the kernels' load options: bits 1 and 2 (the one load walk built), 32 (PDL)
    // カーネルの読み込み設定: bit 1 と 2 (実装した唯一の読み込み方)、32 (PDL)
    const loads: u32 = 35;
    try k.rotIn(stream, x, x_dtype, suh[0..ls.len], xh[0..ls.len], m, ls[0].k, loads & 32 != 0);
    var jobs: exl3.Jobs = .{ .n = @intCast(ls.len) };
    for (ls, 0..) |l, i| {
        const st = l.strides();
        jobs.j[i] = .{ .xh = xh[i], .T = l.words.ptr, .stride_k = st.k, .stride_nb = st.nb, .svh = l.svh.ptr, .y = outs[i], .Z = if (l.sk > 1) s.z[i].ptr else 0, .counters = l.counters.ptr, .y_dtype = @intFromEnum(out_dtypes[i]), .N = @intCast(l.n), .SK = @intCast(l.sk) };
    }
    try k.linear(stream, jobs, m, ls[0].k, ls[0].k2, ls[0].cb, ls[0].wk, loads);
}

test "plan() matches linear.py on GLM-5.3's TP=4 shapes" {
    const t = std.testing;
    // q_b 2048 -> 4096 and o_proj 4096 -> 6144: (4, 4)
    // q_b 2048 -> 4096 と o_proj 4096 -> 6144: (4, 4)
    try t.expectEqual(@as(u32, 4), plan(2048, 4096).sk);
    try t.expectEqual(@as(u32, 4), plan(2048, 4096).wk);
    try t.expectEqual(@as(u32, 4), plan(4096, 6144).sk);
    // q_a 6144 -> 2048 and kv_a 6144 -> 640: (8, 4)
    // q_a 6144 -> 2048 と kv_a 6144 -> 640: (8, 4)
    try t.expectEqual(@as(u32, 8), plan(6144, 2048).sk);
    try t.expectEqual(@as(u32, 8), plan(6144, 640).sk);
    // o_proj partial 512 -> 6144: SK 1
    // o_proj の部分 512 -> 6144: SK 1
    try t.expectEqual(@as(u32, 1), plan(512, 6144).sk);
}
