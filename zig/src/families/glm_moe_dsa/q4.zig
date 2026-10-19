//! The DFlash2 drafter's 4-bit weights, quantized and packed on the host for the 4-bit matmul: affine groups of 64
//! along K, n padded to 128 with zeros. Each fp32 step is one correctly rounded IEEE operation, so the bits equal a
//! GPU quantizer's.
//! DFlash2 ドラフタの 4 bit 重みをホストで量子化し、4 bit 行列積向けにパックする: K 方向 64 個ずつのアフィン group、
//! n はゼロで 128 の倍数に埋める。fp32 の各段は正しく丸めた IEEE 演算 1 回なので、GPU の量子化器と同じビットになる。
const std = @import("std");

pub const gs = 64;
/// Nibble p of a lane's word v holds input 32 v + 2 (lane % 4) + offsets[p] of the group (the order `pair` in
/// qmm_frag.cuh unpacks).
/// lane の語 v の nibble p は group の入力 32 v + 2 (lane % 4) + offsets[p] を持つ (qmm_frag.cuh の `pair` が
/// 展開する順)。
const offsets = [8]u32{ 0, 8, 16, 24, 1, 9, 17, 25 };

pub const Packed = struct {
    words: []u32, // [npad / 64][k / 64][8][32][2]
    scales: []u16, // [k / 64][npad] bf16
    biases: []u16,
    n: usize,
    k: usize,
    npad: usize,

    pub fn deinit(p: *Packed, gpa: std.mem.Allocator) void {
        gpa.free(p.words);
        gpa.free(p.scales);
        gpa.free(p.biases);
    }
};

fn bf16(v: f32) u16 { // round to nearest even; weights hold no NaN / 最近接偶数へ丸める。重みに NaN はない
    const b: u32 = @bitCast(v);
    const r = b +% (0x7FFF + ((b >> 16) & 1));
    return @intCast(r >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Quantizes `w` ([n, k] bf16 rows, k a multiple of 64): scale = bf16(max((hi - lo) / 15, 1e-8)), bias = bf16(lo),
/// q = clamp(round_half_even((x - bias) / scale), 0, 15).
/// `w` ([n, k] の bf16 行、k は 64 の倍数) を量子化する: scale = bf16(max((hi - lo) / 15, 1e-8))、bias = bf16(lo)、
/// q = clamp(round_half_even((x - bias) / scale), 0, 15)。
pub fn quantize(gpa: std.mem.Allocator, w: []const u16, n: usize, k: usize) !Packed {
    if (k % gs != 0 or w.len != n * k) return error.BadShape;
    const kg = k / gs;
    const npad = (n + 127) / 128 * 128;
    // q values [npad][k] (u8); the padding rows are 0, with scale and bias 0
    // q の値 [npad][k] (u8)。埋めた行は 0 で、scale と bias も 0
    const q = try gpa.alloc(u8, npad * k);
    defer gpa.free(q);
    @memset(q, 0);
    const scales = try gpa.alloc(u16, kg * npad);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u16, kg * npad);
    errdefer gpa.free(biases);
    @memset(scales, 0);
    @memset(biases, 0);
    for (0..n) |row| {
        for (0..kg) |g| {
            const xs = w[row * k + g * gs ..][0..gs];
            var lo = f32of(xs[0]);
            var hi = lo;
            for (xs[1..]) |h| {
                const x = f32of(h);
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            const s = bf16(@max((hi - lo) / 15.0, @as(f32, 1e-8)));
            const b = bf16(lo);
            const sf = f32of(s);
            const bf = f32of(b);
            for (xs, 0..) |h, i| {
                const r = roundHalfEven((f32of(h) - bf) / sf);
                q[row * k + g * gs + i] = @intFromFloat(std.math.clamp(r, 0.0, 15.0));
            }
            scales[g * npad + row] = s;
            biases[g * npad + row] = b;
        }
    }
    const words = try gpa.alloc(u32, npad / 64 * kg * 8 * 32 * 2);
    errdefer gpa.free(words);
    // out[T][g][j][r * 4 + c][v] = sum_p q[T * 64 + j * 8 + r][g * 64 + 32 v + 2 c + offsets[p]] << 4 p
    for (0..npad / 64) |t| for (0..kg) |g| for (0..8) |j| for (0..8) |r| for (0..4) |c| for (0..2) |v| {
        const col = t * 64 + j * 8 + r;
        var word: u32 = 0;
        for (offsets, 0..) |o, p| word |= @as(u32, q[col * k + g * gs + 32 * v + 2 * c + o]) << @intCast(4 * p);
        words[((((t * kg + g) * 8 + j) * 32 + r * 4 + c) * 2) + v] = word;
    };
    return .{ .words = words, .scales = scales, .biases = biases, .n = n, .k = k, .npad = npad };
}

/// Rounds to nearest, ties to even.
/// 最近接に丸め、ちょうど半分なら偶数へ。
fn roundHalfEven(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2.0) == 0) f else f + 1;
}

test "round half to even as torch.round" {
    try std.testing.expectEqual(@as(f32, 2), roundHalfEven(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundHalfEven(3.5));
    try std.testing.expectEqual(@as(f32, -2), roundHalfEven(-2.5));
    try std.testing.expectEqual(@as(f32, 1), roundHalfEven(0.7));
}

test "a constant group packs to zeros with its value as bias" {
    const gpa = std.testing.allocator;
    var w: [64]u16 = undefined;
    @memset(&w, bf16(0.5));
    var p = try quantize(gpa, &w, 1, 64);
    defer p.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 128), p.npad);
    for (p.words) |x| try std.testing.expectEqual(@as(u32, 0), x);
    try std.testing.expectEqual(bf16(0.5), p.biases[0]);
    try std.testing.expectEqual(bf16(1e-8), p.scales[0]);
}
