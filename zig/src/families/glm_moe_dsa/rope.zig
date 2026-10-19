//! The RoPE inverse frequencies 1.0 / (theta ** (arange(0, 64, 2) / 64)) as fp32 bits recorded from torch on CUDA
//! (pow, then the reciprocal), and the 128 x 128 Hadamard matrix. A host pow differs in some bits, so another theta or
//! RoPE width is refused, not approximated.
//! RoPE の逆周波数 1.0 / (theta ** (arange(0, 64, 2) / 64)) を torch が CUDA で計算した fp32 ビット
//! (pow のあと逆数) で記録したものと、128 x 128 の Hadamard 行列。ホストの pow では一部のビットが違うので、ほかの
//! theta や RoPE 幅は近似せずに拒否する。
const std = @import("std");

pub const theta: f64 = 8000000.0;
pub const dim = 64;

pub const inv_bits = [dim / 2]u32{
    0x3f800000, 0x3f1bc84c, 0x3ebd9851, 0x3e66bf20, 0x3e0c6a41, 0x3daae464, 0x3d4ffbfb, 0x3cfd2099,
    0x3c9a08c8, 0x3c3b77ab, 0x3be42844, 0x3b8ad6e3, 0x3b28f979, 0x3acda683, 0x3a7a4971, 0x3a184e4a,
    0x39b95d21, 0x396198d7, 0x3909480c, 0x38a71410, 0x384b57bd, 0x37f77a71, 0x379698c3, 0x373748a4,
    0x36df10c4, 0x3687bdae, 0x36253419, 0x35c90f9a, 0x3574b383, 0x3514e825, 0x34b53a20, 0x345c8ff9,
};

/// The table's bytes (fp32 little-endian) for a config with this theta and RoPE width.
/// この theta と RoPE 幅の config 用の、表のバイト列 (fp32、リトルエンディアン)。
pub fn table(cfg_theta: f64, cfg_dim: usize) ![dim * 2]u8 {
    if (cfg_theta != theta or cfg_dim != dim) return error.RopeTableUnrecorded;
    var out: [dim * 2]u8 = undefined;
    for (inv_bits, 0..) |b, i| std.mem.writeInt(u32, out[i * 4 ..][0..4], b, .little);
    return out;
}

/// The 128 x 128 Hadamard matrix in bf16: +1 / -1 by the parity of i & j (exact in bf16).
/// bf16 の 128 x 128 Hadamard 行列: i & j のビット数の偶奇で +1 / -1 (bf16 で厳密)。
pub fn hadamard() [128 * 128 * 2]u8 {
    var out: [128 * 128 * 2]u8 = undefined;
    for (0..128) |i| for (0..128) |j| {
        const odd = @popCount(@as(u32, @intCast(i & j))) & 1 == 1;
        const bits: u16 = if (odd) 0xbf80 else 0x3f80; // bf16 -1.0 or +1.0 / bf16 の -1.0 か +1.0
        std.mem.writeInt(u16, out[(i * 128 + j) * 2 ..][0..2], bits, .little);
    };
    return out;
}

test "the first entry is 1.0 and the table refuses another theta" {
    const t = try table(8000000.0, 64);
    try std.testing.expectEqual(@as(u32, 0x3f800000), std.mem.readInt(u32, t[0..4], .little));
    try std.testing.expectError(error.RopeTableUnrecorded, table(10000.0, 64));
    const h = hadamard();
    try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, h[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0xbf80), std.mem.readInt(u16, h[(1 * 128 + 1) * 2 ..][0..2], .little));
}
