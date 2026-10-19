//! The slot cache's hashes: the token-prefix identity `pid` and the position row hash `H`. Every operation is mod
//! 2^64 (wrapping u64); the slot kernels in glue.cu must give the same bits.
//! スロットキャッシュのハッシュ: トークン prefix の識別子 `pid` と、位置ごとの行ハッシュ `H`。演算はすべて mod 2^64
//! (wrapping u64)。glue.cu のスロット用カーネルも同じビットを出さなければならない。
//!
//!     mix64(x)   splitmix64's finalizer
//!     pid(p)     sum over i <= p of mix64(tok_i ^ i * PHI)
//!     lane(l)    h = mix64((l + 1) * K_LANE); per little-endian word w_j: h = mix64(h ^ w_j ^ (j + 1) * K_WORD)
//!     H(p)       acc = mix64(SEED ^ mix64(p) ^ mix64(pid ^ K_PID) ^ mix64(gen ^ K_GEN));
//!                per lane: acc = mix64(acc ^ lane(l) ^ (l + 1) * K_LAYER)
//!
//!     mix64(x)   splitmix64 の finalizer
//!     pid(p)     i <= p について mix64(tok_i ^ i * PHI) の和
//!     lane(l)    h = mix64((l + 1) * K_LANE); リトルエンディアンの語 w_j ごとに
//!                h = mix64(h ^ w_j ^ (j + 1) * K_WORD)
//!     H(p)       acc = mix64(SEED ^ mix64(p) ^ mix64(pid ^ K_PID) ^ mix64(gen ^ K_GEN));
//!                lane ごとに acc = mix64(acc ^ lane(l) ^ (l + 1) * K_LAYER)
const std = @import("std");

pub const C1: u64 = 0xBF58476D1CE4E5B9;
pub const C2: u64 = 0x94D049BB133111EB;
pub const PHI: u64 = 0x9E3779B97F4A7C15;
pub const SEED: u64 = 0x6A09E667F3BCC908;
pub const K_WORD: u64 = 0xD6E8FEB86659FD93;
pub const K_LANE: u64 = 0xBB67AE8584CAA73B;
pub const K_LAYER: u64 = 0x3C6EF372FE94F82B;
pub const K_PID: u64 = 0xA54FF53A5F1D36F1;
pub const K_GEN: u64 = 0x510E527FADE682D1;

/// GLM-5.3 flagship with FP8 caches: 78 latent rows of 656 B, then 21 index-key rows of 144 B (54,192 B a position).
/// FP8 キャッシュの GLM-5.3 flagship: 656 B の latent 行が 78、続いて 144 B の index キー行が 21 (1 位置 54,192 B)。
pub const FLAGSHIP_WIDTHS: [99]u32 = blk: {
    var w: [99]u32 = undefined;
    for (&w, 0..) |*x, i| x.* = if (i < 78) 656 else 144;
    break :blk w;
};

/// splitmix64's finalizer.
/// splitmix64 の finalizer。
pub fn mix64(x0: u64) u64 {
    var x = (x0 ^ (x0 >> 30)) *% C1;
    x = (x ^ (x >> 27)) *% C2;
    return x ^ (x >> 31);
}

/// pid of every position into `out` (out.len == tokens.len); the tokens start at position 0.
/// 各位置の pid を `out` に書く (out.len == tokens.len)。tokens は位置 0 から始まる。
pub fn pidsInto(out: []u64, tokens: []const i64) void {
    var acc: u64 = 0;
    for (tokens, 0..) |t, i| {
        acc +%= mix64(@as(u64, @bitCast(t)) ^ (@as(u64, i) *% PHI));
        out[i] = acc;
    }
}

/// pid of every position; the caller frees the result.
/// 各位置の pid。結果は呼び出し側が解放する。
pub fn pids(gpa: std.mem.Allocator, tokens: []const i64) ![]u64 {
    const out = try gpa.alloc(u64, tokens.len);
    pidsInto(out, tokens);
    return out;
}

/// lane(l) above: one lane's hash over its 8-byte words.
/// 上の lane(l): 1 つの lane の 8 バイト語に対するハッシュ。
pub fn laneHash(words: []const u64, lane: u64) u64 {
    var h = mix64((lane +% 1) *% K_LANE);
    for (words, 0..) |w, j| h = mix64(h ^ w ^ ((@as(u64, j) +% 1) *% K_WORD));
    return h;
}

/// H of one position whose lanes lie back to back in `row` (`widths` bytes each, multiples of 8).
/// 1 位置の H。lane は `row` に隙間なく並ぶ (それぞれ `widths` バイト、8 の倍数)。
pub fn rowHash(row: []const u8, widths: []const u32, p: i64, pid: u64, gen: i64) error{BadWidths}!u64 {
    var acc = mix64(SEED ^ mix64(@bitCast(p)) ^ mix64(pid ^ K_PID) ^ mix64(@as(u64, @bitCast(gen)) ^ K_GEN));
    var off: usize = 0;
    for (widths, 0..) |width, lane| {
        if (width % 8 != 0) return error.BadWidths;
        if (off + width > row.len) return error.BadWidths;
        var h = mix64((@as(u64, lane) +% 1) *% K_LANE);
        var j: u64 = 0;
        while (j < width / 8) : (j += 1) {
            const w = std.mem.readInt(u64, row[off + 8 * j ..][0..8], .little);
            h = mix64(h ^ w ^ ((j +% 1) *% K_WORD));
        }
        acc = mix64(acc ^ h ^ ((@as(u64, lane) +% 1) *% K_LAYER));
        off += width;
    }
    if (off != row.len) return error.BadWidths;
    return acc;
}
