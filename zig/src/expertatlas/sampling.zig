// Fiore-lab, Expert Atlas.
//! Keyed sampling: the token a sampled request draws at each position, the same on every rank and in every run.
//! The algorithm is TensorFold's (lanes/sampling.zig, after Python's exact_sampling.py); the code is better C in
//! sampling.cpp, and this file is its Zig face, with the API the engine and the server use. The draws match
//! TensorFold's bit for bit; sampling_match_test.zig checks it.
//! 鍵付きのサンプリング: サンプルする要求が各位置で引くトークン。どのランクでもどの実行でも同じ。
//! アルゴリズムは TensorFold のもの。本体は better C の sampling.cpp で、このファイルはその Zig 側の入口。
const std = @import("std");
const Allocator = std.mem.Allocator;

/// sampling.cpp's FioreSampling
const CSampling = extern struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 };

comptime {
    std.debug.assert(@sizeOf(CSampling) == 40 and @offsetOf(CSampling, "top_k") == 16 and @offsetOf(CSampling, "top_p") == 24 and @offsetOf(CSampling, "min_p") == 32);
}

extern fn fiore_sampling_seed_for(tokens: [*]const u32, n: usize, salt: i64) u64;
extern fn fiore_sampling_uniform(seed: u64, position: u64, id: u64) f64;
extern fn fiore_sampling_min_log(min_p: f64) f64;
extern fn fiore_sampling_scratch_bytes(n: usize) usize;
extern fn fiore_sampling_choose(logits: [*]const f64, ids: [*]const u64, n: usize, position: u64, s: CSampling, scratch: *anyopaque) u64;

/// A request's sampling settings; a request without them decodes greedily.
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// ln(min_p), or -inf when min_p is off.
    pub fn minLog(s: Sampling) f64 {
        return fiore_sampling_min_log(s.min_p);
    }

    fn c(s: Sampling) CSampling {
        return .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p };
    }
};

/// A prompt's seed when the request names none: the first 8 bytes (little endian) of the sha256 of the ids written in
/// decimal and joined by commas, then "|" and the salt, kept to 63 bits.
pub fn seedFor(tokens: []const u32, salt: i64) u64 {
    return fiore_sampling_seed_for(tokens.ptr, tokens.len, salt);
}

/// A uniform double in (0, 1] from the key's top 53 bits, offset by half a step.
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    return fiore_sampling_uniform(seed, position, id);
}

/// The token drawn at `position` among the candidates `ids` (at least one) with float64 logits `logits`: sorted, cut
/// to top_k, then top_p (by probability mass) and min_p (against the best), and the Gumbel argmax of the rest under
/// the keyed draws.
pub fn choose(gpa: Allocator, logits: []const f64, ids: []const u64, position: u64, s: Sampling) !u64 {
    std.debug.assert(logits.len == ids.len and ids.len > 0);
    const scratch = try gpa.alloc(u64, (fiore_sampling_scratch_bytes(ids.len) + 7) / 8);
    defer gpa.free(scratch);
    return fiore_sampling_choose(logits.ptr, ids.ptr, ids.len, position, s.c(), scratch.ptr);
}

test "seeds keep 63 bits; uniforms fall in (0, 1]" {
    try std.testing.expect(seedFor(&.{ 1, 2, 3 }, 0) < (@as(u64, 1) << 63));
    const u = uniform(7, 11, 13);
    try std.testing.expect(u > 0.0 and u <= 1.0);
}

test "a greedy-like setting (top_k 1) takes the best logit, the smaller id on a tie" {
    const gpa = std.testing.allocator;
    const s: Sampling = .{ .seed = 1, .top_k = 1 };
    try std.testing.expectEqual(@as(u64, 9), try choose(gpa, &.{ 0.5, 2.0, 2.0 }, &.{ 4, 9, 12 }, 0, s));
}
