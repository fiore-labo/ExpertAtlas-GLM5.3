// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! The keyed host sampler (Python exact_sampling.py) bit for bit: libm's exp and log, numpy's sum order. The code is
//! better C in sampling.cpp and this file is its Zig face, with the API the engines, the drafters and the server use;
//! sampling_match_test.zig checks it against TensorFold 1.0.2's Zig (sampling_ref.zig).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// sampling.cpp's FioreSampling
const CSampling = extern struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 };

comptime {
    std.debug.assert(@sizeOf(CSampling) == 40 and @offsetOf(CSampling, "top_k") == 16 and @offsetOf(CSampling, "top_p") == 24 and @offsetOf(CSampling, "min_p") == 32);
}

extern fn fiore_sampling_seed_for(tokens: [*]const u32, n: usize, salt: i64) u64;
extern fn fiore_sampling_key_bits(seed: u64, position: u64, id: u64) u64;
extern fn fiore_sampling_uniform(seed: u64, position: u64, id: u64) f64;
extern fn fiore_sampling_min_log(min_p: f64) f64;
extern fn fiore_sampling_scratch_bytes(n: usize) usize;
extern fn fiore_sampling_choose(values: [*]const f64, ids: [*]const u64, n: usize, position: u64, s: CSampling, scratch: *anyopaque) u64;

pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// ln(min_p), -inf when off.
    pub fn minLog(s: Sampling) f64 {
        return fiore_sampling_min_log(s.min_p);
    }

    fn c(s: Sampling) CSampling {
        return .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p };
    }
};

/// A reproducible seed from the prompt: sha256 of the ids joined by commas and "|salt", first 8 bytes, 63 bits.
pub fn seedFor(tokens: []const u32, salt: i64) u64 {
    return fiore_sampling_seed_for(tokens.ptr, tokens.len, salt);
}

/// splitmix64 of (seed, position, id): the bits every keyed sampler starts from.
pub fn keyBits(seed: u64, position: u64, id: u64) u64 {
    return fiore_sampling_key_bits(seed, position, id);
}

/// Uniform (0, 1] double from the key's top 53 bits (Python `uniform`).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    return fiore_sampling_uniform(seed, position, id);
}

/// The id drawn among candidates `ids` with float64 logits `values` (Python `choose`, and `choose_rows` row by row).
pub fn choose(gpa: Allocator, values: []const f64, ids: []const u64, position: u64, s: Sampling) !u64 {
    std.debug.assert(values.len == ids.len and ids.len > 0);
    const scratch = try gpa.alloc(u64, (fiore_sampling_scratch_bytes(ids.len) + 7) / 8);
    defer gpa.free(scratch);
    return fiore_sampling_choose(values.ptr, ids.ptr, ids.len, position, s.c(), scratch.ptr);
}

test "seeds keep 63 bits and uniforms fall in (0, 1]" {
    try std.testing.expect(seedFor(&.{ 1, 2, 3 }, 0) < (@as(u64, 1) << 63));
    const u = uniform(7, 11, 13);
    try std.testing.expect(u > 0.0 and u <= 1.0);
}

test "top_k 1 takes the best logit, the smaller id on a tie" {
    const s: Sampling = .{ .seed = 1, .top_k = 1 };
    try std.testing.expectEqual(@as(u64, 9), try choose(std.testing.allocator, &.{ 0.5, 2.0, 2.0 }, &.{ 4, 9, 12 }, 0, s));
}

test {
    _ = @import("sampling_match_test.zig");
}
