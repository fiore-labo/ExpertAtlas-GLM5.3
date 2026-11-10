// Fiore-lab: the better C sampler against TensorFold 1.0.2's Zig, not in TensorFold 1.0.2.
//! The keyed host sampler (sampling.cpp, better C, through sampling.zig) against TensorFold 1.0.2's (sampling_ref.zig),
//! which the engines used until now: the same seeds, key bits, uniforms and draws over many random cases, so sampled
//! replies stay as they were.
const std = @import("std");
const ours = @import("sampling.zig");
const theirs = @import("sampling_ref.zig");

test "seeds match on random prompts and salts" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    var ids: [64]u32 = undefined;
    for (0..2000) |_| {
        const n = r.uintLessThan(usize, ids.len + 1);
        for (ids[0..n]) |*id| id.* = r.int(u32);
        const salt = r.int(i64);
        try std.testing.expectEqual(theirs.seedFor(ids[0..n], salt), ours.seedFor(ids[0..n], salt));
    }
}

test "key bits and uniforms match bit for bit" {
    var prng = std.Random.DefaultPrng.init(0xa11);
    const r = prng.random();
    for (0..100_000) |_| {
        const seed = r.int(u64);
        const pos = r.int(u64);
        const id = r.int(u64);
        try std.testing.expectEqual(theirs.keyBits(seed, pos, id), ours.keyBits(seed, pos, id));
        const a = theirs.uniform(seed, pos, id);
        const b = ours.uniform(seed, pos, id);
        try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
    }
}

test "min_p's log matches, off and on" {
    var prng = std.Random.DefaultPrng.init(0x319);
    const r = prng.random();
    for (0..10_000) |i| {
        const min_p = if (i % 4 == 0) 0.0 else r.float(f64);
        const a = (theirs.Sampling{ .seed = 0, .min_p = min_p }).minLog();
        const b = (ours.Sampling{ .seed = 0, .min_p = min_p }).minLog();
        try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
    }
}

test "draws match over random logits, widths and settings (ties, top_k 0, top_p and min_p on and off)" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd4a3);
    const r = prng.random();
    var logits: [300]f64 = undefined;
    var ids: [300]u64 = undefined;
    for (0..20_000) |case| {
        const n = 1 + r.uintLessThan(usize, logits.len);
        for (logits[0..n], ids[0..n], 0..) |*l, *id, i| {
            // a few repeated values make ties; ids are distinct, as candidate ids are
            l.* = if (r.uintLessThan(u8, 8) == 0) 1.5 else (r.float(f64) - 0.5) * 40.0;
            id.* = @as(u64, i) * 7 + r.uintLessThan(u64, 7);
        }
        const s_ours: ours.Sampling = .{
            .seed = r.int(u64) >> 1,
            .temperature = switch (case % 4) {
                0 => 1.0,
                1 => 0.6,
                2 => 1e-9,
                else => 1.0 + r.float(f64),
            },
            .top_k = switch (case % 3) {
                0 => 20,
                1 => 0,
                else => r.uintLessThan(u32, 400),
            },
            .top_p = switch (case % 5) {
                0 => 0.95,
                1 => 1.0,
                2 => 0.0,
                else => r.float(f64),
            },
            .min_p = if (case % 2 == 0) 0.0 else r.float(f64) * 0.3,
        };
        const s_theirs: theirs.Sampling = .{ .seed = s_ours.seed, .temperature = s_ours.temperature, .top_k = s_ours.top_k, .top_p = s_ours.top_p, .min_p = s_ours.min_p };
        const pos = r.int(u32);
        const a = try theirs.choose(gpa, logits[0..n], ids[0..n], pos, s_theirs);
        const b = try ours.choose(gpa, logits[0..n], ids[0..n], pos, s_ours);
        try std.testing.expectEqual(a, b);
    }
}
