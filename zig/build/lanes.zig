// Fiore-lab: the lane core's module with its keyed sampler in better C, not in TensorFold 1.0.2.
//! The lane core (zig/src/core/lanes/) as one module for the Metal and the CUDA builds, with sampling.cpp, the keyed
//! sampler under sampling.zig. sampling.cpp is better C (C++ without its library or runtime: no exceptions, RTTI or
//! libc++) and keeps the draws' floating point as written: no fused multiply-add, no fast math. The scripts that build
//! the module themselves (tools/zig/test_lanes.sh, zig/tests/server/build.sh and zig_test.sh) pass the same flags.
const std = @import("std");

pub const sampling_flags = [_][]const u8{ "-std=c++20", "-nostdinc++", "-fno-exceptions", "-fno-rtti", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra" };

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    lanes.addCSourceFile(.{ .file = b.path("zig/src/core/lanes/sampling.cpp"), .flags = &sampling_flags });
    return lanes;
}
