// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
const std = @import("std");
const builtin = @import("builtin");
// Fiore-lab: Expert Atlas's CUDA build (GLM-5.3 only, Linux) in place of TensorFold's zig/build/cuda.zig and its Metal build
const expertatlas_build = @import("zig/build/expertatlas.zig");
const dist_build = @import("zig/build/dist.zig");

comptime {
    const required = std.mem.trim(u8, @embedFile(".zig-version"), "\r\n");
    if (!std.mem.eql(u8, builtin.zig_version_string, required))
        @compileError("TensorFold's Zig engine requires Zig " ++ required);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const release_version = b.option([]const u8, "version", "Release version: MAJOR.MINOR.PATCH, or a label rN-M, rN.K-M or rN.K.P-M");
    if (release_version) |v| if (!dist_build.validVersion(v)) @panic("-Dversion must be MAJOR.MINOR.PATCH, rN-M, rN.K-M or rN.K.P-M");
    const build_options = b.addOptions();
    // Keep the existing manifest version for developer builds; releases require an explicit override.
    const manifest = @import("build.zig.zon");
    build_options.addOption([]const u8, "version", release_version orelse manifest.version);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    const test_step = b.step("test", "Host-side unit tests (no GPU work)");
    // Fiore-lab: Linux only; TensorFold's macOS (Metal) build is not part of Expert Atlas
    if (target.result.os.tag == .linux) expertatlas_build.targets(b, target, optimize, build_options);
    dist_build.targets(b, build_options, release_version);
    expertatlas_build.hostTests(b, test_step);
    // the GLM-5.3 slot cache's host-side files alone (no family module)
    const slots = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/slotcache/slotcache_test.zig"), .target = b.graph.host, .optimize = .debug }) });
    b.step("test-glm53-slots", "The GLM-5.3 slot cache's host metadata: allocator, tree, plans, digests against Python").dependOn(&b.addRunArtifact(slots).step);
    // the GLM-5.3 KV pool's host-side planner alone (glm53/POOL.md)
    const pool = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/pool/pool_test.zig"), .target = b.graph.host, .optimize = .debug }) });
    b.step("test-glm53-pool", "The GLM-5.3 KV pool's host planner: worked examples, Python reference traces, invariants").dependOn(&b.addRunArtifact(pool).step);
    // TF_GLM53_MULTI's pace alone: the decode's share of the GPU while a prompt waits or is computed
    const pace = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/pace.zig"), .target = b.graph.host, .optimize = .debug }) });
    b.step("test-glm53-pace", "The GLM-5.3 concurrent server's pace: the decode's share of the GPU beside a prefill").dependOn(&b.addRunArtifact(pace).step);
    // the structured output's grammar engine on the host: the llguidance binding and the counterexample table through
    // the server's schema check (TF_TEST_LLG_LIB and TF_TEST_GLM_VOCAB name the library and the vocabulary; skipped without)
    const schema_check = b.createModule(.{ .root_source_file = b.path("zig/src/server/schema_check.zig"), .target = b.graph.host, .optimize = .debug });
    const structure = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/structure/llg_table_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "schema_check", .module = schema_check }} }) });
    const structure_run = b.addRunArtifact(structure);
    structure_run.has_side_effects = true; // the environment decides what runs, and it is not in the run's cache key
    const structure_step = b.step("test-glm53-structure", "The GLM-5.3 structured output: the llguidance binding, the counterexample table through the server's schema check, and the host's matcher");
    structure_step.dependOn(&structure_run.step);
    // the host's matcher (structure/host.zig) alone and over llguidance (skipped without TF_TEST_LLG_LIB)
    for ([_][]const u8{ "host_test.zig", "host_llg_test.zig" }) |name| {
        const t = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/glm_moe_dsa/structure/{s}", .{name})), .target = b.graph.host, .optimize = .debug, .link_libc = true }) });
        const run = b.addRunArtifact(t);
        run.has_side_effects = true;
        structure_step.dependOn(&run.step);
    }
    const pool_trace = b.addExecutable(.{ .name = "tf-glm53-pool-trace", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/pool/trace.zig"), .target = b.graph.host, .optimize = .debug }) });
    b.step("pool-trace", "tf-glm53-pool-trace: the KV pool planner over a JSON-lines trace (compared with glm53/tools/pool/pool_ref.py)").dependOn(&b.addInstallArtifact(pool_trace, .{}).step);
}

test {
    _ = dist_build;
}
