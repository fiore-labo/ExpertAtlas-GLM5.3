// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "glm53_exl3_linear", .src = "glm53/exl3_linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr", "-DTF_ZIG" } }, // cuda/exl3/linear.py, tensorfold_exl3_linear_v5
    .{ .name = "glm53_native_ops", .src = "glm53/native_ops", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" }, .arch_specific = true }, // ours: the Triton kernels as plain CUDA (TF_GLM53_NATIVE); sm_XXXa for the attention's transposed byte loads (ldmatrix .b8)
    .{ .name = "glm53_glue", .src = "glm53/glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: torch's copies, gathers and the rank-order fp32 sum
    .{ .name = "glm53_exl3_experts", .src = "glm53/exl3_experts", .flags = &.{ "-O3", "-DTF_ZIG" } }, // cuda/exl3/experts.py, tensorfold_exl3_experts_v2
    .{ .name = "glm53_roce", .src = "glm53/roce", .flags = &.{"-O3"} }, // b12x comm/roce _oneshot_cute.py / _allgather_cute.py (TF_GLM53_ROCE)
    .{ .name = "glm53_prompt_experts", .src = "glm53/prompt_experts", .flags = &.{ "-O3", "-DTF_ZIG" } }, // cuda/exl3/prompt_experts.py, tensorfold_exl3_prompt_experts_v1 (TF_EXL3_PROMPT_EXPERTS=1)
    .{ .name = "glm53_qmm_group", .src = "glm53/qmm_group", .flags = &.{ "-O3", "-DTF_ZIG" } }, // cuda/kernels/qmm.py, tensorfold_qmm_v5 (the drafter's 4-bit matmul on sm_12x)
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .tokenizer = tokenizer };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    const tp_runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/tp_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    tp_runner.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-tp-test", .root_module = tp_runner }));
    const glm53_runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_runner.addImport("cuda", cuda);
    const glm = glmModule(b, target, optimize, cuda, mods.core, mods.lanes);
    glm53_runner.addImport("glm_moe_dsa", glm);
    const glm53_run = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_run.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_run.addImport("cuda", cuda);
    glm53_run.addImport("glm_moe_dsa", glm);
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-run", .root_module = glm53_run }));
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-test", .root_module = glm53_runner }));
    const glm53_lock = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_lock.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_lock.addImport("cuda", cuda);
    glm53_lock.addImport("glm_moe_dsa", glm);
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-lock", .root_module = glm53_lock }));
    // Fiore-lab: a decode lane and a prefill lane on the first layers, alone, time-sliced and at once
    const glm53_conc = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_concurrent.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_conc.addImport("cuda", cuda);
    glm53_conc.addImport("glm_moe_dsa", glm);
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-concurrent", .root_module = glm53_conc }));
    const glm53_batch = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_batch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_batch.addImport("cuda", cuda);
    glm53_batch.addImport("glm_moe_dsa", glm);
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-batch", .root_module = glm53_batch }));
    const glm53_q4 = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm53_q4.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm53_q4.addImport("glm_moe_dsa", glm);
    b.installArtifact(b.addExecutable(.{ .name = "tf-glm53-q4", .root_module = glm53_q4 }));
    // host-side tests of the glm_moe_dsa module (config, split, header words, ...): `zig build test-glm53`
    b.step("test-glm53", "glm_moe_dsa host-side unit tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    _ = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, glm, mods.tokenizer, build_options, true);
    serverTests(b, target, optimize, cuda, mods.lanes, mods.nemotron, glm, mods.tokenizer);
}

/// Fiore-lab: the server's unit tests and its frozen parity goldens on the CUDA build host too (build.zig builds them
/// in the Metal build only), as build.zig makes them: `zig build test-server`, `zig build test-server-golden`.
fn serverTests(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, glm: *std.Build.Module, tokenizer: *std.Build.Module) void {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, glm);
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } };
    const server = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = imports });
    const step = b.step("test-server", "The server's unit tests (zig/src/server/root.zig) on this host");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server })).step);
    const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/fake_serve.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "server", .module = server }, .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } },
    }) });
    // libc linked explicitly: Linux, unlike macOS, does not link it by default
    const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/server/golden_check.zig"), .target = target, .optimize = .Debug, .link_libc = true }) });
    const golden_run = b.addRunArtifact(golden_check);
    golden_run.addArtifactArg(fake_serve);
    golden_run.addFileArg(b.path("zig/tests/server/golden/cases.json"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/golden"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    golden_run.setCwd(b.path("."));
    b.step("test-server-golden", "The frozen server parity goldens (fake_serve) on this host").dependOn(&golden_run.step);
}

/// The GLM-5.3 (glm_moe_dsa) family module over the CUDA runtime and the core.
fn glmModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const glm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_moe_dsa/glm_moe_dsa.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm.addImport("core", core);
    glm.addImport("lanes", lanes); // the keyed host sampler (exact_sampling)
    glm.addImport("cuda", cuda);
    return glm;
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, glm: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "glm_moe_dsa", .module = glm } },
    });
    return .{ .api = api, .engines = mod };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, glm: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, glm);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const native = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, glmModule(b, host, .debug, cuda, mods.core, mods.lanes)).engines;
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, glmModule(b, target, .fast, cuda, mods.core, mods.lanes), mods.tokenizer, build_options, false);
}

/// Validate the complete named input set, including images unused by today's server.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
