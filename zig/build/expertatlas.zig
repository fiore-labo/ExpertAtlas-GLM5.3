// Fiore-lab, Expert Atlas.
//! Expert Atlas's CUDA build: the GLM-5.3 kernels (zig/kernels/cuda/glm53) as fatbins, the CUDA runtime that embeds
//! them, the GLM-5.3 engine (glm_moe_dsa, native/glm53_host.zig), the server (tensorfold-native) over Expert Atlas's
//! engine contract (expertatlas/engine_api.zig), the GPU test programs, the host unit tests and the release server.
//! Nothing else is built for CUDA: TensorFold's other kernels and families, its lane host and its CLI are left out.
//! Expert Atlas の CUDA ビルド: GLM-5.3 のカーネル 7 本、それを埋め込む CUDA ランタイム、GLM-5.3 のエンジン、サーバー、
//! GPU の試験プログラム、ホストの単体テスト、配布用のサーバー。TensorFold のほかのカーネルと family は作らない。
const std = @import("std");

const Kernel = struct {
    /// The image's name: the fatbin's file name and its import in zig/src/cuda/kernels.zig.
    name: []const u8,
    /// The .cu under zig/kernels/cuda, without the extension.
    src: []const u8,
    flags: []const []const u8,
    /// SASS for sm_XXXa (architecture-specific instructions) instead of sm_XXX.
    arch_specific: bool = false,
};

/// The kernels the GLM-5.3 engine loads, each with the nvcc flags it was qualified with.
pub const kernels = [_]Kernel{
    .{ .name = "glm53_exl3_linear", .src = "glm53/exl3_linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr", "-DTF_ZIG" } },
    // the attention's transposed byte loads (ldmatrix .b8) need the architecture-specific target
    .{ .name = "glm53_native_ops", .src = "glm53/native_ops", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" }, .arch_specific = true },
    .{ .name = "glm53_glue", .src = "glm53/glue", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } },
    .{ .name = "glm53_exl3_experts", .src = "glm53/exl3_experts", .flags = &.{ "-O3", "-DTF_ZIG" } },
    .{ .name = "glm53_roce", .src = "glm53/roce", .flags = &.{"-O3"} },
    .{ .name = "glm53_prompt_experts", .src = "glm53/prompt_experts", .flags = &.{ "-O3", "-DTF_ZIG" } },
    .{ .name = "glm53_qmm_group", .src = "glm53/qmm_group", .flags = &.{ "-O3", "-DTF_ZIG" } },
};

/// The flags every kernel is compiled with (those torch.utils.cpp_extension passed when the kernels were first built
/// as Python extensions): C++20, and the half/bf16 operators the CUDA headers must not define.
const common_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

const Mode = std.builtin.OptimizeMode;
const Target = std.Build.ResolvedTarget;
const Module = std.Build.Module;

fn module(b: *std.Build, path: []const u8, target: Target, optimize: Mode) *Module {
    return b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = optimize, .link_libc = true });
}

/// `nvcc -fatbin` of one kernel for each SASS target in `sms` (comma separated); nvcc's version text is an input, so a
/// new compiler rebuilds every image.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&common_flags);
    run.addArgs(k.flags);
    const suffix = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, suffix, sm, suffix }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src})));
    return out;
}

/// The CUDA runtime (zig/src/cuda) embedding `images` (one a kernel, in `kernels` order); empty: host-only, no images.
fn runtime(b: *std.Build, target: Target, optimize: Mode, images: []const std.Build.LazyPath) *Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images.len > 0);
    const cuda = module(b, "zig/src/cuda/root.zig", target, optimize);
    cuda.addOptions("kernel_options", options);
    for (images, 0..) |image, i| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{kernels[i].name}), .{ .root_source_file = image });
    return cuda;
}

/// sampling.cpp as better C (C++ without its library or runtime: no exceptions, RTTI or libc++) and with the draws'
/// floating point as written: no fused multiply-add, no fast math.
const sampling_flags = [_][]const u8{ "-std=c++20", "-nostdinc++", "-fno-exceptions", "-fno-rtti", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra" };

/// The modules above the runtime: the tokenizer, the checkpoint core, the sampling (sampling.cpp under its Zig face),
/// the engine contract, the GLM-5.3 family and the engines the server opens.
const Mods = struct { tokenizer: *Module, core: *Module, sampling: *Module, api: *Module, glm: *Module, engines: *Module };

fn mods(b: *std.Build, target: Target, optimize: Mode, cuda: *Module) Mods {
    const tokenizer = module(b, "zig/src/core/tokenizer/tokenizer.zig", target, optimize);
    const core = module(b, "zig/src/core/root.zig", target, optimize);
    core.addImport("tokenizer", tokenizer);
    const sampling = module(b, "zig/src/expertatlas/sampling.zig", target, optimize);
    sampling.addCSourceFile(.{ .file = b.path("zig/src/expertatlas/sampling.cpp"), .flags = &sampling_flags });
    const api = module(b, "zig/src/expertatlas/engine_api.zig", target, optimize);
    api.addImport("sampling", sampling);
    api.addImport("prompt_imprint", module(b, "zig/src/core/prompt_imprint.zig", target, optimize));
    api.addImport("keepalive", module(b, "zig/src/core/keepalive.zig", target, optimize));
    const glm = module(b, "zig/src/families/glm_moe_dsa/glm_moe_dsa.zig", target, optimize);
    glm.addImport("core", core);
    glm.addImport("cuda", cuda);
    glm.addImport("sampling", sampling);
    const engines = module(b, "zig/src/native/expertatlas.zig", target, optimize);
    engines.addImport("cuda", cuda);
    engines.addImport("engine_api", api);
    engines.addImport("glm_moe_dsa", glm);
    return .{ .tokenizer = tokenizer, .core = core, .sampling = sampling, .api = api, .glm = glm, .engines = engines };
}

/// The server over the engines: the HTTP side keeps its safety checks (ReleaseSafe); the engine runs at `optimize`.
fn server(b: *std.Build, target: Target, m: Mods, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = m.api },
            .{ .name = "tokenizer", .module = m.tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "native_engines", .module = m.engines },
            // `tensorfold-native models | info | pull`: the checkpoint subcommands (zig/src/cli)
            .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) },
        },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    return exe;
}

/// The GPU programs under zig/tests/cuda that drive the GLM-5.3 engine, the tensor-parallel exchange and RoCE.
const test_programs = [_]struct { name: []const u8, src: []const u8, glm: bool }{
    .{ .name = "tf-tp-test", .src = "zig/tests/cuda/tp_main.zig", .glm = false },
    .{ .name = "tf-glm53-test", .src = "zig/tests/cuda/glm53_main.zig", .glm = true },
    .{ .name = "tf-glm53-run", .src = "zig/tests/cuda/glm53_run.zig", .glm = true },
    .{ .name = "tf-roce-test", .src = "zig/tests/cuda/roce_main.zig", .glm = true },
    .{ .name = "tf-glm53-lock", .src = "zig/tests/cuda/glm53_lock.zig", .glm = true },
    .{ .name = "tf-glm53-concurrent", .src = "zig/tests/cuda/glm53_concurrent.zig", .glm = true },
    .{ .name = "tf-glm53-batch", .src = "zig/tests/cuda/glm53_batch.zig", .glm = true },
    .{ .name = "tf-glm53-q4", .src = "zig/tests/cuda/glm53_q4.zig", .glm = true },
    // the keyed sampling's candidates picked on the GPU against the host pick (one GPU)
    .{ .name = "tf-glm53-sample-pick", .src = "zig/tests/cuda/glm53_sample_pick.zig", .glm = true },
};

/// `zig build` on Linux: the fatbins (-Dnvcc, else -Dfatbins prebuilt, else host-only), the server (`zig build
/// native`), the GPU test programs and the GLM-5.3 family's host tests (`zig build test-glm53`).
pub fn targets(b: *std.Build, target: Target, optimize: Mode, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    var images: [kernels.len]std.Build.LazyPath = undefined;
    const have_images = prebuilt != null or nvcc != null;
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    if (have_images) {
        const version: ?std.Build.LazyPath = if (prebuilt == null) blk: {
            const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
            run.has_side_effects = true;
            break :blk run.captureStdOut(.{});
        } else null;
        for (kernels, &images) |k, *image| {
            image.* = if (prebuilt) |dir| b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })) else fatbin(b, nvcc.?, version.?, k, sms);
            fatbin_step.dependOn(&b.addInstallFile(image.*, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
        }
    }
    const embedded: []const std.Build.LazyPath = if (have_images) &images else &.{};
    const cuda = runtime(b, target, optimize, embedded);
    const m = mods(b, target, optimize, cuda);
    for (test_programs) |p| {
        const root = module(b, p.src, target, optimize);
        root.addImport("cuda", cuda);
        if (p.glm) root.addImport("glm_moe_dsa", m.glm);
        b.installArtifact(b.addExecutable(.{ .name = p.name, .root_module = root }));
    }
    b.step("test-glm53", "glm_moe_dsa host-side unit tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m.glm })).step);
    const install = b.addInstallArtifact(server(b, target, m, build_options), .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the GLM-5.3 engine into zig-out/native/bin").dependOn(&install.step);
    serverTests(b, target, m);
}

/// The server's unit tests and its frozen parity goldens (fake_serve): `zig build test-server`, `test-server-golden`.
fn serverTests(b: *std.Build, target: Target, m: Mods) void {
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "engine_api", .module = m.api },
        .{ .name = "tokenizer", .module = m.tokenizer },
        .{ .name = "template", .module = template },
        .{ .name = "native_engines", .module = m.engines },
    };
    const srv = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = imports });
    const server_step = b.step("test-server", "The server's unit tests (zig/src/server/root.zig, tool_parse_test.zig, dashboard_test.zig) on this host");
    server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = srv })).step);
    // the parse half's tests and the dashboard's, each its own test binary (TensorFold ran them in its Metal build only)
    const tool_parse = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = imports });
    server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tool_parse })).step);
    const dashboard = b.createModule(.{ .root_source_file = b.path("zig/src/server/dashboard_test.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dashboard })).step);
    const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/fake_serve.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "server", .module = srv },
            .{ .name = "engine_api", .module = m.api },
            .{ .name = "tokenizer", .module = m.tokenizer },
            .{ .name = "template", .module = template },
        },
    }) });
    const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/server/golden_check.zig"), .target = target, .optimize = .Debug, .link_libc = true }) });
    const golden = b.addRunArtifact(golden_check);
    golden.addArtifactArg(fake_serve);
    golden.addFileArg(b.path("zig/tests/server/golden/cases.json"));
    golden.addDirectoryArg(b.path("zig/tests/server/golden"));
    golden.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    golden.setCwd(b.path("."));
    b.step("test-server-golden", "The frozen server parity goldens (fake_serve) on this host").dependOn(&golden.step);
}

/// Host unit tests (no GPU): the sampling and its match against TensorFold's, the engine contract, the engines module
/// and the CUDA runtime.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const m = mods(b, host, .debug, cuda);
    const match = module(b, "zig/src/expertatlas/sampling_match_test.zig", host, .debug);
    match.addImport("sampling", m.sampling);
    // TensorFold 1.0.2's Zig sampler (core/lanes/sampling_ref.zig); core/lanes/sampling.zig is now the same better C
    match.addImport("tf_sampling", module(b, "zig/src/core/lanes/sampling_ref.zig", host, .debug));
    for ([_]*Module{ cuda, m.sampling, match, m.api, m.engines }) |root| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
}

/// The release server for `target`, embedding the prebuilt fatbins in `prebuilt` (null: host-only, no kernels).
pub fn distServer(b: *std.Build, target: Target, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]std.Build.LazyPath = undefined;
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const embedded: []const std.Build.LazyPath = if (prebuilt != null) &images else &.{};
    const cuda = runtime(b, target, .fast, embedded);
    const exe = server(b, target, mods(b, target, .fast, cuda), build_options);
    exe.root_module.strip = true;
    return exe;
}

/// Every image the release server embeds is present and not empty.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
