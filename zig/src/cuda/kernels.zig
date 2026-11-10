// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! Our .cu kernels as fatbins built at build time with each Python extension's nvcc flags, embedded in the binary.

const std = @import("std");
const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no nvcc, no prebuilt fatbins); every image is then empty.
pub const available = options.with_kernels;

// Fiore-lab: the GLM-5.3 kernels only (zig/build/expertatlas.zig builds these seven); TensorFold's others are gone
pub const glm53_exl3_linear: []const u8 = if (available) &Blob("fatbin_glm53_exl3_linear").bytes else &.{};
pub const glm53_glue: []const u8 = if (available) &Blob("fatbin_glm53_glue").bytes else &.{};
pub const glm53_native_ops: []const u8 = if (available) &Blob("fatbin_glm53_native_ops").bytes else &.{};
pub const glm53_exl3_experts: []const u8 = if (available) &Blob("fatbin_glm53_exl3_experts").bytes else &.{};
pub const glm53_qmm_group: []const u8 = if (available) &Blob("fatbin_glm53_qmm_group").bytes else &.{};
pub const glm53_roce: []const u8 = if (available) &Blob("fatbin_glm53_roce").bytes else &.{};
pub const glm53_prompt_experts: []const u8 = if (available) &Blob("fatbin_glm53_prompt_experts").bytes else &.{};
