//! cuBLAS (not Lt) opened at run time: the strided batched GEMM torch.bmm calls, with torch's workspace, for bit-equal bmm.

const std = @import("std");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed };

const S = Status;
const D = abi.DevicePtr;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasGetVersion_v2: *const fn (Handle, *c_int) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, abi.Stream) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, D, usize) callconv(.c) S,
    cublasGemmStridedBatchedEx: *const fn (Handle, c_int, c_int, c_int, c_int, c_int, *const anyopaque, D, c_int, c_int, i64, D, c_int, c_int, i64, *const anyopaque, D, c_int, c_int, i64, c_int, c_int, c_int) callconv(.c) S,
};

/// torch's cuBLAS workspace (CUBLAS_WORKSPACE_CONFIG ":4096:8"): the heuristic sees this cap, so equal picks need it.
pub const torch_workspace: usize = 32 << 20;

/// One strided batched GEMM as cublasGemmStridedBatchedEx takes it (column-major, fp32 alpha/beta).
pub const Gemm = struct {
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    a: D,
    a_type: c_int,
    lda: c_int,
    stride_a: i64,
    b: D,
    b_type: c_int,
    ldb: c_int,
    stride_b: i64,
    c: D,
    c_type: c_int,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
    compute: c_int,
    algo: c_int,
};

pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    pub fn open() Error!Library {
        const path = if (std.c.getenv("TF_CUBLAS_LIB")) |p| std.mem.span(p) else "libcublas.so.13";
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(_: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        std.log.err("{s}: cuBLAS status {d}", .{ what, s });
        return error.CublasFailed;
    }

    /// alpha 1, beta 0, as torch.bmm calls it.
    pub fn gemmStridedBatched(self: *const Library, h: Handle, g: Gemm) Error!void {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        try self.check(self.api.cublasGemmStridedBatchedEx(h, g.transa, g.transb, g.m, g.n, g.k, &alpha, g.a, g.a_type, g.lda, g.stride_a, g.b, g.b_type, g.ldb, g.stride_b, &beta, g.c, g.c_type, g.ldc, g.stride_c, g.batch, g.compute, g.algo), "cublasGemmStridedBatchedEx");
    }
};
