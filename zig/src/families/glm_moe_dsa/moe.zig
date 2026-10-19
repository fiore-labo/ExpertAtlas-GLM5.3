//! A MoE layer's routed experts in the tf layout (TF_GLM53_EXPERTS=tf): their tables on the GPU (trellis pointers and
//! widths, stacked suh / svh), the scratch, and the launch sequence over experts.zig's kernels.
//! tf レイアウト (TF_GLM53_EXPERTS=tf) の MoE 層の routed expert: GPU 上の表 (trellis のポインタと幅、積み重ねた
//! suh / svh)、scratch、experts.zig のカーネルを呼ぶ起動の手順。
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const experts = @import("experts.zig");

const Buf = cuda.DeviceBuffer;

/// The gate/up and down tile settings: (nt, warps, K splits, prefetch).
/// gate/up と down のタイル設定: (nt, warps, K 分割数, prefetch)。
const gateup_cfg = [4]u32{ 8, 4, 4, 1 };
const down_cfg = [4]u32{ 8, 4, 1, 1 };

pub const Routed = struct {
    // gate, up, down: every expert's trellis back to back, each 256-byte aligned
    // gate・up・down: 全 expert の trellis を続けて並べ、それぞれ 256 バイト境界にそろえる
    trellis: [3]Buf,
    ptr: [3]Buf, // int64 [E] into `trellis` / `trellis` を指す int64 [E]
    k2: [3]Buf, // int32 [E], half-bits a value / int32 [E]、1 値あたりの半 bit 数
    suh_g: Buf,
    suh_u: Buf,
    svh_g: Buf,
    svh_u: Buf,
    suh_d: Buf,
    svh_d: Buf,
    count: u32,
    dims: u32, // D
    width: u32, // I on this rank / このランクの I
    cb: u32,
    k2_gu: [2]u32, // (min, max) over gate and up / gate と up を通した (最小, 最大)
    k2_d: [2]u32,
};

fn alignUp(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}

/// This rank's share of `layer`'s routed experts, on the GPU.
/// `layer` の routed expert のうちこのランクの分を GPU に載せる。
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, cfg: Config, layer: usize) !Routed {
    const E = cfg.experts;
    const projs = [_][]const u8{ "gate", "up", "down" };
    var shares: [3][]weights.Share = undefined;
    var scales: [3][2][]weights.Share = undefined;
    defer for (0..3) |p| {
        for (shares[p]) |*s| s.deinit(gpa);
        gpa.free(shares[p]);
        for (0..2) |j| {
            for (scales[p][j]) |*s| s.deinit(gpa);
            gpa.free(scales[p][j]);
        }
    };
    var buf: [192]u8 = undefined;
    for (projs, 0..) |proj, p| {
        shares[p] = try gpa.alloc(weights.Share, E);
        scales[p][0] = try gpa.alloc(weights.Share, E);
        scales[p][1] = try gpa.alloc(weights.Share, E);
        for (0..E) |e| {
            shares[p][e] = try r.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.trellis", .{ layer, e, proj }));
            scales[p][0][e] = try r.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.suh", .{ layer, e, proj }));
            scales[p][1][e] = try r.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.svh", .{ layer, e, proj }));
            if (scales[p][0][e].dtype != .f16 or scales[p][1][e].dtype != .f16) return error.PackedScalesUnsupported;
        }
    }
    const t0 = shares[0][0];
    const D: u32 = @intCast(t0.shape[0] * 16);
    const I: u32 = @intCast(t0.shape[1] * 16);
    var out: Routed = undefined;
    out.count = @intCast(E);
    out.dims = D;
    out.width = I;
    var k2s: [3][2]u32 = .{ .{ 99, 0 }, .{ 99, 0 }, .{ 99, 0 } };
    for (0..3) |p| {
        var total: usize = 0;
        for (shares[p]) |s| total += alignUp(s.bytes.len);
        out.trellis[p] = try Buf.alloc(d, total);
        const ptrs = try gpa.alloc(i64, E);
        defer gpa.free(ptrs);
        const k2 = try gpa.alloc(i32, E);
        defer gpa.free(k2);
        var off: usize = 0;
        for (shares[p], 0..) |s, e| {
            if (s.dtype != .i16 or s.rank != 3 or s.shape[2] % 8 != 0) return error.BadTrellis;
            try out.trellis[p].upload(off, s.bytes);
            ptrs[e] = @intCast(out.trellis[p].ptr + off);
            k2[e] = @intCast(s.shape[2] / 8);
            k2s[p][0] = @min(k2s[p][0], @as(u32, @intCast(k2[e])));
            k2s[p][1] = @max(k2s[p][1], @as(u32, @intCast(k2[e])));
            off += alignUp(s.bytes.len);
        }
        out.ptr[p] = try Buf.fromHost(d, std.mem.sliceAsBytes(ptrs));
        out.k2[p] = try Buf.fromHost(d, std.mem.sliceAsBytes(k2));
    }
    out.k2_gu = .{ @min(k2s[0][0], k2s[1][0]), @max(k2s[0][1], k2s[1][1]) };
    out.k2_d = k2s[2];
    out.suh_g = try stack(gpa, d, scales[0][0]);
    out.suh_u = try stack(gpa, d, scales[1][0]);
    out.svh_g = try stack(gpa, d, scales[0][1]);
    out.svh_u = try stack(gpa, d, scales[1][1]);
    out.suh_d = try stack(gpa, d, scales[2][0]);
    out.svh_d = try stack(gpa, d, scales[2][1]);
    var cbuf: [192]u8 = undefined;
    out.cb = if (r.has(try std.fmt.bufPrint(&cbuf, "model.layers.{d}.mlp.experts.0.gate_proj.mcg", .{layer}))) 1 else if (r.has(try std.fmt.bufPrint(&cbuf, "model.layers.{d}.mlp.experts.0.gate_proj.mul1", .{layer}))) 2 else 0;
    return out;
}

fn stack(gpa: std.mem.Allocator, d: *const cuda.Driver, parts: []const weights.Share) !Buf {
    var total: usize = 0;
    for (parts) |s| total += s.bytes.len;
    const host = try gpa.alloc(u8, total);
    defer gpa.free(host);
    var off: usize = 0;
    for (parts) |s| {
        @memcpy(host[off..][0..s.bytes.len], s.bytes);
        off += s.bytes.len;
    }
    return Buf.fromHost(d, host);
}

/// Scratch for up to `rows` rows of `slots` slots (`members` filled with -1 once).
/// 最大 `rows` 行 x `slots` スロット分の scratch (`members` は最初に一度 -1 で埋める)。
pub const Scratch = struct {
    xg: Buf,
    xu: Buf,
    xd: Buf,
    z: Buf,
    y: Buf,
    ids: Buf,
    count: Buf,
    members: Buf,
    rows: u32,
    slots: u32,

    pub fn init(d: *const cuda.Driver, ex: *const Routed, rows: u32, slots: u32) !Scratch {
        const P: usize = @as(usize, rows) * slots;
        const D: usize = ex.dims;
        const I: usize = ex.width;
        const maxu: usize = @min(P, ex.count);
        var s: Scratch = .{
            .xg = try Buf.alloc(d, P * D * 2),
            .xu = try Buf.alloc(d, P * D * 2),
            .xd = try Buf.alloc(d, P * I * 2),
            .z = try Buf.alloc(d, @max(2 * gateup_cfg[2] * I, down_cfg[2] * D) * P * 4),
            .y = try Buf.alloc(d, P * D * 4),
            .ids = try Buf.alloc(d, maxu * 4),
            .count = try Buf.alloc(d, 16),
            .members = try Buf.alloc(d, maxu * rows * 4),
            .rows = rows,
            .slots = slots,
        };
        inline for (.{ "xg", "xu", "xd", "z", "y", "ids", "count" }) |f| try @field(s, f).fill8(0, null);
        try s.members.fill32(0xffffffff, null);
        return s;
    }
};

/// R rows' routed experts weighted by `wts` into `out` fp32 [R, D] (the down epilogue and combine in one launch).
/// R 行の routed expert を `wts` で重み付けして `out` fp32 [R, D] に出す (down の epilogue と combine は 1 回の起動)。
pub fn routed(k: *const experts.Kernels, stream: cuda.Stream, x: u64, x_stride: u32, pick: u64, wts: u64, ex: *const Routed, s: *const Scratch, out: u64, R: u32) !void {
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const slots = s.slots;
    const P = R * slots;
    if (R > s.rows) return error.ScratchTooSmall;
    const maxu = @min(P, E);
    try k.group(stream, pick, s.ids.ptr, s.count.ptr, s.members.ptr, R, slots, E, R);
    try k.rotIn(stream, x, x_stride, pick, ex.suh_g.ptr, ex.suh_u.ptr, s.xg.ptr, s.xu.ptr, R, D, slots, E);
    try k.grouped(stream, .{ .x0 = s.xg.ptr, .x1 = s.xu.ptr, .tp0 = ex.ptr[0].ptr, .tp1 = ex.ptr[1].ptr, .k2_0 = ex.k2[0].ptr, .k2_1 = ex.k2[1].ptr, .uids = s.ids.ptr, .ucount = s.count.ptr, .members = s.members.ptr, .z = s.z.ptr, .K = D, .N = I, .P = P, .SK = gateup_cfg[2], .maxm = R, .slots = slots, .nexp_max = maxu, .mats = 2, .cb = ex.cb, .nt = gateup_cfg[0], .warps = gateup_cfg[1], .pf = gateup_cfg[3], .lo = ex.k2_gu[0], .hi = ex.k2_gu[1] });
    try k.gateupEpilogue(stream, s.z.ptr, pick, ex.svh_g.ptr, ex.svh_u.ptr, ex.suh_d.ptr, s.xd.ptr, R, P, I, gateup_cfg[2], slots, E, std.math.inf(f32), 1);
    try k.grouped(stream, .{ .x0 = s.xd.ptr, .x1 = s.xd.ptr, .tp0 = ex.ptr[2].ptr, .tp1 = ex.ptr[2].ptr, .k2_0 = ex.k2[2].ptr, .k2_1 = ex.k2[2].ptr, .uids = s.ids.ptr, .ucount = s.count.ptr, .members = s.members.ptr, .z = s.z.ptr, .K = I, .N = D, .P = P, .SK = down_cfg[2], .maxm = R, .slots = slots, .nexp_max = maxu, .mats = 1, .cb = ex.cb, .nt = down_cfg[0], .warps = down_cfg[1], .pf = down_cfg[3], .lo = ex.k2_d[0], .hi = ex.k2_d[1] });
    try k.downCombine(stream, s.z.ptr, pick, ex.svh_d.ptr, s.y.ptr, wts, out, R, P, D, down_cfg[2], slots, E);
}
