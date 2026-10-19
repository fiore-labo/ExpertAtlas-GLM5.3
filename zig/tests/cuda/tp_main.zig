//! Two-or-more-rank NCCL check: `tf-tp-test RANK WORLD HOST:PORT`, one process a GPU; RESULT lines hash each output.
//! tools/zig/tp_oracle.py runs the same collectives on the same inputs from Python; equal hashes mean equal sums.

const std = @import("std");
const cuda = @import("cuda");

const golden: u64 = 0x9E3779B97F4A7C15;

/// splitmix64 of the i-th draw of ``seed``: the same as tp_oracle.py's numpy rows.
fn mix(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% golden;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Exact f32 in [-1, 1): 24 random bits.
fn value(seed: u64, i: u64) f32 {
    const top: f32 = @floatFromInt(mix(seed, i) >> 40);
    return top / 16777216.0 * 2.0 - 1.0;
}

const Kind = enum { f32, bf16 };

/// Little-endian rows: f32 as is, bf16 as the f32's top half (truncated, exact for these 24-bit values' signs and exponents).
fn fill(gpa: std.mem.Allocator, kind: Kind, seed: u64, n: usize) ![]u8 {
    const out = try gpa.alloc(u8, n * width(kind));
    for (0..n) |i| {
        const bits: u32 = @bitCast(value(seed, i));
        switch (kind) {
            .f32 => std.mem.writeInt(u32, out[i * 4 ..][0..4], bits, .little),
            .bf16 => std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate(bits >> 16), .little),
        }
    }
    return out;
}

fn seedOf(op: u64, rank: u32) u64 {
    return (op << 32) | rank;
}

fn dtype(kind: Kind) cuda.nccl.DataType {
    return if (kind == .f32) .f32 else .bf16;
}

fn width(kind: Kind) usize {
    return if (kind == .f32) 4 else 2;
}

const Ctx = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: *const cuda.tp.Group,
    stream: cuda.Stream,

    fn check(c: Ctx, r: cuda.nccl.Result, what: []const u8) !void {
        try c.g.lib.check(r, what);
    }

    fn result(c: Ctx, name: []const u8, buf: cuda.DeviceBuffer, len: usize) !void {
        try c.stream.synchronize();
        const host = try c.gpa.alloc(u8, len);
        defer c.gpa.free(host);
        try buf.download(0, host);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &digest, .{});
        std.debug.print("RESULT {s} {s}\n", .{ name, std.fmt.bytesToHex(digest, .lower) });
    }

    fn input(c: Ctx, kind: Kind, op: u64, n: usize) !cuda.DeviceBuffer {
        const bytes = try fill(c.gpa, kind, seedOf(op, c.g.rank), n);
        defer c.gpa.free(bytes);
        return cuda.DeviceBuffer.fromHost(c.d, bytes);
    }
};

fn allReduce(c: Ctx, name: []const u8, kind: Kind, op: u64, n: usize) !void {
    var send = try c.input(kind, op, n);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(c.d, n * width(kind));
    defer recv.free();
    try c.check(c.g.lib.api.ncclAllReduce(send.ptr, recv.ptr, n, dtype(kind), .sum, c.g.comm, c.stream.handle), "ncclAllReduce");
    try c.result(name, recv, n * width(kind));
}

fn allGather(c: Ctx, op: u64, n: usize) !void {
    var send = try c.input(.bf16, op, n);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(c.d, n * 2 * c.g.world);
    defer recv.free();
    try c.check(c.g.lib.api.ncclAllGather(send.ptr, recv.ptr, n, .bf16, c.g.comm, c.stream.handle), "ncclAllGather");
    try c.result("allgather_bf16", recv, n * 2 * c.g.world);
}

fn reduceScatter(c: Ctx, op: u64, n: usize) !void {
    var send = try c.input(.bf16, op, n * c.g.world);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(c.d, n * 2);
    defer recv.free();
    try c.check(c.g.lib.api.ncclReduceScatter(send.ptr, recv.ptr, n, .bf16, .sum, c.g.comm, c.stream.handle), "ncclReduceScatter");
    try c.result("reducescatter_bf16", recv, n * 2);
}

fn allToAll(c: Ctx, op: u64, n: usize) !void {
    var send = try c.input(.bf16, op, n * c.g.world);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(c.d, n * 2 * c.g.world);
    defer recv.free();
    const api = c.g.lib.api;
    try c.check(api.ncclGroupStart(), "ncclGroupStart");
    for (0..c.g.world) |p| {
        try c.check(api.ncclSend(try send.at(p * n * 2), n, .bf16, @intCast(p), c.g.comm, c.stream.handle), "ncclSend");
        try c.check(api.ncclRecv(try recv.at(p * n * 2), n, .bf16, @intCast(p), c.g.comm, c.stream.handle), "ncclRecv");
    }
    try c.check(api.ncclGroupEnd(), "ncclGroupEnd");
    try c.result("alltoall_bf16", recv, n * 2 * c.g.world);
}

fn timed(c: Ctx, name: []const u8, n: usize, reps: usize) !void {
    var send = try c.input(.bf16, 99, n);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(c.d, n * 2);
    defer recv.free();
    var start = try cuda.Event.init(c.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(c.d, true);
    defer end.deinit();
    for (0..3) |_| try c.check(c.g.lib.api.ncclAllReduce(send.ptr, recv.ptr, n, .bf16, .sum, c.g.comm, c.stream.handle), "warm");
    try start.record(c.stream);
    for (0..reps) |_| try c.check(c.g.lib.api.ncclAllReduce(send.ptr, recv.ptr, n, .bf16, .sum, c.g.comm, c.stream.handle), "timed");
    try end.record(c.stream);
    try end.synchronize();
    const ms = try cuda.Event.elapsedMs(start, end) / @as(f32, @floatFromInt(reps));
    const bus = @as(f64, @floatFromInt(n * 2)) * 2.0 * @as(f64, @floatFromInt(c.g.world - 1)) / @as(f64, @floatFromInt(c.g.world)) / (@as(f64, ms) * 1e6);
    std.debug.print("BW {s} bytes={d} avg_us={d:.1} busbw_GBps={d:.2}\n", .{ name, n * 2, ms * 1000.0, bus });
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: tf-tp-test RANK WORLD HOST:PORT\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const master = try cuda.tp.parseMaster(args[3]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var lib = try cuda.nccl.Library.openEnv();
    defer lib.close();
    std.debug.print("INFO rank {d}/{d} nccl {d}\n", .{ rank, world, try lib.version() });
    var g = try cuda.tp.Group.init(&lib, rank, world, master);
    defer g.deinit();
    const c: Ctx = .{ .gpa = init.gpa, .d = &driver, .g = &g, .stream = try cuda.Stream.init(&driver, true) };
    try allReduce(c, "allreduce_f32", .f32, 1, 1 << 20);
    try allReduce(c, "allreduce_bf16", .bf16, 2, 1 << 20);
    try allReduce(c, "allreduce_bf16_window8", .bf16, 3, 8 * 6144);
    try allGather(c, 4, 1 << 18);
    try reduceScatter(c, 5, 1 << 18);
    try allToAll(c, 6, 1 << 16);
    var pair = try g.split(rank / 2, rank % 2);
    defer pair.deinit();
    const pc: Ctx = .{ .gpa = init.gpa, .d = &driver, .g = &pair, .stream = c.stream };
    std.debug.print("INFO split color {d}: rank {d}/{d}\n", .{ rank / 2, pair.rank, pair.world });
    try allReduce(pc, "split_allreduce_bf16", .bf16, 7, 1 << 16);
    try timed(c, "window8", 8 * 6144, 500);
    try timed(c, "large", 32 << 20, 10);
    std.debug.print("PASS tp {d}/{d}\n", .{ rank, world });
    return 0;
}
