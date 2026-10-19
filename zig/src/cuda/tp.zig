//! Tensor-parallel ranks on CUDA: rank 0's NCCL id over TCP, one communicator a rank, groups split from it.

const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Threaded = std.Io.Threaded;
const nccl = @import("nccl.zig");
const abi = @import("abi.zig");

pub const Error = error{ BadMaster, BadRank, SocketFailed, BindFailed, ListenFailed, ConnectTimeout, AcceptTimeout, ShortTransfer, BadPeer } || nccl.Error;

/// Polls of 200 ms before a peer that never shows up fails the rendezvous (two minutes).
const wait_polls = 600;

/// ``HOST:PORT`` as an address (an IP literal: rendezvous runs on the ranks' own link, no resolver).
pub fn parseMaster(text: []const u8) Error!net.IpAddress {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadMaster;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.BadMaster;
    return net.IpAddress.parse(text[0..colon], port) catch error.BadMaster;
}

/// One NCCL communicator and this process's place in it.
pub const Group = struct {
    lib: *const nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    /// The rendezvous connections, kept open while the group lives (rank 0: one a peer on the master port; others:
    /// theirs to it), so the ranks show as established peers of the master port; -1: none.
    links: [max_links]posix.socket_t = @splat(-1),

    const max_links = 64;

    /// Every rank calls this with the same master; rank 0 makes the id and hands it to the others.
    pub fn init(lib: *const nccl.Library, rank: u32, world: u32, master: net.IpAddress) Error!Group {
        if (world == 0 or rank >= world) return error.BadRank;
        var id: nccl.UniqueId = undefined;
        var links: [max_links]posix.socket_t = @splat(-1);
        if (rank == 0) try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        if (world > 1) {
            if (rank == 0) try serveId(&id, world, master.getPort(), &links) else links[0] = try fetchId(&id, rank, master);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world, .links = links };
    }

    fn closeLinks(g: *Group) void {
        for (&g.links) |*fd| if (fd.* >= 0) {
            _ = posix.system.close(fd.*);
            fd.* = -1;
        };
    }

    /// ncclCommSplit: ranks with one color form a group, ordered by key (every rank of ``g`` must call it).
    pub fn split(g: *const Group, color: u32, key: u32) Error!Group {
        var comm: nccl.Comm = null;
        try g.lib.check(g.lib.api.ncclCommSplit(g.comm, @intCast(color), @intCast(key), &comm, null), "ncclCommSplit");
        var n: c_int = 0;
        var r: c_int = 0;
        try g.lib.check(g.lib.api.ncclCommCount(comm, &n), "ncclCommCount");
        try g.lib.check(g.lib.api.ncclCommUserRank(comm, &r), "ncclCommUserRank");
        return .{ .lib = g.lib, .comm = comm, .rank = @intCast(r), .world = @intCast(n) }; // no links: the parent's
    }

    /// comm.all_to_all: recv[j] <- rank j's send[self.rank], both [world, n] of `dt` contiguous, one NCCL group on
    /// `stream` (graph-capturable). Rows in rank order: a caller summing recv's rows 0 .. world - 1 gets the same bits
    /// on every rank (the exact reduce-scatter).
    pub fn allToAll(g: *const Group, send: u64, recv: u64, n: usize, dt: nccl.DataType, stream: abi.Stream) Error!void {
        const es: usize = switch (dt) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
        const lib = g.lib;
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        for (0..g.world) |j| {
            try lib.check(lib.api.ncclSend(send + j * n * es, n, dt, @intCast(j), g.comm, stream), "ncclSend");
            try lib.check(lib.api.ncclRecv(recv + j * n * es, n, dt, @intCast(j), g.comm, stream), "ncclRecv");
        }
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// Two all-to-all exchanges as one NCCL group (one launch where two calls of allToAll are two): a then b with
    /// every peer, in the same order on every rank.
    pub fn allToAll2(g: *const Group, send_a: u64, recv_a: u64, n_a: usize, dt_a: nccl.DataType, send_b: u64, recv_b: u64, n_b: usize, dt_b: nccl.DataType, stream: abi.Stream) Error!void {
        const ea = elemSize(dt_a);
        const eb = elemSize(dt_b);
        const lib = g.lib;
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        for (0..g.world) |j| {
            try lib.check(lib.api.ncclSend(send_a + j * n_a * ea, n_a, dt_a, @intCast(j), g.comm, stream), "ncclSend");
            try lib.check(lib.api.ncclRecv(recv_a + j * n_a * ea, n_a, dt_a, @intCast(j), g.comm, stream), "ncclRecv");
            try lib.check(lib.api.ncclSend(send_b + j * n_b * eb, n_b, dt_b, @intCast(j), g.comm, stream), "ncclSend");
            try lib.check(lib.api.ncclRecv(recv_b + j * n_b * eb, n_b, dt_b, @intCast(j), g.comm, stream), "ncclRecv");
        }
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    fn elemSize(dt: nccl.DataType) usize {
        return switch (dt) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }

    pub fn deinit(g: *Group) void {
        _ = g.lib.api.ncclCommDestroy(g.comm);
        g.comm = null;
        g.closeLinks();
    }

    /// Ends the communicator without waiting for peers (a failed rank stops every rank's collectives).
    pub fn abort(g: *Group) void {
        _ = g.lib.api.ncclCommAbort(g.comm);
        g.comm = null;
        g.closeLinks();
    }
};

/// Rank 0: accept the other ranks on any address at ``port``; each sends its rank (u32 LE) and receives the id.
fn serveId(id: *const nccl.UniqueId, world: u32, port: u16, keep: []posix.socket_t) Error!void {
    const any: net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = port } };
    const fd = try open(any);
    defer _ = posix.system.close(fd);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
    var storage: Threaded.PosixAddress = undefined;
    const len = Threaded.addressToPosix(&any, &storage);
    if (posix.errno(posix.system.bind(fd, &storage.any, len)) != .SUCCESS) return error.BindFailed;
    if (posix.errno(posix.system.listen(fd, 64)) != .SUCCESS) return error.ListenFailed;
    var seen: [256]bool = @splat(false);
    if (world > seen.len) return error.BadRank;
    var served: u32 = 1;
    var polls: u32 = 0;
    while (served < world) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 200) catch return error.SocketFailed;
        if (ready == 0) {
            polls += 1;
            if (polls >= wait_polls) return error.AcceptTimeout;
            continue;
        }
        const rc = posix.system.accept(fd, null, null);
        if (posix.errno(rc) != .SUCCESS) continue;
        const peer: posix.socket_t = @intCast(rc);
        var kept = false;
        defer if (!kept) {
            _ = posix.system.close(peer);
        };
        var word: [4]u8 = undefined;
        try readAll(peer, &word);
        const r = std.mem.readInt(u32, &word, .little);
        if (r == 0 or r >= world or seen[r]) return error.BadPeer;
        seen[r] = true;
        try writeAll(peer, &id.internal);
        if (r < keep.len) {
            keep[r] = peer;
            kept = true;
        }
        served += 1;
    }
}

/// Ranks 1..: connect to the master (retrying while rank 0 starts), send the rank, read the id.
fn fetchId(id: *nccl.UniqueId, rank: u32, master: net.IpAddress) Error!posix.socket_t {
    var storage: Threaded.PosixAddress = undefined;
    const len = Threaded.addressToPosix(&master, &storage);
    var polls: u32 = 0;
    while (true) : (polls += 1) {
        if (polls >= wait_polls) return error.ConnectTimeout;
        const fd = try open(master);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) {
            _ = posix.system.close(fd);
            var none = [_]posix.pollfd{};
            _ = posix.poll(&none, 200) catch {};
            continue;
        }
        errdefer _ = posix.system.close(fd);
        var word: [4]u8 = undefined;
        std.mem.writeInt(u32, &word, rank, .little);
        try writeAll(fd, &word);
        try readAll(fd, &id.internal);
        return fd; // kept open while the group lives (Group.links)
    }
}

fn open(address: net.IpAddress) Error!posix.socket_t {
    const family: u32 = if (address == .ip4) posix.AF.INET else posix.AF.INET6;
    const rc = posix.system.socket(family, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    return @intCast(rc);
}

fn readAll(fd: posix.socket_t, buf: []u8) Error!void {
    var got: usize = 0;
    while (got < buf.len) {
        const rc = posix.system.read(fd, buf[got..].ptr, buf.len - got);
        if (posix.errno(rc) != .SUCCESS or rc == 0) return error.ShortTransfer;
        got += @intCast(rc);
    }
}

fn writeAll(fd: posix.socket_t, buf: []const u8) Error!void {
    var put: usize = 0;
    while (put < buf.len) {
        const rc = posix.system.write(fd, buf[put..].ptr, buf.len - put);
        if (posix.errno(rc) != .SUCCESS or rc == 0) return error.ShortTransfer;
        put += @intCast(rc);
    }
}

test "master addresses parse as IP:port and refuse the rest" {
    const a = try parseMaster("192.0.2.10:29551");
    try std.testing.expectEqual(@as(u16, 29551), a.getPort());
    try std.testing.expectError(error.BadMaster, parseMaster("192.0.2.10"));
    try std.testing.expectError(error.BadMaster, parseMaster("node0:29551"));
}
