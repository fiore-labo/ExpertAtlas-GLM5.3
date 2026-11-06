//! Tensor-parallel ranks on CUDA: rank 0's NCCL id over TCP, one communicator a rank, groups split from it.

const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Threaded = std.Io.Threaded;
const nccl = @import("nccl.zig");
const abi = @import("abi.zig");

pub const Error = error{ BadMaster, BadRank, SocketFailed, BindFailed, ListenFailed, ConnectTimeout, AcceptTimeout, ShortTransfer, BadPeer, Ring4NeedsRelay, Ring4RelaySpace, Ring4Unsupported, LinkClosed, BadWake, WakeTimeout } || nccl.Error;

/// How the ranks' links are wired. ring4: four nodes cabled as a ring, ranks in ring order (rank r's neighbours are
/// r ^ 1 and r ^ 3, its opposite r ^ 2, which no link reaches: glm53/R3RT-102.md).
/// ランクの結線。ring4: リングに結線した 4 台、ランクはリングの順 (隣は r ^ 1 と r ^ 3、向かい r ^ 2 には線がない)。
pub const Topology = enum { full, ring4 };

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
    /// Set by the host right after init (a split group is always full: its ranks are a cabled pair or one rank).
    /// init の直後にホストが設定する (split した group は常に full: 線でつながった組か 1 ランク)。
    topology: Topology = .full,

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
    /// A ring4 group refuses it (the opposite rank would hang its connection): use allToAllRelay.
    /// ring4 の group では断る (向かいのランクへの接続は止まる): allToAllRelay を使う。
    pub fn allToAll(g: *const Group, send: u64, recv: u64, n: usize, dt: nccl.DataType, stream: abi.Stream) Error!void {
        if (g.topology == .ring4) return error.Ring4NeedsRelay;
        const es = elemSize(dt);
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
        if (g.topology == .ring4) return error.Ring4Unsupported; // the DCP groups (DCP <= 2) are full / DCP の group (DCP <= 2) は full
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

    /// allToAll that works on any topology. Under ring4 the opposite rank's block goes half through each neighbour (the
    /// first floor(n / 2) elements through the sender's r ^ 1, the rest through its r ^ 3) in two NCCL groups on
    /// `stream` (graph-capturable): `recv` ends up byte for byte as a direct exchange leaves it. `relay` (device, at
    /// least n elements of `dt`) holds the halves in transit; it must not overlap send or recv, and both groups must not
    /// be wrapped in an outer NCCL group (the second reads what the first received). Elsewhere: allToAll.
    /// どの結線でも動く allToAll。ring4 では向かいのランクのブロックを半分ずつ両隣を通して送る (先頭の floor(n / 2) 要素は
    /// 送り手の r ^ 1、残りは r ^ 3 を通る)。`stream` 上の NCCL group 2 つ (graph に取れる)。`recv` は直接の交換と
    /// バイト単位で同じになる。`relay` (デバイス、`dt` の n 要素以上) は中継中の半分を置く。send / recv と重ねないこと、
    /// 2 つの group を外側の NCCL group で包まないこと (2 つ目は 1 つ目が受けたものを読む)。それ以外は allToAll。
    pub fn allToAllRelay(g: *const Group, send: u64, recv: u64, n: usize, dt: nccl.DataType, relay: u64, relay_bytes: usize, stream: abi.Stream) Error!void {
        if (g.topology != .ring4) {
            const es = elemSize(dt);
            const lib = g.lib;
            try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
            var post: Post = .{ .g = g, .dt = dt, .stream = stream };
            for (0..g.world) |j| {
                post.send(send + j * n * es, n, @intCast(j));
                post.recv(recv + j * n * es, n, @intCast(j));
            }
            return post.end();
        }
        if (g.world != 4) return error.Ring4Unsupported;
        const es = elemSize(dt);
        if (relay_bytes < n * es) return error.Ring4RelaySpace;
        const p = g.rank;
        const l = p ^ 1;
        const r = p ^ 3;
        const o = p ^ 2;
        const h = n / 2;
        const t = n - h;
        const a = relay; // l's first half for r (h elements) / l の r 宛ての前半 (h 要素)
        const b = relay + h * es; // r's second half for l (t elements) / r の l 宛ての後半 (t 要素)
        const lib = g.lib;
        // 1: this rank's blocks to both neighbours, then the opposite's halves through them; per peer the direct block
        // first, the relayed half second, on both ends (NCCL matches a pair's messages in issue order)
        // 1: 自分のブロックを両隣へ、続けて向かい宛ての半分をそれぞれへ。相手ごとに直接のブロックが先、中継の半分が後を
        // 両端でそろえる (NCCL は 1 組のメッセージを発行の順に対応させる)
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        var one: Post = .{ .g = g, .dt = dt, .stream = stream };
        one.send(send + p * n * es, n, p);
        one.recv(recv + p * n * es, n, p);
        one.send(send + l * n * es, n, l);
        one.send(send + o * n * es, h, l);
        one.send(send + r * n * es, n, r);
        one.send(send + (o * n + h) * es, t, r);
        one.recv(recv + l * n * es, n, l);
        one.recv(a, h, l);
        one.recv(recv + r * n * es, n, r);
        one.recv(b, t, r);
        try one.end();
        // 2: forward the neighbours' halves to each other; the opposite's block arrives, its first half from r, the
        // second from l
        // 2: 両隣の半分を互いへ回す。向かいのブロックが届く: 前半は r から、後半は l から
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        var two: Post = .{ .g = g, .dt = dt, .stream = stream };
        two.send(a, h, r);
        two.recv(recv + o * n * es, h, r);
        two.send(b, t, l);
        two.recv(recv + (o * n + h) * es, t, l);
        return two.end();
    }

    /// The sends and receives of one NCCL group: zero counts skipped (on both ends alike), the first failure kept, the
    /// group always ended.
    /// 1 つの NCCL group の send / recv: 0 要素は省く (両端で同じく)、最初の失敗を残し、group は必ず閉じる。
    const Post = struct {
        g: *const Group,
        dt: nccl.DataType,
        stream: abi.Stream,
        err: ?Error = null,

        fn send(s: *Post, buf: u64, count: usize, peer: u32) void {
            if (count == 0 or s.err != null) return;
            s.g.lib.check(s.g.lib.api.ncclSend(buf, count, s.dt, @intCast(peer), s.g.comm, s.stream), "ncclSend") catch |e| {
                s.err = e;
            };
        }

        fn recv(s: *Post, buf: u64, count: usize, peer: u32) void {
            if (count == 0 or s.err != null) return;
            s.g.lib.check(s.g.lib.api.ncclRecv(buf, count, s.dt, @intCast(peer), s.g.comm, s.stream), "ncclRecv") catch |e| {
                s.err = e;
            };
        }

        fn end(s: *Post) Error!void {
            const ended = s.g.lib.check(s.g.lib.api.ncclGroupEnd(), "ncclGroupEnd");
            if (s.err) |e| return e;
            return ended;
        }
    };

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

    // Fiore-lab (idle park): a parked replica's ranks wait on the rendezvous links, not in a collective. Rank 0 wakes
    // the others with one 'W' on each link and waits for each one's 'R' (it is running again); a rank woken reads the
    // 'W', gets ready, writes the 'R', then waits for rank 0's 'G', which comes only once every rank is ready. So no
    // rank enters a collective while another failed to wake: they see rank 0 end on their links instead (and a rank
    // killed inside a collective can leave its GPU busy). Only the world's group holds the links.
    // Fiore-lab (idle park): 休んでいるレプリカのランクは、集団通信ではなく待ち合わせの接続で待つ。rank 0 は各接続に
    // 'W' を 1 バイト書いて起こし、各ランクの 'R' (また動いている) を待つ。起こされたランクは 'W' を読み、準備して
    // 'R' を書く。接続を持つのは world の group だけ (split した group は持たない)。

    /// Whether this group can park: its rendezvous links are open (rank 0: one to each other rank; others: to rank 0).
    /// World 1 always can (nothing to wake).
    pub fn parkable(g: *const Group) bool {
        if (g.world <= 1) return true;
        if (g.world > max_links) return false;
        if (g.rank != 0) return g.links[0] >= 0;
        for (g.links[1..g.world]) |fd| if (fd < 0) return false;
        return true;
    }

    /// Rank 0: a 'W' to every other rank, then every one's 'R' within about `timeout_ms` (polls of 100 ms). A link
    /// closed or failing: LinkClosed; another byte: BadWake; an 'R' missing at the end: WakeTimeout. A rank that died
    /// while parked is seen before anything is written (its link reads as closed), so then no rank is woken: the
    /// others see rank 0 end instead of waiting in a collective that cannot complete.
    pub fn wakePeers(g: *const Group, timeout_ms: u32) Error!void {
        if (g.rank != 0 or g.world <= 1) return;
        if (!g.parkable()) return error.BadPeer;
        const peers = g.world - 1;
        var dead: [max_links]posix.pollfd = undefined;
        for (g.links[1..g.world], dead[0..peers]) |fd, *f| f.* = .{ .fd = fd, .events = posix.POLL.IN, .revents = 0 };
        if ((posix.poll(dead[0..peers], 0) catch return error.SocketFailed) > 0) for (dead[0..peers]) |f| if (f.revents != 0) {
            _ = (try readByte(f.fd)) orelse return error.LinkClosed;
            return error.BadWake; // a parked rank sends nothing
        };
        for (g.links[1..g.world]) |fd| writeAll(fd, "W") catch return error.LinkClosed;
        var got: [max_links]bool = @splat(false);
        var left: u32 = g.world - 1;
        var polls: u32 = (timeout_ms + 99) / 100;
        while (left > 0) {
            var fds: [max_links]posix.pollfd = undefined;
            var who: [max_links]usize = undefined;
            var n: usize = 0;
            for (1..g.world) |r| if (!got[r]) {
                fds[n] = .{ .fd = g.links[r], .events = posix.POLL.IN, .revents = 0 };
                who[n] = r;
                n += 1;
            };
            const seen = posix.poll(fds[0..n], 100) catch return error.SocketFailed;
            if (seen == 0) {
                if (polls == 0) return error.WakeTimeout;
                polls -= 1;
                continue;
            }
            for (fds[0..n], who[0..n]) |f, r| {
                if (f.revents == 0) continue;
                const b = (try readByte(f.fd)) orelse return error.LinkClosed;
                if (b != 'R') return error.BadWake;
                got[r] = true;
                left -= 1;
            }
        }
        for (g.links[1..g.world]) |fd| writeAll(fd, "G") catch return error.LinkClosed;
    }

    /// Ranks 1..: sleeps until rank 0's 'W' (no time limit: the replica may stay idle for days). The link closed or
    /// failing: LinkClosed (rank 0 is gone); another byte: BadWake.
    pub fn awaitWake(g: *const Group) Error!void {
        if (g.rank == 0 or g.world <= 1) return;
        if (g.links[0] < 0) return error.BadPeer;
        const b = (try readByte(g.links[0])) orelse return error.LinkClosed;
        if (b != 'W') return error.BadWake;
    }

    /// Ranks 1..: the 'R' rank 0 waits for in wakePeers, then rank 0's 'G' (every rank ready), before any collective.
    /// The link closed or failing while waiting: LinkClosed (rank 0 gave up the wake); another byte: BadWake.
    pub fn ready(g: *const Group) Error!void {
        if (g.rank == 0 or g.world <= 1) return;
        if (g.links[0] < 0) return error.BadPeer;
        writeAll(g.links[0], "R") catch return error.LinkClosed;
        const b = (try readByte(g.links[0])) orelse return error.LinkClosed;
        if (b != 'G') return error.BadWake;
    }
};

/// One byte from `fd` (a signal's interruption read again); null at the end of the stream; LinkClosed on an error.
fn readByte(fd: posix.socket_t) Error!?u8 {
    var b: [1]u8 = undefined;
    while (true) {
        const rc = posix.system.read(fd, &b, 1);
        switch (posix.errno(rc)) {
            .SUCCESS => return if (rc == 0) null else b[0],
            .INTR => continue,
            else => return error.LinkClosed,
        }
    }
}

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

/// Two connected stream sockets for the tests.
fn testPair() ![2]posix.socket_t {
    var fds: [2]posix.socket_t = undefined;
    if (posix.errno(posix.system.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketFailed;
    return fds;
}

/// A group with only links (the park handshake uses nothing else).
fn testGroup(rank: u32, world: u32) Group {
    return .{ .lib = undefined, .comm = null, .rank = rank, .world = world };
}

test "park: rank 0 wakes every rank and gets each one's ready" {
    var leader = testGroup(0, 3);
    var followers = [2]Group{ testGroup(1, 3), testGroup(2, 3) };
    for (&followers, 1..) |*f, r| {
        const p = try testPair();
        leader.links[r] = p[0];
        f.links[0] = p[1];
    }
    defer for (&followers) |*f| f.closeLinks();
    defer leader.closeLinks();
    try std.testing.expect(leader.parkable() and followers[0].parkable());
    const F = struct {
        fn go(g: *const Group, out: *?Error) void {
            g.awaitWake() catch |e| {
                out.* = e;
                return;
            };
            g.ready() catch |e| {
                out.* = e;
            };
        }
    };
    var errs: [2]?Error = .{ null, null };
    var threads: [2]std.Thread = undefined;
    for (&threads, &followers, &errs) |*t, *f, *e| t.* = try std.Thread.spawn(.{}, F.go, .{ f, e });
    try leader.wakePeers(5000);
    for (threads) |t| t.join();
    for (errs) |e| try std.testing.expectEqual(@as(?Error, null), e);
}

test "park: a closed link, a wrong byte and a missing ready" {
    // rank 0 gone while a rank sleeps: LinkClosed
    var a = testGroup(1, 2);
    var p = try testPair();
    a.links[0] = p[1];
    _ = posix.system.close(p[0]);
    try std.testing.expectError(error.LinkClosed, a.awaitWake());
    a.closeLinks();
    // a byte that is not 'W': BadWake
    p = try testPair();
    a.links[0] = p[1];
    try writeAll(p[0], "X");
    try std.testing.expectError(error.BadWake, a.awaitWake());
    _ = posix.system.close(p[0]);
    a.closeLinks();
    // rank 0: no 'R' within the time: WakeTimeout
    var leader = testGroup(0, 2);
    p = try testPair();
    leader.links[1] = p[0];
    try std.testing.expectError(error.WakeTimeout, leader.wakePeers(200));
    _ = posix.system.close(p[1]);
    leader.closeLinks();
    // rank 0: the rank takes the 'W' and its link closes (it died before its 'R'): LinkClosed. The close comes after
    // the read, so rank 0's write never meets a closed socket (no SIGPIPE in the test)
    p = try testPair();
    leader.links[1] = p[0];
    var b = testGroup(1, 2);
    b.links[0] = p[1];
    const D = struct {
        fn go(g: *Group) void {
            g.awaitWake() catch {};
            g.closeLinks();
        }
    };
    const t = try std.Thread.spawn(.{}, D.go, .{&b});
    try std.testing.expectError(error.LinkClosed, leader.wakePeers(5000));
    t.join();
    leader.closeLinks();
    // rank 0: one of two ranks died while parked: LinkClosed before any 'W', so the live rank is not woken
    var two = testGroup(0, 3);
    const p1 = try testPair();
    const p2 = try testPair();
    two.links[1] = p1[0];
    two.links[2] = p2[0];
    _ = posix.system.close(p1[1]);
    try std.testing.expectError(error.LinkClosed, two.wakePeers(1000));
    var none = [_]posix.pollfd{.{ .fd = p2[1], .events = posix.POLL.IN, .revents = 0 }};
    try std.testing.expectEqual(@as(usize, 0), try posix.poll(&none, 0));
    _ = posix.system.close(p2[1]);
    two.closeLinks();
    // rank 0 gives up the wake (a rank never answered) and ends: a rank that was ready reads the end, not a 'G'
    var giveup = testGroup(0, 3);
    const q1 = try testPair();
    const q2 = try testPair();
    giveup.links[1] = q1[0];
    giveup.links[2] = q2[0];
    var ok1 = testGroup(1, 3);
    ok1.links[0] = q1[1];
    const R = struct {
        fn go(g: *const Group, out: *?Error) void {
            g.awaitWake() catch |e| {
                out.* = e;
                return;
            };
            g.ready() catch |e| {
                out.* = e;
            };
        }
    };
    var got: ?Error = null;
    const th = try std.Thread.spawn(.{}, R.go, .{ &ok1, &got });
    try std.testing.expectError(error.WakeTimeout, giveup.wakePeers(300));
    giveup.closeLinks();
    th.join();
    try std.testing.expectEqual(@as(?Error, error.LinkClosed), got);
    ok1.closeLinks();
    _ = posix.system.close(q2[1]);
    // world 1 has nothing to wake; a missing link is not parkable
    try std.testing.expect(testGroup(0, 1).parkable());
    try std.testing.expect(!testGroup(0, 2).parkable());
}

test "master addresses parse as IP:port and refuse the rest" {
    const a = try parseMaster("192.0.2.10:29551");
    try std.testing.expectEqual(@as(u16, 29551), a.getPort());
    try std.testing.expectError(error.BadMaster, parseMaster("192.0.2.10"));
    try std.testing.expectError(error.BadMaster, parseMaster("node0:29551"));
}
