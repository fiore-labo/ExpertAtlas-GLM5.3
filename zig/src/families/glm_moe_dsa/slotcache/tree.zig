//! The slot cache's metadata and planner. Every rank makes the same plan from the same metadata and checks this by
//! digest: SHA-256 over the Python repr text of the metadata and the plan. No result depends on the order nodes are
//! visited (every selection key holds the nid).
//! スロットキャッシュのメタデータとプランナ。全ランクが同じメタデータから同じプランを作り、
//! digest (メタデータとプランの Python repr 文字列の SHA-256) で一致を確かめる。
//! 結果はノードを訪れる順に依らない (選択のキーはどれも nid を含む)。
const std = @import("std");
const alloc_mod = @import("alloc.zig");
const hashing = @import("hashing.zig");

pub const Extent = alloc_mod.Extent;
pub const Store = alloc_mod.Store;
pub const Interval = alloc_mod.Interval;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The root's nid.
/// root の nid。
pub const ROOT: i64 = 0;

pub const Error = error{ RequestTooLong, PlanAssert, BadSplit, NoNode } || alloc_mod.Error;

pub const Loc = enum {
    R,
    A,
    S,
    fn name(l: Loc) []const u8 {
        return switch (l) {
            .R => "R",
            .A => "A",
            .S => "S",
        };
    }
};

comptime {
    // ids hash as int64 LE bytes
    // ids は int64 LE のバイト列としてハッシュする
    std.debug.assert(@import("builtin").cpu.arch.endian() == .little);
}

/// SHA-256 of the ids (int64 LE) in lower-case hex, kept per node.
/// ids (int64 LE) の SHA-256 を小文字の 16 進で。ノードごとに持つ。
fn idsHex(ids: []const i64) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(std.mem.sliceAsBytes(ids), &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// A token segment at absolute positions [start, end) with the generation that computed its rows; a path from the
/// root to a leaf is a conversation. `loc`: A on the compute path, S stored (extents congruent to their positions),
/// R the root.
/// 絶対位置 [start, end) のトークン列と、その行を計算した世代。root から葉までのパスが 1 つの会話。
/// `loc`: A は計算パス上、S は保存済み (extent は位置と合同)、R は root。
pub const Node = struct {
    nid: i64,
    parent: ?i64,
    start: i64,
    ids: []i64, // owned; token ids of [start, end) / 所有する。[start, end) のトークン ID
    hex: [64]u8, // idsHex(ids)
    gen: i64,
    loc: Loc,
    ext: std.ArrayList(Extent) = .empty,
    children: std.ArrayList(i64) = .empty,
    last: i64 = 0,
    pinned: bool = false,

    pub fn end(n: *const Node) i64 {
        return n.start + @as(i64, @intCast(n.ids.len));
    }

    fn deinit(n: *Node, gpa: Allocator) void {
        gpa.free(n.ids);
        n.ext.deinit(gpa);
        n.children.deinit(gpa);
    }

    fn clone(n: *const Node, gpa: Allocator) Allocator.Error!Node {
        var c = n.*;
        c.ids = try gpa.dupe(i64, n.ids);
        errdefer gpa.free(c.ids);
        c.ext = try n.ext.clone(gpa);
        errdefer c.ext.deinit(gpa);
        c.children = try n.children.clone(gpa);
        return c;
    }
};

fn removeFirst(list: *std.ArrayList(i64), x: i64) bool {
    const i = std.mem.indexOfScalar(i64, list.items, x) orelse return false;
    _ = list.orderedRemove(i);
    return true;
}

/// The common prefix length of a path's tokens and X, walking the path (no concatenation).
/// パスのトークン列と X の共通 prefix の長さ。連結せずにパスをたどる。
fn lcpPath(m: *const Meta, path: []const i64, X: []const i64) Error!i64 {
    var pos: usize = 0;
    for (path) |nid| {
        for ((try m.atc(nid)).ids) |t| {
            if (pos >= X.len or t != X[pos]) return @intCast(pos);
            pos += 1;
        }
    }
    return @intCast(pos);
}

/// The repr streams: into SHA-256 (digests) or into text (tests).
/// repr の出力先: SHA-256 (digest 用) か文字列 (テスト用)。
const HashSink = struct {
    h: *Sha256,
    fn put(s: HashSink, b: []const u8) error{}!void {
        s.h.update(b);
    }
    fn line(_: HashSink) error{}!void {}
};

const TextSink = struct {
    gpa: Allocator,
    out: *std.ArrayList(u8),
    fn put(s: TextSink, b: []const u8) Allocator.Error!void {
        try s.out.appendSlice(s.gpa, b);
    }
    fn line(s: TextSink) Allocator.Error!void {
        try s.out.append(s.gpa, '\n');
    }
};

fn putInt(sink: anytype, x: i64) !void {
    var buf: [24]u8 = undefined;
    try sink.put(std.fmt.bufPrint(&buf, "{d}", .{x}) catch unreachable);
}

fn putOpt(sink: anytype, x: ?i64) !void {
    if (x) |v| try putInt(sink, v) else try sink.put("None");
}

fn putExtents(sink: anytype, ext: []const Extent) !void {
    try sink.put("[");
    for (ext, 0..) |e, i| {
        if (i > 0) try sink.put(", ");
        try sink.put("(");
        try putInt(sink, e.s);
        try sink.put(", ");
        try putInt(sink, e.a);
        try sink.put(", ");
        try putInt(sink, e.n);
        try sink.put(")");
    }
    try sink.put("]");
}

/// The metadata every rank keeps: the nodes, the compute path (`active`) and the store's free intervals.
/// 全ランクが持つメタデータ: ノード、計算パス (`active`)、ストアの空き区間。
pub const Meta = struct {
    gpa: Allocator,
    // index = nid (null once removed); len == next_id
    // 添字が nid (削除後は null)。len == next_id
    nodes: std.ArrayList(?Node) = .empty,
    next_id: i64 = 1,
    active: std.ArrayList(i64) = .empty,
    store: Store,
    epoch: i64 = 0,
    tick: i64 = 0,

    pub fn init(gpa: Allocator, S: i64, dcp: i64) Error!Meta {
        var m: Meta = .{ .gpa = gpa, .store = try Store.init(gpa, S, dcp) };
        errdefer m.store.deinit();
        try m.nodes.append(gpa, .{ .nid = ROOT, .parent = null, .start = 0, .ids = try gpa.alloc(i64, 0), .hex = idsHex(&.{}), .gen = 0, .loc = .R });
        return m;
    }

    pub fn deinit(m: *Meta) void {
        for (m.nodes.items) |*n| if (n.*) |*x| x.deinit(m.gpa);
        m.nodes.deinit(m.gpa);
        m.active.deinit(m.gpa);
        m.store.deinit();
        m.* = undefined;
    }

    /// A deep copy with its own allocator.
    /// 独自のアロケータを持つ深いコピー。
    pub fn clone(m: *const Meta, gpa: Allocator) Error!Meta {
        var c: Meta = .{ .gpa = gpa, .next_id = m.next_id, .epoch = m.epoch, .tick = m.tick, .store = undefined };
        c.store = .{ .gpa = gpa, .S = m.store.S, .dcp = m.store.dcp, .free = try m.store.free.clone(gpa) };
        errdefer c.deinit();
        c.active = try m.active.clone(gpa);
        try c.nodes.ensureTotalCapacity(gpa, m.nodes.items.len);
        for (m.nodes.items) |n| c.nodes.appendAssumeCapacity(if (n) |x| try x.clone(gpa) else null);
        return c;
    }

    pub fn has(m: *const Meta, nid: i64) bool {
        return nid >= 0 and nid < m.nodes.items.len and m.nodes.items[@intCast(nid)] != null;
    }

    pub fn at(m: *Meta, nid: i64) Error!*Node {
        if (!m.has(nid)) return error.NoNode;
        return &m.nodes.items[@intCast(nid)].?;
    }

    pub fn atc(m: *const Meta, nid: i64) Error!*const Node {
        if (!m.has(nid)) return error.NoNode;
        return &m.nodes.items[@intCast(nid)].?;
    }

    // -- queries

    /// The nids from the root to nid, the root excluded (caller frees).
    /// root から nid までの nid の列。root は含まない (呼び出し側が解放)。
    pub fn path(m: *const Meta, gpa: Allocator, nid0: i64) Error![]i64 {
        var out: std.ArrayList(i64) = .empty;
        errdefer out.deinit(gpa);
        var nid = nid0;
        while (nid != ROOT) {
            try out.append(gpa, nid);
            nid = (try m.atc(nid)).parent orelse return error.NoNode;
        }
        std.mem.reverse(i64, out.items);
        return out.toOwnedSlice(gpa);
    }

    /// The path's token ids concatenated (caller frees).
    /// パスのトークン ID を連結したもの (呼び出し側が解放)。
    pub fn tokens(m: *const Meta, gpa: Allocator, nid: i64) Error![]i64 {
        const p = try m.path(gpa, nid);
        defer gpa.free(p);
        var out: std.ArrayList(i64) = .empty;
        errdefer out.deinit(gpa);
        for (p) |x| try out.appendSlice(gpa, (try m.atc(x)).ids);
        return out.toOwnedSlice(gpa);
    }

    /// The compute path's length in positions.
    /// 計算パスの長さ (位置数)。
    pub fn nAct(m: *const Meta) i64 {
        if (m.active.items.len == 0) return 0;
        return m.nodes.items[@intCast(m.active.items[m.active.items.len - 1])].?.end();
    }

    /// The compute path's last node (the root when it is empty).
    /// 計算パスの最後のノード (空なら root)。
    pub fn actLeaf(m: *const Meta) i64 {
        return if (m.active.items.len == 0) ROOT else m.active.items[m.active.items.len - 1];
    }

    /// The leaves, and the compute path's last node (caller frees).
    /// 葉と、計算パスの最後のノード (呼び出し側が解放)。
    pub fn candidates(m: *const Meta, gpa: Allocator) Error![]i64 {
        var out: std.ArrayList(i64) = .empty;
        errdefer out.deinit(gpa);
        for (m.nodes.items) |n| if (n) |x| if (x.nid != ROOT and x.children.items.len == 0) try out.append(gpa, x.nid);
        if (m.active.items.len > 0) {
            const l = m.actLeaf();
            if (std.mem.indexOfScalar(i64, out.items, l) == null) try out.append(gpa, l);
        }
        return out.toOwnedSlice(gpa);
    }

    fn writeRepr(m: *const Meta, sink: anytype) !void {
        for (m.nodes.items) |nn| {
            const n = nn orelse continue;
            try sink.put("(");
            try putInt(sink, n.nid);
            try sink.put(", ");
            try putOpt(sink, n.parent);
            try sink.put(", ");
            try putInt(sink, n.start);
            try sink.put(", ");
            try putInt(sink, @intCast(n.ids.len));
            try sink.put(", '");
            try sink.put(&n.hex);
            try sink.put("', ");
            try putInt(sink, n.gen);
            try sink.put(", '");
            try sink.put(n.loc.name());
            try sink.put("', ");
            try putExtents(sink, n.ext.items);
            try sink.put(", [");
            // the children in ascending order without allocating (they are distinct)
            // 子を割り当てなしで昇順に出す (子はすべて異なる)
            var prev: ?i64 = null;
            for (0..n.children.items.len) |i| {
                var next: ?i64 = null;
                for (n.children.items) |ch| if ((prev == null or ch > prev.?) and (next == null or ch < next.?)) {
                    next = ch;
                };
                if (i > 0) try sink.put(", ");
                try putInt(sink, next.?);
                prev = next;
            }
            try sink.put("], ");
            try sink.put(if (n.pinned) "True" else "False");
            try sink.put(")");
            try sink.line();
        }
        try sink.put("([");
        for (m.active.items, 0..) |a, i| {
            if (i > 0) try sink.put(", ");
            try putInt(sink, a);
        }
        try sink.put("], [");
        for (m.store.free.items, 0..) |f, i| {
            if (i > 0) try sink.put(", ");
            try sink.put("[");
            try putInt(sink, f.lo);
            try sink.put(", ");
            try putInt(sink, f.hi);
            try sink.put("]");
        }
        try sink.put("], ");
        try putInt(sink, m.store.S);
        try sink.put(", ");
        try putInt(sink, m.epoch);
        try sink.put(", ");
        try putInt(sink, m.next_id);
        try sink.put(")");
    }

    /// SHA-256 of the nodes' reprs (nid order) and of (active, free, S, epoch, next_id); the first 16 B.
    /// ノードの repr (nid 順) と (active, free, S, epoch, next_id) の SHA-256。先頭 16 B。
    pub fn digest(m: *const Meta) [16]u8 {
        var h = Sha256.init(.{});
        m.writeRepr(HashSink{ .h = &h }) catch unreachable;
        return h.finalResult()[0..16].*;
    }

    /// The repr lines `digest` hashes, newline-separated (tests; caller frees).
    /// `digest` がハッシュする repr 行を改行区切りで返す (テスト用。呼び出し側が解放)。
    pub fn reprText(m: *const Meta, gpa: Allocator) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try m.writeRepr(TextSink{ .gpa = gpa, .out = &out });
        return out.toOwnedSlice(gpa);
    }

    /// The extents of every stored node (caller frees).
    /// 保存済みの全ノードの extent (呼び出し側が解放)。
    pub fn storedExtents(m: *const Meta, gpa: Allocator) Allocator.Error![]Extent {
        var out: std.ArrayList(Extent) = .empty;
        errdefer out.deinit(gpa);
        for (m.nodes.items) |n| if (n) |x| if (x.loc == .S) try out.appendSlice(gpa, x.ext.items);
        return out.toOwnedSlice(gpa);
    }

    pub const Summary = struct { stored: i64, leaves: usize, nodes: usize, free: i64, active: i64 };

    /// Stored positions, leaves, nodes (the root excluded), free store positions, compute path length.
    /// 保存済みの位置数、葉の数、ノード数 (root を除く)、ストアの空き位置数、計算パスの長さ。
    pub fn summary(m: *const Meta) Summary {
        var s: Summary = .{ .stored = 0, .leaves = 0, .nodes = 0, .free = m.store.freeTotal(), .active = m.nAct() };
        for (m.nodes.items) |n| if (n) |x| if (x.nid != ROOT) {
            s.nodes += 1;
            if (x.children.items.len == 0) s.leaves += 1;
            if (x.loc == .S) s.stored += @intCast(x.ids.len);
        };
        return s;
    }

    // -- edits

    /// A new child of parent holding ids from parent's end; returns its nid.
    /// parent の末尾から ids を持つ新しい子を作り、その nid を返す。
    pub fn newNode(m: *Meta, parent: i64, ids: []const i64, gen: i64, loc: Loc) Error!i64 {
        const start = (try m.atc(parent)).end();
        const nid = m.next_id;
        const own = try m.gpa.dupe(i64, ids);
        errdefer m.gpa.free(own);
        try m.nodes.ensureUnusedCapacity(m.gpa, 1);
        try (try m.at(parent)).children.append(m.gpa, nid);
        m.nodes.appendAssumeCapacity(.{ .nid = nid, .parent = parent, .start = start, .ids = own, .hex = idsHex(own), .gen = gen, .loc = loc });
        m.next_id += 1;
        return nid;
    }

    /// nid keeps [start, p); a new child takes [p, end) with its generation, location, extents and children.
    /// nid は [start, p) を残し、新しい子が [p, end) を世代・場所・extent・子とともに受け取る。
    pub fn split(m: *Meta, nid: i64, p: i64) Error!i64 {
        const gpa = m.gpa;
        try m.nodes.ensureUnusedCapacity(gpa, 1);
        try m.active.ensureUnusedCapacity(gpa, 1);
        const n0 = try m.atc(nid);
        if (!(n0.start < p and p < n0.end())) return error.BadSplit;
        const k: usize = @intCast(p - n0.start);
        const rid = m.next_id;
        const left_ids = try gpa.dupe(i64, n0.ids[0..k]);
        errdefer gpa.free(left_ids);
        const right_ids = try gpa.dupe(i64, n0.ids[k..]);
        errdefer gpa.free(right_ids);
        var left_ext: std.ArrayList(Extent) = .empty;
        errdefer left_ext.deinit(gpa);
        var right_ext: std.ArrayList(Extent) = .empty;
        errdefer right_ext.deinit(gpa);
        if (n0.loc == .S) {
            for (n0.ext.items) |e| {
                if (e.b() <= p) {
                    try left_ext.append(gpa, e);
                } else if (e.a >= p) {
                    try right_ext.append(gpa, e);
                } else {
                    const lr = try e.split(p);
                    try left_ext.append(gpa, lr[0]);
                    try right_ext.append(gpa, lr[1]);
                }
            }
        } else {
            left_ext = try n0.ext.clone(gpa);
        }
        var left_children: std.ArrayList(i64) = .empty;
        try left_children.append(gpa, rid);
        // no failure past here
        // ここから先は失敗しない
        const n = try m.at(nid);
        const right: Node = .{ .nid = rid, .parent = nid, .start = p, .ids = right_ids, .hex = idsHex(right_ids), .gen = n.gen, .loc = n.loc, .ext = right_ext, .children = n.children, .last = n.last, .pinned = n.pinned };
        gpa.free(n.ids);
        n.ids = left_ids;
        n.hex = idsHex(left_ids);
        n.ext.deinit(gpa);
        n.ext = left_ext;
        n.children = left_children;
        for (right.children.items) |ch| m.nodes.items[@intCast(ch)].?.parent = rid;
        m.nodes.appendAssumeCapacity(right);
        m.next_id += 1;
        if (std.mem.indexOfScalar(i64, m.active.items, nid)) |i| m.active.insertAssumeCapacity(i + 1, rid);
        return rid;
    }

    /// Make p a node boundary on the path (split the first node strictly containing it).
    /// パス上で p をノード境界にする (p を内側に含む最初のノードを分割する)。
    pub fn cut(m: *Meta, path_: []const i64, p: i64) Error!void {
        for (path_) |nid| {
            const n = try m.atc(nid);
            if (n.start < p and p < n.end()) {
                _ = try m.split(nid, p);
                return;
            }
        }
    }

    /// The tree's invariants and the store's partition; error.Check when one fails.
    /// 木の不変条件とストアの分割を確かめる。一つでも崩れていれば error.Check。
    pub fn check(m: *const Meta) Error!void {
        for (m.nodes.items, 0..) |nn, i| {
            const n = nn orelse continue;
            if (n.nid != @as(i64, @intCast(i))) return error.Check;
            if (n.nid == ROOT) {
                if (n.parent != null or n.loc != .R or n.ids.len != 0 or n.ext.items.len != 0) return error.Check;
            } else {
                const par = m.atc(n.parent orelse return error.Check) catch return error.Check;
                // listed once among its parent's children, starting at the parent's end, not empty, not R
                // 親の子リストにちょうど 1 回あり、親の末尾から始まり、空でなく、R でない
                if (std.mem.count(i64, par.children.items, &.{n.nid}) != 1) return error.Check;
                if (n.start != par.end() or n.loc == .R or n.ids.len == 0) return error.Check;
                if (!std.mem.eql(u8, &n.hex, &idsHex(n.ids))) return error.Check;
            }
            for (n.children.items) |ch| {
                const c = m.atc(ch) catch return error.Check;
                if (c.parent != n.nid) return error.Check;
            }
            switch (n.loc) {
                // stored: congruent extents cover [start, end) in order; its children are stored too
                // 保存済み: 合同な extent が [start, end) を順に覆う。子もすべて保存済み
                .S => {
                    if (n.ext.items.len == 0) return error.Check;
                    var a = n.start;
                    for (n.ext.items) |e| {
                        if (e.a != a or e.n <= 0 or @mod(e.s - e.a, m.store.dcp) != 0) return error.Check;
                        a = e.b();
                    }
                    if (a != n.end()) return error.Check;
                    for (n.children.items) |ch| if (m.nodes.items[@intCast(ch)].?.loc != .S) return error.Check;
                },
                .A => { // on the compute path, no extents / 計算パス上にあり、extent を持たない
                    if (n.ext.items.len != 0 or std.mem.indexOfScalar(i64, m.active.items, n.nid) == null) return error.Check;
                },
                .R => if (n.nid != ROOT) return error.Check,
            }
        }
        var prev = ROOT;
        // the compute path: A nodes chained from the root
        // 計算パス: root から続く A ノードの鎖
        for (m.active.items) |a| {
            const n = m.atc(a) catch return error.Check;
            if (n.loc != .A or n.parent != prev) return error.Check;
            prev = a;
        }
        if (m.next_id != @as(i64, @intCast(m.nodes.items.len))) return error.Check;
        const used = try m.storedExtents(m.gpa);
        defer m.gpa.free(used);
        // the free intervals and the stored extents partition the store
        // 空き区間と保存済みの extent がストアをちょうど分け合う
        try m.store.check(used);
    }
};

// ------------------------------------------------------------------------------------------------- identities ---

pub const Ident = struct { pid: []const u64, gen: []const i64 };

/// pid and generation of every position of the conversation ending at nid (caller frees both).
/// nid で終わる会話の全位置の pid と世代 (どちらも呼び出し側が解放)。
pub fn pathIdent(gpa: Allocator, m: *const Meta, nid: i64) Error!struct { pid: []u64, gen: []i64 } {
    const p = try m.path(gpa, nid);
    defer gpa.free(p);
    const toks = try m.tokens(gpa, nid);
    defer gpa.free(toks);
    const pid = try gpa.alloc(u64, toks.len);
    errdefer gpa.free(pid);
    hashing.pidsInto(pid, toks);
    const gen = try gpa.alloc(i64, toks.len);
    var o: usize = 0;
    for (p) |x| {
        const n = try m.atc(x);
        @memset(gen[o .. o + n.ids.len], n.gen);
        o += n.ids.len;
    }
    return .{ .pid = pid, .gen = gen };
}

/// Identity arrays, one pair a path rather than a node (deepest nids first, each covering its ancestors).
/// identity の配列。ノードごとでなくパスごとに 1 組 (深い nid から。各組はその祖先も覆う)。
pub const PathIdents = struct {
    gpa: Allocator,
    paths: std.ArrayList(Path) = .empty,

    const Path = struct { on: []bool, pid: []u64, gen: []i64 };

    pub fn init(gpa: Allocator, m: *const Meta, nids: []const ?i64) Error!PathIdents {
        var pi: PathIdents = .{ .gpa = gpa };
        errdefer pi.deinit();
        var list: std.ArrayList(i64) = .empty;
        defer list.deinit(gpa);
        for (nids) |x| if (x) |v| if (std.mem.indexOfScalar(i64, list.items, v) == null) try list.append(gpa, v);
        for (list.items) |v| _ = try m.atc(v);
        std.mem.sort(i64, list.items, m, struct {
            fn lt(mm: *const Meta, x: i64, y: i64) bool {
                const ex = mm.nodes.items[@intCast(x)].?.end();
                const ey = mm.nodes.items[@intCast(y)].?.end();
                return ex > ey or (ex == ey and x < y);
            }
        }.lt);
        for (list.items) |v| {
            if (pi.find(v) != null) continue;
            const on = try gpa.alloc(bool, m.nodes.items.len);
            @memset(on, false);
            var ok = false;
            defer if (!ok) gpa.free(on);
            const p = try m.path(gpa, v);
            defer gpa.free(p);
            for (p) |x| on[@intCast(x)] = true;
            const id = try pathIdent(gpa, m, v);
            pi.paths.append(gpa, .{ .on = on, .pid = id.pid, .gen = id.gen }) catch |e| {
                gpa.free(id.pid);
                gpa.free(id.gen);
                return e;
            };
            ok = true;
        }
        return pi;
    }

    /// For a plan: every op's nid_in and nid_out, on the plan's post tree.
    /// プラン用: 全 op の nid_in と nid_out を、プランの post の木の上で求める。
    pub fn initPlan(gpa: Allocator, plan: *const Plan) Error!PathIdents {
        var nids: std.ArrayList(?i64) = .empty;
        defer nids.deinit(gpa);
        for (plan.ops) |o| try nids.appendSlice(gpa, &.{ o.nid_in, o.nid_out });
        return init(gpa, &plan.post, nids.items);
    }

    fn find(pi: *const PathIdents, nid: i64) ?*const Path {
        for (pi.paths.items) |*p| if (nid >= 0 and nid < p.on.len and p.on[@intCast(nid)]) return p;
        return null;
    }

    /// The arrays of the first path holding nid (error.NoNode: not on the plan's paths).
    /// nid を含む最初のパスの配列 (プランのパス上になければ error.NoNode)。
    pub fn get(pi: *const PathIdents, nid: i64) Error!Ident {
        const p = pi.find(nid) orelse return error.NoNode;
        return .{ .pid = p.pid, .gen = p.gen };
    }

    pub fn deinit(pi: *PathIdents) void {
        for (pi.paths.items) |p| {
            pi.gpa.free(p.on);
            pi.gpa.free(p.pid);
            pi.gpa.free(p.gen);
        }
        pi.paths.deinit(pi.gpa);
    }
};

// ------------------------------------------------------------------------------------------------------ plans ---

pub const Kind = enum {
    save,
    restore,
    swap,
    fn name(k: Kind) []const u8 {
        return @tagName(k);
    }
};

/// One step of a plan on positions [a, b); src / dst / exp are logical store starts. The rows' identities come
/// from PathIdents.
/// 位置 [a, b) に対するプランの 1 手順。src / dst / exp は論理ストアの開始位置。行の identity は PathIdents から取る。
pub const Op = struct {
    kind: Kind,
    a: i64,
    b: i64,
    src: ?i64,
    dst: ?i64,
    exp: ?i64,
    nid_in: ?i64 = null,
    nid_out: ?i64 = null,
};

pub const Mode = enum { none, plain, swap, drop };

/// One request's plan. `post` is the metadata after the plan (owned); deinit frees everything.
/// 1 要求のプラン。`post` はプラン後のメタデータ (所有する)。deinit ですべて解放する。
pub const Plan = struct {
    gpa: Allocator,
    d0: [16]u8,
    c: i64,
    h: i64,
    t_leaf: i64,
    mode: Mode,
    ops: []Op,
    free_after: []Extent,
    post: Meta,
    evicted: i64 = 0,

    pub fn deinit(p: *Plan) void {
        p.gpa.free(p.ops);
        p.gpa.free(p.free_after);
        p.post.deinit();
        p.* = undefined;
    }

    fn writeRepr(p: *const Plan, sink: anytype) !void {
        try sink.put("(");
        for ([_]i64{ p.c, p.h, p.t_leaf }) |x| {
            try putInt(sink, x);
            try sink.put(", ");
        }
        try sink.put("'");
        try sink.put(@tagName(p.mode));
        try sink.put("', ");
        try putInt(sink, p.evicted);
        try sink.put(", [");
        for (p.ops, 0..) |o, i| {
            if (i > 0) try sink.put(", ");
            try sink.put("('");
            try sink.put(o.kind.name());
            try sink.put("', ");
            try putInt(sink, o.a);
            try sink.put(", ");
            try putInt(sink, o.b);
            for ([_]?i64{ o.src, o.dst, o.exp, o.nid_in, o.nid_out }) |x| {
                try sink.put(", ");
                try putOpt(sink, x);
            }
            try sink.put(")");
        }
        try sink.put("], ");
        try putExtents(sink, p.free_after);
        try sink.put(")");
    }

    /// The plan tuple `planDigest` hashes (tests; caller frees).
    /// `planDigest` がハッシュするプランの組 (テスト用。呼び出し側が解放)。
    pub fn reprText(p: *const Plan, gpa: Allocator) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try p.writeRepr(TextSink{ .gpa = gpa, .out = &out });
        return out.toOwnedSlice(gpa);
    }
};

/// SHA-256(d0 ++ repr(numbers, ops, free_after) ++ post.digest()), the first 16 B.
/// SHA-256(d0 ++ repr(numbers, ops, free_after) ++ post.digest()) の先頭 16 B。
pub fn planDigest(p: *const Plan) [16]u8 {
    var h = Sha256.init(.{});
    h.update(&p.d0);
    p.writeRepr(HashSink{ .h = &h }) catch unreachable;
    h.update(&p.post.digest());
    return h.finalResult()[0..16].*;
}

/// A 16 B digest as 8 int32 of 16 bits each (little-endian pairs).
/// 16 B の digest を 16 bit ずつの int32 8 個にする (リトルエンディアンの 2 バイト組)。
pub fn words(d: [16]u8) [8]i32 {
    var w: [8]i32 = undefined;
    for (&w, 0..) |*x, i| x.* = @as(i32, d[2 * i]) | (@as(i32, d[2 * i + 1]) << 8);
    return w;
}

/// e's pieces outside [lo, hi).
/// e のうち [lo, hi) の外側の片。
fn outsideOverlap(e: Extent, lo: i64, hi: i64) struct { buf: [2]Extent, n: usize } {
    var r: @TypeOf(outsideOverlap(e, lo, hi)) = .{ .buf = undefined, .n = 0 };
    if (e.a < lo) {
        const x: Extent = .{ .s = e.s, .a = e.a, .n = @min(e.b(), lo) - e.a };
        if (x.n > 0) {
            r.buf[r.n] = x;
            r.n += 1;
        }
    }
    if (e.b() > hi) {
        const a = @max(e.a, hi);
        const x: Extent = .{ .s = e.s + (a - e.a), .a = a, .n = e.b() - a };
        if (x.n > 0) {
            r.buf[r.n] = x;
            r.n += 1;
        }
    }
    return r;
}

/// The subtree below nid, last child first (the drop's free_after order). Caller frees.
/// nid より下の部分木。最後の子から並ぶ (drop の free_after の順)。呼び出し側が解放。
fn descendants(gpa: Allocator, m: *const Meta, nid: i64) Error![]i64 {
    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(gpa);
    var stack: std.ArrayList(i64) = .empty;
    defer stack.deinit(gpa);
    if (m.has(nid)) try stack.appendSlice(gpa, (try m.atc(nid)).children.items);
    while (stack.pop()) |x| {
        try out.append(gpa, x);
        try stack.appendSlice(gpa, (try m.atc(x)).children.items);
    }
    return out.toOwnedSlice(gpa);
}

const Pair = struct { nid: i64, e: Extent };

/// The planner's working state.
/// プランナの作業状態。
const Planner = struct {
    gpa: Allocator,
    post: *Meta,
    protected: []bool,
    ops: std.ArrayList(Op) = .empty,
    free_after: std.ArrayList(Extent) = .empty,

    /// A restore op for every extent of the nodes; each extent is freed after the plan.
    /// ノードの extent ごとに restore op を作る。各 extent はプランの後に解放する。
    fn restoreOps(pl: *Planner, nodes: []const i64) Error!void {
        for (nodes) |n| for ((try pl.post.atc(n)).ext.items) |e| {
            try pl.ops.append(pl.gpa, .{ .kind = .restore, .a = e.a, .b = e.b(), .src = e.s, .dst = null, .exp = e.s, .nid_in = n });
            try pl.free_after.append(pl.gpa, e);
        };
    }

    /// Store pieces for the positions [lo, hi) of the A nodes (null: no room, the store unchanged).
    /// A ノードの位置 [lo, hi) を置くストアの片 (null なら空きがなく、ストアは変わらない)。
    fn savePieces(pl: *Planner, nodes: []const i64, lo: i64, hi: i64) Error!?[]Pair {
        var out: std.ArrayList(Pair) = .empty;
        errdefer out.deinit(pl.gpa);
        var trial = try pl.post.store.clone();
        defer trial.deinit();
        for (nodes) |n| {
            const node = try pl.post.atc(n);
            const a = @max(node.start, lo);
            const b = @min(node.end(), hi);
            if (a >= b) continue;
            const pieces = (try trial.allocPieces(a, b - a)) orelse {
                out.deinit(pl.gpa);
                return null;
            };
            defer trial.gpa.free(pieces);
            for (pieces) |e| try out.append(pl.gpa, .{ .nid = n, .e = e });
        }
        std.mem.swap(std.ArrayList(Interval), &pl.post.store.free, &trial.free);
        return try out.toOwnedSlice(pl.gpa);
    }

    /// Drop the least recently used stored leaf branch outside the protected paths.
    /// 保護したパスの外で、最も長く使われていない保存済みの葉の枝を捨てる。
    fn evictOne(pl: *Planner) Error!bool {
        const post = pl.post;
        var victim: ?*const Node = null;
        for (post.nodes.items) |*nn| if (nn.*) |*n| {
            if (n.nid == ROOT or n.children.items.len != 0 or n.loc != .S or pl.protected[@intCast(n.nid)] or n.pinned) continue;
            if (victim == null or n.last < victim.?.last or (n.last == victim.?.last and n.nid < victim.?.nid)) victim = n;
        };
        const v = victim orelse return false;
        var top = v.nid;
        // climb while the parent is stored, unprotected, unpinned and has no other child
        // 親が保存済み・保護なし・pin なしで、ほかに子がない間は上へたどる
        while (true) {
            const par = (try post.atc(top)).parent orelse return error.PlanAssert;
            const pn = try post.atc(par);
            if (pn.nid == ROOT or pn.loc != .S or pl.protected[@intCast(par)] or pn.pinned or pn.children.items.len != 1) break;
            top = par;
        }
        const par = (try post.atc(top)).parent orelse return error.PlanAssert;
        if (!removeFirst(&(try post.at(par)).children, top)) return error.PlanAssert;
        var stack: std.ArrayList(i64) = .empty;
        defer stack.deinit(pl.gpa);
        try stack.append(pl.gpa, top);
        while (stack.pop()) |x| {
            var nx = (try post.at(x)).*;
            post.nodes.items[@intCast(x)] = null;
            defer nx.deinit(post.gpa);
            try stack.appendSlice(pl.gpa, nx.children.items);
            if (nx.loc == .S) for (nx.ext.items) |e| {
                try post.store.release(e);
                try pl.free_after.append(pl.gpa, e);
            };
        }
        return true;
    }
};

/// path(nid) in place of old (freed once the new one exists).
/// old の代わりに path(nid) を返す (新しいものができてから old を解放する)。
fn repath(gpa: Allocator, m: *const Meta, old: []i64, nid: i64) Error![]i64 {
    const p = try m.path(gpa, nid);
    gpa.free(old);
    return p;
}

/// One request's plan on a copy of m (m never changes, on error too). T is the conversation matching the prompt X
/// longest, c the resume point, h = min(c, d) with d where the paths of the compute conversation A and of T part:
/// A's rows from h on are saved (or swapped into T's slots, or dropped) and T's rows [h, c) restored.
/// max_out: rows written past X; cap: the compute cache's positions; tail: prompt rows always recomputed;
/// min_resume: shorter resumes are 0; resume false: c = 0 (A is still copied out when there is room).
/// m のコピー上で 1 要求のプランを作る (m はエラー時も含め変わらない)。T は prompt X と最も長く一致する会話、c は
/// 再開点、h = min(c, d) で、d は計算中の会話 A と T のパスが分かれる位置: A の h 以降の行は保存し (または T の
/// スロットと入れ替え、または捨て)、T の行 [h, c) を復元する。
/// max_out: X の後に書く行数。cap: 計算キャッシュの位置数。tail: 常に再計算する prompt の行数。
/// min_resume: これより短い再開は 0 にする。resume が false なら c = 0 (空きがあれば A はそれでも書き出す)。
pub fn makePlan(gpa: Allocator, m: *const Meta, X: []const i64, max_out: i64, cap: i64, tail: i64, min_resume: i64, resume_: bool) Error!Plan {
    var post = try m.clone(gpa);
    errdefer post.deinit();
    post.tick += 1;
    const L: i64 = @intCast(X.len);
    if (L + max_out > cap) return error.RequestTooLong;
    const A_leaf = post.actLeaf();
    const n_A = post.nAct();
    const A_path0 = if (post.active.items.len > 0) try post.path(gpa, A_leaf) else try gpa.alloc(i64, 0);
    defer gpa.free(A_path0);

    // T: the longest match; ties: the compute path (no copy), then recency, then the smaller nid
    // T: 最長一致。同長なら計算パス (コピー不要)、次に新しさ、次に小さい nid
    var best: ?i64 = null;
    var bk: [4]i64 = undefined; // (lcp, is A, last, -nid)
    if (resume_) {
        const cands = try post.candidates(gpa);
        defer gpa.free(cands);
        for (cands) |nid| {
            const p = try post.path(gpa, nid);
            defer gpa.free(p);
            const key = [4]i64{ try lcpPath(&post, p, X), @intFromBool(nid == A_leaf), (try post.atc(nid)).last, -nid };
            if (best == null or std.mem.order(i64, &key, &bk) == .gt) {
                best = nid;
                bk = key;
            }
        }
    }
    var c: i64 = 0;
    if (best != null) {
        c = @min(bk[0], L - tail);
        if (c < min_resume) c = 0;
    }
    const T_leaf = if (c > 0) best.? else ROOT;

    // d: where the tree paths of A and T part
    // d: A と T の木のパスが分かれる位置
    var common: i64 = 0;
    {
        const T_path0 = if (T_leaf != ROOT) try post.path(gpa, T_leaf) else try gpa.alloc(i64, 0);
        defer gpa.free(T_path0);
        for (A_path0[0..@min(A_path0.len, T_path0.len)], T_path0[0..@min(A_path0.len, T_path0.len)]) |x, y| {
            if (x != y) break;
            common = (try post.atc(x)).end();
        }
    }
    const d = if (T_leaf == A_leaf) n_A else common;
    const h = @min(c, d);
    // A's path matches the prompt up to h
    // A のパスは h まで prompt と一致する
    if (@min(try lcpPath(&post, A_path0, X), h) != h) return error.PlanAssert;

    // node boundaries at h (A's path) and at h and c (T's path)
    // ノード境界を h (A のパス) と、h と c (T のパス) に作る
    try post.cut(A_path0, h);
    var T_path: []i64 = try gpa.alloc(i64, 0);
    defer gpa.free(T_path);
    if (T_leaf != ROOT) { // each cut may split: the path again after it / cut は分割しうるので、毎回パスを取り直す
        T_path = try repath(gpa, &post, T_path, T_leaf);
        try post.cut(T_path, h);
        T_path = try repath(gpa, &post, T_path, T_leaf);
        try post.cut(T_path, c);
        T_path = try repath(gpa, &post, T_path, T_leaf);
    }
    var A_out: std.ArrayList(i64) = .empty;
    defer A_out.deinit(gpa);
    for (post.active.items) |n| if ((try post.atc(n)).start >= h) try A_out.append(gpa, n);
    var T_in: std.ArrayList(i64) = .empty;
    defer T_in.deinit(gpa);
    for (T_path) |n| {
        const x = try post.atc(n);
        if (x.start >= h and x.end() <= c) try T_in.append(gpa, n);
    }
    for (T_in.items) |n| if ((try post.atc(n)).loc != .S) return error.PlanAssert;
    var n_save: i64 = 0;
    for (A_out.items) |n| n_save += @intCast((try post.atc(n)).ids.len);
    const protected = try gpa.alloc(bool, post.nodes.items.len);
    defer gpa.free(protected);
    @memset(protected, false);
    for (T_path) |n| protected[@intCast(n)] = true;
    for (post.active.items) |n| protected[@intCast(n)] = true;

    var pl: Planner = .{ .gpa = gpa, .post = &post, .protected = protected };
    errdefer pl.ops.deinit(gpa);
    errdefer pl.free_after.deinit(gpa);
    var mode: Mode = .none;
    var evicted: i64 = 0;
    var placed: ?[]Pair = null;
    defer if (placed) |x| gpa.free(x);

    if (n_save == 0) {
        mode = if (T_in.items.len == 0) .none else .plain;
        try pl.restoreOps(T_in.items);
        placed = try gpa.alloc(Pair, 0);
    } else {
        placed = try pl.savePieces(A_out.items, h, n_A);
        while (placed == null and try pl.evictOne()) {
            evicted += 1;
            placed = try pl.savePieces(A_out.items, h, n_A);
        }
        if (placed) |pp| {
            mode = .plain;
            for (pp) |x| try pl.ops.append(gpa, .{ .kind = .save, .a = x.e.a, .b = x.e.b(), .src = null, .dst = x.e.s, .exp = null, .nid_in = x.nid, .nid_out = x.nid });
            try pl.restoreOps(T_in.items);
        } else {
            // swap: A's rows at positions T restores go to T's own slots (congruent: same position, same slot)
            // swap: T が復元する位置にある A の行は T 自身のスロットへ行く (合同なので同じ位置は同じスロット)
            const over_hi = @min(n_A, c);
            const rem: ?[]Pair = if (n_A > c) try pl.savePieces(A_out.items, @max(h, c), n_A) else try gpa.alloc(Pair, 0);
            if (rem) |rm| {
                defer gpa.free(rm);
                mode = .swap;
                var swapped: std.ArrayList(Pair) = .empty;
                errdefer swapped.deinit(gpa);
                for (T_in.items) |n| {
                    const node = try post.atc(n);
                    for (node.ext.items) |e| {
                        const piece = e.clip(h, over_hi) orelse continue;
                        // a piece may straddle A node boundaries: one op per A node
                        // 片は A ノードの境界をまたぎうる: A ノードごとに 1 op
                        var p = piece.a;
                        while (p < piece.b()) {
                            var an: ?i64 = null;
                            for (A_out.items) |x| {
                                const xn = try post.atc(x);
                                if (xn.start <= p and p < xn.end()) {
                                    an = x;
                                    break;
                                }
                            }
                            const a_n = an orelse return error.PlanAssert;
                            const q = @min(piece.b(), (try post.atc(a_n)).end());
                            const sub = piece.clip(p, q).?;
                            try pl.ops.append(gpa, .{ .kind = .swap, .a = p, .b = q, .src = sub.s, .dst = sub.s, .exp = sub.s, .nid_in = n, .nid_out = a_n });
                            try swapped.append(gpa, .{ .nid = a_n, .e = sub });
                            p = q;
                        }
                    }
                    // T pieces beyond A's end: restore only
                    // A の末尾より先にある T の片: 復元だけ
                    for (node.ext.items) |e2| {
                        if (e2.clip(@max(h, n_A), c)) |piece2| {
                            try pl.ops.append(gpa, .{ .kind = .restore, .a = piece2.a, .b = piece2.b(), .src = piece2.s, .dst = null, .exp = piece2.s, .nid_in = n });
                            try pl.free_after.append(gpa, piece2);
                        }
                    }
                }
                for (rm) |x| try pl.ops.append(gpa, .{ .kind = .save, .a = x.e.a, .b = x.e.b(), .src = null, .dst = x.e.s, .exp = null, .nid_in = x.nid, .nid_out = x.nid });
                try swapped.appendSlice(gpa, rm);
                placed = try swapped.toOwnedSlice(gpa);
            } else {
                mode = .drop;
                try pl.restoreOps(T_in.items);
            }
        }
    }

    // --- the post state (in swap mode T's slots inside the overlap now hold A's rows: they stay allocated)
    // --- プラン後の状態 (swap モードでは重なり内の T のスロットが A の行を持つので、確保したままにする)
    var new_active: std.ArrayList(i64) = .empty;
    defer new_active.deinit(gpa);
    for (post.active.items) |n| if ((try post.atc(n)).start < h) try new_active.append(gpa, n);
    if (placed == null and n_save != 0) {
        for (A_out.items) |n| { // dropped: A's branch beyond h goes away / drop: h より先の A の枝を消す
            // already gone with an earlier A node's subtree
            // 先の A ノードの部分木と一緒にすでに消えている
            if (!post.has(n)) continue;
            if (post.nodes.items[@intCast(n)].?.parent) |par| {
                if (post.has(par)) _ = removeFirst(&(try post.at(par)).children, n);
            }
            const desc = try descendants(gpa, &post, n);
            defer gpa.free(desc);
            var i: usize = 0;
            while (i <= desc.len) : (i += 1) {
                const x = if (i == 0) n else desc[i - 1];
                if (!post.has(x)) continue;
                var nx = post.nodes.items[@intCast(x)].?;
                post.nodes.items[@intCast(x)] = null;
                defer nx.deinit(post.gpa);
                if (nx.loc == .S) for (nx.ext.items) |e| {
                    try post.store.release(e);
                    try pl.free_after.append(gpa, e);
                };
            }
        }
    } else {
        const pp = placed.?;
        for (A_out.items) |n| {
            const node = try post.at(n);
            node.loc = .S;
            node.ext.clearRetainingCapacity();
            for (pp) |x| if (x.nid == n) try node.ext.append(post.gpa, x.e);
            std.mem.sort(Extent, node.ext.items, {}, struct {
                fn lt(_: void, x: Extent, y: Extent) bool {
                    return x.a < y.a;
                }
            }.lt);
        }
    }
    // T's nodes [h, c) join the compute path; their extents (outside swapped slots) are released
    // T のノード [h, c) は計算パスに入る。その extent (入れ替えたスロットの外) は解放する
    for (T_in.items) |n| {
        const node = try post.at(n);
        for (node.ext.items) |e| {
            if (mode != .swap) {
                try post.store.release(e);
            } else {
                const o = outsideOverlap(e, h, @min(n_A, c));
                for (o.buf[0..o.n]) |piece| try post.store.release(piece);
            }
        }
        node.loc = .A;
        node.ext.clearRetainingCapacity();
    }
    post.active.clearRetainingCapacity();
    try post.active.appendSlice(post.gpa, new_active.items);
    try post.active.appendSlice(post.gpa, T_in.items);
    for (post.active.items) |n| (try post.at(n)).last = post.tick;
    for (A_out.items) |n| if (post.has(n)) {
        (try post.at(n)).last = post.tick;
    };
    post.epoch += 1;
    const ops = try pl.ops.toOwnedSlice(gpa);
    errdefer gpa.free(ops);
    const fa = try pl.free_after.toOwnedSlice(gpa);
    return .{ .gpa = gpa, .d0 = m.digest(), .c = c, .h = h, .t_leaf = T_leaf, .mode = mode, .ops = ops, .free_after = fa, .post = post, .evicted = evicted };
}

// ----------------------------------------------------------------------------------------------------- commit ---

/// Stored leaf siblings of nid whose whole run repeats the start of nid's ids go away.
/// Returns the released extents (gpa's; their expected values must be cleared).
/// nid の兄弟のうち、保存済みの葉で、その全体が nid の ids の先頭と同じものを消す。
/// 解放した extent を返す (gpa のもの。その期待値は消す必要がある)。
pub fn dropDuplicates(m: *Meta, gpa: Allocator, parent: i64, nid: i64) Error![]Extent {
    var released: std.ArrayList(Extent) = .empty;
    errdefer released.deinit(gpa);
    const snap = try gpa.dupe(i64, (try m.atc(parent)).children.items);
    defer gpa.free(snap);
    for (snap) |sib| {
        if (sib == nid) continue;
        const new_ids = (try m.atc(nid)).ids;
        const s = try m.atc(sib);
        if (s.loc == .S and s.children.items.len == 0 and s.ids.len <= new_ids.len and
            std.mem.eql(i64, s.ids, new_ids[0..s.ids.len]) and !s.pinned)
        {
            _ = removeFirst(&(try m.at(parent)).children, sib);
            var sx = s.*;
            m.nodes.items[@intCast(sib)] = null;
            defer sx.deinit(m.gpa);
            for (sx.ext.items) |e| {
                try m.store.release(e);
                try released.append(gpa, e);
            }
        }
    }
    return released.toOwnedSlice(gpa);
}

/// After a request wrote positions [c, c + len(seg)) (generation gen) the segment joins the compute path as one
/// node; tick advances even for an empty segment. Returns the extents of the stored duplicates dropped (gpa's;
/// caller frees).
/// 要求が位置 [c, c + len(seg)) を書いた (世代 gen) 後、その区間を 1 ノードとして計算パスに加える。
/// 区間が空でも tick は進む。捨てた保存済みの重複の extent を返す (gpa のもの。呼び出し側が解放)。
pub fn commitSegment(m: *Meta, gpa: Allocator, c: i64, seg: []const i64, gen: i64) Error![]Extent {
    m.tick += 1;
    if (seg.len == 0) return gpa.alloc(Extent, 0);
    const parent = m.actLeaf();
    if (m.nAct() != c) return error.PlanAssert;
    try m.active.ensureUnusedCapacity(m.gpa, 1);
    const nid = try m.newNode(parent, seg, gen, .A);
    m.active.appendAssumeCapacity(nid);
    (try m.at(nid)).last = m.tick;
    return dropDuplicates(m, gpa, parent, nid);
}
