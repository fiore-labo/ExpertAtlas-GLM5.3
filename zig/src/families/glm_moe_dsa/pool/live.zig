// Fiore-lab: the KV pool's conversations and blocks for the control panel (GET /v1/pool), not in TensorFold 1.0.2.
//! Rank 0's snapshot of the pool's host tables (plan.Meta), taken on the engine thread where the tables change: a
//! request's plan applied (the conversation in use, its whole span taken), its commit, a common cold. Nothing per token
//! and no device read; GET /v1/pool copies the snapshot under the mutex and writes the JSON outside it.
//! 制御パネル用の KV pool の会話とブロック (GET /v1/pool)。rank 0 のホストの表 (plan.Meta) の写しで、表が変わる所で
//! エンジンのスレッドが取る: 要求のプランを適用したとき (使用中の会話。要求の範囲全体を確保済み)、確定したとき、
//! 共通の cold。トークンごとの処理もデバイスの読み出しもない。GET /v1/pool は mutex の下で写しをコピーし、JSON は
//! その外で書く。
const std = @import("std");
const plan = @import("plan.zig");

/// owner's values besides a conversation's number / owner の会話の番号以外の値
pub const free: u16 = 0;
pub const shared: u16 = 0xFFFF;

/// A conversation's number in /v1/pool: its slot id + 1, wrapped below 0xFFFF so owner carries it in 16 bits (0 and
/// 0xFFFF are free and shared). 64 slots at most: two kept at once never get the same number.
/// /v1/pool での会話の番号: スロットの id + 1。owner が 16 bit で持てるよう 0xFFFF 未満で一巡する (0 と 0xFFFF は
/// 空きと共有)。スロットは最大 64 なので、同時にある 2 つが同じ番号になることはない。
pub fn number(id: u64) u16 {
    return @intCast(id % 0xFFFE + 1);
}

pub const Conv = struct {
    id: u64,
    positions: u64, // valid_end
    blocks: u64,
    shared_blocks: u64,
    used_ms: u64, // when it was last used (ms since start) / 最後に使った時刻 (起動からの ms)
    active: bool,
    t0: usize = 0, // its table in Snap.tables (active only) / Snap.tables の中のその表 (使用中のみ)
    tn: usize = 0,
};

const Snap = struct {
    owner: []u16 = &.{},
    convs: []Conv = &.{},
    tables: []u32 = &.{},
    at_ms: u64 = 0,

    fn deinit(s: *Snap, gpa: std.mem.Allocator) void {
        gpa.free(s.owner);
        gpa.free(s.convs);
        gpa.free(s.tables);
        s.* = .{};
    }
};

// a slot's tick and when it moved (engine thread only) / スロットの tick と、それが動いた時刻 (エンジンのスレッドだけ)
const Seen = struct { id: u64, tick: u64, ms: u64 };

pub const PoolLive = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    t0: std.Io.Timestamp,
    cur: Snap = .{},
    snaps: u64 = 0,
    seen: []Seen = &.{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io) !*PoolLive {
        const l = try gpa.create(PoolLive);
        l.* = .{ .gpa = gpa, .io = io, .t0 = std.Io.Clock.awake.now(io) };
        return l;
    }

    pub fn deinit(l: *PoolLive) void {
        l.cur.deinit(l.gpa);
        l.gpa.free(l.seen);
        l.gpa.destroy(l);
    }

    fn nowMs(l: *const PoolLive) u64 {
        const ms = l.t0.durationTo(std.Io.Clock.awake.now(l.io)).toMilliseconds();
        return if (ms < 0) 0 else @intCast(ms);
    }

    /// The tables of `m` now; `active`: the conversations requests run on (slot ids, the one /health's progress tells
    /// first). A snapshot that cannot be made (out of memory) leaves the last one.
    /// `m` の今の表。`active`: 要求が動いている会話 (スロットの id。/health の progress が示すものを先頭に)。写しを
    /// 作れなければ (メモリ不足) 前の写しのまま。
    pub fn snap(l: *PoolLive, m: *const plan.Meta, active: []const u64) void {
        l.snapAt(m, active, l.nowMs()) catch {};
    }

    fn snapAt(l: *PoolLive, m: *const plan.Meta, active: []const u64, now: u64) !void {
        const gpa = l.gpa;
        var s: Snap = .{ .at_ms = now };
        errdefer s.deinit(gpa);
        s.owner = try gpa.alloc(u16, m.ref.len);
        @memset(s.owner, free);
        s.convs = try gpa.alloc(Conv, m.slots.items.len);
        const seen = try gpa.alloc(Seen, m.slots.items.len);
        errdefer gpa.free(seen);
        var nt: usize = 0;
        for (m.slots.items) |x| if (std.mem.indexOfScalar(u64, active, x.id) != null) {
            nt += x.blocks.items.len;
        };
        s.tables = try gpa.alloc(u32, nt);
        nt = 0;
        for (m.slots.items, s.convs, seen) |x, *c, *sn| {
            const on = std.mem.indexOfScalar(u64, active, x.id) != null;
            // the last use: now while a request runs on it, else when its tick last moved (its commit)
            // 最後に使った時刻: 要求が動いている間は今、ほかは tick が最後に動いたとき (確定)
            var used = now;
            if (!on) for (l.seen) |o| if (o.id == x.id and o.tick == x.tick) {
                used = o.ms;
            };
            sn.* = .{ .id = x.id, .tick = x.tick, .ms = used };
            var sh: u64 = 0;
            for (x.blocks.items) |b| {
                if (b >= s.owner.len) continue; // Z / Z ブロック
                const twice = m.ref[b] >= 2;
                sh += @intFromBool(twice);
                s.owner[b] = if (twice) shared else number(x.id);
            }
            c.* = .{ .id = x.id, .positions = x.validEnd(), .blocks = x.blocks.items.len, .shared_blocks = sh, .used_ms = used, .active = on };
            if (on) {
                c.t0 = nt;
                c.tn = x.blocks.items.len;
                @memcpy(s.tables[nt..][0..c.tn], x.blocks.items);
                nt += c.tn;
            }
        }
        // the conversations in use first (in `active`'s order), then the most recently used: the last is the next
        // trimmed (plan.zig's lessOld, the other way round)
        // 使用中の会話を先頭に (`active` の順)、続いて最近使った順: 最後が次に削られる (plan.zig の lessOld の逆)
        const Order = struct {
            m: *const plan.Meta,
            active: []const u64,
            fn tick(o: @This(), id: u64) u64 {
                return o.m.slots.items[o.m.find(id).?].tick;
            }
            fn before(o: @This(), a: Conv, b: Conv) bool {
                const ia = std.mem.indexOfScalar(u64, o.active, a.id) orelse o.active.len;
                const ib = std.mem.indexOfScalar(u64, o.active, b.id) orelse o.active.len;
                if (ia != ib) return ia < ib;
                const ta = o.tick(a.id);
                const tb = o.tick(b.id);
                return ta > tb or (ta == tb and a.id > b.id);
            }
        };
        std.mem.sort(Conv, s.convs, Order{ .m = m, .active = active }, Order.before);
        {
            l.mutex.lockUncancelable(l.io);
            defer l.mutex.unlock(l.io);
            std.mem.swap(Snap, &l.cur, &s);
            l.snaps += 1;
        }
        s.deinit(gpa);
        gpa.free(l.seen);
        l.seen = seen;
    }

    /// GET /v1/pool's body.
    /// GET /v1/pool の本文。
    pub fn json(l: *PoolLive, a: std.mem.Allocator) ![]u8 {
        return l.jsonAt(a, l.nowMs());
    }

    fn jsonAt(l: *PoolLive, a: std.mem.Allocator, now: u64) ![]u8 {
        var s: Snap = .{};
        var snaps: u64 = 0;
        {
            l.mutex.lockUncancelable(l.io);
            defer l.mutex.unlock(l.io);
            s.owner = try a.dupe(u16, l.cur.owner);
            s.convs = try a.dupe(Conv, l.cur.convs);
            s.tables = try a.dupe(u32, l.cur.tables);
            s.at_ms = l.cur.at_ms;
            snaps = l.snaps;
        }
        var out: std.ArrayList(u8) = .empty;
        try out.print(a, "{{\"age_ms\":{d},\"block_positions\":{d},\"blocks\":{d},\"snapshots\":{d},\"convs\":[", .{ now -| s.at_ms, plan.block, s.owner.len, snaps });
        for (s.convs, 0..) |c, i| {
            try out.print(a, "{s}{{\"id\":{d},\"positions\":{d},\"blocks\":{d},\"shared_blocks\":{d},\"idle_ms\":{d},\"active\":{s}", .{
                if (i == 0) "" else ",", number(c.id), c.positions, c.blocks, c.shared_blocks, if (c.active) 0 else now -| c.used_ms, if (c.active) "true" else "false",
            });
            if (c.active) {
                try out.appendSlice(a, ",\"table\":\"");
                try base64(a, &out, u32, s.tables[c.t0..][0..c.tn]);
                try out.append(a, '"');
            }
            try out.append(a, '}');
        }
        try out.appendSlice(a, "],\"owner\":\"");
        try base64(a, &out, u16, s.owner);
        try out.appendSlice(a, "\"}");
        return out.toOwnedSlice(a);
    }
};

/// `v` as little-endian words in base64, onto `out`.
/// `v` をリトルエンディアンの語にして base64 で `out` へ。
fn base64(a: std.mem.Allocator, out: *std.ArrayList(u8), comptime T: type, v: []const T) !void {
    const raw = try a.alloc(u8, v.len * @sizeOf(T));
    defer a.free(raw);
    for (v, 0..) |x, i| std.mem.writeInt(T, raw[i * @sizeOf(T) ..][0..@sizeOf(T)], x, .little);
    const enc = std.base64.standard.Encoder;
    const dst = try out.addManyAsSlice(a, enc.calcSize(raw.len));
    _ = enc.encode(dst, raw);
}

// ------------------------------------------------------------------------------------------------------ tests ---

const t = std.testing;

fn decode(a: std.mem.Allocator, comptime T: type, s: []const u8) ![]T {
    const dec = std.base64.standard.Decoder;
    const raw = try a.alloc(u8, try dec.calcSizeForSlice(s));
    defer a.free(raw);
    try dec.decode(raw, s);
    const v = try a.alloc(T, raw.len / @sizeOf(T));
    for (v, 0..) |*x, i| x.* = std.mem.readInt(T, raw[i * @sizeOf(T) ..][0..@sizeOf(T)], .little);
    return v;
}

const Body = struct {
    age_ms: u64,
    block_positions: u64,
    blocks: u64,
    snapshots: u64,
    convs: []struct { id: u64, positions: u64, blocks: u64, shared_blocks: u64, idle_ms: u64, active: bool, table: ?[]const u8 = null },
    owner: []const u8,
};

test "GET /v1/pool: a shared start, the conversation in use with its whole span, owners, idle times, LRU order" {
    const gpa = t.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try plan.Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const l = try PoolLive.init(gpa, t.io);
    defer l.deinit();
    try l.snapAt(&m, &.{}, 0);

    // conversation A: 1,100 tokens (blocks 0-4), committed at 100 ms
    var sys: [1100]i64 = undefined;
    for (&sys, 0..) |*x, i| x.* = @intCast(i);
    var p1 = try plan.makePlan(&m, &sys, 4, true);
    defer p1.deinit(gpa);
    try plan.apply(&m, &p1, &sys);
    try l.snapAt(&m, &.{p1.X}, 50);
    const seq1 = try std.mem.concat(gpa, i64, &.{ &sys, &.{ 1, 2, 3 } });
    defer gpa.free(seq1);
    try plan.commit(&m, &p1, seq1, sys.len, seq1.len);
    try l.snapAt(&m, &.{}, 100);

    // conversation B shares A's first three blocks; while it runs (at 400 ms), its table is the whole span taken
    var sub: [1200]i64 = undefined;
    @memcpy(sub[0..1000], sys[0..1000]);
    for (sub[1000..], 0..) |*x, i| x.* = 50000 + @as(i64, @intCast(i));
    var p2 = try plan.makePlan(&m, &sub, 600, true);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try plan.apply(&m, &p2, &sub);
    try l.snapAt(&m, &.{p2.X}, 400);
    {
        const b = try std.json.parseFromSliceLeaky(Body, a, try l.jsonAt(a, 450), .{});
        try t.expectEqual(@as(u64, 50), b.age_ms);
        try t.expectEqual(@as(u64, 256), b.block_positions);
        try t.expectEqual(@as(u64, 40), b.blocks);
        try t.expectEqual(@as(usize, 2), b.convs.len);
        // B first (in use), then A
        try t.expectEqual(@as(u64, number(p2.X)), b.convs[0].id);
        try t.expect(b.convs[0].active);
        try t.expectEqual(@as(u64, 0), b.convs[0].idle_ms);
        try t.expectEqual(@as(u64, 768), b.convs[0].positions);
        try t.expectEqual(@as(u64, p2.table.len), b.convs[0].blocks);
        try t.expectEqual(@as(u64, 3), b.convs[0].shared_blocks);
        try t.expectEqualSlices(u32, p2.table, try decode(a, u32, b.convs[0].table.?));
        try t.expectEqual(@as(u64, number(p1.X)), b.convs[1].id);
        try t.expect(!b.convs[1].active);
        try t.expectEqual(@as(u64, 350), b.convs[1].idle_ms);
        try t.expect(b.convs[1].table == null);
        const own = try decode(a, u16, b.owner);
        try t.expectEqual(@as(usize, 40), own.len);
        for (p2.table[0..3]) |x| try t.expectEqual(shared, own[x]);
        for (p2.table[3..]) |x| try t.expectEqual(number(p2.X), own[x]);
        var n_a: usize = 0;
        var n_free: usize = 0;
        for (own) |o| {
            n_a += @intFromBool(o == number(p1.X));
            n_free += @intFromBool(o == free);
        }
        try t.expectEqual(p1.table.len - 3, n_a);
        try t.expectEqual(40 - p1.table.len - (p2.table.len - 3), n_free);
    }

    // B committed at 900 ms: nothing in use, B the most recently used, A's idle time unchanged since its commit
    const seq2 = try std.mem.concat(gpa, i64, &.{ &sub, &.{ 7, 8 } });
    defer gpa.free(seq2);
    try plan.commit(&m, &p2, seq2, sub.len, seq2.len);
    try l.snapAt(&m, &.{}, 900);
    {
        const b = try std.json.parseFromSliceLeaky(Body, a, try l.jsonAt(a, 1000), .{});
        try t.expectEqual(@as(u64, 5), b.snapshots);
        try t.expectEqual(@as(u64, number(p2.X)), b.convs[0].id);
        try t.expect(!b.convs[0].active and !b.convs[1].active);
        try t.expectEqual(@as(u64, 100), b.convs[0].idle_ms);
        try t.expectEqual(@as(u64, 900), b.convs[1].idle_ms);
        try t.expectEqual(@as(u64, 1202), b.convs[0].positions);
        try t.expectEqual(@as(u64, 5), b.convs[0].blocks); // the span past its commit given back / 確定の後ろの分は返した
    }
}

test "conversation numbers fit owner's 16 bits and miss free and shared" {
    try t.expectEqual(@as(u16, 1), number(0));
    try t.expectEqual(@as(u16, 0xFFFE), number(0xFFFD));
    try t.expectEqual(@as(u16, 1), number(0xFFFE));
}
