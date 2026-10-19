//! One rank's share of each checkpoint tensor: full tensors cut by split.zig's rules, and pre-sharded routed experts
//! (`...P_proj.rank{s}.S`) read as stored, or joined when fewer ranks read them.
//! チェックポイントの各テンソルの 1 ランク分: 完全なテンソルは split.zig の規則で切り、事前分割した routed expert
//! (`...P_proj.rank{s}.S`) は保存のまま読むか、読むランクの方が少なければ連結する。
const std = @import("std");
const core = @import("core");
const split = @import("split.zig");

const st = core.safetensors;

pub const Share = struct {
    dtype: st.DType,
    rank: u8,
    shape: [st.max_rank]usize,
    bytes: []u8,

    pub fn deinit(s: *Share, gpa: std.mem.Allocator) void {
        gpa.free(s.bytes);
        s.* = undefined;
    }
};

pub const Reader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    rank: usize,
    world: usize,
    // the ranks pre-sharded experts were saved for (TF_GLM53_EXPERTS_STORED_WORLD)
    // 事前分割した expert の保存時のランク数 (TF_GLM53_EXPERTS_STORED_WORLD)
    stored_world: usize,
    index: std.StringHashMapUnmanaged([]const u8) = .empty, // tensor name -> shard file / テンソル名 -> shard ファイル
    files: std.StringHashMapUnmanaged(*st.File) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: usize, world: usize, stored_world: usize) !Reader {
        if (stored_world % world != 0) return error.StoredWorldNotMultiple;
        var r: Reader = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = dir, .rank = rank, .world = world, .stored_world = stored_world };
        errdefer r.arena.deinit();
        const a = r.arena.allocator();
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = (parsed.object.get("weight_map") orelse return error.BadIndex).object;
        var it = map.iterator();
        while (it.next()) |e| try r.index.put(a, e.key_ptr.*, e.value_ptr.string);
        return r;
    }

    pub fn close(r: *Reader) void {
        var it = r.files.valueIterator();
        while (it.next()) |f| f.*.close(r.io);
        r.arena.deinit();
        r.* = undefined;
    }

    /// Drop the mapped shards from the page cache and unmap them: on a unified-memory GB10 the cache competes with the
    /// device allocations the next layers make. Later gets map them again.
    /// map した shard をページキャッシュから落として unmap する。unified memory の GB10 では、キャッシュが次の層の
    /// デバイス確保と取り合うため。後の get でまた map する。
    pub fn release(r: *Reader) void {
        var it = r.files.valueIterator();
        while (it.next()) |f| {
            _ = std.os.linux.fadvise(f.*.file.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
            f.*.close(r.io);
        }
        r.files.clearRetainingCapacity();
    }

    fn file(r: *Reader, fname: []const u8) !*st.File {
        if (r.files.get(fname)) |f| return f;
        const a = r.arena.allocator();
        const path = try std.fs.path.join(a, &.{ r.dir, fname });
        const f = try a.create(st.File);
        f.* = try st.File.open(r.gpa, r.io, path);
        try r.files.put(a, fname, f);
        return f;
    }

    fn tensor(r: *Reader, stored: []const u8) !st.Tensor {
        const fname = r.index.get(stored) orelse return error.MissingTensor;
        return (try r.file(fname)).get(stored) orelse error.MissingTensor;
    }

    /// The stored name of `name`'s share on stored rank `s` (`...P_proj.rank{s}.S`), in `buf`.
    /// 保存時のランク `s` にある `name` の分の保存名 (`...P_proj.rank{s}.S`) を `buf` に作る。
    fn storedName(buf: []u8, name: []const u8, s: usize) ![]const u8 {
        const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return error.BadName;
        return std.fmt.bufPrint(buf, "{s}.rank{d}{s}", .{ name[0..dot], s, name[dot..] });
    }

    /// Whether the checkpoint has `name` (whole, or as pre-sharded stored ranks).
    /// チェックポイントに `name` があるか (完全な形か、事前分割した保存ランクの形で)。
    pub fn has(r: *const Reader, name: []const u8) bool {
        if (r.index.contains(name)) return true;
        var buf: [256]u8 = undefined;
        const s0 = storedName(&buf, name, 0) catch return false;
        return r.index.contains(s0);
    }

    /// This rank's share of `name`, owned by the caller.
    /// `name` のうちこのランクの分。呼び出し側が所有する。
    pub fn get(r: *Reader, name: []const u8) !Share {
        if (r.index.contains(name)) return cut(r.gpa, try r.tensor(name), try split.rule(name), r.rank, r.world, name);
        const k = r.stored_world / r.world;
        const rule = try split.rule(name);
        var parts: [16]st.Tensor = undefined;
        if (k > parts.len) return error.TooManyStoredRanks;
        for (0..k) |i| {
            var buf: [256]u8 = undefined;
            parts[i] = try r.tensor(try storedName(&buf, name, r.rank * k + i));
        }
        return join(r.gpa, parts[0..k], rule, name);
    }
};

/// A full tensor's share on `rank` of `world`.
/// 完全なテンソルのうち、`world` ランク中の `rank` の分。
fn cut(gpa: std.mem.Allocator, t: st.Tensor, rule: split.Rule, rank: usize, world: usize, name: []const u8) !Share {
    var s: Share = .{ .dtype = t.dtype, .rank = t.rank, .shape = t.shape, .bytes = undefined };
    if (rule == .rep) {
        s.bytes = try gpa.dupe(u8, t.bytes);
        return s;
    }
    try split.checkCut(name, rule, t.shape[0..t.rank], world);
    const sh = split.share(rule, t.shape[0..t.rank], rank, world);
    const n = sh.hi - sh.lo;
    if (rule == .row) {
        const per = t.bytes.len / t.shape[0];
        s.shape[0] = n;
        s.bytes = try gpa.dupe(u8, t.bytes[sh.lo * per .. sh.hi * per]);
        return s;
    }
    // bytes of one [i, j] cell (the axes past the second)
    // [i, j] 1 マス分のバイト数 (3 番目以降の軸)
    const inner = t.bytes.len / (t.shape[0] * t.shape[1]);
    s.shape[1] = n;
    s.bytes = try gpa.alloc(u8, t.shape[0] * n * inner);
    for (0..t.shape[0]) |i| {
        const src = t.bytes[(i * t.shape[1] + sh.lo) * inner ..][0 .. n * inner];
        @memcpy(s.bytes[i * n * inner ..][0 .. n * inner], src);
    }
    return s;
}

/// Stored shares joined into one: concatenated along the split axis, or (replicated) checked byte-equal.
/// 保存された分を 1 つにまとめる: 分割軸に沿って連結する。複製のものはバイト単位で等しいことを確かめる。
fn join(gpa: std.mem.Allocator, parts: []const st.Tensor, rule: split.Rule, name: []const u8) !Share {
    const p0 = parts[0];
    var s: Share = .{ .dtype = p0.dtype, .rank = p0.rank, .shape = p0.shape, .bytes = undefined };
    for (parts[1..]) |p| if (p.dtype != p0.dtype or p.rank != p0.rank or !std.mem.eql(usize, p.shape[0..p.rank], p0.shape[0..p0.rank])) return error.StoredSharesDiffer;
    if (rule == .rep) {
        for (parts[1..]) |p| if (!std.mem.eql(u8, p.bytes, p0.bytes)) {
            std.log.err("{s}: a replicated part differs between stored ranks", .{name});
            return error.ReplicatedPartDiffers;
        };
        s.bytes = try gpa.dupe(u8, p0.bytes);
        return s;
    }
    const total = p0.bytes.len * parts.len;
    s.bytes = try gpa.alloc(u8, total);
    if (rule == .row) {
        s.shape[0] = p0.shape[0] * parts.len;
        for (parts, 0..) |p, i| @memcpy(s.bytes[i * p0.bytes.len ..][0..p0.bytes.len], p.bytes);
        return s;
    }
    // one leading index's bytes in a stored share
    // 保存された分での、先頭の添字 1 つ分のバイト数
    const row = p0.bytes.len / p0.shape[0];
    s.shape[1] = p0.shape[1] * parts.len;
    for (0..p0.shape[0]) |i| for (parts, 0..) |p, j| {
        @memcpy(s.bytes[(i * parts.len + j) * row ..][0..row], p.bytes[i * row ..][0..row]);
    };
    return s;
}

test "cut and join agree: a dim1 share of a full tensor equals its stored halves joined" {
    const gpa = std.testing.allocator;
    var full: [2 * 8 * 3]u8 = undefined;
    for (&full, 0..) |*b, i| b.* = @intCast(i);
    const t: st.Tensor = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 8, 3, 0 }, .bytes = &full };
    var whole = try cut(gpa, t, .dim1, 1, 2, "x.y");
    defer whole.deinit(gpa);
    var q0 = try cut(gpa, t, .dim1, 2, 4, "x.y");
    defer q0.deinit(gpa);
    var q1 = try cut(gpa, t, .dim1, 3, 4, "x.y");
    defer q1.deinit(gpa);
    const parts = [_]st.Tensor{ .{ .dtype = .u8, .rank = 3, .shape = q0.shape, .bytes = q0.bytes }, .{ .dtype = .u8, .rank = 3, .shape = q1.shape, .bytes = q1.bytes } };
    var joined = try join(gpa, &parts, .dim1, "x.y");
    defer joined.deinit(gpa);
    try std.testing.expectEqualSlices(u8, whole.bytes, joined.bytes);
    try std.testing.expectEqual(@as(usize, 4), joined.shape[1]);
}
