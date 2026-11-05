//! One rank's share of each checkpoint tensor: full tensors cut by split.zig's rules, and pre-sharded routed experts
//! (`...P_proj.rank{s}.S`) read as stored, or joined when fewer ranks read them.
//! チェックポイントの各テンソルの 1 ランク分: 完全なテンソルは split.zig の規則で切り、事前分割した routed expert
//! (`...P_proj.rank{s}.S`) は保存のまま読むか、読むランクの方が少なければ連結する。
const std = @import("std");
const core = @import("core");
const split = @import("split.zig");

const st = core.safetensors;

/// This process's bytes of shares read so far (every Reader.get; TF_GLM53_LOAD_PROGRESS reports rank 0's at start).
/// このプロセスがこれまでに読んだ分のバイト数 (Reader.get のたび。TF_GLM53_LOAD_PROGRESS は起動時の rank 0 の分を出す)。
pub var read_bytes: u64 = 0;

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

    /// A reader over one shard file of `dir`, its index taken from the file's own header: the whole checkpoint's
    /// index (model.safetensors.index.json; GLM-5.3: 170 MB, 1.8 M names) parsed holds about 1 GiB of host memory.
    /// `dir` の 1 つの shard ファイルだけを読む reader。索引はファイル自身のヘッダから作る: チェックポイント全体の
    /// 索引 (model.safetensors.index.json。GLM-5.3 では 170 MB、180 万個の名前) を解析すると約 1 GiB のホストメモリを持つ。
    pub fn openFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, fname: []const u8, rank: usize, world: usize, stored_world: usize) !Reader {
        if (stored_world % world != 0) return error.StoredWorldNotMultiple;
        var r: Reader = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = dir, .rank = rank, .world = world, .stored_world = stored_world };
        errdefer r.close();
        const a = r.arena.allocator();
        const own = try a.dupe(u8, fname);
        const f = try r.file(own);
        // the names outlive the file's own arena (release closes it)
        // 名前はファイル自身の arena (release が閉じる) より長く使う
        for (f.names.keys()) |k| try r.index.put(a, try a.dupe(u8, k), own);
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

    /// TF_GLM53_LOAD_READERS: the maps closed and their pages left in the page cache: the load pipe drops the ranges
    /// its readers read once the loader is done with them (load_pipe.Prefetch.done), so the steps read ahead stay.
    /// TF_GLM53_LOAD_READERS: map を閉じ、ページはページキャッシュに残す: ローダーが使い終えたら、load pipe が reader の
    /// 読んだ範囲を落とす (load_pipe.Prefetch.done)。先読みしたステップは残る。
    pub fn releaseKeep(r: *Reader) void {
        var it = r.files.valueIterator();
        while (it.next()) |f| f.*.close(r.io);
        r.files.clearRetainingCapacity();
    }

    /// TF_GLM53_LOAD_*: `release` with the maps closed before the pages are dropped. POSIX_FADV_DONTNEED leaves the
    /// pages still mapped in the cache, so `release` drops only those of earlier maps (10-31 on the ring: the page cache
    /// grew to 55 GB during the load until the kernel reclaimed it).
    /// TF_GLM53_LOAD_*: map を閉じてからページを落とす `release`。POSIX_FADV_DONTNEED は map 中のページをキャッシュに
    /// 残すので、`release` が落とすのは前の map のページだけ (10-31 のリングでは、読み込み中にページキャッシュが
    /// 55 GB まで増え、カーネルが回収するまで残った)。
    pub fn releaseDrop(r: *Reader) void {
        var it = r.files.valueIterator();
        while (it.next()) |f| {
            f.*.map.destroy(r.io);
            _ = std.os.linux.fadvise(f.*.file.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
            f.*.file.close(r.io);
            f.*.arena.deinit();
        }
        r.files.clearRetainingCapacity();
    }

    /// TF_GLM53_LOAD_*: a reader of rank `rank`'s shares over `base`'s index, not parsed again (GLM-5.3's: 170 MB of
    /// JSON, about 1 GiB parsed). `base` outlives it; closing it closes only its own files.
    /// TF_GLM53_LOAD_*: `base` の索引をそのまま使う、ランク `rank` の分の reader (索引を再び解析しない。GLM-5.3 では
    /// 170 MB の JSON、解析すると約 1 GiB)。`base` の方が長く生きる。閉じると自分のファイルだけを閉じる。
    pub fn openSharing(base: *const Reader, rank: usize) Reader {
        return .{ .gpa = base.gpa, .io = base.io, .arena = .init(base.gpa), .dir = base.dir, .rank = rank, .world = base.world, .stored_world = base.stored_world, .index = base.index };
    }

    /// Whether `name` is stored as pre-sharded parts (`...P_proj.rank{s}.S`), not whole.
    /// `name` が完全な形ではなく、事前分割した分 (`...P_proj.rank{s}.S`) として保存されているか。
    pub fn preSharded(r: *const Reader, name: []const u8) bool {
        return !r.index.contains(name);
    }

    /// TF_GLM53_LOAD_PACKERS: this rank's stored parts of the pre-sharded `name`, checked as `join` checks them, with
    /// the joined share's shape and bytes; nothing is read but the files' headers (the parts are slices of the maps,
    /// valid until `release`). `joinInto` makes from them the bytes `get` returns.
    /// TF_GLM53_LOAD_PACKERS: 事前分割した `name` のこのランクの保存された分。`join` と同じく確かめ、連結した分の形と
    /// バイト数を添える。読むのはファイルの header だけ (分は map の一部で、`release` まで有効)。`joinInto` がそこから
    /// `get` と同じバイトを作る。
    pub fn storedParts(r: *Reader, name: []const u8) !Parts {
        const k = r.stored_world / r.world;
        const rule = try split.rule(name);
        var ts: [16]st.Tensor = undefined;
        if (k > ts.len or k == 0) return error.TooManyStoredRanks;
        for (0..k) |i| {
            var buf: [256]u8 = undefined;
            ts[i] = try r.tensor(try storedName(&buf, name, r.rank * k + i));
        }
        const p0 = ts[0];
        for (ts[1..k]) |p| if (p.dtype != p0.dtype or p.rank != p0.rank or !std.mem.eql(usize, p.shape[0..p.rank], p0.shape[0..p0.rank])) return error.StoredSharesDiffer;
        var out: Parts = .{ .bytes = undefined, .k = k, .rule = rule, .dtype = p0.dtype, .rank = p0.rank, .shape = p0.shape, .rows = p0.shape[0], .len = if (rule == .rep) p0.bytes.len else p0.bytes.len * k };
        for (0..k) |i| out.bytes[i] = ts[i].bytes;
        switch (rule) {
            .rep => {},
            .row => out.shape[0] = p0.shape[0] * k,
            .dim1 => out.shape[1] = p0.shape[1] * k,
        }
        return out;
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
        const s = try r.getShare(name);
        read_bytes += s.bytes.len;
        return s;
    }

    fn getShare(r: *Reader, name: []const u8) !Share {
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

/// One rank's stored parts of a pre-sharded tensor (Reader.storedParts): the k parts' bytes, the rule joining them, and the
/// joined share's dtype, shape and bytes.
/// 事前分割したテンソルの 1 ランクの保存された分 (Reader.storedParts): k 個の分のバイト、連結の規則、連結した分の dtype・
/// 形・バイト数。
pub const Parts = struct {
    bytes: [16][]const u8,
    k: usize,
    rule: split.Rule,
    dtype: st.DType,
    rank: u8,
    shape: [st.max_rank]usize,
    rows: usize, // the parts' leading axis (join's rows for dim1) / 分の先頭の軸 (dim1 の join の行数)
    len: usize,
};

/// `join`'s bytes into `dst` (`pt.len` bytes): the same concatenation, row interleave or replicated check. The
/// packers' join (load_pipe.Packers), on any thread.
/// `join` のバイトを `dst` (`pt.len` バイト) に作る: 同じ連結・行の交互の並べ・複製の確認。packer
/// (load_pipe.Packers) の連結で、どのスレッドからでもよい。
pub fn joinInto(dst: []u8, pt: *const Parts) !void {
    if (dst.len != pt.len) return error.JoinSize;
    const k = pt.k;
    const p0 = pt.bytes[0];
    switch (pt.rule) {
        .rep => {
            for (pt.bytes[1..k]) |b| if (!std.mem.eql(u8, b, p0)) return error.ReplicatedPartDiffers;
            @memcpy(dst, p0);
        },
        .row => for (pt.bytes[0..k], 0..) |b, i| @memcpy(dst[i * p0.len ..][0..p0.len], b),
        .dim1 => {
            if (pt.rows == 0) return error.JoinSize;
            const row = p0.len / pt.rows;
            for (0..pt.rows) |i| for (pt.bytes[0..k], 0..) |b, j| {
                @memcpy(dst[(i * k + j) * row ..][0..row], b[i * row ..][0..row]);
            };
        },
    }
}

test "joinInto makes join's bytes (dim1, row, rep)" {
    const gpa = std.testing.allocator;
    var a: [2 * 4 * 3]u8 = undefined;
    var b: [2 * 4 * 3]u8 = undefined;
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = @intCast(i);
        y.* = @intCast(100 + i);
    }
    const parts = [_]st.Tensor{ .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 4, 3, 0 }, .bytes = &a }, .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 4, 3, 0 }, .bytes = &b } };
    for ([_]split.Rule{ .dim1, .row }) |rule| {
        var s = try join(gpa, &parts, rule, "x.y");
        defer s.deinit(gpa);
        var pt: Parts = .{ .bytes = undefined, .k = 2, .rule = rule, .dtype = .u8, .rank = 3, .shape = parts[0].shape, .rows = 2, .len = 2 * a.len };
        pt.bytes[0] = &a;
        pt.bytes[1] = &b;
        var out: [2 * a.len]u8 = undefined;
        try joinInto(&out, &pt);
        try std.testing.expectEqualSlices(u8, s.bytes, &out);
    }
    var rp: Parts = .{ .bytes = undefined, .k = 2, .rule = .rep, .dtype = .u8, .rank = 3, .shape = parts[0].shape, .rows = 2, .len = a.len };
    rp.bytes[0] = &a;
    rp.bytes[1] = &a;
    var ro: [a.len]u8 = undefined;
    try joinInto(&ro, &rp);
    try std.testing.expectEqualSlices(u8, &a, &ro);
    rp.bytes[1] = &b;
    try std.testing.expectError(error.ReplicatedPartDiffers, joinInto(&ro, &rp));
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
