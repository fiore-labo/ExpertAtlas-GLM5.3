//! The device side of the KV pool's map (native_ops.cu PoolMap): its sizes and its bytes. A PoolMap is a 16-byte header
//! (write_lo, write_hi, err, z as int32) and the active slot's table, ceil(bucket(capacity + 1) / 256) entries (the
//! widest key range an indexer scores), the entries past the slot's blocks holding z (the zero block).
//! KV pool の表のデバイス側 (native_ops.cu の PoolMap): 大きさとバイト列。PoolMap は 16 バイトのヘッダ (write_lo、
//! write_hi、err、z の int32) と処理中のスロットの表で、項目数は ceil(bucket(capacity + 1) / 256) (indexer が採点する
//! 最大のキー範囲)。スロットのブロックより後ろの項目は z (ゼロのブロック) を持つ。
//!
//! TF_GLM53_POOL_TEST (P1 of glm53/POOL.md, without the planner): the caches laid out as a pool of
//! ceil(capacity / 256) + spare blocks and Z, the table a fixed permutation, writes allowed over [0, capacity); every
//! request must give the bytes of the contiguous cache. identity / reverse / random / offset.
//! TF_GLM53_POOL_TEST (glm53/POOL.md の P1、プランナなし): キャッシュを ceil(capacity / 256) + 予備のブロックと Z の
//! pool として並べ、表は決まった並べ替え、書き込みは [0, capacity) で許す。どの要求も連続のキャッシュと同じバイトを
//! 出さなければならない。identity / reverse / random / offset。
const std = @import("std");

pub const block: usize = 256;
pub const header: usize = 16;
/// Blocks beyond ceil(capacity / 256) in a test pool (room for offset and permutations off the identity).
/// テスト用 pool で ceil(capacity / 256) より多く持つブロック数 (offset と恒等でない並べ替えの余地)。
pub const test_spare: usize = 3;

pub const TestMode = enum { identity, reverse, random, offset };

/// TF_GLM53_PREFIX=pool and its settings (glm53/POOL.md 7).
/// TF_GLM53_PREFIX=pool とその設定 (glm53/POOL.md 7)。
pub const Env = struct {
    on: bool = false,
    // TF_GLM53_POOL_POSITIONS: the pool (default capacity + TF_GLM53_PREFIX_STORE, else twice the capacity)
    // TF_GLM53_POOL_POSITIONS: pool の位置数 (既定は capacity + TF_GLM53_PREFIX_STORE、なければ capacity の 2 倍)
    positions: usize = 0,
    slots: u32 = 64, // TF_GLM53_POOL_SLOTS
    verify: Verify = .sample, // TF_GLM53_POOL_VERIFY: sample / all / off
    draft_saves: u32 = 0, // TF_GLM53_POOL_DRAFT_SAVES (DFlash2's per-slot ring saves; 0: one live ring as now)

    pub const Verify = enum { sample, all, off };

    pub fn fromEnv(capacity: usize) !Env {
        var v: Env = .{};
        const m = std.c.getenv("TF_GLM53_PREFIX") orelse return v;
        if (!std.mem.eql(u8, std.mem.span(m), "pool")) return v;
        v.on = true;
        v.positions = 2 * capacity;
        if (std.c.getenv("TF_GLM53_PREFIX_STORE")) |x| v.positions = capacity + try std.fmt.parseInt(usize, std.mem.span(x), 10);
        if (std.c.getenv("TF_GLM53_POOL_POSITIONS")) |x| v.positions = try std.fmt.parseInt(usize, std.mem.span(x), 10);
        if (std.c.getenv("TF_GLM53_POOL_SLOTS")) |x| v.slots = try std.fmt.parseInt(u32, std.mem.span(x), 10);
        if (std.c.getenv("TF_GLM53_POOL_VERIFY")) |x| v.verify = std.meta.stringToEnum(Verify, std.mem.span(x)) orelse return error.BadPoolVerify;
        if (std.c.getenv("TF_GLM53_POOL_DRAFT_SAVES")) |x| v.draft_saves = try std.fmt.parseInt(u32, std.mem.span(x), 10);
        return v;
    }

    /// Usable blocks (Z comes after them).
    /// 使えるブロック数 (Z はその後ろ)。
    pub fn blocks(v: Env) usize {
        return (v.positions + block - 1) / block;
    }
};

pub fn testModeFromEnv() !?TestMode {
    const v = std.c.getenv("TF_GLM53_POOL_TEST") orelse return null;
    return std.meta.stringToEnum(TestMode, std.mem.span(v)) orelse error.BadPoolTestMode;
}

/// Table entries for a key range of `keys` positions (the engine's bucket(capacity + 1)).
/// `keys` 位置のキー範囲に要る表の項目数 (エンジンの bucket(capacity + 1))。
pub fn entries(keys: usize) usize {
    return (keys + block - 1) / block;
}

/// The bytes of a PoolMap: the header and `table` (entries past it hold z).
/// PoolMap のバイト列: ヘッダと `table` (それより後ろの項目は z)。
pub fn bytes(gpa: std.mem.Allocator, write_lo: u32, write_hi: u32, z: u32, table: []const u32, n_entries: usize) ![]u8 {
    if (table.len > n_entries) return error.PoolTableTooLong;
    const out = try gpa.alloc(u8, header + 4 * n_entries);
    const w = std.mem.bytesAsSlice(u32, out);
    w[0] = write_lo;
    w[1] = write_hi;
    w[2] = 0;
    w[3] = z;
    @memcpy(w[4 .. 4 + table.len], table);
    @memset(w[4 + table.len ..], z);
    return out;
}

/// A test table of `n` entries over `usable` blocks (n <= usable): identity j, reverse usable - 1 - j, random a fixed
/// permutation (seed 0x9e3779b9), offset j + spare.
/// `usable` ブロック上の `n` 項目のテスト用の表 (n <= usable): identity は j、reverse は usable - 1 - j、random は
/// 決まった並べ替え (seed 0x9e3779b9)、offset は j + 予備の数。
pub fn testTable(gpa: std.mem.Allocator, mode: TestMode, n: usize, usable: usize) ![]u32 {
    if (n > usable) return error.PoolTooSmall;
    const all = try gpa.alloc(u32, usable);
    defer gpa.free(all);
    for (all, 0..) |*x, i| x.* = @intCast(i);
    switch (mode) {
        .identity => {},
        .reverse => std.mem.reverse(u32, all),
        .random => {
            var prng = std.Random.DefaultPrng.init(0x9e3779b9);
            prng.random().shuffle(u32, all);
        },
        .offset => for (all, 0..) |*x, i| {
            x.* = @intCast((i + usable - n) % usable);
        },
    }
    return gpa.dupe(u32, all[0..n]);
}

test "test tables are permutations into the usable blocks" {
    const gpa = std.testing.allocator;
    for ([_]TestMode{ .identity, .reverse, .random, .offset }) |m| {
        const t = try testTable(gpa, m, 10, 13);
        defer gpa.free(t);
        var seen: [13]bool = @splat(false);
        for (t) |b| {
            try std.testing.expect(b < 13 and !seen[b]);
            seen[b] = true;
        }
    }
    const o = try testTable(gpa, .offset, 10, 13);
    defer gpa.free(o);
    try std.testing.expectEqual(@as(u32, 3), o[0]);
    const b = try bytes(gpa, 0, 1000, 13, o, 12);
    defer gpa.free(b);
    const w = std.mem.bytesAsSlice(u32, b);
    try std.testing.expectEqual(@as(u32, 13), w[3]);
    try std.testing.expectEqual(@as(u32, 13), w[4 + 11]);
    try std.testing.expectEqual(@as(u32, 3), w[4]);
}
