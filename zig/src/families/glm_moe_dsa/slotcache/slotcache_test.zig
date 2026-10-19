//! Tests of the slot cache's host side: hashing vectors, the store, worked traces (reprs and digests), golden random
//! walks from a reference planner (testdata/golden_plans.txt) and a random walk checking the tree's invariants after
//! every request. `zig build test-glm53-slots`.
//! スロットキャッシュのホスト側のテスト: ハッシュのベクトル、ストア、手順を追った例 (repr と digest)、参照プランナに
//! よる golden のランダムウォーク (testdata/golden_plans.txt)、要求ごとに木の不変条件を確かめるランダムウォーク。
//! `zig build test-glm53-slots`。
const std = @import("std");
const hashing = @import("hashing.zig");
const alloc_mod = @import("alloc.zig");
const tree = @import("tree.zig");

const Extent = alloc_mod.Extent;
const Store = alloc_mod.Store;
const Meta = tree.Meta;
const t = std.testing;
const gpa = std.testing.allocator;

test {
    _ = alloc_mod;
    _ = hashing;
    _ = tree;
}

fn hex16(s: []const u8) [16]u8 {
    var out: [16]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch unreachable;
    return out;
}

fn range(comptime lo: i64, comptime hi: i64) [hi - lo]i64 {
    var out: [hi - lo]i64 = undefined;
    for (&out, 0..) |*x, i| x.* = lo + @as(i64, @intCast(i));
    return out;
}

// ------------------------------------------------------------------------------------------------- hashing ---

test "mix64, pids and lane_hash hit Python's values" {
    try t.expectEqual(@as(u64, 0), hashing.mix64(0));
    try t.expectEqual(@as(u64, 0x5692161D100B05E5), hashing.mix64(1));
    try t.expectEqual(@as(u64, 0xDBD238973A2B148A), hashing.mix64(2));
    try t.expectEqual(@as(u64, 0xF36CF1164265DD51), hashing.mix64(12345));
    try t.expectEqual(@as(u64, 0xB4D055FCF2CBBD7B), hashing.mix64(std.math.maxInt(u64)));
    const p = try hashing.pids(gpa, &.{ 1, 2, 3 });
    defer gpa.free(p);
    try t.expectEqualSlices(u64, &.{ 0x5692161D100B05E5, 0xEDEA4BFB2CA25CB3, 0xD78A44630898DF7C }, p);
    var z: [1]u64 = undefined;
    hashing.pidsInto(&z, &.{0});
    try t.expectEqual(@as(u64, 0), z[0]);
    var q: [10]u64 = undefined;
    hashing.pidsInto(&q, &range(1000, 1010));
    try t.expectEqual(@as(u64, 0xC2CD542105832FAE), q[9]);
    try t.expectEqual(@as(u64, 0x86FA59FA943FE6F8), hashing.laneHash(&.{}, 0));
    try t.expectEqual(@as(u64, 0x4C4E029500D963FE), hashing.laneHash(&.{ 1, 2 }, 3));
}

test "row hash vectors VEC0, VEC1, VEC2" {
    var row: [40]u8 = undefined;
    for (&row, 0..) |*x, i| x.* = @intCast(i);
    const W = [_]u32{ 16, 16, 8 };
    try t.expectEqual(@as(u64, 0x456D1D2379EFE2CB), try hashing.rowHash(&row, &W, 0, 0, 0));
    try t.expectEqual(@as(u64, 0x52D695A5C059DB77), try hashing.rowHash(&row, &W, 819199, 0xFFFFFFFFFFFFFFFF, (1 << 40) + 3));
    const big = try gpa.alloc(u8, 54192);
    defer gpa.free(big);
    for (big, 0..) |*x, i| x.* = @truncate(i);
    try t.expectEqual(@as(u64, 0x7DEE0E288E59128D), try hashing.rowHash(big, &hashing.FLAGSHIP_WIDTHS, 123457, 0xD78A44630898DF7C, 9));
    try t.expectError(error.BadWidths, hashing.rowHash(row[0..32], &W, 0, 0, 0));
}

test "the row hash sees every kind of change" {
    var row: [40]u8 = undefined;
    var prng: std.Random.DefaultPrng = .init(3);
    prng.random().bytes(&row);
    const W = [_]u32{ 16, 16, 8 };
    const base = try hashing.rowHash(&row, &W, 100, 12345, 7);
    var v = row;
    v[21] ^= 1;
    try t.expect(try hashing.rowHash(&v, &W, 100, 12345, 7) != base);
    v = row;
    @memcpy(v[0..8], row[8..16]);
    @memcpy(v[8..16], row[0..8]);
    try t.expect(try hashing.rowHash(&v, &W, 100, 12345, 7) != base);
    v = row;
    @memcpy(v[0..16], row[16..32]);
    @memcpy(v[16..32], row[0..16]);
    try t.expect(try hashing.rowHash(&v, &W, 100, 12345, 7) != base);
    try t.expect(try hashing.rowHash(&row, &W, 101, 12345, 7) != base);
    try t.expect(try hashing.rowHash(&row, &W, 100, 12346, 7) != base);
    try t.expect(try hashing.rowHash(&row, &W, 100, 12345, 8) != base);
}

// --------------------------------------------------------------------------------------------------- store ---

test "store: congruent first fit, pieces, split and exact frees" {
    try t.expectError(error.InvalidStore, Store.init(gpa, 5, 2));
    try t.expectError(error.InvalidStore, Store.init(gpa, -2, 2));
    var st = try Store.init(gpa, 4, 2);
    defer st.deinit();
    try t.expectEqual(@as(?Extent, null), try st.alloc(1, 4));
    try t.expectEqual(@as(?Extent, Extent{ .s = 1, .a = 1, .n = 3 }), try st.alloc(1, 3));

    var s2 = try Store.init(gpa, 16, 2);
    defer s2.deinit();
    const e = try s2.take(1, 5, 5);
    try t.expectError(error.NotCongruent, s2.take(2, 5, 1));
    const lr = try e.split(7);
    try t.expectEqual(Extent{ .s = 1, .a = 5, .n = 2 }, lr[0]);
    try t.expectEqual(Extent{ .s = 3, .a = 7, .n = 3 }, lr[1]);
    try t.expectError(error.BadSplit, e.split(5));
    try s2.release(lr[0]);
    try t.expectEqualSlices(alloc_mod.Interval, &.{ .{ .lo = 0, .hi = 3 }, .{ .lo = 6, .hi = 16 } }, s2.free.items);
    try t.expectError(error.DoubleFree, s2.release(lr[0]));
    try t.expectError(error.OutOfRange, s2.release(.{ .s = 15, .a = 15, .n = 2 }));
    try s2.check(&.{lr[1]});
    try t.expectEqual(@as(?Extent, Extent{ .s = 4, .a = 6, .n = 2 }), (Extent{ .s = 3, .a = 5, .n = 5 }).clip(6, 8));
    try t.expectEqual(@as(?Extent, null), (Extent{ .s = 3, .a = 5, .n = 5 }).clip(10, 12));

    // pieces across fragmentation: the longest congruent run first, ties the lowest; dcp 2 skips a run too short
    // 断片化した空きからの片: 最長の合同な区間から、同長なら最も低いものから。dcp 2 では短すぎる区間を飛ばす
    var s3 = try Store.init(gpa, 20, 2);
    defer s3.deinit();
    _ = try s3.take(2, 2, 2); // free [0,2) [4,20)
    _ = try s3.take(8, 8, 4); // free [0,2) [4,8) [12,20)
    const ps = (try s3.allocPieces(1, 11)).?;
    defer gpa.free(ps);
    try t.expectEqualSlices(Extent, &.{ .{ .s = 13, .a = 1, .n = 7 }, .{ .s = 4, .a = 8, .n = 4 } }, ps);
    try t.expectEqualSlices(alloc_mod.Interval, &.{ .{ .lo = 0, .hi = 2 }, .{ .lo = 12, .hi = 13 } }, s3.free.items);
    // 1 at s=1, 1 at s=0 (tie: the lowest), then 13 is past [12, 13)
    // s=1 に 1、s=0 に 1 (同長なので最も低いもの)、次の 13 は [12, 13) を越える
    try t.expect((try s3.allocPieces(1, 3)) == null);
    try t.expectEqualSlices(alloc_mod.Interval, &.{ .{ .lo = 0, .hi = 2 }, .{ .lo = 12, .hi = 13 } }, s3.free.items);
}

test "store: random alloc and release stay a partition" {
    var prng: std.Random.DefaultPrng = .init(4);
    const r = prng.random();
    for ([_]i64{ 1, 2, 4 }) |dcp| {
        var st = try Store.init(gpa, 96, dcp);
        defer st.deinit();
        var used: std.ArrayList(Extent) = .empty;
        defer used.deinit(gpa);
        for (0..400) |_| {
            if (used.items.len > 0 and r.float(f64) < 0.45) {
                const e = used.orderedRemove(r.uintLessThan(usize, used.items.len));
                // free a split part sometimes
                // ときどき分割した片を解放する
                if (e.n > 1 and r.boolean()) {
                    const lr = try e.split(e.a + 1 + @as(i64, @intCast(r.uintLessThan(u64, @intCast(e.n - 1)))));
                    try st.release(lr[0]);
                    try used.append(gpa, lr[1]);
                } else try st.release(e);
            } else {
                const a: i64 = @intCast(r.uintLessThan(u64, 100));
                const n: i64 = @intCast(r.intRangeLessThan(u64, 1, 12));
                if (try st.allocPieces(a, n)) |ps| {
                    defer gpa.free(ps);
                    var p = a;
                    for (ps) |e| {
                        try t.expectEqual(p, e.a);
                        p = e.b();
                    }
                    try t.expectEqual(a + n, p);
                    try used.appendSlice(gpa, ps);
                }
            }
            try st.check(used.items);
        }
    }
}

// ------------------------------------------------------------------------------------------------- traces ---

/// One request as the engine runs it: plan, adopt post, commit X ++ out[:-1] from c with the next generation.
/// エンジンと同じ 1 要求: プランを作り、post を採用し、X ++ out[:-1] を c から次の世代で commit する。
const Step = struct {
    X: []const i64,
    out: []const i64,
    d0: []const u8,
    pd: []const u8,
    repr: []const u8,
    post: []const u8,
    free: []const alloc_mod.Interval = &.{},
    epoch: i64 = 0,
    next_id: i64 = 0,
    after: []const u8,
    tick: i64 = 0,
};

fn concat(comptime a: []const i64, comptime b: []const i64) [a.len + b.len]i64 {
    return (a[0..a.len].* ++ b[0..b.len].*);
}

fn runTrace(S: i64, dcp: i64, cap: i64, tail: i64, mn: i64, steps: []const Step, last_post_repr: ?[]const u8, last_words: ?[8]i32) !void {
    var m = try Meta.init(gpa, S, dcp);
    defer m.deinit();
    var gen: i64 = 0;
    for (steps, 0..) |s, i| {
        var p = try tree.makePlan(gpa, &m, s.X, @as(i64, @intCast(s.out.len)) + 2, cap, tail, mn, true);
        defer p.deinit();
        try t.expectEqual(hex16(s.d0), p.d0);
        try t.expectEqual(hex16(s.d0), m.digest());
        const r = try p.reprText(gpa);
        defer gpa.free(r);
        try t.expectEqualStrings(s.repr, r);
        try t.expectEqual(hex16(s.pd), tree.planDigest(&p));
        try t.expectEqual(hex16(s.post), p.post.digest());
        if (s.free.len > 0) try t.expectEqualSlices(alloc_mod.Interval, s.free, p.post.store.free.items);
        if (s.epoch > 0) try t.expectEqual(s.epoch, p.post.epoch);
        if (s.next_id > 0) try t.expectEqual(s.next_id, p.post.next_id);
        if (i == steps.len - 1) if (last_post_repr) |want| {
            const txt = try p.post.reprText(gpa);
            defer gpa.free(txt);
            try t.expectEqualStrings(want, txt);
        };
        try p.post.check();
        std.mem.swap(Meta, &m, &p.post);
        gen += 1;
        const seq = try std.mem.concat(gpa, i64, &.{ s.X, s.out[0 .. s.out.len - 1] });
        defer gpa.free(seq);
        const freed = try tree.commitSegment(&m, gpa, p.c, seq[@intCast(p.c)..], gen);
        gpa.free(freed);
        try t.expectEqual(hex16(s.after), m.digest());
        if (s.tick > 0) try t.expectEqual(s.tick, m.tick);
        try m.check();
    }
    if (last_words) |w| try t.expectEqual(w, tree.words(m.digest()));
}

const root5 = [_]i64{ 1, 2, 3, 4, 5 };
const trA = concat(&root5, &range(100, 130));
const trB = concat(&root5, &range(200, 230));

test "A, B, back to A by a swap (Meta(40, 2), cap 64, tail 2, min_resume 3)" {
    const post3 =
        \\(0, None, 0, 0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 0, 'R', [], [1], False)
        \\(1, 0, 0, 5, 'ea0d7246ca8346f6137619c0aaf7410ae4ffe69d1f0f702d6760e661de6df857', 1, 'A', [], [2, 3], False)
        \\(2, 1, 5, 31, '31e72a6a42a632a7d2665cf3d6ce42c4147006eddd403dc4fb5114912689de47', 1, 'A', [], [], False)
        \\(3, 1, 5, 31, '943e510873ac02c6d2b3a40d641c6e3e6076fdfe2d09c11443988e82626d8ea1', 2, 'S', [(1, 5, 31)], [], False)
        \\([1, 2], [[0, 1], [32, 40]], 40, 3, 4)
    ;
    try runTrace(40, 2, 64, 2, 3, &.{
        .{ .X = &trA, .out = &.{ 7, 8 }, .d0 = "e0fa2a808da1549f9cc34ef4ded0bbea", .pd = "cae539e37f9aff750a2d2e24eb7cc5d5", .repr = "(0, 0, 0, 'none', 0, [], [])", .post = "2f1d07564c279134237bf1130c9298a6", .after = "d97254bce243e3fa462d6bbf60d66922", .tick = 2 },
        .{ .X = &trB, .out = &.{ 9, 9 }, .d0 = "d97254bce243e3fa462d6bbf60d66922", .pd = "2c24da84aa67c77e44b33e7c063109c0", .repr = "(5, 5, 1, 'plain', 0, [('save', 5, 36, None, 1, None, 2, 2)], [])", .post = "1d328288fd190e2dd91bd0c9f0c148e6", .after = "2defce72883e7fb4a07e47bacf643a9e", .tick = 4 },
        .{ .X = &comptime concat(&trA, &.{ 7, 8, 50 }), .out = &.{3}, .d0 = "2defce72883e7fb4a07e47bacf643a9e", .pd = "9a8caaeb7f8bd6909b580b6ffd466fc9", .repr = "(36, 5, 2, 'swap', 0, [('swap', 5, 36, 1, 1, 1, 2, 3)], [])", .post = "34183eb37db02416de7ca9fc1732d77c", .free = &.{ .{ .lo = 0, .hi = 1 }, .{ .lo = 32, .hi = 40 } }, .epoch = 3, .next_id = 4, .after = "f2a5b5613871fc8dd1fac38f1f1891cb", .tick = 6 },
    }, post3, .{ 42482, 25013, 28984, 36348, 64209, 36803, 6175, 52113 });
}

test "a branch switch restores [h, c) and saves A (Meta(64, 2), cap 128, tail 3, min_resume 4)" {
    const X1 = comptime range(100, 140);
    try runTrace(64, 2, 128, 3, 4, &.{
        .{ .X = &X1, .out = &.{ 7, 8, 9 }, .d0 = "94eb968eb8a251bbea9321988864dd33", .pd = "ae36e99d427d3b411d8e6db34c33098e", .repr = "(0, 0, 0, 'none', 0, [], [])", .post = "17b48323434911424e21c1a820f020e8", .after = "f09c1422fae3a6937500a3c970d5a8b2" },
        .{ .X = &comptime concat(&range(100, 120), &range(500, 530)), .out = &.{1}, .d0 = "f09c1422fae3a6937500a3c970d5a8b2", .pd = "ccb6314c5f0277b6df07f1ec5a3dab77", .repr = "(20, 20, 1, 'plain', 0, [('save', 20, 42, None, 0, None, 2, 2)], [])", .post = "72b4c35c713db3350d9048c4cb691ad7", .free = &.{.{ .lo = 22, .hi = 64 }}, .after = "7fa5a245a87d3d95e0b03bbd6873f2f7" },
        .{ .X = &comptime concat(&X1, &.{ 7, 8, 9, 10, 11, 12, 13 }), .out = &.{1}, .d0 = "7fa5a245a87d3d95e0b03bbd6873f2f7", .pd = "9741fca3f11732eb51186931ed05b8be", .repr = "(42, 20, 2, 'plain', 0, [('save', 20, 50, None, 22, None, 3, 3), ('restore', 20, 42, 0, None, 0, 2, None)], [(0, 20, 22)])", .post = "6b24e04b4da99c677b53f3ac6075430d", .free = &.{ .{ .lo = 0, .hi = 22 }, .{ .lo = 52, .hi = 64 } }, .epoch = 3, .next_id = 4, .after = "7f3602399adddccb28afb4f1af3074e9" },
    }, null, null);
}

test "no room drops A (Meta(8, 2))" {
    try runTrace(8, 2, 64, 2, 3, &.{
        .{ .X = &trA, .out = &.{ 7, 8 }, .d0 = "51c4163f8c471a07c8e90b20c096f86c", .pd = "e3604351ae877fb8ea6e8bfd9522cdae", .repr = "(0, 0, 0, 'none', 0, [], [])", .post = "e603bfb97e381fffe7c693050fbd61b8", .after = "46b7eb5c52f7b3d9b6c2aaf1fe4242b7" },
        .{ .X = &trB, .out = &.{ 9, 9 }, .d0 = "46b7eb5c52f7b3d9b6c2aaf1fe4242b7", .pd = "b98f2cc09ac1d02d4afbbd420a802a70", .repr = "(5, 5, 1, 'drop', 0, [], [])", .post = "8654f11aeca402de094028cac26ba593", .free = &.{.{ .lo = 0, .hi = 8 }}, .after = "bc4b4c00e7807b4d6eceb603a99fe359" },
    }, null, null);
}

test "LRU eviction reuses the freed slots (Meta(40, 1))" {
    try runTrace(40, 1, 64, 2, 3, &.{
        .{ .X = &comptime range(1000, 1015), .out = &.{5}, .d0 = "e0fa2a808da1549f9cc34ef4ded0bbea", .pd = "cae539e37f9aff750a2d2e24eb7cc5d5", .repr = "(0, 0, 0, 'none', 0, [], [])", .post = "2f1d07564c279134237bf1130c9298a6", .after = "6b13bb87fc9dd44ffce072a82fcf6313" },
        .{ .X = &comptime range(2000, 2015), .out = &.{5}, .d0 = "6b13bb87fc9dd44ffce072a82fcf6313", .pd = "693b1699c54d492c6652bcea28c96926", .repr = "(0, 0, 0, 'plain', 0, [('save', 0, 15, None, 0, None, 1, 1)], [])", .post = "bfaffd5de4bfb94ccc4ca568d7cdb610", .after = "7b221345b0d6233c7c01d3e7bae7ddb3" },
        .{ .X = &comptime range(3000, 3015), .out = &.{5}, .d0 = "7b221345b0d6233c7c01d3e7bae7ddb3", .pd = "9a044d06fb004a071efafda9d4fd838c", .repr = "(0, 0, 0, 'plain', 0, [('save', 0, 15, None, 15, None, 2, 2)], [])", .post = "ce61c5a19a2419721055ce694e190ece", .after = "ea92212224ce1c78d2b6b63b4307f50a" },
        .{ .X = &comptime range(4000, 4015), .out = &.{5}, .d0 = "ea92212224ce1c78d2b6b63b4307f50a", .pd = "b7602a6290f9bec8f5f70c5e96b1fc4d", .repr = "(0, 0, 0, 'plain', 1, [('save', 0, 15, None, 0, None, 3, 3)], [(0, 0, 15)])", .post = "78f6465d1c11d10cde16380952ef28bc", .free = &.{.{ .lo = 30, .hi = 40 }}, .after = "32d28f688d8962a604ec2802cd442ea8", .tick = 8 },
    }, null, null);
}

// ---------------------------------------------------------------------------------------------- edge cases ---

test "edge cases: empty tree, too long, S = 0, empty commit, resume off, the input never changes" {
    try t.expectError(error.InvalidStore, Meta.init(gpa, 6, 4));
    var m = try Meta.init(gpa, 0, 2);
    defer m.deinit();
    const X = comptime range(10, 60);
    {
        var p = try tree.makePlan(gpa, &m, &X, 5, 64, 17, 16, true);
        defer p.deinit();
        try t.expectEqual(tree.Mode.none, p.mode);
        try t.expectEqual(@as(i64, 0), p.c);
        try t.expectEqual(@as(usize, 0), p.ops.len);
        std.mem.swap(Meta, &m, &p.post);
    }
    gpa.free(try tree.commitSegment(&m, gpa, 0, &X, 1));
    const d = m.digest();
    try t.expectError(error.RequestTooLong, tree.makePlan(gpa, &m, &X, 15, 64, 17, 16, true));
    try t.expectEqual(d, m.digest());
    try t.expectEqual(@as(i64, 2), m.tick);
    // S = 0: the same prompt again resumes at L - tail and drops A's [c, n_A) (single-slot prefix cache)
    // S = 0: 同じ prompt を再び送ると L - tail から再開し、A の [c, n_A) を捨てる (スロット 1 つの prefix キャッシュ)
    {
        var p = try tree.makePlan(gpa, &m, &X, 5, 64, 17, 16, true);
        defer p.deinit();
        try t.expectEqual(tree.Mode.drop, p.mode);
        try t.expectEqual(@as(i64, 33), p.c);
        try t.expectEqual(@as(i64, 33), p.post.nAct());
        try t.expectEqual(d, m.digest());
        try p.post.check();
    }
    // resume off: c = 0, A still leaves (dropped: no store)
    // 再開なし: c = 0 で、A はそれでも外れる (ストアがないので捨てる)
    {
        var p = try tree.makePlan(gpa, &m, &X, 5, 64, 17, 16, false);
        defer p.deinit();
        try t.expectEqual(@as(i64, 0), p.c);
        try t.expectEqual(tree.Mode.drop, p.mode);
        try t.expectEqual(@as(i64, 0), p.post.nAct());
    }
    // short prompts (L < tail) and short matches (< min_resume) resume at 0
    // 短い prompt (L < tail) と短い一致 (< min_resume) は 0 から再開する
    {
        var p = try tree.makePlan(gpa, &m, X[0..10], 5, 64, 17, 1, true);
        defer p.deinit();
        try t.expectEqual(@as(i64, 0), p.c);
        var p2 = try tree.makePlan(gpa, &m, &X, 5, 64, 17, 34, true);
        defer p2.deinit();
        try t.expectEqual(@as(i64, 0), p2.c);
    }
    // an empty segment: tick only
    // 空の区間: tick だけ進む
    const tick = m.tick;
    const e = try tree.commitSegment(&m, gpa, 50, &.{}, 2);
    gpa.free(e);
    try t.expectEqual(tick + 1, m.tick);
    try t.expectEqual(d, m.digest());
    try t.expectError(error.PlanAssert, tree.commitSegment(&m, gpa, 7, &.{1}, 3));
}

test "split: extents cut inside, active insertion, boundary cuts are no-ops" {
    var m = try Meta.init(gpa, 32, 2);
    defer m.deinit();
    const a = try m.newNode(tree.ROOT, &range(0, 10), 1, .A);
    try m.active.append(gpa, a);
    const s = try m.newNode(a, &range(50, 56), 1, .S);
    try (try m.at(s)).ext.appendSlice(gpa, &.{ try m.store.take(0, 10, 4), try m.store.take(20, 14, 2) });
    try m.check();
    try m.cut(&.{ a, s }, 10); // a boundary / 境界
    try m.cut(&.{ a, s }, 16); // past the end / 末尾より先
    try t.expectEqual(@as(i64, 3), m.next_id);
    const r = try m.split(s, 12);
    try t.expectEqualSlices(Extent, &.{.{ .s = 0, .a = 10, .n = 2 }}, (try m.at(s)).ext.items);
    try t.expectEqualSlices(Extent, &.{ .{ .s = 2, .a = 12, .n = 2 }, .{ .s = 20, .a = 14, .n = 2 } }, (try m.at(r)).ext.items);
    const r2 = try m.split(a, 4);
    try t.expectEqualSlices(i64, &.{ a, r2 }, m.active.items);
    try t.expectEqual(r2, (try m.at(s)).parent.?);
    try t.expectError(error.BadSplit, m.split(a, 4));
    try m.check();
    const sm = m.summary();
    try t.expectEqual(@as(i64, 6), sm.stored);
    try t.expectEqual(@as(usize, 4), sm.nodes);
    try t.expectEqual(@as(usize, 1), sm.leaves);
}

// ------------------------------------------------------------------------------------------ golden walks ---

/// The reference walks (S 0/32/160/384 x dcp 1/2/4, and S = cap swap walks), a line each:
/// `s name S dcp cap tail min_resume` starts a Meta; `q resume max_out nX X.. nout out.. P d0 plan_digest c h t_leaf
/// mode evicted n_ops n_free_after post_digest n_released digest_after_commit` (or `.. E digest` for RequestTooLong).
/// The commit is commitSegment(m, c, (X ++ out[:-1])[c:], gen) with gen 1, 2, .. a Meta.
/// 参照のウォーク (S 0/32/160/384 x dcp 1/2/4 と、S = cap の swap ウォーク)。1 行ずつ:
/// `s name S dcp cap tail min_resume` で Meta を始める。`q resume max_out nX X.. nout out.. P d0 plan_digest c h
/// t_leaf mode evicted n_ops n_free_after post_digest n_released digest_after_commit` (RequestTooLong なら
/// `.. E digest`)。commit は commitSegment(m, c, (X ++ out[:-1])[c:], gen) で、gen は Meta ごとに 1, 2, ..。
const golden = @embedFile("testdata/golden_plans.txt");

const Tok = struct {
    it: std.mem.TokenIterator(u8, .scalar),
    fn s(k: *Tok) []const u8 {
        return k.it.next().?;
    }
    fn i(k: *Tok) i64 {
        return std.fmt.parseInt(i64, k.s(), 10) catch unreachable;
    }
    fn list(k: *Tok, out: *std.ArrayList(i64)) !void {
        out.clearRetainingCapacity();
        const n = k.i();
        for (0..@intCast(n)) |_| try out.append(gpa, k.i());
    }
};

test "golden random walks: every plan, digest and commit equals Python's" {
    var lines = std.mem.tokenizeScalar(u8, golden, '\n');
    var m: ?Meta = null;
    defer if (m) |*x| x.deinit();
    var cfg: [5]i64 = undefined;
    var gen: i64 = 0;
    var X: std.ArrayList(i64) = .empty;
    defer X.deinit(gpa);
    var out: std.ArrayList(i64) = .empty;
    defer out.deinit(gpa);
    var n_req: usize = 0;
    var n_res: usize = 0;
    var modes: [4]usize = @splat(0);
    while (lines.next()) |line| {
        var k: Tok = .{ .it = std.mem.tokenizeScalar(u8, line, ' ') };
        const tag = k.s();
        if (std.mem.eql(u8, tag, "s")) {
            _ = k.s();
            for (&cfg) |*x| x.* = k.i();
            if (m) |*x| x.deinit();
            m = try Meta.init(gpa, cfg[0], cfg[1]);
            gen = 0;
            continue;
        }
        const mm = &m.?;
        const resume_ = k.i() != 0;
        const max_out = k.i();
        try k.list(&X);
        try k.list(&out);
        const kind = k.s();
        const before = mm.digest();
        var p = tree.makePlan(gpa, mm, X.items, max_out, cfg[2], cfg[3], cfg[4], resume_) catch |e| {
            try t.expectEqualStrings("E", kind);
            try t.expect(e == error.RequestTooLong);
            try t.expectEqual(hex16(k.s()), before);
            try t.expectEqual(before, mm.digest());
            continue;
        };
        defer p.deinit();
        errdefer std.debug.print("golden line: {s}\n", .{line[0..@min(line.len, 120)]});
        try t.expectEqualStrings("P", kind);
        try t.expectEqual(before, mm.digest());
        try t.expectEqual(hex16(k.s()), p.d0);
        try t.expectEqual(hex16(k.s()), tree.planDigest(&p));
        try t.expectEqual(k.i(), p.c);
        try t.expectEqual(k.i(), p.h);
        try t.expectEqual(k.i(), p.t_leaf);
        try t.expectEqualStrings(k.s(), @tagName(p.mode));
        try t.expectEqual(k.i(), p.evicted);
        try t.expectEqual(k.i(), @as(i64, @intCast(p.ops.len)));
        try t.expectEqual(k.i(), @as(i64, @intCast(p.free_after.len)));
        try t.expectEqual(hex16(k.s()), p.post.digest());
        try p.post.check();
        try t.expectEqual(p.c, p.post.nAct()); // the compute path ends at c / 計算パスは c で終わる
        std.mem.swap(Meta, mm, &p.post);
        gen += 1;
        const seq = try std.mem.concat(gpa, i64, &.{ X.items, out.items[0 .. out.items.len - 1] });
        defer gpa.free(seq);
        const freed = try tree.commitSegment(mm, gpa, p.c, seq[@intCast(p.c)..], gen);
        defer gpa.free(freed);
        try t.expectEqual(k.i(), @as(i64, @intCast(freed.len)));
        try t.expectEqual(hex16(k.s()), mm.digest());
        try mm.check();
        n_req += 1;
        n_res += @intFromBool(p.c > 0);
        modes[@intFromEnum(p.mode)] += 1;
    }
    try t.expect(n_req > 400);
    try t.expect(n_res > 50);
    for (modes) |x| try t.expect(x > 10);
}

// ----------------------------------------------------------------------------------------- the Zig walk ---

/// Every op's identity arrays (PathIdents over post) equal pathIdent of its own node up to the node's end.
/// 各 op の identity 配列 (post 上の PathIdents) は、そのノードの末尾まで、ノード自身の pathIdent と等しい。
fn checkIdents(p: *const tree.Plan) !void {
    var pi = try tree.PathIdents.initPlan(gpa, p);
    defer pi.deinit();
    try t.expect(pi.paths.items.len <= 3);
    for (p.ops) |o| for ([_]?i64{ o.nid_in, o.nid_out }) |x| if (x) |nid| {
        const got = try pi.get(nid);
        const want = try tree.pathIdent(gpa, &p.post, nid);
        defer gpa.free(want.pid);
        defer gpa.free(want.gen);
        try t.expect(got.pid.len >= want.pid.len);
        try t.expectEqualSlices(u64, want.pid, got.pid[0..want.pid.len]);
        try t.expectEqualSlices(i64, want.gen, got.gen[0..want.gen.len]);
        try t.expect(o.b <= @as(i64, @intCast(want.pid.len)));
    };
}

fn randToks(al: std.mem.Allocator, r: std.Random, list: *std.ArrayList(i64), n: usize, lo: i64) !void {
    for (0..n) |_| try list.append(al, lo + @as(i64, @intCast(r.uintLessThan(u64, 50_000))));
}

test "random conversations never break an invariant (S 0/32/160/384, dcp 1/2/4)" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var resumed: usize = 0;
    var swaps: usize = 0;
    for ([_]i64{ 0, 32, 160, 384 }) |S| for ([_]i64{ 1, 2, 4 }) |dcp| for (0..3) |seed| {
        var prng: std.Random.DefaultPrng = .init(@as(u64, @intCast(S * 131 + dcp * 7)) + seed);
        const r = prng.random();
        const cap: i64 = 320;
        const tail: i64 = ([_]i64{ 1, 5, 17 })[r.uintLessThan(usize, 3)];
        var m = try Meta.init(gpa, S, dcp);
        defer m.deinit();
        var roots: [2]std.ArrayList(i64) = .{ .empty, .empty };
        for (&roots, 0..) |*x, i| try randToks(ar, r, x, r.intRangeLessThan(usize, 20, 60), 1000 + 100_000 * @as(i64, @intCast(i)));
        var convs: std.ArrayList([]const i64) = .empty;
        var gen: i64 = 0;
        for (0..40) |_| {
            var X: std.ArrayList(i64) = .empty;
            const kind = r.float(f64);
            if (convs.items.len == 0 or kind < 0.25) {
                if (r.float(f64) < 0.8) try X.appendSlice(ar, roots[r.uintLessThan(usize, 2)].items);
                for (0..r.intRangeLessThan(usize, 5, 80)) |_| try X.append(ar, 1000 + @as(i64, @intCast(r.uintLessThan(u64, 50_000))));
            } else if (kind < 0.75) {
                try X.appendSlice(ar, convs.items[r.uintLessThan(usize, convs.items.len)]);
                for (0..r.intRangeLessThan(usize, 1, 40)) |_| try X.append(ar, 1000 + @as(i64, @intCast(r.uintLessThan(u64, 50_000))));
            } else if (kind < 0.9) {
                const base = convs.items[r.uintLessThan(usize, convs.items.len)];
                try X.appendSlice(ar, base[0..r.intRangeLessThan(usize, 1, base.len + 1)]);
                for (0..r.intRangeLessThan(usize, 1, 30)) |_| try X.append(ar, 1000 + @as(i64, @intCast(r.uintLessThan(u64, 50_000))));
            } else {
                try X.appendSlice(ar, convs.items[r.uintLessThan(usize, convs.items.len)]);
            }
            if (X.items.len > 280) X.shrinkRetainingCapacity(280); // cap - 40
            if (X.items.len < 2) continue;
            var out: std.ArrayList(i64) = .empty;
            for (0..r.intRangeLessThan(usize, 1, 20)) |_| try out.append(ar, 1000 + @as(i64, @intCast(r.uintLessThan(u64, 50_000))));
            const before = m.digest();
            var p = try tree.makePlan(gpa, &m, X.items, @intCast(out.items.len), cap, tail, 16, r.float(f64) < 0.92);
            defer p.deinit();
            try t.expectEqual(before, m.digest()); // the input never changes / 入力は変わらない
            try t.expectEqual(before, p.d0);
            try p.post.check(); // the tree's invariants / 木の不変条件
            try t.expectEqual(p.c, p.post.nAct()); // the compute path ends at c / 計算パスは c で終わる
            try t.expect(p.h <= p.c or p.c == 0);
            if (S == 0) try t.expectEqual(@as(usize, 0), p.ops.len);
            try checkIdents(&p);
            std.mem.swap(Meta, &m, &p.post);
            gen += 1;
            const seq = try std.mem.concat(ar, i64, &.{ X.items, out.items[0 .. out.items.len - 1] });
            gpa.free(try tree.commitSegment(&m, gpa, p.c, seq[@intCast(p.c)..], gen));
            try m.check();
            resumed += @intFromBool(p.c > 0);
            swaps += @intFromBool(p.mode == .swap);
            try convs.append(ar, try std.mem.concat(ar, i64, &.{ X.items, out.items }));
        }
    };
    try t.expect(resumed > 100);
}

test "a plan over a conversation of many turns computes one identity a path" {
    var m = try Meta.init(gpa, 4096, 2);
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(77);
    const r = prng.random();
    var A: std.ArrayList(i64) = .empty;
    defer A.deinit(gpa);
    var gen: i64 = 0;
    for (0..120) |_| { // 120 turns of 15 tokens, each continuing the last / 15 トークンを 120 ターン、前に続ける
        const c0 = A.items.len;
        try randToks(gpa, r, &A, 14, 1000);
        var p = try tree.makePlan(gpa, &m, A.items, 2, 4096, 1, 4, true);
        defer p.deinit();
        std.mem.swap(Meta, &m, &p.post);
        gen += 1;
        try A.append(gpa, 7); // out = [7, x]: 7 joins the record / out = [7, x]: 7 が記録に加わる
        gpa.free(try tree.commitSegment(&m, gpa, p.c, A.items[@intCast(p.c)..], gen));
        _ = c0;
    }
    var B: std.ArrayList(i64) = .empty;
    defer B.deinit(gpa);
    try randToks(gpa, r, &B, 40, 900_000);
    {
        var p = try tree.makePlan(gpa, &m, B.items, 2, 4096, 1, 4, true);
        defer p.deinit();
        std.mem.swap(Meta, &m, &p.post);
        gen += 1;
        gpa.free(try tree.commitSegment(&m, gpa, p.c, B.items[@intCast(p.c)..], gen));
    }
    try A.append(gpa, 99);
    var p = try tree.makePlan(gpa, &m, A.items, 2, 4096, 1, 4, true);
    defer p.deinit();
    var restores: usize = 0;
    for (p.ops) |o| restores += @intFromBool(o.kind == .restore);
    try t.expect(restores > 100);
    var pi = try tree.PathIdents.initPlan(gpa, &p);
    defer pi.deinit();
    try t.expect(pi.paths.items.len <= 2);
    try checkIdents(&p);
}

// ---------------------------------------------------------------------------------------- allocation failures ---

/// One request as the engine runs it (plan, adopt, commit X ++ out[:-1] from c).
/// エンジンと同じ 1 要求 (プランを作り、採用し、X ++ out[:-1] を c から commit する)。
fn serve(m: *Meta, X: []const i64, out: []const i64, gen: i64, cap: i64, tail: i64, mn: i64) !void {
    var p = try tree.makePlan(gpa, m, X, @intCast(out.len + 2), cap, tail, mn, true);
    defer p.deinit();
    std.mem.swap(Meta, m, &p.post);
    const seq = try std.mem.concat(gpa, i64, &.{ X, out[0 .. out.len - 1] });
    defer gpa.free(seq);
    gpa.free(try tree.commitSegment(m, gpa, p.c, seq[@intCast(p.c)..], gen));
}

fn planCommit(al: std.mem.Allocator, m: *const Meta, X: []const i64, cap: i64, tail: i64, mn: i64) !void {
    const d = m.digest();
    var p = try tree.makePlan(al, m, X, 3, cap, tail, mn, true);
    defer p.deinit();
    var pi = try tree.PathIdents.initPlan(al, &p);
    defer pi.deinit();
    var post = try p.post.clone(al);
    defer post.deinit();
    al.free(try tree.commitSegment(&post, al, p.c, X[@intCast(p.c)..], 99));
    try t.expectEqual(d, m.digest());
}

test "every allocation failure in a plan, its identities and a commit leaks nothing and leaves the input alone" {
    var m = try Meta.init(gpa, 40, 2);
    defer m.deinit();
    try serve(&m, &trA, &.{ 7, 8 }, 1, 64, 2, 3);
    try serve(&m, &trB, &.{ 9, 9 }, 2, 64, 2, 3);
    try t.checkAllAllocationFailures(gpa, planCommit, .{ &m, &comptime concat(&trA, &.{ 7, 8, 50 }), 64, 2, 3 }); // swap
    var e = try Meta.init(gpa, 40, 1);
    defer e.deinit();
    inline for (1..4) |i| try serve(&e, &comptime range(1000 * i, 1000 * i + 15), &.{5}, i, 64, 2, 3);
    try t.checkAllAllocationFailures(gpa, planCommit, .{ &e, &comptime range(4000, 4015), 64, 2, 3 }); // evict + plain
    var dr = try Meta.init(gpa, 8, 2);
    defer dr.deinit();
    try serve(&dr, &trA, &.{ 7, 8 }, 1, 64, 2, 3);
    try t.checkAllAllocationFailures(gpa, planCommit, .{ &dr, &trB, 64, 2, 3 }); // drop
}
