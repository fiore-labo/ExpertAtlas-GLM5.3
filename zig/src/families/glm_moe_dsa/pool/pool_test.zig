//! Tests of the KV pool's host side (glm53/POOL.md P2): worked examples with hand-written expected states, the
//! traces of the separate Python reference (testdata/*.jsonl, made by glm53/tools/pool/pool_ref.py), and a random
//! harness-like walk checking the invariants after every step. `zig build test-glm53-pool`.
//! KV pool のホスト側のテスト (glm53/POOL.md の P2): 期待する状態を手で書いた例、別実装の Python 版の trace
//! (testdata/*.jsonl、glm53/tools/pool/pool_ref.py が作る)、要求ごとに不変条件を確かめるハーネス型のランダムウォーク。
//! `zig build test-glm53-pool`。
const std = @import("std");
const plan = @import("plan.zig");

test {
    _ = @import("devmap.zig");
}

const Meta = plan.Meta;
const t = std.testing;
const gpa = std.testing.allocator;

fn toks(comptime n: usize, base: i64) [n]i64 {
    var out: [n]i64 = undefined;
    for (&out, 0..) |*x, i| x.* = base + @as(i64, @intCast(i));
    return out;
}

/// plan, apply, check; the caller frees the plan.
/// plan、apply、check。プランは呼び出し側が解放する。
fn step(m: *Meta, prompt: []const i64, max_tokens: i64) !plan.Plan {
    var p = try plan.makePlan(m, prompt, max_tokens, true);
    errdefer p.deinit(gpa);
    try plan.apply(m, &p, prompt);
    try m.check(p.X);
    for (p.table[p.resume_end / plan.block ..]) |b| try t.expectEqual(@as(u32, 1), m.ref[b]);
    return p;
}

/// commit prompt ++ out[: -1] with kv_end = V.
fn finish(m: *Meta, p: *const plan.Plan, prompt: []const i64, out: []const i64) !void {
    const seq = try std.mem.concat(gpa, i64, &.{ prompt, out[0 .. out.len - @min(out.len, 1)] });
    defer gpa.free(seq);
    try plan.commit(m, p, seq, prompt.len, seq.len);
    try m.check(null);
}

test "startup lower bound: ceil((capacity + 8) / 256) + 1 blocks" {
    try t.expectError(error.PoolTooSmall, (plan.Config{ .n_blocks = 4, .capacity = 768, .max_slots = 4, .mtp = false }).check());
    try (plan.Config{ .n_blocks = 5, .capacity = 768, .max_slots = 4, .mtp = false }).check();
    // 1M: 4,098 <= 7,488 (POOL.md 4.4)
    try (plan.Config{ .n_blocks = 7488, .capacity = 1048576, .max_slots = 64, .mtp = false }).check();
    try t.expectEqual(@as(u64, 4098), plan.ceilDiv(1048576 + 8, 256) + 1);
}

test "NEW, then CONTINUE at the previous prompt end, then a resend resumes n - 17" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(600, 1000);
    var p1 = try step(&m, &a, 10);
    defer p1.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p1.action);
    try t.expectEqual(@as(u64, 0), p1.resume_end);
    try t.expectEqual(@as(u64, 618), p1.write_hi); // 600 + 10 + 8
    try t.expectEqual(@as(usize, 3), p1.table.len);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2 }, p1.alloc);
    const o1 = toks(10, 5000);
    try finish(&m, &p1, &a, &o1);
    try t.expectEqual(@as(u64, 609), m.slots.items[0].validEnd());
    // the next turn sends the output back without its reasoning: match = the previous prompt end
    // 次のターンは出力を思考抜きで送り返す: 一致 = 前回の prompt の終わり
    var b: [700]i64 = undefined;
    @memcpy(b[0..600], &a);
    for (b[600..], 0..) |*x, i| x.* = 9000 + @as(i64, @intCast(i));
    var p2 = try step(&m, &b, 10);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p2.action);
    try t.expectEqual(@as(u64, 600), p2.h);
    try t.expectEqual(@as(u64, 600), p2.c);
    try t.expectEqual(@as(u64, 600), p2.resume_end);
    try t.expectEqual(@as(u64, 0), p2.X);
    try finish(&m, &p2, &b, &toks(5, 7000));
    // resend: c = n - 17 = 683 >= prompt_end - 17 = 683
    var p3 = try step(&m, &b, 10);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 683), p3.resume_end);
}

test "G5: h = n = 260 against prompt_end 1000 is NEW (c = 243 would fall through in v3); the 260 slot is continued after it" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{42});
    var p2 = try step(&m, a[0..260], 1);
    defer p2.deinit(gpa);
    // c = min(260, 260 - 17, 1000) = 243 < 1000 - 17 and < 256: NEW
    try t.expectEqual(@as(u64, 243), p2.c);
    try t.expectEqual(plan.Action.NEW, p2.action);
    try t.expectEqual(@as(u64, 0), p2.resume_end);
    try finish(&m, &p2, a[0..260], &.{43});
    var p3 = try step(&m, a[0..300], 1);
    defer p3.deinit(gpa);
    // the 1000 slot: c = min(300, 283, 1000) = 283, not continuable: a SHARE resuming at 256; the 260 slot: c =
    // min(260, 283, 260) = 260, continuable (260 >= 243), its block 1 private: it resumes at 260 -> slot 1, CONTINUE
    // (v3.5 compared c before rounding and SHAREd from slot 0 at 256)
    // 1000 のスロット: c = 283 だが続けられない (256 からの SHARE)。260 のスロット: c = 260 で続けられ、ブロック 1 は
    // 専有: 260 から再開 -> スロット 1 を CONTINUE (v3.5 は切り下げ前の c で比べ、スロット 0 から 256 で SHARE していた)
    try t.expectEqual(@as(u64, 260), p3.c);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(?u64, 1), p3.S);
    try t.expectEqual(@as(u64, 260), p3.resume_end);
}

test "SHARE shares the full blocks below c; the branch block is recomputed from its start" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    var sys: [1100]i64 = undefined;
    for (&sys, 0..) |*x, i| x.* = @intCast(i);
    var p1 = try step(&m, &sys, 4);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &sys, &.{ 1, 2, 3, 4 });
    var sub: [1200]i64 = undefined;
    @memcpy(sub[0..1000], sys[0..1000]);
    for (sub[1000..], 0..) |*x, i| x.* = 50000 + @as(i64, @intCast(i));
    var p2 = try step(&m, &sub, 4);
    defer p2.deinit(gpa);
    // h = 1000, c = 1000 < prompt_end 1100 - 17: SHARE at 768 (three blocks)
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqual(@as(u64, 768), p2.resume_end);
    try t.expectEqual(@as(u64, 1), p2.X);
    try t.expectEqualSlices(u32, p1.table[0..3], p2.table[0..3]);
    for (p2.table[0..3]) |b| try t.expectEqual(@as(u32, 2), m.ref[b]);
    try finish(&m, &p2, &sub, &.{ 5, 6 });
}

test "POOL.md 4.3: a shared block mid-resume is rounded down; the other slot keeps it" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{9});
    // B branches from A at 900 (c = 883 < 1000 - 17): SHARE of A's first three blocks, resume 768
    var bb: [950]i64 = undefined;
    @memcpy(bb[0..900], a[0..900]);
    for (bb[900..], 0..) |*x, i| x.* = 7000 + @as(i64, @intCast(i));
    var p2 = try step(&m, &bb, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqual(@as(u64, 768), p2.resume_end);
    try finish(&m, &p2, &bb, &.{9});
    // eviction cuts A to valid_end = prompt_end = 768 (the state of the POOL.md example)
    try plan.shrinkSlot(&m, 0, 768);
    plan.recount(&m);
    try m.check(null);
    try t.expectEqual(@as(u64, 768), m.slots.items[0].prompt_end);
    // the 768 prompt again: A has c = 751 >= 751: CONTINUE; block 2 is shared and 751 is mid-block: resume_end
    // 512, A gets a private block 2, B keeps the old one
    var p3 = try step(&m, a[0..768], 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 751), p3.c);
    try t.expectEqual(@as(u64, 512), p3.resume_end);
    try t.expectEqual(@as(u64, 0), p3.X);
    try t.expect(p3.table[2] != p1.table[2]);
    const B = &m.slots.items[m.find(1).?];
    try t.expectEqual(p1.table[2], B.blocks.items[2]);
    try t.expectEqual(@as(u32, 1), m.ref[p1.table[2]]);
    try t.expectEqual(@as(u64, 950), B.validEnd());
}

test "eviction from the tail of the oldest slot; SHARE's source may go in the same plan" {
    // 6 blocks of 256, capacity 1024: lower bound ceil(1032/256)+1 = 6
    var m = try Meta.init(gpa, .{ .n_blocks = 6, .capacity = 1024, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(600, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{9}); // A: 3 blocks (600 rows)
    var bb: [900]i64 = undefined;
    @memcpy(bb[0..600], &a);
    for (bb[600..], 0..) |*x, i| x.* = 3000 + @as(i64, @intCast(i));
    // B shares A's first 2 blocks (c = 600 < 583? no: c = min(600, 883, 600) = 600 >= 583: CONTINUE)
    var p2 = try step(&m, &bb, 100);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p2.action);
    try finish(&m, &p2, &bb, &.{9});
    // C unrelated, 1000 tokens: needs 4 blocks; A is the only other slot and holds 4 (999): all of it goes
    const cc = toks(1000, 90000);
    var p3 = try step(&m, &cc, 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p3.action);
    try t.expectEqual(@as(usize, 4), p3.table.len);
}

test "MTP F: the worked examples of POOL.md 4.6.1" {
    // n = 256, max_tokens 2: one room-0 window -> F 255
    var f = plan.MtpF.begin(0, 256);
    f.chunk(0, 256);
    try t.expectEqual(@as(u64, 255), f.F);
    f.propose(256, 1, 0);
    try t.expectEqual(@as(u64, 255), f.kvEnd(257));
    // n = 256, max_tokens 10, K = 7 all taken, last window room 0: F 256
    f = plan.MtpF.begin(0, 256);
    f.chunk(0, 256);
    f.propose(256, 1, 7);
    f.chain(256, 6);
    try t.expectEqual(@as(u64, 256), f.F);
    f.propose(264, 8, 0);
    try t.expectEqual(@as(u64, 256), f.kvEnd(265));
    // n = 1, max_tokens 5, K = 1 all taken: F 0 from the first propose to the end (S1)
    f = plan.MtpF.begin(0, 1);
    f.chunk(0, 1);
    try t.expectEqual(@as(u64, 0), f.F);
    f.propose(1, 1, 1);
    try t.expectEqual(@as(u64, 0), f.F);
    f.chain(1, 1);
    f.propose(3, 2, 1);
    try t.expectEqual(@as(u64, 0), f.kvEnd(5));
    // a chunked prompt resumed at 512: F follows the chunks that join it
    f = plan.MtpF.begin(512, 5000);
    f.chunk(512, 2560);
    f.chunk(2560, 4608);
    try t.expectEqual(@as(u64, 4608), f.F);
    f.chunk(4608, 5000);
    try t.expectEqual(@as(u64, 4999), f.F);
}

test "DFlash2 ranges: the 20,000 -> 18,944 example is saved, s 14,952" {
    const live = plan.Ring.live(20000, 20000, 0, 8);
    try t.expectEqual(@as(u64, 15912), live.keep_lo);
    const snap = plan.Ring.snap(17000, 0);
    try t.expectEqual(@as(u64, 12904), snap.keep_lo);
    const lv = live.shrink(18944).?;
    const r = plan.resumeAt(16983, 17000, .{ lv.vlo, lv.vhi }, .{ snap.vlo, snap.vhi }, 17000, 2047);
    try t.expectEqual(plan.ResumeKind.saved, r.kind);
    try t.expectEqual(@as(u64, 14952), r.s);
    try t.expectEqual(@as(u64, 16983), r.pos);
    try t.expectEqual([2]u64{ 14952, 15912 }, r.copy.?);
    // EOS mid-window: C 20,008, V 20,001
    const e = plan.Ring.live(20008, 20001, 0, 8);
    try t.expectEqual(@as(u64, 15920), e.keep_lo);
    try t.expectEqual(@as(u64, 20001), e.vhi);
    // cold, empty D, a one-row hole, touching ranges
    try t.expectEqual(plan.ResumeKind.none, plan.resumeAt(0, 100, null, null, 0, 2047).kind);
    try t.expectEqual(plan.ResumeKind.none, plan.resumeAt(100, 3000, null, null, 100, 2047).kind);
    const hole = plan.resumeAt(5000, 5017, .{ 4000, 5017 }, .{ 2000, 3968 }, 5017, 2047);
    try t.expectEqual(plan.ResumeKind.partial, hole.kind);
    try t.expectEqual(@as(u64, 5000), hole.s);
    const touch = plan.resumeAt(5000, 5017, .{ 3970, 5017 }, .{ 2000, 3970 }, 5017, 2047);
    try t.expectEqual(plan.ResumeKind.saved, touch.kind);
    try t.expectEqual(plan.ResumeKind.live, plan.resumeAt(5000, 5017, .{ 2969, 5017 }, null, 5017, 2047).kind);
}

// ---------------------------------------------------------------------------------- random harness walk ---

test "random harness-like conversations never break an invariant" {
    var prng = std.Random.DefaultPrng.init(0x900d);
    const rnd = prng.random();
    for ([_]u32{ 9, 12, 20, 40 }) |nb| for ([_]bool{ false, true }) |mtp| {
        var m = try Meta.init(gpa, .{ .n_blocks = nb, .capacity = 1792, .max_slots = 6, .mtp = mtp });
        defer m.deinit();
        var sys: [600]i64 = undefined;
        for (&sys, 0..) |*x, i| x.* = @intCast(i);
        var convs: [4]std.ArrayList(i64) = @splat(.empty);
        defer for (&convs) |*c| c.deinit(gpa);
        var stats: [3]usize = @splat(0);
        for (0..400) |_| {
            const ci = rnd.uintLessThan(usize, 4);
            var c = &convs[ci];
            if (c.items.len == 0 or rnd.uintLessThan(u32, 10) == 0) {
                c.clearRetainingCapacity();
                const ls = if (rnd.boolean()) sys.len else rnd.uintLessThan(usize, 40);
                try c.appendSlice(gpa, sys[0..ls]);
            }
            const grow = rnd.uintLessThan(usize, 200) + 1;
            for (0..grow) |_| try c.append(gpa, rnd.intRangeAtMost(i64, 1, 50000));
            if (c.items.len >= 1600) {
                c.shrinkRetainingCapacity(rnd.uintLessThan(usize, 600) + 1);
            }
            const mt = rnd.intRangeAtMost(i64, 1, 64);
            var p = try step(&m, c.items, mt);
            defer p.deinit(gpa);
            stats[@intFromEnum(p.action)] += 1;
            const want = plan.wantTokens(1792, c.items.len, mt);
            const nout = rnd.uintLessThan(usize, want) + 1;
            const out = try gpa.alloc(i64, nout);
            defer gpa.free(out);
            for (out) |*x| x.* = rnd.intRangeAtMost(i64, 1, 50000);
            const seq = try std.mem.concat(gpa, i64, &.{ c.items, out[0 .. nout - 1] });
            defer gpa.free(seq);
            const kv = if (mtp) seq.len - @min(seq.len - p.resume_end, rnd.uintLessThan(usize, 9)) else seq.len;
            try plan.commit(&m, &p, seq, c.items.len, kv);
            try m.check(null);
            // the conversation continues with its output, its reasoning dropped (a few tokens)
            // 会話は出力を付けて続く (思考の分の数トークンを落とす)
            const keep = nout - @min(nout, rnd.uintLessThan(usize, 4));
            try c.appendSlice(gpa, out[0..keep]);
        }
        try t.expect(stats[0] > 20 and stats[2] > 5);
    };
}

test "DFlash2 saves: DF A, DF B, serial A, DF A restores A's own rings; eviction and LRU free saves" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 2 });
    defer m.deinit();
    // DF A: 1000-token prompt, 50 out
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 50);
    defer p1.deinit(gpa);
    try t.expect(p1.restore == null);
    const o1 = toks(50, 5000);
    try finish(&m, &p1, &a, &o1);
    const ka = (try plan.commitSave(&m, &p1, plan.Ring.live(1049, 1049, 0, 8), plan.Ring.snap(1000, 0), true)).?;
    try t.expectEqual([2]u64{ ka, 1 }, m.ring.?);
    // DF B: unrelated
    const b = toks(700, 30000);
    var p2 = try step(&m, &b, 10);
    defer p2.deinit(gpa);
    try t.expect(p2.restore == null);
    try finish(&m, &p2, &b, &toks(10, 6000));
    const kb = (try plan.commitSave(&m, &p2, plan.Ring.live(709, 709, 0, 8), plan.Ring.snap(700, 0), true)).?;
    try t.expect(ka != kb);
    // serial A (continues A): restores nothing on the drafter's side, saves nothing; A's save is cut at resume_end
    var a2: [1100]i64 = undefined;
    @memcpy(a2[0..1049], a[0..1000] ++ o1[0..49]);
    for (a2[1049..], 0..) |*x, i| x.* = 8000 + @as(i64, @intCast(i));
    var p3 = try step(&m, &a2, 5);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 1049), p3.resume_end);
    try t.expectEqual(ka, p3.restore.?);
    try t.expect(!p3.skip_copy); // the ring holds B / リングは B のもの
    try finish(&m, &p3, &a2, &toks(5, 9000));
    try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p3, null, null, false));
    try t.expectEqual([2]u64{ kb, 1 }, m.ring.?); // the serial request never touched the ring
    try t.expectEqual(@as(u64, 1049), m.saves[ka].live.?.vhi);
    // DF A again: restores A's save (not B's ring)
    var a3: [1200]i64 = undefined;
    @memcpy(a3[0..1100], &a2);
    for (a3[1100..], 0..) |*x, i| x.* = 11000 + @as(i64, @intCast(i));
    var p4 = try step(&m, &a3, 5);
    defer p4.deinit(gpa);
    try t.expectEqual(ka, p4.restore.?);
    try t.expectEqual(@as(u64, 1049), p4.restore_live.?.vhi);
    const rs = plan.resumeAt(p4.resume_end, 1200, .{ p4.restore_live.?.vlo, p4.restore_live.?.vhi }, .{ p4.restore_snap.?.vlo, p4.restore_snap.?.vhi }, p4.below, 2047);
    // needed [1200 - 2048 -> 0, 1104): live [0, 1049) stops short of r: partial
    try t.expectEqual(plan.ResumeKind.partial, rs.kind);
    try finish(&m, &p4, &a3, &toks(5, 12000));
    _ = try plan.commitSave(&m, &p4, plan.Ring.live(1204, 1204, rs.s, 8), plan.Ring.snap(1200, rs.s), true);
    try t.expectEqual(@as(u64, 1100), p4.resume_end);
    try t.expectEqual(@as(u64, 1100), m.saves[ka].live.?.vlo); // s = r after a partial
    // a third slot's save takes the least recently stored one (B's)
    const c = toks(400, 70000);
    var p5 = try step(&m, &c, 3);
    defer p5.deinit(gpa);
    try finish(&m, &p5, &c, &toks(3, 1));
    try t.expectEqual(kb, (try plan.commitSave(&m, &p5, plan.Ring.live(402, 402, 0, 8), null, true)).?);
    try m.check(null);
}

test "DFlash2: SHARE restores the source's save into the new slot's request; evicting the source frees it" {
    var m = try Meta.init(gpa, .{ .n_blocks = 9, .capacity = 1792, .max_slots = 8, .mtp = false, .draft_saves = 4 });
    defer m.deinit();
    const a = toks(1700, 1);
    var p1 = try step(&m, &a, 2);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{ 7, 8 });
    _ = try plan.commitSave(&m, &p1, plan.Ring.live(1701, 1701, 0, 8), plan.Ring.snap(1700, 0), true);
    // a subagent: shares A's first 4 blocks; the pool (9 blocks) forces A out in the same plan
    var sub: [1500]i64 = undefined;
    @memcpy(sub[0..1100], a[0..1100]);
    for (sub[1100..], 0..) |*x, i| x.* = 40000 + @as(i64, @intCast(i));
    var p2 = try step(&m, &sub, 200);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqual(@as(u64, 1024), p2.resume_end);
    try t.expectEqual(@as(usize, 0), p2.restore.?);
    try t.expect(p2.skip_copy);
    try t.expectEqual(@as(u64, 1100), p2.below);
    try t.expectEqual(@as(usize, 1), p2.shrunk.len);
    try t.expectEqual([2]u64{ 0, 1536 }, p2.shrunk[0]);
    try t.expectEqual(@as(u64, 1536), m.saves[0].live.?.vhi); // cut with its slot / スロットと一緒に切る
    try finish(&m, &p2, &sub, &toks(3, 1));
}

// ------------------------------------------------------------------------------ more P2 scenarios (POOL.md 9) ---
// Expected values below were written by hand from POOL.md before the tests ran.
// 以下の期待値は、試験を走らせる前に POOL.md から手で書いた。

/// Tokens base, base + 1, ... into `out`.
/// base, base + 1, ... を `out` へ。
fn fill(out: []i64, base: i64) void {
    for (out, 0..) |*x, i| x.* = base + @as(i64, @intCast(i));
}

/// The slot with id `id` (it must exist).
/// id が `id` のスロット (あること)。
fn slotOf(m: *Meta, id: u64) *plan.Slot {
    return &m.slots.items[m.find(id).?];
}

/// commit with an explicit prompt_end and kv_end, then check.
/// prompt_end と kv_end を明示して確定し、check。
fn commitKv(m: *Meta, p: *const plan.Plan, seq: []const i64, prompt_end: u64, kv_end: u64) !void {
    try plan.commit(m, p, seq, prompt_end, kv_end);
    try m.check(null);
}

/// plan (with resume_ok), apply, check; the caller frees the plan.
/// plan (resume_ok を指定)、apply、check。プランは呼び出し側が解放する。
fn stepOpt(m: *Meta, prompt: []const i64, max_tokens: i64, resume_ok: bool) !plan.Plan {
    var p = try plan.makePlan(m, prompt, max_tokens, resume_ok);
    errdefer p.deinit(gpa);
    try plan.apply(m, &p, prompt);
    try m.check(p.X);
    for (p.table[p.resume_end / plan.block ..]) |b| try t.expectEqual(@as(u32, 1), m.ref[b]);
    return p;
}

fn expectFree(m: *const Meta, want: []const u32) !void {
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(gpa);
    for (m.ref, 0..) |r, b| if (r == 0) try got.append(gpa, @intCast(b));
    try t.expectEqualSlices(u32, want, got.items);
}

fn expectShrunk(got: []const [2]u64, want: []const [2]u64) !void {
    try t.expectEqual(want.len, got.len);
    for (want, got) |w, g| try t.expectEqual(w, g);
}

/// A slot's valid_end, kv_end, prompt_end, tick and table.
/// スロットの valid_end、kv_end、prompt_end、tick、表。
fn expectSlot(m: *Meta, id: u64, valid_end: u64, kv_end: u64, prompt_end: u64, tick: u64, table: []const u32) !void {
    const i = m.find(id) orelse return error.TestUnexpectedResult;
    const s = &m.slots.items[i];
    try t.expectEqual(valid_end, s.validEnd());
    try t.expectEqual(kv_end, s.kv_end);
    try t.expectEqual(prompt_end, s.prompt_end);
    try t.expectEqual(tick, s.tick);
    try t.expectEqualSlices(u32, table, s.blocks.items);
}

/// A copy of a slot and of its blocks' generations: a plan or a commit on another slot must leave it as it was.
/// スロットとそのブロックの世代の写し: 別のスロットのプランや確定はこれを変えてはいけない。
const Frozen = struct {
    id: u64,
    seq: []i64,
    blocks: []u32,
    gens: []u64,
    kv_end: u64,
    prompt_end: u64,
    tick: u64,

    fn of(m: *Meta, id: u64) !Frozen {
        const s = slotOf(m, id);
        const seq = try gpa.dupe(i64, s.seq.items);
        errdefer gpa.free(seq);
        const blocks = try gpa.dupe(u32, s.blocks.items);
        errdefer gpa.free(blocks);
        const gens = try gpa.alloc(u64, blocks.len);
        for (gens, blocks) |*g, b| g.* = m.gen[b];
        return .{ .id = id, .seq = seq, .blocks = blocks, .gens = gens, .kv_end = s.kv_end, .prompt_end = s.prompt_end, .tick = s.tick };
    }

    fn deinit(f: *Frozen) void {
        gpa.free(f.seq);
        gpa.free(f.blocks);
        gpa.free(f.gens);
    }

    fn expectSame(f: *const Frozen, m: *Meta) !void {
        const i = m.find(f.id) orelse return error.TestUnexpectedResult;
        const s = &m.slots.items[i];
        try t.expectEqualSlices(i64, f.seq, s.seq.items);
        try t.expectEqualSlices(u32, f.blocks, s.blocks.items);
        for (f.gens, f.blocks) |g, b| try t.expectEqual(g, m.gen[b]);
        try t.expectEqual(f.kv_end, s.kv_end);
        try t.expectEqual(f.prompt_end, s.prompt_end);
        try t.expectEqual(f.tick, s.tick);
    }
};

/// A (1,100 tokens, blocks 0..4) and B (a SHARE of A's blocks 0..3 at 1,024, blocks 0..3 + 5), then A's prompt_end
/// set to 300 by hand. The planner alone never makes this state (a share point always lies below its source's
/// prompt_end - 17, and prompt_end only falls with valid_end), but POOL.md 6.1 allows it (prompt_end <= valid_end)
/// and 4.2 says the CONTINUE rule is a policy that safety must not rest on: a CONTINUE inside the shared run must be
/// safe anyway.
/// A (1,100 token、ブロック 0..4) と B (A のブロック 0..3 を 1,024 で SHARE、ブロック 0..3 + 5)。続いて A の prompt_end を
/// 手で 300 にする。プランナだけではこの状態にならない (共有の位置は常に元の prompt_end - 17 より手前で、prompt_end は
/// valid_end と一緒にしか下がらない) が、POOL.md 6.1 は許し (prompt_end <= valid_end)、4.2 は CONTINUE の条件は方針で
/// 安全をそれに頼らないとする: 共有の並びの中での CONTINUE も安全でなければならない。
fn sharedPair(m: *Meta) !void {
    const a = toks(1100, 1);
    var p1 = try step(m, &a, 1);
    defer p1.deinit(gpa);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, p1.alloc);
    try finish(m, &p1, &a, &.{9});
    var b: [1250]i64 = undefined;
    @memcpy(b[0..1050], a[0..1050]);
    fill(b[1050..], 60000);
    var p2 = try step(m, &b, 1);
    defer p2.deinit(gpa);
    // c = min(1050, 1233, 1100) = 1050 < 1100 - 17: SHARE of 4 blocks
    // c = min(1050, 1233, 1100) = 1050 < 1100 - 17: 4 ブロックの SHARE
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqual(@as(u64, 1024), p2.resume_end);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 5 }, p2.table);
    try finish(m, &p2, &b, &.{9});
    try t.expectEqualSlices(u32, &.{ 2, 2, 2, 2, 1, 1 }, m.ref[0..6]);
    slotOf(m, 0).prompt_end = 300;
    try m.check(null);
}

test "P2: CONTINUE inside a shared block with more shared blocks after it: every shared block from j0 on is replaced" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    try sharedPair(&m);
    var fb = try Frozen.of(&m, 1);
    defer fb.deinit();
    const a = toks(1100, 1);
    var q: [450]i64 = undefined;
    @memcpy(q[0..400], a[0..400]);
    fill(q[400..], 70000);
    var p = try step(&m, &q, 600);
    defer p.deinit(gpa);
    // A: c = min(400, 433, 1100) = 400 >= 300 - 17 (continues); B: c = 400 too, does not continue -> A. 400 is inside
    // block 1, shared -> resume 256 (4.3 step 2). write_hi = 450 + 600 + 8 = 1058, 5 blocks: the shared blocks 1, 2, 3
    // are replaced (not only j0, 4.3 step 3), A's own block 4 is kept
    // A: c = 400 >= 300 - 17 (続けられる)。B も c = 400 だが続けられない -> A。400 は共有のブロック 1 の途中 -> 256 から。
    // write_hi = 1058、5 ブロック: 共有のブロック 1, 2, 3 を置き換え (j0 だけでない)、A の専有のブロック 4 は残す
    try t.expectEqual(plan.Action.CONTINUE, p.action);
    try t.expectEqual(@as(u64, 0), p.X);
    try t.expectEqual(@as(?u64, 0), p.S);
    try t.expectEqual(@as(u64, 400), p.h);
    try t.expectEqual(@as(u64, 400), p.c);
    try t.expectEqual(@as(u64, 256), p.resume_end);
    try t.expectEqual(@as(u64, 256), p.write_lo);
    try t.expectEqual(@as(u64, 1058), p.write_hi);
    try t.expectEqualSlices(u32, &.{ 0, 6, 7, 8, 4 }, p.table);
    try t.expectEqualSlices(u32, &.{ 6, 7, 8 }, p.alloc);
    try t.expectEqual(@as(u64, 3), p.released);
    try t.expectEqual(@as(usize, 0), p.evicted.len + p.shrunk.len);
    try t.expectEqualSlices(u32, &.{0}, p.verify);
    try expectSlot(&m, 0, 256, 256, 256, 1, &.{ 0, 6, 7, 8, 4 });
    try t.expectEqualSlices(i64, q[0..256], slotOf(&m, 0).seq.items);
    // B keeps its tokens, table, lengths and generations; its blocks 1..3 are its own now
    // B はトークン列・表・長さ・世代をそのまま持つ。ブロック 1..3 は B の専有になる
    try fb.expectSame(&m);
    try t.expectEqualSlices(u32, &.{ 2, 1, 1, 1, 1, 1, 1, 1, 1, 0 }, m.ref[0..10]);
    // commit V = 749: block 6 ([256, 512)) was written and is complete -> a new generation; 7 is not; 4 and 8 go back
    // 確定 V = 749: ブロック 6 は書いて完成 -> 世代を進める。7 は未完成。4 と 8 は空きに戻る
    const out = toks(300, 5000);
    try finish(&m, &p, &q, &out);
    try expectSlot(&m, 0, 749, 749, 450, 3, &.{ 0, 6, 7 });
    try t.expectEqual(@as(u64, 2), m.gen[0]);
    try t.expectEqual(@as(u64, 2), m.gen[6]);
    try t.expectEqual(@as(u64, 1), m.gen[7]);
    try t.expectEqual(@as(u32, 0), m.ref[4]);
    try t.expectEqual(@as(u32, 0), m.ref[8]);
    try fb.expectSame(&m);
}

test "P2: CONTINUE inside a shared block: the shared blocks past write_hi lose only X's reference" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    try sharedPair(&m);
    var fb = try Frozen.of(&m, 1);
    defer fb.deinit();
    const a = toks(1100, 1);
    var q: [450]i64 = undefined;
    @memcpy(q[0..400], a[0..400]);
    fill(q[400..], 70000);
    var p = try step(&m, &q, 1);
    defer p.deinit(gpa);
    // write_hi 459, 2 blocks: A drops 4 (freed), 3 and 2 (still B's), replaces the shared block 1; the lowest free
    // number is A's old 4
    // write_hi 459、2 ブロック: A は 4 (空く)、3 と 2 (B に残る) を外し、共有のブロック 1 を置き換える。一番小さい空きは
    // A の古い 4
    try t.expectEqual(plan.Action.CONTINUE, p.action);
    try t.expectEqual(@as(u64, 256), p.resume_end);
    try t.expectEqual(@as(u64, 459), p.write_hi);
    try t.expectEqualSlices(u32, &.{ 0, 4 }, p.table);
    try t.expectEqualSlices(u32, &.{4}, p.alloc);
    try t.expectEqual(@as(u64, 4), p.released);
    try t.expectEqualSlices(u32, &.{ 2, 1, 1, 1, 1, 1, 0 }, m.ref[0..7]);
    try t.expectEqual(@as(u64, 2), m.gen[4]); // A's allocation, now X's / A の割り当て、今度は X の
    try fb.expectSame(&m);
    try finish(&m, &p, &q, &.{1});
    try expectSlot(&m, 0, 450, 450, 450, 3, &.{ 0, 4 });
    try fb.expectSame(&m);
}

test "P2: CONTINUE at the start of a shared block with a shared block after it" {
    for ([_]i64{ 300, 1 }) |mt| {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
        defer m.deinit();
        try sharedPair(&m);
        var fb = try Frozen.of(&m, 1);
        defer fb.deinit();
        const a = toks(1100, 1);
        var q: [562]i64 = undefined;
        @memcpy(q[0..512], a[0..512]);
        fill(q[512..], 80000);
        var p = try step(&m, &q, mt);
        defer p.deinit(gpa);
        // A: c = min(512, 545, 1100) = 512 (continues); B: c 512 (does not). 512 is a block start: no rounding, j0 = 2
        // A: c = 512 (続けられる)、B: c 512 (続けられない)。512 はブロックの先頭: 切り下げなし、j0 = 2
        try t.expectEqual(plan.Action.CONTINUE, p.action);
        try t.expectEqual(@as(u64, 512), p.c);
        try t.expectEqual(@as(u64, 512), p.resume_end);
        try t.expectEqualSlices(u32, &.{ 0, 1 }, p.verify);
        if (mt == 300) {
            // write_hi = 562 + 300 + 8 = 870, 4 blocks: block 4 (A's) dropped and freed, the shared 2 and 3 replaced by
            // the lowest free numbers 4 and 6
            // write_hi = 870、4 ブロック: A のブロック 4 は外して空き、共有の 2 と 3 は一番小さい空きの 4 と 6 で置き換え
            try t.expectEqual(@as(u64, 870), p.write_hi);
            try t.expectEqualSlices(u32, &.{ 0, 1, 4, 6 }, p.table);
            try t.expectEqualSlices(u32, &.{ 4, 6 }, p.alloc);
            try t.expectEqual(@as(u64, 3), p.released);
            try t.expectEqualSlices(u32, &.{ 2, 2, 1, 1, 1, 1, 1, 0 }, m.ref[0..8]);
        } else {
            // write_hi 571, 3 blocks: 4 freed, 3 left to B, 2 replaced by 4
            // write_hi 571、3 ブロック: 4 は空き、3 は B に残し、2 を 4 で置き換え
            try t.expectEqual(@as(u64, 571), p.write_hi);
            try t.expectEqualSlices(u32, &.{ 0, 1, 4 }, p.table);
            try t.expectEqualSlices(u32, &.{4}, p.alloc);
            try t.expectEqual(@as(u64, 3), p.released);
            try t.expectEqualSlices(u32, &.{ 2, 2, 1, 1, 1, 1, 0 }, m.ref[0..7]);
        }
        // prompt_end = min(300, 512) = 300 during the request (4.5); blocks 0, 1 stay shared with their generations
        // 要求の間の prompt_end = min(300, 512) = 300 (4.5)。ブロック 0, 1 は共有のまま、世代もそのまま
        try t.expectEqual(@as(u64, 300), slotOf(&m, 0).prompt_end);
        try t.expectEqual(@as(u64, 512), slotOf(&m, 0).kv_end);
        try t.expectEqualSlices(u64, &.{ 2, 2 }, m.gen[0..2]);
        try fb.expectSame(&m);
        if (mt == 300) {
            try finish(&m, &p, &q, &toks(100, 5000)); // V = 661: blocks 0, 1, 4 (6 goes back) / 6 は空きへ
            try expectSlot(&m, 0, 661, 661, 562, 3, &.{ 0, 1, 4 });
        } else {
            try finish(&m, &p, &q, &.{9}); // V = 562
            try expectSlot(&m, 0, 562, 562, 562, 3, &.{ 0, 1, 4 });
        }
        try fb.expectSame(&m);
    }
}

test "P2: POOL.md 4.3 768 -> 751 -> 512 with the cut made by an eviction; a prompt stopped midway frees its reserve" {
    // capacity 2048: lower bound ceil(2056 / 256) + 1 = 10; 11 blocks
    // capacity 2048: 下限 ceil(2056 / 256) + 1 = 10。11 ブロック
    var m = try Meta.init(gpa, .{ .n_blocks = 11, .capacity = 2048, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{9});
    var b: [1000]i64 = undefined;
    @memcpy(b[0..800], a[0..800]);
    fill(b[800..], 50000);
    var p2 = try step(&m, &b, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 4 }, p2.table);
    try finish(&m, &p2, &b, &.{9});
    // C (1,600 tokens) needs 7 blocks, 6 are free: A's tail block 3 goes; A is cut to valid_end = prompt_end = 768
    // C (1,600 token) は 7 ブロック要り、空きは 6: A の末尾のブロック 3 を返し、A は 768 に縮む
    const cc = toks(1600, 90000);
    var p3 = try step(&m, &cc, 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p3.action);
    try t.expectEqual(@as(usize, 0), p3.evicted.len);
    try expectShrunk(p3.shrunk, &.{.{ 0, 768 }});
    try t.expectEqualSlices(u32, &.{ 3, 5, 6, 7, 8, 9, 10 }, p3.alloc);
    try expectSlot(&m, 0, 768, 768, 768, 1, &.{ 0, 1, 2 });
    try t.expectEqualSlices(i64, a[0..768], slotOf(&m, 0).seq.items);
    // C stops in its prompt after 1,024 rows: it keeps 4 blocks, its reserve 8, 9, 10 goes back
    // C は prompt の 1,024 行で止まる: 4 ブロックを残し、予約の 8, 9, 10 は空きに戻る
    try commitKv(&m, &p3, cc[0..1024], 1024, 1024);
    try expectSlot(&m, 2, 1024, 1024, 1024, 3, &.{ 3, 5, 6, 7 });
    try expectFree(&m, &.{ 8, 9, 10 });
    var fb = try Frozen.of(&m, 1);
    defer fb.deinit();
    // the 768 prompt again: A c = min(768, 751, 768) = 751 >= 751 (continues), B c = 751 (does not): A. Block 2 is shared
    // and 751 is inside it: resume 512, A's references to block 2 only is dropped, new blocks 8 and 9
    // 768 の prompt をもう一度: A は c = 751 (続けられる)、B は c = 751 (続けられない): A。ブロック 2 は共有で 751 はその
    // 途中: 512 から、A のブロック 2 の参照だけを外し、新しいブロック 8 と 9
    var p4 = try step(&m, a[0..768], 1);
    defer p4.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p4.action);
    try t.expectEqual(@as(u64, 0), p4.X);
    try t.expectEqual(@as(u64, 768), p4.h);
    try t.expectEqual(@as(u64, 751), p4.c);
    try t.expectEqual(@as(u64, 512), p4.resume_end);
    try t.expectEqual(@as(u64, 777), p4.write_hi);
    try t.expectEqualSlices(u32, &.{ 0, 1, 8, 9 }, p4.table);
    try t.expectEqualSlices(u32, &.{ 8, 9 }, p4.alloc);
    try t.expectEqual(@as(u64, 1), p4.released);
    try t.expectEqual(@as(usize, 0), p4.evicted.len + p4.shrunk.len);
    try t.expectEqual(@as(u32, 1), m.ref[2]);
    try expectSlot(&m, 0, 512, 512, 512, 1, &.{ 0, 1, 8, 9 });
    try fb.expectSame(&m);
    try finish(&m, &p4, a[0..768], &.{9});
    try expectSlot(&m, 0, 768, 768, 768, 4, &.{ 0, 1, 8 });
    try t.expectEqual(@as(u64, 3), m.gen[8]); // C's allocation, A's allocation, A's commit / C の割り当て、A の割り当て、A の確定
    try t.expectEqual(@as(u64, 2), m.gen[9]); // freed again: a free never bumps / 空きに戻しても進めない
    try expectFree(&m, &.{ 9, 10 });
    try fb.expectSame(&m);
}

test "P2: X's old private blocks past write_hi are released before need; the other slot is untouched; numbers reused" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const p = toks(300, 1);
    var p1 = try step(&m, &p, 1000);
    defer p1.deinit(gpa);
    try t.expectEqual(@as(u64, 1308), p1.write_hi);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5 }, p1.alloc);
    const out = toks(900, 5000);
    try finish(&m, &p1, &p, &out); // V = 1199, prompt_end 300: 5 blocks, block 5 back / 5 ブロック、5 は空きへ
    try expectSlot(&m, 0, 1199, 1199, 300, 1, &.{ 0, 1, 2, 3, 4 });
    try t.expectEqualSlices(u64, &.{ 2, 2, 2, 2, 1, 1 }, m.gen[0..6]);
    var b: [370]i64 = undefined;
    @memcpy(b[0..270], p[0..270]);
    fill(b[270..], 7000);
    var p2 = try step(&m, &b, 1);
    defer p2.deinit(gpa);
    // c = min(270, 353, 1199) = 270 < 300 - 17 and >= 256: SHARE of block 0
    // c = 270 < 300 - 17 かつ >= 256: ブロック 0 の SHARE
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqual(@as(u64, 256), p2.resume_end);
    try t.expectEqualSlices(u32, &.{ 0, 5 }, p2.table);
    try finish(&m, &p2, &b, &.{1});
    var fb = try Frozen.of(&m, 1);
    defer fb.deinit();
    // the 300 prompt again: c = min(300, 283, 1199) = 283 >= 283: CONTINUE at 283 (block 1 is A's own). write_hi 309, 2
    // blocks: A's blocks 2, 3, 4 (its old output) are released by the plan; nothing new is needed
    // 300 の prompt をもう一度: c = 283 >= 283: 283 から CONTINUE (ブロック 1 は A の専有)。write_hi 309、2 ブロック:
    // A のブロック 2, 3, 4 (古い出力) はプランで外す。新しいブロックは要らない
    var p3 = try step(&m, &p, 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 300), p3.h);
    try t.expectEqual(@as(u64, 283), p3.resume_end);
    try t.expectEqual(@as(u64, 309), p3.write_hi);
    try t.expectEqualSlices(u32, &.{ 0, 1 }, p3.table);
    try t.expectEqual(@as(usize, 0), p3.alloc.len);
    try t.expectEqual(@as(u64, 3), p3.released);
    try t.expectEqualSlices(u32, &.{0}, p3.verify);
    try expectSlot(&m, 0, 283, 283, 283, 1, &.{ 0, 1 });
    try t.expectEqualSlices(u32, &.{ 2, 1, 0, 0, 0, 1 }, m.ref[0..6]);
    try fb.expectSame(&m);
    try finish(&m, &p3, &p, &.{1});
    try t.expectEqualSlices(u64, &.{ 2, 2, 2, 2, 1, 2 }, m.gen[0..6]); // nothing complete written / 完成したブロックは書いていない
    try fb.expectSame(&m);
    // an unrelated prompt takes the lowest free numbers: the released 2, 3, 4
    // 無関係の prompt は一番小さい空きを取る: 外した 2, 3, 4
    const u = toks(700, 30000);
    var p4 = try step(&m, &u, 1);
    defer p4.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p4.action);
    try t.expectEqualSlices(u32, &.{ 2, 3, 4 }, p4.alloc);
    try t.expectEqualSlices(u64, &.{ 3, 3, 2 }, m.gen[2..5]);
    try fb.expectSame(&m);
}

test "P2: reference count 3; a slot's shared tail frees nothing but the walk goes on; whole slots dropped, the last cut" {
    // capacity 1536: lower bound ceil(1544 / 256) + 1 = 8; 9 blocks
    // capacity 1536: 下限 8。9 ブロック
    var m = try Meta.init(gpa, .{ .n_blocks = 9, .capacity = 1536, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const sys = toks(800, 1);
    var convs: [3][1000]i64 = undefined;
    for (&convs, 0..) |*c, i| {
        @memcpy(c[0..800], &sys);
        fill(c[800..], 10000 * @as(i64, @intCast(i + 1)));
    }
    var p1 = try step(&m, &convs[0], 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &convs[0], &.{9});
    var p2 = try step(&m, &convs[1], 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p2.action);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 4 }, p2.table);
    try finish(&m, &p2, &convs[1], &.{9});
    // the third ties A and B on c = 800 and h = 800 (neither continues); B's tick is newer: SHARE from B
    // 3 本目は A と B が c = 800・h = 800 で同点 (どちらも続けられない)。B の tick が新しい: B から SHARE
    var p3 = try step(&m, &convs[2], 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p3.action);
    try t.expectEqual(@as(?u64, 1), p3.S);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 5 }, p3.table);
    try finish(&m, &p3, &convs[2], &.{9});
    try t.expectEqualSlices(u32, &.{ 3, 3, 3, 1, 1, 1, 0, 0, 0 }, m.ref);
    // E needs 6 blocks, 3 are free. A (oldest): block 3 frees one; 2, 1, 0 free nothing (B and D hold them) but the walk
    // goes on and A goes. B: 4 frees one, 2, 1, 0 nothing: B goes. D: 5 frees the sixth: D is cut to 768
    // E は 6 ブロック要り、空きは 3。A (最古): 3 で 1 つ空く。2, 1, 0 は空かない (B と D が持つ) が次へ進み A は消える。
    // B: 4 で 1 つ、2, 1, 0 は空かない: B も消える。D: 5 で 6 つ目: D は 768 に縮む
    const e = toks(1500, 90000);
    var p4 = try step(&m, &e, 1);
    defer p4.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p4.action);
    try t.expectEqualSlices(u64, &.{ 0, 1 }, p4.evicted);
    try expectShrunk(p4.shrunk, &.{.{ 2, 768 }});
    try t.expectEqualSlices(u32, &.{ 3, 4, 5, 6, 7, 8 }, p4.table);
    try t.expectEqual(@as(usize, 2), m.slots.items.len);
    try expectSlot(&m, 2, 768, 768, 768, 3, &.{ 0, 1, 2 });
    try t.expectEqualSlices(i64, sys[0..768], slotOf(&m, 2).seq.items);
    try t.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, m.ref);
    try t.expectEqualSlices(u64, &.{ 2, 2, 2 }, m.gen[0..3]); // never rewritten / 書き直していない
    try finish(&m, &p4, &e, &.{9});
}

test "P2: a SHARE whose source goes whole in the same plan: X keeps the shared blocks, now its own" {
    var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{9});
    const q = toks(1000, 50000);
    var p2 = try step(&m, &q, 1);
    defer p2.deinit(gpa);
    try finish(&m, &p2, &q, &.{9});
    try expectFree(&m, &.{});
    var b: [1000]i64 = undefined;
    @memcpy(b[0..800], a[0..800]);
    fill(b[800..], 7000);
    // max_tokens 600 -> want 536 (capacity - n), write_hi = min(1544, 1536) = 1536: 6 blocks, 3 shared + 3 new. A
    // (oldest): 3 frees one, 2, 1, 0 stay (T holds them): A goes whole. Q: 7 and 6: Q is cut to 512
    // max_tokens 600 -> want 536、write_hi = 1536: 6 ブロック、共有 3 + 新規 3。A (最古): 3 で 1 つ、2, 1, 0 は T が
    // 持つので残る: A は丸ごと消える。Q: 7 と 6: Q は 512 に縮む
    var p3 = try step(&m, &b, 600);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p3.action);
    try t.expectEqual(@as(?u64, 0), p3.S);
    try t.expectEqual(@as(u64, 2), p3.X);
    try t.expectEqual(@as(u64, 768), p3.resume_end);
    try t.expectEqual(@as(u64, 1536), p3.write_hi);
    try t.expectEqualSlices(u64, &.{0}, p3.evicted);
    try expectShrunk(p3.shrunk, &.{.{ 1, 512 }});
    try t.expectEqualSlices(u32, &.{ 3, 6, 7 }, p3.alloc);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 6, 7 }, p3.table);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2 }, p3.verify);
    try t.expect(m.find(0) == null);
    try expectSlot(&m, 1, 512, 512, 512, 2, &.{ 4, 5 });
    try expectSlot(&m, 2, 768, 768, 0, 0, &.{ 0, 1, 2, 3, 6, 7 });
    try t.expectEqualSlices(i64, b[0..768], slotOf(&m, 2).seq.items);
    // the shared blocks keep their generation; block 3: A's allocation + T's
    // 共有したブロックの世代はそのまま。ブロック 3: A の割り当て + T の割り当て
    try t.expectEqualSlices(u64, &.{ 2, 2, 2, 2 }, m.gen[0..4]);
    try finish(&m, &p3, &b, &.{9});
    try expectSlot(&m, 2, 1000, 1000, 1000, 3, &.{ 0, 1, 2, 3 });
    try expectFree(&m, &.{ 6, 7 });
}

test "P2: slot cap 1: a SHARE drops the only other slot after T's references; freed numbers are reused at once" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 1, .mtp = false });
    defer m.deinit();
    const a = toks(600, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &.{9});
    // a CONTINUE with the cap reached drops nothing
    // 上限に達していても CONTINUE は何も消さない
    var a2: [700]i64 = undefined;
    @memcpy(a2[0..600], &a);
    fill(a2[600..], 3000);
    var p2 = try step(&m, &a2, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p2.action);
    try t.expectEqual(@as(usize, 0), p2.evicted.len);
    try finish(&m, &p2, &a2, &.{9});
    try expectSlot(&m, 0, 700, 700, 700, 2, &.{ 0, 1, 2 });
    // B branches at 400: c 400 < 683: SHARE of block 0; then the cap drops A whole (block 0 stays: T holds it); A's
    // blocks 1 and 2 are free again and the lowest: T takes them in the same plan
    // B は 400 で分かれる: c 400 < 683: ブロック 0 の SHARE。続いて上限で A を丸ごと捨てる (ブロック 0 は T が持つので
    // 残る)。A のブロック 1 と 2 が空いて一番小さいので、同じプランで T が取る
    var b: [700]i64 = undefined;
    @memcpy(b[0..400], a[0..400]);
    fill(b[400..], 9000);
    var p3 = try step(&m, &b, 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p3.action);
    try t.expectEqual(@as(u64, 256), p3.resume_end);
    try t.expectEqualSlices(u64, &.{0}, p3.evicted);
    try t.expectEqual(@as(usize, 0), p3.shrunk.len);
    try t.expectEqualSlices(u32, &.{ 1, 2 }, p3.alloc);
    try t.expectEqualSlices(u32, &.{ 0, 1, 2 }, p3.table);
    try t.expectEqual(@as(usize, 1), m.slots.items.len);
    try t.expectEqualSlices(u64, &.{ 2, 3, 2 }, m.gen[0..3]);
    try finish(&m, &p3, &b, &.{9});
    // an unrelated NEW drops T (everything else)
    // 無関係の NEW は T (ほかの全部) を捨てる
    const c = toks(300, 40000);
    var p4 = try step(&m, &c, 1);
    defer p4.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p4.action);
    try t.expectEqualSlices(u64, &.{1}, p4.evicted);
    try t.expectEqualSlices(u32, &.{ 0, 1 }, p4.alloc);
    try finish(&m, &p4, &c, &.{9});
    try t.expectEqual(@as(usize, 1), m.slots.items.len);
    try expectSlot(&m, 2, 300, 300, 300, 4, &.{ 0, 1 });
}

test "P2: slot cap full: a new slot drops the least recently committed other one, a CONTINUE drops none" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 3, .mtp = false });
    defer m.deinit();
    const a = toks(300, 1);
    const b = toks(300, 1000);
    const c = toks(300, 2000);
    for ([_][]const i64{ &a, &b, &c }) |pr| {
        var p = try step(&m, pr, 1);
        defer p.deinit(gpa);
        try t.expectEqual(plan.Action.NEW, p.action);
        try finish(&m, &p, pr, &.{9});
    }
    var c2: [350]i64 = undefined;
    @memcpy(c2[0..300], &c);
    fill(c2[300..], 5000);
    var p4 = try step(&m, &c2, 1);
    defer p4.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p4.action);
    try t.expectEqual(@as(usize, 0), p4.evicted.len);
    try finish(&m, &p4, &c2, &.{9}); // C's tick 4 / C の tick 4
    // D: the cap drops A (tick 1); D takes A's numbers
    // D: 上限で A (tick 1) を捨てる。D は A の番号を取る
    const d = toks(300, 3000);
    var p5 = try step(&m, &d, 1);
    defer p5.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p5.action);
    try t.expectEqualSlices(u64, &.{0}, p5.evicted);
    try t.expectEqualSlices(u32, &.{ 0, 1 }, p5.alloc);
    try finish(&m, &p5, &d, &.{9});
    // a SHARE at the cap whose source is the oldest (B, tick 2): B goes, its block 2 stays with T, its block 3 is the
    // lowest free number
    // 上限での SHARE で、元が最古 (B、tick 2): B は消え、ブロック 2 は T に残り、ブロック 3 が一番小さい空き
    var f: [400]i64 = undefined;
    @memcpy(f[0..280], b[0..280]);
    fill(f[280..], 6000);
    var p6 = try step(&m, &f, 1);
    defer p6.deinit(gpa);
    try t.expectEqual(plan.Action.SHARE, p6.action);
    try t.expectEqual(@as(?u64, 1), p6.S);
    try t.expectEqual(@as(u64, 4), p6.X);
    try t.expectEqualSlices(u64, &.{1}, p6.evicted);
    try t.expectEqualSlices(u32, &.{ 2, 3 }, p6.table);
    try t.expectEqualSlices(u32, &.{3}, p6.alloc);
    try finish(&m, &p6, &f, &.{9});
    try t.expectEqual(@as(usize, 3), m.slots.items.len);
    for ([_]u64{ 2, 3, 4 }, m.slots.items) |id, s| try t.expectEqual(id, s.id);
}

test "P2: usable blocks at the startup lower bound - 1 / = / + 1, capacities not a multiple of 256" {
    // capacity 1000: ceil(1008 / 256) + 1 = 5; 1020: ceil(1028 / 256) + 1 = 6 (the + 8 crosses a block); 769: 5
    // capacity 1000: 5、1020: 6 (+ 8 でブロックをまたぐ)、769: 5
    for ([_][2]u64{ .{ 1000, 5 }, .{ 1020, 6 }, .{ 769, 5 } }) |row| {
        const lb: u32 = @intCast(row[1]);
        try t.expectError(error.PoolTooSmall, (plan.Config{ .n_blocks = lb - 1, .capacity = row[0], .max_slots = 4, .mtp = false }).check());
        try (plan.Config{ .n_blocks = lb, .capacity = row[0], .max_slots = 4, .mtp = false }).check();
        try (plan.Config{ .n_blocks = lb + 1, .capacity = row[0], .max_slots = 4, .mtp = false }).check();
    }
    try t.expectError(error.PoolTooSmall, Meta.init(gpa, .{ .n_blocks = 4, .capacity = 1000, .max_slots = 4, .mtp = false }));
    try t.expectError(error.PoolTooSmall, (plan.Config{ .n_blocks = 40, .capacity = 0, .max_slots = 4, .mtp = false }).check());
    try t.expectError(error.PoolTooSmall, (plan.Config{ .n_blocks = 40, .capacity = 1000, .max_slots = 0, .mtp = false }).check());
    for ([_]u32{ 5, 6 }) |nb| {
        var m = try Meta.init(gpa, .{ .n_blocks = nb, .capacity = 1000, .max_slots = 8, .mtp = false });
        defer m.deinit();
        // n = 999 = capacity - 1, max_tokens 50 -> want 1, write_hi = min(1008, 1000) = 1000: 4 blocks
        // n = 999 = capacity - 1、max_tokens 50 -> want 1、write_hi = 1000: 4 ブロック
        const a = toks(999, 1);
        var p1 = try step(&m, &a, 50);
        defer p1.deinit(gpa);
        try t.expectEqual(@as(u64, 1000), p1.write_hi);
        try t.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, p1.table);
        try finish(&m, &p1, &a, &.{9});
        const b = toks(999, 20000);
        var p2 = try step(&m, &b, 1);
        defer p2.deinit(gpa);
        // 4 needed: with 5 blocks 1 is free, A gives 3 (cut to 256); with 6, 2 are free, A gives 2 (cut to 512)
        // 4 つ要る: 5 ブロックなら空き 1、A が 3 つ返す (256 に)。6 なら空き 2、A が 2 つ (512 に)
        if (nb == 5) {
            try expectShrunk(p2.shrunk, &.{.{ 0, 256 }});
            try t.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, p2.table);
        } else {
            try expectShrunk(p2.shrunk, &.{.{ 0, 512 }});
            try t.expectEqualSlices(u32, &.{ 2, 3, 4, 5 }, p2.table);
        }
        try finish(&m, &p2, &b, &.{9});
        // n = capacity and n = 0 are refused; the metadata does not change
        // n = capacity と n = 0 は断る。メタデータは変わらない
        const d0 = m.digest();
        const big = toks(1000, 1);
        try t.expectError(error.BadRequest, plan.makePlan(&m, &big, 1, true));
        try t.expectError(error.BadRequest, plan.makePlan(&m, &[_]i64{}, 1, true));
        try t.expectEqual(d0, m.digest());
        if (nb == 5) {
            // A's 256 prefix + more: c = min(256, 583, 256) = 256 >= 239: CONTINUE at 256; 2 new blocks, none free: B
            // gives its tail 4, 3 (cut to 512)
            // A の 256 の先頭 + 続き: c = 256 >= 239: 256 から CONTINUE。新しく 2 つ要り空きなし: B が末尾の 4, 3 を返す
            var p3 = try step(&m, a[0..600], 1);
            defer p3.deinit(gpa);
            try t.expectEqual(plan.Action.CONTINUE, p3.action);
            try t.expectEqual(@as(u64, 256), p3.resume_end);
            try expectShrunk(p3.shrunk, &.{.{ 1, 512 }});
            try t.expectEqualSlices(u32, &.{ 0, 3, 4 }, p3.table);
            try finish(&m, &p3, a[0..600], &.{9});
        }
    }
}

test "P2: n = 1 .. 17 have c = 0 and continue their own slot from 0; n = 18 resumes 1 (with and without MTP)" {
    for ([_]bool{ false, true }) |mtp| for (1..19) |nn| {
        var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = mtp });
        defer m.deinit();
        var buf: [18]i64 = undefined;
        const pr = buf[0..nn];
        fill(pr, 100);
        var p1 = try step(&m, pr, 1);
        defer p1.deinit(gpa);
        try t.expectEqual(plan.Action.NEW, p1.action);
        try t.expectEqual(@as(?u64, null), p1.S);
        try t.expectEqual(@as(u64, nn + 9), p1.write_hi);
        try t.expectEqualSlices(u32, &.{0}, p1.table);
        try finish(&m, &p1, pr, &.{7});
        // c = min(h - d, max(0, n - 17), kv_end): 0 for n <= 17; 0 >= max(0, n - 17): CONTINUE from 0 in its own block
        // n <= 17 では c = 0。0 >= max(0, n - 17): 自分のブロックで 0 から CONTINUE
        var p2 = try step(&m, pr, 1);
        defer p2.deinit(gpa);
        const want_c: u64 = if (nn <= 17) 0 else nn - 17;
        try t.expectEqual(plan.Action.CONTINUE, p2.action);
        try t.expectEqual(@as(u64, nn), p2.h);
        try t.expectEqual(want_c, p2.c);
        try t.expectEqual(want_c, p2.resume_end);
        try t.expectEqualSlices(u32, &.{0}, p2.table);
        try t.expectEqual(@as(usize, 0), p2.alloc.len);
        try expectSlot(&m, 0, want_c, want_c, want_c, 1, &.{0});
        try finish(&m, &p2, pr, &.{7});
        try expectSlot(&m, 0, nn, nn, nn, 2, &.{0});
    };
}

test "P2: prompt_end under 17: an unrelated prompt continues that slot from 0 even if it is older; a stop before the first chunk drops it" {
    var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const short = toks(10, 1);
    var p1 = try step(&m, &short, 1);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &short, &.{2});
    // a cold request (resume_ok false) makes a newer, longer slot: NEW, S null
    // キャッシュを使わない要求 (resume_ok false) で新しく長いスロットを作る: NEW、S は null
    const long0 = toks(600, 50000);
    var p2 = try stepOpt(&m, &long0, 1, false);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.NEW, p2.action);
    try t.expectEqual(@as(?u64, null), p2.S);
    try t.expectEqual(@as(u64, 0), p2.c);
    try t.expectEqualSlices(u32, &.{ 1, 2, 3 }, p2.table);
    try finish(&m, &p2, &long0, &.{2});
    // an unrelated 400-token prompt: c = 0 everywhere; the short slot continues (0 >= max(0, 10 - 17)), the long one
    // does not (0 < 583): the older short slot wins the tie (4.1), CONTINUE at 0 in its own block 0 + a new block 4
    // 無関係の 400 token: どこも c = 0。短いスロットは続けられ、長い方は続けられない: 古い短いスロットが同点で勝つ。
    // 自分のブロック 0 と新しいブロック 4 で 0 から CONTINUE
    const u = toks(400, 90000);
    var p3 = try step(&m, &u, 1);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 0), p3.X);
    try t.expectEqual(@as(u64, 0), p3.h);
    try t.expectEqual(@as(u64, 0), p3.resume_end);
    try t.expectEqualSlices(u32, &.{ 0, 4 }, p3.table);
    try t.expectEqualSlices(u32, &.{4}, p3.alloc);
    try expectSlot(&m, 0, 0, 0, 0, 1, &.{ 0, 4 });
    // stopped before the first chunk: nothing committed from 0, the slot goes (POOL.md 4.4, v3.5)
    // 最初のチャンクの前で止まった: 0 から何も確定しないので、スロットを消す (POOL.md 4.4、v3.5)
    try commitKv(&m, &p3, &[_]i64{}, 0, 0);
    try t.expect(m.find(0) == null);
    try t.expectEqual(@as(usize, 1), m.slots.items.len);
    try expectFree(&m, &.{ 0, 4, 5, 6, 7 });
}

test "P2: h != c: c = 255 / 256 / 257 without MTP; with MTP h = 256 / 257 / 258 against kv_end 255 / 256 / 257" {
    const a = toks(1000, 1);
    const Case = struct { mtp: bool, h: usize, kv: u64, c: u64 };
    for ([_]Case{
        .{ .mtp = false, .h = 255, .kv = 1000, .c = 255 },
        .{ .mtp = false, .h = 256, .kv = 1000, .c = 256 },
        .{ .mtp = false, .h = 257, .kv = 1000, .c = 257 },
        .{ .mtp = true, .h = 256, .kv = 256, .c = 255 }, // h - 1
        .{ .mtp = true, .h = 257, .kv = 256, .c = 256 },
        .{ .mtp = true, .h = 258, .kv = 257, .c = 257 },
        .{ .mtp = true, .h = 258, .kv = 255, .c = 255 }, // kv_end before the block end: block 0 is not shareable
        .{ .mtp = true, .h = 257, .kv = 255, .c = 255 },
        .{ .mtp = true, .h = 1000, .kv = 256, .c = 256 }, // kv_end caps a long h / kv_end が長い h を抑える
    }) |cs| {
        var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = cs.mtp });
        defer m.deinit();
        var p1 = try step(&m, &a, 1);
        defer p1.deinit(gpa);
        try commitKv(&m, &p1, &a, 1000, cs.kv);
        var pr: [1030]i64 = undefined;
        @memcpy(pr[0..cs.h], a[0..cs.h]);
        fill(pr[cs.h .. cs.h + 30], 70000);
        const q = pr[0 .. cs.h + 30];
        var p2 = try step(&m, q, 1);
        defer p2.deinit(gpa);
        try t.expectEqual(@as(u64, cs.h), p2.h);
        try t.expectEqual(cs.c, p2.c);
        try t.expectEqual(@as(?u64, 0), p2.S);
        if (cs.c >= 256) {
            try t.expectEqual(plan.Action.SHARE, p2.action);
            try t.expectEqual(@as(u64, 256), p2.resume_end);
            try t.expectEqual(@as(u32, 0), p2.table[0]);
            try t.expectEqual(@as(u32, 2), m.ref[0]);
        } else {
            try t.expectEqual(plan.Action.NEW, p2.action);
            try t.expectEqual(@as(u64, 0), p2.resume_end);
            try t.expectEqual(@as(u32, 4), p2.table[0]);
            try t.expectEqual(@as(u32, 1), m.ref[0]);
        }
    }
}

test "P2: the largest c wins over a longer exact h (every candidate's exact match is computed first)" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = true });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 1);
    defer p1.deinit(gpa);
    try commitKv(&m, &p1, &a, 1000, 255); // MTP rows right only below 255 / MTP の行は 255 まで
    var b: [1000]i64 = undefined;
    @memcpy(b[0..500], a[0..500]);
    fill(b[500..], 30000);
    var p2 = try step(&m, &b, 1);
    defer p2.deinit(gpa);
    // against A: h 500, c = min(499, 983, 255) = 255 < 256: NEW
    // A に対して: h 500、c = 255 < 256: NEW
    try t.expectEqual(plan.Action.NEW, p2.action);
    try t.expectEqual(@as(u64, 255), p2.c);
    try finish(&m, &p2, &b, &.{9});
    var q: [650]i64 = undefined;
    @memcpy(q[0..600], a[0..600]);
    fill(q[600..], 40000);
    var p3 = try step(&m, &q, 1);
    defer p3.deinit(gpa);
    // A: h 600, c = min(599, 633, 255) = 255; B: h 500, c = min(499, 633, 1000) = 499 -> B, SHARE at 256 from B's block
    // A: h 600、c = 255。B: h 500、c = 499 -> B。B のブロックで 256 から SHARE
    try t.expectEqual(plan.Action.SHARE, p3.action);
    try t.expectEqual(@as(?u64, 1), p3.S);
    try t.expectEqual(@as(u64, 500), p3.h);
    try t.expectEqual(@as(u64, 499), p3.c);
    try t.expectEqual(@as(u64, 256), p3.resume_end);
    try t.expectEqual(slotOf(&m, 1).blocks.items[0], p3.table[0]);
}

test "P2: G4: a block complete before and overwritten from its middle gets a new generation; untouched blocks keep theirs" {
    for ([_]bool{ false, true }) |mtp| {
        var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = mtp });
        defer m.deinit();
        const p = toks(300, 1);
        var p1 = try step(&m, &p, 300);
        defer p1.deinit(gpa);
        try t.expectEqualSlices(u32, &.{ 0, 1, 2 }, p1.alloc);
        try finish(&m, &p1, &p, &toks(220, 5000)); // V = 519: blocks 0, 1 complete / ブロック 0, 1 が完成
        try t.expectEqualSlices(u64, &.{ 2, 2, 1 }, m.gen[0..3]);
        var p2 = try step(&m, &p, 300);
        defer p2.deinit(gpa);
        // c = min(300 - d, 283, 519) = 283: CONTINUE at 283 inside block 1 (A's own, complete since the last commit)
        // c = 283: ブロック 1 (A の専有で、前の確定から完成している) の途中の 283 から CONTINUE
        try t.expectEqual(plan.Action.CONTINUE, p2.action);
        try t.expectEqual(@as(u64, 283), p2.resume_end);
        try t.expectEqualSlices(u32, &.{ 0, 1, 2 }, p2.table);
        try t.expectEqual(@as(usize, 0), p2.alloc.len);
        try t.expectEqualSlices(u64, &.{ 2, 2, 1 }, m.gen[0..3]);
        const seq = try std.mem.concat(gpa, i64, &.{ &p, &toks(219, 8000) });
        defer gpa.free(seq);
        if (!mtp) {
            // V = kv_end = 519: block 1 was written in [283, 512) and is wholly below kv_end -> generation 3 (4.5)
            // V = kv_end = 519: ブロック 1 は [283, 512) を書き kv_end の手前に丸ごと入る -> 世代 3 (4.5)
            try commitKv(&m, &p2, seq, 300, 519);
            try t.expectEqualSlices(u64, &.{ 2, 3, 1 }, m.gen[0..3]);
        } else {
            // kv_end 510 (MTP rows short of the block end): block 1 is not wholly below kv_end: no new generation
            // kv_end 510 (MTP の行がブロックの終わりに届かない): ブロック 1 は kv_end の手前に丸ごと入らない: 世代はそのまま
            try commitKv(&m, &p2, seq, 300, 510);
            try t.expectEqualSlices(u64, &.{ 2, 2, 1 }, m.gen[0..3]);
        }
    }
}

test "P2: shrinking a slot never raises prompt_end" {
    var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = false });
    defer m.deinit();
    const p = toks(300, 1);
    var p1 = try step(&m, &p, 1000);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &p, &toks(700, 5000)); // V 999, prompt_end 300
    try expectSlot(&m, 0, 999, 999, 300, 1, &.{ 0, 1, 2, 3 });
    const q = toks(1000, 50000);
    var p2 = try step(&m, &q, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(@as(usize, 0), p2.evicted.len + p2.shrunk.len);
    try finish(&m, &p2, &q, &.{9});
    // P needs 2 blocks, none free: A gives 3 and 2 -> valid_end 512, prompt_end stays 300 (not 512)
    // P は 2 ブロック要り空きなし: A が 3 と 2 を返す -> valid_end 512、prompt_end は 300 のまま (512 にしない)
    const pp = toks(260, 90000);
    var p3 = try step(&m, &pp, 1);
    defer p3.deinit(gpa);
    try expectShrunk(p3.shrunk, &.{.{ 0, 512 }});
    try expectSlot(&m, 0, 512, 512, 300, 1, &.{ 0, 1 });
    try finish(&m, &p3, &pp, &.{9});
    // R needs 1: A gives 1 -> valid_end 256, prompt_end min(300, 256) = 256
    // R は 1 つ要る: A が 1 を返す -> valid_end 256、prompt_end = min(300, 256) = 256
    const r = toks(10, 95000);
    var p4 = try step(&m, &r, 1);
    defer p4.deinit(gpa);
    try expectShrunk(p4.shrunk, &.{.{ 0, 256 }});
    try expectSlot(&m, 0, 256, 256, 256, 1, &.{0});
    try t.expectEqualSlices(i64, p[0..256], slotOf(&m, 0).seq.items);
}

// ------------------------------------------------------------ harness-like walk against the POOL.md 4.1 - 4.6.2 rules ---

/// What POOL.md 4.1 - 4.3 make of a prompt on the metadata before the plan: every slot's exact h and c_S, the best by
/// (c, continuable, h, newer tick, smaller id), the action from c, the resume point with 4.3 step 2's rounding.
/// POOL.md 4.1〜4.3 がプラン前のメタデータで prompt から決めるもの: 全スロットの正確な h と c_S、(c, 続けられる, h,
/// 新しい tick, 小さい id) で最良のもの、c からの動作、4.3 の 2 の切り下げをした再開位置。
const Want = struct { action: plan.Action, S: ?u64, h: u64, c: u64, r: u64 };

fn wantOf(m: *const Meta, prompt: []const i64) Want {
    const n: u64 = prompt.len;
    const d: u64 = @intFromBool(m.cfg.mtp);
    const cap: u64 = if (n > 17) n - 17 else 0;
    var w: Want = .{ .action = .NEW, .S = null, .h = 0, .c = 0, .r = 0 };
    var bi: ?usize = null;
    var bcont = false;
    var be: u64 = 0;
    for (m.slots.items, 0..) |*s, i| {
        var h: u64 = 0;
        while (h < s.seq.items.len and h < n and s.seq.items[h] == prompt[h]) h += 1;
        const c = @min(if (h > d) h - d else 0, cap, s.kv_end);
        const cont = c + 17 >= s.prompt_end; // c >= max(0, prompt_end - 17)
        // where the request would resume from this slot (POOL.md 4.1 v3.6): c when continued (a shared branch block
        // rounds it down), c rounded down for a SHARE, 0 for a NEW
        // このスロットからの再開位置 (POOL.md 4.1 v3.6): 続けるなら c (分かれ目のブロックが共有なら切り下げ)、SHARE なら
        // c の切り下げ、NEW なら 0
        const e: u64 = if (cont)
            (if (c % 256 != 0 and c / 256 < s.blocks.items.len and m.ref[s.blocks.items[c / 256]] >= 2) c / 256 * 256 else c)
        else if (c >= 256) c / 256 * 256 else 0;
        var better = bi == null;
        if (bi) |j| {
            const b = &m.slots.items[j];
            if (e != be) {
                better = e > be;
            } else if (cont != bcont) {
                better = cont;
            } else if (c != w.c) {
                better = c > w.c;
            } else if (h != w.h) {
                better = h > w.h;
            } else if (s.tick != b.tick) {
                better = s.tick > b.tick;
            } else better = s.id < b.id;
        }
        if (better) {
            bi = i;
            bcont = cont;
            be = e;
            w.h = h;
            w.c = c;
            w.S = s.id;
        }
    }
    if (bi) |j| {
        const s = &m.slots.items[j];
        if (bcont) {
            w.action = .CONTINUE;
            w.r = w.c;
            const k = w.c / 256;
            if (w.c % 256 != 0 and m.ref[s.blocks.items[k]] >= 2) w.r = k * 256;
        } else if (w.c >= 256) {
            w.action = .SHARE;
            w.r = w.c / 256 * 256;
        }
    }
    return w;
}

/// The metadata before a step: every slot frozen, the generations, the saves, the live ring's identity.
/// ステップ前のメタデータ: 全スロットの写し、世代、写しの記録、live のリングの身元。
const Before = struct {
    slots: []Frozen,
    gen: []u64,
    saves: []plan.Save,
    ring: ?[2]u64,
    next_id: u64,

    fn take(m: *Meta) !Before {
        const slots = try gpa.alloc(Frozen, m.slots.items.len);
        var k: usize = 0;
        errdefer {
            for (slots[0..k]) |*f| f.deinit();
            gpa.free(slots);
        }
        for (m.slots.items) |s| {
            slots[k] = try Frozen.of(m, s.id);
            k += 1;
        }
        const gen = try gpa.dupe(u64, m.gen);
        errdefer gpa.free(gen);
        const saves = try gpa.dupe(plan.Save, m.saves);
        return .{ .slots = slots, .gen = gen, .saves = saves, .ring = m.ring, .next_id = m.next_id };
    }

    fn deinit(b: *Before) void {
        for (b.slots) |*f| f.deinit();
        gpa.free(b.slots);
        gpa.free(b.gen);
        gpa.free(b.saves);
    }

    fn find(b: *const Before, id: u64) ?*const Frozen {
        for (b.slots) |*f| if (f.id == id) return f;
        return null;
    }

    fn saveOf(b: *const Before, id: u64) ?usize {
        for (b.saves, 0..) |sv, i| if (sv.owner != null and sv.owner.? == id) return i;
        return null;
    }
};

/// A save after a plan (POOL.md 4.6.2): freed with its slot, its vhi cut with its slot (vlo never lowered; a ring with
/// vhi <= vlo is gone), else unchanged.
/// プラン後の写し (POOL.md 4.6.2): スロットと一緒に空く、スロットと一緒に vhi を切る (vlo は下げない。vhi <= vlo の
/// リングはなくなる)、それ以外は変わらない。
fn expectSaveAfter(old: plan.Save, now: plan.Save, dropped: bool, cut: ?u64) !void {
    if (dropped) {
        try t.expectEqual(@as(?u64, null), now.owner);
        try t.expectEqual(old.gen, now.gen);
        try t.expect(now.live == null and now.snap == null);
        return;
    }
    if (cut) |v| {
        const lv: ?plan.Ring = if (old.live) |g| g.shrink(v) else null;
        const sn: ?plan.Ring = if (old.snap) |g| g.shrink(v) else null;
        try t.expectEqual(old.gen, now.gen);
        try t.expectEqual(lv, now.live);
        try t.expectEqual(sn, now.snap);
        if (lv != null or sn != null) {
            try t.expectEqual(old.owner, now.owner);
            try t.expectEqual(old.used, now.used);
        }
        if (lv) |g| try t.expect(g.vlo == old.live.?.vlo and g.vhi <= v);
        return;
    }
    try t.expectEqual(old, now);
}

const Session = struct {
    prompt: std.ArrayList(i64) = .empty, // its last prompt / 最後の prompt
    out: std.ArrayList(i64) = .empty, // its last output / 最後の出力
    seq: std.ArrayList(i64) = .empty, // what its slot held after its commit / 確定後にスロットが持っていたもの
    sid: ?u64 = null,
    pe: u64 = 0,
    kv: u64 = 0,
    done: bool = false, // the last request did not stop in its prompt / 最後の要求は prompt の途中で止まっていない

    fn deinit(s: *Session) void {
        s.prompt.deinit(gpa);
        s.out.deinit(gpa);
        s.seq.deinit(gpa);
    }

    /// Its slot still holds exactly what its last commit left (not cut, not taken over by another session).
    /// スロットが最後の確定のままのものを持つ (縮んでいない、別のセッションに引き継がれていない)。
    fn intact(s: *const Session, m: *Meta) bool {
        const id = s.sid orelse return false;
        const i = m.find(id) orelse return false;
        const x = &m.slots.items[i];
        return std.mem.eql(i64, x.seq.items, s.seq.items) and x.prompt_end == s.pe and x.kv_end == s.kv;
    }
};

fn userText(pr: *std.ArrayList(i64), rnd: std.Random, k: usize) !void {
    for (0..k) |_| try pr.append(gpa, rnd.intRangeAtMost(i64, 1000, 50000));
}

test "P2: harness-like request sequences: the 4.1 - 4.3 choice, the 4.4 - 4.5 effects and the 4.6.2 saves at every step" {
    var prng = std.Random.DefaultPrng.init(0x5eed9002);
    const rnd = prng.random();
    const window: u64 = 300; // a short drafter window so that D = [n - 301, r) is often not all of [0, r)
    const Cfg = struct { nb: u32, cap: u64, slots: u32, mtp: bool, saves: u32 };
    var shares: usize = 0;
    // capacity 1792: lower bound ceil(1800 / 256) + 1 = 9; 1800: ceil(1808 / 256) + 1 = 9; 1000: 5
    // capacity 1792: 下限 9。1800: 9。1000: 5
    for ([_]Cfg{
        .{ .nb = 9, .cap = 1792, .slots = 6, .mtp = false, .saves = 3 },
        .{ .nb = 9, .cap = 1800, .slots = 3, .mtp = true, .saves = 2 },
        .{ .nb = 14, .cap = 1792, .slots = 8, .mtp = true, .saves = 8 },
        .{ .nb = 40, .cap = 1792, .slots = 2, .mtp = false, .saves = 1 },
        .{ .nb = 6, .cap = 1000, .slots = 64, .mtp = false, .saves = 0 },
        .{ .nb = 20, .cap = 1800, .slots = 1, .mtp = true, .saves = 1 },
    }) |cf| {
        var m = try Meta.init(gpa, .{ .n_blocks = cf.nb, .capacity = cf.cap, .max_slots = cf.slots, .mtp = cf.mtp, .draft_saves = cf.saves });
        defer m.deinit();
        var sys: [600]i64 = undefined;
        fill(&sys, 0); // user text is drawn from 1,000 .. 50,000 / 利用者の文は 1,000〜50,000 から
        var sess: [5]Session = @splat(.{});
        defer for (&sess) |*s| s.deinit();
        var pr: std.ArrayList(i64) = .empty;
        defer pr.deinit(gpa);
        var out: std.ArrayList(i64) = .empty;
        defer out.deinit(gpa);
        var seq: std.ArrayList(i64) = .empty;
        defer seq.deinit(gpa);
        var uniq: i64 = 1 << 40; // title tokens: never seen anywhere else / title のトークン: ほかに出てこない
        var acts: [3]usize = @splat(0);
        for (0..600) |_| {
            const si = rnd.uintLessThan(usize, sess.len);
            const se = &sess[si];
            pr.clearRetainingCapacity();
            // 0 a new main conversation, 1 its next turn (or a return to it), 2 a subagent, 3 a title request,
            // 4 a resend, 5 the prompt grown by 1..16 tokens, 6 another session's prompt
            // 0 新しいメインの会話、1 次のターン (戻りも)、2 サブエージェント、3 title、4 再送、5 1〜16 token 伸びた prompt、
            // 6 別のセッションの prompt
            var kind: usize = if (se.prompt.items.len == 0) 0 else rnd.uintLessThan(usize, 7);
            var src: ?*Session = null;
            if (kind == 6) {
                const oi = rnd.uintLessThan(usize, sess.len);
                if (oi != si and sess[oi].prompt.items.len > 0) src = &sess[oi] else kind = 0;
            }
            switch (kind) {
                1 => if (se.done) {
                    try pr.appendSlice(gpa, se.prompt.items);
                    try pr.appendSlice(gpa, se.out.items[0 .. se.out.items.len - @min(se.out.items.len, rnd.uintLessThan(usize, 4))]);
                    try userText(&pr, rnd, 17 + rnd.uintLessThan(usize, 184));
                } else try pr.appendSlice(gpa, se.prompt.items), // sent again after a stop / 止まった後の送り直し
                2 => {
                    try pr.appendSlice(gpa, &sys);
                    try userText(&pr, rnd, 1 + rnd.uintLessThan(usize, 300));
                },
                3 => for (0..20 + rnd.uintLessThan(usize, 61)) |_| {
                    try pr.append(gpa, uniq);
                    uniq += 1;
                },
                4 => try pr.appendSlice(gpa, se.prompt.items),
                5 => {
                    try pr.appendSlice(gpa, se.prompt.items);
                    try userText(&pr, rnd, 1 + rnd.uintLessThan(usize, 16));
                },
                6 => try pr.appendSlice(gpa, src.?.prompt.items),
                else => {},
            }
            if (kind == 0 or pr.items.len + 1 >= cf.cap) {
                kind = 0;
                pr.clearRetainingCapacity();
                try pr.appendSlice(gpa, sys[0..if (rnd.boolean()) sys.len else rnd.uintLessThan(usize, 40)]);
                try userText(&pr, rnd, 1 + rnd.uintLessThan(usize, 300));
            }
            const n: u64 = pr.items.len;
            // the design-level expectations of POOL.md 1.2 / 4.2 for a slot that still holds its last commit
            // スロットが最後の確定のままのときの POOL.md 1.2 / 4.2 の期待
            const owner: ?*Session = if (kind == 6) src else if (kind == 1 or kind == 4 or kind == 5) se else null;
            const whole = if (owner) |o| o.done and o.intact(&m) else false;

            const w = wantOf(&m, pr.items);
            var before = try Before.take(&m);
            defer before.deinit();
            const mt = rnd.intRangeAtMost(i64, -2, 64);
            var p = try plan.makePlan(&m, pr.items, mt, true);
            defer p.deinit(gpa);
            try t.expectEqual(w.action, p.action);
            try t.expectEqual(w.S, p.S);
            try t.expectEqual(w.h, p.h);
            try t.expectEqual(w.c, p.c);
            try t.expectEqual(w.r, p.resume_end);
            const r = p.resume_end;
            try t.expectEqual(r, p.write_lo);
            const want = plan.wantTokens(cf.cap, n, mt);
            try t.expectEqual(@min(n + want + 8, cf.cap), p.write_hi);
            try t.expectEqual(plan.ceilDiv(p.write_hi, 256), p.table.len);
            try t.expectEqual(if (p.action == .CONTINUE) w.S.? else before.next_id, p.X);
            try t.expectEqual(p.action != .CONTINUE, p.new_x);
            const j0: usize = @intCast(r / 256);
            if (p.action != .NEW) try t.expectEqualSlices(u32, before.find(w.S.?).?.blocks[0..j0], p.table[0..j0]);
            for (p.alloc[@min(p.alloc.len, 1)..], 0..) |b, i| try t.expect(p.alloc[i] < b);
            switch (kind) {
                3 => {
                    try t.expectEqual(@as(u64, 0), p.h);
                    try t.expectEqual(@as(u64, 0), r);
                },
                1, 4, 5, 6 => if (whole and n > 17) {
                    // a resend, a prompt grown by <= 16, the same prompt from another session: the owner's c is the cap
                    // n - 17 (when its kv_end reaches it) and it continues: CONTINUE with c = n - 17 (POOL.md 1.2, 4.2).
                    // The next turn without MTP: CONTINUE (no other slot holds this session's output). With MTP the
                    // owner's c is capped by its kv_end and a sibling holding the same prompt can win by c (see the
                    // MTP test below)
                    // 再送・16 以下だけ伸びた prompt・別のセッションの同じ prompt: 持ち主の c は上限 n - 17 (kv_end が
                    // 届けば) で続けられる: c = n - 17 の CONTINUE (POOL.md 1.2、4.2)。MTP なしの次のターン: CONTINUE
                    // (このセッションの出力をほかのスロットは持たない)。MTP では持ち主の c は kv_end で抑えられ、同じ
                    // prompt を持つ別のスロットが c で勝ちうる (下の MTP の試験)
                    if (kind != 1 and owner.?.kv + 17 >= n) {
                        try t.expectEqual(plan.Action.CONTINUE, p.action);
                        try t.expectEqual(n - 17, p.c);
                    }
                    if (kind == 1 and !cf.mtp) try t.expectEqual(plan.Action.CONTINUE, p.action);
                },
                else => {},
            }
            // 4.6.2: the save restored at the start (S's, for CONTINUE and SHARE with something to resume)
            // 4.6.2: 要求の初めに戻す写し (CONTINUE と SHARE で再開するものがあるとき、S のもの)
            if (p.action != .NEW and r > 0 and before.saveOf(w.S.?) != null) {
                const k = before.saveOf(w.S.?).?;
                try t.expectEqual(@as(?usize, k), p.restore);
                try t.expectEqual(before.saves[k].gen, p.restore_gen);
                try t.expectEqual(before.saves[k].live, p.restore_live);
                try t.expectEqual(before.saves[k].snap, p.restore_snap);
                try t.expectEqual(p.h, p.below);
                const same = if (before.ring) |g| g[0] == k and g[1] == before.saves[k].gen else false;
                try t.expectEqual(same, p.skip_copy);
            } else try t.expectEqual(@as(?usize, null), p.restore);

            try plan.apply(&m, &p, pr.items);
            try m.check(p.X);
            acts[@intFromEnum(p.action)] += 1;
            // 4.3: X's blocks from j0 on are its own; the blocks taken were the lowest free; only they got a generation
            // 4.3: j0 以降の X のブロックは専有。割り当ては一番小さい空きから。世代が進むのはそれだけ
            for (p.table[j0..]) |b| try t.expectEqual(@as(u32, 1), m.ref[b]);
            if (p.alloc.len > 0) for (m.ref, 0..) |rc, b| if (rc == 0) try t.expect(b > p.alloc[p.alloc.len - 1]);
            for (before.gen, m.gen, 0..) |g0, g1, b| try t.expectEqual(g0 + @intFromBool(std.mem.indexOfScalar(u32, p.alloc, @intCast(b)) != null), g1);
            // 4.4: every other slot is dropped, or cut from its tail (tokens a prefix, prompt_end and kv_end never raised,
            // tick kept), or untouched
            // 4.4: ほかのスロットは消える、末尾から縮む (トークン列は先頭部分、prompt_end と kv_end は増えない、tick は
            // そのまま)、または変わらない
            for (before.slots) |*f| {
                if (f.id == p.X) continue;
                const gone = std.mem.indexOfScalar(u64, p.evicted, f.id) != null;
                const fi = m.find(f.id);
                try t.expectEqual(gone, fi == null);
                if (gone) continue;
                const s = &m.slots.items[fi.?];
                const v = s.validEnd();
                var cut: ?u64 = null;
                for (p.shrunk) |e| if (e[0] == f.id) {
                    cut = e[1];
                };
                try t.expectEqual(cut orelse f.seq.len, v);
                if (cut) |cv| try t.expect(cv < f.seq.len);
                try t.expectEqualSlices(i64, f.seq[0..v], s.seq.items);
                try t.expectEqualSlices(u32, f.blocks[0..plan.ceilDiv(v, 256)], s.blocks.items);
                try t.expectEqual(@min(f.prompt_end, v), s.prompt_end);
                try t.expectEqual(@min(f.kv_end, v), s.kv_end);
                try t.expectEqual(f.tick, s.tick);
            }
            for (p.shrunk) |e| try t.expect(e[0] != p.X and std.mem.indexOfScalar(u64, p.evicted, e[0]) == null);
            // 4.5: X during the request holds the first resume_end tokens of the prompt
            // 4.5: 要求の間の X は prompt の先頭 resume_end 個を持つ
            {
                const x = slotOf(&m, p.X);
                try t.expectEqualSlices(i64, pr.items[0..r], x.seq.items);
                try t.expectEqual(r, x.kv_end);
                const pe0: u64 = if (p.new_x) 0 else @min(before.find(p.X).?.prompt_end, r);
                try t.expectEqual(pe0, x.prompt_end);
                try t.expectEqualSlices(u32, p.table, x.blocks.items);
            }
            for (before.saves, m.saves) |old, now| {
                var dropped = false;
                var cut: ?u64 = null;
                if (old.owner) |o| {
                    if (std.mem.indexOfScalar(u64, p.evicted, o) != null) {
                        dropped = true;
                    } else {
                        for (p.shrunk) |e| if (e[0] == o) {
                            cut = e[1];
                        };
                        if (o == p.X) cut = r;
                    }
                }
                try expectSaveAfter(old, now, dropped, cut);
            }

            // the drafter's range-based resume (S2): pos is r; s is 0 cold, L when restored, r otherwise
            // drafter の範囲付きの再開 (S2): pos は r。s は cold で 0、戻せたら L、それ以外は r
            const drafted = rnd.boolean();
            var s_valid: u64 = 0;
            if (drafted) {
                const rg = plan.resumeAt(r, n, if (p.restore_live) |g| .{ g.vlo, g.vhi } else null, if (p.restore_snap) |g| .{ g.vlo, g.vhi } else null, p.below, window);
                const L = if (n > window + 1) n - (window + 1) else 0;
                try t.expectEqual(r, rg.pos);
                switch (rg.kind) {
                    .none => try t.expectEqual(if (r == 0) 0 else r, rg.s),
                    .live, .saved => try t.expectEqual(L, rg.s),
                    .partial => try t.expectEqual(r, rg.s),
                }
                if (rg.kind == .none) try t.expect(r == 0 or L >= r);
                if (rg.kind != .none) try t.expect(r > 0 and L < r);
                s_valid = rg.s;
            }
            var mid = try Before.take(&m);
            defer mid.deinit();

            // how it ends: stopped in its prompt after some chunks (V in [r, n)), or the prompt and every output but
            // the last (MTP: kv_end a few rows short)
            // 終わり方: prompt の途中で止まる (V は [r, n))、または prompt と最後以外の出力 (MTP: kv_end は数行手前)
            const stopped = rnd.uintLessThan(u32, 8) == 0;
            out.clearRetainingCapacity();
            seq.clearRetainingCapacity();
            var pe: u64 = undefined;
            var kv: u64 = undefined;
            if (stopped) {
                const done_rows = r + rnd.uintLessThan(u64, n - r);
                try seq.appendSlice(gpa, pr.items[0..done_rows]);
                pe = done_rows;
                kv = done_rows;
            } else {
                try userText(&out, rnd, 1 + rnd.uintLessThan(usize, want));
                try seq.appendSlice(gpa, pr.items);
                try seq.appendSlice(gpa, out.items[0 .. out.items.len - 1]);
                const vv: u64 = seq.items.len;
                pe = n;
                kv = if (cf.mtp) vv - rnd.uintLessThan(u64, @min(9, vv - r + 1)) else vv;
            }
            const V: u64 = seq.items.len;
            try plan.commit(&m, &p, seq.items, pe, kv);
            try m.check(null);
            if (V == 0) {
                try t.expect(m.find(p.X) == null);
            } else {
                try expectSlot(&m, p.X, V, kv, pe, m.clock, p.table[0..plan.ceilDiv(V, 256)]);
                try t.expectEqualSlices(i64, seq.items, slotOf(&m, p.X).seq.items);
            }
            // 4.5: the blocks written from j0 on and wholly below kv_end get a new generation, no other block
            // 4.5: j0 以降で書いて kv_end の手前に丸ごと入るブロックだけ世代を進める
            for (mid.gen, m.gen, 0..) |g0, g1, b| {
                var bump: u64 = 0;
                for (p.table[j0..], j0..) |pb, j| if (pb == b and (j + 1) * 256 <= kv) {
                    bump = 1;
                };
                try t.expectEqual(g0 + bump, g1);
            }
            for (mid.slots) |*f| if (f.id != p.X) try f.expectSame(&m);
            for (mid.saves, m.saves) |old, now| {
                if (V == 0 and old.owner != null and old.owner.? == p.X) {
                    try t.expectEqual(@as(?u64, null), now.owner);
                } else try t.expectEqual(old, now);
            }

            // 4.6.2 the save table: a drafted request that reached the decode saves into X's save, else a free one,
            // else the least recently stored; a serial one or one stopped in its prompt saves nothing
            // 4.6.2 の保存の表: decode まで進んだ下書き付きの要求は X の写し、なければ空き、なければ最も前に保存したもの
            // へ保存する。逐次モードと prompt の途中で止まった要求は保存しない
            const saves0 = try gpa.dupe(plan.Save, m.saves);
            defer gpa.free(saves0);
            const ring0 = m.ring;
            if (drafted and !stopped) {
                const C = V + rnd.uintLessThan(u64, 8); // EOS may leave kept rows past V / EOS で V より後ろに行が残りうる
                const live = plan.Ring.live(C, V, s_valid, 8);
                const snap = plan.Ring.snap(n, s_valid);
                const lv = live.shrink(V);
                const sn = snap.shrink(V);
                var target: ?usize = null;
                if (cf.saves > 0 and (lv != null or sn != null)) {
                    target = m.saveOf(p.X) orelse blk: {
                        for (m.saves, 0..) |sv, i| if (sv.owner == null) break :blk i;
                        var o: usize = 0;
                        for (m.saves, 0..) |sv, i| if (sv.used < m.saves[o].used) {
                            o = i;
                        };
                        break :blk o;
                    };
                }
                const got = try plan.commitSave(&m, &p, live, snap, true);
                try t.expectEqual(target, got);
                for (saves0, m.saves, 0..) |old, now, i| {
                    if (target != null and target.? == i) {
                        try t.expectEqual(@as(?u64, p.X), now.owner);
                        try t.expectEqual(old.gen + 1, now.gen);
                        try t.expectEqual(m.clock, now.used);
                        try t.expectEqual(lv, now.live);
                        try t.expectEqual(sn, now.snap);
                    } else try t.expectEqual(old, now);
                }
                if (target != null and lv != null) {
                    try t.expectEqual(@as(?[2]u64, .{ target.?, m.saves[target.?].gen }), m.ring);
                } else try t.expectEqual(@as(?[2]u64, null), m.ring);
            } else {
                try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p, null, null, drafted));
                for (saves0, m.saves) |old, now| try t.expectEqual(old, now);
                try t.expectEqual(if (drafted) null else ring0, m.ring);
            }
            try m.check(null);

            se.prompt.clearRetainingCapacity();
            try se.prompt.appendSlice(gpa, pr.items);
            se.out.clearRetainingCapacity();
            try se.out.appendSlice(gpa, out.items);
            se.seq.clearRetainingCapacity();
            try se.seq.appendSlice(gpa, seq.items);
            se.sid = if (V > 0) p.X else null;
            se.pe = pe;
            se.kv = kv;
            se.done = !stopped;
        }
        try t.expect(acts[0] > 50 and acts[2] > 10);
        shares += acts[1];
    }
    try t.expect(shares > 30);
}

// ------------------------------------------------------------------------------ MTP F with hand-written values ---
// Engine events of POOL.md 4.6.1 for n = 256 and K drafts: the prompt's last chunk leaves F = n - 1; window w at P
// rewrites its backlog [P - m, P) when room > 0 (the first window's backlog is row n - 1), its chain rows
// [P, P + k - 1) hold draft-time values; a taken drafts makes P += a + 1 and the next backlog a + 1. room =
// min(K, n + max_tokens - 2 - P). V = n + outputs - 1 (the first output comes from the prefill).
// POOL.md 4.6.1 のエンジンの出来事 (n = 256、下書き K 個): prompt の最後のチャンクで F = n - 1。P の窓は room > 0 なら
// バックログ [P - m, P) を書き直し (最初の窓のバックログは行 n - 1)、連鎖の行 [P, P + k - 1) は下書きの時の値。下書きを
// a 個採用すると P += a + 1、次のバックログ a + 1。room = min(K, n + max_tokens - 2 - P)。V = n + 出力数 - 1 (最初の
// 出力は prefill から)。

test "P2: MTP F: a rejection at each position, EOS at each position, K = 1 .. 7 all taken" {
    // max_tokens 100, K = 7. Window 1 rewrites row 255 -> F 256; its 6 chain rows lie at or past F: F stays 256. a of
    // the 7 taken: P = 257 + a. Window 2 rewrites [256, 257 + a) -> F = 257 + a. Stopped after window 2's verify (all 7
    // taken): V = P = 265 + a, kv_end = 257 + a
    // max_tokens 100、K = 7。窓 1 は行 255 を書き直す -> F 256。連鎖の 6 行は F 以降: F は 256 のまま。7 個中 a 個採用:
    // P = 257 + a。窓 2 は [256, 257 + a) を書き直す -> F = 257 + a。窓 2 の verify の後 (7 個採用) で止める:
    // V = P = 265 + a、kv_end = 257 + a
    for (0..8) |aa| {
        const a: u64 = aa;
        var f = plan.MtpF.begin(0, 256);
        f.chunk(0, 256);
        try t.expectEqual(@as(u64, 255), f.F);
        f.propose(256, 1, 7);
        try t.expectEqual(@as(u64, 256), f.F);
        f.chain(256, 6);
        try t.expectEqual(@as(u64, 256), f.F);
        f.propose(257 + a, a + 1, 7);
        try t.expectEqual(257 + a, f.F);
        f.chain(257 + a, 6);
        try t.expectEqual(257 + a, f.F);
        try t.expectEqual(257 + a, f.kvEnd(265 + a));
    }
    // EOS at position e of window 1's 8 kept rows (7 taken): V = 256 + (1 + e + 1) - 1 = 257 + e while P = 264; F 256
    // 窓 1 で残した 8 行 (7 個採用) の e 番目に EOS: V = 257 + e、P = 264。F 256
    for (0..8) |ee| {
        const e: u64 = ee;
        var f = plan.MtpF.begin(0, 256);
        f.chunk(0, 256);
        f.propose(256, 1, 7);
        f.chain(256, 6);
        try t.expectEqual(@as(u64, 256), f.kvEnd(257 + e));
    }
    // EOS at position e of window 2 (window 1 took all 7: P 264, backlog 8): window 2 rewrites [256, 264) -> F 264;
    // V = 256 + (1 + 8 + e + 1) - 1 = 265 + e
    // 窓 2 の e 番目に EOS (窓 1 は 7 個採用: P 264、バックログ 8): 窓 2 は [256, 264) を書き直す -> F 264。V = 265 + e
    for (0..8) |ee| {
        const e: u64 = ee;
        var f = plan.MtpF.begin(0, 256);
        f.chunk(0, 256);
        f.propose(256, 1, 7);
        f.chain(256, 6);
        f.propose(264, 8, 7);
        f.chain(264, 6);
        try t.expectEqual(@as(u64, 264), f.kvEnd(265 + e));
    }
    // the first output is EOS (or max_tokens 1): no window, F = n - 1 = 255, V = 256
    // 最初の出力が EOS (または max_tokens 1): 窓なし、F = n - 1 = 255、V = 256
    var f0 = plan.MtpF.begin(0, 256);
    f0.chunk(0, 256);
    try t.expectEqual(@as(u64, 255), f0.kvEnd(256));
    // K = 1 .. 7, every draft taken, stopped after window 2's verify: window 2 rewrites [256, 257 + K) -> F = 257 + K;
    // V = 256 + (1 + (K + 1) + (K + 1)) - 1 = 258 + 2K
    // K = 1〜7 で全採用、窓 2 の verify の後で止める: F = 257 + K、V = 258 + 2K
    for (1..8) |kk| {
        const K: u64 = kk;
        var f = plan.MtpF.begin(0, 256);
        f.chunk(0, 256);
        f.propose(256, 1, K);
        f.chain(256, K - 1);
        try t.expectEqual(@as(u64, 256), f.F);
        f.propose(257 + K, K + 1, K);
        f.chain(257 + K, K - 1);
        try t.expectEqual(257 + K, f.kvEnd(258 + 2 * K));
    }
}

test "P2: MTP F at cancels: after a propose before its verify, at room 0, at chunk boundaries, before the first chunk" {
    // after window 1's propose, before its verify: one output (the prefill's): V = 256 = F
    // 窓 1 の propose の後・verify の前: 出力は 1 個 (prefill のもの): V = 256 = F
    var f = plan.MtpF.begin(0, 256);
    f.chunk(0, 256);
    f.propose(256, 1, 7);
    f.chain(256, 6);
    try t.expectEqual(@as(u64, 256), f.kvEnd(256));
    // window 1 took 3 (P 260, backlog 4); window 2 rewrites [256, 260) -> F 260; stopped before its verify: V = 260
    // 窓 1 は 3 個採用 (P 260、バックログ 4)。窓 2 は [256, 260) を書き直す -> F 260。verify の前で止める: V = 260
    f.propose(260, 4, 7);
    f.chain(260, 6);
    try t.expectEqual(@as(u64, 260), f.kvEnd(260));
    // room 0: max_tokens 10, K = 7 all taken (P 264, backlog 8), window 2 has room 10 - 2 - 8 = 0 (no propose); stopped
    // there: V = 264, kv_end stays 256
    // room 0: max_tokens 10、K = 7 全採用 (P 264、バックログ 8)、窓 2 は room 0 (propose なし)。そこで止める: V = 264、
    // kv_end は 256 のまま
    f = plan.MtpF.begin(0, 256);
    f.chunk(0, 256);
    f.propose(256, 1, 7);
    f.chain(256, 6);
    f.propose(264, 8, 0);
    try t.expectEqual(@as(u64, 256), f.kvEnd(264));
    // the prompt resumed at 512, chunks of 2,048 over n = 5,000: stopped before the first chunk F = resume_end; after a
    // chunk F = its end (its last row was written with the true next token)
    // 512 から再開した prompt、n = 5,000 を 2,048 ずつ: 最初のチャンクの前で止まれば F = resume_end、チャンクの後なら
    // F = その終わり (最終行も本物の次の token で書いた)
    f = plan.MtpF.begin(512, 5000);
    try t.expectEqual(@as(u64, 512), f.kvEnd(512));
    f.chunk(512, 2560);
    try t.expectEqual(@as(u64, 2560), f.kvEnd(2560));
    f.chunk(2560, 4608);
    try t.expectEqual(@as(u64, 4608), f.kvEnd(4608));
    // S1's general rule: a chunk that does not join F does not extend it, the last chunk's placeholder row still lowers
    // it (B03); a chain row below F lowers it (B02); a canonical rewrite inside [0, F) never lowers it
    // S1 の一般の規則: F につながらないチャンクは伸ばさない。最後のチャンクの仮の行は下げる (B03)。F より前の連鎖の行は
    // 下げる (B02)。[0, F) の中の正しい書き直しは下げない
    f = plan.MtpF.begin(0, 1000);
    f.chunk(768, 1000);
    try t.expectEqual(@as(u64, 0), f.F);
    f = plan.MtpF.begin(512, 5000);
    f.chain(100, 3);
    try t.expectEqual(@as(u64, 100), f.F);
    f = plan.MtpF.begin(512, 5000);
    f.chunk(0, 128);
    try t.expectEqual(@as(u64, 512), f.F);
}

test "P2: MTP n = 1 (S1): kv_end 0 whatever the decode; n = 2 for contrast; the next long prompt resumes at 0" {
    // n = 1, max_tokens 1: no window -> F = n - 1 = 0, V = 1
    // n = 1、max_tokens 1: 窓なし -> F = 0、V = 1
    var f = plan.MtpF.begin(0, 1);
    f.chunk(0, 1);
    try t.expectEqual(@as(u64, 0), f.kvEnd(1));
    // n = 1, max_tokens 2: one room-0 window (no propose): F 0, V 2
    f.propose(1, 1, 0);
    try t.expectEqual(@as(u64, 0), f.kvEnd(2));
    // n = 1, K = 7, max_tokens 20: window 1 (P 1, room 7) writes row 0 from a step -> F 0; 7 taken (P 9, backlog 8);
    // window 2 rewrites [1, 9): does not join F = 0; stopped right after that propose: F = 0, V = 9
    // n = 1、K = 7、max_tokens 20: 窓 1 は step で行 0 を書く -> F 0。7 個採用 (P 9、バックログ 8)。窓 2 は [1, 9) を
    // 書き直すが F = 0 につながらない。その propose の直後に止める: F = 0、V = 9
    f = plan.MtpF.begin(0, 1);
    f.chunk(0, 1);
    f.propose(1, 1, 7);
    f.chain(1, 6);
    try t.expectEqual(@as(u64, 0), f.F);
    f.propose(9, 8, 7);
    try t.expectEqual(@as(u64, 0), f.kvEnd(9));
    // n = 2, max_tokens 5, K = 1: the last chunk leaves F 1; window 1 rewrites row 1 -> F 2; 1 taken (P 4, backlog 2);
    // window 2 (room 5 + 2 - 2 - 4 = 1) rewrites [2, 4) -> F 4; 1 taken: 5 outputs, V = 6
    // n = 2、max_tokens 5、K = 1: 最後のチャンクで F 1。窓 1 は行 1 を書き直す -> F 2。1 個採用 (P 4、バックログ 2)。
    // 窓 2 は [2, 4) を書き直す -> F 4。1 個採用: 出力 5 個、V = 6
    f = plan.MtpF.begin(0, 2);
    f.chunk(0, 2);
    try t.expectEqual(@as(u64, 1), f.F);
    f.propose(2, 1, 1);
    try t.expectEqual(@as(u64, 2), f.F);
    f.propose(4, 2, 1);
    try t.expectEqual(@as(u64, 4), f.kvEnd(6));

    // the planner after the n = 1 request (V 9, kv_end 0): a long prompt extending it has c = min(8, 92, 0) = 0, and
    // 0 >= max(0, 1 - 17): CONTINUE at 0 in the slot's own block; its prefill makes row 0 again by the prompt rule
    // n = 1 の要求 (V 9、kv_end 0) の後のプランナ: それを伸ばした長い prompt は c = 0、0 >= max(0, 1 - 17): 自分の
    // ブロックで 0 から CONTINUE。prefill が prompt の規則で行 0 を作り直す
    var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = true });
    defer m.deinit();
    const one = [_]i64{77};
    var p1 = try step(&m, &one, 20);
    defer p1.deinit(gpa);
    const seq1 = [_]i64{77} ++ toks(8, 500);
    try commitKv(&m, &p1, &seq1, 1, 0);
    try expectSlot(&m, 0, 9, 0, 1, 1, &.{0});
    var long: [109]i64 = undefined;
    @memcpy(long[0..9], &seq1);
    fill(long[9..], 9000);
    var p2 = try step(&m, &long, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p2.action);
    try t.expectEqual(@as(u64, 9), p2.h);
    try t.expectEqual(@as(u64, 0), p2.c);
    try t.expectEqual(@as(u64, 0), p2.resume_end);
    try t.expectEqualSlices(u32, &.{0}, p2.table);
    f = plan.MtpF.begin(p2.resume_end, long.len);
    f.chunk(0, long.len);
    try t.expectEqual(@as(u64, 108), f.F);
}

test "P2: MTP serial mode: F stays n - 1 through a long output; the next turn resumes at n - 1; capacity room 0" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = true });
    defer m.deinit();
    const p = toks(300, 1);
    var p1 = try step(&m, &p, 600);
    defer p1.deinit(gpa);
    var f = plan.MtpF.begin(p1.resume_end, 300);
    f.chunk(0, 300); // serial decode writes no MTP row / 逐次モードの decode は MTP の行を書かない
    const seq = p ++ toks(499, 5000); // 500 outputs: V = 799
    try commitKv(&m, &p1, &seq, 300, f.kvEnd(seq.len));
    try expectSlot(&m, 0, 799, 299, 300, 1, &.{ 0, 1, 2, 3 });
    // the next turn sends the output back: h 799, c = min(798, 882, 299) = 299 >= 283: CONTINUE at 299, not 798
    // 次のターンは出力を送り返す: h 799、c = 299 >= 283: 798 でなく 299 から CONTINUE
    var nx: [899]i64 = undefined;
    @memcpy(nx[0..799], &seq);
    fill(nx[799..], 9000);
    var p2 = try step(&m, &nx, 1);
    defer p2.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p2.action);
    try t.expectEqual(@as(u64, 799), p2.h);
    try t.expectEqual(@as(u64, 299), p2.c);
    try t.expectEqual(@as(u64, 299), p2.resume_end);
    try finish(&m, &p2, &nx, &.{9});
    // capacity room 0: n = capacity - 2 -> want 2, write_hi = min(4094 + 2 + 8, 4096) = 4096; the one window has room 0
    // (no propose): F = n - 1 = 4093, V = n + 1 = 4095
    // capacity による room 0: n = capacity - 2 -> want 2、write_hi = 4096。窓は room 0 の 1 回 (propose なし):
    // F = 4093、V = 4095
    var m2 = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = true });
    defer m2.deinit();
    var big: [4094]i64 = undefined;
    fill(&big, 1);
    var p3 = try step(&m2, &big, 100);
    defer p3.deinit(gpa);
    try t.expectEqual(@as(u64, 4096), p3.write_hi);
    f = plan.MtpF.begin(0, 4094);
    f.chunk(0, 2048);
    f.chunk(2048, 4094);
    f.propose(4094, 1, 0);
    const seq3 = try std.mem.concat(gpa, i64, &.{ &big, &[_]i64{5} });
    defer gpa.free(seq3);
    try t.expectEqual(@as(u64, 4093), f.kvEnd(seq3.len));
    try commitKv(&m2, &p3, seq3, 4094, f.kvEnd(seq3.len));
    try expectSlot(&m2, 0, 4095, 4093, 4094, 1, m2.slots.items[0].blocks.items);
    try t.expectEqual(@as(usize, 16), m2.slots.items[0].blocks.items.len);
}

// --------------------------------------------------------------------------------- DFlash2 records and saves ---

test "P2: DFlash2 records around the ring's wrap, the live ring's lowest block, EOS rows past the output" {
    // live keep_lo = max(0, C - 4,096 + 8) (the draft pass pre-writes [C, C + 8)): C 4,088 -> 0, 4,089 -> 1, one lap
    // 4,096 -> 8, two laps 8,192 -> 4,104
    // live の keep_lo = max(0, C - 4,096 + 8) (draft pass が [C, C + 8) に先書きする): 4,088 -> 0、4,089 -> 1、1 周
    // 4,096 -> 8、2 周 8,192 -> 4,104
    try t.expectEqual(@as(u64, 0), plan.Ring.live(4088, 4088, 0, 8).keep_lo);
    try t.expectEqual(@as(u64, 1), plan.Ring.live(4089, 4089, 0, 8).keep_lo);
    try t.expectEqual(@as(u64, 8), plan.Ring.live(4096, 4096, 0, 8).keep_lo);
    try t.expectEqual(@as(u64, 4104), plan.Ring.live(8192, 8192, 0, 8).keep_lo);
    // the snapshot has no pre-written block: n 4,096 -> 0, 4,097 -> 1, 10,000 -> 5,904 (the live ring at 10,000: 5,912)
    // prompt 後の写しには先書きの余白がない: 4,096 -> 0、4,097 -> 1、10,000 -> 5,904 (live の 10,000 は 5,912)
    try t.expectEqual(@as(u64, 0), plan.Ring.snap(4096, 0).keep_lo);
    try t.expectEqual(@as(u64, 1), plan.Ring.snap(4097, 0).keep_lo);
    try t.expectEqual(@as(u64, 5904), plan.Ring.snap(10000, 0).keep_lo);
    const g = plan.Ring.live(10000, 10000, 0, 8);
    try t.expectEqual(plan.Ring{ .phys_end = 10000, .keep_lo = 5912, .vlo = 5912, .vhi = 10000 }, g);
    try t.expectEqual(@as(u64, 7000), plan.Ring.live(10000, 10000, 7000, 8).vlo); // vlo = max(keep_lo, s)
    try t.expectEqual(@as(u64, 7000), plan.Ring.snap(10000, 7000).vlo);
    // EOS: 8 rows kept at P 20,000 and 3 outputs (EOS last): C 20,008, V 20,002 -> vhi 20,002, phys_end 20,008
    // EOS: P 20,000 で 8 行を残し出力 3 個 (最後が EOS): C 20,008、V 20,002 -> vhi 20,002、phys_end 20,008
    try t.expectEqual(plan.Ring{ .phys_end = 20008, .keep_lo = 15920, .vlo = 15920, .vhi = 20002 }, plan.Ring.live(20008, 20002, 0, 8));
    // a resume needing a row of the live ring's lowest block is not live: live [5,912, 10,000); n 7,959 -> L 5,911:
    // partial, s = r; n 7,960 -> L 5,912: live, s = L
    // live の最も低い block の行が要る再開は live にならない: n 7,959 -> L 5,911: partial、s = r。n 7,960 -> live、s = L
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 7942, .pos = 7942 }, plan.resumeAt(7942, 7959, .{ g.vlo, g.vhi }, null, 10000, 2047));
    try t.expectEqual(plan.Resume{ .kind = .live, .s = 5912, .pos = 7943 }, plan.resumeAt(7943, 7960, .{ g.vlo, g.vhi }, null, 10000, 2047));
    // a snapshot of the same physical end keeps 5,904 up: it covers L 5,911
    // 同じ物理の終わりの prompt 後の写しは 5,904 から持つ: L 5,911 を覆う
    const sn = plan.Ring.snap(10000, 0);
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 5911, .pos = 7942, .copy = .{ 5911, 7942 } }, plan.resumeAt(7942, 7959, null, .{ sn.vlo, sn.vhi }, 10000, 2047));
    // a cut lowers only vhi; vhi <= vlo leaves nothing
    // 縮めると vhi だけ下がる。vhi <= vlo なら何も残らない
    try t.expectEqual(plan.Ring{ .phys_end = 10000, .keep_lo = 5912, .vlo = 5912, .vhi = 6000 }, g.shrink(6000).?);
    try t.expect(g.shrink(5912) == null);
}

test "P2: DFlash2 resumeAt S2: no save is never none when rows are needed; adjacent / overlapping / one-row-gap ranges" {
    // cold: r 0 -> none, s 0, pos 0 (whatever the saves)
    // cold: r 0 -> none、s 0、pos 0 (写しによらない)
    try t.expectEqual(plan.Resume{ .kind = .none, .s = 0, .pos = 0 }, plan.resumeAt(0, 5000, .{ 0, 5000 }, null, 5000, 2047));
    // no save and D empty (L >= r): none, not partial; s = pos = r. L == r at n = r + 2,048; at n = r + 2,047 one row is
    // needed: partial (B01). n < 2,048 with no save: D = [0, r): partial
    // 写しなしで D が空 (L >= r): none (partial でない)、s = pos = r。n = r + 2,048 で L == r。n = r + 2,047 では 1 行要る:
    // partial (B01)。n < 2,048 で写しなし: D = [0, r): partial
    try t.expectEqual(plan.Resume{ .kind = .none, .s = 100, .pos = 100 }, plan.resumeAt(100, 3000, null, null, 0, 2047));
    try t.expectEqual(plan.Resume{ .kind = .none, .s = 1000, .pos = 1000 }, plan.resumeAt(1000, 3048, null, null, 0, 2047));
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 1000, .pos = 1000 }, plan.resumeAt(1000, 3047, null, null, 0, 2047));
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 1000, .pos = 1000 }, plan.resumeAt(1000, 1017, null, null, 0, 2047));
    // n 3,000, r 2,983: D = [952, 2,983)
    const R = struct {
        fn at(live: ?[2]u64, snap: ?[2]u64, below: u64) plan.Resume {
            return plan.resumeAt(2983, 3000, live, snap, below, 2047);
        }
    };
    const part: plan.Resume = .{ .kind = .partial, .s = 2983, .pos = 2983 };
    try t.expectEqual(plan.Resume{ .kind = .live, .s = 952, .pos = 2983 }, R.at(.{ 952, 2983 }, null, 3000));
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 952, 2983 } }, R.at(null, .{ 952, 2983 }, 3000));
    // live from below L, the snapshot above: adjacent or overlapping -> saved (copy what live lacks); a one-row gap ->
    // partial
    // live が L の下から、prompt 後の写しが上: 接する・重なる -> saved (live に足りない分をコピー)。1 行の穴 -> partial
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 2000, 2983 } }, R.at(.{ 900, 2000 }, .{ 2000, 2983 }, 3000));
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 2000, 2983 } }, R.at(.{ 900, 2000 }, .{ 1500, 3000 }, 3000));
    try t.expectEqual(part, R.at(.{ 900, 2000 }, .{ 2001, 2983 }, 3000));
    // live up to r from inside D, the snapshot below
    // live が D の中から r まで、prompt 後の写しが下
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 952, 1500 } }, R.at(.{ 1500, 3000 }, .{ 952, 1500 }, 3000));
    try t.expectEqual(part, R.at(.{ 1500, 3000 }, .{ 953, 1500 }, 3000));
    // live in the middle, touching neither end: the snapshot must cover all of D
    // live が真ん中 (どちらの端にも届かない): prompt 後の写しが D 全部を覆う必要がある
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 952, 2983 } }, R.at(.{ 1200, 2500 }, .{ 900, 3000 }, 3000));
    try t.expectEqual(part, R.at(.{ 1200, 2500 }, .{ 952, 2500 }, 3000));
    // `below` cuts both ranges: rows at or past it belong to other tokens
    // `below` は両方の範囲を切る: それ以降の行は別のトークン列のもの
    try t.expectEqual(part, R.at(.{ 952, 3000 }, .{ 0, 3000 }, 2900));
    // a live range wholly below L (or ending at L) gives nothing
    // L より下 (または L で終わる) live の範囲は何も足さない
    try t.expectEqual(plan.Resume{ .kind = .saved, .s = 952, .pos = 2983, .copy = .{ 952, 2983 } }, R.at(.{ 0, 900 }, .{ 952, 2983 }, 3000));
    try t.expectEqual(part, R.at(.{ 0, 952 }, null, 3000));
}

test "P2: DFlash2 partial -> save -> resume never makes missing rows valid; a long decode moves vlo up; 10K -> 40K -> 20K" {
    // request 1: r 5,000, n 5,017, no save: partial, s = r
    // 要求 1: r 5,000、n 5,017、写しなし: partial、s = r
    const q1 = plan.resumeAt(5000, 5017, null, null, 5017, 2047);
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 5000, .pos = 5000 }, q1);
    const lv1 = plan.Ring.live(5100, 5100, q1.s, 8);
    const sn1 = plan.Ring.snap(5017, q1.s);
    try t.expectEqual(@as(u64, 5000), lv1.vlo);
    try t.expectEqual(@as(u64, 5000), sn1.vlo);
    // request 2: the 5,100 prompt again, r 5,083: D = [3,052, 5,083); live lacks [3,052, 5,000), the snapshot does not
    // have it: partial, s = 5,083 (neither 3,052 nor 5,000)
    // 要求 2: 5,100 の prompt をもう一度、r 5,083: D = [3,052, 5,083)。live に [3,052, 5,000) がなく写しにもない: partial、
    // s = 5,083 (3,052 でも 5,000 でもない)
    const q2 = plan.resumeAt(5083, 5100, .{ lv1.vlo, lv1.vhi }, .{ sn1.vlo, sn1.vhi }, 5100, 2047);
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 5083, .pos = 5083 }, q2);
    // its 17-row prefill and a short decode are saved from 5,083; the same resend stays partial
    // 17 行の prefill と短い decode は 5,083 から保存される。同じ再送は partial のまま
    const lv2 = plan.Ring.live(5110, 5110, q2.s, 8);
    const sn2 = plan.Ring.snap(5100, q2.s);
    try t.expectEqual(@as(u64, 5083), lv2.vlo);
    try t.expectEqual(plan.ResumeKind.partial, plan.resumeAt(5083, 5100, .{ lv2.vlo, lv2.vhi }, .{ sn2.vlo, sn2.vhi }, 5110, 2047).kind);
    // a long decode from there (C = V = 9,600): keep_lo 9,600 + 8 - 4,096 = 5,512 passes s: vlo 5,512; its resend
    // (r 9,583, L 7,552) is live, s = L
    // そこから長い decode (C = V = 9,600): keep_lo 5,512 が s を越える: vlo 5,512。その再送 (r 9,583、L 7,552) は live、
    // s = L
    const lv3 = plan.Ring.live(9600, 9600, q2.s, 8);
    try t.expectEqual(plan.Ring{ .phys_end = 9600, .keep_lo = 5512, .vlo = 5512, .vhi = 9600 }, lv3);
    try t.expectEqual(plan.Resume{ .kind = .live, .s = 7552, .pos = 9583 }, plan.resumeAt(9583, 9600, .{ lv3.vlo, lv3.vhi }, null, 9600, 2047));
    try t.expect(lv3.shrink(5512) == null);
    // POOL.md 4.6.2: prompt 10,000, output to 40,000, the next prompt branches inside the old output at 20,000: snapshot
    // [5,904, 10,000), live [35,912, 40,000); needed [17,969, 20,000): partial, s = r
    // POOL.md 4.6.2: prompt 10,000、出力は 40,000 まで、次の prompt は古い出力の中の 20,000 で分かれる: 写し
    // [5,904, 10,000)、live [35,912, 40,000)。必要な [17,969, 20,000) はどちらにもない: partial、s = r
    const lv = plan.Ring.live(40000, 40000, 0, 8);
    const sn = plan.Ring.snap(10000, 0);
    try t.expectEqual(@as(u64, 35912), lv.vlo);
    try t.expectEqual(@as(u64, 5904), sn.vlo);
    try t.expectEqual(plan.Resume{ .kind = .partial, .s = 20000, .pos = 20000 }, plan.resumeAt(20000, 20017, .{ lv.vlo, lv.vhi }, .{ sn.vlo, sn.vhi }, 20000, 2047));
}

test "P2: DFlash2 saves 0 / 1 / 8 and beyond; with 1 save it changes hands by LRU and is never written as its old owner's" {
    // 0 saves: nothing saved, nothing restored
    // 写し 0 個: 何も保存せず、何も戻さない
    {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 0 });
        defer m.deinit();
        const a = toks(1000, 1);
        var p1 = try step(&m, &a, 10);
        defer p1.deinit(gpa);
        try finish(&m, &p1, &a, &toks(10, 5000));
        try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p1, plan.Ring.live(1009, 1009, 0, 8), plan.Ring.snap(1000, 0), true));
        try t.expectEqual(@as(?[2]u64, null), m.ring);
        var p2 = try step(&m, &a, 10);
        defer p2.deinit(gpa);
        try t.expectEqual(plan.Action.CONTINUE, p2.action);
        try t.expectEqual(@as(?usize, null), p2.restore);
        try finish(&m, &p2, &a, &.{9});
    }
    // 1 save: A saves into it; B (unrelated) takes it over by LRU; B's next turn restores it without a copy (the live
    // ring holds it), serial, so the ring stays B's; A's next turn has nothing to restore and takes the save back as A's
    // 写し 1 個: A が保存。無関係の B が LRU で引き継ぐ。B の次のターンはコピーなしで戻す (live のリングが持つ)、逐次
    // なのでリングは B のまま。A の次のターンは戻すものがなく、写しを A のものとして取り戻す
    {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 1 });
        defer m.deinit();
        const a = toks(1000, 1);
        const oa = toks(10, 5000);
        var p1 = try step(&m, &a, 10);
        defer p1.deinit(gpa);
        try finish(&m, &p1, &a, &oa); // V 1009
        try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p1, plan.Ring.live(1009, 1009, 0, 8), plan.Ring.snap(1000, 0), true));
        const b = toks(700, 30000);
        const ob = toks(5, 6000);
        var p2 = try step(&m, &b, 10);
        defer p2.deinit(gpa);
        try t.expect(p2.restore == null);
        try finish(&m, &p2, &b, &ob); // V 704
        try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p2, plan.Ring.live(704, 704, 0, 8), plan.Ring.snap(700, 0), true));
        try t.expectEqual(@as(?u64, 1), m.saves[0].owner);
        try t.expectEqual(@as(u64, 2), m.saves[0].gen);
        try t.expectEqual(@as(?usize, null), m.saveOf(0));
        const b2 = b ++ ob[0..4].* ++ toks(50, 8000);
        var p3 = try step(&m, &b2, 10);
        defer p3.deinit(gpa);
        try t.expectEqual(plan.Action.CONTINUE, p3.action);
        try t.expectEqual(@as(u64, 704), p3.resume_end);
        try t.expectEqual(@as(?usize, 0), p3.restore);
        try t.expectEqual(@as(u64, 2), p3.restore_gen);
        try t.expect(p3.skip_copy);
        try t.expectEqual(@as(u64, 704), p3.below);
        try finish(&m, &p3, &b2, &.{9});
        try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p3, null, null, false));
        try t.expectEqual(@as(?[2]u64, .{ 0, 2 }), m.ring);
        const a2 = a ++ oa[0..9].* ++ toks(50, 9000);
        var p4 = try step(&m, &a2, 10);
        defer p4.deinit(gpa);
        try t.expectEqual(plan.Action.CONTINUE, p4.action);
        try t.expectEqual(@as(u64, 1009), p4.resume_end);
        try t.expectEqual(@as(?usize, null), p4.restore);
        const rg = plan.resumeAt(p4.resume_end, a2.len, null, null, 0, 2047);
        try t.expectEqual(plan.Resume{ .kind = .partial, .s = 1009, .pos = 1009 }, rg);
        try finish(&m, &p4, &a2, &toks(5, 7000)); // V 1063
        try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p4, plan.Ring.live(1063, 1063, rg.s, 8), plan.Ring.snap(a2.len, rg.s), true));
        try t.expectEqual(plan.Save{ .owner = 0, .gen = 3, .used = 4, .live = .{ .phys_end = 1063, .keep_lo = 0, .vlo = 1009, .vhi = 1063 }, .snap = .{ .phys_end = 1059, .keep_lo = 0, .vlo = 1009, .vhi = 1059 } }, m.saves[0]);
        try t.expectEqual(@as(?usize, null), m.saveOf(1));
        try t.expectEqual(@as(?[2]u64, .{ 0, 3 }), m.ring);
        try m.check(null);
    }
    // 8 saves, 9 slots saving once each: the free ones first, then the 9th takes the least recently stored (slot 0's)
    // 写し 8 個に 9 スロットが 1 回ずつ保存: まず空きから、9 本目は最も前に保存したもの (スロット 0 の) を取る
    {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 16, .mtp = false, .draft_saves = 8 });
        defer m.deinit();
        for (0..9) |i| {
            const pr = toks(300, 1000 * @as(i64, @intCast(i + 1)));
            var p = try step(&m, &pr, 1);
            defer p.deinit(gpa);
            try t.expectEqual(plan.Action.NEW, p.action);
            try finish(&m, &p, &pr, &.{9});
            const k = try plan.commitSave(&m, &p, plan.Ring.live(300, 300, 0, 8), plan.Ring.snap(300, 0), true);
            try t.expectEqual(@as(?usize, if (i < 8) i else 0), k);
        }
        try t.expectEqual(@as(?usize, null), m.saveOf(0));
        try t.expectEqual(@as(?usize, 0), m.saveOf(8));
        try t.expectEqual(@as(?usize, 1), m.saveOf(1));
        const pr1 = toks(300, 2000);
        var p = try step(&m, &pr1, 1);
        defer p.deinit(gpa);
        try t.expectEqual(@as(u64, 1), p.X);
        try t.expectEqual(@as(?usize, 1), p.restore);
        try finish(&m, &p, &pr1, &.{9});
        try m.check(null);
    }
}

test "P2: DFlash2: a SHARE whose source goes whole in the same plan restores the record the plan carries" {
    var m = try Meta.init(gpa, .{ .n_blocks = 8, .capacity = 1536, .max_slots = 8, .mtp = false, .draft_saves = 1 });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 8);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &toks(8, 5000)); // V 1007
    try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p1, plan.Ring.live(1008, 1007, 0, 8), plan.Ring.snap(1000, 0), true));
    const q = toks(1000, 50000);
    var p2 = try step(&m, &q, 1);
    defer p2.deinit(gpa);
    try finish(&m, &p2, &q, &.{9});
    try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p2, null, null, false)); // serial: the ring stays A's
    var b: [1000]i64 = undefined;
    @memcpy(b[0..800], a[0..800]);
    fill(b[800..], 7000);
    var p3 = try step(&m, &b, 600);
    defer p3.deinit(gpa);
    // the plan of the test above: SHARE from A, A goes whole, Q is cut to 512
    // 上の試験と同じプラン: A から SHARE、A は丸ごと消え、Q は 512 に縮む
    try t.expectEqual(plan.Action.SHARE, p3.action);
    try t.expectEqualSlices(u64, &.{0}, p3.evicted);
    // the restore is planned from A's save as it was (index 0, generation 1, its records, below = h 800, no copy: the
    // live ring holds it); the metadata frees the save with A (the copy runs before any later store, POOL.md 12.3)
    // 復元は A の写しの計画時の記録から (番号 0、世代 1、記録、below = h 800、コピーなし: live のリングが持つ)。
    // メタデータでは写しは A と一緒に空く (コピーは後の保存より先に走る、POOL.md 12.3)
    try t.expectEqual(@as(?usize, 0), p3.restore);
    try t.expectEqual(@as(u64, 1), p3.restore_gen);
    try t.expectEqual(@as(?plan.Ring, plan.Ring{ .phys_end = 1008, .keep_lo = 0, .vlo = 0, .vhi = 1007 }), p3.restore_live);
    try t.expectEqual(@as(?plan.Ring, plan.Ring{ .phys_end = 1000, .keep_lo = 0, .vlo = 0, .vhi = 1000 }), p3.restore_snap);
    try t.expectEqual(@as(u64, 800), p3.below);
    try t.expect(p3.skip_copy);
    try t.expectEqual(@as(?u64, null), m.saves[0].owner);
    try t.expectEqual(@as(u64, 1), m.saves[0].gen);
    // the drafter resumes from the carried record: D = [0, 768), live cut at below 800 covers it
    // drafter は運んだ記録から再開する: D = [0, 768)、below 800 で切った live が覆う
    const rg = plan.resumeAt(p3.resume_end, 1000, .{ p3.restore_live.?.vlo, p3.restore_live.?.vhi }, .{ p3.restore_snap.?.vlo, p3.restore_snap.?.vhi }, p3.below, 2047);
    try t.expectEqual(plan.Resume{ .kind = .live, .s = 0, .pos = 768 }, rg);
    try finish(&m, &p3, &b, &toks(5, 1)); // V 1004
    // T saves into the freed save as T's (generation 2)
    // T は空いた写しへ T のものとして保存する (世代 2)
    try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p3, plan.Ring.live(1008, 1004, rg.s, 8), plan.Ring.snap(1000, rg.s), true));
    try t.expectEqual(@as(?u64, 2), m.saves[0].owner);
    try t.expectEqual(@as(u64, 2), m.saves[0].gen);
    try t.expectEqual(@as(?[2]u64, .{ 0, 2 }), m.ring);
    try m.check(null);
}

test "P2: DFlash2: a request stopped in its prompt keeps X's old save cut at resume_end and loses the ring's identity" {
    var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 2 });
    defer m.deinit();
    const a = toks(1000, 1);
    var p1 = try step(&m, &a, 50);
    defer p1.deinit(gpa);
    try finish(&m, &p1, &a, &toks(50, 5000)); // V 1049
    try t.expectEqual(@as(?usize, 0), try plan.commitSave(&m, &p1, plan.Ring.live(1049, 1049, 0, 8), plan.Ring.snap(1000, 0), true));
    try t.expectEqual(@as(?[2]u64, .{ 0, 1 }), m.ring);
    // the 1,000 prompt again: CONTINUE at 983, restored from the record before the cut; the save is cut at 983
    // 1,000 の prompt をもう一度: 983 から CONTINUE、縮める前の記録から戻す。写しは 983 で切る
    var p2 = try step(&m, &a, 50);
    defer p2.deinit(gpa);
    try t.expectEqual(@as(u64, 983), p2.resume_end);
    try t.expectEqual(@as(?usize, 0), p2.restore);
    try t.expect(p2.skip_copy);
    try t.expectEqual(@as(u64, 1049), p2.restore_live.?.vhi);
    try t.expectEqual(@as(u64, 983), m.saves[0].live.?.vhi);
    try t.expectEqual(@as(u64, 983), m.saves[0].snap.?.vhi);
    // the drafted request stops in its prompt at 990: nothing saved, the old save stays (cut), the ring is unknown
    // 下書き付きの要求が prompt の 990 で止まる: 保存しない、古い写しは (切って) 残る、リングの中身は不明
    try commitKv(&m, &p2, a[0..990], 990, 990);
    try t.expectEqual(@as(?usize, null), try plan.commitSave(&m, &p2, null, null, true));
    try t.expectEqual(@as(?[2]u64, null), m.ring);
    try t.expectEqual(@as(?u64, 0), m.saves[0].owner);
    try t.expectEqual(@as(u64, 1), m.saves[0].gen);
    try t.expectEqual(@as(u64, 983), m.saves[0].live.?.vhi);
    try m.check(null);
    // sent again: c = min(990, 983, 990) = 983 >= 973: CONTINUE at 983; the save restored with a copy, live
    // 送り直し: c = 983 >= 973: 983 から CONTINUE。写しをコピーして戻す、live
    var p3 = try step(&m, &a, 50);
    defer p3.deinit(gpa);
    try t.expectEqual(plan.Action.CONTINUE, p3.action);
    try t.expectEqual(@as(u64, 983), p3.resume_end);
    try t.expectEqual(@as(?usize, 0), p3.restore);
    try t.expect(!p3.skip_copy);
    const rg = plan.resumeAt(983, 1000, .{ p3.restore_live.?.vlo, p3.restore_live.?.vhi }, .{ p3.restore_snap.?.vlo, p3.restore_snap.?.vhi }, p3.below, 2047);
    try t.expectEqual(plan.Resume{ .kind = .live, .s = 0, .pos = 983 }, rg);
    try finish(&m, &p3, &a, &.{9});
}

test "P2: two ranks whose saves differ only in generation get different metadata digests; the plan digest has restore_gen" {
    var m1 = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 2 });
    defer m1.deinit();
    var m2 = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = false, .draft_saves = 2 });
    defer m2.deinit();
    const a = toks(1000, 1);
    for ([_]*Meta{ &m1, &m2 }) |m| {
        var p = try step(m, &a, 10);
        defer p.deinit(gpa);
        try finish(m, &p, &a, &toks(10, 5000));
        _ = try plan.commitSave(m, &p, plan.Ring.live(1009, 1009, 0, 8), plan.Ring.snap(1000, 0), true);
    }
    try t.expectEqual(m1.digest(), m2.digest());
    // the same slot, ranges and ring identity; only the save's generation differs on one rank
    // 同じスロット・範囲・リングの身元で、1 ランクだけ写しの世代が違う
    m2.saves[0].gen += 1;
    m2.ring = .{ 0, m2.saves[0].gen };
    try t.expect(!std.mem.eql(u8, &m1.digest(), &m2.digest()));
    var q1 = try plan.makePlan(&m1, &a, 10, true);
    defer q1.deinit(gpa);
    var q2 = try plan.makePlan(&m2, &a, 10, true);
    defer q2.deinit(gpa);
    try t.expectEqual(q1.restore, q2.restore);
    try t.expect(q1.restore_gen != q2.restore_gen);
    const d = m1.digest();
    try t.expect(!std.mem.eql(u8, &q1.digest(d), &q2.digest(d)));
}

test "P2: the reused full blocks checked before the first token: all up to 9, else the last and 8 spread; all; off" {
    for ([_]plan.Verify{ .sample, .all, .off }) |vm| for ([_]usize{ 2400, 5200 }) |len| {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 8192, .max_slots = 8, .mtp = false, .verify = vm });
        defer m.deinit();
        var buf: [5200]i64 = undefined;
        const a = buf[0..len];
        fill(a, 1);
        var p1 = try step(&m, a, 1);
        defer p1.deinit(gpa);
        try finish(&m, &p1, a, &.{9});
        var p2 = try step(&m, a, 1);
        defer p2.deinit(gpa);
        // the resend resumes at n - 17: 2,383 -> 9 full blocks, 5,183 -> 20
        // 再送は n - 17 から: 2,383 -> 完成ブロック 9 個、5,183 -> 20 個
        const full: u32 = @intCast(p2.resume_end / 256);
        try t.expectEqual(@as(u32, if (len == 2400) 9 else 20), full);
        switch (vm) {
            .off => try t.expectEqual(@as(usize, 0), p2.verify.len),
            .all => {
                try t.expectEqual(@as(usize, full), p2.verify.len);
                for (p2.verify, 0..) |j, i| try t.expectEqual(@as(u32, @intCast(i)), j);
            },
            .sample => if (full <= 9) {
                try t.expectEqual(@as(usize, full), p2.verify.len);
                for (p2.verify, 0..) |j, i| try t.expectEqual(@as(u32, @intCast(i)), j);
            } else {
                try t.expectEqual(@as(usize, 9), p2.verify.len);
                try t.expectEqual(@as(u32, 0), p2.verify[0]);
                try t.expectEqual(full - 1, p2.verify[8]);
                for (1..9) |i| try t.expect(p2.verify[i - 1] < p2.verify[i]);
            },
        }
    };
}

test "P2: MTP: the continuable slot beats a sibling with a larger c that would only SHARE (POOL.md 12.2e finding 3)" {
    // With MTP the session's own slot has c capped by its kv_end (4 rows short of V), while a sibling that holds the
    // same prompt as a prefix has c = h - 1. Comparing c before rounding (POOL.md 4.1 up to v3.5) SHAREd at 768
    // instead of continuing at 871 (103 more rows, and the session's own slot left as a duplicate); the planner now
    // compares the position each slot would really resume from, so the session's slot is continued at 871. Without
    // MTP the two tie and the continuable slot wins as before.
    // MTP では会話自身のスロットの c は kv_end (V の 4 行手前) で抑えられ、同じ prompt を先頭に持つ別のスロットは
    // c = h - 1。切り下げる前の c で比べると (v3.5 までの POOL.md 4.1)、871 から続けずに 768 から SHARE していた (計算
    // し直しが 103 行増え、自身のスロットは重複として残る)。今は実際に再開する位置で比べるので、会話自身のスロットを
    // 871 から続ける。MTP なしでは同点になり、これまでどおり続けられるスロットが勝つ
    for ([_]bool{ true, false }) |mtp| {
        var m = try Meta.init(gpa, .{ .n_blocks = 40, .capacity = 4096, .max_slots = 8, .mtp = mtp });
        defer m.deinit();
        const P = toks(875, 1);
        // Y: P + 129 more (prompt_end 1,004), 43 outputs kept (V 1,047)
        // Y: P + 129 (prompt_end 1,004)、出力 43 個 (V 1,047)
        const yp = P ++ toks(129, 20000);
        var p1 = try step(&m, &yp, 50);
        defer p1.deinit(gpa);
        const ys = yp ++ toks(43, 40000);
        try commitKv(&m, &p1, &ys, 1004, if (mtp) 1042 else 1047);
        // X sends P: Y's c = 858 < 1,004 - 17: SHARE at 768; X ends with V 875 (MTP kv_end 871)
        // X が P を送る: Y の c = 858 < 1,004 - 17: 768 から SHARE。X は V 875 で終わる (MTP の kv_end 871)
        var p2 = try step(&m, &P, 1);
        defer p2.deinit(gpa);
        try t.expectEqual(plan.Action.SHARE, p2.action);
        try t.expectEqual(@as(u64, 858), p2.c);
        try t.expectEqual(@as(u64, 768), p2.resume_end);
        try commitKv(&m, &p2, &P, 875, if (mtp) 871 else 875);
        // X's next turn: P, its last output, 164 new tokens (n 1,040); both slots match h = 875
        // X の次のターン: P、最後の出力、新しい 164 token (n 1,040)。どちらのスロットも h = 875
        const xp = P ++ [_]i64{99999} ++ toks(164, 30000);
        var p3 = try step(&m, &xp, 1);
        defer p3.deinit(gpa);
        try t.expectEqual(@as(u64, 875), p3.h);
        if (mtp) {
            // Y: c = min(874, 1023, 1042) = 874, not continuable: it would SHARE at 768; X: c = min(874, 1023, 871) =
            // 871, continuable, its block 3 private: it resumes at 871 -> X, CONTINUE at 871
            // Y: c = 874 (続けられない、768 で SHARE になる)。X: c = 871 (続けられる、ブロック 3 は専有): 871 から再開 ->
            // X を 871 から CONTINUE
            try t.expectEqual(plan.Action.CONTINUE, p3.action);
            try t.expectEqual(@as(?u64, 1), p3.S);
            try t.expectEqual(@as(u64, 871), p3.c);
            try t.expectEqual(@as(u64, 871), p3.resume_end);
            try t.expectEqual(@as(u64, 1), p3.X);
        } else {
            // both c = 875; the continuable X wins the tie: CONTINUE at 875
            // どちらも c = 875。続けられる X が同点で勝つ: 875 から CONTINUE
            try t.expectEqual(plan.Action.CONTINUE, p3.action);
            try t.expectEqual(@as(?u64, 1), p3.S);
            try t.expectEqual(@as(u64, 875), p3.resume_end);
        }
    }
}
