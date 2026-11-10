// Fiore-lab, Expert Atlas.
//! `tf-glm53-sample-pick`: the keyed sampling's candidates picked on the GPU (forward.Glue.sampleCandidates and the
//! host merge of engine.sampleRowsGpu) against the host pick of engine.sampleRows (the m best of the whole row by
//! value descending, id ascending, -0 equal to +0), over GLM-5.3's vocabulary split as 1, 2, 4 and 8 ranks would hold
//! it, with ties, signed zeros and wide ranges. One GPU; exits 1 on any difference.
//! GPU で選ぶキー付きサンプリングの候補を、ホストで選ぶ候補 (行全体の上位 m 個) と比べる。1・2・4・8 ランクに語彙を
//! 分けた形で、同点・符号付きゼロ・広い範囲を混ぜる。違いが 1 つでもあれば 1 で終わる。
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");

const vocab = 154_880;

const Cand = struct { v: f32, id: u64 };

/// a before b: higher value, then lower id (-0 and +0 compare equal, as f32 does)
fn before(_: void, a: Cand, b: Cand) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

/// The host pick: the m best of a whole row.
fn hostPick(gpa: std.mem.Allocator, row: []const f32, m: usize) ![]Cand {
    const all = try gpa.alloc(Cand, row.len);
    defer gpa.free(all);
    for (all, row, 0..) |*c, v, i| c.* = .{ .v = v, .id = i };
    std.mem.sort(Cand, all, {}, before);
    return gpa.dupe(Cand, all[0..m]);
}

fn decode(key: i64) Cand {
    const ord: i32 = @truncate(key >> 32);
    const bits: i32 = if (ord >= 0) ord else ord ^ 0x7FFFFFFF;
    return .{ .v = @bitCast(bits), .id = @intCast(0x7FFFFFFF - (key & 0xFFFFFFFF)) };
}

pub fn main(init: std.process.Init) !u8 {
    _ = init;
    const gpa = std.heap.page_allocator;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const glue = try glm.forward.Glue.load(&driver);

    var prng = std.Random.DefaultPrng.init(0x5a3b1e);
    const r = prng.random();
    const max_rows = 8;
    const logits = try gpa.alloc(f32, max_rows * vocab);
    defer gpa.free(logits);
    var dev_logits = try cuda.DeviceBuffer.alloc(&driver, max_rows * vocab * 4);
    defer dev_logits.free();
    var packed_rows = try cuda.DeviceBuffer.alloc(&driver, max_rows * vocab * 8);
    defer packed_rows.free();
    var idx = try cuda.DeviceBuffer.alloc(&driver, max_rows * 1024 * 4);
    defer idx.free();
    var sel = try cuda.DeviceBuffer.alloc(&driver, max_rows * 1024 * 8);
    defer sel.free();
    const share_rows = try gpa.alloc(f32, max_rows * vocab);
    defer gpa.free(share_rows);
    const got = try gpa.alloc(i64, 8 * max_rows * 1024);
    defer gpa.free(got);

    var cases: usize = 0;
    var bad: usize = 0;
    const worlds = [_]usize{ 1, 2, 4, 8 };
    const ms = [_]usize{ 9, 28, 264, 1024 };
    for (0..48) |trial| {
        const W = worlds[trial % worlds.len];
        const vs = vocab / W;
        const R: usize = 1 + r.uintLessThan(usize, max_rows);
        const m = @min(ms[(trial / worlds.len) % ms.len], W * vs);
        // logits: a wide spread, a run of exact ties, signed zeros, a few large magnitudes
        for (logits[0 .. R * vocab]) |*x| {
            x.* = switch (r.uintLessThan(u8, 64)) {
                0 => 3.25,
                1 => 0.0,
                2 => -0.0,
                3 => -3.25,
                4 => (r.float(f32) - 0.5) * 2e4,
                else => (r.float(f32) - 0.5) * 30.0,
            };
        }
        const k = @min(m, vs);
        for (0..W) |rank| {
            // rank's share as the engine holds it: [R, vs], ids rank * vs + j
            for (0..R) |i| @memcpy(share_rows[i * vs ..][0..vs], logits[i * vocab + rank * vs ..][0..vs]);
            try dev_logits.upload(0, std.mem.sliceAsBytes(share_rows[0 .. R * vs]));
            try glue.sampleCandidates(stream, dev_logits.ptr, packed_rows.ptr, idx.ptr, sel.ptr, @intCast(R), @intCast(vs), @intCast(k), @intCast(rank * vs));
            try stream.synchronize();
            try sel.download(0, std.mem.sliceAsBytes(got[rank * R * k ..][0 .. R * k]));
        }
        const keys = try gpa.alloc(i64, W * k);
        defer gpa.free(keys);
        for (0..R) |i| {
            for (0..W) |rank| @memcpy(keys[rank * k ..][0..k], got[(rank * R + i) * k ..][0..k]);
            std.mem.sort(i64, keys, {}, std.sort.desc(i64));
            const want = try hostPick(gpa, logits[i * vocab ..][0..vocab], m);
            defer gpa.free(want);
            for (want, keys[0..m], 0..) |w, key, j| {
                const g = decode(key);
                if (g.id != w.id or g.v != w.v) {
                    if (bad < 10) std.debug.print("trial {d} W {d} row {d} m {d} #{d}: gpu ({d}, {d}) host ({d}, {d})\n", .{ trial, W, i, m, j, g.id, g.v, w.id, w.v });
                    bad += 1;
                    break;
                }
            }
            cases += 1;
        }
    }
    std.debug.print("[sample-pick] {d} rows compared, {d} differ\n", .{ cases, bad });
    return if (bad == 0) 0 else 1;
}
