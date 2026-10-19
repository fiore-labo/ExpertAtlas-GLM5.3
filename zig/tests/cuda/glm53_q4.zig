//! `tf-glm53-q4 DRAFT_DIR RANK WORLD ORACLE.json`: RANK's DFlash2 drafter weights quantized to 4 bits on the host, each
//! tensor's sha256 compared with ORACLE.json, made by a Python reference implementation that this repository does not
//! include. No GPU work.
//! RANK の DFlash2 drafter の重みをホストで 4 bit に量子化し、各テンソルの sha256 を ORACLE.json と比べる。
//! ORACLE.json は Python の参照実装が作るもので、参照実装はこのリポジトリに含まない。GPU は使わない。
const std = @import("std");
const glm = @import("glm_moe_dsa");

fn hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("usage: tf-glm53-q4 DRAFT_DIR RANK WORLD ORACLE.json\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(usize, args[2], 10);
    const world = try std.fmt.parseInt(usize, args[3], 10);
    const t0 = std.Io.Clock.awake.now(io);
    var w = try glm.draft_weights.load(gpa, io, args[1], rank, world);
    defer w.close(io);
    std.debug.print("[q4] rank {d}: drafter quantized in {d} ms ({d} layers, taps {any})\n", .{ rank, t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(), w.layers.len, w.cfg.taps[0..w.cfg.n_taps] });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[4], a, .limited(1 << 26));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    const shared = parsed.value.object.get("shared").?.object;
    var rk: [8]u8 = undefined;
    const mine = parsed.value.object.get("ranks").?.object.get(try std.fmt.bufPrint(&rk, "{d}", .{rank})).?.object;
    var ok: usize = 0;
    var bad: usize = 0;
    const Check = struct {
        fn one(name: []const u8, p: glm.q4.Packed, ref: std.json.ObjectMap, good: *usize, wrong: *usize) void {
            const parts = [3]struct { k: []const u8, b: []const u8 }{
                .{ .k = "weight", .b = std.mem.sliceAsBytes(p.words) },
                .{ .k = "scales", .b = std.mem.sliceAsBytes(p.scales) },
                .{ .k = "biases", .b = std.mem.sliceAsBytes(p.biases) },
            };
            for (parts) |x| {
                const h = hex(x.b);
                if (std.mem.eql(u8, &h, ref.get(x.k).?.string)) good.* += 1 else {
                    wrong.* += 1;
                    std.debug.print("[q4] {s}.{s}: zig {s} python {s}\n", .{ name, x.k, h[0..16], ref.get(x.k).?.string[0..16] });
                }
            }
        }
    };
    if (rank == 0) Check.one("fc", w.fc, shared.get("fc").?.object, &ok, &bad);
    var nb: [32]u8 = undefined;
    for (w.layers, 0..) |L, i| {
        if (rank == 0) {
            Check.one("a_kp", L.a_kp, shared.get(try std.fmt.bufPrint(&nb, "{d}.a_kp", .{i})).?.object, &ok, &bad);
            Check.one("m_kp", L.m_kp, shared.get(try std.fmt.bufPrint(&nb, "{d}.m_kp", .{i})).?.object, &ok, &bad);
        }
        const fields = .{ .{ "qkv", L.qkv }, .{ "kv", L.kv }, .{ "o", L.o }, .{ "gu", L.gu }, .{ "down", L.down } };
        inline for (fields) |fp| Check.one(fp[0], fp[1], mine.get(try std.fmt.bufPrint(&nb, "{d}.{s}", .{ i, fp[0] })).?.object, &ok, &bad);
    }
    std.debug.print("[q4] rank {d}: {d} tensors equal, {d} different -> Q4 GATE {s}\n", .{ rank, ok, bad, if (bad == 0) "PASS" else "FAIL" });
    return if (bad == 0) 0 else 1;
}
