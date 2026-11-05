//! The RoCE one-shot runtime (glm_moe_dsa/roce.zig, zig/native/roce_proxy.c) between ranks without a model:
//! `tf-roce-test RANK WORLD HOST:PORT [ITER]`, one process a GPU. Every op's output is checked on the host: an
//! all-gather against the ranks' inputs in rank order, an all-reduce against the rank-order sum done the kernel's way
//! (fp32 adds from rank 0; bf16 inputs widened, the fp32 sum rounded to nearest even). Each op's inputs differ (op,
//! rank), so a stale slot shows. TF_TP_RING4=1: the world is a ring of four in ring order (ABI 6 relays,
//! glm53/R3RT-102.md). Then the time an op. RDMA devices from B12X_ROCE_HCA / NCCL_IB_HCA, the GID index from
//! B12X_ROCE_GID_INDEX (default -1), the library from TF_ROCE_PROXY_LIB.
//! RoCE の one-shot ランタイムをモデルなしで試す。出力は毎回ホストで確かめる (all-gather は入力をランク順に並べたもの、
//! all-reduce はカーネルと同じやり方のランク順の和)。TF_TP_RING4=1 で 4 台のリング (ABI 6 の中継)。最後に 1 回の時間。

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");
const roce = glm.roce;

const golden: u64 = 0x9E3779B97F4A7C15;

fn mix(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% golden;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Exact f32 in [-1, 1) (24 random bits), as tp_main.zig.
fn value(seed: u64, i: u64) f32 {
    const top: f32 = @floatFromInt(mix(seed, i) >> 40);
    return top / 16777216.0 * 2.0 - 1.0;
}

const Kind = enum { gather, f32, bf16 };

/// Rank `rank`'s input of op `op`: n bytes (f32 values, bf16 = their top halves, or raw bytes for a gather).
fn input(out: []u8, kind: Kind, op: u64, rank: u32) void {
    const seed = (op << 32) | rank;
    switch (kind) {
        .gather => for (out, 0..) |*b, i| {
            b.* = @truncate(mix(seed, i / 8) >> @intCast(8 * (i % 8)));
        },
        .f32 => for (0..out.len / 4) |i| std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(value(seed, i)), .little),
        .bf16 => for (0..out.len / 2) |i| {
            const bits: u32 = @bitCast(value(seed, i));
            std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate(bits >> 16), .little);
        },
    }
}

/// bf16 of an fp32, round to nearest even (cvt.rn.bf16.f32 for the finite values here).
fn toBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% 0x7FFF +% ((bits >> 16) & 1)) >> 16);
}

/// What every rank must hold after op `op`: the gather's concatenation or the rank-order sum.
fn expected(gpa: std.mem.Allocator, kind: Kind, op: u64, world: u32, n: usize) ![]u8 {
    const one = try gpa.alloc(u8, n);
    defer gpa.free(one);
    if (kind == .gather) {
        const out = try gpa.alloc(u8, n * world);
        for (0..world) |r| input(out[r * n ..][0..n], .gather, op, @intCast(r));
        return out;
    }
    const elems = if (kind == .f32) n / 4 else n / 2;
    const acc = try gpa.alloc(f32, elems);
    defer gpa.free(acc);
    for (0..world) |r| {
        input(one, kind, op, @intCast(r));
        for (0..elems) |i| {
            const v: f32 = if (kind == .f32) @bitCast(std.mem.readInt(u32, one[i * 4 ..][0..4], .little)) else @bitCast(@as(u32, std.mem.readInt(u16, one[i * 2 ..][0..2], .little)) << 16);
            acc[i] = if (r == 0) v else acc[i] + v;
        }
    }
    const out = try gpa.alloc(u8, n);
    for (0..elems) |i| {
        if (kind == .f32) std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(acc[i]), .little) else std.mem.writeInt(u16, out[i * 2 ..][0..2], toBf16(acc[i]), .little);
    }
    return out;
}

/// A wrong output, described: for a gather each source rank's block that differs (bytes, first and last offset, and
/// whether it holds that rank's input of the previous op or the one before, a stale slot); for a reduction the
/// differing elements' range (which stripe half).
/// 間違った出力の説明: gather なら違う送り元のブロックごとに (バイト数、最初と最後の位置、1 つ前・2 つ前の操作の入力と
/// 同じか = 古い slot)、reduce なら違う要素の範囲 (どちらの stripe の半分か)。
fn diagnose(gpa: std.mem.Allocator, kind: Kind, op: u64, world: u32, n: usize, got: []const u8, want: []const u8, rank: u32) !void {
    const old = try gpa.alloc(u8, n);
    defer gpa.free(old);
    if (kind == .gather) {
        for (0..world) |r| {
            const g = got[r * n ..][0..n];
            const w = want[r * n ..][0..n];
            if (std.mem.eql(u8, g, w)) continue;
            var cnt: usize = 0;
            var first: usize = n;
            var last: usize = 0;
            for (g, w, 0..) |a, b, i| if (a != b) {
                cnt += 1;
                first = @min(first, i);
                last = i;
            };
            input(old, .gather, op - 1, @intCast(r));
            const prev1 = std.mem.eql(u8, g, old);
            input(old, .gather, op - 2, @intCast(r));
            const prev2 = std.mem.eql(u8, g, old);
            const zero = std.mem.allEqual(u8, g, 0);
            std.debug.print("DIFF rank {d} op {d} gather {d} B: block of rank {d}: {d} bytes differ [{d}..{d}] (op-1 {s}, op-2 {s}, zero {s})\n", .{ rank, op, n, r, cnt, first, last, if (prev1) "yes" else "no", if (prev2) "yes" else "no", if (zero) "yes" else "no" });
        }
        return;
    }
    var cnt: usize = 0;
    var first: usize = n;
    var last: usize = 0;
    for (got[0..n], want[0..n], 0..) |a, b, i| if (a != b) {
        cnt += 1;
        first = @min(first, i);
        last = i;
    };
    std.debug.print("DIFF rank {d} op {d} {s} {d} B: {d} bytes differ [{d}..{d}] (half at {d})\n", .{ rank, op, @tagName(kind), n, cnt, first, last, (n / 16 + 1) / 2 * 16 });
}

fn launch(rt: *const roce.Runtime, s: cuda.Stream, kind: Kind, in: u64, out: u64, n: u32) !void {
    switch (kind) {
        .gather => try rt.allGather(s, in, out, n),
        .f32 => try rt.allReduce(s, in, out, n, false),
        .bf16 => try rt.allReduce(s, in, out, n, true),
    }
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: tf-roce-test RANK WORLD HOST:PORT [ITER]\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const master = try cuda.tp.parseMaster(args[3]);
    const iters: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 100;
    const env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
        }
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var lib = try cuda.nccl.Library.openEnv();
    defer lib.close();
    var g = try cuda.tp.Group.init(&lib, rank, world, master);
    defer g.deinit();
    const ring4 = if (env.get("TF_TP_RING4")) |v| std.mem.eql(u8, v, "1") else false;
    if (ring4) g.topology = .ring4;
    var names: [4][:0]const u8 = undefined;
    var nh: usize = 0;
    var it = std.mem.tokenizeScalar(u8, env.get("B12X_ROCE_HCA") orelse env.get("NCCL_IB_HCA") orelse return error.RoceNoHca, ',');
    while (it.next()) |h| : (nh += 1) {
        if (nh == names.len) break;
        names[nh] = try init.arena.allocator().dupeSentinel(u8, std.mem.trimStart(u8, h, "=^"), 0);
    }
    const gid = try std.fmt.parseInt(i32, env.get("B12X_ROCE_GID_INDEX") orelse "-1", 10);
    const stream = try cuda.Stream.init(&driver, true);
    const ks = try roce.Kernels.load(&driver);
    var rt = try roce.Runtime.init(gpa, &driver, &g, &ks, env.get("TF_ROCE_PROXY_LIB") orelse "./libroce_proxy.so", names[0..nh], gid, 300_000_000, ring4, stream);
    defer rt.deinit();
    std.debug.print("INFO rank {d}/{d} {s} hcas {d}\n", .{ rank, world, if (ring4) "ring4" else "full", nh });

    const max: usize = roce.max_bytes;
    var din = try cuda.DeviceBuffer.alloc(&driver, max);
    defer din.free();
    var dout = try cuda.DeviceBuffer.alloc(&driver, max * world);
    defer dout.free();
    const host_in = try gpa.alloc(u8, max);
    defer gpa.free(host_in);
    const host_out = try gpa.alloc(u8, max * world);
    defer gpa.free(host_out);
    const sizes = [_]u32{ 16, 48, 4096, 12288, 98304, 1 << 20 };
    var op: u64 = 1;
    // a wrong output is counted and described, and the run goes on (a rank that stopped would leave the others
    // waiting); the verdict comes at the end
    // 間違いは数えて説明し、続ける (止めたランクは他のランクを待たせる)。判定は最後
    var wrong: usize = 0;
    for (sizes) |n| {
        for ([_]Kind{ .gather, .f32, .bf16 }) |kind| {
            var bad: usize = 0;
            for (0..iters) |_| {
                op += 1;
                input(host_in[0..n], kind, op, rank);
                // on the kernels' stream: a plain upload (the legacy stream) may still be copying when a
                // non-blocking stream's kernel stages the input (a stale or half-new block went out)
                // カーネルと同じ stream で: 普通の upload (既定の stream) はコピーの途中で、non-blocking の
                // stream のカーネルが入力を写すことがある (古い・半分新しいブロックが出ていった)
                try din.uploadAsync(0, host_in[0..n], stream.handle);
                try launch(&rt, stream, kind, din.ptr, dout.ptr, n);
                try stream.synchronize();
                if (!rt.healthy()) return error.RoceUnhealthy;
                const out_n: usize = if (kind == .gather) n * world else n;
                try dout.download(0, host_out[0..out_n]);
                const want = try expected(gpa, kind, op, world, n);
                defer gpa.free(want);
                if (!std.mem.eql(u8, host_out[0..out_n], want)) {
                    bad += 1;
                    try diagnose(gpa, kind, op, world, n, host_out[0..out_n], want, rank);
                }
            }
            std.debug.print("{s} {s} {d} B: {d} ops, {d} wrong\n", .{ if (bad == 0) "CHECK" else "FAIL", @tagName(kind), n, iters, bad });
            wrong += bad;
        }
    }
    // a CUDA graph as a decode window uses them: six ops of mixed kinds and sizes (consecutive ops in the other slot,
    // each a different size) captured once, replayed with new inputs, every output checked (the sequence advances on
    // the device)
    // decode ウィンドウと同じく CUDA graph で: 種類と大きさの違う 6 つの操作 (続く操作は別の slot、大きさも別) を一度
    // capture し、入力を変えて再生し、出力を全部確かめる (seq はデバイス上で進む)
    const plan = [_]struct { kind: Kind, n: u32 }{ .{ .kind = .bf16, .n = 12288 }, .{ .kind = .gather, .n = 4096 }, .{ .kind = .f32, .n = 98304 }, .{ .kind = .bf16, .n = 48 }, .{ .kind = .bf16, .n = 1 << 20 }, .{ .kind = .gather, .n = 16 } };
    var gin: [plan.len]cuda.DeviceBuffer = undefined;
    var gout: [plan.len]cuda.DeviceBuffer = undefined;
    for (plan, 0..) |p, i| {
        gin[i] = try cuda.DeviceBuffer.alloc(&driver, p.n);
        gout[i] = try cuda.DeviceBuffer.alloc(&driver, if (p.kind == .gather) p.n * world else p.n);
    }
    defer for (&gin, &gout) |*a, *b| {
        a.free();
        b.free();
    };
    try cuda.graph.beginCapture(stream, .thread_local);
    for (plan, 0..) |p, i| try launch(&rt, stream, p.kind, gin[i].ptr, gout[i].ptr, p.n);
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    var exec = try graph.instantiate();
    defer exec.deinit();
    const replays = iters / 2;
    var gbad: usize = 0;
    for (0..replays) |t| {
        for (plan, 0..) |p, i| {
            input(host_in[0..p.n], p.kind, 1_000_000 + t * 16 + i, rank);
            try gin[i].uploadAsync(0, host_in[0..p.n], stream.handle);
        }
        try exec.launchOn(stream);
        try stream.synchronize();
        if (!rt.healthy()) return error.RoceUnhealthy;
        for (plan, 0..) |p, i| {
            const out_n: usize = if (p.kind == .gather) p.n * world else p.n;
            try gout[i].download(0, host_out[0..out_n]);
            const want = try expected(gpa, p.kind, 1_000_000 + t * 16 + i, world, p.n);
            defer gpa.free(want);
            if (!std.mem.eql(u8, host_out[0..out_n], want)) {
                gbad += 1;
                try diagnose(gpa, p.kind, 1_000_000 + t * 16 + i, world, p.n, host_out[0..out_n], want, rank);
            }
        }
    }
    std.debug.print("{s} graph of {d} mixed ops: {d} replays, {d} wrong outputs\n", .{ if (gbad == 0) "CHECK" else "FAIL", plan.len, replays, gbad });
    wrong += gbad;
    // the time an op, outputs unchecked (the same input each time)
    var start = try cuda.Event.init(&driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(&driver, true);
    defer end.deinit();
    for ([_]struct { kind: Kind, n: u32 }{ .{ .kind = .bf16, .n = 12288 }, .{ .kind = .bf16, .n = 98304 }, .{ .kind = .gather, .n = 4096 }, .{ .kind = .bf16, .n = 1 << 20 } }) |t| {
        const reps: usize = if (t.n >= 1 << 20) 200 else 2000;
        for (0..20) |_| try launch(&rt, stream, t.kind, din.ptr, dout.ptr, t.n);
        try start.record(stream);
        for (0..reps) |_| try launch(&rt, stream, t.kind, din.ptr, dout.ptr, t.n);
        try end.record(stream);
        try end.synchronize();
        const ms = try cuda.Event.elapsedMs(start, end) / @as(f32, @floatFromInt(reps));
        std.debug.print("TIME {s} {d} B: {d:.1} us an op\n", .{ @tagName(t.kind), t.n, ms * 1000.0 });
    }
    if (!rt.healthy()) return error.RoceUnhealthy;
    // every rank done before any tears its runtime down (a neighbour's last relay may still be in flight to a rank
    // whose own ops are complete): an all-gather of one word over NCCL
    // 全ランクが終えてから runtime を壊す (自分の操作が終わっても、隣への最後の中継はまだ途中のことがある)
    try lib.check(lib.api.ncclAllGather(din.ptr, dout.ptr, 4, .u8, g.comm, stream.handle), "ncclAllGather");
    try stream.synchronize();
    if (wrong != 0) {
        std.debug.print("FAIL roce {d}/{d}: {d} wrong outputs\n", .{ rank, world, wrong });
        return 1;
    }
    std.debug.print("PASS roce {d}/{d}\n", .{ rank, world });
    return 0;
}
