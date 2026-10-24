//! `tf-glm53-batch RANK WORLD MASTER CKPT STORED_WORLD LAYERS` (one process a GPU): decode windows
//! over several conversations at once. The first LAYERS layers loaded once; CONVS conversations of
//! CONTEXT positions each prefilled (different prompts) and copied into caches of their own; conversation j decodes at
//! position CONTEXT - 32 - 13 j. For every window size r of SHAPES: each conversation alone (`windowStep`, r rows);
//! for every shape c x r: one window over conversations 0 .. c - 1, r rows each (`windowMulti`). Rank 0 prints a P0C
//! line a measurement (the round's time; per conversation and in total against r rows alone); every rank prints its
//! P0C check line (each conversation's picks in the shared window against the same conversation alone).
//! 1 GPU に 1 プロセス。複数の会話をまとめた decode ウィンドウ。先頭 LAYERS 層を
//! 1 回だけ読み込む。CONTEXT 位置の会話を CONVS 個 prefill し (prompt はそれぞれ別)、それぞれ専用のキャッシュに写す。
//! 会話 j は位置 CONTEXT - 32 - 13 j で decode する。SHAPES の各ウィンドウの大きさ r について各会話の単独
//! (`windowStep`、r 行)、各形 c x r について会話 0 .. c - 1 を r 行ずつまとめたウィンドウ 1 つ (`windowMulti`)。rank 0
//! は測定ごとに P0C 行 (1 ラウンドの時間、r 行の単独に対する 1 会話あたりと合計)、全ランクは P0C check 行 (まとめた
//! ウィンドウでの各会話の選択を、同じ会話の単独と比べる) を出す。
//!
//! Environment (TF_GLM53_BATCH_*): CONTEXT (32768), CHUNK (the engine's prompt rows), SHAPES
//! (1x1,2x1,3x1,4x1,1x4,2x4,4x4,1x8,2x8), CONVS (the widest c of SHAPES), ROUNDS (50), CHECK_ROUNDS (16), COMM
//! (per,one,one_absorb,par: the `forward.MultiComm` variants each shape runs; DCP 1 runs `per` alone). A shape takes
//! up to forward.multi_rows rows (c x r), past 16 under `par` alone. COMM_PROBE (a list of row counts, e.g.
//! 8,16,24,32): the window's collectives alone at each count (`commProbe`), then exit (no conversations). AR2_ROWS:
//! windows of this many rows or more reduce in two shots (forward.Ctx.ar2_rows; default off). PROFILE=1: each shape's
//! shared window alone (the first COMM listed, 5 + ROUNDS rounds, no alone windows or checks), for a profiler.
//! 環境変数 (TF_GLM53_BATCH_*): CONTEXT (32768)、CHUNK (エンジンの prompt の行数)、SHAPES
//! (1x1,2x1,3x1,4x1,1x4,2x4,4x4,1x8,2x8)、CONVS (SHAPES の最大の c)、ROUNDS (50)、CHECK_ROUNDS (16)、COMM
//! (per,one,one_absorb,par: 各形で回す `forward.MultiComm` の種類。DCP 1 は `per` だけ)。形は forward.multi_rows 行
//! (c x r) まで。16 行を超える形は `par` だけ。COMM_PROBE (行数のリスト、例 8,16,24,32): 各行数でウィンドウの集団
//! 通信だけを測り (`commProbe`)、終わる (会話は作らない)。AR2_ROWS: この行数以上のウィンドウは 2 段で reduce する
//! (forward.Ctx.ar2_rows。既定は使わない)。PROFILE=1: 各形のまとめたウィンドウだけ (COMM の最初の種類、5 + ROUNDS
//! ラウンド、単独のウィンドウも照合もなし)。プロファイラ用。
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");
const forward = glm.forward;
const moe = glm.moe;

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn envStr(name: [:0]const u8, default: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else default;
}

fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds);
}

fn promptIds(out: []i64, seed: u64) void {
    const golden: u64 = 0x9E3779B97F4A7C15;
    for (out, 0..) |*v, i| {
        var z = seed +% (@as(u64, i) +% 1) *% golden;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z ^= z >> 31;
        v.* = @intCast(z % 149000 + 1000);
    }
}

/// /proc/meminfo's MemAvailable in MiB (0 if unreadable).
/// /proc/meminfo の MemAvailable (MiB。読めなければ 0)。
extern "c" fn fgets(s: [*]u8, n: c_int, stream: *std.c.FILE) ?[*]u8;

fn memAvailableMiB() usize {
    const f = std.c.fopen("/proc/meminfo", "r") orelse return 0;
    defer _ = std.c.fclose(f);
    var line: [256]u8 = undefined;
    while (fgets(&line, line.len, f) != null) {
        const l = std.mem.sliceTo(&line, 0);
        if (!std.mem.startsWith(u8, l, "MemAvailable:")) continue;
        var it = std.mem.tokenizeAny(u8, l[13..], " \tkB\n");
        const kb = std.fmt.parseInt(usize, it.next() orelse return 0, 10) catch return 0;
        return kb >> 10;
    }
    return 0;
}

// roce.max_bytes: the largest RoCE one-shot block (the exchange groups of `commProbe`, as forward's xGroups)
// roce.max_bytes: RoCE の one-shot の最大ブロック (`commProbe` の交換のまとまり。forward の xGroups と同じ)
const roce_max_bytes = 1 << 20;

/// TF_GLM53_BATCH_COMM_PROBE: the window's collectives alone at R rows, for each R of `list`: a graph of `n` of one
/// kind timed over M launches: the RoCE all-reduce of the bf16 partials [R, D], and over the DCP pair the candidate
/// gather [R, index_topk] int64, the query gather [R, H, 576] bf16 and the exchange (blocks of xBytes, 8-row
/// conversations grouped within roce_max_bytes). Rank 0 prints a P0C comm line an R (microseconds an operation, and
/// a 78-layer round of 2 all-reduces and one of each DCP collective a layer).
/// TF_GLM53_BATCH_COMM_PROBE: `list` の各 R について、R 行のウィンドウの集団通信だけ: 同じ種類 `n` 個の graph を
/// M 回起動して測る。bf16 の部分和 [R, D] の RoCE all-reduce、DCP の組での候補の gather [R, index_topk] int64、
/// query の gather [R, H, 576] bf16、交換 (xBytes のブロック。8 行の会話を roce_max_bytes に収まるようにまとめる)。
/// rank 0 は R ごとに P0C comm 行を出す (1 回あたりのマイクロ秒と、1 層に all-reduce 2 回と DCP の集団通信を各 1 回
/// として 78 層の 1 ラウンド)。
fn commProbe(l: *Lane, gpa: std.mem.Allocator, rank: u32, list: []const u8, M: usize) !void {
    const c = &l.c;
    const D: u32 = @intCast(c.cfg.hidden);
    const n = 64;
    const lat = try gpa.alloc(u64, @max(M, 1));
    defer gpa.free(lat);
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |w| {
        const R = try std.fmt.parseInt(u32, w, 10);
        var us: [5]f64 = @splat(0);
        for (0..5) |kind| {
            if (kind > 0 and c.dfast == null) continue;
            try cuda.graph.beginCapture(l.s, .thread_local);
            probeOps(l, kind, R, n) catch |err| {
                if (cuda.graph.endCapture(l.s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return err;
            };
            var g = try cuda.graph.endCapture(l.s);
            defer g.deinit();
            var x = try g.instantiate();
            defer x.deinit();
            for (0..3) |_| {
                try x.launchOn(l.s);
                try l.s.synchronize();
            }
            for (lat[0..M]) |*v| {
                const t0 = nowNs();
                try x.launchOn(l.s);
                try l.s.synchronize();
                v.* = nowNs() - t0;
            }
            if (c.fast) |f| if (!f.healthy()) return error.RoceFailed;
            if (c.dfast) |f| if (!f.healthy()) return error.RoceFailed;
            us[kind] = stat(gpa, lat[0..M]).mean * 1000.0 / n;
        }
        if (rank == 0) std.debug.print("P0C comm R={d}: all-reduce {d:.1} us ({d} B; two shots {d:.1} us), candidates {d:.1} us, queries {d:.1} us, exchange {d:.1} us; a 78-layer round {d:.2} ms (two shots {d:.2} ms)\n", .{ R, us[0], R * D * 2, us[4], us[1], us[2], us[3], 78.0 * (2 * us[0] + us[1] + us[2] + us[3]) / 1000.0, 78.0 * (2 * us[4] + us[1] + us[2] + us[3]) / 1000.0 });
    }
}

fn probeOps(l: *Lane, kind: usize, R: u32, n: usize) !void {
    const c = &l.c;
    const b = l.b;
    const s = l.s;
    const D: u32 = @intCast(c.cfg.hidden);
    const K: u32 = @intCast(c.cfg.index_topk);
    const H: u32 = c.heads;
    const G: u32 = if (c.dgroup) |dg| dg.world else 1;
    const xBytes = struct {
        fn f(g: u32, h: u32, r: u32) u32 {
            return g * r * h * 1024 + (g * r * h * 4 + 15) / 16 * 16;
        }
    }.f;
    for (0..n) |_| switch (kind) {
        0 => try c.fast.?.allReduce(s, b.dpart.?.ptr, b.dred.?.ptr, R * D * 2, true),
        1 => try c.dfast.?.allGather(s, b.dmine.?.ptr, b.cand.?.ptr, R * K * 8),
        2 => try c.dfast.?.allGather(s, b.qpack.?.ptr, b.qall.?.ptr, R * H * 576 * 2),
        4 => try c.fast.?.allReduce2(s, b.dpart.?.ptr, b.dmid.?.ptr, b.dred.?.ptr, R * D * 2, true),
        else => {
            const cr: u32 = @min(R, 8);
            var o: u32 = 0;
            var gn: u32 = 0;
            var off: u64 = 0;
            while (o < R) : (o += cr) {
                if (gn > 0 and xBytes(G, H, gn + cr) > roce_max_bytes) {
                    try c.dfast.?.allGather(s, b.xsend.?.ptr + off, b.xrecv.?.ptr + G * off, xBytes(G, H, gn));
                    off += xBytes(G, H, gn);
                    gn = 0;
                }
                gn += cr;
            }
            try c.dfast.?.allGather(s, b.xsend.?.ptr + off, b.xrecv.?.ptr + G * off, xBytes(G, H, gn));
        },
    };
}

const Stat = struct { n: usize = 0, p50: f64 = 0, p95: f64 = 0, max: f64 = 0, mean: f64 = 0 };

fn stat(gpa: std.mem.Allocator, ns: []const u64) Stat {
    if (ns.len == 0) return .{};
    const v = gpa.dupe(u64, ns) catch return .{};
    defer gpa.free(v);
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    var sum: f64 = 0;
    for (v) |x| sum += @floatFromInt(x);
    const ms = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    }.f;
    return .{ .n = v.len, .p50 = ms(v[v.len / 2]), .p95 = ms(v[@min(v.len - 1, v.len * 95 / 100)]), .max = ms(v[v.len - 1]), .mean = sum / @as(f64, @floatFromInt(v.len)) / 1e6 };
}

const max_shapes = 16;
const max_convs = 8;
const Shape = struct { c: u32, r: u32 };

fn parseShapes(list: []const u8, out: []Shape) !usize {
    var it = std.mem.tokenizeScalar(u8, list, ',');
    var n: usize = 0;
    while (it.next()) |w| {
        if (n == out.len) return error.TooManyShapes;
        const x = std.mem.indexOfScalar(u8, w, 'x') orelse return error.BadShape;
        const c = try std.fmt.parseInt(u32, w[0..x], 10);
        const r = try std.fmt.parseInt(u32, w[x + 1 ..], 10);
        if (c < 1 or c > max_convs or r < 1 or c * r > forward.multi_rows) return error.BadShape;
        out[n] = .{ .c = c, .r = r };
        n += 1;
    }
    return n;
}

/// Conversation j's starting ids: the same on every rank, and for the conversation alone and in a shared window.
/// 会話 j の最初の ids: 全ランクで同じ。単独でもまとめたウィンドウでも同じ。
fn startId(j: usize, i: usize) i64 {
    return @intCast(5000 + 37 * i + 1009 * j);
}

/// The window's stream, context and buffers (the engine's layers and counters: nothing else runs meanwhile).
/// ウィンドウの stream・文脈・バッファ (層とカウンタはエンジンのもの: 同時に他は走らない)。
const Lane = struct {
    e: *glm.engine.Engine,
    c: forward.Ctx,
    s: cuda.Stream,
    b: *forward.Buffers,

    fn capture(l: *Lane, seqs: []const forward.WindowSeq, multi: bool, comm: forward.MultiComm) !cuda.graph.Exec {
        const e = l.e;
        try cuda.graph.beginCapture(l.s, .thread_local);
        const r = if (multi)
            l.c.windowMulti(seqs, e.embed.ptr, e.layers, e.final_norm.ptr, e.lm_head.ptr, e.vshare, comm)
        else
            l.c.windowStep(seqs[0].rows, e.embed.ptr, e.layers, seqs[0].kc, seqs[0].ic, seqs[0].pos, seqs[0].t, e.final_norm.ptr, e.lm_head.ptr, e.vshare);
        r catch |err| {
            if (cuda.graph.endCapture(l.s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(l.s);
        defer g.deinit();
        return g.instantiate();
    }

    /// One round: the graph, the picks down, the RoCE health. Returns its wall time.
    /// 1 ラウンド: graph、選択の取り出し、RoCE の健全性。経過時間を返す。
    fn round(l: *Lane, x: *cuda.graph.Exec, R: u32, picks: []i32) !u64 {
        const t0 = nowNs();
        try x.launchOn(l.s);
        try l.b.tok_out.downloadAsync(0, std.mem.sliceAsBytes(picks[0..R]), l.s.handle);
        try l.s.synchronize();
        if (l.e.fast) |*f| if (!f.healthy()) return error.RoceFailed;
        if (l.e.dfast) |*f| if (!f.healthy()) return error.RoceFailed;
        return nowNs() - t0;
    }

    /// The chain of k rounds from the conversations' starting ids: one hash a conversation (`convs`, r rows each).
    /// 会話の最初の ids から k ラウンドの列: 会話ごとに 1 つのハッシュ (`convs`、それぞれ r 行)。
    fn chain(l: *Lane, x: *cuda.graph.Exec, convs: []const usize, r: u32, k: usize, out: []u64) !void {
        var ids: [forward.multi_rows]i64 = undefined;
        for (convs, 0..) |j, q| for (0..r) |i| {
            ids[q * r + i] = startId(j, i);
        };
        const R: u32 = @intCast(convs.len * r);
        try l.b.ids.uploadAsync(0, std.mem.sliceAsBytes(ids[0..R]), l.s.handle);
        var hs: [max_convs]std.hash.Wyhash = undefined;
        for (hs[0..convs.len]) |*h| h.* = std.hash.Wyhash.init(7);
        var pk: [forward.multi_rows]i32 = undefined;
        for (0..k) |_| {
            _ = try l.round(x, R, &pk);
            for (hs[0..convs.len], 0..) |*h, q| h.update(std.mem.sliceAsBytes(pk[q * r .. (q + 1) * r]));
        }
        for (hs[0..convs.len], 0..) |*h, q| out[q] = h.final();
    }

    fn timed(l: *Lane, x: *cuda.graph.Exec, R: u32, lat: []u64) !void {
        var pk: [forward.multi_rows]i32 = undefined;
        for (0..5) |_| _ = try l.round(x, R, &pk);
        for (lat) |*v| v.* = try l.round(x, R, &pk);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 7) {
        std.debug.print("usage: tf-glm53-batch RANK WORLD MASTER CKPT STORED_WORLD LAYERS\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const cp = envInt("TF_GLM53_BATCH_CONTEXT", 32768);
    var shapes_buf: [max_shapes]Shape = undefined;
    const shapes = shapes_buf[0..try parseShapes(envStr("TF_GLM53_BATCH_SHAPES", "1x1,2x1,3x1,4x1,1x4,2x4,4x4,1x8,2x8"), &shapes_buf)];
    var widest: u32 = 1;
    var wrows: u32 = 16; // the window buffers' rows: 16, or the widest shape past it / ウィンドウのバッファの行数: 16 か、それを超える最大の形
    for (shapes) |s| {
        widest = @max(widest, s.c);
        wrows = @max(wrows, s.c * s.r);
    }
    const probe = envStr("TF_GLM53_BATCH_COMM_PROBE", "");
    {
        var pit = std.mem.tokenizeScalar(u8, probe, ',');
        while (pit.next()) |w| wrows = @max(wrows, std.fmt.parseInt(u32, w, 10) catch return error.BadProbeRows);
        if (wrows > forward.multi_rows) return error.BadProbeRows;
    }
    const nconv: usize = @min(max_convs, @max(widest, envInt("TF_GLM53_BATCH_CONVS", widest)));
    const M = envInt("TF_GLM53_BATCH_ROUNDS", 50);
    const CR = envInt("TF_GLM53_BATCH_CHECK_ROUNDS", 16);
    const comm = envStr("TF_GLM53_BATCH_COMM", "per,one,one_absorb,par");

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var nccl = try cuda.nccl.Library.openEnv();
    defer nccl.close();
    var group = try cuda.tp.Group.init(&nccl, rank, world, try cuda.tp.parseMaster(args[3]));
    defer group.deinit();
    const t_load = nowNs();
    var o = glm.engine.withEnv(.{
        .ckpt = args[4],
        .stored_world = try std.fmt.parseInt(usize, args[5], 10),
        .layers = try std.fmt.parseInt(usize, args[6], 10),
        .capacity = 0,
        // the engine's own decode graphs (every key bucket up to the capacity) are never launched here: off unless
        // TF_GLM53_BATCH_ENGINE_GRAPHS=1 (their memory matters at long contexts)
        // エンジン自身の decode の graph (容量までの全キー bucket) はここでは使わない: TF_GLM53_BATCH_ENGINE_GRAPHS=1
        // でなければ作らない (長い文脈ではそのメモリが効く)
        .graphs = envInt("TF_GLM53_BATCH_ENGINE_GRAPHS", 0) == 1,
        .kv_fp8 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp8") or std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .kv_fp4 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .dcp = @intCast(envInt("TF_GLM53_DCP", 1)),
    });
    const n = envInt("TF_GLM53_BATCH_CHUNK", o.prompt_rows);
    o.capacity = cp + 64;
    if (n < forward.sp_min_rows or n > o.prompt_rows or cp % n != 0) {
        std.debug.print("[p0c] CHUNK {d} must be in [{d}, {d}] and divide CONTEXT {d}\n", .{ n, forward.sp_min_rows, o.prompt_rows, cp });
        return 2;
    }
    const e = try glm.engine.Engine.create(gpa, io, &driver, &ctx, &group, o);
    defer e.destroy();
    if (e.n_sp == 0 or e.fast == null) {
        std.debug.print("[p0c] needs the sequence-parallel parts (TF_GLM53_PROMPT_SP=1, SP_DCP=1, DCP 2) and RoCE (TF_GLM53_ROCE=1)\n", .{});
        return 2;
    }
    if (cp < e.cfg.index_topk + 32 + 13 * max_convs + 16) {
        std.debug.print("[p0c] CONTEXT {d} must pass index_topk {d} (+ 32 + 13 x conversations + 16)\n", .{ cp, e.cfg.index_topk });
        return 2;
    }
    // the prefill on NCCL alone (as the release's prompt chunks), the window on RoCE
    // prefill は NCCL だけ (配布版の prompt チャンクと同じ)、ウィンドウは RoCE
    for (e.sp_parts[0..e.n_sp]) |*h| {
        h.main.fast = null;
        h.main.dfast = null;
        h.side.fast = null;
        h.side.dfast = null;
    }
    if (rank == 0) std.debug.print("[p0c] engine up in {d} ms: world {d}, {d} layers, dcp {d}, chunk {d}, context {d}, conversations {d}\n", .{ (nowNs() - t_load) / 1000000, world, e.layers.len, e.dcp, n, cp, nconv });

    // the window's own buffers (wrows rows), experts scratch, stream and context
    // ウィンドウ専用のバッファ (wrows 行)、expert の作業領域、stream、文脈
    var first_moe: ?*const moe.Routed = null;
    for (e.layers) |*L| if (L.experts) |*x| {
        first_moe = x;
        break;
    };
    const keys: usize = glm.engine.bucket(e.capacity + 1, e.cfg.index_topk) orelse 2 * e.cfg.index_topk;
    const inv = try glm.rope.table(e.cfg.rope_theta, e.cfg.qk_rope);
    const had = glm.rope.hadamard();
    var b = try forward.Buffers.init(&driver, e.cfg, e.heads, e.world, wrows, e.vshare, &inv, &had, keys, e.dcp, true, wrows);
    if (e.decode_bf16) {
        b.dpart = try cuda.DeviceBuffer.alloc(&driver, @as(usize, wrows) * e.cfg.hidden * 2);
        b.dred = try cuda.DeviceBuffer.alloc(&driver, @as(usize, wrows) * e.cfg.hidden * 2);
        b.dmid = try cuda.DeviceBuffer.alloc(&driver, (@as(usize, wrows) * e.cfg.hidden * 2 / e.world + 15) / 16 * 16);
    }
    var xs = try moe.Scratch.init(&driver, first_moe orelse return error.NoMoeLayer, wrows, @intCast(e.cfg.top_k));
    const s = try cuda.Stream.initPriority(&driver, 0);
    var c = e.fw(true);
    c.stream = s;
    c.ops.stream = s;
    c.ops.pmap = 0;
    c.b = &b;
    c.xs = &xs;
    c.ucache = null;
    c.sid = 0;
    c.pek = null;
    c.pes = null;
    c.pe_ok = &.{};
    c.probe = null;
    c.tap_slot = &.{};
    c.n_taps = 0;
    c.sp = false;
    // `par`: a stream a conversation for the graph's parallel branches, the fork and join events
    // `par`: graph の並列の枝のための会話ごとの stream、分岐と合流の event
    var par_streams: [max_convs]cuda.Stream = undefined;
    var par_events: [max_convs + 1]cuda.Event = undefined;
    for (&par_streams) |*x| x.* = try cuda.Stream.initPriority(&driver, 0);
    for (&par_events) |*x| x.* = try cuda.Event.init(&driver, false);
    c.par_streams = &par_streams;
    c.par_events = &par_events;
    c.ar2_rows = @intCast(envInt("TF_GLM53_BATCH_AR2_ROWS", std.math.maxInt(u32)));
    if (rank == 0 and c.ar2_rows <= forward.multi_rows) std.debug.print("[p0c] windows of {d} rows or more: two-shot all-reduce\n", .{c.ar2_rows});
    var lane: Lane = .{ .e = e, .c = c, .s = s, .b = &b };
    if (probe.len > 0) {
        try commProbe(&lane, gpa, rank, probe, M);
        std.debug.print("P0C check rank={d}: all equal true; comm probe only\n", .{rank});
        return 0;
    }

    // the conversations: each prefilled straight into caches of its own (the SP parts write the caches they are given)
    // 会話: それぞれ専用のキャッシュへ直接 prefill する (SP の part は渡されたキャッシュに書く)
    const states = try gpa.alloc(forward.State, nconv);
    const P = try gpa.alloc(usize, nconv);
    {
        const ids = try gpa.alloc(i64, cp);
        defer gpa.free(ids);
        const done = try cuda.Event.init(&driver, false);
        var bytes: usize = 0;
        for (0..nconv) |j| {
            const t0 = nowNs();
            promptIds(ids, 7_000_000 + j);
            states[j] = try forward.State.init(gpa, &driver, e.cfg, e.layers, (e.capacity + e.dcp - 1) / e.dcp, e.kv8, e.kv4);
            var pa: usize = 0;
            while (pa < cp) : (pa += n) {
                const end = pa + n;
                const t: ?u32 = if (end > e.cfg.index_topk) @intCast(end) else null;
                _ = try forward.promptSp(e.sp_parts[0..e.n_sp], e.sp_want, e.ready, ids[pa..end], @intCast(pa), e.embed.ptr, e.layers, states[j].kc, states[j].ic, t, null, 0, false, null);
                try done.record(e.sp_parts[0].main.stream);
                try done.synchronize();
            }
            try ctx.synchronize();
            bytes = 0;
            for (states[j].kc) |x| bytes += x.len;
            for (states[j].ic) |x| if (x) |y| {
                bytes += y.len;
            };
            P[j] = cp - 32 - 13 * j;
            try states[j].pos.fill32(@intCast(P[j]), null);
            try ctx.synchronize();
            if (rank == 0) std.debug.print("[p0c] conversation {d}: {d} positions prefilled in {d} ms, decodes at {d}; its caches {d} MiB a rank; MemAvailable {d} MiB\n", .{ j, cp, (nowNs() - t0) / 1000000, P[j], bytes >> 20, memAvailableMiB() });
        }
    }

    const lat = try gpa.alloc(u64, @max(M, 1));
    var alone_ms: [17]f64 = @splat(0);
    // alone[r][j]: conversation j's chain alone with r rows
    // alone[r][j]: r 行で単独の会話 j の列
    var alone: [17][max_convs]u64 = @splat(@splat(0));
    var all_ok = true;
    var log: std.ArrayList(u8) = .empty;

    // ---- PROFILE=1: the shared windows alone ----
    // ---- PROFILE=1: まとめたウィンドウだけ ----
    if (envInt("TF_GLM53_BATCH_PROFILE", 0) == 1) {
        var cit = std.mem.tokenizeScalar(u8, comm, ',');
        const mode = std.meta.stringToEnum(forward.MultiComm, cit.next() orelse "par") orelse return error.BadComm;
        for (shapes) |sh| {
            var seqs: [max_convs]forward.WindowSeq = undefined;
            for (0..sh.c) |j| seqs[j] = .{ .rows = sh.r, .kc = states[j].kc, .ic = states[j].ic, .pos = states[j].pos.ptr, .t = glm.engine.bucket(P[j] + sh.r, e.cfg.index_topk) };
            var x = try lane.capture(seqs[0..sh.c], true, mode);
            defer x.deinit();
            try lane.timed(&x, sh.c * sh.r, lat[0..M]);
            const st = stat(gpa, lat[0..M]);
            if (rank == 0) std.debug.print("P0C profile shape={d}x{d} comm={s}: n={d} mean={d:.2} ms\n", .{ sh.c, sh.r, @tagName(mode), st.n, st.mean });
        }
        std.debug.print("P0C check rank={d}: all equal true; profile only\n", .{rank});
        return 0;
    }

    // ---- each conversation alone, every window size of SHAPES ----
    // ---- 各会話の単独、SHAPES の各ウィンドウの大きさ ----
    for (1..17) |rr| {
        const r: u32 = @intCast(rr);
        var cmax: u32 = 0;
        for (shapes) |sh| if (sh.r == r) {
            cmax = @max(cmax, sh.c);
        };
        if (cmax == 0) continue;
        for (0..cmax) |j| {
            const seq = [1]forward.WindowSeq{.{ .rows = r, .kc = states[j].kc, .ic = states[j].ic, .pos = states[j].pos.ptr, .t = glm.engine.bucket(P[j] + r, e.cfg.index_topk) }};
            var x = try lane.capture(&seq, false, .per);
            defer x.deinit();
            var h: [1]u64 = undefined;
            try lane.chain(&x, &.{j}, r, CR, &h);
            alone[rr][j] = h[0];
            var again: [1]u64 = undefined;
            try lane.chain(&x, &.{j}, r, CR, &again);
            if (again[0] != h[0]) all_ok = false;
            log.print(gpa, " alone r={d} conv{d} {x} repeat-same {any};", .{ r, j, h[0], again[0] == h[0] }) catch {};
            if (j == 0) {
                try lane.timed(&x, r, lat[0..M]);
                const st = stat(gpa, lat[0..M]);
                alone_ms[rr] = st.mean;
                if (rank == 0) std.debug.print("P0C alone r={d}: n={d} mean={d:.2} p50={d:.2} p95={d:.2} max={d:.2} ms\n", .{ r, st.n, st.mean, st.p50, st.p95, st.max });
            }
        }
    }

    // ---- control: conversation j's ids on conversation j + 1's caches and position must give another chain (else
    // the checks below could not see a window that reads the wrong conversation); rank 0 prints each conversation's
    // first picks alone ----
    // ---- 対照: 会話 j の ids を会話 j + 1 のキャッシュと位置で回すと別の列になること (そうでなければ、下の照合は
    // 別の会話を読むウィンドウを見分けられない)。rank 0 は各会話の単独の最初の選択を出す ----
    // the widest r whose alone chains cover every conversation (a shape c x r with c = nconv)
    // 単独の列が全会話にある最大の r (c = nconv の形 c x r がある)
    var r_ctl: u32 = 0;
    for (shapes) |sh| if (sh.c >= nconv) {
        r_ctl = @max(r_ctl, sh.r);
    };
    if (nconv > 1 and r_ctl > 0) {
        var differ: usize = 0;
        for (0..nconv) |j| {
            const k = (j + 1) % nconv;
            const seq = [1]forward.WindowSeq{.{ .rows = r_ctl, .kc = states[k].kc, .ic = states[k].ic, .pos = states[k].pos.ptr, .t = glm.engine.bucket(P[k] + r_ctl, e.cfg.index_topk) }};
            var x = try lane.capture(&seq, false, .per);
            defer x.deinit();
            var h: [1]u64 = undefined;
            try lane.chain(&x, &.{j}, r_ctl, CR, &h);
            if (h[0] != alone[r_ctl][j]) differ += 1;
        }
        log.print(gpa, " control r={d}: {d}/{d} chains differ on another conversation's caches;", .{ r_ctl, differ, nconv }) catch {};
        if (rank == 0) std.debug.print("P0C control r={d}: {d}/{d} chains differ on another conversation's caches\n", .{ r_ctl, differ, nconv });
    }
    {
        // each conversation's first 12 picks alone (r = 1), every rank alike
        // 各会話の単独 (r = 1) の最初の 12 個の選択 (全ランク同じ)
        var line: std.ArrayList(u8) = .empty;
        for (0..nconv) |j| {
            const seq = [1]forward.WindowSeq{.{ .rows = 1, .kc = states[j].kc, .ic = states[j].ic, .pos = states[j].pos.ptr, .t = glm.engine.bucket(P[j] + 1, e.cfg.index_topk) }};
            var x = try lane.capture(&seq, false, .per);
            defer x.deinit();
            var ids: [1]i64 = .{startId(j, 0)};
            try lane.b.ids.uploadAsync(0, std.mem.sliceAsBytes(&ids), lane.s.handle);
            var pk: [forward.multi_rows]i32 = undefined;
            line.print(gpa, " conv{d}:", .{j}) catch {};
            for (0..12) |_| {
                _ = try lane.round(&x, 1, &pk);
                line.print(gpa, " {d}", .{pk[0]}) catch {};
            }
            line.print(gpa, ";", .{}) catch {};
        }
        if (rank == 0) std.debug.print("P0C first picks alone (r=1):{s}\n", .{line.items});
    }

    // ---- shared windows ----
    // ---- まとめたウィンドウ ----
    for (shapes) |sh| for ([_]forward.MultiComm{ .per, .one, .one_absorb, .par }) |mode| {
        const tag: []const u8 = @tagName(mode);
        var listed = false;
        var it = std.mem.tokenizeScalar(u8, comm, ',');
        while (it.next()) |w| listed = listed or std.mem.eql(u8, w, tag);
        if (!listed or (mode != .per and e.dcp == 1)) continue;
        // past 16 rows: `par` alone (windowMulti)
        // 16 行を超える形は `par` だけ (windowMulti)
        if (sh.c * sh.r > 16 and mode != .par) continue;
        var seqs: [max_convs]forward.WindowSeq = undefined;
        var convs: [max_convs]usize = undefined;
        for (0..sh.c) |j| {
            seqs[j] = .{ .rows = sh.r, .kc = states[j].kc, .ic = states[j].ic, .pos = states[j].pos.ptr, .t = glm.engine.bucket(P[j] + sh.r, e.cfg.index_topk) };
            convs[j] = j;
        }
        var x = try lane.capture(seqs[0..sh.c], true, mode);
        defer x.deinit();
        var h: [max_convs]u64 = undefined;
        try lane.chain(&x, convs[0..sh.c], sh.r, CR, &h);
        var eq: usize = 0;
        for (0..sh.c) |j| {
            if (h[j] == alone[sh.r][j]) eq += 1;
        }
        if (eq != sh.c) all_ok = false;
        log.print(gpa, " shape {d}x{d} comm={s}: {d}/{d} equal alone;", .{ sh.c, sh.r, tag, eq, sh.c }) catch {};
        const R = sh.c * sh.r;
        try lane.timed(&x, R, lat[0..M]);
        const st = stat(gpa, lat[0..M]);
        const base = alone_ms[sh.r];
        if (rank == 0) std.debug.print("P0C shape={d}x{d} comm={s} rows={d}: n={d} mean={d:.2} p50={d:.2} p95={d:.2} max={d:.2} ms; against r={d} alone ({d:.2} ms): per conversation {d:.3}, total {d:.3}\n", .{ sh.c, sh.r, tag, R, st.n, st.mean, st.p50, st.p95, st.max, sh.r, base, base / st.mean, @as(f64, @floatFromInt(sh.c)) * base / st.mean });
    };
    std.debug.print("P0C check rank={d}: all equal {any};{s}\n", .{ rank, all_ok, log.items });
    return if (all_ok) 0 else 1;
}
