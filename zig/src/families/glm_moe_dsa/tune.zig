//! Picks each EXL3 linear shape's (K splits, warps) by timing, and one warps count for a group of one-input linears
//! when one launch beats the layers alone. The picks set the reduction order (per shape, so rows stay independent) and
//! vary by rank and start; a table written by TF_GLM53_TUNE_DUMP and read by TF_GLM53_TUNE=<prefix> repeats them.
//! EXL3 linear の形ごとに (K 分割数, warp 数) を計測で選び、同じ入力の linear の組は 1 回の起動が層ごとより速ければ
//! 共通の warp 数にする。選択は足し込み順を決め (形ごとなので行どうしは独立)、ランクと起動ごとに変わる。
//! TF_GLM53_TUNE_DUMP で書いた表を TF_GLM53_TUNE=<prefix> で読めば同じ選択になる。
const std = @import("std");
const cuda = @import("cuda");
const linear = @import("linear.zig");
const exl3 = @import("exl3.zig");
const model = @import("model.zig");

const rows = 3; // the decode window's rows / decode ウィンドウの行数
const iters = 20;
const sks = [_]u32{ 1, 2, 4, 8, 16, 32, 64 };
const wks = [_]u32{ 2, 4, 8 };

pub const Mode = union(enum) { plan, measure, table: []const u8 };

pub fn modeFromEnv() Mode {
    const v = std.mem.span(std.c.getenv("TF_GLM53_TUNE") orelse return .measure);
    if (std.mem.eql(u8, v, "0")) return .plan;
    if (std.mem.eql(u8, v, "1") or v.len == 0) return .measure;
    return .{ .table = v };
}

const Shape = struct { k: u32, n: u32, k2: u32, cb: u32 };

fn shapeOf(l: *const linear.Linear) Shape {
    return .{ .k = l.k, .n = l.n, .k2 = l.k2, .cb = l.cb };
}

/// A layer's linears, in a fixed order.
/// 層の linear を決まった順に並べる。
fn linearsOf(L: *model.Layer, out: *std.ArrayList(*linear.Linear), gpa: std.mem.Allocator) !void {
    try out.appendSlice(gpa, &.{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down });
    if (L.indexer) |*ix| try out.append(gpa, &ix.wq_b);
}

/// A layer's groups: the linears that can run as one launch.
/// 層の組: 1 回の起動で走らせうる linear。
fn groupsOf(L: *model.Layer, out: *std.ArrayList([3]?*linear.Linear), gpa: std.mem.Allocator) !void {
    try out.append(gpa, .{ &L.q_a, &L.kv_a, null });
    try out.append(gpa, .{ &L.mlp.gate, &L.mlp.up, null });
    if (L.indexer) |*ix| try out.append(gpa, .{ &ix.wq_b, &L.q_b, null });
}

fn glen(g: [3]?*linear.Linear) usize {
    return if (g[2] != null) 3 else if (g[1] != null) 2 else 1;
}

fn valid(k: u32, sk: u32, wk: u32) bool {
    const kt = k / 16;
    return kt % (sk * wk) == 0 and kt / (sk * wk) >= 2;
}

const Bench = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    x3: *const exl3.Kernels,
    stream: cuda.Stream,
    scratch: linear.Scratch,
    x: cuda.DeviceBuffer, // bf16 [rows, max K]
    y: []cuda.DeviceBuffer, // bf16 [rows, max N], one a pool member and position / pool の要素と位置ごとに 1 つ
    e0: cuda.Event,
    e1: cuda.Event,

    fn call(b: *Bench, ls: []const *const linear.Linear, outs: []const u64) !void {
        var dts: [exl3.max_jobs]exl3.DType = @splat(.bf16);
        try linear.call(b.x3, b.stream, &b.scratch, ls, b.x.ptr, .bf16, rows, outs, dts[0..ls.len]);
    }

    fn elapsedUs(b: *Bench) !f64 {
        try b.e1.synchronize();
        var ms: f32 = 0;
        try b.d.check(b.d.api.cuEventElapsedTime(&ms, b.e0.handle, b.e1.handle), "cuEventElapsedTime");
        return @as(f64, ms) * 1e3;
    }
};

/// Microseconds a call of `step` replayed from a CUDA graph, the faster of two timings of `iters` replays.
/// `step` 1 回を CUDA graph から再生したときのマイクロ秒。`iters` 回の再生を 2 度計り、速い方を取る。
fn graphUs(b: *Bench, ctx: anytype, comptime step: fn (@TypeOf(ctx), *Bench) anyerror!void) !f64 {
    try step(ctx, b);
    try b.stream.synchronize();
    try cuda.graph.beginCapture(b.stream, .thread_local);
    step(ctx, b) catch |err| {
        if (cuda.graph.endCapture(b.stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    var g = try cuda.graph.endCapture(b.stream);
    defer g.deinit();
    var x = try g.instantiate();
    defer x.deinit();
    try x.launchOn(b.stream);
    try b.stream.synchronize();
    var best: ?f64 = null;
    for (0..2) |_| {
        try b.e0.record(b.stream);
        for (0..iters) |_| try x.launchOn(b.stream);
        try b.e1.record(b.stream);
        const t = try b.elapsedUs() / iters;
        best = if (best) |v| @min(v, t) else t;
    }
    return best.?;
}

pub const Result = struct { linears: usize, retiled: usize, groups: usize, grouped: usize };

/// Picks (or reads) every linear's (sk, wk) on this rank. `layers` are this rank's; returns counts for the log.
/// このランクの全 linear の (sk, wk) を選ぶ (または読む)。`layers` はこのランクのもの。ログ用の件数を返す。
pub fn run(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, x3: *const exl3.Kernels, stream: cuda.Stream, layers: []model.Layer, mode: Mode, rank: u32) !Result {
    var lins: std.ArrayList(*linear.Linear) = .empty;
    defer lins.deinit(gpa);
    var groups: std.ArrayList([3]?*linear.Linear) = .empty;
    defer groups.deinit(gpa);
    for (layers) |*L| {
        try linearsOf(L, &lins, gpa);
        try groupsOf(L, &groups, gpa);
    }
    switch (mode) {
        .plan => return .{ .linears = 0, .retiled = 0, .groups = 0, .grouped = 0 },
        .table => |prefix| return readTable(gpa, io, prefix, rank, lins.items, groups.items),
        .measure => {},
    }
    var max_k: u32 = 0;
    var max_n: u32 = 0;
    for (lins.items) |l| {
        max_k = @max(max_k, l.k);
        max_n = @max(max_n, l.n);
    }
    var ny: usize = 0; // the output buffers allocated (only those are freed) / 確保した出力バッファ数 (それだけ解放)
    var b: Bench = .{
        .gpa = gpa, .d = d, .x3 = x3, .stream = stream,
        .scratch = try linear.Scratch.init(d, rows, max_k, @as(usize, 64) * rows * max_n),
        .x = try cuda.DeviceBuffer.alloc(d, @as(usize, rows) * max_k * 2),
        .y = try gpa.alloc(cuda.DeviceBuffer, 16 * exl3.max_jobs),
        .e0 = try cuda.Event.init(d, true), .e1 = try cuda.Event.init(d, true),
    };
    defer {
        for (&b.scratch.xh) |*x| x.free();
        for (&b.scratch.z) |*z| z.free();
        b.x.free();
        for (b.y[0..ny]) |*y| y.free();
        gpa.free(b.y);
        b.e0.deinit();
        b.e1.deinit();
    }
    for (b.y) |*y| {
        y.* = try cuda.DeviceBuffer.alloc(d, @as(usize, rows) * max_n * 2);
        ny += 1;
    }
    // normal draws * 0.1 as bf16 from a fixed seed (the values do not matter for timing)
    // 固定 seed の正規乱数 * 0.1 を bf16 で (計測では値は関係しない)
    {
        var prng = std.Random.DefaultPrng.init(0x7f4a7c15);
        const host = try gpa.alloc(u16, @as(usize, rows) * max_k);
        defer gpa.free(host);
        for (host) |*h| h.* = @truncate(@as(u32, @bitCast(prng.random().floatNorm(f32) * 0.1)) >> 16);
        try b.x.upload(0, std.mem.sliceAsBytes(host));
    }
    var res: Result = .{ .linears = 0, .retiled = 0, .groups = 0, .grouped = 0 };
    // linears: shapes in first-seen order, up to 16 layers a shape timed in rotation (DRAM, not L2)
    // linear: 形を初めて出た順に、形ごとに最大 16 層を順に回して計る (L2 でなく DRAM から読ませる)
    var done: std.ArrayList(Shape) = .empty;
    defer done.deinit(gpa);
    for (lins.items) |l0| {
        const sh = shapeOf(l0);
        if (for (done.items) |x| {
            if (std.meta.eql(x, sh)) break true;
        } else false) continue;
        try done.append(gpa, sh);
        var pool: [16]*linear.Linear = undefined;
        var np: usize = 0;
        for (lins.items) |l| if (std.meta.eql(shapeOf(l), sh) and np < pool.len) {
            pool[np] = l;
            np += 1;
        };
        const default = [2]u32{ l0.sk, l0.wk };
        var best = default;
        var best_t: ?f64 = null;
        for (sks) |sk| for (wks) |wk| {
            if (!valid(sh.k, sk, wk)) continue;
            for (pool[0..np]) |l| {
                l.sk = sk;
                l.wk = wk;
            }
            // a tiling the kernel refuses
            // カーネルが受け付けない分割
            const t = timeEach(&b, pool[0..np]) catch continue;
            if (best_t == null or t < best_t.?) {
                best = .{ sk, wk };
                best_t = t;
            }
        };
        for (lins.items) |l| if (std.meta.eql(shapeOf(l), sh)) {
            l.sk = best[0];
            l.wk = best[1];
        };
        res.linears += 1;
        if (best[0] != default[0] or best[1] != default[1]) res.retiled += 1;
    }
    // groups: kinds of groups (their layers' shapes), up to 8 groups a kind
    // 組: 組の種類 (層の形の並び) ごとに、最大 8 組を計る
    var kinds_done: std.ArrayList([3]?Shape) = .empty;
    defer kinds_done.deinit(gpa);
    for (groups.items) |g0| {
        const n = glen(g0);
        var kind: [3]?Shape = .{ null, null, null };
        for (0..n) |i| kind[i] = shapeOf(g0[i].?);
        if (for (kinds_done.items) |x| {
            if (std.meta.eql(x, kind)) break true;
        } else false) continue;
        try kinds_done.append(gpa, kind);
        var members: std.ArrayList([3]?*linear.Linear) = .empty;
        defer members.deinit(gpa);
        for (groups.items) |g| {
            var gk: [3]?Shape = .{ null, null, null };
            for (0..glen(g)) |i| gk[i] = shapeOf(g[i].?);
            if (std.meta.eql(gk, kind)) try members.append(gpa, g);
        }
        const pool = members.items[0..@min(8, members.items.len)];
        var own: [3][2]u32 = undefined;
        for (0..n) |i| own[i] = .{ pool[0][i].?.sk, pool[0][i].?.wk };
        const best = tuneGroup(&b, members.items, pool, n, own) catch null;
        const tiles = best orelse own;
        for (members.items) |g| for (0..n) |i| {
            g[i].?.sk = tiles[i][0];
            g[i].?.wk = tiles[i][1];
        };
        res.groups += 1;
        if (best != null) res.grouped += 1;
    }
    return res;
}

/// One call of each pool layer, `iters` times, timed with events (plain launches, not a graph).
/// pool の各層を 1 回ずつ呼ぶのを `iters` 回くり返し、event で計る (graph でなく通常の起動)。
fn timeEach(b: *Bench, pool: []const *linear.Linear) !f64 {
    for (pool, 0..) |l, i| try b.call(&.{l}, &.{b.y[i].ptr});
    try b.stream.synchronize();
    try b.e0.record(b.stream);
    for (0..iters) |_| for (pool, 0..) |l, i| try b.call(&.{l}, &.{b.y[i].ptr});
    try b.e1.record(b.stream);
    return try b.elapsedUs() / @as(f64, @floatFromInt(iters * pool.len));
}

const GroupCtx = struct {
    pool: []const [3]?*linear.Linear,
    n: usize,
    // one position of every group, a layer alone; when null, `grouped` decides
    // 各組の 1 か所だけを層単独で走らせる。null なら `grouped` で決める
    only: ?usize = null,
    // every group as one launch, else every layer alone
    // 各組を 1 回の起動で走らせる。false なら全層を単独で
    grouped: bool = false,

    fn run(c: GroupCtx, b: *Bench) anyerror!void {
        for (c.pool, 0..) |g, gi| {
            var ls: [3]*const linear.Linear = undefined;
            var outs: [3]u64 = undefined;
            for (0..c.n) |i| {
                ls[i] = g[i].?;
                outs[i] = b.y[gi * exl3.max_jobs + i].ptr;
            }
            if (c.only) |i| {
                try b.call(&.{ls[i]}, &.{outs[i]});
            } else if (c.grouped) {
                try b.call(ls[0..c.n], outs[0..c.n]);
            } else for (0..c.n) |i| try b.call(&.{ls[i]}, &.{outs[i]});
        }
    }
};

/// One kind of group: the best tiles at one warps-a-block when the grouped launch beats the layers alone.
/// 組の 1 種類: まとめた起動が層単独より速いとき、1 つの warp 数での最良のタイル。
fn tuneGroup(b: *Bench, members: []const [3]?*linear.Linear, pool: []const [3]?*linear.Linear, n: usize, own: [3][2]u32) !?[3][2]u32 {
    const apply = struct {
        fn f(ms: []const [3]?*linear.Linear, nn: usize, tiles: [3][2]u32) void {
            for (ms) |g| for (0..nn) |i| {
                g[i].?.sk = tiles[i][0];
                g[i].?.wk = tiles[i][1];
            };
        }
    }.f;
    errdefer apply(members, n, own);
    var best: ?[3][2]u32 = null;
    var best_t = try graphUs(b, GroupCtx{ .pool = pool, .n = n }, GroupCtx.run);
    const k = pool[0][0].?.k;
    for (wks) |wk| {
        var tiles: [3][2]u32 = undefined;
        var complete = true;
        // each layer's fastest K splits at this warps-a-block
        // この warp 数での各層の最速の K 分割数
        for (0..n) |i| {
            var fastest: ?[2]u32 = null;
            var fastest_t: f64 = 0;
            for (sks) |sk| {
                if (!valid(k, sk, wk)) continue;
                for (pool) |g| {
                    g[i].?.sk = sk;
                    g[i].?.wk = wk;
                }
                const t = try graphUs(b, GroupCtx{ .pool = pool, .n = n, .only = i }, GroupCtx.run);
                if (fastest == null or t < fastest_t) {
                    fastest = .{ sk, wk };
                    fastest_t = t;
                }
            }
            if (fastest) |f| tiles[i] = f else complete = false;
        }
        if (!complete) continue;
        apply(members, n, tiles);
        const t = try graphUs(b, GroupCtx{ .pool = pool, .n = n, .grouped = true }, GroupCtx.run);
        if (t < best_t) {
            best = tiles;
            best_t = t;
        }
    }
    apply(members, n, best orelse own);
    return best;
}

/// Writes this rank's choices to <prefix>-r<rank>.json.
/// このランクの選択を <prefix>-r<rank>.json に書く。
pub fn writeTable(gpa: std.mem.Allocator, io: std.Io, prefix: []const u8, rank: u32, layers: []model.Layer) !void {
    var lins: std.ArrayList(*linear.Linear) = .empty;
    defer lins.deinit(gpa);
    var groups: std.ArrayList([3]?*linear.Linear) = .empty;
    defer groups.deinit(gpa);
    for (layers) |*L| {
        try linearsOf(L, &lins, gpa);
        try groupsOf(L, &groups, gpa);
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.appendSlice(gpa, "{\"linears\": [");
    var seen: std.ArrayList(Shape) = .empty;
    defer seen.deinit(gpa);
    for (lins.items) |l| {
        const sh = shapeOf(l);
        if (for (seen.items) |x| {
            if (std.meta.eql(x, sh)) break true;
        } else false) continue;
        try seen.append(gpa, sh);
        try out.print(gpa, "{s}[{d}, {d}, {d}, {d}]", .{ if (seen.items.len == 1) "" else ", ", l.k, l.n, l.sk, l.wk });
    }
    try out.appendSlice(gpa, "], \"groups\": [");
    var kinds: std.ArrayList([3]?Shape) = .empty;
    defer kinds.deinit(gpa);
    for (groups.items) |g| {
        const n = glen(g);
        var kind: [3]?Shape = .{ null, null, null };
        for (0..n) |i| kind[i] = shapeOf(g[i].?);
        if (for (kinds.items) |x| {
            if (std.meta.eql(x, kind)) break true;
        } else false) continue;
        try kinds.append(gpa, kind);
        try out.appendSlice(gpa, if (kinds.items.len == 1) "{\"shapes\": [" else ", {\"shapes\": [");
        for (0..n) |i| try out.print(gpa, "{s}[{d}, {d}]", .{ if (i == 0) "" else ", ", g[i].?.k, g[i].?.n });
        try out.appendSlice(gpa, "], \"tiles\": [");
        for (0..n) |i| try out.print(gpa, "{s}[{d}, {d}]", .{ if (i == 0) "" else ", ", g[i].?.sk, g[i].?.wk });
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "]}\n");
    var name: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&name, "{s}-r{d}.json", .{ prefix, rank });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
}

const TableJson = struct {
    linears: []const [4]u32,
    groups: []const struct { shapes: []const [2]u32, tiles: []const [2]u32 },
};

/// <prefix>-r<rank>.json: every linear of a listed (k, n) takes its tiles, then every group of listed shapes its own.
/// <prefix>-r<rank>.json: 載っている (k, n) の linear はそのタイルを、続いて載っている形の組はその組のタイルを取る。
fn readTable(gpa: std.mem.Allocator, io: std.Io, prefix: []const u8, rank: u32, lins: []*linear.Linear, groups: []const [3]?*linear.Linear) !Result {
    var name: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&name, "{s}-r{d}.json", .{ prefix, rank });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(TableJson, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var res: Result = .{ .linears = parsed.value.linears.len, .retiled = 0, .groups = parsed.value.groups.len, .grouped = 0 };
    for (parsed.value.linears) |e| {
        var changed = false;
        for (lins) |l| if (l.k == e[0] and l.n == e[1]) {
            changed = changed or l.sk != e[2] or l.wk != e[3];
            l.sk = e[2];
            l.wk = e[3];
        };
        res.retiled += @intFromBool(changed);
    }
    // one launch when the tiles share their warps (linear.groupable)
    // タイルの warp 数が同じなら 1 回の起動 (linear.groupable)
    for (parsed.value.groups) |t| {
        const one = for (t.tiles) |x| {
            if (x[1] != t.tiles[0][1]) break false;
        } else true;
        res.grouped += @intFromBool(one);
    }
    for (parsed.value.groups) |t| for (groups) |g| {
        const n = glen(g);
        if (n != t.shapes.len) continue;
        const same = for (0..n) |i| {
            if (g[i].?.k != t.shapes[i][0] or g[i].?.n != t.shapes[i][1]) break false;
        } else true;
        if (!same) continue;
        for (0..n) |i| {
            g[i].?.sk = t.tiles[i][0];
            g[i].?.wk = t.tiles[i][1];
        }
    };
    return res;
}
