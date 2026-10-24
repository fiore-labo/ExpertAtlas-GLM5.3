//! `tf-glm53-concurrent RANK WORLD MASTER CKPT STORED_WORLD LAYERS` (one process a GPU): a decode
//! lane and a prefill lane at once. The first LAYERS layers loaded once; a decode lane with its own copy of the layers' EXL3 linear counters, its own
//! window buffers, experts scratch, KV caches, stream and graphs, on the RoCE runtimes alone; a prefill lane on the
//! engine's prompt buffers and sequence-parallel parts, on NCCL alone (the parts' RoCE runtimes set to null). Then,
//! for decode windows of R rows (TF_GLM53_CONCURRENT_R, default 1,8) at a context of CONTEXT positions and prefill
//! chunks of CHUNK rows written after the same context: each lane alone, time-slicing (a chunk, then q decode rounds),
//! both lanes at once with the decode stream at the greatest priority, and both at once on green contexts (the decode
//! lane on DECODE_SM SMs, the prefill lane on the rest). Rank 0 prints a P0B line a measurement; every rank prints its
//! P0B check line (decode picks and the prefill's KV hash against the lanes alone).
//! 1 GPU に 1 プロセス。decode と prefill のレーンを同時に。先頭 LAYERS 層を 1 回だけ読み込む。decode のレーンは、層の EXL3
//! linear のカウンタの自分の写し、ウィンドウのバッファ、expert の作業領域、KV キャッシュ、stream、graph を持ち、RoCE
//! だけを使う。prefill のレーンはエンジンの prompt のバッファと sequence-parallel の part を使い、NCCL だけを使う
//! (part の RoCE は null にする)。そのうえで、文脈 CONTEXT 位置での R 行の decode ウィンドウ (TF_GLM53_CONCURRENT_R、
//! 既定 1,8) と、同じ文脈の後ろに書く CHUNK 行の prefill のチャンクについて、各レーン単独、時分割 (チャンク 1 つの後に
//! decode を q ラウンド)、decode の stream を最高の優先度にした同時、green context での同時 (decode のレーンは
//! DECODE_SM 個の SM、prefill のレーンは残り) を測る。rank 0 は測定ごとに P0B 行を、全ランクは P0B check 行 (decode の
//! 選択と prefill の KV のハッシュを各レーン単独と比べる) を出す。
//!
//! Environment (TF_GLM53_CONCURRENT_*): CONTEXT (32768), CHUNK (the engine's prompt rows), CHUNKS (4, the prefill
//! chunks a measurement), ROUNDS (200, decode rounds alone), R (1,8), QUOTA (1,4,8, time-slicing's rounds a chunk),
//! MODES (alone,timeslice,priority,green), DECODE_SM (8), GREEN_FLAGS (0; 1: IGNORE_SM_COSCHEDULING),
//! DECODE_PRIORITY (the greatest), CHECK_ROUNDS (16).
//! 環境変数 (TF_GLM53_CONCURRENT_*): CONTEXT (32768)、CHUNK (エンジンの prompt の行数)、CHUNKS (4、測定 1 回の prefill
//! のチャンク数)、ROUNDS (200、decode 単独のラウンド数)、R (1,8)、QUOTA (1,4,8、時分割でチャンクごとのラウンド数)、
//! MODES (alone,timeslice,priority,green)、DECODE_SM (8)、GREEN_FLAGS (0。1 は IGNORE_SM_COSCHEDULING)、
//! DECODE_PRIORITY (最高)、CHECK_ROUNDS (16)。
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");
const forward = glm.forward;
const linear = glm.linear;
const model = glm.model;
const moe = glm.moe;

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn envStr(name: [:0]const u8, default: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else default;
}

fn listHas(list: []const u8, item: []const u8) bool {
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |x| if (std.mem.eql(u8, x, item)) return true;
    return false;
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

/// The driver's green-context entry points, looked up here (the engine's driver table does not carry them).
/// driver の green context の関数。ここで引く (エンジンの driver の表には無い)。
const DevResource = extern struct {
    typ: c_int,
    pad: [92]u8,
    sm_count: u32,
    sm_min: u32,
    sm_align: u32,
    rest: [36]u8,
};
comptime {
    std.debug.assert(@sizeOf(DevResource) == 144);
}
const GreenLanes = struct { decode: cuda.Stream, main: cuda.Stream, comm: cuda.Stream, decode_sms: u32, rest_sms: u32 };
const Green = struct {
    lib: std.DynLib,
    get_res: *const fn (c_int, *DevResource, c_int) callconv(.c) c_int,
    split: *const fn (?*DevResource, *c_uint, *const DevResource, ?*DevResource, c_uint, c_uint) callconv(.c) c_int,
    gen_desc: *const fn (*?*anyopaque, *DevResource, c_uint) callconv(.c) c_int,
    create: *const fn (*?*anyopaque, ?*anyopaque, c_int, c_uint) callconv(.c) c_int,
    stream_create: *const fn (*cuda.abi.Stream, ?*anyopaque, c_uint, c_int) callconv(.c) c_int,

    fn open() !Green {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        return .{
            .get_res = lib.lookup(@FieldType(Green, "get_res"), "cuDeviceGetDevResource") orelse return error.NoGreen,
            .split = lib.lookup(@FieldType(Green, "split"), "cuDevSmResourceSplitByCount") orelse return error.NoGreen,
            .gen_desc = lib.lookup(@FieldType(Green, "gen_desc"), "cuDevResourceGenerateDesc") orelse return error.NoGreen,
            .create = lib.lookup(@FieldType(Green, "create"), "cuGreenCtxCreate") orelse return error.NoGreen,
            .stream_create = lib.lookup(@FieldType(Green, "stream_create"), "cuGreenCtxStreamCreate") orelse return error.NoGreen,
            .lib = lib,
        };
    }

    /// Two green contexts' streams: `want` SMs (the decode lane, at priority `prio`) and the rest (main at 0 and comm
    /// at `comm_prio`, the prefill lane).
    /// green context 2 つの stream: `want` 個の SM (decode のレーン、優先度 `prio`) と残り (prefill のレーン、main は 0、
    /// comm は `comm_prio`)。
    fn lanes(g: *const Green, d: *const cuda.Driver, dev: c_int, want: u32, flags: u32, prio: i32, comm_prio: i32) !GreenLanes {
        var all: DevResource = std.mem.zeroes(DevResource);
        if (g.get_res(dev, &all, 1) != 0) return error.GreenResource;
        var grp: DevResource = std.mem.zeroes(DevResource);
        var rem: DevResource = std.mem.zeroes(DevResource);
        var n1: c_uint = 1;
        if (g.split(&grp, &n1, &all, &rem, flags, want) != 0) return error.GreenSplit;
        var da: ?*anyopaque = null;
        var db: ?*anyopaque = null;
        if (g.gen_desc(&da, &grp, 1) != 0 or g.gen_desc(&db, &rem, 1) != 0) return error.GreenDesc;
        var ga: ?*anyopaque = null;
        var gb: ?*anyopaque = null;
        if (g.create(&ga, da, dev, 1) != 0 or g.create(&gb, db, dev, 1) != 0) return error.GreenCreate;
        var s: [3]cuda.abi.Stream = .{ null, null, null };
        if (g.stream_create(&s[0], ga, 1, prio) != 0 or g.stream_create(&s[1], gb, 1, 0) != 0 or g.stream_create(&s[2], gb, 1, comm_prio) != 0) return error.GreenStream;
        return .{ .decode = .{ .d = d, .handle = s[0] }, .main = .{ .d = d, .handle = s[1] }, .comm = .{ .d = d, .handle = s[2] }, .decode_sms = grp.sm_count, .rest_sms = rem.sm_count };
    }
};

const Stat = struct { n: usize = 0, p50: f64 = 0, p95: f64 = 0, p99: f64 = 0, max: f64 = 0, mean: f64 = 0 };

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
    return .{ .n = v.len, .p50 = ms(v[v.len / 2]), .p95 = ms(v[@min(v.len - 1, v.len * 95 / 100)]), .p99 = ms(v[@min(v.len - 1, v.len * 99 / 100)]), .max = ms(v[v.len - 1]), .mean = sum / @as(f64, @floatFromInt(v.len)) / 1e6 };
}

/// The decode lane: copies of the layers with counters of their own, window buffers, experts scratch, caches; per
/// stream a context and the graphs of each window size.
/// decode のレーン: カウンタを自分で持つ層の写し、ウィンドウのバッファ、expert の作業領域、キャッシュ。stream ごとに
/// context と各ウィンドウの大きさの graph。
const Decode = struct {
    layers: []model.Layer,
    b: forward.Buffers,
    xs: moe.Scratch,
    state: forward.State,
    P: usize,
    rows: []const u32,
    // a second decode lane's own RoCE runtimes (null: the engine's)
    // 2 本目の decode のレーン自身の RoCE ランタイム (null ならエンジンのもの)
    fast: @FieldType(forward.Ctx, "fast") = null,
    dfast: @FieldType(forward.Ctx, "dfast") = null,

    const Lane = struct {
        stream: cuda.Stream,
        c: forward.Ctx,
        execs: [8]cuda.graph.Exec = undefined,
    };

    /// A decode lane over the engine's weights: the layers' structs copied with every EXL3 linear's split-K counters
    /// made anew, window buffers and experts scratch of `wrows` rows, caches of the engine's size.
    /// エンジンの重みの上の decode のレーン: 層の構造体を写して全 EXL3 linear の split-K のカウンタを作り直し、`wrows`
    /// 行のウィンドウのバッファと expert の作業領域、エンジンと同じ大きさのキャッシュ。
    fn make(gpa: std.mem.Allocator, d: *const cuda.Driver, e: *glm.engine.Engine, wrows: u32, keys: usize, inv: []const u8, had: []const u8, first_moe: *const moe.Routed, P: usize, rows: []const u32) !struct { dl: Decode, counters: usize } {
        var dl: Decode = .{
            .layers = try gpa.dupe(model.Layer, e.layers),
            .b = try forward.Buffers.init(d, e.cfg, e.heads, e.world, wrows, e.vshare, inv, had, keys, e.dcp, true, wrows),
            .xs = try moe.Scratch.init(d, first_moe, wrows, @intCast(e.cfg.top_k)),
            .state = try forward.State.init(gpa, d, e.cfg, e.layers, (e.capacity + e.dcp - 1) / e.dcp, e.kv8, e.kv4),
            .P = P,
            .rows = rows,
        };
        if (e.decode_bf16) {
            dl.b.dpart = try cuda.DeviceBuffer.alloc(d, 16 * e.cfg.hidden * 2);
            dl.b.dred = try cuda.DeviceBuffer.alloc(d, 16 * e.cfg.hidden * 2);
        }
        var n: usize = 0;
        for (dl.layers) |*L| {
            var ls: [10]?*linear.Linear = .{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down, null, null, null };
            if (L.indexer) |*ix| ls[7] = &ix.wq_b;
            if (L.peer_q_b) |*pq| ls[8] = pq;
            for (ls) |maybe| if (maybe) |l| {
                l.counters = try cuda.DeviceBuffer.alloc(d, 8 * (l.n / 128) * 4);
                try l.counters.fill8(0, null);
                n += 1;
            };
        }
        return .{ .dl = dl, .counters = n };
    }

    /// The engine's caches (the context the engine prefilled) copied into this lane's.
    /// エンジンのキャッシュ (エンジンが prefill した文脈) をこのレーンのものへ写す。
    fn copyContext(dl: *Decode, e: *glm.engine.Engine) !void {
        for (e.state.kc, dl.state.kc) |src, dst| try dst.copyFrom(0, src.ptr, src.len, null);
        for (e.state.ic, dl.state.ic) |src, dst| if (src) |sb| try dst.?.copyFrom(0, sb.ptr, sb.len, null);
    }

    fn ctxOn(dl: *Decode, e: *glm.engine.Engine, s: cuda.Stream) forward.Ctx {
        var c = e.fw(true);
        c.stream = s;
        c.ops.stream = s;
        c.ops.pmap = 0;
        c.b = &dl.b;
        c.xs = &dl.xs;
        c.ucache = null;
        c.sid = 0;
        c.pek = null;
        c.pes = null;
        c.pe_ok = &.{};
        c.probe = null;
        c.tap_slot = &.{};
        c.n_taps = 0;
        c.sp = false;
        if (dl.fast != null) {
            c.fast = dl.fast;
            c.dfast = dl.dfast;
        }
        return c;
    }

    fn lane(dl: *Decode, e: *glm.engine.Engine, s: cuda.Stream) !Lane {
        var l: Lane = .{ .stream = s, .c = dl.ctxOn(e, s) };
        try dl.state.pos.fill32(@intCast(dl.P), s.handle);
        for (dl.rows, 0..) |R, i| {
            const t = glm.engine.bucket(dl.P + R, e.cfg.index_topk);
            try cuda.graph.beginCapture(s, .thread_local);
            l.c.windowStep(R, e.embed.ptr, dl.layers, dl.state.kc, dl.state.ic, dl.state.pos.ptr, t, e.final_norm.ptr, e.lm_head.ptr, e.vshare) catch |err| {
                if (cuda.graph.endCapture(s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return err;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            l.execs[i] = try g.instantiate();
        }
        try s.synchronize();
        return l;
    }

    /// One round: the window's graph, its picks down, the RoCE health. Returns its wall time.
    /// 1 ラウンド: ウィンドウの graph、選択の取り出し、RoCE の健全性。経過時間を返す。
    fn round(dl: *Decode, e: *glm.engine.Engine, l: *Lane, ri: usize, picks: []i32) !u64 {
        const t0 = nowNs();
        try l.execs[ri].launchOn(l.stream);
        last_launch_ns = nowNs() - t0;
        try dl.b.tok_out.downloadAsync(0, std.mem.sliceAsBytes(picks[0..dl.rows[ri]]), l.stream.handle);
        try l.stream.synchronize();
        if (dl.fast) |f| {
            if (!f.healthy()) return error.RoceFailed;
            if (dl.dfast) |g| if (!g.healthy()) return error.RoceFailed;
        } else {
            if (e.fast) |*f| if (!f.healthy()) return error.RoceFailed;
            if (e.dfast) |*f| if (!f.healthy()) return error.RoceFailed;
        }
        return nowNs() - t0;
    }

    /// The window's ids reset to fixed tokens (every rank alike), so a run of rounds picks the same chain.
    /// ウィンドウの ids を決まったトークンに戻す (全ランク同じ)。これで続くラウンドは同じ列を選ぶ。
    fn resetIds(dl: *Decode, l: *Lane) !void {
        var ids: [16]i64 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(5000 + 37 * i);
        try dl.b.ids.uploadAsync(0, std.mem.sliceAsBytes(&ids), l.stream.handle);
    }
};

/// The host time the last round's graph launch took (a launch that waits on a driver lock shows here).
/// 直前のラウンドの graph の投入にかかったホストの時間 (driver のロックを待った投入はここに出る)。
threadlocal var last_launch_ns: u64 = 0;

/// The prefill lane: chunks of `n` rows written at [cp, cp + n) (the same rows each time), on the parts' streams.
/// prefill のレーン: [cp, cp + n) に書く n 行のチャンク (毎回同じ行)。part の stream で。
const Prefill = struct {
    e: *glm.engine.Engine,
    ids: []const i64,
    cp: usize,
    n: usize,
    done: cuda.Event,

    /// Enqueues the chunk at `a` (every layer, both streams); returns the host time it took.
    /// 位置 `a` のチャンクを投入する (全層、両 stream)。かかったホストの時間を返す。
    fn enqueue(p: *Prefill, a: usize, rows: usize) !u64 {
        const e = p.e;
        const t0 = nowNs();
        const end = a + rows;
        const t: ?u32 = if (end > e.cfg.index_topk) @intCast(end) else null;
        _ = try forward.promptSp(e.sp_parts[0..e.n_sp], e.sp_want, e.ready, p.ids[a..end], @intCast(a), e.embed.ptr, e.layers, e.state.kc, e.state.ic, t, null, 0, false, null);
        try p.done.record(e.sp_parts[0].main.stream);
        return nowNs() - t0;
    }

    fn wait(p: *Prefill) !void {
        try p.done.synchronize();
    }

    /// The streams the parts run on (main for every part's main context, comm for every side), and cuBLAS's.
    /// part が走る stream (各 part の main の context は main、side は comm) と cuBLAS の stream。
    fn setStreams(p: *Prefill, ms: cuda.Stream, cs: cuda.Stream) !void {
        const e = p.e;
        for (e.sp_parts[0..e.n_sp]) |*h| {
            h.main.stream = ms;
            h.main.ops.stream = ms;
            h.side.stream = cs;
            h.side.ops.stream = cs;
        }
        // cublasSetStream resets the workspace: set it after / cublasSetStream は作業領域を戻すので、その後に設定する
        try e.blas.check(e.blas.api.cublasSetStream_v2(e.handle, ms.handle), "cublasSetStream");
        try e.blas.check(e.blas.api.cublasSetWorkspace_v2(e.handle, e.workspace.ptr, cuda.cublas.torch_workspace), "cublasSetWorkspace");
    }
};

/// The prefill thread of a concurrent measurement: `k` chunks, one at a time; their times and the span.
/// 同時の測定の prefill のスレッド: `k` 個のチャンクを 1 つずつ。各時間と全体の区間。
const PrefillRun = struct {
    p: *Prefill,
    ctx: *const cuda.Context,
    k: usize,
    chunk_ns: []u64,
    enq_ns: []u64,
    enq_at: [16][2]u64 = undefined,
    t0: u64 = 0,
    t1: u64 = 0,
    started: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,

    fn go(r: *PrefillRun) void {
        r.ctx.makeCurrent() catch |err| {
            r.err = err;
            r.started.store(true, .release);
            return;
        };
        r.t0 = nowNs();
        // the decode lane starts with the prefill's first enqueue, not after it (a wide chunk's enqueue holds this
        // thread for most of the chunk while the GPU's queue is full)
        // decode のレーンは prefill の最初の投入と同時に始める (投入の後ではなく。広いチャンクの投入は、GPU の待ち行列が
        // 満杯の間、チャンクの大半このスレッドを塞ぐ)
        r.started.store(true, .release);
        for (0..r.k) |i| {
            const a = nowNs();
            r.enq_ns[i] = r.p.enqueue(r.p.cp, r.p.n) catch |err| {
                r.err = err;
                return;
            };
            if (i < 16) r.enq_at[i] = .{ a, nowNs() };
            r.p.wait() catch |err| {
                r.err = err;
                return;
            };
            r.chunk_ns[i] = nowNs() - a;
        }
        r.t1 = nowNs();
    }
};

/// Every rank's word gathered; returns rank 0's (NCCL on the engine's stream, nothing else running).
/// 全ランクの語を集め、rank 0 のものを返す (エンジンの stream の NCCL。他に何も動いていないとき)。
fn fromRank0(e: *glm.engine.Engine, v: u32) !u32 {
    if (e.world == 1) return v;
    const w = [4]i32{ @intCast(v), 0, 0, 0 };
    try e.chk_send.uploadAsync(0, std.mem.sliceAsBytes(&w), e.stream.handle);
    const lib = e.group.lib;
    try lib.check(lib.api.ncclAllGather(e.chk_send.ptr, e.chk_recv.ptr, 4, .i32, e.group.comm, e.stream.handle), "ncclAllGather");
    var all: [4]i32 = undefined;
    try e.chk_recv.downloadAsync(0, std.mem.sliceAsBytes(&all), e.stream.handle);
    try e.stream.synchronize();
    return @intCast(all[0]);
}

fn kvHash(gpa: std.mem.Allocator, e: *glm.engine.Engine) ![16]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const last = e.layers.len - 1;
    for ([_]usize{ 0, last }) |i| {
        const host = try gpa.alloc(u8, e.state.kc[i].len);
        defer gpa.free(host);
        try e.state.kc[i].download(0, host);
        h.update(host);
        if (e.state.ic[i]) |b| {
            const hi = try gpa.alloc(u8, b.len);
            defer gpa.free(hi);
            try b.download(0, hi);
            h.update(hi);
        }
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out[0..8].*, .lower);
}

const CheckLog = struct {
    lines: std.ArrayList(u8) = .empty,
    fn add(c: *CheckLog, gpa: std.mem.Allocator, comptime f: []const u8, args: anytype) void {
        c.lines.print(gpa, f, args) catch {};
    }
};

/// The decode chain's hash: k rounds after resetting the window's ids.
/// decode の列のハッシュ: ウィンドウの ids を戻してから k ラウンド。
const Seq = struct {
    fn run(d: *Decode, en: *glm.engine.Engine, l: *Decode.Lane, ri: usize, k: usize, pk: *[16]i32) !u64 {
        try d.resetIds(l);
        var h = std.hash.Wyhash.init(7);
        for (0..k) |_| {
            _ = try d.round(en, l, ri, pk);
            h.update(std.mem.sliceAsBytes(pk[0..d.rows[ri]]));
        }
        return h.final();
    }
};

/// One decode lane's thread of a two-lane measurement: both threads start together (host side), each runs its check
/// chain, then `m` timed rounds (the same count on every rank).
/// 2 レーンの測定の decode のレーン 1 本のスレッド: 両スレッドはホスト側でそろって始め、照合の列のあと `m` ラウンドを
/// 計時する (数は全ランクで同じ)。
const DecodeRun = struct {
    dl: *Decode,
    e: *glm.engine.Engine,
    lane: *Decode.Lane,
    ri: usize,
    m: usize,
    cr: usize,
    ctx: *const cuda.Context,
    lat: []u64,
    ends: []u64,
    go_flag: *std.atomic.Value(u32),
    seq: u64 = 0,
    t_loop: u64 = 0,
    t1: u64 = 0,
    err: ?anyerror = null,

    fn run(r: *DecodeRun) void {
        r.ctx.makeCurrent() catch |err| {
            r.err = err;
        };
        _ = r.go_flag.fetchAdd(1, .acq_rel);
        while (r.go_flag.load(.acquire) < 2) std.atomic.spinLoopHint();
        if (r.err != null) return;
        var pk: [16]i32 = undefined;
        r.seq = Seq.run(r.dl, r.e, r.lane, r.ri, r.cr, &pk) catch |err| {
            r.err = err;
            return;
        };
        r.t_loop = nowNs();
        for (r.lat, r.ends) |*x, *en| {
            x.* = r.dl.round(r.e, r.lane, r.ri, &pk) catch |err| {
                r.err = err;
                return;
            };
            en.* = nowNs();
        }
        r.t1 = nowNs();
    }
};

/// The RoCE settings the engine reads (B12X_ROCE_HCA / NCCL_IB_HCA, GID index, spin limit, proxy library).
/// エンジンが読む RoCE の設定 (B12X_ROCE_HCA / NCCL_IB_HCA、GID index、spin の上限、proxy ライブラリ)。
const RoceEnv = struct { names: [2][:0]const u8, nh: usize, gid: u32, spin: u32, lib: []const u8 };

fn roceEnv(gpa: std.mem.Allocator) !RoceEnv {
    const F = struct {
        fn env(a: [:0]const u8, b: [:0]const u8) ?[]const u8 {
            if (std.c.getenv(a)) |v| return std.mem.span(v);
            if (std.c.getenv(b)) |v| return std.mem.span(v);
            return null;
        }
    };
    var r: RoceEnv = .{ .names = undefined, .nh = 0, .gid = 3, .spin = 20000000, .lib = "./libroce_proxy.so" };
    var it = std.mem.tokenizeScalar(u8, F.env("B12X_ROCE_HCA", "NCCL_IB_HCA") orelse return error.RoceNoHca, ',');
    while (it.next()) |h| {
        if (r.nh == 2) break;
        const name = std.mem.trimStart(u8, h, "=^");
        r.names[r.nh] = try gpa.dupeSentinel(u8, name[0 .. std.mem.indexOfScalar(u8, name, ':') orelse name.len], 0);
        r.nh += 1;
    }
    r.gid = try std.fmt.parseInt(u32, F.env("B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX") orelse "3", 10);
    r.spin = try std.fmt.parseInt(u32, F.env("B12X_ROCE_SPIN_LIMIT", "B12X_ROCE_SPIN_LIMIT") orelse "20000000", 10);
    r.lib = F.env("TF_ROCE_PROXY_LIB", "TF_ROCE_PROXY_LIB") orelse r.lib;
    return r;
}

/// Whether R is in the comma list (empty: every R).
/// R がカンマ区切りの一覧にあるか (空ならどの R も)。
fn rcHas(list: []const u8, R: u32) bool {
    if (list.len == 0) return true;
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |x| if ((std.fmt.parseInt(u32, x, 10) catch 0) == R) return true;
    return false;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 7) {
        std.debug.print("usage: tf-glm53-concurrent RANK WORLD MASTER CKPT STORED_WORLD LAYERS\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const ckpt = args[4];
    const cp = envInt("TF_GLM53_CONCURRENT_CONTEXT", 32768);

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
        .ckpt = ckpt,
        .stored_world = try std.fmt.parseInt(usize, args[5], 10),
        .layers = try std.fmt.parseInt(usize, args[6], 10),
        .capacity = 0,
        .graphs = true,
        .kv_fp8 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp8") or std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .kv_fp4 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .dcp = @intCast(envInt("TF_GLM53_DCP", 1)),
    });
    const n = envInt("TF_GLM53_CONCURRENT_CHUNK", o.prompt_rows);
    o.capacity = cp + n + 64;
    if (n < forward.sp_min_rows or n > o.prompt_rows or cp % n != 0) {
        std.debug.print("[p0b] CHUNK {d} must be in [{d}, {d}] and divide CONTEXT {d}\n", .{ n, forward.sp_min_rows, o.prompt_rows, cp });
        return 2;
    }
    const e = try glm.engine.Engine.create(gpa, io, &driver, &ctx, &group, o);
    defer e.destroy();
    if (e.n_sp == 0) {
        std.debug.print("[p0b] the prefill lane needs sequence-parallel parts (TF_GLM53_PROMPT_SP=1, SP_DCP=1, DCP 2)\n", .{});
        return 2;
    }
    if (e.fast == null) {
        std.debug.print("[p0b] the decode lane needs the RoCE runtimes (TF_GLM53_ROCE=1)\n", .{});
        return 2;
    }
    // the prefill lane on NCCL alone: its parts' RoCE runtimes go
    // prefill のレーンは NCCL だけ: part の RoCE を外す
    for (e.sp_parts[0..e.n_sp]) |*h| {
        h.main.fast = null;
        h.main.dfast = null;
        h.side.fast = null;
        h.side.dfast = null;
    }
    if (rank == 0) std.debug.print("[p0b] engine up in {d} ms: world {d}, {d} layers, dcp {d}, sp parts {d}, chunk {d}, context {d}, capacity {d}\n", .{ (nowNs() - t_load) / 1000000, world, e.layers.len, e.dcp, e.n_sp, n, cp, e.capacity });

    // the decode lane's own state
    // decode のレーン自身の状態
    var rows_buf: [8]u32 = undefined;
    var nr: usize = 0;
    {
        var it = std.mem.tokenizeScalar(u8, envStr("TF_GLM53_CONCURRENT_R", "1,8"), ',');
        while (it.next()) |x| : (nr += 1) {
            if (nr == 8) break;
            rows_buf[nr] = try std.fmt.parseInt(u32, x, 10);
            if (rows_buf[nr] < 1 or rows_buf[nr] > 16) return error.BadWindowRows;
        }
    }
    const rows = rows_buf[0..nr];
    var first_moe: ?*const moe.Routed = null;
    for (e.layers) |*L| if (L.experts) |*x| {
        first_moe = x;
        break;
    };
    const keys: usize = glm.engine.bucket(e.capacity + 1, e.cfg.index_topk) orelse 2 * e.cfg.index_topk;
    const inv = try glm.rope.table(e.cfg.rope_theta, e.cfg.qk_rope);
    const had = glm.rope.hadamard();
    // the split-K counters of every EXL3 linear: the decode lane's own (the weights stay shared)
    // 全 EXL3 linear の split-K のカウンタ: decode のレーン専用 (重みは共有のまま)
    var wrows: u32 = 1;
    for (rows) |r| wrows = @max(wrows, r);
    const fm = first_moe orelse return error.NoMoeLayer;
    const made = try Decode.make(gpa, &driver, e, wrows, keys, &inv, &had, fm, cp, rows);
    var dl = made.dl;
    const n_counters = made.counters;
    try ctx.synchronize();

    // the context: CONTEXT positions prefilled once (serially), copied into the decode lane's caches
    // 文脈: CONTEXT 位置を 1 回 (逐次に) prefill し、decode のレーンのキャッシュに写す
    const ids = try gpa.alloc(i64, cp + n);
    promptIds(ids, 7_000_000);
    var pf: Prefill = .{ .e = e, .ids = ids, .cp = cp, .n = n, .done = try cuda.Event.init(&driver, false) };
    {
        const t0 = nowNs();
        var pa: usize = 0;
        while (pa < cp) : (pa += n) {
            _ = try pf.enqueue(pa, n);
            try pf.wait();
        }
        try dl.copyContext(e);
        try ctx.synchronize();
        if (rank == 0) std.debug.print("[p0b] context of {d} prefilled in {d} ms; the decode lane has {d} linear counters of its own\n", .{ cp, (nowNs() - t0) / 1000000, n_counters });
    }

    // the lanes' streams and graphs: priority (primary context) and green
    // レーンの stream と graph: 優先度 (primary context) と green
    var lo: c_int = 0;
    var hi: c_int = 0;
    {
        // not in the engine's driver table: looked up here
        // エンジンの driver の表には無いので、ここで引く
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        defer lib.close();
        const f = lib.lookup(*const fn (*c_int, *c_int) callconv(.c) c_int, "cuCtxGetStreamPriorityRange") orelse return error.MissingSymbol;
        if (f(&lo, &hi) != 0) return error.CudaFailed;
    }
    const dprio: i32 = if (std.c.getenv("TF_GLM53_CONCURRENT_DECODE_PRIORITY")) |v| try std.fmt.parseInt(i32, std.mem.span(v), 10) else hi;
    const modes = envStr("TF_GLM53_CONCURRENT_MODES", "alone,timeslice,priority,green");
    const base_main = e.sp_parts[0].main.stream;
    const base_comm = e.sp_parts[0].side.stream;
    var lane_p = try dl.lane(e, try cuda.Stream.initPriority(&driver, dprio));
    var green_lanes: ?GreenLanes = null;
    var lane_g: ?Decode.Lane = null;
    if (listHas(modes, "green")) {
        const g = try Green.open();
        const gl = try g.lanes(&driver, ctx.device, @intCast(envInt("TF_GLM53_CONCURRENT_DECODE_SM", 8)), @intCast(envInt("TF_GLM53_CONCURRENT_GREEN_FLAGS", 0)), dprio, o.comm_priority);
        green_lanes = gl;
        lane_g = try dl.lane(e, gl.decode);
        if (rank == 0) std.debug.print("[p0b] green: decode lane {d} SMs, prefill lane {d} SMs\n", .{ gl.decode_sms, gl.rest_sms });
    }
    if (rank == 0) std.debug.print("[p0b] streams: decode priority {d} (range {d}..{d}), prefill main 0, comm {d}\n", .{ dprio, lo, hi, o.comm_priority });

    const K = envInt("TF_GLM53_CONCURRENT_CHUNKS", 4);
    // the window sizes the time-slicing and concurrent modes run (empty: all of R)
    // 時分割と同時のモードで回すウィンドウの大きさ (空なら R のすべて)
    const rc_list = envStr("TF_GLM53_CONCURRENT_RC", "");
    const M0 = envInt("TF_GLM53_CONCURRENT_ROUNDS", 200);
    const CR = envInt("TF_GLM53_CONCURRENT_CHECK_ROUNDS", 16);
    var picks: [16]i32 = undefined;
    var check: CheckLog = .{};
    const lat = try gpa.alloc(u64, 200000);
    const ends = try gpa.alloc(u64, 200000);
    const launch = try gpa.alloc(u64, 200000);
    const chunk_ns = try gpa.alloc(u64, K);
    const enq_ns = try gpa.alloc(u64, K);
    var round_alone: [8]f64 = @splat(0);
    var chunk_alone: f64 = 0;
    var seq_alone: [8]u64 = @splat(0);
    var kv_alone: [16]u8 = @splat('-');


    // ---- alone ----
    {
        for (rows, 0..) |R, ri| {
            for (0..5) |_| _ = try dl.round(e, &lane_p, ri, &picks);
            for (lat[0..M0]) |*x| x.* = try dl.round(e, &lane_p, ri, &picks);
            const s = stat(gpa, lat[0..M0]);
            round_alone[ri] = s.mean;
            seq_alone[ri] = try Seq.run(&dl, e, &lane_p, ri, CR, &picks);
            const again = try Seq.run(&dl, e, &lane_p, ri, CR, &picks);
            check.add(gpa, " decode R={d}: alone {x} repeat-same {any}", .{ R, seq_alone[ri], again == seq_alone[ri] });
            if (rank == 0) std.debug.print("P0B mode=alone lane=decode R={d}: n={d} p50={d:.2} p95={d:.2} p99={d:.2} max={d:.2} mean={d:.2} ms, {d:.1} rounds/s\n", .{ R, s.n, s.p50, s.p95, s.p99, s.max, s.mean, 1000.0 / s.mean });
        }
        _ = try pf.enqueue(cp, n);
        try pf.wait();
        for (0..K) |i| {
            const t0 = nowNs();
            enq_ns[i] = try pf.enqueue(cp, n);
            try pf.wait();
            chunk_ns[i] = nowNs() - t0;
        }
        const s = stat(gpa, chunk_ns);
        const se = stat(gpa, enq_ns);
        chunk_alone = s.mean;
        kv_alone = try kvHash(gpa, e);
        _ = try pf.enqueue(cp, n);
        try pf.wait();
        const kv_again = try kvHash(gpa, e);
        check.add(gpa, " kv: alone {s} repeat-same {any}", .{ &kv_alone, std.mem.eql(u8, &kv_alone, &kv_again) });
        if (rank == 0) std.debug.print("P0B mode=alone lane=prefill: chunks={d} of {d} rows: mean={d:.1} max={d:.1} ms, {d:.0} tok/s; host enqueue mean={d:.2} ms\n", .{ K, n, s.mean, s.max, @as(f64, @floatFromInt(n)) * 1000.0 / s.mean, se.mean });
    }

    // ---- time-slicing: a chunk, then q rounds ----
    if (listHas(modes, "timeslice")) {
        var qit = std.mem.tokenizeScalar(u8, envStr("TF_GLM53_CONCURRENT_QUOTA", "1,4,8"), ',');
        while (qit.next()) |qs| {
            const q = try std.fmt.parseInt(usize, qs, 10);
            for (rows, 0..) |R, ri| {
                if (!rcHas(rc_list, R)) continue;
                _ = try fromRank0(e, 0);
                var nl: usize = 0;
                var gap: u64 = 0;
                const t0 = nowNs();
                var last_end = t0;
                for (0..K) |_| {
                    _ = try pf.enqueue(cp, n);
                    try pf.wait();
                    for (0..q) |_| {
                        lat[nl] = try dl.round(e, &lane_p, ri, &picks);
                        const now = nowNs();
                        gap = @max(gap, now - last_end);
                        last_end = now;
                        nl += 1;
                    }
                }
                const total = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
                const s = stat(gpa, lat[0..nl]);
                if (rank == 0) std.debug.print("P0B mode=timeslice q={d} R={d}: decode {d:.1} rounds/s (alone {d:.1}), round p50={d:.2} ms, longest gap between rounds {d:.1} ms; prefill {d:.0} tok/s (alone {d:.0})\n", .{ q, R, @as(f64, @floatFromInt(nl)) * 1000.0 / total, 1000.0 / round_alone[ri], s.p50, @as(f64, @floatFromInt(gap)) / 1e6, @as(f64, @floatFromInt(K * n)) * 1000.0 / total, @as(f64, @floatFromInt(n)) * 1000.0 / chunk_alone });
            }
        }
    }

    // ---- layer-slicing: one decode round after every LAYER_STEP prefill layers, on the prefill's main stream ----
    // ---- 層単位の時分割: prefill の LAYER_STEP 層ごとに decode を 1 ラウンド、prefill の main stream で ----
    if (listHas(modes, "layerslice")) {
        const evs = try gpa.alloc(cuda.Event, K * e.layers.len + 2);
        for (evs) |*ev| ev.* = try cuda.Event.init(&driver, true);
        const main_s = e.sp_parts[0].main.stream;
        var sit = std.mem.tokenizeScalar(u8, envStr("TF_GLM53_CONCURRENT_LAYER_STEP", "1"), ',');
        while (sit.next()) |ss| for (rows, 0..) |R, ri| {
            if (!rcHas(rc_list, R)) continue;
            const step = @max(1, try std.fmt.parseInt(usize, ss, 10));
            _ = try fromRank0(e, 0);
            try dl.resetIds(&lane_p);
            try lane_p.stream.synchronize();
            var nd: usize = 0;
            try evs[0].record(main_s);
            for (0..K) |_| {
                const tc = if (cp + n > e.cfg.index_topk) @as(?u32, @intCast(cp + n)) else null;
                var ch = try forward.SpChunk.begin(e.sp_parts[0..e.n_sp], e.sp_want, e.ready, ids[cp .. cp + n], @intCast(cp), e.embed.ptr, e.layers, e.state.kc, e.state.ic, tc, null, 0, false, null);
                for (0..e.layers.len) |li| {
                    try ch.layer(li);
                    if ((li + 1) % step == 0 or li + 1 == e.layers.len) {
                        try lane_p.execs[ri].launchOn(main_s);
                        try evs[1 + nd].record(main_s);
                        nd += 1;
                    }
                }
                _ = try ch.finish();
            }
            try evs[1 + nd].record(main_s);
            try evs[1 + nd].synchronize();
            if (e.fast) |*f| if (!f.healthy()) return error.RoceFailed;
            var picks_ls: [16]i32 = undefined;
            try dl.b.tok_out.download(0, std.mem.sliceAsBytes(picks_ls[0..R]));
            const total = try cuda.Event.elapsedMs(evs[0], evs[1 + nd]);
            var gaps: std.ArrayList(u64) = .empty;
            defer gaps.deinit(gpa);
            for (1..nd) |k| try gaps.append(gpa, @intFromFloat(1e6 * try cuda.Event.elapsedMs(evs[k], evs[k + 1])));
            const sg = stat(gpa, gaps.items);
            // the same number of rounds alone from the same ids: the last picks must agree
            // 同じ ids から同じ数のラウンドを単独で: 最後の選択が一致すること
            try dl.resetIds(&lane_p);
            for (0..nd) |_| _ = try dl.round(e, &lane_p, ri, &picks);
            const same = std.mem.eql(i32, picks[0..R], picks_ls[0..R]);
            check.add(gpa, " layerslice step {d} R={d}: decode same {any}", .{ step, R, same });
            const rps = @as(f64, @floatFromInt(nd)) * 1000.0 / total;
            const ptok = @as(f64, @floatFromInt(K * n)) * 1000.0 / total;
            if (rank == 0) std.debug.print("P0B mode=layerslice step={d} R={d}: decode {d:.1} rounds/s ({d:.0}% of alone), interval between rounds p50={d:.1} p99={d:.1} max={d:.1} ms; prefill {d:.0} tok/s ({d:.0}% of alone); sum {d:.2}\n", .{ step, R, rps, 100.0 * rps * round_alone[ri] / 1000.0, sg.p50, sg.p99, sg.max, ptok, 100.0 * ptok * chunk_alone / (1000.0 * @as(f64, @floatFromInt(n))), rps * round_alone[ri] / 1000.0 + ptok * chunk_alone / (1000.0 * @as(f64, @floatFromInt(n))) });
        };
    }

    // ---- concurrent: priority (primary context), green ----
    const Conc = struct { name: []const u8, lane: *Decode.Lane, main: cuda.Stream, comm: cuda.Stream };
    var concs: [2]Conc = undefined;
    var nc: usize = 0;
    if (listHas(modes, "priority")) {
        concs[nc] = .{ .name = "priority", .lane = &lane_p, .main = base_main, .comm = base_comm };
        nc += 1;
    }
    if (lane_g != null) {
        concs[nc] = .{ .name = "green", .lane = &lane_g.?, .main = green_lanes.?.main, .comm = green_lanes.?.comm };
        nc += 1;
    }
    for (concs[0..nc]) |cm| {
        try pf.setStreams(cm.main, cm.comm);
        defer pf.setStreams(base_main, base_comm) catch {};
        // the prefill lane alone on these streams (green: on the SMs left to it)
        // この stream で prefill のレーン単独 (green では残りの SM で)
        _ = try pf.enqueue(cp, n);
        try pf.wait();
        for (0..K) |i| {
            const t0 = nowNs();
            _ = try pf.enqueue(cp, n);
            try pf.wait();
            chunk_ns[i] = nowNs() - t0;
        }
        const cna = stat(gpa, chunk_ns).mean;
        if (rank == 0) std.debug.print("P0B mode={s} lane=prefill alone on these streams: chunk mean={d:.1} ms, {d:.0} tok/s ({d:.0}% of the primary context's)\n", .{ cm.name, cna, @as(f64, @floatFromInt(n)) * 1000.0 / cna, 100.0 * chunk_alone / cna });
        for (rows, 0..) |R, ri| {
            if (!rcHas(rc_list, R)) continue;
            // the decode lane alone on its stream (green: on its SMs)
            // decode のレーン単独を自分の stream で (green では自分の SM で)
            for (0..5) |_| _ = try dl.round(e, cm.lane, ri, &picks);
            for (lat[0..M0]) |*x| x.* = try dl.round(e, cm.lane, ri, &picks);
            const sa = stat(gpa, lat[0..M0]);
            if (rank == 0) std.debug.print("P0B mode={s} lane=decode R={d} alone on its stream: n={d} p50={d:.2} p99={d:.2} max={d:.2} mean={d:.2} ms ({d:.1} rounds/s)\n", .{ cm.name, R, sa.n, sa.p50, sa.p99, sa.max, sa.mean, 1000.0 / sa.mean });
            // decode rounds to outlast the prefill: rank 0's estimate, shared before the prefill thread starts (no
            // collective of the engine's stream may run beside it)
            // prefill より長く続く decode のラウンド数: rank 0 の見積もりを、prefill のスレッドを始める前に共有する
            // (その横でエンジンの stream の集団通信を動かしてはならない)
            const est: u32 = @intCast(@min(lat.len - CR, @max(50, @as(usize, @intFromFloat(1.5 * @as(f64, @floatFromInt(K)) * cna / @max(sa.mean, 0.01))))));
            const M = try fromRank0(e, est);
            var run: PrefillRun = .{ .p = &pf, .ctx = &ctx, .k = K, .chunk_ns = chunk_ns, .enq_ns = enq_ns };
            const th = try std.Thread.spawn(.{}, PrefillRun.go, .{&run});
            while (!run.started.load(.acquire)) std.atomic.spinLoopHint();
            // the check chain first (inside the first chunk), then the timed rounds
            // 先に照合の列 (最初のチャンクの中で)、続いて計時するラウンド
            const seq = Seq.run(&dl, e, cm.lane, ri, CR, &picks) catch |err| {
                th.join();
                return err;
            };
            var i: usize = 0;
            const t_loop = nowNs();
            while (i < M) : (i += 1) {
                lat[i] = dl.round(e, cm.lane, ri, &picks) catch |err| {
                    th.join();
                    return err;
                };
                ends[i] = nowNs();
                launch[i] = last_launch_ns;
            }
            th.join();
            if (run.err) |err| return err;
            // rounds that ended while the prefill ran
            // prefill の実行中に終わったラウンド
            var dur: std.ArrayList(u64) = .empty;
            defer dur.deinit(gpa);
            var gap: u64 = 0;
            var gap_at: u64 = 0;
            var prev: u64 = @max(run.t0, t_loop);
            var after: usize = 0;
            // rounds that ended inside a chunk's host enqueue, and the launches that waited
            // チャンクのホストの投入の最中に終わったラウンドと、待たされた投入
            var in_enq: std.ArrayList(u64) = .empty;
            defer in_enq.deinit(gpa);
            var launch_max: u64 = 0;
            var slow_launch: usize = 0;
            for (lat[0..M], ends[0..M], launch[0..M]) |x, en, ln| {
                if (en >= run.t0 and en <= run.t1) {
                    try dur.append(gpa, x);
                    if (en - prev > gap) {
                        gap = en - prev;
                        gap_at = prev;
                    }
                    prev = en;
                    launch_max = @max(launch_max, ln);
                    if (ln > 1000000) slow_launch += 1;
                    for (run.enq_at[0..@min(K, 16)]) |w| if (en >= w[0] and en <= w[1]) {
                        try in_enq.append(gpa, x);
                        break;
                    };
                } else if (en > run.t1) after += 1;
            }
            const gap_in_enq = for (run.enq_at[0..@min(K, 16)]) |w| {
                if (gap_at + gap > w[0] and gap_at < w[1]) break true;
            } else false;
            const se = stat(gpa, in_enq.items);
            const sen = stat(gpa, run.enq_ns);
            const s = stat(gpa, dur.items);
            const sc = stat(gpa, chunk_ns);
            const span = @as(f64, @floatFromInt(run.t1 - run.t0)) / 1e6;
            const rps = @as(f64, @floatFromInt(s.n)) * 1000.0 / span;
            check.add(gpa, " {s} R={d}: decode same {any}", .{ cm.name, R, seq == seq_alone[ri] });
            if (rank == 0) std.debug.print("P0B mode={s} R={d} detail: longest gap starts {d:.0} ms after the prefill start, overlaps a host enqueue {any}; rounds ending inside a host enqueue n={d} mean={d:.2} max={d:.2} ms; graph launch max {d:.2} ms, {d} launches over 1 ms; host enqueue mean {d:.0} ms; check chain {d:.0} ms\n", .{ cm.name, R, @as(f64, @floatFromInt(gap_at -| run.t0)) / 1e6, gap_in_enq, se.n, se.mean, se.max, @as(f64, @floatFromInt(launch_max)) / 1e6, slow_launch, sen.mean, @as(f64, @floatFromInt(t_loop -| run.t0)) / 1e6 });
            if (rank == 0) std.debug.print("P0B mode={s} R={d} concurrent: decode during prefill n={d} p50={d:.2} p95={d:.2} p99={d:.2} max={d:.2} mean={d:.2} ms, {d:.1} rounds/s ({d:.0}% of alone on its stream, {d:.0}% of alone on all SMs), longest gap {d:.1} ms, {d} rounds after; prefill chunks={d} mean={d:.1} ms, {d:.0} tok/s ({d:.0}% of alone on these streams, {d:.0}% of alone on all SMs), span {d:.0} ms\n", .{ cm.name, R, s.n, s.p50, s.p95, s.p99, s.max, s.mean, rps, 100.0 * rps * sa.mean / 1000.0, 100.0 * rps * round_alone[ri] / 1000.0, @as(f64, @floatFromInt(gap)) / 1e6, after, K, sc.mean, @as(f64, @floatFromInt(n)) * 1000.0 / sc.mean, 100.0 * cna / sc.mean, 100.0 * chunk_alone / sc.mean, span });
        }
        // the prefill's KV under this mode: one chunk with decode rounds (R = the first) alongside, their count rank
        // 0's, shared before the thread starts
        // このモードでの prefill の KV: decode のラウンド (R は先頭のもの) を並べた 1 チャンク。ラウンド数は rank 0 の
        // もので、スレッドを始める前に共有する
        const mk = try fromRank0(e, @intCast(@min(lat.len, @max(8, @as(usize, @intFromFloat(0.8 * cna / @max(round_alone[0], 0.01)))))));
        var run: PrefillRun = .{ .p = &pf, .ctx = &ctx, .k = 1, .chunk_ns = chunk_ns[0..1], .enq_ns = enq_ns[0..1] };
        const th = try std.Thread.spawn(.{}, PrefillRun.go, .{&run});
        while (!run.started.load(.acquire)) std.atomic.spinLoopHint();
        for (0..mk) |_| _ = dl.round(e, cm.lane, 0, &picks) catch |err| {
            th.join();
            return err;
        };
        th.join();
        if (run.err) |err| return err;
        const kv = try kvHash(gpa, e);
        check.add(gpa, " {s} kv same {any}", .{ cm.name, std.mem.eql(u8, &kv, &kv_alone) });
    }
    // ---- two decode lanes at once (c = 2 without batching): each lane its own RoCE runtimes, stream and thread ----
    // ---- decode のレーン 2 本を同時に (まとめない c = 2): 各レーンが自分の RoCE ランタイム・stream・スレッドを持つ ----
    if (listHas(modes, "dual")) {
        const RT = @typeInfo(@typeInfo(@FieldType(forward.Ctx, "fast")).optional.child).pointer.child;
        const re = try roceEnv(gpa);
        var fast2 = try RT.init(gpa, &driver, &group, &e.rk, re.lib, re.names[0..re.nh], re.gid, re.spin, e.stream);
        var dfast2: ?RT = null;
        if (e.dfast != null) dfast2 = try RT.init(gpa, &driver, &e.dgroup.?, &e.rk, re.lib, re.names[0..re.nh], re.gid, re.spin, e.stream);
        const made2 = try Decode.make(gpa, &driver, e, wrows, keys, &inv, &had, fm, cp, rows);
        var dl2 = made2.dl;
        dl2.fast = &fast2;
        dl2.dfast = if (dfast2) |*f| f else null;
        try dl2.copyContext(e);
        try ctx.synchronize();
        var lane2 = try dl2.lane(e, try cuda.Stream.initPriority(&driver, dprio));
        if (rank == 0) std.debug.print("[p0b] dual: a second decode lane with its own RoCE runtimes (world{s}), {d} counters\n", .{ if (dfast2 != null) ", DCP group" else "", made2.counters });
        const rd_list = envStr("TF_GLM53_CONCURRENT_RD", "1");
        for (rows, 0..) |R, ri| {
            if (!rcHas(rd_list, R)) continue;
            for (0..5) |_| _ = try dl2.round(e, &lane2, ri, &picks);
            for (lat[0..M0]) |*x| x.* = try dl2.round(e, &lane2, ri, &picks);
            const s2a = stat(gpa, lat[0..M0]);
            const seq2a = try Seq.run(&dl2, e, &lane2, ri, CR, &picks);
            const M = try fromRank0(e, @intCast(@min(lat.len / 2, @max(100, @as(usize, @intFromFloat(4000.0 / @max(round_alone[ri], 0.01)))))));
            var go = std.atomic.Value(u32).init(0);
            var ra: DecodeRun = .{ .dl = &dl, .e = e, .lane = &lane_p, .ri = ri, .m = M, .cr = CR, .ctx = &ctx, .lat = lat[0..M], .ends = ends[0..M], .go_flag = &go };
            var rb: DecodeRun = .{ .dl = &dl2, .e = e, .lane = &lane2, .ri = ri, .m = M, .cr = CR, .ctx = &ctx, .lat = lat[M .. 2 * M], .ends = ends[M .. 2 * M], .go_flag = &go };
            const th = try std.Thread.spawn(.{}, DecodeRun.run, .{&ra});
            DecodeRun.run(&rb);
            th.join();
            if (ra.err) |err| return err;
            if (rb.err) |err| return err;
            // the rounds of each lane that ended while both ran
            // 両方が走っていた間に終わった各レーンのラウンド
            const w0 = @max(ra.t_loop, rb.t_loop);
            const w1 = @min(ra.t1, rb.t1);
            const span = @as(f64, @floatFromInt(w1 -| w0)) / 1e6;
            var la: std.ArrayList(u64) = .empty;
            defer la.deinit(gpa);
            var lb: std.ArrayList(u64) = .empty;
            defer lb.deinit(gpa);
            for (ra.lat, ra.ends) |x, en| if (en >= w0 and en <= w1) try la.append(gpa, x);
            for (rb.lat, rb.ends) |x, en| if (en >= w0 and en <= w1) try lb.append(gpa, x);
            const sa = stat(gpa, la.items);
            const sb = stat(gpa, lb.items);
            const rpa = @as(f64, @floatFromInt(sa.n)) * 1000.0 / @max(span, 1);
            const rpb = @as(f64, @floatFromInt(sb.n)) * 1000.0 / @max(span, 1);
            const pa = rpa * round_alone[ri] / 1000.0;
            const pb = rpb * s2a.mean / 1000.0;
            check.add(gpa, " dual R={d}: lane 1 same {any}, lane 2 same {any}", .{ R, ra.seq == seq_alone[ri], rb.seq == seq2a });
            if (rank == 0) std.debug.print("P0B mode=dual R={d}: lane 1 round mean={d:.2} p99={d:.2} ms ({d:.0}% of alone), lane 2 mean={d:.2} p99={d:.2} ms ({d:.0}% of its alone {d:.2} ms); sum {d:.2}; overlap {d:.0} ms\n", .{ R, sa.mean, sa.p99, 100.0 * pa, sb.mean, sb.p99, 100.0 * pb, s2a.mean, pa + pb, span });
        }
    }
    std.debug.print("P0B check rank {d}:{s}\n", .{ rank, check.lines.items });
    if (e.fast) |*f| if (!f.healthy()) return 5;
    if (e.dfast) |*f| if (!f.healthy()) return 5;
    return 0;
}
