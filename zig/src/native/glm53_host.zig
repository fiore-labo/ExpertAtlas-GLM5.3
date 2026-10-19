//! GLM-5.3 behind the native server: one process a GPU, the ranks of one tensor-parallel replica. Rank 0 serves HTTP
//! and runs one request at a time (foreground first) on every rank. A request samples or decodes greedily, may resume
//! from the slot cache and draft (MTP, DFlash2); a cancel or stop string stops every rank at the next agreement point.
//! ネイティブサーバーで動く GLM-5.3: GPU ごとに 1 プロセス、1 つの tensor-parallel レプリカのランク群。rank 0 が
//! HTTP を受け、要求を 1 つずつ (foreground を先に) 全ランクで実行する。要求はサンプルか greedy で decode し、
//! スロットキャッシュからの再開と draft (MTP、DFlash2) もできる。cancel や stop 文字列は次の合意点で全ランクを止める。
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const glm = @import("glm_moe_dsa");
const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    // a stop string matched or the client left: the rest of the reply is dropped
    // stop 文字列に一致したか、クライアントが去った: 応答の残りは捨てる
    quiet: bool = false,
    queued: i96 = 0,
    began: i96 = 0,
    prefilled: ?i96 = null,
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    nccl: cuda.nccl.Library,
    group: cuda.tp.Group,
    eng: *glm.engine.Engine,
    ch: glm.lockstep.Channel,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    running: ?*Job = null,
    closing: bool = false,
    // requests that may wait beside the running one (TF_GLM53_QUEUE); past it the server answers 503 (busy)
    // 実行中のもののほかに待てる要求の数 (TF_GLM53_QUEUE)。超えるとサーバーは 503 (busy) を返す
    queue_max: usize = 32,
    // rank 0: the engine thread; others: the follower
    // rank 0 ではエンジンのスレッド、他のランクではフォロワー
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_prompt: u64 = 0,
    live_generated: u64 = 0,

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn } };
    }

    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    fn now(h: *Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        // only rank 0 takes requests: the others follow it and never run a queue of their own
        // 要求を受けるのは rank 0 だけ。他のランクは rank 0 に従い、自分の列を処理しない
        if (h.group.rank != 0) return error.Closed;
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .request = request, .sink = sink, .queued = h.now() };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (h.queued.items.len >= h.queue_max) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        // foreground before background, each in arrival order
        // foreground を background より先に、それぞれ到着順に並べる
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
            h.finish(job, .cancelled, .{}, "");
            h.lock();
            return;
        };
        h.cancels.append(h.gpa, id) catch {};
        // the running request: ask every rank to stop at its next agreement point (a prompt chunk, a decode round)
        // 実行中の要求: 次の合意点 (prompt チャンク、decode ラウンド) で止まるよう全ランクに求める
        if (h.running) |job| if (job.id == id) h.eng.stop_req.store(true, .monotonic);
    }

    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= t - window_ns) tokens += m.tokens;
        }
        var n: usize = 0;
        var generation_tokens: u64 = 0;
        if (h.running != null and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_generated);
            generation_tokens = h.live_generated;
            n = 1;
        }
        out.* = .{
            .pool = if (h.eng.pool) |p| blk: {
                const s = p.status();
                break :blk .{ .blocks = s.blocks, .free_blocks = s.free_blocks, .slots = s.slots, .shared_blocks = s.shared_blocks, .evicted_total = s.evicted_total, .cold_total = s.cold_total };
            } else null,
            .running = @intFromBool(h.running != null),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = generation_tokens,
        };
    }

    fn memoryFn(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        // before finished is out: status must not see a freed job
        // finished を出す前に外す: status が解放済みの job を見てはならない
        if (h.running == job) h.running = null;
        h.unlock();
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        job.emitted.deinit(h.gpa);
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *Host, n: usize) void {
        const t = h.now();
        h.lock();
        defer h.unlock();
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < t - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = t, .tokens = n }) catch {};
    }

    fn run(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        while (true) {
            h.lock();
            while (h.queued.items.len == 0 and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            if (h.closing) {
                const left = h.gpa.dupe(*Job, h.queued.items) catch &.{};
                h.queued.clearRetainingCapacity();
                h.unlock();
                for (left) |job| h.finish(job, .cancelled, .{}, "");
                h.gpa.free(left);
                glm.lockstep.shutdown(&h.ch) catch |err| std.log.err("glm53: telling the other ranks to stop failed: {s}", .{@errorName(err)});
                return;
            }
            const job = h.queued.orderedRemove(0);
            h.running = job;
            // with `running`, under the lock: a cancel from now on is kept
            // `running` と一緒にロックの下でリセットする: これ以降の cancel は失われない
            h.eng.stop_req.store(false, .monotonic);
            h.live_prompt = job.request.prompt.len;
            h.live_generated = 0;
            h.cancels.clearRetainingCapacity();
            h.unlock();
            h.serve(job);
        }
    }

    /// The reply's callbacks from Engine.generate (engine thread).
    /// Engine.generate から呼ばれる応答のコールバック (エンジンのスレッド)。
    const Out = struct {
        h: *Host,
        job: *Job,

        fn prefilled(ctx: *anyopaque) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            const done = h.now();
            h.lock();
            // the slot cache's resume point (this engine thread set it for this prefill)
            // スロットキャッシュの再開位置 (このエンジンのスレッドがこの prefill のために設定した)
            const cached = h.eng.start;
            if (done > c.job.began) h.prefill_rate = @as(f64, @floatFromInt(c.job.request.prompt.len - cached)) / (@as(f64, @floatFromInt(done - c.job.began)) / 1e9);
            h.prefill_at = done;
            c.job.prefilled = done;
            h.unlock();
            emit(c.job, .{ .prefilled = @intCast(cached) });
        }

        fn tokens(ctx: *anyopaque, toks: []const u32) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const job = c.job;
            const h = c.h;
            h.noteDecoded(toks.len);
            if (job.quiet) return;
            h.lock();
            const gone = h.closing or std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
            h.unlock();
            if (gone) {
                job.quiet = true;
                h.eng.stop_req.store(true, .monotonic);
                return;
            }
            var n: usize = 0;
            // stop strings are checked after each token
            // stop 文字列はトークンごとに確かめる
            for (toks) |t| {
                // out of host memory: the reply is lost, so every rank stops
                // ホストのメモリ不足: 応答は失われるので全ランクを止める
                job.emitted.append(h.gpa, t) catch {
                    job.quiet = true;
                    h.eng.stop_req.store(true, .monotonic);
                    break;
                };
                n += 1;
                if (job.request.stop) |s| if (s.check(s.ctx, job.emitted.items)) {
                    job.quiet = true;
                    // the reply is complete: the ranks need not go on
                    // 応答は完成した: ランクは続ける必要がない
                    h.eng.stop_req.store(true, .monotonic);
                    break;
                };
            }
            if (n > 0) emit(job, .{ .tokens = toks[0..n] });
            h.lock();
            h.live_generated = @intCast(job.emitted.items.len);
            h.unlock();
        }
    };

    fn serve(h: *Host, job: *Job) void {
        const r = job.request;
        job.began = h.now();
        if (r.prompt.len == 0 or r.prompt.len > h.eng.promptLimit()) {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, "the prompt does not fit this server's context window");
        }
        const prompt = h.gpa.alloc(i64, r.prompt.len) catch {
            emit(job, .{ .prefilled = 0 });
            return h.finish(job, .failed, .{}, "out of host memory");
        };
        defer h.gpa.free(prompt);
        for (prompt, r.prompt) |*p, t| p.* = t;
        var out: Out = .{ .h = h, .job = job };
        // the request's EOS switch: every rank stops at the model's EOS ids (the server's set also holds the
        // tokenizer's; for GLM-5.3 they agree)
        // 要求の EOS 指定: 全ランクがモデルの EOS id で止まる (サーバーの集合はトークナイザーのものも含むが、GLM-5.3
        // では一致する)
        // drafts when the request allows them ("draft": false: the serial rounds): the MTP layer's, or DFlash2's
        // 要求が許せば draft する ("draft": false なら逐次ラウンド): MTP 層のもの、なければ DFlash2 のもの
        const mode: u8 = if (!r.drafts) 0 else if (h.eng.mtp != null) glm.lockstep.mode_mtp else if (h.eng.drafter != null) glm.lockstep.mode_dflash else 0;
        var header: glm.lockstep.Header = .{ .max_tokens = @intCast(@min(r.max_tokens, std.math.maxInt(i32))), .stop_eos = r.eos.len > 0, .mode = mode };
        // the server resolved the model's defaults and the prompt's seed
        // モデルの既定値と prompt の seed はサーバーが決めてある
        if (r.sampling) |s| if (s.temperature > 0) {
            header.seed = s.seed;
            header.temperature = s.temperature;
            header.top_k = @intCast(s.top_k);
            header.top_p = s.top_p;
            header.min_p = s.min_p;
        };
        // from here every rank runs the request: a failure leaves the others waiting in a collective
        // ここから全ランクが要求を実行する: 失敗すると他のランクは collective 通信で待ったままになる
        const st = glm.lockstep.lead(&h.ch, h.eng, header, prompt, .{ .ctx = @ptrCast(&out), .tokens = Out.tokens, .prefilled = Out.prefilled }, null) catch |err| {
            // refused before it was shared: every rank goes on serving
            // 共有の前に断った: 全ランクがそのまま処理を続ける
            if (err == error.ShapeNotVerified) {
                emit(job, .{ .prefilled = 0 });
                return h.finish(job, .failed, .{}, "this prompt needs a kernel shape the server did not verify at start-up (see the server log)");
            }
            fatal("serving a shared request", err);
        };
        // the cancel list under the lock: a cancel arriving now may grow it
        // cancel の一覧はロックの下で読む: いま届く cancel が一覧を伸ばしうる
        const cancelled = if (job.quiet) blk: {
            h.lock();
            defer h.unlock();
            break :blk std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        } else false;
        const reason: api.Reason = if (job.quiet) (if (cancelled) .cancelled else .stop) else switch (st.finish) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        const rounds = if (st.rounds > 0) st.rounds else st.generated -| 1;
        h.finish(job, reason, .{ .rounds = rounds, .drafted = st.drafted, .accepted = st.accepted, .min_rows = if (st.drafted > 0) 2 else 1, .prefill_seconds = @as(f64, @floatFromInt(st.prefill_ns)) / 1e9 }, "");
    }

    fn follow(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        glm.lockstep.follow(&h.ch, h.eng, null, null) catch |err| fatal("following rank 0", err);
        std.log.info("glm53: rank {d} stopped following (rank 0 shut down)", .{h.group.rank});
    }
};

/// A failure once a request is shared ends this rank at once with exit code 70; the launcher's watch stops the rest
/// (they wait in a collective).
/// 要求の共有後の失敗では、このランクを直ちに終了コード 70 で終える。残りのランク (collective 通信で待っている)
/// はランチャーの監視が止める。
fn fatal(what: []const u8, err: anyerror) noreturn {
    std.log.err("glm53: {s} failed ({s}); exiting with 70 so the watch stops every rank", .{ what, @errorName(err) });
    std.process.exit(70);
}

/// The build identity every rank must share: sha256 of this executable and of the engine settings (every
/// TF_GLM53_* / TF_EXL3_* variable but TF_GLM53_RANK, sorted).
/// 全ランクで同じでなければならないビルド ID: この実行ファイルとエンジン設定 (TF_GLM53_RANK を除く
/// TF_GLM53_* / TF_EXL3_* の全変数を整列したもの) の sha256。
fn buildIdentity(gpa: std.mem.Allocator, io: std.Io) ![]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const exe = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/exe", gpa, .limited(1 << 30));
    h.update(exe);
    gpa.free(exe);
    var vars: std.ArrayList([]const u8) = .empty;
    defer vars.deinit(gpa);
    var i: usize = 0;
    while (std.c.environ[i]) |e| : (i += 1) {
        const kv = std.mem.span(e);
        if (!(std.mem.startsWith(u8, kv, "TF_GLM53_") or std.mem.startsWith(u8, kv, "TF_EXL3_"))) continue;
        if (std.mem.startsWith(u8, kv, "TF_GLM53_RANK=")) continue;
        try vars.append(gpa, kv);
    }
    std.mem.sort([]const u8, vars.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    for (vars.items) |v| {
        h.update(v);
        h.update("\n");
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.allocPrint(gpa, "{x}", .{d});
}

fn envOr(name: [:0]const u8, default: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else default;
}

fn envNum(name: [:0]const u8, default: usize) !usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

/// This process's rank of the replica: loads it, checks with the others that every rank loaded the same, and starts
/// rank 0's engine thread or a follower. `window`: positions a request may reach (prompt + reply).
/// TF_GLM53_MASTER: rank 0's HOST:PORT for the NCCL id; TF_GLM53_STORED_WORLD: the experts' split on disk.
/// このプロセスが受け持つレプリカのランク: 読み込み、全ランクが同じものを読み込んだかを他と確かめ、rank 0 なら
/// エンジンのスレッドを、他ならフォロワーを起動する。`window`: 要求が届きうる位置の数 (prompt + 応答)。
/// TF_GLM53_MASTER: NCCL id 用の rank 0 の HOST:PORT。TF_GLM53_STORED_WORLD: ディスク上の expert の分割数。
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: usize) !*Host {
    const rank: u32 = @intCast(try envNum("TF_GLM53_RANK", 0));
    const world: u32 = @intCast(try envNum("TF_GLM53_WORLD", 1));
    const master = std.c.getenv("TF_GLM53_MASTER") orelse return error.NoMaster;
    const queue_max = try envNum("TF_GLM53_QUEUE", 32);
    if (queue_max == 0) return error.QueueZero;
    // the engine and the channel keep pointers to the driver, context and group: they live in h from the start
    // エンジンとチャネルは driver・context・group へのポインタを持つので、それらは最初から h に置く
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa, .io = io, .driver = undefined, .ctx = undefined, .nccl = undefined, .group = undefined,
        .eng = undefined, .ch = undefined, .info_ = .{ .name = "glm53-zig", .lanes = 1, .context_window = @intCast(window), .cuda_fields = true },
        .queue_max = queue_max,
    };
    h.driver = try cuda.Driver.open();
    h.ctx = try cuda.Context.init(&h.driver, @intCast(try envNum("TF_CUDA_DEVICE", 0)));
    h.nccl = try cuda.nccl.Library.openEnv();
    h.group = try cuda.tp.Group.init(&h.nccl, rank, world, try cuda.tp.parseMaster(std.mem.span(master)));
    const t0 = std.Io.Clock.awake.now(io);
    h.eng = try glm.engine.Engine.create(gpa, io, &h.driver, &h.ctx, &h.group, glm.engine.withEnv(.{
        .ckpt = dir,
        .stored_world = try envNum("TF_GLM53_STORED_WORLD", 8),
        .layers = if (std.c.getenv("TF_GLM53_LAYERS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null,
        .capacity = window,
        .graphs = !std.mem.eql(u8, envOr("TF_GLM53_GRAPHS", "1"), "0"),
        .drafter = if (std.c.getenv("TF_GLM53_DFLASH")) |v| std.mem.span(v) else null,
        .kv_fp8 = std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp8") or std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp4"),
        .kv_fp4 = std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp4"),
        .dcp = @intCast(try envNum("TF_GLM53_DCP", 1)),
    }));
    // every rank's checkpoint, executable and engine env compared before the prewarm, whose collectives would hang
    // on ranks set up differently
    // 全ランクのチェックポイント・実行ファイル・エンジンの env を prewarm の前に比べる。設定の違うランクがあると
    // prewarm の collective 通信が止まるため
    h.ch = glm.lockstep.Channel.init(gpa, h.eng);
    const ident = try buildIdentity(gpa, io);
    defer gpa.free(ident);
    const fp = try glm.lockstep.Fingerprint.of(io, gpa, dir, ident, world, rank);
    fp.agree(&h.ch) catch |err| {
        std.log.err("glm53: the ranks loaded different checkpoints, executables, settings or ranks ({s}); refusing to start", .{@errorName(err)});
        return err;
    };
    var mt: usize = 0; // TF_GLM53_MEMTRACE
    glm.engine.memMark(&h.ctx, rank, "-", &mt);
    {
        const tw = std.Io.Clock.awake.now(io);
        const n = try h.eng.prewarm();
        std.log.info("glm53: rank {d}: {d} decode graphs captured up front in {d:.1} s", .{ rank, n, @as(f64, @floatFromInt(tw.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0 });
    }
    // the slot cache, when TF_GLM53_PREFIX=slots
    // スロットキャッシュ (TF_GLM53_PREFIX=slots のとき)
    glm.engine.memMark(&h.ctx, rank, "decode graphs (prewarm)", &mt);
    try glm.slots_serve.setup(&h.ch, h.eng, &h.ctx, try glm.slots.Env.fromEnv(), &.{ dir, envOr("TF_GLM53_EXPERTS_DIR", ""), envOr("TF_GLM53_DFLASH", "") });
    try glm.pool_serve.setup(&h.ch, h.eng); // TF_GLM53_PREFIX=pool
    glm.engine.memMark(&h.ctx, rank, "slot cache store", &mt);
    // /health shows the store's positions: 0 when a rank could not allocate it
    // /health にストアの位置数を出す: どこかのランクで確保できなかったときは 0
    if (h.eng.slots) |s| h.info_.slot_positions = @intCast(s.S);
    if (h.eng.pool) |p| h.info_.slot_positions = @intCast(p.env.positions);
    std.log.info("glm53: rank {d}/{d} up in {d:.1} s ({d} layers, window {d})", .{ rank, world, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0, h.eng.layers.len, window });
    h.thread = if (rank == 0) try std.Thread.spawn(.{ .stack_size = 16 << 20 }, Host.run, .{h}) else try std.Thread.spawn(.{ .stack_size = 16 << 20 }, Host.follow, .{h});
    return h;
}

/// Rank 0: stops admitting, tells the followers to stop, joins. Others: the follower thread ends when rank 0's stop
/// arrives; closing before that would leave rank 0 waiting, so a follower's server waits for it.
/// rank 0: 受け付けをやめ、フォロワーに停止を伝えて join する。他: rank 0 の停止が届くとフォロワーのスレッドが
/// 終わる。その前に閉じると rank 0 が待ち続けるので、フォロワーのサーバーはそれを待つ。
pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    if (h.group.rank == 0) {
        h.lock();
        h.closing = true;
        h.wake.broadcast(h.io);
        h.unlock();
    }
    if (h.thread) |t| t.join();
    h.ch.deinit();
    h.eng.destroy();
    h.group.deinit();
    h.nccl.close();
    h.queued.deinit(h.gpa);
    h.cancels.deinit(h.gpa);
    h.decoded.deinit(h.gpa);
    h.gpa.destroy(h);
}
