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
    // its progress for /metrics (the engine thread writes it under the host's mutex): when the prefill started, the
    // prompt rows the cache resumed, those in the caches so far and when they were (the last chunk's end: the prefill
    // seconds move with the rows, so tokens over seconds is the chunks' own rate whenever a monitor reads it), the
    // decode rounds and drafts
    // /metrics 用の進み具合 (エンジンのスレッドがホストの mutex の下で書く): prefill を始めた時刻、キャッシュから
    // 再開した prompt の行数、いまキャッシュにある行数とその時刻 (最後のチャンクの終わり: prefill の秒数は行と一緒に
    // 進むので、モニタがいつ読んでもトークン ÷ 秒はチャンクそのものの速さになる)、decode のラウンド数と draft の数
    prefill_at: ?i96 = null,
    cached: u64 = 0,
    done: u64 = 0,
    done_at: i96 = 0,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    // TF_GLM53_MULTI: decoded as one of the conversations of the batched rounds; `stop`: rank 0's stop for it (the
    // client left, a stop string matched, a cancel), sent with the next round
    // TF_GLM53_MULTI: まとめたラウンドの会話の 1 つとして decode する。`stop`: それへの rank 0 の停止 (クライアントが
    // 去った、stop 文字列に一致、cancel)。次のラウンドで送る
    multi: bool = false,
    stop: std.atomic.Value(bool) = .init(false),
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
    // the requests that ran: their totals, which status adds the running request's progress to (under the mutex, so
    // each total only grows)
    // 実行を終えた要求の合計。status はこれに実行中の要求の進み具合を足す (mutex の下なので、どの合計も減らない)
    totals: api.Progress = .{},
    labels: []u8 = &.{},
    // TF_GLM53_MULTI: conversations decoded together (0: one request at a time), the job of each, its outputs, and
    // the job whose prompt is being prefilled
    // TF_GLM53_MULTI: 一緒に decode する会話の数 (0 なら 1 要求ずつ)、それぞれの job と出力、prompt を prefill 中の job
    multi_n: usize = 0,
    /// rank 0: the memory the control panel shows (GET /v1/memory), its snapshot refreshed between requests.
    /// rank 0: 制御パネルが出すメモリ (GET /v1/memory)。写しは要求の合間に更新する。
    mem: ?*glm.memory_live.MemLive = null,
    /// rank 0: a change of the routed experts' bits asked for (POST /v1/experts/swap; the engine thread runs it between
    /// requests), its state for GET /v1/experts/swap, and the checkpoint the bits are saved against (under the mutex)
    /// rank 0: 求められた routed expert の bit の変更 (POST /v1/experts/swap。エンジンのスレッドが要求の合間に実行)、
    /// GET /v1/experts/swap 用のその状態、保存する bit の基のチェックポイント (mutex の下)
    swap_want: ?[]u8 = null,
    /// saves so far, for each save's own temporary file (under the mutex) / 保存の回数 (一時ファイルの名前に。mutex の下)
    save_n: u64 = 0,
    swap: Swap = .{},
    ckpt: []const u8 = "",
    act: [glm.engine.multi_max]?*Job = @splat(null),
    outs: [glm.engine.multi_max]Out = undefined,
    prefilling: ?*Job = null,

    const Mark = struct { at: i96, tokens: u64 };
    const Swap = struct {
        seq: u32 = 0,
        state: enum { idle, waiting, running, done, failed } = .idle,
        done: u32 = 0,
        layers: u32 = 0,
        stats: glm.expert_swap.Stats = .{},
        err: []const u8 = "",
    };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .experts = expertsFn, .experts_control = controlFn } };
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
        if (h.prefilling) |job| if (job.id == id) h.stopJob(job);
        for (h.act[0..h.multi_n]) |a| if (a) |job| if (job.id == id) h.stopJob(job);
    }

    /// A job's stop: its conversation's bit in the next batched round, and the prefill's stop while its prompt runs
    /// (the agreement points of every prompt chunk read the engine's stop_req).
    /// job の停止: 次のまとめたラウンドでのその会話のビット。prompt の実行中は prefill の停止も (各 prompt チャンクの
    /// 合意点がエンジンの stop_req を読む)。
    fn stopJob(h: *Host, job: *Job) void {
        job.stop.store(true, .monotonic);
        if (h.multi_n == 0 or h.prefilling == job) h.eng.stop_req.store(true, .monotonic);
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
                break :blk .{
                    .blocks = s.blocks, .free_blocks = s.free_blocks, .slots = s.slots, .shared_blocks = s.shared_blocks, .evicted_total = s.evicted_total, .cold_total = s.cold_total,
                    .block_positions = s.block_positions, .continued_total = s.continued_total, .shared_total = s.shared_total, .new_total = s.new_total,
                    .check_seconds_total = s.check_seconds_total, .hash_seconds_total = s.hash_seconds_total,
                };
            } else null,
            .running = @intCast(@as(usize, @intFromBool(h.running != null)) + @intFromBool(h.prefilling != null) + blk: {
                var a: usize = 0;
                for (h.act[0..h.multi_n]) |x| a += @intFromBool(x != null);
                break :blk a;
            }),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = generation_tokens,
            .progress = h.progress(t),
        };
    }

    /// The totals with the running request's progress in them, and its phase (under the mutex).
    /// 実行中の要求の進み具合を足した合計と、その要求の段階 (mutex の下で)。
    fn progress(h: *Host, t: i96) api.Progress {
        var p = h.totals;
        p.phase = .idle;
        p.prompt = 0;
        p.cached = 0;
        p.prompt_done = 0;
        p.generated = 0;
        p.queued_prompt = 0;
        for (h.queued.items) |q| p.queued_prompt += q.request.prompt.len;
        const job = h.running orelse return p;
        p.phase = if (job.prefilled == null) .prefill else .decode;
        p.prompt = job.request.prompt.len;
        const start = job.prefill_at orelse return p; // still planning: nothing counted yet / まだ計画中: 何も数えない
        p.cached = job.cached;
        p.prompt_done = job.done;
        p.generated = h.live_generated;
        h.addRunning(&p, job, start, t);
        return p;
    }

    /// The running request's progress at time `t` added to the totals in `p`.
    /// 時刻 `t` での実行中の要求の進み具合を `p` の合計に足す。
    fn addRunning(h: *Host, p: *api.Progress, job: *const Job, start: i96, t: i96) void {
        p.prompt_total += job.request.prompt.len;
        p.cached_total += job.cached;
        p.computed_total += job.done -| job.cached;
        p.generated_total += h.live_generated;
        p.prefill_seconds_total += secs(job.done_at - start);
        if (job.prefilled) |f| p.decode_seconds_total += secs(t - f);
        p.rounds_total += job.rounds;
        p.drafted_total += job.drafted;
        p.accepted_total += job.accepted;
    }

    /// A finished TF_GLM53_MULTI job's progress added to the totals (as addRunning, with its own tokens).
    /// 終わった TF_GLM53_MULTI の job の進み具合を合計に足す (addRunning と同じ、トークンはその job のもの)。
    fn addFinished(h: *Host, p: *api.Progress, job: *const Job, start: i96, t: i96) void {
        _ = h;
        p.prompt_total += job.request.prompt.len;
        p.cached_total += job.cached;
        p.computed_total += job.done -| job.cached;
        p.generated_total += job.emitted.items.len;
        p.prefill_seconds_total += secs(job.done_at - start);
        if (job.prefilled) |f| p.decode_seconds_total += secs(t - f);
        p.rounds_total += job.rounds;
        p.drafted_total += job.drafted;
        p.accepted_total += job.accepted;
    }

    fn secs(ns: i96) f64 {
        return @max(0, @as(f64, @floatFromInt(ns)) / 1e9);
    }

    /// TF_GLM53_EXPERT_LIVE: the counts the engine thread keeps on rank 0 (its own mutex; no GPU).
    /// TF_GLM53_EXPERT_LIVE: エンジンのスレッドが rank 0 に残す数 (専用の mutex。GPU には触れない)。
    fn expertsFn(ctx: *anyopaque, a: Allocator, q: api.ExpertsQuery) ?[]u8 {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (q == .memory) return if (h.mem) |m| m.json(a) catch null else null;
        if (q == .swap) return h.swapJson(a) catch null;
        const lv = h.eng.live orelse return null;
        return switch (q) {
            .usage => |w| lv.usageJson(a, w) catch null,
            .recent => |r| lv.recentJson(a, r.after, r.max, if (r.text) |tx| .{ .ctx = tx.ctx, .bytes = tx.bytes } else null) catch null,
            .prefill => |r| lv.prefillJson(a, r.after, r.max) catch null,
            .memory, .swap => null,
        };
    }

    /// GET /v1/experts/swap: whether a change can be asked for, the banks, and the last change's state.
    /// GET /v1/experts/swap: 変更を求められるか、バンク、最後の変更の状態。
    fn swapJson(h: *Host, a: Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        const w = &out.writer;
        h.lock();
        defer h.unlock();
        const s = h.swap;
        try w.print("{{\"available\": {}, \"plan\": ", .{h.eng.banks != null and h.multi_n == 0});
        try planJson(h, w);
        try w.writeAll(", \"banks\": {");
        var first = true;
        if (h.eng.banks) |b| for (b.dirs, 0..) |d, bits| if (d) |dir| {
            try w.print("{s}\"{d}\": ", .{ if (first) "" else ", ", bits });
            try std.json.Stringify.value(dir, .{}, w);
            first = false;
        };
        try w.print("}}, \"seq\": {d}, \"state\": \"{s}\", \"done\": {d}, \"layers\": {d}, \"experts\": {d}, \"up\": {d}, \"down\": {d}, \"read_bytes\": {d}, \"ms\": {d}, \"error\": \"{s}\"}}", .{ s.seq, @tagName(s.state), s.done, s.layers, s.stats.experts, s.stats.up, s.stats.down, s.stats.read_bytes, s.stats.ms, s.err });
        return out.toOwnedSlice();
    }

    /// The expert plan the routed experts were loaded by (null without one): name, digest, banks, counts by bits.
    /// routed expert を読み込んだ計画 (無ければ null): 名前、ダイジェスト、バンク、bit ごとの数。
    fn planJson(h: *Host, w: *std.Io.Writer) !void {
        const p = h.eng.plan orelse return w.writeAll("null");
        try w.writeAll("{\"name\": ");
        try std.json.Stringify.value(p.name, .{}, w);
        try w.print(", \"digest\": \"{x}\", \"banks\": [", .{p.digest});
        for (p.banks, 0..) |b, i| {
            try w.print("{s}{{\"id\": ", .{if (i == 0) "" else ", "});
            try std.json.Stringify.value(b.id, .{}, w);
            try w.print(", \"bits\": {d}, \"calibration\": ", .{b.bits});
            try std.json.Stringify.value(b.calibration, .{}, w);
            try w.writeAll("}");
        }
        const c = glm.expert_plan.counts(&p);
        try w.writeAll("], \"counts\": {");
        var first = true;
        for (c, 0..) |n, b| if (n > 0) {
            try w.print("{s}\"{d}\": {d}", .{ if (first) "" else ", ", b, n });
            first = false;
        };
        try w.writeAll("}}");
    }

    /// POST /v1/experts/swap: taken only while nothing runs or waits (the engine thread runs it next);
    /// POST /v1/experts/save: the bits in use, as a named set (TF_GLM53_EXPERT_SETS/<name>.json).
    /// Neither with TF_GLM53_MULTI (UnsupportedWithMulti, 409: its scheduler runs no swap: not supported for
    /// now).
    /// POST /v1/experts/swap: 何も実行・待機していないときだけ受ける (次にエンジンのスレッドが実行)。
    /// POST /v1/experts/save: 使用中の bit を名前つきのセットとして (TF_GLM53_EXPERT_SETS/<name>.json)。
    /// TF_GLM53_MULTI ではどちらも断る (UnsupportedWithMulti、409: そのスケジューラは入れ替えをしない。今は対応
    /// しない)。
    fn controlFn(ctx: *anyopaque, a: Allocator, c: api.ExpertsControl) api.ControlError![]u8 {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (h.group.rank != 0) return error.Unsupported;
        if (h.multi_n > 0) return error.UnsupportedWithMulti;
        switch (c) {
            .swap => |bits| {
                if (h.eng.banks == null) return error.Unsupported;
                if (bits.len != h.eng.xbits.len) return error.BadRequest;
                for (bits) |b| if (b < 2 or b > 8) return error.BadRequest;
                const want = try h.gpa.dupe(u8, bits);
                h.lock();
                defer h.unlock();
                if (h.running != null or h.queued.items.len > 0 or h.swap_want != null or h.swap.state == .running or h.closing) {
                    h.gpa.free(want);
                    return error.Busy;
                }
                h.swap_want = want;
                h.swap = .{ .seq = h.swap.seq + 1, .state = .waiting };
                h.wake.signal(h.io);
                return std.fmt.allocPrint(a, "{{\"seq\": {d}, \"state\": \"waiting\"}}", .{h.swap.seq});
            },
            .save => |name| return h.saveSet(a, name),
        }
    }

    fn saveSet(h: *Host, a: Allocator, name: []const u8) api.ControlError![]u8 {
        const dir = std.mem.span(std.c.getenv("TF_GLM53_EXPERT_SETS") orelse return error.Unsupported);
        if (name.len == 0 or name.len > 64) return error.BadRequest;
        for (name) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_' and ch != '.') return error.BadRequest;
        if (name[0] == '.') return error.BadRequest;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ dir, name });
        const cwd = std.Io.Dir.cwd();
        if (cwd.openFile(h.io, path, .{})) |f| {
            f.close(h.io);
            return error.Exists;
        } else |_| {}
        h.lock();
        // the bits change only in a swap, which the engine thread runs with the state at .running
        // bit が変わるのは swap の中だけ。エンジンのスレッドはその間 state を .running にしている
        if (h.swap.state == .running or h.swap_want != null) {
            h.unlock();
            return error.Busy;
        }
        const bits = a.dupe(u8, h.eng.xbits) catch |err| {
            h.unlock();
            return err;
        };
        h.unlock();
        const body = setJson(h, a, name, bits) catch return error.OutOfMemory;
        cwd.createDirPath(h.io, dir) catch return error.Failed;
        // written whole to a file of its own, then linked to the name, which fails if the name exists: two saves of
        // one name never both win, and no reader sees half a file
        // 自分だけのファイルに全部書いてから名前へ link する (名前があれば失敗): 同じ名前の 2 つの保存が両方勝つことは
        // なく、読み手が書きかけを見ることもない
        h.lock();
        h.save_n += 1;
        const n = h.save_n;
        h.unlock();
        const tmpz = std.fmt.allocPrintSentinel(a, "{s}/.{s}.json.tmp{d}", .{ dir, name, n }, 0) catch return error.OutOfMemory;
        const pathz = a.dupeSentinel(u8, path, 0) catch return error.OutOfMemory;
        cwd.writeFile(h.io, .{ .sub_path = tmpz, .data = body }) catch return error.Failed;
        defer _ = std.c.unlink(tmpz.ptr);
        if (std.c.link(tmpz.ptr, pathz.ptr) != 0) return if (std.c._errno().* == @intFromEnum(std.c.E.EXIST)) error.Exists else error.Failed;
        std.log.info("glm53: expert bits saved as {s}", .{path});
        return std.fmt.allocPrint(a, "{{\"saved\": \"{s}\"}}", .{name});
    }

    /// glm53-expert-bits v1: the bits (base64, one byte an expert, the MoE layers from first_layer), their counts, the
    /// checkpoint they were loaded from and the banks.
    /// glm53-expert-bits v1: bit (base64、expert ごとに 1 バイト、first_layer からの MoE 層)、その数、読み込んだ
    /// チェックポイントとバンク。
    fn setJson(h: *Host, a: Allocator, name: []const u8, bits: []const u8) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        const w = &out.writer;
        const E = h.eng.cfg.experts;
        var counts: [9]u64 = @splat(0);
        var sum: u64 = 0;
        for (bits) |b| {
            counts[@min(b, 8)] += 1;
            sum += b;
        }
        const t = std.Io.Clock.real.now(h.io).toSeconds();
        try w.print("{{\n  \"format\": \"glm53-expert-bits\",\n  \"version\": 1,\n  \"name\": \"{s}\",\n  \"saved_unix\": {d},\n  \"checkpoint\": ", .{ name, t });
        try std.json.Stringify.value(h.ckpt, .{}, w);
        try w.print(",\n  \"first_layer\": {d},\n  \"layers\": {d},\n  \"experts\": {d},\n  \"mean_bits\": {d:.4},\n  \"counts\": {{", .{ h.eng.xfirst, bits.len / E, E, @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(@max(bits.len, 1))) });
        var first = true;
        for (counts, 0..) |n, b| if (n > 0) {
            try w.print("{s}\"{d}\": {d}", .{ if (first) "" else ", ", b, n });
            first = false;
        };
        try w.writeAll("},\n  \"banks\": {");
        first = true;
        if (h.eng.banks) |bk| for (bk.dirs, 0..) |d, b| if (d) |dir| {
            try w.print("{s}\"{d}\": ", .{ if (first) "" else ", ", b });
            try std.json.Stringify.value(dir, .{}, w);
            first = false;
        };
        try w.writeAll("},\n  \"bits\": \"");
        const enc = std.base64.standard.Encoder;
        const b64 = try a.alloc(u8, enc.calcSize(bits.len));
        _ = enc.encode(b64, bits);
        try w.writeAll(b64);
        try w.writeAll("\"\n}\n");
        return out.toOwnedSlice();
    }

    /// Engine thread: the change asked for, on every rank (lockstep.swap), its progress and outcome in `swap`.
    /// エンジンのスレッド: 求められた変更を全ランクで (lockstep.swap)。進み具合と結果は `swap` に。
    fn runSwap(h: *Host, want: []u8) void {
        defer h.gpa.free(want);
        const Prog = struct {
            fn f(ctx: *anyopaque, done: u32, layers: u32) void {
                const hh: *Host = @ptrCast(@alignCast(ctx));
                hh.lock();
                defer hh.unlock();
                hh.swap.done = done;
                hh.swap.layers = layers;
            }
        };
        std.log.info("glm53: expert swap {d}: start", .{h.swap.seq});
        var st: glm.expert_swap.Stats = .{};
        const r = glm.lockstep.swap(&h.ch, h.eng, want, .{ .ctx = h, .f = Prog.f }, &st);
        h.lock();
        h.swap.stats = st;
        if (r) |_| {
            h.swap.state = .done;
        } else |err| {
            h.swap.state = .failed;
            h.swap.err = @errorName(err);
        }
        const s = h.swap;
        h.unlock();
        // a swap that may have left the ranks apart (or the cached prefixes stale): every rank stops, as after a
        // failed request (the followers return the error from follow and stop too)
        // ランクが食い違ったかもしれない (または保存した prefix が古い) 入れ替え: 失敗した要求の後と同じく全ランクが
        // 止まる (フォロワーも follow からエラーを返して止まる)
        if (r) |_| {} else |err| if (!st.alike) fatal("an expert swap that may have left the ranks apart", err);
        if (r) |_| {
            std.log.info("glm53: expert swap {d}: {d} experts in {d} layers ({d} up, {d} down), {d} MiB read, {d} ms", .{ s.seq, s.stats.experts, s.stats.layers, s.stats.up, s.stats.down, s.stats.read_bytes >> 20, s.stats.ms });
        } else |err| std.log.err("glm53: expert swap {d}: {s}", .{ s.seq, @errorName(err) });
        if (h.mem) |m| m.setExperts(expertBytes(h.eng));
    }

    /// rank 0, engine thread: the memory snapshot (tracked bytes and the driver's free / total).
    /// rank 0、エンジンのスレッド: メモリの写し (数えているバイトとドライバの free / total)。
    fn memSnap(h: *Host) void {
        const m = h.mem orelse return;
        const u = cuda.usage(false);
        const mi = h.ctx.memInfo() catch return;
        m.snap(u.device, u.host, u.peak, mi.free, mi.total);
    }

    /// The process's tracked device bytes (DeviceBuffers) and their peak; no GPU call (/health, /metrics).
    /// プロセスが数えている device のバイト (DeviceBuffer) とそのピーク。GPU は呼ばない (/health、/metrics)。
    fn memoryFn(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        // the request's progress joins the totals as it leaves `running`, so the totals status reads never fall back
        // 要求の進み具合は `running` から外すときに合計へ移す。status が読む合計は後戻りしない
        if (h.running == job) {
            if (job.prefill_at) |start| {
                h.addRunning(&h.totals, job, start, h.now());
                h.totals.requests_total += 1;
            }
        }
        if (job.multi) if (job.prefill_at) |start| {
            h.addFinished(&h.totals, job, start, h.now());
            h.totals.requests_total += 1;
        };
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
            while (h.queued.items.len == 0 and h.swap_want == null and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            if (h.swap_want) |want| if (!h.closing) {
                h.swap_want = null;
                h.swap.state = .running;
                h.unlock();
                h.runSwap(want);
                h.memSnap();
                continue;
            };
            if (h.closing) {
                if (h.swap_want) |want| h.gpa.free(want);
                h.swap_want = null;
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
            h.memSnap();
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
            if (c.job.prefill_at == null) {
                c.job.prefill_at = c.job.began;
                c.job.cached = cached;
            }
            c.job.done = c.job.request.prompt.len;
            c.job.done_at = done;
            h.unlock();
            emit(c.job, .{ .prefilled = @intCast(cached) });
        }

        fn chunk(ctx: *anyopaque, done: usize) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            const t = h.now();
            h.lock();
            defer h.unlock();
            // the first chunk: the prefill starts at the rows the cache resumed
            // 最初のチャンク: prefill はキャッシュから再開した行から始まる
            if (c.job.prefill_at == null) {
                c.job.prefill_at = t;
                c.job.cached = done;
            }
            c.job.done = done;
            c.job.done_at = t;
        }

        fn round(ctx: *anyopaque, drafted: usize, accepted: usize) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            h.lock();
            defer h.unlock();
            c.job.rounds += 1;
            c.job.drafted += drafted;
            c.job.accepted += accepted;
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
                h.stopJob(job);
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
                    h.stopJob(job);
                    break;
                };
                n += 1;
                if (job.request.stop) |s| if (s.check(s.ctx, job.emitted.items)) {
                    job.quiet = true;
                    // the reply is complete: the ranks need not go on
                    // 応答は完成した: ランクは続ける必要がない
                    h.stopJob(job);
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
        const sink: glm.engine.Sink = .{ .ctx = @ptrCast(&out), .tokens = Out.tokens, .prefilled = Out.prefilled, .chunk = Out.chunk, .round = Out.round };
        // TF_GLM53_MULTI: the followers run events; this request is one of them (single)
        // TF_GLM53_MULTI: フォロワーは出来事を実行する。この要求はその 1 つ (single)
        const st = (if (h.multi_n > 0) glm.lockstep.leadSingle(&h.ch, h.eng, header, prompt, sink, null) else glm.lockstep.lead(&h.ch, h.eng, header, prompt, sink, null)) catch |err| {
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
        h.lock();
        const computed = job.done -| job.cached;
        h.unlock();
        h.finish(job, reason, .{
            .rounds = rounds, .drafted = st.drafted, .accepted = st.accepted, .min_rows = if (st.drafted > 0) 2 else 1,
            .prefill_seconds = @as(f64, @floatFromInt(st.prefill_ns)) / 1e9,
            .decode_seconds = if (st.generated > 0) @as(f64, @floatFromInt(st.decode_ns)) / 1e9 else null,
            .computed = computed,
            .queue_seconds = secs(job.began - job.queued),
        }, "");
    }

    // ----------------------------------------------- TF_GLM53_MULTI: batched conversations / まとめた会話 ---

    fn sinkOf(out: *Out) glm.engine.Sink {
        return .{ .ctx = @ptrCast(out), .tokens = Out.tokens, .prefilled = Out.prefilled, .chunk = Out.chunk, .round = Out.round };
    }

    fn liveCount(h: *Host) usize {
        var n: usize = 0;
        for (h.act[0..h.multi_n]) |a| n += @intFromBool(a != null);
        return n;
    }

    /// Rank 0's scheduler with TF_GLM53_MULTI: admit the head of the queue into a free conversation (its prefill runs
    /// alone, the others wait), then one batched round over the live conversations, then release the ones done; a
    /// sampled request runs the old way once no conversation is live.
    /// TF_GLM53_MULTI での rank 0 のスケジューラ: 待ち行列の先頭を空いた会話に受け入れ (その prefill は単独で走り、他は
    /// 待つ)、使用中の会話をまとめたラウンドを 1 回、終わった会話を返す。サンプリングの要求は、使用中の会話が無くなって
    /// から従来の方法で実行する。
    fn runMulti(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        while (true) {
            h.lock();
            while (h.queued.items.len == 0 and h.liveCount() == 0 and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            if (h.closing) {
                const left = h.gpa.dupe(*Job, h.queued.items) catch &.{};
                h.queued.clearRetainingCapacity();
                for (h.act[0..h.multi_n]) |a| if (a) |job| job.stop.store(true, .monotonic);
                h.unlock();
                for (left) |job| h.finish(job, .cancelled, .{}, "");
                h.gpa.free(left);
                // the live conversations stop (one round of stops) and commit
                // 使用中の会話は止め (停止のラウンド 1 回)、確定する
                h.roundOnce();
                glm.lockstep.shutdownMulti(&h.ch) catch |err| std.log.err("glm53: telling the other ranks to stop failed: {s}", .{@errorName(err)});
                return;
            }
            // greedy and sampled requests alike (a sampled conversation draws its rows with its own keyed sampler)
            // greedy も sampled も同じく (sampled の会話は自分のキー付きの抽選で行を選ぶ)
            var pick: ?*Job = null;
            if (h.queued.items.len > 0 and h.eng.multiFree() != null) pick = h.queued.items[0];
            if (pick != null) _ = h.queued.orderedRemove(0);
            h.unlock();
            if (pick) |job| h.admitJob(job);
            h.roundOnce();
        }
    }

    /// The job into a free conversation: its plan beside the live ones, its prefill and first token. No room in the
    /// pool beside them: the job goes back to the head of the queue.
    /// job を空いた会話へ: 使用中の会話の横でのプラン、prefill、最初のトークン。その横に pool の余地がなければ、job は
    /// 待ち行列の先頭へ戻る。
    fn admitJob(h: *Host, job: *Job) void {
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
        for (prompt, r.prompt) |*pp, t| pp.* = t;
        const k = h.eng.multiFree() orelse return h.requeue(job);
        const mode: u8 = if (!r.drafts) 0 else if (h.eng.drafter != null) glm.lockstep.mode_dflash else 0;
        var header: glm.lockstep.Header = .{ .max_tokens = @intCast(@min(r.max_tokens, std.math.maxInt(i32))), .stop_eos = r.eos.len > 0, .mode = mode };
        // the server resolved the model's defaults and the prompt's seed (as serve)
        // モデルの既定値と prompt の seed はサーバーが決めてある (serve と同じ)
        if (r.sampling) |s| if (s.temperature > 0) {
            header.seed = s.seed;
            header.temperature = s.temperature;
            header.top_k = @intCast(s.top_k);
            header.top_p = s.top_p;
            header.min_p = s.min_p;
        };
        h.outs[k] = .{ .h = h, .job = job };
        job.multi = true;
        job.stop.store(false, .monotonic);
        h.lock();
        h.prefilling = job;
        h.eng.stop_req.store(false, .monotonic);
        h.unlock();
        glm.lockstep.leadAdmit(&h.ch, h.eng, k, header, prompt, sinkOf(&h.outs[k]), .{ .ctx = @ptrCast(h), .f = liveFor, .release = releaseFor }) catch |err| {
            h.lock();
            h.prefilling = null;
            h.unlock();
            switch (err) {
                // refused before it was shared: every rank goes on serving
                // 共有の前に断った: 全ランクがそのまま処理を続ける
                error.NoRoom => return h.requeue(job),
                error.ShapeNotVerified => {
                    job.multi = false;
                    emit(job, .{ .prefilled = 0 });
                    return h.finish(job, .failed, .{}, "this prompt needs a kernel shape the server did not verify at start-up (see the server log)");
                },
                else => fatal("admitting a shared request", err),
            }
        };
        h.lock();
        h.prefilling = null;
        h.act[k] = job;
        h.unlock();
        // done at once (its first token ended it, or it stopped in the prompt)
        // すぐに終わった (最初のトークンで終わった、または prompt の途中で止まった)
        if (h.eng.multi.?.convs[k].done) h.releaseConv(k);
    }

    /// The live conversations not done but `skip` (the one being admitted), for the rounds inside its prefill.
    /// prefill 中の会話 `skip` を除く、未完了の使用中の会話 (その prefill の中のラウンド用)。
    fn liveFor(ctx: *anyopaque, skip: usize, ks: []u8, stops: []bool, sinks: []?glm.engine.Sink) usize {
        const h = self(ctx);
        const m = h.eng.multi.?;
        var n: usize = 0;
        for (h.act[0..h.multi_n], 0..) |a, k| if (a) |job| {
            if (k == skip or m.convs[k].done) continue;
            ks[n] = @intCast(k);
            stops[n] = job.stop.load(.monotonic);
            sinks[n] = sinkOf(&h.outs[k]);
            n += 1;
        };
        return n;
    }

    fn releaseFor(ctx: *anyopaque, k: usize) void {
        self(ctx).releaseConv(k);
    }

    fn requeue(h: *Host, job: *Job) void {
        job.multi = false;
        h.lock();
        defer h.unlock();
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, .{}, "out of host memory");
            h.lock();
        };
    }

    /// One batched round over the live conversations not done (rank 0's stop bit for each), then the done ones
    /// released.
    /// 未完了の使用中の会話をまとめたラウンド 1 回 (それぞれに rank 0 の停止ビット)、続いて終わった会話を返す。
    fn roundOnce(h: *Host) void {
        const m = h.eng.multi.?;
        var ks: [glm.engine.multi_max]u8 = undefined;
        var stops: [glm.engine.multi_max]bool = undefined;
        var sinks: [glm.engine.multi_max]?glm.engine.Sink = undefined;
        var n: usize = 0;
        for (h.act[0..h.multi_n], 0..) |a, k| if (a) |job| {
            if (m.convs[k].done) continue;
            ks[n] = @intCast(k);
            stops[n] = job.stop.load(.monotonic);
            sinks[n] = sinkOf(&h.outs[k]);
            n += 1;
        };
        if (n > 0) glm.lockstep.leadRound(&h.ch, h.eng, ks[0..n], stops[0..n], sinks[0..n]) catch |err| fatal("a batched round", err);
        for (h.act[0..h.multi_n], 0..) |a, k| if (a != null and m.convs[k].done) h.releaseConv(k);
    }

    /// Conversation k done: its commit on every rank, the job finished with its stats.
    /// 会話 k が終わった: 全ランクで確定し、統計を付けて job を終える。
    fn releaseConv(h: *Host, k: usize) void {
        const job = h.act[k].?;
        const st = glm.lockstep.leadRelease(&h.ch, h.eng, k) catch |err| fatal("releasing a conversation", err);
        h.lock();
        h.act[k] = null;
        const cancelled = job.quiet and std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        const computed = job.done -| job.cached;
        h.unlock();
        const reason: api.Reason = if (job.quiet) (if (cancelled) .cancelled else .stop) else switch (st.finish) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        const rounds = if (st.rounds > 0) st.rounds else st.generated -| 1;
        h.finish(job, reason, .{
            .rounds = rounds, .drafted = st.drafted, .accepted = st.accepted, .min_rows = if (st.drafted > 0) 2 else 1,
            .prefill_seconds = @as(f64, @floatFromInt(st.prefill_ns)) / 1e9,
            .decode_seconds = if (st.generated > 0) @as(f64, @floatFromInt(st.decode_ns)) / 1e9 else null,
            .computed = computed,
            .queue_seconds = secs(job.began - job.queued),
        }, "");
    }

    fn followMulti(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        glm.lockstep.followMulti(&h.ch, h.eng) catch |err| fatal("following rank 0's events", err);
        std.log.info("glm53: rank {d} stopped following (rank 0 shut down)", .{h.group.rank});
    }

    fn follow(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        glm.lockstep.follow(&h.ch, h.eng, null, null) catch |err| fatal("following rank 0", err);
        std.log.info("glm53: rank {d} stopped following (rank 0 shut down)", .{h.group.rank});
    }
};

/// The routed experts' device bytes on this rank (slabs and scales).
/// このランクの routed expert のデバイスのバイト (置き場とスケール)。
fn expertBytes(e: *const glm.engine.Engine) u64 {
    var ex: u64 = 0;
    for (e.layers) |*L| if (L.experts) |*x| {
        for (x.trellis) |b| ex += b.len;
        for ([_]cuda.DeviceBuffer{ x.suh_g, x.suh_u, x.svh_g, x.svh_u, x.suh_d, x.svh_d }) |b| ex += b.len;
    };
    return ex;
}

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
    // TF_GLM53_EXPERT_BITS (a saved set, applied before the first request) with TF_GLM53_MULTI, whose scheduler changes
    // no bits (not supported for now): refused before anything loads, on every rank alike (the set would wait for ever
    // while the old bits served)
    // TF_GLM53_EXPERT_BITS (保存したセット。最初の要求の前に当てる) と、bit を変えないスケジューラの TF_GLM53_MULTI の
    // 組み合わせ (今は対応しない): 何も読む前に、全ランクで同じく断る
    if (std.c.getenv("TF_GLM53_EXPERT_BITS") != null and try envNum("TF_GLM53_MULTI", 0) > 0) {
        std.log.err("glm53: TF_GLM53_EXPERT_BITS is not supported with TF_GLM53_MULTI (its scheduler changes no bits): start with those bits as the expert plan (the launcher's allocation) instead", .{});
        return error.ExpertBitsWithMulti;
    }
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
    const ident0 = try buildIdentity(gpa, io);
    defer gpa.free(ident0);
    // the expert plan's contents count too, not only its path (its banks' manifests were checked against it)
    // エキスパートの計画はパスだけでなく中身も比べる (バンクの manifest は計画と照らし済み)
    const ident = if (h.eng.plan) |p| try std.fmt.allocPrint(gpa, "{s} plan {x}", .{ ident0, p.digest }) else try gpa.dupe(u8, ident0);
    defer gpa.free(ident);
    const fp = try glm.lockstep.Fingerprint.of(io, gpa, dir, ident, world, rank);
    fp.agree(&h.ch) catch |err| {
        std.log.err("glm53: the ranks loaded different checkpoints, executables, settings or ranks ({s}); refusing to start", .{@errorName(err)});
        return err;
    };
    var mt: usize = 0; // TF_GLM53_MEMTRACE
    glm.engine.memMark(&h.ctx, rank, "-", &mt);
    if (h.eng.load_progress) |*p| p.setPhase(.prewarm);
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
    // TF_GLM53_MULTI=n: up to n conversations decoded together (needs the pool; engine.multiSetup)
    // TF_GLM53_MULTI=n: 最大 n 個の会話を一緒に decode する (pool が要る。engine.multiSetup)
    const multi_n = try envNum("TF_GLM53_MULTI", 0);
    if (multi_n > 0) {
        try h.eng.multiSetup(multi_n);
        h.multi_n = multi_n;
        h.info_.lanes = @intCast(multi_n);
    }
    glm.engine.memMark(&h.ctx, rank, "slot cache store", &mt);
    // /health shows the store's positions: 0 when a rank could not allocate it
    // /health にストアの位置数を出す: どこかのランクで確保できなかったときは 0
    if (h.eng.slots) |s| h.info_.slot_positions = @intCast(s.S);
    if (h.eng.pool) |p| h.info_.slot_positions = @intCast(p.env.positions);
    // /metrics' engine_info: the settings a monitor shows beside the numbers
    // /metrics の engine_info: モニタが数値と並べて見せる設定
    const kv = envOr("TF_GLM53_KV", "bf16");
    const kv_name = if (std.mem.eql(u8, kv, "fp4")) "fp4" else if (std.mem.eql(u8, kv, "fp8")) "fp8" else "bf16";
    const cache_name = if (h.eng.pool != null) "pool" else if (h.eng.slots != null) "slots" else "off";
    const drafter_name = if (h.eng.mtp != null) "mtp" else if (h.eng.drafter != null) "dflash2" else "none";
    h.labels = try std.fmt.allocPrint(gpa, "tp=\"{d}\",dcp=\"{d}\",kv=\"{s}\",cache=\"{s}\",drafter=\"{s}\",context=\"{d}\",queue=\"{d}\"", .{ world, h.eng.dcp, kv_name, cache_name, drafter_name, window, queue_max });
    h.info_.labels = h.labels;
    std.log.info("glm53: rank {d}/{d} up in {d:.1} s ({d} layers, window {d})", .{ rank, world, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0, h.eng.layers.len, window });
    if (h.eng.load_progress) |*p| p.setPhase(.ready);
    // TF_GLM53_EXPERT_LIVE: the warm-up's rows are not usage; requests start from zero
    // TF_GLM53_EXPERT_LIVE: 暖機の行は使用ではない。要求は 0 から数える
    if (h.eng.live) |lv| lv.reset();
    // rank 0: the memory the control panel shows: the start-up steps, the routed experts' bytes, a first snapshot
    // rank 0: 制御パネルが出すメモリ: 起動の段階、routed expert のバイト、最初の写し
    if (rank == 0) {
        h.mem = try glm.memory_live.MemLive.init(gpa, io, rank, glm.engine.mem_steps[0..glm.engine.mem_n], expertBytes(h.eng));
        h.memSnap();
        h.ckpt = try gpa.dupe(u8, dir);
        // TF_GLM53_EXPERT_BITS=<set>.json (saved by POST /v1/experts/save): the experts changed to it first, before
        // any request
        // TF_GLM53_EXPERT_BITS=<set>.json (POST /v1/experts/save で保存したもの): どの要求よりも先に、expert をそれに変える
        if (std.c.getenv("TF_GLM53_EXPERT_BITS")) |v| {
            h.swap_want = try glm.expert_swap.readSet(gpa, io, std.mem.span(v), h.eng);
            h.swap = .{ .seq = 1, .state = .waiting };
            std.log.info("glm53: expert bits from {s} (the first job)", .{std.mem.span(v)});
        }
    }
    const spawn: std.Thread.SpawnConfig = .{ .stack_size = 16 << 20 };
    h.thread = if (rank == 0)
        (if (h.multi_n > 0) try std.Thread.spawn(spawn, Host.runMulti, .{h}) else try std.Thread.spawn(spawn, Host.run, .{h}))
    else if (h.multi_n > 0) try std.Thread.spawn(spawn, Host.followMulti, .{h}) else try std.Thread.spawn(spawn, Host.follow, .{h});
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
    h.gpa.free(h.labels);
    h.gpa.destroy(h);
}
