// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! Prometheus text for GET /metrics, family for family as the Python server writes it.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");

const prefix = "tensorfold:";
/// Upper edges shared by the request and time-to-first-token histograms; +Inf is added when rendered.
pub const buckets = [_]f64{ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0 };
pub const tpot_buckets = [_]f64{ 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0 };

fn HistogramType(comptime edges: []const f64) type {
    return struct {
        pub const bucket_edges = edges;
        counts: [edges.len + 1]u64 = @splat(0),
        total: f64 = 0,
        n: u64 = 0,

        pub fn observe(h: *@This(), raw: f64) void {
            const value = @max(0, raw);
            h.n += 1;
            h.total += value;
            for (edges, 0..) |edge, i| if (value <= edge) {
                h.counts[i] += 1;
                return;
            };
            h.counts[edges.len] += 1;
        }
    };
}

pub const Histogram = HistogramType(&buckets);
pub const TpotHistogram = HistogramType(&tpot_buckets);

/// Counters of finished requests; gauges are read from the engine at scrape time.
pub const Metrics = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    prompt: u64 = 0,
    generation: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    rounds: u64 = 0,
    cached: u64 = 0, // prompt tokens the engine found cached (finished requests)
    cached_known: bool = false, // some request's engine said how many (an engine with a prompt cache)
    disconnects: u64 = 0,
    // requests refused because the engine's queue was full (503); queue waits of finished requests
    rejected: u64 = 0,
    queue: Histogram = .{},
    latency: Histogram = .{},
    ttft: Histogram = .{},
    decode: Histogram = .{},
    prefill: Histogram = .{},
    tpot: TpotHistogram = .{},
    requests: std.ArrayList(struct { key: []const u8, status: u16, count: u64 }) = .empty,

    pub fn note(m: *Metrics, io: std.Io, prompt: usize, generation: usize, drafted: u64, accepted: u64, rounds: u64, latency: f64, ttft: ?f64, decode: ?f64, prefill: ?f64, tpot: ?f64, cached: ?usize, queue: ?f64) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.prompt += prompt;
        if (cached) |c| {
            m.cached += c;
            m.cached_known = true;
        }
        m.generation += generation;
        m.drafted += drafted;
        m.accepted += accepted;
        m.rounds += rounds;
        m.latency.observe(latency);
        if (ttft) |t| m.ttft.observe(t);
        if (decode) |d| m.decode.observe(d);
        if (prefill) |p| m.prefill.observe(p);
        if (tpot) |t| m.tpot.observe(t);
        if (queue) |q| m.queue.observe(q);
    }

    /// A request the engine refused because its queue was full (the 503 reply).
    pub fn refused(m: *Metrics, io: std.Io) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.rejected += 1;
    }

    pub fn tpotValue(_: *const Metrics, first: ?i96, last: ?i96, generated: usize) ?f64 {
        if (generated < 2 or first == null or last == null) return null;
        return @max(0, @as(f64, @floatFromInt(last.? - first.?)) / 1e9) / @as(f64, @floatFromInt(generated - 1));
    }

    /// One HTTP reply under its key label, counted once per request.
    pub fn httpRequest(m: *Metrics, io: std.Io, key: []const u8, status: u16) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        for (m.requests.items) |*r| if (r.status == status and std.mem.eql(u8, r.key, key)) {
            r.count += 1;
            return;
        };
        const owned = m.gpa.dupe(u8, key) catch return;
        m.requests.append(m.gpa, .{ .key = owned, .status = status, .count = 1 }) catch m.gpa.free(owned);
    }

    pub fn disconnected(m: *Metrics, io: std.Io) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.disconnects += 1;
    }

    /// Finished requests' totals, as the Python CUDA server's /health counts them (cuda/health.py).
    pub const Totals = struct { requests: u64, prompt: u64, generation: u64, prefill_s: f64, decode_s: f64, cached: u64, cached_known: bool, rounds: u64, drafted: u64, accepted: u64 };

    pub fn totals(m: *Metrics, io: std.Io) Totals {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        return .{ .requests = m.latency.n, .prompt = m.prompt, .generation = m.generation, .prefill_s = m.prefill.total, .decode_s = m.decode.total, .cached = m.cached, .cached_known = m.cached_known, .rounds = m.rounds, .drafted = m.drafted, .accepted = m.accepted };
    }

    /// The scrape body, ending in a newline.
    pub fn render(m: *Metrics, io: std.Io, w: *std.Io.Writer, engine: api.Engine, window: u32) !void {
        var status: api.Status = .{};
        var streams: [512]u32 = undefined;
        engine.status(&status, &streams);
        var snap: Metrics = undefined;
        const requests = blk: {
            m.mutex.lockUncancelable(io);
            defer m.mutex.unlock(io);
            snap = m.*;
            break :blk try m.gpa.dupe(@TypeOf(m.requests.items[0]), m.requests.items);
        };
        defer m.gpa.free(requests);
        std.mem.sort(@TypeOf(requests[0]), requests, {}, struct {
            fn less(_: void, x: @TypeOf(requests[0]), y: @TypeOf(requests[0])) bool {
                const o = std.mem.order(u8, x.key, y.key);
                return o == .lt or (o == .eq and x.status < y.status);
            }
        }.less);
        if (requests.len > 0) {
            try family(w, "requests_total", "counter", "HTTP replies by API key label and status.");
            for (requests) |r| try w.print("{s}requests_total{{key=\"{s}\",status=\"{d}\"}} {d}\n", .{ prefix, r.key, r.status, r.count });
        }
        try gauge(w, "requests_running", "gauge", "Requests in prefill or decode.", status.running);
        try gauge(w, "requests_waiting", "gauge", "Requests queued or held until a lane is free.", status.waiting);
        try gauge(w, "generation_tokens_running", "gauge", "Generated tokens held by live streams.", status.generation_tokens);
        try gauge(w, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.", snap.prompt);
        try gauge(w, "generation_tokens_total", "counter", "Generated tokens of finished requests.", snap.generation);
        const live = streams[0..@min(status.streams, streams.len)];
        try family(w, "kv_cache_usage_ratio", "gauge", "Tokens in a stream cache divided by that stream's context window.");
        try pools(w, "kv_cache_usage_ratio", "pool", live, window);
        try gauge(w, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.", snap.drafted);
        try gauge(w, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try histogram(w, "request_latency_seconds", "Seconds from arrival to the reply leaving.", snap.latency);
        try histogram(w, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", snap.ttft);
        try histogram(w, "request_decode_seconds", "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.", snap.decode);
        try histogram(w, "request_time_per_output_token_seconds", "Seconds from the first generated token to the last, divided by the tokens less one.", snap.tpot);
        try gauge(w, "decode_rounds_total", "counter", "Decode rounds used by finished requests.", snap.rounds);
        try histogram(w, "request_prefill_seconds", "Seconds a finished request's prompt pass took.", snap.prefill);
        if (footprint()) |bytes| try gauge(w, "process_footprint_bytes", "gauge", "This process's physical footprint as the OS counts it, Metal buffers included; only where the platform reports one (macOS).", bytes);
        if (engine.memory(false)) |mem| {
            try gauge(w, "device_memory_bytes", "gauge", "Device memory the engine holds now: weights, caches and live streams; only where the backend counts it (CUDA).", mem.active);
            try gauge(w, "device_memory_peak_bytes", "gauge", "The most device memory the engine has held since start or the last /health?reset_peak=1.", mem.peak);
        }
        try gauge(w, "num_requests_running", "gauge", "Requests in prefill or decode. A mirror of tensorfold:requests_running.", status.running);
        try gauge(w, "num_requests_waiting", "gauge", "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.", status.waiting);
        try family(w, "kv_cache_usage_perc", "gauge", "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.");
        try pools(w, "kv_cache_usage_perc", "stream", live, window);
        try gauge(w, "spec_decode_num_draft_tokens_total", "counter", "Draft tokens verified on finished requests, this server's single draft counter.", snap.drafted);
        try gauge(w, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try histogram(w, "e2e_request_latency_seconds", "Seconds from arrival to the reply leaving, under vLLM's name.", snap.latency);
        try histogram(w, "request_decode_time_seconds", "Seconds a finished request spent decoding, under vLLM's name.", snap.decode);
        try histogram(w, "request_prefill_time_seconds", "Seconds a finished request's prompt pass took, under vLLM's name.", snap.prefill);
        try gauge(w, "client_disconnections_total", "counter", "Requests a client left before the reply left the server.", snap.disconnects);
        if (status.preemptions) |p| try gauge(w, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.", p);
        // server/metrics.py HEALTH: the CUDA server's /health totals and readings under tensorfold_health: (monitors
        // take the prefill / decode rates from these seconds); a field this server cannot count
        // (cached_tokens_total until an engine reports its prompt cache's hits, e.g. GLM-5.3's slot cache) is left out, as there
        var buf: [40]u8 = undefined;
        try healthFamily(w, "requests_total", "counter", "Finished requests (as /health's requests_total).", "{d}", snap.latency.n);
        if (status.progress) |p| {
            // Fiore-lab: an engine that reports its own totals (the GLM-5.3 host) gives them with the running request's
            // progress in them, read under one lock: every pair a monitor divides (tokens over seconds) moves together
            // and no total falls back when a request ends
            try healthFamily(w, "completion_tokens_total", "counter", "Reply tokens, the running reply's tokens so far included.", "{d}", p.generated_total);
            try healthFamily(w, "rounds_total", "counter", "Decode rounds, the running reply's so far included.", "{d}", p.rounds_total);
            try healthFamily(w, "drafted_total", "counter", "Draft tokens verified, the running reply's so far included.", "{d}", p.drafted_total);
            try healthFamily(w, "accepted_total", "counter", "Draft tokens kept, the running reply's so far included.", "{d}", p.accepted_total);
            try healthFamily(w, "prompt_tokens_total", "counter", "Prompt tokens, a running request's counted as its prefill starts.", "{d}", p.prompt_total);
            try healthFamily(w, "cached_tokens_total", "counter", "Prompt tokens resumed from the prompt cache, a running request's counted as its prefill starts.", "{d}", p.cached_total);
            try healthFamily(w, "prefill_tokens_total", "counter", "Prompt tokens the prefill computed (resumed ones excluded), the running prefill's chunks so far included.", "{d}", p.computed_total);
            try healthFamily(w, "prefill_seconds_total", "counter", "Engine prefill seconds, the running prefill's so far included.", "{s}", num(&buf, p.prefill_seconds_total));
            try healthFamily(w, "decode_seconds_total", "counter", "Engine decode seconds, the running reply's so far included.", "{s}", num(&buf, p.decode_seconds_total));
        } else {
            try healthFamily(w, "completion_tokens_total", "counter", "Reply tokens, the running replies' tokens so far included.", "{d}", snap.generation + status.generation_tokens);
            try healthFamily(w, "rounds_total", "counter", "Decode rounds of finished requests.", "{d}", snap.rounds);
            try healthFamily(w, "prefill_seconds_total", "counter", "Engine prefill seconds of finished requests.", "{s}", num(&buf, snap.prefill.total));
            try healthFamily(w, "decode_seconds_total", "counter", "Engine decode seconds of finished requests.", "{s}", num(&buf, snap.decode.total));
            if (snap.cached_known) try healthFamily(w, "cached_tokens_total", "counter", "Prompt tokens found cached, finished requests.", "{d}", snap.cached);
        }
        try healthFamily(w, "context_length", "gauge", "Tokens a request's prompt and reply may hold (the server's effective window).", "{d}", window);
        try healthFamily(w, "streams_max", "gauge", "Streams the concurrent engine decodes at most (--parallel).", "{d}", engine.info().lanes);
        if (status.progress) |p| try renderProgress(w, &snap, p, status.pool, engine.info());
    }
};

/// Fiore-lab: what an engine that reports its progress adds (the GLM-5.3 host): the running request's phase and
/// position, the queue, refusals, the KV pool, and the engine's settings.
fn renderProgress(w: *std.Io.Writer, snap: *const Metrics, p: api.Progress, pool: ?api.PoolStatus, info: api.Info) !void {
    var buf: [40]u8 = undefined;
    try gauge(w, "requests_prefilling", "gauge", "1 while a request's prompt is being prefilled (or planned), else 0.", @intFromBool(p.phase == .prefill));
    try gauge(w, "requests_decoding", "gauge", "1 while a request is decoding its reply, else 0.", @intFromBool(p.phase == .decode));
    try gauge(w, "prompt_tokens_running", "gauge", "The running request's prompt tokens (0: none).", p.prompt);
    try gauge(w, "prompt_tokens_cached_running", "gauge", "Of them, the tokens resumed from the prompt cache.", p.cached);
    try gauge(w, "prefill_tokens_remaining", "gauge", "Prompt tokens the running prefill has still to compute.", if (p.phase == .prefill) p.prompt -| @max(p.prompt_done, p.cached) else 0);
    try gauge(w, "queued_prompt_tokens", "gauge", "Prompt tokens of the requests waiting in the queue.", p.queued_prompt);
    try gauge(w, "requests_rejected_total", "counter", "Requests refused with 503 because the engine's queue was full.", snap.rejected);
    try histogram(w, "queue_wait_seconds", "Seconds a finished request waited in the engine's queue before it ran.", snap.queue);
    if (pool) |ps| {
        try gauge(w, "pool_blocks", "gauge", "Blocks of the KV pool.", ps.blocks);
        try gauge(w, "pool_block_positions", "gauge", "Positions one block holds.", ps.block_positions);
        try gauge(w, "pool_free_blocks", "gauge", "Blocks no conversation holds.", ps.free_blocks);
        try gauge(w, "pool_shared_blocks", "gauge", "Blocks two or more conversations hold.", ps.shared_blocks);
        try gauge(w, "pool_conversations", "gauge", "Conversations kept in the pool.", ps.slots);
        try family(w, "pool_usage_ratio", "gauge", "Blocks in use over all blocks.");
        try w.print("{s}pool_usage_ratio {s}\n", .{ prefix, num(&buf, if (ps.blocks == 0) 0 else @as(f64, @floatFromInt(ps.blocks -| ps.free_blocks)) / @as(f64, @floatFromInt(ps.blocks))) });
        try gauge(w, "pool_evicted_total", "counter", "Conversations evicted to make room.", ps.evicted_total);
        try gauge(w, "pool_cold_total", "counter", "Requests that dropped every conversation first (a check failed).", ps.cold_total);
        try family(w, "pool_requests_total", "counter", "Requests by what the pool did: continue (their own conversation), share (another's start), new.");
        try w.print("{s}pool_requests_total{{action=\"continue\"}} {d}\n{s}pool_requests_total{{action=\"share\"}} {d}\n{s}pool_requests_total{{action=\"new\"}} {d}\n", .{ prefix, ps.continued_total, prefix, ps.shared_total, prefix, ps.new_total });
        try family(w, "pool_check_seconds_total", "counter", "Seconds the pool spent checking the rows requests resumed.");
        try w.print("{s}pool_check_seconds_total {s}\n", .{ prefix, num(&buf, ps.check_seconds_total) });
        try family(w, "pool_hash_seconds_total", "counter", "Seconds the pool spent hashing the rows requests wrote.");
        try w.print("{s}pool_hash_seconds_total {s}\n", .{ prefix, num(&buf, ps.hash_seconds_total) });
    }
    if (info.labels.len > 0) {
        try family(w, "engine_info", "gauge", "The engine's settings as labels (always 1).");
        try w.print("{s}engine_info{{{s}}} 1\n", .{ prefix, info.labels });
    }
}

const health_prefix = "tensorfold_health:";

fn healthFamily(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, comptime fmt: []const u8, value: anytype) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n{s}{s} " ++ fmt ++ "\n", .{ health_prefix, name, help, health_prefix, name, kind, health_prefix, name, value });
}

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n", .{ prefix, name, help, prefix, name, kind });
}

fn gauge(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, value: u64) !void {
    try family(w, name, kind, help);
    try w.print("{s}{s} {d}\n", .{ prefix, name, value });
}

fn pools(w: *std.Io.Writer, name: []const u8, label: []const u8, lengths: []const u32, window: u32) !void {
    if (lengths.len == 0) return w.print("{s}{s}{{{s}=\"0\"}} 0\n", .{ prefix, name, label });
    for (lengths, 0..) |n, i| {
        const ratio: f64 = if (window == 0) 0 else @min(1.0, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(window)));
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}{{{s}=\"{d}\"}} {s}\n", .{ prefix, name, label, i, num(&buf, ratio) });
    }
}

fn histogram(w: *std.Io.Writer, name: []const u8, help: []const u8, h: anytype) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} histogram\n", .{ prefix, name, help, prefix, name });
    var cumulative: u64 = 0;
    const edges = @TypeOf(h).bucket_edges;
    for (edges, 0..) |edge, i| {
        cumulative += h.counts[i];
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}_bucket{{le=\"{s}\"}} {d}\n", .{ prefix, name, trimmed(&buf, edge, 4), cumulative });
    }
    try w.print("{s}{s}_bucket{{le=\"+Inf\"}} {d}\n", .{ prefix, name, cumulative + h.counts[edges.len] });
    var buf: [40]u8 = undefined;
    try w.print("{s}{s}_sum {s}\n{s}{s}_count {d}\n", .{ prefix, name, num(&buf, h.total), prefix, name, h.n });
}

/// ``_num``: an integral value as an int, else six decimals without trailing zeros.
fn num(buf: []u8, value: f64) []const u8 {
    if (value == @trunc(value) and @abs(value) < 1e18) return std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(value))}) catch "0";
    return trimmed(buf, value, 6);
}

fn trimmed(buf: []u8, value: f64, comptime places: u8) []const u8 {
    const text = std.fmt.bufPrint(buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{places}) ++ "}", .{value}) catch return "0";
    return std.mem.trimEnd(u8, std.mem.trimEnd(u8, text, "0"), ".");
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// The process's physical footprint where macOS reports one (rusage_info_v4).
fn footprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

test "edges and numbers" {
    var buf: [40]u8 = undefined;
    try std.testing.expectEqualStrings("0.01", trimmed(&buf, 0.01, 4));
    try std.testing.expectEqualStrings("120", trimmed(&buf, 120.0, 4));
    try std.testing.expectEqualStrings("0.333333", num(&buf, 1.0 / 3.0));
    try std.testing.expectEqualStrings("2", num(&buf, 2.0));
}

test "metrics record rounds prefill and per-token timing" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 3, 0, 0, 2, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 1_024_000_000, 3), null, null);
    try std.testing.expectEqual(@as(u64, 2), m.rounds);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.n);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.n);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.counts[3]);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.counts[4]);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try histogram(&out.writer, "request_prefill_seconds", "prefill", m.prefill);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "request_prefill_seconds_sum 0.25") != null);
}

const RenderStub = struct {
    pub fn info(ctx: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.info_value;
    }
    pub fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    pub fn cancel(_: *anyopaque, _: api.Id) void {}
    pub fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        out.* = s.status_value;
    }
    pub fn memory(ctx: *anyopaque, _: bool) ?api.Memory {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.memory_value;
    }
    status_value: api.Status = .{ .running = 1, .waiting = 2, .generation_tokens = 3 },
    memory_value: ?api.Memory = null,
    info_value: api.Info = .{},

    pub fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
};

test "render exposes live counters, rounds, prefill and TPOT without cache family" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 3, 0, 0, 2, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 3_000_000_000, 3), null, null);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:generation_tokens_running 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:decode_rounds_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prefill_seconds_sum 0.25") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_time_per_output_token_seconds_sum 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "prompt_tokens_cached_total") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "device_memory_bytes") == null); // unknown memory: no gauge
    try std.testing.expect(std.mem.indexOf(u8, body, "requests_prefilling") == null); // no progress: none of its families
    try std.testing.expect(std.mem.indexOf(u8, body, "engine_info") == null);
    stub.memory_value = .{ .active = 7 << 30, .peak = 8 << 30 };
    var known: std.Io.Writer.Allocating = .init(gpa);
    defer known.deinit();
    try m.render(std.testing.io, &known.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_bytes 7516192768\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_peak_bytes 8589934592\n") != null);
}

test "an engine's progress gives live totals, the running request's phase, the queue, the pool and engine_info" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var m: Metrics = .{ .gpa = gpa };
    defer m.requests.deinit(gpa);
    m.note(io, 10, 3, 0, 0, 2, 1.0, null, null, 0.25, null, 4, 0.5);
    m.refused(io);
    var stub: RenderStub = .{};
    stub.status_value.progress = .{ .phase = .prefill, .prompt = 1000, .cached = 200, .prompt_done = 600, .queued_prompt = 50, .prompt_total = 1010, .cached_total = 204, .computed_total = 406, .generated_total = 7, .prefill_seconds_total = 1.5, .decode_seconds_total = 0.25, .rounds_total = 3, .drafted_total = 6, .accepted_total = 4 };
    stub.status_value.pool = .{ .blocks = 100, .free_blocks = 75, .block_positions = 256, .continued_total = 1, .new_total = 2, .check_seconds_total = 0.002 };
    stub.info_value = .{ .labels = "tp=\"8\",dcp=\"2\"" };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    for ([_][]const u8{
        "tensorfold:requests_prefilling 1\n",            "tensorfold:requests_decoding 0\n",
        "tensorfold:prefill_tokens_remaining 400\n",     "tensorfold:queued_prompt_tokens 50\n",
        "tensorfold:requests_rejected_total 1\n",        "tensorfold:queue_wait_seconds_count 1\n",
        "tensorfold_health:completion_tokens_total 7\n", "tensorfold_health:prefill_tokens_total 406\n",
        "tensorfold_health:prompt_tokens_total 1010\n",  "tensorfold_health:cached_tokens_total 204\n",
        "tensorfold_health:prefill_seconds_total 1.5\n", "tensorfold_health:decode_seconds_total 0.25\n",
        "tensorfold_health:drafted_total 6\n",           "tensorfold_health:accepted_total 4\n",
        "tensorfold:pool_usage_ratio 0.25\n",            "tensorfold:pool_requests_total{action=\"new\"} 2\n",
        "tensorfold:pool_check_seconds_total 0.002\n",   "tensorfold:engine_info{tp=\"8\",dcp=\"2\"} 1\n",
        "tensorfold:prompt_tokens_total 10\n", // the finished requests' families keep their meaning
    }) |line| {
        if (std.mem.indexOf(u8, body, line) == null) {
            std.debug.print("missing: {s}", .{line});
            return error.TestExpectedEqual;
        }
    }
    // decoding: nothing left to prefill
    stub.status_value.progress.?.phase = .decode;
    var out2: std.Io.Writer.Allocating = .init(gpa);
    defer out2.deinit();
    try m.render(io, &out2.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "tensorfold:prefill_tokens_remaining 0\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "tensorfold:requests_decoding 1\n") != null);
}

test "metrics snapshot allocation failure releases the mutex" {
    const io = std.testing.io;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer {
        for (m.requests.items) |r| gpa.free(r.key);
        m.requests.deinit(gpa);
    }
    m.httpRequest(io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, m.render(io, &out.writer, stub.engine(), 4096));
    const available = m.mutex.tryLock();
    // No other thread uses m: either tryLock acquired it, or render retained it.
    // Release before asserting so the unfixed regression terminates without blocking.
    m.mutex.unlock(io);
    try std.testing.expect(available);
    try std.testing.expect(failing.has_induced_failure);
    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(io, "client", 200);
    m.note(io, 10, 3, 0, 0, 2, 1.0, null, null, 0.25, null, null, null);
    try m.render(io, &out.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "requests_total{key=\"client\",status=\"200\"} 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "prompt_tokens_total 10") != null);
}

test "cached_tokens_total only once an engine reports its prompt cache" {
    const io = std.testing.io;
    var m: Metrics = .{ .gpa = std.testing.allocator };
    defer m.requests.deinit(std.testing.allocator);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    m.note(io, 10, 3, 0, 0, 2, 1.0, null, null, 0.25, null, null, null);
    try m.render(io, &out.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "cached_tokens_total") == null);
    m.note(io, 100, 3, 0, 0, 2, 1.0, null, null, 0.25, null, 64, null);
    m.note(io, 100, 3, 0, 0, 2, 1.0, null, null, 0.25, null, 0, null);
    var out2: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out2.deinit();
    try m.render(io, &out2.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "tensorfold_health:cached_tokens_total 64\n") != null);
}

test "request counter append failure frees its label" {
    var bytes: [1024]u8 = undefined;
    var buffer = std.heap.FixedBufferAllocator.init(&bytes);
    var failing = std.testing.FailingAllocator.init(buffer.allocator(), .{ .fail_index = 1 });
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer {
        for (m.requests.items) |r| gpa.free(r.key);
        m.requests.deinit(gpa);
    }
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 6), failing.allocated_bytes);
    try std.testing.expectEqual(@as(usize, 0), m.requests.items.len);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);

    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    try std.testing.expectEqualStrings("client", m.requests.items[0].key);
    const allocations = failing.allocations;
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(u64, 2), m.requests.items[0].count);
    try std.testing.expectEqual(allocations, failing.allocations);
}
