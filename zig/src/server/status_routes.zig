// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const live = @import("live.zig");
const routes = @import("routes.zig");
const auth = @import("auth.zig");
const http_body = @import("http_body.zig");
const server_mod = @import("server.zig");
const Server = server_mod.Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names nothing about the model.
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    if (srv.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    try o.put(a, "warming", .{ .bool = status.warming });
    const memory = try json.newObject(a);
    if (srv.engine.memory(std.mem.indexOf(u8, conn.path, "reset_peak=1") != null)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
    }
    try o.put(a, "memory", .{ .object = memory });
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    if (srv.info.slot_positions) |n| try o.put(a, "slot_cache_positions", try json.intValue(a, n));
    if (status.pool) |ps| {
        const po = try json.newObject(a);
        inline for (.{ "blocks", "free_blocks", "slots", "shared_blocks", "evicted_total", "cold_total" }) |f| try po.put(a, f, try json.intValue(a, @field(ps, f)));
        if (status.progress != null) {
            inline for (.{ "block_positions", "continued_total", "shared_total", "new_total" }) |f| try po.put(a, f, try json.intValue(a, @field(ps, f)));
            inline for (.{ "check_seconds_total", "hash_seconds_total" }) |f| try po.put(a, f, .{ .float = @round(@field(ps, f) * 1e6) / 1e6 });
        }
        try o.put(a, "pool", .{ .object = po });
    }
    // the Python CUDA server's /health (cuda/health.py snapshot): ok / backend / busy / requests_running and the
    // finished requests' totals (completion tokens with the live replies' so far), context_length; monitors read these
    if (srv.info.cuda_fields) try cudaHealth(srv, a, o, &status);
    routes.sendValue(conn, a, 200, .{ .object = o });
}

fn cudaHealth(srv: *Server, a: Allocator, o: *json.Object, status: *const api.Status) !void {
    const t = srv.metrics.totals(srv.io);
    try o.put(a, "ok", .{ .bool = true });
    try o.put(a, "backend", .{ .string = "tensorfold" });
    try o.put(a, "busy", .{ .bool = status.running > 0 });
    try o.put(a, "requests_running", try json.intValue(a, status.running));
    try o.put(a, "requests_total", try json.intValue(a, t.requests));
    // Fiore-lab: the thinking defaults now (POST /v1/thinking changes them), on a model that thinks
    // Fiore-lab: 今の思考の既定 (POST /v1/thinking が変える)。思考するモデルだけ
    if (srv.think_close.len > 0) {
        const th = try json.newObject(a);
        try putThinking(a, th, srv.thinkingNow());
        try o.put(a, "thinking", .{ .object = th });
    }
    if (status.progress) |p| {
        // Fiore-lab: the engine's own totals, the running request's progress in them (as /metrics' tensorfold_health:
        // families), and that request's position
        try o.put(a, "prompt_tokens_total", try json.intValue(a, p.prompt_total));
        try o.put(a, "completion_tokens_total", try json.intValue(a, p.generated_total));
        try o.put(a, "prefill_seconds_total", .{ .float = @round(p.prefill_seconds_total * 1e6) / 1e6 });
        try o.put(a, "decode_seconds_total", .{ .float = @round(p.decode_seconds_total * 1e6) / 1e6 });
        try o.put(a, "cached_tokens_total", try json.intValue(a, p.cached_total));
        try o.put(a, "prefill_tokens_total", try json.intValue(a, p.computed_total));
        try o.put(a, "rounds_total", try json.intValue(a, p.rounds_total));
        try o.put(a, "drafted_total", try json.intValue(a, p.drafted_total));
        try o.put(a, "accepted_total", try json.intValue(a, p.accepted_total));
        try o.put(a, "busy_total", try json.intValue(a, p.busy_total));
        // Fiore-lab: TF_GLM53_MULTI as this engine runs it (the launcher's status.multi is its memory of the start)
        // Fiore-lab: このエンジンが動かしている TF_GLM53_MULTI (ランチャーの status.multi は起動の記録)
        if (p.multi > 0) {
            const m = try json.newObject(a);
            try m.put(a, "n", try json.intValue(a, p.multi));
            try m.put(a, "wait_s", try json.intValue(a, p.multi_wait_s));
            try putSchedule(a, m, p.multi_schedule, false);
            try o.put(a, "multi", .{ .object = m });
        }
        // Fiore-lab: where the RoCE proxy threads poll (TF_GLM53_ROCE_PROXY_CORES): "proxy_cores" (a725, x925, os) and
        // each rank's proxies ("world", then "dcp" when there is one): "cpu" (rank 0's read now, "live"; another's
        // where it was kept, null when the scheduler's) and "core" (A725, X925, or null)
        // Fiore-lab: RoCE の proxy スレッドが待つ所 (TF_GLM53_ROCE_PROXY_CORES): "proxy_cores" (a725、x925、os) と各ランクの
        // proxy ("world"、DCP があれば "dcp"): "cpu" (rank 0 のものは今読んだ値で "live"。ほかは留めた所、スケジューラ任せ
        // なら null) と "core" (A725、X925、null)
        if (p.roce.world > 0) {
            const ro = try json.newObject(a);
            try ro.put(a, "proxy_cores", .{ .string = @tagName(p.roce.mode) });
            const ranks = try a.alloc(Value, p.roce.world);
            for (ranks, p.roce.proxies[0..p.roce.world], 0..) |*slot, prs, rk| {
                const ro_r = try json.newObject(a);
                try ro_r.put(a, "rank", try json.intValue(a, rk));
                var n: usize = 0;
                for (prs) |q| n += @intFromBool(q.present);
                const list = try a.alloc(Value, n);
                var i: usize = 0;
                for (prs, [2][]const u8{ "world", "dcp" }) |q, g| {
                    if (!q.present) continue;
                    const e = try json.newObject(a);
                    try e.put(a, "group", .{ .string = g });
                    try e.put(a, "cpu", if (q.cpu) |c| try json.intValue(a, c) else .null);
                    try e.put(a, "core", switch (q.core) {
                        .unknown => .null,
                        .a725 => .{ .string = "A725" },
                        .x925 => .{ .string = "X925" },
                    });
                    try e.put(a, "live", .{ .bool = q.live });
                    list[i] = .{ .object = e };
                    i += 1;
                }
                try ro_r.put(a, "proxies", .{ .array = list });
                slot.* = .{ .object = ro_r };
            }
            try ro.put(a, "ranks", .{ .array = ranks });
            try o.put(a, "roce", .{ .object = ro });
        }
        const r = try json.newObject(a);
        try r.put(a, "phase", .{ .string = @tagName(p.phase) });
        try r.put(a, "prompt_tokens", try json.intValue(a, p.prompt));
        try r.put(a, "cached_tokens", try json.intValue(a, p.cached));
        try r.put(a, "prompt_done", try json.intValue(a, p.prompt_done));
        try r.put(a, "prefill_tokens_remaining", try json.intValue(a, if (p.phase == .prefill) p.prompt -| @max(p.prompt_done, p.cached) else 0));
        try r.put(a, "generated_tokens", try json.intValue(a, p.generated));
        try r.put(a, "queued_prompt_tokens", try json.intValue(a, p.queued_prompt));
        // Fiore-lab: the requests not admitted yet (TF_GLM53_MULTI: also those waiting for room in the pool) and the
        // oldest one's wait
        // Fiore-lab: まだ受け付けていない要求 (TF_GLM53_MULTI では pool の空きを待つものも) と、最も古いものの待ち時間
        try r.put(a, "waiting", try json.intValue(a, p.waiting));
        try r.put(a, "waiting_ms", try json.intValue(a, p.waiting_ms));
        // Fiore-lab: each request running now, the oldest admitted first (several when conversations decode
        // together; the single fields above tell the first): its slot of the decode window, its conversation's
        // number in GET /v1/pool, its stage and position
        // Fiore-lab: いま走っている要求それぞれ、受け付けの古い順 (会話を一緒に decode するときは複数。上の単独の
        // 項目は最初のもの): decode ウィンドウの枠、GET /v1/pool での会話の番号、段階と位置
        const reqs = try a.alloc(Value, p.n_requests);
        for (p.requests[0..p.n_requests], reqs) |q, *slot| {
            const e = try json.newObject(a);
            try e.put(a, "conv", try json.intValue(a, q.conv));
            if (q.pool_id) |id| try e.put(a, "pool_id", try json.intValue(a, id));
            try e.put(a, "phase", .{ .string = @tagName(q.phase) });
            try e.put(a, "prompt_tokens", try json.intValue(a, q.prompt));
            try e.put(a, "cached_tokens", try json.intValue(a, q.cached));
            try e.put(a, "prompt_done", try json.intValue(a, q.prompt_done));
            try e.put(a, "generated_tokens", try json.intValue(a, q.generated));
            try e.put(a, "elapsed_ms", try json.intValue(a, q.elapsed_ms));
            if (q.prefill_ms) |v| try e.put(a, "prefill_ms", try json.intValue(a, v));
            slot.* = .{ .object = e };
        }
        try r.put(a, "requests", .{ .array = reqs });
        try o.put(a, "progress", .{ .object = r });
    } else {
        try o.put(a, "prompt_tokens_total", try json.intValue(a, t.prompt));
        try o.put(a, "completion_tokens_total", try json.intValue(a, t.generation + status.generation_tokens));
        try o.put(a, "prefill_seconds_total", .{ .float = @round(t.prefill_s * 1e6) / 1e6 });
        try o.put(a, "decode_seconds_total", .{ .float = @round(t.decode_s * 1e6) / 1e6 });
        if (t.cached_known) try o.put(a, "cached_tokens_total", try json.intValue(a, t.cached));
        try o.put(a, "rounds_total", try json.intValue(a, t.rounds));
        try o.put(a, "drafted_total", try json.intValue(a, t.drafted));
        try o.put(a, "accepted_total", try json.intValue(a, t.accepted));
    }
    if (srv.info.context_window > 0) try o.put(a, "context_length", try json.intValue(a, srv.info.context_window));
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "created", try json.intValue(a, created));
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

/// Fiore-lab: GET /v1/experts/usage[?window=SECONDS], /v1/experts/recent[?after=SEQ&max=N] and
/// /v1/experts/prefill[?after=SEQ&max=N] (max 4 by default, at most 64), the live expert usage
/// (TF_GLM53_EXPERT_LIVE), GET /v1/memory (rank 0's memory) and GET /v1/pool (the KV pool's conversations and each
/// block's owner, TF_GLM53_PREFIX=pool). They show what the model computed for recent requests: with no API key on,
/// only a loopback client gets them.
pub fn experts(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!keyed and !auth.loopback(conn.peer)) return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/experts answers this node only (a loopback client, or an API key when the server has keys): from another machine use the launcher's port, PORT + 10\"}}");
    var window: u32 = 60;
    var after: ?u64 = null;
    var max: ?u32 = null;
    if (std.mem.indexOfScalar(u8, conn.path, '?')) |q| {
        var it = std.mem.tokenizeScalar(u8, conn.path[q + 1 ..], '&');
        while (it.next()) |kv| {
            const eq = std.mem.indexOfScalar(u8, kv, '=') orelse continue;
            const k = kv[0..eq];
            const v = kv[eq + 1 ..];
            if (std.mem.eql(u8, k, "window")) window = @min(std.fmt.parseInt(u32, v, 10) catch window, 3600);
            if (std.mem.eql(u8, k, "after")) after = std.fmt.parseInt(u64, v, 10) catch after;
            if (std.mem.eql(u8, k, "max")) max = @min(std.fmt.parseInt(u32, v, 10) catch (max orelse 256), 4096);
        }
    }
    const q: api.ExpertsQuery = if (std.mem.eql(u8, route, "/v1/memory")) .memory else if (std.mem.eql(u8, route, "/v1/pool")) .pool else if (std.mem.eql(u8, route, "/v1/experts/swap")) .swap else if (std.mem.eql(u8, route, "/v1/experts/usage")) .{ .usage = window } else if (std.mem.eql(u8, route, "/v1/experts/recent")) .{ .recent = .{ .after = after, .max = max orelse 256, .text = .{ .ctx = srv, .bytes = tokenBytes } } } else if (std.mem.eql(u8, route, "/v1/experts/prefill")) .{ .prefill = .{ .after = after, .max = @min(max orelse 4, 64) } } else return routes.unknown(conn, a);
    const body = srv.engine.experts(a, q) orelse return routes.unknown(conn, a);
    conn.sendJson(200, body);
}

/// Fiore-lab: GET /v1/experts/usage[?window=SECONDS] on the read-only listener (TENSORFOLD_METRICS_LISTEN), with no
/// key: the usage counts only (no token text, unlike /v1/experts/recent), for a monitor that keeps them.
pub fn usageReadOnly(srv: *Server, conn: *Conn, a: Allocator) void {
    var window: u32 = 60;
    if (std.mem.indexOfScalar(u8, conn.path, '?')) |q| {
        var it = std.mem.tokenizeScalar(u8, conn.path[q + 1 ..], '&');
        while (it.next()) |kv| {
            if (std.mem.startsWith(u8, kv, "window=")) window = @min(std.fmt.parseInt(u32, kv["window=".len..], 10) catch window, 3600);
        }
    }
    const body = srv.engine.experts(a, .{ .usage = window }) orelse return routes.unknown(conn, a);
    conn.sendJson(200, body);
}

/// The byte-level BPE byte alphabet backwards (the vocabulary: printable Latin-1 stands for itself, the other 68
/// bytes are U+0100 onwards), -1 for a character outside it.
/// byte-level BPE のバイトの表の逆 (語彙: 印字できる Latin-1 はそのまま、ほかの 68 バイトは U+0100 から)。
/// 表に無い文字は -1。
const char_bytes: [0x144]i16 = blk: {
    var table: [0x144]i16 = @splat(-1);
    var next: usize = 256;
    for (0..256) |b| {
        const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
        if (printable) table[b] = b else {
            table[next] = b;
            next += 1;
        }
    }
    break :blk table;
};

/// A token's raw bytes for /v1/experts/recent's text: null for an added token (the chat template's markers,
/// special or not: <|assistant|>, <think>, </think> ...), so the text is the reply's own; else the vocabulary string
/// mapped back through that byte alphabet; a string with a character outside it (not byte-level) as it is.
/// /v1/experts/recent の文字のための token の生のバイト。追加 token (チャットテンプレートの印。special でも
/// そうでなくても: <|assistant|>、<think>、</think> など) は null で、文字は返答そのものになる。それ以外は語彙の文字列を
/// バイトの表で戻す。表に無い文字を含む文字列 (byte-level でない) はそのまま。
fn tokenBytes(ctx: *anyopaque, a: Allocator, id: u32) ?[]const u8 {
    const srv: *Server = @ptrCast(@alignCast(ctx));
    if (id >= srv.text.vocabSize() or srv.text.isAdded(id)) return null;
    const s = srv.text.tokenString(a, id) catch return null;
    const out = a.alloc(u8, s.len) catch return null;
    var n: usize = 0;
    const view = std.unicode.Utf8View.init(s) catch return s;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (cp >= char_bytes.len or char_bytes[cp] < 0) return s;
        out[n] = @intCast(char_bytes[cp]);
        n += 1;
    }
    return out[0..n];
}

/// Fiore-lab: POST /v1/experts/swap {"bits": base64, one byte an expert, the MoE layers in order} (run between
/// requests; 409 while another change runs or a request does) and POST /v1/experts/save {"name": ...} (the bits in
/// use as a named set); the same access as GET /v1/experts.
pub fn expertsPost(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!keyed and !auth.loopback(conn.peer)) {
        routes.discardBody(conn, a);
        return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/experts answers this node only (a loopback client, or an API key when the server has keys): from another machine use the launcher's port, PORT + 10\"}}");
    }
    const bad = "{\"error\": {\"message\": \"a JSON object: {\\\"bits\\\": base64} for /v1/experts/swap, {\\\"name\\\": ...} for /v1/experts/save\"}}";
    const raw = switch (http_body.read(conn, a, 1 << 20) catch return) {
        .ok => |b| b,
        .refused => return conn.sendJson(400, bad),
    };
    const In = struct { bits: ?[]const u8 = null, name: ?[]const u8 = null };
    const in = std.json.parseFromSliceLeaky(In, a, raw, .{ .ignore_unknown_fields = true }) catch return conn.sendJson(400, bad);
    const c: api.ExpertsControl = if (std.mem.eql(u8, route, "/v1/experts/swap")) blk: {
        const b64 = in.bits orelse return conn.sendJson(400, bad);
        const dec = std.base64.standard.Decoder;
        const n = dec.calcSizeForSlice(b64) catch return conn.sendJson(400, bad);
        const bytes = a.alloc(u8, n) catch return;
        dec.decode(bytes, b64) catch return conn.sendJson(400, bad);
        break :blk .{ .swap = bytes };
    } else if (std.mem.eql(u8, route, "/v1/experts/save")) .{ .save = in.name orelse return conn.sendJson(400, bad) } else return routes.unknown(conn, a);
    const body = srv.engine.expertsControl(a, c) catch |err| {
        const code: u16 = switch (err) {
            error.Unsupported => 404,
            error.Busy, error.Exists, error.UnsupportedWithMulti => 409,
            error.BadRequest => 400,
            error.Failed, error.OutOfMemory => 500,
        };
        const msg = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{@errorName(err)}) catch return;
        return conn.sendJson(code, msg);
    };
    conn.sendJson(200, body);
}

/// Fiore-lab: TF_GLM53_MULTI's schedule as JSON: "decode_share" (percent), "short_prompt" (rows), "short_preempt",
/// "measured" (percent, or null) and "contended"; with `start` (GET / POST /v1/schedule, not /health) the values the
/// engine started with under "start".
fn putSchedule(a: Allocator, o: *json.Object, s: api.Schedule, start: bool) !void {
    try o.put(a, "decode_share", try json.intValue(a, s.decode_share));
    try o.put(a, "short_prompt", try json.intValue(a, s.short_prompt));
    try o.put(a, "short_preempt", .{ .bool = s.short_preempt });
    try o.put(a, "measured", if (s.measured) |m| try json.intValue(a, m) else .null);
    try o.put(a, "contended", .{ .bool = s.contended });
    if (start) if (s.start) |st| {
        const so = try json.newObject(a);
        try so.put(a, "decode_share", try json.intValue(a, st.decode_share));
        try so.put(a, "short_prompt", try json.intValue(a, st.short_prompt));
        try so.put(a, "short_preempt", .{ .bool = st.short_preempt });
        try o.put(a, "start", .{ .object = so });
    };
}

/// Fiore-lab: GET /v1/schedule, how TF_GLM53_MULTI shares the GPU's time between a prompt being computed and the live
/// conversations' decode while both want it, and POST /v1/schedule {"decode_share": 0 .. 100, "short_prompt": rows,
/// "short_preempt": bool} (any of them) to change it while the engine runs (0: the prompts first, 100: the decode
/// first; a running prefill takes it at its next pause. A prompt with at most short_prompt rows to compute goes before
/// the longer ones and is not held by the share; 0: none. short_preempt: a longer prefill running gives way to it).
/// Both answer the schedule in use. The same access as /v1/experts; 404 without TF_GLM53_MULTI.
pub fn schedule(srv: *Server, conn: *Conn, a: Allocator, post: bool) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!keyed and !auth.loopback(conn.peer)) {
        if (post) routes.discardBody(conn, a);
        return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/schedule answers this node only (a loopback client, or an API key when the server has keys): from another machine use the launcher's port, PORT + 10\"}}");
    }
    const bad = "{\"error\": {\"message\": \"a JSON object with any of {\\\"decode_share\\\": 0 .. 100, \\\"short_prompt\\\": 0 .. 1048576, \\\"short_preempt\\\": true or false}\"}}";
    var set: api.ScheduleSet = .{};
    if (post) {
        const raw = switch (http_body.read(conn, a, 1 << 12) catch return) {
            .ok => |b| b,
            .refused => return conn.sendJson(400, bad),
        };
        set = std.json.parseFromSliceLeaky(api.ScheduleSet, a, raw, .{ .ignore_unknown_fields = true }) catch return conn.sendJson(400, bad);
        if (set.decode_share == null and set.short_prompt == null and set.short_preempt == null) return conn.sendJson(400, bad);
        if (set.decode_share) |v| if (v > 100) return conn.sendJson(400, bad);
        if (set.short_prompt) |v| if (v > api.short_prompt_max) return conn.sendJson(400, bad);
    }
    const s = srv.engine.schedule(set) catch |err| return switch (err) {
        error.BadRequest => conn.sendJson(400, bad),
        error.Unsupported => conn.sendJson(404, "{\"error\": {\"message\": \"/v1/schedule needs TF_GLM53_MULTI (conversations decoded together)\"}}"),
        else => conn.sendJson(500, "{\"error\": {\"message\": \"the schedule could not be read\"}}"),
    };
    const o = json.newObject(a) catch return;
    putSchedule(a, o, s, true) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Fiore-lab: the thinking defaults as JSON: "share" (percent of a reply's max_tokens), "budget" (tokens; 0: the share
/// decides), "fold" (off, recent, checkpoint), "at_close_share" (percent of the thinking's limit; 0: never), "fold_share" (a fold no later than this
/// percent of a reply's max_tokens, at least 1024 tokens; 0: none), "fold_max" (a reply's folds at most; 0: no limit),
/// "at_budget" (1: the summary at the budget's end when a request names no steps), "fold_margin" (how far before that
/// end it starts, a percent of the budget), "fold_prompt" and "close_prompt" (what the model is asked there and at its
/// own close; empty in a POST: back to the built-in text, the "_default" fields).
fn putThinking(a: Allocator, o: *json.Object, t: server_mod.Thinking) !void {
    try o.put(a, "share", try json.intValue(a, t.share));
    try o.put(a, "budget", try json.intValue(a, t.budget));
    try o.put(a, "fold", .{ .string = @tagName(t.fold) });
    try o.put(a, "at_close_share", try json.intValue(a, t.at_close_share));
    try o.put(a, "fold_share", try json.intValue(a, t.fold_share));
    try o.put(a, "fold_max", try json.intValue(a, t.fold_max));
    try o.put(a, "at_budget", try json.intValue(a, t.at_budget));
    try o.put(a, "fold_margin", try json.intValue(a, t.fold_margin));
    try o.put(a, "fold_prompt", .{ .string = t.fold_prompt });
    try o.put(a, "fold_prompt_default", .{ .string = server_mod.fold_prompt_default });
    try o.put(a, "close_prompt", .{ .string = t.close_prompt });
    try o.put(a, "close_prompt_default", .{ .string = server_mod.close_prompt_default });
}

/// What a POST /v1/thinking body may change, as the request fields check them: share 0 .. 100, fold one of off, recent,
/// checkpoint, branch (a fold needs the model's fold texts: `can_fold`, for branch `can_branch`), at_close_share 0 .. 100
/// (above 0 it needs the summary's text: `can_close`), fold_share 0 .. 100, fold_max 0 .. 1000; at least one of
/// them. Null when the body holds none of them right.
pub fn thinkingCheck(raw: []const u8, a: Allocator, can_fold: bool, can_close: bool, can_branch: bool) ?server_mod.ThinkingSet {
    const Body = struct { share: ?i64 = null, fold: ?[]const u8 = null, at_close_share: ?i64 = null, fold_share: ?i64 = null, fold_max: ?i64 = null, at_budget: ?i64 = null, fold_margin: ?i64 = null, fold_prompt: ?[]const u8 = null, close_prompt: ?[]const u8 = null };
    const b = std.json.parseFromSliceLeaky(Body, a, raw, .{ .ignore_unknown_fields = true }) catch return null;
    if (b.share == null and b.fold == null and b.at_close_share == null and b.fold_share == null and b.fold_max == null and b.at_budget == null and b.fold_margin == null and b.fold_prompt == null and b.close_prompt == null) return null;
    var set: server_mod.ThinkingSet = .{};
    if (b.share) |v| {
        if (v < 0 or v > 100) return null;
        set.share = v;
    }
    if (b.fold) |v| {
        const f = std.meta.stringToEnum(api.Fold, v) orelse return null;
        if (f != .off and !(if (f == .branch) can_branch else can_fold)) return null;
        set.fold = f;
    }
    if (b.at_close_share) |v| {
        if (v < 0 or v > 100 or (v > 0 and !can_close)) return null;
        set.at_close_share = v;
    }
    if (b.fold_share) |v| {
        if (v < 0 or v > 100) return null;
        set.fold_share = v;
    }
    if (b.fold_max) |v| {
        if (v < 0 or v > 1000) return null;
        set.fold_max = v;
    }
    if (b.at_budget) |v| {
        if (v != 0 and v != 1) return null;
        set.at_budget = v;
    }
    if (b.fold_margin) |v| {
        if (v < 0 or v > 50) return null;
        set.fold_margin = v;
    }
    if (b.fold_prompt) |v| {
        if (v.len > 4096) return null;
        set.fold_prompt = v;
    }
    if (b.close_prompt) |v| {
        if (v.len > 4096) return null;
        set.close_prompt = v;
    }
    return set;
}

/// Fiore-lab: GET /v1/thinking, the thinking defaults a request takes when it names none, and POST /v1/thinking
/// {"share": 0 .. 100, "fold": "off" | "recent" | "checkpoint", "at_close_share": 0 .. 100} (any of them) to change them while the
/// engine runs: the requests that start after it take them (a request's own thinking_budget, thinking_fold and
/// thinking_fold_at_close still come first; the running ones keep theirs). A share sets the token budget to 0, since a
/// budget would win over it. Both answer the values in use, "start" (what the server started with), and, read only,
/// "fold_window", "fold_cap", "notice" and "max_tokens" (the reply's limit when the request names none). The same
/// access as /v1/schedule; 404 on a model that does not think.
/// Fiore-lab: GET /v1/thinking は要求が指定しないときの思考の既定。POST /v1/thinking で動作中に変える (その後に始まる要求
/// から。要求自身の thinking_budget などが先、走っている要求はそのまま)。share を変えるとトークンの予算は 0 (予算が
/// share より先に効くため)。どちらも今の値・起動時の値 (start)・読むだけの fold_window などを返す。/v1/schedule と同じ制限
pub fn thinking(srv: *Server, conn: *Conn, a: Allocator, post: bool) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!keyed and !auth.loopback(conn.peer)) {
        if (post) routes.discardBody(conn, a);
        return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/thinking answers this node only (a loopback client, or an API key when the server has keys): from another machine use the launcher's port, PORT + 10\"}}");
    }
    if (srv.think_close.len == 0) {
        if (post) routes.discardBody(conn, a);
        return conn.sendJson(404, "{\"error\": {\"message\": \"/v1/thinking: this model does not think\"}}");
    }
    const bad = "{\"error\": {\"message\": \"a JSON object with any of {\\\"share\\\": 0 .. 100, \\\"fold\\\": \\\"off\\\" | \\\"recent\\\" | \\\"checkpoint\\\" | \\\"branch\\\", \\\"at_close_share\\\": 0 .. 100, \\\"fold_share\\\": 0 .. 100, \\\"fold_max\\\": 0 .. 1000, \\\"at_budget\\\": 0 | 1, \\\"fold_margin\\\": 0 .. 50, \\\"fold_prompt\\\": text, \\\"close_prompt\\\": text}\"}}";
    var now: server_mod.Thinking = undefined;
    if (post) {
        const raw = switch (http_body.read(conn, a, 1 << 14) catch return) {
            .ok => |b| b,
            .refused => return conn.sendJson(400, bad),
        };
        const set = thinkingCheck(raw, a, srv.canFold(.checkpoint), srv.fold_close_open.len > 0, srv.canFold(.branch)) orelse return conn.sendJson(400, bad);
        now = srv.thinkingSet(set);
    } else now = srv.thinkingNow();
    const o = json.newObject(a) catch return;
    putThinking(a, o, now) catch return;
    const so = json.newObject(a) catch return;
    putThinking(a, so, srv.thinking_start) catch return;
    o.put(a, "start", .{ .object = so }) catch return;
    o.put(a, "fold_window", json.intValue(a, srv.config.thinking_fold_window) catch return) catch return;
    o.put(a, "fold_cap", json.intValue(a, srv.config.thinking_fold_cap) catch return) catch return;
    o.put(a, "fold_cap_share", json.intValue(a, srv.config.thinking_fold_cap_share) catch return) catch return;
    o.put(a, "notice", json.intValue(a, srv.config.thinking_notice) catch return) catch return;
    o.put(a, "max_tokens", json.intValue(a, srv.config.default_max_tokens) catch return) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Fiore-lab: GET /panel, the control panel page (glm53/tools/expert_usage/atlas, build.py --panel), served when the
/// engine keeps the live data (TF_GLM53_EXPERT_LIVE on the glm53 host). The page itself holds no data; it reads the
/// endpoints above, with the API key the operator gives it.
pub fn panel(srv: *Server, conn: *Conn, a: Allocator) void {
    if (srv.engine.experts(a, .memory) == null) return routes.unknown(conn, a);
    // TF_PANEL_LAUNCHER_PORT (tf-glm53 sets it on rank 0 while the lane's launcher runs): the panel's entrance is the
    // launcher, which starts and stops the engine and reads its counts from inside the node, so this page sends the
    // browser there, on the host the browser used
    // TF_PANEL_LAUNCHER_PORT (レーンのランチャーが動いている時に tf-glm53 が rank 0 に設定する): パネルの入口はランチャー
    // (エンジンを起動・停止し、回数をノードの中から読む)。このページはブラウザをそこへ送る (ブラウザが使ったホストのまま)
    if (std.c.getenv("TF_PANEL_LAUNCHER_PORT")) |p| if (launcherPanel(a, conn.header("host"), std.mem.span(p))) |url| {
        conn.startResponse(302, "Found") catch return;
        conn.addHeader("Location", url) catch return;
        conn.addHeader("Content-Length", "0") catch return;
        conn.finish("") catch {};
        return;
    };
    // TF_PANEL_HTML (development): the page read from this file on each request instead of the built-in one
    // TF_PANEL_HTML (開発用): 組み込みのページの代わりに、要求ごとにこのファイルから読む
    const page = if (std.c.getenv("TF_PANEL_HTML")) |p| std.Io.Dir.cwd().readFileAlloc(srv.io, std.mem.span(p), a, .limited(8 << 20)) catch panel_page else panel_page;
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{page.len}) catch return) catch return;
    conn.finish(page) catch {};
}
const panel_page = @embedFile("panel.html");

/// The launcher's panel for the browser's Host header: http://HOST:PORT/panel, its port replaced by `port`; null when
/// either is missing or not plain (a port of digits; a name of letters, digits, dots and hyphens, or an IPv6 address
/// in brackets).
/// ブラウザの Host ヘッダーに対するランチャーのパネル: http://HOST:PORT/panel (ポートを `port` に替える)。どちらかが
/// 無いか素直でない時は null (ポートは数字だけ。名前は英数字・点・ハイフン、または角括弧の IPv6 アドレス)。
fn launcherPanel(a: Allocator, host_header: ?[]const u8, port: []const u8) ?[]const u8 {
    const h = host_header orelse return null;
    if (port.len == 0 or port.len > 5) return null;
    for (port) |ch| if (!std.ascii.isDigit(ch)) return null;
    if (std.mem.startsWith(u8, h, "[")) {
        const end = std.mem.indexOfScalar(u8, h, ']') orelse return null;
        for (h[1..end]) |ch| if (!(std.ascii.isHex(ch) or ch == ':' or ch == '.')) return null;
        return std.fmt.allocPrint(a, "http://{s}:{s}/panel", .{ h[0 .. end + 1], port }) catch null;
    }
    const name = if (std.mem.indexOfScalar(u8, h, ':')) |i| h[0..i] else h;
    if (name.len == 0) return null;
    for (name) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-')) return null;
    return std.fmt.allocPrint(a, "http://{s}:{s}/panel", .{ name, port }) catch null;
}

test "thinkingCheck takes the fields a POST /v1/thinking may change, as the request fields check them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = thinkingCheck("{\"share\": 60, \"fold\": \"checkpoint\", \"at_close_share\": 5}", a, true, true, true).?;
    try std.testing.expectEqual(@as(?i64, 60), s.share);
    try std.testing.expectEqual(@as(?api.Fold, .checkpoint), s.fold);
    try std.testing.expectEqual(@as(?i64, 5), s.at_close_share);
    try std.testing.expectEqual(@as(?i64, null), thinkingCheck("{\"fold\": \"off\"}", a, false, false, false).?.share);
    try std.testing.expect(thinkingCheck("{}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"share\": 101}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"share\": -1}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"fold\": \"half\"}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"fold\": \"recent\"}", a, false, true, false) == null);
    try std.testing.expect(thinkingCheck("{\"at_close_share\": -5}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"at_close_share\": 101}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"at_close_share\": 1}", a, true, false, true) == null);
    try std.testing.expectEqual(@as(?i64, 0), thinkingCheck("{\"at_close_share\": 0}", a, false, false, false).?.at_close_share);
    try std.testing.expect(thinkingCheck("not json", a, true, true, true) == null);
    try std.testing.expectEqual(@as(?i64, 0), thinkingCheck("{\"at_budget\": 0}", a, false, false, false).?.at_budget);
    try std.testing.expect(thinkingCheck("{\"at_budget\": 2}", a, true, true, true) == null);
    try std.testing.expectEqual(@as(?i64, 10), thinkingCheck("{\"fold_margin\": 10}", a, false, false, false).?.fold_margin);
    try std.testing.expect(thinkingCheck("{\"fold_margin\": 51}", a, true, true, true) == null);
    try std.testing.expectEqualStrings("Sort it.", thinkingCheck("{\"fold_prompt\": \"Sort it.\"}", a, false, false, false).?.fold_prompt.?);
    try std.testing.expectEqualStrings("", thinkingCheck("{\"close_prompt\": \"\"}", a, false, false, false).?.close_prompt.?);
    try std.testing.expectEqual(@as(?i64, 25), thinkingCheck("{\"fold_share\": 25}", a, false, false, false).?.fold_share);
    try std.testing.expect(thinkingCheck("{\"fold_share\": 101}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"fold_share\": -1}", a, true, true, true) == null);
    try std.testing.expectEqual(@as(?i64, 3), thinkingCheck("{\"fold_max\": 3}", a, false, false, false).?.fold_max);
    try std.testing.expectEqual(@as(?i64, 0), thinkingCheck("{\"fold_max\": 0}", a, false, false, false).?.fold_max);
    try std.testing.expect(thinkingCheck("{\"fold_max\": 1001}", a, true, true, true) == null);
    try std.testing.expect(thinkingCheck("{\"fold_max\": -1}", a, true, true, true) == null);
    try std.testing.expectEqual(@as(?api.Fold, .branch), thinkingCheck("{\"fold\": \"branch\"}", a, false, false, true).?.fold);
    try std.testing.expect(thinkingCheck("{\"fold\": \"branch\"}", a, true, true, false) == null);
}

test "launcherPanel sends the browser to the launcher on the host it used" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("http://203.0.113.200:8944/panel", launcherPanel(a, "203.0.113.200:8934", "8944").?);
    try std.testing.expectEqualStrings("http://node-0.example.ts.net:8944/panel", launcherPanel(a, "node-0.example.ts.net:8934", "8944").?);
    try std.testing.expectEqualStrings("http://localhost:8944/panel", launcherPanel(a, "localhost", "8944").?);
    try std.testing.expectEqualStrings("http://[::1]:8944/panel", launcherPanel(a, "[::1]:8934", "8944").?);
    try std.testing.expect(launcherPanel(a, null, "8944") == null);
    try std.testing.expect(launcherPanel(a, "evil.example/x:80", "8944") == null);
    try std.testing.expect(launcherPanel(a, "host\r\nSet-Cookie: x", "8944") == null);
    try std.testing.expect(launcherPanel(a, "203.0.113.200:8934", "89a4") == null);
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
}
