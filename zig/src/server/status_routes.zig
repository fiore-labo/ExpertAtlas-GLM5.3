// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const live = @import("live.zig");
const routes = @import("routes.zig");
const auth = @import("auth.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
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
        const r = try json.newObject(a);
        try r.put(a, "phase", .{ .string = @tagName(p.phase) });
        try r.put(a, "prompt_tokens", try json.intValue(a, p.prompt));
        try r.put(a, "cached_tokens", try json.intValue(a, p.cached));
        try r.put(a, "prompt_done", try json.intValue(a, p.prompt_done));
        try r.put(a, "prefill_tokens_remaining", try json.intValue(a, if (p.phase == .prefill) p.prompt -| @max(p.prompt_done, p.cached) else 0));
        try r.put(a, "generated_tokens", try json.intValue(a, p.generated));
        try r.put(a, "queued_prompt_tokens", try json.intValue(a, p.queued_prompt));
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
/// (TF_GLM53_EXPERT_LIVE), and GET /v1/memory (rank 0's memory). They show what the model computed for recent
/// requests: with no API key on, only a loopback client gets them.
pub fn experts(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!keyed and !auth.loopback(conn.peer)) return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/experts needs an API key on this server, or a loopback client\"}}");
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
    const q: api.ExpertsQuery = if (std.mem.eql(u8, route, "/v1/memory")) .memory else if (std.mem.eql(u8, route, "/v1/experts/swap")) .swap else if (std.mem.eql(u8, route, "/v1/experts/usage")) .{ .usage = window } else if (std.mem.eql(u8, route, "/v1/experts/recent")) .{ .recent = .{ .after = after, .max = max orelse 256, .text = .{ .ctx = srv, .bytes = tokenBytes } } } else if (std.mem.eql(u8, route, "/v1/experts/prefill")) .{ .prefill = .{ .after = after, .max = @min(max orelse 4, 64) } } else return routes.unknown(conn, a);
    const body = srv.engine.experts(a, q) orelse return routes.unknown(conn, a);
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
        return conn.sendJson(403, "{\"error\": {\"message\": \"/v1/experts needs an API key on this server, or a loopback client\"}}");
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

/// Fiore-lab: GET /panel, the control panel page (glm53/tools/expert_usage/atlas, build.py --panel), served when the
/// engine keeps the live data (TF_GLM53_EXPERT_LIVE on the glm53 host). The page itself holds no data; it reads the
/// endpoints above, with the API key the operator gives it.
pub fn panel(srv: *Server, conn: *Conn, a: Allocator) void {
    if (srv.engine.experts(a, .memory) == null) return routes.unknown(conn, a);
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
