// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const prompt_mod = @import("prompt.zig");
const log = @import("log.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const server_mod = @import("server.zig");
const Server = server_mod.Server;
const chunk_plan = @import("chunk_plan.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };

/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
};

/// A streamed piece: content text (a string) or a delta object (reasoning or tool calls). ``ping``, when the stream
/// has one, keeps a silent stream alive (a long prefill sends nothing for minutes; Claude Code drops a stream after
/// 300 s of silence): ``Generation.loop`` calls it after ``pingInterval`` without a write.
pub const Sink = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, delta: Value) error{Closed}!void,
    ping: ?*const fn (ctx: *anyopaque) error{Closed}!void = null,
};

/// TF_SSE_PING_S: seconds of stream silence before a keep-alive (default 10; 0 turns it off), in nanoseconds.
fn pingInterval() i96 {
    const v = std.c.getenv("TF_SSE_PING_S") orelse return 10 * std.time.ns_per_s;
    const s = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return 10 * std.time.ns_per_s;
    return @as(i96, s) * std.time.ns_per_s;
}

pub const Reply = struct {
    content: []const u8,
    stop_sequence: ?[]const u8,
    reasoning: ?[]const u8,
    tool_calls_streamed: bool,
    finish_reason: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    runtime: Value,
    speculative: Value,
    /// A folded reply's reasoning as the model's context held it (the head, the label, the last checkpoint and the
    /// thinking after it), for the client to send back next turn in place of ``reasoning``; null: not folded.
    context_reasoning: ?[]const u8 = null,
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

/// Events the engine thread hands this request, read by the request's own thread.
const Mailbox = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    tokens: std.ArrayList(u32) = .empty,
    chunks: std.ArrayList(usize) = .empty, // where each round's tokens end
    cached: ?u32 = null,
    prefilled_ns: ?i96 = null,
    finished: bool = false,
    reason: api.Reason = .stop,
    stats: api.Stats = .{},
    widths: []u32 = &.{},
    raised: []bool = &.{},
    telemetry: []u8 = "",
    message: []u8 = "",
    context: []u32 = &.{}, // a folded reply as the model's context held it (api.Stats.context_reply)

    fn onEvent(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
        const m: *Mailbox = @ptrCast(@alignCast(ctx));
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        switch (event.*) {
            .prefilled => |cached| {
                m.cached = cached;
                m.prefilled_ns = std.Io.Clock.awake.now(m.io).toNanoseconds();
            },
            .tokens => |t| {
                m.tokens.appendSlice(m.gpa, t) catch {};
                m.chunks.append(m.gpa, m.tokens.items.len) catch {};
            },
            .finished => |f| {
                m.finished = true;
                m.reason = f.reason;
                m.stats = f.stats;
                m.widths = m.gpa.dupe(u32, f.stats.prefill_widths) catch &.{};
                m.raised = m.gpa.dupe(bool, f.stats.prefill_raised) catch &.{};
                m.telemetry = m.gpa.dupe(u8, f.stats.telemetry_json) catch "";
                m.message = m.gpa.dupe(u8, f.message) catch "";
                m.context = m.gpa.dupe(u32, f.stats.context_reply) catch &.{};
            },
        }
        m.cond.signal(m.io);
    }

    fn deinit(m: *Mailbox) void {
        m.tokens.deinit(m.gpa);
        m.chunks.deinit(m.gpa);
        m.gpa.free(m.widths);
        m.gpa.free(m.raised);
        m.gpa.free(m.telemetry);
        m.gpa.free(m.message);
        m.gpa.free(m.context);
    }
};

const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// Python's ``bool(fields.get(key))`` on the sampling fields.
fn flag(f: Value, key: []const u8) ?bool {
    const v = f.get(key) orelse return null;
    if (v == .null) return null;
    return v.truthy();
}

/// A request rendered and checked but not submitted; it holds the preparing count until generate or release.
pub const Prepared = struct {
    input: Input,
    request: api.Request,
    prompt_len: usize,
    received: i96,
    thinking: bool,
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    stops: fields_mod.Stops,
    preparing: bool,
};

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.
pub fn prepare(srv: *Server, cx: *Cx, input: Input, gone: anytype) Failure!Prepared {
    const a = cx.a;
    const io = srv.io;
    const received = nowNs(io);
    const f = input.fields;
    const stops_opt = try fields_mod.stopOptions(cx, f);
    var limit: i64 = @max(1, if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens);
    const priority = f.get("priority");
    const background = prompt_mod.isTitle(input.messages, input.tools.len > 0) or
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background"));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer release(srv, preparing);
    if (gone.check()) return error.Cancelled;
    var thinking = flag(f, "enable_thinking") orelse srv.config.enable_thinking;
    if (input.prompt != null) thinking = false;
    const effort = srv.effortFor(if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null);
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    // the fold asked for (the request's thinking_fold, else the server's): with it the context holds the prompt, the
    // thinking since the last fold and the answer, not the whole reply, so max_tokens is checked against that below
    // the server's thinking defaults as they are now (POST /v1/thinking changes them while the engine runs), read once
    const tn = srv.thinkingNow();
    const fold_req: api.Fold = if (!thinking) .off else if (f.get("thinking_fold")) |v| (if (v == .string) std.meta.stringToEnum(api.Fold, v.string) orelse return cx.refuse("thinking_fold must be off, recent, checkpoint or branch") else return cx.refuse("thinking_fold must be off, recent, checkpoint or branch")) else tn.fold;
    const fold_on = fold_req != .off and srv.canFold(fold_req) and srv.think_close.len > 0;
    // the budget's summary empties the context too, so a reply with one may also pass the context's length
    // 予算の要約も文脈を空けるので、それがある応答も文脈の長さを越えてよい
    const steps_field = f.get("thinking_fold_steps");
    const at_budget_on = thinking and srv.think_close.len > 0 and (if (steps_field) |v| v == .object and v.get("budget") != null else tn.fold_step != null and (flag(f, "thinking_fold_at_budget") orelse (tn.at_budget != 0)));
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.context_fitted) ", the most this server's memory budget fits" else "", n });
        if (!fold_on and !at_budget_on) {
            if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
            limit = @min(limit, room);
        }
    }
    const system_len: usize = if (input.prompt != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .shared_prefixes = shared.items,
        // a cut just before the conversation's own text: fresh sessions resume their whole harness
        .chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, rendered.ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, rendered.ids.len, srv.chunks.min_chunk),
        .tools_json = if (input.tools.len > 0) try json.stringify(a, .{ .array = @constCast(input.tools) }, .{ .ascii = false }) else "",
    };
    try srv.checkFeatures(cx, f, input.tools.len > 0, thinking, rendered.ids, input.tools, &request);
    // the thinking's share and the answer's part: the share is of max_tokens, at most what the prompt leaves
    // of the context (`room`), and the rest of that is the answer's. Without folds the thinking's budget is the share;
    // with them the thinking goes on past the context (what is folded leaves it), its budget max_tokens less the
    // answer's part, and it folds when it fills its share of the room
    // 思考の割合と答えの分: 割合は max_tokens の、ただし prompt が文脈に残す分 (`room`) までの割合で、残りが
    // 答えの分。畳み込みなしでは思考の予算はその割合。ありでは思考は文脈を越えて続き (畳んだ分は文脈から出る)、予算は
    // max_tokens から答えの分を引いたもの。畳むのは思考が room の割合を埋めた時
    const room: i64 = if (window > 0) window - n else limit;
    const split = thinkingSplit(room, limit, tn.share, fold_on, srv.config.thinking_fold_window, tn.fold_share);
    var asked_window = false;
    if (thinking) {
        // the request's budget, else --thinking-budget, else the share
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else if (tn.budget > 0) tn.budget else split.budget;
        const notice: i64 = srv.config.thinking_notice;
        // the fold: the request's thinking_fold / _window / _keep / _max, else the server's (the window: thinkingSplit)
        const fold = fold_req;
        const window_default = split.window;
        const asked = f.get("thinking_fold_window");
        asked_window = asked != null;
        const fold_window: i64 = if (asked) |v| v.int64() orelse window_default else window_default;
        const keep0: i64 = if (f.get("thinking_fold_keep")) |v| v.int64() orelse srv.config.thinking_fold_keep else srv.config.thinking_fold_keep;
        const keep = if (keep0 > 0 and (asked_window or keep0 < fold_window)) keep0 else @divTrunc(fold_window, 2);
        const fold_max: i64 = if (f.get("thinking_fold_max")) |v| v.int64() orelse tn.fold_max else tn.fold_max;
        if (fold_max < 0) return cx.refuse("thinking_fold_max must be a count from 0");
        const folding = fold != .off and srv.canFold(fold);
        // the summary at the think block's close: the request's thinking_fold_at_close (tokens), else the server's
        // percent of the thinking's limit (0: none)
        const close_default = closeAt(tn.at_close_share, if (budget > 0) budget else limit);
        const at_close: i64 = if (f.get("thinking_fold_at_close")) |v| v.int64() orelse close_default else close_default;
        if (at_close < 0) return cx.refuse("thinking_fold_at_close must be a token count from 0");
        // the server's summary at the close is a step like the budget's, unless the request brings its own texts or steps
        // サーバーの閉じる時の要約は予算と同じ手順。要求が自分の文か手順を持つときを除く
        const close_by_step = at_close > 0 and f.get("thinking_fold_texts") == null and steps_field == null and tn.close_step != null;
        const closing = at_close > 0 and !close_by_step and srv.fold_close_open.len > 0;
        // a request's own checkpoint texts, to compare wordings without a rebuild: thinking_fold_texts
        // {"label": text, "stages": [{"open": text, "resume": text}, ...], "at_close": {"open": text, "budget_open":
        // text}} (1 to 3 stages; at_close and its budget_open may be left out; the end stays the marker)
        const texts = if (folding or closing) try foldTexts(srv, cx, f.get("thinking_fold_texts")) else null;
        // a request's own steps, to compare ways of summing up without a rebuild (foldSteps)
        const steps = try foldSteps(srv, cx, steps_field);
        if (steps) |st| if (st.midway != null and fold != .branch) return cx.refuse("thinking_fold_steps: midway needs thinking_fold branch");
        // the summary at the budget's end: where it starts, and whether the server's runs when the request has no steps
        const margin: i64 = if (f.get("thinking_fold_margin")) |v| v.int64() orelse tn.fold_margin else tn.fold_margin;
        if (margin < 0 or margin > 50) return cx.refuse("thinking_fold_margin must be a percent from 0 to 50");
        const stages: i64 = if (f.get("thinking_fold_stages")) |v| v.int64() orelse srv.config.thinking_fold_stages else if (texts) |t| @intCast(t.opens.len) else srv.config.thinking_fold_stages;
        if (folding and (stages < 1 or stages > server_mod.fold_stages)) return cx.refuse("thinking_fold_stages must be 1, 2 or 3");
        // the summary's cap: the request's tokens, else the server's percent of the thinking it organises, within that
        // percent of the most a fold holds (Job reads foldCapOf), else the server's tokens
        // 要約の上限: 要求のトークン数、なければサーバーの「整理する思考の %」(1 回が持つ最大のその % まで)、なければサーバーのトークン数
        const cap_share: i64 = if (f.get("thinking_fold_cap") != null) 0 else srv.config.thinking_fold_cap_share;
        const cap_most: i64 = if (folding) fold_window else if (budget > 0) budget else limit;
        const cap: i64 = if (cap_share > 0) @min(std.math.maxInt(u32), @max(128, @divTrunc(cap_most * cap_share, 100))) else if (f.get("thinking_fold_cap")) |v| v.int64() orelse srv.config.thinking_fold_cap else srv.config.thinking_fold_cap;
        request.think_fold_cap_share = @intCast(cap_share);
        if (cap_share > 0) request.think_fold_cap = @intCast(cap);
        // 0 with steps: no cap, a rewrite is at most the thinking it rewrites (Job.stepCap)
        if (cap_share == 0 and (folding or closing or steps != null) and !(steps != null and cap == 0) and (cap < 128 or cap > 16384)) return cx.refuse("thinking_fold_cap must be from 128 to 16384");
        if (folding and asked_window and (fold_window < 1024 or keep >= fold_window)) return cx.refuse("thinking_fold_window must be at least 1024 and thinking_fold_keep less than it");
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        if (folding) {
            request.think_fold = fold;
            request.think_fold_window = @intCast(@min(@max(0, fold_window), std.math.maxInt(u32)));
            request.think_fold_max = @intCast(@min(fold_max, std.math.maxInt(u32)));
            request.think_fold_region = @intCast(@min(@max(0, split.region), std.math.maxInt(u32)));
            request.think_fold_keep = @intCast(keep);
            request.think_fold_stages = @intCast(stages);
            request.think_fold_cap = @intCast(cap);
            request.think_fold_opens = if (texts) |t| t.opens else srv.fold_opens[0..];
            request.think_fold_label = if (texts) |t| t.label else srv.fold_label;
            request.think_fold_resumes = if (texts) |t| t.resumes else srv.fold_resumes[0..];
            request.think_fold_closes = if (texts) |t| t.closes else srv.fold_closes[0..];
            if (fold == .branch) {
                request.think_fold_branch_ask = srv.fold_branch_ask;
                request.think_fold_branch_label = srv.fold_branch_label;
                request.think_fold_branch_resume = srv.fold_branch_resume;
                request.think_fold_branch_stops = srv.fold_branch_stops;
                request.think_fold_branch_open = srv.fold_branch_open;
            }
        }
        if (closing) {
            request.think_fold_at_close = @intCast(@min(at_close, std.math.maxInt(u32)));
            request.think_fold_close_open = if (texts) |t| (t.close_open orelse srv.fold_close_open) else srv.fold_close_open;
            request.think_fold_close_budget_open = if (texts) |t| (t.close_budget_open orelse srv.fold_close_budget_open) else srv.fold_close_budget_open;
            request.think_fold_cap = @intCast(cap);
            request.think_fold_label = if (texts) |t| t.label else srv.fold_label;
        }
        if (steps) |st| {
            request.think_step_midway = st.midway;
            request.think_step_budget = st.budget;
            request.think_step_close = st.close;
            request.think_unclosed = st.unclosed;
            request.think_fold_cap = @intCast(cap);
            request.think_fold_branch_stops = srv.fold_branch_stops;
            request.think_fold_branch_open = srv.fold_branch_open;
        } else {
            if (at_budget_on and budget > 0) request.think_step_budget = tn.fold_step;
            if (close_by_step) {
                request.think_step_close = tn.close_step;
                request.think_step_close.?.min = @intCast(@min(at_close, std.math.maxInt(u32)));
            }
            if (request.think_step_budget != null or request.think_step_close != null) {
                request.think_unclosed = .close;
                request.think_fold_branch_stops = srv.fold_branch_stops;
                request.think_fold_branch_open = srv.fold_branch_open;
            }
        }
        request.think_step_margin = @intCast(margin);
        request.think_answer = @intCast(@min(@max(0, split.answer), std.math.maxInt(u32)));
        if ((budget > 0 or notice > 0 or folding or closing or close_by_step or steps != null or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
            request.think_wrap = srv.think_wrap;
            request.think_breaks = srv.think_breaks;
            if (notice > 0) {
                request.think_notice_every = @intCast(@min(notice, std.math.maxInt(u32)));
                request.think_notices = try thinkNotices(srv, a, notice, @max(0, budget), limit);
            }
        }
    }
    // with folds, the context holds the prompt, the thinking at its most (the head kept, the label, the fold window,
    // the line break waited for, the checkpoint's request, cap and close) and the answer past the thinking budget:
    // that has to fit, not max_tokens (the host caps each segment at the context left, so the context never overruns).
    // A window of the thinking's share is cut to what fits; one that fits less than 1024 tokens leaves this request
    // without folds (its budget the share again, max_tokens at most the room)
    // 畳み込みありでは、文脈は prompt、思考の最大 (残す先頭、見出し、窓、待つ改行、チェックポイントの依頼と上限と終わり)、
    // 思考の予算より後の答えを持つ。収まるべきはそれで、max_tokens ではない (ホストが区間ごとに残りの文脈で切るので、文脈は
    // 越えない)。思考の割合からの窓は収まる大きさに切る。1024 未満しか収まらなければこの要求は畳まない (予算は割合に戻し、
    // max_tokens は room まで)
    if (window > 0 and request.think_fold != .off and request.think_end != null) {
        const slack = foldSlack(&request);
        const answer: i64 = if (request.think_budget > 0) @max(0, limit - request.think_budget) else 0;
        // with a percent cap, what a fold holds and its summary have to fit together
        // % の上限では、1 回が持つ分とその要約が一緒に収まること
        if (request.think_fold_cap_share > 0) {
            const share: i64 = request.think_fold_cap_share;
            const most = @divTrunc(@max(0, room - answer - @as(i64, @intCast(slack))) * 100, 100 + share);
            request.think_fold_cap = @intCast(@min(@as(i64, request.think_fold_cap), @max(128, @divTrunc(most * share, 100))));
        }
        const fits = room - answer - request.think_fold_cap - @as(i64, @intCast(slack));
        // the region (where the summaries stack) is what fits beside the answer and the summary being written
        // 要約が積み上がる領域は、答えと書いている要約の横に収まる分
        if (fits >= 1024 and request.think_fold_region > fits) request.think_fold_region = @intCast(@min(fits, std.math.maxInt(u32)));
        if (!asked_window) {
            if (fits < 1024) {
                request.think_fold = .off;
                limit = @min(limit, room);
                request.max_tokens = @intCast(@min(limit, std.math.maxInt(u32)));
                if (request.think_budget > 0 and f.get("thinking_budget") == null and tn.budget == 0) request.think_budget = @intCast(thinkingSplit(room, limit, tn.share, false, 0, 0).budget);
                // the limit moved: the server's percent for the summary at the close follows it
                if (f.get("thinking_fold_at_close") == null and tn.at_close_share > 0) {
                    const at: u32 = @intCast(@min(closeAt(tn.at_close_share, if (request.think_budget > 0) request.think_budget else limit), std.math.maxInt(u32)));
                    if (request.think_fold_at_close > 0) request.think_fold_at_close = at;
                    if (request.think_step_close) |*st| st.min = at;
                }
                if (request.think_notice_every > 0) request.think_notices = try thinkNotices(srv, cx.a, request.think_notice_every, request.think_budget, limit);
            } else if (request.think_fold_window > fits) {
                request.think_fold_window = @intCast(@min(fits, std.math.maxInt(u32)));
                if (request.think_fold_keep >= request.think_fold_window) request.think_fold_keep = request.think_fold_window / 2;
            }
        }
    }
    if (window > 0 and request.think_fold != .off and request.think_end != null) {
        const slack = foldSlack(&request);
        const peak: i64 = @as(i64, request.think_fold_window) + request.think_fold_cap + @as(i64, @intCast(slack));
        const answer: i64 = if (request.think_budget > 0) @max(0, limit - request.think_budget) else 0;
        const need = @min(limit, peak + answer);
        if (n + need > window) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens; with thinking folds the context also holds up to {d} thinking tokens (the fold window, the checkpoint and its texts) and the answer past the thinking budget ({d} tokens). Reduce the prompt to at most {d} prompt tokens, or lower thinking_fold_window, thinking_fold_cap or max_tokens.", .{ errors.context_limit, window, n, peak, answer, @max(0, window - need) });
    }
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .effort = effort, .sampled = sampling != null, .drafts = drafts, .stops = stops_opt, .preparing = preparing };
}

/// Where the summary at the think block's close starts: `share` percent of the thinking's limit, at least one token;
/// share 0: never.
pub fn closeAt(share: i64, thinking_limit: i64) i64 {
    if (share <= 0) return 0;
    return @max(1, @divTrunc(@max(0, thinking_limit) * share, 100));
}

test "closeAt: a percent of the thinking's limit, 0 for never" {
    try std.testing.expectEqual(@as(i64, 0), closeAt(0, 464486));
    try std.testing.expectEqual(@as(i64, 23224), closeAt(5, 464486));
    try std.testing.expectEqual(@as(i64, 1440), closeAt(5, 28800));
    try std.testing.expectEqual(@as(i64, 1), closeAt(1, 10));
    try std.testing.expectEqual(@as(i64, 1), closeAt(50, 0));
}

/// A reply's thinking split: `share` percent of its max_tokens `limit`, at most of the context `room` the
/// prompt leaves, is the thinking's and the rest the answer's. Without folds the budget is the thinking's part; with
/// them the thinking goes on past the context (what is folded leaves it), its budget `limit` less the answer's part,
/// and its window (where it folds) the share of the room, no later than `early` tokens or `fold_share` percent of
/// `limit` (at least 1024) when those are above 0. Share 0: no budget, the window all the room. The window is cut
/// further to what fits beside the answer and the summary (prepare).
/// 応答の思考の分け方: max_tokens `limit` の `share` %、ただし prompt が残す文脈 `room` までの割合が思考の分、
/// 残りが答えの分。畳み込みなしでは予算は思考の分。ありでは思考は文脈を越えて続き (畳んだ分は文脈から出る)、予算は
/// `limit` から答えの分を引いたもの、窓 (畳む所) は room の割合。`early` トークンか `limit` の `fold_share` % (1024 以上)
/// が 0 より大きければそれより遅くはしない。share 0: 予算なし、窓は room 全部。窓は答えと要約の横に収まるようさらに切る
/// (prepare)。
pub const Split = struct { budget: i64, answer: i64, window: i64, region: i64 };

pub fn thinkingSplit(room: i64, limit: i64, share0: i64, folds: bool, early0: i64, fold_share: i64) Split {
    const share = @min(100, @max(0, share0));
    const base = @min(limit, room);
    const answer: i64 = if (share > 0) base - @divTrunc(base * share, 100) else 0;
    const budget: i64 = if (share == 0) 0 else if (folds) limit - answer else base - answer;
    const region: i64 = if (share > 0) @divTrunc(room * share, 100) else room;
    var early: i64 = early0;
    if (fold_share > 0) {
        const by_share = @max(1024, @divTrunc(limit * @min(100, fold_share), 100));
        early = if (early > 0) @min(early, by_share) else by_share;
    }
    return .{ .budget = budget, .answer = answer, .window = if (early > 0) @min(early, region) else region, .region = region };
}

test "thinkingSplit: the share of max_tokens at most of the room; with folds the budget runs past the room" {
    const t = std.testing;
    // pi on a 1M lane: max_tokens 276,480 below the room 988,576; share 81: the same budget folded or not
    var s = thinkingSplit(988_576, 276_480, 81, true, 0, 0);
    try t.expectEqual(@as(i64, 52_532), s.answer);
    try t.expectEqual(@as(i64, 223_948), s.budget);
    try t.expectEqual(@as(i64, 800_746), s.window);
    try t.expectEqual(@as(i64, 800_746), s.region);
    try t.expectEqual(@as(i64, 223_948), thinkingSplit(988_576, 276_480, 81, false, 0, 0).budget);
    // max_tokens past the room (65,436): the answer's part is of the room; with folds the thinking takes the rest of
    // max_tokens, without them its share of the room
    s = thinkingSplit(65_436, 200_000, 50, true, 0, 0);
    try t.expectEqual(@as(i64, 32_718), s.answer);
    try t.expectEqual(@as(i64, 167_282), s.budget);
    try t.expectEqual(@as(i64, 32_718), s.window);
    try t.expectEqual(@as(i64, 32_718), thinkingSplit(65_436, 200_000, 50, false, 0, 0).budget);
    // an earlier fold: --thinking-fold-window, fold_share percent of max_tokens (at least 1024), the smaller
    try t.expectEqual(@as(i64, 71_680), thinkingSplit(988_576, 276_480, 81, true, 71_680, 0).window);
    try t.expectEqual(@as(i64, 800_746), thinkingSplit(988_576, 276_480, 81, true, 71_680, 0).region);
    try t.expectEqual(@as(i64, 27_648), thinkingSplit(988_576, 276_480, 81, true, 71_680, 10).window);
    try t.expectEqual(@as(i64, 1_024), thinkingSplit(988_576, 4_000, 81, true, 0, 1).window);
    try t.expectEqual(@as(i64, 32_718), thinkingSplit(65_436, 200_000, 50, true, 100_000, 0).window);
    // share 0: no budget, the answer no part, the window all the room
    s = thinkingSplit(10_000, 4_096, 0, true, 0, 0);
    try t.expectEqual(@as(i64, 0), s.budget);
    try t.expectEqual(@as(i64, 0), s.answer);
    try t.expectEqual(@as(i64, 10_000), s.window);
}

/// The tokens a fold may hold past its window: the head kept, the label, the line break waited for, the longest
/// request, close and resume, and (thinking_fold branch) the summary's request, label and resume line and a second cap
/// (the answer may think first).
/// 畳み込みが窓を越えて持ちうるトークン: 残す先頭、見出し、待つ改行、いちばん長い依頼・終わり・続き、(branch) 要約の依頼と
/// 見出しと続きの一行と、もう 1 つの上限 (答えが先に考えうる)。
fn foldSlack(r: *const api.Request) usize {
    return r.think_fold_head + r.think_fold_label.len + 257 + longest(r.think_fold_opens) + longest(r.think_fold_closes) + longest(r.think_fold_resumes) +
        r.think_fold_branch_ask.len + r.think_fold_branch_label.len + r.think_fold_branch_resume.len + (if (r.think_fold == .branch) r.think_fold_cap else 0) +
        (if (r.think_step_midway) |st| st.ask.len + st.label.len + st.heading.len + st.after.len else 0);
}

/// The longest of `texts`, in tokens.
fn longest(texts: []const []const u32) usize {
    var m: usize = 0;
    for (texts) |t| m = @max(m, t.len);
    return m;
}

/// The think block's notices for a reply (GLM-5.3): the k-th after k * `every` thinking tokens, while that is short
/// of the budget (no budget: of the reply's `limit`), at most 256. Each tells the model, in its own words, what it
/// used and what is left; past half the budget it asks for focus, past three quarters for the answer soon.
fn thinkNotices(srv: *Server, a: Allocator, every: i64, budget: i64, limit: i64) Allocator.Error![]const []const u32 {
    var out: std.ArrayList([]const u32) = .empty;
    const end = if (budget > 0) budget else limit;
    var used = every;
    while (used < end and out.items.len < 256) : (used += every) {
        const left = end - used;
        const line = if (budget > 0)
            try std.fmt.allocPrint(a, "(Thinking budget check: I have used about {d} of my {d} thinking tokens; about {d} are left{s})\n", .{ used, budget, left, if (left * 4 <= budget) ", so I should wrap up and move on to the answer soon." else if (left * 2 <= budget) ", so I should stay focused on what the answer needs." else "." })
        else
            try std.fmt.allocPrint(a, "(Thinking check: I have used about {d} thinking tokens; the whole reply, answer included, may use at most {d} tokens.)\n", .{ used, limit });
        // plain text: only memory can fail; anything else leaves the notices out
        const toks = srv.text.encode(a, line, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return &.{},
        };
        try out.append(a, toks);
    }
    return out.items;
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const thinking = prepared.thinking;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer release(srv, preparing);
    if (request.background) {
        while (true) {
            const now = nowNs(io);
            const waiting = now < received + 150 * std.time.ns_per_ms or (srv.preparing.load(.acquire) > 0 and now < received + 2 * std.time.ns_per_s);
            if (!waiting) break;
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
    }
    if (gone.check()) return error.Cancelled;
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    if (stops_opt.strings.len > 0) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var fold_hook: FoldHook = .{ .srv = srv };
    if (request.think_fold == .checkpoint or request.think_fold_at_close > 0) request.think_fold_end = .{ .ctx = &fold_hook, .check = FoldHook.check };
    var canon_hook: CanonHook = .{ .srv = srv };
    if (request.think_end != null) request.think_canon = .{ .ctx = &canon_hook, .run = CanonHook.run };
    request.token_vocab = .{ .ctx = &canon_hook, .bytes = CanonHook.tokenBytes, .added = CanonHook.tokenAdded };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    const submitted = nowNs(io);
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => blk: {
            srv.metrics.refused(io);
            break :blk cx.fail(.capacity, "the engine is busy; retry shortly", .{});
        },
        error.Closed => cx.fail(.other, "the scheduler is closed", .{}),
    };
    release(srv, preparing); // a background request waits only while a foreground one prepares
    preparing = false;
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .reply_id = input.id, .sink = sink, .thinking = thinking or reply_text.isChannel(srv.markers), .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .ping_ns = pingInterval(), .last_write = submitted };
    defer srv.noteRequest(prepared.prompt_len, gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds, if (box.cached) |c| @as(usize, c) else null, box.stats.decode_seconds, box.stats.queue_seconds);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    const result: Failure!Reply = blk: {
        if (sink != null and input.tools.len > 0) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        break :blk gen.finish(cx, prepared.prompt_len, received, submitted, thinking, effort, prepared.sampled, prepared.drafts);
    };
    if (result) |_| {} else |e| logEnded(input.id, e, cx.message, prepared.prompt_len, gen.collected.items.len, seconds(nowNs(io) - received));
    return result;
}

/// The ``ended`` line of a submitted reply that ends without one: its client left, or it failed.
fn logEnded(id: []const u8, e: Failure, message: []const u8, prompt: usize, tokens: usize, after: f64) void {
    const why: ?[]const u8 = switch (e) {
        error.Cancelled => null,
        error.Refused => message, // once submitted, only the engine's own failure refuses
        else => @errorName(e),
    };
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, why, prompt, tokens, after)});
}

/// The reply to ``input``; ``sink`` hears the stream (null: not streamed). ``gone`` says the client left.
pub fn run(srv: *Server, cx: *Cx, input: Input, sink: ?Sink, gone: anytype) Failure!Reply {
    if (srv.config.compact_at == null) return generate(srv, cx, try prepare(srv, cx, input, gone), sink, gone);
    return @import("compact.zig").run(srv, cx, input, sink, gone);
}

/// Give back a foreground request's preparing count: at its submit, or when it is never generated.
pub fn release(srv: *Server, preparing: bool) void {
    if (preparing) _ = srv.preparing.fetchSub(1, .acq_rel);
}

/// The text a stream has sent: Python's ``streamed = visible``, grown by its delta while it only grows.
const Shown = struct {
    text: std.ArrayList(u8) = .empty,
    chars: usize = 0,
    at: ?[*]const u8 = null, // where the last shown text lay; the decode buffer only grows, so that prefix holds
    extends: bool = false,

    /// Python's ``now[len(shown):]``; ``stable``: ``now`` lies in the append-only decode buffer.
    fn after(s: *Shown, now: []const u8, stable: bool) []const u8 {
        const sent = s.text.items;
        s.extends = (stable and s.at == now.ptr and now.len >= sent.len) or std.mem.startsWith(u8, now, sent);
        if (!s.extends) return reply_text.afterChars(now, s.chars);
        if (stable) s.at = now.ptr; // the buffer moved but kept its bytes: the next check is free again
        return now[sent.len..];
    }

    fn set(s: *Shown, a: Allocator, now: []const u8, delta: []const u8) Allocator.Error!void {
        if (s.extends) {
            try s.text.appendSlice(a, delta);
            s.chars += reply_text.charCount(delta);
        } else {
            s.text = .empty;
            try s.text.appendSlice(a, now);
            s.chars = reply_text.charCount(now);
        }
        s.at = now.ptr;
    }
};

/// The token loop and the text it streams.
const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    sink: ?Sink,
    thinking: bool,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: Shown = .{},
    streaming_done: bool = false,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    ping_ns: i96 = 0, // silence before a keep-alive (0: none)
    last_write: i96 = 0, // when the stream last carried something

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// What closes a call the model's end token left open (``tool_parse.closeCall``); nothing when a stop string, the length or ``ignore_eos`` ended the reply instead.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const all = try g.visible.extend(g.a, g.srv.text, decodeText, fresh.items);
        const text = g.stops.visible(all, true);
        if (g.visible.pending()) return; // a character still split across tokens
        var answer = text;
        if (g.thinking) {
            const split = try reply_text.splitThinking(g.a, text, false, g.srv.markers);
            const piece = g.streamed_reasoning.after(split.reasoning, true);
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, split.reasoning, piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
            }
            answer = split.answer;
        }
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false) else answer;
        const vis = reply_text.streamingVisible(shown);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, .{ .string = delta });
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.last_write = nowNs(g.srv.io);
    }

    /// A keep-alive on a stream silent for ``ping_ns`` (the prefill, a long tool-call argument held back): the
    /// connection thread writes it between its waits, so nothing else writes at the same time.
    fn keepAlive(g: *Generation) Failure!void {
        if (g.ping_ns == 0) return;
        const sink = g.sink orelse return;
        const ping = sink.ping orelse return;
        const now = nowNs(g.srv.io);
        if (now - g.last_write < g.ping_ns) return;
        ping(sink.ctx) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.last_write = now;
    }

    /// Ends the request and waits for the engine; one it ended unfinished counts as a disconnect, as the Mac scheduler counts it.
    fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    fn loop(g: *Generation, gone: anytype) Failure!void {
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                g.chunk_index += 1;
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch null;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check()) {
                g.cancel();
                return error.Cancelled;
            }
            const tokens = chunk orelse {
                try g.keepAlive();
                continue;
            };
            if (g.reason == null) try g.commit(tokens);
        }
    }

    fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, submitted: i96, thinking: bool, effort: ?[]const u8, exact: bool, drafts: bool) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done) g.drain();
        const reason: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
            .busy => return cx.fail(.capacity, "{s}", .{if (m.message.len > 0) m.message else "the engine is busy; retry shortly"}),
        };
        const finished_ns = nowNs(g.srv.io);
        const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
        const raw = try g.srv.text.decode(a, content_tokens);
        const visible = g.stops.visible(raw, false);
        // closed before the think split, so a call ending an unclosed think block is the answer
        const close = try g.closeCall(visible);
        const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
        var content: []const u8 = text;
        var reasoning: ?[]const u8 = null;
        if (g.thinking) {
            const split = try reply_text.splitThinking(a, text, true, g.srv.markers);
            const r = reply_text.pyStrip(split.reasoning);
            reasoning = if (r.len > 0) r else null;
            content = split.answer;
        } else {
            const h = reply_text.parseHarmony(text);
            content = h.content;
            reasoning = h.reasoning;
        }
        if (g.sink) |sink| {
            const thought = g.streamed_reasoning.text.items;
            if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
            const shown = if (g.calls != null) try reply_text.hideToolCalls(a, content, true) else content;
            const sent = g.streamed.text.items;
            if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) try g.emit(sink, .{ .string = shown[sent.len..] });
            if (g.calls) |*c| if (close.len > 0) {
                // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                var out: std.ArrayList(Value) = .empty;
                try c.feed(if (g.thinking) content else text, &out);
                for (out.items) |d| try g.emit(sink, d);
            };
        }
        const prefilled = m.prefilled_ns;
        const total = seconds(finished_ns - submitted);
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, total) });
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        if (m.telemetry.len > 0) if ((try json.parse(a, m.telemetry)) == .ok) try spec.put(a, "proposer", (try json.parse(a, m.telemetry)).ok);
        // the Python CUDA engine's stats under the same names (glm_moe_dsa engine.py / runner.py "tensorfold"): clients and
        // scripts written against it read prefill_s, decode_s, tokens, rounds, tokens_per_round, accept, tok_s, cached
        const n_out = g.collected.items.len;
        if (g.srv.info.cuda_fields) {
            try runtime.put(a, "prefill_s", if (m.stats.prefill_seconds) |p| .{ .float = @round(p * 1e3) / 1e3 } else .null);
            try runtime.put(a, "decode_s", .{ .float = @round(decode_s * 1e3) / 1e3 });
            try runtime.put(a, "tokens", try json.intValue(a, n_out));
            try runtime.put(a, "rounds", try json.intValue(a, s.rounds));
            try runtime.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @round(@as(f64, @floatFromInt(n_out -| 1)) / @as(f64, @floatFromInt(s.rounds)) * 1e3) / 1e3 else 0 });
            try runtime.put(a, "accept", if (s.drafted > 0) .{ .float = @round(@as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) * 1e3) / 1e3 } else .null);
            try runtime.put(a, "tok_s", .{ .float = if (decode_s > 0 and n_out > 1) @round(@as(f64, @floatFromInt(n_out - 1)) / decode_s * 100) / 100 else 0 });
            try runtime.put(a, "cached", try json.intValue(a, m.cached orelse 0));
            try runtime.put(a, "drafted", try json.intValue(a, s.drafted));
            try runtime.put(a, "accepted", try json.intValue(a, s.accepted));
        }
        const think_end: ?u32 = if (thinking) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            // TF_GLM53_BYTE_RESUME: the engine's prompt may be some ids longer than the one rendered
            // TF_GLM53_BYTE_RESUME: エンジンの prompt は描画したものより数 id 長いことがある
            .cached_tokens = @min(m.cached orelse 0, prompt_len),
            .completion_tokens = g.collected.items.len,
            .reasoning_tokens = reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .context_reasoning = if (thinking and m.context.len > 0) try contextReasoning(g, a, m.context) else null,
        };
        if (std.mem.eql(u8, reason, "length") and thinking and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted });
        return reply;
    }
};

/// A folded reply's reasoning from the reply as the model's context held it (the think block's text, split as the
/// reply's own); null when it has none.
fn contextReasoning(g: *Generation, a: Allocator, context: []const u32) Failure!?[]const u8 {
    // the text between the think block's tokens exactly as the context held it: no display parsing, no strip, for
    // the template's '<think>' + reasoning + '</think>' renders it back as it is; a reply still thinking at its end
    // has no block to send back
    const close = g.srv.text.tokenId(g.srv.markers.close) orelse return null;
    const end = std.mem.indexOfScalar(u32, context, close) orelse return null;
    const open = g.srv.text.tokenId("<think>");
    const start: usize = if (open != null and end > 0 and context[0] == open.?) 1 else 0;
    return try g.srv.text.decode(a, context[start..end]);
}

/// Rank 0's remaking of a reply's think ids (Request.think_canon): the ids decoded and the text encoded again as one,
/// the way the next turn's prompt tokenizes a client's copy; null when that changes the text (a character cut
/// between tokens) or its structural tokens (special ids and the think markers keep their count).
const CanonHook = struct {
    srv: *Server,

    fn run(ctx: *anyopaque, a: Allocator, think: []const u32) ?[]u32 {
        const h: *CanonHook = @ptrCast(@alignCast(ctx));
        const text = h.srv.text.decode(a, think) catch return null;
        if (std.mem.indexOf(u8, text, "\u{FFFD}") != null) return null;
        const out = h.srv.text.encode(a, text, false) catch return null;
        const back = h.srv.text.decode(a, out) catch return null;
        if (!std.mem.eql(u8, back, text)) return null;
        if (h.structural(think) != h.structural(out)) return null;
        return out;
    }

    fn tokenBytes(ctx: *anyopaque, a: Allocator, id: u32) ?[]const u8 {
        const h: *CanonHook = @ptrCast(@alignCast(ctx));
        return h.srv.text.tokenBytes(a, id) catch null;
    }

    fn tokenAdded(ctx: *anyopaque, id: u32) bool {
        const h: *CanonHook = @ptrCast(@alignCast(ctx));
        return h.srv.text.isAdded(id);
    }

    fn structural(h: *const CanonHook, toks: []const u32) usize {
        const open = h.srv.text.tokenId("<think>");
        const close = h.srv.text.tokenId(h.srv.markers.close);
        var n: usize = 0;
        for (toks) |t| n += @intFromBool(h.srv.text.isSpecial(t) or (open != null and t == open.?) or (close != null and t == close.?));
        return n;
    }
};

/// The request's stop strings as the engine checks them after each token: the newest tokens' text holds one.
const StopHook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    fn check(ctx: *anyopaque, emitted: []const u32) bool {
        const h: *StopHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const tail = emitted[emitted.len -| h.stops.tail()..];
        const text = h.srv.text.decode(arena.allocator(), tail) catch return false;
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};

/// A request's thinking_fold_texts tokenized (null: none given): the stages' open, resume and close texts and the
/// label. The close is the marker's line before the stage's resume, as the server's own.
const FoldTexts = struct { opens: []const []const u32, resumes: []const []const u32, closes: []const []const u32, label: []const u32, close_open: ?[]const u32 = null, close_budget_open: ?[]const u32 = null };

fn foldTexts(srv: *Server, cx: *Cx, field: ?Value) errors.Refused!?FoldTexts {
    const v = field orelse return null;
    const bad = "thinking_fold_texts must be {\"label\": text, \"stages\": [{\"open\": text, \"resume\": text}, ...], \"at_close\": {\"open\": text, \"budget_open\": text}} with 1 to 3 stages (at_close and its budget_open may be left out)";
    const label = v.get("label") orelse return cx.refuse(bad);
    const stages = v.get("stages") orelse return cx.refuse(bad);
    if (label != .string or stages != .array or stages.array.len < 1 or stages.array.len > server_mod.fold_stages) return cx.refuse(bad);
    const a = cx.a;
    const enc = struct {
        fn f(s: *Server, c: *Cx, text: []const u8) errors.Refused![]const u32 {
            return s.text.encode(c.a, text, false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return c.refuse("thinking_fold_texts: a text could not be tokenized"),
            };
        }
    }.f;
    const marker = try enc(srv, cx, server_mod.fold_end_marker ++ "\n");
    const opens = try a.alloc([]const u32, stages.array.len);
    const resumes = try a.alloc([]const u32, stages.array.len);
    const closes = try a.alloc([]const u32, stages.array.len);
    for (stages.array, 0..) |st, k| {
        const o = st.get("open") orelse return cx.refuse(bad);
        const r = st.get("resume") orelse return cx.refuse(bad);
        if (o != .string or r != .string or o.string.len == 0) return cx.refuse(bad);
        opens[k] = try enc(srv, cx, server_mod.inserted(o.string));
        resumes[k] = try enc(srv, cx, server_mod.inserted(r.string));
        closes[k] = try std.mem.concat(a, u32, &.{ marker, resumes[k] });
    }
    var close_open: ?[]const u32 = null;
    var close_budget_open: ?[]const u32 = null;
    if (v.get("at_close")) |c| {
        const o = c.get("open") orelse return cx.refuse(bad);
        if (o != .string or o.string.len == 0) return cx.refuse(bad);
        close_open = try enc(srv, cx, server_mod.inserted(o.string));
        if (c.get("budget_open")) |b| {
            if (b != .string or b.string.len == 0) return cx.refuse(bad);
            close_budget_open = try enc(srv, cx, server_mod.inserted(b.string));
        }
    }
    return .{ .opens = opens, .resumes = resumes, .closes = closes, .label = try enc(srv, cx, server_mod.inserted(label.string)), .close_open = close_open, .close_budget_open = close_budget_open };
}

/// A request's thinking_fold_steps tokenized (null: none given): {"midway": step, "budget": step, "close": step,
/// "unclosed": "keep" | "close"}, any of them, where a step is {"ask": text, "form": "turn" | "inline", "think":
/// "closed" | "open" | "none", "heading": text, "label": text, "resume": text, "take": "after" | "before", "place":
/// "replace" | "add" | "show", "min": tokens} and only "ask" is needed. "turn" asks in a user turn after the block
/// closed, the answer begun as "think" says ("closed": <think></think>, "open": <think>) and at "heading"; "inline"
/// writes "ask" in the open block in the model's voice, with "heading" after it.
const FoldSteps = struct { midway: ?api.ThinkStep = null, budget: ?api.ThinkStep = null, close: ?api.ThinkStep = null, unclosed: api.Unclosed = .keep };

const fold_steps_bad = "thinking_fold_steps must be {\"midway\": step, \"budget\": step, \"close\": step, \"unclosed\": \"keep\" | \"close\"} with step {\"ask\": text, \"form\": \"turn\" | \"inline\", \"think\": \"closed\" | \"open\" | \"none\", \"heading\": text, \"label\": text, \"resume\": text, \"take\": \"after\" | \"before\", \"place\": \"replace\" | \"add\" | \"show\", \"min\": tokens} (only ask is needed)";

fn foldSteps(srv: *Server, cx: *Cx, field: ?Value) errors.Refused!?FoldSteps {
    const v = field orelse return null;
    if (v != .object) return cx.refuse(fold_steps_bad);
    if (srv.fold_branch_ask.len == 0 or srv.think_close.len == 0) return cx.refuse("thinking_fold_steps: this model has no think block to rewrite");
    var out: FoldSteps = .{};
    if (v.get("midway")) |s| out.midway = try foldStep(srv, cx, s);
    // shown only, a summary midway would leave the thinking to fold again at once
    if (out.midway) |m| if (m.place == .show) return cx.refuse("thinking_fold_steps: midway cannot be place show");
    if (v.get("budget")) |s| out.budget = try foldStep(srv, cx, s);
    if (v.get("close")) |s| out.close = try foldStep(srv, cx, s);
    if (v.get("unclosed")) |u| {
        if (u != .string) return cx.refuse(fold_steps_bad);
        out.unclosed = std.meta.stringToEnum(api.Unclosed, u.string) orelse return cx.refuse(fold_steps_bad);
    }
    return out;
}

fn foldStep(srv: *Server, cx: *Cx, v: Value) errors.Refused!api.ThinkStep {
    if (v != .object) return cx.refuse(fold_steps_bad);
    const a = cx.a;
    const Texts = struct {
        fn of(c: *Cx, o: Value, key: []const u8) errors.Refused![]const u8 {
            const t = o.get(key) orelse return "";
            if (t != .string) return c.refuse(fold_steps_bad);
            return t.string;
        }
        fn enc(s: *Server, c: *Cx, text: []const u8) errors.Refused![]const u32 {
            if (text.len == 0) return &.{};
            return s.text.encode(c.a, text, false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return c.refuse("thinking_fold_steps: a text could not be tokenized"),
            };
        }
    };
    const ask = try Texts.of(cx, v, "ask");
    if (ask.len == 0) return cx.refuse(fold_steps_bad);
    const form = try Texts.of(cx, v, "form");
    const in_line = std.mem.eql(u8, form, "inline");
    if (!in_line and form.len > 0 and !std.mem.eql(u8, form, "turn")) return cx.refuse(fold_steps_bad);
    const think = try Texts.of(cx, v, "think");
    const heading = try Texts.enc(srv, cx, try Texts.of(cx, v, "heading"));
    var asked: []const u32 = undefined;
    if (in_line) {
        asked = try std.mem.concat(a, u32, &.{ try Texts.enc(srv, cx, server_mod.inserted(ask)), heading });
    } else {
        const end = srv.think_close_end orelse return cx.refuse("thinking_fold_steps: this model has no think block to rewrite");
        const open = srv.fold_branch_open orelse return cx.refuse("thinking_fold_steps: this model has no think block to rewrite");
        const user = srv.text.tokenId("<|user|>") orelse return cx.refuse("thinking_fold_steps: this model has no turns to ask in");
        const assistant = srv.text.tokenId("<|assistant|>") orelse return cx.refuse("thinking_fold_steps: this model has no turns to ask in");
        const start: []const u32 = if (think.len == 0 or std.mem.eql(u8, think, "closed"))
            try a.dupe(u32, &.{ open, end })
        else if (std.mem.eql(u8, think, "open"))
            try a.dupe(u32, &.{open})
        else if (std.mem.eql(u8, think, "none"))
            &.{}
        else
            return cx.refuse(fold_steps_bad);
        asked = try std.mem.concat(a, u32, &.{ &.{ end, user }, try Texts.enc(srv, cx, ask), &.{assistant}, start, heading });
    }
    var st: api.ThinkStep = .{
        .ask = asked,
        .heading = heading,
        .label = try Texts.enc(srv, cx, server_mod.inserted(try Texts.of(cx, v, "label"))),
        .after = try Texts.enc(srv, cx, server_mod.inserted(try Texts.of(cx, v, "resume"))),
    };
    if (v.get("take")) |t| {
        if (t != .string) return cx.refuse(fold_steps_bad);
        st.take = std.meta.stringToEnum(api.ThinkStep.Take, t.string) orelse return cx.refuse(fold_steps_bad);
    }
    if (v.get("place")) |t| {
        if (t != .string) return cx.refuse(fold_steps_bad);
        st.place = std.meta.stringToEnum(api.ThinkStep.Place, t.string) orelse return cx.refuse(fold_steps_bad);
    }
    if (v.get("min")) |t| {
        const n = t.int64() orelse return cx.refuse(fold_steps_bad);
        if (n < 0) return cx.refuse(fold_steps_bad);
        st.min = @intCast(@min(n, std.math.maxInt(u32)));
    }
    return st;
}

/// The checkpoint fold's end (GLM-5.3): whether the checkpoint written so far ends with the marker (its last tokens
/// decoded; the engine thread calls it after each token of the checkpoint).
const FoldHook = struct {
    srv: *Server,

    fn check(ctx: *anyopaque, written: []const u32) bool {
        const h: *FoldHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const text = h.srv.text.decode(arena.allocator(), written[written.len -| 24..]) catch return false;
        return std.mem.indexOf(u8, text, server_mod.fold_end_marker) != null;
    }
};

fn decodeText(t: anytype, a: Allocator, tokens: []const u32) ![]u8 {
    return t.decode(a, tokens);
}

/// ``{key: text}``: a streamed delta (role, content or reasoning_content).
pub fn deltaOf(a: Allocator, key: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = text });
    return .{ .object = o };
}

/// The reply's token ids hashed: drafted and ``"draft": false`` replies must match.
pub fn tokenSha(tokens: []const u32) [12]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [16]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    var out: [12]u8 = undefined;
    const hex = std.fmt.bytesToHex(d[0..6].*, .lower);
    @memcpy(&out, &hex);
    return out;
}
