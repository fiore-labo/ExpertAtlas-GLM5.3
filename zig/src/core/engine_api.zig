// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! The engine a server drives (``Engine``), and ``LaneHost``: the lane core served on one thread, rounds left to it.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// The server's name for one request, unique while the process lives.
pub const Id = u64;

/// One request's sampling (Python's exact_sampling.Sampling); a request without one decodes greedily.
pub const Sampling = lanes.Sampling;

/// A prompt's reproducible seed (``seed_for``), for requests that name none.
pub const seedFor = lanes.sampling.seedFor;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar },
    /// The schema (JSON text), regex, choices (a JSON array) or EBNF; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
};

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    max_tokens: u32,
    sampling: ?Sampling = null,
    /// Tokens that end the reply as ``stop`` (the token is delivered); empty when the request ignores EOS.
    eos: []const u32 = &.{},
    stop: ?Stop = null,
    /// false: one token a round and no drafts, the serial reference drafted replies must equal.
    drafts: bool = true,
    /// Background work yields to foreground arrivals.
    background: bool = false,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then shared system blocks.
    history_len: u32 = 0,
    shared_prefixes: []const u32 = &.{},
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at ``Info.prefill_step``); empty: the engine's own.
    chunks: []const u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
};

pub const Reason = enum { stop, length, cancelled, failed };

/// What a finished reply's rounds did, for its runtime fields and /metrics.
pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// Decode duration measured by the engine host (the first token to the last), in seconds; null: not measured.
    decode_seconds: ?f64 = null,
    /// Prompt tokens the prefill computed (a prompt cache's resumed ones excluded); null: not reported.
    computed: ?u64 = null,
    /// Seconds the request waited in the engine's queue before it ran; null: not measured.
    queue_seconds: ?f64 = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

pub const Event = union(enum) {
    /// The prompt is in the cache: ``cached`` of its tokens came from a kept prefix.
    prefilled: u32,
    /// Tokens committed for this request, in order; valid only during the call.
    tokens: []const u32,
    /// The reply ended; nothing follows. ``message`` says why a failed reply failed.
    finished: struct { reason: Reason, stats: Stats = .{}, message: []const u8 = "" },
};

/// Where a request's events go. Called on the engine's thread: it must return at once and not re-enter the engine.
pub const Sink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: Id, event: *const Event) void,
};

/// What the engine serves, fixed once it has loaded.
pub const Info = struct {
    name: []const u8 = "lanes",
    lanes: u32 = 1,
    /// Prompt plus reply tokens one request may use (0: no limit).
    context_window: u32 = 0,
    context_fitted: bool = false,
    /// What the engine enforces; the server refuses requests that need more.
    call_gates: bool = false,
    /// The engine observes loop_guard requests in its lane stream.
    loop_guard: bool = false,
    structures: bool = false,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine cuts prompts itself).
    prefill_step: u32 = 0,
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    startup: []const u8 = "",
    /// Positions kept for earlier conversations beside the running one (a slot cache's store), shown on /health;
    /// 0: the store could not be allocated; null: the engine keeps no such store.
    slot_positions: ?u64 = null,
    /// The engine's settings as Prometheus labels for /metrics' engine_info (`tp="8",dcp="2"`); empty: no such family.
    labels: []const u8 = "",
    /// The Python CUDA server's fields too (cuda/health.py's /health totals and /v1/health, the engine stats names in
    /// a reply's "tensorfold" object), for clients and monitors written against it; off: the Mac server's shapes only.
    cuda_fields: bool = false,
};

/// A checkpoint family an engine reads: its config ``model_type`` and weight formats, as gate entries name them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an engine reads.
pub const Open = struct {
    dir: []const u8,
    model_type: []const u8,
    context: ?i64 = null,
    lanes: u32 = 8,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses instead of serving fewer.
    lanes_fixed: bool = false,
    drafts: bool = true,
    speed_up: ?[]const u8 = null,
    prompt_cache_gib: ?f64 = null,
    prompt_cache_over_cap: bool = false,
    /// --learn: where shared prompt states are kept on disk for later sessions and servers (null: off).
    learn: ?[]const u8 = null,
    /// --learn-gib: what learned states may take on disk, every model and build together.
    learn_gib: f64 = 32,
    /// --device and --segments (CUDA); null: the backend's environment fallback, then its default.
    device: ?u32 = null,
    segments: ?u32 = null,
};

/// An opened engine; ``close`` stops its thread and frees its backend.
pub const Opened = struct { engine: Engine, close: *const fn (ctx: *anyopaque) void, ctx: *anyopaque };

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0 };

/// Fiore-lab: GET /v1/experts/usage?window=SECONDS, GET /v1/experts/recent?after=SEQ&max=N, GET /v1/memory,
/// GET /v1/experts/swap (the last change of the experts' bits and its progress), or GET /v1/experts/prefill?after=SEQ&max=N
/// (the prompt chunks' pick counts).
pub const ExpertsQuery = union(enum) { usage: u32, recent: struct { after: ?u64, max: u32, text: ?TokenBytes = null }, memory: void, swap: void, prefill: struct { after: ?u64, max: u32 } };

/// Fiore-lab (the control panel): a token's raw bytes for the pick log's text (`records[].text`), from the server's
/// tokenizer: null for a special token, else the bytes (allocated in `a`; byte-level BPE: the vocabulary string mapped
/// back through the byte-level byte alphabet, so a token may end inside a UTF-8 character).
/// Fiore-lab (コントロールパネル): 選択ログの文字 (`records[].text`) のための token の生のバイト。サーバーのトークナイザー
/// から: 特殊 token は null、それ以外はバイト (`a` に確保。byte-level BPE: 語彙の文字列をバイトの表で戻す
/// ので、token が UTF-8 の文字の途中で終わることがある)。
pub const TokenBytes = struct {
    ctx: *anyopaque,
    bytes: *const fn (ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8,
};

/// Fiore-lab: POST /v1/experts/swap (the bits wanted: one byte an expert, the MoE layers in order; run between
/// requests) or POST /v1/experts/save (a name for the bits in use).
pub const ExpertsControl = union(enum) { swap: []const u8, save: []const u8 };
pub const ControlError = error{ Unsupported, UnsupportedWithMulti, Busy, BadRequest, Exists, Failed, OutOfMemory };

/// A backend's words for its own refusals (a request it cannot serve); null: the error's name.
pub const Explain = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, err: anyerror) ?[]const u8,
};

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

pub const Status = struct {
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// A KV pool's state (the GLM-5.3 engine's TF_GLM53_PREFIX=pool), shown on /health; null: no pool.
    pool: ?PoolStatus = null,
    /// The engine's own totals with the running request's progress in them, and that request's phase (the GLM-5.3
    /// host); null: not reported, and /metrics and /health leave these out.
    progress: ?Progress = null,
};

pub const PoolStatus = struct {
    blocks: u64 = 0,
    free_blocks: u64 = 0,
    slots: u64 = 0,
    shared_blocks: u64 = 0,
    evicted_total: u64 = 0,
    cold_total: u64 = 0,
    /// Positions one block holds.
    block_positions: u64 = 0,
    /// Requests by what the pool did with them: resumed their own conversation, shared another's start, or began a
    /// new one (a cold request counts as new, and in cold_total).
    continued_total: u64 = 0,
    shared_total: u64 = 0,
    new_total: u64 = 0,
    /// Seconds the pool spent checking the rows a request resumed, and hashing the rows it wrote.
    check_seconds_total: f64 = 0,
    hash_seconds_total: f64 = 0,
};

/// Totals since the engine started, the running request's progress included, read under one lock so that each
/// only grows; and the running request's own position.
pub const Progress = struct {
    phase: enum { idle, prefill, decode } = .idle,
    /// The running request: its prompt tokens, those a prompt cache resumed, those in the caches so far (resumed
    /// ones included), and the reply tokens so far.
    prompt: u64 = 0,
    cached: u64 = 0,
    prompt_done: u64 = 0,
    generated: u64 = 0,
    /// Prompt tokens of the requests waiting in the queue.
    queued_prompt: u64 = 0,
    /// Every request so far, the running one's progress included: prompt tokens (counted as its prefill starts),
    /// those resumed, those computed, reply tokens, prefill and decode seconds, decode rounds, drafts verified and
    /// drafts kept.
    prompt_total: u64 = 0,
    cached_total: u64 = 0,
    computed_total: u64 = 0,
    generated_total: u64 = 0,
    prefill_seconds_total: f64 = 0,
    decode_seconds_total: f64 = 0,
    rounds_total: u64 = 0,
    drafted_total: u64 = 0,
    accepted_total: u64 = 0,
    requests_total: u64 = 0,
};

pub const SubmitError = error{ Closed, Busy };

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        info: *const fn (ctx: *anyopaque) Info,
        submit: *const fn (ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void,
        /// Ends a request between rounds; its ``finished`` (cancelled) still arrives. Unknown ids are ignored.
        cancel: *const fn (ctx: *anyopaque, id: Id) void,
        status: *const fn (ctx: *anyopaque, out: *Status, stream_tokens: []u32) void,
        /// Device memory in bytes, the peak since the last reset; null when the backend keeps no count.
        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// The engine's queue as a keepalive target, or null when the backend is not Metal.
        keepalive: ?*const fn (ctx: *anyopaque) ?keepalive.Target = null,
        /// One logit per label, and the vocabulary logsumexp. Null until a family scores decisions.
        score: ?*const fn (ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64) error{Failed}!f64 = null,
        /// Fiore-lab: the live expert usage as a JSON body (GET /v1/experts/...), null when the backend keeps none.
        /// Read from HTTP threads, so it must not touch the GPU.
        experts: ?*const fn (ctx: *anyopaque, a: std.mem.Allocator, q: ExpertsQuery) ?[]u8 = null,
        /// Fiore-lab: a change of the experts' bits asked for, or the bits in use saved; a JSON body.
        experts_control: ?*const fn (ctx: *anyopaque, a: std.mem.Allocator, c: ExpertsControl) ControlError![]u8 = null,
    };

    pub fn expertsControl(e: Engine, a: std.mem.Allocator, c: ExpertsControl) ControlError![]u8 {
        const f = e.vtable.experts_control orelse return error.Unsupported;
        return f(e.ctx, a, c);
    }

    pub fn experts(e: Engine, a: std.mem.Allocator, q: ExpertsQuery) ?[]u8 {
        const f = e.vtable.experts orelse return null;
        return f(e.ctx, a, q);
    }

    pub fn info(e: Engine) Info {
        return e.vtable.info(e.ctx);
    }
    pub fn submit(e: Engine, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    pub fn cancel(e: Engine, id: Id) void {
        e.vtable.cancel(e.ctx, id);
    }
    pub fn status(e: Engine, out: *Status, stream_tokens: []u32) void {
        e.vtable.status(e.ctx, out, stream_tokens);
    }
    pub fn memory(e: Engine, reset_peak: bool) ?Memory {
        return e.vtable.memory(e.ctx, reset_peak);
    }
    /// The engine's queue as a keepalive target, or null when the backend is not Metal.
    pub fn keepaliveTarget(e: Engine) ?keepalive.Target {
        const f = e.vtable.keepalive orelse return null;
        return f(e.ctx);
    }

    /// The decision hook, or Unsupported when this engine does not score.
    pub fn score(e: Engine, prompt: []const u32, labels: []const u32, logits: []f64) error{ Failed, Unsupported }!f64 {
        const f = e.vtable.score orelse return error.Unsupported;
        return f(e.ctx, prompt, labels, logits);
    }
};

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    /// Prefills `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core (true).
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
};

/// The lane core served to the HTTP threads (lane_host.zig).
pub const LaneHost = @import("lane_host.zig").LaneHost;

/// Exact prompt reuse between requests, for any family (prompt_cache.zig).
pub const prompt_cache = @import("prompt_cache.zig");

/// Learned prompt-cache states on disk (prompt_imprint.zig).
pub const prompt_imprint = @import("prompt_imprint.zig");

/// The idle keepalive's ticker and target contract.
pub const keepalive = @import("keepalive.zig");

test {
    _ = @import("lane_host.zig");
    _ = @import("lane_host_reuse_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
}
