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

/// A reply's think ids as the server's tokenizer cuts their text (decoded, then encoded as one text), into `a`; null
/// when that would change the text or its structural tokens. The engine thread calls it when it edits the context.
/// Fiore-lab: a token's raw bytes (before the decoder's lossy UTF-8; null: unknown) and whether it is an added token.
pub const TokenVocab = struct {
    ctx: *anyopaque,
    bytes: *const fn (ctx: *anyopaque, a: Allocator, id: u32) ?[]const u8,
    added: *const fn (ctx: *anyopaque, id: u32) bool,
};

pub const Canon = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, a: Allocator, ids: []const u32) ?[]u32,
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

/// GLM-5.3: how a long think block is folded (Request.think_fold).
pub const Fold = enum { off, recent, checkpoint, branch };

/// GLM-5.3: one moment the think block is rewritten from what the model writes beside the reply (thinking_fold_steps,
/// to try ways of summing up without a rebuild): the segment runs from the context and ``ask``; what comes back goes
/// into the block as ``place`` says, between ``label`` + ``heading`` and ``after``.
pub const ThinkStep = struct {
    /// The tokens after the block in the branch's segment (their last are ``heading``, which the model goes on from).
    ask: []const u32 = &.{},
    heading: []const u32 = &.{},
    label: []const u32 = &.{},
    after: []const u32 = &.{},
    take: Take = .after,
    place: Place = .replace,
    /// At the block's end: the think tokens since the last rewrite the step needs (fewer: it does not run).
    min: u32 = 1024,

    /// Which part is the result: what follows the last ``think_end`` (what precedes it if that is too short), or what
    /// precedes the first (the branch ends there; all of it if none comes).
    pub const Take = enum { after, before };
    /// What the result does: takes the place of the thinking since the last rewrite, follows it, or is only shown.
    pub const Place = enum { replace, add, show };
};

/// GLM-5.3: a reply that ends with its think block open (its answer written there): ``keep`` leaves the content empty,
/// ``close`` closes the block and lets the model answer.
pub const Unclosed = enum { keep, close };

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
    /// GLM-5.3: after every ``think_notice_every`` tokens of the think block (0: none), at the next line break,
    /// ``think_notices[k - 1]`` (the k-th) goes into the reply as the model's own text, telling it how much it used.
    think_notice_every: u32 = 0,
    think_notices: []const []const u32 = &.{},
    /// GLM-5.3: what the budget's cut writes, at the next line break (a closing sentence, then the think block's
    /// end); empty: ``think_close``.
    think_wrap: []const u32 = &.{},
    /// GLM-5.3: the tokens holding a line break, sorted: a notice or the cut waits for one (at most 256 tokens).
    think_breaks: []const u32 = &.{},
    /// GLM-5.3: the think block folded once ``think_fold_window`` of its tokens are in the context past the last fold,
    /// at a line break; the reply the client gets is unchanged, the context the model goes on from is shorter.
    /// ``recent``: the block's first ``think_fold_head`` tokens and its last ``think_fold_keep``. ``checkpoint``: the
    /// model first writes its conclusions (``think_fold_opens`` asks, ``think_fold_end`` sees the end marker, past
    /// ``think_fold_cap`` tokens ``think_fold_closes`` ends it); then the head, ``think_fold_label`` and the checkpoint,
    /// and ``think_fold_resumes`` goes on. The texts come by stage (``think_fold_stages`` of them, 1 to 3: organise,
    /// converge, commit), picked by the folds so far and the thinking budget left. ``branch`` (Fiore-lab):
    /// the summary is written beside the reply, not in it: the context so far, ``think_fold_opens[0]`` (the block
    /// closed, a user turn asking for the summary, an answer begun without thinking) and the model's answer up to its
    /// EOS (or ``think_fold_cap`` tokens) is the summary; the block then keeps its head, the summaries before, then
    /// ``think_fold_label``, this one and ``think_fold_resumes[0]``, and goes on.
    think_fold: Fold = .off,
    think_fold_window: u32 = 0,
    /// GLM-5.3: the folds a reply may make at most (0: no limit); past them, or when the summaries kept and the next
    /// one would pass ``think_fold_region``, the block folds no more and closes as at the end of ``think_budget`` when
    /// it fills the region.
    think_fold_max: u32 = 0,
    /// GLM-5.3: the thinking's room in the context, the share of what the prompt leaves (0: none). The
    /// block folds once ``think_fold_window`` tokens came since the last fold or all its tokens in the context reach
    /// this, whichever is first; the summaries stack in it.
    think_fold_region: u32 = 0,
    think_fold_keep: u32 = 0,
    think_fold_head: u32 = 100,
    think_fold_cap: u32 = 1024,
    /// Above 0: a summary's tokens at most are this percent of the thinking it organises (at least 128), within
    /// ``think_fold_cap`` (foldCapOf).
    think_fold_cap_share: u8 = 0,
    think_fold_stages: u8 = 1,
    think_fold_opens: []const []const u32 = &.{},
    think_fold_label: []const u32 = &.{},
    think_fold_resumes: []const []const u32 = &.{},
    think_fold_closes: []const []const u32 = &.{},
    think_fold_end: ?Stop = null,
    /// GLM-5.3, ``think_fold`` branch: the request after the block so far, the label the summary is kept under in the
    /// block, and the line the thinking goes on after.
    think_fold_branch_ask: []const u32 = &.{},
    think_fold_branch_label: []const u32 = &.{},
    think_fold_branch_resume: []const u32 = &.{},
    /// GLM-5.3, ``think_fold`` branch: the turn and tool tokens that end the summary as its EOS does (such a token
    /// stays out of the summary), and the think block's opening: the answer may think first (``<think>`` dropped,
    /// what precedes a ``think_end`` its thinking), and the summary is then what follows the last close.
    think_fold_branch_stops: []const u32 = &.{},
    think_fold_branch_open: ?u32 = null,
    /// GLM-5.3, ``thinking_fold_steps``: the midway step in place of the summary beside the reply (``think_fold`` branch);
    /// the budget's step in place of ``think_wrap`` (it starts so that the block closes within the budget); the step at
    /// the model's own close. The last two close the block after them. With them ``think_fold_cap`` 0 means no cap: a
    /// result is at most the thinking it rewrites, and nothing is set aside of the budget ahead.
    think_step_midway: ?ThinkStep = null,
    think_step_budget: ?ThinkStep = null,
    think_step_close: ?ThinkStep = null,
    think_unclosed: Unclosed = .keep,
    /// The budget's step starts this percent before the budget's end. Past it the model goes on until its thinking in
    /// the context fills ``think_budget`` again, ``think_answer`` tokens of the context and the reply staying the answer's.
    think_step_margin: u32 = 5,
    think_answer: u32 = 0,
    /// GLM-5.3: the summary at the think block's close. When the model closes the block with at least
    /// ``think_fold_at_close`` of its tokens in the context past the last fold (0: never), the close waits: the model
    /// first writes a checkpoint (``think_fold_close_open`` asks; the end marker, the model's own close or
    /// ``think_fold_cap`` ends it), the block folds to its head, ``think_fold_label`` and that checkpoint and then
    /// closes, so the answer and the next turns go on from the summary. Works with ``think_fold`` off too. When the
    /// thinking budget's cut closes the block instead, ``think_fold_close_budget_open`` asks for the same summary in
    /// place of ``think_wrap`` (empty: the cut writes ``think_wrap`` and closes at once). Either is asked for only when
    /// it, a summary of ``think_fold_cap`` tokens and an answer as long still fit in ``max_tokens``.
    think_fold_at_close: u32 = 0,
    think_fold_close_open: []const u32 = &.{},
    think_fold_close_budget_open: []const u32 = &.{},
    /// Remakes the context's think ids where the engine edited it (folds, inserted texts, the close), so a client's
    /// copy of the text renders to them next turn; null: the ids stay as they were made.
    think_canon: ?Canon = null,
    /// Reads tokens as bytes, to match a prompt with a kept slot past the first id that differs (TF_GLM53_BYTE_RESUME).
    token_vocab: ?TokenVocab = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
};

/// `busy`: taken but refused for want of room (the engine's queue gave up on it): the client gets 503 / overloaded and
/// may retry.
pub const Reason = enum { stop, length, cancelled, failed, busy };

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
    /// The reply as the model's context held it at the end, when thinking folds made it differ from the tokens sent
    /// (empty: the same); valid only during the call.
    context_reply: []const u32 = &.{},
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
/// GET /v1/experts/swap (the last change of the experts' bits and its progress), GET /v1/experts/prefill?after=SEQ&max=N
/// (the prompt chunks' pick counts), or GET /v1/pool (the KV pool's conversations and each block's owner).
pub const ExpertsQuery = union(enum) { usage: u32, recent: struct { after: ?u64, max: u32, text: ?TokenBytes = null }, memory: void, swap: void, prefill: struct { after: ?u64, max: u32 }, pool: void };

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

/// Fiore-lab: how TF_GLM53_MULTI shares the GPU's time between a prompt being computed and the live conversations'
/// decode while both want it (GET / POST /v1/schedule, changed while the engine runs; the family's pace.zig).
/// `decode_share`: the decode's percent of that time, 0 (the prompts first: no round inside a prefill, the queue
/// admitted back to back) .. 100 (the decode first: a prompt waits while a conversation decodes). `measured`: the
/// decode's percent over the last seconds both had work (kept after), null before they ever did; `contended`: both
/// have work now. `short_prompt`: a prompt with at most this many rows to compute (past what the cache reuses) goes
/// before the longer ones waiting and is not held by the decode share (0: none is short). `short_preempt`: a longer
/// prefill running gives way to it too (its chunks done kept, resumed afterwards).
/// `start`: the values the engine started with (MULTI_DECODE_SHARE, MULTI_SHORT_PROMPT, MULTI_SHORT_PREEMPT).
pub const Schedule = struct { decode_share: u8 = 30, short_prompt: u32 = 8192, short_preempt: bool = false, measured: ?u8 = null, contended: bool = false, start: ?ScheduleStart = null };

pub const ScheduleStart = struct { decode_share: u8 = 30, short_prompt: u32 = 8192, short_preempt: bool = false };

/// Fiore-lab: where the RoCE proxy threads poll (TF_GLM53_ROCE_PROXY_CORES; /health "roce"). `world` 0: no RoCE.
/// `proxies[rank]`: the world's proxy, then the DCP group's; `present` false: none; `cpu`: the one it polls on (`live`:
/// read now, rank 0's; else where it was kept, null when the scheduler's); `core`: that CPU's kind.
pub const Roce = struct {
    mode: RoceMode = .os,
    world: u8 = 0,
    proxies: [roce_ranks][2]RoceProxy = @splat(@splat(.{})),
};
pub const RoceMode = enum { a725, x925, os };
pub const CoreKind = enum { unknown, a725, x925 };
pub const RoceProxy = struct { present: bool = false, cpu: ?u16 = null, core: CoreKind = .unknown, live: bool = false };
pub const roce_ranks = 8;

/// A change of the schedule (POST /v1/schedule): the fields given are set, the others kept.
pub const ScheduleSet = struct { decode_share: ?u8 = null, short_prompt: ?u32 = null, short_preempt: ?bool = null };

/// The largest `short_prompt` taken (rows).
pub const short_prompt_max: u32 = 1 << 20;

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

/// A request's stage: none running, its prompt, its reply.
pub const Phase = enum { idle, prefill, decode };

/// The requests a status can list one by one (an engine decoding several together lists up to this many).
pub const max_requests = 4;

/// One request running now: the slot of the decode window its conversation has (the `conv` of the expert records),
/// its conversation's number in the prompt cache's pool (null without one, or before its plan), its stage, its
/// prompt tokens, those a prompt cache resumed, those in the caches so far (resumed ones included), its reply tokens
/// so far, and the milliseconds since it was admitted.
pub const RequestProgress = struct {
    conv: u16 = 0,
    pool_id: ?u64 = null,
    phase: Phase = .idle,
    prompt: u64 = 0,
    cached: u64 = 0,
    prompt_done: u64 = 0,
    generated: u64 = 0,
    elapsed_ms: u64 = 0,
    /// Fiore-lab: the time its prefill chunks computed so far (ms), the waits left out (the decode's rounds inside it,
    /// a hold by the GPU split, the queue after it gave way); kept after its prefill. null before its first chunk.
    prefill_ms: ?u64 = null,
};

/// Totals since the engine started, the running requests' progress included, read under one lock so that each
/// only grows; and the running requests' own positions (the fields below `phase` tell the first of them).
pub const Progress = struct {
    phase: Phase = .idle,
    /// The running request: its prompt tokens, those a prompt cache resumed, those in the caches so far (resumed
    /// ones included), and the reply tokens so far.
    prompt: u64 = 0,
    cached: u64 = 0,
    prompt_done: u64 = 0,
    generated: u64 = 0,
    /// Prompt tokens of the requests waiting in the queue.
    queued_prompt: u64 = 0,
    /// The requests waiting in the queue (not admitted yet: under TF_GLM53_MULTI also those waiting for room in the
    /// KV pool) and how long the oldest of them has waited (ms).
    waiting: u64 = 0,
    waiting_ms: u64 = 0,
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
    /// Requests refused as busy: no room in the KV pool within TF_GLM53_MULTI_WAIT_S (503 / overloaded).
    busy_total: u64 = 0,
    /// TF_GLM53_MULTI: the conversations decoded together and the seconds a request may wait for room (0: no limit);
    /// multi 0: one at a time (/health then leaves `multi` out).
    multi: u8 = 0,
    multi_wait_s: u64 = 0,
    /// TF_GLM53_MULTI: the schedule in use now (GET / POST /v1/schedule).
    multi_schedule: Schedule = .{},
    /// Fiore-lab: where the RoCE proxies poll (world 0: no RoCE).
    roce: Roce = .{},
    /// The requests running now, the oldest admitted first: every conversation admitted when several decode
    /// together, else the one running. Not the ones waiting for admission.
    requests: [max_requests]RequestProgress = @splat(.{}),
    n_requests: u8 = 0,
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
        /// Fiore-lab: TF_GLM53_MULTI's schedule, the fields of `set` given set first; the schedule in use after.
        schedule: ?*const fn (ctx: *anyopaque, set: ScheduleSet) ControlError!Schedule = null,
    };

    pub fn schedule(e: Engine, set: ScheduleSet) ControlError!Schedule {
        const f = e.vtable.schedule orelse return error.Unsupported;
        return f(e.ctx, set);
    }

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

/// A summary's tokens at most for `thinking` tokens organised: `cap`, or with `share` above 0 that percent of them (at
/// least 128) within `cap`.
pub fn foldCapOf(share: u8, cap: u32, thinking: u64) u64 {
    if (share == 0) return cap;
    return @min(@as(u64, cap), @max(128, thinking * share / 100));
}

test "foldCapOf: a percent of the thinking, at least 128, within the cap" {
    const t = @import("std").testing;
    try t.expectEqual(@as(u64, 4096), foldCapOf(0, 4096, 100_000));
    try t.expectEqual(@as(u64, 50_000), foldCapOf(50, 200_000, 100_000));
    try t.expectEqual(@as(u64, 128), foldCapOf(50, 200_000, 100));
    try t.expectEqual(@as(u64, 60_000), foldCapOf(50, 60_000, 200_000));
}
