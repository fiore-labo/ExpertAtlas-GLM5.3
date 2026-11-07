// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! The server: one engine and tokenizer behind the HTTP routes, one thread a connection.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const auth = @import("auth.zig");
const metrics = @import("metrics.zig");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const http_conn = @import("http_conn.zig");
const listener_mod = @import("listener.zig");
const routes = @import("routes.zig");
const responses = @import("responses.zig");
const grammar = @import("grammar.zig");
const schema_check = @import("schema_check.zig");
const clock = @import("clock.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const chunk_plan = @import("chunk_plan.zig");
const log = @import("log.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// auto compacts near the window. fraction compacts past that share of it.
pub const CompactAt = union(enum) { auto, fraction: f64 };

pub const Config = struct {
    served_name: []const u8,
    /// The served name first, then aliases, without repeats.
    model_ids: []const []const u8,
    default_max_tokens: i64 = 4096,
    enable_thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    /// The percent of a reply's max_tokens its think block may use when neither the request nor --thinking-budget
    /// sets a budget (0: none).
    thinking_share: i64 = 0,
    /// Tokens of the think block between notices of how much it used (0: none; GLM-5.3).
    thinking_notice: i64 = 0,
    /// How a long think block is folded (GLM-5.3): off, recent (its head and last tokens kept) or checkpoint (the
    /// model writes its conclusions, which replace the rest). It folds when the thinking fills its share of the context
    /// (thinking_share percent of what the prompt leaves); thinking_fold_window, above 0, folds no later than that many
    /// of its tokens in the context, and thinking_fold_keep is how many the recent fold keeps (0: half the window).
    thinking_fold: api.Fold = .off,
    thinking_fold_window: i64 = 0,
    /// The same as a percent of a reply's max_tokens (at least 1024 tokens) when the request names none; 0: none.
    /// Changed while the engine runs (Server.thinking).
    thinking_fold_share: i64 = 0,
    /// A reply's folds at most (0: no limit): past them, a thinking that fills its share closes as at its budget's end.
    /// Changed while the engine runs (Server.thinking).
    thinking_fold_max: i64 = 0,
    /// The summary at the budget's end when a request names no steps (0: none), and how far before that end it starts,
    /// a percent of the budget. Past it the model goes on until the thinking fills its memory again. Changed while the
    /// engine runs (Server.thinking).
    thinking_fold_at_budget: i64 = 1,
    thinking_fold_margin: i64 = 10,
    /// What the model is asked at the budget's end and at its own close (empty: the built-in texts).
    thinking_fold_prompt: []const u8 = "",
    thinking_close_prompt: []const u8 = "",
    thinking_fold_keep: i64 = 0,
    /// The checkpoint's stages (1: one text every time; 3: organise, converge, commit by the folds and budget left).
    thinking_fold_stages: i64 = 1,
    /// The summary at the think block's close (GLM-5.3): when the model closes a block with at least this many of
    /// its tokens in the context past the last fold, it first writes a checkpoint that replaces the block (0: never).
    thinking_fold_at_close_share: i64 = 0,
    /// A checkpoint's tokens at most (a fold's and the summary at the close), when the request names none (GLM-5.3).
    thinking_fold_cap: i64 = 4096,
    /// Above 0: a summary's tokens at most as a percent of the thinking it organises, in place of thinking_fold_cap.
    thinking_fold_cap_share: i64 = 0,
    loop_guard: bool = false,
    /// Seconds the idle keepalive runs after the last request ends (0: off).
    keep_warm_s: i64 = 900,
    dashboard: bool = false,
    /// Sampling when a request names none (generation_config.json and the serve flags), or null for greedy.
    default_sampling: ?Value = null,
    use_drafts: bool = true,
    seed_salt: i64 = 0,
    /// TENSORFOLD_REQUEST_LOG: where chat and completion bodies are appended; null: nowhere.
    request_log: ?[]const u8 = null,
    /// Null keeps every reply identical to a server that has no compaction.
    compact_at: ?CompactAt = null,
    /// Tokens of the recent tail kept whole. Null uses min(20000, a quarter of the window).
    compact_keep: ?u32 = null,
    /// Directory of the stored note, or null when notes are not stored.
    compact_memory: ?[]const u8 = null,
    timeouts: http_conn.Timeouts = .{},
};

/// What the thinking budget's cut writes before the think block's end (GLM-5.3), as the model's own words.
pub const think_wrap_text = "\n\nI have used up my thinking budget, so I will stop thinking here and write my final answer now, based on what I have worked out so far.";

/// The checkpoint fold (GLM-5.3), in the model's own words: the engine asks for the conclusions so far in a fixed
/// form ending with fold_end_marker, keeps them under the label in place of the reasoning before, and goes on after
/// the resume line, thinking only about what is still open.
const fold_open_organise = "\n\n(Checkpoint. My reasoning above has grown long and will be dropped from my context; only this checkpoint stays in view. I write it in this form, then think only about what is still open:\nCONCLUSIONS: what I have settled, with exact numbers, names and paths, including any earlier checkpoint's conclusions.\nRULED OUT: approaches that failed or turned out wrong, one line each.\nOPEN: what is still undecided.\nNEXT: the one thing to examine next.\n" ++ fold_end_marker ++ ")\n\nCONCLUSIONS:\n";
pub const fold_label_text = "\n\n(A checkpoint of my earlier reasoning, which is no longer in view:)\n\nCONCLUSIONS:\n";
const fold_resume_organise = "\n\n(Continuing from the checkpoint: I think only about what is OPEN, starting with NEXT, and I do not redo what is CONCLUDED or RULED OUT.)\n\n";
/// The checkpoint's stages: organise (the first fold with most of the budget left), converge (a later fold, or half
/// the budget spent: the best answer so far and the shortest path to the final one), commit (a quarter of the budget
/// left: the final candidate and one last check, then the answer). Drafts until the tested wording replaces them.
pub const fold_stages = 3;
const fold_open_converge = "\n\n(Checkpoint. My reasoning above will be dropped from my context; only this checkpoint stays in view. It is time to converge: I write it in this form, then spend the rest only on reaching the final answer:\nCONCLUSIONS: what I have settled, with exact numbers, names and paths, including any earlier checkpoint's conclusions.\nRULED OUT: approaches that failed or turned out wrong, one line each.\nBEST ANSWER SO FAR: my current best final answer, and how sure I am of it.\nSHORTEST PATH: the fewest steps that would confirm or correct it; side questions are dropped.\n" ++ fold_end_marker ++ ")\n\nCONCLUSIONS:\n";
const fold_open_commit = "\n\n(Checkpoint. My thinking budget is nearly spent and my reasoning above will be dropped; only this checkpoint stays in view. I commit now: I write it in this form, do only the one check it names, then write the answer:\nCONCLUSIONS: what I have settled, with exact numbers.\nFINAL CANDIDATE: the answer I will give.\nLAST CHECK: the single check worth doing before I answer, or none.\n" ++ fold_end_marker ++ ")\n\nCONCLUSIONS:\n";
const fold_resume_converge = "\n\n(Continuing from the checkpoint: I follow SHORTEST PATH toward the final answer, and I do not redo what is CONCLUDED or RULED OUT.)\n\n";
const fold_resume_commit = "\n\n(Continuing from the checkpoint: I do the LAST CHECK, if any, and then I write the final answer.)\n\n";
pub const fold_open_texts = [fold_stages][]const u8{ fold_open_organise, fold_open_converge, fold_open_commit };
pub const fold_resume_texts = [fold_stages][]const u8{ fold_resume_organise, fold_resume_converge, fold_resume_commit };
pub const fold_end_marker = "END CHECKPOINT";
/// The summary at the think block's close (thinking_fold_at_close_share), in the model's own words: the thinking is done,
/// and what it learned replaces it under the same label as a fold's checkpoint, for the answer and the later turns.
/// It keeps the knowledge, not the history: the facts, the conditional results, every idea ruled out with its reason
/// (dropping one would let the next reasoning repeat it), the mistakes corrected and the hypotheses still open, each
/// in a line or two; the back-and-forth, the musing and the restatements go.
const fold_close_view = " My reasoning above will be dropped from my context before I answer; only this checkpoint stays in view, and my answer and the later turns are written from it alone.";
const fold_close_keep = " I keep every piece of knowledge my reasoning earned, not its history: a long path that ended in a result becomes one line with that result and how I established it. I write it in this form:\nCONCLUSIONS: the facts I established and the results I settled, with exact numbers, names, paths and code decisions.\nDERIVED: intermediate results worth keeping, each with its condition (if X, then Y).\nRULED OUT: every approach, cause or idea I rejected, each with why it fails and how I showed it (verified, or only argued). I never leave one out, however long it took to reach.\nCORRECTED: mistakes I made and fixed, each with the right version.\n";
const fold_close_drop = "I leave out the back-and-forth, the musing and the restatements, keeping only what I learned from them, and I write \"none\" under a heading with nothing in it.";
const fold_close_end = "\n" ++ fold_end_marker ++ ")\n\nCONCLUSIONS:\n";
pub const fold_close_open_text = "\n\n(Checkpoint. My thinking is complete." ++ fold_close_view ++ fold_close_keep ++
    "OPEN: hypotheses not yet confirmed, what each rests on, and what would settle it.\n" ++
    "PLAN: the answer as I will write it, part by part in order, specific enough to write it from this alone.\n" ++
    "CHECKS: the points to keep right while writing.\n" ++ fold_close_drop ++ fold_close_end;
/// The same summary when the thinking budget's cut closes the block (in place of think_wrap_text). The thinking
/// stops unfinished and cannot go on, so it also records where it stopped (for a later turn to pick up), and decides
/// now, for each point left unsettled, its best answer, how sure it is and what the answer does about it; the plan
/// writes the settled parts in full and states the assumption under each part resting on an unsettled point, and the
/// answer is written from the checkpoint rather than starting the analysis over.
pub const fold_close_budget_open_text = "\n\n(Checkpoint. I have used up my thinking budget, so I stop thinking here, before I have finished." ++ fold_close_view ++
    " I cannot think any further, so I decide now what the answer does with each point still open." ++ fold_close_keep ++
    "STOPPED AT: what I was working on when the budget ran out, how far I got, and the step I would have taken next.\n" ++
    "UNSETTLED: each point I could not settle: my best answer to it so far, how sure I am, and what the answer does about it (takes that choice and says so, or leaves it marked as open).\n" ++
    "PLAN: the answer as I will write it, part by part in order, specific enough to write it from this alone: the settled parts in full, and under each part that rests on an unsettled point, the assumption it makes.\n" ++
    "CHECKS: the points to keep right while writing.\n" ++ fold_close_drop ++ " In the answer I write from this checkpoint and do not start the analysis over." ++ fold_close_end;
/// Fiore-lab: the summary beside the reply (thinking_fold branch): after the block
/// so far, the block closed, a user turn asking for a summary of the reasoning since its last one, and an answer begun
/// without thinking at the form's first heading; the model's answer up to its EOS is the summary (an answer always ends,
/// where a checkpoint written in the block seldom wrote its end marker). It goes into the block under
/// fold_branch_label_text, which ends with the same heading, and the thinking goes on after fold_branch_resume_text.
/// Fiore-lab: what the model is asked at the budget's end, and at its own close when that is on; each can be
/// changed while the engine runs and goes back to this when set empty. tf_launcher.py's SETTINGS holds the same text.
pub const fold_prompt_default = "Organize your thoughts so far. Keep code, references and numbers exactly as they are.";
pub const close_prompt_default = fold_prompt_default;
const fold_branch_ask_text = "Your thinking above has grown long and will be dropped from your context; only a summary of it stays in view. Write that summary for yourself, of the part after your last summary (an earlier summary stays in view as it is), so that you can go on thinking from it alone. It is only for your thinking: do not write the answer, its code or anything else the user asked for here; that comes later, in your reply. Write each decision as a decision, not as finished code or text. Keep every piece of knowledge the reasoning earned, not its history: under CONCLUSIONS the facts and results you settled, with exact numbers, names and paths, the conditional results with their conditions and the mistakes you corrected; under RULED OUT every idea you rejected, with the reason; under OPEN what is still undecided (none if nothing); under NEXT the one thing to examine next. Leave out the back-and-forth and the restatements. Use this form and write nothing else:\nCONCLUSIONS: ...\nRULED OUT: ...\nOPEN: ...\nNEXT: ...";
const fold_branch_heading = "CONCLUSIONS:\n";
pub const fold_branch_label_text = "\n\n(My reasoning grew long, so I summarized it once; what I reasoned before this summary is no longer in view:)\n\n" ++ fold_branch_heading;
const fold_branch_resume_text = "\n\n(Continuing from the summary: if nothing is OPEN, I close my thinking now and write the answer; otherwise I think only about what is OPEN, starting with NEXT, and I do not redo what is CONCLUDED or RULED OUT.)\n\n";

/// An inserted text as the engine writes it: right after a line break the model wrote (a notice, the budget's cut, a
/// fold's texts all wait for one), without line breaks of its own in front, which the tokenizer would join with that
/// break into other ids than the ones written (so the next turn's prompt would not resume the cache there).
pub fn inserted(text: []const u8) []const u8 {
    return std.mem.trimStart(u8, text, "\n");
}

/// The ids whose text holds a line break, sorted (each token decoded alone; a scratch arena).
fn lineBreaks(gpa: Allocator, a: Allocator, text: model_text.Text) ![]const u32 {
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    var out: std.ArrayList(u32) = .empty;
    var id: u32 = 0;
    while (id < text.vocabSize()) : (id += 1) {
        if (text.isSpecial(id)) continue;
        const piece = try text.decode(scratch.allocator(), &.{id});
        if (std.mem.indexOfScalar(u8, piece, '\n') != null) try out.append(a, id);
        if (id % 4096 == 4095) _ = scratch.reset(.retain_capacity);
    }
    return out.items;
}

/// Fiore-lab: the thinking defaults POST /v1/thinking changes (Server.thinking).
pub const Thinking = struct { share: i64 = 0, budget: i64 = 0, fold: api.Fold = .off, at_close_share: i64 = 0, fold_share: i64 = 0, fold_max: i64 = 0, at_budget: i64 = 1, fold_margin: i64 = 10, fold_prompt: []const u8 = fold_prompt_default, close_prompt: []const u8 = close_prompt_default, fold_step: ?api.ThinkStep = null, close_step: ?api.ThinkStep = null };
/// What POST /v1/thinking asks to change (each optional), checked by status_routes.thinkingCheck.
pub const ThinkingSet = struct { share: ?i64 = null, fold: ?api.Fold = null, at_close_share: ?i64 = null, fold_share: ?i64 = null, fold_max: ?i64 = null, at_budget: ?i64 = null, fold_margin: ?i64 = null, fold_prompt: ?[]const u8 = null, close_prompt: ?[]const u8 = null };

pub const Server = struct {
    gpa: Allocator,
    io: std.Io,
    engine: api.Engine,
    info: api.Info,
    text: model_text.Text,
    config: Config,
    keys: ?*auth.Store,
    late_system: []const u8 = "system",
    needs_user_after_tool: bool = false,
    markers: reply_text.Markers = reply_text.think_markers,
    chunks: chunk_plan.Plan = .{},
    effort_levels: []const []const u8 = &.{},
    eos: []const u32 = &.{},
    think_close: []const u32 = &.{},
    think_close_end: ?u32 = null,
    /// The budget's cut on GLM-5.3: a closing sentence, then the think block's end; and the tokens holding a line
    /// break, sorted (a notice or the cut waits for one).
    think_wrap: []const u32 = &.{},
    think_breaks: []const u32 = &.{},
    /// The checkpoint fold's texts (GLM-5.3): the request for it, the label it is kept under, the line the thinking
    /// goes on after, and the end the engine writes when the model's checkpoint runs past its cap.
    fold_opens: [fold_stages][]const u32 = @splat(&.{}),
    fold_label: []const u32 = &.{},
    fold_resumes: [fold_stages][]const u32 = @splat(&.{}),
    fold_closes: [fold_stages][]const u32 = @splat(&.{}),
    fold_close_open: []const u32 = &.{},
    fold_close_budget_open: []const u32 = &.{},
    /// Fiore-lab, the summary beside the reply (thinking_fold branch; GLM-5.3's turns): its request after the block
    /// so far, the label it is kept under in the block and the line the thinking goes on after; empty without the
    /// template's turn tokens.
    fold_branch_ask: []const u32 = &.{},
    fold_branch_label: []const u32 = &.{},
    fold_branch_resume: []const u32 = &.{},
    /// The tokens that end the summary beside the reply as its EOS does: the other turns' tokens and a tool call's
    /// opening (an answer that calls a tool has finished its summary); and the think block's opening, dropped from it.
    fold_branch_stops: []const u32 = &.{},
    fold_branch_open: ?u32 = null,
    metrics: metrics.Metrics,
    store: responses.Store,
    next_id: std.atomic.Value(u64) = .init(1),
    /// Connections open now; a stop waits for them before freeing what they read.
    open_connections: std.atomic.Value(u32) = .init(0),
    /// The idle keepalive on the engine's queue, armed by --keep-warm; null when off or not Metal.
    keepalive: ?*api.keepalive.Keepalive = null,
    preparing: std.atomic.Value(i64) = .init(0),
    /// Fiore-lab: the thinking defaults a request takes when it names none, changed while the engine runs by POST
    /// /v1/thinking; each request's admission reads them whole (thinkingNow), so a change takes the requests that start
    /// after it. `thinking_start`: what the server started with (the serve flags)
    /// Fiore-lab: 要求が指定しないときの思考の既定。POST /v1/thinking が動作中に変える。要求の受け付けごとにまとめて読む
    /// ので、変更はその後に始まる要求から効く。`thinking_start` は起動時の値
    thinking: Thinking = .{},
    thinking_start: Thinking = .{},
    thinking_mutex: std.Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,

    /// The thinking defaults now (share: percent of a reply's max_tokens, at most what the prompt leaves of the context;
    /// budget: tokens, 0 none and the share then decides; fold; at_close_share: percent of the thinking's limit, 0 never; fold_share: a fold no later
    /// than this percent of max_tokens, 0 none; fold_max: a reply's folds at most, 0 no limit).
    pub fn thinkingNow(srv: *Server) Thinking {
        srv.thinking_mutex.lockUncancelable(srv.io);
        defer srv.thinking_mutex.unlock(srv.io);
        return srv.thinking;
    }

    /// The thinking defaults changed (each field given; a share sets the budget to 0, as a budget would win over it);
    /// the values in use answered.
    pub fn thinkingSet(srv: *Server, set: ThinkingSet) Thinking {
        srv.thinking_mutex.lockUncancelable(srv.io);
        defer srv.thinking_mutex.unlock(srv.io);
        if (set.share) |v| {
            srv.thinking.share = v;
            srv.thinking.budget = 0;
        }
        if (set.fold) |v| srv.thinking.fold = v;
        if (set.at_close_share) |v| srv.thinking.at_close_share = v;
        if (set.fold_share) |v| srv.thinking.fold_share = v;
        if (set.fold_max) |v| srv.thinking.fold_max = v;
        if (set.at_budget) |v| srv.thinking.at_budget = v;
        if (set.fold_margin) |v| srv.thinking.fold_margin = v;
        // kept for good: replies in flight hold them, and they change by hand
        // 捨てない: 動いている応答が持っている。変更は手でするもの
        if (set.fold_prompt) |t| {
            const text = if (t.len == 0) fold_prompt_default else srv.gpa.dupe(u8, t) catch return srv.thinking;
            if (srv.promptStep(srv.gpa, text) catch null) |st| {
                srv.thinking.fold_prompt = text;
                srv.thinking.fold_step = st;
            }
        }
        if (set.close_prompt) |t| {
            const text = if (t.len == 0) close_prompt_default else srv.gpa.dupe(u8, t) catch return srv.thinking;
            if (srv.promptStep(srv.gpa, text) catch null) |st| {
                srv.thinking.close_prompt = text;
                srv.thinking.close_step = st;
            }
        }
        return srv.thinking;
    }

    /// The step that asks `text` beside the reply: the block closed, a user turn, an answer without thinking; the
    /// result is what precedes the first close. Null when the model has no such turns.
    /// `text` を応答の横で求める手順: ブロックを閉じ、ユーザーの番、思考なしの答え。結果は最初の閉じるトークンの前まで。
    fn promptStep(srv: *Server, a: Allocator, text: []const u8) !?api.ThinkStep {
        const end = srv.think_close_end orelse return null;
        const user = srv.text.tokenId("<|user|>") orelse return null;
        const assistant = srv.text.tokenId("<|assistant|>") orelse return null;
        const open = srv.text.tokenId("<think>") orelse return null;
        return .{ .ask = try std.mem.concat(a, u32, &.{ &.{ end, user }, try srv.text.encode(a, text, false), &.{ assistant, open, end } }), .take = .before };
    }

    /// Whether this model can fold a think block the way `fold` asks (its texts were made at init).
    pub fn canFold(srv: *const Server, fold: api.Fold) bool {
        return switch (fold) {
            .off => false,
            .branch => srv.fold_branch_ask.len > 0,
            .recent, .checkpoint => srv.fold_opens[0].len > 0,
        };
    }

    fn grammarBytes(ctx: *anyopaque, a: Allocator, id: u32) ?[]const u8 {
        const srv: *Server = @ptrCast(@alignCast(ctx));
        return srv.text.tokenBytes(a, id) catch null;
    }

    fn grammarAdded(ctx: *anyopaque, id: u32) bool {
        const srv: *Server = @ptrCast(@alignCast(ctx));
        return srv.text.isAdded(id);
    }

    /// Reads what the template and tokenizer decide once: late system role, think markers, efforts, the forced close.
    pub fn init(gpa: Allocator, io: std.Io, engine: api.Engine, text: model_text.Text, config: Config, keys: ?*auth.Store) !*Server {
        const srv = try gpa.create(Server);
        srv.* = .{ .gpa = gpa, .io = io, .engine = engine, .info = engine.info(), .text = text, .config = config, .keys = keys, .metrics = .{ .gpa = gpa }, .store = .{ .gpa = gpa }, .arena = .init(gpa) };
        srv.thinking = .{ .share = config.thinking_share, .budget = config.thinking_budget, .fold = config.thinking_fold, .at_close_share = config.thinking_fold_at_close_share, .fold_share = config.thinking_fold_share, .fold_max = config.thinking_fold_max, .at_budget = config.thinking_fold_at_budget, .fold_margin = config.thinking_fold_margin, .fold_prompt = if (config.thinking_fold_prompt.len > 0) config.thinking_fold_prompt else fold_prompt_default, .close_prompt = if (config.thinking_close_prompt.len > 0) config.thinking_close_prompt else close_prompt_default };
        srv.thinking_start = srv.thinking;
        if (config.keep_warm_s > 0) if (engine.keepaliveTarget()) |target| {
            srv.keepalive = api.keepalive.Keepalive.start(gpa, io, target, @as(i64, config.keep_warm_s) * std.time.ns_per_s) catch |e| blk: {
                log.line("idle keepalive off: {s}", .{@errorName(e)});
                break :blk null;
            };
        };
        const a = srv.arena.allocator();
        srv.eos = try a.dupe(u32, text.eosIds());
        // Fiore-lab: the grammar vocabulary, once (VTable.grammar_init): every id's raw bytes as this tokenizer knows
        // them and the model's end tokens; structured output stays off when it cannot be built
        // Fiore-lab: 文法の語彙を 1 回 (VTable.grammar_init): この tokenizer が知る全 id の生のバイトとモデルの終わりの
        // token。作れなければ構造化出力はオフのまま
        if (engine.vtable.grammar_init != null) {
            engine.grammarInit(.{ .vocab = .{ .ctx = srv, .bytes = grammarBytes, .added = grammarAdded }, .size = text.vocabSize(), .eos = srv.eos }) catch |e|
                log.line("structured output off: the grammar vocabulary could not be built ({s})", .{@errorName(e)});
            srv.info = engine.info();
        }
        if (text.tokenId(reply_text.channel_markers.close) != null) srv.markers = reply_text.channel_markers;
        srv.effort_levels = try fields.effortLevels(a, text.templateSource());
        srv.late_system = try lateSystem(a, text);
        srv.needs_user_after_tool = needsUserAfterTool(text.templateSource());
        if (text.tokenId("</think>")) |end| {
            // the forced close ends at </think>: the next turn's template renders an answer right after it (stripped),
            // so a line break the engine wrote past it would differ there and cost the cache the whole answer (11-02:
            // 4,446 of 12,955 resumed after a summary at the close)
            // 強制の閉じは </think> で終える: 次のターンのテンプレートは答えをその直後に (前後の空白を除いて) 組むので、
            // エンジンがその後ろに書いた改行はそこで食い違い、答え全部のキャッシュを失う (11-02: 思考完了時要約の後、12,955 のうち
            // 4,446 しか再開しなかった)
            const lead = try text.encode(a, "\n", false);
            srv.think_close = try std.mem.concat(a, u32, &.{ lead, &.{end} });
            srv.think_close_end = end;
            srv.think_wrap = try std.mem.concat(a, u32, &.{ try text.encode(a, inserted(think_wrap_text), false), &.{end} });
            srv.think_breaks = try lineBreaks(gpa, a, text);
            srv.fold_label = try text.encode(a, inserted(fold_label_text), false);
            const marker = try text.encode(a, fold_end_marker ++ "\n", false);
            for (0..fold_stages) |k| {
                srv.fold_opens[k] = try text.encode(a, inserted(fold_open_texts[k]), false);
                srv.fold_resumes[k] = try text.encode(a, inserted(fold_resume_texts[k]), false);
                srv.fold_closes[k] = try std.mem.concat(a, u32, &.{ marker, srv.fold_resumes[k] });
            }
            srv.fold_close_open = try text.encode(a, inserted(fold_close_open_text), false);
            srv.fold_close_budget_open = try text.encode(a, inserted(fold_close_budget_open_text), false);
            // the summary beside the reply: the block closed, a user turn and an answer without thinking, as GLM-5.3's
            // template renders them (<|user|>text, <|assistant|><think></think>answer), the answer begun at the
            // form's first heading
            // 応答の横での要約: ブロックを閉じ、ユーザーの番と思考なしの答え (GLM-5.3 のテンプレートの形
            // <|user|>本文、<|assistant|><think></think>答え)。答えは型の最初の見出しから始める
            if (text.tokenId("<|user|>")) |user| if (text.tokenId("<|assistant|>")) |assistant| if (text.tokenId("<think>")) |open| {
                srv.fold_branch_ask = try std.mem.concat(a, u32, &.{ &.{ end, user }, try text.encode(a, fold_branch_ask_text, false), &.{ assistant, open, end }, try text.encode(a, fold_branch_heading, false) });
                srv.fold_branch_label = try text.encode(a, inserted(fold_branch_label_text), false);
                srv.fold_branch_resume = try text.encode(a, inserted(fold_branch_resume_text), false);
                srv.thinking.fold_step = try srv.promptStep(a, srv.thinking.fold_prompt);
                srv.thinking.close_step = try srv.promptStep(a, srv.thinking.close_prompt);
                srv.thinking_start = srv.thinking;
                var stops: std.ArrayList(u32) = .empty;
                try stops.appendSlice(a, &.{ assistant, user });
                srv.fold_branch_open = open;
                for ([_][]const u8{ "<|system|>", "<|observation|>", "<tool_call>" }) |s| if (text.tokenId(s)) |id| try stops.append(a, id);
                srv.fold_branch_stops = stops.items;
            };
        }
        if (srv.info.prefill_step > 0) {
            srv.chunks = try chunk_plan.markers(srv, a, srv.info.prefill_step);
            log.line("prompt chunks of up to {d},{d:0>3} tokens, cut at replies {d}+ tokens apart", .{ srv.info.prefill_step / 1000, srv.info.prefill_step % 1000, srv.chunks.min_chunk });
        }
        return srv;
    }

    pub fn deinit(srv: *Server) void {
        if (srv.keepalive) |k| k.stop(); // before the engine's queue goes away
        srv.arena.deinit();
        srv.store.deinit();
        srv.gpa.destroy(srv);
    }

    /// The effort the template hears: the request's name, or this server's default.
    pub fn effortFor(srv: *const Server, explicit: ?[]const u8) ?[]const u8 {
        const e = explicit orelse srv.config.reasoning_effort orelse return null;
        return fields.coerceEffort(e, srv.effort_levels);
    }

    /// ``_resolve_sampling``: omitted or null fields keep the defaults; an omitted seed is keyed to the prompt.
    pub fn resolveSampling(srv: *Server, cx: *Cx, f: Value, temperature: f64, prompt_ids: []const u32) errors.Refused!?api.Sampling {
        _ = temperature; // the request's own temperature field decides, as in the Python server
        const options = try json.newObject(cx.a);
        if (srv.config.default_sampling) |d| try merge(options, cx, try fields.parseNumbers(cx, d));
        try merge(options, cx, try fields.parseNumbers(cx, f));
        const temp = number(options.get("temperature")) orelse 0;
        if (temp <= 0) return null;
        const seed: u64 = if (options.get("seed")) |s| seedBits(s) else api.seedFor(prompt_ids, srv.config.seed_salt);
        return .{
            .seed = seed,
            .temperature = @floatCast(temp),
            .top_k = @intCast(@min(@max(0, (if (options.get("top_k")) |k| k.int64() else null) orelse 0), std.math.maxInt(u32))),
            .top_p = @floatCast(number(options.get("top_p")) orelse 1.0),
            .min_p = @floatCast(number(options.get("min_p")) orelse 0.0),
        };
    }

    /// The call gate and grammar a request asks for, or its refusal when the engine cannot enforce them.
    pub fn checkFeatures(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, prompt_ids: []const u32, tools: []const Value, request: *api.Request) errors.Refused!void {
        _ = prompt_ids;
        const required = if (f.get("tool_call_required")) |r| r.truthy() else false;
        if (has_tools and required) {
            const form = try srv.callForm(cx);
            const opener = if (form) |fm| srv.text.tokenId(fm.opener) else null;
            if (opener == null) return cx.refuse("tool_choice \"required\" or a named function needs a chat template that marks tool calls (<tool_call> or <|tool_call>), and this one does not: send \"auto\"");
            if (!srv.info.call_gates) return cx.refuse("tool_choice \"required\" or a named function is not supported by this engine yet: send \"auto\"");
            var names: std.ArrayList([]const u8) = .empty;
            if (form.?.lead != null) for (tools) |t| try names.append(cx.a, try tool_specs.toolName(cx.a, t));
            const think_open = if (srv.markers.open.len > 0) blk: {
                const ids = srv.text.encode(cx.a, srv.markers.open, false) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Template => &[_]u32{},
                };
                break :blk if (ids.len > 0) ids[0] else null;
            } else srv.text.tokenId("<think>");
            request.call = .{ .opener = opener.?, .lead = form.?.lead orelse "", .tail = form.?.tail orelse "", .names = names.items, .think_open = think_open, .think_end = srv.text.tokenId(srv.markers.close) };
        }
        if (try grammar.spec(cx, f)) |s| {
            if (!srv.info.structures) return cx.refuse("structured output (response_format and the guided_* fields) is not supported by this engine yet");
            // Fiore-lab: structured output takes json_object and json_schema, the schema checked and rewritten for
            // llguidance (schema_check: what llguidance would weaken without an error is a 400 here); it is compiled on
            // this thread just before submit (generate), so a refusal still comes before any header
            // Fiore-lab: 構造化出力は json_object と json_schema。schema は llguidance 向けに確かめて書き直す (schema_check:
            // llguidance がエラーなしに弱めるものはここで 400)。compile は submit の直前にこのスレッドで (generate) するので、
            // 断る時もヘッダの前
            const text: []const u8 = switch (s.kind) {
                .json => schema_check.json_object_text,
                .json_schema => blk: {
                    const value = switch (try json.parseText(cx.a, s.text)) {
                        .ok => |v| v,
                        .err => return cx.fail(.request, "{s} is not valid JSON", .{s.field}),
                    };
                    break :blk switch (try schema_check.check(cx.a, value)) {
                        .text => |t| t,
                        .refused => |r| return cx.fail(.request, "{s}: {s}", .{ s.field, r.message }),
                    };
                },
                else => return cx.fail(.request, "{s}: regex, choice and grammar are not supported yet; use response_format json_object or json_schema", .{s.field}),
            };
            if (has_tools) return cx.refuse("structured output together with tools is not supported yet: send the tools or the response_format, not both");
            request.structure = .{ .kind = s.kind, .text = text, .after = if (thinking) srv.text.tokenId(srv.markers.close) else null };
        }
    }

    const Form = struct { opener: []const u8, lead: ?[]const u8, tail: ?[]const u8 };

    /// (opener, lead, tail) of this template's tool calls, read from a rendered probe call.
    fn callForm(srv: *Server, cx: *Cx) errors.Refused!?Form {
        const probe = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"tfprobe_fn\", \"arguments\": {}}}]}]")).ok;
        var scratch: Cx = .{ .a = cx.a };
        const text = if (prompt.renderIds(srv, &scratch, probe, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |_| "";
        if (std.mem.lastIndexOf(u8, text, "tfprobe_fn")) |at| {
            var best: ?usize = null;
            var opener: []const u8 = "";
            for (reply_text.calls) |c| if (std.mem.lastIndexOf(u8, text[0..at], c[0])) |s| {
                if (best == null or s > best.? or (s == best.? and std.mem.order(u8, c[0], opener) == .gt)) {
                    best = s;
                    opener = c[0];
                }
            };
            if (best) |s| {
                const end = at + "tfprobe_fn".len;
                return .{ .opener = opener, .lead = text[s + opener.len .. at], .tail = if (end < text.len) text[end .. end + 1] else "" };
            }
        }
        for (reply_text.calls) |c| if (srv.text.tokenId(c[0]) != null) return .{ .opener = c[0], .lead = null, .tail = null };
        return null;
    }

    /// Counts one finished request for /metrics. `decode` / `queue`: the engine's own decode and queue seconds, when it
    /// measures them (the decode otherwise runs from the first token's arrival to now).
    pub fn noteRequest(srv: *Server, prompt_len: usize, generated: usize, drafted: u64, accepted: u64, rounds: u64, received: i96, first: ?i96, last: ?i96, prefill: ?f64, cached: ?usize, decode: ?f64, queue: ?f64) void {
        const now = clock.nowNs(srv.io);
        const tpot = srv.metrics.tpotValue(first, last, generated);
        srv.metrics.note(srv.io, prompt_len, generated, drafted, accepted, rounds, @max(0, clock.seconds(now - received)), if (first) |f| clock.seconds(f - received) else null, decode orelse if (first) |f| @max(0, clock.seconds(now - f)) else null, prefill, tpot, cached, queue);
    }

    /// Accepts connections until ``stop`` is set, each on its own thread.
    pub fn serve(srv: *Server, listener: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        while (listener.accept(stop)) |c| {
            const t = std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ srv, c }) catch {
                _ = std.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    fn connection(srv: *Server, accepted: listener_mod.Listener.Accepted) void {
        _ = srv.open_connections.fetchAdd(1, .acq_rel);
        defer _ = srv.open_connections.fetchSub(1, .acq_rel);
        defer _ = std.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        const peer = listener_mod.peerText(&peer_buf, accepted.peer);
        var conn = http_conn.Conn.init(srv.gpa, accepted.fd, peer) catch return;
        defer conn.deinit();
        conn.timeouts = srv.config.timeouts;
        conn.auth_enabled = srv.keys != null and srv.keys.?.enabled();
        if (conn.auth_enabled) conn.hook = .{ .ctx = srv, .call = countReply };
        var arena: std.heap.ArenaAllocator = .init(srv.gpa);
        defer arena.deinit();
        while (true) {
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();
            const outcome = conn.readRequest(a, true) catch break;
            switch (outcome) {
                .closed => break,
                .handled => if (conn.close) break else continue,
                .ready => {},
            }
            if (!srv.authenticate(&conn, a)) break;
            if (conn.header("Expect")) |expect| if (std.ascii.eqlIgnoreCase(expect, "100-continue") and std.mem.order(u8, conn.version, "HTTP/1.1") != .lt) conn.sendContinue();
            routes.dispatch(srv, &conn, a);
            if (conn.close or conn.broken) break;
        }
    }

    /// Fiore-lab: the read-only listener (TENSORFOLD_METRICS_LISTEN), until ``stop`` is set: /metrics and /health for
    /// monitors on other hosts while the API listens on loopback. Its connections are not logged; a stop waits for
    /// them as for the API's.
    pub fn serveMetrics(srv: *Server, listener: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        while (listener.accept(stop)) |c| {
            const t = std.Thread.spawn(.{ .stack_size = 1 << 20 }, metricsConnection, .{ srv, c }) catch {
                _ = std.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    fn metricsConnection(srv: *Server, accepted: listener_mod.Listener.Accepted) void {
        // counted with the API's connections, so a stop does not free the server under one (release review 10-24)
        // API の接続といっしょに数える: 停止がこの接続の下でサーバーを解放しないように
        _ = srv.open_connections.fetchAdd(1, .acq_rel);
        defer _ = srv.open_connections.fetchSub(1, .acq_rel);
        defer _ = std.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        const peer = listener_mod.peerText(&peer_buf, accepted.peer);
        var conn = http_conn.Conn.init(srv.gpa, accepted.fd, peer) catch return;
        defer conn.deinit();
        conn.timeouts = srv.config.timeouts;
        var arena: std.heap.ArenaAllocator = .init(srv.gpa);
        defer arena.deinit();
        while (true) {
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();
            const outcome = conn.readRequest(a, true) catch break;
            switch (outcome) {
                .closed => break,
                .handled => if (conn.close) break else continue,
                .ready => {},
            }
            conn.quiet_log = true;
            routes.metricsOnly(srv, &conn, a);
            if (conn.close or conn.broken) break;
        }
    }

    fn countReply(ctx: *anyopaque, conn: *http_conn.Conn, code: u16) void {
        const srv: *Server = @ptrCast(@alignCast(ctx));
        srv.metrics.httpRequest(srv.io, conn.key_label orelse "unauthenticated", code);
    }

    /// The key gate before any route: 401 for a gated path without a valid key (``Authenticated._authenticate``).
    fn authenticate(srv: *Server, conn: *http_conn.Conn, a: Allocator) bool {
        const keys = srv.keys orelse return true;
        if (std.mem.eql(u8, conn.method, "OPTIONS") or !keys.enabled()) return true;
        conn.key_label = keys.match(a, single(conn, "Authorization"), single(conn, "x-api-key"));
        if (!auth.gated(conn.path, keys.metrics_open) or conn.key_label != null) return true;
        const path = auth.routePath(conn.path);
        const messages = for ([_][]const u8{ "/v1/messages", "/v1/messages/count_tokens", "/messages", "/messages/count_tokens" }) |p| {
            if (std.mem.eql(u8, path, p)) break true;
        } else false;
        const body = if (messages)
            "{\"type\": \"error\", \"error\": {\"type\": \"authentication_error\", \"message\": \"A valid API key is required\"}}"
        else
            "{\"error\": {\"message\": \"A valid API key is required\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": \"invalid_api_key\"}}";
        conn.close = true;
        conn.startResponse(401, null) catch return false;
        conn.addHeader("WWW-Authenticate", "Bearer") catch return false;
        conn.addHeader("Content-Type", "application/json") catch return false;
        var len: [24]u8 = undefined;
        conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return false;
        conn.addHeader("Connection", "close") catch return false;
        conn.finish(body) catch {};
        return false;
    }
};

/// A header that appears exactly once; none or several read as absent (as the key match reads them).
fn single(conn: *const http_conn.Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}

fn merge(into: *json.Object, cx: *Cx, from: Value) errors.Refused!void {
    for (from.object.keys(), from.object.values()) |k, v| if (v != .null) try into.put(cx.a, k, v);
}

fn number(v: ?Value) ?f64 {
    const x = v orelse return null;
    return if (x == .bool) null else x.float64();
}

/// A request seed as the engine's u64: negative and huge ints wrap as two's complement bits.
fn seedBits(v: Value) u64 {
    if (v != .int) return 0;
    if (std.fmt.parseInt(i128, v.int, 10)) |n| return @truncate(@as(u128, @bitCast(n))) else |_| return 0;
}

/// ``template_late_system``: "system" when the template keeps a later system message in place, else "user".
fn lateSystem(a: Allocator, text: model_text.Text) ![]const u8 {
    const probe_text = "tensorfold-late-system-probe";
    const probe = (try json.parse(a, "[{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"u\"}, {\"role\": \"assistant\", \"content\": \"a\"}, {\"role\": \"system\", \"content\": \"tensorfold-late-system-probe\"}, {\"role\": \"user\", \"content\": \"v\"}]")).ok;
    var problem: []const u8 = "";
    const rendered = text.render(a, probe, .{ .add_generation_prompt = false }, &problem) catch return "user";
    return if (std.mem.indexOf(u8, rendered, probe_text) != null) "system" else "user";
}

/// ``needs_user_after_tool``: the template raises "No user query found" when the conversation has no user query, which a conversation whose user turn became tool results no longer has; such templates gain a user turn after a trailing tool message.
fn needsUserAfterTool(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "No user query found") != null;
}
