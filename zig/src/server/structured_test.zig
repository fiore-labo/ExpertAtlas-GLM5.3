//! Fiore-lab: structured output's server side against a fake engine with the grammar hooks: what the engine is
//! given (the checked schema, the compiled handle), what is refused before anything is compiled or submitted, the
//! handle freed when submit does not take it, the answer from api.Event.answer on, and an engine failure winning.
//! Fiore-lab: 構造化出力のサーバー側を、文法の hook を持つ偽のエンジンで試す: エンジンが受け取るもの (確かめた
//! schema、compile した handle)、compile や submit の前に断るもの、submit が受け取らなかった handle の解放、
//! api.Event.answer からの答え、エンジンの失敗が勝つこと。
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const errors = @import("errors.zig");
const schema_check = @import("schema_check.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens); 0 is the model's end token.
const TestText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine with the grammar hooks: it counts compiles, frees and submits, keeps the text it compiled and the handle
/// it was given, and answers with some free text, the answer event, then `answer` (or fails after it).
const GrammarEngine = struct {
    inited: bool = false,
    vocab_size: u32 = 0,
    compiles: usize = 0,
    frees: usize = 0,
    submits: usize = 0,
    refuse_compile: bool = false,
    busy: bool = false,
    fail_after: bool = false,
    compiled_text: [512]u8 = undefined,
    compiled_len: usize = 0,
    handed: ?*anyopaque = null,
    token: u8 = 0, // the handle's storage
    answer: []const u8 = "{\"a\":1}",

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .grammar_init = grammarInit, .structure = structure, .structure_free = structureFree } };
    }
    fn self(ctx: *anyopaque) *GrammarEngine {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .context_window = 4096, .structures = self(ctx).inited };
    }
    fn grammarInit(ctx: *anyopaque, v: api.GrammarVocab) error{ Unavailable, OutOfMemory }!void {
        const e = self(ctx);
        e.inited = true;
        e.vocab_size = v.size;
    }
    fn structure(ctx: *anyopaque, a: std.mem.Allocator, s: *const api.Structure) std.mem.Allocator.Error!api.Compiled {
        const e = self(ctx);
        e.compiles += 1;
        @memcpy(e.compiled_text[0..s.text.len], s.text);
        e.compiled_len = s.text.len;
        if (e.refuse_compile) return .{ .refused = try a.dupe(u8, "the fake compiler refuses it") };
        return .{ .ok = &e.token };
    }
    fn structureFree(ctx: *anyopaque, handle: *anyopaque) void {
        const e = self(ctx);
        std.debug.assert(handle == @as(*anyopaque, &e.token));
        e.frees += 1;
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = self(ctx);
        if (e.busy) return error.Busy;
        e.submits += 1;
        e.handed = if (request.structure) |st| st.compiled else null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'x', 'y' } }); // before the answer: not the answer's
        sink.event(sink.ctx, id, &.answer);
        var ids: [64]u32 = undefined;
        for (e.answer, 0..) |c, i| ids[i] = c;
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..e.answer.len] });
        if (e.fail_after) {
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = "the grammar refused a token" } });
        } else {
            sink.event(sink.ctx, id, &.{ .tokens = &.{0} });
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
        }
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A whole reply's status, its content and its error message.
const Seen = struct {
    status: ?u16 = null,
    content: [256]u8 = undefined,
    content_len: usize = 0,
    message: [512]u8 = undefined,
    message_len: usize = 0,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        if (payload.get("error")) |err| if (err.get("message")) |m| if (m == .string) {
            const n = @min(m.string.len, c.message.len);
            @memcpy(c.message[0..n], m.string[0..n]);
            c.message_len = n;
        };
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const msg = choices.array[0].get("message") orelse return;
        const content = msg.get("content") orelse return;
        if (content != .string) return;
        const n = @min(content.string.len, c.content.len);
        @memcpy(c.content[0..n], content.string[0..n]);
        c.content_len = n;
    }
    fn contentText(c: *const Seen) []const u8 {
        return c.content[0..c.content_len];
    }
    fn messageText(c: *const Seen) []const u8 {
        return c.message[0..c.message_len];
    }
};

fn run(backend: *GrammarEngine, body: []const u8) !Seen {
    var text: TestText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = true,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const json_object_body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_object\"}}";

test "structured: the server builds the grammar vocabulary at start, compiles json_object's checked text before submit and hands the handle to the engine" {
    var e: GrammarEngine = .{};
    const seen = try run(&e, json_object_body);
    try std.testing.expect(e.inited);
    try std.testing.expectEqual(@as(u32, 256), e.vocab_size);
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expectEqual(@as(usize, 1), e.compiles);
    try std.testing.expectEqualStrings(schema_check.json_object_text, e.compiled_text[0..e.compiled_len]);
    try std.testing.expectEqual(@as(usize, 1), e.submits);
    try std.testing.expect(e.handed == @as(*anyopaque, &e.token));
    try std.testing.expectEqual(@as(usize, 0), e.frees); // the engine owns it after submit
    // the answer is what came from api.Event.answer on; the text before it is not
    try std.testing.expectEqualStrings("{\"a\":1}", seen.contentText());
}

test "structured: a json_schema goes to the engine as schema_check's text" {
    var e: GrammarEngine = .{};
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}}}}";
    const seen = try run(&e, body);
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const want = switch (try schema_check.check(arena.allocator(), (try json.parse(arena.allocator(), "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}")).ok)) {
        .text => |t| t,
        .refused => return error.UnexpectedRefusal,
    };
    try std.testing.expectEqualStrings(want, e.compiled_text[0..e.compiled_len]);
}

test "structured: refusals before any compile (the schema check, tools, stop strings, ignore_eos, regex)" {
    const bodies = [_][]const u8{
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\",\"nullable\":true}}}}}}",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_object\"},\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_object\"},\"stop\":[\"}\"]}",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"response_format\":{\"type\":\"json_object\"},\"ignore_eos\":true}",
        "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"guided_regex\":\"[0-9]+\"}",
    };
    const words = [_][]const u8{ "nullable", "tools", "stop strings", "ignore_eos", "regex" };
    for (bodies, words) |body, word| {
        var e: GrammarEngine = .{};
        const seen = try run(&e, body);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        if (std.mem.indexOf(u8, seen.messageText(), word) == null) {
            std.debug.print("expected \"{s}\" in: {s}\n", .{ word, seen.messageText() });
            return error.TestUnexpectedMessage;
        }
        try std.testing.expectEqual(@as(usize, 0), e.compiles);
        try std.testing.expectEqual(@as(usize, 0), e.submits);
    }
}

test "structured: a compile refusal is a 400 with the compiler's reason, nothing submitted or freed" {
    var e: GrammarEngine = .{ .refuse_compile = true };
    const seen = try run(&e, json_object_body);
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(std.mem.indexOf(u8, seen.messageText(), "the fake compiler refuses it") != null);
    try std.testing.expectEqual(@as(usize, 1), e.compiles);
    try std.testing.expectEqual(@as(usize, 0), e.submits);
    try std.testing.expectEqual(@as(usize, 0), e.frees);
}

test "structured: a busy engine leaves the compiled grammar with the server, which frees it once" {
    var e: GrammarEngine = .{ .busy = true };
    const seen = try run(&e, json_object_body);
    try std.testing.expectEqual(@as(?u16, 503), seen.status);
    try std.testing.expectEqual(@as(usize, 1), e.compiles);
    try std.testing.expectEqual(@as(usize, 1), e.frees);
}

test "structured: the engine's failure wins over the tokens that came before it" {
    var e: GrammarEngine = .{ .fail_after = true };
    const seen = try run(&e, json_object_body);
    try std.testing.expect(seen.status != null and seen.status.? >= 500);
    try std.testing.expect(std.mem.indexOf(u8, seen.messageText(), "the grammar refused a token") != null);
}
