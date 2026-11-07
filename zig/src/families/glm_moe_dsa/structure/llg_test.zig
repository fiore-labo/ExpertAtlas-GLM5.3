//! Fiore-lab: the llguidance binding against a real library and the GLM-5.3 vocabulary. TF_TEST_LLG_LIB names the
//! library (libllguidance.dylib on the Mac) and TF_TEST_GLM_VOCAB the vocabulary file (tokenizer.json's byte-level
//! BPE strings decoded: u32 count, then per id u32 length and raw bytes, added tokens empty); without them the tests
//! are skipped.
//! The fixture and its walk are public for the counterexample table (llg_table_test.zig).
//! Fiore-lab: 実際のライブラリと GLM-5.3 の語彙で llguidance の binding を試す。TF_TEST_LLG_LIB にライブラリ、
//! TF_TEST_GLM_VOCAB に語彙のファイル。なければ飛ばす。fixture と歩き方は反例の表 (llg_table_test.zig) のために公開する。

const std = @import("std");
const llg = @import("llg.zig");

pub const vocab_size: u32 = 154880;
pub const eos = [_]llg.Token{ 154820, 154827, 154829 };
pub const added_first: u32 = 154820;
pub const padding_first: u32 = 154856;

/// The 36 added tokens' names (tokenizer.json added_tokens, ids 154820..154855).
const added_names = [_][]const u8{
    "<|endoftext|>",        "[MASK]",          "[gMASK]",       "[sMASK]",           "<sop>",           "<eop>",
    "<|system|>",           "<|user|>",        "<|assistant|>", "<|observation|>",   "<|begin_of_image|>", "<|end_of_image|>",
    "<|begin_of_video|>",   "<|end_of_video|>", "<|begin_of_audio|>", "<|end_of_audio|>", "<|begin_of_transcription|>", "<|end_of_transcription|>",
    "<|code_prefix|>",      "<|code_middle|>", "<|code_suffix|>", "<think>",          "</think>",        "<tool_call>",
    "</tool_call>",         "<tool_response>", "</tool_response>", "<arg_key>",       "</arg_key>",      "<arg_value>",
    "</arg_value>",         "/nothink",        "<|begin_of_box|>", "<|end_of_box|>",  "<|image|>",       "<|video|>",
};

/// The vocabulary file as a VocabSource: BPE bytes from the file, added tokens by name, padding without bytes.
const FileVocab = struct {
    toks: []const []const u8,

    fn bytes(ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8 {
        _ = a;
        const f: *FileVocab = @ptrCast(@alignCast(ctx));
        if (id >= added_first and id < padding_first) return added_names[id - added_first];
        if (id >= f.toks.len) return null;
        return f.toks[id];
    }
    fn added(ctx: *anyopaque, id: u32) bool {
        _ = ctx;
        return id >= added_first and id < padding_first;
    }
};

pub const Fixture = struct {
    lib: llg.Library,
    vocab: llg.Vocab,
    tok: *llg.Tokenizer,
    file: []u8,
    toks: [][]const u8,

    pub fn open(a: std.mem.Allocator) !Fixture {
        const lib_path = std.c.getenv("TF_TEST_LLG_LIB") orelse return error.SkipZigTest;
        const vocab_path = std.c.getenv("TF_TEST_GLM_VOCAB") orelse return error.SkipZigTest;
        var lib = try llg.Library.openPath(std.mem.span(lib_path));
        errdefer lib.close();
        const file = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(vocab_path), a, .limited(64 << 20));
        errdefer a.free(file);
        const n = std.mem.readInt(u32, file[0..4], .little);
        const toks = try a.alloc([]const u8, n);
        errdefer a.free(toks);
        var at: usize = 4;
        for (toks) |*t| {
            const len = std.mem.readInt(u32, file[at..][0..4], .little);
            t.* = file[at + 4 ..][0..len];
            at += 4 + len;
        }
        var fv: FileVocab = .{ .toks = toks };
        var vocab = try llg.Vocab.build(a, .{ .ctx = &fv, .bytes = FileVocab.bytes, .added = FileVocab.added }, vocab_size);
        errdefer vocab.deinit(a);
        var err: [256]u8 = undefined;
        const tok = llg.newTokenizer(&lib.api, &vocab, &eos, &err) orelse {
            std.debug.print("tokenizer: {s}\n", .{std.mem.sliceTo(&err, 0)});
            return error.TokenizerFailed;
        };
        return .{ .lib = lib, .vocab = vocab, .tok = tok, .file = file, .toks = toks };
    }

    pub fn close(f: *Fixture, a: std.mem.Allocator) void {
        f.lib.api.llg_free_tokenizer(f.tok);
        f.vocab.deinit(a);
        a.free(f.toks);
        a.free(f.file);
        f.lib.close();
    }

    pub fn matcher(f: *Fixture, schema: [:0]const u8) !*llg.Matcher {
        const init = llg.constraintInit(&f.lib.api, f.tok);
        return f.lib.api.llg_new_matcher(&init, "json_schema", schema.ptr) orelse error.NoMatcher;
    }

    /// Writes `text` with the longest allowed token whose bytes come next at each step; the steps, or null at a dead
    /// end. Every mask: no added id but the ends, no padding, an end only while accepting.
    pub fn walk(f: *Fixture, m: *llg.Matcher, text: []const u8, bad: *usize) !?usize {
        var at: usize = 0;
        return f.walkTo(m, text, bad, &at);
    }

    /// `walk`, and in `at_out` how many bytes of `text` were written (where a dead end stopped it).
    pub fn walkTo(f: *Fixture, m: *llg.Matcher, text: []const u8, bad: *usize, at_out: *usize) !?usize {
        const api = &f.lib.api;
        var row: [llg.maskWords(vocab_size)]u32 = undefined;
        var at: usize = 0;
        defer at_out.* = at;
        var steps: usize = 0;
        while (at < text.len) {
            if (api.llg_matcher_compute_mask_into(m, &row, @sizeOf(@TypeOf(row))) != 0) return null;
            for (added_first..vocab_size) |i| {
                const id: llg.Token = @intCast(i);
                if (llg.allowed(&row, id) and std.mem.indexOfScalar(llg.Token, &eos, id) == null) bad.* += 1;
            }
            if (llg.allowed(&row, eos[0]) and !api.llg_matcher_is_accepting(m)) bad.* += 1;
            var best: ?u32 = null;
            var best_len: usize = 0;
            for (f.toks[0..added_first], 0..) |t, i| {
                if (t.len > best_len and t.len <= text.len - at and llg.allowed(&row, @intCast(i)) and std.mem.eql(u8, t, text[at..][0..t.len])) {
                    best = @intCast(i);
                    best_len = t.len;
                }
            }
            const id = best orelse return null;
            if (api.llg_matcher_consume_token(m, id) != 0) return null;
            at += best_len;
            steps += 1;
        }
        return steps;
    }
};

test "llguidance structs: llg_constraint_init_set_defaults' values read back through the Zig copies" {
    const a = std.testing.allocator;
    var f = try Fixture.open(a);
    defer f.close(a);
    var init: llg.ConstraintInit = undefined;
    f.lib.api.llg_constraint_init_set_defaults(&init, f.tok);
    try std.testing.expect(init.tokenizer == f.tok);
    try std.testing.expectEqual(@as(u32, 1), init.log_stderr_level);
    try std.testing.expectEqual(@as(usize, 2000), init.limits.max_items_in_row);
    try std.testing.expectEqual(@as(u64, 200_000), init.limits.step_lexer_fuel);
    try std.testing.expect(init.limits.verbose_errors);
    // the sizes the C header gives on arm64 (TokenizerInitV2 96, ConstraintInit 80, ParserLimits 56)
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(llg.TokenizerInitV2));
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(llg.ConstraintInit));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(llg.ParserLimits));
}

test "llguidance vocabulary: added tokens as 0xFF and their name, padding empty, BPE bytes as they are" {
    const a = std.testing.allocator;
    var f = try Fixture.open(a);
    defer f.close(a);
    try std.testing.expectEqual(@as(usize, vocab_size), f.vocab.lens.len);
    var at: usize = 0;
    for (f.vocab.lens, 0..) |len, i| {
        const got = f.vocab.bytes[at..][0..len];
        at += len;
        if (i >= padding_first) {
            try std.testing.expectEqual(@as(u32, 0), len);
        } else if (i >= added_first) {
            try std.testing.expectEqual(@as(u8, 0xFF), got[0]);
            try std.testing.expectEqualStrings(added_names[i - added_first], got[1..]);
        } else {
            try std.testing.expectEqualSlices(u8, f.toks[i], got);
        }
    }
    try std.testing.expectEqual(f.vocab.bytes.len, at);
}

test "llguidance matcher: json_object writes any object, refuses a root array, stops by itself, never allows added or padding ids" {
    const a = std.testing.allocator;
    var f = try Fixture.open(a);
    defer f.close(a);
    const api = &f.lib.api;
    try std.testing.expectEqual(@as(usize, 19360), blk: {
        const m = try f.matcher("{\"type\":\"object\"}");
        defer api.llg_free_matcher(m);
        break :blk api.llg_matcher_get_mask_byte_size(m);
    });
    var bad: usize = 0;
    {
        const m = try f.matcher("{\"type\":\"object\"}");
        defer api.llg_free_matcher(m);
        try std.testing.expect((try f.walk(m, "{\"a\": [1, {\"b\": \"\u{6771}\u{4eac}\"}]}", &bad)) != null);
        try std.testing.expect(api.llg_matcher_is_accepting(m));
        try std.testing.expect(api.llg_matcher_is_stopped(m));
        // the stop was never consumed: consuming one now is an error (the Host does not)
        try std.testing.expect(api.llg_matcher_consume_token(m, eos[0]) != 0);
        try std.testing.expect(api.llg_matcher_is_error(m));
    }
    {
        const m = try f.matcher("{\"type\":\"object\"}");
        defer api.llg_free_matcher(m);
        try std.testing.expect((try f.walk(m, "[1, 2]", &bad)) == null);
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "llguidance matcher: a refused schema gives its reason, an escaped declared key is refused without u" {
    const a = std.testing.allocator;
    var f = try Fixture.open(a);
    defer f.close(a);
    const api = &f.lib.api;
    {
        const m = try f.matcher("{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"uniqueItems\":true}");
        defer api.llg_free_matcher(m);
        try std.testing.expect(api.llg_matcher_is_error(m));
        const why = std.mem.span(api.llg_matcher_get_error(m) orelse "");
        try std.testing.expect(std.mem.indexOf(u8, why, "uniqueItems") != null);
    }
    var bad: usize = 0;
    const open = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}";
    const no_u = "{\"x-guidance\":{\"json_allowed_escapes\":\"nrbtf\\\\\\\"\"},\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}";
    {
        const m = try f.matcher(open);
        defer api.llg_free_matcher(m);
        try std.testing.expect((try f.walk(m, "{\"\\u0061\": \"x\"}", &bad)) != null); // the default lets it through
    }
    {
        const m = try f.matcher(no_u);
        defer api.llg_free_matcher(m);
        try std.testing.expect(!api.llg_matcher_is_error(m));
        try std.testing.expect((try f.walk(m, "{\"\\u0061\": \"x\"}", &bad)) == null);
    }
    {
        const m = try f.matcher(no_u);
        defer api.llg_free_matcher(m);
        try std.testing.expect((try f.walk(m, "{\"a\": 1, \"b\": \"x\"}", &bad)) != null);
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "llguidance matcher: rollback of three tokens gives the same mask; validate leaves the state as it was" {
    const a = std.testing.allocator;
    var f = try Fixture.open(a);
    defer f.close(a);
    const api = &f.lib.api;
    const m = try f.matcher("{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}");
    defer api.llg_free_matcher(m);
    var bad: usize = 0;
    try std.testing.expect((try f.walk(m, "{\"name\": \"", &bad)) != null);
    var before: [llg.maskWords(vocab_size)]u32 = undefined;
    var after: [llg.maskWords(vocab_size)]u32 = undefined;
    try std.testing.expectEqual(@as(i32, 0), api.llg_matcher_compute_mask_into(m, &before, @sizeOf(@TypeOf(before))));
    var ids: [3]llg.Token = undefined;
    for ("abc", 0..) |c, k| {
        for (f.toks[0..added_first], 0..) |t, i| if (t.len == 1 and t[0] == c) {
            ids[k] = @intCast(i);
            break;
        };
    }
    try std.testing.expectEqual(@as(i32, 3), api.llg_matcher_validate_tokens(m, &ids, ids.len));
    try std.testing.expectEqual(@as(i32, 0), api.llg_matcher_compute_mask_into(m, &after, @sizeOf(@TypeOf(after))));
    try std.testing.expectEqualSlices(u32, &before, &after);
    try std.testing.expectEqual(@as(i32, 0), api.llg_matcher_consume_tokens(m, &ids, ids.len));
    try std.testing.expectEqual(@as(i32, 0), api.llg_matcher_rollback(m, 3));
    try std.testing.expectEqual(@as(i32, 0), api.llg_matcher_compute_mask_into(m, &after, @sizeOf(@TypeOf(after))));
    try std.testing.expectEqualSlices(u32, &before, &after);
}
