//! Fiore-lab: the structured host's compile/fill/accept against the optional real library and vocabulary, using
//! llg_test's fixture.
//! Fiore-lab: llg_test の fixture で、任意の実 library・語彙に対して構造化出力の host の compile/fill/accept を試す。
const std = @import("std");
const host = @import("host.zig");
const fixture = @import("llg_test.zig");

fn walk(f: *fixture.Fixture, state: *host.State, text: []const u8) !bool {
    var row: [host.words]u32 = undefined;
    var at: usize = 0;
    while (at < text.len) {
        if (!state.fill(&row)) return false;
        for (fixture.padding_first..fixture.vocab_size) |id| try std.testing.expect(!host.llg.allowed(&row, @intCast(id)));
        var best: ?u32 = null;
        var n: usize = 0;
        for (f.toks[0..fixture.added_first], 0..) |tok, id| {
            if (tok.len > n and tok.len <= text.len - at and std.mem.eql(u8, tok, text[at..][0..tok.len]) and host.llg.allowed(&row, @intCast(id))) {
                best = @intCast(id);
                n = tok.len;
            }
        }
        const id = best orelse return false;
        try std.testing.expectEqual(host.State.Accept.token, state.accept(id));
        at += n;
    }
    return true;
}

test "structured host: real compile, refusal, object fallback, server schema, and accepting EOS" {
    const a = std.testing.allocator;
    var f = try fixture.Fixture.open(a);
    defer f.close(a);
    var eos = fixture.eos;
    // Borrowed for this test; Fixture owns the library and tokenizer.
    // この試験の間だけ借りる。library と tokenizer は Fixture が持つ。
    var runtime: host.Runtime = .{ .library = f.lib, .tokenizer = f.tok, .eos = &eos, .real_size = fixture.padding_first };
    {
        const bad = try runtime.compile(a, "{\"type\":\"array\",\"uniqueItems\":true}");
        switch (bad) {
            .ok => |m| {
                f.lib.api.llg_free_matcher(m);
                return error.ExpectedRefusal;
            },
            .refused => |why| {
                defer a.free(why);
                try std.testing.expect(why.len > 0);
                try std.testing.expect(std.mem.indexOfAny(u8, why, "\r\n") == null);
            },
        }
    }
    const schemas = [_][]const u8{ host.object_schema, "{\"type\":\"object\",\"properties\":{\"answer\":{\"const\":42}},\"required\":[\"answer\"],\"additionalProperties\":false}" };
    const answers = [_][]const u8{ "{\"\u{6771}\u{4eac}\": [1, true, null]}", "{\"answer\":42}" };
    for (schemas, answers) |schema, answer| {
        const compiled = try runtime.compile(a, schema);
        const m = switch (compiled) {
            .ok => |m| m,
            .refused => |why| {
                defer a.free(why);
                std.debug.print("{s}\n", .{why});
                return error.UnexpectedRefusal;
            },
        };
        defer f.lib.api.llg_free_matcher(m);
        var state = runtime.state(m);
        try std.testing.expect(try walk(&f, &state, answer));
        var row: [host.words]u32 = undefined;
        try std.testing.expect(state.fill(&row));
        for (eos) |id| try std.testing.expect(host.llg.allowed(&row, id));
        try std.testing.expectEqual(host.State.Accept.done, state.accept(eos[0]));
        try std.testing.expect(!f.lib.api.llg_matcher_is_error(m));
    }
}

/// The fixture's table as the server's tokenizer would give it: ids below `size` only, an added token's name without
/// llguidance's 0xFF mark.
const ServerVocab = struct {
    v: *const host.llg.Vocab,
    starts: []usize,

    fn bytes(ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8 {
        _ = a;
        const s: *ServerVocab = @ptrCast(@alignCast(ctx));
        const got = s.v.bytes[s.starts[id]..][0..s.v.lens[id]];
        if (got.len == 0) return null;
        return if (got[0] == 0xFF) got[1..] else got;
    }
    fn added(ctx: *anyopaque, id: u32) bool {
        const s: *ServerVocab = @ptrCast(@alignCast(ctx));
        return s.v.lens[id] > 0 and s.v.bytes[s.starts[id]] == 0xFF;
    }
};

test "structured host: Runtime.init takes the server tokenizer's 154,856 ids and gives llguidance the model's 154,880, padding never allowed" {
    const a = std.testing.allocator;
    var f = try fixture.Fixture.open(a);
    defer f.close(a);
    const starts = try a.alloc(usize, f.vocab.lens.len);
    defer a.free(starts);
    var at: usize = 0;
    for (f.vocab.lens, 0..) |len, i| {
        starts[i] = at;
        at += len;
    }
    var sv: ServerVocab = .{ .v = &f.vocab, .starts = starts };
    // the library is the fixture's (borrowed): the runtime's tokenizer and ends are freed here, not with deinit
    // library は fixture のもの (借りる): runtime の tokenizer と終わりの token は deinit でなくここで放す
    var runtime: host.Runtime = .{ .library = f.lib };
    defer {
        if (runtime.tokenizer) |t| f.lib.api.llg_free_tokenizer(t);
        a.free(runtime.eos);
    }
    try std.testing.expectError(error.Unavailable, runtime.init(a, .{ .ctx = &sv, .bytes = ServerVocab.bytes, .added = ServerVocab.added }, fixture.vocab_size + 1, &fixture.eos));
    try runtime.init(a, .{ .ctx = &sv, .bytes = ServerVocab.bytes, .added = ServerVocab.added }, fixture.padding_first, &fixture.eos);
    try std.testing.expectEqual(fixture.padding_first, runtime.real_size);
    const compiled = try runtime.compile(a, host.object_schema);
    const m = switch (compiled) {
        .ok => |m| m,
        .refused => |why| {
            defer a.free(why);
            std.debug.print("{s}\n", .{why});
            return error.UnexpectedRefusal;
        },
    };
    defer f.lib.api.llg_free_matcher(m);
    try std.testing.expectEqual(@as(usize, host.bytes), f.lib.api.llg_matcher_get_mask_byte_size(m));
    var state = runtime.state(m);
    try std.testing.expect(try walk(&f, &state, "{\"a\": [1, {\"b\": \"\u{6771}\u{4eac}\"}]}"));
}
