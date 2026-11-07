// Modified by Fiore-lab from TensorFold 1.0.2: as in TensorFold at commit c126fe16, then GLM values that hold their closers (NOTICE lists the changed files).
//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;
const repairGlmKeys = tool_parse.repairGlmKeys;
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const glm_rescue = @import("glm_rescue.zig");

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", "</arg_value></tool_call>", "{\"n\":7}", "Trying." }, // a GLM value left open (#285)
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}

test "GLM calls keep a declared name with spaces, dots, colons or unicode; an undeclared one stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"Get Classes List\"}},{\"type\":\"function\",\"function\":{\"name\":\"search.web\"}},{\"type\":\"function\",\"function\":{\"name\":\"ns:lookup\"}},{\"type\":\"function\",\"function\":{\"name\":\"Caf\u{e9} Lookup\"}}]")).ok.array;
    const cases = [_][3][]const u8{
        .{ "<tool_call>Get Classes List<arg_key>class_type</arg_key><arg_value>ACTIVE</arg_value></tool_call>", "Get Classes List", "{\"class_type\":\"ACTIVE\"}" },
        .{ "<tool_call>get classes list<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>", "Get Classes List", "{\"n\":\"2\"}" },
        .{ "<tool_call>search.web<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>", "search.web", "{\"q\":\"x\"}" },
        .{ "<tool_call>ns:lookup<arg_key>q</arg_key><arg_value>y</arg_value></tool_call>", "ns:lookup", "{\"q\":\"y\"}" },
        .{ "<tool_call>Caf\u{e9} Lookup<arg_key>q</arg_key><arg_value>z</arg_value></tool_call>", "Caf\u{e9} Lookup", "{\"q\":\"z\"}" },
    };
    for (cases) |c| {
        for ([_]?usize{ null, 1 }) |max_calls| {
            const r = try parse(a, c[0], tools, max_calls);
            const f = r.calls.?[0].get("function").?;
            try std.testing.expectEqualStrings(c[1], f.get("name").?.string);
            try std.testing.expectEqualStrings(c[2], f.get("arguments").?.string);
            try std.testing.expectEqualStrings("", r.content);
        }
    }
    const open = "<tool_call>Get Classes List<arg_key>class_type</arg_key><arg_value>ACTIVE</arg_value>";
    try std.testing.expectEqualStrings("</tool_call>", try closeCall(a, open, tools));
    const other = "<tool_call>Get Other List<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>";
    const kept = try parse(a, other, tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings(other, kept.content);
    const prose = "<tool_call>two\nlines<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>";
    try std.testing.expect((try parse(a, prose, tools, null)).calls == null);
}

const dsml_prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">" ++
    "<｜DSML｜parameter name=\"q\" string=\"true\">";
const dsml_suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
const nested_calls = [_][]const u8{
    "<tool_call><function=lookup></function></tool_call>",
    "<tool_call><function=lookup></function>",
    "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>",
    "<tool_call>{\"name\":\"lookup\",\"arguments\":{}",
};

test "DSML parameters contain complete and unfinished legacy calls without consuming adjacent calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const adjacent = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":7}}</tool_call>";
    for (nested_calls) |literal| {
        const text = try std.mem.concat(a, u8, &.{ "Before.", dsml_prefix, literal, dsml_suffix, adjacent, "After." });
        for ([_]?usize{ null, 1 }) |quota| {
            const parsed = try parse(a, text, tools, quota);
            try std.testing.expectEqual(@as(usize, if (quota == null) 2 else 1), parsed.calls.?.len);
            const outer = parsed.calls.?[0].get("function").?;
            try std.testing.expectEqualStrings("lookup", outer.strField("name").?);
            const arguments = (try json.parse(a, outer.strField("arguments").?)).ok;
            try std.testing.expectEqualStrings(literal, arguments.strField("q").?);
            if (quota == null) {
                try std.testing.expectEqualStrings("{\"n\":7}", parsed.calls.?[1].get("function").?.strField("arguments").?);
            }
            try std.testing.expectEqualStrings("Before.After.", parsed.content);
            try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        }
    }
}

test "an unfinished DSML envelope never parses or autocompletes its nested legacy call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    for (nested_calls) |literal| {
        const text = try std.mem.concat(a, u8, &.{ "Before.", dsml_prefix, literal });
        for ([_]?usize{ null, 1 }) |quota| {
            const parsed = try parse(a, text, tools, quota);
            try std.testing.expect(parsed.calls == null);
            try std.testing.expectEqualStrings(text, parsed.content);
        }
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
    }
}

test "closeCall skips a closed DSML parameter's unfinished legacy opener and completes the adjacent call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const text = dsml_prefix ++ "<tool_call><function=lookup></function>" ++ dsml_suffix ++
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":7}";
    const tail = try closeCall(a, text, tools);
    try std.testing.expectEqualStrings("}</tool_call>", tail);
    const parsed = try parse(a, try std.mem.concat(a, u8, &.{ text, tail }), tools, null);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
    try std.testing.expectEqualStrings("{\"n\":7}", parsed.calls.?[1].get("function").?.strField("arguments").?);
}

test "bare JSON calls keep literal unfinished DSML markers as argument data" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const call = "{\"name\":\"lookup\",\"arguments\":{\"q\":\"<｜DSML｜tool_calls>\"}}";
    const inputs = [_][]const u8{ call, "[" ++ call ++ "]", "```json\n" ++ call ++ "\n```" };
    for (inputs) |text| for ([_]?usize{ null, 1 }) |quota| {
        const parsed = try parse(a, text, tools, quota);
        try std.testing.expectEqual(@as(usize, 1), parsed.calls.?.len);
        const arguments = (try json.parse(a, parsed.calls.?[0].get("function").?.strField("arguments").?)).ok;
        try std.testing.expectEqualStrings("<｜DSML｜tool_calls>", arguments.strField("q").?);
        try std.testing.expectEqualStrings("", parsed.content);
    };
}

test "a GLM call whose <arg_key> the model wrote as whitespace gets it back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const repaired = [_][2][]const u8{ // the block as written, with its keys put back
        .{ "read_file path</arg_key><arg_value>/x</arg_value>", "read_file <arg_key>path</arg_key><arg_value>/x</arg_value>" },
        .{ "read_file\npath</arg_key><arg_value>/x</arg_value>", "read_file\n<arg_key>path</arg_key><arg_value>/x</arg_value>" },
        .{ "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\nlimit</arg_key><arg_value>3</arg_value>", "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\n<arg_key>limit</arg_key><arg_value>3</arg_value>" },
        .{ "read_file path</arg_key><arg_value>/x</arg_value> limit</arg_key><arg_value>3</arg_value>", "read_file <arg_key>path</arg_key><arg_value>/x</arg_value> <arg_key>limit</arg_key><arg_value>3</arg_value>" },
    };
    for (repaired) |c| {
        try std.testing.expectEqualStrings(c[1], try repairGlmKeys(a, c[0]));
        try std.testing.expectEqualStrings(c[1], try repairGlmKeys(a, c[1]));
    }
    const unchanged = [_][]const u8{
        "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>", // nothing missing
        "read_file",
        "read_filepath</arg_key><arg_value>/x</arg_value>", // no space: no name left before the key
        " path</arg_key><arg_value>/x</arg_value>", // no name at all
        "{\"name\": \"run\", \"arguments\": {\"command\": \"ls\"}}",
    };
    for (unchanged) |b| try std.testing.expectEqualStrings(b, try repairGlmKeys(a, b));
}

test "a repaired GLM call parses, in both modes and once the end token closed it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"properties\":{\"filePath\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]")).ok.array;
    // before: the first came back as text, the second lost its second argument
    const cases = [_][2][]const u8{
        .{ "<tool_call>read_file filePath</arg_key><arg_value>/home/a/main.go</arg_value></tool_call>", "{\"filePath\":\"/home/a/main.go\"}" },
        .{ "<tool_call>read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value>\nlimit</arg_key><arg_value>3</arg_value></tool_call>", "{\"filePath\":\"/a\",\"limit\":3}" },
    };
    for (cases) |c| for ([_]?usize{ null, 1 }) |max_calls| {
        const r = try parse(a, c[0], tools, max_calls);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings("", r.content);
    };
    const open = "<tool_call>read_file filePath</arg_key><arg_value>/x</arg_value>";
    try std.testing.expectEqualStrings("</tool_call>", try closeCall(a, open, tools));
}

test "a GLM call the end token left open closes as #285 closed it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"read_file","parameters":{"properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}},
        \\ {"type":"function","function":{"name":"run","parameters":{"properties":{"command":{"type":"string"}}}}},
        \\ {"type":"function","function":{"name":"now","parameters":{"properties":{}}}}]
    )).ok.array;
    const closes = [_][3][]const u8{ // the reply, what closes it, the tool the last call names (tests/test_glm_tool_call_end.py)
        .{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>", "</tool_call>", "read_file" },
        .{ "<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", "</arg_value></tool_call>", "run" },
        .{ "Let me look:<tool_call>now", "</tool_call>", "now" },
        .{ "<tool_call>now</tool_call>\n<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>\n", "</tool_call>", "read_file" },
        .{ "<tool_call>read_file path</arg_key><arg_value>x", "</arg_value></tool_call>", "read_file" }, // and repaired
    };
    for (closes) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("name").?.string);
    }
    const left_open = [_][]const u8{
        "no call at all",
        "<tool_call>now</tool_call>", // closed already
        "<tool_call>launch<arg_key>x</arg_key><arg_value>1", // a tool the client did not offer
        "<tool_call>run<arg_key>comm", // a key with no value
        "<tool_call>run<arg_key>command</arg_key>", // ... or no value yet
        "<tool_call>run command=ls", // not GLM's markup
        "<tool_call>", // nothing written
        "<tool_call>now</think>The answer.", // the think block went on past it
        "<tool_call>run<arg_key>command</arg_key><arg_value>ls</arg_value> and then", // text after an argument
    };
    for (left_open) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

// GLM values that hold </arg_value> or </tool_call> (Fiore-lab, glm_rescue.zig): what changed,
// what did not, and the stream's hiding of the same replies.

const closer_tools =
    \\[{"type":"function","function":{"name":"write","parameters":{"properties":{"content":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"f","parameters":{"properties":{"a":{"type":"string"},"c":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"g","parameters":{"properties":{"b":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"lookup","parameters":{"properties":{"a":{"type":"string"},"b":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"next","parameters":{"properties":{"b":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"read_file","parameters":{"properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}},
    \\ {"type":"function","function":{"name":"call:lookup","parameters":{"properties":{"q":{"type":"string"}}}}},
    \\ {"type":"function","function":{"name":"\"f","parameters":{"properties":{"a":{"type":"string"}}}}}]
;

/// Each call as ``name args``, one per line, then the content: what a reading came to.
fn reading(a: std.mem.Allocator, r: tool_parse.Parsed) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    if (r.calls) |list| for (list) |c| {
        const f = c.get("function").?;
        try out.print(a, "{s} {s}\n", .{ f.get("name").?.string, f.get("arguments").?.string });
    };
    try out.print(a, "content={s}", .{r.content});
    return out.items;
}

test "GLM values that hold their closers read whole, in both modes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, closer_tools)).ok.array;
    const example = "<tool_call>x<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>";
    const cases = [_][3][]const u8{ // the reply, its reading with calls unlimited, with one call
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value></tool_call>", "write {\"content\":\"a</tool_call>b\"}\ncontent=", "write {\"content\":\"a</tool_call>b\"}\ncontent=" },
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>see " ++ example ++ " end</arg_value></tool_call>", "write {\"content\":\"see " ++ example ++ " end\"}\ncontent=", "write {\"content\":\"see " ++ example ++ " end\"}\ncontent=" },
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>see " ++ example ++ " and <tool_call>write<arg_key>content</arg_key><arg_value>w</arg_value></tool_call></arg_value></tool_call>", "write {\"content\":\"see " ++ example ++ " and <tool_call>write<arg_key>content</arg_key><arg_value>w</arg_value></tool_call>\"}\ncontent=", "write {\"content\":\"see " ++ example ++ " and <tool_call>write<arg_key>content</arg_key><arg_value>w</arg_value></tool_call>\"}\ncontent=" },
        // two calls; one call keeps the first, now whole (today it dropped the cut one and kept the second)
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>p</tool_call>q</arg_value></tool_call><tool_call>write<arg_key>content</arg_key><arg_value>r</arg_value></tool_call>", "write {\"content\":\"p</tool_call>q\"}\nwrite {\"content\":\"r\"}\ncontent=", "write {\"content\":\"p</tool_call>q\"}\ncontent=" },
        // the key repair no longer reaches into the value
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>A<arg_value>B</arg_value> k</arg_key><arg_value>C</arg_value>D</arg_value></tool_call>", "write {\"content\":\"A<arg_value>B</arg_value> k</arg_key><arg_value>C</arg_value>D\"}\ncontent=", "write {\"content\":\"A<arg_value>B</arg_value> k</arg_key><arg_value>C</arg_value>D\"}\ncontent=" },
        .{ "<tool_call>call:lookup<arg_key>q</arg_key><arg_value>A</tool_call>B</arg_value></tool_call>", "call:lookup {\"q\":\"A</tool_call>B\"}\ncontent=", "call:lookup {\"q\":\"A</tool_call>B\"}\ncontent=" },
        // a lone opener in one value and a lone closer in the next read as nesting
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y</arg_value></tool_call><tool_call>write<arg_key>content</arg_key><arg_value>z</arg_value>t</arg_value></tool_call>", "write {\"content\":\"x<arg_value>y</arg_value></tool_call><tool_call>write<arg_key>content</arg_key><arg_value>z</arg_value>t\"}\ncontent=", "write {\"content\":\"x<arg_value>y</arg_value></tool_call><tool_call>write<arg_key>content</arg_key><arg_value>z</arg_value>t\"}\ncontent=" },
        .{ "Done: <tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value></tool_call> ok", "write {\"content\":\"a</tool_call>b\"}\ncontent=Done:  ok", "write {\"content\":\"a</tool_call>b\"}\ncontent=Done:  ok" },
    };
    for (cases) |c| {
        try std.testing.expectEqualStrings(c[1], try reading(a, try parse(a, c[0], tools, null)));
        try std.testing.expectEqualStrings(c[2], try reading(a, try parse(a, c[0], tools, 1)));
        // the finished stream hides what the parse took as the call
        try std.testing.expectEqualStrings(try reading(a, .{ .content = (try parse(a, c[0], tools, null)).content, .calls = null }), try std.fmt.allocPrint(a, "content={s}", .{reply_text.pyStrip(try reply_text.hideToolCalls(a, c[0], true))}));
    }
}

/// Replies the re-read leaves as the first closer reads them, each with the reading the parser gave before the re-read
/// existed where it pins a case near the re-read's edges.
const kept_readings = [_][2][]const u8{
    .{ "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"<arg_key>k</arg_key><arg_value>x\"}}</tool_call><tool_call>write<arg_key>content</arg_key><arg_value>y</arg_value></tool_call>", "write {\"content\":\"<arg_key>k</arg_key><arg_value>x\"}\nwrite {\"content\":\"y\"}\ncontent=" },
    .{ "<tool_call>lookup<arg_key>a</arg_key><arg_value>x</arg_value>NOTE<arg_key>b</arg_key><arg_value>y</arg_value></tool_call>", "lookup {\"a\":\"x\",\"b\":\"y\"}\ncontent=" },
    .{ "<tool_call>lookup<arg_key>a</arg_key><arg_value>x</arg_value>NOTE</tool_call><tool_call>next<arg_key>b</arg_key><arg_value>y</arg_value></tool_call>", "lookup {\"a\":\"x\"}\nnext {\"b\":\"y\"}\ncontent=" },
    .{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>/a</arg_value> limit</arg_key><arg_value>3</arg_value></tool_call>", "read_file {\"path\":\"/a\",\"limit\":3}\ncontent=" },
    .{ "<tool_call>write<arg_key><arg_value></arg_key><arg_value>x</arg_value></tool_call>TEXT</arg_value><tool_call>write<arg_key>content</arg_key><arg_value>y</arg_value></tool_call>", "write {\"<arg_value>\":\"x\"}\nwrite {\"content\":\"y\"}\ncontent=TEXT</arg_value>" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y</arg_value></tool_call>", "write {\"content\":\"x<arg_value>y\"}\ncontent=" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y</arg_value></tool_call><tool_call>write<arg_key>content</arg_key><arg_value>z</arg_value></tool_call>", "write {\"content\":\"x<arg_value>y\"}\nwrite {\"content\":\"z\"}\ncontent=" },
    .{ "<tool_call>write content</arg_key><arg_value>A</tool_call>SECRET</arg_value></tool_call>", "write {}\ncontent=SECRET</arg_value></tool_call>" },
    .{ "<x:tool_call>write<arg_key>content</arg_key><arg_value>A<arg_value>B</arg_value>C</arg_value></x:tool_call>", "write {\"content\":\"A<arg_value>B\"}\ncontent=" },
    .{ "<|tool_call>write<arg_key>content</arg_key><arg_value>A<arg_value>B</arg_value>C</arg_value><tool_call|>", "write {\"content\":\"A<arg_value>B\"}\ncontent=" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</arg_value>b</arg_value></tool_call>", "write {\"content\":\"a\"}\ncontent=" },
    .{ "Before<tool_call>write<arg_key>content</arg_key><arg_value>A</tool_call>SECRET", "write {}\ncontent=BeforeSECRET" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>A</tool_call><｜DSML｜tool_calls><｜DSML｜invoke name=\"write\"><｜DSML｜parameter name=\"content\" string=\"true\">B</arg_value></tool_call><tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"danger\"}}</tool_call></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>", "write {}\nwrite {\"content\":\"B</arg_value></tool_call><tool_call>{\\\"name\\\":\\\"write\\\",\\\"arguments\\\":{\\\"content\\\":\\\"danger\\\"}}</tool_call>\"}\ncontent=" },
    .{ "<tool_call>f<arg_key>a</arg_key><arg_value>x<arg_value>y</arg_value><arg_key>b</arg_value><arg_key>c</arg_key><arg_value>z</arg_value></tool_call>", "f {\"a\":\"x<arg_value>y\",\"b</arg_value><arg_key>c\":\"z\"}\ncontent=" },
    .{ "<tool_call>\"f<arg_key>a</arg_key><arg_value>x\"</tool_call>y</arg_value></tool_call>", "content=<tool_call>\"f<arg_key>a</arg_key><arg_value>x\"</tool_call>y</arg_value></tool_call>" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value>stray</tool_call>", "" },
    .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value><|tool_call>call:g{b:<|\"|>1<|\"|>}<tool_call|></tool_call>", "" },
};

test "replies the re-read leaves as the first closer reads them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, closer_tools)).ok.array;
    for (kept_readings) |c| for ([_]?usize{ null, 1 }) |max| {
        const now = try reading(a, try parse(a, c[0], tools, max));
        try std.testing.expectEqualStrings(try reading(a, try tool_parse.parseWith(a, c[0], tools, max, .{ .rescue = false })), now);
        if (max == null and c[1].len > 0) try std.testing.expectEqualStrings(c[1], now);
    };
}

test "a reply the end token left inside a GLM call closes as before, and a nested value now closes too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, closer_tools)).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, its reading with calls unlimited, with one call
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y</arg_value></tool_call>", "", "write {\"content\":\"x<arg_value>y\"}\ncontent=", "write {\"content\":\"x<arg_value>y\"}\ncontent=" },
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y", "</arg_value></tool_call>", "write {\"content\":\"x<arg_value>y\"}\ncontent=", "write {\"content\":\"x<arg_value>y\"}\ncontent=" },
        // new: today the call was lost and the reply was all text
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>see <arg_value>v</arg_value> more</arg_value>", "</tool_call>", "write {\"content\":\"see <arg_value>v</arg_value> more\"}\ncontent=", "write {\"content\":\"see <arg_value>v</arg_value> more\"}\ncontent=" },
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b", "", "write {}\ncontent=b", "content=b" },
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</arg_value>literal", "", "content=<tool_call>write<arg_key>content</arg_key><arg_value>a</arg_value>literal", "content=<tool_call>write<arg_key>content</arg_key><arg_value>a</arg_value>literal" },
        .{ "<tool_call>f<arg_key>a</arg_key><arg_value>x<arg_value>y</arg_value><arg_key>b</arg_value><arg_key>c</arg_key><arg_value>z</arg_value>", "</tool_call>", "f {\"a\":\"x<arg_value>y\",\"b</arg_value><arg_key>c\":\"z\"}\ncontent=", "f {\"a\":\"x<arg_value>y\",\"b</arg_value><arg_key>c\":\"z\"}\ncontent=" },
        // the closed call's own reading holds; the whole reply reads its first call on into it
        .{ "<tool_call>f<arg_key>a</arg_key><arg_value>x<arg_value>y</arg_value></tool_call><tool_call>g<arg_key>b</arg_value><arg_key>c</arg_key><arg_value>z</arg_value>", "</tool_call>", "f {\"a\":\"x<arg_value>y</arg_value></tool_call><tool_call>g<arg_key>b\",\"c\":\"z\"}\ncontent=", "f {\"a\":\"x<arg_value>y</arg_value></tool_call><tool_call>g<arg_key>b\",\"c\":\"z\"}\ncontent=" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const whole = try std.mem.concat(a, u8, &.{ c[0], close });
        try std.testing.expectEqualStrings(c[2], try reading(a, try parse(a, whole, tools, null)));
        try std.testing.expectEqualStrings(c[3], try reading(a, try parse(a, whole, tools, 1)));
    }
}

/// Every split of ``reply``: the hidden stream never shows the call's bytes, never takes back what it showed, and
/// ends where the parse's content does; the one-call filter, fed the same bytes, ends there too (or at ``filtered``,
/// for markup it does not hide).
fn expectStream(a: std.mem.Allocator, reply: []const u8, shown: []const u8) !void {
    return expectStreams(a, reply, shown, shown);
}

fn expectStreams(a: std.mem.Allocator, reply: []const u8, shown: []const u8, filtered: []const u8) !void {
    var last: []const u8 = "";
    for (0..reply.len + 1) |n| {
        const now = try reply_text.hideToolCalls(a, reply[0..n], false);
        try std.testing.expect(std.mem.startsWith(u8, now, last));
        try std.testing.expect(std.mem.startsWith(u8, shown, now));
        last = try a.dupe(u8, now);
    }
    try std.testing.expectEqualStrings(shown, try reply_text.hideToolCalls(a, reply, true));
    for ([_]usize{ 1, 2, 3, 5, 11, 64 }) |step| {
        var f: tool_stream.TextFilter = .{};
        var out: std.ArrayList(u8) = .empty;
        var at: usize = 0;
        while (at < reply.len) : (at += step) {
            try out.appendSlice(a, try f.feed(a, reply[at..@min(reply.len, at + step)]));
            try out.appendSlice(a, try f.feed(a, ""));
            try std.testing.expect(std.mem.startsWith(u8, filtered, out.items));
        }
        try out.appendSlice(a, try f.finish(a));
        try std.testing.expectEqualStrings(filtered, out.items);
    }
}

test "the stream hides a GLM call to its real end at every split, and lets go of what it held when the reply ends" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const example = "<tool_call>x<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>";
    try expectStream(a, "Go <tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value></tool_call> on", "Go  on");
    try expectStream(a, "<tool_call>write<arg_key>content</arg_key><arg_value>see " ++ example ++ " and " ++ example ++ "</arg_value></tool_call>Tail", "Tail");
    try expectStream(a, "<tool_call>write<arg_key>content</arg_key><arg_value>A<arg_value>B</arg_value> k</arg_key><arg_value>C</arg_value>D</arg_value></tool_call>x", "x");
    // a value left open: held until the reply ends, then shown as the first closer reads it (as today)
    try expectStream(a, "Before<tool_call>write<arg_key>content</arg_key><arg_value>A</tool_call>SECRET", "BeforeSECRET");
    try expectStream(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call><tool_call>g<arg_key>b</arg_key><arg_value>y</tool_call>SECRET", "SECRET");
    try expectStream(a, "<tool_call>write<arg_key>content</arg_key><arg_value>x<arg_value>y</arg_value></tool_call>after", "after");
    try expectStream(a, "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value>stray</tool_call>c", "b</arg_value>stray</tool_call>c");
    // another family inside: the first closer's reading (the one-call filter never hid Gemma's markup)
    try expectStreams(a, "<tool_call>w<arg_key>c</arg_key><arg_value>a</tool_call><|tool_call>call:g{}<tool_call|>b</arg_value></tool_call>", "b</arg_value></tool_call>", "<|tool_call>call:g{}<tool_call|>b</arg_value></tool_call>");
}

test "a bare JSON call keeps its literal GLM text whole, while the stream hides that text's call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, closer_tools)).ok.array;
    const reply = "{\"name\":\"write\",\"arguments\":{\"content\":\"<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>SECRET</arg_value></tool_call>\"}}";
    try std.testing.expectEqualStrings("write {\"content\":\"<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>SECRET</arg_value></tool_call>\"}\ncontent=", try reading(a, try parse(a, reply, tools, null)));
    // the stream hid the text's call before as well, to its first closer
    try std.testing.expectEqualStrings("{\"name\":\"write\",\"arguments\":{\"content\":\"\"}}", try reply_text.hideToolCalls(a, reply, true));
}

/// The one-call filter fed ``head``, then ``unit`` ``n`` times a byte at a time, then ``tail``: what it showed.
fn fedBytes(a: std.mem.Allocator, head: []const u8, unit: []const u8, n: usize, tail: []const u8) ![]const u8 {
    var f: tool_stream.TextFilter = .{};
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, try f.feed(a, head));
    for (0..n) |_| for (unit) |*b| try out.appendSlice(a, try f.feed(a, b[0..1]));
    try out.appendSlice(a, try f.feed(a, tail));
    try out.appendSlice(a, try f.finish(a));
    return out.items;
}

/// The re-read's work for ``n`` and for ``2 n`` units: twice the reply, not four times the work.
fn expectLinear(a: std.mem.Allocator, head: []const u8, unit: []const u8, tail: []const u8, shown: []const u8) !void {
    var work: [2]usize = undefined;
    for ([_]usize{ 2000, 4000 }, 0..) |n, k| {
        glm_rescue.work = 0;
        try std.testing.expectEqualStrings(shown, try fedBytes(a, head, unit, n, tail));
        work[k] = glm_rescue.work;
    }
    try std.testing.expect(work[1] < 3 * work[0]);
}

test "a streamed GLM call read again goes on from where it stopped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cut_value = "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>y</arg_value>";
    // a long key, many arguments, a long tag name that could still become a namespaced one
    try expectLinear(a, cut_value ++ "<arg_key>", "k", "</arg_key><arg_value>z</arg_value></tool_call>Tail", "Tail");
    try expectLinear(a, cut_value, "<arg_key>b</arg_key><arg_value>z</arg_value>", "</tool_call>Tail", "Tail");
    try expectLinear(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call><tool_call><", "a", "></tool_call>y</arg_value></tool_call>Tail", "Tail");
    // ... and inside a key, where it ends as text or as another family's tag
    try expectLinear(a, cut_value ++ "<arg_key><", "a", "></arg_key><arg_value>z</arg_value></tool_call>Tail", "Tail");
    try expectLinear(a, cut_value ++ "<arg_key><", "a", ":tool_call></arg_key><arg_value>z</arg_value></tool_call>Tail", "y</arg_value><arg_key>");
    // calls left open, each read once: at the end of the reply (byte by byte), and in one look at a streamed reply
    try expectLinear(a, "", "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>", "SECRET", "SECRET");
    // the hide's own search for each family's opener walks the rest of the reply per call: a short reply
    var work: [2]usize = undefined;
    for ([_]usize{ 250, 500 }, 0..) |n, k| {
        var reply: std.ArrayList(u8) = .empty;
        for (0..n) |_| try reply.appendSlice(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>");
        try reply.appendSlice(a, "<|tool_call>");
        glm_rescue.work = 0;
        try std.testing.expectEqualStrings("", try reply_text.hideToolCalls(a, reply.items, false));
        work[k] = glm_rescue.work;
    }
    try std.testing.expect(work[1] < 3 * work[0]);
}

test "the stream lets go of text as soon as a GLM call cannot read whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const deltas = [_][]const u8{ "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>", "y</arg_value>X", "Y" };
    const shown = [_][]const u8{ "", "y</arg_value>X", "y</arg_value>XY" };
    var f: tool_stream.TextFilter = .{};
    var reply: std.ArrayList(u8) = .empty;
    var out: std.ArrayList(u8) = .empty;
    for (deltas, shown) |d, want| {
        try reply.appendSlice(a, d);
        try out.appendSlice(a, try f.feed(a, d));
        try std.testing.expectEqualStrings(want, out.items);
        try std.testing.expectEqualStrings(want, try reply_text.hideToolCalls(a, reply.items, false));
    }
    try std.testing.expectEqualStrings("", try f.finish(a));
    // a reply that ends inside a tag, after a call left open
    try expectStream(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call><a", "<a");
    try expectStream(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>y</arg_val", "y</arg_val");
    try expectStreams(a, "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call><a:tool_call>", "<a:tool_call>", "");
}

test "a GLM call that does not read again leaves the next call to read on its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, closer_tools)).ok.array;
    const reply = "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>y</arg_value>BAD</tool_call><tool_call>g<arg_key>b</arg_key><arg_value>u</tool_call>v</arg_value></tool_call>";
    try std.testing.expectEqualStrings("f {}\ng {\"b\":\"u</tool_call>v\"}\ncontent=y</arg_value>BAD</tool_call>", try reading(a, try parse(a, reply, tools, null)));
    try std.testing.expectEqualStrings("g {\"b\":\"u</tool_call>v\"}\ncontent=y</arg_value>BAD</tool_call>", try reading(a, try parse(a, reply, tools, 1)));
}

test "GLM calls written by the template read back as written" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a,
        \\[{"type":"function","function":{"name":"write","parameters":{"properties":{"content":{"type":"string"},"path":{"type":"string"}}}}},
        \\ {"type":"function","function":{"name":"run","parameters":{"properties":{"command":{"type":"string"}}}}}]
    )).ok.array;
    var prng: std.Random.DefaultPrng = .init(0x7c105e);
    const r = prng.random();
    const plain = [_][]const u8{ "a", "b c", "\n", "日本語", "{\"x\":1}", "<b>", "x < y > z", "  " };
    for (0..400) |_| {
        var reply: std.ArrayList(u8) = .empty;
        var prose: std.ArrayList(u8) = .empty;
        var want: std.ArrayList(u8) = .empty;
        const n_calls = r.intRangeAtMost(usize, 1, 3);
        for (0..n_calls) |_| {
            const words = [_][]const u8{ "Next.", "ok ", "結果 " };
            const said = words[r.uintLessThan(usize, words.len)];
            try reply.appendSlice(a, said);
            try prose.appendSlice(a, said);
            const name: []const u8 = if (r.boolean()) "write" else "run";
            try reply.print(a, "<tool_call>{s}", .{name});
            var args = try json.newObject(a);
            const keys: []const []const u8 = if (std.mem.eql(u8, name, "write")) &.{ "path", "content" } else &.{"command"};
            for (keys) |k| {
                var value: std.ArrayList(u8) = .empty;
                try valueOf(a, r, &value, plain[0..], 2);
                try reply.print(a, "<arg_key>{s}</arg_key><arg_value>{s}</arg_value>", .{ k, value.items });
                try args.put(a, k, .{ .string = value.items });
            }
            try reply.appendSlice(a, "</tool_call>");
            try want.print(a, "{s} {s}\n", .{ name, try json.stringify(a, .{ .object = args }, .{ .ascii = false, .compact = true }) });
        }
        try want.print(a, "content={s}", .{reply_text.pyStrip(prose.items)});
        try std.testing.expectEqualStrings(want.items, try reading(a, try parse(a, reply.items, tools, null)));
    }
}

/// A value of balanced pieces: text, ``</tool_call>``, nested values and example calls (key tags come only inside an
/// example call, so no key swallows a nested value's tail).
fn valueOf(a: std.mem.Allocator, r: std.Random, out: *std.ArrayList(u8), plain: []const []const u8, depth: usize) !void {
    for (0..r.intRangeAtMost(usize, 0, 4)) |_| switch (r.uintLessThan(u8, if (depth == 0) 2 else 4)) {
        0 => try out.appendSlice(a, plain[r.uintLessThan(usize, plain.len)]),
        1 => try out.appendSlice(a, "</tool_call>"),
        2 => {
            try out.appendSlice(a, "<arg_value>");
            try valueOf(a, r, out, plain, depth - 1);
            try out.appendSlice(a, "</arg_value>");
        },
        else => {
            try out.appendSlice(a, "<tool_call>x<arg_key>k</arg_key><arg_value>");
            try valueOf(a, r, out, plain, depth - 1);
            try out.appendSlice(a, "</arg_value></tool_call>");
        },
    };
}

test "a block that opens like DSML but is shorter than its opener and closer is no DSML call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{}}}}]")).ok.array;
    const open = "<｜DSML｜tool_calls>";
    const close = "</｜DSML｜tool_calls>";
    for ([_][]const u8{ open, open ++ "x", open ++ close[0 .. close.len - 1] }) |payload| for ([_][]const u8{ "<tool_call>", "<|tool_call>", "<x:tool_call>" }, [_][]const u8{ "</tool_call>", "<tool_call|>", "</x:tool_call>" }) |o, c| {
        const reply = try std.mem.concat(a, u8, &.{ "Hi ", o, payload, c });
        // calls unlimited keep the block as text, one call drops it, as any block that is not a call
        const all = try parse(a, reply, tools, null);
        try std.testing.expect(all.calls == null);
        try std.testing.expectEqualStrings(reply, all.content);
        const one = try parse(a, reply, tools, 1);
        try std.testing.expect(one.calls == null);
        try std.testing.expectEqualStrings("Hi", one.content);
    };
}

