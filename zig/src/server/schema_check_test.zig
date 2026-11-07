//! Fiore-lab: the schema acceptance check (schema_check.zig) rule by rule: accepted and refused examples, the JSON
//! Pointer the refusal names, and the rewritten text (the client's keys and order kept, x-guidance first, objects with
//! properties closed).
//! Fiore-lab: schema の受付の検査 (schema_check.zig) を決まりごとに: 受ける例と断る例、断る文の JSON Pointer、書き換えた文
//! (クライアントのキーと順はそのまま、先頭に x-guidance、properties のある object は閉じる)。

const std = @import("std");
const schema_check = @import("schema_check.zig");
const json = @import("json.zig");

const head = "{\"x-guidance\":" ++ schema_check.x_guidance;

/// The text llguidance would get for `schema` (JSON text), failing the test when the check refuses it.
fn accepted(a: std.mem.Allocator, schema: []const u8) ![:0]const u8 {
    const v = (try json.parse(a, schema)).ok;
    switch (try schema_check.check(a, v)) {
        .text => |t| return t,
        .refused => |r| {
            std.debug.print("refused {s}: {s}\n", .{ schema, r.message });
            return error.TestUnexpectedResult;
        },
    }
}

/// Fails the test unless the check refuses `schema` at `pointer`, with a message naming the place and `word`.
fn refused(a: std.mem.Allocator, schema: []const u8, pointer: []const u8, word: []const u8) !void {
    const v = (try json.parse(a, schema)).ok;
    switch (try schema_check.check(a, v)) {
        .text => |t| {
            std.debug.print("accepted {s} as {s}\n", .{ schema, t });
            return error.TestUnexpectedResult;
        },
        .refused => |r| {
            errdefer std.debug.print("{s}: pointer {s}, message {s}\n", .{ schema, r.pointer, r.message });
            try std.testing.expectEqualStrings(pointer, r.pointer);
            try std.testing.expect(std.mem.indexOf(u8, r.message, pointer) != null);
            try std.testing.expect(std.mem.indexOf(u8, r.message, word) != null);
        },
    }
}

/// `body` as llguidance gets it: the server's x-guidance, then the client's keys.
fn with(comptime body: []const u8) []const u8 {
    return if (body.len == 0) head ++ "}" else head ++ "," ++ body ++ "}";
}

test "schema_check root: an object or true (as {}); false, arrays, numbers, strings and null refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings(with("\"type\":\"integer\""), try accepted(a, "{\"type\": \"integer\"}"));
    try std.testing.expectEqualStrings(with(""), try accepted(a, "true"));
    try std.testing.expectEqualStrings(with(""), try accepted(a, "{}"));
    try refused(a, "false", "", "not false");
    try refused(a, "[]", "", "not an array");
    try refused(a, "1", "", "not a number");
    try refused(a, "\"x\"", "", "not a string");
    try refused(a, "null", "", "not null");
}

test "schema_check x-guidance: the client's at the root refused, the server's put first; one below the root is llguidance's to ignore" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "{\"x-guidance\": {\"lenient\": true}, \"type\": \"array\"}", "/x-guidance", "x-guidance at /x-guidance");
    try refused(a, "{\"type\": \"string\", \"x-guidance\": {}}", "/x-guidance", "server sets it");
    try std.testing.expectEqualStrings(with("\"properties\":{\"a\":{\"x-guidance\":{\"lenient\":true}}},\"additionalProperties\":false"), try accepted(a, "{\"properties\": {\"a\": {\"x-guidance\": {\"lenient\": true}}}}"));
    // every field written, u left out of the escapes, the whitespace bounded
    const x = (try json.parse(a, schema_check.x_guidance)).ok;
    try std.testing.expectEqual(@as(usize, 7), x.object.count());
    try std.testing.expectEqualStrings("nrbtf\\\"", x.strField("json_allowed_escapes").?);
    try std.testing.expectEqualStrings("[\\x20\\x09]{1,64}|\\x0D?\\x0A[\\x20\\x09]{0,64}", x.strField("whitespace_pattern").?);
    try std.testing.expect(x.get("lenient").?.bool == false and x.get("coerce_one_of").?.bool == false and x.get("json_allow_general_unicode_escapes").?.bool == false);
    try std.testing.expectEqualStrings(",", x.strField("item_separator").?);
    try std.testing.expectEqualStrings(":", x.strField("key_separator").?);
    // json_object: {"type": "object"} with the same x-guidance
    try std.testing.expectEqualStrings(with("\"type\":\"object\""), schema_check.json_object_text);
    try std.testing.expectEqualStrings(try accepted(a, "{\"type\": \"object\"}"), schema_check.json_object_text);
}

test "schema_check size: 64 KiB of compact JSON taken, a byte more refused (the request's own spaces not counted)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const frame = "{\"type\":\"integer\",\"description\":\"\"}".len;
    for ([_]usize{ schema_check.max_bytes - 1, schema_check.max_bytes, schema_check.max_bytes + 1 }) |n| {
        const pad = try a.alloc(u8, n - frame);
        @memset(pad, 'x');
        const spaced = try std.fmt.allocPrint(a, "{{ \"type\" : \"integer\" , \"description\" : \"{s}\" }}", .{pad});
        if (n <= schema_check.max_bytes) {
            _ = try accepted(a, spaced);
        } else try refused(a, spaced, "", "65537 bytes");
    }
}

test "schema_check nesting: 32 levels of objects and arrays taken, 33 refused at the 33rd, data included" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 31, 32, 33 }) |levels| {
        var text: std.ArrayList(u8) = .empty;
        for (0..levels - 1) |_| try text.appendSlice(a, "{\"items\":");
        try text.appendSlice(a, "{}");
        for (0..levels - 1) |_| try text.append(a, '}');
        var at: std.ArrayList(u8) = .empty;
        for (0..levels - 1) |_| try at.appendSlice(a, "/items");
        if (levels <= schema_check.max_depth) {
            _ = try accepted(a, text.items);
        } else try refused(a, text.items, at.items, "deeper than 32");
    }
    // a const's data counts: the root, const's array and 31 more arrays inside it is 33
    var deep: std.ArrayList(u8) = .empty;
    try deep.appendSlice(a, "{\"const\":");
    for (0..32) |_| try deep.append(a, '[');
    for (0..32) |_| try deep.append(a, ']');
    try deep.append(a, '}');
    var deep_at: std.ArrayList(u8) = .empty;
    try deep_at.appendSlice(a, "/const");
    for (0..31) |_| try deep_at.appendSlice(a, "/0");
    try refused(a, deep.items, deep_at.items, "deeper than 32");
}

test "schema_check text: a lone surrogate in a key or a string refused, a pair taken; NUL written as \\u0000" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "{\"properties\": {\"\\ud800\": {}}}", "/properties/\xed\xa0\x80", "lone surrogate");
    try refused(a, "{\"enum\": [\"a\", \"\\udc00\"]}", "/enum/1", "lone surrogate");
    try refused(a, "{\"description\": \"\\ud83d\"}", "/description", "lone surrogate");
    const pair = try accepted(a, "{\"const\": \"\\ud83d\\ude00\"}");
    try std.testing.expectEqualStrings(with("\"const\":\"\xf0\x9f\x98\x80\""), pair);
    const nul = try accepted(a, "{\"properties\": {\"a\\u0000b\": {\"const\": \"\\u0000\"}}, \"additionalProperties\": false}");
    try std.testing.expectEqualStrings(with("\"properties\":{\"a\\u0000b\":{\"const\":\"\\u0000\"}},\"additionalProperties\":false"), nul);
    try std.testing.expect(std.mem.indexOfScalar(u8, nul, 0) == null);
    try std.testing.expectEqual(@as(u8, 0), nul[nul.len]);
    try std.testing.expect((try json.parse(a, nul)) == .ok);
}

test "schema_check $schema: absent or 2020-12 (a trailing # too), anywhere; another draft or a non-string refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try accepted(a, "{\"$schema\": \"https://json-schema.org/draft/2020-12/schema\", \"type\": \"array\"}");
    _ = try accepted(a, "{\"$schema\": \"https://json-schema.org/draft/2020-12/schema#\"}");
    try refused(a, "{\"$schema\": \"http://json-schema.org/draft-07/schema#\"}", "/$schema", "$schema at /$schema");
    try refused(a, "{\"$schema\": \"https://json-schema.org/draft/2019-09/schema\"}", "/$schema", "2020-12");
    try refused(a, "{\"properties\": {\"a\": {\"$schema\": \"http://json-schema.org/draft-04/schema#\"}}}", "/properties/a/$schema", "only");
    try refused(a, "{\"$schema\": 2020}", "/$schema", "only");
}

test "schema_check arrays: additionalItems and an array of items refused wherever they are; prefixItems and a schema in items taken" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // additionalItems true beside prefixItems would let [2] through
    try refused(a, "{\"$schema\": \"https://json-schema.org/draft/2020-12/schema\", \"type\": \"array\", \"prefixItems\": [{\"const\": 1}], \"additionalItems\": true}", "/additionalItems", "additionalItems at /additionalItems");
    try refused(a, "{\"items\": {\"additionalItems\": false}}", "/items/additionalItems", "prefixItems");
    try refused(a, "{\"items\": [{\"type\": \"integer\"}]}", "/items", "items at /items");
    try refused(a, "{\"anyOf\": [{\"type\": \"null\"}, {\"items\": [true]}]}", "/anyOf/1/items", "is an array");
    _ = try accepted(a, "{\"type\": \"array\", \"prefixItems\": [{\"const\": 1}], \"items\": {\"type\": \"string\"}}");
    _ = try accepted(a, "{\"type\": \"array\", \"items\": false}");
}

test "schema_check $ref: \"#\", #/$defs/<name> and #/definitions/<name> (~0, ~1) only; other places, anchors, percent and missing names refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try accepted(a, "{\"$defs\": {\"a\": {\"type\": \"integer\"}}, \"$ref\": \"#/$defs/a\"}");
    _ = try accepted(a, "{\"definitions\": {\"a\": true}, \"properties\": {\"x\": {\"$ref\": \"#/definitions/a\"}}}");
    _ = try accepted(a, "{\"$defs\": {\"a/b\": {\"const\": 1}, \"a~b\": {\"const\": 2}}, \"anyOf\": [{\"$ref\": \"#/$defs/a~1b\"}, {\"$ref\": \"#/$defs/a~0b\"}]}");
    _ = try accepted(a, "{\"anyOf\": [{\"type\": \"null\"}, {\"type\": \"array\", \"items\": {\"$ref\": \"#\"}}]}");
    // cycles end: the walk visits a node once
    _ = try accepted(a, "{\"$ref\": \"#\"}");
    _ = try accepted(a, "{\"$defs\": {\"a\": {\"$ref\": \"#/$defs/b\"}, \"b\": {\"$ref\": \"#/$defs/a\"}}, \"$ref\": \"#/$defs/a\"}");
    // a $ref into default, which held a minProperties on an open object
    try refused(a, "{\"$ref\": \"#/default\", \"default\": {\"type\": \"object\", \"minProperties\": 2}}", "/$ref", "not \"#/default\"");
    try refused(a, "{\"properties\": {\"a\": {\"type\": \"integer\"}, \"b\": {\"$ref\": \"#/properties/a\"}}}", "/properties/b/$ref", "#/$defs/<name>");
    try refused(a, "{\"$ref\": \"#missing\"}", "/$ref", "not \"#missing\"");
    try refused(a, "{\"$defs\": {\"T\": {\"const\": 1}}, \"$ref\": \"#%2F$defs%2FT\"}", "/$ref", "must be");
    try refused(a, "{\"$defs\": {\"a b\": {}}, \"$ref\": \"#/$defs/a%20b\"}", "/$ref", "must be");
    try refused(a, "{\"$defs\": {\"a\": {\"properties\": {\"b\": {}}}}, \"$ref\": \"#/$defs/a/properties/b\"}", "/$ref", "must be");
    try refused(a, "{\"$defs\": {\"a~x\": {}}, \"$ref\": \"#/$defs/a~x\"}", "/$ref", "must be");
    try refused(a, "{\"$ref\": \"other.json#/$defs/a\"}", "/$ref", "must be");
    try refused(a, "{\"$ref\": \"urn:example:a\"}", "/$ref", "must be");
    try refused(a, "{\"$ref\": \"\"}", "/$ref", "must be");
    try refused(a, "{\"$ref\": \"#/$defs/\"}", "/$ref", "must be");
    try refused(a, "{\"$ref\": 1}", "/$ref", "must be a string");
    try refused(a, "{\"$defs\": {\"a\": {}}, \"items\": {\"$ref\": \"#/$defs/b\"}}", "/items/$ref", "names #/$defs/b, which the schema does not have");
    try refused(a, "{\"$ref\": \"#/definitions/a\", \"$defs\": {\"a\": {}}}", "/$ref", "does not have");
    // a $ref resolves against the nearest $id: "#/$defs/a" under {"$id": ...} is that node's $defs, not the root's
    _ = try accepted(a, "{\"properties\": {\"p\": {\"$id\": \"http://example.com/p\", \"$defs\": {\"a\": {}}, \"$ref\": \"#/$defs/a\"}}}");
    try refused(a, "{\"$defs\": {\"a\": {}}, \"properties\": {\"p\": {\"$id\": \"http://example.com/p\", \"$ref\": \"#/$defs/a\"}}}", "/properties/p/$ref", "does not have");
}

test "schema_check integers past 2^53: refused in the bounds, multipleOf, const and enum (by their decimal text), not in default or examples" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try accepted(a, "{\"type\": \"integer\", \"minimum\": 9007199254740992, \"maximum\": -9007199254740992}");
    // 9007199254740993 rounded to 9007199254740992 let 9007199254740992 through
    try refused(a, "{\"type\": \"integer\", \"minimum\": 9007199254740993}", "/minimum", "9007199254740993 is past 2^53");
    try refused(a, "{\"maximum\": -9007199254740993}", "/maximum", "past 2^53");
    try refused(a, "{\"exclusiveMinimum\": 100000000000000000000}", "/exclusiveMinimum", "past 2^53");
    try refused(a, "{\"properties\": {\"n\": {\"exclusiveMaximum\": 18446744073709551616}}}", "/properties/n/exclusiveMaximum", "past 2^53");
    try refused(a, "{\"const\": 9007199254740993}", "/const", "const at /const");
    try refused(a, "{\"enum\": [1, {\"a\": [2, -9007199254740993]}]}", "/enum/1/a/1", "enum at /enum/1/a/1");
    try refused(a, "{\"const\": {\"minimum\": 9007199254740993}}", "/const/minimum", "const at");
    // doubles stay doubles: a decimal bound is compared as llguidance compares it
    _ = try accepted(a, "{\"minimum\": 1e300, \"const\": 9007199254740993.0}");
    // data that is not const or enum is not checked (an unreferenced default too)
    _ = try accepted(a, "{\"type\": \"integer\", \"default\": 9007199254740993, \"examples\": [9007199254740993]}");
    _ = try accepted(a, "{\"type\": \"integer\", \"default\": {\"type\": \"object\", \"minProperties\": 2}}");
}

test "schema_check multipleOf alone: an integer from 1 to 65536, or a decimal llguidance reads as one at 8 places or fewer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // llguidance's doubles: 0.01 is 1 x 10^-2, 6.5536 is 65536 x 10^-4, 0.00065536 is 65536 x 10^-8, 65536.0 is 65536
    for ([_][]const u8{ "1", "65536", "0.01", "0.5", "0.25", "1.5", "2.0", "65536.0", "6.5536", "655.36", "0.00065536", "1e-8", "3e-8" }) |m|
        _ = try accepted(a, try std.fmt.allocPrint(a, "{{\"type\": \"number\", \"multipleOf\": {s}}}", .{m}));
    for ([_][]const u8{ "65537", "2147483647", "2147483648", "9007199254740993", "0", "-3", "0.0", "-0.5", "100000.0", "1e300", "0.07", "1.23456789", "0.123456789", "1e-9", "7e-9", "3e-10", "\"5\"", "null", "true" }) |m|
        try refused(a, try std.fmt.allocPrint(a, "{{\"type\": \"number\", \"multipleOf\": {s}}}", .{m}), "/multipleOf", "from 1 to 65536");
    // the message says how llguidance reads a decimal: 0.07 times ten in doubles first has no fraction at 7000000000000001
    const reads = [_]struct { []const u8, []const u8 }{
        .{ "0.07", "reads 0.07 as 7000000000000001 x 10^-17" },
        .{ "1.23456789", "reads 1.23456789 as 12345678899999998 x 10^-16" },
        .{ "3e-10", "reads 3e-10 as 3 x 10^-10" },
        .{ "100000.0", "reads 100000.0 as 100000" },
        .{ "65537", "not 65537" },
    };
    for (reads) |r| try refused(a, try std.fmt.allocPrint(a, "{{\"multipleOf\": {s}}}", .{r[0]}), "/multipleOf", r[1]);
}

test "schema_check NaN and Infinity: the server's parser takes them, the check refuses them anywhere (not JSON)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "{\"type\": \"number\", \"minimum\": NaN}", "/minimum", "is NaN");
    try refused(a, "{\"multipleOf\": Infinity}", "/multipleOf", "is Infinity");
    try refused(a, "{\"default\": [1, -Infinity]}", "/default/1", "is -Infinity");
}

test "schema_check multipleOf twice: two that can land on one value refused, through allOf, anyOf, $ref, patternProperties and items" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // lcm(65537, 65537) wrapped in u32 to 1; 65536 twice wraps too (65536 x 65536 is 2^32)
    try refused(a, "{\"allOf\": [{\"type\": \"integer\", \"multipleOf\": 65536}, {\"multipleOf\": 65536}]}", "/allOf/1/multipleOf", "same value as /allOf/0/multipleOf");
    try refused(a, "{\"multipleOf\": 3, \"allOf\": [{\"multipleOf\": 5}]}", "/allOf/0/multipleOf", "same value as /multipleOf");
    try refused(a, "{\"multipleOf\": 3, \"$ref\": \"#/$defs/m\", \"$defs\": {\"m\": {\"multipleOf\": 5}}}", "/$defs/m/multipleOf", "same value as /multipleOf");
    try refused(a, "{\"allOf\": [{\"$ref\": \"#/$defs/m\"}, {\"$ref\": \"#/$defs/m\"}], \"$defs\": {\"m\": {\"multipleOf\": 65536}}}", "/$defs/m/multipleOf", "same value as /$defs/m/multipleOf");
    try refused(a, "{\"anyOf\": [{\"multipleOf\": 2}, {\"multipleOf\": 3}]}", "/anyOf/1/multipleOf", "same value");
    try refused(a, "{\"properties\": {\"n_a\": {\"multipleOf\": 2}}, \"patternProperties\": {\"^n_\": {\"multipleOf\": 3}}}", "/patternProperties/^n_/multipleOf", "same value as /properties/n_a/multipleOf");
    try refused(a, "{\"prefixItems\": [{\"multipleOf\": 2}], \"items\": {\"multipleOf\": 3}}", "/items/multipleOf", "same value");
    try refused(a, "{\"allOf\": [{\"properties\": {\"a\": {\"multipleOf\": 2}}}, {\"properties\": {\"a\": {\"multipleOf\": 3}}}]}", "/allOf/1/properties/a/multipleOf", "same value");
    // different values: taken, $defs reached from two properties included
    _ = try accepted(a, "{\"properties\": {\"price\": {\"multipleOf\": 0.01}, \"qty\": {\"multipleOf\": 1}, \"tags\": {\"items\": {\"multipleOf\": 2}}}}");
    _ = try accepted(a, "{\"properties\": {\"price\": {\"$ref\": \"#/$defs/money\"}, \"tax\": {\"$ref\": \"#/$defs/money\"}}, \"$defs\": {\"money\": {\"multipleOf\": 0.01}}}");
    _ = try accepted(a, "{\"prefixItems\": [{\"multipleOf\": 2}, {\"multipleOf\": 3}]}");
    _ = try accepted(a, "{\"multipleOf\": 7, \"$defs\": {\"unused\": {\"multipleOf\": 5}}}");
    // a $ref cycle repeats what it reaches: one multipleOf in it is taken, a second one anywhere is not
    _ = try accepted(a, "{\"$defs\": {\"node\": {\"properties\": {\"v\": {\"multipleOf\": 0.5}, \"kids\": {\"items\": {\"$ref\": \"#/$defs/node\"}}}}}, \"$ref\": \"#/$defs/node\"}");
    try refused(a, "{\"$defs\": {\"node\": {\"properties\": {\"v\": {\"multipleOf\": 0.5}, \"w\": {\"multipleOf\": 3}, \"kids\": {\"items\": {\"$ref\": \"#/$defs/node\"}}}}}, \"$ref\": \"#/$defs/node\"}", "/$defs/node/properties/v/multipleOf", "same value as /$defs/node/properties/w/multipleOf");
    _ = try accepted(a, "{\"properties\": {\"n\": {\"multipleOf\": 2}, \"more\": {\"$ref\": \"#\"}}}");
    try refused(a, "{\"properties\": {\"n\": {\"multipleOf\": 2}, \"m\": {\"multipleOf\": 3}, \"more\": {\"$ref\": \"#\"}}}", "/properties/n/multipleOf", "same value as /properties/m/multipleOf");
}

test "schema_check nullable: refused in any schema node, not as a property name or in data" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refused(a, "{\"type\": \"string\", \"nullable\": true}", "/nullable", "nullable at /nullable");
    try refused(a, "{\"anyOf\": [{\"type\": \"integer\", \"nullable\": false}]}", "/anyOf/0/nullable", "\"null\"");
    try refused(a, "{\"$defs\": {\"a\": {\"nullable\": true}}}", "/$defs/a/nullable", "OpenAPI");
    try refused(a, "{\"properties\": {\"a\": {\"additionalProperties\": {\"nullable\": true}}}}", "/properties/a/additionalProperties/nullable", "nullable");
    _ = try accepted(a, "{\"type\": \"object\", \"properties\": {\"nullable\": {\"type\": \"boolean\"}}, \"required\": [\"nullable\"]}");
    _ = try accepted(a, "{\"const\": {\"nullable\": true}, \"default\": {\"nullable\": true}, \"examples\": [{\"nullable\": true}]}");
}

test "schema_check minProperties and maxProperties: only on a closed object (the server closes one with properties)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try accepted(a, "{\"type\": \"object\", \"properties\": {\"a\": {}, \"b\": {}}, \"additionalProperties\": false, \"minProperties\": 1, \"maxProperties\": 2}");
    try std.testing.expectEqualStrings(
        with("\"type\":\"object\",\"properties\":{\"a\":{}},\"minProperties\":1,\"additionalProperties\":false"),
        try accepted(a, "{\"type\": \"object\", \"properties\": {\"a\": {}}, \"minProperties\": 1}"),
    );
    try refused(a, "{\"type\": \"object\", \"minProperties\": 2}", "/minProperties", "minProperties at /minProperties");
    try refused(a, "{\"properties\": {\"a\": {}}, \"additionalProperties\": true, \"maxProperties\": 1}", "/maxProperties", "additionalProperties false");
    try refused(a, "{\"properties\": {\"a\": {}}, \"additionalProperties\": {\"type\": \"integer\"}, \"minProperties\": 1}", "/minProperties", "open object");
    try refused(a, "{\"properties\": {\"a\": {}}, \"patternProperties\": {\"^x\": {}}, \"additionalProperties\": false, \"minProperties\": 1}", "/minProperties", "patternProperties");
    try refused(a, "{\"properties\": {\"a\": {\"type\": \"object\", \"maxProperties\": 3}}}", "/properties/a/maxProperties", "maxProperties at /properties/a/maxProperties");
    try refused(a, "{\"items\": {\"$ref\": \"#/$defs/o\"}, \"$defs\": {\"o\": {\"minProperties\": 1}}}", "/$defs/o/minProperties", "minProperties");
    // an allOf branch stays open (the exception to closing), so its minProperties is an open object's
    try refused(a, "{\"allOf\": [{\"properties\": {\"a\": {}, \"b\": {}}, \"minProperties\": 1}]}", "/allOf/0/minProperties", "open object");
}

test "schema_check closing and allOf: the branches of an allOf, and what a $ref from one leads to, stay open; other properties nodes close" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // closing both branches would leave {"a": 1, "b": 2} no key: each would refuse the other's
    try std.testing.expectEqualStrings(
        with("\"allOf\":[{\"properties\":{\"a\":{\"type\":\"integer\"}}},{\"properties\":{\"b\":{\"type\":\"integer\"}}}]"),
        try accepted(a, "{\"allOf\": [{\"properties\": {\"a\": {\"type\": \"integer\"}}}, {\"properties\": {\"b\": {\"type\": \"integer\"}}}]}"),
    );
    // a plain properties node beside them closes; a branch's own properties values are not branches; a $ref from a
    // branch (and on along $ref) leaves its target open; a $defs node no branch refers to closes
    try std.testing.expectEqualStrings(with("\"anyOf\":[" ++
        "{\"properties\":{\"p\":{}},\"additionalProperties\":false}," ++
        "{\"allOf\":[{\"properties\":{\"a\":{\"properties\":{\"in\":{}},\"additionalProperties\":false}}},{\"$ref\":\"#/$defs/B\"},{\"$ref\":\"#/$defs/R\"}]}]," ++
        "\"$defs\":{\"B\":{\"properties\":{\"b\":{}}},\"R\":{\"$ref\":\"#/$defs/S\"},\"S\":{\"properties\":{\"s\":{}}},\"C\":{\"properties\":{\"c\":{}},\"additionalProperties\":false}}"), try accepted(a, "{\"anyOf\": [" ++
        "{\"properties\": {\"p\": {}}}, " ++
        "{\"allOf\": [{\"properties\": {\"a\": {\"properties\": {\"in\": {}}}}}, {\"$ref\": \"#/$defs/B\"}, {\"$ref\": \"#/$defs/R\"}]}], " ++
        "\"$defs\": {\"B\": {\"properties\": {\"b\": {}}}, \"R\": {\"$ref\": \"#/$defs/S\"}, \"S\": {\"properties\": {\"s\": {}}}, \"C\": {\"properties\": {\"c\": {}}}}}"));
    // a node one branch refers to stays open where else it is used too: the text holds it once
    try std.testing.expectEqualStrings(
        with("\"properties\":{\"x\":{\"$ref\":\"#/$defs/B\"}},\"allOf\":[{\"$ref\":\"#/$defs/B\"}],\"$defs\":{\"B\":{\"properties\":{\"b\":{}}}},\"additionalProperties\":false"),
        try accepted(a, "{\"properties\": {\"x\": {\"$ref\": \"#/$defs/B\"}}, \"allOf\": [{\"$ref\": \"#/$defs/B\"}], \"$defs\": {\"B\": {\"properties\": {\"b\": {}}}}}"),
    );
}

test "schema_check closing: additionalProperties false added where properties stand alone, the client's keys and order kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const given = "{\"title\": \"T\", \"type\": \"object\", \"properties\": {\"z\": {\"type\": \"object\", \"properties\": {\"y\": {\"type\": \"integer\"}}}, \"a\": {\"type\": \"object\"}, \"b\": {\"properties\": {}}}, \"required\": [\"z\"], \"$defs\": {\"d\": {\"properties\": {\"k\": true}, \"description\": \"x\"}}}";
    const v = (try json.parse(a, given)).ok;
    const out = switch (try schema_check.check(a, v)) {
        .text => |t| t,
        .refused => return error.TestUnexpectedResult,
    };
    try std.testing.expectEqualStrings(with("\"title\":\"T\",\"type\":\"object\",\"properties\":{" ++
        "\"z\":{\"type\":\"object\",\"properties\":{\"y\":{\"type\":\"integer\"}},\"additionalProperties\":false}," ++
        "\"a\":{\"type\":\"object\"}," ++
        "\"b\":{\"properties\":{},\"additionalProperties\":false}}," ++
        "\"required\":[\"z\"],\"$defs\":{\"d\":{\"properties\":{\"k\":true},\"description\":\"x\",\"additionalProperties\":false}}," ++
        "\"additionalProperties\":false"), out);
    // the client's value is not changed
    try std.testing.expect(!v.object.contains("additionalProperties"));
    try std.testing.expect(!v.get("properties").?.get("z").?.object.contains("additionalProperties"));
    // what says something of other keys is left as written; data is not rewritten
    try std.testing.expectEqualStrings(with("\"properties\":{\"a\":{}},\"additionalProperties\":true"), try accepted(a, "{\"properties\": {\"a\": {}}, \"additionalProperties\": true}"));
    try std.testing.expectEqualStrings(with("\"properties\":{\"a\":{}},\"patternProperties\":{\"^b\":{}}"), try accepted(a, "{\"properties\": {\"a\": {}}, \"patternProperties\": {\"^b\": {}}}"));
    try std.testing.expectEqualStrings(with("\"type\":\"object\",\"additionalProperties\":{\"properties\":{\"a\":{}},\"additionalProperties\":false}"), try accepted(a, "{\"type\": \"object\", \"additionalProperties\": {\"properties\": {\"a\": {}}}}"));
    try std.testing.expectEqualStrings(with("\"const\":{\"properties\":{\"a\":{}}},\"default\":{\"properties\":{}}"), try accepted(a, "{\"const\": {\"properties\": {\"a\": {}}}, \"default\": {\"properties\": {}}}"));
}

test "schema_check positions: keywords inside data, property names and unknown keywords are not read as schemas" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try accepted(a, "{\"const\": {\"$ref\": \"#/x\", \"additionalItems\": 1, \"items\": [1], \"$schema\": \"draft-07\", \"minProperties\": 1}}");
    _ = try accepted(a, "{\"enum\": [{\"nullable\": true, \"multipleOf\": 0}]}");
    _ = try accepted(a, "{\"default\": {\"$ref\": \"#/default\"}, \"examples\": [{\"additionalItems\": true}]}");
    _ = try accepted(a, "{\"properties\": {\"$ref\": {\"type\": \"string\"}, \"items\": {}, \"additionalItems\": {}, \"$schema\": {}}}");
    _ = try accepted(a, "{\"x-note\": {\"nullable\": true, \"$ref\": \"#/nowhere\"}, \"type\": \"string\"}");
    // the positions are walked: each kind holds a refused keyword once
    const kinds = [_]struct { []const u8, []const u8 }{
        .{ "{\"properties\": {\"p\": {\"nullable\": 1}}}", "/properties/p/nullable" },
        .{ "{\"patternProperties\": {\"^a/b~\": {\"nullable\": 1}}}", "/patternProperties/^a~1b~0/nullable" },
        .{ "{\"additionalProperties\": {\"nullable\": 1}}", "/additionalProperties/nullable" },
        .{ "{\"items\": {\"nullable\": 1}}", "/items/nullable" },
        .{ "{\"prefixItems\": [true, {\"nullable\": 1}]}", "/prefixItems/1/nullable" },
        .{ "{\"allOf\": [{\"nullable\": 1}]}", "/allOf/0/nullable" },
        .{ "{\"anyOf\": [{}, {\"nullable\": 1}]}", "/anyOf/1/nullable" },
        .{ "{\"oneOf\": [{\"nullable\": 1}]}", "/oneOf/0/nullable" },
        .{ "{\"$defs\": {\"d\": {\"nullable\": 1}}}", "/$defs/d/nullable" },
        .{ "{\"definitions\": {\"d\": {\"nullable\": 1}}}", "/definitions/d/nullable" },
    };
    for (kinds) |k| try refused(a, k[0], k[1], "nullable");
}
