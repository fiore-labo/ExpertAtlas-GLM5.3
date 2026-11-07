//! Fiore-lab: the counterexample table through the server's acceptance check and the llguidance binding: the cases
//! llguidance was evaluated on (schemas a grammar engine may weaken), four it weakened itself and their controls, and
//! rows for the limits and rules the server adds. A schema goes through schema_check first, so the x-guidance is the
//! server's (no u escape, bounded whitespace), then to llg_new_matcher; json_object is schema_check's text, a regex
//! goes to llguidance as it is. The walk is llg_test.zig's (the longest allowed token at each step, every mask checked
//! for added and padding ids and for an end only while accepting). A row's outcome: refused by the check (the
//! server's 400), refused by llguidance (its compile error), dead (the target cannot be written as a whole output: no
//! allowed token at some step, or all written but not accepting) or pass (written and accepting). The expectations
//! were set before the runs; a row llguidance answers otherwise is printed and fails the test, it is not made to agree.
//! TF_TEST_LLG_LIB and TF_TEST_GLM_VOCAB as in llg_test.zig; without them the table is skipped.
//! Fiore-lab: 反例の表を、サーバーの受付の検査と llguidance の binding で回す: llguidance を評価した case (文法エンジンが
//! schema を弱めうる例)、llguidance 自身が弱めた 4 つとその対照、サーバーが足す上限と決まりの行。schema はまず
//! schema_check を通るので、x-guidance はサーバーのもの (u なし、空白に上限)。その後 llg_new_matcher へ。json_object は
//! schema_check の文、regex はそのまま。歩き方は llg_test.zig のもの。結果は 4 つ: 検査が断る (サーバーの 400)、
//! llguidance が断る (compile のエラー)、DEAD (目標を出力の全体として書けない: どこかで許された token がない、または
//! 書き終えても受理でない)、PASS (書けて受理)。期待値は流す前に決めた。llguidance の答えが違う行は表に出して試験を失敗に
//! し、期待値を合わせない。

const std = @import("std");
const schema_check = @import("schema_check");
const json = schema_check.json;
const llg = @import("llg.zig");
const llg_test = @import("llg_test.zig");

const Kind = enum { json_schema, json_object, regex };

/// A row's outcome; `finite` is only an expectation: whatever the outcome, the compile returns (a $ref cycle, the
/// largest schema the check takes).
const Outcome = enum { refused_by_check, refused_by_llg, dead, pass, finite };

/// Rows whose schema and target are built here (too long to write out).
const Generated = enum { none, forty_objects, indent_ten, spaces_64, spaces_65, depth_32, depth_33, bytes_64k, bytes_64k_1, properties_64k };

const Row = struct {
    name: []const u8,
    kind: Kind,
    schema: []const u8,
    target: []const u8,
    expect: Outcome,
    /// Run again with a client's x-guidance (no u escape), which the check refuses: the server's alone is taken.
    client_escapes: bool = false,
    generated: Generated = .none,
};

/// The cases llguidance was evaluated on, in their order, with the outcome each must have under the server's settings
/// (no u escape: an extra key by an escaped plain letter and a control character in a key or a string are dead, its
/// cost; the old array forms, $ref outside the root and $defs, integers past 2^53 and multipleOf are the check's).
const probe = [_]Row{
    .{ .name = "object-any (builtin -> json_object)", .kind = .json_object, .schema = "", .target = "{\"name\": \"Tokyo\", \"temps\": [12.5, 14, -3], \"ok\": true, \"next\": null}", .expect = .pass },
    .{ .name = "object-any additionalProperties", .kind = .json_schema, .schema = "{\"type\":\"object\",\"additionalProperties\":true}", .target = "{\"name\": \"Tokyo\", \"nested\": {\"a\": [1, {\"b\": \"c\"}]}}", .expect = .pass },
    .{ .name = "object strict default (no properties)", .kind = .json_schema, .schema = "{\"type\":\"object\"}", .target = "{\"a\": 1}", .expect = .pass },
    .{ .name = "person", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\",\"minimum\":0},\"city\":{\"type\":\"string\",\"enum\":[\"\u{6771}\u{4eac}\",\"\u{5927}\u{962a}\",\"\u{9577}\u{91ce}\"]}},\"required\":[\"name\",\"age\",\"city\"]}", .target = "{\"name\": \"\u{5c71}\u{7530} \u{82b1}\u{5b50}\", \"age\": 34, \"city\": \"\u{9577}\u{91ce}\"}", .expect = .pass },
    .{ .name = "uniqueItems (warned only?)", .kind = .json_schema, .schema = "{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"uniqueItems\":true}", .target = "[1, 1]", .expect = .refused_by_llg },
    .{ .name = "40 nested objects", .kind = .json_schema, .schema = "", .target = "", .expect = .pass, .generated = .forty_objects },
    .{ .name = "regex date", .kind = .regex, .schema = "\\d{4}-\\d{2}-\\d{2}", .target = "1999-12-31", .expect = .pass },
    .{ .name = "builtin: root array (json_object)", .kind = .json_object, .schema = "", .target = "[1, 2]", .expect = .dead },
    .{ .name = "enum+minimum: 0", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"enum\":[0,1],\"minimum\":1}", .target = "0", .expect = .dead },
    .{ .name = "type+anyOf minimum: 0", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"anyOf\":[{\"minimum\":1}]}", .target = "0", .expect = .dead },
    .{ .name = "integer allOf string", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"allOf\":[{\"type\":\"string\"}]}", .target = "\"s\"", .expect = .refused_by_llg },
    .{ .name = "minLength alone: empty string", .kind = .json_schema, .schema = "{\"minLength\":2}", .target = "\"\"", .expect = .dead },
    .{ .name = "required without properties: {}", .kind = .json_schema, .schema = "{\"type\":\"object\",\"required\":[\"a\"]}", .target = "{}", .expect = .dead },
    .{ .name = "ref a~1b", .kind = .json_schema, .schema = "{\"$defs\":{\"a/b\":{\"const\":1},\"a~1b\":{\"const\":2}},\"$ref\":\"#/$defs/a~1b\"}", .target = "2", .expect = .dead },
    .{ .name = "ref a~1b, the right one", .kind = .json_schema, .schema = "{\"$defs\":{\"a/b\":{\"const\":1},\"a~1b\":{\"const\":2}},\"$ref\":\"#/$defs/a~1b\"}", .target = "1", .expect = .pass },
    .{ .name = "declared key, plain", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\": 1}", .expect = .pass, .client_escapes = true },
    .{ .name = "declared and an additional key", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\": 1, \"b\": \"x\"}", .expect = .pass, .client_escapes = true },
    .{ .name = "additional key holding a quote", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\": 1, \"x\\\"y\": 2}", .expect = .pass, .client_escapes = true },
    .{ .name = "additional key by an escape of a plain letter (dead without the u escape)", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\": 1, \"\\u0062\": 2}", .expect = .dead, .client_escapes = true },
    .{ .name = "Japanese declared key, plain", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\u{540d}\u{5b57}\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\u{540d}\u{5b57}\": 1}", .expect = .pass, .client_escapes = true },
    .{ .name = "Japanese declared key as an additional key (bypass)", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\u{540d}\u{5b57}\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\u{540d}\u{5b57}\": \"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "Japanese declared key spelled by \\u escapes as an additional key", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\u{540d}\u{5b57}\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\\u540d\\u5b57\": \"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "declared a/b by an escaped slash (bypass)", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a/b\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\\/b\": \"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "additional key with a control escape", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"a\": 1, \"x\\u001Fy\": 2}", .expect = .dead, .client_escapes = true },
    .{ .name = "a declared backslash key, then \\u0061", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\\\\\":{\"type\":\"integer\"},\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\\u0061\":\"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "the declared backslash key itself", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\\\\\":{\"type\":\"integer\"},\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\\\\\": 1}", .expect = .pass, .client_escapes = true },
    .{ .name = "declared key holding a control char, written as \\u0001", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\\u0001\":{\"type\":\"integer\"}},\"additionalProperties\":false}", .target = "{\"\\u0001\": 1}", .expect = .pass, .client_escapes = true },
    .{ .name = "string value holding a control char, written as \\u0001", .kind = .json_schema, .schema = "{\"type\":\"string\"}", .target = "\"a\\u0001b\"", .expect = .dead, .client_escapes = true },
    .{ .name = "raw UTF-8 surrogate for a declared U+FFFD x3", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"\\ufffd\\ufffd\\ufffd\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\xed\xa0\x80\":\"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "const objects holding title", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"const\":{\"title\":1}},\"b\":{\"const\":{\"title\":2}}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}", .target = "{\"a\":{\"title\":1},\"b\":{\"title\":1}}", .expect = .dead },
    .{ .name = "const objects holding title, the right one", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"const\":{\"title\":1}},\"b\":{\"const\":{\"title\":2}}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}", .target = "{\"a\":{\"title\":1},\"b\":{\"title\":2}}", .expect = .pass },
    .{ .name = "closed objects with a title property", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"integer\"}},\"required\":[\"title\"],\"additionalProperties\":false},\"b\":{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"}},\"required\":[\"title\"],\"additionalProperties\":false}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}", .target = "{\"a\":{\"title\":1},\"b\":{\"title\":2}}", .expect = .dead },
    .{ .name = "closed objects with a title property, the right one", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"integer\"}},\"required\":[\"title\"],\"additionalProperties\":false},\"b\":{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"}},\"required\":[\"title\"],\"additionalProperties\":false}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}", .target = "{\"a\":{\"title\":1},\"b\":{\"title\":\"x\"}}", .expect = .pass },
    .{ .name = "description as a property name", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"description\":{\"type\":\"integer\"},\"title\":{\"type\":\"string\"}},\"required\":[\"description\",\"title\"],\"additionalProperties\":false}", .target = "{\"description\":\"x\",\"title\":\"y\"}", .expect = .dead },
    .{ .name = "description as a property name, the right one", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"description\":{\"type\":\"integer\"},\"title\":{\"type\":\"string\"}},\"required\":[\"description\",\"title\"],\"additionalProperties\":false}", .target = "{\"description\":1,\"title\":\"y\"}", .expect = .pass },
    .{ .name = "percent-encoded ref", .kind = .json_schema, .schema = "{\"$defs\":{\"T\":{\"const\":1}},\"$ref\":\"#%2F$defs%2FT\"}", .target = "2", .expect = .refused_by_check },
    .{ .name = "percent-encoded ref, the right one", .kind = .json_schema, .schema = "{\"$defs\":{\"T\":{\"const\":1}},\"$ref\":\"#%2F$defs%2FT\"}", .target = "1", .expect = .refused_by_check },
    .{ .name = "anchor ref", .kind = .json_schema, .schema = "{\"$ref\":\"#missing\"}", .target = "2", .expect = .refused_by_check },
    .{ .name = "uuid format with a pattern", .kind = .json_schema, .schema = "{\"type\":\"string\",\"format\":\"uuid\",\"pattern\":\"^x$\"}", .target = "\"12345678-1234-1234-1234-123456789abc\"", .expect = .refused_by_llg },
    .{ .name = "uuid format alone", .kind = .json_schema, .schema = "{\"type\":\"string\",\"format\":\"uuid\"}", .target = "\"12345678-1234-1234-1234-123456789abc\"", .expect = .pass },
    .{ .name = "json-pointer format with a raw quote", .kind = .json_schema, .schema = "{\"type\":\"string\",\"format\":\"json-pointer\"}", .target = "\"/a\"b\"", .expect = .refused_by_llg },
    .{ .name = "minLength with a raw TAB", .kind = .json_schema, .schema = "{\"type\":\"string\",\"minLength\":1}", .target = "\"\t\"", .expect = .dead },
    .{ .name = "prefixItems with additionalItems (2020-12)", .kind = .json_schema, .schema = "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"type\":\"array\",\"prefixItems\":[{\"const\":1}],\"additionalItems\":true}", .target = "[2]", .expect = .refused_by_check },
    .{ .name = "control: prefixItems alone", .kind = .json_schema, .schema = "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"type\":\"array\",\"prefixItems\":[{\"const\":1}]}", .target = "[2]", .expect = .dead },
    .{ .name = "$ref into default, duplicate key for minProperties", .kind = .json_schema, .schema = "{\"$ref\":\"#/default\",\"default\":{\"type\":\"object\",\"minProperties\":2}}", .target = "{\"x\":1,\"x\":2}", .expect = .refused_by_check },
    .{ .name = "minimum above 2^53", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"minimum\":9007199254740993}", .target = "9007199254740992", .expect = .refused_by_check },
    .{ .name = "const above 2^53", .kind = .json_schema, .schema = "{\"const\":9007199254740993}", .target = "9007199254740992", .expect = .refused_by_check },
    .{ .name = "multipleOf 65537 twice", .kind = .json_schema, .schema = "{\"allOf\":[{\"type\":\"integer\",\"multipleOf\":65537},{\"multipleOf\":65537}]}", .target = "1", .expect = .refused_by_check },
    .{ .name = "control: multipleOf 65537 once", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":65537}", .target = "1", .expect = .refused_by_check },
    .{ .name = "minLength with an escaped TAB", .kind = .json_schema, .schema = "{\"type\":\"string\",\"minLength\":1}", .target = "\"\\t\"", .expect = .pass },
    .{ .name = "escaped key bypass", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"additionalProperties\":true}", .target = "{\"\\u0061\": \"x\"}", .expect = .dead, .client_escapes = true },
    .{ .name = "builtin: root number (json_object)", .kind = .json_object, .schema = "", .target = "1", .expect = .dead },
    .{ .name = "integer that could go on: 12 then the stop", .kind = .json_schema, .schema = "{\"type\":\"integer\"}", .target = "12", .expect = .pass },
    .{ .name = "regex \\d+ that could go on: 12 then the stop", .kind = .regex, .schema = "\\d+", .target = "12", .expect = .pass },
    .{ .name = "object strict default: root array", .kind = .json_schema, .schema = "{\"type\":\"object\"}", .target = "[]", .expect = .dead },
    .{ .name = "additionalProperties: root array", .kind = .json_schema, .schema = "{\"type\":\"object\",\"additionalProperties\":true}", .target = "[]", .expect = .dead },
};

/// The evaluation's lenient cases: each holds a client's x-guidance.
const lenient = [_]Row{
    .{ .name = "uniqueItems [lenient]", .kind = .json_schema, .schema = "{\"x-guidance\":{\"lenient\":true},\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"uniqueItems\":true}", .target = "[1, 1]", .expect = .refused_by_check },
    .{ .name = "anchor ref [lenient]", .kind = .json_schema, .schema = "{\"x-guidance\":{\"lenient\":true},\"$ref\":\"#missing\"}", .target = "2", .expect = .refused_by_check },
    .{ .name = "uuid format with a pattern [lenient]", .kind = .json_schema, .schema = "{\"x-guidance\":{\"lenient\":true},\"type\":\"string\",\"format\":\"uuid\",\"pattern\":\"^x$\"}", .target = "\"12345678-1234-1234-1234-123456789abc\"", .expect = .refused_by_check },
    .{ .name = "json-pointer format with a raw quote [lenient]", .kind = .json_schema, .schema = "{\"x-guidance\":{\"lenient\":true},\"type\":\"string\",\"format\":\"json-pointer\"}", .target = "\"/a\"b\"", .expect = .refused_by_check },
    .{ .name = "integer allOf string [lenient]", .kind = .json_schema, .schema = "{\"x-guidance\":{\"lenient\":true},\"type\":\"integer\",\"allOf\":[{\"type\":\"string\"}]}", .target = "\"s\"", .expect = .refused_by_check },
};

const object_a = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}";
const tree = "{\"anyOf\":[{\"type\":\"null\"},{\"type\":\"array\",\"items\":{\"$ref\":\"#\"}}]}";
const pattern_n = "{\"type\":\"object\",\"patternProperties\":{\"^n_\":{\"type\":\"integer\"}}}";
const declared_n = "{\"type\":\"object\",\"properties\":{\"n_a\":{\"type\":\"number\"}},\"patternProperties\":{\"^n_\":{\"type\":\"integer\"}}}";
const nested_id = "{\"type\":\"object\",\"properties\":{\"p\":{\"$id\":\"http://example.com/p\",\"$defs\":{\"a\":{\"type\":\"string\"}},\"$ref\":\"#/$defs/a\"}},\"$defs\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"p\"]}";

/// The rows added to the evaluation's cases: whitespace limits, the check's rules at their edges (accepted ones
/// compiled too), $ref cycles and recursion, decimal multipleOf and the edges of the multipleOf bound, the
/// patternProperties types, objects closed to their declared properties (an allOf's branches open).
/// 評価の case に足す行: 空白の上限、検査の決まりの境目 (受けたものは compile も)、$ref の循環と再帰、小数の
/// multipleOf と multipleOf の上限の端、patternProperties の型、宣言したキーだけに閉じる object (allOf の枝は open)。
const extra = [_]Row{
    // whitespace: once per gap, at most one newline and 64 spaces or tabs, none before the first { or after the last }
    .{ .name = "ws: 10 nested objects indented by two spaces", .kind = .json_schema, .schema = "", .target = "", .expect = .pass, .generated = .indent_ten },
    .{ .name = "ws: 64 spaces in a row", .kind = .json_schema, .schema = "", .target = "", .expect = .pass, .generated = .spaces_64 },
    .{ .name = "ws: 65 spaces in a row", .kind = .json_schema, .schema = "", .target = "", .expect = .dead, .generated = .spaces_65 },
    .{ .name = "ws: two newlines in a row", .kind = .json_schema, .schema = object_a, .target = "{\n\n  \"a\": 1\n}", .expect = .dead },
    .{ .name = "ws: CR LF and two spaces", .kind = .json_schema, .schema = object_a, .target = "{\r\n  \"a\": 1\r\n}", .expect = .pass },
    .{ .name = "ws: a tab indent", .kind = .json_schema, .schema = object_a, .target = "{\n\t\"a\": 1\n}", .expect = .pass },
    .{ .name = "ws: a space before the first {", .kind = .json_schema, .schema = object_a, .target = " {\"a\": 1}", .expect = .dead },
    .{ .name = "ws: a newline before the first {", .kind = .json_schema, .schema = object_a, .target = "\n{\"a\": 1}", .expect = .dead },
    .{ .name = "ws: a space after the last }", .kind = .json_schema, .schema = object_a, .target = "{\"a\": 1} ", .expect = .dead },
    // the check's rules through llguidance: refused, and the accepted side of each edge compiled and written
    .{ .name = "check: root true writes any JSON", .kind = .json_schema, .schema = "true", .target = "[1, {\"a\": null}]", .expect = .pass },
    .{ .name = "check: root false", .kind = .json_schema, .schema = "false", .target = "1", .expect = .refused_by_check },
    .{ .name = "check: 32 levels of nesting (31 arrays)", .kind = .json_schema, .schema = "", .target = "", .expect = .pass, .generated = .depth_32 },
    .{ .name = "check: 33 levels of nesting", .kind = .json_schema, .schema = "", .target = "", .expect = .refused_by_check, .generated = .depth_33 },
    .{ .name = "check: 65536 bytes (a long description)", .kind = .json_schema, .schema = "", .target = "", .expect = .pass, .generated = .bytes_64k },
    .{ .name = "check: 65537 bytes", .kind = .json_schema, .schema = "", .target = "", .expect = .refused_by_check, .generated = .bytes_64k_1 },
    .{ .name = "check: about 64 KiB of integer properties (llguidance's own limits may refuse)", .kind = .json_schema, .schema = "", .target = "", .expect = .finite, .generated = .properties_64k },
    .{ .name = "check: nullable", .kind = .json_schema, .schema = "{\"type\":\"string\",\"nullable\":true}", .target = "null", .expect = .refused_by_check },
    .{ .name = "check: draft-07 $schema", .kind = .json_schema, .schema = "{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"array\",\"items\":[{\"const\":1}]}", .target = "[2]", .expect = .refused_by_check },
    .{ .name = "check: items as an array", .kind = .json_schema, .schema = "{\"type\":\"array\",\"items\":[{\"const\":1}]}", .target = "[2]", .expect = .refused_by_check },
    .{ .name = "check: minProperties on an object without properties (open)", .kind = .json_schema, .schema = "{\"type\":\"object\",\"minProperties\":2}", .target = "{\"x\":1,\"x\":2}", .expect = .refused_by_check },
    .{ .name = "check: minProperties on an object the server closes", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"],\"minProperties\":2}", .target = "{\"a\": 1, \"b\": 2}", .expect = .pass },
    .{ .name = "check: the $ref-into-default data in a default nobody references", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"default\":{\"type\":\"object\",\"minProperties\":2}}", .target = "1", .expect = .pass },
    .{ .name = "check: minimum 2^53 itself", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"minimum\":9007199254740992}", .target = "9007199254740992", .expect = .pass },
    .{ .name = "check: minimum 2^53, one below", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"minimum\":9007199254740992}", .target = "9007199254740991", .expect = .dead },
    .{ .name = "check: $ref to an $id's URI", .kind = .json_schema, .schema = "{\"$defs\":{\"a\":{\"$id\":\"urn:example:a\",\"type\":\"integer\"}},\"$ref\":\"urn:example:a\"}", .target = "1", .expect = .refused_by_check },
    .{ .name = "check: #/$defs/a under a nested $id is that node's (string)", .kind = .json_schema, .schema = nested_id, .target = "{\"p\": \"x\"}", .expect = .pass },
    .{ .name = "check: #/$defs/a under a nested $id is not the root's (integer)", .kind = .json_schema, .schema = nested_id, .target = "{\"p\": 1}", .expect = .dead },
    // $ref cycles: refused or a finite grammar; recursion with content passes
    .{ .name = "ref: {\"$ref\":\"#\"}", .kind = .json_schema, .schema = "{\"$ref\":\"#\"}", .target = "1", .expect = .finite },
    .{ .name = "ref: a -> b -> a with nothing else", .kind = .json_schema, .schema = "{\"$defs\":{\"a\":{\"$ref\":\"#/$defs/b\"},\"b\":{\"$ref\":\"#/$defs/a\"}},\"$ref\":\"#/$defs/a\"}", .target = "1", .expect = .finite },
    .{ .name = "ref: anyOf null or an array of #", .kind = .json_schema, .schema = tree, .target = "[null, [[], null]]", .expect = .pass },
    .{ .name = "ref: anyOf null or an array of #, a number inside", .kind = .json_schema, .schema = tree, .target = "[[1]]", .expect = .dead },
    // multipleOf: one is an integer from 1 to 65536, or a decimal llguidance reads as one at 8 places or fewer (the
    // range llguidance computes right, measured below); two on a value are refused
    .{ .name = "mo: 0.01 takes 0.25", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.01}", .target = "0.25", .expect = .pass },
    .{ .name = "mo: 0.01 refuses 0.257", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.01}", .target = "0.257", .expect = .dead },
    .{ .name = "mo: 65536 takes 65536", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":65536}", .target = "65536", .expect = .pass },
    .{ .name = "mo: 65536 takes 131072", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":65536}", .target = "131072", .expect = .pass },
    .{ .name = "mo: 65536 refuses 65535", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":65536}", .target = "65535", .expect = .dead },
    .{ .name = "mo: 65537 (above 65536)", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":65537}", .target = "65537", .expect = .refused_by_check },
    .{ .name = "mo: 2147483647 takes itself", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":2147483647}", .target = "2147483647", .expect = .refused_by_check },
    .{ .name = "mo: 2147483647 takes 4294967294", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":2147483647}", .target = "4294967294", .expect = .refused_by_check },
    .{ .name = "mo: 2147483647 refuses 2147483646", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":2147483647}", .target = "2147483646", .expect = .refused_by_check },
    .{ .name = "mo: 2147483647 refuses 7", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":2147483647}", .target = "7", .expect = .refused_by_check },
    // Measured on the Mac before the bound, now refused by the check: the compile's time grows with the
    // coefficient (65537 took 0.1 s, 1,000,003 1.8 s, 10,000,019 more than 45 s); multipleOf 429496729 written as 6
    // PASSED (10 x 429496729 + 6 wraps in u32 to 0) after about two minutes; 429496728 did not finish its compile in
    // 9.5 minutes (5.5 GB); 2147483647 ran out of the lexer's fuel at once.
    .{ .name = "mo: 2147483648 (2^31)", .kind = .json_schema, .schema = "{\"type\":\"integer\",\"multipleOf\":2147483648}", .target = "2147483648", .expect = .refused_by_check },
    .{ .name = "mo: 1.23456789 takes 2.46913578", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":1.23456789}", .target = "2.46913578", .expect = .refused_by_check },
    .{ .name = "mo: 1.23456789 refuses 2.46913579", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":1.23456789}", .target = "2.46913579", .expect = .refused_by_check },
    .{ .name = "mo: 0.123456789 takes 0.246913578", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.123456789}", .target = "0.246913578", .expect = .refused_by_check },
    .{ .name = "mo: 0.123456789 refuses 0.246913579", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.123456789}", .target = "0.246913579", .expect = .refused_by_check },
    .{ .name = "mo: 0.07 takes 0.21", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.07}", .target = "0.21", .expect = .refused_by_check },
    .{ .name = "mo: 3e-10 (3 at 10 places, llguidance wraps: 0.6 refused)", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":3e-10}", .target = "0.6", .expect = .refused_by_check },
    .{ .name = "mo: 7e-9 (7 at 9 places, llguidance wraps: 5.6 refused)", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":7e-9}", .target = "5.6", .expect = .refused_by_check },
    .{ .name = "mo: 3e-8 (3 at 8 places) takes 9.3", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":3e-8}", .target = "9.3", .expect = .pass },
    .{ .name = "mo: 3e-8 refuses 9.30000001", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":3e-8}", .target = "9.30000001", .expect = .dead },
    .{ .name = "mo: 0.00065536 (65536 at 8 places) takes 0.00131072", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.00065536}", .target = "0.00131072", .expect = .pass },
    .{ .name = "mo: 0.00065536 refuses 9.17503999", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.00065536}", .target = "9.17503999", .expect = .dead },
    .{ .name = "mo: 0.25 takes 0.75", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.25}", .target = "0.75", .expect = .pass },
    // a known narrowing, not a weakening: llguidance's regex library (derivre, simplify.rs: "TODO: trim trailing
    // zeros?") makes a decimal multiple's remaining places a fixed string, trailing 0 included, so 0.5 is written only
    // as 0.50; what it writes is always a multiple
    // 弱めではなく既知の狭め: llguidance の regex の部品 (derivre、simplify.rs) は小数の倍数の残りの桁を末尾の 0 も含めた
    // 固定の文字列にするので、0.5 は 0.50 としか書けない。書くものは必ず倍数
    .{ .name = "mo: 0.25 does not end at 0.5 (only 0.50: derivre keeps the trailing 0)", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.25}", .target = "0.5", .expect = .dead },
    .{ .name = "mo: 0.25 takes 0.50", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.25}", .target = "0.50", .expect = .pass },
    .{ .name = "mo: 0.1234567891 (10 digits)", .kind = .json_schema, .schema = "{\"type\":\"number\",\"multipleOf\":0.1234567891}", .target = "0.2469135782", .expect = .refused_by_check },
    .{ .name = "mo: 2 and 3 on one value", .kind = .json_schema, .schema = "{\"allOf\":[{\"multipleOf\":2},{\"multipleOf\":3}]}", .target = "6", .expect = .refused_by_check },
    .{ .name = "mo: 0.01 on two properties through one $defs", .kind = .json_schema, .schema = "{\"type\":\"object\",\"properties\":{\"price\":{\"$ref\":\"#/$defs/m\"},\"tax\":{\"$ref\":\"#/$defs/m\"}},\"$defs\":{\"m\":{\"type\":\"number\",\"multipleOf\":0.01}}}", .target = "{\"price\": 1.25, \"tax\": 0.1}", .expect = .pass },
    // patternProperties: a matching key keeps the pattern's type, also beside a declared one; others are extra keys
    .{ .name = "pp: a matching key, an integer", .kind = .json_schema, .schema = pattern_n, .target = "{\"n_a\": 1}", .expect = .pass },
    .{ .name = "pp: a matching key, a string", .kind = .json_schema, .schema = pattern_n, .target = "{\"n_a\": \"x\"}", .expect = .dead },
    .{ .name = "pp: a key the pattern does not match", .kind = .json_schema, .schema = pattern_n, .target = "{\"b\": \"x\"}", .expect = .pass },
    .{ .name = "pp: a declared number key the pattern makes an integer, 1.5", .kind = .json_schema, .schema = declared_n, .target = "{\"n_a\": 1.5}", .expect = .dead },
    .{ .name = "pp: a declared number key the pattern makes an integer, 2", .kind = .json_schema, .schema = declared_n, .target = "{\"n_a\": 2}", .expect = .pass },
    // closed objects: properties without additionalProperties take only the declared keys
    .{ .name = "closed: a declared key", .kind = .json_schema, .schema = object_a, .target = "{\"a\": 1}", .expect = .pass },
    .{ .name = "closed: a key not declared", .kind = .json_schema, .schema = object_a, .target = "{\"a\": 1, \"b\": 2}", .expect = .dead },
    .{ .name = "closed: allOf of {properties a} and {properties b} stays open: both keys", .kind = .json_schema, .schema = "{\"allOf\":[{\"properties\":{\"a\":{\"type\":\"integer\"}}},{\"properties\":{\"b\":{\"type\":\"integer\"}}}]}", .target = "{\"a\": 1, \"b\": 2}", .expect = .pass },
    .{ .name = "closed: the same allOf, a key neither declares", .kind = .json_schema, .schema = "{\"allOf\":[{\"properties\":{\"a\":{\"type\":\"integer\"}}},{\"properties\":{\"b\":{\"type\":\"integer\"}}}]}", .target = "{\"a\": 1, \"b\": 2, \"c\": 3}", .expect = .pass },
    .{ .name = "closed: the same allOf, a declared key keeps its type", .kind = .json_schema, .schema = "{\"allOf\":[{\"properties\":{\"a\":{\"type\":\"integer\"}}},{\"properties\":{\"b\":{\"type\":\"integer\"}}}]}", .target = "{\"a\": \"x\", \"b\": 2}", .expect = .dead },
};

/// The whole table: the evaluation's cases, their runs with a client's x-guidance, its lenient cases, the extra rows.
fn table(a: std.mem.Allocator) ![]Row {
    var rows: std.ArrayList(Row) = .empty;
    for (probe) |r| try rows.append(a, try build(a, r));
    for (probe) |r| if (r.client_escapes) try rows.append(a, .{
        .name = try std.fmt.allocPrint(a, "{s} [the client's x-guidance: no u]", .{r.name}),
        .kind = .json_schema,
        .schema = try std.fmt.allocPrint(a, "{{\"x-guidance\":{{\"json_allowed_escapes\":\"nrbtf\\\\\\\"\"}},{s}", .{r.schema[1..]}),
        .target = r.target,
        .expect = .refused_by_check,
    });
    try rows.appendSlice(a, &lenient);
    for (extra) |r| try rows.append(a, try build(a, r));
    return rows.items;
}

/// A generated row's schema and target.
fn build(a: std.mem.Allocator, r: Row) !Row {
    var out = r;
    var s: std.ArrayList(u8) = .empty;
    var t: std.ArrayList(u8) = .empty;
    switch (r.generated) {
        .none => return r,
        .forty_objects => { // the evaluation's 40 objects of an id, tags and a note, and their text
            try s.appendSlice(a, "{\"type\":\"object\",\"properties\":{");
            try t.append(a, '{');
            for (0..40) |i| {
                try s.print(a, "{s}\"f{d}\":{{\"type\":\"object\",\"properties\":{{\"id\":{{\"type\":\"integer\"}},\"tags\":{{\"type\":\"array\",\"items\":{{\"type\":\"string\",\"enum\":[\"alpha\",\"beta\",\"gamma\"]}}}},\"note\":{{\"type\":\"string\",\"maxLength\":40}}}},\"required\":[\"id\",\"tags\",\"note\"]}}", .{ if (i > 0) "," else "", i });
                try t.print(a, "{s}\"f{d}\": {{\"id\": {d}, \"tags\": [\"beta\", \"alpha\"], \"note\": \"item {d} is fine\"}}", .{ if (i > 0) ", " else "", i, i * 7, i });
            }
            try s.appendSlice(a, "},\"required\":[\"f0\",\"f1\"]}");
            try t.append(a, '}');
        },
        .indent_ten => { // {"a": {"a": ... 1}} ten deep, a newline and two more spaces for each level
            for (0..10) |_| try s.appendSlice(a, "{\"type\":\"object\",\"properties\":{\"a\":");
            try s.appendSlice(a, "{\"type\":\"integer\"}");
            for (0..10) |_| try s.appendSlice(a, "},\"required\":[\"a\"]}");
            for (0..10) |l| {
                try t.appendSlice(a, "{\n");
                try t.appendNTimes(a, ' ', 2 * (l + 1));
                try t.appendSlice(a, "\"a\": ");
            }
            try t.append(a, '1');
            for (0..10) |l| {
                try t.append(a, '\n');
                try t.appendNTimes(a, ' ', 2 * (9 - l));
                try t.append(a, '}');
            }
        },
        .spaces_64, .spaces_65 => {
            try s.appendSlice(a, object_a);
            try t.append(a, '{');
            try t.appendNTimes(a, ' ', if (r.generated == .spaces_64) 64 else 65);
            try t.appendSlice(a, "\"a\": 1}");
        },
        .depth_32, .depth_33 => { // the root object and nested items: 32 (33) levels, 31 (32) arrays in the output
            const n: usize = if (r.generated == .depth_32) 31 else 32;
            for (0..n) |_| try s.appendSlice(a, "{\"type\":\"array\",\"items\":");
            try s.appendSlice(a, "{\"type\":\"integer\"}");
            for (0..n) |_| try s.append(a, '}');
            try t.appendNTimes(a, '[', n);
            try t.append(a, '1');
            try t.appendNTimes(a, ']', n);
        },
        .bytes_64k, .bytes_64k_1 => {
            const want: usize = if (r.generated == .bytes_64k) schema_check.max_bytes else schema_check.max_bytes + 1;
            const frame = "{\"type\":\"integer\",\"description\":\"\"}";
            try s.appendSlice(a, frame[0 .. frame.len - 2]);
            try s.appendNTimes(a, 'x', want - frame.len);
            try s.appendSlice(a, "\"}");
            try t.appendSlice(a, "1");
        },
        .properties_64k => { // "pNNNN":{"type":"integer"} until the next one would pass 64 KiB
            try s.appendSlice(a, "{\"type\":\"object\",\"properties\":{");
            var i: usize = 0;
            while (true) : (i += 1) {
                const one = try std.fmt.allocPrint(a, "{s}\"p{d}\":{{\"type\":\"integer\"}}", .{ if (i > 0) "," else "", i });
                if (s.items.len + one.len + 2 > schema_check.max_bytes) break;
                try s.appendSlice(a, one);
            }
            try s.appendSlice(a, "}}");
            out.name = try std.fmt.allocPrint(a, "{s}: {d} properties, {d} bytes", .{ r.name, i, s.items.len });
            try t.appendSlice(a, "{\"p0\": 1, \"p1\": 2}");
        },
    }
    out.schema = s.items;
    out.target = t.items;
    return out;
}

const Result = struct { got: Outcome, detail: []const u8, bad: usize = 0, compile_ms: f64 = 0, walk_ms: f64 = 0 };

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
}

fn ms(from: i96, to: i96) f64 {
    return @as(f64, @floatFromInt(to - from)) / 1e6;
}

/// One row: the check (json_schema), the compile, the walk; and the end's mask (no added or padding id, the three
/// ends while accepting and none otherwise). An error from llguidance at the first mask is its refusal too (the
/// server computes the first mask when it checks the compile); one later in the walk is a dead end.
fn run(f: *llg_test.Fixture, a: std.mem.Allocator, row: Row) !Result {
    const api = &f.lib.api;
    const text: [:0]const u8 = switch (row.kind) {
        .regex => try a.dupeSentinel(u8, row.schema, 0),
        .json_object => schema_check.json_object_text,
        .json_schema => blk: {
            const v = switch (try json.parse(a, row.schema)) {
                .ok => |v| v,
                .err => |e| {
                    std.debug.print("{s}: the row's schema is not JSON: {s}\n", .{ row.name, e });
                    return error.BadRow;
                },
            };
            switch (try schema_check.check(a, v)) {
                .text => |t| break :blk t,
                .refused => |r| return .{ .got = .refused_by_check, .detail = r.message },
            }
        },
    };
    const init = llg.constraintInit(api, f.tok);
    const t0 = nowNs();
    const m = api.llg_new_matcher(&init, if (row.kind == .regex) "regex" else "json_schema", text.ptr) orelse return error.NoMatcher;
    defer api.llg_free_matcher(m);
    const t1 = nowNs();
    if (api.llg_matcher_is_error(m))
        return .{ .got = .refused_by_llg, .compile_ms = ms(t0, t1), .detail = try a.dupe(u8, std.mem.span(api.llg_matcher_get_error(m) orelse "")) };
    var bad: usize = 0;
    var at: usize = 0;
    const walked = try f.walkTo(m, row.target, &bad, &at);
    const t2 = nowNs();
    const steps = walked orelse {
        const rest = std.zig.fmtString(row.target[at..@min(row.target.len, at + 24)]);
        if (api.llg_matcher_is_error(m)) {
            const why = std.mem.span(api.llg_matcher_get_error(m) orelse "");
            return .{ .got = if (at == 0) .refused_by_llg else .dead, .bad = bad, .compile_ms = ms(t0, t1), .walk_ms = ms(t1, t2), .detail = try std.fmt.allocPrint(a, "llguidance's error at byte {d} (before {f}): {s}", .{ at, rest, why }) };
        }
        return .{ .got = .dead, .bad = bad, .compile_ms = ms(t0, t1), .walk_ms = ms(t1, t2), .detail = try std.fmt.allocPrint(a, "no allowed token at byte {d}, before {f}", .{ at, rest }) };
    };
    var mask: [llg.maskWords(llg_test.vocab_size)]u32 = undefined;
    const accepting = api.llg_matcher_is_accepting(m);
    if (api.llg_matcher_compute_mask_into(m, &mask, @sizeOf(@TypeOf(mask))) != 0) return error.MaskAtTheEnd;
    var ends: usize = 0;
    for (llg_test.added_first..llg_test.vocab_size) |i| {
        const id: llg.Token = @intCast(i);
        if (!llg.allowed(&mask, id)) continue;
        if (std.mem.indexOfScalar(llg.Token, &llg_test.eos, id) == null) bad += 1 else ends += 1;
    }
    if (ends != (if (accepting) llg_test.eos.len else 0)) bad += 1;
    return .{
        .got = if (accepting) .pass else .dead,
        .bad = bad,
        .compile_ms = ms(t0, t1),
        .walk_ms = ms(t1, t2),
        .detail = try std.fmt.allocPrint(a, "{d} tokens, {s}", .{ steps, if (accepting) "accepting" else "all written but not accepting" }),
    };
}

test {
    _ = llg_test;
}

test "structured counterexample table: the evaluation's cases and the added rows through the check and llguidance" {
    const gpa = std.testing.allocator;
    var f = try llg_test.Fixture.open(gpa);
    defer f.close(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = try table(a);
    var counts: [4]usize = @splat(0);
    var wrong: usize = 0;
    var bad: usize = 0;
    var finite: usize = 0;
    for (rows) |row| {
        const r = try run(&f, a, row);
        counts[@backingInt(r.got)] += 1;
        bad += r.bad;
        if (row.expect == .finite) finite += 1;
        const ok = row.expect == .finite or r.got == row.expect;
        if (!ok) wrong += 1;
        const detail = r.detail[0..@min(r.detail.len, 140)];
        std.debug.print("{s} {s:<16} {s:<16} {s} | {s} | compile {d:.1} ms, walk {d:.1} ms\n", .{ if (ok) "  " else "NG", @tagName(row.expect), @tagName(r.got), row.name, detail, r.compile_ms, r.walk_ms });
    }
    std.debug.print("{d} rows: refused_by_check {d}, refused_by_llg {d}, dead {d}, pass {d} ({d} rows expected finite); {d} not as expected; masks with an added or padding id, or a wrong end: {d}\n", .{ rows.len, counts[0], counts[1], counts[2], counts[3], finite, wrong, bad });
    try std.testing.expectEqual(@as(usize, 0), bad);
    try std.testing.expectEqual(@as(usize, 0), wrong);
}
