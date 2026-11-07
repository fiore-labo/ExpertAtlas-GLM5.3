//! Fiore-lab: the server's acceptance check of a request's JSON schema, before llguidance compiles it. llguidance
//! itself refuses the keywords it does not implement and intersects those of one node; the server refuses what
//! llguidance would weaken without an error: the client's x-guidance, nullable (ignored, so null would not be
//! written), minProperties and maxProperties on an open object (an extra key can be written twice), a draft other
//! than 2020-12 and its old array forms, a $ref anywhere but the root and $defs, integers past 2^53 (rounded to
//! doubles) and two multipleOf on one value (their least common multiple wraps in u32). What passes comes back as
//! the text llguidance compiles: the server's x-guidance first, and every object that declares properties without
//! saying what other keys may do closed to them, but for the branches of an allOf. Pure Zig on the server's JSON
//! values; nothing here calls llguidance.
//! Fiore-lab: llguidance が compile する前の、要求の JSON schema のサーバーの受付の検査。
//! 実装していないキーワードは llguidance 自身が断り、1 つの節点のキーワードは交わす。サーバーが断るのは、llguidance が
//! エラーなしに弱めるもの: クライアントの x-guidance、nullable (無視されて null が書けない)、open な object の
//! minProperties・maxProperties (追加のキーは同じキーを 2 回書ける)、2020-12 以外の版と古い配列の形、根と $defs
//! 以外への $ref、2^53 を超える整数 (double に丸められる)、1 つの値にかかる 2 つの multipleOf (最小公倍数が
//! u32 であふれる)。通ったものは llguidance が compile する文にして返す: 先頭にサーバーの x-guidance、properties を
//! 宣言してほかのキーの扱いを書いていない object は宣言したキーだけに閉じる (allOf の枝は除く)。サーバーの
//! JSON の値の上の Zig だけで、llguidance は呼ばない。

const std = @import("std");
pub const json = @import("json.zig");
const Value = json.Value;
const Object = json.Object;
const Allocator = std.mem.Allocator;

/// The largest schema the server takes, in bytes of the client's schema written as compact JSON (UTF-8), and its
/// deepest nesting of objects and arrays (the root object is 1; data such as const counts too), so that llguidance's
/// recursion stays far from the end of Rust's stack (an overflow aborts the process).
/// サーバーが受ける schema の最大: クライアントの schema を compact な JSON (UTF-8) で書いたバイト数と、object と配列の
/// 入れ子の最も深い段 (根の object が 1。const などのデータも数える)。llguidance の再帰が Rust の stack の
/// 終わり (溢れるとプロセスごと abort) から遠くにあるように。
pub const max_bytes = 64 * 1024;
pub const max_depth = 32;

/// The single multipleOf llguidance computes right: its coefficient from 1 to 65536 and at most 8 decimal places.
/// llguidance's regex library keeps the remainder in u32 and adds a digit times 10^places to ten times it (derivre's
/// deriv.rs and relevance.rs), which fits for 65536 at 8 places and wraps at 9 (multipleOf 7e-9 refused 5.6); the
/// compile's time grows with the coefficient (65537 took 0.1 s, 1,000,003 took 1.8 s, 10,000,019 more than 45 s on
/// the Mac).
/// llguidance が正しく計算する 1 つの multipleOf: 係数は 1 から 65536、小数の桁は 8 まで。
/// llguidance の regex のライブラリは余りを u32 で持ち、その 10 倍に「桁 × 10^桁数」を足す (derivre の deriv.rs と
/// relevance.rs)。65536 で 8 桁なら収まり、9 桁であふれる (multipleOf 7e-9 で 5.6 を断った)。compile の時間は係数に比例して
/// 伸びる (Mac で 65537 は 0.1 s、1,000,003 は 1.8 s、10,000,019 は 45 s を超えた)。
pub const max_multiple = 65536;
pub const max_multiple_places = 8;

/// The one draft $schema may name (referencing's spelling; a trailing # is the same draft to it).
/// $schema に書ける唯一の版 (referencing の綴り。末尾の # は同じ版とみなされる)。
pub const draft_2020_12 = "https://json-schema.org/draft/2020-12/schema";

/// The x-guidance the server puts first in every schema, every field written, since llguidance takes the field
/// defaults for the ones left out. No u escape: a declared key has one spelling, so an extra key cannot spell it
/// otherwise; the cost is that control characters without a short escape cannot be written.
/// Whitespace once between two tokens: at most one newline and 64 spaces or tabs, so two-space indents reach 32 levels.
/// サーバーがどの schema にも先頭に入れる x-guidance。llguidance は書いていない欄を既定にするので
/// 全部の欄を書く。u のエスケープはなし: 宣言したキーの綴りは 1 通りで、追加のキーがほかの綴りで書けない。
/// 費用は、短いエスケープのない制御文字を書けないこと。空白は token の間に 1 回: 改行は高々 1 つ、空白とタブは高々 64 で、
/// 2 つの空白の字下げなら 32 段まで。
pub const x_guidance =
    \\{"lenient":false,"coerce_one_of":false,"json_allowed_escapes":"nrbtf\\\"","json_allow_general_unicode_escapes":false,"whitespace_pattern":"[\\x20\\x09]{1,64}|\\x0D?\\x0A[\\x20\\x09]{0,64}","item_separator":",","key_separator":":"}
;

/// json_object's schema: any object, with the same x-guidance as a json_schema request.
/// json_object の schema: 任意の object。x-guidance は json_schema の要求と同じ。
pub const json_object_text: [:0]const u8 = "{\"x-guidance\":" ++ x_guidance ++ ",\"type\":\"object\"}";

/// Why the server refuses a schema: the place as a JSON Pointer into the client's schema, the keyword's own place
/// (such as /properties/a/minProperties; empty for the whole schema), and the 400's text, which names both.
/// サーバーが schema を断る理由: クライアントの schema の中の場所 (JSON Pointer。キーワードそのものの場所で、
/// /properties/a/minProperties など。schema 全体なら空) と、その両方を書いた 400 の文。
pub const Refusal = struct { pointer: []const u8, message: []const u8 };

pub const Checked = union(enum) {
    /// The schema llguidance compiles as "json_schema": compact JSON, the x-guidance first, a NUL after the end.
    /// llguidance が "json_schema" として compile する schema: compact な JSON、先頭に x-guidance、終わりの後に NUL。
    text: [:0]const u8,
    refused: Refusal,
};

/// Checks the request's schema and gives the text llguidance compiles, or the refusal. `a` is the request's arena: the
/// rewritten copy and the scratch stay in it. The client's value is not changed.
/// 要求の schema を確かめ、llguidance が compile する文か、断る理由を返す。`a` は要求の arena で、書き換えた写しと作業の
/// 領域はそこに残る。クライアントの値は変えない。
pub fn check(a: Allocator, schema: Value) Allocator.Error!Checked {
    var c: Checker = .{ .a = a };
    const text = c.run(schema) catch |e| switch (e) {
        error.Refused => return .{ .refused = c.refusal },
        error.OutOfMemory => return error.OutOfMemory,
    };
    return .{ .text = text };
}

const Fail = error{ Refused, OutOfMemory };

/// One step of a place in the schema document.
const Step = union(enum) { key: []const u8, index: usize };

/// One step of a place in the output, from its root: a declared key, any key (additionalProperties,
/// patternProperties), a prefixItems index, any index (items).
/// 出力の中の場所の 1 歩 (根から): 宣言したキー、任意のキー (additionalProperties、patternProperties)、prefixItems の番号、
/// 任意の番号 (items)。
const Hop = union(enum) { key: []const u8, any_key, index: usize, any_index };

/// A multipleOf and the value of the output it lands on; `anywhere` once a $ref cycle repeats it.
const Landing = struct { pointer: []const u8, hops: []const Hop, anywhere: bool };

/// The node a "#" means in a $ref at or below it (the nearest one with $id, else the root) and the length of its place.
/// その節点とその下の $ref の "#" が指す節点 ($id のある一番近い節点、なければ根) と、その場所の長さ。
const Base = struct { node: *Object, depth: usize };

/// A $ref's target: the node and its place in the document.
const Target = struct { value: Value, steps: []const Step };

/// A $ref target being followed by the multipleOf walk; `again` when a $ref under it leads back to it.
const Frame = struct { node: *const Object, start: usize, again: bool };

/// The multipleOf walk's limits: schema nodes it visits (a $ref can lead to the same node from many places, and
/// many $ref to many places) and landings it compares, pair by pair. Only a schema with multipleOf takes this walk.
/// multipleOf の歩き方の上限: 訪れる schema の節点の数 ($ref は同じ節点に多くの場所から至りうる) と、2 つずつ比べる着地の数。
/// この歩き方をするのは multipleOf のある schema だけ。
const max_visits = 100_000;
const max_landings = 1000;

const Checker = struct {
    a: Allocator,
    path: std.ArrayList(Step) = .empty,
    seen: std.AutoHashMapUnmanaged(*const Object, void) = .empty,
    marked: std.AutoHashMapUnmanaged(*const Object, void) = .empty, // walked by `openBranches`
    kept_open: std.AutoHashMapUnmanaged(*const Object, void) = .empty, // allOf's branches and their $ref targets: open
    targets: std.ArrayList(Target) = .empty,
    bases: std.ArrayList(Base) = .empty, // targets[i] is under bases[i]
    multiples: usize = 0,
    hops: std.ArrayList(Hop) = .empty,
    landings: std.ArrayList(Landing) = .empty,
    cycle: std.ArrayList(Frame) = .empty,
    visits: usize = 0,
    refusal: Refusal = undefined,

    fn run(c: *Checker, schema: Value) Fail![:0]const u8 {
        switch (schema) {
            .object => {},
            .bool => |b| if (!b) return c.stop("", "the schema must be an object or true, not false"),
            else => return c.stop("", try std.fmt.allocPrint(c.a, "the schema must be an object or true, not {s}", .{kindName(schema)})),
        }
        const given: Value = if (schema == .bool) .{ .object = try json.newObject(c.a) } else schema;
        const size = compactSize(given);
        if (size > max_bytes)
            return c.stop("", try std.fmt.allocPrint(c.a, "the schema is {d} bytes as compact JSON, above the {d} the server takes", .{ size, max_bytes }));
        try c.document(given, 1);
        if (given.object.contains("x-guidance")) {
            try c.push(.{ .key = "x-guidance" });
            return c.refuse("x-guidance", "the server sets it (lenient, the escapes and the whitespace are fixed)", .{});
        }
        const root = (try json.deepCopy(c.a, given)).object;
        const base: Base = .{ .node = root, .depth = 0 };
        try c.openBranches(.{ .object = root }, root);
        try c.node(.{ .object = root }, base);
        var i: usize = 0;
        while (i < c.targets.items.len) : (i += 1) { // what the $ref lead to, walked like any schema node (once)
            c.path.clearRetainingCapacity();
            try c.path.appendSlice(c.a, c.targets.items[i].steps);
            try c.node(c.targets.items[i].value, c.bases.items[i]);
        }
        if (c.multiples > 0) {
            c.path.clearRetainingCapacity();
            try c.cycle.append(c.a, .{ .node = root, .start = 0, .again = false });
            try c.land(.{ .object = root }, base);
            if (c.cycle.pop().?.again) try c.repeated(0);
        }
        const body = try json.stringify(c.a, .{ .object = root }, .{ .ascii = false, .compact = true });
        const head = "{\"x-guidance\":" ++ x_guidance;
        const parts: []const []const u8 = if (root.count() == 0) &.{ head, "}" } else &.{ head, ",", body[1..] };
        return std.mem.concatWithSentinel(c.a, u8, parts, 0);
    }

    /// The client's whole schema, data included: no object or array deeper than max_depth, no string or key that is
    /// not Unicode text (a lone surrogate from a \ud800-style escape: llguidance takes UTF-8 only)
    /// and no NaN or Infinity (the server's parser takes them as Python's does; the text would not be JSON).
    /// クライアントの schema の全体 (データも): max_depth より深い object・配列がないこと、Unicode の文でない文字列・キー
    /// (\ud800 のようなエスケープから来た片方だけの surrogate) がないこと (llguidance は UTF-8 だけを受ける)、
    /// NaN と Infinity がないこと (サーバーの parser は Python と同じく受けるが、文が JSON でなくなる)。
    fn document(c: *Checker, v: Value, depth: usize) Fail!void {
        switch (v) {
            .string => |s| if (!std.unicode.utf8ValidateSlice(s)) return c.notText("string"),
            .float => |f| if (!std.math.isFinite(f)) {
                const at = try c.pointer();
                return c.stop(at, try std.fmt.allocPrint(c.a, "the number at {s} is {s}, which JSON does not have", .{ at, if (std.math.isNan(f)) "NaN" else if (f > 0) "Infinity" else "-Infinity" }));
            },
            .array => |items| {
                if (depth > max_depth) return c.tooDeep();
                for (items, 0..) |x, i| {
                    try c.push(.{ .index = i });
                    try c.document(x, depth + 1);
                    c.pop();
                }
            },
            .object => |o| {
                if (depth > max_depth) return c.tooDeep();
                for (o.keys(), o.values()) |k, x| {
                    try c.push(.{ .key = k });
                    if (!std.unicode.utf8ValidateSlice(k)) return c.notText("key");
                    try c.document(x, depth + 1);
                    c.pop();
                }
            },
            else => {},
        }
    }

    fn tooDeep(c: *Checker) Fail {
        const at = try c.pointer();
        return c.stop(at, try std.fmt.allocPrint(c.a, "the schema nests objects and arrays deeper than {d} levels at {s}", .{ max_depth, at }));
    }

    fn notText(c: *Checker, what: []const u8) Fail {
        const at = try c.pointer();
        return c.stop(at, try std.fmt.allocPrint(c.a, "the {s} at {s} holds a lone surrogate (a \\ud800-\\udfff escape without its pair), which is not Unicode text", .{ what, at }));
    }

    /// One schema node at the place `c.path` names: its keywords, then the schema positions under it. A node is
    /// checked once, however many $ref lead to it.
    /// `c.path` の場所の schema の節点 1 つ: そのキーワード、次にその下の schema の位置。いくつの $ref が指しても 1 回だけ。
    fn node(c: *Checker, v: Value, base: Base) Fail!void {
        const o = switch (v) {
            .object => |o| o,
            else => return, // true and false; anything else is llguidance's to refuse
        };
        if ((try c.seen.getOrPut(c.a, o)).found_existing) return;
        const here: Base = if (hasId(o)) .{ .node = o, .depth = c.path.items.len } else base;
        try c.keywords(o, here);
        try c.positions(o, here);
    }

    fn keywords(c: *Checker, o: *Object, base: Base) Fail!void {
        if (o.get("$schema")) |s| if (!(s == .string and std.mem.eql(u8, std.mem.trimEnd(u8, s.string, "#"), draft_2020_12)))
            return c.refuseKey("$schema", "only " ++ draft_2020_12 ++ " is taken (or no $schema)", .{});
        if (o.contains("nullable"))
            return c.refuseKey("nullable", "is OpenAPI's, not JSON Schema's (llguidance would ignore it and write no null): write \"type\": [..., \"null\"]", .{});
        if (o.contains("additionalItems"))
            return c.refuseKey("additionalItems", "is not taken: write prefixItems and items (2020-12)", .{});
        if (o.get("items")) |it| if (it == .array)
            return c.refuseKey("items", "is an array: write prefixItems (2020-12)", .{});
        if (o.get("$ref")) |r| {
            const t = try c.resolve(r, base);
            try c.targets.append(c.a, t);
            try c.bases.append(c.a, base);
        }
        for ([_][]const u8{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" }) |k| if (o.get(k)) |n| if (n == .int and pastDouble(n.int))
            return c.refuseKey(k, "{s} is past 2^53 in absolute value, where llguidance rounds it to a double", .{n.int});
        if (o.get("multipleOf")) |m| {
            if (try multipleRefusal(c.a, m)) |why| return c.refuseKey("multipleOf", "{s}", .{why});
            c.multiples += 1;
        }
        for ([_][]const u8{ "const", "enum" }) |k| if (o.get(k)) |d| {
            try c.push(.{ .key = k });
            try c.data(k, d);
            c.pop();
        };
        // an object that declares properties and says nothing of other keys takes only its declared keys, but for an
        // allOf's branches (`openBranches`). One without properties keeps any keys, so {"type": "object"} still
        // writes any object.
        // properties を宣言してほかのキーについて何も書いていない object は宣言したキーだけ (allOf の枝は除く:
        // `openBranches`)。properties のない object はどのキーも受けるので、{"type": "object"} は今までどおり任意の object。
        if (o.contains("properties") and !o.contains("additionalProperties") and !o.contains("patternProperties") and !c.kept_open.contains(o))
            try o.put(c.a, "additionalProperties", .{ .bool = false });
        for ([_][]const u8{ "minProperties", "maxProperties" }) |k| if (o.contains(k) and open(o))
            return c.refuseKey(k, "needs additionalProperties false and no patternProperties (an open object can write one key twice, and llguidance counts it twice)", .{});
    }

    /// The schema positions under a node, in its keys' order: the values of properties, patternProperties, $defs and
    /// definitions, items and additionalProperties as schemas, the elements of prefixItems, allOf, anyOf and oneOf.
    /// Nothing else is read as a schema: not const, enum, default or examples (data), not the property names, not a
    /// keyword llguidance refuses or ignores.
    /// 節点の下の schema の位置 (キーの順): properties・patternProperties・$defs・definitions の値、schema としての items と
    /// additionalProperties、prefixItems・allOf・anyOf・oneOf の要素。ほかは schema として読まない: const・enum・default・
    /// examples (データ)、property の名前、llguidance が断るか無視するキーワード。
    fn positions(c: *Checker, o: *Object, base: Base) Fail!void {
        for (o.keys(), o.values()) |k, v| {
            const shape = position(k) orelse continue;
            try c.push(.{ .key = k });
            switch (shape) {
                .one => try c.node(v, base),
                .map => if (v == .object) for (v.object.keys(), v.object.values()) |name, sub| {
                    try c.push(.{ .key = name });
                    try c.node(sub, base);
                    c.pop();
                },
                .list => if (v == .array) for (v.array, 0..) |sub, i| {
                    try c.push(.{ .index = i });
                    try c.node(sub, base);
                    c.pop();
                },
            }
            c.pop();
        }
    }

    /// const and enum data: only the integers' size is looked at (past 2^53 they are rounded to doubles); nothing in
    /// it is read as a keyword.
    /// const と enum のデータ: 見るのは整数の大きさだけ (2^53 を超えると double に丸められる)。中のものをキーワード
    /// として読まない。
    fn data(c: *Checker, keyword: []const u8, d: Value) Fail!void {
        switch (d) {
            .int => |t| if (pastDouble(t)) return c.refuse(keyword, "the integer {s} is past 2^53 in absolute value, where llguidance rounds it to a double", .{t}),
            .array => |items| for (items, 0..) |x, i| {
                try c.push(.{ .index = i });
                try c.data(keyword, x);
                c.pop();
            },
            .object => |o| for (o.keys(), o.values()) |k, x| {
                try c.push(.{ .key = k });
                try c.data(keyword, x);
                c.pop();
            },
            else => {},
        }
    }

    /// A $ref's target: "#" is the base itself, "#/$defs/<name>" and "#/definitions/<name>" a value of the
    /// base's $defs or definitions (the name one JSON Pointer token: ~0 and ~1, no / and no %, since a percent-encoded
    /// name would be read one way here and maybe another by llguidance's resolver). Anything else is refused, as is
    /// a name the base does not have.
    /// $ref の先: "#" は base そのもの、"#/$defs/<名前>" と "#/definitions/<名前>" は base の $defs か definitions
    /// の値 (名前は JSON Pointer の 1 つの token: ~0 と ~1。/ と % はなし。percent-encoding はここと llguidance の resolver
    /// で読み方が違いうる)。ほかの形と、base にない名前は断る。
    fn resolve(c: *Checker, r: Value, base: Base) Fail!Target {
        try c.push(.{ .key = "$ref" });
        const s = switch (r) {
            .string => |s| s,
            else => return c.refuse("$ref", "must be a string", .{}),
        };
        const named = switch (try lookup(c.a, s, base.node)) {
            .named => |n| n,
            .bad_form => return c.refuse("$ref", "must be \"#\", \"#/$defs/<name>\" or \"#/definitions/<name>\", not \"{s}\"", .{s}),
            .missing => return c.refuse("$ref", "names {s}, which the schema does not have", .{s}),
        };
        c.pop();
        var steps: std.ArrayList(Step) = .empty;
        try steps.appendSlice(c.a, c.path.items[0..base.depth]);
        if (named.container) |k| {
            try steps.append(c.a, .{ .key = k });
            try steps.append(c.a, .{ .key = named.name });
        }
        return .{ .value = named.value, .steps = steps.items };
    }

    /// The exception to closing objects, found before the check (which needs to know whether a node is closed): the
    /// elements of every allOf, and the node a $ref from one leads to (and on along $ref), stay open. Closing each
    /// branch of an allOf would make the intersection refuse every key another branch declares: {"allOf":
    /// [{"properties": {"a": ...}}, {"properties": {"b": ...}}]} would write only {}. An open node still writes each
    /// key one way only, since the x-guidance has no u escape (an extra key cannot spell a declared one otherwise).
    /// 閉じることの例外。検査より先に求める (検査は節点が閉じているかを知る必要がある): 全部の allOf の要素と、そこから $ref で
    /// 至る節点 ($ref をたどった先も) は open のまま。allOf の枝をそれぞれ閉じると、交わりはほかの枝が宣言したキーを全部断る:
    /// {"allOf": [{"properties": {"a": ...}}, {"properties": {"b": ...}}]} は {} しか書けなくなる。open な節点でもキーの綴りは
    /// 1 通り (x-guidance に u のエスケープがないので、追加のキーが宣言したキーを別の綴りで書けない)。
    fn openBranches(c: *Checker, v: Value, base: *Object) Allocator.Error!void {
        const o = switch (v) {
            .object => |o| o,
            else => return,
        };
        if ((try c.marked.getOrPut(c.a, o)).found_existing) return;
        const here = if (hasId(o)) o else base;
        for (o.keys(), o.values()) |k, sub| {
            const shape = position(k) orelse continue;
            switch (shape) {
                .one => try c.openBranches(sub, here),
                .map => if (sub == .object) for (sub.object.values()) |x| try c.openBranches(x, here),
                .list => if (sub == .array) for (sub.array) |x| {
                    if (std.mem.eql(u8, k, "allOf")) try c.keepOpen(x, here);
                    try c.openBranches(x, here);
                },
            }
        }
    }

    /// An allOf element kept open, then the node each $ref leads to from there (a $ref the check will refuse ends it).
    fn keepOpen(c: *Checker, v: Value, base: *Object) Allocator.Error!void {
        var at = v;
        var b = base;
        while (at == .object) {
            const o = at.object;
            if ((try c.kept_open.getOrPut(c.a, o)).found_existing) return;
            if (hasId(o)) b = o;
            const r = o.get("$ref") orelse return;
            if (r != .string) return;
            at = switch (try lookup(c.a, r.string, b)) {
                .named => |n| n.value,
                else => return,
            };
        }
    }

    /// The multipleOf walk: from the root along the positions that hold a value of the output (not $defs)
    /// and through every $ref, the place each multipleOf lands on. Two that can land on one value are refused, the
    /// alternatives of an anyOf or oneOf included (the walk does not try to prove two branches are never
    /// intersected). A $ref cycle is followed once, and what it reached lands anywhere, since the cycle repeats it.
    /// multipleOf の歩き方: 根から、出力の値を持つ位置 ($defs は除く) と全部の $ref を通って、各 multipleOf の着地を
    /// 求める。1 つの値に着地しうる 2 つは断る。anyOf・oneOf の枝どうしも (2 つの枝が交わらないことは証明しない)。
    /// $ref の循環は 1 回だけたどり、そこで届いたものはどこにでも着地するとみなす (循環が繰り返すので)。
    fn land(c: *Checker, v: Value, base: Base) Fail!void {
        const o = switch (v) {
            .object => |o| o,
            else => return,
        };
        c.visits += 1;
        if (c.visits > max_visits)
            return c.refuse("multipleOf", "the schema's $ref lead to more than {d} schema nodes, past what the server checks multipleOf over", .{max_visits});
        const here: Base = if (hasId(o)) .{ .node = o, .depth = c.path.items.len } else base;
        if (o.contains("multipleOf")) try c.landing();
        for (o.keys(), o.values()) |k, sub| {
            if (std.mem.eql(u8, k, "$ref")) {
                try c.follow(sub, here);
                continue;
            }
            const shape = position(k) orelse continue;
            if (std.mem.eql(u8, k, "$defs") or std.mem.eql(u8, k, "definitions")) continue;
            try c.push(.{ .key = k });
            switch (shape) {
                .one => try c.landUnder(sub, here, if (std.mem.eql(u8, k, "items")) .any_index else .any_key),
                .map => if (sub == .object) for (sub.object.keys(), sub.object.values()) |name, x| {
                    try c.push(.{ .key = name });
                    try c.landUnder(x, here, if (std.mem.eql(u8, k, "properties")) .{ .key = name } else .any_key);
                    c.pop();
                },
                .list => if (sub == .array) for (sub.array, 0..) |x, i| {
                    try c.push(.{ .index = i });
                    if (std.mem.eql(u8, k, "prefixItems")) try c.landUnder(x, here, .{ .index = i }) else try c.land(x, here);
                    c.pop();
                },
            }
            c.pop();
        }
    }

    fn landUnder(c: *Checker, v: Value, base: Base, hop: Hop) Fail!void {
        try c.hops.append(c.a, hop);
        try c.land(v, base);
        _ = c.hops.pop();
    }

    fn landing(c: *Checker) Fail!void {
        try c.push(.{ .key = "multipleOf" });
        const new: Landing = .{ .pointer = try c.pointer(), .hops = try c.a.dupe(Hop, c.hops.items), .anywhere = false };
        for (c.landings.items) |old| if (meet(old, new)) return c.twice(new.pointer, old.pointer);
        if (c.landings.items.len == max_landings)
            return c.refuse("multipleOf", "lands on a value after {d} others, past what the server compares", .{max_landings});
        try c.landings.append(c.a, new);
        c.pop();
    }

    fn follow(c: *Checker, r: Value, base: Base) Fail!void {
        const t = try c.resolve(r, base);
        const o = switch (t.value) {
            .object => |o| o,
            else => return,
        };
        for (c.cycle.items) |*f| if (f.node == o) {
            f.again = true;
            return;
        };
        const saved = try c.a.dupe(Step, c.path.items);
        c.path.clearRetainingCapacity();
        try c.path.appendSlice(c.a, t.steps);
        try c.cycle.append(c.a, .{ .node = o, .start = c.landings.items.len, .again = false });
        try c.land(t.value, base);
        const f = c.cycle.pop().?;
        if (f.again) try c.repeated(f.start);
        c.path.clearRetainingCapacity();
        try c.path.appendSlice(c.a, saved);
    }

    /// The landings from `start` on are repeated by a $ref cycle: they land anywhere, so any other landing meets them.
    /// `start` 以降の着地は $ref の循環で繰り返される: どこにでも着地するので、ほかのどの着地とも出会う。
    fn repeated(c: *Checker, start: usize) Fail!void {
        const all = c.landings.items;
        for (all[start..]) |*l| l.anywhere = true;
        if (start == all.len or all.len < 2) return;
        return c.twice(all[start].pointer, all[if (start > 0) 0 else 1].pointer);
    }

    fn twice(c: *Checker, at: []const u8, other: []const u8) Fail {
        return c.stop(at, try std.fmt.allocPrint(c.a, "multipleOf at {s}: can land on the same value as {s}; write one multipleOf for a value (llguidance's least common multiple of two wraps)", .{ at, other }));
    }

    fn push(c: *Checker, s: Step) Allocator.Error!void {
        try c.path.append(c.a, s);
    }

    fn pop(c: *Checker) void {
        _ = c.path.pop();
    }

    /// `c.path` as a JSON Pointer (RFC 6901: ~ written ~0, / written ~1), empty at the root.
    fn pointer(c: *Checker) Allocator.Error![]const u8 {
        var out: std.Io.Writer.Allocating = .init(c.a);
        for (c.path.items) |s| writeStep(&out.writer, s) catch return error.OutOfMemory;
        return out.toOwnedSlice();
    }

    /// Refuses `keyword`, the last step of `c.path`: "keyword at /pointer: why".
    fn refuse(c: *Checker, keyword: []const u8, comptime why: []const u8, args: anytype) Fail {
        const at = try c.pointer();
        return c.stop(at, try std.fmt.allocPrint(c.a, "{s} at {s}: " ++ why, .{ keyword, at } ++ args));
    }

    fn refuseKey(c: *Checker, keyword: []const u8, comptime why: []const u8, args: anytype) Fail {
        try c.push(.{ .key = keyword });
        return c.refuse(keyword, why, args);
    }

    fn stop(c: *Checker, at: []const u8, message: []const u8) Fail {
        c.refusal = .{ .pointer = at, .message = message };
        return error.Refused;
    }
};

const Shape = enum { one, map, list };

/// The schema positions a keyword holds (see `Checker.positions`), null for a keyword that holds none.
fn position(k: []const u8) ?Shape {
    const table = [_]struct { []const u8, Shape }{
        .{ "properties", .map }, .{ "patternProperties", .map },    .{ "$defs", .map },        .{ "definitions", .map },
        .{ "items", .one },      .{ "additionalProperties", .one }, .{ "prefixItems", .list }, .{ "allOf", .list },
        .{ "anyOf", .list },     .{ "oneOf", .list },
    };
    for (table) |t| if (std.mem.eql(u8, t[0], k)) return t[1];
    return null;
}

fn hasId(o: *Object) bool {
    const id = o.get("$id") orelse return false;
    return id == .string;
}

/// An object node that takes keys besides its properties: patternProperties, an additionalProperties other than false,
/// or none at all (once objects with properties are closed, a node without properties).
/// properties のほかのキーも受ける object の節点: patternProperties がある、additionalProperties が false でない、
/// またはない (properties のある object を閉じた後では、properties のない節点)。
fn open(o: *Object) bool {
    if (o.contains("patternProperties")) return true;
    const extra = o.get("additionalProperties") orelse return true;
    return !(extra == .bool and !extra.bool);
}

/// Whether an integer's decimal text is past 2^53 = 9007199254740992 in absolute value (the server's JSON keeps an
/// integer as its exact decimal text, without leading zeros).
/// 整数の 10 進の字句が、絶対値で 2^53 = 9007199254740992 を超えるか (サーバーの JSON は整数を正確な字句で持つ。頭の 0 はない)。
fn pastDouble(t: []const u8) bool {
    const digits = if (t.len > 0 and t[0] == '-') t[1..] else t;
    const limit = "9007199254740992";
    return digits.len > limit.len or (digits.len == limit.len and std.mem.order(u8, digits, limit) == .gt);
}

/// Why the server refuses a single multipleOf, or null when it takes it: an integer from 1 to max_multiple, or a
/// decimal llguidance reads as such a coefficient with at most max_multiple_places decimal places.
/// サーバーが 1 つの multipleOf を断る理由。受けるなら null: 1 から max_multiple の整数か、llguidance が読むとそういう係数で
/// 小数の桁が max_multiple_places までになる小数。
fn multipleRefusal(a: Allocator, m: Value) Allocator.Error!?[]const u8 {
    const rule = std.fmt.comptimePrint("must be an integer from 1 to {d}, or a decimal llguidance reads as such an integer times 10^-k with k at most {d}", .{ max_multiple, max_multiple_places });
    const text = try json.stringify(a, m, .{ .ascii = false, .compact = true });
    switch (m) {
        .int => |t| {
            const n = std.fmt.parseInt(u64, t, 10) catch 0;
            return if (n >= 1 and n <= max_multiple) null else try std.fmt.allocPrint(a, rule ++ ", not {s}", .{text});
        },
        .float => |f| {
            if (!(f > 0) or !std.math.isFinite(f)) return try std.fmt.allocPrint(a, rule ++ ", not {s}", .{text});
            const d = llgDecimal(f);
            if (d.coef >= 1 and d.coef <= max_multiple and d.exp <= max_multiple_places) return null;
            var buf: [40]u8 = undefined;
            const coef = if (d.coef < 1e18) try std.fmt.allocPrint(a, "{d}", .{@as(u64, @intFromFloat(d.coef))}) else json.floatRepr(&buf, d.coef);
            if (d.exp == 0) return try std.fmt.allocPrint(a, rule ++ "; llguidance reads {s} as {s}", .{ text, coef });
            return try std.fmt.allocPrint(a, rule ++ "; llguidance reads {s} as {s} x 10^-{d}", .{ text, coef, d.exp });
        },
        else => return try std.fmt.allocPrint(a, rule ++ ", not {s}", .{text}),
    }
}

/// llguidance's reading of a positive, finite multipleOf, double for double as Decimal::try_from does it
/// (parser/src/json/numeric.rs, after serde's as_f64 and abs): times ten until the fraction is zero, then
/// Decimal::new's trailing zeros taken off. A coefficient past u32 is one llguidance refuses ("too many digits").
/// The loop ends: a double of 2^52 or more has no fraction, and ten times one below that is still finite.
/// 正で有限の multipleOf を llguidance が読んだもの。Decimal::try_from (parser/src/json/numeric.rs。serde の as_f64 と abs の
/// 後) と double の計算を 1 つずつ同じに: 小数部が 0 になるまで 10 倍し、Decimal::new のように末尾の 0 を取る。係数が u32 を
/// 超えるものは llguidance が断る ("too many digits")。2^52 以上の double に小数部はなく、それ未満の 10 倍も有限なので終わる。
fn llgDecimal(f: f64) struct { coef: f64, exp: u32 } {
    var v = f;
    var exp: u32 = 0;
    while (v - @trunc(v) != 0) {
        v *= 10.0;
        exp += 1;
    }
    if (v > std.math.maxInt(u32)) return .{ .coef = v, .exp = exp };
    var coef: u32 = @intFromFloat(v);
    if (coef == 0) return .{ .coef = 0, .exp = 0 };
    while (exp > 0 and coef % 10 == 0) {
        coef /= 10;
        exp -= 1;
    }
    return .{ .coef = @floatFromInt(coef), .exp = exp };
}

/// What a $ref string names under `base`, without refusing (see `Checker.resolve`): the target, and the two steps
/// from the base to it ("#" has none).
const Named = struct { value: Value, container: ?[]const u8 = null, name: []const u8 = "" };
const Lookup = union(enum) { named: Named, bad_form, missing };

fn lookup(a: Allocator, s: []const u8, base: *Object) Allocator.Error!Lookup {
    if (std.mem.eql(u8, s, "#")) return .{ .named = .{ .value = .{ .object = base } } };
    const container: []const u8 = for ([_][]const u8{ "$defs", "definitions" }) |k| {
        if (s.len > 3 + k.len and std.mem.startsWith(u8, s, "#/") and std.mem.startsWith(u8, s[2..], k) and s[2 + k.len] == '/') break k;
    } else return .bad_form;
    const name = try pointerToken(a, s[3 + container.len ..]) orelse return .bad_form;
    const defs = base.get(container);
    const found = if (defs != null and defs.? == .object) defs.?.object.get(name) else null;
    return .{ .named = .{ .value = found orelse return .missing, .container = container, .name = name } };
}

/// One JSON Pointer token decoded (~0 is ~, ~1 is /), or null when it is not a single token the server reads one way:
/// a /, a %, or a ~ not followed by 0 or 1.
/// JSON Pointer の 1 つの token を解いたもの (~0 は ~、~1 は /)。サーバーが 1 通りに読める 1 つの token でなければ null:
/// / か % がある、~ の後が 0 でも 1 でもない。
fn pointerToken(a: Allocator, t: []const u8) Allocator.Error!?[]const u8 {
    if (std.mem.indexOfAny(u8, t, "/%") != null) return null;
    if (std.mem.indexOfScalar(u8, t, '~') == null) return t;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < t.len) : (i += 1) {
        if (t[i] != '~') {
            try out.append(a, t[i]);
            continue;
        }
        if (i + 1 == t.len or (t[i + 1] != '0' and t[i + 1] != '1')) return null;
        try out.append(a, if (t[i + 1] == '0') '~' else '/');
        i += 1;
    }
    return out.items;
}

fn writeStep(w: *std.Io.Writer, s: Step) std.Io.Writer.Error!void {
    try w.writeByte('/');
    switch (s) {
        .index => |i| try w.print("{d}", .{i}),
        .key => |k| for (k) |ch| switch (ch) {
            '~' => try w.writeAll("~0"),
            '/' => try w.writeAll("~1"),
            else => try w.writeByte(ch),
        },
    }
}

/// Whether two landings can be one value of the output: either is repeated by a cycle, or they have as many steps
/// and each pair can be the same key or index (any key meets every key, any index every index).
/// 2 つの着地が出力の 1 つの値になりうるか: どちらかが循環で繰り返される、または歩数が同じで、各歩が同じキーか番号に
/// なりうる (任意のキーはどのキーとも、任意の番号はどの番号とも出会う)。
fn meet(x: Landing, y: Landing) bool {
    if (x.anywhere or y.anywhere) return true;
    if (x.hops.len != y.hops.len) return false;
    for (x.hops, y.hops) |p, q| if (!hopMeet(p, q)) return false;
    return true;
}

fn hopMeet(p: Hop, q: Hop) bool {
    return switch (p) {
        .key => |k| switch (q) {
            .key => |l| std.mem.eql(u8, k, l),
            .any_key => true,
            else => false,
        },
        .any_key => q == .key or q == .any_key,
        .index => |i| switch (q) {
            .index => |j| i == j,
            .any_index => true,
            else => false,
        },
        .any_index => q == .index or q == .any_index,
    };
}

fn compactSize(v: Value) u64 {
    var buf: [512]u8 = undefined;
    var d: std.Io.Writer.Discarding = .init(&buf);
    json.write(&d.writer, v, .{ .ascii = false, .compact = true }) catch unreachable; // a discarding writer never fails
    return d.fullCount();
}

fn kindName(v: Value) []const u8 {
    return switch (v) {
        .null => "null",
        .bool => "a boolean",
        .int, .float => "a number",
        .string => "a string",
        .array => "an array",
        .object => "an object",
    };
}
