// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! Feature probes from tools/template_parity.py; expected strings come from Python jinja2 in transformers' environment.
const std = @import("std");
const template = @import("template.zig");

const Outcome = struct { text: ?[]const u8, raised: ?[]const u8 };

fn render(source: []const u8, vars: []const u8, diag: *template.Diag) ![]u8 {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, vars, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const generation = if (obj.get("add_generation_prompt")) |g| g.bool else true;
    return template.renderChat(a, source, obj.get("messages") orelse .null, obj.get("tools") orelse .null, parsed.value, generation, diag);
}

fn expect(source: []const u8, vars: []const u8, want: Outcome) !void {
    const a = std.testing.allocator;
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    const out = render(source, vars, &diag) catch |err| {
        if (err != error.TemplateFailed or want.text != null) {
            std.debug.print("{s}\nfailed: {s}\n", .{ source, diag.msg });
            return error.TestUnexpectedError;
        }
        if (want.raised) |msg| {
            try std.testing.expect(diag.raised);
            try std.testing.expectEqualStrings(msg, diag.msg);
        }
        return;
    };
    defer a.free(out);
    if (want.text == null) {
        std.debug.print("{s}\nrendered instead of failing: {s}\n", .{ source, out });
        return error.TestExpectedError;
    }
    try std.testing.expectEqualStrings(want.text.?, out);
}

const shared = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}, {\"role\": \"assistant\", \"content\": \" Hello \"}, {\"role\": \"user\", \"content\": \"Bye\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}], \"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}";

const probes = [_]struct { []const u8, ?[]const u8, ?[]const u8 }{
    .{ "{% for m in messages %}{{ loop.index }}/{{ loop.length }} {{ loop.revindex0 }} {{ loop.first }} {{ loop.last }} {{ loop.previtem.role if loop.previtem }}|{{ loop.nextitem.role if loop.nextitem is defined }}\n{% endfor %}", "1/3 2 True False |assistant\n2/3 1 False False user|user\n3/3 0 False True assistant|\n", null },
    .{ "{% set x = 1 %}{% for i in [1, 2] %}{{ x }}{% set x = x + i %}{{ x }}{% endfor %}{{ x }}", "12131", null },
    .{ "{% set ns = namespace(n=0, s='') %}{% for i in range(5) if i is odd %}{% set ns.n = ns.n + i %}{% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }}:{{ ns.s }}:{{ ns }}", "4:13:<Namespace {'n': 4, 's': '13'}>", null },
    .{ "{% macro m(a, b=2, c='x') %}[{{ a }}|{{ b }}|{{ c }}|{{ d }}]{% endmacro %}{{ m(1) }}{{ m(1, 3) }}{{ m(1, c='z') }}{{ m() }}{% set d = 'late' %}{{ m(4) }}", "[1|2|x|][1|3|x|][1|2|z|][|2|x|][4|2|x|late]", null },
    .{ "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}", null, null },
    .{ "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, b=2) }}", null, null },
    .{ "{% for i in [3, 1, 2] %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% else %}empty{% endfor %}", "32", null },
    .{ "{% for i in [] %}x{% else %}empty{% endfor %}|{% for i in [1] %}{% continue %}{% else %}else-after-continue{% endfor %}", "empty|else-after-continue", null },
    .{ "{%- if true -%}   trimmed   {%- endif -%}  |  {% if true %}\n  kept\n{% endif %}\n    {% if true %}lstrip{% endif %}  {%+ if true %}plus{% endif %}  {{- ' dash' }}  {#- comment -#}  end", "trimmed|    kept\nlstrip  plus dashend", null },
    .{ "{{ 'a\\tb\\n\\x41\u{e9}\\101\\\\q' if false else 'esc:\\t\\x41\u{e9}\\101\\q' }}|{{ \"it's\" }}|{{ 'say \"hi\"' }}", "esc:\tA\u{e9}A\\q|it's|say \"hi\"", null },
    .{ "{{ [1, 'a', none, true, 1.5, [2], {'k': 'v'}, ('t',), (1, 2)] }}|{{ {'a': 1, 'b': [1.0, 1e16, -0.0]} }}|{{ (1,) | string }}", "[1, 'a', None, True, 1.5, [2], {'k': 'v'}, ('t',), (1, 2)]|{'a': 1, 'b': [1.0, 1e+16, -0.0]}|(1,)", null },
    .{ "{{ messages | tojson }}|{{ tools | tojson(indent=2) }}|{{ {'b': 1, 'a': '\u{e9}'} | tojson(sort_keys=true, ensure_ascii=true) }}|{{ [1, [2]] | tojson(separators=(',', ':')) }}", "[{\"role\": \"user\", \"content\": \"Hi\"}, {\"role\": \"assistant\", \"content\": \" Hello \"}, {\"role\": \"user\", \"content\": \"Bye\"}]|[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"weather\",\n      \"description\": \"Get weather\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"city\": {\n            \"type\": \"string\"\n          }\n        },\n        \"required\": [\n          \"city\"\n        ]\n      }\n    }\n  }\n]|{\"a\": \"\\u00e9\", \"b\": 1}|[1,[2]]", null },
    .{ "{{ 'Hello World'[::-1] }}|{{ 'h\u{e9}llo'[1:4] }}|{{ [1, 2, 3, 4][-2:] }}|{{ 'abc'[5:] }}|{{ (1, 2, 3)[::2] }}|{{ 'abc'[-1] }}|{{ 'abc'[10] is defined }}", "dlroW olleH|\u{e9}ll|[3, 4]||(1, 3)|c|False", null },
    .{ "{{ '  x y  '.strip() }}|{{ 'xxhixx'.strip('x') }}|{{ 'a,b,,c'.split(',') }}|{{ ' a  b '.split() }}|{{ 'a b c'.split(None, 1) }}|{{ 'a b c'.rsplit(' ', 1) }}|{{ 'aaa'.replace('a', 'b', 2) }}|{{ 'abc'.replace('', '-') }}", "x y|hi|['a', 'b', '', 'c']|['a', 'b']|['a', 'b c']|['a b', 'c']|bba|-a-b-c-", null },
    .{ "{{ 'hello'.find('l') }}|{{ 'hello'.rfind('l') }}|{{ 'h\u{e9}llo'.find('l') }}|{{ 'hello'.startswith(('x', 'he')) }}|{{ 'hello'.endswith('lo', 0, 5) }}|{{ 'a-b'.count('-') }}|{{ 'x'.join(['a', 'b']) }}|{{ 'pre_x'.removeprefix('pre_') }}", "2|3|2|True|True|1|axb|x", null },
    .{ "{{ '\u{c6}\u{d8}\u{c5} stra\u{df}e'.lower() }}|{{ '\u{c6}\u{d8}\u{c5} stra\u{df}e'.upper() }}|{{ 'abc' | upper }}|{{ none | upper }}|{{ 'ABC' | lower }}|{{ 42 | string }}|{{ 1.0 | string }}|{{ [1] | string }}", "\u{e6}\u{f8}\u{e5} stra\u{df}e|\u{c6}\u{d8}\u{c5} STRASSE|ABC|NONE|abc|42|1.0|[1]", null },
    .{ "{{ '\u{3a3}\u{391}\u{3a3} \u{39f}\u{394}\u{3a5}\u{3a3}\u{3a3}\u{395}\u{3a5}\u{3a3} A\u{3a3}\\'b' | lower }}|{{ ['\u{3a3}a', '\u{3c3}b', '\u{3a3}A'] | sort }}|{{ {'\u{c4}rger': 1, '\u{e4}pfel': 2, 'Zed': 3} | dictsort }}", "\u{3c3}\u{3b1}\u{3c2} \u{3bf}\u{3b4}\u{3c5}\u{3c3}\u{3c3}\u{3b5}\u{3c5}\u{3c2} a\u{3c3}'b|['\u{3a3}a', '\u{3a3}A', '\u{3c3}b']|[('Zed', 3), ('\u{e4}pfel', 2), ('\u{c4}rger', 1)]", null },
    .{ "{{ undefined_name }}|{{ undefined_name is defined }}|{{ undefined_name | default('dflt') }}|{{ '' | default('empty', true) }}|{{ none | default('n') }}|{{ undefined_name | length }}|{{ undefined_name ~ 'x' }}", "|False|dflt|empty|None|0|x", null },
    .{ "{{ x.y }}", null, null },
    .{ "{{ 'a' + undefined_name }}", null, null },
    .{ "{{ messages[0].nope.deeper }}", null, null },
    .{ "{{ messages[0]['role'] }}|{{ messages[0].role }}|{{ messages.0.role }}|{{ messages[0].items is defined }}|{{ messages[0].get('role') }}|{{ messages[0].get('nope', 'd') }}|{{ messages[0].pop is defined }}", "user|user|user|True|user|d|False", null },
    .{ "{{ [3, 1, 2] | sort }}|{{ ['b', 'A', 'a'] | sort }}|{{ ['b', 'A', 'a'] | sort(case_sensitive=true) }}|{{ ['b', 'a'] | sort(reverse=true) }}|{{ ['a', 'A', 'b'] | unique | list }}|{{ [1, 2] | reverse | list }}|{{ 'abc' | reverse }}", "[1, 2, 3]|['A', 'a', 'b']|['A', 'a', 'b']|['b', 'a']|['a', 'b']|[2, 1]|cba", null },
    .{ "{{ {'b': 1, 'A': 2, 'a': 3} | dictsort }}|{{ {'b': 1, 'a': 2} | dictsort(by='value', reverse=true) }}|{{ {'x': 1} | items | list }}|{{ [1, 2, 3] | first }}|{{ [1, 2, 3] | last }}|{{ [] | first is defined }}", "[('A', 2), ('a', 3), ('b', 1)]|[('a', 2), ('b', 1)]|[('x', 1)]|1|3|False", null },
    .{ "{{ ['a', 'b'] | map('upper') | list }}|{{ messages | map(attribute='role') | join(',') }}|{{ messages | selectattr('role', 'equalto', 'user') | list | length }}|{{ [0, 1, 2] | select | list }}|{{ [0, 1, 2] | reject('odd') | list }}|{{ messages | rejectattr('content') | list | length }}", "['A', 'B']|user,assistant,user|2|[1, 2]|[0, 2]|0", null },
    .{ "{{ 'a\\nb\\n\\nc' | indent(2) }}|{{ 'a\\nb' | indent(2, true) }}|{{ 'a\\n\\nb' | indent('> ', blank=true) }}|{{ 'x&<>\"\\'' | e }}|{{ ('<b>' | safe) + '<i>' }}|{{ '<i>' + ('<b>' | safe) }}|{{ 'x' | replace('x', 'y') }}", "a\n  b\n\n  c|  a\n  b|a\n> \n> b|x&amp;&lt;&gt;&#34;&#39;|<b>&lt;i&gt;|&lt;i&gt;<b>|y", null },
    .{ "{{ 1 + 2 }}|{{ 7 // 2 }}|{{ -7 // 2 }}|{{ 7 % 3 }}|{{ -7 % 3 }}|{{ 7 / 2 }}|{{ 2 ** 10 }}|{{ 2 ** -1 }}|{{ 1.5 * 2 }}|{{ -7.5 // 2 }}|{{ -7.5 % 2 }}|{{ true + 1 }}|{{ 'ab' * 2 }}|{{ [1] * 2 }}|{{ 10 / 4 }}|{{ 0.1 + 0.2 }}|{{ 1e300 * 1e10 }}", "3|3|-4|1|2|3.5|1024|0.5|3.0|-4.0|0.5|2|abab|[1, 1]|2.5|0.30000000000000004|inf", null },
    .{ "{{ 1 == 1.0 }}|{{ 'a' < 'b' }}|{{ [1, 2] < [1, 3] }}|{{ 1 < 2 < 3 }}|{{ 3 > 2 > 2 }}|{{ 'a' in 'cat' }}|{{ 'x' not in ['x'] }}|{{ 'k' in {'k': 1} }}|{{ none == none }}|{{ undefined_name == undefined_other }}|{{ (1, 2) == [1, 2] }}", "True|True|True|True|False|True|False|True|True|True|False", null },
    .{ "{{ 1 < 'a' }}", null, null },
    .{ "{{ 'a' in 1 }}", null, null },
    .{ "{{ none is none }}|{{ 1 is number }}|{{ true is number }}|{{ true is integer }}|{{ 1.0 is float }}|{{ 'x' is string }}|{{ {} is mapping }}|{{ {} is sequence }}|{{ 'x' is sequence }}|{{ undefined_name is sequence }}|{{ undefined_name is iterable }}|{{ none is iterable }}|{{ 3 is odd }}|{{ 4 is even }}|{{ 9 is divisibleby 3 }}|{{ 'a' is in 'abc' }}|{{ 1 is eq 1 }}|{{ 2 is gt 1 }}|{{ false is sameas false }}|{{ 'tojson' is filter }}|{{ 'odd' is test }}|{{ x is callable }}|{{ range is callable }}", "True|True|True|False|True|True|True|True|True|True|True|False|True|True|True|True|True|True|True|True|True|True|True", null },
    .{ "{{ range(3) | list }}|{{ range(1, 10, 3) | list }}|{{ range(5, 0, -2) | list }}|{{ range(3) }}|{{ range(0, 10, 2) }}|{{ range(3) | length }}", "[0, 1, 2]|[1, 4, 7]|[5, 3, 1]|range(0, 3)|range(0, 10, 2)|3", null },
    .{ "{% set captured %}  inner {{ 1 + 1 }}  {% endset %}[{{ captured }}]|{% set up | upper %}shout{% endset %}{{ up }}", "[  inner 2  ]|SHOUT", null },
    .{ "{% set a, b = [1, 2] %}{{ a }}{{ b }}|{% for k, v in {'x': 1, 'y': 2} | items %}{{ k }}={{ v }};{% endfor %}|{% for a, (b, c) in [(1, (2, 3))] %}{{ a }}{{ b }}{{ c }}{% endfor %}", "12|x=1;y=2;|123", null },
    .{ "{% set a, b = [1] %}", null, null },
    .{ "{{ raise_exception('custom failure: ' ~ 42) }}", null, "custom failure: 42" },
    .{ "{{ messages[0].content if messages else 'none' }}|{{ 'yes' if '' else 'no' }}|{{ ('a' if false) is defined }}|{{ none or 'fallback' }}|{{ 0 and 'never' }}|{{ not none }}", "Hi|no|False|fallback|0|True", null },
    .{ "{% for i in range(3) %}{{ loop.cycle('odd', 'even') }}{{ loop.changed(i // 2) }}{% endfor %}", "oddTrueevenFalseoddTrue", null },
    .{ "{% generation %}gen {{ 1 }}{% endgeneration %}|{% raw %}{{ raw }} {% not a tag %}{% endraw %}", "gen 1|{{ raw }} {% not a tag %}", null },
    .{ "{% macro outer() %}{{ inner() }}{% endmacro %}{% macro inner() %}inner-ok{% endmacro %}{{ outer() }}", "inner-ok", null },
    .{ "{% for m in messages %}{{ late_var }}{% endfor %}{% set late_var = 'root' %}{{ late_var }}", "root", null },
    .{ "{% if false %}{% set y = 1 %}{% endif %}{{ y }}|{% if true %}{% set y = 2 %}{% endif %}{{ y }}", "|2", null },
    .{ "{{ messages | length }}|{{ messages[-1].role }}|{{ messages[99] is defined }}|{{ messages[0]['nope'] is defined }}|{{ (messages | first).role }}|{{ tools | length if tools else 0 }}|{{ documents }}|{{ bos_token }}{{ eos_token }}", "3|user|False|False|user|1|None|<s></s>", null },
    .{ "{{ messages[0].content.type }}|{{ 'abc'.nope is defined }}|{{ none.x is defined }}|{{ (1).real is defined }}|{{ {}.items() | list }}|{{ {'a': 1}.items() }}|{{ {'a': 1}.keys() | list }}|{{ {'a': 1}.values() | list }}", "|False|False|True|[]|dict_items([('a', 1)])|['a']|[1]", null },
    .{ "{{ [1, 2, 3] | join('-') }}|{{ [1, none, 'x'] | join }}|{{ 'abc' | list }}|{{ {'a': 1, 'b': 2} | list }}|{{ 'h\u{e9}llo' | length }}|{{ {'a': 1} | length }}|{{ 'a b' | trim }}|{{ ' x ' | trim('x ') }}", "1-2-3|1Nonex|['a', 'b', 'c']|['a', 'b']|5|1|a b|", null },
    .{ "{{ x | nofilter }}", null, null },
    .{ "{{ 'a' ~ 1 ~ none ~ 1.5 ~ [1] ~ true }}|{{ -1 }}|{{ -(1) }}|{{ +1 }}|{{ - 1.5 }}|{{ not true }}|{{ 'x' * 0 }}|{{ 1.0 == 1 }}|{{ 1e2 }}|{{ 1_000 }}|{{ 0x1F }}|{{ 1.5e-7 }}|{{ 12.5E3 }}", "a1None1.5[1]True|-1|-1|1|-1.5|False||True|100.0|1000|31|1.5e-07|12500.0", null },
    .{ "{{ dict(a=1, b='x') }}|{{ namespace(a=1).a }}|{{ namespace({'q': 2}).q }}|{% set n = namespace() %}{% set n.v = [1] %}{{ n.v }}", "{'a': 1, 'b': 'x'}|1|2|[1]", null },
    .{ "{% set t = 'x' %}{% set t.attr = 1 %}", null, null },
    .{ "{{ messages[0].content[0] }}|{{ 'abc'[0:2] }}|{{ 'abc'[:-1] }}|{{ 'abc'['upper']() }}", "H|ab|ab|ABC", null },
    .{ "{{ ['b', 'a'] | sort | join }}{{ [{'n': 'b'}, {'n': 'a'}] | sort(attribute='n') | map(attribute='n') | join }}{{ [{'n': 1}] | map(attribute='m', default='d') | join }}", "ababd", null },
    .{ "{% for x in 'ab' %}{{ x }}{{ loop.index0 }}{% endfor %}|{% for k in {'p': 1, 'q': 2} %}{{ k }}{% endfor %}|{% for x in none %}{% endfor %}", null, null },
    .{ "{%- for m in messages -%}\n    {%- set content = m.content | trim -%}\n    {{- '<' + m.role + '>' + content -}}\n{%- endfor %}\nx", "<user>Hi<assistant>Hello<user>Byex", null },
};

test "feature probes match Python jinja2" {
    for (probes) |p| try expect(p[0], shared, .{ .text = p[1], .raised = p[2] });
}

test "engine bridge: NUL content, merged system prompt and generation prompt" {
    try expect("{% for m in messages %}{{m.role}}:{{m.content}}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
        \\{"messages":[{"role":"system","content":"first\n\nsecond"},{"role":"user","content":"a\u0000b"}]}
    , .{ .text = "system:first\n\nseconduser:a\x00bassistant:", .raised = null });
}

test "engine context: thinking switches and effort defaults" {
    const source = "{{ 'on' if enable_thinking else 'off' }}:{{ reasoning_effort|default('unset') }}";
    try expect(source, "{\"enable_thinking\":true,\"reasoning_effort\":\"medium\"}", .{ .text = "on:medium", .raised = null });
    try expect(source, "{\"enable_thinking\":false}", .{ .text = "off:unset", .raised = null });
}

test "numeric members, undefined tools and documents" {
    try expect("{{ messages.0.role }}|{{ tools }}|{{ documents }}|{{ tools is defined }}", "{\"messages\":[{\"role\":\"user\"}]}", .{ .text = "user|None|None|True", .raised = null });
}

test "syntax and unknown filters fail at compile time" {
    try expect("{{ x | nofilter }}", shared, .{ .text = null, .raised = null });
    try expect("{% if %}", shared, .{ .text = null, .raised = null });
    try expect("{% for x in y %}", shared, .{ .text = null, .raised = null });
}

/// Loops over range(...), counted and not built: expected strings from Python jinja2 3.1.6 in transformers' environment
/// (sandboxed, trim and lstrip blocks, loop controls).
const range_probes = [_]struct { []const u8, ?[]const u8, ?[]const u8 }{
    .{ "{% for i in range(2, 9, 3) %}{{ i }}:{{ loop.index0 }}/{{ loop.index }}/{{ loop.length }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }};{% endfor %}", "2:0/1/3/3/2/True/False;5:1/2/3/2/1/False/False;8:2/3/3/1/0/False/True;", null },
    .{ "{% for i in range(3) %}{{ loop.previtem if loop.previtem is defined else 'start' }}-{{ loop.nextitem if loop.nextitem is defined else 'end' }};{% endfor %}|{% for i in range(3) %}{{ loop.nextitem }}{% endfor %}", "start-1;0-2;1-end;|12", null },
    .{ "{% for i in range(5, 0, -2) %}{{ i }}{{ loop.length }}{% endfor %}|{% for i in range(3, 3) %}x{% else %}empty{% endfor %}|{% for i in range(0) %}{% else %}e2{% endfor %}|{% for i in range(3, 0) %}{% else %}e3{% endfor %}", "533313|empty|e2|e3", null },
    .{ "{% for i in range(4) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{% for j in range(i) %}{{ i }}{{ j }}{{ loop.length }}{% endfor %};{% endfor %}", ";202212;", null },
    .{ "{% for i in range(2) %}{{ loop | length }}{{ 'T' if loop else 'F' }}{{ loop }}{% endfor %}", "2T<LoopContext 1/2>2T<LoopContext 2/2>", null },
    .{ "{% macro range(n) %}m{{ n }}{% endmacro %}{% for c in range(2) %}{{ c }}{% endfor %}", "m2", null },
    .{ "{% for i in range(6) if i is even %}{{ i }}{{ loop.length }}{{ loop.last }}{% endfor %}", "03False23False43True", null },
    .{ "{% set s %}{% for i in range(3) %}{{ i }}{% endfor %}{% endset %}{{ s }}|{% macro m() %}{% for i in range(2) %}{{ i }}{% endfor %}{% endmacro %}{{ m() }}|{% for i in range(3) %}{% endfor %}{{ i is defined }}", "012|01|False", null },
    .{ "{% for j in range(1, messages | length) %}{{ messages[j].role }}{% endfor %}|{% for j in range(0, 100000) %}{{ j }}{% break %}{% endfor %}|{% for i in range(-2, 2) %}{{ i }}{% endfor %}", "assistantuser|0|-2-101", null },
    .{ "{% for i in range(2) %}{% for k in range(i, 3) %}{{ loop.revindex }}{% endfor %}{{ loop.cycle('a', 'b') }}{% endfor %}", "321a21b", null },
    .{ "{% for i in range(2) %}{% break %}{% else %}E{% endfor %}|{% for i in range(2) %}{% continue %}{% else %}E{% endfor %}|{% for i in [1, 2] %}{% continue %}{% else %}E{% endfor %}", "E|E|E", null },
    .{ "{% for i in range(1, 2, 0) %}{% endfor %}", null, null }, // jinja2: ValueError: range() arg 3 must not be zero
    .{ "{% for i in range(200000) %}{% endfor %}", null, null }, // jinja2: OverflowError: Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).
    .{ "{% for i in range(stop=3) %}{% endfor %}", null, null }, // jinja2: TypeError: safe_range() got an unexpected keyword argument 'stop'
    .{ "{% for i in range() %}{% endfor %}", null, null }, // jinja2: TypeError: range expected at least 1 argument, got 0
    .{ "{% for i in range('a') %}{% endfor %}", null, null }, // jinja2: TypeError: 'str' object cannot be interpreted as an integer
};

test "range loops match Python jinja2" {
    for (range_probes) |p| try expect(p[0], shared, .{ .text = p[1], .raised = p[2] });
}

/// The text, or the error's message and whether raise_exception raised it, as `source` renders with no variables.
fn expectSame(source: []const u8, text: ?[]const u8, msg: ?[]const u8, raised: bool) !void {
    const a = std.testing.allocator;
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    const out = template.renderChat(a, source, .null, .null, .null, false, &diag) catch |err| {
        if (err != error.TemplateFailed or text != null) {
            std.debug.print("{s}\nfailed: {s}\n", .{ source, diag.msg });
            return error.TestUnexpectedError;
        }
        try std.testing.expectEqualStrings(msg.?, diag.msg);
        try std.testing.expectEqual(raised, diag.raised);
        return;
    };
    defer a.free(out);
    if (text == null) {
        std.debug.print("{s}\nrendered instead of failing: {s}\n", .{ source, out });
        return error.TestExpectedError;
    }
    try std.testing.expectEqualStrings(text.?, out);
}

/// Loops over range(...) and what is near them, with the outputs and errors (message, raised) the engine gave before
/// range loops were counted rather than built.
const range_golden = [_]struct { []const u8, ?[]const u8, ?[]const u8, bool }{
    .{ "{% set ns = namespace(n=0) %}{% macro tick() %}{% set ns.n = ns.n + 1 %}{% endmacro %}{% macro r(n) %}ab{% endmacro %}{% for x in (tick() or r)(2) %}{{ x }}{% endfor %}{{ ns.n }}", "ab1", null, false },
    .{ "{% set ns = namespace(range=range) %}{% macro change() %}{% set ns.range = 'x' %}2{% endmacro %}{% for i in ns.range(change()|length) %}{{ i }}{% endfor %}", "0", null, false },
    .{ "{% for i in range(1) %}{% break %}{% else %}E{% endfor %}|{% for i in range(2) %}{% continue %}{% else %}E{% endfor %}|{% for i in range(2) %}{% if i == 1 %}{% break %}{% endif %}X{% else %}E{% endfor %}|{% for i in range(0) %}{% else %}E{% endfor %}", "E|E|X|E", null, false },
    .{ "{% for i in range('bad', 2, 0, stop=3) %}{% endfor %}", null, "range() takes no keyword arguments", false },
    .{ "{% for i in range(100001) %}{% break %}{% endfor %}", null, "Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).", false },
    .{ "{% for i in range(100000) %}{{ i }}{% break %}{% endfor %}|{% for i in range(200000, 200002) %}{{ i }}{% endfor %}", "0|200000200001", null, false },
    .{ "{% for i in range(stop=raise_exception('kw')) %}{% endfor %}", null, "kw", true },
    .{ "{% for i in range('bad', raise_exception('second')) %}{% endfor %}", null, "second", true },
    .{ "{% for i in range() %}{% endfor %}", null, "range expected 1 to 3 arguments, got 0", false },
    .{ "{% for i in range(1, 2, 3, 4) %}{% endfor %}", null, "range expected 1 to 3 arguments, got 4", false },
    .{ "{% for i in range(1.5) %}{% endfor %}", null, "'float' object cannot be interpreted as an integer", false },
    .{ "{% for i in range('x') %}{% endfor %}", null, "'str' object cannot be interpreted as an integer", false },
    .{ "{% for i in range(nothing) %}{% endfor %}", null, "'Undefined' object cannot be interpreted as an integer", false },
    .{ "{% for i in range(1, 2, 0) %}{% endfor %}", null, "range() arg 3 must not be zero", false },
    .{ "{% set i = 3 %}{% for i in range(i) %}{{ i }}{% endfor %}:{{ i }}|{% for range in range(2) %}{{ range }}{% endfor %}", "012:3|01", null, false },
    .{ "{% set x = 10 %}{% for i in range(2) %}{{ x }}{% set x = x + 1 %}{{ x }};{% endfor %}{{ x }}", "1011;1011;10", null, false },
    .{ "{% set range = 'ab'.upper %}{% for x in range() %}{{ x }}{% endfor %}", "AB", null, false },
    .{ "{% set range = 3 %}{% for i in range(raise_exception('arg')) %}{% endfor %}", null, "arg", true },
    .{ "{% set range = 3 %}{% for i in range(2) %}{% endfor %}", null, "'int' object is not callable", false },
    .{ "{% macro f(range) %}{% for i in range(2) %}{{ i }}{% endfor %}{% endmacro %}{{ f('ab'.upper) }}", null, "lower() takes at most 0 arguments (1 given)", false },
    .{ "{% set ns = namespace(saved=none) %}{% for i in range(3) %}{% set ns.saved = loop %}{% endfor %}{{ ns.saved }}:{{ ns.saved.last }}:{{ ns.saved.previtem }}", "<LoopContext 3/3>:True:1", null, false },
    .{ "{% set ns = namespace(prev=none) %}{% for i in range(3) %}{{ loop.changed(1) }}/{{ loop.cycle('a', 'b') }}/{{ loop == ns.prev }};{% set ns.prev = loop %}{% endfor %}", "True/a/False;False/b/True;False/a/True;", null, false },
    .{ "{% for i in range(3, 0, -1) %}{{ loop.previtem }}|{{ loop.nextitem }};{% endfor %}", "|2;3|1;2|;", null, false },
    .{ "{% for i in range(2) %}{{ loop.previtem + 1 }}{% endfor %}", null, "there is no previous item", false },
    .{ "{% for i in range(2) %}{{ loop.nextitem }}{% endfor %}{% for i in range(1) %}{{ loop.nextitem + 1 }}{% endfor %}", null, "there is no next item", false },
    .{ "{% for i in range(2) %}{{ loop['length'] }}{{ loop | count }}{{ [loop] }}{% endfor %}", "22[<LoopContext 1/2>]22[<LoopContext 2/2>]", null, false },
    .{ "{% for i in range(1) %}{{ loop[0] is defined }}{% endfor %}", "False", null, false },
    .{ "{% for i in range(1) %}{{ loop[:] }}{% endfor %}", null, "'LoopContext' object is not subscriptable", false },
    .{ "{% for i in range(1) %}{{ loop | list }}{% endfor %}", null, "'LoopContext' object is not iterable", false },
    .{ "{% for i in range(1) %}{{ loop([]) }}{% endfor %}", null, "Tried to call non recursive loop.  Maybe you forgot the 'recursive' modifier.", false },
    .{ "{% for i in range(9223372036854775806, 9223372036854775807, 2) %}{{ i }}{% endfor %}", "9223372036854775806", null, false },
    .{ "{% for a, b in range(1) %}{% endfor %}", null, "'int' object is not iterable", false },
    .{ "{% for a, b in range(0) %}{% endfor %}empty", "empty", null, false },
    .{ "{% for i in range(5, 0, 2) %}x{% else %}E{% endfor %}|{% for i in range(-3) %}x{% else %}E{% endfor %}", "E|E", null, false },
    .{ "{% for i in range(3) if i > 0 %}{{ i }}{{ loop.length }}{% endfor %}", "1222", null, false },
    .{ "{% set r = range %}{% for i in r(2) %}{{ i }}{% endfor %}", "01", null, false },
};

test "range loops render and fail as before" {
    for (range_golden) |g| try expectSame(g[0], g[1], g[2], g[3]);
}

