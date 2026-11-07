// Fiore-lab, 2026 (NOTICE): GLM tool calls whose values hold </arg_value> or </tool_call> (glm53 tool closer, design v4.1).
//! A GLM call the first-closer reading cuts, read again by its values' depth: the part ``tool_parse.envelopes``,
//! ``reply_text.hideInto`` and ``tool_stream.TextFilter`` share.
//!
//! GLM-5.3 writes ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`` with values raw, so a
//! value that holds ``</arg_value>`` or ``</tool_call>`` (a call written as an example, a file about the format) ends
//! early when the first closer ends it. Only a call that reading shows cut is read again, and the new reading replaces
//! it only when it reads the whole call, nothing left over and no other family's markup inside.
const std = @import("std");
const builtin = @import("builtin");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;

const space = reply_text.isSpaceByte;

const key_open = "<arg_key>";
const key_close = "</arg_key>";
const value_open = "<arg_value>";
const value_close = "</arg_value>";
const call_close = "</tool_call>";
/// the other families' wrappers, exact as ``tool_parse.envelopes`` finds them; the namespaced one is matched apart
const foreign = [_][]const u8{ "<|tool_call>", "<tool_call|>", "<\u{ff5c}DSML\u{ff5c}tool_calls>", "</\u{ff5c}DSML\u{ff5c}tool_calls>" };
const exact = [_]struct { Tag, []const u8 }{ .{ .key_open, key_open }, .{ .key_close, key_close }, .{ .value_open, value_open }, .{ .value_close, value_close } };
const none = std.math.maxInt(usize);

/// Bytes the re-read looked at, counted in tests only: they must grow with the reply, not with its square.
pub var work: usize = 0;

inline fn count(n: usize) void {
    if (builtin.is_test) work += n;
}

/// Bytes of the text: a name, a key or a value.
pub const Span = struct { start: usize, end: usize };
pub const Pair = struct { key: Span, value: Span };
/// A call read again: its name and its keys and values as the model wrote them.
pub const Call = struct { name: Span, pairs: []const Pair };

/// What the call at a ``<tool_call>`` comes to: the first-closer reading, a re-read that ends at ``end`` (past its
/// ``</tool_call>``), or a re-read that needs more of a reply still streaming.
pub const Outcome = union(enum) { legacy, end: usize, need_more };

const Tag = enum { key_open, key_close, value_open, value_close, call_close, foreign, partial, other };

/// How far a namespaced tag that a streamed text ended inside was read, so the next look goes on from there.
const Resume = struct { at: usize = none, j: usize = 0 };

/// The tag at ``text[i]``; ``partial`` when an unfinished text ends inside what could still become one.
fn tagAt(text: []const u8, i: usize, finished: bool, ns: *Resume) Tag {
    if (text[i] != '<') return .other;
    const rest = text[i..];
    count(1);
    for (exact) |t| if (std.mem.startsWith(u8, rest, t[1])) return t[0];
    if (std.ascii.startsWithIgnoreCase(rest, call_close)) return .call_close;
    for (foreign) |f| if (std.mem.startsWith(u8, rest, f)) return .foreign;
    const n = namespaced(rest, if (ns.at == i) ns.j else 0);
    if (n.kind == .whole) return .foreign;
    if (finished) return .other;
    if (n.kind == .partial) {
        ns.* = .{ .at = i, .j = n.reached };
        return .partial;
    }
    for (exact) |t| if (rest.len < t[1].len and std.mem.startsWith(u8, t[1], rest)) return .partial;
    if (rest.len < call_close.len and std.ascii.startsWithIgnoreCase(call_close, rest)) return .partial;
    for (foreign) |f| if (rest.len < f.len and std.mem.startsWith(u8, f, rest)) return .partial;
    return .other;
}

/// ``<NAME:tool_call>`` or ``</NAME:tool_call>`` (any case) at the start of ``rest``, NAME as ``envelopes`` reads it;
/// ``from`` goes on past the name bytes an earlier look read.
fn namespaced(rest: []const u8, from: usize) struct { kind: enum { whole, partial, no }, reached: usize } {
    var j = from;
    if (j == 0) {
        j = 1;
        if (j < rest.len and rest[j] == '/') j += 1;
        if (j >= rest.len) return .{ .kind = .partial, .reached = 0 };
        if (!(std.ascii.isAlphabetic(rest[j]) or rest[j] == '_')) return .{ .kind = .no, .reached = 0 };
        j += 1;
    }
    const start = j;
    while (j < rest.len and (std.ascii.isAlphanumeric(rest[j]) or rest[j] == '_' or rest[j] >= 0x80 or rest[j] == '.' or rest[j] == '-')) j += 1;
    count(j - start);
    const tail = ":tool_call>";
    if (std.ascii.startsWithIgnoreCase(rest[j..], tail)) return .{ .kind = .whole, .reached = j };
    if (rest.len - j < tail.len and std.ascii.startsWithIgnoreCase(tail, rest[j..])) return .{ .kind = .partial, .reached = j };
    return .{ .kind = .no, .reached = j };
}

/// Every value tag of a text paired as brackets, and where the other families' tags are: a value's end is its
/// opener's pair (the tags before an opener do not change which closer brings its depth back to 0), so several calls
/// read again never read the same value twice. A text that grows at its end extends it.
pub const Matches = struct {
    /// the ``<`` of every exact ``<arg_value>`` and ``</arg_value>``, in order
    at: std.ArrayList(usize) = .empty,
    /// for an opener, the index in ``at`` of the closer it pairs with; ``none`` for a closer or an opener still open
    pair: std.ArrayList(usize) = .empty,
    open: std.ArrayList(usize) = .empty,
    foreign: std.ArrayList(usize) = .empty,
    scanned: usize = 0,
    ns: Resume = .{},

    pub fn extend(m: *Matches, a: Allocator, text: []const u8, finished: bool) Allocator.Error!void {
        var i = m.scanned;
        while (std.mem.indexOfScalarPos(u8, text, i, '<')) |lt| {
            count(lt - i);
            switch (tagAt(text, lt, finished, &m.ns)) {
                .partial => {
                    m.scanned = lt;
                    return;
                },
                .value_open => {
                    try m.open.append(a, m.at.items.len);
                    try m.at.append(a, lt);
                    try m.pair.append(a, none);
                    i = lt + value_open.len;
                },
                .value_close => {
                    if (m.open.pop()) |o| m.pair.items[o] = m.at.items.len;
                    try m.at.append(a, lt);
                    try m.pair.append(a, none);
                    i = lt + value_close.len;
                },
                .foreign => {
                    try m.foreign.append(a, lt);
                    i = lt + 1;
                },
                else => i = lt + 1,
            }
        }
        count(text.len - i);
        m.scanned = text.len;
    }

    fn lowerBound(items: []const usize, pos: usize) usize {
        var lo: usize = 0;
        var hi = items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (items[mid] < pos) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    fn foreignFrom(m: *const Matches, from: usize, to: usize) bool {
        const k = lowerBound(m.foreign.items, from);
        return k < m.foreign.items.len and m.foreign.items[k] < to;
    }
};

const Value = union(enum) { close: struct { at: usize, nested: bool }, need_more, fail };

/// The value whose ``<arg_value>`` is at ``v``: the closer that brings its depth back to 0, from the pair table.
fn valueEnd(v: usize, finished: bool, m: *const Matches) Value {
    const k = Matches.lowerBound(m.at.items, v);
    if (k == m.at.items.len or m.at.items[k] != v) return if (finished) .fail else .need_more;
    const c = m.pair.items[k];
    if (c == none) return if (finished or m.foreignFrom(v, none)) .fail else .need_more;
    if (m.foreignFrom(v, m.at.items[c])) return .fail;
    return .{ .close = .{ .at = m.at.items[c], .nested = c - k > 1 } };
}

/// G: today's payload ``text[body..first_close]`` looks like ``NAME<arg_key>`` with a name ``glmCall`` keeps that no
/// other family reads first (Gemma needs a ``{`` after its name, JSON a ``"`` before a ``<``, Qwen a leading ``<``).
fn gate(text: []const u8, body: usize, first_close: usize) ?struct { name: Span, first_key: usize, p: Span } {
    const p = reply_text.pyStrip(text[body..first_close]);
    if (p.len == 0 or p[0] == '{' or p[0] == '[' or p[0] == '<') return null;
    const lt = std.mem.indexOfScalar(u8, p, '<') orelse return null;
    if (!std.mem.startsWith(u8, p[lt..], key_open)) return null;
    const name = reply_text.pyStrip(p[0..lt]);
    if (name.len == 0) return null;
    for (name) |ch| switch (ch) {
        '<', '>', '\n', '\r', '{', '}', '[', ']', '"' => return null,
        else => {},
    };
    const ps = @intFromPtr(p.ptr) - @intFromPtr(text.ptr);
    const ns = @intFromPtr(name.ptr) - @intFromPtr(text.ptr);
    return .{ .name = .{ .start = ns, .end = ns + name.len }, .first_key = ps + lt, .p = .{ .start = ps, .end = ps + p.len } };
}

/// T: today's reading of ``p`` cuts a value (an ``<arg_value>`` while one is open, or one left open at the end).
fn cut(text: []const u8, p: Span) bool {
    var open = false;
    var i = p.start;
    while (std.mem.indexOfScalarPos(u8, text[0..p.end], i, '<')) |lt| {
        if (std.mem.startsWith(u8, text[lt..p.end], value_open)) {
            if (open) return true;
            open = true;
            i = lt + value_open.len;
        } else if (std.mem.startsWith(u8, text[lt..p.end], value_close)) {
            open = false;
            i = lt + value_close.len;
        } else i = lt + 1;
    }
    return open;
}

/// Where R stands in a call, so a streamed call goes on from there: in a key (``at`` is where its closer's search
/// resumes), before a value, or after one.
const Progress = struct {
    phase: enum { key, value, next } = .key,
    at: usize,
    key: Span,
    nested: bool = false,
    ns: Resume = .{},
};

const Read = union(enum) { end: struct { at: usize, nested: bool }, need_more, fail };

/// R: the call from its first ``<arg_key>``, nothing left over: keys to their first ``</arg_key>``, values by depth.
fn read(a: Allocator, text: []const u8, p: *Progress, finished: bool, m: *const Matches, pairs: ?*std.ArrayList(Pair)) Allocator.Error!Read {
    const more: Read = if (finished) .fail else .need_more;
    while (true) switch (p.phase) {
        .key => {
            const ke = while (std.mem.indexOfScalarPos(u8, text, p.at, '<')) |lt| {
                count(lt + 1 - p.at);
                switch (tagAt(text, lt, finished, &p.ns)) {
                    .key_close => break lt,
                    .call_close, .foreign => return .fail,
                    .partial => {
                        p.at = lt;
                        return .need_more;
                    },
                    else => p.at = lt + 1,
                }
            } else {
                count(text.len - p.at);
                p.at = text.len;
                return more;
            };
            p.key.end = ke;
            p.phase = .value;
            p.at = ke + key_close.len;
        },
        .value => {
            while (p.at < text.len and space(text[p.at])) p.at += 1;
            if (p.at == text.len) return more;
            switch (tagAt(text, p.at, finished, &p.ns)) {
                .value_open => {},
                .partial => return .need_more,
                else => return .fail,
            }
            const close = switch (valueEnd(p.at, finished, m)) {
                .close => |c| c,
                .need_more => return .need_more,
                .fail => return .fail,
            };
            p.nested = p.nested or close.nested;
            if (pairs) |list| try list.append(a, .{ .key = p.key, .value = .{ .start = p.at + value_open.len, .end = close.at } });
            p.phase = .next;
            p.at = close.at + value_close.len;
        },
        .next => {
            while (p.at < text.len and space(text[p.at])) p.at += 1;
            if (p.at == text.len) return more;
            switch (tagAt(text, p.at, finished, &p.ns)) {
                .key_open => {
                    p.phase = .key;
                    p.at += key_open.len;
                    p.key = .{ .start = p.at, .end = p.at };
                },
                .call_close => return .{ .end = .{ .at = p.at + call_close.len, .nested = p.nested } },
                .partial => return .need_more,
                else => return .fail,
            }
        },
    };
}

/// A call the first closer cuts that may read whole (G and T), and how far its re-read has come.
pub const Candidate = struct { name: Span, first_close: usize, progress: Progress };

/// The call whose ``<tool_call>`` ends at ``body`` and whose first ``</tool_call>`` is at ``first_close``, when that
/// reading cuts one of its values; null keeps the first closer's reading. Its bytes are fixed once the closer is.
pub fn candidate(text: []const u8, body: usize, first_close: usize) ?Candidate {
    const g = gate(text, body, first_close) orelse return null;
    if (!cut(text, g.p)) return null;
    const k = g.first_key + key_open.len;
    return .{ .name = g.name, .first_close = first_close, .progress = .{ .at = k, .key = .{ .start = k, .end = k } } };
}

/// A candidate read by depth, on from where it stopped: the re-read's end when it reads whole and ends past the first
/// closer or holds a nested value. ``m`` (extended here) holds the text's value pairs; ``call`` gets the name and the
/// pairs of a candidate read from its start.
pub fn reread(a: Allocator, text: []const u8, c: *Candidate, finished: bool, m: *Matches, call: ?*Call) Allocator.Error!Outcome {
    try m.extend(a, text, finished);
    var pairs: std.ArrayList(Pair) = .empty;
    switch (try read(a, text, &c.progress, finished, m, if (call != null) &pairs else null)) {
        .fail => return .legacy,
        .need_more => return .need_more,
        .end => |e| {
            if (e.at == c.first_close + call_close.len and !e.nested) return .legacy;
            if (call) |out| out.* = .{ .name = c.name, .pairs = pairs.items };
            return .{ .end = e.at };
        },
    }
}

/// ``candidate`` then ``reread``, from the call's start.
pub fn rescue(a: Allocator, text: []const u8, body: usize, first_close: usize, finished: bool, m: *Matches, call: ?*Call) Allocator.Error!Outcome {
    var c = candidate(text, body, first_close) orelse return .legacy;
    return reread(a, text, &c, finished, m, call);
}

test "a GLM value that holds the closers reads whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8 }{ // the reply, the bytes the re-read ends after (null: the first closer's reading)
        .{ "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value></tool_call>Tail", "</arg_value></tool_call>" },
        .{ "<tool_call>w<arg_key>c</arg_key><arg_value>A<arg_value>B</arg_value>C</arg_value></tool_call>", "C</arg_value></tool_call>" },
        .{ "<tool_call>w<arg_key>c</arg_key><arg_value>x<arg_value>y</arg_value></tool_call>", null }, // a lone opener: never back to 0
        .{ "<tool_call>w<arg_key><arg_value></arg_key><arg_value>x</arg_value></tool_call>", null }, // the key's tag is not counted
        .{ "<tool_call>w<arg_key>c</arg_key><arg_value>a</tool_call><|tool_call>b</arg_value></tool_call>", null }, // another family inside
        .{ "<tool_call>w<arg_key>c</arg_key><arg_value>a</tool_call><x:tool_call>b</arg_value></tool_call>", null },
        .{ "<tool_call>{\"name\":\"w\"}</tool_call>", null },
        .{ "<tool_call>\"w<arg_key>c</arg_key><arg_value>x\"</tool_call>y</arg_value></tool_call>", null }, // a JSON string
        .{ "<tool_call>w<arg_key>c</arg_key><arg_value>a</tool_call>b</arg_value>stray</tool_call>", null }, // something left over
    };
    for (cases) |c| {
        var m: Matches = .{};
        const first = std.mem.indexOf(u8, c[0], call_close).?;
        const got = try rescue(a, c[0], 11, first, true, &m, null);
        if (c[1]) |tail| {
            const want = std.mem.lastIndexOf(u8, c[0], tail).? + tail.len;
            try std.testing.expectEqual(Outcome{ .end = want }, got);
        } else try std.testing.expectEqual(Outcome.legacy, got);
    }
}

test "a streamed GLM call waits while it could still read whole, at every split, from where it stopped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<tool_call>write<arg_key>content</arg_key><arg_value>a</tool_call>b</arg_value><arg_key>n</arg_key> <arg_value>c</arg_value></tool_call>Tail";
    const first = std.mem.indexOf(u8, reply, call_close).?;
    const done = std.mem.lastIndexOf(u8, reply, call_close).? + call_close.len;
    var m: Matches = .{};
    var kept = candidate(reply, 11, first).?;
    for (first + call_close.len..reply.len + 1) |n| {
        var fresh: Matches = .{};
        const got = try rescue(a, reply[0..n], 11, first, false, &fresh, null);
        if (n < done) try std.testing.expectEqual(Outcome.need_more, got) else try std.testing.expectEqual(Outcome{ .end = done }, got);
        // the table and the re-read, both carried on as the text grows, agree
        if (n <= done) try std.testing.expectEqual(got, try reread(a, reply[0..n], &kept, false, &m, null));
    }
    // a value closed, then text: the re-read fails there and then, not when the reply ends
    const x = "<tool_call>f<arg_key>a</arg_key><arg_value>x</tool_call>y</arg_value>X";
    var t: Matches = .{};
    try std.testing.expectEqual(Outcome.legacy, try rescue(a, x, 11, std.mem.indexOf(u8, x, call_close).?, false, &t, null));
}
