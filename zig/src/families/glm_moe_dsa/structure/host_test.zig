//! Fiore-lab: the structured output's host logic with a fake C matcher; no CUDA or external library needed.
//! Fiore-lab: 偽の C matcher による構造化出力の host の試験。CUDA・外部 library 不要。
const std = @import("std");
const host = @import("host.zig");
const llg = host.llg;
const t = std.testing;

const Fake = struct {
    accepting: bool = false,
    reject: bool = false,
    mask_error: bool = false,
    only_padding: bool = false,
    consumed: usize = 0,
    fills: usize = 0,
    dest: ?[*]u32 = null,
    last: u32 = 0,
    fn self(m: *llg.Matcher) *Fake {
        return @ptrCast(@alignCast(m));
    }
    fn compute(m: *llg.Matcher, out: [*]u32, n: usize) callconv(.c) i32 {
        const f = self(m);
        f.fills += 1;
        f.dest = out;
        if (f.mask_error) return -1;
        std.debug.assert(n == host.bytes);
        @memset(out[0..host.words], 0);
        const id: u32 = if (f.only_padding) 154856 else 10;
        out[id / 32] = @as(u32, 1) << @intCast(id % 32);
        out[host.words - 1] = if (f.only_padding) 0xffffff00 else std.math.maxInt(u32);
        return 0;
    }
    fn consume(m: *llg.Matcher, id: u32) callconv(.c) i32 {
        const f = self(m);
        f.consumed += 1;
        f.last = id;
        return if (f.reject) -1 else 0;
    }
    fn acceptingFn(m: *llg.Matcher) callconv(.c) bool {
        return self(m).accepting;
    }
    fn err(_: *llg.Matcher) callconv(.c) ?[*:0]const u8 {
        return "first line\nprivate parser state";
    }
    fn api() llg.Api {
        var a: llg.Api = undefined;
        a.llg_matcher_compute_mask_into = compute;
        a.llg_matcher_consume_token = consume;
        a.llg_matcher_is_accepting = acceptingFn;
        a.llg_matcher_get_error = err;
        return a;
    }
    fn state(f: *Fake, a: *const llg.Api) host.State {
        return .{ .api = a, .matcher = @ptrCast(f), .eos = &.{ 154820, 154827, 154829 }, .real_size = 154856 };
    }
};

test "structured host: fill writes into the supplied row, clears padding, stops on an error or empty mask" {
    const a = Fake.api();
    var f: Fake = .{};
    var s = f.state(&a);
    var row: [host.words]u32 = undefined;
    try t.expect(s.fill(&row));
    try t.expect(f.dest.? == &row);
    try t.expect(llg.allowed(&row, 10));
    for (154856..154880) |id| try t.expect(!llg.allowed(&row, @intCast(id)));
    f.only_padding = true;
    try t.expect(!s.fill(&row));
    try t.expect(s.failed());
    const calls = f.fills;
    try t.expect(!s.fill(&row));
    try t.expectEqual(calls, f.fills);
    s = f.state(&a);
    f.mask_error = true;
    try t.expect(!s.fill(&row));
    try t.expectEqualStrings("first line", s.message());
    s.fail("later reason");
    try t.expectEqualStrings("first line", s.message());
}

test "structured host: all EOS ids are checked for accepting and never consumed" {
    const a = Fake.api();
    var f: Fake = .{};
    for ([_]u32{ 154820, 154827, 154829 }) |id| {
        var s = f.state(&a);
        f.accepting = false;
        try t.expectEqual(host.State.Accept.failed, s.accept(id));
        try t.expectEqual(@as(usize, 0), f.consumed);
        s = f.state(&a);
        f.accepting = true;
        try t.expectEqual(host.State.Accept.done, s.accept(id));
        try t.expect(s.done);
        try t.expectEqual(@as(usize, 0), f.consumed);
    }
}

test "structured host: a committed token consumed once; rejected token fails before delivery" {
    const a = Fake.api();
    var f: Fake = .{};
    var s = f.state(&a);
    try t.expectEqual(host.State.Accept.token, s.accept(17));
    try t.expectEqual(@as(usize, 1), f.consumed);
    f.reject = true;
    try t.expectEqual(host.State.Accept.failed, s.accept(18));
    try t.expectEqual(@as(usize, 2), f.consumed);
    try t.expectEqual(host.State.Accept.failed, s.accept(19));
    try t.expectEqual(@as(usize, 2), f.consumed);
    try t.expectEqualStrings("first line", s.message());
}

test "structured host: the grammar starts only at segment heads after thinking, no summary branch, answer once" {
    const a = Fake.api();
    var f: Fake = .{};
    var s = f.state(&a);
    try t.expect(!s.segment(true, false, false));
    try t.expect(!s.active);
    try t.expect(!s.segment(true, false, true)); // dropped close / summary branch
    try t.expect(!s.active);
    try t.expect(s.segment(true, true, false)); // canon/fold completed
    try t.expect(s.active);
    try t.expect(!s.segment(true, true, false));
    s = f.state(&a);
    try t.expect(s.segment(false, false, false));
    try t.expect(s.active);
}

test "structured host: the verdict over all ranks: an empty share is fine, global empty or allowed NaN fails" {
    try t.expect(!host.numericFailed(&.{ 4, 5 }));
    try t.expect(host.numericFailed(&.{ 4, 4 }));
    try t.expect(host.numericFailed(&.{ 5, 6 }));
    try t.expect(!host.numericFailed(&.{ 0, 0 }));
}

test "structured host: padding clears exactly the ids at or past the real vocabulary" {
    var row: [3]u32 = @splat(std.math.maxInt(u32));
    host.clearPadding(&row, 33);
    try t.expectEqualSlices(u32, &.{ std.math.maxInt(u32), 1, 0 }, &row);
    host.clearPadding(&row, 0);
    try t.expectEqualSlices(u32, &.{ 0, 0, 0 }, &row);
}

test "structured host: finish: failed before cancellation and done, partial length, unexpected stop fails" {
    const a = Fake.api();
    var f: Fake = .{};
    var s = f.state(&a);
    try t.expectEqual(.length, s.finishReason(true, false));
    try t.expectEqual(.cancelled, s.finishReason(false, true));
    s.done = true;
    try t.expectEqual(.stop, s.finishReason(false, false));
    try t.expectEqual(.cancelled, s.finishReason(false, true));
    s.fail("mask failed");
    try t.expectEqual(.failed, s.finishReason(true, true));
    s = f.state(&a);
    try t.expectEqual(.failed, s.finishReason(false, false));
    try t.expect(s.failed());
}
