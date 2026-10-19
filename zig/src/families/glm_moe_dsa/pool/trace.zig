//! tf-glm53-pool-trace IN OUT: runs a planner / MTP / DFlash2 trace (JSON lines, the format of
//! glm53/tools/pool/FORMAT.md; the kind from the name: .plan. / .mtp. / .dfl.) through pool/plan.zig and writes one
//! output line per input line, for comparison with the Python reference (glm53/tools/pool/pool_ref.py).
//! tf-glm53-pool-trace IN OUT: プランナ / MTP / DFlash2 の trace (JSON lines、glm53/tools/pool/FORMAT.md の形式。種類は
//! 名前の .plan. / .mtp. / .dfl. で決める) を pool/plan.zig で実行し、入力 1 行ごとに 1 行を書く。Python 版
//! (glm53/tools/pool/pool_ref.py) と比べるため。
const std = @import("std");
const plan = @import("plan.zig");

const Value = std.json.Value;

fn int(v: Value, key: []const u8) !i64 {
    return switch (v.object.get(key) orelse return error.MissingKey) {
        .integer => |x| x,
        else => error.NotAnInteger,
    };
}

fn uint(v: Value, key: []const u8) !u64 {
    return @intCast(try int(v, key));
}

fn tokens(gpa: std.mem.Allocator, v: Value, key: []const u8) ![]i64 {
    const arr = (v.object.get(key) orelse return error.MissingKey).array.items;
    const out = try gpa.alloc(i64, arr.len);
    for (arr, out) |x, *o| o.* = x.integer;
    return out;
}

fn range(v: Value, key: []const u8) ?[2]u64 {
    const x = v.object.get(key) orelse return null;
    if (x == .null) return null;
    return .{ @intCast(x.array.items[0].integer), @intCast(x.array.items[1].integer) };
}

const Out = struct {
    gpa: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    fn s(o: *Out, t: []const u8) !void {
        try o.buf.appendSlice(o.gpa, t);
    }
    fn i(o: *Out, x: anytype) !void {
        try o.buf.print(o.gpa, "{d}", .{x});
    }
    fn list(o: *Out, xs: anytype) !void {
        try o.s("[");
        for (xs, 0..) |x, k| {
            if (k > 0) try o.s(",");
            try o.i(x);
        }
        try o.s("]");
    }
};

fn state(o: *Out, m: *const plan.Meta) !void {
    try o.s("{\"clock\":");
    try o.i(m.clock);
    try o.s(",\"free\":[");
    var first = true;
    for (m.ref, 0..) |r, b| if (r == 0) {
        if (!first) try o.s(",");
        first = false;
        try o.i(b);
    };
    try o.s("],\"gen\":");
    try o.list(m.gen);
    try o.s(",\"next_id\":");
    try o.i(m.next_id);
    try o.s(",\"ref\":");
    try o.list(m.ref);
    try o.s(",\"slots\":[");
    for (m.slots.items, 0..) |sl, k| {
        if (k > 0) try o.s(",");
        try o.s("{\"id\":");
        try o.i(sl.id);
        try o.s(",\"kv_end\":");
        try o.i(sl.kv_end);
        try o.s(",\"prompt_end\":");
        try o.i(sl.prompt_end);
        try o.s(",\"seq\":");
        try o.list(sl.seq.items);
        try o.s(",\"table\":");
        try o.list(sl.blocks.items);
        try o.s(",\"tick\":");
        try o.i(sl.tick);
        try o.s(",\"valid_end\":");
        try o.i(sl.validEnd());
        try o.s("}");
    }
    try o.s("]}");
}

fn ring(o: *Out, g: ?plan.Ring) !void {
    const x = g orelse return o.s("null");
    try o.buf.print(o.gpa, "{{\"keep_lo\":{d},\"phys_end\":{d},\"vhi\":{d},\"vlo\":{d}}}", .{ x.keep_lo, x.phys_end, x.vhi, x.vlo });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("usage: tf-glm53-pool-trace IN OUT\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(1 << 30));
    const kind: u8 = if (std.mem.indexOf(u8, args[1], ".plan.") != null) 'p' else if (std.mem.indexOf(u8, args[1], ".mtp.") != null) 'm' else 'd';
    var o: Out = .{ .gpa = gpa };
    var m: ?plan.Meta = null;
    var p: ?plan.Plan = null;
    var pending: ?u64 = null; // X between its plan and its commit / 計画から確定までの X
    var f: plan.MtpF = undefined;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var ln: usize = 0;
    while (lines.next()) |line| {
        ln += 1;
        var parsed = try std.json.parseFromSlice(Value, gpa, line, .{});
        defer parsed.deinit();
        const v = parsed.value;
        const op = v.object.get("op").?.string;
        errdefer std.debug.print("{s}:{d}: op {s}\n", .{ args[1], ln, op });
        if (kind == 'p') {
            if (std.mem.eql(u8, op, "init")) {
                const cfg: plan.Config = .{ .n_blocks = @intCast(try uint(v, "n_blocks")), .capacity = try uint(v, "capacity"), .max_slots = @intCast(try uint(v, "max_slots")), .mtp = v.object.get("mtp").?.bool };
                if (plan.Meta.init(gpa, cfg)) |mm| {
                    m = mm;
                    try o.s("{\"ok\":true}");
                } else |err| switch (err) {
                    error.PoolTooSmall => {
                        try o.s("{\"error\":\"pool_too_small\"}\n");
                        break;
                    },
                    else => return err,
                }
            } else if (std.mem.eql(u8, op, "plan")) {
                const prompt = try tokens(gpa, v, "prompt");
                defer gpa.free(prompt);
                const rok = v.object.get("resume_ok").?.bool;
                if (plan.makePlan(&m.?, prompt, try int(v, "max_tokens"), rok)) |pp| {
                    if (p) |*old| old.deinit(gpa);
                    p = pp;
                    const q = &p.?;
                    try plan.apply(&m.?, q, prompt);
                    try m.?.check(q.X);
                    pending = q.X;
                    try o.s("{\"plan\":{\"action\":\"");
                    try o.s(@tagName(q.action));
                    try o.s("\",\"S\":");
                    if (q.S) |sid| try o.i(sid) else try o.s("null");
                    try o.s(",\"X\":");
                    try o.i(q.X);
                    try o.s(",\"h\":");
                    try o.i(q.h);
                    try o.s(",\"c\":");
                    try o.i(q.c);
                    try o.s(",\"resume_end\":");
                    try o.i(q.resume_end);
                    try o.s(",\"write_lo\":");
                    try o.i(q.write_lo);
                    try o.s(",\"write_hi\":");
                    try o.i(q.write_hi);
                    try o.s(",\"table\":");
                    try o.list(q.table);
                    try o.s(",\"evicted\":");
                    try o.list(q.evicted);
                    try o.s(",\"shrunk\":[");
                    for (q.shrunk, 0..) |e, k| {
                        if (k > 0) try o.s(",");
                        try o.buf.print(gpa, "[{d},{d}]", .{ e[0], e[1] });
                    }
                    try o.s("],\"alloc\":");
                    try o.list(q.alloc);
                    try o.s("},\"state\":");
                    try state(&o, &m.?);
                    try o.s("}");
                } else |err| switch (err) {
                    error.BadRequest => try o.s("{\"error\":\"bad_request\"}"),
                    else => return err,
                }
            } else if (std.mem.eql(u8, op, "commit")) {
                const seq = try tokens(gpa, v, "seq");
                defer gpa.free(seq);
                try plan.commit(&m.?, &p.?, seq, try uint(v, "prompt_end"), try uint(v, "kv_end"));
                try m.?.check(null);
                pending = null;
                try o.s("{\"state\":");
                try state(&o, &m.?);
                try o.s("}");
            } else if (std.mem.eql(u8, op, "check")) {
                if (m.?.check(pending)) |_| try o.s("{\"ok\":true}") else |_| try o.s("{\"ok\":false,\"why\":\"invariant\"}");
            } else return error.UnknownOp;
        } else if (kind == 'm') {
            if (std.mem.eql(u8, op, "begin")) {
                f = plan.MtpF.begin(try uint(v, "resume_end"), try uint(v, "n"));
            } else if (std.mem.eql(u8, op, "chunk")) {
                const a0 = try uint(v, "a");
                const e0 = try uint(v, "e");
                if (a0 < e0) f.chunk(a0, e0);
            } else if (std.mem.eql(u8, op, "propose")) {
                f.propose(try uint(v, "P"), try uint(v, "m"), try uint(v, "room"));
            } else if (std.mem.eql(u8, op, "chain")) {
                f.chain(try uint(v, "q0"), try uint(v, "k"));
            } else if (std.mem.eql(u8, op, "kept")) {} else if (std.mem.eql(u8, op, "end")) {
                try o.buf.print(gpa, "{{\"F\":{d},\"kv_end\":{d}}}\n", .{ f.F, f.kvEnd(try uint(v, "V")) });
                continue;
            } else return error.UnknownOp;
            try o.buf.print(gpa, "{{\"F\":{d}}}", .{f.F});
        } else {
            if (std.mem.eql(u8, op, "resume")) {
                const r = plan.resumeAt(try uint(v, "r"), try uint(v, "n"), range(v, "live"), range(v, "snap"), try uint(v, "below"), try uint(v, "window"));
                try o.buf.print(gpa, "{{\"kind\":\"{s}\",\"pos\":{d},\"s\":{d}}}", .{ @tagName(r.kind), r.pos, r.s });
            } else if (std.mem.eql(u8, op, "save_live")) {
                try ring(&o, plan.Ring.live(try uint(v, "C"), try uint(v, "V"), try uint(v, "s"), try uint(v, "B")));
            } else if (std.mem.eql(u8, op, "save_snap")) {
                try ring(&o, plan.Ring.snap(try uint(v, "n"), try uint(v, "s")));
            } else if (std.mem.eql(u8, op, "shrink")) {
                const rv = v.object.get("rec").?;
                const g: plan.Ring = .{ .phys_end = try uint(rv, "phys_end"), .keep_lo = try uint(rv, "keep_lo"), .vlo = try uint(rv, "vlo"), .vhi = try uint(rv, "vhi") };
                try ring(&o, g.shrink(try uint(v, "valid_end")));
            } else return error.UnknownOp;
        }
        try o.s("\n");
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = o.buf.items });
    return 0;
}
