//! TF_GLM53_POOL_POISON=<byte> (tests, glm53/POOL.md P1): cache rows no request may read hold this byte (0xff: NaN in
//! FP8 / bf16 / fp32, 0x7f: the largest finite values), so a kernel that reads a row outside its table, a future row
//! or a freed block changes the output. Prefix off (with or without TF_GLM53_POOL_TEST): every latent and index-key
//! row before every request. Pool: every row at start, then per request the blocks the plan allocates and the rows of
//! the resume block from resume_end, before the prompt, and the blocks the request freed, after the commit. Slots: not
//! used (the compute cache keeps the conversation it continues).
//! TF_GLM53_POOL_POISON=<byte> (試験用、glm53/POOL.md の P1): どの要求も読んではいけないキャッシュの行をこのバイトで
//! 埋める (0xff: FP8 / bf16 / fp32 で NaN、0x7f: 最大の有限値)。表の外の行・未来の行・解放したブロックを読むカーネルが
//! あれば出力が変わる。prefix off (TF_GLM53_POOL_TEST の有無によらず): 要求ごとに latent と index キーの全行。pool: 起動時
//! に全行、要求ごとにプランが割り当てるブロックと再開ブロックの resume_end 以降の行を prompt の前に、その要求が解放した
//! ブロックを確定の後に。slots: 使わない (計算用のキャッシュは続ける会話を持っている)。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const forward = @import("forward.zig");

const Buf = cuda.DeviceBuffer;

var parsed = false;
var value: ?u8 = null;

pub fn byte() ?u8 {
    if (!parsed) {
        parsed = true;
        if (std.c.getenv("TF_GLM53_POOL_POISON")) |v| value = std.fmt.parseInt(u8, std.mem.span(v), 0) catch 0xff;
    }
    return value;
}

fn rowBytes(e: *const engine.Engine) [2]usize {
    return .{ forward.latRow(e.cfg, e.kv8, e.kv4), if (e.kv8) 144 else e.cfg.index_dim * 2 };
}

fn fillRange(e: *engine.Engine, b: Buf, off: usize, n: usize, v: u8) !void {
    if (n == 0) return;
    if (off + n > b.len) return error.PoisonRange;
    try b.d.check(b.d.api.cuMemsetD8Async(b.ptr + off, v, n, e.stream.handle), "cuMemsetD8Async");
}

/// Every latent and index-key row of every cache layer.
/// 全キャッシュ層の latent と index キーの全行。
pub fn all(e: *engine.Engine, v: u8) !void {
    for (e.state.kc) |b| try b.fill8(v, e.stream.handle);
    for (e.state.ic) |b| if (b) |x| try x.fill8(v, e.stream.handle);
}

/// Local rows [r0, r1) of every cache layer (latent and index keys).
/// 全キャッシュ層 (latent と index キー) のローカル行 [r0, r1)。
pub fn rows(e: *engine.Engine, r0: usize, r1: usize, v: u8) !void {
    if (r1 <= r0) return;
    const rb = rowBytes(e);
    for (e.state.kc) |b| try fillRange(e, b, r0 * rb[0], (r1 - r0) * rb[0], v);
    for (e.state.ic) |b| if (b) |x| try fillRange(e, x, r0 * rb[1], (r1 - r0) * rb[1], v);
}

/// The rows of physical blocks `blocks` (any order; consecutive numbers merged), `rows_blk` local rows a block.
/// 物理ブロック `blocks` の行 (順不同。連続する番号はまとめる)。1 ブロック `rows_blk` ローカル行。
pub fn blocks(e: *engine.Engine, gpa: std.mem.Allocator, list: []const u32, rows_blk: usize, v: u8) !void {
    if (list.len == 0) return;
    const s = try gpa.dupe(u32, list);
    defer gpa.free(s);
    std.mem.sort(u32, s, {}, std.sort.asc(u32));
    var k: usize = 0;
    while (k < s.len) {
        var m: usize = 1;
        while (k + m < s.len and s[k + m] == s[k] + m) m += 1;
        try rows(e, @as(usize, s[k]) * rows_blk, (@as(usize, s[k]) + m) * rows_blk, v);
        k += m;
    }
}
