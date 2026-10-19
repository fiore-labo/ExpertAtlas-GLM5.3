//! TF_GLM53_KV_DIGEST=1 (tests, glm53/POOL.md P1 / P3): after every request each rank prints digests of its cache rows
//! in logical order - the latent rows of every layer but the MTP one, the index-key rows, the MTP layer's rows up to its
//! kv_end, and DFlash2's ring rows in its window - so runs that should hold the same bytes (prefix off, a test table,
//! slots, pool resumed where the others resume) compare row by row, not only by their tokens. The rows are the ones a
//! later request may reuse: positions [0, V) with V the prompt and every emitted token but the pending last one.
//! TF_GLM53_KV_DIGEST=1 (試験用、glm53/POOL.md の P1 / P3): 要求ごとに各ランクが、キャッシュの行を論理の順に並べた
//! digest を出す (MTP 以外の全層の latent の行、index キーの行、MTP 層の kv_end までの行、DFlash2 のリングの window の
//! 行)。同じバイトを持つはずの実行 (prefix off、試験の表、slots、他と同じ位置から再開した pool) を、トークンだけでなく
//! 行で比べられる。対象は後の要求が再利用しうる行: 位置 [0, V)、V は prompt と、未処理の最後の 1 つを除く出力の全トークン。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const devmap = @import("pool/devmap.zig");
const drafter_mod = @import("drafter.zig");

const Buf = cuda.DeviceBuffer;

/// Rows a lanes launch takes at most / lane の起動 1 回の最大行数
const chunk: usize = 16384;

const State = struct {
    act: Buf, // u64 [L]: every cache layer's latent, then the index keys / 全キャッシュ層の latent、続いて index キー
    words: Buf, // int32 [L]
    lanes: Buf, // u64 [L, chunk]
    L: u32,
    nk: usize, // latent lanes / latent の lane 数
    mtp_lane: [2]?usize, // the MTP layer's latent and index-key lanes / MTP 層の latent と index キーの lane
    host: []u64,
    // DFlash2's rings: per layer, K then V, one lane a head / DFlash2 のリング: 層ごとに K、V、head ごとに 1 lane
    dact: ?Buf = null,
    dwords: ?Buf = null,
    dL: u32 = 0,
};

var state: ?State = null;
var n_req: usize = 0;

pub fn on() bool {
    return std.c.getenv("TF_GLM53_KV_DIGEST") != null;
}

fn init(e: *engine.Engine) !State {
    const gpa = e.gpa;
    const lat_row: usize = forward.latRow(e.cfg, e.kv8, e.kv4);
    const idx_row: usize = if (e.kv8) 144 else e.cfg.index_dim * 2;
    var ptrs: std.ArrayList(u64) = .empty;
    defer ptrs.deinit(gpa);
    var wds: std.ArrayList(i32) = .empty;
    defer wds.deinit(gpa);
    const nk = e.state.kc.len;
    const mtp_layer: ?usize = if (e.mtp != null) nk - 1 else null;
    var mtp_lane: [2]?usize = .{ mtp_layer, null };
    for (e.state.kc) |b| {
        try ptrs.append(gpa, b.ptr);
        try wds.append(gpa, @intCast(lat_row / 8));
    }
    for (e.state.ic, 0..) |b, li| if (b) |x| {
        if (mtp_layer != null and li == mtp_layer.?) mtp_lane[1] = ptrs.items.len;
        try ptrs.append(gpa, x.ptr);
        try wds.append(gpa, @intCast(idx_row / 8));
    };
    const L: u32 = @intCast(ptrs.items.len);
    var s: State = .{
        .act = try Buf.fromHost(e.d, std.mem.sliceAsBytes(ptrs.items)), .words = try Buf.fromHost(e.d, std.mem.sliceAsBytes(wds.items)),
        .lanes = try Buf.alloc(e.d, @as(usize, L) * chunk * 8), .L = L, .nk = nk, .mtp_lane = mtp_lane,
        .host = try gpa.alloc(u64, @as(usize, L) * chunk),
    };
    if (e.drafter) |dr| {
        var dp: std.ArrayList(u64) = .empty;
        defer dp.deinit(gpa);
        var dw: std.ArrayList(i32) = .empty;
        defer dw.deinit(gpa);
        const row = dr.cfg.head_dim * 2;
        for (dr.layers) |*ly| for ([_]Buf{ ly.kc, ly.vc }) |b| {
            const heads = b.len / (drafter_mod.ring_rows * row);
            for (0..heads) |h| {
                try dp.append(gpa, b.ptr + h * drafter_mod.ring_rows * row);
                try dw.append(gpa, @intCast(row / 8));
            }
        };
        s.dact = try Buf.fromHost(e.d, std.mem.sliceAsBytes(dp.items));
        s.dwords = try Buf.fromHost(e.d, std.mem.sliceAsBytes(dw.items));
        s.dL = @intCast(dp.items.len);
        if (@as(usize, s.dL) > L) return error.DigestLanes;
    }
    return s;
}

/// One run of `n` physical rows from `phys0` hashed lane by lane into hs (rows of a lane below `lim[lane]` of the
/// run only).
/// `phys0` からの物理行 `n` 本の並びを lane ごとに hs へ (lane ごとに、並びの先頭から `lim[lane]` 行まで)。
fn feed(e: *engine.Engine, s: *State, act: u64, words: u64, L: u32, phys0: u64, n: usize, hs: []std.hash.XxHash64, lim: []const usize) !void {
    try e.glue.slotLanes(e.stream, act, 0, words, L, @intCast(phys0), 0, @intCast(n), s.lanes.ptr);
    try e.stream.synchronize();
    try s.lanes.download(0, std.mem.sliceAsBytes(s.host[0 .. @as(usize, L) * n]));
    for (0..L) |l| {
        const m = @min(n, lim[l]);
        hs[l].update(std.mem.sliceAsBytes(s.host[l * n ..][0..m]));
    }
}

fn fold(hs: []std.hash.XxHash64, pick: anytype, ctx: anytype) u64 {
    var h = std.hash.XxHash64.init(0);
    for (hs, 0..) |*x, l| if (pick(ctx, l)) {
        const v = x.final();
        h.update(std.mem.asBytes(&v));
    };
    return h.final();
}

/// After a request (every rank, both streams done or not): V valid positions, the MTP layer's kv_end, and the
/// logical block -> physical block table the rows sit at (null: the rows are contiguous from 0).
/// 要求の後 (全ランク): 有効な位置 V、MTP 層の kv_end、論理ブロック -> 物理ブロックの表 (null: 行は 0 から連続)。
pub fn report(e: *engine.Engine, V: u64, P: u64, mtp_end: u64, table: ?[]const u32) !void {
    if (state == null) state = try init(e);
    const s = &state.?;
    const gpa = e.gpa;
    try e.stream.synchronize();
    if (e.n_sp > 0) try e.comm.synchronize();
    const dcp: u64 = e.dcp;
    const drank: u64 = e.rank % e.dcp;
    const rows_blk: u64 = devmap.block / dcp;
    const nloc: u64 = if (V > drank) (V - drank + dcp - 1) / dcp else 0;
    const nloc_m: u64 = if (mtp_end > drank) @min(nloc, (mtp_end - drank + dcp - 1) / dcp) else 0;
    const hs = try gpa.alloc(std.hash.XxHash64, s.L);
    defer gpa.free(hs);
    for (hs) |*h| h.* = std.hash.XxHash64.init(0);
    const lim = try gpa.alloc(usize, s.L);
    defer gpa.free(lim);
    // the same lanes over the prompt's rows alone, [0, P) / 同じ lane を prompt の行 [0, P) だけで
    const hp = try gpa.alloc(std.hash.XxHash64, s.L);
    defer gpa.free(hp);
    for (hp) |*h| h.* = std.hash.XxHash64.init(0);
    const limp = try gpa.alloc(usize, s.L);
    defer gpa.free(limp);
    const nloc_p: u64 = if (@min(P, V) > drank) (@min(P, V) - drank + dcp - 1) / dcp else 0;
    // TF_GLM53_KV_DIGEST=2: also a digest of the latent lanes per block of 256 positions (rows_blk local rows)
    // TF_GLM53_KV_DIGEST=2: latent の lane の digest を 256 位置 (rows_blk ローカル行) のブロックごとにも
    const per_block = if (std.c.getenv("TF_GLM53_KV_DIGEST")) |v| v[0] == '2' else false;
    var blk_line: std.ArrayList(u8) = .empty;
    defer blk_line.deinit(gpa);
    var bh = std.hash.XxHash64.init(0);
    var bi: u64 = 0;
    var i: u64 = 0;
    while (i < nloc) {
        const ph = struct {
            fn of(t: ?[]const u32, rb: u64, k: u64) u64 {
                return if (t) |tb| @as(u64, tb[@intCast(k / rb)]) * rb + k % rb else k;
            }
        }.of;
        const p0 = ph(table, rows_blk, i);
        var n: u64 = 1;
        while (i + n < nloc and n < chunk and ph(table, rows_blk, i + n) == p0 + n) n += 1;
        for (lim, 0..) |*x, l| {
            const is_m = (s.mtp_lane[0] != null and l == s.mtp_lane[0].?) or (s.mtp_lane[1] != null and l == s.mtp_lane[1].?);
            x.* = if (is_m) @intCast(if (nloc_m > i) nloc_m - i else 0) else @intCast(n);
        }
        try feed(e, s, s.act.ptr, s.words.ptr, s.L, p0, @intCast(n), hs, lim);
        if (per_block) for (0..@intCast(n)) |k| {
            for (0..s.nk) |l| bh.update(std.mem.asBytes(&s.host[l * @as(usize, @intCast(n)) + k]));
            bi += 1;
            if (bi % rows_blk == 0 or i + k + 1 == nloc) {
                try blk_line.print(gpa, " {x:0>4}", .{@as(u16, @truncate(bh.final()))});
                bh = std.hash.XxHash64.init(0);
            }
        };
        // the prompt part from the lanes just downloaded / いま読んだ lane から prompt の部分
        const mp: usize = @intCast(if (nloc_p > i) @min(n, nloc_p - i) else 0);
        if (mp > 0) for (0..s.L) |l| {
            limp[l] = @min(mp, lim[l]);
            hp[l].update(std.mem.sliceAsBytes(s.host[l * @as(usize, @intCast(n)) ..][0..limp[l]]));
        };
        i += n;
    }
    const Pick = struct {
        s: *const State,
        fn main(c: @This(), l: usize) bool {
            return l < c.s.nk and !(c.s.mtp_lane[0] != null and l == c.s.mtp_lane[0].?);
        }
        fn index(c: @This(), l: usize) bool {
            return l >= c.s.nk and !(c.s.mtp_lane[1] != null and l == c.s.mtp_lane[1].?);
        }
        fn mtp(c: @This(), l: usize) bool {
            return (c.s.mtp_lane[0] != null and l == c.s.mtp_lane[0].?) or (c.s.mtp_lane[1] != null and l == c.s.mtp_lane[1].?);
        }
    };
    const pk: Pick = .{ .s = s };
    const d_main = fold(hs, Pick.main, pk);
    const d_index = fold(hs, Pick.index, pk);
    const d_mtp = fold(hs, Pick.mtp, pk);
    const p_main = fold(hp, Pick.main, pk);
    const p_index = fold(hp, Pick.index, pk);
    // DFlash2: the ring rows of positions [context_end - window, context_end), ring slot = position mod ring_rows
    // DFlash2: 位置 [context_end - window, context_end) のリングの行。リングの位置 = 位置 mod ring_rows
    var d_dr: u64 = 0;
    var ce: usize = 0;
    var lo: usize = 0;
    if (e.drafter) |dr| if (s.dact) |da| {
        ce = dr.context_end;
        lo = ce - @min(ce, dr.cfg.window);
        const dh = hs[0..s.dL];
        for (dh) |*h| h.* = std.hash.XxHash64.init(0);
        @memset(lim[0..s.dL], std.math.maxInt(usize));
        var q = lo;
        while (q < ce) {
            const slot = q % drafter_mod.ring_rows;
            const n = @min(ce - q, drafter_mod.ring_rows - slot, chunk);
            try feed(e, s, da.ptr, s.dwords.?.ptr, s.dL, slot, n, dh, lim[0..s.dL]);
            q += n;
        }
        var h = std.hash.XxHash64.init(0);
        for (dh) |*x| {
            const v = x.final();
            h.update(std.mem.asBytes(&v));
        }
        d_dr = h.final();
    };
    std.debug.print("[kvdigest] rank {d} req {d}: V {d} (local {d}), main {x:0>16}, index {x:0>16}, mtp {x:0>16} to {d}, drafter {x:0>16} [{d}, {d}), prompt {d}: main {x:0>16}, index {x:0>16}\n", .{ e.rank, n_req, V, nloc, d_main, d_index, d_mtp, mtp_end, d_dr, lo, ce, @min(P, V), p_main, p_index });
    if (per_block) std.debug.print("[kvblocks] rank {d} req {d}:{s}\n", .{ e.rank, n_req, blk_line.items });
    n_req += 1;
}

/// The valid positions after a request (as pool_serve.zig commits them): the prompt and every emitted token but the
/// pending last one; a request stopped in its prompt: the chunks done.
/// 要求の後の有効な位置 (pool_serve.zig の確定と同じ): prompt と、未処理の最後の 1 つを除く出力。prompt の途中で
/// 止まった要求は終わったチャンクまで。
pub fn validOf(e: *const engine.Engine, prompt_len: usize, st: engine.Stats) u64 {
    if (st.finish == .cancelled and st.generated == 0) return e.prefill_done;
    return prompt_len + st.generated - @min(st.generated, 1);
}

var test_tbl: ?[]u32 = null;

/// TF_GLM53_POOL_TEST's table (the PoolMap's entries, read once), null without one.
/// TF_GLM53_POOL_TEST の表 (PoolMap の項目、1 回だけ読む)。なければ null。
pub fn testTable(e: *engine.Engine) !?[]const u32 {
    const m = e.pmap orelse return null;
    if (test_tbl == null) {
        const n = (m.len - devmap.header) / 4;
        const t = try e.gpa.alloc(u32, n);
        try e.stream.synchronize();
        try m.download(devmap.header, std.mem.sliceAsBytes(t));
        test_tbl = t;
    }
    return test_tbl;
}
