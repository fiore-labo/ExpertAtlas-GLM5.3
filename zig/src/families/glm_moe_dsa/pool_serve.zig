//! The KV pool's part of serving (TF_GLM53_PREFIX=pool, glm53/POOL.md): the metadata and the per-rank row hashes set
//! up at start; per request, rank 0's plan before the request is shared, then on every rank the plan's agreement,
//! the reused rows checked against their hashes before the first token (or a common cold), the active slot's table
//! into the PoolMap, the prompt from resume_end, the kernels' write check, the commit and the new rows' hashes.
//! KV pool の担当分 (TF_GLM53_PREFIX=pool、glm53/POOL.md): 起動時にメタデータとランクごとの行ハッシュを用意する。
//! 要求ごとに、共有前に rank 0 がプランを作り、続いて全ランクでプランの合意、再利用する行を最初のトークンの前に
//! ハッシュで照合 (または全ランク共通の cold)、処理中のスロットの表を PoolMap へ、resume_end からの prompt、カーネルの
//! 書き込みの検査、確定と新しい行のハッシュを行う。
//!
//! Row hashes: every local physical row of a block wholly or partly below a slot's kv_end has an expected value
//! (slotcache/hashing.zig H, bound to the position, the token prefix and the block's generation), on the device, a
//! rank's own. They are made at the commit for the rows from the block holding resume_end to kv_end, and checked
//! before a request reuses them.
//! 行ハッシュ: スロットの kv_end より手前に (全部か一部が) 入るブロックのローカルの物理行は、期待値を持つ
//! (slotcache/hashing.zig の H。位置・トークンの prefix・ブロックの世代に結びつける)。デバイス上にあり、ランクごとに
//! 違う。確定のときに resume_end を含むブロックから kv_end までの行について作り、要求が再利用する前に照合する。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const lockstep = @import("lockstep.zig");
const plan = @import("pool/plan.zig");
const devmap = @import("pool/devmap.zig");
const hashing = @import("slotcache/hashing.zig");
const tree = @import("slotcache/tree.zig");
const kvdigest = @import("kvdigest.zig");
const poison = @import("poison.zig");

const Buf = cuda.DeviceBuffer;

/// Local rows a hash launch takes at most (bounds the lanes scratch, rows x lanes x 8 B).
/// ハッシュ 1 回の起動で扱う最大のローカル行数 (lane の作業域 rows x lanes x 8 B を抑える)。
const hash_rows: usize = 16384;

/// The drafter's confidence, the same on every rank (slots_serve.zig).
/// ドラフタの confidence。全ランクで同じ値 (slots_serve.zig)。
const dflash_confidence: f64 = 0.4;

/// TF_GLM53_POOL_FAULT=KIND:RANK (tests, glm53/POOL.md P3): one fault, once, on rank RANK. hash: before the first
/// request that checks reused rows, the expected hash of the first row checked is flipped (that rank's check fails:
/// every rank must go cold). lo / z / oob: on the first request with a prompt to prefill, that rank's PoolMap gets
/// write_lo dcp past resume_end / the block of resume_end mapped to Z / to a block number past Z (the kernels must
/// refuse the write; how the other ranks end is what the test looks at).
/// TF_GLM53_POOL_FAULT=KIND:RANK (試験用、glm53/POOL.md の P3): rank RANK に 1 回だけ異常を起こす。hash: 再利用する行を
/// 照合する最初の要求の前に、最初に照合する行の期待値を反転する (そのランクの照合が失敗し、全ランクが cold になるはず)。
/// lo / z / oob: prefill する prompt がある最初の要求で、そのランクの PoolMap の write_lo を resume_end の dcp 先に / resume_end
/// のブロックを Z に / Z より先のブロック番号にする (カーネルが書き込みを拒むはず。他のランクがどう終わるかを見る)。
const Fault = struct {
    kind: enum { none, hash, lo, z, oob } = .none,
    rank: u32 = 0,
    done: bool = false,

    fn fromEnv() !Fault {
        const v = std.mem.span(std.c.getenv("TF_GLM53_POOL_FAULT") orelse return .{});
        var it = std.mem.tokenizeScalar(u8, v, ':');
        var f: Fault = .{};
        f.kind = std.meta.stringToEnum(@FieldType(Fault, "kind"), it.next() orelse return error.BadPoolFault) orelse return error.BadPoolFault;
        f.rank = try std.fmt.parseInt(u32, it.next() orelse return error.BadPoolFault, 10);
        return f;
    }
};

/// The PoolMap layout and the row-hash scheme (native_ops.cu PoolMap, pool/devmap.zig): checked across ranks at start.
/// PoolMap の並びと行ハッシュの方式 (native_ops.cu の PoolMap、pool/devmap.zig): 起動時にランク間で照合する。
const abi_version: i32 = 1;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    meta: plan.Meta,
    env: devmap.Env,
    n_entries: usize,
    dcp: u32,
    drank: u32,
    rows_blk: usize, // local rows a block: 256 / dcp / 1 ブロックのローカル行数
    L: u32,
    words: Buf, // int32 [L]: a lane's row in 8-byte words / lane の 1 行の 8 バイト語の数
    act: Buf, // u64 [L]: the lanes' base pointers (every cache layer's latent, then index keys) / lane の先頭
    E: Buf, // u64 [local physical rows]: expected row hashes (0: none) / 行ハッシュの期待値
    lanes: Buf, // u64 [L, hash_rows]
    ids: Buf, // int64 [2, hash_rows]: pid, gen of the rows hashed / ハッシュする行の pid と gen
    bad: Buf, // u64 [1]
    host_map: []u8, // the PoolMap staged / PoolMap の転送用
    host_id: []i64, // [2, hash_rows]
    fault: Fault = .{},
    // DFlash2 without per-slot saves: the tokens of the request the live ring last held
    // スロットごとの写しなしの DFlash2: live のリングが最後に持った要求のトークン列
    ring_seq: std.ArrayList(i64) = .empty,
    // /health (core/engine_api.zig PoolStatus), set after every request, read by the server's thread
    // /health (core/engine_api.zig の PoolStatus)。要求ごとに更新し、サーバーのスレッドが読む
    st_free: std.atomic.Value(u64) = .init(0),
    st_slots: std.atomic.Value(u64) = .init(0),
    st_shared: std.atomic.Value(u64) = .init(0),
    st_evicted: std.atomic.Value(u64) = .init(0),
    st_cold: std.atomic.Value(u64) = .init(0),

    pub const Stats = struct { blocks: u64, free_blocks: u64, slots: u64, shared_blocks: u64, evicted_total: u64, cold_total: u64 };

    pub fn status(p: *const Pool) Stats {
        return .{ .blocks = p.meta.cfg.n_blocks, .free_blocks = p.st_free.load(.monotonic), .slots = p.st_slots.load(.monotonic), .shared_blocks = p.st_shared.load(.monotonic), .evicted_total = p.st_evicted.load(.monotonic), .cold_total = p.st_cold.load(.monotonic) };
    }

    fn noteStats(p: *Pool, evicted: usize, cold_: bool) void {
        var shared: u64 = 0;
        for (p.meta.ref) |r| shared += @intFromBool(r >= 2);
        p.st_free.store(p.meta.freeCount(), .monotonic);
        p.st_slots.store(p.meta.slots.items.len, .monotonic);
        p.st_shared.store(shared, .monotonic);
        _ = p.st_evicted.fetchAdd(evicted, .monotonic);
        if (cold_) _ = p.st_cold.fetchAdd(1, .monotonic);
    }

    fn digestWords(d: [32]u8) [8]i32 {
        return tree.words(d[0..16].*);
    }

    /// The rows of logical blocks `js` (ascending) of a slot whose table is `table`, below position `lim`, hashed with
    /// the tokens' pids and each block's generation: into E (exp false) or compared with E, mismatches into bad.
    /// Logical blocks next to each other at consecutive physical blocks go in one launch (their rows and positions are
    /// both contiguous).
    /// 表が `table` のスロットの論理ブロック `js` (昇順) のうち位置 `lim` より手前の行を、トークンの pid と各ブロックの
    /// 世代でハッシュする: E へ書く (exp が false) か、E と比べて不一致を bad に数える。隣り合う論理ブロックが連続した
    /// 物理ブロックにあれば 1 回の起動で扱う (行も位置も連続するため)。
    fn hashBlocks(p: *Pool, e: *engine.Engine, pids: []const u64, table: []const u32, js: []const usize, lim: u64, exp: bool) !void {
        var k: usize = 0;
        while (k < js.len) {
            // a run: js[k], js[k] + 1, ... at table[js[k]], table[js[k]] + 1, ...; every block but the last is whole
            // 並び: js[k], js[k] + 1, ... が table[js[k]], table[js[k]] + 1, ... にある。最後のブロック以外は丸ごと
            const j0 = js[k];
            var m: usize = 1;
            while (k + m < js.len and js[k + m] == j0 + m and table[j0 + m] == table[j0] + m and (m + 1) * p.rows_blk <= hash_rows and (j0 + m) * devmap.block < lim) m += 1;
            k += m;
            const p0: u64 = j0 * devmap.block + p.drank;
            if (lim <= p0) continue;
            const n: usize = @intCast(@min(m * p.rows_blk, (lim - p0 + p.dcp - 1) / p.dcp));
            for (0..n) |i| {
                p.host_id[i] = @bitCast(pids[@intCast(p0 + i * p.dcp)]);
                p.host_id[hash_rows + i] = @intCast(p.meta.gen[table[j0 + i / p.rows_blk]]);
            }
            try p.ids.uploadAsync(0, std.mem.sliceAsBytes(p.host_id[0..n]), e.stream.handle);
            try p.ids.uploadAsync(hash_rows * 8, std.mem.sliceAsBytes(p.host_id[hash_rows..][0..n]), e.stream.handle);
            const row0: u64 = @as(u64, table[j0]) * p.rows_blk;
            try e.glue.slotLanes(e.stream, p.act.ptr, 0, p.words.ptr, p.L, @intCast(row0), 0, @intCast(n), p.lanes.ptr);
            const eptr = p.E.ptr + row0 * 8;
            try e.glue.slotFold(e.stream, p.lanes.ptr, p.L, @intCast(n), @intCast(p0), p.dcp, p.ids.ptr, p.ids.ptr + hash_rows * 8, if (exp) 0 else eptr, if (exp) eptr else 0, p.bad.ptr);
        }
    }

    fn pidsOf(p: *Pool, seq: []const i64) ![]u64 {
        return hashing.pids(p.gpa, seq);
    }

    /// Before the first token: the plan's verify blocks wholly and the block holding resume_end up to it, against
    /// this rank's expected values (no sync; `checkOk` reads the verdict).
    /// 最初のトークンの前: プランの照合ブロック全体と、resume_end を含むブロックのそこまでを、このランクの期待値と
    /// 照合する (同期しない。判定は `checkOk` で読む)。
    fn verify(p: *Pool, e: *engine.Engine, q: *const plan.Plan, prompt: []const i64) !void {
        try p.bad.fill8(0, e.stream.handle);
        const r = q.resume_end;
        if (r == 0) return;
        const pids = try p.pidsOf(prompt[0..r]);
        defer p.gpa.free(pids);
        var js: std.ArrayList(usize) = .empty;
        defer js.deinit(p.gpa);
        for (q.verify) |j| try js.append(p.gpa, j);
        const j0 = r / devmap.block;
        if (r % devmap.block != 0 and p.env.verify != .off) try js.append(p.gpa, j0);
        if (p.fault.kind == .hash and !p.fault.done and js.items.len > 0) {
            p.fault.done = true;
            if (e.rank == p.fault.rank) {
                const off: usize = @as(usize, q.table[js.items[0]]) * p.rows_blk * 8;
                var w: [8]u8 = undefined;
                try e.stream.synchronize();
                try p.E.download(off, &w);
                w[0] ^= 1;
                try p.E.upload(off, &w);
                std.debug.print("[tensorfold] pool: TF_GLM53_POOL_FAULT=hash: rank {d} flipped the expected hash of logical block {d}'s first row\n", .{ e.rank, js.items[0] });
            }
        }
        try p.hashBlocks(e, pids, q.table, js.items, r, true);
    }

    fn checkOk(p: *Pool, e: *engine.Engine) !bool {
        var v: [1]u64 = undefined;
        try p.bad.downloadAsync(0, std.mem.sliceAsBytes(&v), e.stream.handle);
        try e.stream.synchronize();
        return v[0] == 0;
    }

    /// The active slot's table and write range into the PoolMap, after both streams finished the request before.
    /// 処理中のスロットの表と書き込み範囲を PoolMap へ。前の要求で両方の stream が終わってから書く。
    fn writeMap(p: *Pool, e: *engine.Engine, q: *const plan.Plan, n_prompt: usize) !void {
        try e.stream.synchronize();
        if (e.n_sp > 0) try e.comm.synchronize();
        var lo: u64 = q.write_lo;
        const tbl = try p.gpa.dupe(u32, q.table);
        defer p.gpa.free(tbl);
        switch (p.fault.kind) {
            .lo, .z, .oob => if (!p.fault.done and q.resume_end < n_prompt) {
                p.fault.done = true;
                if (e.rank == p.fault.rank) {
                    const j0: usize = @intCast(q.resume_end / devmap.block);
                    switch (p.fault.kind) {
                        .lo => lo = q.resume_end + p.dcp, // past the first position of either DCP parity / どちらの DCP 偶奇でも最初の位置より先
                        .z => tbl[j0] = p.meta.cfg.z(),
                        else => tbl[j0] = p.meta.cfg.z() + 3,
                    }
                    std.debug.print("[tensorfold] pool: TF_GLM53_POOL_FAULT={s}: rank {d}'s map changed (resume_end {d})\n", .{ @tagName(p.fault.kind), e.rank, q.resume_end });
                }
            },
            else => {},
        }
        const raw = try devmap.bytes(p.gpa, @intCast(lo), @intCast(q.write_hi), p.meta.cfg.z(), tbl, p.n_entries);
        defer p.gpa.free(raw);
        try e.pmap.?.upload(0, raw);
    }

    /// A common cold: every slot dropped (the rows stay, nothing claims them).
    /// 共通の cold: 全スロットを捨てる (行は残るが、どこからも参照しない)。
    fn cold(p: *Pool) !void {
        const cfg = p.meta.cfg;
        p.meta.deinit();
        p.meta = try plan.Meta.init(p.gpa, cfg);
        p.ring_seq.clearRetainingCapacity();
    }
};

fn verifyMode(v: devmap.Env.Verify) plan.Verify {
    return switch (v) {
        .sample => .sample,
        .all => .all,
        .off => .off,
    };
}

/// Every rank once after the engine is up (TF_GLM53_PREFIX=pool): the metadata and the row hashes' buffers.
/// エンジン起動後に全ランクが 1 回呼ぶ (TF_GLM53_PREFIX=pool): メタデータと行ハッシュのバッファ。
pub fn setup(ch: *lockstep.Channel, e: *engine.Engine) !void {
    const env = try devmap.Env.fromEnv(e.capacity);
    if (!env.on) return;
    const gpa = e.gpa;
    const p = try gpa.create(Pool);
    errdefer gpa.destroy(p);
    const lat_row: usize = forward.latRow(e.cfg, e.kv8, e.kv4);
    const idx_row: usize = if (e.kv8) 144 else e.cfg.index_dim * 2;
    var ptrs: std.ArrayList(u64) = .empty;
    defer ptrs.deinit(gpa);
    var wds: std.ArrayList(i32) = .empty;
    defer wds.deinit(gpa);
    for (e.state.kc) |b| {
        try ptrs.append(gpa, b.ptr);
        try wds.append(gpa, @intCast(lat_row / 8));
    }
    for (e.state.ic) |b| if (b) |x| {
        try ptrs.append(gpa, x.ptr);
        try wds.append(gpa, @intCast(idx_row / 8));
    };
    const L: u32 = @intCast(ptrs.items.len);
    const rows_blk = devmap.block / e.dcp;
    const local_rows = e.state.kc[0].len / lat_row;
    // DFlash2's per-slot saves (POOL.md 4.6.2): 0 keeps the live ring alone, as with the slots (a switch to another
    // conversation resumes the drafter partially)
    // DFlash2 のスロットごとの写し (POOL.md 4.6.2): 0 なら slots と同じく live のリング 1 本だけ (別の会話に切り替える
    // と drafter の再開は partial になる)
    const saves: u32 = if (e.drafter != null) env.draft_saves else 0;
    if (saves > 0) try e.drafter.?.initSaves(saves);
    const cfg: plan.Config = .{ .n_blocks = @intCast(env.blocks()), .capacity = e.capacity, .max_slots = env.slots, .mtp = e.mtp != null, .draft_saves = saves, .verify = verifyMode(env.verify) };
    // POOL.md 4.7: the values the environment alone does not fix, equal on every rank (else no rank starts)
    // POOL.md 4.7: 環境変数だけでは決まらない値。全ランクで同じでなければどのランクも起動しない
    // (the slot cap and the verify mode too: ranks that disagree on them would plan or check differently and every
    // request would go cold without a word, POOL.md 12.2e finding 2)
    // (スロット数の上限と照合の方式も: ランク間で違うと計画や照合が食い違い、全要求が黙って cold になる。POOL.md
    // 12.2e の見つかったこと 2)
    const mine = [_]i32{ abi_version, @intCast(cfg.n_blocks), @intCast((e.pmap.?.len - devmap.header) / 4), @intCast(L), @intCast(lat_row), @intCast(idx_row), @intCast(saves), @intFromBool(cfg.mtp), @intCast(e.dcp), @intCast(cfg.max_slots), @intFromEnum(cfg.verify) };
    var all: [64 * mine.len]i32 = undefined;
    try ch.allGather(&mine, all[0 .. e.world * mine.len]);
    for (0..e.world) |r| if (!std.mem.eql(i32, all[r * mine.len ..][0..mine.len], &mine)) {
        std.debug.print("[tensorfold] pool: rank {d}'s pool values {any} differ from rank {d}'s {any}\n", .{ e.rank, mine, r, all[r * mine.len ..][0..mine.len] });
        return error.PoolValuesDiffer;
    };
    var meta = try plan.Meta.init(gpa, cfg);
    errdefer meta.deinit();
    var E = try Buf.alloc(e.d, local_rows * 8);
    errdefer E.free();
    try E.fill8(0, null);
    p.* = .{
        .gpa = gpa, .meta = meta, .env = env, .n_entries = (e.pmap.?.len - devmap.header) / 4, .dcp = e.dcp, .drank = e.rank % e.dcp, .rows_blk = rows_blk, .L = L,
        .words = try Buf.fromHost(e.d, std.mem.sliceAsBytes(wds.items)), .act = try Buf.fromHost(e.d, std.mem.sliceAsBytes(ptrs.items)), .E = E,
        .lanes = try Buf.alloc(e.d, @as(usize, L) * hash_rows * 8), .ids = try Buf.alloc(e.d, 2 * hash_rows * 8), .bad = try Buf.alloc(e.d, 8),
        .host_map = &.{}, .host_id = try gpa.alloc(i64, 2 * hash_rows),
    };
    p.fault = try Fault.fromEnv();
    if (poison.byte()) |v| {
        try poison.all(e, v);
        try e.stream.synchronize();
    }
    e.pool = p;
    if (e.rank == 0) std.debug.print("[tensorfold] pool: {d} blocks of 256 positions + Z ({d} positions, capacity {d}), {d} slots at most, verify {s}, DFlash2 saves {d}, MTP rows cached {}\n", .{ cfg.n_blocks, env.positions, e.capacity, env.slots, @tagName(env.verify), env.draft_saves, cfg.mtp });
}

/// Rank 0, before the request is shared: its plan (the metadata unchanged) and the header's slot words; r0 =
/// resume_end.
/// 要求を共有する前の rank 0: プラン (メタデータは変えない) とヘッダのスロット用の語。r0 = resume_end。
pub fn lead(e: *engine.Engine, h: *lockstep.Header, prompt: []const i64) !plan.Plan {
    const p = e.pool.?;
    var q = try plan.makePlan(&p.meta, prompt, h.max_tokens, true);
    errdefer q.deinit(p.gpa);
    const d0 = p.meta.digest();
    var x: [21]i32 = undefined;
    x[0] = 1; // the pool reuses the target's rows in the serial mode too / pool は逐次モードでもターゲットの行を再利用する
    x[1..9].* = Pool.digestWords(d0);
    x[9..17].* = Pool.digestWords(q.digest(d0));
    x[17] = if (e.drafter != null) @intCast(lockstep.dflash_depth) else 0;
    lockstep.f64Words(if (e.drafter != null) dflash_confidence else 0, x[18..21]);
    h.extra = x;
    h.r0 = @intCast(q.resume_end);
    return q;
}

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

/// On every rank once the request is shared (see the file's comment). Any error is fatal to the caller.
/// 要求の共有後に全ランクで (ファイル冒頭の説明)。エラーはどれも呼び出し側にとって致命的。
pub fn serve(ch: *lockstep.Channel, e: *engine.Engine, h: lockstep.Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer, plan0: ?*plan.Plan, run: anytype) !engine.Stats {
    const p = e.pool.?;
    const gpa = p.gpa;
    // a prompt the engine refuses (empty, or no room for the reply) is refused as without the pool, on every rank
    // エンジンが断る prompt (空、または応答の余地がない) は、pool がないときと同じく全ランクで断る
    if (prompt.len == 0 or prompt.len >= e.capacity) return run(e, h, prompt, sink, prop);
    const x = h.extra orelse return error.MissingSlotWords;
    var own: ?plan.Plan = null;
    defer if (own) |*q| q.deinit(gpa);
    var q: ?*plan.Plan = plan0;
    if (q == null) {
        const d0 = p.meta.digest();
        if (std.mem.eql(i32, &Pool.digestWords(d0), x[1..9])) {
            if (plan.makePlan(&p.meta, prompt, h.max_tokens, x[0] != 0)) |qq| {
                own = qq;
                if (std.mem.eql(i32, &Pool.digestWords(own.?.digest(d0)), x[9..17])) q = &own.?;
            } else |err| std.debug.print("[tensorfold] pool: rank {d} could not plan ({s})\n", .{ e.rank, @errorName(err) });
        }
        if (q == null) std.debug.print("[tensorfold] pool: rank {d} differs from rank 0's plan - every rank goes cold\n", .{e.rank});
    }
    // TF_GLM53_POOL_POISON: the blocks this request frees are poisoned after the commit
    // TF_GLM53_POOL_POISON: この要求が解放するブロックは確定の後に毒で埋める
    const ref0: ?[]u32 = if (poison.byte() != null) try gpa.dupe(u32, p.meta.ref) else null;
    defer if (ref0) |r0| gpa.free(r0);
    const t0 = std.Io.Clock.awake.now(e.io);
    var why: ?[]const u8 = null;
    if (!try ch.allOk(q != null)) why = "plan" else {
        try plan.apply(&p.meta, q.?, prompt);
        try p.meta.check(q.?.X);
        try p.verify(e, q.?, prompt);
        if (!try ch.allOk(try p.checkOk(e))) why = "check";
    }
    // cold: every rank drops every slot and plans the request again from nothing (NEW, from 0)
    // cold: 全ランクが全スロットを捨て、何もない所から計画し直す (NEW、0 から)
    var cold_plan: ?plan.Plan = null;
    defer if (cold_plan) |*cq| cq.deinit(gpa);
    if (why != null) {
        try p.cold();
        cold_plan = try plan.makePlan(&p.meta, prompt, h.max_tokens, false);
        q = &cold_plan.?;
        try plan.apply(&p.meta, q.?, prompt);
    }
    const qp = q.?;
    try p.writeMap(e, qp, prompt.len);
    if (poison.byte()) |v| {
        // the blocks taken, and the resume block's rows from resume_end (its stale tail; written by this request)
        // 割り当てたブロックと、再開ブロックの resume_end 以降の行 (古い残り。この要求が書く)
        try poison.blocks(e, gpa, qp.alloc, p.rows_blk, v);
        const j0: usize = @intCast(qp.resume_end / devmap.block);
        if (qp.resume_end % devmap.block != 0 and j0 < qp.table.len) {
            const into = qp.resume_end - j0 * devmap.block;
            const k0: u64 = if (into > p.drank) (into - p.drank + p.dcp - 1) / p.dcp else 0;
            const b0: usize = qp.table[j0];
            if (b0 < p.meta.cfg.z()) try poison.rows(e, b0 * p.rows_blk + @as(usize, @intCast(k0)), (b0 + 1) * p.rows_blk, v);
        }
    }
    const check_ms = ms(e.io, t0);
    const r: usize = @intCast(qp.resume_end);
    e.start = r;
    defer e.start = 0;
    const drafted = h.mode == lockstep.mode_dflash and e.drafter != null;
    const saves = p.meta.saves.len > 0;
    var s_valid: u64 = 0; // the drafter's valid start this request (POOL.md 4.6.2 s) / この要求の drafter の正しい部分の先頭
    if (saves and drafted) {
        // the slot's save back into the live rings (unless they hold it), then the range-based resume
        // スロットの写しを live のリングへ戻し (すでに持っていなければ)、範囲付きで再開する
        if (qp.restore) |k| if (!qp.skip_copy) try e.drafter.?.restoreFrom(k);
        const rg = plan.resumeAt(r, prompt.len, if (qp.restore_live) |g| .{ g.vlo, g.vhi } else null, if (qp.restore_snap) |g| .{ g.vlo, g.vhi } else null, qp.below, e.drafter.?.cfg.window);
        s_valid = rg.s;
        e.dr_pool = .{ .kind = switch (rg.kind) {
            .none => .none,
            .live => .live,
            .saved => .saved,
            .partial => .partial,
        }, .copy = if (rg.copy) |c| .{ @intCast(c[0]), @intCast(c[1]) } else null };
    } else {
        // DFlash2 without per-slot saves: the live ring holds the last drafted request's rows; below = its match
        // スロットごとの写しなしの DFlash2: live のリングは最後の下書き付きの要求の行を持つ。below はそれとの一致
        const nb = @min(p.ring_seq.items.len, prompt.len);
        const below = std.mem.findDiff(i64, p.ring_seq.items[0..nb], prompt[0..nb]) orelse nb;
        e.dr_ring = if (r > 0) .{ e.dr_live_end, e.dr_snap_end, below } else null;
    }
    defer e.dr_pool = null;
    e.dr_live_end = null;
    e.dr_snap_end = null;
    e.dr_resume = null;
    e.out.clearRetainingCapacity();
    var st = try run(e, h, prompt, sink, prop);
    // commit: the prompt and every emitted token but the pending last one; stopped in the prompt: its chunks done
    // 確定: prompt と出力した全トークン。ただし未処理の最後の 1 つは除く。prompt の途中で止まったら終わったチャンクだけ
    const out = e.out.items;
    const in_prompt = st.finish == .cancelled and st.generated == 0;
    const seq = if (in_prompt) try gpa.dupe(i64, prompt[0..e.prefill_done]) else blk: {
        const s = try gpa.alloc(i64, prompt.len + out.len - @min(out.len, 1));
        @memcpy(s[0..prompt.len], prompt);
        if (out.len > 1) @memcpy(s[prompt.len..], out[0 .. out.len - 1]);
        break :blk s;
    };
    defer gpa.free(seq);
    const V: u64 = seq.len;
    const kv_end: u64 = if (e.mtp) |*m| m.track.kvEnd(V) else V;
    try plan.commit(&p.meta, qp, seq, if (in_prompt) e.prefill_done else prompt.len, @max(kv_end, qp.resume_end));
    try p.meta.check(null);
    // the new rows' expected hashes: from the block holding resume_end to kv_end
    // 新しい行の期待値: resume_end を含むブロックから kv_end まで
    const xi = p.meta.find(qp.X);
    const ke: u64 = if (xi) |i| p.meta.slots.items[i].kv_end else 0;
    const th = std.Io.Clock.awake.now(e.io);
    if (ke > 0) {
        const xs = &p.meta.slots.items[xi.?];
        const pids = try p.pidsOf(seq[0..@intCast(ke)]);
        defer gpa.free(pids);
        var js: std.ArrayList(usize) = .empty;
        defer js.deinit(gpa);
        var j: usize = @intCast(qp.resume_end / devmap.block);
        while (j * devmap.block < ke) : (j += 1) try js.append(gpa, j);
        try p.hashBlocks(e, pids, xs.blocks.items, js.items, ke, false);
    }
    try e.stream.synchronize();
    const hash_ms = ms(e.io, th);
    // DFlash2 with per-slot saves: the table of POOL.md 4.6.2 (a drafted request that reached the decode saves its live
    // ring and its snapshot into X's save; one stopped in the prompt or a serial one saves nothing)
    // スロットごとの写しのある DFlash2: POOL.md 4.6.2 の表 (decode まで進んだ下書き付きの要求は live のリングと snapshot
    // を X の写しへ保存する。prompt の途中で止まった要求と逐次モードの要求は何も保存しない)
    if (saves) {
        const dr = e.drafter.?;
        const took_snap = e.dr_snap_end != null;
        if (drafted and !in_prompt) {
            const live = plan.Ring.live(dr.context_end, V, s_valid, dr.cfg.block);
            const snap: ?plan.Ring = if (took_snap) plan.Ring.snap(prompt.len, s_valid) else null;
            if (try plan.commitSave(&p.meta, qp, live, snap, true)) |k| try dr.saveTo(k);
        } else _ = try plan.commitSave(&p.meta, qp, null, null, drafted);
        try p.meta.check(null);
    }
    if (ref0) |r0| if (poison.byte()) |v| {
        var freed: std.ArrayList(u32) = .empty;
        defer freed.deinit(gpa);
        for (r0, p.meta.ref, 0..) |a, b, k| if (a > 0 and b == 0) try freed.append(gpa, @intCast(k));
        try poison.blocks(e, gpa, freed.items, p.rows_blk, v);
    };
    // TF_GLM53_KV_DIGEST: the rows the slot keeps, [0, its kv_end) / スロットが持つ行 [0, kv_end)
    if (kvdigest.on()) if (p.meta.find(qp.X)) |i| {
        const xs2 = &p.meta.slots.items[i];
        try kvdigest.report(e, @min(V, xs2.kv_end), if (in_prompt) e.prefill_done else prompt.len, kv_end, xs2.blocks.items);
    };
    // the drafter's ring after the request (as slots_serve.zig)
    // 要求の後の drafter のリング (slots_serve.zig と同じ)
    if (in_prompt or h.mode != lockstep.mode_dflash) {
        e.dr_live_end = null;
        e.dr_snap_end = null;
        p.ring_seq.clearRetainingCapacity();
    } else {
        e.dr_live_end = if (e.drafter) |dr| dr.context_end else null;
        p.ring_seq.clearRetainingCapacity();
        try p.ring_seq.appendSlice(gpa, seq);
    }
    st.cached = r;
    p.noteStats(if (why == null) qp.evicted.len else 0, why != null);
    if (e.rank == 0) {
        var shared: usize = 0;
        if (p.meta.find(qp.X)) |i| for (p.meta.slots.items[i].blocks.items) |b| {
            shared += @intFromBool(p.meta.ref[b] >= 2);
        };
        std.debug.print("[tensorfold] pool: prompt {d}, {s}{s} slot {d} (h {d}, c {d}), resumed {d}, checks {d:.1} ms, hashes {d:.1} ms, blocks +{d} (shared {d}), evicted {d}, shrunk {d}, prefilled {d}{s} in {d:.2}s, kv_end {d}/{d}, drafter {s}; {d} slots, {d} free blocks\n", .{
            prompt.len, if (why != null) "cold: " else "", if (why) |w| w else @tagName(qp.action), qp.X, qp.h, qp.c, r, check_ms, hash_ms, qp.alloc.len, shared, qp.evicted.len, qp.shrunk.len,
            e.prefill_done - r, if (st.finish == .cancelled) (if (st.generated == 0) " (stopped in the prompt)" else " (stopped while decoding)") else "", @as(f64, @floatFromInt(st.prefill_ns)) / 1e9, ke, V, if (e.dr_resume) |rr| @tagName(rr) else "-", p.meta.slots.items.len, p.meta.freeCount(),
        });
    }
    return st;
}
