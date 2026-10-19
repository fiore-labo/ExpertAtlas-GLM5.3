//! The slot cache's part of serving: the store set up at start; per request, rank 0's plan before the request is
//! shared, then on every rank the plan, the copies and their checks, publish or a common cold, the prompt from c, the
//! commit.
//! スロットキャッシュの担当分: 起動時にストアを用意する。要求ごとに、共有前に rank 0 がプランを作り、続いて全ランクで
//! プラン、コピーとその検査、publish か全ランク共通の cold、c からの prompt、commit を行う。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const lockstep = @import("lockstep.zig");
const slots = @import("slots.zig");
const tree = @import("slotcache/tree.zig");

pub const Plan = tree.Plan;

/// The drafter's confidence, the same on every rank.
/// ドラフタの confidence。全ランクで同じ値。
const dflash_confidence: f64 = 0.4;

fn wantTokens(e: *const engine.Engine, max_tokens: i32, prompt_len: usize) usize {
    const m: usize = @intCast(@max(1, max_tokens));
    return @max(1, @min(m, e.capacity - prompt_len));
}

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

/// posix_fadvise(DONTNEED) on every file under `dir` (depth-first, `depth` levels), so their clean page cache goes
/// back to free memory: on GB10's unified memory the device allocator does not reclaim it. Adds to files / bytes.
/// `dir` 以下の全ファイルに posix_fadvise(DONTNEED) をかけ (深さ優先、`depth` 段まで)、きれいなページキャッシュを
/// 空きメモリに戻す: GB10 の統合メモリではデバイスのアロケータがこれを回収しない。files / bytes に加算する。
pub fn dropFileCache(io: std.Io, dir: []const u8, depth: u32, files: *usize, bytes: *u64) void {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    while (it.next(io) catch null) |ent| {
        var buf: [4096]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, ent.name }) catch continue;
        switch (ent.kind) {
            .directory => if (depth > 0) dropFileCache(io, path, depth - 1, files, bytes),
            .file, .sym_link => {
                const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch continue;
                defer f.close(io);
                _ = std.os.linux.fadvise(f.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
                files.* += 1;
                bytes.* += (f.stat(io) catch continue).size;
            },
            else => {},
        }
    }
}

/// The store sized from the smallest free device memory over the ranks (after the checkpoint's page cache is
/// dropped), allocated on every rank or on none (then S = 0: one conversation kept). Every rank calls it once after
/// the engine is up.
/// 全ランクで最小の空きデバイスメモリからストアの大きさを決め (チェックポイントのページキャッシュを捨てた後)、
/// 全ランクで確保するか、どこでも確保しない (そのときは S = 0 で会話を 1 つだけ保持)。
/// エンジン起動後に全ランクが 1 回呼ぶ。
pub fn setup(ch: *lockstep.Channel, e: *engine.Engine, ctx: *const cuda.Context, env: slots.Env, dirs: []const []const u8) !void {
    if (!env.on) return;
    // the store copies the compute cache's rows as contiguous runs: not under TF_GLM53_POOL_TEST's permuted rows
    // ストアは計算キャッシュの行を連続した並びとしてコピーする: TF_GLM53_POOL_TEST の並べ替えた行では使えない
    if (e.pmap != null) return error.PoolTestWithSlots;
    const W = e.world;
    var files: usize = 0;
    var dropped: u64 = 0;
    for (dirs) |dir| if (dir.len > 0) dropFileCache(e.io, dir, 3, &files, &dropped);
    try e.stream.synchronize();
    const free = (try ctx.memInfo()).free;
    var all: [64]i32 = undefined;
    try ch.allGather(&.{@intCast(free >> 20)}, all[0..W]);
    var free_min: u64 = std.math.maxInt(u64);
    for (all[0..W]) |v| free_min = @min(free_min, @as(u64, @intCast(v)) << 20);
    const lat_row: usize = forward.latRow(e.cfg, e.kv8, e.kv4);
    const idx_row: usize = if (e.kv8) 144 else e.cfg.index_dim * 2;
    var lanes: usize = e.state.kc.len;
    for (e.state.ic) |b| lanes += @intFromBool(b != null);
    var row_bytes: usize = e.state.kc.len * lat_row;
    for (e.state.ic) |b| row_bytes += if (b != null) idx_row else 0;
    const sz = slots.storePositions(env, free_min, row_bytes, lanes, e.dcp, e.capacity);
    const got = try ch.share(if (e.rank == 0) &.{@as(i32, @intCast(sz[0]))} else null);
    var S: i64 = got[0];
    ch.gpa.free(got);
    const drank: u32 = if (e.dgroup) |g| @intCast(g.rank) else 0;
    var store: ?*slots.Store = slots.Store.init(e.gpa, e.d, &e.glue, e.stream, &e.state, lat_row, idx_row, e.dcp, drank, e.capacity, S, e.rank, env) catch |err| blk: {
        std.debug.print("[tensorfold] slots: rank {d} could not allocate a store of {d} positions ({s})\n", .{ e.rank, S, @errorName(err) });
        break :blk null;
    };
    // out of memory somewhere: every rank goes without
    // どこかでメモリ不足: 全ランクがストアなしにする
    if (!try ch.allOk(store != null)) {
        if (store) |s| s.deinit();
        if (e.rank == 0) std.debug.print("[tensorfold] slot cache: warning: a rank could not allocate the store of {d} positions, so there is none: a request resumes only the conversation just before it (lower TF_GLM53_PREFIX_STORE or free memory)\n", .{S});
        S = 0;
        store = try slots.Store.init(e.gpa, e.d, &e.glue, e.stream, &e.state, lat_row, idx_row, e.dcp, drank, e.capacity, 0, e.rank, env);
    }
    e.slots = store;
    if (e.rank == 0) {
        const g: f64 = 1 << 30;
        const after = (try ctx.memInfo()).free;
        std.debug.print("[tensorfold] slot cache: store of {d} positions beside the compute cache's {d} ({d:.2} GiB a rank; free memory sizes it {d}{s}), smallest free {d:.1} GiB before it, {d:.1} after (rank 0), reserve {d} GiB; page cache of {d} checkpoint files ({d:.0} GiB) dropped; resumes of at least {d} tokens\n", .{
            S, e.capacity, @as(f64, @floatFromInt(store.?.nbytes())) / g, sz[1], if (env.store != null) ", TF_GLM53_PREFIX_STORE" else "",
            @as(f64, @floatFromInt(free_min)) / g, @as(f64, @floatFromInt(after)) / g, env.peak_gib + env.margin_gib, files, @as(f64, @floatFromInt(dropped)) / g, env.min,
        });
    }
}

/// Rank 0, before the request is shared: its plan (on a copy of the metadata) and the header's slot words; r0 = c.
/// 要求を共有する前の rank 0: プラン (メタデータのコピー上で作る) とヘッダのスロット用の語。r0 = c。
pub fn lead(e: *engine.Engine, h: *lockstep.Header, prompt: []const i64) !Plan {
    const sl = e.slots.?;
    const want = wantTokens(e, h.max_tokens, prompt.len);
    // with a drafter loaded, a request without drafts (the serial mode) prefills from 0
    // ドラフタがあるとき、ドラフトなしの要求 (逐次モード) は 0 から prefill する
    const resume_ok = h.mode == lockstep.mode_dflash or e.drafter == null;
    var p = try sl.plan(prompt, want, resume_ok);
    errdefer p.deinit();
    var x: [21]i32 = undefined;
    x[0] = @intFromBool(resume_ok);
    x[1..9].* = tree.words(p.d0);
    x[9..17].* = tree.words(tree.planDigest(&p));
    x[17] = if (e.drafter != null) @intCast(lockstep.dflash_depth) else 0;
    lockstep.f64Words(if (e.drafter != null) dflash_confidence else 0, x[18..21]);
    h.extra = x;
    h.r0 = @intCast(p.c);
    return p;
}

pub const Runner = *const fn (e: *engine.Engine, h: lockstep.Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) anyerror!engine.Stats;

/// On every rank once the request is shared: rank 0's plan (followers make the same one from the same metadata, or
/// vote for cold), the copies and their checks, publish (every rank succeeded) or a common cold, the prompt from c,
/// the commit. Any error is fatal to the caller (the ranks may disagree from here on).
/// 要求の共有後に全ランクで: rank 0 のプラン (他ランクは同じメタデータから同じプランを作るか、cold に投票する)、
/// コピーとその検査、publish (全ランク成功時) か全ランク共通の cold、c からの prompt、commit。
/// エラーはどれも呼び出し側にとって致命的 (ここから先はランク間で状態が食い違いうる)。
pub fn serve(ch: *lockstep.Channel, e: *engine.Engine, h: lockstep.Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer, plan0: ?*Plan, run: Runner) !engine.Stats {
    const sl = e.slots.?;
    const x = h.extra orelse return error.MissingSlotWords;
    const want = wantTokens(e, h.max_tokens, prompt.len);
    var own: ?Plan = null;
    defer if (own) |*p| p.deinit();
    var plan: ?*Plan = plan0;
    if (plan == null) {
        if (std.mem.eql(i32, &tree.words(sl.meta.digest()), x[1..9])) {
            if (sl.plan(prompt, want, x[0] != 0)) |p| {
                own = p;
                if (std.mem.eql(i32, &tree.words(tree.planDigest(&own.?)), x[9..17])) plan = &own.?;
            } else |err| std.debug.print("[tensorfold] slots: rank {d} could not plan ({s})\n", .{ e.rank, @errorName(err) });
        }
        if (plan == null) std.debug.print("[tensorfold] slots: rank {d} differs from rank 0's plan - every rank goes cold\n", .{e.rank});
    }
    const t0 = std.Io.Clock.awake.now(e.io);
    var why: ?[]const u8 = null;
    var c: usize = 0;
    var hh: usize = 0;
    if (!try ch.allOk(plan != null)) {
        why = "plan";
    } else {
        try sl.execute(plan.?);
        if (try ch.allOk(try sl.ok())) {
            try sl.publish(plan.?);
            c = @intCast(plan.?.c);
            hh = @intCast(plan.?.h);
        } else why = "check";
    }
    if (why != null) try sl.cold();
    const copy_ms = ms(e.io, t0);
    e.start = c;
    defer e.start = 0;
    e.dr_ring = if (c > 0) .{ e.dr_live_end, e.dr_snap_end, hh } else null;
    e.dr_live_end = null;
    e.dr_resume = null;
    e.out.clearRetainingCapacity();
    var st = try run(e, h, prompt, sink, prop);
    // commit: the prompt and every emitted token but the pending last one (not yet in the caches)
    // commit: prompt と出力した全トークン。ただし未処理の最後の 1 つ (まだキャッシュにない) は除く
    const out = e.out.items;
    // stopped in the prompt: only its chunks done are in the caches
    // prompt の途中で止まった: キャッシュにあるのは終わったチャンクだけ
    if (st.finish == .cancelled and st.generated == 0) {
        try sl.commit(c, prompt[0..e.prefill_done]);
        e.dr_live_end = null;
        e.dr_snap_end = null;
    } else {
        const seq = try e.gpa.alloc(i64, prompt.len + out.len - @min(out.len, 1));
        defer e.gpa.free(seq);
        @memcpy(seq[0..prompt.len], prompt);
        if (out.len > 1) @memcpy(seq[prompt.len..], out[0 .. out.len - 1]);
        try sl.commit(c, seq);
        e.dr_live_end = if (h.mode == lockstep.mode_dflash) if (e.drafter) |dr| dr.context_end else null else null;
    }
    st.cached = c;
    if (e.rank == 0) {
        var copied: i64 = 0;
        if (why == null) for (plan.?.ops) |op| {
            copied += op.b - op.a;
        };
        const s = sl.meta.summary();
        std.debug.print("[tensorfold] slots: prompt {d}, resumed {d} ({s}{s}), copies + checks {d:.0} ms, copied {d}, evicted {d}, prefilled {d}{s} in {d:.2}s, drafter {s}; store {d}/{d}, {d} leaves, {d} nodes\n", .{
            prompt.len, c, if (why != null) "cold: " else @tagName(plan.?.mode), why orelse "", copy_ms, copied, if (why == null) plan.?.evicted else 0,
            e.prefill_done - c, if (st.finish == .cancelled) (if (st.generated == 0) " (stopped in the prompt)" else " (stopped while decoding)") else "", @as(f64, @floatFromInt(st.prefill_ns)) / 1e9, if (e.dr_resume) |r| @tagName(r) else "-", s.stored, sl.S, s.leaves, s.nodes,
        });
    }
    return st;
}
