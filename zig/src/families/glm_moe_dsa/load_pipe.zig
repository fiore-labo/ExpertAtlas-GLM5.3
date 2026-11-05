// Fiore-lab: the start's weight load in parallel (TF_GLM53_LOAD_*), not in TensorFold 1.0.2.
//! The start's weight load (Engine.create) in parallel, on by default since 10-31 (READERS 16, PACKERS 8, AHEAD 4: on
//! the ring the weights 267 -> 40.5 s a rank, the same bytes). READERS=0 PACKERS=0 runs the load on one thread as
//! before (each tensor read through the mapped shard by page faults of 128 KiB, joined on the host and copied to the
//! GPU one tensor at a time; 10-31 on the ring: 92.9 GB a rank in about 270 s, 0.35 GB/s).
//!   TF_GLM53_LOAD_READERS=N   (default 16; 0: none) N threads read each load step's byte ranges (this rank's shares
//!                             only: the plan's banks' expert parts, the checkpoint's tensors) into the page cache
//!                             ahead of the loader, with
//!                             positional reads of up to 2 MiB; the loader waits for its step, then maps the pages
//!                             read; once it unmapped them the step's pages are dropped (the ranges only, so the
//!                             steps read ahead stay)
//!   TF_GLM53_LOAD_AHEAD=K     steps read beyond the one loading (default 4: the page cache holds about 5 layers'
//!                             shares, 1.2 GB each, instead of everything read since the last reclaim; 40.5 s against
//!                             42.2 s with 2 on the ring)
//!   TF_GLM53_LOAD_PACKERS=M   (default 8; 0: none) M threads join the routed experts' stored parts
//!                             (moe.loadFromPacked) straight into pinned chunks, which the GPU copies from
//!                             asynchronously (one copy an expert, at the same offsets as moe.loadFrom's): no host
//!                             share a tensor, no pageable copy
//!   TF_GLM53_LOAD_PINNED_MB=P the packers' pinned memory, all chunks together (default 192; 2 M chunks)
//!   TF_GLM53_LOAD_STATS=1     the weights' seconds (with or without the pipe), and rank 0's line a step
//!   TF_GLM53_LOAD_DIGEST=1    every loaded weight hashed after the load, a line a layer a rank (load digest), to
//!                             compare two starts' weights bit for bit
//! Every rank sets the pipe up or none does (Engine.create agrees on it as on the plan's checks); everything it holds
//! (threads, buffers, pinned memory, open files, the pages it read) is gone when the weights are loaded, before the
//! tuning, the buffers and the KV caches.
//! 起動時の重みの読み込み (Engine.create) を並列にする。10-31 から既定で on (READERS 16、PACKERS 8、AHEAD 4。リングで
//! 1 ランクの重みが 267 → 40.5 秒、バイトは同じ)。READERS=0 PACKERS=0 なら従来どおり 1 スレッドで読む (各テンソルを
//! map した shard から 128 KiB のページフォールトで読み、ホストで連結し、1 テンソルずつ GPU に写す。10-31 のリングでは
//! 1 ランク 92.9 GB を約 270 秒、0.35 GB/s)。
//!   TF_GLM53_LOAD_READERS=N   (既定 16、0 はなし) N 個のスレッドが、読み込みの各ステップのバイト範囲 (このランクの
//!                             分だけ: 計画のバンクの expert の分、チェックポイントのテンソル) を、最大 2 MiB の位置指定の読み込みでローダーより
//!                             先にページキャッシュへ読む。ローダーは自分のステップを待ってから読まれたページを map
//!                             する。ローダーが unmap したら、そのステップのページを落とす (範囲だけなので、先読み
//!                             したステップは残る)
//!   TF_GLM53_LOAD_AHEAD=K     読み込み中のステップの先に読むステップ数 (既定 4: ページキャッシュは約 5 層分、1 層
//!                             1.2 GB を持つ。前回の回収以降に読んだものすべてではなく)
//!   TF_GLM53_LOAD_PACKERS=M   (既定 8、0 はなし) M 個のスレッドが routed expert の保存された分
//!                             (moe.loadFromPacked) を pin したチャンクへ直接連結し、GPU はそこから非同期に写す
//!                             (expert ごとに 1 回、moe.loadFrom と同じ位置へ)。テンソルごとのホストの写しも、
//!                             pageable のコピーも無い
//!   TF_GLM53_LOAD_PINNED_MB=P packer の pin したメモリ (全チャンクの合計。既定 192、チャンクは 2 M 個)
//!   TF_GLM53_LOAD_STATS=1     重みの秒数 (pipe の有無によらず) と、rank 0 のステップごとの 1 行
//!   TF_GLM53_LOAD_DIGEST=1    読み込みの後に全重みのハッシュを取り、ランクごと・層ごとに 1 行出す (load digest)。2 回の
//!                             起動の重みがビットまで同じかを比べるため
//! pipe は全ランクで用意するか、どのランクでも用意しない (Engine.create が計画の検査と同じく合意する)。pipe が持つもの
//! (スレッド、バッファ、pin したメモリ、開いたファイル、読んだページ) は、重みを読み終えた時点で、調整・バッファ・
//! KV キャッシュより前にすべて無くなる。
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const split = @import("split.zig");
const model = @import("model.zig");
const moe = @import("moe.zig");
const linear = @import("linear.zig");
const expert_plan = @import("expert_plan.zig");

const st = core.safetensors;

/// The largest positional read (a reader's buffer).
/// 位置指定の読み込みの最大 (reader のバッファ)。
const piece_bytes: usize = 2 << 20;
/// Ranges of a file closer than this are read as one (the bytes between come along).
/// これより近いファイルの範囲は 1 つとして読む (間のバイトも読む)。
const merge_gap: usize = 64 << 10;
const page: usize = 4096;

pub const Env = struct {
    readers: usize = 0,
    ahead: usize = 2,
    packers: usize = 0,
    pinned_mb: usize = 192,
    stats: bool = false,
    digest: bool = false,

    /// The TF_GLM53_LOAD_* settings; a value that is not a number (or out of range) is refused.
    /// TF_GLM53_LOAD_* の設定。数でない値 (または範囲外) は断る。
    pub fn fromEnv() !Env {
        var e: Env = .{};
        e.readers = try num("TF_GLM53_LOAD_READERS", 16);
        e.ahead = try num("TF_GLM53_LOAD_AHEAD", 4);
        e.packers = try num("TF_GLM53_LOAD_PACKERS", 8);
        e.pinned_mb = try num("TF_GLM53_LOAD_PINNED_MB", 192);
        e.stats = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_GLM53_LOAD_STATS") orelse "0"), "1");
        e.digest = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_GLM53_LOAD_DIGEST") orelse "0"), "1");
        if (e.readers > 64 or e.packers > 64 or e.ahead == 0 or e.ahead > 16) return error.BadLoadSetting;
        if (e.packers > 0 and (e.pinned_mb < 16 or e.pinned_mb > 4096)) return error.BadLoadSetting;
        return e;
    }

    /// Whether anything runs beside the loader (statistics and digests alone do not).
    /// ローダーの横で何かが動くか (統計とダイジェストだけなら動かない)。
    pub fn on(e: Env) bool {
        return e.readers > 0 or e.packers > 0;
    }
};

fn num(name: [:0]const u8, default: usize) !usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch {
        std.log.err("glm53: {s}={s}: not a number", .{ name, std.mem.span(v) });
        return error.BadLoadSetting;
    };
}

/// The order Engine.create loads in, as steps: the decoder layers, the MTP layer, the DCP partner's q_b / kv_b of each
/// layer (TF_GLM53_PEER_Q; the MTP layer's last), then the embedding, lm_head and final norm.
/// Engine.create が読む順をステップにしたもの: decoder の層、MTP 層、各層の DCP の相手の q_b / kv_b
/// (TF_GLM53_PEER_Q。MTP 層のものが最後)、そして embedding、lm_head、最後の norm。
pub const Layout = struct {
    n: usize,
    mtp: bool,
    peer: bool,

    pub fn layer(_: Layout, i: usize) usize {
        return i;
    }
    pub fn mtpStep(l: Layout) usize {
        return l.n;
    }
    pub fn peerStep(l: Layout, i: usize) usize {
        return l.n + @intFromBool(l.mtp) + i;
    }
    pub fn tail(l: Layout) usize {
        return l.peerStep(if (l.peer) l.n + @intFromBool(l.mtp) else 0);
    }
    pub fn count(l: Layout) usize {
        return l.tail() + 1;
    }
};

/// What Engine.create runs the load through: inactive (both null) unless TF_GLM53_LOAD_READERS or _PACKERS is set.
/// Engine.create が読み込みに使うもの: TF_GLM53_LOAD_READERS か _PACKERS が無ければ不活性 (両方 null)。
pub const Pipe = struct {
    env: Env = .{},
    lay: Layout = .{ .n = 0, .mtp = false, .peer = false },
    pf: ?*Prefetch = null,
    pk: ?*Packers = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, env: Env, reader: *const weights.Reader, plan: ?*const expert_plan.Plan, cfg: Config, lay: Layout, rank: u32) !Pipe {
        var p: Pipe = .{ .env = env, .lay = lay };
        errdefer p.deinit();
        if (env.readers > 0) p.pf = try Prefetch.init(gpa, io, env, reader, plan, cfg, lay);
        if (env.packers > 0) p.pk = try Packers.init(gpa, io, d, env.packers, env.pinned_mb);
        if (rank == 0) {
            if (p.pf) |f| std.log.info("glm53: load pipe: {d} readers, {d} steps ahead of the loader ({d} steps)", .{ env.readers, f.ahead, f.st.len });
            if (p.pk) |k| std.log.info("glm53: load pipe: {d} packers into {d} pinned chunks of {d} MiB (TF_GLM53_LOAD_PACKERS)", .{ env.packers, k.slots.len, k.chunk >> 20 });
        }
        return p;
    }

    pub fn active(p: *const Pipe) bool {
        return p.pf != null or p.pk != null;
    }

    /// Waits until the readers have read step `step` (nothing without readers).
    /// ステップ `step` を reader が読み終えるまで待つ (reader が無ければ何もしない)。
    pub fn wait(p: *Pipe, step: usize) void {
        if (p.pf) |f| f.wait(step);
    }

    /// The loader is done with `step` and `r`'s maps: closed, then the step's pages dropped (with readers: the ranges
    /// they read, so the steps read ahead stay; without: `r`'s files whole, after the unmap).
    /// ローダーは `step` と `r` の map を使い終えた: 閉じてから、ステップのページを落とす (reader があれば読んだ範囲
    /// だけ。先読みしたステップは残る。無ければ unmap の後に `r` のファイル全体)。
    pub fn finish(p: *Pipe, r: *weights.Reader, step: usize) void {
        if (p.pf) |f| {
            r.releaseKeep();
            f.done(step);
        } else r.releaseDrop();
    }

    /// The load is over: the statistics (TF_GLM53_LOAD_STATS, or the pipe on), then everything the pipe held freed.
    /// 読み込みが終わった: 統計を出し (TF_GLM53_LOAD_STATS、または pipe が有効なとき)、pipe が持っていたものをすべて返す。
    pub fn end(p: *Pipe, io: std.Io, rank: u32, t0: std.Io.Timestamp) void {
        const ms: i64 = @intCast(@max(0, t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds()));
        if (p.env.stats or p.active()) std.log.info("glm53: rank {d}: weights loaded in {d:.1} s{s}", .{ rank, @as(f64, @floatFromInt(ms)) / 1000.0, if (p.active()) " (TF_GLM53_LOAD_READERS / _PACKERS)" else "" });
        if (p.pf) |f| f.report(rank, ms);
        if (p.pk) |k| k.report(rank);
        p.deinit();
    }

    pub fn deinit(p: *Pipe) void {
        if (p.pf) |f| f.deinit();
        p.pf = null;
        if (p.pk) |k| k.deinit();
        p.pk = null;
    }
};

// ------------------------------------------------------------------------- readers / reader (先読み) ---

/// One positional read: `len` bytes at `off` of `fd`.
/// 位置指定の読み込み 1 つ: `fd` の `off` から `len` バイト。
const Range = struct {
    fd: std.c.fd_t,
    off: usize,
    len: usize,

    fn lessThan(_: void, x: Range, y: Range) bool {
        return if (x.fd != y.fd) x.fd < y.fd else x.off < y.off;
    }
};

/// A checkpoint tensor a step reads: `name` in `file` (of the checkpoint's index), rank `share`'s part of it, or all
/// of it (`whole`: a pre-sharded part, read as stored).
/// ステップが読むチェックポイントのテンソル: `file` (チェックポイントの索引のもの) の `name`。ランク `share` の分か、
/// 全体 (`whole`: 事前分割した分で、保存のまま読む)。
const Want = struct {
    name: []const u8,
    file: []const u8,
    share: usize,
    whole: bool,

    fn byFile(_: void, x: Want, y: Want) bool {
        return std.mem.order(u8, x.file, y.file) == .lt;
    }
};

const Step = struct {
    wants: []Want,
    // an MoE layer the plan covers: its banks' layer files are read too
    // 計画が覆う MoE 層: そのバンクの層ファイルも読む
    layer: ?usize,
};

const Phase = enum { pending, expanding, reading, ready };

const State = struct {
    phase: Phase = .pending,
    pieces: []Range = &.{},
    fds: []std.c.fd_t = &.{},
    next: usize = 0, // the next piece a reader takes / reader が次に取る piece
    left: usize = 0, // pieces not read yet (taken or not) / まだ読み終えていない piece
};

const Expanded = struct { pieces: []Range, fds: []std.c.fd_t };

pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    plan: ?*const expert_plan.Plan,
    rank: usize,
    world: usize,
    k: usize, // stored shares a rank joins / 1 ランクが連結する保存された分の数
    ahead: usize,
    steps: []Step,
    st: []State,
    bufs: [][]u8,
    threads: []std.Thread,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    // the step the loader is on: readers work on [consumed, consumed + 1 + ahead)
    // ローダーがいるステップ: reader は [consumed, consumed + 1 + ahead) を扱う
    consumed: usize = 0,
    stop: bool = false,
    // under the mutex: bytes read, failed reads and expansions (the loader's own faults read those bytes instead)
    // mutex の下で: 読んだバイト、失敗した読み込みと展開 (そのバイトはローダー自身のフォールトが読む)
    bytes: u64 = 0,
    read_errors: usize = 0,
    expand_errors: usize = 0,
    first_err: ?anyerror = null,
    // the loader's side (its thread only): time waited, the current step's ready time
    // ローダー側 (そのスレッドだけ): 待った時間、今のステップが揃った時刻
    waited_ns: u64 = 0,
    stats: bool = false,
    t_ready: std.Io.Timestamp = undefined,
    step_wait_ns: u64 = 0,

    /// The steps from `reader`'s index (this rank's shares of every tensor Engine.create reads, by step), then the
    /// readers started.
    /// `reader` の索引からステップを作り (Engine.create が読む全テンソルのこのランクの分をステップごとに)、reader を
    /// 起動する。
    pub fn init(gpa: std.mem.Allocator, io: std.Io, env: Env, reader: *const weights.Reader, plan: ?*const expert_plan.Plan, cfg: Config, lay: Layout) !*Prefetch {
        const p = try gpa.create(Prefetch);
        errdefer gpa.destroy(p);
        const k = reader.stored_world / reader.world;
        const steps = try buildSteps(gpa, reader, plan, cfg, lay, k);
        errdefer freeSteps(gpa, steps);
        const sts = try gpa.alloc(State, steps.len);
        errdefer gpa.free(sts);
        @memset(sts, .{});
        const bufs = try gpa.alloc([]u8, env.readers);
        errdefer gpa.free(bufs);
        var nb: usize = 0;
        errdefer for (bufs[0..nb]) |b| gpa.free(b);
        for (bufs) |*b| {
            b.* = try gpa.alloc(u8, piece_bytes);
            nb += 1;
        }
        const threads = try gpa.alloc(std.Thread, env.readers);
        errdefer gpa.free(threads);
        p.* = .{
            .gpa = gpa, .io = io, .dir = reader.dir, .plan = plan, .rank = reader.rank, .world = reader.world, .k = k,
            .ahead = env.ahead, .steps = steps, .st = sts, .bufs = bufs, .threads = threads[0..0], .stats = env.stats,
        };
        errdefer p.stopJoin();
        for (threads, 0..) |*t, i| {
            // 4 MiB: a step's expansion parses a bank file's 2.8 MiB header (std.json) on the reader's stack frames
            // 4 MiB: ステップの展開はバンクの層ファイルの 2.8 MiB の header を reader のスタック上で解析する (std.json)
            t.* = try std.Thread.spawn(.{ .stack_size = 4 << 20 }, worker, .{ p, bufs[i] });
            p.threads = threads[0 .. i + 1];
        }
        return p;
    }

    fn stopJoin(p: *Prefetch) void {
        p.mutex.lockUncancelable(p.io);
        p.stop = true;
        p.cond.broadcast(p.io);
        p.mutex.unlock(p.io);
        for (p.threads) |t| t.join();
        p.threads = p.threads[0..0];
    }

    /// The readers stopped, every step's pages still held dropped, files closed, memory freed.
    /// reader を止め、まだ持っている各ステップのページを落とし、ファイルを閉じ、メモリを返す。
    pub fn deinit(p: *Prefetch) void {
        const gpa = p.gpa;
        const all = p.threads.ptr[0..p.bufs.len];
        p.stopJoin();
        for (p.st) |*s| release(gpa, s);
        gpa.free(all);
        for (p.bufs) |b| gpa.free(b);
        gpa.free(p.bufs);
        gpa.free(p.st);
        freeSteps(gpa, p.steps);
        gpa.destroy(p);
    }

    /// Blocks until step `i` is read (or could not be: the loader's faults read it then).
    /// ステップ `i` が読まれるまで止まる (読めなかったときも返る: そのときはローダーのフォールトが読む)。
    pub fn wait(p: *Prefetch, i: usize) void {
        const t = std.Io.Clock.awake.now(p.io);
        p.mutex.lockUncancelable(p.io);
        // only the loader's own step can be waited for (the readers stay in its window); another is not waited for
        // 待てるのはローダー自身のステップだけ (reader はその窓の中にいる)。それ以外は待たない
        if (i < p.st.len and i == p.consumed) {
            while (p.st[i].phase != .ready) p.cond.waitUncancelable(p.io, &p.mutex);
        }
        p.mutex.unlock(p.io);
        p.t_ready = std.Io.Clock.awake.now(p.io);
        p.step_wait_ns = @intCast(@max(0, t.durationTo(p.t_ready).toNanoseconds()));
        p.waited_ns += p.step_wait_ns;
    }

    /// The loader finished step `i` and unmapped its files: the readers move past it and its pages are dropped.
    /// ローダーがステップ `i` を終え、そのファイルを unmap した: reader は先へ進み、そのページを落とす。
    pub fn done(p: *Prefetch, i: usize) void {
        if (i >= p.st.len) return;
        p.mutex.lockUncancelable(p.io);
        if (i + 1 > p.consumed) p.consumed = i + 1;
        const ready = p.st[i].phase == .ready;
        var s = p.st[i];
        if (ready) p.st[i] = .{ .phase = .ready };
        p.cond.broadcast(p.io);
        p.mutex.unlock(p.io);
        if (p.stats and p.rank == 0) {
            const ms = @max(0, p.t_ready.durationTo(std.Io.Clock.awake.now(p.io)).toMilliseconds());
            var b: usize = 0;
            for (s.pieces) |r| b += r.len;
            std.log.info("glm53: load step {d}: {d} MiB read ahead, waited {d} ms, loaded in {d} ms", .{ i, b >> 20, p.step_wait_ns / std.time.ns_per_ms, ms });
        }
        // a step not read yet stays with the readers until deinit
        // まだ読み終えていないステップは deinit まで reader に残す
        if (ready) release(p.gpa, &s);
    }

    fn report(p: *Prefetch, rank: u32, ms: i64) void {
        p.mutex.lockUncancelable(p.io);
        const bytes = p.bytes;
        const re = p.read_errors;
        const ee = p.expand_errors;
        const fe = p.first_err;
        p.mutex.unlock(p.io);
        const gib = @as(f64, @floatFromInt(bytes)) / (1 << 30);
        const s = @as(f64, @floatFromInt(ms)) / 1000.0;
        std.log.info("glm53: rank {d}: load readers: {d:.1} GiB read ahead ({d:.2} GiB/s over the load), the loader waited {d:.1} s for them", .{ rank, gib, if (s > 0) gib / s else 0, @as(f64, @floatFromInt(p.waited_ns)) / 1e9 });
        if (re + ee > 0) std.log.warn("glm53: rank {d}: load readers: {d} reads and {d} steps failed ({s}): the loader read those bytes itself", .{ rank, re, ee, if (fe) |e| @errorName(e) else "-" });
    }

    fn worker(p: *Prefetch, buf: []u8) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        while (!p.stop) {
            const hi = @min(p.st.len, p.consumed + 1 + p.ahead);
            var i = p.consumed;
            const did = while (i < hi) : (i += 1) {
                const s = &p.st[i];
                if (s.phase == .reading and s.next < s.pieces.len) {
                    const r = s.pieces[s.next];
                    s.next += 1;
                    p.mutex.unlock(p.io);
                    const ok = if (preadAll(r.fd, buf[0..r.len], r.off)) |_| true else |_| false;
                    p.mutex.lockUncancelable(p.io);
                    if (ok) p.bytes += r.len else p.read_errors += 1;
                    s.left -= 1;
                    if (s.left == 0) {
                        s.phase = .ready;
                        p.cond.broadcast(p.io);
                    }
                    break true;
                }
                if (s.phase == .pending) {
                    s.phase = .expanding;
                    p.mutex.unlock(p.io);
                    const got = p.expand(i);
                    p.mutex.lockUncancelable(p.io);
                    if (got) |x| {
                        s.pieces = x.pieces;
                        s.fds = x.fds;
                        s.left = x.pieces.len;
                        s.phase = if (x.pieces.len == 0) .ready else .reading;
                    } else |err| {
                        p.expand_errors += 1;
                        if (p.first_err == null) p.first_err = err;
                        s.phase = .ready;
                    }
                    p.cond.broadcast(p.io);
                    break true;
                }
            } else false;
            if (!did) p.cond.waitUncancelable(p.io, &p.mutex);
        }
    }

    /// Step `i`'s reads: each file's header parsed (the checkpoint's shards, the plan's bank layer files) and the
    /// ranges of this rank's shares in it, merged and cut into pieces of at most piece_bytes.
    /// ステップ `i` の読み込み: 各ファイルの header を解析し (チェックポイントの shard、計画のバンクの層ファイル)、
    /// その中のこのランクの分の範囲を、まとめてから piece_bytes 以下の piece に切る。
    fn expand(p: *Prefetch, i: usize) !Expanded {
        const gpa = p.gpa;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var ranges: std.ArrayList(Range) = .empty;
        defer ranges.deinit(gpa);
        var fds: std.ArrayList(std.c.fd_t) = .empty;
        errdefer {
            for (fds.items) |fd| _ = std.c.close(fd);
            fds.deinit(gpa);
        }
        const s = p.steps[i];
        // the checkpoint's tensors, a file at a time (the wants are sorted by file)
        // チェックポイントのテンソルを、ファイルごとに (want はファイル順に並べてある)
        var j: usize = 0;
        while (j < s.wants.len) {
            const file = s.wants[j].file;
            var end = j + 1;
            while (end < s.wants.len and std.mem.eql(u8, s.wants[end].file, file)) end += 1;
            try fds.ensureUnusedCapacity(gpa, 1);
            const h = try openHead(a, try std.fs.path.joinZ(a, &.{ p.dir, file }));
            fds.appendAssumeCapacity(h.fd);
            for (s.wants[j..end]) |w| {
                const ent = h.hdr.get(w.name) orelse continue;
                const sp = if (w.whole) Span{ .lo = ent.begin, .hi = ent.end } else shareSpan(ent, split.rule(w.name) catch .rep, w.share, p.world);
                try ranges.append(gpa, .{ .fd = h.fd, .off = h.data + sp.lo, .len = sp.hi - sp.lo });
            }
            j = end;
        }
        // an MoE layer of the plan: this rank's stored parts of the experts each bank serves, and the file's header
        // (Reader.openFile parses it)
        // 計画の MoE 層: 各バンクが受け持つ expert のこのランクの保存された分と、ファイルの header (Reader.openFile が
        // 解析する)
        if (s.layer) |L| if (p.plan) |pl| if (pl.row(L)) |pick| {
            for (pl.banks, 0..) |b, bi| {
                if (!pl.uses(L, bi)) continue;
                var nb: [64]u8 = undefined;
                const fname = try std.fmt.bufPrint(&nb, "experts-layer-{d:0>3}.safetensors", .{L});
                try fds.ensureUnusedCapacity(gpa, 1);
                const h = try openHead(a, try std.fs.path.joinZ(a, &.{ b.dir, fname }));
                fds.appendAssumeCapacity(h.fd);
                try ranges.append(gpa, .{ .fd = h.fd, .off = 0, .len = h.data });
                const s0 = p.rank * p.k;
                for (pick, 0..) |pb, e| {
                    if (pb != bi) continue;
                    for ([_][]const u8{ "gate", "up", "down" }) |proj| for (s0..s0 + p.k) |sr| for ([_][]const u8{ "trellis", "suh", "svh" }) |kind| {
                        var buf: [192]u8 = undefined;
                        const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.rank{d}.{s}", .{ L, e, proj, sr, kind });
                        const ent = h.hdr.get(name) orelse continue;
                        try ranges.append(gpa, .{ .fd = h.fd, .off = h.data + ent.begin, .len = ent.end - ent.begin });
                    };
                }
            }
        };
        const pieces = try mergeSplit(gpa, ranges.items, merge_gap, piece_bytes);
        errdefer gpa.free(pieces);
        return .{ .pieces = pieces, .fds = try fds.toOwnedSlice(gpa) };
    }
};

/// A step's pages dropped (page-aligned outward: a boundary page shared with a neighbour is read again if needed),
/// its files closed and its lists freed.
/// ステップのページを落とし (ページ境界の外側へ広げる: 隣と共有する境界のページは、要れば読み直す)、ファイルを閉じ、
/// 一覧を返す。
fn release(gpa: std.mem.Allocator, s: *State) void {
    for (s.pieces) |r| {
        const lo = std.mem.alignBackward(usize, r.off, page);
        const hi = std.mem.alignForward(usize, r.off + r.len, page);
        _ = std.os.linux.fadvise(@intCast(r.fd), @intCast(lo), @intCast(hi - lo), std.os.linux.POSIX_FADV.DONTNEED);
    }
    for (s.fds) |fd| _ = std.c.close(fd);
    gpa.free(s.pieces);
    gpa.free(s.fds);
    s.pieces = &.{};
    s.fds = &.{};
}

fn buildSteps(gpa: std.mem.Allocator, reader: *const weights.Reader, plan: ?*const expert_plan.Plan, cfg: Config, lay: Layout, k: usize) ![]Step {
    const lists = try gpa.alloc(std.ArrayList(Want), lay.count());
    for (lists) |*l| l.* = .empty;
    defer {
        for (lists) |*l| l.deinit(gpa);
        gpa.free(lists);
    }
    const rank = reader.rank;
    const partner = rank ^ 1;
    var it = reader.index.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const file = e.value_ptr.*;
        const sr = storedRank(name);
        const mine = if (sr) |s| s / k == rank else true;
        if (layerOf(name)) |L| {
            const expert = std.mem.indexOf(u8, name, ".mlp.experts.") != null;
            const planned = if (plan) |pl| pl.row(L) != null else false;
            if (L < lay.n and mine and !(expert and planned)) try lists[lay.layer(L)].append(gpa, .{ .name = name, .file = file, .share = rank, .whole = sr != null });
            if (lay.mtp and L == cfg.layers and mine) try lists[lay.mtpStep()].append(gpa, .{ .name = name, .file = file, .share = rank, .whole = sr != null });
            const theirs = if (sr) |s| s / k == partner else true;
            if (lay.peer and !expert and theirs and isPeerName(name)) {
                if (L < lay.n) {
                    try lists[lay.peerStep(L)].append(gpa, .{ .name = name, .file = file, .share = partner, .whole = sr != null });
                } else if (lay.mtp and L == cfg.layers) {
                    try lists[lay.peerStep(lay.n)].append(gpa, .{ .name = name, .file = file, .share = partner, .whole = sr != null });
                }
            }
        } else if (mine and isTailName(name)) try lists[lay.tail()].append(gpa, .{ .name = name, .file = file, .share = rank, .whole = sr != null });
    }
    const steps = try gpa.alloc(Step, lists.len);
    var made: usize = 0;
    errdefer {
        for (steps[0..made]) |s| gpa.free(s.wants);
        gpa.free(steps);
    }
    for (lists, steps, 0..) |*l, *s, i| {
        std.mem.sort(Want, l.items, {}, Want.byFile);
        const planned = if (plan) |pl| pl.row(i) != null else false;
        s.* = .{ .wants = try l.toOwnedSlice(gpa), .layer = if (i < lay.n and planned) i else null };
        made += 1;
    }
    return steps;
}

fn freeSteps(gpa: std.mem.Allocator, steps: []Step) void {
    for (steps) |s| gpa.free(s.wants);
    gpa.free(steps);
}

/// The decoder layer of `model.layers.N...`, or null.
/// `model.layers.N...` の decoder 層の番号。それ以外は null。
fn layerOf(name: []const u8) ?usize {
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(usize, rest[0..dot], 10) catch null;
}

/// The stored rank of a pre-sharded part's name (`...P_proj.rank{s}.S`, weights.Reader.storedName), or null.
/// 事前分割した分の名前 (`...P_proj.rank{s}.S`、weights.Reader.storedName) の保存時のランク。それ以外は null。
fn storedRank(name: []const u8) ?usize {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return null;
    const stem = name[0..dot];
    const at = std.mem.lastIndexOf(u8, stem, ".rank") orelse return null;
    return std.fmt.parseInt(usize, stem[at + 5 ..], 10) catch null;
}

/// What model.loadPeerQuery reads of a layer.
/// model.loadPeerQuery が層から読むもの。
fn isPeerName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".self_attn.q_b_proj.") != null or std.mem.endsWith(u8, name, ".self_attn.kv_b_proj.weight");
}

fn isTailName(name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embed_tokens.weight") or std.mem.eql(u8, name, "lm_head.weight") or std.mem.eql(u8, name, "model.norm.weight");
}

const Span = struct { lo: usize, hi: usize };

/// The bytes (from the data region's start) holding rank `r`'s share of a full tensor as weights.cut reads it: the
/// rows of `row`, from the first row's columns to the last row's of `dim1` (the other ranks' columns between them
/// come along), all of `rep` or of a tensor that cannot be cut so.
/// 完全なテンソルのうち weights.cut が読むランク `r` の分を含むバイト (データ領域の先頭から): `row` ならその行、`dim1`
/// なら最初の行の列から最後の行の列まで (間の他ランクの列も含む)、`rep` やそう切れないテンソルなら全体。
fn shareSpan(e: st.Entry, rule: split.Rule, r: usize, world: usize) Span {
    const total = e.end - e.begin;
    switch (rule) {
        .rep => {},
        .row => if (e.rank >= 1 and e.shape[0] > 0 and e.shape[0] % world == 0) {
            const per = total / e.shape[0];
            const n = e.shape[0] / world;
            return .{ .lo = e.begin + r * n * per, .hi = e.begin + (r + 1) * n * per };
        },
        .dim1 => if (e.rank >= 2 and e.shape[0] > 0 and e.shape[1] > 0 and e.shape[1] % world == 0) {
            const inner = total / (e.shape[0] * e.shape[1]);
            const n = e.shape[1] / world;
            const lo = r * n;
            return .{ .lo = e.begin + lo * inner, .hi = e.begin + ((e.shape[0] - 1) * e.shape[1] + lo + n) * inner };
        },
    }
    return .{ .lo = e.begin, .hi = e.end };
}

const Head = struct { fd: std.c.fd_t, data: usize, hdr: st.Header };

/// A safetensors file opened for positional reads (through the page cache) and its header parsed into `a`.
/// safetensors のファイルを位置指定の読み込み用に開き (ページキャッシュ経由)、header を `a` に解析する。
fn openHead(a: std.mem.Allocator, path: [:0]const u8) !Head {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    errdefer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 8) return error.BadSafetensors;
    const size: usize = @intCast(end);
    var lb: [8]u8 = undefined;
    try preadAll(fd, &lb, 0);
    const n: usize = @intCast(std.mem.readInt(u64, &lb, .little));
    if (n > size - 8) return error.BadSafetensors;
    const json = try a.alloc(u8, n);
    try preadAll(fd, json, 8);
    return .{ .fd = fd, .data = 8 + n, .hdr = try st.parseHeader(a, json, size - 8 - n) };
}

/// `buf.len` bytes at `off`, through the page cache (the pages stay there for the loader's map).
/// `off` から `buf.len` バイト。ページキャッシュを通る (ページはローダーの map のために残る)。
fn preadAll(fd: std.c.fd_t, buf: []u8, off: usize) !void {
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.pread(fd, buf.ptr + got, buf.len - got, @intCast(off + got));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        got += @intCast(n);
    }
}

/// `ranges` sorted, those of a file closer than `gap` merged, and cut into pieces of at most `piece`.
/// `ranges` を並べ、同じファイルで `gap` より近いものをまとめ、`piece` 以下の piece に切る。
fn mergeSplit(gpa: std.mem.Allocator, ranges: []Range, gap: usize, piece: usize) ![]Range {
    std.mem.sort(Range, ranges, {}, Range.lessThan);
    var out: std.ArrayList(Range) = .empty;
    errdefer out.deinit(gpa);
    var cur: ?Range = null;
    for (ranges) |r| {
        if (r.len == 0) continue;
        if (cur) |*c| {
            if (c.fd == r.fd and r.off <= c.off + c.len + gap) {
                c.len = @max(c.off + c.len, r.off + r.len) - c.off;
                continue;
            }
            try cutInto(gpa, &out, c.*, piece);
        }
        cur = r;
    }
    if (cur) |c| try cutInto(gpa, &out, c, piece);
    return out.toOwnedSlice(gpa);
}

fn cutInto(gpa: std.mem.Allocator, out: *std.ArrayList(Range), r: Range, piece: usize) !void {
    var o = r.off;
    const end = r.off + r.len;
    while (o < end) {
        const n = @min(piece, end - o);
        try out.append(gpa, .{ .fd = r.fd, .off = o, .len = n });
        o += n;
    }
}

// ----------------------------------------------------------------------------------- packers / packer ---

/// One routed expert's trellis share for the packers: its stored parts (weights.Parts) and where the joined share
/// goes in the projection's device buffer (moe.loadFrom's offsets).
/// packer に渡す routed expert 1 つの trellis の分: 保存された分 (weights.Parts) と、連結した分を置く射影の
/// デバイスバッファ内の位置 (moe.loadFrom と同じ位置)。
pub const Piece = struct {
    pt: weights.Parts,
    off: usize,
};

const SlotState = enum { free, queued, packing, ready, copying };

/// A pinned chunk: a run of pieces [a, b) packed there, then copied on its own stream.
/// pin したチャンク: piece の並び [a, b) をそこに詰め、自分のストリームで写す。
const Slot = struct {
    bytes: []u8,
    stream: cuda.Stream,
    state: SlotState = .free,
    a: usize = 0,
    b: usize = 0,
};

pub const Packers = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    pinned: cuda.HostBuffer,
    slots: []Slot,
    chunk: usize,
    threads: []std.Thread,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    pieces: []const Piece = &.{},
    err: ?anyerror = null,
    stop: bool = false,
    // the loader's side: bytes copied, time waited for a packed chunk
    // ローダー側: 写したバイト、詰め終えたチャンクを待った時間
    bytes: u64 = 0,
    waited_ns: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, n: usize, pinned_mb: usize) !*Packers {
        const pk = try gpa.create(Packers);
        errdefer gpa.destroy(pk);
        const nslots = 2 * n;
        const chunk = std.mem.alignBackward(usize, (pinned_mb << 20) / nslots, page);
        if (chunk < 1 << 20) return error.LoadPinnedTooSmall;
        var pinned = try cuda.HostBuffer.alloc(d, chunk * nslots);
        errdefer pinned.free();
        const slots = try gpa.alloc(Slot, nslots);
        errdefer gpa.free(slots);
        var made: usize = 0;
        errdefer for (slots[0..made]) |*s| s.stream.deinit();
        for (slots, 0..) |*s, i| {
            s.* = .{ .bytes = pinned.bytes[i * chunk ..][0..chunk], .stream = try cuda.Stream.init(d, true) };
            made += 1;
        }
        const threads = try gpa.alloc(std.Thread, n);
        errdefer gpa.free(threads);
        pk.* = .{ .gpa = gpa, .io = io, .pinned = pinned, .slots = slots, .chunk = chunk, .threads = threads[0..0] };
        errdefer pk.stopJoin();
        for (threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, packer, .{pk});
            pk.threads = threads[0 .. i + 1];
        }
        return pk;
    }

    fn stopJoin(pk: *Packers) void {
        pk.mutex.lockUncancelable(pk.io);
        pk.stop = true;
        pk.cond.broadcast(pk.io);
        pk.mutex.unlock(pk.io);
        for (pk.threads) |t| t.join();
        pk.threads = pk.threads[0..0];
    }

    pub fn deinit(pk: *Packers) void {
        const gpa = pk.gpa;
        const all = pk.threads.ptr[0 .. pk.slots.len / 2];
        pk.stopJoin();
        for (pk.slots) |*s| {
            s.stream.synchronize() catch {};
            s.stream.deinit();
        }
        gpa.free(all);
        gpa.free(pk.slots);
        pk.pinned.free();
        gpa.destroy(pk);
    }

    fn report(pk: *Packers, rank: u32) void {
        std.log.info("glm53: rank {d}: load packers: {d:.1} GiB of expert trellis joined into pinned chunks, the loader waited {d:.1} s for them", .{ rank, @as(f64, @floatFromInt(pk.bytes)) / (1 << 30), @as(f64, @floatFromInt(pk.waited_ns)) / 1e9 });
    }

    /// `pieces` joined by the packers and copied into `dst` at their offsets (one copy a piece, as moe.loadFrom
    /// uploads them; the alignment gaps between them are never written). Returns when every copy has finished.
    /// `pieces` を packer が連結し、`dst` のそれぞれの位置へ写す (piece ごとに 1 回。moe.loadFrom と同じく、間の
    /// 整列の隙間には書かない)。すべてのコピーが終わってから返る。
    pub fn run(pk: *Packers, pieces: []const Piece, dst: cuda.DeviceBuffer) !void {
        if (pieces.len == 0) return;
        for (pieces) |pc| if (pc.pt.len > pk.chunk) {
            std.log.err("glm53: an expert share of {d} MiB is wider than a pinned chunk ({d} MiB): raise TF_GLM53_LOAD_PINNED_MB", .{ pc.pt.len >> 20, pk.chunk >> 20 });
            return error.LoadChunkTooSmall;
        };
        pk.mutex.lockUncancelable(pk.io);
        pk.pieces = pieces;
        pk.err = null;
        pk.mutex.unlock(pk.io);
        errdefer pk.drain();
        var next: usize = 0; // the first piece not given to a slot / まだスロットに渡していない最初の piece
        var copied: usize = 0;
        while (copied < pieces.len) {
            // every free slot takes the next run of pieces that fits it (its last copies finished first)
            // 空いたスロットごとに、入るだけの次の piece の並びを渡す (そのスロットの前回のコピーを先に終える)
            for (pk.slots) |*s| {
                if (next == pieces.len) break;
                pk.mutex.lockUncancelable(pk.io);
                const free = s.state == .free;
                pk.mutex.unlock(pk.io);
                if (!free) continue;
                try s.stream.synchronize();
                const a = next;
                const base = pieces[a].off;
                var b = a + 1;
                while (b < pieces.len and pieces[b].off + pieces[b].pt.len - base <= pk.chunk) b += 1;
                next = b;
                pk.mutex.lockUncancelable(pk.io);
                s.a = a;
                s.b = b;
                s.state = .queued;
                pk.cond.broadcast(pk.io);
                pk.mutex.unlock(pk.io);
            }
            // a packed slot's pieces copied on its stream
            // 詰め終えたスロットの piece を、そのストリームで写す
            const t = std.Io.Clock.awake.now(pk.io);
            pk.mutex.lockUncancelable(pk.io);
            var got: ?*Slot = null;
            while (got == null) {
                if (pk.err) |err| {
                    pk.mutex.unlock(pk.io);
                    return err;
                }
                for (pk.slots) |*s| {
                    if (s.state == .ready) {
                        got = s;
                        break;
                    }
                }
                if (got == null) pk.cond.waitUncancelable(pk.io, &pk.mutex);
            }
            const s = got.?;
            s.state = .copying;
            pk.mutex.unlock(pk.io);
            pk.waited_ns += @intCast(@max(0, t.durationTo(std.Io.Clock.awake.now(pk.io)).toNanoseconds()));
            const base = pieces[s.a].off;
            for (pieces[s.a..s.b]) |pc| {
                try dst.uploadAsync(pc.off, s.bytes[pc.off - base ..][0..pc.pt.len], s.stream.handle);
                pk.bytes += pc.pt.len;
            }
            copied += s.b - s.a;
            pk.mutex.lockUncancelable(pk.io);
            s.state = .free;
            pk.mutex.unlock(pk.io);
        }
        for (pk.slots) |*s| try s.stream.synchronize();
        pk.mutex.lockUncancelable(pk.io);
        pk.pieces = &.{};
        pk.mutex.unlock(pk.io);
    }

    /// After a failure: slots not taken yet given back, the packs under way finished, the copies drained.
    /// 失敗の後: まだ取られていないスロットを戻し、途中の詰めを終わらせ、コピーを空にする。
    fn drain(pk: *Packers) void {
        pk.mutex.lockUncancelable(pk.io);
        for (pk.slots) |*s| {
            if (s.state == .queued) s.state = .free;
        }
        while (pk.packing()) pk.cond.waitUncancelable(pk.io, &pk.mutex);
        for (pk.slots) |*s| s.state = .free;
        pk.pieces = &.{};
        pk.mutex.unlock(pk.io);
        for (pk.slots) |*s| s.stream.synchronize() catch {};
    }

    /// Whether a packer is still joining into a slot (under the mutex).
    /// packer がまだスロットに連結しているか (mutex の下で)。
    fn packing(pk: *const Packers) bool {
        for (pk.slots) |s| {
            if (s.state == .packing) return true;
        }
        return false;
    }

    fn packer(pk: *Packers) void {
        pk.mutex.lockUncancelable(pk.io);
        defer pk.mutex.unlock(pk.io);
        while (true) {
            var got: ?*Slot = null;
            for (pk.slots) |*s| {
                if (s.state == .queued) {
                    got = s;
                    break;
                }
            }
            if (got) |s| {
                s.state = .packing;
                const pieces = pk.pieces;
                pk.mutex.unlock(pk.io);
                const base = pieces[s.a].off;
                var failed: ?anyerror = null;
                for (pieces[s.a..s.b]) |*pc| {
                    weights.joinInto(s.bytes[pc.off - base ..][0..pc.pt.len], &pc.pt) catch |err| {
                        failed = err;
                        break;
                    };
                }
                pk.mutex.lockUncancelable(pk.io);
                if (failed) |err| {
                    if (pk.err == null) pk.err = err;
                }
                s.state = .ready;
                pk.cond.broadcast(pk.io);
                continue;
            }
            if (pk.stop) return;
            pk.cond.waitUncancelable(pk.io, &pk.mutex);
        }
    }
};

// ---------------------------------------------------------------------------- digest / ダイジェスト ---

/// TF_GLM53_LOAD_DIGEST=1: every loaded weight's bytes hashed (wyhash, from the device), a line a layer and one for
/// the rest (the MTP layer's extras, embedding, lm_head, final norm) on every rank. Only what the loader wrote is
/// hashed: the routed experts' trellis per expert (its offset and width, not the alignment gaps), the linears' words,
/// scales and shape (not their zeroed counters). Two starts loaded the same weights when their lines match.
/// TF_GLM53_LOAD_DIGEST=1: 読み込んだ全重みのバイトのハッシュ (wyhash、デバイスから読む)。全ランクで、層ごとに 1 行と、
/// 残り (MTP 層の付属、embedding、lm_head、最後の norm) に 1 行。ローダーが書いたものだけを読む: routed expert の
/// trellis は expert ごと (位置と幅。整列の隙間は読まない)、linear は語・スケール・形 (0 にした counters は読まない)。
/// 2 回の起動の行が一致すれば、同じ重みを読み込んだ。
pub fn digest(gpa: std.mem.Allocator, rank: u32, layers: []const model.Layer, mtp_layer: ?*const model.Layer, rest: []const cuda.DeviceBuffer) !void {
    const host = try gpa.alloc(u8, 64 << 20);
    defer gpa.free(host);
    var all = std.hash.Wyhash.init(0);
    for (layers) |*L| {
        const h = try layerDigest(host, L);
        std.log.info("glm53: load digest rank {d} layer {d}: {x:0>16}", .{ rank, L.index, h });
        all.update(std.mem.asBytes(&h));
    }
    if (mtp_layer) |L| {
        const h = try layerDigest(host, L);
        std.log.info("glm53: load digest rank {d} layer {d} (MTP): {x:0>16}", .{ rank, L.index, h });
        all.update(std.mem.asBytes(&h));
    }
    var x = std.hash.Wyhash.init(0);
    for (rest) |b| try bufInto(&x, host, b, 0, b.len);
    const hr = x.final();
    std.log.info("glm53: load digest rank {d} rest: {x:0>16}", .{ rank, hr });
    all.update(std.mem.asBytes(&hr));
    std.log.info("glm53: load digest rank {d}: {x:0>16} (all)", .{ rank, all.final() });
}

fn layerDigest(host: []u8, L: *const model.Layer) !u64 {
    var h = std.hash.Wyhash.init(0);
    for ([_]cuda.DeviceBuffer{ L.input_norm, L.post_attn_norm, L.q_a_norm, L.kv_a_norm, L.wk, L.wv }) |b| try bufInto(&h, host, b, 0, b.len);
    for ([_]*const linear.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down }) |l| try linInto(&h, host, l);
    if (L.indexer) |*ix| {
        try linInto(&h, host, &ix.wq_b);
        for ([_]cuda.DeviceBuffer{ ix.wk, ix.weights_proj, ix.k_norm_w, ix.k_norm_b }) |b| try bufInto(&h, host, b, 0, b.len);
    }
    if (L.router_w) |b| try bufInto(&h, host, b, 0, b.len);
    if (L.router_bias) |b| try bufInto(&h, host, b, 0, b.len);
    if (L.peer_q_b) |*l| try linInto(&h, host, l);
    if (L.peer_wk) |b| try bufInto(&h, host, b, 0, b.len);
    if (L.experts) |*x| try routedInto(&h, host, x);
    return h.final();
}

fn linInto(h: *std.hash.Wyhash, host: []u8, l: *const linear.Linear) !void {
    try bufInto(h, host, l.words, 0, l.words.len);
    try bufInto(h, host, l.suh, 0, l.suh.len);
    try bufInto(h, host, l.svh, 0, l.svh.len);
    const dims = [_]u32{ l.k, l.n, l.k2, l.cb, l.sk, l.wk };
    h.update(std.mem.sliceAsBytes(&dims));
}

fn routedInto(h: *std.hash.Wyhash, host: []u8, x: *const moe.Routed) !void {
    const E: usize = x.count;
    var ptrs: [1024]i64 = undefined;
    var k2: [1024]i32 = undefined;
    if (E > ptrs.len) return error.TooManyExperts;
    for (0..3) |p| {
        try x.ptr[p].download(0, std.mem.sliceAsBytes(ptrs[0..E]));
        try x.k2[p].download(0, std.mem.sliceAsBytes(k2[0..E]));
        for (0..E) |e| {
            // trellis bytes: (D / 16) x (I / 16) tiles of 16 x bits int16 = D x I x k2 / 16, k2 = 2 x bits
            // trellis のバイト: (D / 16) x (I / 16) タイル、各 16 x bits 個の int16 = D x I x k2 / 16 (k2 = 2 x bits)
            const off: usize = @intCast(@as(u64, @bitCast(ptrs[e])) - x.trellis[p].ptr);
            const len: usize = @as(usize, x.dims) * x.width * @as(usize, @intCast(k2[e])) / 16;
            h.update(std.mem.asBytes(&off));
            h.update(std.mem.asBytes(&k2[e]));
            try bufInto(h, host, x.trellis[p], off, len);
        }
    }
    for ([_]cuda.DeviceBuffer{ x.suh_g, x.suh_u, x.svh_g, x.svh_u, x.suh_d, x.svh_d }) |b| try bufInto(h, host, b, 0, b.len);
    const meta = [_]u32{ x.count, x.dims, x.width, x.cb, x.k2_gu[0], x.k2_gu[1], x.k2_d[0], x.k2_d[1] };
    h.update(std.mem.sliceAsBytes(&meta));
}

fn bufInto(h: *std.hash.Wyhash, host: []u8, b: cuda.DeviceBuffer, off: usize, len: usize) !void {
    h.update(std.mem.asBytes(&len));
    var o: usize = 0;
    while (o < len) {
        const n = @min(host.len, len - o);
        try b.download(off + o, host[0..n]);
        h.update(host[0..n]);
        o += n;
    }
}

// ------------------------------------------------------------------------------------------- tests ---

test "layer and stored rank from a tensor name" {
    try std.testing.expectEqual(@as(?usize, 12), layerOf("model.layers.12.self_attn.q_b_proj.trellis"));
    try std.testing.expectEqual(@as(?usize, null), layerOf("model.embed_tokens.weight"));
    try std.testing.expectEqual(@as(?usize, 5), storedRank("model.layers.3.mlp.experts.0.gate_proj.rank5.trellis"));
    try std.testing.expectEqual(@as(?usize, null), storedRank("model.layers.3.self_attn.q_a_proj.trellis"));
    try std.testing.expect(isPeerName("model.layers.3.self_attn.q_b_proj.svh"));
    try std.testing.expect(isPeerName("model.layers.3.self_attn.kv_b_proj.weight"));
    try std.testing.expect(!isPeerName("model.layers.3.self_attn.q_a_proj.svh"));
}

test "steps in Engine.create's order" {
    const l: Layout = .{ .n = 78, .mtp = true, .peer = true };
    try std.testing.expectEqual(@as(usize, 78), l.mtpStep());
    try std.testing.expectEqual(@as(usize, 79), l.peerStep(0));
    try std.testing.expectEqual(@as(usize, 79 + 78), l.peerStep(78));
    try std.testing.expectEqual(@as(usize, 79 + 79), l.tail());
    try std.testing.expectEqual(@as(usize, 159), l.count());
    const bare: Layout = .{ .n = 4, .mtp = false, .peer = false };
    try std.testing.expectEqual(@as(usize, 4), bare.tail());
    try std.testing.expectEqual(@as(usize, 5), bare.count());
}

test "a share's span covers every byte weights.cut reads of it" {
    // [4, 8, 3] u8: dim1 over 4 ranks, row over 4 ranks
    // [4, 8, 3] の u8: 4 ランクで dim1、4 ランクで row
    const e: st.Entry = .{ .dtype = .u8, .rank = 3, .shape = .{ 4, 8, 3, 1 }, .begin = 100, .end = 100 + 96 };
    for (0..4) |r| {
        const d = shareSpan(e, .dim1, r, 4);
        // rows i = 0..3, columns [2r, 2r + 2) of 8, 3 bytes each: the first byte at 100 + 6r, the last at
        // 100 + (3 * 8 + 2r + 2) * 3 - 1
        try std.testing.expectEqual(@as(usize, 100 + 6 * r), d.lo);
        try std.testing.expectEqual(@as(usize, 100 + (3 * 8 + 2 * r + 2) * 3), d.hi);
        const w = shareSpan(e, .row, r, 4);
        try std.testing.expectEqual(@as(usize, 100 + 24 * r), w.lo);
        try std.testing.expectEqual(@as(usize, 100 + 24 * (r + 1)), w.hi);
    }
    const rep = shareSpan(e, .rep, 1, 4);
    try std.testing.expectEqual(@as(usize, 100), rep.lo);
    try std.testing.expectEqual(@as(usize, 196), rep.hi);
    // an axis that does not divide: the whole tensor / 割り切れない軸: 全体
    const odd = shareSpan(e, .row, 0, 3);
    try std.testing.expectEqual(@as(usize, 100), odd.lo);
    try std.testing.expectEqual(@as(usize, 196), odd.hi);
}

test "ranges merged within the gap and cut into pieces" {
    const gpa = std.testing.allocator;
    var rs = [_]Range{
        .{ .fd = 4, .off = 5000, .len = 100 },
        .{ .fd = 3, .off = 0, .len = 10 },
        .{ .fd = 3, .off = 20, .len = 30 },
        .{ .fd = 3, .off = 1000, .len = 2500 },
        .{ .fd = 4, .off = 0, .len = 0 },
    };
    const out = try mergeSplit(gpa, &rs, 100, 1024);
    defer gpa.free(out);
    // fd 3: [0, 50) merged (gap 10), [1000, 3500) apart (gap 950) cut into 1024 + 1024 + 452; fd 4: [5000, 5100)
    // fd 3: [0, 50) はまとめる (隙間 10)。[1000, 3500) は離れていて (隙間 950)、1024 + 1024 + 452 に切る。fd 4: [5000, 5100)
    try std.testing.expectEqual(@as(usize, 5), out.len);
    try std.testing.expectEqual(@as(usize, 50), out[0].len);
    try std.testing.expectEqual(@as(usize, 1000), out[1].off);
    try std.testing.expectEqual(@as(usize, 1024), out[1].len);
    try std.testing.expectEqual(@as(usize, 2024), out[2].off);
    try std.testing.expectEqual(@as(usize, 452), out[3].len);
    try std.testing.expectEqual(@as(std.c.fd_t, 4), out[4].fd);
    try std.testing.expectEqual(@as(usize, 100), out[4].len);
}
