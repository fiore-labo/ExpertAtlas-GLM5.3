//! The slot cache's store (TF_GLM53_PREFIX=slots): conversations kept beside the compute cache, so a prompt that
//! continues one resumes where it matches. Every rank runs the same plan (slotcache/tree.zig) on its own rows; with
//! DCP a rank holds the positions p % dcp == drank, at local row p / dcp and at a constant shift of it in the store.
//! スロットキャッシュのストア (TF_GLM53_PREFIX=slots): 計算キャッシュの横に会話を保持し、そのどれかに続く prompt は一致
//! した所から再開する。全ランクが同じプラン (slotcache/tree.zig) を自分の行に実行する: DCP では位置 p % dcp == drank を
//! ローカル行 p / dcp に持ち、ストアではそれを一定量ずらした所に置く。
//!
//! A copy is checked by hash: a saved row's expected value comes from the compute row; a restored row is re-hashed
//! where it lands and compared with the published value. A mismatch on any rank drops every conversation on every
//! rank (a common cold) and the request runs from 0.
//! コピーはハッシュで検査する: 保存する行の期待値は計算行から取り、復元した行は置いた先で再ハッシュして publish 済みの
//! 値と比べる。どのランクで食い違っても、全ランクが全会話を捨て (共通の cold)、要求は 0 から実行する。
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("forward.zig");
const tree = @import("slotcache/tree.zig");
const alloc_mod = @import("slotcache/alloc.zig");

const Buf = cuda.DeviceBuffer;

/// Prompt rows a resumed request still computes (more than forward.zig's small_rows, so they stay a prompt window).
/// 再開した要求でもまだ計算する prompt の行数 (forward.zig の small_rows より多く、prompt の window のままにする)。
pub const tail: usize = 17;
/// Rows per hash launch (bounds the lane-hash scratch, rows x lanes x 8 B).
/// ハッシュ 1 回の起動で扱う行数 (lane ハッシュの作業領域 rows x lanes x 8 B を抑える)。
pub const chunk_rows: usize = 32768;

/// The slot cache's settings from the environment.
/// 環境変数から読むスロットキャッシュの設定。
pub const Env = struct {
    on: bool = false, // TF_GLM53_PREFIX=slots
    min: usize = 256, // TF_GLM53_PREFIX_MIN: shorter resumes are not taken / これより短い再開はしない
    // TF_GLM53_PREFIX_STORE: positions kept (null: auto, from free memory)
    // TF_GLM53_PREFIX_STORE: 保持する位置数 (null は空きメモリから自動で決める)
    store: ?i64 = null,
    peak_gib: f64 = 8, // TF_GLM53_PREFIX_PEAK_GIB: a long prompt's transient memory / 長い prompt の一時メモリ
    margin_gib: f64 = 5, // TF_GLM53_PREFIX_MARGIN_GIB
    scratch: usize = 16384, // TF_GLM53_PREFIX_SCRATCH: local rows a swap block moves / swap 1 ブロックの行数

    pub fn fromEnv() !Env {
        var v: Env = .{};
        if (std.c.getenv("TF_GLM53_PREFIX")) |m| {
            const s = std.mem.span(m);
            // pool: pool/devmap.zig and pool_serve.zig instead of this store
            // pool: このストアの代わりに pool/devmap.zig と pool_serve.zig
            if (std.mem.eql(u8, s, "slots")) v.on = true else if (!std.mem.eql(u8, s, "off") and !std.mem.eql(u8, s, "pool")) return error.UnportedPrefixMode;
        }
        if (std.c.getenv("TF_GLM53_PREFIX_MIN")) |s| v.min = try std.fmt.parseInt(usize, std.mem.span(s), 10);
        if (std.c.getenv("TF_GLM53_PREFIX_STORE")) |s| {
            const t = std.mem.span(s);
            if (!std.mem.eql(u8, t, "auto")) v.store = try std.fmt.parseInt(i64, t, 10);
        }
        if (std.c.getenv("TF_GLM53_PREFIX_PEAK_GIB")) |s| v.peak_gib = try std.fmt.parseFloat(f64, std.mem.span(s));
        if (std.c.getenv("TF_GLM53_PREFIX_MARGIN_GIB")) |s| v.margin_gib = try std.fmt.parseFloat(f64, std.mem.span(s));
        if (std.c.getenv("TF_GLM53_PREFIX_SCRATCH")) |s| v.scratch = try std.fmt.parseInt(usize, std.mem.span(s), 10);
        // a swap moves scratch rows a block: 0 never advances
        // swap は 1 ブロックで scratch 行ずつ動かす: 0 では進まない
        if (v.scratch == 0) return error.PrefixScratchZero;
        return v;
    }
};

/// (S, S from free memory): the store's positions, a multiple of dcp, from the smallest free device memory over the
/// ranks after the reserve for a long prompt's peak and the margin, the scratch and the hashing buffers (a local row
/// costs its bytes and its expected values, published and in flight), at most cap. An explicit TF_GLM53_PREFIX_STORE
/// replaces it, cap or not.
/// (S, 空きメモリから決めた S): ストアの位置数 (dcp の倍数)。全ランクで最小の空きデバイスメモリから、長い prompt の
/// ピーク分と margin の予約、scratch、ハッシュ用バッファを引いて決め (ローカル 1 行はそのバイト数と、publish 済み・
/// 仮置きの期待値を使う)、最大 cap。TF_GLM53_PREFIX_STORE を明示すれば cap に関係なくそれを使う。
pub fn storePositions(env: Env, free_min: u64, row_bytes: u64, lanes: u64, dcp: u64, cap: u64) [2]i64 {
    const fixed: i128 = @intCast(env.scratch * row_bytes + chunk_rows * (lanes + 4) * 8);
    const reserve: i128 = @intFromFloat((env.peak_gib + env.margin_gib) * (1 << 30));
    const room: i128 = @as(i128, free_min) - reserve - fixed;
    const per: i128 = @max(0, @divFloor(room, @as(i128, row_bytes + 16)));
    const d: i128 = dcp;
    const auto: i64 = @intCast(@divFloor(@min(@as(i128, cap), per * d), d) * d);
    if (env.store) |s| return .{ @intCast(@divFloor(@max(0, s), @as(i64, @intCast(dcp))) * @as(i64, @intCast(dcp))), auto };
    return .{ auto, auto };
}

/// One side's lanes (the compute cache's, the store's or a scratch's): a device table of the lanes' base pointers.
/// 片側 (計算キャッシュ・ストア・scratch のいずれか) の lane: 各 lane の先頭ポインタを並べたデバイス上の表。
const Side = struct {
    table: Buf,
    rows: usize,
    // owned (store, scratch); empty for the compute cache's
    // 所有する (ストア・scratch)。計算キャッシュでは空
    bufs: []Buf = &.{},

    fn deinit(s: *Side, gpa: std.mem.Allocator) void {
        s.table.free();
        for (s.bufs) |*b| b.free();
        if (s.bufs.len != 0) gpa.free(s.bufs);
    }
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    glue: *const forward.Glue,
    stream: cuda.Stream,
    dcp: u32,
    r: u32, // drank, the rank within DCP / drank、DCP 内の順位
    cap: usize,
    S: i64,
    rank: u32,
    min_resume: usize,
    L: u32,
    max_words: u32,
    row_bytes: usize,
    words: Buf, // int32 [L]
    act: Side,
    sto: ?Side = null,
    scr: ?Side = null,
    // int64 [max(S / dcp, 1)]: published expected values, a store row each (0: unused)
    // int64 [max(S / dcp, 1)]: publish 済みの期待値。ストア行ごとに 1 つ (0 は未使用)
    E: Buf,
    etx: Buf, // in flight until publish (the plan's saves) / publish までの仮置き (プランの保存分)
    lanes: Buf, // u64 [L, chunk_rows]
    // int64 [4, id_rows]: pid / gen of the rows arriving, pid / gen of the rows leaving
    // int64 [4, id_rows]: 入ってくる行の pid / gen、出ていく行の pid / gen
    ids: Buf,
    id_rows: usize,
    bad: Buf, // u64 [1]
    // (store row, rows) published with the plan
    // プランと一緒に publish する (ストア行, 行数)
    tx: std.ArrayList([2]i64) = .empty,
    meta: tree.Meta,
    gen: i64 = 0,
    host_id: []i64, // [2, id_rows] staging / [2, id_rows] の転送用

    /// `kc` / `ic`: the compute cache's per-layer buffers (rows of `lat_row` / `idx_row` bytes, capacity / dcp + 1
    /// rows); `cap`: positions a request may reach; `S`: the store's positions (0: one conversation, no device store).
    /// `kc` / `ic`: 計算キャッシュの層ごとのバッファ (1 行 `lat_row` / `idx_row` バイト、capacity / dcp + 1 行)。`cap`:
    /// 要求が届きうる位置数。`S`: ストアの位置数 (0 なら会話 1 つで、デバイス上のストアなし)。
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, glue: *const forward.Glue, stream: cuda.Stream, state: *const forward.State, lat_row: usize, idx_row: usize, dcp: u32, drank: u32, cap: usize, S: i64, rank: u32, env: Env) !*Store {
        // the lanes: every layer's latent rows, then every full-indexer layer's index-key rows, ascending
        // lane の並び: 全層の latent 行、続いて full-indexer 層の index-key 行。それぞれ昇順
        var ptrs: std.ArrayList(u64) = .empty;
        defer ptrs.deinit(gpa);
        var widths: std.ArrayList(u32) = .empty;
        defer widths.deinit(gpa);
        for (state.kc) |b| {
            try ptrs.append(gpa, b.ptr);
            try widths.append(gpa, @intCast(lat_row));
        }
        for (state.ic) |b| if (b) |x| {
            try ptrs.append(gpa, x.ptr);
            try widths.append(gpa, @intCast(idx_row));
        };
        const L: u32 = @intCast(ptrs.items.len);
        var row_bytes: usize = 0;
        var max_words: u32 = 0;
        const wds = try gpa.alloc(i32, L);
        defer gpa.free(wds);
        for (widths.items, wds) |w, *o| {
            if (w % 16 != 0) return error.LaneWidth;
            row_bytes += w;
            o.* = @intCast(w / 8);
            max_words = @max(max_words, w / 8);
        }
        const act_rows = state.kc[0].len / lat_row;
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        const n_local: usize = @intCast(@divFloor(S, @as(i64, dcp)));
        const id_rows = (cap + dcp - 1) / dcp + 1;
        // every allocation undone on a failure (an out-of-memory store must leave the memory for the S = 0 retry)
        // 失敗時は確保をすべて戻す (メモリ不足のストアは、S = 0 でのやり直しのためにメモリを残す)
        var words = try Buf.fromHost(d, std.mem.sliceAsBytes(wds));
        errdefer words.free();
        var table = try Buf.fromHost(d, std.mem.sliceAsBytes(ptrs.items));
        errdefer table.free();
        var E = try Buf.alloc(d, @max(n_local, 1) * 8);
        errdefer E.free();
        var etx = try Buf.alloc(d, @max(n_local, 1) * 8);
        errdefer etx.free();
        var lanes = try Buf.alloc(d, chunk_rows * L * 8);
        errdefer lanes.free();
        var ids = try Buf.alloc(d, 4 * id_rows * 8);
        errdefer ids.free();
        var bad = try Buf.alloc(d, 8);
        errdefer bad.free();
        var meta = try tree.Meta.init(gpa, S, dcp);
        errdefer meta.deinit();
        const host_id = try gpa.alloc(i64, 2 * id_rows);
        errdefer gpa.free(host_id);
        try E.fill8(0, null);
        var sto: ?Side = null;
        var scr: ?Side = null;
        errdefer if (sto) |*x| x.deinit(gpa);
        errdefer if (scr) |*x| x.deinit(gpa);
        if (S > 0) {
            sto = try side(gpa, d, widths.items, n_local);
            scr = try side(gpa, d, widths.items, env.scratch);
        }
        s.* = .{
            .gpa = gpa, .d = d, .glue = glue, .stream = stream, .dcp = dcp, .r = drank, .cap = cap, .S = S, .rank = rank,
            .min_resume = env.min, .L = L, .max_words = max_words, .row_bytes = row_bytes,
            .words = words, .act = .{ .table = table, .rows = act_rows }, .sto = sto, .scr = scr,
            .E = E, .etx = etx, .lanes = lanes, .ids = ids, .id_rows = id_rows, .bad = bad, .meta = meta, .host_id = host_id,
        };
        return s;
    }

    fn side(gpa: std.mem.Allocator, d: *const cuda.Driver, widths: []const u32, rows: usize) !Side {
        const bufs = try gpa.alloc(Buf, widths.len);
        errdefer gpa.free(bufs);
        const ptrs = try gpa.alloc(u64, widths.len);
        defer gpa.free(ptrs);
        var made: usize = 0; // the lanes allocated so far (freed on a failure) / 確保済みの lane 数 (失敗時に解放)
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, widths, ptrs) |*b, w, *p| {
            b.* = try Buf.alloc(d, rows * w);
            made += 1;
            p.* = b.ptr;
        }
        return .{ .table = try Buf.fromHost(d, std.mem.sliceAsBytes(ptrs)), .rows = rows, .bufs = bufs };
    }

    pub fn deinit(s: *Store) void {
        const gpa = s.gpa;
        s.act.table.free();
        if (s.sto) |*x| x.deinit(gpa);
        if (s.scr) |*x| x.deinit(gpa);
        s.words.free();
        s.E.free();
        s.etx.free();
        s.lanes.free();
        s.ids.free();
        s.bad.free();
        s.tx.deinit(gpa);
        s.meta.deinit();
        gpa.free(s.host_id);
        gpa.destroy(s);
    }

    /// Device bytes the store holds (the store, its scratch, the expected values).
    /// ストアが持つデバイスのバイト数 (ストア、scratch、期待値)。
    pub fn nbytes(s: *const Store) usize {
        const rows = (if (s.sto) |x| x.rows else 0) + (if (s.scr) |x| x.rows else 0);
        return rows * s.row_bytes + s.E.len * 2;
    }

    /// The request's plan on a copy of this rank's metadata (never changes it).
    /// このランクのメタデータのコピー上で作る要求のプラン (元のメタデータは変えない)。
    pub fn plan(s: *const Store, X: []const i64, max_out: usize, resume_ok: bool) !tree.Plan {
        return tree.makePlan(s.gpa, &s.meta, X, @intCast(max_out), @intCast(s.cap), @intCast(tail), @intCast(s.min_resume), resume_ok);
    }

    /// This rank's positions of [a, b): first, count, its compute row, and its store row when [a, b) sits at logical
    /// store start `st`.
    /// [a, b) のうちこのランクの位置: 最初の位置、個数、計算行、[a, b) が論理ストアの `st` から置かれるときのストア行。
    fn span(s: *const Store, a: i64, b: i64, st: ?i64) [4]i64 {
        const dcp: i64 = s.dcp;
        const p0 = a + @mod(@as(i64, s.r) - a, dcp);
        const n = @max(0, -@divFloor(-(b - p0), dcp));
        return .{ p0, n, @divFloor(p0, dcp), if (st) |x| @divFloor(x + p0 - a, dcp) else -1 };
    }

    /// The identities (pid, gen) of positions p0, p0 + dcp, ... < b of the path ending at nid, into ids block `blk`
    /// (0: arriving, 1: leaving).
    /// nid で終わるパスの位置 p0, p0 + dcp, ... < b の identity (pid, gen) を ids のブロック `blk` に入れる
    /// (0: 入ってくる行、1: 出ていく行)。
    fn ident(s: *Store, paths: *tree.PathIdents, nid: i64, p0: i64, n: usize, blk: usize) !void {
        const id = try paths.get(nid);
        const pid = s.host_id[0..n];
        const gen = s.host_id[s.id_rows..][0..n];
        for (0..n) |k| {
            const p: usize = @intCast(p0 + @as(i64, @intCast(k)) * s.dcp);
            pid[k] = @bitCast(id.pid[p]);
            gen[k] = id.gen[p];
        }
        // pageable: the driver stages the bytes before returning, so the host rows may be reused at once
        // pageable メモリ: ドライバが戻る前にバイトを中継バッファへ写すので、ホストの行はすぐ再利用できる
        try s.ids.uploadAsync((2 * blk) * s.id_rows * 8, std.mem.sliceAsBytes(pid), s.stream.handle);
        try s.ids.uploadAsync((2 * blk + 1) * s.id_rows * 8, std.mem.sliceAsBytes(gen), s.stream.handle);
    }

    fn idPtr(s: *const Store, blk: usize, which: usize, k0: usize) u64 {
        return s.ids.ptr + ((2 * blk + which) * s.id_rows + k0) * 8;
    }

    /// H (slotcache/hashing.zig: a row's bytes bound to its position, pid and generation) of rows [r0, r0 + n) of
    /// `src` holding positions p0, p0 + dcp, ... with identities block `blk` from row k0: into out (or compared with
    /// exp, mismatches into bad); with dst the rows also go to its rows d0 ... in the same pass.
    /// 位置 p0, p0 + dcp, ... を持つ `src` の行 [r0, r0 + n) の H (slotcache/hashing.zig: 行のバイトを位置・pid・世代と
    /// 結びつける) を、ids のブロック `blk` の k0 行目からの identity で求め、out に書く (または exp と比べ、食い違いを
    /// bad に数える)。dst があれば、同じ処理で行を dst の行 d0 ... へも写す。
    fn hashes(s: *Store, src: u64, r0: i64, n: usize, p0: i64, blk: usize, k0: usize, dst: u64, d0: i64, out: u64, exp: u64) !void {
        var i: usize = 0;
        while (i < n) : (i += chunk_rows) {
            const k = @min(chunk_rows, n - i);
            const ii: i64 = @intCast(i);
            try s.glue.slotLanes(s.stream, src, dst, s.words.ptr, s.L, r0 + ii, d0 + ii, @intCast(k), s.lanes.ptr);
            try s.glue.slotFold(s.stream, s.lanes.ptr, s.L, @intCast(k), p0 + ii * s.dcp, s.dcp, s.idPtr(blk, 0, k0 + i), s.idPtr(blk, 1, k0 + i), if (out != 0) out + i * 8 else 0, if (exp != 0) exp + i * 8 else 0, s.bad.ptr);
        }
    }

    fn copy(s: *Store, src: u64, r0: i64, dst: u64, d0: i64, n: usize) !void {
        try s.glue.slotCopy(s.stream, src, dst, s.words.ptr, s.L, s.max_words, r0, d0, @intCast(n));
    }

    /// Every op of the plan on this rank's rows, queued on the stream (no sync): saves hash the compute rows they copy
    /// (expected values in flight until publish), restores and swaps hash the rows where they landed against the
    /// published values; mismatches count into `bad` (read by `ok`).
    /// プランの全 op をこのランクの行に対しストリームへ積む (同期しない): save はコピーする計算行をハッシュし (期待値は
    /// publish まで仮置き)、restore と swap は置いた先の行をハッシュして publish 済みの値と比べる。食い違いは `bad` に
    /// 数える (`ok` が読む)。
    pub fn execute(s: *Store, p: *const tree.Plan) !void {
        s.tx.clearRetainingCapacity();
        try s.bad.fill8(0, s.stream.handle);
        if (p.ops.len == 0) return;
        const sto = s.sto orelse return error.NoStore;
        const scr = s.scr.?;
        var paths = try tree.PathIdents.initPlan(s.gpa, p);
        defer paths.deinit();
        const act = s.act.table.ptr;
        for (p.ops) |op| {
            const st: ?i64 = switch (op.kind) {
                .save => op.dst,
                .restore, .swap => op.src,
            };
            const sp = s.span(op.a, op.b, st);
            const p0 = sp[0];
            const n: usize = @intCast(sp[1]);
            const arow = sp[2];
            const srow = sp[3];
            if (n == 0) continue;
            try s.ident(&paths, op.nid_in.?, p0, n, 0);
            switch (op.kind) {
                .save => {
                    try s.hashes(act, arow, n, p0, 0, 0, sto.table.ptr, srow, s.etx.ptr + @as(u64, @intCast(srow)) * 8, 0);
                    try s.tx.append(s.gpa, .{ srow, @intCast(n) });
                },
                .restore => {
                    const erow = s.span(op.a, op.b, op.exp.?)[3];
                    try s.copy(sto.table.ptr, srow, act, arow, n);
                    try s.hashes(act, arow, n, p0, 0, 0, 0, 0, 0, s.E.ptr + @as(u64, @intCast(erow)) * 8);
                },
                // T's slots <-> the compute rows, block by block through the scratch
                // T のスロットと計算行を、scratch を通してブロックごとに入れ替える
                .swap => {
                    try s.ident(&paths, op.nid_out.?, p0, n, 1);
                    var k0: usize = 0;
                    while (k0 < n) : (k0 += scr.rows) {
                        const k = @min(scr.rows, n - k0);
                        const kk: i64 = @intCast(k0);
                        const pk = p0 + kk * s.dcp;
                        try s.hashes(act, arow + kk, k, pk, 1, k0, scr.table.ptr, 0, s.etx.ptr + @as(u64, @intCast(srow + kk)) * 8, 0);
                        try s.tx.append(s.gpa, .{ srow + kk, @intCast(k) });
                        try s.copy(sto.table.ptr, srow + kk, act, arow + kk, k);
                        try s.hashes(act, arow + kk, k, pk, 0, k0, 0, 0, 0, s.E.ptr + @as(u64, @intCast(srow + kk)) * 8);
                        try s.copy(scr.table.ptr, 0, sto.table.ptr, srow + kk, k);
                    }
                },
            }
        }
    }

    /// This rank's verdict on the plan's checks: every arriving row matched (waits for the copies).
    /// プランの検査に対するこのランクの判定: 入ってきた行がすべて一致したか (コピーの完了を待つ)。
    pub fn ok(s: *Store) !bool {
        var v: [1]u64 = undefined;
        try s.bad.downloadAsync(0, std.mem.sliceAsBytes(&v), s.stream.handle);
        try s.stream.synchronize();
        return v[0] == 0;
    }

    fn clear(s: *Store, e: alloc_mod.Extent) !void {
        const sp = s.span(e.a, e.a + e.n, e.s);
        if (sp[1] > 0) try s.glue.slotZero(s.stream, s.E.ptr + @as(u64, @intCast(sp[3])) * 8, @intCast(sp[1]));
    }

    /// Every rank succeeded: the released extents' expected values cleared, then the saved rows' published (a save may
    /// reuse an evicted extent's slots), the plan's tree and store taken (`p.post` moves here).
    /// 全ランクが成功した: 解放した extent の期待値を消し、次に保存した行の期待値を publish する (save は追い出した
    /// extent のスロットを再利用しうる)。プランの木とストアを引き継ぐ (`p.post` をここへ移す)。
    pub fn publish(s: *Store, p: *tree.Plan) !void {
        for (p.free_after) |e| try s.clear(e);
        for (s.tx.items) |t| {
            const off: u64 = @intCast(t[0] * 8);
            try s.E.copyFrom(off, s.etx.ptr + off, @intCast(t[1] * 8), s.stream.handle);
        }
        s.tx.clearRetainingCapacity();
        // the old metadata goes with the plan
        // 古いメタデータはプランの側へ移る
        std.mem.swap(tree.Meta, &s.meta, &p.post);
    }

    /// A common cold: every conversation dropped (the rows stay, nothing claims them); the generation counter goes on.
    /// 共通の cold: 全会話を捨てる (行は残るが、どこからも参照しない)。世代の番号はそのまま進む。
    pub fn cold(s: *Store) !void {
        s.meta.deinit();
        s.meta = try tree.Meta.init(s.gpa, s.S, s.dcp);
        try s.E.fill8(0, s.stream.handle);
        s.tx.clearRetainingCapacity();
    }

    /// After the request, positions [c, seq.len) of the token sequence `seq` hold its new rows (a new generation).
    /// 要求の後、トークン列 `seq` の位置 [c, seq.len) が新しい行 (新しい世代) を持つ。
    pub fn commit(s: *Store, c: usize, seq: []const i64) !void {
        s.gen += 1;
        const released = try tree.commitSegment(&s.meta, s.gpa, @intCast(c), seq[c..], s.gen);
        defer s.gpa.free(released);
        for (released) |e| try s.clear(e);
    }
};
