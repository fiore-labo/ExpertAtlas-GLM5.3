//! The KV pool's host metadata and planner (TF_GLM53_PREFIX=pool, glm53/POOL.md): one pool of 256-position blocks,
//! a conversation (slot) is a table of logical block -> pool block. Full blocks below a slot's kv_end are shared by
//! reference; a request writes only blocks private to its slot X, inside [write_lo, write_hi). Every rank makes the
//! same plan from the same metadata (a pure function) and checks it by digest.
//! KV pool のホスト側のメタデータとプランナ (TF_GLM53_PREFIX=pool、glm53/POOL.md): 256 位置のブロックの pool が 1 つ、
//! 会話 (スロット) は論理ブロック -> pool のブロックの表。スロットの kv_end より手前の完成したブロックは参照で共有する。
//! 要求はそのスロット X の専有ブロックの [write_lo, write_hi) にだけ書く。全ランクが同じメタデータから同じプランを
//! 作り (純粋な関数)、digest で確かめる。
const std = @import("std");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Positions a block holds.
/// 1 ブロックの位置数。
pub const block: u64 = 256;
/// Prompt rows a resumed request still computes (slots.zig's tail: more than forward.zig's small_rows).
/// 再開した要求でもまだ計算する prompt の行数 (slots.zig の tail と同じ。forward.zig の small_rows より多い)。
pub const tail: u64 = 17;
/// Rows a decode window writes at most (engine.zig's vrows): write_hi leaves this much past n + max_tokens.
/// decode の窓が書く最大の行数 (engine.zig の vrows)。write_hi は n + max_tokens の後ろにこれだけ残す。
pub const vrows: u64 = 8;

pub const Error = error{ PoolTooSmall, BadRequest, BadCommit, Invariant } || Allocator.Error;

pub fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

pub const Config = struct {
    n_blocks: u32, // usable blocks, ids 0 .. n_blocks - 1; Z is n_blocks / 使えるブロック数。Z は n_blocks 番
    capacity: u64, // positions a request may reach / 要求が届きうる位置数
    max_slots: u32,
    mtp: bool, // the MTP layer is cached: a resume stops one row before the branch (d = 1) / MTP の層をキャッシュする
    draft_saves: u32 = 0, // DFlash2 per-slot ring saves (TF_GLM53_POOL_DRAFT_SAVES) / DFlash2 のスロットごとの写しの数
    verify: Verify = .sample, // reused blocks hash-checked before the first token (TF_GLM53_POOL_VERIFY) / 生成前の照合

    /// The startup lower bound (POOL.md 4.4): X never needs more than ceil(capacity / 256) blocks, kept with a margin.
    /// 起動時の下限 (POOL.md 4.4): X が使うのは ceil(capacity / 256) ブロック以下。余裕を持たせた下限。
    pub fn check(c: Config) Error!void {
        if (c.max_slots == 0 or c.capacity == 0) return error.PoolTooSmall;
        if (@as(u64, c.n_blocks) < ceilDiv(c.capacity + vrows, block) + 1) return error.PoolTooSmall;
    }

    pub fn z(c: Config) u32 {
        return c.n_blocks;
    }
};

pub const Verify = enum { sample, all, off };

/// A DFlash2 save (POOL.md 4.6.2): the live ring and the post-prompt snapshot of one slot, with their ranges. The
/// bytes differ per rank; the record is common.
/// DFlash2 の写し (POOL.md 4.6.2): 1 つのスロットの live のリングと prompt 後の写し、とその範囲。バイトはランクごとに
/// 違い、記録は共通。
pub const Save = struct {
    owner: ?u64 = null,
    gen: u64 = 0, // bumped at every store into it / 保存するたびに進める
    used: u64 = 0, // the clock of its last store (LRU) / 最後に保存した論理時刻 (LRU)
    live: ?Ring = null,
    snap: ?Ring = null,

    fn cut(sv: *Save, v: u64) void {
        sv.live = if (sv.live) |g| g.shrink(v) else null;
        sv.snap = if (sv.snap) |g| g.shrink(v) else null;
        if (sv.live == null and sv.snap == null) sv.* = .{ .gen = sv.gen };
    }
};

pub const Slot = struct {
    id: u64,
    blocks: std.ArrayList(u32) = .empty, // logical -> pool block / 論理 -> pool のブロック
    seq: std.ArrayList(i64) = .empty, // committed tokens, [0, valid_end) / 確定したトークン列
    kv_end: u64 = 0, // every cached layer's rows are right in [0, kv_end) / 全キャッシュ層の行が正しい長さ
    prompt_end: u64 = 0,
    tick: u64 = 0,

    pub fn validEnd(s: *const Slot) u64 {
        return s.seq.items.len;
    }

    fn deinit(s: *Slot, gpa: Allocator) void {
        s.blocks.deinit(gpa);
        s.seq.deinit(gpa);
    }
};

pub const Meta = struct {
    gpa: Allocator,
    cfg: Config,
    ref: []u32, // per block: the slot tables it appears in / ブロックごとに、出てくるスロットの表の数
    gen: []u64, // per block: its content's generation / ブロックごとの内容の世代
    slots: std.ArrayList(Slot) = .empty, // ascending id / id の昇順
    saves: []Save = &.{}, // cfg.draft_saves of them / cfg.draft_saves 個
    // what the drafter's live ring holds: the save (index, generation) it was saved to or restored from
    // drafter の live のリングの中身: 保存先か復元元の写し (番号, 世代)
    ring: ?[2]u64 = null,
    next_id: u64 = 0,
    clock: u64 = 0,

    pub fn init(gpa: Allocator, cfg: Config) Error!Meta {
        try cfg.check();
        const ref = try gpa.alloc(u32, cfg.n_blocks);
        errdefer gpa.free(ref);
        const gen = try gpa.alloc(u64, cfg.n_blocks);
        errdefer gpa.free(gen);
        const saves = try gpa.alloc(Save, cfg.draft_saves);
        @memset(ref, 0);
        @memset(gen, 0);
        @memset(saves, .{});
        return .{ .gpa = gpa, .cfg = cfg, .ref = ref, .gen = gen, .saves = saves };
    }

    pub fn deinit(m: *Meta) void {
        for (m.slots.items) |*s| s.deinit(m.gpa);
        m.slots.deinit(m.gpa);
        m.gpa.free(m.ref);
        m.gpa.free(m.gen);
        m.gpa.free(m.saves);
    }

    /// The save a slot owns.
    /// スロットが持つ写し。
    pub fn saveOf(m: *const Meta, id: u64) ?usize {
        for (m.saves, 0..) |sv, i| if (sv.owner != null and sv.owner.? == id) return i;
        return null;
    }

    pub fn find(m: *const Meta, id: u64) ?usize {
        for (m.slots.items, 0..) |s, i| if (s.id == id) return i;
        return null;
    }

    pub fn freeCount(m: *const Meta) u64 {
        var n: u64 = 0;
        for (m.ref) |r| n += @intFromBool(r == 0);
        return n;
    }

    /// SHA-256 over the canonical text of the common metadata (POOL.md 4.7): blocks' refs and generations, every
    /// slot's table, lengths, tokens and tick, the next id, the clock. The per-rank row hashes are not in it.
    /// 共通のメタデータの決まった文字列の SHA-256 (POOL.md 4.7): ブロックの参照数と世代、各スロットの表・長さ・
    /// トークン列・tick、次の id、論理時刻。ランクごとの行ハッシュは含めない。
    pub fn digest(m: *const Meta) [32]u8 {
        var h = Sha256.init(.{});
        put(&h, "pool1");
        putInt(&h, m.cfg.n_blocks);
        putInt(&h, m.cfg.capacity);
        putInt(&h, m.cfg.max_slots);
        putInt(&h, @intFromBool(m.cfg.mtp));
        putInt(&h, m.next_id);
        putInt(&h, m.clock);
        for (m.ref, m.gen) |r, g| {
            putInt(&h, r);
            putInt(&h, g);
        }
        for (m.saves) |sv| {
            put(&h, "save");
            putInt(&h, if (sv.owner) |o| o + 1 else 0);
            putInt(&h, sv.gen);
            putInt(&h, sv.used);
            for ([_]?Ring{ sv.live, sv.snap }) |g| if (g) |x| {
                for ([_]u64{ 1, x.phys_end, x.keep_lo, x.vlo, x.vhi }) |v| putInt(&h, v);
            } else putInt(&h, 0);
        }
        if (m.ring) |x| {
            putInt(&h, x[0] + 1);
            putInt(&h, x[1]);
        } else putInt(&h, 0);
        for (m.slots.items) |s| {
            put(&h, "slot");
            putInt(&h, s.id);
            putInt(&h, s.validEnd());
            putInt(&h, s.kv_end);
            putInt(&h, s.prompt_end);
            putInt(&h, s.tick);
            putInt(&h, s.blocks.items.len);
            for (s.blocks.items) |b| putInt(&h, b);
            h.update(std.mem.sliceAsBytes(s.seq.items));
        }
        var d: [32]u8 = undefined;
        h.final(&d);
        return d;
    }

    /// The invariants of POOL.md 6.1 that hold between requests and right after a plan is applied (`x`: the slot of
    /// the request in flight, whose table may be longer than its valid end).
    /// リクエストの間と、プランを適用した直後に成り立つ POOL.md 6.1 の不変条件 (`x`: 処理中の要求のスロット。表は確定長より
    /// 長くてよい)。
    pub fn check(m: *const Meta, x: ?u64) Error!void {
        const gpa = m.gpa;
        const count = try gpa.alloc(u32, m.cfg.n_blocks);
        defer gpa.free(count);
        @memset(count, 0);
        var prev: ?u64 = null;
        for (m.slots.items) |s| {
            if (prev) |p| if (s.id <= p) return error.Invariant;
            prev = s.id;
            if (s.id >= m.next_id) return error.Invariant;
            const v = s.validEnd();
            if (s.kv_end > v or s.prompt_end > v) return error.Invariant;
            const k = s.blocks.items.len;
            if (x != null and x.? == s.id) {
                if (k < ceilDiv(v, block)) return error.Invariant;
            } else if (k != ceilDiv(v, block)) return error.Invariant;
            for (s.blocks.items, 0..) |b, j| {
                if (b >= m.cfg.n_blocks) return error.Invariant; // Z or beyond / Z か範囲外
                for (s.blocks.items[0..j]) |b2| if (b2 == b) return error.Invariant;
                count[b] += 1;
            }
        }
        for (m.ref, count) |r, c| if (r != c) return error.Invariant;
        // a save belongs to one live slot; its ranges lie inside the ring's physical range and the slot's valid end
        // 写しは生きている 1 つのスロットのもの。範囲はリングの物理の範囲とスロットの確定長の中
        for (m.saves, 0..) |sv, i| if (sv.owner) |o| {
            const si = m.find(o) orelse return error.Invariant;
            for (m.saves[0..i]) |sv2| if (sv2.owner != null and sv2.owner.? == o) return error.Invariant;
            const v = m.slots.items[si].validEnd();
            for ([_]?Ring{ sv.live, sv.snap }) |g| if (g) |rg| {
                if (rg.keep_lo > rg.vlo or rg.vlo >= rg.vhi or rg.vhi > rg.phys_end or rg.vhi > v) return error.Invariant;
            };
        };
        // a shared block lies wholly below kv_end in every slot that holds it
        // 共有ブロックは、それを持つどのスロットでも kv_end より手前に丸ごと入る
        for (m.slots.items) |s| for (s.blocks.items, 0..) |b, j| {
            if (m.ref[b] >= 2 and (j + 1) * block > s.kv_end) return error.Invariant;
        };
    }
};

fn put(h: *Sha256, s: []const u8) void {
    h.update(s);
    h.update(&.{0});
}

fn putInt(h: *Sha256, x: u64) void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, x, .little);
    h.update(&b);
}

pub const Action = enum { CONTINUE, SHARE, NEW };

/// A request's plan: everything the execution needs, made from the metadata without changing it.
/// 要求のプラン: 実行に要るものすべて。メタデータを変えずに作る。
pub const Plan = struct {
    action: Action,
    S: ?u64, // the slot whose match is used (null: no slot, or reuse not allowed) / 一致を使うスロット
    X: u64, // the slot the request writes (S for CONTINUE, a new id otherwise) / 要求が書くスロット
    new_x: bool,
    n: u64,
    h: u64, // S's exact match length / S との正確な一致の長さ
    c: u64, // c_S of POOL.md 4.1 before rounding / 切り下げる前の c_S
    resume_end: u64,
    write_lo: u64,
    write_hi: u64,
    table: []u32, // X's logical blocks 0 .. ceil(write_hi / 256) / X の論理ブロック
    evicted: []u64, // slots removed, in order / 消したスロット (順に)
    shrunk: [][2]u64, // (slot id, new valid end), in order / (スロット id, 新しい確定長)
    alloc: []u32, // blocks taken, in order / 割り当てたブロック (順に)
    released: u64, // X's own references dropped (shared blocks replaced, blocks past write_hi) / 外した X の参照の数
    verify: []u32, // X's logical blocks hash-checked before the first token / 生成の前に照合する X の論理ブロック
    // DFlash2: the save restored before the prompt (S's, for CONTINUE and SHARE), its record as planned (the plan
    // may free or cut it; the copy runs before any later store), the match with its tokens, and whether the live
    // ring already holds it
    // DFlash2: prompt の前に復元する写し (CONTINUE と SHARE では S のもの)、計画時の記録 (プランで空けたり切ったり
    // しうる。コピーは後の保存より先に走る)、写しのトークン列との一致、live のリングがすでにそれを持つか
    restore: ?usize = null,
    restore_gen: u64 = 0,
    restore_live: ?Ring = null,
    restore_snap: ?Ring = null,
    below: u64 = 0,
    skip_copy: bool = false,

    pub fn deinit(p: *Plan, gpa: Allocator) void {
        gpa.free(p.verify);
        gpa.free(p.table);
        gpa.free(p.evicted);
        gpa.free(p.shrunk);
        gpa.free(p.alloc);
    }

    /// SHA-256 over the metadata digest before the plan and every field the execution uses (POOL.md 4.7); the
    /// metadata after the plan is a function of these two.
    /// プラン前のメタデータの digest と、実行が使う全フィールドの SHA-256 (POOL.md 4.7)。プラン後のメタデータは
    /// この 2 つの関数。
    pub fn digest(p: *const Plan, d0: [32]u8) [32]u8 {
        var h = Sha256.init(.{});
        put(&h, "plan1");
        h.update(&d0);
        putInt(&h, @intFromEnum(p.action));
        putInt(&h, if (p.S) |s| s + 1 else 0);
        putInt(&h, p.X);
        putInt(&h, @intFromBool(p.new_x));
        for ([_]u64{ p.n, p.h, p.c, p.resume_end, p.write_lo, p.write_hi, p.released }) |v| putInt(&h, v);
        putInt(&h, p.table.len);
        for (p.table) |b| putInt(&h, b);
        putInt(&h, p.evicted.len);
        for (p.evicted) |v| putInt(&h, v);
        putInt(&h, p.shrunk.len);
        for (p.shrunk) |v| {
            putInt(&h, v[0]);
            putInt(&h, v[1]);
        }
        putInt(&h, p.alloc.len);
        for (p.alloc) |b| putInt(&h, b);
        putInt(&h, p.verify.len);
        for (p.verify) |b| putInt(&h, b);
        putInt(&h, if (p.restore) |i| i + 1 else 0);
        putInt(&h, p.restore_gen);
        for ([_]?Ring{ p.restore_live, p.restore_snap }) |g| if (g) |x| {
            for ([_]u64{ 1, x.phys_end, x.keep_lo, x.vlo, x.vhi }) |v| putInt(&h, v);
        } else putInt(&h, 0);
        putInt(&h, p.below);
        putInt(&h, @intFromBool(p.skip_copy));
        var d: [32]u8 = undefined;
        h.final(&d);
        return d;
    }
};

/// The token count want of slots_serve.zig: max_tokens normalized to [1, capacity - n].
/// slots_serve.zig の want: max_tokens を [1, capacity - n] に正規化したもの。
pub fn wantTokens(capacity: u64, n: u64, max_tokens: i64) u64 {
    const m: u64 = @intCast(@max(1, max_tokens));
    return @max(1, @min(m, capacity - n));
}

fn lessOld(a: *const Slot, b: *const Slot) bool {
    return a.tick < b.tick or (a.tick == b.tick and a.id < b.id);
}

/// The plan of a request for `prompt` (n tokens, 1 <= n < capacity) on `m`, which it does not change. `resume_ok`
/// false: the request reuses no cache (every c_S is 0).
/// `m` 上の `prompt` (n トークン、1 <= n < capacity) の要求のプラン。`m` は変えない。`resume_ok` が false なら
/// キャッシュを使わない (どの c_S も 0)。
pub fn makePlan(m: *const Meta, prompt: []const i64, max_tokens: i64, resume_ok: bool) Error!Plan {
    const gpa = m.gpa;
    const cfg = m.cfg;
    const n: u64 = prompt.len;
    if (n < 1 or n >= cfg.capacity) return error.BadRequest;
    const want = wantTokens(cfg.capacity, n, max_tokens);
    const write_hi = @min(n + want + vrows, cfg.capacity);
    const d: u64 = @intFromBool(cfg.mtp);
    const cap_n: u64 = if (n > tail) n - tail else 0;

    // 4.1: every slot's exact match h_S, c_S = min(h_S - d, max(0, n - TAIL), kv_end) and the position the request
    // would really resume from with it, e_S (4.2 / 4.3: c_S if the slot is continued, c_S rounded down to 256 when
    // the branch block is shared; c_S rounded down for a SHARE; 0 for a NEW); the largest e_S wins (comparing c_S
    // before rounding let a SHARE at 768 beat a CONTINUE at 871 with MTP: POOL.md 12.2e finding 3). Ties: the slot
    // the request would continue (a SHARE from its twin would keep two copies of the same rows), the larger c, the
    // longer h, the newer tick, the smaller id
    // 4.1: 各スロットの正確な一致 h_S、c_S と、それで実際に再開する位置 e_S (4.2 / 4.3: 続けるなら c_S、分かれ目の
    // ブロックが共有なら 256 単位に切り下げ。SHARE なら c_S を切り下げ。NEW なら 0)。e_S が最大のものを選ぶ (切り下げる
    // 前の c_S で比べると、MTP で 768 からの SHARE が 871 からの CONTINUE に勝った: POOL.md 12.2e の見つかったこと 3)。
    // 同点は、要求がそのまま続けられるスロット (同じ中身の別スロットから SHARE すると同じ行を 2 組持つ)、c の大きい方、
    // h の長い方、tick の新しい方、id の小さい方
    var best: ?usize = null;
    var best_h: u64 = 0;
    var best_c: u64 = 0;
    var best_e: u64 = 0;
    var best_cont = false;
    if (resume_ok) for (m.slots.items, 0..) |*s, i| {
        const seq = s.seq.items;
        const lim = @min(seq.len, prompt.len);
        const h: u64 = std.mem.findDiff(i64, seq[0..lim], prompt[0..lim]) orelse lim;
        const c: u64 = @min(if (h > d) h - d else 0, cap_n, s.kv_end);
        const cont = c >= (if (s.prompt_end > tail) s.prompt_end - tail else 0);
        const e: u64 = if (cont) blk: {
            const j = c / block;
            const shared = c % block != 0 and j < s.blocks.items.len and m.ref[s.blocks.items[j]] >= 2;
            break :blk if (shared) j * block else c;
        } else if (c >= block) c / block * block else 0;
        const better = if (best) |bi| blk: {
            const b = &m.slots.items[bi];
            if (e != best_e) break :blk e > best_e;
            if (cont != best_cont) break :blk cont;
            if (c != best_c) break :blk c > best_c;
            if (h != best_h) break :blk h > best_h;
            if (s.tick != b.tick) break :blk s.tick > b.tick;
            break :blk s.id < b.id;
        } else true;
        if (better) {
            best = i;
            best_h = h;
            best_c = c;
            best_e = e;
            best_cont = cont;
        }
    };

    // 4.2: the action, in order on c
    // 4.2: 動作。c で上から順に判定する
    var action: Action = .NEW;
    if (best != null) {
        if (best_cont) action = .CONTINUE else if (best_c >= block) action = .SHARE;
    }
    var r: u64 = switch (action) {
        .CONTINUE => best_c,
        .SHARE => best_c / block * block,
        .NEW => 0,
    };

    // shadow refs: the plan's changes are simulated here, never on m
    // 影の参照数: プランの変更はここで試し、m には触れない
    const ref = try gpa.dupe(u32, m.ref);
    defer gpa.free(ref);
    var xb: std.ArrayList(u32) = .empty; // X's table being built / 作っている X の表
    defer xb.deinit(gpa);
    const x_id: u64 = if (action == .CONTINUE) m.slots.items[best.?].id else m.next_id;
    if (action == .CONTINUE) {
        const S = &m.slots.items[best.?];
        try xb.appendSlice(gpa, S.blocks.items);
        // 4.3 step 2: the block holding the resume point is shared: recompute it from its start
        // 4.3 の 2: 再開位置を含むブロックが共有なら、そのブロックの先頭から計算し直す
        const j = r / block;
        if (r % block != 0 and j < xb.items.len and m.ref[xb.items[j]] >= 2) r = j * block;
    } else if (action == .SHARE) {
        const S = &m.slots.items[best.?];
        for (S.blocks.items[0 .. r / block]) |b| {
            try xb.append(gpa, b);
            ref[b] += 1;
        }
    }
    const j0 = r / block;
    const kx = ceilDiv(write_hi, block);

    // 4.3 step 3: X's table made [0, kx): its blocks past kx dropped, its shared blocks from j0 on replaced
    // 4.3 の 3: X の表を [0, kx) にそろえる。kx より後ろは外し、j0 以降の共有ブロックは置き換える
    var released: u64 = 0;
    const need_new = try gpa.alloc(bool, kx);
    defer gpa.free(need_new);
    @memset(need_new, false);
    while (xb.items.len > kx) {
        ref[xb.pop().?] -= 1;
        released += 1;
    }
    var need: u64 = 0;
    for (j0..kx) |j| {
        if (j < xb.items.len) {
            const b = xb.items[j];
            if (ref[b] >= 2) {
                ref[b] -= 1;
                released += 1;
                need_new[j] = true;
                need += 1;
            }
        } else {
            need_new[j] = true;
            need += 1;
        }
    }

    // the slot cap (3), then 4.4: other slots by (tick, id), tail block first, until enough blocks are free
    // スロットの上限 (3)、続いて 4.4: ほかのスロットを (tick, id) の順に、末尾のブロックから、空きが足りるまで
    const ns = m.slots.items.len;
    const order = try gpa.alloc(usize, ns);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, m, struct {
        fn f(mm: *const Meta, a: usize, b: usize) bool {
            return lessOld(&mm.slots.items[a], &mm.slots.items[b]);
        }
    }.f);
    const len = try gpa.alloc(usize, ns); // blocks each slot keeps / 各スロットに残すブロック数
    defer gpa.free(len);
    for (len, m.slots.items) |*l, s| l.* = s.blocks.items.len;
    var evicted: std.ArrayList(u64) = .empty;
    defer evicted.deinit(gpa);
    var shrunk: std.ArrayList([2]u64) = .empty;
    defer shrunk.deinit(gpa);
    const gone = try gpa.alloc(bool, ns);
    defer gpa.free(gone);
    @memset(gone, false);
    var alive: u64 = ns + @intFromBool(action != .CONTINUE);
    for (order) |i| {
        if (alive <= cfg.max_slots) break;
        const s = &m.slots.items[i];
        if (s.id == x_id) continue;
        for (s.blocks.items[0..len[i]]) |b| ref[b] -= 1;
        len[i] = 0;
        gone[i] = true;
        try evicted.append(gpa, s.id);
        alive -= 1;
    }
    var free: u64 = 0;
    for (ref) |c| free += @intFromBool(c == 0);
    // an empty slot (a request stopped before its first chunk) met while blocks are short goes too
    // 空きが足りない間に出会った空のスロット (最初のチャンクの前に止まった要求のもの) も消す
    for (order) |i| {
        if (free >= need) break;
        const s = &m.slots.items[i];
        if (s.id == x_id or gone[i]) continue;
        while (len[i] > 0 and free < need) {
            len[i] -= 1;
            const b = s.blocks.items[len[i]];
            ref[b] -= 1;
            free += @intFromBool(ref[b] == 0);
        }
        if (len[i] == 0) try evicted.append(gpa, s.id) else try shrunk.append(gpa, .{ s.id, @min(s.validEnd(), len[i] * block) });
    }
    if (free < need) return error.Invariant; // the startup bound makes this unreachable / 起動時の下限により起きない

    // 4.3 step 5: the lowest free blocks, in logical order
    // 4.3 の 5: 番号の小さい空きブロックを、論理の順に
    var alloc: std.ArrayList(u32) = .empty;
    defer alloc.deinit(gpa);
    try xb.resize(gpa, kx);
    var next: u32 = 0;
    for (0..kx) |j| if (need_new[j]) {
        while (ref[next] != 0) next += 1;
        ref[next] = 1;
        xb.items[j] = next;
        try alloc.append(gpa, next);
    };

    // 6.2: the reused full blocks checked before the first token: all, or the last one and 8 spread over the rest
    // 6.2: 生成の前に照合する再利用の完成ブロック: 全部か、最後の 1 つと残りに散らした 8 つ
    var ver: std.ArrayList(u32) = .empty;
    defer ver.deinit(gpa);
    const full: u32 = @intCast(r / block);
    switch (cfg.verify) {
        .off => {},
        .all => for (0..full) |j| try ver.append(gpa, @intCast(j)),
        .sample => if (full <= 9) {
            for (0..full) |j| try ver.append(gpa, @intCast(j));
        } else {
            for (0..8) |i| try ver.append(gpa, @intCast(i * (full - 1) / 8));
            try ver.append(gpa, full - 1);
        },
    }

    const table = try xb.toOwnedSlice(gpa);
    errdefer gpa.free(table);
    const ev = try evicted.toOwnedSlice(gpa);
    errdefer gpa.free(ev);
    const sh = try shrunk.toOwnedSlice(gpa);
    errdefer gpa.free(sh);
    const al = try alloc.toOwnedSlice(gpa);
    errdefer gpa.free(al);
    var p: Plan = .{
        .action = action, .S = if (best) |bi| m.slots.items[bi].id else null, .X = x_id, .new_x = action != .CONTINUE,
        .n = n, .h = best_h, .c = best_c, .resume_end = r, .write_lo = r, .write_hi = write_hi,
        .table = table, .evicted = ev, .shrunk = sh, .alloc = al, .released = released,
        .verify = try ver.toOwnedSlice(gpa),
    };
    if (action != .NEW and r > 0) if (m.saveOf(m.slots.items[best.?].id)) |i| {
        const sv = m.saves[i];
        p.restore = i;
        p.restore_gen = sv.gen;
        p.restore_live = sv.live;
        p.restore_snap = sv.snap;
        p.below = best_h;
        p.skip_copy = if (m.ring) |g| g[0] == i and g[1] == sv.gen else false;
    };
    return p;
}

/// Applies a plan made on `m` (the metadata changes before generation, POOL.md 4.5): evictions and shrinks, X's
/// table, its length cut to resume_end, the refs recounted from every table, the generations of the blocks taken.
/// `m` 上で作ったプランを適用する (生成の前のメタデータの変更、POOL.md 4.5): 消す・縮める、X の表、X の長さを
/// resume_end に、全部の表から参照数を数え直す、割り当てたブロックの世代。
pub fn apply(m: *Meta, p: *const Plan, prompt: []const i64) Error!void {
    const gpa = m.gpa;
    for (p.evicted) |id| {
        const i = m.find(id) orelse return error.Invariant;
        var s = m.slots.orderedRemove(i);
        s.deinit(gpa);
        if (m.saveOf(id)) |k| m.saves[k] = .{ .gen = m.saves[k].gen };
    }
    for (p.shrunk) |e| try shrinkSlot(m, e[0], e[1]);
    m.clock += 1;
    const r = p.resume_end;
    if (p.new_x) {
        if (p.X != m.next_id) return error.Invariant;
        m.next_id += 1;
        var s: Slot = .{ .id = p.X };
        errdefer s.deinit(gpa);
        try s.seq.appendSlice(gpa, prompt[0..r]);
        try m.slots.append(gpa, s); // the largest id: stays ascending / 最大の id なので昇順のまま
    }
    const x = &m.slots.items[m.find(p.X) orelse return error.Invariant];
    x.blocks.clearRetainingCapacity();
    try x.blocks.appendSlice(gpa, p.table);
    x.seq.shrinkRetainingCapacity(r);
    // until the commit X holds only what the request reuses; its tick moves at the commit
    // 確定までの X は要求が再利用する分だけを持つ。tick は確定のときに進める
    x.kv_end = if (p.new_x) r else @min(x.kv_end, r);
    x.prompt_end = if (p.new_x) 0 else @min(x.prompt_end, r);
    // X's old save keeps only what is below resume_end (replaced at the commit if the request saves)
    // X の古い写しは resume_end より手前だけ残す (要求が保存すれば確定のときに置き換わる)
    if (m.saveOf(p.X)) |k| m.saves[k].cut(r);
    recount(m);
    for (p.alloc) |b| m.gen[b] += 1;
}

/// Cuts a slot to valid end v (POOL.md 4.4): its table to ceil(v / 256) blocks, prompt_end and kv_end never above v.
/// The refs are recounted by the caller.
/// スロットを確定長 v に縮める (POOL.md 4.4): 表を ceil(v / 256) 個に、prompt_end と kv_end は v 以下に。参照数は
/// 呼び出し側が数え直す。
pub fn shrinkSlot(m: *Meta, id: u64, v: u64) Error!void {
    const s = &m.slots.items[m.find(id) orelse return error.Invariant];
    if (v > s.validEnd()) return error.Invariant;
    s.blocks.shrinkRetainingCapacity(ceilDiv(v, block));
    s.seq.shrinkRetainingCapacity(v);
    s.prompt_end = @min(s.prompt_end, v);
    s.kv_end = @min(s.kv_end, v);
    if (m.saveOf(id)) |k| m.saves[k].cut(v);
}

pub fn recount(m: *Meta) void {
    @memset(m.ref, 0);
    for (m.slots.items) |s| for (s.blocks.items) |b| {
        m.ref[b] += 1;
    };
}

/// The commit of X after the request (POOL.md 2.4, 4.5): `seq` the committed tokens (V of them, its first
/// resume_end equal to what X holds), `prompt_end`, `kv_end` (min(F, V) with MTP, else V). The blocks written and
/// wholly below kv_end get a new generation (their row hashes are made again); X's table is cut to ceil(V / 256).
/// 要求の後の X の確定 (POOL.md 2.4、4.5): `seq` は確定トークン列 (V 個、先頭 resume_end 個は X の持つものと同じ)、
/// `prompt_end`、`kv_end` (MTP があれば min(F, V)、なければ V)。書いたブロックのうち kv_end より手前に丸ごと入るものは
/// 世代を進める (行ハッシュを作り直す)。X の表は ceil(V / 256) 個に切る。
pub fn commit(m: *Meta, p: *const Plan, seq: []const i64, prompt_end: u64, kv_end: u64) Error!void {
    const gpa = m.gpa;
    const x = &m.slots.items[m.find(p.X) orelse return error.Invariant];
    const r = p.resume_end;
    const V: u64 = seq.len;
    if (x.validEnd() != r or V < r or V > p.write_hi or kv_end < r or kv_end > V or prompt_end > V) return error.BadCommit;
    if (!std.mem.eql(i64, x.seq.items, seq[0..r])) return error.BadCommit;
    try x.seq.appendSlice(gpa, seq[r..]);
    x.kv_end = kv_end;
    x.prompt_end = prompt_end;
    x.tick = m.clock;
    var j = r / block;
    while ((j + 1) * block <= kv_end) : (j += 1) m.gen[x.blocks.items[j]] += 1;
    const k = ceilDiv(V, block);
    while (x.blocks.items.len > k) {
        const b = x.blocks.pop().?;
        m.ref[b] -= 1;
    }
    // nothing committed (stopped before its first chunk, from 0): the slot goes (an empty slot would only linger)
    // 何も確定しなかった (0 から始めて最初のチャンクの前に止まった): スロットを消す (空のスロットは残っても役に立たない)
    if (V == 0) {
        var s = m.slots.orderedRemove(m.find(p.X).?);
        s.deinit(gpa);
        if (m.saveOf(p.X)) |i| m.saves[i] = .{ .gen = m.saves[i].gen };
    }
}

/// DFlash2 after the commit of a drafted request that reached the decode (POOL.md 4.6.2, the save table): the live
/// ring and the post-prompt snapshot go to X's save, else a free one, else the least recently stored; returns its
/// index (null: no save configured or nothing valid). `touched`: the request ran the drafter (any DFlash2 request).
/// A request that saves nothing (serial, stopped in the prompt) calls it with both null: X keeps its old save cut
/// at resume_end, and the live ring's identity is lost when the drafter ran.
/// 下書き付きで decode まで進んだ要求の確定の後の DFlash2 (POOL.md 4.6.2 の保存の表): live のリングと prompt 後の写しを
/// X の写し、なければ空き、なければ最も前に保存したものへ。その番号を返す (写しなし・有効な範囲なしは null)。
/// `touched`: 要求が drafter を動かした (DFlash2 の要求はすべて)。何も保存しない要求 (逐次モード・prompt の途中で
/// 止まった) は両方 null で呼ぶ: X は resume_end で切った古い写しを残し、drafter を動かしたならリングの中身は不明になる。
pub fn commitSave(m: *Meta, p: *const Plan, live: ?Ring, snap: ?Ring, touched: bool) Error!?usize {
    if (touched) m.ring = null;
    const xi = m.find(p.X) orelse return null; // nothing committed: the slot went / 何も確定しなかった: スロットはない
    const v = m.slots.items[xi].validEnd();
    const lv: ?Ring = if (live) |g| g.shrink(v) else null;
    const sn: ?Ring = if (snap) |g| g.shrink(v) else null;
    if (m.saves.len == 0 or (lv == null and sn == null)) return null;
    const k = m.saveOf(p.X) orelse blk: {
        for (m.saves, 0..) |sv, i| if (sv.owner == null) break :blk i;
        var o: usize = 0;
        for (m.saves, 0..) |sv, i| if (sv.used < m.saves[o].used) {
            o = i;
        };
        break :blk o;
    };
    const sv = &m.saves[k];
    sv.* = .{ .owner = p.X, .gen = sv.gen + 1, .used = m.clock, .live = lv, .snap = sn };
    if (lv != null) m.ring = .{ k, sv.gen };
    return k;
}

// ------------------------------------------------------------------------------------------- MTP F counter ---

/// Where the MTP rows are right (POOL.md 4.6.1): [0, F) unbroken. A canonical write [a, b) extends F only when it
/// joins it (a <= F); a non-canonical write at q lowers it to q.
/// MTP の行が正しい範囲 (POOL.md 4.6.1): [0, F) が途切れずに正しい。正しい書き込み [a, b) は F につながるとき (a <= F)
/// だけ F を伸ばす。正しくない書き込みが位置 q に入ったら F を q まで下げる。
pub const MtpF = struct {
    F: u64,
    n: u64,

    pub fn begin(resume_end: u64, n: u64) MtpF {
        return .{ .F = resume_end, .n = n };
    }

    fn canonical(t: *MtpF, a: u64, b: u64) void {
        if (a <= t.F) t.F = @max(t.F, b);
    }

    fn wrong(t: *MtpF, q: u64) void {
        t.F = @min(t.F, q);
    }

    /// A prompt chunk [a, e) finished: its rows are canonical but the last chunk's last row (placeholder token).
    /// prompt のチャンク [a, e) が終わった: 行は正しい。ただし最後のチャンクの最終行は仮の token。
    pub fn chunk(t: *MtpF, a: u64, e: u64) void {
        if (e < t.n) return t.canonical(a, e);
        t.canonical(a, e - 1);
        t.wrong(e - 1);
    }

    /// A window's propose: with room > 0 it rewrites the backlog [P - m, P) with the true next tokens; row 0 is
    /// never canonical when a step writes it (no zeroed embedding).
    /// 窓の propose: room > 0 ならバックログ [P - m, P) を本物の次の token で書き直す。step が書く行 0 は正しくない
    /// (埋め込みをゼロにしない)。
    pub fn propose(t: *MtpF, P: u64, m: u64, room: u64) void {
        if (room == 0 or m == 0) return;
        const a = P - m;
        if (a == 0) {
            t.wrong(0);
            t.canonical(1, P);
        } else t.canonical(a, P);
    }

    /// Chain rows [q0, q0 + k) hold draft-time values.
    /// 連鎖の行 [q0, q0 + k) は下書きの時の値。
    pub fn chain(t: *MtpF, q0: u64, k: u64) void {
        if (k > 0) t.wrong(q0);
    }

    pub fn kvEnd(t: *const MtpF, V: u64) u64 {
        return @min(t.F, V);
    }
};

// ------------------------------------------------------------------------------- DFlash2 ring ranges ---

/// A ring's saved record (POOL.md 4.6.2): its physical write end, the lowest position still physically there,
/// and the logical range [vlo, vhi) known right.
/// リングの写しの記録 (POOL.md 4.6.2): 物理的な書き込みの終わり、物理的に残っている下限、正しい論理の範囲 [vlo, vhi)。
pub const Ring = struct {
    phys_end: u64,
    keep_lo: u64,
    vlo: u64,
    vhi: u64,

    /// The live ring after a request: C its context end, V the committed length, s the valid start, B the block.
    /// 要求後の live のリング: C は context の終わり、V は確定長、s は正しい部分の先頭、B は block。
    pub fn live(C: u64, V: u64, s: u64, B: u64) Ring {
        const keep_lo = if (C + B > ring_rows) C + B - ring_rows else 0;
        return .{ .phys_end = C, .keep_lo = keep_lo, .vlo = @max(keep_lo, s), .vhi = @min(C, V) };
    }

    /// The post-prompt snapshot: taken when the prefill of n ends.
    /// prompt 後の写し: n の prefill を終えた時点で取る。
    pub fn snap(n: u64, s: u64) Ring {
        const keep_lo = if (n > ring_rows) n - ring_rows else 0;
        return .{ .phys_end = n, .keep_lo = keep_lo, .vlo = @max(keep_lo, s), .vhi = n };
    }

    /// The record after the slot is cut to `valid_end` (only vhi comes down); null when nothing is left.
    /// スロットを `valid_end` に縮めた後の記録 (下げるのは vhi だけ)。何も残らなければ null。
    pub fn shrink(g: Ring, valid_end: u64) ?Ring {
        var o = g;
        o.vhi = @min(o.vhi, valid_end);
        return if (o.vhi <= o.vlo) null else o;
    }
};

/// drafter.zig's ring: 4,096 rows (window 2,047 + block 8 + a tap batch, rounded up).
/// drafter.zig のリング: 4,096 行 (窓 2,047 + block 8 + tap 1 回分を切り上げ)。
pub const ring_rows: u64 = 4096;

pub const ResumeKind = enum { none, live, saved, partial };

pub const Resume = struct {
    kind: ResumeKind,
    s: u64, // the valid start the next save uses / 次の保存に使う正しい部分の先頭
    pos: u64, // the ring's context end and device pos / リングの context の終わりと device の pos
    copy: ?[2]u64 = null, // the snapshot's positions copied into the live ring / prompt 後の写しから live へコピーする位置
};

/// The range-based resumeAt (POOL.md 4.6.2): the drafter needs D = [max(0, n - (window + 1)), r); live and snap are
/// the saved valid ranges, cut at `below` (where the saved tokens stop matching the prompt).
/// 範囲付きの resumeAt (POOL.md 4.6.2): drafter に要るのは D = [max(0, n - (window + 1)), r)。live と snap は写しの
/// 正しい範囲で、`below` (写しのトークン列が prompt と一致しなくなる所) で切る。
pub fn resumeAt(r: u64, n: u64, live: ?[2]u64, snap: ?[2]u64, below: u64, window: u64) Resume {
    if (r == 0) return .{ .kind = .none, .s = 0, .pos = 0 };
    const L = if (n > window + 1) n - (window + 1) else 0;
    if (L >= r) return .{ .kind = .none, .s = r, .pos = r };
    // rows at or past `below` belong to other tokens
    // `below` 以降の行は別のトークン列のもの
    const I1: ?[2]u64 = if (live) |g| clip(g, below) else null;
    const I2: ?[2]u64 = if (snap) |g| clip(g, below) else null;
    // restored: the next save trusts [L, r) only (rows below L may be kept, never needed)
    // 戻せた: 次の保存が信じるのは [L, r) だけ (L より下の行は残っていても要らない)
    if (I1) |a| if (a[0] <= L and a[1] >= r) return .{ .kind = .live, .s = L, .pos = r };
    // what live lacks of D: one interval when live reaches an end of D, else all of D
    // live に足りない D の部分: live が D の端に届いていれば 1 つの区間、そうでなければ D 全部
    var need: [2]u64 = .{ L, r };
    if (I1) |a| {
        if (a[0] <= L and a[1] > L) {
            need = .{ a[1], r };
        } else if (a[0] > L and a[0] < r and a[1] >= r) need = .{ L, a[0] };
    }
    if (I2) |b| if (b[0] <= need[0] and b[1] >= need[1]) return .{ .kind = .saved, .s = L, .pos = r, .copy = need };
    return .{ .kind = .partial, .s = r, .pos = r };
}

fn clip(g: [2]u64, below: u64) ?[2]u64 {
    const hi = @min(g[1], below);
    return if (hi > g[0]) .{ g[0], hi } else null;
}
