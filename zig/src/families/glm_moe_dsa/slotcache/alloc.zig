//! The slot cache's store addresses: positions [a, b) of a conversation go to logical store positions [s, s + b - a)
//! with s == a (mod dcp), so a rank's store rows are a constant shift of its compute rows. Positions are i64, divided
//! with floor (@mod / @divFloor).
//! スロットキャッシュのストアの番地: 会話の位置 [a, b) は論理ストア位置 [s, s + b - a) (s == a (mod dcp)) に置く。
//! そのためランクのストア行は計算行を一定量ずらしたものになる。位置は i64 で、除算は floor (@mod / @divFloor)。
const std = @import("std");

/// x / d rounded up: -floor(-x / d).
/// x / d の切り上げ: -floor(-x / d)。
pub fn ceilDiv(x: i64, d: i64) i64 {
    return -@divFloor(-x, d);
}

/// The local rows of DCP rank r (drank) for the positions [lo, hi).
/// 位置 [lo, hi) のうち DCP の順位 r (drank) が持つローカル行。
pub fn localRows(lo: i64, hi: i64, r: i64, dcp: i64) [2]i64 {
    return .{ ceilDiv(lo - r, dcp), ceilDiv(hi - r, dcp) };
}

/// Logical store positions [s, s + n) holding absolute positions [a, a + n); s == a (mod dcp).
/// 絶対位置 [a, a + n) を持つ論理ストア位置 [s, s + n)。s == a (mod dcp)。
pub const Extent = struct {
    s: i64,
    a: i64,
    n: i64,

    pub fn b(e: Extent) i64 {
        return e.a + e.n;
    }

    /// The left and right pieces at absolute position p (a < p < b), each congruent.
    /// 絶対位置 p (a < p < b) で分けた左右の片。どちらも合同。
    pub fn split(e: Extent, p: i64) error{BadSplit}![2]Extent {
        if (!(e.a < p and p < e.b())) return error.BadSplit;
        const k = p - e.a;
        return .{ .{ .s = e.s, .a = e.a, .n = k }, .{ .s = e.s + k, .a = p, .n = e.n - k } };
    }

    /// The piece holding [lo, hi) ∩ [a, b), or null.
    /// [lo, hi) ∩ [a, b) を持つ片。なければ null。
    pub fn clip(e: Extent, lo_: i64, hi_: i64) ?Extent {
        const lo = @max(lo_, e.a);
        const hi = @min(hi_, e.b());
        if (lo >= hi) return null;
        return .{ .s = e.s + (lo - e.a), .a = lo, .n = hi - lo };
    }
};

/// A free interval [lo, hi) of the logical store.
/// 論理ストアの空き区間 [lo, hi)。
pub const Interval = struct { lo: i64, hi: i64 };

pub const Error = error{ InvalidStore, NotCongruent, NotFree, OutOfRange, DoubleFree, EmptyExtent, Check } || std.mem.Allocator.Error;

/// The free intervals of the logical store [0, S), host bookkeeping only: kept sorted, disjoint and never adjacent
/// (they enter the metadata digest as they are).
/// 論理ストア [0, S) の空き区間 (ホスト側の管理のみ)。昇順・互いに素・隣接なしに保つ (そのままメタデータの digest に
/// 入る)。
pub const Store = struct {
    gpa: std.mem.Allocator,
    S: i64,
    dcp: i64,
    free: std.ArrayList(Interval) = .empty,

    pub fn init(gpa: std.mem.Allocator, S: i64, dcp: i64) Error!Store {
        if (dcp < 1 or S < 0 or @mod(S, dcp) != 0) return error.InvalidStore;
        var st: Store = .{ .gpa = gpa, .S = S, .dcp = dcp };
        if (S > 0) try st.free.append(gpa, .{ .lo = 0, .hi = S });
        return st;
    }

    pub fn deinit(st: *Store) void {
        st.free.deinit(st.gpa);
    }

    pub fn clone(st: *const Store) Error!Store {
        return .{ .gpa = st.gpa, .S = st.S, .dcp = st.dcp, .free = try st.free.clone(st.gpa) };
    }

    /// Free positions in all.
    /// 空き位置の総数。
    pub fn freeTotal(st: *const Store) i64 {
        var t: i64 = 0;
        for (st.free.items) |f| t += f.hi - f.lo;
        return t;
    }

    /// Replace free[k] by its parts outside [s, s + n).
    /// free[k] を [s, s + n) の外側の部分で置き換える。
    fn carve(st: *Store, k: usize, s: i64, n: i64) Error!void {
        const f = st.free.items[k];
        _ = st.free.orderedRemove(k);
        var at = k;
        if (s > f.lo) {
            try st.free.insert(st.gpa, at, .{ .lo = f.lo, .hi = s });
            at += 1;
        }
        if (s + n < f.hi) try st.free.insert(st.gpa, at, .{ .lo = s + n, .hi = f.hi });
    }

    /// (tests only) First fit: the lowest free interval with room at a start congruent to a.
    /// (テスト専用) first fit: a と合同な開始位置に空きがある、最も低い空き区間から取る。
    pub fn alloc(st: *Store, a: i64, n: i64) Error!?Extent {
        if (n <= 0) return error.EmptyExtent;
        for (st.free.items, 0..) |f, k| {
            const s = f.lo + @mod(a - f.lo, st.dcp);
            if (s + n <= f.hi) {
                try st.carve(k, s, n);
                return .{ .s = s, .a = a, .n = n };
            }
        }
        return null;
    }

    /// n positions from a, each piece from the longest congruent run free right now (ties: the lowest interval), in
    /// ascending a; null and nothing taken when they do not fit. The slice is the store's gpa's.
    /// a から n 位置を取る。各片はその時点で最長の合同な空き区間から取り (同長なら最も低い区間)、a の昇順に並ぶ。
    /// 入りきらなければ null で、何も取らない。スライスはストアの gpa のもの。
    pub fn allocPieces(st: *Store, a: i64, n: i64) Error!?[]Extent {
        var trial = try st.clone();
        defer trial.deinit();
        var out: std.ArrayList(Extent) = .empty;
        errdefer out.deinit(st.gpa);
        var p = a;
        while (p < a + n) {
            var best: ?Interval = null; // (s, hi)
            for (trial.free.items) |f| {
                const s = f.lo + @mod(p - f.lo, st.dcp);
                if (s < f.hi and (best == null or f.hi - s > best.?.hi - best.?.lo)) best = .{ .lo = s, .hi = f.hi };
            }
            const bst = best orelse {
                out.deinit(st.gpa);
                return null;
            };
            const take_n = @min(bst.hi - bst.lo, a + n - p);
            try out.append(st.gpa, try trial.take(bst.lo, p, take_n));
            p += take_n;
        }
        std.mem.swap(std.ArrayList(Interval), &st.free, &trial.free);
        return try out.toOwnedSlice(st.gpa);
    }

    /// Claim exactly [s, s + n) (free, congruent) for absolute positions [a, a + n).
    /// 絶対位置 [a, a + n) のために [s, s + n) (空きで合同) をちょうど確保する。
    pub fn take(st: *Store, s: i64, a: i64, n: i64) Error!Extent {
        if (@mod(s - a, st.dcp) != 0) return error.NotCongruent;
        for (st.free.items, 0..) |f, k| {
            if (f.lo <= s and s + n <= f.hi) {
                try st.carve(k, s, n);
                return .{ .s = s, .a = a, .n = n };
            }
        }
        return error.NotFree;
    }

    /// Free exactly e's logical interval (no rounding), merging with free neighbours.
    /// e の論理区間をちょうど (丸めずに) 解放し、隣の空き区間と結合する。
    pub fn release(st: *Store, e: Extent) Error!void {
        const lo = e.s;
        const hi = e.s + e.n;
        if (lo < 0 or hi > st.S) return error.OutOfRange;
        for (st.free.items) |f| if (lo < f.hi and f.lo < hi) return error.DoubleFree;
        try st.free.append(st.gpa, .{ .lo = lo, .hi = hi });
        std.mem.sort(Interval, st.free.items, {}, struct {
            fn lt(_: void, x: Interval, y: Interval) bool {
                return x.lo < y.lo or (x.lo == y.lo and x.hi < y.hi);
            }
        }.lt);
        var w: usize = 0;
        for (st.free.items) |f| {
            if (w > 0 and st.free.items[w - 1].hi == f.lo) {
                st.free.items[w - 1].hi = f.hi;
            } else {
                st.free.items[w] = f;
                w += 1;
            }
        }
        st.free.shrinkRetainingCapacity(w);
    }

    /// Every logical position is in exactly one of the free intervals and the used extents (each congruent).
    /// どの論理位置も、空き区間と使用中の extent (それぞれ合同) のちょうど一つに入っている。
    pub fn check(st: *const Store, used: []const Extent) Error!void {
        const marks = try st.gpa.alloc(u32, @intCast(st.S));
        defer st.gpa.free(marks);
        @memset(marks, 0);
        for (st.free.items) |f| {
            if (f.lo < 0 or f.hi > st.S or f.lo >= f.hi) return error.Check;
            for (@intCast(f.lo)..@intCast(f.hi)) |x| marks[x] += 1;
        }
        for (used) |e| {
            if (@mod(e.s - e.a, st.dcp) != 0 or e.n <= 0 or e.s < 0 or e.s + e.n > st.S) return error.Check;
            for (@intCast(e.s)..@intCast(e.s + e.n)) |x| marks[x] += 1;
        }
        for (marks) |m| if (m != 1) return error.Check;
        // the normal form: ascending, non-empty, neither overlapping nor adjacent
        // 正規形: 昇順・空でない・重なりも隣接もない
        for (st.free.items[0..st.free.items.len -| 1], 1..) |f, k| if (f.hi >= st.free.items[k].lo) return error.Check;
    }
};

test "ceil_div and local_rows follow Python's floor division" {
    try std.testing.expectEqual(@as(i64, -1), ceilDiv(-3, 2));
    try std.testing.expectEqual(@as(i64, 2), ceilDiv(3, 2));
    try std.testing.expectEqual([2]i64{ 0, 2 }, localRows(0, 5, 1, 2));
    try std.testing.expectEqual([2]i64{ 1, 3 }, localRows(1, 5, 0, 2));
}
