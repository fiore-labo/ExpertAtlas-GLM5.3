//! EXL3 weights unpacked to fp16 W_q for prompt GEMMs, kept for the next calls (the parts of a sequence-parallel chunk
//! read the same weights; TF_GLM53_UNPACK_CACHE_MB). The weights never change, so a kept copy gives the same result.
//! The main and comm streams share it, ordered by events.
//! prompt GEMM 用に fp16 の W_q へ展開した EXL3 重みを、次の呼び出しのために保持する (sequence-parallel チャンクの
//! 各部分は同じ重みを読む。TF_GLM53_UNPACK_CACHE_MB)。重みは変わらないので、保持した写しでも結果は同じ。
//! main と comm のストリームで共有し、順序は event で守る。
const std = @import("std");
const cuda = @import("cuda");
const linear = @import("linear.zig");
const exl3 = @import("exl3.zig");
const model = @import("model.zig");

pub const streams = 2; // forward.Ctx.sid: 0 main, 1 the SP comm stream / 0 は main、1 は SP の comm ストリーム

const Slot = struct {
    buf: cuda.DeviceBuffer,
    // the packed weight's device address (a Linear may be a by-value copy, so its own address is no key)
    // パック済み重みのデバイスアドレス (Linear は値のコピーのことがあり、それ自体のアドレスはキーにならない)
    owner: u64 = 0,
    stamp: u64 = 0,
    ready: cuda.Event,
    used: [streams]cuda.Event,
};

pub const Cache = struct {
    slots: []Slot,
    clock: u64 = 0,
    bytes: usize,

    /// Slots of the widest prompt-GEMM weight among `layers` (q_a, kv_a, q_b, o_proj, the indexer's wq_b, the MLPs),
    /// as many as fit `budget` bytes (at least 2).
    /// `layers` の prompt GEMM 重み (q_a・kv_a・q_b・o_proj・indexer の wq_b・MLP) で最大のものの大きさのスロットを、
    /// `budget` バイトに入るだけ (最低 2 個) 作る。
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, layers: []const model.Layer, budget: usize) !Cache {
        var widest: usize = 0;
        for (layers) |*L| {
            const ls = [_]*const linear.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down };
            for (ls) |l| widest = @max(widest, @as(usize, l.k) * l.n * 2);
            if (L.indexer) |*ix| widest = @max(widest, @as(usize, ix.wq_b.k) * ix.wq_b.n * 2);
        }
        const n = @max(2, budget / @max(widest, 1));
        const slots = try gpa.alloc(Slot, n);
        for (slots) |*s| {
            s.* = .{ .buf = try cuda.DeviceBuffer.alloc(d, widest), .ready = try cuda.Event.init(d, false), .used = undefined };
            for (&s.used) |*e| e.* = try cuda.Event.init(d, false);
            // record every event once so the first waits on them are no-ops
            // 各 event を一度記録しておき、最初の待ちがすぐ通るようにする
            try s.ready.record(.{ .d = d, .handle = null });
            for (&s.used) |*e| try e.record(.{ .d = d, .handle = null });
        }
        return .{ .slots = slots, .bytes = widest };
    }

    /// `l`'s unpacked W_q [k, n] fp16 for a reader on `stream` (stream index `sid`): a kept copy (waiting for its
    /// unpack), else the least recently used slot overwritten (after every stream's readers of it). The caller runs its
    /// GEMM, then `done`.
    /// `stream` (ストリーム番号 `sid`) 上の読み手に `l` の展開済み W_q [k, n] fp16 を返す: 保持した写し (その展開を
    /// 待つ)、なければ最も長く使われていないスロットを (全ストリームの読み手を待って) 上書きする。呼び出し側は GEMM を
    /// 発行してから `done` を呼ぶ。
    pub fn get(c: *Cache, x3: *const exl3.Kernels, stream: cuda.Stream, sid: usize, l: *const linear.Linear) !*Slot {
        c.clock += 1;
        for (c.slots) |*s| if (s.owner == l.words.ptr) {
            s.stamp = c.clock;
            try stream.wait(s.ready);
            return s;
        };
        var lru = &c.slots[0];
        for (c.slots) |*s| if (s.stamp < lru.stamp) {
            lru = s;
        };
        for (&lru.used) |e| try stream.wait(e);
        const st = l.strides();
        try x3.unpackW(stream, l.words.ptr, lru.buf.ptr, l.k, l.n, l.k2, l.cb, st.k, st.nb);
        try lru.ready.record(stream);
        lru.owner = l.words.ptr;
        lru.stamp = c.clock;
        _ = sid;
        return lru;
    }

    /// The reader on stream `sid` has issued its GEMM over `s`.
    /// ストリーム `sid` の読み手が `s` を読む GEMM を発行し終えた。
    pub fn done(_: *Cache, s: *Slot, stream: cuda.Stream, sid: usize) !void {
        try s.used[sid].record(stream);
    }
};
