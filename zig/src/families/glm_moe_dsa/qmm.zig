//! The DFlash2 drafter's 4-bit matmul on GB10: qmm_group.cu's group_kernel, one projection a launch. The tile follows
//! the rows; K is split in `sk` slices summed inside a cluster of sk blocks. The slices come from the shape alone, so
//! no row affects another.
//! GB10 上の DFlash2 drafter の 4 ビット行列積: qmm_group.cu の group_kernel で、1 回の launch に 1 projection。
//! タイルは行数で選び、K を `sk` 個に分けて sk ブロックのクラスタ内で足す。分け方は形だけで決まるので、行どうしは
//! 影響しない。
const std = @import("std");
const cuda = @import("cuda");
const q4 = @import("q4.zig");

/// qmm_group.cu Part / Parts (passed by value as a __grid_constant__ kernel argument).
/// qmm_group.cu の Part / Parts (__grid_constant__ のカーネル引数として値で渡す)。
const Part = extern struct {
    w: u64,
    scales: u64,
    biases: u64,
    out: u64,
    n: c_int,
    npad: c_int,
    sk: c_int,
    tiles: c_int,
    first: c_int,
};
const Parts = extern struct {
    p: [4]Part,
    count: c_int,
};

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

/// One packed matrix on the device.
/// デバイス上の packed 行列 1 つ。
pub const Matrix = struct {
    words: cuda.DeviceBuffer,
    scales: cuda.DeviceBuffer,
    biases: cuda.DeviceBuffer,
    n: u32,
    k: u32,
    npad: u32,
    sk: u32,

    pub fn upload(d: *const cuda.Driver, p: q4.Packed) !Matrix {
        return .{
            .words = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.words)),
            .scales = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.scales)),
            .biases = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.biases)),
            .n = @intCast(p.n),
            .k = @intCast(p.k),
            .npad = @intCast(p.npad),
            .sk = splitK(p.n, p.k),
        };
    }
};

/// K slices of a shape: the table's, else K halved until the column tiles times slices reach 192.
/// 形ごとの K の分割数: 表にあればその値、なければ列タイル数 x 分割数が 192 に届くまで K を半分にしていく。
pub fn splitK(n: usize, k: usize) u32 {
    const table = [_]struct { n: usize, k: usize, sk: u32 }{
        .{ .n = 12576, .k = 4096, .sk = 4 }, .{ .n = 4096, .k = 4096, .sk = 2 }, .{ .n = 2048, .k = 4096, .sk = 4 },
        .{ .n = 8192, .k = 1536, .sk = 4 },  .{ .n = 8192, .k = 512, .sk = 4 },  .{ .n = 4096, .k = 8192, .sk = 8 },
    };
    for (table) |t| if (t.n == n and t.k == k) return t.sk;
    const col_tiles = (n + 63) / 64;
    const groups = k / q4.gs;
    var sk: u32 = 1;
    while (sk < 8 and col_tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

const Tile = struct { bm: u32, bn: u32, threads: u32, smem: u32 };

/// LaneTile<64, BM, BN, WM, WN, STAGES>: threads and dynamic shared bytes (stages, or the cluster sum's partials).
/// LaneTile<64, BM, BN, WM, WN, STAGES> のスレッド数と動的共有メモリのバイト数 (ステージ分かクラスタ和の部分和)。
fn tileOf(bm: u32, bn: u32, wm: u32, wn: u32, stages: u32) Tile {
    const threads = wm * wn * 32;
    const mt = bm / wm / 16;
    const nt = bn / wn / 8;
    const stage = bm * 128 + bn * 32 + 2 * bn * 2 + bm * 4;
    const partials = mt * nt * 4 * threads * 4;
    return .{ .bm = bm, .bn = bn, .threads = threads, .smem = @max(stages * stage, partials) };
}

const tiles = [4]Tile{ tileOf(16, 64, 1, 4, 8), tileOf(32, 64, 1, 4, 4), tileOf(64, 64, 1, 4, 4), tileOf(64, 128, 2, 4, 3) };
const names = [4][2][:0]const u8{
    .{ "_ZN7tf_qmmg12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii", "_ZN7tf_qmmg12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii" },
    .{ "_ZN7tf_qmmg12group_kernelILi64ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii", "_ZN7tf_qmmg12group_kernelILi64ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii" },
    .{ "_ZN7tf_qmmg12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii", "_ZN7tf_qmmg12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii" },
    .{ "_ZN7tf_qmmg12group_kernelILi64ELi64ELi128ELi2ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii", "_ZN7tf_qmmg12group_kernelILi64ELi64ELi128ELi2ELi4ELi3ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii" },
};

pub const Kernels = struct {
    module: cuda.Module,
    fns: [4][2]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_qmm_group);
        var fns: [4][2]cuda.Function = undefined;
        for (0..4) |t| for (0..2) |f| {
            fns[t][f] = try m.function(names[t][f]);
            try fns[t][f].allowDynamicShared(tiles[t].smem);
        };
        return .{ .module = m, .fns = fns };
    }

    /// out [m, n] (bf16, or fp32 with `wide`) = x [m, k] bf16 (rows `ldx` apart, 16-byte aligned) @ w^T; `xs` fp32
    /// [m, k / 64]: the input's 64-group sums.
    /// out [m, n] (bf16、`wide` なら fp32) = x [m, k] bf16 (行の間隔 `ldx`、16 バイト境界) @ w^T。`xs` は fp32
    /// [m, k / 64]: 入力の 64 要素グループごとの和。
    pub fn matmul(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: u32, xs: u64, w: *const Matrix, out: u64, m: u32, wide: bool) !void {
        const t: usize = if (m <= 16) 0 else if (m <= 32) 1 else if (m <= 64) 2 else 3;
        const tile = tiles[t];
        const rows_t = (m + tile.bm - 1) / tile.bm;
        const ntiles = (w.n + tile.bn - 1) / tile.bn;
        const C = w.sk; // one part: the cluster is its K slices / part は 1 つ: クラスタはその K スライス
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = w.words.ptr, .scales = w.scales.ptr, .biases = w.biases.ptr, .out = out, .n = @intCast(w.n), .npad = @intCast(w.npad), .sk = @intCast(w.sk), .tiles = @intCast(ntiles), .first = 0 };
        const clusters = rows_t * ((ntiles + C / w.sk - 1) / (C / w.sk));
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        a.add(@as(c_int, @intCast(m)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(if (m == 1) w.k else ldx)));
        a.add(@as(c_int, @intCast(rows_t)));
        a.add(@as(c_int, @intCast(C)));
        try cuda.launch.launch(ks.fns[t][@intFromBool(wide)], .{
            .grid = .{ .x = clusters * C, .y = 1, .z = 1 },
            .block = .{ .x = tile.threads, .y = 1, .z = 1 },
            .shared = tile.smem,
            .cluster = if (C > 1) .{ .x = C, .y = 1, .z = 1 } else null,
            .pdl = true, // GB10: may start while the previous kernel finishes / 前のカーネルの終了前に始めてよい
        }, stream, &a);
    }
};

test "split_k follows the shape rule" {
    try std.testing.expectEqual(@as(u32, 2), splitK(4096, 4096));
    // fc: 6144 x 36864 -> 96 tiles; 576 groups: 1 -> 2 (192 reached)
    // fc: 6144 x 36864 -> 96 タイル。576 グループ: 1 -> 2 (192 に届く)
    try std.testing.expectEqual(@as(u32, 2), splitK(6144, 36864));
    // kv: 256 x 6144 -> 4 tiles, 96 groups: 2, 4, 8 (the cap)
    // kv: 256 x 6144 -> 4 タイル、96 グループ: 2, 4, 8 (上限)
    try std.testing.expectEqual(@as(u32, 8), splitK(256, 6144));
}
