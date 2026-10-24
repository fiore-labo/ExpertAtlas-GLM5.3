//! The decode-window collectives over RoCE (TF_GLM53_ROCE), after b12x's RoCEnante: a pinned host region that the
//! integrated GPU reads and writes in place and the ConnectX-7 RDMA-writes into, b12x's RDMA proxy thread
//! (_roce_proxy.c, built as a shared library and opened here) and one kernel a collective (kernels/cuda/glm53/roce.cu).
//! The all-reduce sums in fixed rank order: every rank stores the same bits, and fp32 inputs give the bits of an NCCL
//! all-gather + rank-order sum. Capturable in CUDA graphs: the sequence number (the epoch) lives on the device.
//! decode window 用の RoCE 上の集団通信 (TF_GLM53_ROCE)。b12x の RoCEnante に倣う: 統合 GPU がその場で読み書きし
//! ConnectX-7 が RDMA で書き込む pin 留めのホスト領域、b12x の RDMA proxy スレッド (_roce_proxy.c を共有ライブラリに
//! ビルドしてここで開く)、集団通信ごとに 1 つのカーネル (kernels/cuda/glm53/roce.cu)。
//! all-reduce はランク順に固定して足す: どのランクも同じビットを持ち、fp32 入力なら NCCL の all-gather + ランク順の
//! 和と同じビットになる。CUDA graph に capture できる: シーケンス番号 (epoch) はデバイス上にある。
const std = @import("std");
const cuda = @import("cuda");

const Buf = cuda.DeviceBuffer;

// the largest all-reduce or all-gather, in bytes (decode windows, argmax rows)
// all-reduce と all-gather の最大バイト数 (decode window、argmax の行)
pub const max_bytes = 1 << 20;
const slot_alignment = 4096;
const threads = 512; // threads a block / ブロックあたりのスレッド数
const max_blocks = 8; // a power of two: the arrival counters' modulus / 2 のべき乗: 到着カウンタの法
const counter_classes = 4; // one class a grid size: 1, 2, 4, 8 blocks / grid の大きさごとに 1 つ: 1, 2, 4, 8 ブロック
const abi_version = 4; // 4: the share mode (allReduce2) / 4: 取り分のモード (allReduce2)
// sizeof(roce_blob_t) on aarch64 (checked against the library's roce_blob_bytes)
// aarch64 での sizeof(roce_blob_t) (ライブラリの roce_blob_bytes と照合する)
const blob_bytes = 192;

/// roce.cu's Region, passed by value.
/// roce.cu の Region。値渡し。
const Region = extern struct {
    recv: u64,
    flag: u64,
    send: u64,
    ctrl: u64,
    slot_bytes: u64,
    epoch: u64,
    stage_ctr: u64,
    tail_ctr: u64,
    poison: u64,
    spin_limit: u32,
    world: c_int,
    rank: c_int,
    slots: c_int,
    hcas: c_int,
    flag_stride: c_int,
};

comptime {
    std.debug.assert(@sizeOf(Region) == 96);
}

const Ctx = opaque {};

/// The proxy library's functions.
/// proxy ライブラリの関数。
const Proxy = struct {
    lib: std.DynLib,
    abi_version: *const fn () callconv(.c) c_int,
    layout: *const fn (c_int, u64, *[7]u64) callconv(.c) c_int,
    blob_bytes: *const fn () callconv(.c) u64,
    create: *const fn (c_int, c_int, [*]const [*:0]const u8, c_int, c_int, ?*anyopaque, u64, u64, [*]u8, u64) callconv(.c) ?*Ctx,
    local_blob: *const fn (*Ctx, ?*anyopaque, u64) callconv(.c) c_int,
    connect: *const fn (*Ctx, ?*const anyopaque, u64) callconv(.c) c_int,
    start: *const fn (*Ctx) callconv(.c) c_int,
    failed: *const fn (*Ctx) callconv(.c) c_int,
    err: *const fn (*Ctx) callconv(.c) [*:0]const u8,
    destroy: *const fn (*Ctx) callconv(.c) void,

    fn open(path: []const u8) !Proxy {
        var lib = std.DynLib.open(path) catch {
            std.log.err("roce: cannot open the proxy library {s} (build it from _roce_proxy.c with -libverbs)", .{path});
            return error.RoceProxyUnavailable;
        };
        errdefer lib.close();
        const L = struct {
            fn get(l: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
                return l.lookup(T, name) orelse error.RoceProxySymbol;
            }
        };
        var p: Proxy = undefined;
        p.lib = lib;
        p.abi_version = try L.get(&p.lib, @TypeOf(p.abi_version), "roce_abi_version");
        p.layout = try L.get(&p.lib, @TypeOf(p.layout), "roce_layout");
        p.blob_bytes = try L.get(&p.lib, @TypeOf(p.blob_bytes), "roce_blob_bytes");
        p.create = try L.get(&p.lib, @TypeOf(p.create), "roce_create");
        p.local_blob = try L.get(&p.lib, @TypeOf(p.local_blob), "roce_local_blob");
        p.connect = try L.get(&p.lib, @TypeOf(p.connect), "roce_connect");
        p.start = try L.get(&p.lib, @TypeOf(p.start), "roce_start");
        p.failed = try L.get(&p.lib, @TypeOf(p.failed), "roce_failed");
        p.err = try L.get(&p.lib, @TypeOf(p.err), "roce_error");
        p.destroy = try L.get(&p.lib, @TypeOf(p.destroy), "roce_destroy");
        if (p.abi_version() != abi_version) return error.RoceProxyAbi;
        return p;
    }
};

pub const Kernels = struct {
    module: cuda.Module,
    reduce_f32: cuda.Function,
    reduce_bf16: cuda.Function,
    gather: cuda.Function,
    scatter_f32: cuda.Function,
    scatter_bf16: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm53_roce);
        return .{ .module = m, .reduce_f32 = try m.function("tf_glm53_roce_allreduce_f32"), .reduce_bf16 = try m.function("tf_glm53_roce_allreduce_bf16"), .gather = try m.function("tf_glm53_roce_allgather"), .scatter_f32 = try m.function("tf_glm53_roce_reducescatter_f32"), .scatter_bf16 = try m.function("tf_glm53_roce_reducescatter_bf16") };
    }
};

/// One runtime over a communicator's ranks (the world, or a DCP group).
/// 1 つの communicator のランクにまたがるランタイム (world 全体か DCP グループ)。
pub const Runtime = struct {
    proxy: Proxy,
    ctx: *Ctx,
    region: cuda.HostBuffer,
    // epoch, stage counters [classes], tail counters [classes], poison
    // epoch、stage カウンタ [classes]、tail カウンタ [classes]、poison
    counters: Buf,
    reg: Region,
    ks: *const Kernels,

    /// Every rank of `group` calls this together (the blobs and configurations are exchanged over it). `hcas`: the
    /// RDMA devices (B12X_ROCE_HCA / NCCL_IB_HCA), `gid`: their GID index. Fails on every rank alike.
    /// `group` の全ランクが一緒に呼ぶ (blob と設定はこの group で交換する)。`hcas`: RDMA デバイス (B12X_ROCE_HCA /
    /// NCCL_IB_HCA)、`gid`: その GID index。失敗するときは全ランクがそろって失敗する。
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, group: *const cuda.tp.Group, ks: *const Kernels, lib_path: []const u8, hcas: []const [:0]const u8, gid: u32, spin_limit: u32, stream: cuda.Stream) !Runtime {
        const world = group.world;
        // a rank that cannot open the proxy still takes part in the exchange below (with a failed status), so every
        // rank reaches the same verdict instead of the others waiting in the collective
        // proxy を開けないランクも下の交換には (失敗の状態で) 加わる。そうすれば他のランクが集団通信で待ち続けず、
        // 全ランクが同じ判定になる
        const words = 8;
        const rec = (words * 4 + blob_bytes + 15) / 16 * 16;
        const opened: ?Proxy = Proxy.open(lib_path) catch null;
        if (opened == null or opened.?.blob_bytes() != blob_bytes) {
            const zero = try gpa.alloc(u8, rec);
            defer gpa.free(zero);
            @memset(zero, 0);
            const all = try exchange(gpa, d, group, zero, stream);
            gpa.free(all);
            return error.RoceProxyUnavailable;
        }
        var proxy = opened.?;
        errdefer proxy.lib.close();
        const slot_bytes: u64 = (max_bytes + slot_alignment - 1) / slot_alignment * slot_alignment;
        var lay: [7]u64 = undefined;
        if (proxy.layout(@intCast(world), slot_bytes, &lay) != 0) return error.RoceGeometry;
        // recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots
        var region = try cuda.HostBuffer.allocMapped(d, lay[4]);
        errdefer region.free();
        @memset(region.bytes, 0);
        const host: u64 = @intFromPtr(region.bytes.ptr);
        if (try region.device() != host) return error.RoceNeedsUnifiedAddressing;
        var counters = try Buf.alloc(d, 4 * (2 + 2 * counter_classes));
        errdefer counters.free();
        try counters.fill8(0, null);
        var names: [2][*:0]const u8 = undefined;
        const nh = @min(hcas.len, 2);
        for (0..nh) |i| names[i] = hcas[i].ptr;
        var err: [512]u8 = @splat(0);
        const ctx = proxy.create(@intCast(world), @intCast(group.rank), &names, @intCast(nh), @intCast(gid), region.bytes.ptr, lay[4], slot_bytes, &err, err.len);
        // every rank's blob and status in one exchange: [ok, config words..., blob]
        // 全ランクの blob と状態を 1 回の交換で: [ok, 設定の語..., blob]
        const blob_n: usize = blob_bytes;
        const mine = try gpa.alloc(u8, rec);
        defer gpa.free(mine);
        @memset(mine, 0);
        const cfg = [words]u32{ @intFromBool(ctx != null), @intCast(nh), @intCast(slot_bytes / 4096), @intCast(lay[5]), @intCast(lay[6]), spin_limit, threads, max_blocks };
        @memcpy(mine[0 .. words * 4], std.mem.sliceAsBytes(&cfg));
        if (ctx) |c| if (proxy.local_blob(c, mine[words * 4 ..].ptr, blob_n) != 0) {
            mine[0] = 0;
        };
        if (ctx == null) std.log.err("roce: rank {d}: {s}", .{ group.rank, std.mem.sliceTo(&err, 0) });
        const all = try exchange(gpa, d, group, mine, stream);
        defer gpa.free(all);
        var ok = true;
        for (0..world) |r| {
            const w = std.mem.bytesAsSlice(u32, all[r * rec ..][0 .. words * 4]);
            if (w[0] != 1) ok = false;
            if (!std.mem.eql(u32, @alignCast(w[1..]), cfg[1..])) ok = false;
        }
        if (!ok) {
            if (ctx) |c| proxy.destroy(c);
            return error.RoceSetupFailed;
        }
        const blobs = try gpa.alloc(u8, world * blob_n);
        defer gpa.free(blobs);
        for (0..world) |r| @memcpy(blobs[r * blob_n ..][0..blob_n], all[r * rec + words * 4 ..][0..blob_n]);
        var good: u8 = 1;
        if (proxy.connect(ctx.?, blobs.ptr, blobs.len) != 0 or proxy.start(ctx.?) != 0) {
            std.log.err("roce: rank {d}: connect: {s}", .{ group.rank, std.mem.span(proxy.err(ctx.?)) });
            good = 0;
        }
        var verdict = [16]u8{ good, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        const vs = try exchange(gpa, d, group, &verdict, stream);
        defer gpa.free(vs);
        for (0..world) |r| if (vs[r * 16] != 1) {
            proxy.destroy(ctx.?);
            return error.RoceConnectFailed;
        };
        const ep = counters.ptr;
        return .{
            .proxy = proxy,
            .ctx = ctx.?,
            .region = region,
            .counters = counters,
            .ks = ks,
            .reg = .{
                .recv = host + lay[0], .flag = host + lay[1], .send = host + lay[2], .ctrl = host + lay[3], .slot_bytes = slot_bytes,
                .epoch = ep, .stage_ctr = ep + 4, .tail_ctr = ep + 4 * (1 + counter_classes), .poison = ep + 4 * (1 + 2 * counter_classes),
                .spin_limit = spin_limit, .world = @intCast(world), .rank = @intCast(group.rank), .slots = @intCast(lay[6]), .hcas = @intCast(nh), .flag_stride = @intCast(lay[5]),
            },
        };
    }

    /// Every rank's `local` bytes in rank order over NCCL (the setup exchange; staged through device memory).
    /// 全ランクの `local` のバイト列を NCCL でランク順に集める (準備用の交換。デバイスメモリを経由する)。
    fn exchange(gpa: std.mem.Allocator, d: *const cuda.Driver, group: *const cuda.tp.Group, local: []const u8, stream: cuda.Stream) ![]u8 {
        var src = try Buf.fromHost(d, local);
        defer src.free();
        var dst = try Buf.alloc(d, local.len * group.world);
        defer dst.free();
        try group.lib.check(group.lib.api.ncclAllGather(src.ptr, dst.ptr, local.len, .u8, group.comm, stream.handle), "ncclAllGather");
        try stream.synchronize();
        const out = try gpa.alloc(u8, local.len * group.world);
        errdefer gpa.free(out);
        try dst.download(0, out);
        return out;
    }

    /// A power-of-two grid of about two 16-byte packs a thread, at most max_blocks, and its counters.
    /// スレッドあたり 16 バイトの pack 約 2 個となる 2 のべき乗の grid (max_blocks まで) と、そのカウンタ。
    fn grid(r: *const Runtime, packs: u32) struct { blocks: u32, reg: Region } {
        const per = 2 * threads;
        const need = @max(1, (packs + per - 1) / per);
        const blocks = @min(std.math.ceilPowerOfTwo(u32, need) catch max_blocks, max_blocks);
        // widened: log2_int's u5 would wrap 4 * (1 + 4 + 3)
        // 幅を広げる: log2_int の u5 のままだと 4 * (1 + 4 + 3) があふれる
        const class: u64 = std.math.log2_int(u32, blocks);
        var g = r.reg;
        g.stage_ctr = r.reg.epoch + 4 * (1 + class);
        g.tail_ctr = r.reg.epoch + 4 * (1 + counter_classes + class);
        return .{ .blocks = blocks, .reg = g };
    }

    /// out = the rank-order sum of every rank's `in` (`nbytes`, a multiple of 16, 16-byte aligned; fp32 or bf16).
    /// out = 全ランクの `in` をランク順に足した和 (`nbytes` は 16 の倍数、16 バイト境界。fp32 か bf16)。
    pub fn allReduce(r: *const Runtime, s: cuda.Stream, in: u64, out: u64, nbytes: u32, bf16: bool) !void {
        if (nbytes % 16 != 0 or nbytes > max_bytes or in % 16 != 0 or out % 16 != 0) return error.RoceShape;
        const packs = nbytes / 16;
        const g = r.grid(packs);
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(c_int, @intCast(packs)));
        a.add(nbytes);
        a.add(g.reg);
        try cuda.launch.launch(if (bf16) r.ks.reduce_bf16 else r.ks.reduce_f32, .{ .grid = .{ .x = g.blocks, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    /// `allReduce` in two shots: a reduce-scatter (each peer is sent its share of `in` alone, this rank's share of the
    /// rank-order sum into `mid`, nbytes / world) and an all-gather of the shares into `out`. 2 (W - 1) / W of the
    /// bytes on this rank's wire instead of W - 1 times, the same bits; two round trips. nbytes a multiple of
    /// 16 x world.
    /// 2 段の `allReduce`: reduce-scatter (各 peer には `in` のその取り分だけを送り、ランク順の和のこのランクの取り分を
    /// `mid` へ、nbytes / world) と、取り分の all-gather で `out` へ。このランクの回線のバイトは W - 1 倍ではなく
    /// 2 (W - 1) / W、同じビット。往復は 2 回。nbytes は 16 x world の倍数。
    pub fn allReduce2(r: *const Runtime, s: cuda.Stream, in: u64, mid: u64, out: u64, nbytes: u32, bf16: bool) !void {
        const W: u32 = @intCast(r.reg.world);
        if (nbytes % (16 * W) != 0 or nbytes > max_bytes or in % 16 != 0 or mid % 16 != 0 or out % 16 != 0) return error.RoceShape;
        const packs = nbytes / 16;
        const g = r.grid(packs);
        var a: cuda.Args = .{};
        a.add(in);
        a.add(mid);
        a.add(@as(c_int, @intCast(packs)));
        a.add(nbytes);
        a.add(g.reg);
        try cuda.launch.launch(if (bf16) r.ks.scatter_bf16 else r.ks.scatter_f32, .{ .grid = .{ .x = g.blocks, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
        return r.allGather(s, mid, out, nbytes / W);
    }

    /// out = every rank's `in` (nbytes each) concatenated in rank order (dim 0).
    /// out = 全ランクの `in` (各 nbytes) をランク順につないだもの (dim 0)。
    pub fn allGather(r: *const Runtime, s: cuda.Stream, in: u64, out: u64, nbytes: u32) !void {
        if (nbytes % 16 != 0 or nbytes > max_bytes or in % 16 != 0 or out % 16 != 0) return error.RoceShape;
        const packs = nbytes / 16;
        const g = r.grid(packs);
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(c_int, @intCast(packs)));
        a.add(nbytes);
        a.add(@as(c_int, @intCast(packs)));
        a.add(g.reg);
        try cuda.launch.launch(r.ks.gather, .{ .grid = .{ .x = g.blocks, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    /// False after a wait timed out (the control record's error word) or the proxy failed; logs which.
    /// 待ちがタイムアウトした (制御レコードの error 語) か proxy が失敗したら false。どちらかをログに出す。
    pub fn healthy(r: *const Runtime) bool {
        const ctrl: *const volatile [7]u32 = @ptrFromInt(r.reg.ctrl);
        if (ctrl[2] != 0) std.log.err("roce (world {d}, rank {d}): wait timed out at seq {d} on peer {d} hca {d}", .{ r.reg.world, r.reg.rank, ctrl[2], ctrl[3], ctrl[6] });
        if (r.proxy.failed(r.ctx) != 0) std.log.err("roce (world {d}, rank {d}): proxy failed: {s}", .{ r.reg.world, r.reg.rank, std.mem.span(r.proxy.err(r.ctx)) });
        return ctrl[2] == 0 and r.proxy.failed(r.ctx) == 0;
    }

    pub fn deinit(r: *Runtime) void {
        r.proxy.destroy(r.ctx);
        r.region.free();
        r.counters.free();
        r.proxy.lib.close();
    }
};
