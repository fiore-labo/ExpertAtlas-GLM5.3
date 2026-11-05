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
// 4: the share mode (allReduce2); 5: each peer's stripes over the HCAs in its subnet (up to 4 HCAs, a triangle of
// direct links)
// 4: 取り分のモード (allReduce2)。5: 相手ごとの stripe はそのサブネットの HCA だけ (HCA 4 つまで、直結の三角形)
// 6: a ring of four (roce_set_ring4): the opposite rank's payload relayed by both neighbours (glm53/R3RT-102.md)
// 6: 4 台のリング (roce_set_ring4): 向かいのランクの payload は両隣が中継する
// 7: roce_pin and roce_tid (the proxy thread kept on one CPU, pinProxies; its id for the CPU it runs on, cpuOf)
// 7: roce_pin と roce_tid (proxy のスレッドを 1 つの CPU に留める pinProxies、動いている CPU を見る cpuOf 用の id)
const abi_version = 7;
const max_hcas = 4;
// sizeof(roce_blob_t) on aarch64 (checked against the library's roce_blob_bytes)
// aarch64 での sizeof(roce_blob_t) (ライブラリの roce_blob_bytes と照合する)
const blob_bytes = 392;

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

/// roce.cu's Pf (TF_GLM53_L2PF): up to 4 byte ranges the threads that do not poll ask the L2 for while a collective
/// waits on its peers (gr8's paced L2 prefetch of the next weights, here in the wait itself). Writes nothing.
/// roce.cu の Pf (TF_GLM53_L2PF): 集団通信が peer を待つ間に、poll しないスレッドが L2 に求める最大 4 つのバイト
/// 範囲 (gr8 の次の重みの L2 先読みを、ここでは待ちの中で)。何も書かない。
pub const Prefetch = extern struct {
    p: [4]u64 = @splat(0),
    n: [4]u32 = @splat(0),
    count: c_int = 0,

    /// `bytes` at `ptr` as one more range (the first 4 only; a range is cut at 4 GiB).
    /// `ptr` の `bytes` を範囲に加える (最初の 4 つまで。範囲は 4 GiB で切る)。
    pub fn add(pf: *Prefetch, ptr: u64, bytes: usize) void {
        if (pf.count >= 4 or bytes == 0 or ptr == 0) return;
        const i: usize = @intCast(pf.count);
        pf.p[i] = ptr;
        pf.n[i] = @intCast(@min(bytes, std.math.maxInt(u32)));
        pf.count += 1;
    }
};

comptime {
    std.debug.assert(@sizeOf(Prefetch) == 56);
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
    stripes: *const fn (*Ctx) callconv(.c) c_int,
    stripe_hca: *const fn (*Ctx, c_int, c_int) callconv(.c) c_int,
    set_ring4: *const fn (*Ctx, c_int) callconv(.c) c_int,
    pin: *const fn (*Ctx, c_int) callconv(.c) c_int,
    tid: *const fn (*Ctx) callconv(.c) c_int,
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
        p.stripes = try L.get(&p.lib, @TypeOf(p.stripes), "roce_stripes");
        p.stripe_hca = try L.get(&p.lib, @TypeOf(p.stripe_hca), "roce_stripe_hca");
        p.set_ring4 = try L.get(&p.lib, @TypeOf(p.set_ring4), "roce_set_ring4");
        p.pin = try L.get(&p.lib, @TypeOf(p.pin), "roce_pin");
        p.tid = try L.get(&p.lib, @TypeOf(p.tid), "roce_tid");
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

/// TF_GLM53_ROCE_PROXY_CORES: the cores the proxy threads poll on. They spin while the decode runs; on the fast
/// Cortex-X925 cores the scheduler often gives them they heat the SoC (measured 10-31 on four GB10: TSOC 5-16 °C lower on
/// A725, decode unchanged; GitHub MiaAI-Lab DeepSeek-V4.1 TensorFold issue 40). `a725` (the default) / `x925`: the first
/// core of that kind and the one halfway down their list (GB10: A725 0 and 10, X925 5 and 15, one a cluster), the
/// world's proxy then the DCP group's; `os`: left to the scheduler.
/// TF_GLM53_ROCE_PROXY_CORES: proxy のスレッドが待つコア。decode の間ずっと回るので、スケジューラが選びがちな速い
/// Cortex-X925 では SoC を熱する (10-31 に GB10 4 台で実測: A725 では TSOC が 5〜16 °C 下がり、decode は変わらない。
/// GitHub MiaAI-Lab DeepSeek-V4.1 TensorFold の issue 40)。`a725` (既定) / `x925`: その種類の最初のコアと、その一覧の
/// 半分の所 (GB10: A725 は 0 と 10、X925 は 5 と 15、クラスタに 1 つずつ)。world の proxy、次に DCP グループのもの。
/// `os`: スケジューラに任せる。
pub const Mode = enum { a725, x925, os };

/// A core's kind from /proc/cpuinfo's CPU part (Cortex-A725 0xd87, Cortex-X925 0xd85).
/// /proc/cpuinfo の CPU part によるコアの種類 (Cortex-A725 は 0xd87、Cortex-X925 は 0xd85)。
pub const Kind = enum { unknown, a725, x925 };

pub const max_cpus = 256;

/// Every CPU's kind (unknown past what /proc/cpuinfo names).
/// 各 CPU の種類 (/proc/cpuinfo にないものは unknown)。
pub fn coreKinds(io: std.Io) [max_cpus]Kind {
    var kinds: [max_cpus]Kind = @splat(.unknown);
    var buf: [64 * 1024]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/cpuinfo", &buf) catch return kinds;
    var cpu: ?u16 = null;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.mem.eql(u8, key, "processor")) {
            cpu = std.fmt.parseInt(u16, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "CPU part")) {
            const c = cpu orelse continue;
            if (c >= max_cpus) continue;
            kinds[c] = if (std.mem.eql(u8, val, "0xd87")) .a725 else if (std.mem.eql(u8, val, "0xd85")) .x925 else .unknown;
        }
    }
    return kinds;
}

/// Fiore-lab: "a725" when every CPU in `allowed` (a sched_getaffinity mask) is a Cortex-A725, else "os": another kind
/// among them, or a kind not known.
/// Fiore-lab: `allowed` (sched_getaffinity のマスク) の CPU がすべて Cortex-A725 なら "a725"、それ以外は "os":
/// 別の種類が混ざる、または種類が分からない。
pub fn coresOf(kinds: *const [max_cpus]Kind, allowed: []const usize) []const u8 {
    var any = false;
    for (allowed, 0..) |word, i| for (0..@bitSizeOf(usize)) |b| {
        if (word >> @intCast(b) & 1 == 0) continue;
        const c = i * @bitSizeOf(usize) + b;
        if (c >= max_cpus or kinds[c] != .a725) return "os";
        any = true;
    };
    return if (any) "a725" else "os";
}

/// Where this process may run, read from the process, not from the settings (/health engine_cores).
/// このプロセスが動ける所。設定ではなくプロセスから読む (/health engine_cores)。
pub fn engineCores(io: std.Io) []const u8 {
    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    const rc = std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &set);
    if (std.os.linux.errno(rc) != .SUCCESS) return "os";
    const kinds = coreKinds(io);
    return coresOf(&kinds, &set);
}

test "the engine's cores: A725 only when every allowed CPU is one" {
    var kinds: [max_cpus]Kind = @splat(.unknown);
    for (0..5) |c| kinds[c] = .a725;
    for (5..10) |c| kinds[c] = .x925;
    for (10..15) |c| kinds[c] = .a725;
    for (15..20) |c| kinds[c] = .x925;
    try std.testing.expectEqualStrings("a725", coresOf(&kinds, &[_]usize{0b0111_1100_0001_1111}));
    try std.testing.expectEqualStrings("a725", coresOf(&kinds, &[_]usize{0b1}));
    // every core / 全コア
    try std.testing.expectEqualStrings("os", coresOf(&kinds, &[_]usize{(1 << 20) - 1}));
    // one X925 among them / X925 が 1 つ
    try std.testing.expectEqualStrings("os", coresOf(&kinds, &[_]usize{0b10_0000}));
    // a kind not known, or none allowed / 種類が不明、または 1 つもない
    try std.testing.expectEqualStrings("os", coresOf(&kinds, &[_]usize{1 << 30}));
    try std.testing.expectEqualStrings("os", coresOf(&kinds, &[_]usize{0}));
}

/// The two CPUs `mode` gives the proxies (null: none, the scheduler's choice).
/// `mode` が proxy に与える 2 つの CPU (null はなし、スケジューラが選ぶ)。
pub fn proxyCpus(kinds: *const [max_cpus]Kind, mode: Mode) [2]?u16 {
    const want: Kind = switch (mode) {
        .a725 => .a725,
        .x925 => .x925,
        .os => return .{ null, null },
    };
    var list: [max_cpus]u16 = undefined;
    var n: usize = 0;
    for (kinds, 0..) |k, c| if (k == want) {
        list[n] = @intCast(c);
        n += 1;
    };
    if (n == 0) return .{ null, null };
    return .{ list[0], list[n / 2] };
}

/// The proxies of `rs` (the world's, the DCP group's) kept on `cpus` (this rank's own threads). Returns where each is
/// kept: its CPU, -1 when left to the scheduler (asked to, or the pin failed: logged), -2 when there is no such proxy.
/// `rs` の proxy (world のもの、DCP グループのもの) を `cpus` に留める (このランク自身のスレッド)。それぞれをどこに
/// 留めたかを返す: その CPU、スケジューラ任せなら -1 (そう指定されたか、留められなかった: ログに出す)、その proxy が
/// なければ -2。
pub fn pinProxies(rank: usize, mode: Mode, rs: [2]?*Runtime, cpus: [2]?u16) [2]i16 {
    var out: [2]i16 = .{ -2, -2 };
    const what = [2][]const u8{ "the world's", "the DCP group's" };
    for (rs, cpus, what, &out) |r, c, w, *o| {
        const rt = r orelse continue;
        o.* = -1;
        const cpu = c orelse {
            std.log.info("roce: rank {d}: {s} proxy polls where the scheduler puts it ({s})", .{ rank, w, @tagName(mode) });
            continue;
        };
        const rc = rt.proxy.pin(rt.ctx, cpu);
        if (rc != 0) {
            std.log.warn("roce: rank {d}: {s} proxy not kept on CPU {d} (error {d}); it polls where the scheduler puts it", .{ rank, w, cpu, rc });
            continue;
        }
        o.* = @intCast(cpu);
        std.log.info("roce: rank {d}: {s} proxy polls on CPU {d} ({s})", .{ rank, w, cpu, @tagName(mode) });
    }
    return out;
}

/// The CPU thread `tid` of this process last ran on (/proc/self/task/<tid>/stat, field 39), null if unknown.
/// このプロセスのスレッド `tid` が最後に動いた CPU (/proc/self/task/<tid>/stat の 39 番目)。分からなければ null。
pub fn cpuOf(io: std.Io, tid: i32) ?u16 {
    if (tid <= 0) return null;
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/self/task/{d}/stat", .{tid}) catch return null;
    var buf: [1024]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, path, &buf) catch return null;
    // the fields after the command (which may hold spaces) start past its closing parenthesis, at field 3
    // コマンド名 (空白を含みうる) より後の項目は、その閉じ括弧の後の 3 番目から始まる
    const close = std.mem.lastIndexOfScalar(u8, text, ')') orelse return null;
    var it = std.mem.tokenizeScalar(u8, text[close + 1 ..], ' ');
    var field: usize = 3;
    while (it.next()) |f| : (field += 1) {
        if (field == 39) return std.fmt.parseInt(u16, f, 10) catch null;
    }
    return null;
}

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
    /// RDMA devices (B12X_ROCE_HCA / NCCL_IB_HCA, up to 4), `gid`: their GID index (< 0: each device's last IPv4
    /// GID, in TF_ROCE_ADDR_RANGE when set). Each peer is reached over the devices in its subnet (the library's
    /// stripes). Fails on every rank alike.
    /// `group` の全ランクが一緒に呼ぶ (blob と設定はこの group で交換する)。`hcas`: RDMA デバイス (B12X_ROCE_HCA /
    /// NCCL_IB_HCA、4 つまで)、`gid`: その GID index (負ならデバイスごとに最後の IPv4 の GID、TF_ROCE_ADDR_RANGE が
    /// あればその範囲で)。相手にはそのサブネットのデバイスで送る (ライブラリの stripe)。失敗するときは全ランクが
    /// そろって失敗する。
    /// `ring4`: the group is a ring of four in ring order (tp.Group.topology): the opposite rank is reached through
    /// both neighbours' relays; the share mode (allReduce2) fails there.
    /// `ring4`: group はリング順の 4 台のリング: 向かいのランクには両隣の中継で届く。取り分のモード (allReduce2) は失敗する。
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, group: *const cuda.tp.Group, ks: *const Kernels, lib_path: []const u8, hcas: []const [:0]const u8, gid: i32, spin_limit: u32, ring4: bool, stream: cuda.Stream) !Runtime {
        const world = group.world;
        // a rank that cannot open the proxy still takes part in the exchange below (with a failed status), so every
        // rank reaches the same verdict instead of the others waiting in the collective
        // proxy を開けないランクも下の交換には (失敗の状態で) 加わる。そうすれば他のランクが集団通信で待ち続けず、
        // 全ランクが同じ判定になる
        const words = 9;
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
        var names: [max_hcas][*:0]const u8 = undefined;
        const nh = @min(hcas.len, max_hcas);
        for (0..nh) |i| names[i] = hcas[i].ptr;
        var err: [512]u8 = @splat(0);
        const ctx = proxy.create(@intCast(world), @intCast(group.rank), &names, @intCast(nh), gid, region.bytes.ptr, lay[4], slot_bytes, &err, err.len);
        // every rank's blob and status in one exchange: [ok, config words..., blob]
        // 全ランクの blob と状態を 1 回の交換で: [ok, 設定の語..., blob]
        const blob_n: usize = blob_bytes;
        const mine = try gpa.alloc(u8, rec);
        defer gpa.free(mine);
        @memset(mine, 0);
        const cfg = [words]u32{ @intFromBool(ctx != null), @intCast(nh), @intCast(slot_bytes / 4096), @intCast(lay[5]), @intCast(lay[6]), spin_limit, threads, max_blocks, @intFromBool(ring4) };
        @memcpy(mine[0 .. words * 4], std.mem.sliceAsBytes(&cfg));
        if (ctx) |c| if (ring4 and proxy.set_ring4(c, 1) != 0) {
            std.log.err("roce: rank {d}: {s}", .{ group.rank, std.mem.span(proxy.err(c)) });
            mine[0] = 0;
        };
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
        // every peer's stripes (the GPU waits for that many flags a peer): the same count on every rank
        // 相手ごとの stripe (GPU は相手ごとにその数の flag を待つ): どのランクでも同じ数
        const stripes: u8 = if (good == 1) @intCast(@max(0, proxy.stripes(ctx.?))) else 0;
        var verdict = [16]u8{ good, stripes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        const vs = try exchange(gpa, d, group, &verdict, stream);
        defer gpa.free(vs);
        for (0..world) |r| if (vs[r * 16] != 1 or vs[r * 16 + 1] != stripes or stripes == 0) {
            if (vs[r * 16] == 1 and vs[r * 16 + 1] != stripes) std.log.err("roce: rank {d} reaches each peer over {d} HCAs, rank {d} over {d}", .{ group.rank, stripes, r, vs[r * 16 + 1] });
            proxy.destroy(ctx.?);
            return error.RoceConnectFailed;
        };
        logStripes(&proxy, ctx.?, group, hcas[0..nh], stripes);
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
                .spin_limit = spin_limit, .world = @intCast(world), .rank = @intCast(group.rank), .slots = @intCast(lay[6]), .hcas = stripes, .flag_stride = @intCast(lay[5]),
            },
        };
    }

    /// One line: the devices each peer is reached over.
    /// 1 行: 相手ごとに使うデバイス。
    fn logStripes(proxy: *const Proxy, ctx: *Ctx, group: *const cuda.tp.Group, hcas: []const [:0]const u8, stripes: u8) void {
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (0..group.world) |p| {
            if (p == group.rank) continue;
            w.print(" peer {d}:", .{p}) catch break;
            for (0..stripes) |s| {
                const h = proxy.stripe_hca(ctx, @intCast(p), @intCast(s));
                w.print(" {s}", .{if (h >= 0 and h < hcas.len) hcas[@intCast(h)] else if (h < 0) "relay" else "?"}) catch break;
            }
        }
        std.log.info("roce (world {d}, rank {d}): {d} HCAs a peer;{s}", .{ group.world, group.rank, stripes, w.buffered() });
    }

    /// Every rank's `local` bytes in rank order over NCCL (the setup exchange; staged through device memory).
    /// 全ランクの `local` のバイト列を NCCL でランク順に集める (準備用の交換。デバイスメモリを経由する)。
    pub fn exchange(gpa: std.mem.Allocator, d: *const cuda.Driver, group: *const cuda.tp.Group, local: []const u8, stream: cuda.Stream) ![]u8 {
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

    /// The proxy thread's id (0 before it runs).
    /// proxy のスレッドの id (動く前は 0)。
    pub fn proxyTid(r: *const Runtime) i32 {
        return r.proxy.tid(r.ctx);
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
        return r.allReducePf(s, in, out, nbytes, bf16, .{});
    }

    /// `allReduce`, prefetching `pf` into L2 while it waits on its peers (TF_GLM53_L2PF).
    /// peer を待つ間に `pf` を L2 に先読みする `allReduce` (TF_GLM53_L2PF)。
    pub fn allReducePf(r: *const Runtime, s: cuda.Stream, in: u64, out: u64, nbytes: u32, bf16: bool, pf: Prefetch) !void {
        if (nbytes % 16 != 0 or nbytes > max_bytes or in % 16 != 0 or out % 16 != 0) return error.RoceShape;
        const packs = nbytes / 16;
        const g = r.grid(packs);
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(c_int, @intCast(packs)));
        a.add(nbytes);
        a.add(g.reg);
        a.add(pf);
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
        return r.allGatherPf(s, in, out, nbytes, .{});
    }

    /// `allGather`, prefetching `pf` into L2 while it waits on its peers (TF_GLM53_L2PF).
    /// peer を待つ間に `pf` を L2 に先読みする `allGather` (TF_GLM53_L2PF)。
    pub fn allGatherPf(r: *const Runtime, s: cuda.Stream, in: u64, out: u64, nbytes: u32, pf: Prefetch) !void {
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
        a.add(pf);
        try cuda.launch.launch(r.ks.gather, .{ .grid = .{ .x = g.blocks, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 } }, s, &a);
    }

    /// False after a wait timed out (the control record's error word) or the proxy failed; logs which.
    /// 待ちがタイムアウトした (制御レコードの error 語) か proxy が失敗したら false。どちらかをログに出す。
    pub fn healthy(r: *const Runtime) bool {
        const ctrl: *const volatile [7]u32 = @ptrFromInt(r.reg.ctrl);
        if (ctrl[2] != 0) std.log.err("roce (world {d}, rank {d}): wait timed out at seq {d} on peer {d} stripe {d}", .{ r.reg.world, r.reg.rank, ctrl[2], ctrl[3], ctrl[6] });
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
