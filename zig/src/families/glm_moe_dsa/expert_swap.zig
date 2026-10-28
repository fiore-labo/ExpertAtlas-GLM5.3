// Fiore-lab: the routed experts' bits changed while the engine runs (TF_GLM53_EXPERT_BANKS), not in TensorFold 1.0.2.
//! Change routed experts' bits between requests. For each MoE layer whose target differs, each projection gets a new
//! slab: unchanged experts copied on the device, changed ones read from the bank of their new width (a checkpoint of
//! the same layout with every expert at that width). The device tables (ptr, k2) are rewritten in place - decode
//! graphs hold their addresses, not their entries - the changed experts' scales are overwritten, and the old slab is
//! freed. Every rank runs the same table at the same point (rank 0 shares it: lockstep.swap); cached prefixes are then
//! dropped, being computed with the old bits.
//! routed expert の bit をリクエストの合間に変える。目標が違う MoE 層ごとに、各射影に新しい置き場を作る: 変わらない
//! expert はデバイス上で写し、変わる expert は新しい幅のバンク (全 expert がその幅の、同じ並びのチェックポイント) から
//! 読む。デバイスの表 (ptr, k2) はその場で書き換え (decode のグラフが持つのは表のアドレスで中身ではない)、変わる expert
//! のスケールを上書きして、古い置き場を解放する。全ランクが同じ表を同じ時点で実行する (rank 0 が配る:
//! lockstep.swap)。その後、古い bit で計算した保存済みの prefix は捨てる。
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const moe = @import("moe.zig");
const memory_live = @import("memory_live.zig");
const prompt_experts = @import("prompt_experts.zig");
const experts = @import("experts.zig");
const Engine = @import("engine.zig").Engine;

const Buf = cuda.DeviceBuffer;
pub const max_bits = 8;

/// TF_GLM53_EXPERT_BANKS=3:/dir3,4:/dir4 - a checkpoint directory for each width.
/// TF_GLM53_EXPERT_BANKS=3:/dir3,4:/dir4 - 幅ごとのチェックポイントのディレクトリ。
pub const Banks = struct {
    dirs: [max_bits + 1]?[]const u8 = @splat(null),

    pub fn parse(gpa: std.mem.Allocator, spec: []const u8) !Banks {
        var b: Banks = .{};
        var it = std.mem.tokenizeScalar(u8, spec, ',');
        while (it.next()) |item| {
            const colon = std.mem.indexOfScalar(u8, item, ':') orelse return error.BadExpertBanks;
            const bits = std.fmt.parseInt(u8, item[0..colon], 10) catch return error.BadExpertBanks;
            if (bits < 2 or bits > max_bits) return error.BadExpertBanks;
            b.dirs[bits] = try gpa.dupe(u8, item[colon + 1 ..]);
        }
        return b;
    }
};

/// A saved set's bits (glm53-expert-bits v1: POST /v1/experts/save), checked against the engine's MoE layers and banks.
/// 保存したセットの bit (glm53-expert-bits v1: POST /v1/experts/save)。エンジンの MoE 層とバンクに照らして確かめる。
pub fn readSet(gpa: std.mem.Allocator, io: std.Io, path: []const u8, e: *const Engine) ![]u8 {
    if (e.banks == null) return error.ExpertBitsNeedBanks;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const Set = struct { format: []const u8, version: u32, first_layer: usize, layers: usize, experts: usize, bits: []const u8 };
    const parsed = try std.json.parseFromSlice(Set, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const v = parsed.value;
    if (!std.mem.eql(u8, v.format, "glm53-expert-bits") or v.version != 1) return error.ExpertBitsFormat;
    if (v.first_layer != e.xfirst or v.experts != e.cfg.experts or v.layers * v.experts != e.xbits.len) return error.ExpertBitsShape;
    const dec = std.base64.standard.Decoder;
    const bits = try gpa.alloc(u8, try dec.calcSizeForSlice(v.bits));
    errdefer gpa.free(bits);
    try dec.decode(bits, v.bits);
    if (bits.len != e.xbits.len) return error.ExpertBitsShape;
    return bits;
}

/// `alike`: the ranks stopped in the same state (a refusal before anything changed, or a lack of memory agreed on
/// before a layer changed), so serving may go on; false when a rank may differ (a failure inside a layer's change,
/// a failed agreement or cache drop): the caller stops every rank.
/// `alike`: 全ランクが同じ状態で止まった (何も変える前の拒否か、層を変える前に合意したメモリ不足) ので、続けて
/// 推論してよい。false はランクが食い違いうる (層の変更中の失敗、合意や cache の破棄の失敗): 呼ぶ側が全ランクを止める。
pub const Stats = struct { layers: u32 = 0, experts: u32 = 0, up: u32 = 0, down: u32 = 0, read_bytes: u64 = 0, ms: u64 = 0, alike: bool = true };

/// Every rank's agreement at a layer boundary (lockstep: Channel.allOk): true when every rank passed ok.
/// 層の境目での全ランクの合意 (lockstep: Channel.allOk): 全ランクが ok を渡せば true。
pub const Sync = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
};

/// Called after each MoE layer (rank 0: the panel shows the progress).
/// MoE 層ごとに呼ばれる (rank 0: パネルが進み具合を出す)。
pub const Progress = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, done: u32, layers: u32) void,
};

/// The checks every rank makes before any rank changes anything: the table's size, a bank for every width it asks
/// for, the widths the kernels take (the prompt kernel: 2, 3, 4 bits; 5 with TF_GLM53_PE_K10=1), with an expert plan
/// (TF_GLM53_EXPERT_PLAN) its bank of each new width being that width's bank, and no new width in a layer whose decode
/// graphs run the all-4-bit kernel.
/// どのランクも何かを変える前に行う確認: 表の大きさ、求める幅すべてのバンク、カーネルが受ける幅 (prompt カーネル:
/// 2・3・4 bit。TF_GLM53_PE_K10=1 なら 5 も)、計画 (TF_GLM53_EXPERT_PLAN) で起動したときは新しい幅の計画のバンクが
/// その幅のバンクと同じこと、全 4 bit 用カーネルで decode グラフが動く層に新しい幅を入れないこと。
pub fn check(e: *const Engine, target: []const u8) !void {
    const E = e.cfg.experts;
    const banks = e.banks orelse return error.SwapNoBanks;
    if (target.len != e.xbits.len) return error.SwapTableSize;
    const widest: u8 = if (prompt_experts.k10Enabled()) 5 else 4;
    for (target, e.xbits) |t, c| {
        if (t == c) continue;
        if (t < 2 or t > widest) return error.SwapWidthUnsupported;
        const dir = banks.dirs[t] orelse return error.SwapNoBank;
        // a plan names its banks by id; a swap reads a width from its one bank: the two must be the same directory
        // 計画はバンクを ID で指す。入れ替えは幅ごとに 1 つのバンクから読む: 両者は同じディレクトリでなければならない
        if (e.plan) |*p| for (p.banks) |b| if (b.bits == t and !sameDir(b.dir, dir)) return error.SwapPlanBankDiffers;
    }
    for (e.layers) |*L| if (L.experts) |*x| {
        const li = L.index - e.xfirst;
        const tgt = target[li * E ..][0..E];
        if (std.mem.eql(u8, tgt, e.xbits[li * E ..][0..E])) continue;
        // the all-4-bit decode kernel traps on any other K2: such a layer stays all 4 bit
        // 全 4 bit 用の decode カーネルは他の K2 で止まる: そういう層は全 4 bit のまま
        // (TF_GLM53_EXPERTS_GENERIC=1: no layer runs it, experts.genericOnly)
        // (TF_GLM53_EXPERTS_GENERIC=1 ならどの層も使わない、experts.genericOnly)
        if (!experts.genericOnly() and ((x.k2_gu[0] == 8 and x.k2_gu[1] == 8) or (x.k2_d[0] == 8 and x.k2_d[1] == 8))) for (tgt) |t| if (t != 4) return error.SwapLayerAll4Bit;
        // the layer's file in each width it takes experts from: present on this rank. A missing one refuses the change
        // before anything changes, on every rank (lockstep.swapEvery agrees on the check), so the engine goes on
        // serving
        // その層が expert を取る各幅の層ファイルが、このランクにあること。無ければ何も変える前に全ランクで断る
        // (lockstep.swapEvery が確認に合意する)。エンジンはそのまま推論を続ける
        var need: [max_bits + 1]bool = @splat(false);
        for (tgt, e.xbits[li * E ..][0..E]) |t, c| if (t != c) {
            need[t] = true;
        };
        for (need, 0..) |n, t| if (n) {
            var buf: [4096]u8 = undefined;
            const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/experts-layer-{d:0>3}.safetensors", .{ banks.dirs[t].?, L.index }) catch return error.SwapBankFileMissing;
            buf[s.len] = 0;
            const path: [:0]const u8 = buf[0..s.len :0];
            if (std.c.access(path.ptr, 0) != 0) {
                std.log.err("glm53: rank {d}: the {d}-bit bank has no {s}: the change is refused, nothing changed", .{ e.rank, t, path });
                return error.SwapBankFileMissing;
            }
        };
    };
    try checkMemory(e, target);
}

fn sameDir(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, std.mem.trimEnd(u8, a, "/"), std.mem.trimEnd(u8, b, "/"));
}

/// Each layer, in the order `apply` takes them, must fit in MemAvailable with TF_GLM53_SWAP_MARGIN_GIB (default 2) to
/// spare: its new slabs (made beside the old ones) and its reads on the host, on top of the growth of the layers changed
/// before it. On GB10 a device allocation past it livelocks the node. (The net growth alone is not enough: a re-sort
/// that keeps every width's count can grow its first layers by most of a GiB a rank and shrink the later ones.)
/// `apply` の順で各層が、TF_GLM53_SWAP_MARGIN_GIB (既定 2) を残して MemAvailable に収まること: その層の新しい置き場
/// (古いものと並べて作る) とホスト上の読み込みを、それより前に変えた層で増えた分の上に。GB10 ではそれを超えるデバイスの
/// 確保でノードがライブロックする。(差し引きの増分だけでは足りない: 幅ごとの数を保つ並べ替えでも、前の層で 1 ランク
/// 1 GiB 近く増え、後の層で減ることがある。)
fn checkMemory(e: *const Engine, target: []const u8) !void {
    const E = e.cfg.experts;
    var peak: u64 = 0;
    var peak_layer: usize = 0;
    var peak_slabs: u64 = 0;
    var peak_grown: u64 = 0;
    var grow: i64 = 0;
    for (0..2) |pass| for (e.layers) |*L| if (L.experts) |*x| {
        const li = L.index - e.xfirst;
        const tgt = target[li * E ..][0..E];
        const cur = e.xbits[li * E ..][0..E];
        if (std.mem.eql(u8, tgt, cur) or swapPass(tgt, cur) != pass) continue;
        const slabs = layerNeed(x, tgt, cur);
        const grown: u64 = @intCast(@max(grow, 0));
        if (grown + slabs > peak) {
            peak = grown + slabs;
            peak_layer = L.index;
            peak_slabs = slabs;
            peak_grown = grown;
        }
        const unit = expertUnit(x);
        for (tgt, cur) |t, c| grow += @as(i64, @intCast(unit * t)) - @as(i64, @intCast(unit * c));
    };
    const need = peak + marginBytes();
    const avail = availBytes(e.io);
    if (avail < need) {
        std.log.err("glm53: rank {d}: expert swap needs {d} MiB available (layer {d}: its slabs {d} MiB after a growth of {d} MiB, margin {d} MiB), MemAvailable {d} MiB", .{ e.rank, need >> 20, peak_layer, peak_slabs >> 20, peak_grown >> 20, marginBytes() >> 20, avail >> 20 });
        return error.SwapLowMemory;
    }
}

/// The pass of `apply` a changed layer goes in: 0 for one whose bits do not grow (its old slabs freed early), 1 for one
/// that grows. By the bits, not the bytes, so every rank takes the layers in the same order.
/// `apply` で変わる層を回すパス: bit が増えない層は 0 (古い置き場を先に解放する)、増える層は 1。バイトでなく bit で
/// 決めるので、どのランクも同じ順で層を回す。
fn swapPass(tgt: []const u8, cur: []const u8) usize {
    var d: i64 = 0;
    for (tgt, cur) |t, c| d += @as(i64, t) - @as(i64, c);
    return @intFromBool(d > 0);
}

/// One expert's bytes per bit on this rank: three projections of dims x width, K2 = 2 x bits, dims x width / 16 x K2.
/// このランクの 1 expert の 1 bit あたりのバイト: dims x width の射影 3 つ、K2 = 2 x bit、dims x width / 16 x K2。
fn expertUnit(x: *const moe.Routed) u64 {
    return 3 * @as(u64, x.dims) * x.width / 16 * 2;
}

/// A layer's new slabs (made beside the old ones) and its reads on the host.
/// 1 層の新しい置き場 (古いものと並べて作る) とホスト上の読み込み。
fn layerNeed(x: *const moe.Routed, tgt: []const u8, cur: []const u8) u64 {
    const unit = expertUnit(x);
    var n: u64 = 0;
    for (tgt, cur) |t, c| n += unit * t * @as(u64, if (t != c) 2 else 1);
    return n;
}

fn marginBytes() u64 {
    const gib: u64 = if (std.c.getenv("TF_GLM53_SWAP_MARGIN_GIB")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 2 else 2;
    return gib << 30;
}

fn availBytes(io: std.Io) u64 {
    var v: [1]u64 = undefined;
    memory_live.MemLive.meminfo(io, &.{"MemAvailable:"}, &v);
    return v[0];
}

/// Whether MemAvailable reaches `need` within `ms` (polled every 20 ms).
/// MemAvailable が `ms` 以内に `need` に届くか (20 ms ごとに見る)。
fn waitAvail(io: std.Io, need: u64, ms: u32) bool {
    var waited: u32 = 0;
    while (availBytes(io) < need) : (waited += 20) {
        if (waited >= ms) return false;
        std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
    }
    return true;
}

/// Every MoE layer to `target` (one byte an expert, the MoE layers in order), `check`ed first by the caller (on every
/// rank, agreed: a second check here could differ between ranks and leave one out of the agreements below). The
/// engine must be idle.
/// 全 MoE 層を `target` にする (expert ごとに 1 バイト、MoE 層の順)。呼び出し側が先に `check` すること (全ランクで、
/// 合意して: ここでもう一度確かめるとランクごとに結果が分かれ、下の合意から外れるランクが出うる)。エンジンは止まっていること。
/// With `sync`, every rank waits for its memory and agrees before each changed layer, and agrees after it: a rank short
/// of memory (or failing) stops every rank at the same layer, the layers before it changed on all.
/// `sync` があれば、変わる層ごとに全ランクがメモリを待って合意し、終えた後にも合意する: メモリが足りない (または
/// 失敗した) ランクがあれば全ランクが同じ層で止まり、それより前の層は全ランクで変わっている。
/// `st` counts what changed, also when it stops part-way. A layer's experts are read from the banks' file for it,
/// `experts-layer-NNN.safetensors` (the layout of these checkpoints), and the file is closed after the layer.
/// `st` は変わったものを数える (途中で止まった場合も)。1 層の expert はバンクのその層のファイル
/// `experts-layer-NNN.safetensors` (これらのチェックポイントの並び) から読み、層を終えたら閉じる。
pub fn apply(e: *Engine, target: []const u8, progress: ?Progress, sync: ?Sync, st: *Stats) !void {
    const E = e.cfg.experts;
    const margin = marginBytes();
    const trace = if (std.c.getenv("TF_GLM53_SWAP_TRACE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const t0 = std.Io.Clock.awake.now(e.io);
    try e.stream.synchronize(); // nothing in flight reads an old slab / 古い置き場を読む処理が残っていない
    var readers: [max_bits + 1]?weights.Reader = @splat(null);
    defer closeReaders(&readers);
    defer st.ms = @intCast(@max(0, t0.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds()));
    const n_moe: u32 = @intCast(e.xbits.len / E);
    var done: u32 = 0;
    // the layers that do not grow first, then the ones that grow (swapPass): the memory a re-sort frees comes before
    // the memory it takes
    // 増えない層を先に、増える層を後に (swapPass): 並べ替えで空くメモリを、使うメモリより先に
    for (0..2) |pass| for (e.layers) |*L| {
        const x = if (L.experts) |*xx| xx else continue;
        const li = L.index - e.xfirst;
        const tgt = target[li * E ..][0..E];
        const cur = e.xbits[li * E ..][0..E];
        if (swapPass(tgt, cur) != pass) continue;
        if (!std.mem.eql(u8, tgt, cur)) {
            // the old slabs' memory comes back to GB10 some time after they are freed: wait for this layer's
            // 古い置き場のメモリは解放の少し後で GB10 に戻る: この層の分を待つ
            const need = layerNeed(x, tgt, cur) + margin;
            const a0 = availBytes(e.io);
            const ready = waitAvail(e.io, need, 10_000);
            const a1 = availBytes(e.io);
            if (sync) |s| {
                if (!try s.f(s.ctx, ready)) return if (ready) error.SwapLowMemoryOnAnotherRank else error.SwapLowMemory;
            } else if (!ready) return error.SwapLowMemory;
            var why: ?anyerror = null;
            swapLayer(e, x, L.index, tgt, cur, &readers, st) catch |err| {
                why = err;
            };
            // the banks' files closed and out of the page cache before the next layer (GB10: it competes with the
            // device memory)
            // 次の層の前にバンクのファイルを閉じ、ページキャッシュから落とす (GB10 ではデバイスのメモリと取り合う)
            closeReaders(&readers);
            if (sync) |s| {
                if (!try s.f(s.ctx, why == null)) return why orelse error.SwapFailedOnAnotherRank;
            } else if (why) |err| return err;
            if (trace) std.log.info("glm53: rank {d}: swap layer {d}: need {d} MiB, MemAvailable {d} -> {d} MiB (waited), {d} MiB after", .{ e.rank, L.index, need >> 20, a0 >> 20, a1 >> 20, availBytes(e.io) >> 20 });
            @memcpy(cur, tgt);
            st.layers += 1;
        }
        done += 1;
        if (progress) |p| p.f(p.ctx, done, n_moe);
    };
}

fn closeReaders(readers: *[max_bits + 1]?weights.Reader) void {
    for (readers) |*r| if (r.*) |*rr| {
        rr.release();
        rr.close();
        r.* = null;
    };
}

fn alignUp(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}

fn reader(e: *Engine, readers: *[max_bits + 1]?weights.Reader, bits: u8, layer: usize) !*weights.Reader {
    if (readers[bits] == null) {
        var b: [64]u8 = undefined;
        const fname = try std.fmt.bufPrint(&b, "experts-layer-{d:0>3}.safetensors", .{layer});
        readers[bits] = try weights.Reader.openFile(e.gpa, e.io, e.banks.?.dirs[bits].?, fname, e.rank, e.world, e.stored_world);
    }
    return &readers[bits].?;
}

/// One expert's three projections from a bank, read and checked on the host.
/// バンクから読んで確かめた 1 expert の 3 射影 (ホスト上)。
const Read = struct {
    tr: [3]weights.Share,
    su: [3]weights.Share,
    sv: [3]weights.Share,

    fn deinit(r: *Read, gpa: std.mem.Allocator) void {
        for (0..3) |p| {
            r.tr[p].deinit(gpa);
            r.su[p].deinit(gpa);
            r.sv[p].deinit(gpa);
        }
    }
};

/// The codebook of expert `j`'s three projections in the bank (0 none, 1 mcg, 2 mul1); the projections must agree, as
/// the layer's kernels decode them all with one (the plan load checks every projection as well).
/// バンクにある expert `j` の 3 つの射影の codebook (0 なし、1 mcg、2 mul1)。層のカーネルは 1 つで全部を復号するので
/// 射影どうしが一致すること (計画での読み込みも全射影を確かめる)。
fn bankCb(r: *weights.Reader, layer: usize, j: usize) !u32 {
    var b: [192]u8 = undefined;
    var cb: ?u32 = null;
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        const v: u32 = if (r.has(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.experts.{d}.{s}_proj.mcg", .{ layer, j, proj }))) 1 else if (r.has(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.experts.{d}.{s}_proj.mul1", .{ layer, j, proj }))) 2 else 0;
        if (cb) |c| if (c != v) return error.SwapCodebookDiffers;
        cb = v;
    }
    return cb.?;
}

/// One layer in three steps, so that a failure leaves it as it was: the changed experts read and checked, the three
/// new slabs allocated, then the device written (slabs, scales, tables) and the old slabs freed.
/// 1 層を 3 段で行い、失敗してもその層は元のまま: 変わる expert を読んで確かめ、新しい置き場 3 つを確保し、それから
/// デバイスに書いて (置き場・スケール・表) 古い置き場を解放する。
fn swapLayer(e: *Engine, x: *moe.Routed, layer: usize, tgt: []const u8, cur: []const u8, readers: *[max_bits + 1]?weights.Reader, st: *Stats) !void {
    const gpa = e.gpa;
    const E = tgt.len;
    const projs = [_][]const u8{ "gate", "up", "down" };
    // one expert's trellis on this rank: (dims / 16) x (width / 16) tiles of K2 x 8 i16 each
    // このランクの 1 expert の trellis: (dims / 16) x (width / 16) タイル、各 K2 x 8 個の i16
    const per_k2: usize = @as(usize, x.dims) * x.width / 16;
    const scale_bufs = [3][2]Buf{ .{ x.suh_g, x.svh_g }, .{ x.suh_u, x.svh_u }, .{ x.suh_d, x.svh_d } };
    var name: [192]u8 = undefined;

    // 1. read and check / 読んで確かめる
    const reads = try gpa.alloc(?Read, E);
    @memset(reads, null);
    defer {
        for (reads) |*r| if (r.*) |*rr| rr.deinit(gpa);
        gpa.free(reads);
    }
    for (0..E) |j| if (tgt[j] != cur[j]) {
        const r = try reader(e, readers, tgt[j], layer);
        if (try bankCb(r, layer, j) != x.cb) return error.SwapCodebookDiffers;
        const k2 = @as(usize, tgt[j]) * 2;
        var rd: Read = undefined;
        var got: usize = 0; // projections read so far / 読み終えた射影の数
        errdefer for (0..got) |p| {
            rd.tr[p].deinit(gpa);
            rd.su[p].deinit(gpa);
            rd.sv[p].deinit(gpa);
        };
        for (projs, 0..) |proj, p| {
            var tr = try r.get(try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.experts.{d}.{s}_proj.trellis", .{ layer, j, proj }));
            errdefer tr.deinit(gpa);
            var su = try r.get(try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.experts.{d}.{s}_proj.suh", .{ layer, j, proj }));
            errdefer su.deinit(gpa);
            var sv = try r.get(try std.fmt.bufPrint(&name, "model.layers.{d}.mlp.experts.{d}.{s}_proj.svh", .{ layer, j, proj }));
            errdefer sv.deinit(gpa);
            const dims: usize = if (p == 2) x.width else x.dims;
            const width: usize = if (p == 2) x.dims else x.width;
            if (tr.dtype != .i16 or tr.rank != 3 or tr.shape[0] * 16 != dims or tr.shape[1] * 16 != width or tr.shape[2] != k2 * 8) return error.SwapBadTrellis;
            if (su.dtype != .f16 or sv.dtype != .f16 or su.bytes.len * E != scale_bufs[p][0].len or sv.bytes.len * E != scale_bufs[p][1].len) return error.SwapBadScales;
            rd.tr[p] = tr;
            rd.su[p] = su;
            rd.sv[p] = sv;
            got += 1;
        }
        reads[j] = rd;
    };

    // 2. the new slabs / 新しい置き場
    var old_ptrs: [3][]i64 = undefined;
    var old_k2: [3][]i32 = undefined;
    var n_tbl: usize = 0;
    defer for (0..n_tbl) |p| {
        gpa.free(old_ptrs[p]);
        gpa.free(old_k2[p]);
    };
    for (0..3) |p| {
        old_ptrs[p] = try gpa.alloc(i64, E);
        old_k2[p] = gpa.alloc(i32, E) catch |err| {
            gpa.free(old_ptrs[p]);
            return err;
        };
        n_tbl += 1;
        try x.ptr[p].download(0, std.mem.sliceAsBytes(old_ptrs[p]));
        try x.k2[p].download(0, std.mem.sliceAsBytes(old_k2[p]));
    }
    var nb: [3]Buf = undefined;
    var n_nb: usize = 0;
    errdefer for (nb[0..n_nb]) |*b| b.free();
    for (0..3) |p| {
        var total: usize = 0;
        for (0..E) |j| total += alignUp(if (reads[j]) |rd| rd.tr[p].bytes.len else per_k2 * @as(usize, @intCast(old_k2[p][j])));
        nb[p] = try Buf.alloc(e.d, total);
        n_nb += 1;
    }

    // 3. the device / デバイス
    const ptrs = try gpa.alloc(i64, E);
    defer gpa.free(ptrs);
    const k2v = try gpa.alloc(i32, E);
    defer gpa.free(k2v);
    for (0..3) |p| {
        var lo: u32 = 99;
        var hi: u32 = 0;
        var off: usize = 0;
        for (0..E) |j| {
            if (reads[j]) |rd| {
                try nb[p].upload(off, rd.tr[p].bytes);
                try scale_bufs[p][0].upload(j * rd.su[p].bytes.len, rd.su[p].bytes);
                try scale_bufs[p][1].upload(j * rd.sv[p].bytes.len, rd.sv[p].bytes);
                k2v[j] = @intCast(rd.tr[p].shape[2] / 8);
                ptrs[j] = @intCast(nb[p].ptr + off);
                off += alignUp(rd.tr[p].bytes.len);
                st.read_bytes += rd.tr[p].bytes.len + rd.su[p].bytes.len + rd.sv[p].bytes.len;
            } else {
                const n = per_k2 * @as(usize, @intCast(old_k2[p][j]));
                try nb[p].copyFrom(off, @intCast(old_ptrs[p][j]), n, null);
                k2v[j] = old_k2[p][j];
                ptrs[j] = @intCast(nb[p].ptr + off);
                off += alignUp(n);
            }
            lo = @min(lo, @as(u32, @intCast(k2v[j])));
            hi = @max(hi, @as(u32, @intCast(k2v[j])));
        }
        // the tables in place: the graphs hold their addresses / 表はその場で: グラフが持つのはそのアドレス
        try x.ptr[p].upload(0, std.mem.sliceAsBytes(ptrs));
        try x.k2[p].upload(0, std.mem.sliceAsBytes(k2v));
        // the K2 ranges only widen: a captured graph keeps the kernel chosen from the range at capture (check keeps
        // the all-4-bit kernel's layers all 4 bit, every other range takes 2 .. 5 bits)
        // K2 の範囲は広がるだけ: capture 済みのグラフは capture 時の範囲で選んだカーネルのまま (check が全 4 bit
        // カーネルの層を全 4 bit に保ち、それ以外の範囲は 2 .. 5 bit を受ける)
        const rg = if (p == 2) &x.k2_d else &x.k2_gu;
        rg.* = .{ @min(rg[0], lo), @max(rg[1], hi) };
    }
    for (0..3) |p| {
        x.trellis[p].free();
        x.trellis[p] = nb[p];
    }
    n_nb = 0;
    for (0..E) |j| if (tgt[j] != cur[j]) {
        st.experts += 1;
        if (tgt[j] > cur[j]) st.up += 1 else st.down += 1;
    };
}
