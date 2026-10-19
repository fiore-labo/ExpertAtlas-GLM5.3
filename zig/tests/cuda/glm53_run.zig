//! `tf-glm53-run RANK WORLD MASTER CKPT STORED_WORLD LAYERS - PROMPT_LEN SEED OUT` (one process a GPU): a seeded prompt
//! through the first LAYERS layers, then greedy decode, each layer's attention / ffn / residual and the head hashed for
//! comparison with a Python reference implementation, which this repository does not include.
//! 1 GPU に 1 プロセス。seed から作った prompt を先頭 LAYERS 層に通して greedy に decode し、各層の attention / ffn /
//! residual と head をハッシュする。Python の参照実装と比べるためのもので、参照実装はこのリポジトリに含まない。
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");

const Probe = struct {
    gpa: std.mem.Allocator,
    lines: std.ArrayList(u8) = .empty,
    n: usize = 0,

    fn hook(ctx: *anyopaque, layer: usize, what: []const u8, rows: u32, ptr: u64, bytes: usize) anyerror!void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const host = try p.gpa.alloc(u8, bytes);
        defer p.gpa.free(host);
        try copyDown(host, ptr);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        try p.lines.print(p.gpa, "{s}\"L{d}:{s}:{d}:{s}\"", .{ if (p.n == 0) "" else ",\n", layer, what, rows, hex[0..24] });
        p.n += 1;
    }
};

var g_driver: *const cuda.Driver = undefined;

fn copyDown(out: []u8, ptr: u64) !void {
    try g_driver.check(g_driver.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "cuMemcpyDtoH");
}

/// TF_ZIG_TOKENS (default 32): tokens to generate; TF_ZIG_PROBE=0: no hashes (timing runs).
/// TF_ZIG_TOKENS (既定 32): 生成するトークン数。TF_ZIG_PROBE=0: ハッシュを取らない (時間計測用)。
fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

/// Stop before the unified memory runs out (an OOM livelock needs a cold power cycle): MemAvailable under 8 GiB.
/// unified memory が尽きる前に止める (OOM のライブロックは電源の入れ直しが要る): MemAvailable が 8 GiB 未満で。
fn memCheck(io: std.Io, layer: usize) !void {
    var buf: [4096]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf);
    const at = std.mem.indexOf(u8, text, "MemAvailable:") orelse return;
    var it = std.mem.tokenizeAny(u8, text[at + 13 ..], " \n");
    const kib = try std.fmt.parseInt(u64, it.next().?, 10);
    if (kib < 8 << 20) {
        std.debug.print("[zig] MemAvailable {d} MiB after layer {d}: stopping\n", .{ kib >> 10, layer });
        return error.LowMemory;
    }
}

/// The indexer's key range for a window ending at position t - 1 (null while t <= topk, else a power of two).
/// 位置 t - 1 で終わる window の indexer のキー範囲 (t <= topk の間は null、それ以外は 2 のべき)。
fn bucket(t: usize, topk: usize) ?u32 {
    if (t <= topk) return null;
    const p2 = std.math.ceilPowerOfTwo(usize, t) catch unreachable;
    return @intCast(@max(2 * topk, p2));
}

/// The head hash: sha256 of the head's input row (bf16) then this rank's logits share (fp32), "mode:1:hex".
/// head のハッシュ: head の入力行 (bf16)、続いてこのランクの logits の分 (fp32) の sha256。形式は "mode:1:hex"。
fn headHash(a: std.mem.Allocator, out: *std.ArrayList(u8), mode: []const u8, x_row: u64, b: *glm.forward.Buffers, cfg: glm.config.Config, vshare: u32, stream: cuda.Stream) !void {
    // the copies below run on the legacy stream: wait for the head's launches first
    // 下のコピーは legacy stream で走るので、先に head の起動を待つ
    try stream.synchronize();
    const xb = try a.alloc(u8, cfg.hidden * 2);
    const lb = try a.alloc(u8, @as(usize, vshare) * 4);
    try copyDown(xb, x_row);
    try b.lpart.download(0, lb);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(xb);
    h.update(lb);
    var d: [32]u8 = undefined;
    h.final(&d);
    var dx: [32]u8 = undefined;
    var dl: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(xb, &dx, .{});
    std.crypto.hash.sha2.Sha256.hash(lb, &dl, .{});
    const hx = std.fmt.bytesToHex(dx, .lower);
    const hl = std.fmt.bytesToHex(dl, .lower);
    try out.print(a, "{s}\"{s}:1:{s}:x={s}:l={s}\"", .{ if (out.items.len == 0) "" else ",\n", mode, std.fmt.bytesToHex(d, .lower), hx[0..16], hl[0..16] });
}

/// TF_ZIG_MICRO: a graph of 2000 one-thread kernels replayed 20 times; prints microseconds a kernel
/// (launch gap + a tiny body).
/// TF_ZIG_MICRO: 1 スレッドのカーネル 2000 個の graph を 20 回再生し、カーネル 1 つあたりのマイクロ秒を出す
/// (起動の間隔 + 小さな本体)。
fn micro(driver: *const cuda.Driver, rank: usize, label: []const u8) !void {
    if (envInt("TF_ZIG_MICRO", 0) == 0) return;
    var s = try cuda.Stream.init(driver, true);
    defer s.deinit();
    var glue = try glm.forward.Glue.load(driver);
    const buf = try cuda.DeviceBuffer.alloc(driver, 256);
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..2000) |i| try glue.setI32(s, buf.ptr, @intCast(i));
    var g = try cuda.graph.endCapture(s);
    defer g.deinit();
    var e = try g.instantiate();
    defer e.deinit();
    for (0..3) |_| try e.launchOn(s);
    try s.synchronize();
    var t0 = try cuda.Event.init(driver, true);
    var t1 = try cuda.Event.init(driver, true);
    try t0.record(s);
    for (0..20) |_| try e.launchOn(s);
    try t1.record(s);
    try s.synchronize();
    const ms = try cuda.Event.elapsedMs(t0, t1);
    std.debug.print("[zig] rank {d}: micro {s}: {d:.3} us a kernel\n", .{ rank, label, ms * 1000.0 / 40000.0 });
}

/// Prompt ids: splitmix64 draws of `seed`, ids in [1000, 150000).
/// prompt の id: `seed` からの splitmix64 乱数で、id は [1000, 150000)。
fn promptIds(out: []i64, seed: u64) void {
    const golden: u64 = 0x9E3779B97F4A7C15;
    for (out, 0..) |*v, i| {
        var z = seed +% (@as(u64, i) +% 1) *% golden;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z ^= z >> 31;
        v.* = @intCast(z % 149000 + 1000);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 11) {
        std.debug.print("usage: tf-glm53-run RANK WORLD MASTER CKPT STORED_WORLD LAYERS UNUSED PROMPT_LEN SEED OUT\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const ckpt = args[4];
    const stored = try std.fmt.parseInt(usize, args[5], 10);
    const nlayers = try std.fmt.parseInt(usize, args[6], 10);
    const prompt_len = try std.fmt.parseInt(usize, args[8], 10);
    const seed = try std.fmt.parseInt(u64, args[9], 10);

    var driver = try cuda.Driver.open();
    defer driver.close();
    g_driver = &driver;
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (std.c.getenv("TF_ZIG_TOPK")) |v| glm.forward.topk_select = !std.mem.eql(u8, std.mem.span(v), "sort");
    try micro(&driver, rank, "start");
    var nccl = try cuda.nccl.Library.openEnv();
    defer nccl.close();
    var group = try cuda.tp.Group.init(&nccl, rank, world, try cuda.tp.parseMaster(args[3]));
    defer group.deinit();
    try micro(&driver, rank, "nccl");
    var blas = try cuda.cublas.Library.open();
    defer blas.close();
    var handle: cuda.cublas.Handle = null;
    try blas.check(blas.api.cublasCreate_v2(&handle), "cublasCreate");
    const ws = try cuda.DeviceBuffer.alloc(&driver, cuda.cublas.torch_workspace);
    try blas.check(blas.api.cublasSetWorkspace_v2(handle, ws.ptr, cuda.cublas.torch_workspace), "cublasSetWorkspace");
    var stream = try cuda.Stream.init(&driver, true);
    try blas.check(blas.api.cublasSetStream_v2(handle, stream.handle), "cublasSetStream");

    const cfg = try glm.config.Config.read(a, io, ckpt);
    const heads: u32 = @intCast(cfg.heads / world);
    var reader = try glm.weights.Reader.open(gpa, io, ckpt, rank, world, stored);
    defer reader.close();
    const t0 = std.Io.Clock.awake.now(io);
    const layers = try a.alloc(glm.model.Layer, nlayers);
    for (layers, 0..) |*L, i| {
        L.* = try glm.model.loadLayer(gpa, &driver, &reader, cfg, i, heads);
        if (cfg.isMoe(i)) L.experts = try glm.moe.load(gpa, &driver, &reader, cfg, i);
        reader.release();
        try memCheck(io, i);
    }
    var embed_share = try reader.get("model.embed_tokens.weight");
    var embed = try cuda.DeviceBuffer.fromHost(&driver, embed_share.bytes);
    embed_share.deinit(gpa);
    reader.release();
    std.debug.print("[zig] rank {d}/{d}: {d} layers + embed loaded in {d} ms\n", .{ rank, world, nlayers, t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });

    var nat = try glm.native_ops.Kernels.load(&driver);
    var x3 = try glm.exl3.Kernels.load(&driver);
    defer x3.deinit();
    var glue = try glm.forward.Glue.load(&driver);
    _ = &glue;
    const inv = try glm.rope.table(cfg.rope_theta, cfg.qk_rope);
    const had = glm.rope.hadamard();
    var lm = try reader.get("lm_head.weight");
    const vshare: u32 = @intCast(lm.shape[0]);
    var lm_head = try cuda.DeviceBuffer.fromHost(&driver, lm.bytes);
    lm.deinit(gpa);
    var fnorm = try reader.get("model.norm.weight");
    var final_norm = try cuda.DeviceBuffer.fromHost(&driver, fnorm.bytes);
    fnorm.deinit(gpa);
    reader.release();
    _ = &lm_head;
    _ = &final_norm;
    var bufs = try glm.forward.Buffers.init(&driver, cfg, heads, world, glm.forward.max_rows, vshare, &inv, &had, glm.forward.max_keys, 1, true, glm.forward.max_rows);
    // the decode buffers: windows of 1 row
    // decode 用のバッファ: 1 行の window
    var vbufs = try glm.forward.Buffers.init(&driver, cfg, heads, world, 1, vshare, &inv, &had, glm.forward.max_keys, 1, true, 1);
    var exk = try glm.experts.Kernels.load(&driver, @intCast(try ctx.attribute(.max_shared_memory_per_block_optin)));
    defer exk.deinit();
    var first_moe: ?*const glm.moe.Routed = null;
    for (layers) |*L| if (L.experts) |*e| {
        first_moe = e;
        break;
    };
    // the experts' scratch, sized from the first MoE layer (prompt rows x top-k slots)
    // expert の scratch。最初の MoE 層から大きさを決める (prompt 行数 x top-k スロット)
    const xs = if (first_moe) |e| try glm.moe.Scratch.init(&driver, e, glm.forward.max_rows, @intCast(cfg.top_k)) else undefined;
    const vxs = if (first_moe) |e| try glm.moe.Scratch.init(&driver, e, 1, @intCast(cfg.top_k)) else undefined;
    var state = try glm.forward.State.init(gpa, &driver, cfg, layers, 8192, false, false);

    try micro(&driver, rank, "loaded");
    // the init-time legacy-stream memsets (caches, scratch) land before the non-blocking stream reads them
    // 初期化時の legacy stream の memset (キャッシュ、scratch) を、non-blocking stream が読む前に終わらせる
    try ctx.synchronize();
    var probe: Probe = .{ .gpa = gpa };
    const probing = envInt("TF_ZIG_PROBE", 1) == 1;
    const max_tokens = envInt("TF_ZIG_TOKENS", 32);
    const fw: glm.forward.Ctx = .{
        .cfg = cfg, .heads = heads, .world = world, .rank = rank, .stream = stream,
        .ops = .{ .k = &nat, .stream = stream }, .x3 = &x3, .glue = &glue, .blas = &blas, .handle = handle,
        .group = &group, .b = &bufs, .ex = &exk, .xs = &xs, .probe = if (probing) Probe.hook else null, .probe_ctx = @ptrCast(&probe),
    };
    // equal chunks of at most 2048 rows
    // 最大 2048 行の等しいチャンクに分ける
    const n_chunks = (prompt_len + 2047) / 2048;
    const step = (prompt_len + n_chunks - 1) / n_chunks;
    const R: u32 = @intCast(@min(step, prompt_len));
    const ids = try a.alloc(i64, prompt_len);
    promptIds(ids, seed);
    // every chunk [start, e) at pos = start; t = e past index_topk, else null (no selection)
    // 各チャンク [start, e) を pos = start で。t は e が index_topk を超えたら e、でなければ null (選択なし)
    var start: usize = 0;
    while (start < prompt_len) : (start += step) {
        const e = @min(start + step, prompt_len);
        const rows: u32 = @intCast(e - start);
        const t: ?u32 = if (e > cfg.index_topk) @intCast(e) else null;
        // stream-ordered: the legacy-stream upload / memset do not wait for the previous chunk on the
        // non-blocking stream
        // stream の順に並べる: legacy stream の upload / memset は non-blocking stream 上の前のチャンクを待たないため
        try bufs.ids.uploadAsync(0, std.mem.sliceAsBytes(ids[start..e]), stream.handle);
        try state.pos.fill32(@intCast(start), stream.handle);
        try fw.embed(embed.ptr, rows);
        for (layers, 0..) |*L, i| try fw.layerPrompt(L, rows, state.kc[i].ptr, if (state.ic[i]) |b| b.ptr else null, state.pos.ptr, t);
        if (e == prompt_len) try fw.headLogits(final_norm.ptr, lm_head.ptr, rows - 1, 1, vshare);
    }
    try micro(&driver, rank, "prefilled");
    var heads_out: std.ArrayList(u8) = .empty;
    if (probing) try headHash(a, &heads_out, "full", try bufs.x.at(@as(usize, R - 1) * cfg.hidden * 2), &bufs, cfg, vshare, stream);
    var tokens: std.ArrayList(u32) = .empty;
    var tok = try fw.pickArgmax(gpa, vshare);
    try tokens.append(a, tok);
    // decode without drafts: one row a round at P, t = bucket(P + 1), the argmax head
    // draft なしの decode: 1 ラウンド 1 行を位置 P で、t = bucket(P + 1)、head は argmax
    var vfw = fw;
    vfw.b = &vbufs;
    vfw.xs = &vxs;
    var P: usize = prompt_len;
    const td = std.Io.Clock.awake.now(io);
    if (envInt("TF_ZIG_GRAPH", 0) == 1) {
        // one CUDA graph a key-range bucket: embed .. layers .. head .. pick, replayed with the position written first
        // キー範囲の bucket ごとに CUDA graph 1 つ: embed .. 層 .. head .. pick。位置を先に書いてから再生する
        if (probing) return error.GraphWithProbe;
        const first = [1]i64{tok};
        try vbufs.ids.upload(0, std.mem.sliceAsBytes(&first));
        var execs: [8]?cuda.graph.Exec = @splat(null);
        var keys: [8]?u32 = @splat(null);
        var n_graphs: usize = 0;
        // 1: a second created stream; 2: the legacy default stream
        // 1: 作成した 2 本目の stream、2: legacy の既定 stream
        const stream_mode = envInt("TF_ZIG_GRAPH_STREAM", 0);
        var replay_stream: ?cuda.Stream = switch (stream_mode) {
            1 => try cuda.Stream.init(&driver, true),
            2 => cuda.Stream{ .d = &driver, .handle = null },
            else => null,
        };
        defer if (stream_mode == 1) replay_stream.?.deinit();
        // NCCL keeps a persistent plan per captured collective until its exec is gone; ncclCommDestroy waits on them
        // NCCL は capture した collective ごとの plan を exec が消えるまで持ち、ncclCommDestroy はそれを待つ
        defer for (execs[0..n_graphs]) |*e| e.*.?.deinit();
        while (tokens.items.len < max_tokens) : (P += 1) {
            if (envInt("TF_ZIG_POS_KERNEL", 0) == 1) try glue.setI32(stream, state.pos.ptr, @intCast(P)) else try state.pos.fill32(@intCast(P), stream.handle);
            const t = bucket(P + 1, cfg.index_topk);
            const slot = for (0..n_graphs) |i| {
                if (keys[i] == t) break i;
            } else blk: {
                try cuda.graph.beginCapture(stream, .thread_local);
                try vfw.decodeStep(embed.ptr, layers, state.kc, state.ic, state.pos.ptr, t, final_norm.ptr, lm_head.ptr, vshare);
                var g = try cuda.graph.endCapture(stream);
                if (std.c.getenv("TF_ZIG_GRAPH_DOT")) |path| try g.dot(path, 0);
                defer g.deinit();
                execs[n_graphs] = try g.instantiateFlags(@intCast(envInt("TF_ZIG_GRAPH_FLAGS", 0)));
                if (envInt("TF_ZIG_GRAPH_UPLOAD", 0) == 1) try execs[n_graphs].?.upload(stream);
                keys[n_graphs] = t;
                n_graphs += 1;
                std.debug.print("[zig] rank {d}: captured the decode graph for keys {?d} ({d} nodes)\n", .{ rank, t, try g.nodeCount() });
                break :blk n_graphs - 1;
            };
            // TF_ZIG_GRAPH_STREAM=1: replay on a second stream
            // TF_ZIG_GRAPH_STREAM=1: 2 本目の stream で再生する
            const ls = if (replay_stream) |*r| r.* else stream;
            if (replay_stream != null) try stream.synchronize();
            try execs[slot].?.launchOn(ls);
            try ls.synchronize();
            var got: [1]i32 = undefined;
            if (envInt("TF_ZIG_ASYNC_DL", 0) == 1) {
                try vbufs.tok_out.downloadAsync(0, std.mem.sliceAsBytes(&got), ls.handle);
                try ls.synchronize();
            } else try vbufs.tok_out.download(0, std.mem.sliceAsBytes(&got));
            tok = @intCast(got[0]);
            try tokens.append(a, tok);
        }
    }
    while (tokens.items.len < max_tokens) : (P += 1) {
        const one = [1]i64{tok};
        try vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(&one), stream.handle);
        try state.pos.fill32(@intCast(P), stream.handle);
        const t = bucket(P + 1, cfg.index_topk);
        try vfw.embed(embed.ptr, 1);
        for (layers, 0..) |*L, i| try vfw.layerDecode(L, 1, state.kc[i].ptr, if (state.ic[i]) |b| b.ptr else null, state.pos.ptr, t);
        try vfw.headLogits(final_norm.ptr, lm_head.ptr, 0, 1, vshare);
        if (probing) try headHash(a, &heads_out, "argmax", vbufs.x.ptr, &vbufs, cfg, vshare, stream);
        tok = try vfw.pickArgmax(gpa, vshare);
        try tokens.append(a, tok);
    }
    try stream.synchronize();
    const dec_ms = td.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    std.debug.print("[zig] rank {d}: decode {d} tokens in {d} ms = {d:.2} tok/s (probe {})\n", .{ rank, tokens.items.len - 1, dec_ms, @as(f64, @floatFromInt(tokens.items.len - 1)) * 1000.0 / @as(f64, @floatFromInt(@max(dec_ms, 1))), probing });
    var out: std.ArrayList(u8) = .empty;
    var toks: std.ArrayList(u8) = .empty;
    for (tokens.items, 0..) |t, i| try toks.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", t });
    try out.print(a, "{{\"rank\": {d}, \"tokens\": [{s}], \"heads\": [\n{s}\n], \"layers\": [\n{s}\n]}}\n", .{ rank, toks.items, heads_out.items, probe.lines.items });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[10], .data = out.items });
    std.debug.print("[zig] rank {d}: prompt in chunks of {d} rows through {d} layers, {d} hashes -> {s}\n", .{ rank, R, nlayers, probe.n, args[10] });
    _ = &embed;
    return 0;
}
