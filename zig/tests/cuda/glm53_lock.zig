//! `tf-glm53-lock RANK WORLD MASTER CKPT STORED_WORLD LAYERS - CASES OUT` (one process a GPU): the ranks agree on what
//! they loaded, then rank 0 sends each case of CASES (`LEN:SEED:MAX[:EOS],...`) and generates while the others follow.
//! Rank 0 writes the tokens and prompt digests it sent to OUT, followers the headers and prompt digests they received.
//! 1 GPU に 1 プロセス。ランクどうしが読み込んだものを照合し、rank 0 が CASES の各ケースを送って生成し、他のランクは
//! それに従う。rank 0 は送ったトークンと prompt のダイジェストを、follower は受け取ったヘッダと prompt のダイジェストを
//! OUT に書く。
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

/// TF_ZIG_LAYERS=1 (rank 0): each layer step's output hashed, in the reference implementation's format.
/// TF_ZIG_LAYERS=1 (rank 0): 各層の段の出力を、参照実装と同じ形式でハッシュする。
const Probe = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lines: std.ArrayList(u8) = .empty,
    n: usize = 0,
    sel: usize = 0,
    rank: u32 = 0,

    fn hook(ctx: *anyopaque, layer: usize, what: []const u8, rows: u32, ptr: u64, bytes: usize) anyerror!void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const host = try p.gpa.alloc(u8, bytes);
        defer p.gpa.free(host);
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH");
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (layer == std.math.maxInt(usize) and (std.mem.eql(u8, what, "tok") or std.mem.eql(u8, what, "cnt"))) {
            if (std.c.getenv("TF_ZIG_SELECT_DUMP")) |dump| {
                if (std.mem.eql(u8, what, "tok")) p.sel += 1;
                if (p.sel <= 2) {
                    var name: [256]u8 = undefined;
                    const path = try std.fmt.bufPrintSentinel(&name, "{s}-r{d}-{s}{d}.bin", .{ std.mem.span(dump), p.rank, what, p.sel }, 0);
                    const f = std.c.fopen(path, "wb") orelse return error.DumpOpen;
                    _ = std.c.fwrite(host.ptr, 1, host.len, f);
                    _ = std.c.fclose(f);
                }
            }
        }
        if (layer == std.math.maxInt(usize)) {
            try p.lines.print(p.gpa, "{s}\"S:{s}:{d}:{s}\"", .{ if (p.n == 0) "" else ", ", what, rows, hex[0..24] });
        } else try p.lines.print(p.gpa, "{s}\"L{d}:{s}:{d}:{s}\"", .{ if (p.n == 0) "" else ", ", layer, what, rows, hex[0..24] });
        p.n += 1;
    }
};

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

/// A test proposer (TF_ZIG_FED=PATH, the reference's JSON of the same cases): the reference reply's next tokens as
/// drafts, every third round with its last draft made wrong and every fifth with its first, so rounds keep, cut and
/// reject drafts. Every rank reads the same file and finds the request's case by its prompt.
/// テスト用の proposer (TF_ZIG_FED=PATH、同じケースの参照 JSON): 参照の返答の次のトークンを draft にし、3 ラウンドに
/// 1 回は最後の draft を、5 ラウンドに 1 回は最初の draft をわざと間違える。これでラウンドが draft を残す・切る・
/// 退ける場合をすべて通る。全ランクが同じファイルを読み、要求のケースを prompt で見つける。
const Fed = struct {
    gpa: std.mem.Allocator,
    refs: []Ref,
    cur: ?*const Ref = null,
    prompt_len: usize = 0,

    const Ref = struct { len: usize, seed: u64, tokens: []u32 };

    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Fed {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        var refs: std.ArrayList(Ref) = .empty;
        for (parsed.value.object.get("runs").?.array.items) |r| {
            const c = r.object.get("case").?.array.items;
            const toks = r.object.get("tokens").?.array.items;
            const t = try gpa.alloc(u32, toks.len);
            for (toks, t) |v, *o| o.* = @intCast(v.integer);
            try refs.append(gpa, .{ .len = @intCast(c[0].integer), .seed = @intCast(c[1].integer), .tokens = t });
        }
        return .{ .gpa = gpa, .refs = refs.items };
    }

    fn begin(ctx: *anyopaque, prompt: []const i64) anyerror!void {
        const f: *Fed = @ptrCast(@alignCast(ctx));
        f.cur = null;
        f.prompt_len = prompt.len;
        const ids = try f.gpa.alloc(i64, prompt.len);
        defer f.gpa.free(ids);
        for (f.refs) |*r| {
            if (r.len != prompt.len) continue;
            promptIds(ids, r.seed);
            if (std.mem.eql(i64, ids, prompt)) f.cur = r;
        }
        if (f.cur == null) return error.NoReference;
    }

    fn propose(ctx: *anyopaque, tok: u32, P: usize, room: usize, out: []u32) anyerror!usize {
        _ = tok;
        const f: *Fed = @ptrCast(@alignCast(ctx));
        const r = f.cur.?;
        // the reply index of the first draft (the pending token is reply index at - 1)
        // 最初の draft の返答内の添字 (pending トークンは添字 at - 1)
        const at = P - f.prompt_len + 1;
        if (at >= r.tokens.len) return 0;
        const n = @min(room, r.tokens.len - at);
        @memcpy(out[0..n], r.tokens[at..][0..n]);
        if (P % 3 == 0 and n >= 2) out[n - 1] +%= 1;
        if (P % 5 == 0) out[0] +%= 1;
        return n;
    }

    fn proposer(f: *Fed) glm.engine.Proposer {
        return .{ .ctx = @ptrCast(f), .propose = propose, .begin = begin };
    }
};

const Collect = struct {
    gpa: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    fn tokens(ctx: *anyopaque, new: []const u32) void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(c.gpa, new) catch {};
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 10) {
        std.debug.print("usage: tf-glm53-lock RANK WORLD MASTER CKPT STORED_WORLD LAYERS UNUSED CASES OUT\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const ckpt = args[4];

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var nccl = try cuda.nccl.Library.openEnv();
    defer nccl.close();
    var group = try cuda.tp.Group.init(&nccl, rank, world, try cuda.tp.parseMaster(args[3]));
    defer group.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    const eng = try glm.engine.Engine.create(gpa, io, &driver, &ctx, &group, glm.engine.withEnv(.{
        .ckpt = ckpt,
        .stored_world = try std.fmt.parseInt(usize, args[5], 10),
        .layers = try std.fmt.parseInt(usize, args[6], 10),
        .capacity = envInt("TF_ZIG_CAPACITY", 8192),
        .graphs = envInt("TF_ZIG_GRAPH", 1) == 1,
        .drafter = if (std.c.getenv("TF_ZIG_DFLASH")) |v| std.mem.span(v) else null,
        .kv_fp8 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp8") or std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .kv_fp4 = if (std.c.getenv("TF_GLM53_KV")) |v| std.mem.eql(u8, std.mem.span(v), "fp4") else false,
        .dcp = @intCast(envInt("TF_GLM53_DCP", 1)),
    }));
    defer eng.destroy();
    std.debug.print("[lock] rank {d}/{d}: engine up in {d} ms\n", .{ rank, world, t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });

    if (envInt("TF_ZIG_ROCE_TEST", 0) > 0) if (eng.fast) |*f| {
        // RoCE self-test: all-reduces of decode-window sizes (and a 16-byte one) against the host's rank-order sum
        // RoCE の自己テスト: decode ウィンドウの大きさ (と 16 バイト) の all-reduce を、ホストのランク順の和と比べる
        const iters = envInt("TF_ZIG_ROCE_TEST", 0);
        const sizes = [_]u32{ 1536 * 16, 16, 3072 * 16, 10752 * 16, 64 * 16, 1536 * 16, 12288 * 16 };
        var in = try cuda.DeviceBuffer.alloc(&driver, 12288 * 16);
        var out = try cuda.DeviceBuffer.alloc(&driver, 12288 * 16);
        const host = try gpa.alloc(f32, 12288 * 4);
        const got = try gpa.alloc(f32, 12288 * 4);
        var bad: usize = 0;
        for (0..iters) |it| {
            const nb = sizes[it % sizes.len];
            const n = nb / 4;
            for (host[0..n], 0..) |*v, i| v.* = @floatFromInt((rank + 1) * 1000 + (i + it) % 997);
            try in.uploadAsync(0, std.mem.sliceAsBytes(host[0..n]), eng.stream.handle);
            try f.allReduce(eng.stream, in.ptr, out.ptr, nb, false);
            try eng.stream.synchronize();
            {
                const ctrl: *const volatile [7]u32 = @ptrFromInt(f.reg.ctrl);
                var ep: [10]u32 = undefined;
                try f.counters.download(0, std.mem.sliceAsBytes(&ep));
                if (rank == 0 and it < 0) std.debug.print("[roce-test] it {d} size {d}: ctrl seq {d} epoch {d} stage {any} tail {any}\n", .{ it, nb, ctrl[0], ep[0], ep[1..5], ep[5..9] });
            }
            if (!f.healthy()) {
                std.debug.print("[roce-test] rank {d}: unhealthy at iter {d} (size {d}), {d} wrong so far\n", .{ rank, it, nb, bad });
                return 4;
            }
            try out.download(0, std.mem.sliceAsBytes(got[0..n]));
            var bad_it: usize = 0;
            for (got[0..n], 0..) |v, i| {
                var want: f32 = 0;
                for (0..world) |r| want += @floatFromInt((r + 1) * 1000 + (i + it) % 997);
                if (v != want) {
                    if (bad_it == 0 and rank == 0) std.debug.print("[roce-test] it {d} size {d}: first wrong i {d}: got {d} want {d}\n", .{ it, nb, i, v, want });
                    bad_it += 1;
                }
            }
            if (bad_it > 0 and rank == 0) std.debug.print("[roce-test] it {d} size {d}: {d} of {d} wrong\n", .{ it, nb, bad_it, n });
            bad += bad_it;
        }
        std.debug.print("[roce-test] rank {d}: {d} all-reduces, {d} wrong values\n", .{ rank, iters, bad });
    };
    if (envInt("TF_ZIG_TOPK_BENCH", 0) > 0) {
        // decode-window top-k at long context: topk_values over [R, Tl] packed keys (random), timed with events
        // 長い文脈での decode ウィンドウの top-k: [R, Tl] の詰めたキー (乱数) に topk_values をかけ、event で計る
        const Tl: u32 = @intCast(envInt("TF_ZIG_TOPK_BENCH", 0));
        const host = try gpa.alloc(i64, 8 * @as(usize, Tl));
        var rng = std.Random.DefaultPrng.init(42);
        for (host, 0..) |*v, i| v.* = (@as(i64, rng.random().int(i32)) << 32) | @as(i64, @intCast(0x7FFFFFFF - (i % Tl)));
        const sc = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(host));
        const out = try cuda.DeviceBuffer.alloc(&driver, 8 * 2048 * 8);
        var ev0 = try cuda.Event.init(&driver, true);
        var ev1 = try cuda.Event.init(&driver, true);
        var state = try cuda.DeviceBuffer.alloc(&driver, 16 * 1040);
        try state.fill8(0, null);
        const cand = try cuda.DeviceBuffer.alloc(&driver, 8 * @as(usize, Tl) * 8);
        const out2 = try cuda.DeviceBuffer.alloc(&driver, 8 * 2048 * 8);
        try ctx.synchronize();
        // warm load (a clock governor may raise the cap only after it sees the GPU busy for a while)
        // 暖機の負荷 (クロック制御は GPU がしばらく忙しいのを見てから上限を上げることがある)
        {
            const tw = std.Io.Clock.awake.now(io);
            while (tw.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() < 15000) {
                for (0..20) |_| try eng.glue.topkValues(eng.stream, sc.ptr, out.ptr, 8, Tl, 2048);
                try eng.stream.synchronize();
            }
        }
        for ([_]u32{ 1, 4, 8 }) |R| {
            try eng.glue.topkValues(eng.stream, sc.ptr, out.ptr, R, Tl, 2048);
            try ev0.record(eng.stream);
            for (0..20) |_| try eng.glue.topkValues(eng.stream, sc.ptr, out.ptr, R, Tl, 2048);
            try ev1.record(eng.stream);
            try ev1.synchronize();
            const old_ms = (try cuda.Event.elapsedMs(ev0, ev1)) / 20.0;
            try eng.glue.topkValuesWide(eng.stream, sc.ptr, out2.ptr, R, Tl, 2048, state.ptr, cand.ptr);
            try ev0.record(eng.stream);
            for (0..20) |_| try eng.glue.topkValuesWide(eng.stream, sc.ptr, out2.ptr, R, Tl, 2048, state.ptr, cand.ptr);
            try ev1.record(eng.stream);
            try ev1.synchronize();
            const new_ms = (try cuda.Event.elapsedMs(ev0, ev1)) / 20.0;
            var stw: [260]i32 = undefined;
            try state.download(0, std.mem.sliceAsBytes(&stw));
            std.debug.print("[topk-bench] row 0 state: prefix {d} need {d} n_out {d} n_cand {d}\n", .{ stw[256], stw[257], stw[258], stw[259] });
            const xa = try gpa.alloc(i64, R * 2048);
            const xb = try gpa.alloc(i64, R * 2048);
            try out.download(0, std.mem.sliceAsBytes(xa));
            try out2.download(0, std.mem.sliceAsBytes(xb));
            var same = true;
            for (0..R) |r| {
                std.mem.sort(i64, xa[r * 2048 ..][0..2048], {}, std.sort.asc(i64));
                std.mem.sort(i64, xb[r * 2048 ..][0..2048], {}, std.sort.asc(i64));
                if (!std.mem.eql(i64, xa[r * 2048 ..][0..2048], xb[r * 2048 ..][0..2048])) same = false;
            }
            std.debug.print("[topk-bench] rank {d}: Tl {d} rows {d}: old {d:.3} ms, wide {d:.3} ms, same set {any}\n", .{ rank, Tl, R, old_ms, new_ms, same });
        }
    }
    if (envInt("TF_ZIG_PREWARM", 0) == 1) {
        const tw = std.Io.Clock.awake.now(io);
        const ng = try eng.prewarm();
        std.debug.print("[lock] rank {d}: {d} graphs captured up front in {d} ms\n", .{ rank, ng, tw.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
    }
    var ch = glm.lockstep.Channel.init(gpa, eng);
    var fp = try glm.lockstep.Fingerprint.of(io, gpa, ckpt, "tf-glm53-lock", world, rank);
    // TF_ZIG_FP_BREAK_RANK=r: perturb rank r's fingerprint; every rank must then refuse to start
    // TF_ZIG_FP_BREAK_RANK=r: rank r の fingerprint を乱す。そのとき全ランクが起動を拒まなければならない
    if (envInt("TF_ZIG_FP_BREAK_RANK", 999) == rank) fp.words[0] ^= 1;
    fp.agree(&ch) catch |err| {
        std.debug.print("[lock] rank {d}: the start-up check refused ({s})\n", .{ rank, @errorName(err) });
        return 3;
    };

    try glm.slots_serve.setup(&ch, eng, &ctx, try glm.slots.Env.fromEnv(), &.{ckpt}); // TF_GLM53_PREFIX=slots
    try glm.pool_serve.setup(&ch, eng); // TF_GLM53_PREFIX=pool

    // Fiore-lab: TF_GLM53_EXPERT_BITS=<set>.json (glm53-expert-bits v1, the same file on every node; the banks from
    // TF_GLM53_EXPERT_BANKS): every rank changes the routed experts to it before anything runs (lockstep.swapEvery)
    // Fiore-lab: TF_GLM53_EXPERT_BITS=<set>.json (glm53-expert-bits v1、全ノードに同じファイル。バンクは
    // TF_GLM53_EXPERT_BANKS): 何かを実行する前に、全ランクが routed expert をそれに変える (lockstep.swapEvery)
    if (std.c.getenv("TF_GLM53_EXPERT_BITS")) |p| {
        const bits = try glm.expert_swap.readSet(gpa, io, std.mem.span(p), eng);
        defer gpa.free(bits);
        var st: glm.expert_swap.Stats = .{};
        try glm.lockstep.swapEvery(&ch, eng, bits, null, &st);
        std.debug.print("[lock] rank {d}: expert bits from {s}: {d} experts in {d} layers ({d} up, {d} down), {d} MiB read, {d} ms\n", .{ rank, std.mem.span(p), st.experts, st.layers, st.up, st.down, st.read_bytes >> 20, st.ms });
    }

    // TF_ZIG_SCORE=a.npy[,b.npy..] (for KLD): each token file (npy, int32 / int64, one row) teacher-forced on every
    // rank (TF_ZIG_SCORE_P0 rows through the prefill, then decode windows of TF_ZIG_SCORE_WIN); rank 0 writes the
    // logits of positions p0 .. len - 2 as fp32 [rows, world * vshare] to <OUT>-<i>.f32
    // TF_ZIG_SCORE=a.npy[,b.npy..] (KLD 用): 各トークンファイル (npy、int32 / int64、1 行) を全ランクで teacher
    // forcing する (TF_ZIG_SCORE_P0 行を prefill で、残りを TF_ZIG_SCORE_WIN 行の decode ウィンドウで)。rank 0 は位置
    // p0 .. len - 2 の logits を fp32 [rows, world * vshare] で <OUT>-<i>.f32 に書く
    if (std.c.getenv("TF_ZIG_SCORE")) |list| {
        const p0 = envInt("TF_ZIG_SCORE_P0", 1024);
        const win: u32 = @intCast(envInt("TF_ZIG_SCORE_WIN", 16));
        // TF_ZIG_SELECT_DUMP=PREFIX: the first `dcp` ranks (one of each DCP member) write every decode window's picks
        // to PREFIX-r<rank>-<file>.sel - a record a window past index_topk of context: int32 P (first position),
        // R (rows), layers (index layers), K (index_topk), 8 (rows a layer's block holds); then cnt [layers][8] int32
        // and tok [layers][8][K] int32 (this member's local slots, ascending; global position =
        // slot * dcp + rank % dcp)
        // TF_ZIG_SELECT_DUMP=PREFIX: 先頭の `dcp` ランク (DCP の各メンバー 1 つずつ) が、decode ウィンドウごとの選択を
        // PREFIX-r<rank>-<file>.sel に書く。文脈が index_topk を超えたウィンドウごとに 1 レコード: int32 の
        // P (先頭位置)、R (行数)、layers (index 層の数)、K (index_topk)、8 (層の block が持つ行数)。続いて
        // cnt [layers][8] int32 と tok [layers][8][K] int32 (このメンバーのローカルスロット、昇順。全体の位置 =
        // slot * dcp + rank % dcp)
        const SelSink = struct {
            f: ?*std.c.FILE = null,
            k: u32,
            fn window(c: *anyopaque, P: usize, R: u32, layers: u32, tok: []const u8, cnt: []const u8) anyerror!void {
                const self: *@This() = @ptrCast(@alignCast(c));
                const f = self.f orelse return;
                const head = [5]i32{ @intCast(P), @intCast(R), @intCast(layers), @intCast(self.k), 8 };
                _ = std.c.fwrite(std.mem.asBytes(&head), 1, 20, f);
                _ = std.c.fwrite(cnt.ptr, 1, cnt.len, f);
                _ = std.c.fwrite(tok.ptr, 1, tok.len, f);
            }
        };
        var ss: SelSink = .{ .k = @intCast(eng.cfg.index_topk) };
        const sel_dump = if (std.c.getenv("TF_ZIG_SELECT_DUMP")) |v| (if (rank < eng.dcp) std.mem.span(v) else null) else null;
        if (sel_dump != null) {
            try eng.selLogEnable();
            eng.sel_sink = SelSink.window;
            eng.sel_sink_ctx = @ptrCast(&ss);
        }
        var fit = std.mem.tokenizeScalar(u8, std.mem.span(list), ',');
        var fi: usize = 0;
        while (fit.next()) |path| : (fi += 1) {
            const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
            if (raw.len < 10 or !std.mem.eql(u8, raw[1..6], "NUMPY")) return error.NotNpy;
            const hl = std.mem.readInt(u16, raw[8..10], .little);
            const hdr_s = raw[10..][0..hl];
            const body = raw[10 + hl ..];
            const wide = std.mem.indexOf(u8, hdr_s, "<i8") != null;
            if (!wide and std.mem.indexOf(u8, hdr_s, "<i4") == null) return error.NpyDtype;
            const ntok = body.len / @as(usize, if (wide) 8 else 4);
            const toks = try gpa.alloc(i64, ntok);
            for (toks, 0..) |*t, i| t.* = if (wide) std.mem.readInt(i64, body[i * 8 ..][0..8], .little) else std.mem.readInt(i32, body[i * 4 ..][0..4], .little);
            const Sink2 = struct {
                rows: std.ArrayList(u8) = .empty,
                gpa: std.mem.Allocator,
                keep: bool,
                first: usize = std.math.maxInt(usize),
                fn row(c: *anyopaque, pos: usize, lg: []const f32) anyerror!void {
                    const self: *@This() = @ptrCast(@alignCast(c));
                    if (self.first == std.math.maxInt(usize)) self.first = pos;
                    if (self.keep) try self.rows.appendSlice(self.gpa, std.mem.sliceAsBytes(lg));
                }
            };
            var sk: Sink2 = .{ .gpa = gpa, .keep = rank == 0 };
            if (sel_dump) |pfx| {
                var nb: [512]u8 = undefined;
                const spath = try std.fmt.bufPrintSentinel(&nb, "{s}-r{d}-{d}.sel", .{ pfx, rank, fi }, 0);
                ss.f = std.c.fopen(spath, "wb") orelse return error.DumpOpen;
            }
            const ts = std.Io.Clock.awake.now(io);
            try eng.scoreForced(toks, p0, win, @ptrCast(&sk), Sink2.row);
            if (ss.f) |f| {
                _ = std.c.fclose(f);
                ss.f = null;
            }
            if (rank == 0) {
                const name = try std.fmt.allocPrint(gpa, "{s}-{d}.f32", .{ args[9], fi });
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = sk.rows.items });
                std.debug.print("[score] {s}: {d} tokens, positions {d}..{d} ({d} rows of {d}) in {d} ms -> {s}\n", .{ path, ntok, sk.first, ntok - 2, sk.rows.items.len / 4 / (eng.world * eng.vshare), eng.world * eng.vshare, ts.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(), name });
            }
            sk.rows.deinit(gpa);
            gpa.free(toks);
            gpa.free(raw);
        }
        return 0;
    }
    var probe: Probe = .{ .gpa = gpa, .d = &driver, .rank = rank };
    if ((rank == 0 or std.c.getenv("TF_ZIG_SELECT_DUMP") != null) and envInt("TF_ZIG_LAYERS", 0) == 1) {
        eng.probe = Probe.hook;
        eng.probe_ctx = @ptrCast(&probe);
        eng.probe_select = envInt("TF_ZIG_SELECT", 0) == 1;
    }
    var fed: ?Fed = null;
    if (std.c.getenv("TF_ZIG_FED")) |path| fed = try Fed.load(gpa, io, std.mem.span(path));
    const prop: ?glm.engine.Proposer = if (fed) |*f| f.proposer() else null;
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{{\"rank\": {d}, \"requests\": [\n", .{rank});
    if (rank == 0) {
        var it = std.mem.tokenizeScalar(u8, args[8], ',');
        // TF_SLOTS_CONV: one entry a request, "-" (alone), "K" (the conversation K: its history - every earlier prompt
        // and reply of K - then this case's ids) or "K/P" (a new conversation K starting from conversation P's first
        // prompt)
        // TF_SLOTS_CONV: 要求ごとに 1 項目。"-" (単独)、"K" (会話 K: その履歴 = K のそれまでの全 prompt と返答、続いて
        // このケースの id)、"K/P" (会話 P の最初の prompt から始まる新しい会話 K)
        var conv_it = std.mem.tokenizeScalar(u8, if (std.c.getenv("TF_SLOTS_CONV")) |v| std.mem.span(v) else "", ',');
        var hist: [16]std.ArrayList(i64) = @splat(.empty);
        var first_prompt: [16][]const i64 = @splat(&.{});
        var first = true;
        while (it.next()) |c| {
            var f = std.mem.tokenizeScalar(u8, c, ':');
            const len = try std.fmt.parseInt(usize, f.next().?, 10);
            const seed = try std.fmt.parseInt(u64, f.next().?, 10);
            const max = try std.fmt.parseInt(i32, f.next().?, 10);
            const eos = if (f.next()) |v| std.mem.eql(u8, v, "1") else false;
            const mode: u8 = if (f.next()) |v| (if (std.mem.eql(u8, v, "d")) glm.lockstep.mode_dflash else if (std.mem.eql(u8, v, "m")) glm.lockstep.mode_mtp else 0) else 0;
            const own = try a.alloc(i64, len);
            promptIds(own, seed);
            var conv: ?usize = null;
            var prompt: []i64 = own;
            if (conv_it.next()) |cv| if (!std.mem.eql(u8, cv, "-")) {
                var cf = std.mem.tokenizeScalar(u8, cv, '/');
                const k = try std.fmt.parseInt(usize, cf.next().?, 10);
                if (cf.next()) |pp| {
                    const par = try std.fmt.parseInt(usize, pp, 10);
                    hist[k].clearRetainingCapacity();
                    try hist[k].appendSlice(a, first_prompt[par]);
                }
                prompt = try a.alloc(i64, hist[k].items.len + len);
                @memcpy(prompt[0..hist[k].items.len], hist[k].items);
                @memcpy(prompt[hist[k].items.len..], own);
                if (first_prompt[k].len == 0) first_prompt[k] = prompt;
                conv = k;
            };
            var col: Collect = .{ .gpa = gpa };
            probe.lines.clearRetainingCapacity();
            probe.n = 0;
            var hdr: glm.lockstep.Header = .{ .max_tokens = max, .stop_eos = eos, .mode = mode };
            // TF_ZIG_SAMPLE=temperature,top_p,top_k,min_p: keyed sampling, seeded from the prompt (seedFor)
            // TF_ZIG_SAMPLE=temperature,top_p,top_k,min_p: keyed サンプリング。seed は prompt から作る (seedFor)
            if (std.c.getenv("TF_ZIG_SAMPLE")) |sv| {
                var sf = std.mem.tokenizeScalar(u8, std.mem.span(sv), ',');
                hdr.temperature = try std.fmt.parseFloat(f64, sf.next().?);
                hdr.top_p = if (sf.next()) |v| try std.fmt.parseFloat(f64, v) else 1.0;
                hdr.top_k = if (sf.next()) |v| try std.fmt.parseInt(i32, v, 10) else 0;
                hdr.min_p = if (sf.next()) |v| try std.fmt.parseFloat(f64, v) else 0.0;
                const ids32 = try a.alloc(u32, prompt.len);
                for (ids32, prompt) |*d, t| d.* = @intCast(t);
                hdr.seed = glm.engine.seedFor(ids32, 0);
            }
            const st = try glm.lockstep.lead(&ch, eng, hdr, prompt, .{ .ctx = @ptrCast(&col), .tokens = Collect.tokens }, prop);
            if (conv) |k| {
                hist[k].clearRetainingCapacity();
                try hist[k].appendSlice(a, prompt);
                for (col.toks.items) |t| try hist[k].append(a, t);
            }
            var d: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(prompt), &d, .{});
            var toks: std.ArrayList(u8) = .empty;
            for (col.toks.items, 0..) |t, i| try toks.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", t });
            const dec_s = @as(f64, @floatFromInt(st.decode_ns)) / 1e9;
            try out.print(a, "{s}{{\"case\": [{d}, {d}, {d}, {d}], \"prompt_sha256\": \"{s}\", \"tokens\": [{s}], \"finish\": \"{s}\", \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"cached\": {d}, \"prompt_len\": {d}, \"mode\": {d}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"layers\": [{s}]}}", .{
                if (first) "" else ",\n", len, seed, max, @intFromBool(eos), std.fmt.bytesToHex(d, .lower), toks.items, @tagName(st.finish),
                @as(f64, @floatFromInt(st.prefill_ns)) / 1e9, dec_s, st.cached, prompt.len, mode, st.rounds, st.drafted, st.accepted, probe.lines.items,
            });
            if (st.rounds > 0) std.debug.print("[lock] round phases ms/round: agree {d:.2} propose {d:.2} agree2 {d:.2} window {d:.2} kept {d:.2}\n", .{
                @as(f64, @floatFromInt(eng.round_ns[0])) / 1e6 / @as(f64, @floatFromInt(st.rounds)), @as(f64, @floatFromInt(eng.round_ns[1])) / 1e6 / @as(f64, @floatFromInt(st.rounds)),
                @as(f64, @floatFromInt(eng.round_ns[2])) / 1e6 / @as(f64, @floatFromInt(st.rounds)), @as(f64, @floatFromInt(eng.round_ns[3])) / 1e6 / @as(f64, @floatFromInt(st.rounds)),
                @as(f64, @floatFromInt(eng.round_ns[4])) / 1e6 / @as(f64, @floatFromInt(st.rounds)),
            });
            eng.round_ns = @splat(0);
            std.debug.print("[lock] case {s}: {d} tokens ({s}, rounds {d}, drafted {d}, accepted {d}), prefill {d:.2} s, decode {d:.2} tok/s\n", .{ c, col.toks.items.len, @tagName(st.finish), st.rounds, st.drafted, st.accepted, @as(f64, @floatFromInt(st.prefill_ns)) / 1e9, @as(f64, @floatFromInt(st.generated -| 1)) / @max(dec_s, 1e-9) });
            first = false;
        }
        try glm.lockstep.shutdown(&ch);
    } else {
        var seen: std.ArrayList(glm.lockstep.Received) = .empty;
        try glm.lockstep.follow(&ch, eng, &seen, prop);
        if (std.c.getenv("TF_ZIG_SELECT_DUMP")) |dump| if (probe.n > 0) {
            var name: [256]u8 = undefined;
            const path = try std.fmt.bufPrint(&name, "{s}-r{d}-log.txt", .{ std.mem.span(dump), rank });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = probe.lines.items });
        };
        for (seen.items, 0..) |r, i| {
            const w = r.header.encode();
            var hw: std.ArrayList(u8) = .empty;
            for (w, 0..) |v, j| try hw.print(a, "{s}{d}", .{ if (j == 0) "" else ", ", v });
            try out.print(a, "{s}{{\"header\": [{s}], \"prompt_sha256\": \"{s}\", \"generated\": {d}}}", .{ if (i == 0) "" else ",\n", hw.items, std.fmt.bytesToHex(r.prompt_sha256, .lower), r.stats.generated });
        }
    }
    try out.print(a, "\n]}}\n", .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[9], .data = out.items });
    std.debug.print("[lock] rank {d}: done -> {s}\n", .{ rank, args[9] });
    return 0;
}
