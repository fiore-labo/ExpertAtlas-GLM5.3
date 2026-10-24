// Fiore-lab: the start-up's progress for the panel's launcher (TF_GLM53_LOAD_PROGRESS), not in TensorFold 1.0.2.
//! TF_GLM53_LOAD_PROGRESS=<path>, the same on every rank: every rank agrees after each layer it loads (one all-gather
//! a layer, with whether the layer loaded: a rank that fails stops every rank at the same layer), and rank 0 writes the
//! start-up's progress to <path> as JSON - written beside it and renamed, so a reader never sees half a file. The
//! launcher (glm53/launcher) serves it as GET /v1/launch/status.
//! TF_GLM53_LOAD_PROGRESS=<path> (全ランクで同じ): 全ランクが層を読み込むごとに合意し (1 層に 1 回の all-gather。その層を
//! 読めたかを添える: 失敗したランクがあれば全ランクが同じ層で止まる)、rank 0 が起動の進み具合を <path> に JSON で書く
//! (横に書いてから名前を変えるので、読む側が書きかけを見ることはない)。ランチャー (glm53/launcher) はそれを
//! GET /v1/launch/status として出す。
const std = @import("std");
const cuda = @import("cuda");
const lockstep = @import("lockstep.zig");
const weights = @import("weights.zig");

/// weights: the checkpoint and the banks read; kv: kernels, tuning, buffers, KV caches, drafter; prewarm: the
/// start-up's prompts and the decode graphs; ready: the engine is up (the server starts next).
/// weights: チェックポイントとバンクを読む。kv: カーネル、調整、バッファ、KV キャッシュ、drafter。prewarm: 起動時の
/// prompt と decode のグラフ。ready: エンジンが起動した (次にサーバーが始まる)。
pub const Phase = enum { weights, kv, prewarm, ready };

pub const Progress = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ch: lockstep.Channel,
    path: ?[]const u8, // rank 0 only / rank 0 だけ
    t0: std.Io.Timestamp,
    phase: Phase = .weights,
    first: usize, // the first MoE layer / 最初の MoE 層
    done: []u8, // one byte a MoE layer: 1 = loaded on every rank / MoE 層ごとに 1 バイト: 1 は全ランクで読み込み済み
    total_bytes: u64, // rank 0's bytes to read, 0 unknown / rank 0 が読むバイト数。0 は不明
    plan: []const u8 = "",
    plan_digest: [32]u8 = @splat(0),
    warned: bool = false,

    /// null when TF_GLM53_LOAD_PROGRESS is not set (then nothing is agreed or written).
    /// TF_GLM53_LOAD_PROGRESS が無ければ null (そのときは合意も書き込みもしない)。
    /// Every rank makes one, set or not (a variable on some ranks only would leave one rank in an all-gather the others
    /// never join); only rank 0 with TF_GLM53_LOAD_PROGRESS writes the file.
    /// 変数があってもなくても全ランクが作る (一部のランクだけに変数があると、そのランクだけが他の入らない all-gather に
    /// 入ってしまう)。ファイルを書くのは TF_GLM53_LOAD_PROGRESS のある rank 0 だけ。
    pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, group: *const cuda.tp.Group, stream: cuda.Stream, first: usize, moe_layers: usize, total_bytes: u64) !?Progress {
        const v = std.c.getenv("TF_GLM53_LOAD_PROGRESS");
        const done = try gpa.alloc(u8, moe_layers);
        @memset(done, 0);
        var p: Progress = .{
            .gpa = gpa, .io = io, .ch = .{ .gpa = gpa, .d = d, .group = group, .stream = stream },
            .path = if (group.rank == 0) if (v) |s| std.mem.span(s) else null else null,
            .t0 = std.Io.Clock.awake.now(io),
            .first = first, .done = done, .total_bytes = total_bytes,
        };
        p.write();
        return p;
    }

    /// Whether every rank passed `ok` (the plan's checks before the first layer).
    /// 全ランクが `ok` を渡したか (最初の層の前の、計画の検査)。
    pub fn agree(p: *Progress, ok: bool) !bool {
        return p.ch.allOk(ok);
    }

    /// After layer `layer` (`ok`: it loaded here): whether every rank loaded it. A MoE layer every rank loaded is
    /// marked done and the file rewritten.
    /// 層 `layer` の後で (`ok`: ここで読めた): 全ランクが読めたか。全ランクが読めた MoE 層は済みにして、ファイルを書き直す。
    pub fn layerDone(p: *Progress, layer: usize, moe: bool, ok: bool) !bool {
        if (!try p.ch.allOk(ok)) return false;
        if (moe) {
            p.done[layer - p.first] = 1;
            p.write();
        }
        return true;
    }

    pub fn setPhase(p: *Progress, phase: Phase) void {
        p.phase = phase;
        p.write();
    }

    /// The channel's device buffers back (the load is over); the file stays and is rewritten on each phase.
    /// チャネルのデバイスバッファを返す (読み込みは終わった)。ファイルは残し、段階ごとに書き直す。
    pub fn endLoad(p: *Progress) void {
        p.ch.deinit();
    }

    pub fn deinit(p: *Progress) void {
        p.ch.deinit();
        p.gpa.free(p.done);
    }

    fn write(p: *Progress) void {
        const path = p.path orelse return;
        p.writeFile(path) catch |err| if (!p.warned) {
            p.warned = true;
            std.log.warn("glm53: load progress: cannot write {s}: {s}", .{ path, @errorName(err) });
        };
    }

    fn writeFile(p: *Progress, path: []const u8) !void {
        var n: usize = 0;
        for (p.done) |x| n += x;
        const b64 = std.base64.standard.Encoder;
        const done64 = try p.gpa.alloc(u8, b64.calcSize(p.done.len));
        defer p.gpa.free(done64);
        _ = b64.encode(done64, p.done);
        const ms = @max(0, p.t0.durationTo(std.Io.Clock.awake.now(p.io)).toMilliseconds());
        var aw: std.Io.Writer.Allocating = .init(p.gpa);
        defer aw.deinit();
        const w = &aw.writer;
        try w.print("{{\"format\": \"glm53-load-progress\", \"version\": 1, \"pid\": {d}, \"phase\": \"{s}\", \"layers\": {d}, \"layers_done\": {d}, \"done\": \"{s}\", \"experts_done\": \"\", \"read_bytes\": {d}, \"total_bytes\": {d}, \"engine_ms\": {d}, \"plan\": ", .{ std.c.getpid(), @tagName(p.phase), p.done.len, n, done64, weights.read_bytes, p.total_bytes, ms });
        try std.json.Stringify.value(p.plan, .{}, w);
        if (p.plan.len > 0) try w.print(", \"plan_digest\": \"{x}\"}}\n", .{p.plan_digest}) else try w.writeAll(", \"plan_digest\": \"\"}\n");
        var tbuf: [4096]u8 = undefined;
        const tmp = try std.fmt.bufPrint(&tbuf, "{s}.tmp", .{path});
        const cwd = std.Io.Dir.cwd();
        try cwd.writeFile(p.io, .{ .sub_path = tmp, .data = aw.written() });
        try cwd.rename(tmp, cwd, path, p.io);
    }
};
