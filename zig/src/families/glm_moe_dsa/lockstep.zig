//! How the ranks agree on a request: rank 0's header, then its prompt ids, each shared by int32 all-gathers, length
//! first. At start-up the ranks also check that they loaded the same things.
//! ランクどうしが要求をそろえる手順: rank 0 のヘッダ、続いて prompt の id を、それぞれ int32 の all-gather で配る
//! (長さが先)。起動時には、全ランクが同じものを読み込んだかも確かめる。
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const slots = @import("slots_serve.zig");
const pool_serve = @import("pool_serve.zig");
const kvdigest = @import("kvdigest.zig");
const poison = @import("poison.zig");
const pool_plan = @import("pool/plan.zig");
const expert_swap = @import("expert_swap.zig");
const pace_mod = @import("pace.zig");

/// A request's settings as 17 non-negative words: max_tokens, stop_eos, k | mode << 8, seed (3), temperature (3),
/// top_k, top_p (3), min_p (3), r0 (the slot cache's resume point). max_tokens < 0 tells the followers to stop.
/// 要求の設定を負でない 17 語で表す: max_tokens、stop_eos、k | mode << 8、seed (3)、temperature (3)、top_k、
/// top_p (3)、min_p (3)、r0 (スロットキャッシュの再開位置)。max_tokens < 0 はフォロワーへの停止指示。
pub const Header = struct {
    max_tokens: i32,
    stop_eos: bool = true,
    k: u8 = 0,
    mode: u8 = 0,
    seed: u64 = 0,
    temperature: f64 = 0,
    top_k: i32 = 0,
    top_p: f64 = 1,
    min_p: f64 = 0,
    r0: i32 = 0,
    /// The slot cache's 21 words, present when the engine keeps slots: resume flag, metadata digest (8), plan digest
    /// (8), drafter depth, confidence (3).
    /// スロットキャッシュの 21 語。エンジンがスロットを持つときだけある: resume フラグ、メタデータのダイジェスト (8)、
    /// 計画のダイジェスト (8)、drafter の深さ、信頼度 (3)。
    extra: ?[21]i32 = null,
    /// Fiore-lab, rank 0 only (not one of the words): the pool slot this request's reply wrote before a re-admission,
    /// which the pool continues (pool_serve.lead puts it into the slot words' first).
    /// Fiore-lab、rank 0 だけ (語には入らない): この要求の応答が受け入れ直しの前に書いた pool のスロット。pool はそれを
    /// 続ける (pool_serve.lead がスロットの語の先頭に入れる)。
    own_slot: ?u64 = null,

    pub const words = 17;

    /// The 17 words, then the slot cache's 21 when present.
    /// 17 語。スロットキャッシュの 21 語があれば続ける。
    pub fn encodeAll(h: Header, out: *[38]i32) []const i32 {
        out[0..17].* = h.encode();
        if (h.extra) |x| {
            out[17..38].* = x;
            return out[0..38];
        }
        return out[0..17];
    }

    pub fn encode(h: Header) [words]i32 {
        var w: [words]i32 = undefined;
        w[0] = h.max_tokens;
        w[1] = @intFromBool(h.stop_eos);
        w[2] = @as(i32, h.k) | (@as(i32, h.mode) << 8);
        w[3] = @intCast(h.seed & 0x7FFFFFFF);
        w[4] = @intCast((h.seed >> 31) & 0x7FFFFFFF);
        w[5] = @intCast(h.seed >> 62);
        f64Words(h.temperature, w[6..9]);
        w[9] = h.top_k;
        f64Words(h.top_p, w[10..13]);
        f64Words(h.min_p, w[13..16]);
        w[16] = h.r0;
        return w;
    }

    pub fn decode(w: []const i32) !Header {
        if (w.len < words) return error.ShortHeader;
        return .{
            .max_tokens = w[0],
            .stop_eos = w[1] != 0,
            .k = @intCast(w[2] & 0xFF),
            .mode = @intCast((w[2] >> 8) & 0xFF),
            .seed = @as(u64, @intCast(w[3])) | (@as(u64, @intCast(w[4])) << 31) | (@as(u64, @intCast(w[5])) << 62),
            .temperature = wordsF64(w[6..9]),
            .top_k = w[9],
            .top_p = wordsF64(w[10..13]),
            .min_p = wordsF64(w[13..16]),
            .r0 = w[16],
            .extra = if (w.len >= 38) w[17..38].* else null,
        };
    }
};

/// An f64's bits as three non-negative words: low 31, next 31, top 2.
/// f64 のビットを負でない 3 語に分ける: 下位 31、次の 31、上位 2 ビット。
pub fn f64Words(v: f64, out: []i32) void {
    const b: u64 = @bitCast(v);
    out[0] = @intCast(b & 0x7FFFFFFF);
    out[1] = @intCast((b >> 31) & 0x7FFFFFFF);
    out[2] = @intCast(b >> 62);
}

pub fn wordsF64(w: []const i32) f64 {
    const b = @as(u64, @intCast(w[0])) | (@as(u64, @intCast(w[1])) << 31) | (@as(u64, @intCast(w[2])) << 62);
    return @bitCast(b);
}

/// The all-gathers' device buffers (grown as needed) on the engine's stream.
/// all-gather のデバイスバッファ (必要に応じて拡張)。エンジンの stream で動く。
pub const Channel = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    group: *const cuda.tp.Group,
    stream: cuda.Stream,
    send: ?cuda.DeviceBuffer = null,
    recv: ?cuda.DeviceBuffer = null,
    cap: usize = 0, // int32 words per rank / ランクごとの int32 の語数

    pub fn init(gpa: std.mem.Allocator, e: *const engine.Engine) Channel {
        return .{ .gpa = gpa, .d = e.d, .group = e.group, .stream = e.stream };
    }

    fn reserve(ch: *Channel, n: usize) !void {
        if (n <= ch.cap) return;
        const want = @max(n, 1024);
        // the new pair, then the old one freed (each all-gather ends with a synchronize): a prompt longer than any
        // before grows the pair, and each conversation turn can be one
        // 新しい組を確保してから古い組を解放する (all-gather はどれも synchronize で終わる): それまでより長い prompt が
        // 来るたびに組は大きくなり、会話の 1 ターンごとにそうなりうる
        var send = try cuda.DeviceBuffer.alloc(ch.d, want * 4);
        errdefer send.free();
        const recv = try cuda.DeviceBuffer.alloc(ch.d, want * 4 * ch.group.world);
        ch.deinit();
        ch.send = send;
        ch.recv = recv;
        ch.cap = want;
    }

    /// Frees the device buffers; the channel grows them again when used.
    /// デバイスバッファを解放する。使えばチャネルはまた確保する。
    pub fn deinit(ch: *Channel) void {
        if (ch.send) |*b| b.free();
        if (ch.recv) |*b| b.free();
        ch.send = null;
        ch.recv = null;
        ch.cap = 0;
    }

    /// Every rank's `mine` (n words each) gathered in rank order into `out` (n * world words).
    /// 各ランクの `mine` (各 n 語) をランク順に `out` (n * world 語) へ集める。
    pub fn allGather(ch: *Channel, mine: []const i32, out: []i32) !void {
        const n = mine.len;
        if (out.len != n * ch.group.world) return error.BadLength;
        try ch.reserve(n);
        try ch.send.?.uploadAsync(0, std.mem.sliceAsBytes(mine), ch.stream.handle);
        const lib = ch.group.lib;
        try lib.check(lib.api.ncclAllGather(ch.send.?.ptr, ch.recv.?.ptr, n, .i32, ch.group.comm, ch.stream.handle), "ncclAllGather");
        try ch.recv.?.downloadAsync(0, std.mem.sliceAsBytes(out), ch.stream.handle);
        try ch.stream.synchronize();
    }

    /// Rank 0's `values` to every rank (others pass null), length first; the caller owns the result.
    /// rank 0 の `values` を全ランクへ配る (他のランクは null を渡す)。長さが先。結果は呼び出し側が持つ。
    pub fn share(ch: *Channel, values: ?[]const i32) ![]i32 {
        const W = ch.group.world;
        const lens = try ch.gpa.alloc(i32, W);
        defer ch.gpa.free(lens);
        const mine_len: i32 = if (values) |v| @intCast(v.len) else 0;
        try ch.allGather(&.{mine_len}, lens);
        const n: usize = @intCast(lens[0]);
        const out = try ch.gpa.alloc(i32, n);
        errdefer ch.gpa.free(out);
        if (n == 0) return out;
        const mine = try ch.gpa.alloc(i32, n);
        defer ch.gpa.free(mine);
        if (values) |v| @memcpy(mine, v) else @memset(mine, 0);
        const all = try ch.gpa.alloc(i32, n * W);
        defer ch.gpa.free(all);
        try ch.allGather(mine, all);
        @memcpy(out, all[0..n]);
        return out;
    }

    /// Whether every rank's flag is set.
    /// 全ランクのフラグが立っているか。
    pub fn allOk(ch: *Channel, ok: bool) !bool {
        var all: [64]i32 = undefined;
        try ch.allGather(&.{@intFromBool(ok)}, all[0..ch.group.world]);
        for (all[0..ch.group.world]) |v| if (v == 0) return false;
        return true;
    }
};

/// What a rank loaded (sha256 of config.json, of the weights index and of the build id), world and rank; the ranks
/// start only when the digests agree everywhere and the ranks are 0 .. world - 1 in order.
/// ランクが読み込んだもの (config.json・重みの index・ビルド ID の sha256) と world・rank。ダイジェストが全ランクで
/// 一致し、ランクが 0 .. world - 1 の順に並ぶときだけ起動する。
pub const Fingerprint = struct {
    words: [19]i32,

    pub fn of(io: std.Io, gpa: std.mem.Allocator, ckpt: []const u8, build_id: []const u8, world: u32, rank: u32) !Fingerprint {
        var f: Fingerprint = .{ .words = undefined };
        try fileDigest(io, gpa, ckpt, "config.json", f.words[0..4]);
        try fileDigest(io, gpa, ckpt, "model.safetensors.index.json", f.words[4..8]);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(build_id, &d, .{});
        digestWords(d, f.words[8..17]);
        f.words[17] = @intCast(world);
        f.words[18] = @intCast(rank);
        return f;
    }

    /// Every rank calls it; error.Mismatch on all of them when any differs.
    /// 全ランクが呼ぶ。どれかが違えば全ランクで error.Mismatch になる。
    pub fn agree(f: Fingerprint, ch: *Channel) !void {
        const W = ch.group.world;
        const all = try ch.gpa.alloc(i32, 19 * W);
        defer ch.gpa.free(all);
        try ch.allGather(&f.words, all);
        for (0..W) |r| {
            const row = all[r * 19 ..][0..19];
            if (!std.mem.eql(i32, row[0..18], all[0..18]) or row[18] != @as(i32, @intCast(r))) return error.Mismatch;
        }
    }
};

fn digestWords(d: [32]u8, out: []i32) void {
    for (out, 0..) |*w, i| w.* = @bitCast(std.mem.readInt(u32, d[(i * 4) % 32 ..][0..4], .little) & 0x7FFFFFFF);
}

fn fileDigest(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, name: []const u8, out: []i32) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    var d: [32]u8 = @splat(0);
    if (std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30))) |text| {
        defer gpa.free(text);
        std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    } else |err| if (err != error.FileNotFound) return err;
    digestWords(d, out);
}

/// The header mode that asks for DFlash2 drafts; 0 decodes serially.
/// DFlash2 の draft を求めるヘッダの mode。0 は逐次 decode。
pub const mode_dflash: u8 = 9;
/// The header mode that asks for drafts from the checkpoint's MTP layer (TF_GLM53_DRAFTER=mtp).
/// チェックポイントの MTP 層の draft を求めるヘッダの mode (TF_GLM53_DRAFTER=mtp)。
pub const mode_mtp: u8 = 4;
/// DFlash2's draft tokens a round, the same on every rank.
/// DFlash2 が 1 ラウンドに出す draft トークン数。全ランクで同じ。
pub const dflash_depth: usize = 7;

/// The slot words 17..21 rank 0 sends: DFlash2's depth and chain confidence (Engine.dflash_conf), or with the MTP
/// layer depth 0 and its chain confidence (Engine.mtp_conf, TF_GLM53_MTP_CONF: 0 when off, the words as before); 0
/// without either. The two are never loaded together (engine error.TwoDrafters).
/// rank 0 が送るスロットの語 17..21: DFlash2 の深さと連鎖の信頼度 (Engine.dflash_conf)。MTP 層なら深さ 0 とその連鎖の
/// 信頼度 (Engine.mtp_conf、TF_GLM53_MTP_CONF: 無効なら 0 で語は従来と同じ)。どちらもなければ 0。両方を同時に読み込む
/// ことはない (engine の error.TwoDrafters)。
pub fn draftWords(e: *const engine.Engine, out: []i32) void {
    out[0] = if (e.drafter != null) @intCast(dflash_depth) else 0;
    f64Words(if (e.drafter != null) e.dflash_conf else if (e.mtp != null) e.mtp_conf else 0, out[1..4]);
}

/// On every rank, rank 0's chain confidence from the slot words, so the ranks draft alike whatever their own
/// TF_GLM53_DFLASH_CONF / TF_GLM53_MTP_CONF (an MTP chain's steps are collectives: the ranks must stop it alike).
/// 全ランクで、スロットの語から rank 0 の連鎖の信頼度を取る。各ランクの TF_GLM53_DFLASH_CONF / TF_GLM53_MTP_CONF に
/// よらず同じ draft になる (MTP の連鎖の段は collective なので、全ランクが同じ所で止めねばならない)。
pub fn takeDraftWords(e: *engine.Engine, x: []const i32) void {
    if (e.drafter != null and x[0] != 0) e.dflash_conf = wordsF64(x[1..4]);
    if (e.drafter == null and e.mtp != null) e.mtp_conf = wordsF64(x[1..4]);
}

/// One request on this rank: the slot cache's agreement first when there is one, then the decode the header asks for.
/// このランクでの 1 要求: スロットキャッシュがあればその合意を先に行い、次にヘッダが求める decode を行う。
fn run(ch: *Channel, e: *engine.Engine, h: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer, plan: ?*slots.Plan, pplan: ?*pool_plan.Plan) !engine.Stats {
    if (e.pool != null) return pool_serve.serve(ch, e, h, prompt, sink, prop, pplan, runPlain);
    // TF_GLM53_POOL_POISON (tests): prefix off poisons every row before every request (poison.zig)
    // TF_GLM53_POOL_POISON (試験用): prefix off は要求ごとに全行を毒で埋める (poison.zig)
    if (e.slots == null) if (poison.byte()) |v| try poison.all(e, v);
    const st = if (e.slots != null) try slots.serve(ch, e, h, prompt, sink, prop, plan, runPlain) else try runPlain(e, h, prompt, sink, prop);
    if (kvdigest.on()) {
        // the rows at [0, V) contiguous, or through TF_GLM53_POOL_TEST's fixed table
        // 行は [0, V) に連続、または TF_GLM53_POOL_TEST の決まった表を通す
        const V = kvdigest.validOf(e, prompt.len, st);
        try kvdigest.report(e, V, @min(V, prompt.len), if (e.mtp) |*m| m.track.kvEnd(V) else V, try kvdigest.testTable(e));
    }
    return st;
}

/// temperature > 0: keyed draws (the same tokens on every rank, drafted or not); else greedy (null).
/// temperature > 0 ならキー付きの抽選 (draft の有無によらず全ランクで同じトークン)。他は greedy (null)。
fn samplingOf(h: Header) ?engine.Sampling {
    return if (h.temperature > 0) .{ .seed = h.seed, .temperature = h.temperature, .top_k = @intCast(@max(0, h.top_k)), .top_p = h.top_p, .min_p = h.min_p } else null;
}

fn runPlain(e: *engine.Engine, h: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) !engine.Stats {
    e.samp = samplingOf(h);
    defer e.samp = null;
    if (h.mode == mode_dflash) {
        const p = prop orelse e.drafterProposer() orelse return error.NoDrafter;
        return e.generateDrafted(prompt, @intCast(h.max_tokens), h.stop_eos, sink, p, dflash_depth);
    }
    if (h.mode == mode_mtp) {
        const p = prop orelse e.mtpProposer() orelse return error.NoMtp;
        return e.generateDrafted(prompt, @intCast(h.max_tokens), h.stop_eos, sink, p, e.mtp.?.k);
    }
    if (h.mode != 0) return error.UnportedMode;
    return e.generate(prompt, @intCast(h.max_tokens), h.stop_eos, sink);
}

/// Rank 0: shares one request (header, then prompt) with every rank, then runs its own part.
/// rank 0: 1 つの要求 (ヘッダ、続いて prompt) を全ランクと共有し、自分の分を実行する。
pub fn lead(ch: *Channel, e: *engine.Engine, h0: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) !engine.Stats {
    var h = h0;
    // the slot cache's plan comes before anything is shared, so a failure here touches no rank
    // スロットキャッシュの計画は共有の前に立てるので、ここで失敗してもどのランクにも影響しない
    var plan: ?slots.Plan = null;
    defer if (plan) |*p| p.deinit();
    var pplan: ?pool_plan.Plan = null;
    defer if (pplan) |*p| p.deinit(e.gpa);
    if (e.pool != null) pplan = pool_serve.lead(e, &h, prompt) catch |err| blk: {
        // go on cold: words that match no rank's metadata
        // cold で続行: どのランクのメタデータとも一致しない語を送る
        std.debug.print("[tensorfold] pool: rank 0 could not plan ({s}) - every rank goes cold\n", .{@errorName(err)});
        var x: [21]i32 = @splat(0);
        x[0] = 1;
        h.extra = x;
        h.r0 = 0;
        break :blk null;
    };
    if (e.slots != null) plan = slots.lead(e, &h, prompt) catch |err| blk: {
        // go on cold: slot words that match no rank's metadata
        // cold で続行: どのランクのメタデータとも一致しないスロットの語を送る
        std.debug.print("[tensorfold] slots: rank 0 could not plan ({s}) - every rank goes cold\n", .{@errorName(err)});
        var x: [21]i32 = @splat(0);
        x[0] = 1;
        h.extra = x;
        h.r0 = 0;
        break :blk null;
    };
    // refuse prompt chunk shapes not run at start-up, before the followers see the request
    // 起動時に試していない prompt チャンクの形は、フォロワーが要求を見る前に断る
    try e.admit(prompt.len, @intCast(@max(0, h.r0)));
    var buf: [38]i32 = undefined;
    ch.gpa.free(try ch.share(h.encodeAll(&buf)));
    const ids = try ch.gpa.alloc(i32, prompt.len);
    defer ch.gpa.free(ids);
    for (ids, prompt) |*w, t| w.* = @intCast(t);
    ch.gpa.free(try ch.share(ids));
    return run(ch, e, h, prompt, sink, prop, if (plan) |*p| p else null, if (pplan) |*p| p else null);
}

/// Rank 0: tells the followers to return from `follow`.
/// rank 0: フォロワーに `follow` から戻るよう伝える。
pub fn shutdown(ch: *Channel) !void {
    const hw = (Header{ .max_tokens = -1 }).encode();
    ch.gpa.free(try ch.share(&hw));
}

/// The header's max_tokens for a change of the routed experts' bits (expert_swap); the table follows it.
/// routed expert の bit の変更 (expert_swap) を示すヘッダの max_tokens。表がその後に続く。
pub const swap_tokens: i32 = -2;

/// Rank 0, between requests: every rank's routed experts to `target` (one byte an expert, the MoE layers in order;
/// expert_swap), then the cached prefixes dropped. A table any rank refuses changes nothing; a rank short of memory or
/// failing stops every rank at the same layer (the layers before it changed on all; the error says why).
/// rank 0、要求の合間に: 全ランクの routed expert を `target` にし (expert ごとに 1 バイト、MoE 層の順。expert_swap)、
/// 保存した prefix を捨てる。どれかのランクが断る表は何も変えない。メモリ不足や失敗のランクがあれば全ランクが同じ層で
/// 止まる (それより前の層は全ランクで変わっている。エラーが理由を示す)。
pub fn swap(ch: *Channel, e: *engine.Engine, target: []const u8, progress: ?expert_swap.Progress, st: *expert_swap.Stats) !void {
    // a failure before every rank is in swapEvery leaves the followers waiting on rank 0: not alike
    // 全ランクが swapEvery に入る前の失敗では、フォロワーが rank 0 を待ったまま: alike ではない
    st.alike = false;
    const hw = (Header{ .max_tokens = swap_tokens }).encode();
    ch.gpa.free(try ch.share(&hw));
    return swapShared(ch, e, target, progress, st);
}

/// `swap` with TF_GLM53_MULTI, nothing live or waiting: the followers hear of it as an event (followMulti).
/// TF_GLM53_MULTI での `swap`。使用中も待ちもないとき: フォロワーには出来事として伝える (followMulti)。
pub fn swapMulti(ch: *Channel, e: *engine.Engine, target: []const u8, progress: ?expert_swap.Progress, st: *expert_swap.Stats) !void {
    st.alike = false;
    const w = [1]i32{ev_swap};
    ch.gpa.free(try ch.share(&w));
    return swapShared(ch, e, target, progress, st);
}

/// Rank 0: the table shared, then the change on every rank.
/// rank 0: 表を配り、全ランクで変える。
fn swapShared(ch: *Channel, e: *engine.Engine, target: []const u8, progress: ?expert_swap.Progress, st: *expert_swap.Stats) !void {
    const words = try ch.gpa.alloc(i32, (target.len + 3) / 4);
    defer ch.gpa.free(words);
    @memset(words, 0);
    @memcpy(std.mem.sliceAsBytes(words)[0..target.len], target);
    ch.gpa.free(try ch.share(words));
    return swapEvery(ch, e, target, progress, st);
}

/// A follower, told of a swap: rank 0's table, the change. An error only when the ranks may differ.
/// フォロワー、入れ替えを告げられて: rank 0 の表を受けて変える。エラーはランクが食い違いうるときだけ。
fn followSwap(ch: *Channel, e: *engine.Engine) !void {
    const words = try ch.share(null);
    defer ch.gpa.free(words);
    const bytes = std.mem.sliceAsBytes(words);
    var st: expert_swap.Stats = .{};
    swapEvery(ch, e, bytes[0..@min(bytes.len, e.xbits.len)], null, &st) catch |err| {
        std.log.err("glm53: rank {d}: expert swap: {s}", .{ e.rank, @errorName(err) });
        // the ranks may differ: stop (the host's fatal), never serve on / ランクが食い違いうる: 止める
        if (!st.alike) return err;
        return;
    };
    std.log.info("glm53: rank {d}: expert swap: {d} experts in {d} layers ({d} up, {d} down), {d} ms", .{ e.rank, st.experts, st.layers, st.up, st.down, st.ms });
}

/// Every rank with the same `target` at the same point (the followers of `swap`, and harnesses whose ranks all run
/// the same code): check and agree, change layer by layer in agreement, drop the cached prefixes.
/// 全ランクが同じ `target` を同じ時点で (`swap` のフォロワーと、全ランクが同じコードを走らせるハーネス): 確かめて
/// 合意し、層ごとに合意しながら変え、保存した prefix を捨てる。
pub fn swapEvery(ch: *Channel, e: *engine.Engine, target: []const u8, progress: ?expert_swap.Progress, st: *expert_swap.Stats) !void {
    // a deadline: a rank stuck past it (a storage read or an agreement that never returns) ends its process, so the
    // lane's watch stops every rank instead of the lane waiting for ever
    // 期限: それを過ぎて止まったランク (戻らないストレージの読み込みや合意) はプロセスを終える。レーンの watch が全ランクを
    // 止め、レーンがいつまでも待ち続けることはない
    const dl = try Deadline.start(e.gpa, e.io, e.rank);
    defer dl.done();
    st.alike = true;
    var why: ?anyerror = null;
    expert_swap.check(e, target) catch |err| {
        why = err;
    };
    const all_ok = ch.allOk(why == null) catch |err| {
        st.alike = false;
        return err;
    };
    if (!all_ok) return why orelse error.SwapRefusedByAnotherRank;
    // every changed layer agreed on by all ranks (expert_swap.Sync), so a stop leaves them alike
    // 変わる層ごとに全ランクで合意する (expert_swap.Sync)。止まっても全ランクが同じ状態
    const Agree = struct {
        fn f(ctx: *anyopaque, ok: bool) anyerror!bool {
            const c: *Channel = @ptrCast(@alignCast(ctx));
            return c.allOk(ok);
        }
    };
    expert_swap.apply(e, target, progress, .{ .ctx = ch, .f = Agree.f }, st) catch |err| {
        std.log.err("glm53: rank {d}: changing the experts' bits stopped after {d} layers: {s}", .{ e.rank, st.layers, @errorName(err) });
        // a lack of memory agreed on before a layer changed leaves every rank alike; any other failure may leave a
        // rank's layer half-changed or the ranks apart (a layer changed on the ranks that read it, not on the one
        // that failed)
        // 層を変える前に合意したメモリ不足なら全ランクは同じ。それ以外の失敗では、あるランクの層が途中まで変わって
        // いるか、ランクが食い違う (読めたランクでは層が変わり、失敗したランクでは変わっていない)
        st.alike = err == error.SwapLowMemory or err == error.SwapLowMemoryOnAnotherRank;
        e.dropCaches() catch |e2| {
            std.log.err("glm53: rank {d}: the cached prefixes were not dropped after the swap: {s}", .{ e.rank, @errorName(e2) });
            st.alike = false;
        };
        return err;
    };
    e.dropCaches() catch |err| {
        st.alike = false;
        return err;
    };
}

/// TF_GLM53_SWAP_DEADLINE_S (default 600; a full change of 75 layers took 83-90 s on TP4): a detached watch of
/// one swap that ends the process (exit 71) once it is past; `done` lets the watch go (it frees itself).
/// TF_GLM53_SWAP_DEADLINE_S (既定 600。75 層全部の変更は TP4 で 83-90 秒): 1 回の入れ替えの切り離した見張り。
/// 過ぎたらプロセスを終える (exit 71)。`done` で見張りを放す (見張りは自分で解放する)。
const Deadline = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rank: u32,
    limit_s: u32,
    over: std.atomic.Value(bool) = .init(false),

    fn start(gpa: std.mem.Allocator, io: std.Io, rank: u32) !*Deadline {
        const limit: u32 = if (std.c.getenv("TF_GLM53_SWAP_DEADLINE_S")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 600 else 600;
        const d = try gpa.create(Deadline);
        d.* = .{ .gpa = gpa, .io = io, .rank = rank, .limit_s = limit };
        // 1 MiB as the server's other small threads: 64 KiB is under aarch64 glibc's PTHREAD_STACK_MIN (128 KiB) and the
        // spawn panics / サーバーのほかの小さなスレッドと同じ 1 MiB: 64 KiB は aarch64 glibc の下限より小さい
        const t = std.Thread.spawn(.{ .stack_size = 1 << 20 }, watch, .{d}) catch |err| {
            gpa.destroy(d);
            return err;
        };
        t.detach();
        return d;
    }

    fn done(d: *Deadline) void {
        d.over.store(true, .release);
    }

    fn watch(d: *Deadline) void {
        var s: u32 = 0;
        while (!d.over.load(.acquire)) : (s += 1) {
            if (s >= d.limit_s) {
                std.log.err("glm53: rank {d}: an expert swap is still running after {d} s: ending the process (the lane's watch stops every rank)", .{ d.rank, d.limit_s });
                std.process.exit(71);
            }
            std.Io.sleep(d.io, .fromMilliseconds(1000), .awake) catch {};
        }
        d.gpa.destroy(d);
    }
};

/// What a follower received, for tests that check every rank got the same request.
/// フォロワーが受け取ったもの。全ランクが同じ要求を受けたかを確かめるテスト用。
pub const Received = struct {
    header: Header,
    prompt_sha256: [32]u8,
    stats: engine.Stats,
};

/// Ranks above 0: run each request as rank 0 does, without output, until rank 0 says stop. `seen` (if any) gets each
/// request's header and the digest of the prompt received.
/// rank 1 以上: rank 0 が停止を告げるまで、各要求を rank 0 と同じく出力なしで実行する。`seen` があれば、各要求の
/// ヘッダと受け取った prompt のダイジェストを渡す。
pub fn follow(ch: *Channel, e: *engine.Engine, seen: ?*std.ArrayList(Received), prop: ?engine.Proposer) !void {
    while (true) {
        const hw = try ch.share(null);
        defer ch.gpa.free(hw);
        const h = try Header.decode(hw);
        if (h.max_tokens == swap_tokens) {
            try followSwap(ch, e);
            continue;
        }
        if (h.max_tokens < 0) return;
        const words = try ch.share(null);
        defer ch.gpa.free(words);
        const prompt = try ch.gpa.alloc(i64, words.len);
        defer ch.gpa.free(prompt);
        for (prompt, words) |*t, w| t.* = w;
        const st = try run(ch, e, h, prompt, null, prop, null, null);
        if (seen) |list| {
            var d: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(prompt), &d, .{});
            try list.append(ch.gpa, .{ .header = h, .prompt_sha256 = d, .stats = st });
        }
    }
}

// ------------------------------------------------- several conversations at once (TF_GLM53_MULTI) / 複数の会話 ---
//
// Rank 0 shares each step as an event, in the order it runs it; every rank runs the same steps in that order:
// admit (conversation k's header and prompt: its plan, prefill and first token), round (the live conversations and
// rank 0's stop for each: one batched round), release (conversation k's commit), single (one request the old way:
// header, prompt, run), shutdown.
// rank 0 は各段を出来事として実行する順に配り、全ランクが同じ段を同じ順に実行する: admit (会話 k のヘッダと prompt:
// プラン、prefill、最初のトークン)、round (使用中の会話とそれぞれへの rank 0 の停止: まとめたラウンド 1 回)、release
// (会話 k の確定)、single (従来の 1 要求: ヘッダ、prompt、実行)、shutdown。

pub const ev_shutdown: i32 = 0;
pub const ev_admit: i32 = 1;
pub const ev_round: i32 = 2;
pub const ev_release: i32 = 3;
pub const ev_single: i32 = 4;
pub const ev_mid_end: i32 = 5; // the end of a round inside a prefill (and its releases) / prefill の中のラウンド (と確定) の終わり
pub const ev_swap: i32 = 6; // the routed experts' bits changed, nothing live or waiting / routed expert の bit の変更。使用中も待ちもない

/// Rank 0's live conversations for a round inside another's prefill (the host fills `ks`, `stops`, `sinks`; returns
/// how many): every one live and not done, the conversation being admitted left out.
/// 別の会話の prefill の中のラウンドに使う、rank 0 の使用中の会話 (ホストが `ks`・`stops`・`sinks` を埋め、数を返す):
/// 使用中で未完了のものすべて。受け付け中の会話は除く。
pub const Live = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, skip: usize, ks: []u8, stops: []bool, sinks: []?engine.Sink) usize,
    // a conversation the round finished: the host releases it (leadRelease) at once, its reply not held for the prefill
    // ラウンドで終わった会話: ホストがすぐに返す (leadRelease)。応答を prefill の終わりまで待たせない
    release: *const fn (ctx: *anyopaque, k: usize) void,
    // the host's counting of the time between the prefill and those rounds, and the decode's share (percent, read
    // anew at each round so a change reaches a running prefill); null: no pause, no round
    // prefill とそのラウンドの時間を数えるホストの Pace と、decode の取り分 (%。ラウンドごとに読み直すので、変更は実行中の
    // prefill に届く)。null: 止まらず、ラウンドもない
    pace: ?*pace_mod.Pace = null,
    share: ?*const std.atomic.Value(u8) = null,
    // a short prompt (the host's TF_GLM53_MULTI_SHORT_PROMPT): its prefill runs to its end with no pause (its time
    // still counted)
    // 短い prompt (ホストの TF_GLM53_MULTI_SHORT_PROMPT): その prefill は止まらずに終わりまで進む (時間は数える)
    straight: bool = false,
    // whether this prefill, `left` rows from its end, gives way to a short prompt waiting (true: the host has taken
    // it as given way and puts the request back in its queue when the prefill ends); null: never
    // この prefill (終わりまで `left` 行) が、待っている短い prompt に譲るか (true: ホストは譲ったものとして扱い、prefill
    // が終わったら要求を待ち行列へ戻す)。null: 譲らない
    yield: ?*const fn (ctx: *anyopaque, left: u64) bool = null,
    // the time this prefill spent paused for the others (the rounds, a hold by the share), added at each pause (rank
    // 0's scheduler thread): its chunks' own time is the rest
    // この prefill がほかのために止まっていた時間 (ラウンド、取り分による待ち)。止まる所ごとに足す (rank 0 のスケジューラ
    // のスレッド): チャンク自体の時間は残り
    paused: ?*i96 = null,
};

const LeadMid = struct {
    ch: *Channel,
    e: *engine.Engine,
    k: usize,
    live: Live,

    fn now(m: *const LeadMid) i96 {
        return std.Io.Clock.awake.now(m.e.io).toNanoseconds();
    }

    fn share(m: *const LeadMid) u8 {
        return if (m.live.share) |s| s.load(.monotonic) else 0;
    }

    fn liveCount(m: *LeadMid) usize {
        var ks: [engine.multi_max]u8 = undefined;
        var stops: [engine.multi_max]bool = undefined;
        var sinks: [engine.multi_max]?engine.Sink = undefined;
        return m.live.f(m.live.ctx, m.k, &ks, &stops, &sinks);
    }

    /// Whether this prefill gives way to a short prompt waiting (the host's `yield`): never a short prompt itself;
    /// only with more rows left than the drafter's window (its resume then needs none of the drafter's saved rows);
    /// and, unless the share holds the prefill anyway (100), only once a chunk of this admission is done, so each
    /// admission gets on however many short prompts come.
    /// この prefill が待っている短い prompt に譲るか (ホストの `yield`): 短い prompt 自身は譲らない。残りの行が drafter の
    /// ウィンドウより多いときだけ (それなら再開に drafter の保存した行が要らない)。取り分がどうせ prefill を止めている
    /// (100) とき以外は、この受け入れでチャンクが 1 つ終わってから (短い prompt がいくつ来ても、受け入れごとに進む)。
    fn yields(m: *LeadMid) bool {
        if (m.live.straight) return false;
        const f = m.live.yield orelse return false;
        const e = m.e;
        const cv = &e.multi.?.convs[m.k];
        const left = cv.prompt.items.len -| e.prefill_done;
        const win: usize = if (cv.drafted) (if (e.drafter) |dr| @as(usize, @intCast(dr.cfg.window)) + 1 else 0) else 0;
        if (left <= win) return false;
        if (e.prefill_done <= e.start and m.share() < 100) return false;
        return f(m.live.ctx, left);
    }

    /// A chunk's start: the segment that ran to the last chunk's end counted (not as a layer's measure: it stopped
    /// at the chunk's end), the layers before this chunk's first pause (0, none, for a short prompt: its chunk is
    /// counted whole and measures no layer, so a long prompt's measure is kept); null: it gives way before this chunk.
    /// チャンクの始め: 前のチャンクの終わりまで進んだ区間を数え (チャンクの終わりで止まったので 1 層の目安にはしない)、
    /// このチャンクで最初に止まるまでの層数を返す (短い prompt は 0 で止まらない。チャンクは丸ごと数え、1 層は測らないので、
    /// 長い prompt で測った値が残る)。null: このチャンクの前で譲る。
    fn first(ctx: *anyopaque, rows: u32) ?u8 {
        const m: *LeadMid = @ptrCast(@alignCast(ctx));
        const pc = m.live.pace orelse return 0;
        const s = m.share();
        const t = m.now();
        pc.cut(s, t);
        if (m.yields()) return null;
        if (!m.live.straight) pc.chunk(rows);
        if (m.liveCount() == 0) {
            pc.idle();
            return 0;
        }
        if (m.live.straight) {
            pc.begin(t, 0);
            return 0;
        }
        const n = pc.layers(s, true);
        pc.begin(t, n);
        return n;
    }

    /// A pause: the segment counted, then rounds of the other conversations while the decode is behind its share
    /// (100: while any decodes, unless this prompt's request is being stopped), then the layers to the next pause,
    /// shared with the end. null: the prefill gives way here (a short prompt waits), this chunk's work dropped; -1 in
    /// the end tells the other ranks.
    /// 止まる所: 区間を数え、decode が取り分に届かない間はほかの会話のラウンドを回し (100 ならどれかが decode している間。
    /// ただしこの prompt の要求が止められているときは回さない)、次に止まるまでの層数を終わりの合図と一緒に配る。null:
    /// prefill はここで譲る (短い prompt が待っている)。このチャンクでしたことは捨てる。終わりの合図の -1 でほかのランクに
    /// 伝える。
    fn run(ctx: *anyopaque) anyerror!?u8 {
        const m: *LeadMid = @ptrCast(@alignCast(ctx));
        const t_in = m.now();
        defer if (m.live.paused) |p| {
            p.* += m.now() - t_in;
        };
        var ks: [engine.multi_max]u8 = undefined;
        var stops: [engine.multi_max]bool = undefined;
        var sinks: [engine.multi_max]?engine.Sink = undefined;
        var n = m.live.f(m.live.ctx, m.k, &ks, &stops, &sinks);
        if (m.live.pace) |pc| pc.end(m.share(), m.now());
        var give = m.yields();
        while (n > 0 and !give) {
            const s = m.share();
            const pc = m.live.pace orelse break;
            if (!pc.behind(s) or m.e.stop_req.load(.monotonic)) break;
            const t0 = m.now();
            try leadRound(m.ch, m.e, ks[0..n], stops[0..n], sinks[0..n]);
            for (ks[0..n]) |k| if (m.e.multi.?.convs[k].done) m.live.release(m.live.ctx, k);
            pc.addDecode(s, m.now() - t0);
            n = m.live.f(m.live.ctx, m.k, &ks, &stops, &sinks);
            // a short prompt that came during the rounds (a prefill held by share 100 waits here)
            // ラウンドの間に来た短い prompt (取り分 100 で止められている prefill はここで待っている)
            give = m.yields();
        }
        if (give) {
            const w = [2]i32{ ev_mid_end, -1 };
            m.ch.gpa.free(try m.ch.share(&w));
            if (m.e.drafter) |dr| dr.use(m.k);
            return null;
        }
        var next: u8 = 0;
        if (m.live.pace) |pc| {
            next = pc.layers(m.share(), n > 0);
            if (next > 0) pc.begin(m.now(), next) else pc.idle();
        }
        const w = [2]i32{ ev_mid_end, next };
        m.ch.gpa.free(try m.ch.share(&w));
        // the prefill goes on with the admitted conversation's drafter context
        // prefill は受け付け中の会話の drafter の context で続ける
        if (m.e.drafter) |dr| dr.use(m.k);
        return next;
    }
};

const FollowMid = struct {
    ch: *Channel,
    e: *engine.Engine,
    k: usize,

    /// Rank 0 decides (chunkAgreed carries its value and its giving way): this rank's is not read.
    /// rank 0 が決める (chunkAgreed がその値と譲るかを運ぶ): このランクの値は読まれない。
    fn first(_: *anyopaque, _: u32) ?u8 {
        return 0;
    }

    fn run(ctx: *anyopaque) anyerror!?u8 {
        const m: *FollowMid = @ptrCast(@alignCast(ctx));
        // the rounds, the releases of the conversations they finished, the end with the layers to the next pause (-1:
        // the prefill gives way)
        // ラウンド、それで終わった会話の確定、次に止まるまでの層数の付いた終わり (-1: prefill は譲る)
        var next: ?u8 = 0;
        while (true) {
            const w = try m.ch.share(null);
            defer m.ch.gpa.free(w);
            if (w.len == 0) return error.EmptyEvent;
            switch (w[0]) {
                ev_round => try followRound(m.e, w),
                ev_release => _ = try pool_serve.commitMulti(m.e, @intCast(w[1])),
                ev_mid_end => {
                    if (w.len < 2 or w[1] < -1 or w[1] > 255) return error.BadEvent;
                    next = if (w[1] < 0) null else @intCast(w[1]);
                    break;
                },
                else => return error.BadEvent,
            }
        }
        if (m.e.drafter) |dr| dr.use(m.k);
        return next;
    }
};

fn followRound(e: *engine.Engine, w: []const i32) !void {
    const n: usize = @intCast(w[1]);
    if (n > engine.multi_max or w.len < 2 + 2 * n) return error.BadEvent;
    var ks: [engine.multi_max]u8 = undefined;
    var stops: [engine.multi_max]bool = undefined;
    const sinks: [engine.multi_max]?engine.Sink = @splat(null);
    for (0..n) |i| {
        ks[i] = @intCast(w[2 + 2 * i]);
        stops[i] = w[3 + 2 * i] != 0;
    }
    try e.multiRound(ks[0..n], stops[0..n], sinks[0..n]);
}

/// Rank 0: conversation k's admission. The plan beside the live conversations comes first, before anything is shared:
/// error.NoRoom (the pool cannot hold it beside them) or error.ShapeNotVerified touch no rank and leave the job to
/// the caller. Then every rank applies the plan, prefills, and conversation k holds its first token (to `sink`).
/// rank 0: 会話 k の受け付け。使用中の会話の横でのプランを最初に、何かを共有する前に立てる: error.NoRoom (pool がそれらの
/// 横に置けない) と error.ShapeNotVerified はどのランクにも影響せず、要求は呼び出し側に戻る。次に全ランクがプランを適用して
/// prefill し、会話 k は最初のトークンを持つ (`sink` へ)。
pub fn leadAdmit(ch: *Channel, e: *engine.Engine, k: usize, h0: Header, prompt: []const i64, sink: ?engine.Sink, live: ?Live) !void {
    var h = h0;
    var busy_buf: [engine.multi_max]u64 = undefined;
    const busy = e.multiBusy(&busy_buf);
    var q = pool_serve.leadBusy(e, &h, prompt, busy) catch |err| return if (err == error.Invariant) error.NoRoom else err;
    var moved = false;
    defer if (!moved) q.deinit(e.gpa);
    try e.admit(prompt.len, @intCast(@max(0, h.r0)));
    var w: [2 + 38]i32 = undefined;
    w[0] = ev_admit;
    w[1] = @intCast(k);
    var hb: [38]i32 = undefined;
    const hw = h.encodeAll(&hb);
    @memcpy(w[2..][0..hw.len], hw);
    ch.gpa.free(try ch.share(w[0 .. 2 + hw.len]));
    const ids = try ch.gpa.alloc(i32, prompt.len);
    defer ch.gpa.free(ids);
    for (ids, prompt) |*v, t| v.* = @intCast(t);
    ch.gpa.free(try ch.share(ids));
    moved = true;
    try pool_serve.admitMulti(ch, e, k, h, prompt, q);
    // the other conversations' rounds inside this prefill (when: rank 0's pace, which every rank follows)
    // この prefill の中のほかの会話のラウンド (いつ: rank 0 の pace。全ランクがそれに従う)
    var lm: LeadMid = .{ .ch = ch, .e = e, .k = k, .live = live orelse .{ .ctx = undefined, .f = noLive, .release = noRelease } };
    e.mid = .{ .ctx = @ptrCast(&lm), .first = LeadMid.first, .f = LeadMid.run };
    defer e.mid = null;
    try e.multiStart(k, prompt, @intCast(h.max_tokens), h.stop_eos, h.mode == mode_dflash or h.mode == mode_mtp, samplingOf(h), sink);
}

fn noLive(_: *anyopaque, _: usize, _: []u8, _: []bool, _: []?engine.Sink) usize {
    return 0;
}

fn noRelease(_: *anyopaque, _: usize) void {}

/// Rank 0: one batched round over conversations `ks` (rank 0's stop for each in `stops`, its outputs in `sinks`).
/// rank 0: 会話 `ks` をまとめたラウンド 1 回 (`stops` にそれぞれへの rank 0 の停止、`sinks` に出力)。
pub fn leadRound(ch: *Channel, e: *engine.Engine, ks: []const u8, stops: []const bool, sinks: []const ?engine.Sink) !void {
    var w: [2 + 2 * engine.multi_max]i32 = undefined;
    w[0] = ev_round;
    w[1] = @intCast(ks.len);
    for (ks, stops, 0..) |k, stp, i| {
        w[2 + 2 * i] = k;
        w[3 + 2 * i] = @intFromBool(stp);
    }
    ch.gpa.free(try ch.share(w[0 .. 2 + 2 * ks.len]));
    try e.multiRound(ks, stops, sinks);
}

/// Rank 0: conversation k is done: its commit on every rank; its stats.
/// rank 0: 会話 k が終わった: 全ランクで確定し、統計を返す。
pub fn leadRelease(ch: *Channel, e: *engine.Engine, k: usize) !engine.Stats {
    const w = [2]i32{ ev_release, @intCast(k) };
    ch.gpa.free(try ch.share(&w));
    return pool_serve.commitMulti(e, k);
}

/// Rank 0: one request the old way (sampled requests, run while no conversation is live).
/// rank 0: 従来の 1 要求 (サンプリングの要求。使用中の会話がないときに実行する)。
pub fn leadSingle(ch: *Channel, e: *engine.Engine, h: Header, prompt: []const i64, sink: ?engine.Sink, prop: ?engine.Proposer) !engine.Stats {
    const w = [1]i32{ev_single};
    ch.gpa.free(try ch.share(&w));
    return lead(ch, e, h, prompt, sink, prop);
}

/// Rank 0: tells the followers of `followMulti` to return.
/// rank 0: `followMulti` のフォロワーに戻るよう伝える。
pub fn shutdownMulti(ch: *Channel) !void {
    const w = [1]i32{ev_shutdown};
    ch.gpa.free(try ch.share(&w));
}

/// Ranks above 0 with TF_GLM53_MULTI: every event rank 0 shares, run as rank 0 runs it, without output.
/// TF_GLM53_MULTI での rank 1 以上: rank 0 が配る出来事を、rank 0 と同じく出力なしで実行する。
pub fn followMulti(ch: *Channel, e: *engine.Engine) !void {
    while (true) {
        const w = try ch.share(null);
        defer ch.gpa.free(w);
        if (w.len == 0) return error.EmptyEvent;
        switch (w[0]) {
            ev_shutdown => return,
            ev_swap => try followSwap(ch, e),
            ev_admit => {
                const k: usize = @intCast(w[1]);
                const h = try Header.decode(w[2..]);
                const words = try ch.share(null);
                defer ch.gpa.free(words);
                const prompt = try ch.gpa.alloc(i64, words.len);
                defer ch.gpa.free(prompt);
                for (prompt, words) |*t, v| t.* = v;
                try pool_serve.admitMulti(ch, e, k, h, prompt, null);
                var fm: FollowMid = .{ .ch = ch, .e = e, .k = k };
                e.mid = .{ .ctx = @ptrCast(&fm), .first = FollowMid.first, .f = FollowMid.run };
                defer e.mid = null;
                try e.multiStart(k, prompt, @intCast(h.max_tokens), h.stop_eos, h.mode == mode_dflash or h.mode == mode_mtp, samplingOf(h), null);
            },
            ev_round => try followRound(e, w),
            ev_release => _ = try pool_serve.commitMulti(e, @intCast(w[1])),
            ev_single => {
                const hw = try ch.share(null);
                defer ch.gpa.free(hw);
                const h = try Header.decode(hw);
                const words = try ch.share(null);
                defer ch.gpa.free(words);
                const prompt = try ch.gpa.alloc(i64, words.len);
                defer ch.gpa.free(prompt);
                for (prompt, words) |*t, v| t.* = v;
                _ = try run(ch, e, h, prompt, null, null, null, null);
            },
            else => return error.BadEvent,
        }
    }
}

test "the header round-trips through its words" {
    const h: Header = .{ .max_tokens = 300, .stop_eos = false, .k = 2, .mode = 9, .seed = 0x7FFF_FFFF_1234_5678, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0.05, .r0 = 17 };
    const w = h.encode();
    for (w) |v| try std.testing.expect(v >= 0);
    const back = try Header.decode(&w);
    try std.testing.expectEqualDeep(h, back);
    // 1.0's bits split low 31 / next 31 / top 2
    // 1.0 のビットを下位 31・次の 31・上位 2 に分ける
    var one: [3]i32 = undefined;
    f64Words(1.0, &one);
    const b: u64 = @bitCast(@as(f64, 1.0));
    try std.testing.expectEqual(@as(i32, @intCast(b & 0x7FFFFFFF)), one[0]);
    try std.testing.expectEqual(@as(i32, @intCast((b >> 31) & 0x7FFFFFFF)), one[1]);
    try std.testing.expectEqual(@as(i32, @intCast(b >> 62)), one[2]);
}
