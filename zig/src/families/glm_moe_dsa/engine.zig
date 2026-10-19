//! One rank of a GLM-5.3 replica: its weight shares, buffers, caches and decode graphs, and the request every rank runs
//! alike: equal prompt chunks, then one row a round (or a verify window with drafts). Every rank computes every pick
//! alike; the only control the ranks exchange is rank 0's stop flag and the draft checks, at their agreement points.
//! GLM-5.3 レプリカの 1 ランク: 重みの分担・バッファ・キャッシュ・decode グラフと、全ランクが同じく実行する要求。
//! 要求は等分した prompt チャンクのあと、1 ラウンドに 1 行 (または draft 付きの verify ウィンドウ) ずつ進む。選択は
//! 全ランクで同じく計算する。ランク間の制御は、合意点での rank 0 の停止フラグと draft の照合だけ。
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const model = @import("model.zig");
const moe = @import("moe.zig");
const exl3 = @import("exl3.zig");
const experts = @import("experts.zig");
const forward = @import("forward.zig");
const rope = @import("rope.zig");
const dw = @import("draft_weights.zig");
const drafter_mod = @import("drafter.zig");
const mtp_mod = @import("mtp.zig");
const prompt_experts = @import("prompt_experts.zig");
const roce = @import("roce.zig");
const native_ops = @import("native_ops.zig");
const pool_devmap = @import("pool/devmap.zig");
const pool_serve = @import("pool_serve.zig");
const slots_serve_mod = @import("slots_serve.zig");
const unpack_cache = @import("unpack_cache.zig");
const slots_mod = @import("slots.zig");
const tune = @import("tune.zig");
const linear = @import("linear.zig");
const lanes = @import("lanes");
/// A request's sampling: keyed draws by (seed, position, id), so drafted and serial replies match.
/// 要求のサンプリング: (seed, 位置, id) をキーにした抽選なので、draft ありと逐次の応答は一致する。
pub const Sampling = lanes.Sampling;
const sampler = lanes.sampling;
/// A prompt's reproducible seed (salt 0).
/// prompt から再現できる seed (salt 0)。
pub const seedFor = lanes.sampling.seedFor;
/// Candidates read past top_k (past 256 when top_k is 0), so ties at the edge resolve exactly.
/// top_k (top_k が 0 なら 256) を超えて読む候補の数。境界の同点を正確に扱うため。
const sample_margin = 8;

pub const Options = struct {
    ckpt: []const u8,
    // the checkpoint's expert split; a smaller world reads its shares as virtual ranks
    // チェックポイントの expert の分割数。world がそれより小さければ仮想ランクとして読む
    stored_world: usize,
    layers: ?usize = null, // the first N layers only (development); null: all / 先頭 N 層だけ (開発用)。null は全層
    capacity: usize = 8192, // positions a request may reach (prompt + reply) / 要求が届きうる位置の数 (prompt + 応答)
    // the decode rounds as CUDA graphs, one a key-range bucket
    // decode ラウンドを CUDA グラフで実行する (キー範囲の bucket ごとに 1 つ)
    graphs: bool = true,
    // the DFlash2 drafter's directory (TF_GLM53_DFLASH): taps on, mode 9 drafts
    // DFlash2 の drafter のディレクトリ (TF_GLM53_DFLASH): tap を有効にし、mode 9 で draft する
    drafter: ?[]const u8 = null,
    /// TF_GLM53_DRAFTER=mtp: drafts from the checkpoint's MTP layer (mode 4), TF_GLM53_MTP_K of them a round
    /// TF_GLM53_DRAFTER=mtp: チェックポイントの MTP 層で draft する (mode 4)。1 ラウンドの数は TF_GLM53_MTP_K
    mtp: bool = false,
    mtp_k: usize = 3,
    // TF_GLM53_KV=fp8: e4m3 latent and index-key caches
    // TF_GLM53_KV=fp8: e4m3 の latent と index キーのキャッシュ
    kv_fp8: bool = false,
    // TF_GLM53_KV=fp4: 4-bit latent rows (kv_fp8 also set; the index keys stay FP8)
    // TF_GLM53_KV=fp4: 4 ビットの latent 行 (kv_fp8 も立つ。index の key は FP8 のまま)
    kv_fp4: bool = false,
    // TF_GLM53_DCP: the caches split over groups of this many consecutive ranks (decode context parallelism)
    // TF_GLM53_DCP: キャッシュを、連続するこの数のランクのグループで分割する (decode context parallelism)
    dcp: u32 = 1,
    /// TF_GLM53_PROMPT_SP=1 and TF_GLM53_SP_DCP=1: sequence-parallel prompt chunks in TF_GLM53_SP_PARTS parts (0: off;
    /// DCP > 1 only) on a comm stream of priority `comm_priority` (TF_GLM53_COMM_PRIORITY)
    /// TF_GLM53_PROMPT_SP=1 かつ TF_GLM53_SP_DCP=1: prompt チャンクを TF_GLM53_SP_PARTS 個に分けて sequence-parallel
    /// で実行する (0 で無効、DCP > 1 のみ)。comm stream の優先度は `comm_priority` (TF_GLM53_COMM_PRIORITY)
    sp_parts: u32 = 0,
    comm_priority: i32 = 0,
    /// The widest prompt chunk (TF_GLM53_PROMPT_ROWS / DCP_PROMPT_ROWS); above forward.max_rows only with sequence
    /// parallelism (each part stays within max_rows)
    /// prompt チャンクの最大幅 (TF_GLM53_PROMPT_ROWS / DCP_PROMPT_ROWS)。forward.max_rows を超えるのは
    /// sequence parallel のときだけ (各部分は max_rows 以内)
    prompt_rows: usize = forward.max_rows,
    /// TF_EXL3_PROMPT_EXPERTS=1: prompt chunks' routed experts on prompt_experts (non-deterministic atomics)
    /// TF_EXL3_PROMPT_EXPERTS=1: prompt チャンクの routed expert を prompt_experts で計算する (atomic で非決定的)
    prompt_experts: bool = false,
    /// TF_GLM53_PREFILL_REDUCE=ring (the default; rs: the exact sum): SP prompt partials reduce-scattered as bf16
    /// TF_GLM53_PREFILL_REDUCE=ring (既定。rs なら正確な和): SP の prompt 部分和を bf16 で reduce-scatter する
    ring: bool = false,
    /// TF_GLM53_TOPK_WORDS (default 1): the DCP selection's candidates of prompt rows from int32 order words; the same
    /// picks as the packed int64 path, which 0 keeps
    /// TF_GLM53_TOPK_WORDS (既定 1): DCP の選択で、prompt 行の候補を int32 の順序語から作る。選ぶものは packed int64
    /// の経路と同じで、0 ならそちらを使う
    topk_words: bool = true,
    /// TF_GLM53_TOPK_ANY (default 1): sequence-parallel candidates from two passes in any order (forward.Ctx.topk_any)
    /// TF_GLM53_TOPK_ANY (既定 1): sequence-parallel の候補を、順不同の 2 パスで集める (forward.Ctx.topk_any)
    topk_any: bool = true,
    /// TF_GLM53_PE_F16 (default 1): the prompt experts' slot sums in fp16 in sequence-parallel parts (forward.Ctx.pe_f16;
    /// KLD against the reference unchanged, TP4.md §AF.9)
    /// TF_GLM53_PE_F16 (既定 1): sequence-parallel の part で prompt expert のスロットの和を fp16 で (forward.Ctx.pe_f16。
    /// 参照に対する KLD は変わらない、TP4.md §AF.9)
    pe_f16: bool = true,
    /// TF_GLM53_TOPK16 (default 1): the any-order candidates from the order words' top 16 bits (forward.Ctx.topk16)
    /// TF_GLM53_TOPK16 (既定 1): 順不同の候補を順序語の上位 16 ビットから (forward.Ctx.topk16)
    topk16: bool = true,
    /// TF_GLM53_FUSE (default 1): steps folded into their neighbours where the native kernels run (forward.Ctx.fuse)
    /// TF_GLM53_FUSE (既定 1): ネイティブカーネルで、ステップを隣の処理にまとめる (forward.Ctx.fuse)
    fuse: bool = true,
    /// TF_GLM53_XGROUP (default 1): a prompt chunk's DCP exchange (outputs, log-sum-exps) as one NCCL group
    /// TF_GLM53_XGROUP (既定 1): prompt チャンクの DCP の交換 (出力と log-sum-exp) を 1 つの NCCL group にする
    xgroup: bool = true,
    /// TF_GLM53_PEER_Q (default 1; DCP 2 with TF_GLM53_FUSE): load the DCP partner's q_b / wk and compute its queries
    /// of a prompt chunk here instead of gathering them over the group
    /// TF_GLM53_PEER_Q (既定 1、DCP 2 かつ TF_GLM53_FUSE のとき): DCP の相手の q_b / wk も読み込み、prompt チャンクの
    /// 相手の query をグループで集めずにここで計算する
    peer_q: bool = true,
    /// TF_GLM53_INDEX_FP8 (default 0): the indexer's scores on the FP8 tensor-core product with the queries quantized
    /// to e4m3; the scores differ from the default path's, so the picks can differ (forward.Ctx.index_fp8)
    /// TF_GLM53_INDEX_FP8 (既定 0): query を e4m3 に量子化し、indexer のスコアを FP8 tensor core の積で求める。既定の
    /// 経路とスコアが違うので、選ばれるものも変わりうる (forward.Ctx.index_fp8)
    index_fp8: bool = false,
    /// TF_GLM53_UNPACK_CACHE_MB (default 384; 0: off): prompt GEMMs' decoded weights kept for the next calls
    /// TF_GLM53_UNPACK_CACHE_MB (既定 384、0 で無効): prompt の GEMM が展開した重みを次の呼び出しのために残す
    unpack_cache_mb: usize = 0,
    /// TF_GLM53_ROCE=1: decode windows' reductions and gathers on the RoCE one-shot runtimes (world, and with
    /// TF_GLM53_DCP_ROCE (default on) the DCP group's); TF_GLM53_DECODE_BF16=1: the window partials reduced as bf16
    /// TF_GLM53_ROCE=1: decode ウィンドウの reduce と gather を RoCE の one-shot ランタイムで行う (world、
    /// TF_GLM53_DCP_ROCE (既定で有効) なら DCP グループも)。TF_GLM53_DECODE_BF16=1: ウィンドウの部分和を bf16 で
    /// reduce する
    roce: bool = false,
    dcp_roce: bool = true,
    decode_bf16: bool = false,
};

/// `o` with the settings read from the environment (see Options). With DCP > 1, TF_GLM53_PROMPT_SP=1 and
/// TF_GLM53_SP_DCP=1 give SP_PARTS parts (default 2) and chunk rows min(PROMPT_ROWS (4096), DCP_PROMPT_ROWS (1024)).
/// 環境変数の設定を反映した `o` (Options を参照)。DCP > 1 で TF_GLM53_PROMPT_SP=1 かつ TF_GLM53_SP_DCP=1 なら、
/// SP_PARTS 個 (既定 2) に分け、チャンクは min(PROMPT_ROWS (4096), DCP_PROMPT_ROWS (1024)) 行。
pub fn withEnv(o: Options) Options {
    const F = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
        }
        fn num(name: [:0]const u8, default: i64) i64 {
            const v = get(name) orelse return default;
            return std.fmt.parseInt(i64, v, 10) catch default;
        }
    };
    var r = o;
    r.prompt_experts = std.mem.eql(u8, F.get("TF_EXL3_PROMPT_EXPERTS") orelse "0", "1");
    r.mtp = std.mem.eql(u8, F.get("TF_GLM53_DRAFTER") orelse "", "mtp");
    r.mtp_k = @intCast(std.math.clamp(F.num("TF_GLM53_MTP_K", 3), 1, mtp_mod.max_rows - 1));
    r.ring = std.mem.eql(u8, F.get("TF_GLM53_PREFILL_REDUCE") orelse "ring", "ring");
    r.topk_words = !std.mem.eql(u8, F.get("TF_GLM53_TOPK_WORDS") orelse "1", "0");
    r.topk_any = !std.mem.eql(u8, F.get("TF_GLM53_TOPK_ANY") orelse "1", "0");
    r.pe_f16 = !std.mem.eql(u8, F.get("TF_GLM53_PE_F16") orelse "1", "0");
    r.topk16 = !std.mem.eql(u8, F.get("TF_GLM53_TOPK16") orelse "1", "0");
    r.fuse = !std.mem.eql(u8, F.get("TF_GLM53_FUSE") orelse "1", "0");
    r.xgroup = !std.mem.eql(u8, F.get("TF_GLM53_XGROUP") orelse "1", "0");
    r.peer_q = !std.mem.eql(u8, F.get("TF_GLM53_PEER_Q") orelse "1", "0");
    r.index_fp8 = std.mem.eql(u8, F.get("TF_GLM53_INDEX_FP8") orelse "0", "1");
    r.unpack_cache_mb = @intCast(@max(0, F.num("TF_GLM53_UNPACK_CACHE_MB", 384)));
    r.roce = std.mem.eql(u8, F.get("TF_GLM53_ROCE") orelse "0", "1");
    r.dcp_roce = !std.mem.eql(u8, F.get("TF_GLM53_DCP_ROCE") orelse "1", "0");
    r.decode_bf16 = std.mem.eql(u8, F.get("TF_GLM53_DECODE_BF16") orelse "0", "1");
    const on = std.mem.eql(u8, F.get("TF_GLM53_PROMPT_SP") orelse "0", "1") and std.mem.eql(u8, F.get("TF_GLM53_SP_DCP") orelse "0", "1");
    // SP splits a chunk across the DCP pair: with DCP 1 it is left off, said once (POOL.md 12.2e finding 5)
    // SP はチャンクを DCP の組に分ける: DCP 1 では使わず、そのことを一度だけ出す (POOL.md 12.2e の見つかったこと 5)
    if (on and o.dcp < 2) std.log.warn("glm53: TF_GLM53_PROMPT_SP=1 needs DCP 2; with DCP {d} the prompt chunks run without sequence-parallel parts", .{o.dcp});
    if (!on or o.dcp < 2) return r;
    r.sp_parts = @intCast(@max(2, F.num("TF_GLM53_SP_PARTS", 2)));
    r.comm_priority = @intCast(F.num("TF_GLM53_COMM_PRIORITY", 0));
    r.prompt_rows = @intCast(@min(F.num("TF_GLM53_PROMPT_ROWS", 4096), F.num("TF_GLM53_DCP_PROMPT_ROWS", 1024)));
    return r;
}

/// Rows a verify window may hold: the pending token and up to 7 drafts.
/// verify ウィンドウの最大行数: 保留中のトークンと最大 7 個の draft。
pub const vrows = 8;

/// A prompt chunk's shape class: its row count up to 16 (each its own row block), the router's BM class, the path
/// (row-exact linears up to 128 rows, the wide GEMMs, SP in 2 or 4 parts) and whether it selects (end past index_topk).
/// prompt チャンクの形の分類: 16 行までは行数そのもの (それぞれ固有の行ブロック)、router の BM 分類、経路 (128 行
/// までは行ごとに正確な linear、その先は幅広の GEMM、SP は 2 か 4 分割)、選択の有無 (終端が index_topk を超えるか)。
pub const ShapeKey = struct { small: u8, bm: u8, path: u8, select: bool };

/// Where a drafted round's drafts come from (the DFlash2 drafter, the MTP layer, or a test's fixed continuation): up
/// to `room` drafts for the positions after the pending token `tok` at P, into `out`; returns how many. Every rank
/// must return the same.
/// draft ラウンドの draft の出どころ (DFlash2 の drafter、MTP 層、テストの固定の続き): 位置 P の保留中のトークン
/// `tok` に続く位置へ最大 `room` 個の draft を `out` に書き、その数を返す。全ランクが同じものを返さねばならない。
pub const Proposer = struct {
    ctx: *anyopaque,
    propose: *const fn (ctx: *anyopaque, tok: u32, P: usize, room: usize, out: []u32) anyerror!usize,
    /// The kept rows (pending + accepted drafts) of a round, taps in the verify buffers' rows 0 .. n - 1.
    /// ラウンドで残った行 (保留中と採用された draft)。tap は verify バッファの行 0 .. n - 1 にある。
    kept: ?*const fn (ctx: *anyopaque, n: usize) anyerror!void = null,
    /// The same rows' picks (the token after each kept row): the MTP layer's next backlog.
    /// 同じ行の選択 (残った各行の次のトークン): MTP 層の次のバックログ。
    kept_ids: ?*const fn (ctx: *anyopaque, picks: []const i32) anyerror!void = null,
    /// A request starts (before its prefill): its prompt.
    /// 要求の開始 (prefill の前): その prompt。
    begin: ?*const fn (ctx: *anyopaque, prompt: []const i64) anyerror!void = null,
    /// A prompt chunk's rows are in the caches: their taps (device, [n, n_taps * D] bf16).
    /// prompt チャンクの行がキャッシュに入った: その tap (デバイス上、[n, n_taps * D] bf16)。
    taps: ?*const fn (ctx: *anyopaque, src: u64, n: usize) anyerror!void = null,
};

/// What a request did (rank 0 reports it; every rank computes the same).
/// 要求の結果 (rank 0 が報告する。全ランクが同じ値を計算する)。
pub const Stats = struct {
    prompt: usize = 0,
    generated: usize = 0,
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    // prompt rows the slot cache resumed (not prefilled)
    // スロットキャッシュから再開した prompt の行数 (prefill しない)
    cached: usize = 0,
    // cancelled: rank 0 asked every rank to stop (stop_req)
    // cancelled: rank 0 が全ランクに停止を求めた (stop_req)
    finish: enum { length, stop, cancelled } = .length,
};

/// Receives the reply's tokens as they are picked (rank 0; followers pass none). Must not block.
/// 選ばれた応答のトークンを順に受け取る (rank 0。フォロワーは渡さない)。ブロックしてはならない。
pub const Sink = struct {
    ctx: *anyopaque,
    tokens: *const fn (ctx: *anyopaque, new: []const u32) void,
    prefilled: ?*const fn (ctx: *anyopaque) void = null,
};

/// The indexer's key range for a window ending at position t - 1 (none while t <= topk, else a power of two, at least
/// twice topk).
/// 位置 t - 1 で終わるウィンドウの indexer のキー範囲 (t <= topk の間はなし。それ以外は 2 のべき乗で、topk の 2 倍
/// 以上)。
pub fn bucket(t: usize, topk: usize) ?u32 {
    if (t <= topk) return null;
    const p2 = std.math.ceilPowerOfTwo(usize, t) catch unreachable;
    return @intCast(@max(2 * topk, p2));
}

// a graph per (rows, key bucket): 1..8 rows x the buckets up to the capacity (10 buckets at 768K: 80)
// (行数, キーの bucket) ごとに 1 つ: 1..8 行 x 容量までの bucket (768K では bucket 10 個で 80)
const max_graphs = 128;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    rank: u32,
    world: u32,
    cfg: Config,
    heads: u32,
    group: *const cuda.tp.Group,
    blas: cuda.cublas.Library,
    handle: cuda.cublas.Handle,
    workspace: cuda.DeviceBuffer,
    stream: cuda.Stream,
    x3: exl3.Kernels,
    glue: forward.Glue,
    nat: native_ops.Kernels,
    exk: experts.Kernels,
    layers: []model.Layer,
    embed: cuda.DeviceBuffer,
    lm_head: cuda.DeviceBuffer,
    final_norm: cuda.DeviceBuffer,
    vshare: u32,
    bufs: forward.Buffers, // prompt chunks / prompt チャンク用
    // verify windows of up to `vrows` rows (small is 16 either way, so one-row rounds are unchanged)
    // 最大 `vrows` 行の verify ウィンドウ用 (small はどちらも 16 なので、1 行のラウンドは変わらない)
    vbufs: forward.Buffers,
    xs: moe.Scratch,
    vxs: moe.Scratch,
    state: forward.State,
    capacity: usize,
    graphs: bool,
    kv8: bool,
    kv4: bool,
    dcp: u32,
    // DCP > 1: this rank's group (ncclCommSplit color rank / dcp, key rank % dcp)
    // DCP > 1: このランクのグループ (ncclCommSplit の color は rank / dcp、key は rank % dcp)
    dgroup: ?cuda.tp.Group = null,
    execs: [max_graphs]cuda.graph.Exec = undefined,
    keys: [max_graphs]?u32 = undefined,
    key_rows: [max_graphs]u32 = undefined, // the window's rows (1: serial rounds) / ウィンドウの行数 (1 は逐次)
    // per layer its tap column block, -1 none (DFlash2's target_layer_ids)
    // 層ごとの tap の列ブロック。-1 はなし (DFlash2 の target_layer_ids)
    tap_slot: []i8 = &.{},
    n_taps: u32 = 0,
    drafter: ?*drafter_mod.Drafter = null,
    // TF_GLM53_DRAFTER=mtp: its caches are State entry layers.len
    // TF_GLM53_DRAFTER=mtp: そのキャッシュは State のエントリ layers.len
    mtp: ?mtp_mod.Mtp = null,
    chk_send: cuda.DeviceBuffer = undefined, // the agreement points' words (int32) / 合意点の語 (int32)
    chk_recv: cuda.DeviceBuffer = undefined,
    n_graphs: usize = 0,
    // sequence-parallel prompt chunks (Options.sp_parts): parts 1 .. of their own buffers (part 0: bufs / xs), the
    // comm stream, the event handing it the main stream's progress
    // sequence-parallel の prompt チャンク (Options.sp_parts): 部分 1 .. は専用のバッファを持つ (部分 0 は bufs / xs)。
    // comm stream と、main stream の進み具合をそれに渡す event
    sp_want: u32 = 0,
    n_sp: usize = 0,
    sp_parts: [forward.sp_max_parts]forward.SpPart = undefined,
    sp_bufs: [forward.sp_max_parts - 1]forward.Buffers = undefined,
    sp_xs: [forward.sp_max_parts - 1]moe.Scratch = undefined,
    comm: cuda.Stream = undefined,
    ready: cuda.Event = undefined,
    prompt_rows: usize = forward.max_rows,
    /// The rows of the main buffers and of every SP part's (partRows): forward.max_rows, or the widest part of a
    /// prompt chunk when its parts are wider.
    /// main のバッファと SP の各 part のバッファの行数 (partRows): forward.max_rows、または part の方が広ければ
    /// prompt チャンクの最も広い part。
    part_rows: usize = forward.max_rows,
    /// The rows of the routed experts' scratch (moe.Scratch, about 600 KB a row): MAX_LIN_ROWS when prompt_experts
    /// takes every MoE layer's windows wider than that (the scratch then serves decode windows and short chunks
    /// only; SP parts are always wider), else part_rows.
    /// routed expert の作業領域 (moe.Scratch、1 行 約 600 KB) の行数: prompt_experts が全 MoE 層でそれより広い
    /// ウィンドウを受け持つなら MAX_LIN_ROWS (作業領域を使うのは decode ウィンドウと短いチャンクだけ。SP の part は
    /// 常にそれより広い)、そうでなければ part_rows。
    xs_rows: usize = forward.max_rows,
    /// forward.Ctx.probe for every context (tests: per-layer hashes; syncs after each layer step)
    /// 全 context の forward.Ctx.probe (テスト用: 層ごとのハッシュ。各層のステップ後に同期する)
    probe: ?*const fn (ctx: *anyopaque, layer: usize, what: []const u8, rows: u32, ptr: u64, bytes: usize) anyerror!void = null,
    probe_ctx: *anyopaque = undefined,
    probe_select: bool = false,
    /// tests (selLogEnable): the device log of a decode window's picks (forward.Ctx.sel_tok / sel_cnt), the calls of
    /// `select` the last capture made, the index layers, and the host copy scoreForced hands to sel_sink after each
    /// window past index_topk (P: the window's first position, R: its rows)
    /// テスト用 (selLogEnable): decode ウィンドウの選択のデバイス上のログ (forward.Ctx.sel_tok / sel_cnt)、直近の
    /// capture が行った `select` の呼び出し数、index 層の数、index_topk を超えた各ウィンドウのあとに scoreForced が
    /// sel_sink に渡すホスト側の写し (P はウィンドウの先頭位置、R は行数)
    sel_tok: ?cuda.DeviceBuffer = null,
    sel_cnt: ?cuda.DeviceBuffer = null,
    sel_n: u32 = 0,
    sel_layers: u32 = 0,
    sel_host: []u8 = &.{},
    sel_sink: ?*const fn (ctx: *anyopaque, P: usize, R: u32, layers: u32, tok: []const u8, cnt: []const u8) anyerror!void = null,
    sel_sink_ctx: *anyopaque = undefined,
    /// drafted rounds' host phase times (ns), slots 1 .. 4: propose, agree, window (launch + sync), kept (0 unused)
    /// draft ラウンドのホスト側の段階ごとの時間 (ns)。1 .. 4 は提案、合意、ウィンドウ (起動 + 同期)、kept (0 は未使用)
    round_ns: [5]u64 = @splat(0),
    pek: ?prompt_experts.Kernels = null,
    pes: prompt_experts.Scratch = undefined,
    pe_ok: []bool = &.{},
    ring: bool = false,
    topk_words: bool = true,
    topk_any: bool = true,
    pe_f16: bool = true,
    topk16: bool = true,
    fuse: bool = true,
    xgroup: bool = true,
    peer_q: bool = false, // the partner's q_b / wk are loaded (forward.Ctx.peer_q) / 相手の q_b / wk を読み込み済み
    index_fp8: bool = false,
    ucache: ?unpack_cache.Cache = null,
    rk: roce.Kernels = undefined,
    fast: ?roce.Runtime = null,
    dfast: ?roce.Runtime = null,
    decode_bf16: bool = false,
    // the slot cache (TF_GLM53_PREFIX=slots, slots.zig): the store; this request's resume point and the drafter's
    // window sources (Drafter.resume); the previous request's drafter ends; the reply's tokens for the commit
    // スロットキャッシュ (TF_GLM53_PREFIX=slots、slots.zig): ストア、この要求の再開位置と drafter のウィンドウの
    // 出どころ (Drafter.resume)、前の要求の drafter の終端、commit 用の応答のトークン
    slots: ?*slots_mod.Store = null,
    // TF_GLM53_POOL_TEST (pool/devmap.zig): the device PoolMap every KV kernel maps rows through (header + table)
    // TF_GLM53_POOL_TEST (pool/devmap.zig): KV カーネルが行を引くデバイスの PoolMap (ヘッダ + 表)
    pmap: ?cuda.DeviceBuffer = null,
    // TF_GLM53_PREFIX=pool (pool_serve.zig): the metadata and the row hashes
    // TF_GLM53_PREFIX=pool (pool_serve.zig): メタデータと行ハッシュ
    pool: ?*pool_serve.Pool = null,
    start: usize = 0,
    dr_ring: ?[3]?usize = null,
    dr_live_end: ?usize = null,
    dr_snap_end: ?usize = null,
    dr_resume: ?drafter_mod.Resume = null,
    // TF_GLM53_PREFIX=pool with per-slot saves: the range-based resume pool_serve.zig decided (the kind, the snapshot
    // positions copied into the live rings)
    // TF_GLM53_PREFIX=pool でスロットごとの写しがあるとき: pool_serve.zig が決めた範囲付きの再開 (種類と、snapshot から
    // live のリングへコピーする位置)
    dr_pool: ?struct { kind: drafter_mod.Resume, copy: ?[2]usize } = null,
    out: std.ArrayList(i64) = .empty,
    // this request's sampling (null: greedy)
    // この要求のサンプリング (null なら greedy)
    samp: ?Sampling = null,
    // stopping a request mid-way (the client left, or a stop string matched): rank 0's wish, which every rank reads at
    // the request's agreement points (each prompt chunk, each decode round) and acts on alike
    // 要求の途中停止 (クライアントが去った、または stop 文字列に一致): rank 0 の希望。全ランクが要求の合意点
    // (各 prompt チャンク、各 decode ラウンド) で読み、同じように従う
    stop_req: std.atomic.Value(bool) = .init(false),
    // the prompt rows in the caches when the prefill ended (all of them, or a stop's point)
    // prefill を終えた時点でキャッシュにある prompt の行 (全部、または停止した位置まで)
    prefill_done: usize = 0,
    // the prompt chunk shapes the start-up prewarm ran: `admit` refuses, before it is shared, a request needing another
    // 起動時の prewarm が実行した prompt チャンクの形。それ以外を要する要求は、共有の前に `admit` が断る
    shapes: [96]ShapeKey = undefined,
    n_shapes: usize = 0,
    recording_shapes: bool = false,
    lgath: cuda.DeviceBuffer = undefined, // fp32 [world, rows, vshare]: every rank's logits / 全ランクの logits
    host_logits: []f32 = &.{},

    /// Loads this rank's shares (layer by layer, returning each layer's file pages before the next) and sets up the
    /// buffers. `group` is the world communicator every rank built; the Engine lives where it is created (the forward
    /// contexts point into it).
    /// このランクの分担を読み込み (層ごとに、次の層の前にその層のファイルのページを返す)、バッファを用意する。
    /// `group` は全ランクが作った world の communicator。Engine は作られた場所に置いたままにする (forward の context が
    /// そこを指す)。
    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, group: *const cuda.tp.Group, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const rank: u32 = @intCast(group.rank);
        const world: u32 = @intCast(group.world);
        var mt: usize = 0; // TF_GLM53_MEMTRACE (memMark)
        memMark(ctx, rank, "-", &mt);
        const cfg = try Config.read(gpa, io, o.ckpt);
        const heads: u32 = @intCast(cfg.heads / world);
        const n = o.layers orelse cfg.layers;
        if (n > cfg.layers) return error.TooManyLayers;

        var blas = try cuda.cublas.Library.open();
        errdefer blas.close();
        var handle: cuda.cublas.Handle = null;
        try blas.check(blas.api.cublasCreate_v2(&handle), "cublasCreate");
        const ws = try cuda.DeviceBuffer.alloc(d, cuda.cublas.torch_workspace);
        try blas.check(blas.api.cublasSetWorkspace_v2(handle, ws.ptr, cuda.cublas.torch_workspace), "cublasSetWorkspace");
        const stream = try cuda.Stream.init(d, true);
        try blas.check(blas.api.cublasSetStream_v2(handle, stream.handle), "cublasSetStream");
        memMark(ctx, rank, "cuBLAS workspace + stream", &mt);

        var reader = try weights.Reader.open(gpa, io, o.ckpt, rank, world, o.stored_world);
        defer reader.close();
        const layers = try gpa.alloc(model.Layer, n);
        for (layers, 0..) |*L, i| {
            L.* = try model.loadLayer(gpa, d, &reader, cfg, i, heads);
            if (cfg.isMoe(i)) L.experts = try moe.load(gpa, d, &reader, cfg, i);
            reader.release();
            try memCheck(io, i);
        }
        memMark(ctx, rank, "layers (weights, routed experts included)", &mt);
        var mtp: ?mtp_mod.Mtp = null;
        if (o.mtp) {
            if (o.drafter != null) return error.TwoDrafters;
            mtp = try mtp_mod.Mtp.load(gpa, d, &reader, cfg, heads, n, o.mtp_k, @max(o.prompt_rows, forward.max_rows));
            reader.release();
            try memCheck(io, n);
        }
        memMark(ctx, rank, "MTP layer", &mt);
        const peer_q = o.peer_q and o.fuse and o.dcp == 2 and world % 2 == 0;
        if (peer_q) {
            var peer = try weights.Reader.open(gpa, io, o.ckpt, rank ^ 1, world, o.stored_world);
            defer peer.close();
            for (layers, 0..) |*L, i| {
                const pq = try model.loadPeerQuery(gpa, d, &peer, cfg, i, heads);
                L.peer_q_b = pq.q_b;
                L.peer_wk = pq.wk;
                peer.release();
            }
            if (mtp) |*m| {
                const pq = try model.loadPeerQuery(gpa, d, &peer, cfg, cfg.layers, heads);
                m.layer.peer_q_b = pq.q_b;
                m.layer.peer_wk = pq.wk;
                peer.release();
            }
            std.log.info("glm53: the DCP partner's q_b / wk loaded (rank {d}): its queries of prompt chunks are computed here (TF_GLM53_PEER_Q)", .{rank ^ 1});
        }
        memMark(ctx, rank, "the DCP partner's q_b / wk (PEER_Q)", &mt);
        const embed = try upload(gpa, d, &reader, "model.embed_tokens.weight");
        var lm = try reader.get("lm_head.weight");
        const vshare: u32 = @intCast(lm.shape[0]);
        const lm_head = try cuda.DeviceBuffer.fromHost(d, lm.bytes);
        lm.deinit(gpa);
        const final_norm = try upload(gpa, d, &reader, "model.norm.weight");
        reader.release();
        memMark(ctx, rank, "embed + lm_head + final norm", &mt);

        // the widest key range a window may score: the bucket of the capacity (a power of two, at least 2 topk)
        // ウィンドウが採点しうる最大のキー範囲: 容量の bucket (2 のべき乗で、topk の 2 倍以上)
        const keys: usize = bucket(o.capacity + 1, cfg.index_topk) orelse 2 * cfg.index_topk;
        const inv = try rope.table(cfg.rope_theta, cfg.qk_rope);
        const had = rope.hadamard();
        var first_moe: ?*const moe.Routed = null;
        for (layers) |*L| if (L.experts) |*x| {
            first_moe = x;
            break;
        };
        // TF_GLM53_TUNE (default on): each linear shape's (K splits, warps) chosen by timing, then the groups; before
        // the buffers, whose split-K scratch the chosen tiles size
        // TF_GLM53_TUNE (既定で有効): linear の形ごとに (K 分割数, warp 数) を計時で選び、次にグループを選ぶ。選んだ
        // タイルが split-K の作業領域の大きさを決めるので、バッファより先に行う
        const x3 = try exl3.Kernels.load(d);
        const tmode = tune.modeFromEnv();
        const tr = try tune.run(gpa, io, d, &x3, stream, layers, tmode, rank);
        var zn: usize = forward.MAX_LIN_ROWS * 8 * 6144;
        // the MTP layer's linears keep plan()'s tiles (not tuned): its splits size the scratch too
        // MTP 層の linear は plan() のタイルのまま (調整しない)。その分割数も作業領域の大きさに入れる
        const mtp_one: []const model.Layer = if (mtp) |*m| (&m.layer)[0..1] else &.{};
        for ([_][]const model.Layer{ layers, mtp_one }) |list| for (list) |*L| {
            const ls = [_]*const linear.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down };
            for (ls) |l| if (l.sk > 1) {
                zn = @max(zn, @as(usize, l.sk) * forward.MAX_LIN_ROWS * l.n);
            };
            if (L.indexer) |*ix| if (ix.wq_b.sk > 1) {
                zn = @max(zn, @as(usize, ix.wq_b.sk) * forward.MAX_LIN_ROWS * ix.wq_b.n);
            };
        };
        forward.lin_z_floats = zn;
        if (tmode != .plan) std.log.info("glm53: rank {d}: tuned {d} linear shapes ({d} retiled), {d} groups ({d} as one launch){s}", .{ rank, tr.linears, tr.retiled, tr.groups, tr.grouped, if (tmode == .table) " - read from the table" else "" });
        if (std.c.getenv("TF_GLM53_TUNE_DUMP")) |p| try tune.writeTable(gpa, io, std.mem.span(p), rank, layers);
        memMark(ctx, rank, "EXL3 kernels + tuning", &mt);
        // the attention and index kernels read FP8 caches only
        // attention と index のカーネルは FP8 のキャッシュしか読まない
        if (!o.kv_fp8) return error.Bf16KvUnsupported;
        // the caches: one entry a decoder layer, then the MTP layer's (so the slot cache keeps its rows too)
        // キャッシュ: decoder 層ごとに 1 エントリ、続いて MTP 層のもの (スロットキャッシュがその行も保つように)
        const cache_layers = if (mtp) |*m| blk: {
            const all = try gpa.alloc(model.Layer, layers.len + 1);
            @memcpy(all[0..layers.len], layers);
            all[layers.len] = m.layer;
            break :blk all;
        } else layers;
        defer if (mtp != null) gpa.free(cache_layers);
        const glue = try forward.Glue.load(d);
        const nat = try native_ops.Kernels.load(d);
        const exk = try experts.Kernels.load(d, @intCast(try ctx.attribute(.max_shared_memory_per_block_optin)));
        memMark(ctx, rank, "glue / native / experts kernel modules", &mt);
        const bufs = try forward.Buffers.init(d, cfg, heads, world, @intCast(partRows(o, world)), vshare, &inv, &had, keys, o.dcp, true, if (o.sp_parts > 0 and o.ring) forward.sp_min_rows else @intCast(partRows(o, world)));
        memMark(ctx, rank, "main buffers (prompt part 0)", &mt);
        const vbufs = try forward.Buffers.init(d, cfg, heads, world, vrows, vshare, &inv, &had, keys, o.dcp, true, vrows);
        const vxs = if (first_moe) |x| try moe.Scratch.init(d, x, vrows, @intCast(cfg.top_k)) else undefined;
        memMark(ctx, rank, "decode window buffers + experts scratch", &mt);
        // TF_GLM53_POOL_TEST: the caches as a pool of 256-position blocks (the usable ones, then Z), rows through a fixed
        // permutation (P1 of glm53/POOL.md)
        // TF_GLM53_POOL_TEST: キャッシュを 256 位置のブロックの pool に (使えるブロック、続いて Z)。行は決まった並べ替えで
        // 引く (glm53/POOL.md の P1)
        // TF_GLM53_PREFIX=pool: the same layout over the pool's blocks; until the first request's plan the table maps
        // the capacity's blocks one to one (the start-up prompts write there)
        // TF_GLM53_PREFIX=pool: 同じ並びを pool のブロックで。最初の要求のプランまでは、表は容量分のブロックをそのまま
        // 指す (起動時の prompt がそこに書く)
        const pool_test = try pool_devmap.testModeFromEnv();
        const pool_env = try pool_devmap.Env.fromEnv(o.capacity);
        if (pool_test != null and pool_env.on) return error.PoolTestWithPool;
        if ((pool_test != null or pool_env.on) and o.dcp != 1 and o.dcp != 2) return error.PoolNeedsDcp1Or2;
        const cap_blocks = (o.capacity + pool_devmap.block - 1) / pool_devmap.block;
        const pool_usable = if (pool_env.on) pool_env.blocks() else cap_blocks + pool_devmap.test_spare;
        // POOL.md 4.4: X's table never needs more than ceil(capacity / 256) blocks; the start-up bound keeps a margin
        // POOL.md 4.4: X の表は ceil(capacity / 256) ブロックを超えない。起動時の下限は余裕を持たせる
        if (pool_env.on and pool_usable < (o.capacity + 8 + pool_devmap.block - 1) / pool_devmap.block + 1) return error.PoolTooSmall;
        const state_rows = if (pool_test != null or pool_env.on) (pool_usable + 1) * pool_devmap.block / o.dcp - 1 else (o.capacity + o.dcp - 1) / o.dcp;
        // the pool is the window and the store at once, allocated here: the checkpoint's page cache goes first (GB10's
        // device allocator does not reclaim it; the slot store drops it the same way before its own allocation)
        // pool は窓とストアを一度にここで確保する: 先にチェックポイントのページキャッシュを捨てる (GB10 のデバイスの
        // アロケータはそれを回収しない。スロットのストアも自分の確保の前に同じく捨てる)
        if (pool_env.on) {
            var files: usize = 0;
            var dropped: u64 = 0;
            slots_serve_mod.dropFileCache(io, o.ckpt, 3, &files, &dropped);
            if (std.c.getenv("TF_GLM53_EXPERTS_DIR")) |x| slots_serve_mod.dropFileCache(io, std.mem.span(x), 3, &files, &dropped);
            if (rank == 0) std.log.info("glm53: pool: page cache of {d} checkpoint files ({d:.0} GiB) dropped before the pool", .{ files, @as(f64, @floatFromInt(dropped)) / (1 << 30) });
        }
        const state = try forward.State.init(gpa, d, cfg, cache_layers, state_rows, o.kv_fp8, o.kv_fp4);
        memMark(ctx, rank, "KV caches (the window)", &mt);
        e.* = .{
            .gpa = gpa, .io = io, .d = d, .rank = rank, .world = world, .cfg = cfg, .heads = heads, .group = group,
            .blas = blas, .handle = handle, .workspace = ws, .stream = stream,
            .x3 = x3,
            .glue = glue,
            .nat = nat,
            .exk = exk,
            .layers = layers, .embed = embed, .lm_head = lm_head, .final_norm = final_norm, .vshare = vshare,
            // with SP under ring, only windows below sp_min_rows reduce through these (wider prompt chunks go to
            // promptSp, whose parts reduce through hpart)
            // ring の SP では、これで reduce するのは sp_min_rows 未満のウィンドウだけ (広い prompt チャンクは promptSp
            // へ行き、その part は hpart で reduce する)
            .part_rows = partRows(o, world),
            .bufs = bufs,
            .vbufs = vbufs,
            // the routed experts' scratch, sized from the first MoE layer (rows x top-k slots)
            // routed expert の作業領域。最初の MoE 層から大きさを決める (行数 x top-k スロット)
            .xs = undefined, // after pe_ok (xs_rows) / pe_ok の後で (xs_rows)
            .vxs = vxs,
            // DCP: position p on member p % dcp at local row p / dcp (ceil(capacity / dcp) + 1 rows a member)
            // DCP: 位置 p はメンバー p % dcp のローカル行 p / dcp に置く (メンバーごとに ceil(capacity / dcp) + 1 行)
            .state = state,
            .mtp = mtp,
            .kv8 = o.kv_fp8,
            .kv4 = o.kv_fp4,
            .dcp = o.dcp,
            .capacity = o.capacity,
            .graphs = o.graphs,
        };
        if (o.dcp > 1) {
            if (world % o.dcp != 0) return error.DcpDoesNotDivideWorld;
            e.dgroup = try group.split(rank / o.dcp, rank % o.dcp);
        }
        if (pool_test != null or pool_env.on) {
            const mode = pool_test orelse .identity;
            const n_ent = pool_devmap.entries(keys);
            const tbl = try pool_devmap.testTable(gpa, mode, cap_blocks, pool_usable);
            defer gpa.free(tbl);
            // TF_GLM53_POOL_TEST_HI (tests): a narrower write range, so that a longer prompt's writes are refused
            // TF_GLM53_POOL_TEST_HI (試験用): 狭い書き込み範囲。それより長い prompt の書き込みは拒まれる
            const hi: usize = if (std.c.getenv("TF_GLM53_POOL_TEST_HI")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else o.capacity;
            const raw = try pool_devmap.bytes(gpa, 0, @intCast(hi), @intCast(pool_usable), tbl, n_ent);
            defer gpa.free(raw);
            e.pmap = try cuda.DeviceBuffer.fromHost(d, raw);
            if (rank == 0) std.log.info("glm53: KV caches as a pool of {d} blocks + Z ({d} local rows), {d} table entries, writes [0, {d}){s}{s}", .{ pool_usable, state_rows + 1, n_ent, o.capacity, if (pool_test != null) ", TF_GLM53_POOL_TEST=" else "", if (pool_test) |m| @tagName(m) else "" });
        }
        e.lgath = try cuda.DeviceBuffer.alloc(d, @as(usize, world) * vrows * e.vshare * 4);
        e.host_logits = try gpa.alloc(f32, @as(usize, world) * vrows * e.vshare);
        e.chk_send = try cuda.DeviceBuffer.alloc(d, 64 * 4);
        e.chk_recv = try cuda.DeviceBuffer.alloc(d, 64 * 4 * world);
        memMark(ctx, rank, "DCP group + logits gather", &mt);
        if (o.drafter) |dir| {
            // DFlash2: the target keeps the residual after each tap layer (b.taps)
            // DFlash2: ターゲットは tap 層ごとに、その後の residual を残す (b.taps)
            const dc = try dw.Config.read(gpa, io, dir, world);
            e.n_taps = @intCast(dc.n_taps);
            e.tap_slot = try gpa.alloc(i8, n);
            @memset(e.tap_slot, -1);
            for (dc.taps[0..dc.n_taps], 0..) |l, slot| if (l < n) {
                e.tap_slot[l] = @intCast(slot);
            };
            const row_bytes = dc.n_taps * cfg.hidden * 2;
            // a whole prompt chunk's taps (sequence-parallel parts write their rows into it)
            // prompt チャンク全体の tap (sequence-parallel の各部分が自分の行を書き込む)
            e.bufs.taps = try cuda.DeviceBuffer.alloc(d, @max(forward.max_rows, o.prompt_rows) * row_bytes);
            e.vbufs.taps = try cuda.DeviceBuffer.alloc(d, vrows * row_bytes);
            e.drafter = try drafter_mod.Drafter.create(gpa, io, d, dir, .{
                .stream = e.stream, .ops = .{ .k = &e.nat, .stream = e.stream, .kv8 = e.kv8, .kv4 = e.kv4 }, .glue = &e.glue, .group = group,
                .blas = &e.blas, .handle = e.handle, .embed = e.embed.ptr, .lm_head = e.lm_head.ptr, .vshare = vshare,
                .rank = rank, .world = world,
            });
            e.drafter.?.graphs = o.graphs;
        }
        memMark(ctx, rank, "DFlash2 drafter + taps", &mt);
        if (o.prompt_experts) if (first_moe) |fm| {
            e.pek = try prompt_experts.Kernels.load(d);
            e.pes = try prompt_experts.Scratch.init(d, fm, @intCast(e.part_rows), @intCast(cfg.top_k));
            e.pe_ok = try gpa.alloc(bool, n);
            for (layers, 0..) |*L, i| e.pe_ok[i] = if (L.experts) |*x| try prompt_experts.supported(gpa, x) else false;
        };
        e.xs_rows = e.part_rows;
        if (e.pek != null) {
            const all_pe = for (layers, 0..) |*L, i| {
                if (L.experts != null and !e.pe_ok[i]) break false;
            } else true;
            if (all_pe) e.xs_rows = @min(e.part_rows, forward.MAX_LIN_ROWS);
        }
        if (first_moe) |x| e.xs = try moe.Scratch.init(d, x, @intCast(e.xs_rows), @intCast(cfg.top_k));
        memMark(ctx, rank, "prompt experts scratch + routed experts scratch", &mt);
        if (o.unpack_cache_mb > 0) e.ucache = try unpack_cache.Cache.init(gpa, d, e.layers, o.unpack_cache_mb << 20);
        memMark(ctx, rank, "unpack cache", &mt);
        if (o.roce) try e.initRoce(o);
        memMark(ctx, rank, "RoCE", &mt);
        e.prompt_rows = forward.max_rows;
        if (o.sp_parts > 0) try e.initSp(o, &inv, &had, keys, first_moe orelse return error.SpNeedsMoeLayer);
        memMark(ctx, rank, "sequence-parallel parts (buffers of parts 1..)", &mt);
        if (o.prompt_rows > forward.max_rows and e.n_sp == 0) return error.PromptRowsNeedSp;
        // the init-time legacy-stream memsets land before the non-blocking stream reads them
        // 初期化時の legacy stream の memset を、non-blocking stream が読む前に終わらせる
        try ctx.synchronize();
        return e;
    }

    /// The RoCE setup: the world's runtime, then (DCP_ROCE) one a DCP group; a runtime every rank built or none
    /// (Runtime.init fails on every rank alike).
    /// RoCE の準備: world のランタイム、続いて (DCP_ROCE なら) DCP グループごとに 1 つ。全ランクが作るか、どのランクも
    /// 作らないかのどちらか (Runtime.init は全ランクで同じく失敗する)。
    fn initRoce(e: *Engine, o: Options) !void {
        const F = struct {
            fn env(a: [:0]const u8, b: [:0]const u8) ?[]const u8 {
                if (std.c.getenv(a)) |v| return std.mem.span(v);
                if (std.c.getenv(b)) |v| return std.mem.span(v);
                return null;
            }
        };
        const hca_env = F.env("B12X_ROCE_HCA", "NCCL_IB_HCA") orelse return error.RoceNoHca;
        var names: [2][:0]const u8 = undefined;
        var nh: usize = 0;
        var it = std.mem.tokenizeScalar(u8, hca_env, ',');
        while (it.next()) |h| : (nh += 1) {
            if (nh == 2) break;
            const name = std.mem.trimStart(u8, h, "=^");
            names[nh] = try e.gpa.dupeSentinel(u8, name[0 .. std.mem.indexOfScalar(u8, name, ':') orelse name.len], 0);
        }
        const gid = try std.fmt.parseInt(u32, F.env("B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX") orelse "3", 10);
        const spin = try std.fmt.parseInt(u32, F.env("B12X_ROCE_SPIN_LIMIT", "B12X_ROCE_SPIN_LIMIT") orelse "20000000", 10);
        const lib = F.env("TF_ROCE_PROXY_LIB", "TF_ROCE_PROXY_LIB") orelse "./libroce_proxy.so";
        e.rk = try roce.Kernels.load(e.d);
        e.fast = try roce.Runtime.init(e.gpa, e.d, e.group, &e.rk, lib, names[0..nh], gid, spin, e.stream);
        if (e.dgroup != null and o.dcp_roce) e.dfast = try roce.Runtime.init(e.gpa, e.d, &e.dgroup.?, &e.rk, lib, names[0..nh], gid, spin, e.stream);
        e.decode_bf16 = o.decode_bf16;
        if (e.decode_bf16) {
            // dpart / dred: [small, D] bf16 in every buffer set (prompt chunks of up to 16 rows too)
            // dpart / dred: すべてのバッファの組に [small, D] bf16 (16 行までの prompt チャンクにも)
            const n = 16 * e.cfg.hidden * 2;
            for ([_]*forward.Buffers{ &e.vbufs, &e.bufs }) |b| {
                b.dpart = try cuda.DeviceBuffer.alloc(e.d, n);
                b.dred = try cuda.DeviceBuffer.alloc(e.d, n);
            }
        }
        // the drafter's row and candidate gathers too
        // drafter の行と候補の gather も
        if (e.drafter) |dr| dr.t.fast = &e.fast.?;
        if (e.rank == 0) std.log.info("glm53: decode collectives on RoCE one-shot (world{s}{s})", .{ if (e.dfast != null) ", DCP group" else "", if (e.decode_bf16) ", bf16 reduce" else "" });
    }

    fn lap(e: *Engine, t: *std.Io.Timestamp, i: usize) void {
        const now = std.Io.Clock.awake.now(e.io);
        e.round_ns[i] += @intCast(t.durationTo(now).toNanoseconds());
        t.* = now;
    }

    /// RoCE health (a timed-out wait or a failed proxy poisons the runtime): checked after each synchronized round.
    /// RoCE の健全性 (待ちのタイムアウトや proxy の失敗はランタイムを poison する)。同期した各ラウンドのあとに
    /// 確かめる。
    fn roceCheck(e: *const Engine) !void {
        if (e.fast) |*f| if (!f.healthy()) return error.RoceFailed;
        if (e.dfast) |*f| if (!f.healthy()) return error.RoceFailed;
    }

    /// The rows the main buffers and each SP part's hold: forward.max_rows, or the widest part of a prompt chunk of
    /// prompt_rows rows in sp_parts parts (padded to the world) when wider: chunks of 12,288 rows in 3 parts give
    /// parts of 4,096, whose routed experts read and decode each expert's weights for twice the rows of 2,048.
    /// main のバッファと SP の各 part のバッファの行数: forward.max_rows、または prompt_rows 行のチャンクを sp_parts に
    /// 分けた最も広い part (world に揃える) の方が広ければそれ。12,288 行のチャンクを 3 つに分けると part は 4,096 行で、
    /// routed expert は各 expert の重みを読んで復号する 1 回で 2,048 行の倍の行を処理する。
    fn partRows(o: Options, world: usize) usize {
        if (o.sp_parts == 0) return forward.max_rows;
        const pr = (o.prompt_rows + o.sp_parts - 1) / o.sp_parts;
        return @max(forward.max_rows, (pr + world - 1) / world * world);
    }

    /// The sequence-parallel prompt setup; each part's padded rows must fit its buffers' part_rows.
    /// sequence-parallel の prompt の準備。各部分の padding 後の行数は、そのバッファの max_rows に収まらねばならない。
    fn initSp(e: *Engine, o: Options, inv: []const u8, had: []const u8, keys: usize, first_moe: *const moe.Routed) !void {
        if (o.dcp < 2) return error.SpNeedsDcp;
        if (o.sp_parts < 2 or o.sp_parts > forward.sp_max_parts) return error.SpParts;
        const d = e.d;
        e.sp_want = o.sp_parts;
        e.ring = o.ring;
        e.topk_words = o.topk_words;
        e.fuse = o.fuse;
        e.xgroup = o.xgroup;
        e.topk_any = o.topk_any;
        e.pe_f16 = o.pe_f16;
        e.topk16 = o.topk16;
        e.index_fp8 = o.index_fp8;
        e.peer_q = e.layers.len > 0 and e.layers[0].peer_q_b != null;
        e.n_sp = o.sp_parts;
        e.prompt_rows = o.prompt_rows;
        e.comm = try cuda.Stream.initPriority(d, o.comm_priority);
        e.ready = try cuda.Event.init(d, false);
        const W: usize = e.world;
        const xo_rows = (e.part_rows + W - 1) / W;
        // the extra parts run sequence-parallel chunks alone; when the direct attention takes those whole (dcpAttend:
        // one chunk of keys, FP8 KV, 16 heads a program), nothing reads their partials
        // 追加の部分は sequence-parallel チャンクだけを実行する。直接の attention がそれを丸ごと扱うとき (dcpAttend:
        // キーのチャンク 1 つ、FP8 KV、プログラムあたり 16 head)、その部分和は誰も読まない
        const ops0: @FieldType(forward.Ctx, "ops") = .{ .k = &e.nat, .stream = e.stream, .kv8 = e.kv8, .kv4 = e.kv4, .dcp = e.dcp, .drank = e.rank % e.dcp, .pmap = e.pmapPtr() };
        const direct_only = e.fuse and e.cfg.index_topk == 2048 and ops0.canAttnDirect(e.dcp, e.heads);
        for (0..e.n_sp) |k| {
            const b: *forward.Buffers = if (k == 0) &e.bufs else &e.sp_bufs[k - 1];
            const xs: *moe.Scratch = if (k == 0) &e.xs else &e.sp_xs[k - 1];
            if (k > 0) {
                // under ring these parts reduce through hpart: their TP reduce scratch serves only the head's argmax
                // (pickArgmax: one row) when the prompt's last row falls in them
                // ring ではこれらの part は hpart で reduce する: TP の reduce の作業域を使うのは、prompt の最後の行が
                // 入ったときの head の argmax (pickArgmax: 1 行) だけ
                b.* = try forward.Buffers.init(d, e.cfg, e.heads, e.world, @intCast(e.part_rows), e.vshare, inv, had, keys, e.dcp, !direct_only, if (o.ring) 16 else @intCast(e.part_rows));
                xs.* = try moe.Scratch.init(d, first_moe, @intCast(e.xs_rows), @intCast(e.cfg.top_k));
                b.taps = e.bufs.taps;
                // the indexer's scores (sel_rows x the key bucket / dcp x 8 bytes: 512 MiB at a window of 512K) only
                // live within spSelect, which every part runs on the comm stream one after another: part 0's buffer
                // serves them all (the same size, the scores and top-k of one part done before the next's start)
                // indexer の点数 (sel_rows x キーの bucket / dcp x 8 バイト。ウィンドウ 512K で 512 MiB) は spSelect の
                // 中でしか使わず、spSelect はどの part も通信ストリームで順に走る: part 0 のバッファを全 part で使う
                // (大きさは同じで、ある part の点数と top-k は次の part が始まる前に終わる)
                b.sc.free();
                b.sc = e.bufs.sc;
            }
            b.xo = try cuda.DeviceBuffer.alloc(d, xo_rows * e.cfg.hidden * 2);
            b.ppos = try cuda.DeviceBuffer.alloc(d, 64);
            if (o.ring) {
                b.hpart = try cuda.DeviceBuffer.alloc(d, e.part_rows * e.cfg.hidden * 2);
                b.hred = try cuda.DeviceBuffer.alloc(d, xo_rows * e.cfg.hidden * 2);
            }
            try b.ppos.?.fill8(0, null);
            e.sp_parts[k] = .{ .main = e.ctxOf(b, xs, e.stream), .side = e.ctxOf(b, xs, e.comm), .done = try cuda.Event.init(d, false) };
            e.sp_parts[k].main.sp = true;
            e.sp_parts[k].side.sp = true;
        }
        // the largest chunk's parts must fit: ceil(ceil(rows / P) / W) W <= part_rows
        // 最大のチャンクの各部分が収まること: ceil(ceil(rows / P) / W) W <= max_rows
        const pr = (e.prompt_rows + e.sp_want - 1) / e.sp_want;
        if ((pr + W - 1) / W * W > e.part_rows) return error.SpPartTooWide;
    }

    /// Destroys the decode graphs (before the communicator: NCCL holds a captured collective's plan until its graph
    /// is gone, and ncclCommDestroy waits on them). The device buffers go with the context.
    /// decode グラフを破棄する (communicator より先に: NCCL は capture した collective の計画をグラフが消えるまで持ち、
    /// ncclCommDestroy はそれを待つ)。デバイスバッファは context とともに消える。
    pub fn destroy(e: *Engine) void {
        if (e.drafter) |dr| dr.destroyGraphs();
        for (e.execs[0..e.n_graphs]) |*x| x.deinit();
        e.n_graphs = 0;
        if (e.dfast) |*f| f.deinit();
        if (e.fast) |*f| f.deinit();
        if (e.dgroup) |*g| g.deinit();
        if (e.slots) |sl| sl.deinit();
        e.out.deinit(e.gpa);
        e.blas.close();
        e.gpa.destroy(e);
    }

    /// The forward context over the prompt buffers (`decode` false) or the decode buffers.
    /// prompt バッファ (`decode` が false) か decode バッファの上の forward context。
    fn pmapPtr(e: *const Engine) u64 {
        return if (e.pmap) |b| b.ptr else 0;
    }

    /// The pool map's error flag after a request (both streams done): a write the kernels refused (outside
    /// [write_lo, write_hi), or into Z) is fatal. Every rank's flag is gathered, so a write one rank refused stops
    /// every rank here together (not that rank alone, the others waiting in the next request: POOL.md 12.2e finding 1).
    /// 要求の後の pool の表の異常フラグ (両方の stream が終わってから): カーネルが拒んだ書き込み ([write_lo, write_hi) の外、
    /// または Z への書き込み) は致命的。全ランクのフラグを集めるので、1 ランクが拒んだ書き込みでも全ランクがここで
    /// そろって止まる (そのランクだけが止まり、他は次の要求で待ち続けることはない: POOL.md 12.2e の見つかったこと 1)。
    pub fn poolCheck(e: *Engine) !void {
        const m = e.pmap orelse return;
        try e.stream.synchronize();
        if (e.n_sp > 0) try e.comm.synchronize();
        var err: [4]u8 = undefined;
        try m.download(8, &err);
        const v = std.mem.readInt(i32, &err, .little);
        if (v != 0) std.log.err("glm53: rank {d}: the pool map's error flag is {d} (1: a write outside [write_lo, write_hi), 2: into the zero block)", .{ e.rank, v });
        var bad: u32 = @intFromBool(v != 0);
        if (e.world > 1) {
            try e.chk_send.uploadAsync(0, std.mem.asBytes(&v), e.stream.handle);
            const lib = e.group.lib;
            try lib.check(lib.api.ncclAllGather(e.chk_send.ptr, e.chk_recv.ptr, 1, .i32, e.group.comm, e.stream.handle), "ncclAllGather");
            var all: [64]i32 = undefined;
            try e.chk_recv.downloadAsync(0, std.mem.sliceAsBytes(all[0..e.world]), e.stream.handle);
            try e.stream.synchronize();
            bad = 0;
            for (all[0..e.world], 0..) |f, r| if (f != 0) {
                bad += 1;
                if (v == 0) std.log.err("glm53: rank {d}: rank {d} refused a pool write (flag {d}); every rank stops", .{ e.rank, r, f });
            };
        }
        if (bad != 0) return error.PoolWriteRefused;
    }

    pub fn fw(e: *Engine, decode: bool) forward.Ctx {
        return if (decode) e.ctxOf(&e.vbufs, &e.vxs, e.stream) else e.ctxOf(&e.bufs, &e.xs, e.stream);
    }

    fn ctxOf(e: *Engine, b: *forward.Buffers, xs: *const moe.Scratch, stream: cuda.Stream) forward.Ctx {
        return .{
            .ucache = if (e.ucache) |*u| u else null, .sid = if (e.n_sp > 0 and stream.handle == e.comm.handle) 1 else 0,
            .cfg = e.cfg, .heads = e.heads, .world = e.world, .rank = e.rank, .stream = stream,
            .ops = .{ .k = &e.nat, .stream = stream, .kv8 = e.kv8, .kv4 = e.kv4, .dcp = e.dcp, .drank = e.rank % e.dcp, .pmap = e.pmapPtr() }, .x3 = &e.x3, .glue = &e.glue, .blas = &e.blas,
            .handle = e.handle, .group = e.group, .b = b, .ex = &e.exk,
            .xs = xs, .tap_slot = e.tap_slot, .n_taps = e.n_taps,
            .dgroup = if (e.dgroup) |*g| g else null, .probe = e.probe, .probe_ctx = e.probe_ctx,
            .pek = if (e.pek) |*k| k else null, .pes = &e.pes, .pe_ok = e.pe_ok, .ring = e.ring, .topk_words = e.topk_words, .topk_any = e.topk_any, .pe_f16 = e.pe_f16, .topk16 = e.topk16, .fuse = e.fuse, .xgroup = e.xgroup, .peer_q = e.peer_q, .index_fp8 = e.index_fp8, .probe_select = e.probe_select,
            .fast = if (e.fast) |*f| f else null, .dfast = if (e.dfast) |*f| f else null, .decode_bf16 = e.decode_bf16,
        };
    }

    /// The longest prompt a request may have (the reply needs at least one position).
    /// 要求が持てる最長の prompt (応答に少なくとも 1 位置が要る)。
    pub fn promptLimit(e: *const Engine) usize {
        return e.capacity - 1;
    }

    /// tests: from here every decode window's graph also copies each index layer's picks to a log (DCP > 1 only),
    /// and scoreForced hands the log to sel_sink. Graphs captured before this call lack the copies: they are dropped.
    /// テスト用: これ以降、各 decode ウィンドウのグラフは index 層ごとの選択をログにも写し (DCP > 1 のみ)、scoreForced
    /// がそのログを sel_sink に渡す。この呼び出し前に capture したグラフは写しを持たないので捨てる。
    pub fn selLogEnable(e: *Engine) !void {
        if (e.dgroup == null) return error.SelLogNeedsDcp;
        comptime std.debug.assert(vrows <= forward.sel_log_rows);
        var n: u32 = 0;
        for (e.layers) |*L| {
            if (L.indexer != null) n += 1;
        }
        const K: usize = e.cfg.index_topk;
        const tok_bytes = @as(usize, n) * forward.sel_log_rows * K * 4;
        const cnt_bytes = @as(usize, n) * forward.sel_log_rows * 4;
        e.sel_tok = try cuda.DeviceBuffer.alloc(e.d, tok_bytes);
        e.sel_cnt = try cuda.DeviceBuffer.alloc(e.d, cnt_bytes);
        try e.sel_tok.?.fill8(0, e.stream.handle);
        try e.sel_cnt.?.fill8(0, e.stream.handle);
        e.sel_host = try e.gpa.alloc(u8, tok_bytes + cnt_bytes);
        e.sel_layers = n;
        try e.stream.synchronize();
        for (e.execs[0..e.n_graphs]) |*x| x.deinit();
        e.n_graphs = 0;
    }

    /// tests: teacher-forced scoring. tokens[0 .. p0] through the prefill (the request path's chunks), then the rest
    /// through decode windows of up to `win` rows whose ids are the reference tokens (a verify window whose drafts are
    /// all right): row i of a window at P is position P + i, its logits predict token P + i + 1. Every rank calls it
    /// alike; `row` gets each position's full-vocabulary fp32 logits (world * vshare, rank-major as the all-gather
    /// lays them out) on every rank, positions p0 .. tokens.len - 2.
    /// テスト用: teacher forcing による採点。tokens[0 .. p0] を prefill (要求と同じチャンク) に通し、残りを最大 `win`
    /// 行の decode ウィンドウに通す。ウィンドウの id は参照トークンそのもの (draft が全部正しい verify ウィンドウ)。
    /// 位置 P のウィンドウの行 i は位置 P + i で、その logits はトークン P + i + 1 を予測する。全ランクが同じく呼ぶ。
    /// `row` は全ランクで、位置 p0 .. tokens.len - 2 それぞれの語彙全体の fp32 logits (world * vshare、all-gather
    /// が並べるランク順) を受け取る。
    pub fn scoreForced(e: *Engine, tokens: []const i64, p0: usize, win: u32, ctx: *anyopaque, row: *const fn (ctx: *anyopaque, pos: usize, logits: []const f32) anyerror!void) !void {
        if (p0 == 0 or p0 >= tokens.len or tokens.len > e.capacity) return error.PromptLength;
        if (!e.graphs) return error.DraftsNeedGraphs;
        if (win == 0 or win > vrows) return error.TooManyDrafts;
        e.start = 0;
        e.dr_snap_end = null;
        _ = (try e.prefill(tokens[0..p0], null)) orelse return error.Stopped;
        const W = e.world;
        const vs: usize = e.vshare;
        const full = try e.gpa.alloc(f32, W * vs);
        defer e.gpa.free(full);
        const lib = e.group.lib;
        var P = p0;
        while (P + 1 < tokens.len) {
            const R: u32 = @intCast(@min(win, tokens.len - 1 - P));
            try e.vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(tokens[P..][0..R]), e.stream.handle);
            try e.state.pos.fill32(@intCast(P), e.stream.handle);
            const t = bucket(P + R, e.cfg.index_topk);
            const slot = for (0..e.n_graphs) |i| {
                if (e.keys[i] == t and e.key_rows[i] == R) break i;
            } else try e.capture(R, t);
            try e.execs[slot].launchOn(e.stream);
            const n = @as(usize, R) * vs;
            try lib.check(lib.api.ncclAllGather(e.vbufs.lpart.ptr, e.lgath.ptr, n, .f32, e.group.comm, e.stream.handle), "ncclAllGather");
            try e.lgath.downloadAsync(0, std.mem.sliceAsBytes(e.host_logits[0 .. W * n]), e.stream.handle);
            try e.stream.synchronize();
            try e.roceCheck();
            if (e.sel_sink) |f| if (t != null) {
                const cb = @as(usize, e.sel_layers) * forward.sel_log_rows * 4;
                try e.sel_tok.?.download(0, e.sel_host[0 .. e.sel_host.len - cb]);
                try e.sel_cnt.?.download(0, e.sel_host[e.sel_host.len - cb ..]);
                try f(e.sel_sink_ctx, P, R, e.sel_layers, e.sel_host[0 .. e.sel_host.len - cb], e.sel_host[e.sel_host.len - cb ..]);
            };
            for (0..R) |i| {
                for (0..W) |r| @memcpy(full[r * vs ..][0..vs], e.host_logits[(r * R + i) * vs ..][0..vs]);
                try row(ctx, P + i, full);
            }
            P += R;
        }
    }

    /// One request on every rank (the same arguments everywhere: lockstep.zig shares them first). `max_tokens` counts
    /// the first token; `stop_eos` ends the reply at (and including) an EOS id. The reply goes to `sink` (rank 0).
    /// 全ランクで 1 つの要求を実行する (引数はどこでも同じ: lockstep.zig が先に共有する)。`max_tokens` は最初の
    /// トークンも数える。`stop_eos` なら EOS id で応答を終える (EOS も含む)。応答は `sink` へ (rank 0)。
    pub fn generate(e: *Engine, prompt: []const i64, max_tokens: usize, stop_eos: bool, sink: ?Sink) !Stats {
        if (prompt.len == 0 or prompt.len > e.promptLimit()) return error.PromptLength;
        const want = @max(1, @min(max_tokens, e.capacity - prompt.len));
        var st: Stats = .{ .prompt = prompt.len };
        // the drafter sees none of this request's rows: its ring and snapshot are stale after it
        // drafter はこの要求の行を見ないので、要求のあとはその ring と snapshot が古くなる
        e.dr_snap_end = null;
        const t0 = std.Io.Clock.awake.now(e.io);
        var tok = (try e.prefill(prompt, null)) orelse {
            try e.poolCheck();
            return e.stoppedInPrefill(st, t0);
        };
        st.prefill_ns = @intCast(t0.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds());
        if (sink) |s| if (s.prefilled) |f| f(s.ctx);
        const t1 = std.Io.Clock.awake.now(e.io);
        var P = prompt.len;
        var n: usize = 1;
        try e.emit(sink, tok);
        var stop = stop_eos and e.cfg.isEos(tok);
        var stopped = false; // the ranks agreed to stop (rank 0's stop_req) / 停止に合意した (rank 0 の stop_req)
        if (e.graphs and !stop and n < want) {
            const first = [1]i64{tok};
            try e.vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(&first), e.stream.handle);
        }
        // tests: TF_ZIG_PROF_SERIAL=a,b - a CUDA profiler range (cuProfilerStart / Stop) around tokens a..b
        // テスト用: TF_ZIG_PROF_SERIAL=a,b - トークン a..b を CUDA プロファイラの範囲 (cuProfilerStart / Stop) で囲む
        var prof = [2]usize{ 0, 0 };
        if (std.c.getenv("TF_ZIG_PROF_SERIAL")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            for (&prof) |*x| x.* = std.fmt.parseInt(usize, it.next() orelse "0", 10) catch 0;
        }
        while (!stop and n < want) : (P += 1) {
            if (try e.stopAgreed()) {
                stopped = true;
                break;
            }
            if (prof[1] > 0 and n == prof[0]) _ = e.d.api.cuProfilerStart();
            if (prof[1] > 0 and n == prof[1]) _ = e.d.api.cuProfilerStop();
            tok = if (e.graphs) try e.roundGraph(P) else try e.roundEager(P, tok);
            // the row at P predicts P + 1; the next round's graph reads its token from ids[0]
            // 位置 P の行は P + 1 を予測する。次のラウンドのグラフはトークンを ids[0] から読む
            if (e.samp) |s| {
                var one: [1]u32 = undefined;
                _ = try e.sampleRows(e.vbufs.lpart.ptr, 1, P + 1, s, &.{}, &one);
                tok = one[0];
                if (e.graphs) {
                    const next = [1]i64{tok};
                    try e.vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(&next), e.stream.handle);
                }
            }
            n += 1;
            try e.emit(sink, tok);
            stop = stop_eos and e.cfg.isEos(tok);
        }
        st.generated = n;
        st.finish = if (stopped) .cancelled else if (stop) .stop else .length;
        st.decode_ns = @intCast(t1.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds());
        try e.poolCheck();
        return st;
    }

    /// Drafted decoding: each round the proposer's drafts after the pending token, one verify window of R = 1 + drafts
    /// rows at P (key range bucket(P + R)), the longest run of drafts equal to the window's picks kept, then the first
    /// pick past it. The reply equals `generate`'s: every token is the target's own pick, and rows of a window of up to
    /// 16 take the same kernels as one row. Rejected drafts' cache rows lie past the new P; the next window overwrites
    /// them before any row reads them.
    /// draft 付きの decode: 各ラウンドで、保留中のトークンに続く proposer の draft を受け、位置 P で R = 1 + draft 数の
    /// verify ウィンドウを 1 つ実行し (キー範囲 bucket(P + R))、ウィンドウの選択と一致する draft の最長の並びと、その
    /// 次の選択を残す。応答は `generate` と同じ: どのトークンもターゲット自身の選択で、16 行までのウィンドウの行は 1 行
    /// と同じカーネルを通る。棄却した draft のキャッシュ行は新しい P より後ろにあり、どの行が読むよりも前に次の
    /// ウィンドウが上書きする。
    pub fn generateDrafted(e: *Engine, prompt: []const i64, max_tokens: usize, stop_eos: bool, sink: ?Sink, prop: Proposer, depth: usize) !Stats {
        if (prompt.len == 0 or prompt.len > e.promptLimit()) return error.PromptLength;
        if (!e.graphs) return error.DraftsNeedGraphs;
        const want = @max(1, @min(max_tokens, e.capacity - prompt.len));
        var st: Stats = .{ .prompt = prompt.len };
        if (prop.begin) |f| try f(prop.ctx, prompt);
        const t0 = std.Io.Clock.awake.now(e.io);
        var tok = (try e.prefill(prompt, prop)) orelse {
            try e.poolCheck();
            return e.stoppedInPrefill(st, t0);
        };
        st.prefill_ns = @intCast(t0.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds());
        if (sink) |s| if (s.prefilled) |f| f(s.ctx);
        // tests: TF_ZIG_PROF_DECODE=1 - a CUDA profiler range (cuProfilerStart / Stop) around the drafted rounds only
        // テスト用: TF_ZIG_PROF_DECODE=1 - draft ラウンドだけを CUDA プロファイラの範囲 (cuProfilerStart / Stop) で囲む
        const prof = std.c.getenv("TF_ZIG_PROF_DECODE") != null;
        if (prof) _ = e.d.api.cuProfilerStart();
        defer if (prof) {
            _ = e.d.api.cuProfilerStop();
        };
        const t1 = std.Io.Clock.awake.now(e.io);
        var P = prompt.len;
        var n: usize = 1;
        try e.emit(sink, tok);
        var stop = stop_eos and e.cfg.isEos(tok);
        var stopped = false; // the ranks agreed to stop (rank 0's stop_req) / 停止に合意した (rank 0 の stop_req)
        var window: [vrows]i64 = undefined;
        var drafts: [vrows]u32 = undefined;
        var picks: [vrows]i32 = undefined;
        const depth_max = @min(depth, vrows - 1);
        while (!stop and n < want) {
            const room = @min(depth_max, want - n - 1, e.capacity - P - 1);
            // every rank proposed from the same state and holds the same drafts, checked once after the proposal: the
            // drafter's collectives have one shape whatever P / tok / room are, so a rank that differed cannot hang
            // them, and the window (whose shape the drafts set) only starts after the check
            // 全ランクは同じ状態から提案し、同じ draft を持つ。提案のあとに一度だけ照合する: drafter の collective は
            // P / tok / room によらず形が 1 つなので、違うランクがあっても止まらない。ウィンドウ (形は draft で
            // 決まる) は照合のあとにだけ始まる
            var tp = std.Io.Clock.awake.now(e.io);
            const nd = if (room > 0) try prop.propose(prop.ctx, tok, P, room, drafts[0..room]) else 0;
            e.lap(&tp, 1);
            if (nd > room) return error.TooManyDrafts;
            // [rank 0's stop, P, tok, room, nd, drafts ..] (16 words: whole 16-byte units for RoCE)
            // [rank 0 の停止, P, tok, room, nd, draft ..] (16 語: RoCE の 16 バイト単位にそろう)
            var words: [16]i32 = @splat(-1);
            words[0] = @intFromBool(e.stop_req.load(.monotonic));
            words[1] = @intCast(P);
            words[2] = @intCast(tok);
            words[3] = @intCast(room);
            words[4] = @intCast(nd);
            for (drafts[0..nd], 5..) |dd, i| words[i] = @intCast(dd);
            const halt = try e.agree(&words);
            e.lap(&tp, 2);
            if (halt) {
                stopped = true;
                break;
            }
            const R: u32 = @intCast(1 + nd);
            window[0] = tok;
            for (drafts[0..nd], 1..) |d, i| window[i] = d;
            try e.vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(window[0..R]), e.stream.handle);
            try e.state.pos.fill32(@intCast(P), e.stream.handle);
            const t = bucket(P + R, e.cfg.index_topk);
            const slot = for (0..e.n_graphs) |i| {
                if (e.keys[i] == t and e.key_rows[i] == R) break i;
            } else try e.capture(R, t);
            try e.execs[slot].launchOn(e.stream);
            try e.vbufs.tok_out.downloadAsync(0, std.mem.sliceAsBytes(picks[0..R]), e.stream.handle);
            try e.stream.synchronize();
            try e.roceCheck();
            var np: usize = R;
            // row i drawn at position P + i + 1, stopping after the first that differs from its draft
            // 行 i は位置 P + i + 1 で抽選し、draft と違った最初の行のあとで止める
            if (e.samp) |s| {
                var sp: [vrows]u32 = undefined;
                np = try e.sampleRows(e.vbufs.lpart.ptr, R, P + 1, s, drafts[0..nd], &sp);
                for (sp[0..np], 0..) |st_, i| picks[i] = @intCast(st_);
            }
            e.lap(&tp, 3);
            var k: usize = 0; // drafts equal to the picks before them / 手前の選択と一致した draft の数
            while (k < nd and k + 1 < np and drafts[k] == @as(u32, @intCast(picks[k]))) k += 1;
            st.rounds += 1;
            st.drafted += nd;
            st.accepted += k;
            if (prop.kept) |f| try f(prop.ctx, k + 1);
            if (prop.kept_ids) |f| try f(prop.ctx, picks[0 .. k + 1]);
            e.lap(&tp, 4);
            // emitted one by one: EOS and max_tokens end the reply inside a round
            // 1 つずつ出す: EOS と max_tokens はラウンドの途中でも応答を終える
            for (picks[0 .. k + 1]) |p| {
                tok = @intCast(p);
                n += 1;
                try e.emit(sink, tok);
                stop = stop_eos and e.cfg.isEos(tok);
                if (stop or n >= want) break;
            }
            P += k + 1;
        }
        st.generated = n;
        st.finish = if (stopped) .cancelled else if (stop) .stop else .length;
        st.decode_ns = @intCast(t1.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds());
        try e.poolCheck();
        return st;
    }

    /// A token of the reply: to the sink (rank 0), and kept for the slot cache's commit.
    /// 応答のトークン 1 つ: sink へ渡し (rank 0)、スロットキャッシュの commit 用に残す。
    fn emit(e: *Engine, sink: ?Sink, tok: u32) !void {
        if (sink) |s| s.tokens(s.ctx, &.{tok});
        if (e.slots != null or e.pool != null) try e.out.append(e.gpa, tok);
    }

    /// Every rank's `words` gathered: words[0] is rank 0's say (stop_req) and is returned, the rest must equal rank
    /// 0's on every rank (error.RanksDiverged otherwise).
    /// 全ランクの `words` を集める: words[0] は rank 0 の意思 (stop_req) で、それを返す。残りは全ランクで rank 0 の
    /// ものと一致しなければならない (違えば error.RanksDiverged)。
    fn agree(e: *Engine, words: []const i32) !bool {
        if (e.world == 1) return words[0] != 0;
        const n = words.len;
        try e.chk_send.uploadAsync(0, std.mem.sliceAsBytes(words), e.stream.handle);
        const lib = e.group.lib;
        if (e.fast != null and (n * 4) % 16 == 0) {
            try e.fast.?.allGather(e.stream, e.chk_send.ptr, e.chk_recv.ptr, @intCast(n * 4));
        } else try lib.check(lib.api.ncclAllGather(e.chk_send.ptr, e.chk_recv.ptr, n, .i32, e.group.comm, e.stream.handle), "ncclAllGather");
        var all: [64 * 16]i32 = undefined;
        try e.chk_recv.downloadAsync(0, std.mem.sliceAsBytes(all[0 .. n * e.world]), e.stream.handle);
        try e.stream.synchronize();
        // a poisoned one-shot writes nothing: the words would be a previous round's
        // poison された one-shot は何も書かないので、語は前のラウンドのものになってしまう
        try e.roceCheck();
        for (1..e.world) |r| if (!std.mem.eql(i32, all[r * n + 1 ..][0 .. n - 1], all[1..n])) return error.RanksDiverged;
        return all[0] != 0;
    }

    /// A prompt chunk's rows a .. a + rows - 1 (their last residual at `x`) into the MTP layer's caches (a chunk
    /// without sequence-parallel parts; promptSp fills them itself).
    /// prompt チャンクの行 a .. a + rows - 1 (最後の residual は `x`) を MTP 層のキャッシュへ入れる (sequence-parallel
    /// の分割がないチャンク用。promptSp は自分で埋める)。
    fn mtpPrompt(e: *Engine, m: *mtp_mod.Mtp, c: *const forward.Ctx, x: u64, prompt: []const i64, a: usize, rows: u32) !void {
        try m.prompt(c, e.state.kc[m.index].ptr, if (e.state.ic[m.index]) |b| b.ptr else null, e.embed.ptr, e.final_norm.ptr, x, prompt, a, rows);
    }

    /// The MTP layer as the drafted rounds' proposer (null without it: TF_GLM53_DRAFTER=mtp).
    /// MTP 層を draft ラウンドの proposer にする (MTP 層がなければ null: TF_GLM53_DRAFTER=mtp)。
    pub fn mtpProposer(e: *Engine) ?Proposer {
        if (e.mtp == null) return null;
        const F = struct {
            fn propose(ctx: *anyopaque, tok: u32, P: usize, room: usize, out: []u32) anyerror!usize {
                const en: *Engine = @ptrCast(@alignCast(ctx));
                const c = en.fw(true);
                const m = &en.mtp.?;
                const kc = en.state.kc[m.index].ptr;
                const ic = if (en.state.ic[m.index]) |b| b.ptr else null;
                return m.propose(&c, kc, ic, en.embed.ptr, en.lm_head.ptr, en.vshare, tok, P, room, bucket(P + room, en.cfg.index_topk), out);
            }
            fn keptIds(ctx: *anyopaque, picks: []const i32) anyerror!void {
                const en: *Engine = @ptrCast(@alignCast(ctx));
                try en.mtp.?.kept(en.stream, en.vbufs.fnormed.ptr, en.cfg.hidden, picks);
            }
        };
        return .{ .ctx = e, .propose = F.propose, .kept_ids = F.keptIds };
    }

    /// The DFlash2 drafter as the drafted rounds' proposer (null without one): depth from the caller, confidence 0.4.
    /// DFlash2 の drafter を draft ラウンドの proposer にする (drafter がなければ null): 深さは呼び出し側が決め、
    /// 信頼度は 0.4。
    pub fn drafterProposer(e: *Engine) ?Proposer {
        if (e.drafter == null) return null;
        const F = struct {
            fn propose(ctx: *anyopaque, tok: u32, P: usize, room: usize, out: []u32) anyerror!usize {
                _ = P;
                const en: *Engine = @ptrCast(@alignCast(ctx));
                return en.drafter.?.propose(tok, room, 0.4, en.samp, out[0..room]);
            }
            fn kept(ctx: *anyopaque, n: usize) anyerror!void {
                const en: *Engine = @ptrCast(@alignCast(ctx));
                try en.drafter.?.addTaps(en.vbufs.taps.?.ptr, n);
            }
            fn begin(ctx: *anyopaque, prompt: []const i64) anyerror!void {
                const en: *Engine = @ptrCast(@alignCast(ctx));
                en.dr_resume = null;
                if (en.dr_pool) |dp| if (en.start > 0) {
                    try en.drafter.?.resumeRange(en.start, dp.copy);
                    en.dr_resume = dp.kind;
                    return;
                };
                if (en.start > 0) { // the drafter continues at the resume point / drafter は再開位置から続ける
                    const rg = en.dr_ring orelse [3]?usize{ null, null, null };
                    en.dr_resume = try en.drafter.?.resumeAt(en.start, prompt.len, rg[0], rg[1], rg[2]);
                } else try en.drafter.?.reset();
            }
            fn taps(ctx: *anyopaque, src: u64, n: usize) anyerror!void {
                const en: *Engine = @ptrCast(@alignCast(ctx));
                try en.drafter.?.addTaps(src, n);
            }
        };
        return .{ .ctx = @ptrCast(e), .propose = F.propose, .kept = F.kept, .begin = F.begin, .taps = F.taps };
    }

    /// A prompt chunk's shape class (ShapeKey) by its rows and whether it selects.
    /// prompt チャンクの形の分類 (ShapeKey)。行数と選択の有無で決まる。
    fn shapeOf(e: *const Engine, rows: usize, select: bool) ShapeKey {
        const path: u8 = if (rows <= 16) 0 else if (rows <= 128) 1 else if (e.n_sp == 0 or rows < forward.sp_min_rows) 2 else if (e.sp_want >= 4 and rows >= 2 * forward.sp_min_rows) 4 else 3;
        const bm: u8 = if (rows <= 16) 16 else if (rows <= 32) 32 else if (rows <= 64) 64 else 128;
        return .{ .small = if (rows <= 16) @intCast(rows) else 0, .bm = bm, .path = path, .select = select };
    }

    fn noteShape(e: *Engine, k: ShapeKey) void {
        for (e.shapes[0..e.n_shapes]) |x| if (std.meta.eql(x, k)) return;
        if (e.n_shapes < e.shapes.len) {
            e.shapes[e.n_shapes] = k;
            e.n_shapes += 1;
        }
    }

    /// Before a request is shared (rank 0): every prompt chunk it will run - from `start`, the prefill's equal chunks
    /// - is of a shape the start-up prewarm ran; error.ShapeNotVerified otherwise (the request fails alone, every rank
    /// keeps serving). Without a prewarm (tests) everything passes.
    /// 要求を共有する前に (rank 0): 実行する prompt チャンク (`start` からの prefill の等分チャンク) がすべて、起動時の
    /// prewarm が実行した形であることを確かめる。違えば error.ShapeNotVerified (その要求だけが失敗し、全ランクは処理を
    /// 続ける)。prewarm がなければ (テスト) すべて通す。
    pub fn admit(e: *const Engine, prompt_len: usize, start: usize) !void {
        if (e.n_shapes == 0 or prompt_len <= start) return;
        const L0 = prompt_len - start;
        const chunks = (L0 + e.prompt_rows - 1) / e.prompt_rows;
        const step = (L0 + chunks - 1) / chunks;
        var a = start;
        while (a < prompt_len) : (a += step) {
            const end = @min(a + step, prompt_len);
            const k = e.shapeOf(end - a, end > e.cfg.index_topk);
            for (e.shapes[0..e.n_shapes]) |x| {
                if (std.meta.eql(x, k)) break;
            } else {
                std.log.warn("glm53: refusing a {d}-token prompt from {d}: its {d}-row chunk (class {any}) was not run at start-up", .{ prompt_len, start, end - a, k });
                return error.ShapeNotVerified;
            }
        }
    }

    /// A request the ranks stopped during its prompt: no token, the rows done so far in the caches (prefill_done).
    /// prompt の途中でランクが止めた要求: トークンはなく、ここまでの行はキャッシュにある (prefill_done)。
    fn stoppedInPrefill(e: *Engine, st0: Stats, t0: std.Io.Timestamp) Stats {
        var st = st0;
        st.prefill_ns = @intCast(t0.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds());
        st.finish = .cancelled;
        st.generated = 0;
        return st;
    }

    /// The request's stop point agreement: rank 0's stop_req to every rank (one all-gather).
    /// 要求の停止の合意: rank 0 の stop_req を全ランクへ配る (all-gather 1 回)。
    fn stopAgreed(e: *Engine) !bool {
        var w = [4]i32{ @intFromBool(e.stop_req.load(.monotonic)), 0, 0, 0 };
        return e.agree(&w);
    }

    /// The prompt in equal chunks of at most prompt_rows (key range: the chunk's end once past index_topk), the head on
    /// the last row; returns the first token, or null when the ranks agreed to stop between chunks.
    /// prompt を最大 prompt_rows 行の等分チャンクで実行し (キー範囲: index_topk を超えたらチャンクの終端)、最終行で
    /// head を計算する。最初のトークンを返す。チャンクの間でランクが停止に合意したら null。
    fn prefill(e: *Engine, prompt: []const i64, prop: ?Proposer) !?u32 {
        const c = e.fw(false);
        var head = &c;
        // rows below e.start are in the caches (the slot cache's resume); the rest takes the equal chunks a prompt of
        // len - start rows takes
        // e.start より前の行はキャッシュにある (スロットキャッシュの再開)。残りは、長さ len - start の prompt と同じ
        // 等分チャンクで実行する
        const L0 = prompt.len - e.start;
        if (L0 == 0) return error.NothingToPrefill;
        if (e.mtp) |*m| {
            m.begin();
            m.track = .begin(e.start, prompt.len);
        }
        const chunks = (L0 + e.prompt_rows - 1) / e.prompt_rows;
        const step = (L0 + chunks - 1) / chunks;
        var a: usize = e.start;
        e.prefill_done = e.start;
        // tests: TF_ZIG_PROF_PREFILL=a,b - a CUDA profiler range around prompt chunks a..b (0-based, inclusive); the
        // device is drained at both ends so the range holds exactly those chunks' work
        // テスト用: TF_ZIG_PROF_PREFILL=a,b - prompt チャンク a..b (0 始まり、両端を含む) を CUDA プロファイラの範囲で
        // 囲む。両端でデバイスを空にするので、範囲にはそのチャンクの処理だけが入る
        var pprof = [2]usize{ 0, 0 };
        const pprof_on = if (std.c.getenv("TF_ZIG_PROF_PREFILL")) |v| blk: {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            for (&pprof) |*x| x.* = std.fmt.parseInt(usize, it.next() orelse "0", 10) catch 0;
            break :blk true;
        } else false;
        // tests: TF_ZIG_PREFILL_LOG=1 - rank 0 prints each prompt chunk's wall time (its device drained at the chunk's
        // end, so a chunk's time holds that chunk's work alone)
        // テスト用: TF_ZIG_PREFILL_LOG=1 - rank 0 が prompt チャンクごとの経過時間を出す (チャンクの終わりでデバイスを
        // 空にするので、その時間にはそのチャンクの処理だけが入る)
        const plog = e.rank == 0 and std.c.getenv("TF_ZIG_PREFILL_LOG") != null;
        var plog_t = std.Io.Clock.awake.now(e.io);
        var ci: usize = 0;
        while (a < prompt.len) : ({
            a += step;
            ci += 1;
        }) {
            if (plog and ci > 0) {
                _ = e.d.api.cuCtxSynchronize();
                const tn = std.Io.Clock.awake.now(e.io);
                std.debug.print("[prefill] chunk {d} to row {d}: {d} us\n", .{ ci - 1, a, @divTrunc(plog_t.durationTo(tn).toNanoseconds(), 1000) });
                plog_t = tn;
            }
            if (pprof_on and (ci == pprof[0] or ci == pprof[1] + 1)) {
                _ = e.d.api.cuCtxSynchronize();
                _ = if (ci == pprof[0]) e.d.api.cuProfilerStart() else e.d.api.cuProfilerStop();
            }
            // rows [start, a) are in the caches (prefill_done)
            // 行 [start, a) はキャッシュにある (prefill_done)
            if (try e.stopAgreed()) return null;
            const end = @min(a + step, prompt.len);
            const rows: u32 = @intCast(end - a);
            const t: ?u32 = if (end > e.cfg.index_topk) @intCast(end) else null;
            if (e.recording_shapes) e.noteShape(e.shapeOf(rows, t != null));
            defer e.prefill_done = end;
            // the chunk's MTP rows are written (by mtpPrompt or the parts' SpExtra) once it ends
            // チャンクの MTP の行は、チャンクが終わった時点で書けている (mtpPrompt か part の SpExtra)
            defer if (e.mtp) |*m| m.track.chunk(a, end);
            if (e.n_sp > 0 and rows >= forward.sp_min_rows) {
                const tap_row: usize = @as(usize, e.n_taps) * e.cfg.hidden * 2;
                // the MTP layer's caches filled inside the chunk, each rank its own rows (mtp.Mtp.spInput)
                // MTP 層のキャッシュはチャンクの中で埋める。各ランクが自分の行を受け持つ (mtp.Mtp.spInput)
                const extra: ?forward.SpExtra = if (e.mtp) |*m| blk: {
                    m.spBegin(prompt, a, rows, e.embed.ptr, e.final_norm.ptr);
                    break :blk m.spExtra(e.state.kc[m.index].ptr, if (e.state.ic[m.index]) |b| b.ptr else null);
                } else null;
                const last = try forward.promptSp(e.sp_parts[0..e.n_sp], e.sp_want, e.ready, prompt[a..end], @intCast(a), e.embed.ptr, e.layers, e.state.kc, e.state.ic, t, if (e.bufs.taps) |x| x.ptr else null, tap_row, end == prompt.len, extra);
                if (end == prompt.len) {
                    try last.main.headLogits(e.final_norm.ptr, e.lm_head.ptr, last.R - 1, 1, e.vshare);
                    head = &last.main;
                }
                if (prop) |p| if (p.taps) |f| try f(p.ctx, e.bufs.taps.?.ptr, rows);
                continue;
            }
            if (rows > forward.max_rows) return error.ChunkTooWide;
            // stream-ordered: a legacy-stream upload / memset would not wait for the previous chunk
            // stream の順序に従う: legacy stream の upload / memset では前のチャンクを待たない
            try e.bufs.ids.uploadAsync(0, std.mem.sliceAsBytes(prompt[a..end]), e.stream.handle);
            try e.state.pos.fill32(@intCast(a), e.stream.handle);
            try c.embed(e.embed.ptr, rows);
            for (e.layers, 0..) |*L, i| try c.layerPrompt(L, rows, e.state.kc[i].ptr, if (e.state.ic[i]) |b| b.ptr else null, e.state.pos.ptr, t);
            if (end == prompt.len) try c.headLogits(e.final_norm.ptr, e.lm_head.ptr, rows - 1, 1, e.vshare);
            if (prop) |p| if (p.taps) |f| try f(p.ctx, e.bufs.taps.?.ptr, rows);
            if (e.mtp) |*m| try e.mtpPrompt(m, &c, e.bufs.x.ptr, prompt, a, rows);
            head = &c;
        }
        if (pprof_on and ci == pprof[1] + 1) {
            _ = e.d.api.cuCtxSynchronize();
            _ = e.d.api.cuProfilerStop();
        }
        // the first round's backlog row
        // 最初のラウンドのバックログの行
        if (e.mtp) |*m| try m.setCarry(e.stream, head.b.fnormed.ptr, e.cfg.hidden);
        if (plog) {
            _ = e.d.api.cuCtxSynchronize();
            std.debug.print("[prefill] chunk {d} to row {d}: {d} us\n", .{ ci - 1, prompt.len, @divTrunc(plog_t.durationTo(std.Io.Clock.awake.now(e.io)).toNanoseconds(), 1000) });
        }
        // the drafter's ring copied after the prompt (a later resume inside it gets back the window rows its decoding
        // overwrites)
        // prompt のあとに drafter の ring を写しておく (その中から後で再開するとき、decode が上書きしたウィンドウの行を
        // 取り戻す)
        if ((e.slots != null or e.pool != null) and prop != null) if (e.drafter) |dr| {
            e.dr_snap_end = try dr.snapshot();
        };
        if (e.samp) |s| { // the first token sits at position prompt.len / 最初のトークンは位置 prompt.len
            var one: [1]u32 = undefined;
            _ = try e.sampleRows(head.b.lpart.ptr, 1, prompt.len, s, &.{}, &one);
            return one[0];
        }
        return try head.pickArgmax(e.gpa, e.vshare);
    }

    /// Keyed sampling over rows 0 .. R of `lpart` (this rank's fp32 logits share, [R, vshare]): every rank's share
    /// all-gathered (so the ranks draw alike), row i's (top_k, else 256) + sample_margin best candidates by
    /// (value, id), drawn at position pos0 + i. Rows are drawn in order and stop after the first whose draw differs
    /// from its draft; returns how many were drawn into `out`.
    /// `lpart` の行 0 .. R (このランクの fp32 logits の分担、[R, vshare]) をキー付きでサンプルする: 全ランクの分担を
    /// all-gather し (全ランクが同じく抽選する)、行 i の上位 (top_k、0 なら 256) + sample_margin 個の候補を (値, id)
    /// の順で選び、位置 pos0 + i で抽選する。行は順に抽選し、draft と違った最初の行のあとで止める。`out` に抽選した
    /// 数を返す。
    fn sampleRows(e: *Engine, lpart: u64, R: u32, pos0: usize, s: Sampling, drafts: []const u32, out: []u32) !usize {
        const W = e.world;
        const vs: usize = e.vshare;
        const n = @as(usize, R) * vs;
        const lib = e.group.lib;
        try lib.check(lib.api.ncclAllGather(lpart, e.lgath.ptr, n, .f32, e.group.comm, e.stream.handle), "ncclAllGather");
        try e.lgath.downloadAsync(0, std.mem.sliceAsBytes(e.host_logits[0 .. W * n]), e.stream.handle);
        try e.stream.synchronize();
        const V = W * vs;
        const m = @min(V, (if (s.top_k != 0) @as(usize, s.top_k) else 256) + sample_margin);
        const Cand = struct { v: f32, id: u64 };
        const heap = try e.gpa.alloc(Cand, m); // the m best so far, worst at the root / ここまでの上位 m 個、根が最下位
        defer e.gpa.free(heap);
        const vals = try e.gpa.alloc(f64, m);
        defer e.gpa.free(vals);
        const ids = try e.gpa.alloc(u64, m);
        defer e.gpa.free(ids);
        const H = struct {
            // a ranks below b in (value desc, id asc)
            // (値の降順, id の昇順) で a が b より下
            fn worse(a: Cand, b: Cand) bool {
                return a.v < b.v or (a.v == b.v and a.id > b.id);
            }
            fn down(h: []Cand, at: usize) void {
                var i = at;
                while (true) {
                    const l = 2 * i + 1;
                    if (l >= h.len) return;
                    var w = l;
                    if (l + 1 < h.len and worse(h[l + 1], h[l])) w = l + 1;
                    if (!worse(h[w], h[i])) return;
                    std.mem.swap(Cand, &h[w], &h[i]);
                    i = w;
                }
            }
        };
        var drawn: usize = 0;
        for (0..R) |i| {
            var len: usize = 0;
            for (0..W) |r| {
                const row = e.host_logits[(r * R + i) * vs ..][0..vs];
                for (row, 0..) |v, j| {
                    const c: Cand = .{ .v = v, .id = r * vs + j };
                    if (len < m) {
                        heap[len] = c;
                        len += 1;
                        if (len == m) {
                            var k = m / 2;
                            while (k > 0) {
                                k -= 1;
                                H.down(heap, k);
                            }
                        }
                    } else if (H.worse(heap[0], c)) {
                        heap[0] = c;
                        H.down(heap, 0);
                    }
                }
            }
            for (heap[0..len], vals[0..len], ids[0..len]) |c, *v, *d| {
                v.* = c.v;
                d.* = c.id;
            }
            const tok = try sampler.choose(e.gpa, vals[0..len], ids[0..len], pos0 + i, s);
            out[i] = @intCast(tok);
            drawn += 1;
            if (i >= drafts.len or drafts[i] != out[i]) break;
        }
        return drawn;
    }

    /// One decode round without graphs: `tok` at position P.
    /// グラフを使わない decode ラウンド 1 回: 位置 P の `tok`。
    fn roundEager(e: *Engine, P: usize, tok: u32) !u32 {
        const c = e.fw(true);
        const one = [1]i64{tok};
        try e.vbufs.ids.uploadAsync(0, std.mem.sliceAsBytes(&one), e.stream.handle);
        try e.state.pos.fill32(@intCast(P), e.stream.handle);
        const t = bucket(P + 1, e.cfg.index_topk);
        try c.embed(e.embed.ptr, 1);
        for (e.layers, 0..) |*L, i| try c.layerDecode(L, 1, e.state.kc[i].ptr, if (e.state.ic[i]) |b| b.ptr else null, e.state.pos.ptr, t);
        try c.headLogits(e.final_norm.ptr, e.lm_head.ptr, 0, 1, e.vshare);
        return c.pickArgmax(e.gpa, e.vshare);
    }

    /// One decode round as the bucket's graph (captured on first use): the graph embeds vbufs.ids[0], which the
    /// previous round's pick (or generate, for the first round) left there, and picks into it again.
    /// bucket のグラフで decode ラウンドを 1 回実行する (初回に capture): グラフは、前のラウンドの選択 (最初の
    /// ラウンドでは generate) が vbufs.ids[0] に置いたトークンを埋め込み、選択を再びそこへ書く。
    fn roundGraph(e: *Engine, P: usize) !u32 {
        try e.state.pos.fill32(@intCast(P), e.stream.handle);
        const t = bucket(P + 1, e.cfg.index_topk);
        const slot = for (0..e.n_graphs) |i| {
            if (e.keys[i] == t and e.key_rows[i] == 1) break i;
        } else try e.capture(1, t);
        try e.execs[slot].launchOn(e.stream);
        var got: [1]i32 = undefined;
        try e.vbufs.tok_out.downloadAsync(0, std.mem.sliceAsBytes(&got), e.stream.handle);
        try e.stream.synchronize();
        try e.roceCheck();
        return @intCast(got[0]);
    }

    /// Every decode graph up front - each index-key bucket up to the capacity (and none, below index_topk), windows of
    /// 1 .. vrows rows with a drafter (else 1) - and the drafter's graphs, so no request waits for a capture. Captures
    /// run nothing. Returns the graphs captured.
    /// decode グラフを前もってすべて作る: 容量までの index キーの bucket ごと (index_topk 未満の「なし」も)、drafter
    /// があれば 1 .. vrows 行のウィンドウ (なければ 1 行)、さらに drafter のグラフ。要求が capture を待たないように
    /// する。capture は何も実行しない。capture したグラフの数を返す。
    pub fn prewarm(e: *Engine) !usize {
        if (!e.graphs) return 0;
        const before = e.n_graphs;
        const top = bucket(e.capacity + 1, e.cfg.index_topk) orelse 2 * @as(u32, @intCast(e.cfg.index_topk));
        const rows: u32 = if (e.drafter != null) vrows else 1;
        var t: ?u32 = null;
        while (true) {
            var R: u32 = 1;
            while (R <= rows) : (R += 1) {
                const have = for (0..e.n_graphs) |i| {
                    if (e.keys[i] == t and e.key_rows[i] == R) break true;
                } else false;
                if (!have) _ = try e.capture(R, t);
            }
            const next: u32 = if (t) |x| x * 2 else 2 * @as(u32, @intCast(e.cfg.index_topk));
            if (next > top) break;
            t = next;
        }
        if (e.drafter) |dr| try dr.prewarm();
        try e.prewarmPrompts();
        if (e.rank == 0) std.log.info("glm53: {d} prompt chunk shape classes run at start-up (admit lets only these through)", .{e.n_shapes});
        return e.n_graphs - before;
    }

    /// The prompt shapes a request can bring, run once at start-up, so a shape the kernels cannot run fails here,
    /// before any request is shared, instead of ending every rank mid-request: up to 16 rows each count its own
    /// (row block), the router's BM classes (16 / 32 / 64 / 128), 1 and multiples of 16, the 128-row linear bound
    /// and the SP parts (258 / 516), each with and without a selection (past index_topk, as in a resumed prompt).
    /// 要求が持ち込みうる prompt の形を起動時に一度ずつ実行する。カーネルが実行できない形は、要求を共有して全ランクを
    /// 途中で止める前に、ここで失敗させる: 16 行までは行数ごと (行ブロック)、router の BM 分類 (16 / 32 / 64 / 128)、
    /// 1 と 16 の倍数、128 行の linear の境界、SP の分割 (258 / 516)。それぞれ選択あり (index_topk 超え。再開した
    /// prompt のチャンクはこれ) となしの両方。
    fn prewarmPrompts(e: *Engine) !void {
        const sizes = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 31, 32, 33, 42, 47, 48, 63, 64, 65, 79, 80, 100, 127, 128, 129, 144, 200, 257, 258, 272, 300, 515, 516, 528, 600 };
        const past: usize = 2100; // > index_topk: the rows after it select / index_topk より大: 以降の行は選択する
        const ids = try e.gpa.alloc(i64, past + 600);
        defer e.gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(1000 + i % 100000);
        const start = e.start;
        defer e.start = start;
        // the chunk shapes run here are the ones `admit` lets through
        // ここで実行したチャンクの形だけを `admit` が通す
        e.recording_shapes = true;
        defer e.recording_shapes = false;
        for ([_]bool{ false, true }) |select| {
            e.start = 0;
            if (select) _ = try e.prefill(ids[0..past], null);
            for (sizes) |n| {
                const base: usize = if (select) past else 0;
                if (base + n > e.promptLimit()) break;
                // a resumed prompt computes at least slots.tail (17) rows
                // 再開した prompt は少なくとも slots.tail (17) 行を計算する
                if (select and n < 17) continue;
                e.start = base;
                _ = e.prefill(ids[0 .. base + n], null) catch |err| {
                    std.log.err("glm53: rank {d}: a {d}-row prompt chunk{s} cannot run ({s}) while the start-up warms its shapes", .{ e.rank, n, if (select) " after 2100 rows (a selection)" else "", @errorName(err) });
                    return err;
                };
            }
        }
    }

    fn capture(e: *Engine, R: u32, t: ?u32) !usize {
        if (e.n_graphs == max_graphs) return error.TooManyGraphs;
        var c = e.fw(true);
        // tests: this window's picks into the log, a copy an index layer
        // テスト用: このウィンドウの選択をログへ。index 層ごとに 1 つ写す
        if (e.sel_tok) |b| {
            e.sel_n = 0;
            c.sel_tok = b.ptr;
            c.sel_cnt = e.sel_cnt.?.ptr;
            c.sel_n = &e.sel_n;
        }
        try cuda.graph.beginCapture(e.stream, .thread_local);
        c.windowStep(R, e.embed.ptr, e.layers, e.state.kc, e.state.ic, e.state.pos.ptr, t, e.final_norm.ptr, e.lm_head.ptr, e.vshare) catch |err| {
            if (cuda.graph.endCapture(e.stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        if (e.sel_tok != null and t != null and e.sel_n != e.sel_layers) {
            if (cuda.graph.endCapture(e.stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return error.SelLogLayers;
        }
        var g = try cuda.graph.endCapture(e.stream);
        defer g.deinit();
        e.execs[e.n_graphs] = try g.instantiate();
        e.keys[e.n_graphs] = t;
        e.key_rows[e.n_graphs] = R;
        e.n_graphs += 1;
        return e.n_graphs - 1;
    }
};

fn upload(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, name: []const u8) !cuda.DeviceBuffer {
    var s = try r.get(name);
    defer s.deinit(gpa);
    return cuda.DeviceBuffer.fromHost(d, s.bytes);
}

/// Stops the load when MemAvailable falls under 8 GiB, before the unified memory runs out (an OOM livelock on GB10
/// needs a cold power cycle).
/// MemAvailable が 8 GiB を切ったら読み込みを止める。unified memory が尽きる前に止めるため (GB10 で OOM による
/// livelock が起きると電源の入れ直しが要る)。
/// TF_GLM53_MEMTRACE=1: the device memory a startup step took (cuMemGetInfo's free before and after; on GB10 it
/// leaves out the page cache), one line a step a rank. `last` holds the previous reading (0: the first, not printed).
/// TF_GLM53_MEMTRACE=1: 起動の各段階が取ったデバイスメモリ (cuMemGetInfo の free の前後。GB10 ではページキャッシュを
/// 含まない)。段階ごと・ランクごとに 1 行。`last` は前回の値 (0 は最初で、表示しない)。
pub fn memMark(ctx: *const cuda.Context, rank: u32, label: []const u8, last: *usize) void {
    const on = std.c.getenv("TF_GLM53_MEMTRACE") orelse return;
    if (!std.mem.eql(u8, std.mem.span(on), "1")) return;
    ctx.synchronize() catch {};
    const f = (ctx.memInfo() catch return).free;
    // the DeviceBuffers this process holds: what the step allocated, unlike `free` blind to the page cache moving
    // このプロセスが持つ DeviceBuffer: その段階が確保した量。ページキャッシュの増減に左右される `free` と違う
    const held = cuda.usage(false).device;
    const g: f64 = 1 << 30;
    if (last.* != 0) std.log.info("glm53: memtrace rank {d}: {s}: allocated {d:.3} GiB (held {d:.2} GiB), free {d:.3} -> {d:.2} GiB", .{ rank, label, (@as(f64, @floatFromInt(held)) - @as(f64, @floatFromInt(mt_held))) / g, @as(f64, @floatFromInt(held)) / g, (@as(f64, @floatFromInt(last.*)) - @as(f64, @floatFromInt(f))) / g, @as(f64, @floatFromInt(f)) / g });
    last.* = f;
    mt_held = held;
}
var mt_held: u64 = 0;

fn memCheck(io: std.Io, layer: usize) !void {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return;
    const at = std.mem.indexOf(u8, text, "MemAvailable:") orelse return;
    var it = std.mem.tokenizeAny(u8, text[at + 13 ..], " \n");
    const kib = std.fmt.parseInt(u64, it.next() orelse return, 10) catch return;
    if (kib < 8 << 20) {
        std.log.err("glm53: MemAvailable {d} MiB after layer {d}: stopping the load", .{ kib >> 10, layer });
        return error.LowMemory;
    }
}

test "bucket rounds up to a power of two above the chunk" {
    try std.testing.expectEqual(@as(?u32, null), bucket(2048, 2048));
    try std.testing.expectEqual(@as(?u32, 4096), bucket(2049, 2048));
    try std.testing.expectEqual(@as(?u32, 4096), bucket(4096, 2048));
    try std.testing.expectEqual(@as(?u32, 8192), bucket(4097, 2048));
}
