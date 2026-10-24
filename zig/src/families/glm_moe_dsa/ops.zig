//! The engine's element-wise, attention, index and routing kernels (kernels/cuda/glm53/native_ops.cu), call by call.
//! Shapes outside what a kernel covers fail with error.UnsupportedShape instead of running anything else.
//! エンジンの要素ごと・attention・index・routing のカーネル (kernels/cuda/glm53/native_ops.cu) を呼び出し単位で包む。
//! カーネルが扱わない形は、別の処理を走らせずに error.UnsupportedShape で失敗する。
const std = @import("std");
const cuda = @import("cuda");

const native = @import("native_ops.zig");

const scale_256: u32 = 0x3d800000; // (nope + rope) ** -0.5 = 256 ** -0.5
// 128 ** -0.5 (the Hadamard scale, the indexer's query scale)
// 128 ** -0.5 (Hadamard のスケール、indexer の query のスケール)
const scale_128: u32 = 0x3db504f3;
// 32 ** -0.5 (the indexer's weight scale)
// 32 ** -0.5 (indexer の重みのスケール)
const scale_32: u32 = 0x3e3504f3;

fn f(b: u32) f32 {
    return @bitCast(b);
}

fn unsupported(name: []const u8) error{UnsupportedShape} {
    std.log.err("glm53: {s}: no kernel for this shape", .{name});
    return error.UnsupportedShape;
}

pub const Ops = struct {
    k: *const native.Kernels,
    stream: cuda.Stream,
    /// FP8 caches: latent rows of 512 e4m3 codes, a scale (byte 512) and the bf16 RoPE (byte 528), 656 bytes; index-key
    /// rows of 128 codes and a scale, 144 bytes.
    /// FP8 キャッシュ: latent 行は e4m3 コード 512 個、スケール (512 バイト目)、bf16 の RoPE (528 バイト目) で
    /// 656 バイト。index key 行はコード 128 個とスケールで 144 バイト。
    kv8: bool = false,
    /// TF_GLM53_KV=fp4 (with kv8): latent rows of 432 bytes, e2m1 codes with a power-of-two exponent each 16; the index
    /// keys stay FP8.
    /// TF_GLM53_KV=fp4 (kv8 と一緒に): latent 行は 432 バイト、e2m1 コードと 16 個ごとの 2 のべき乗の指数。index の
    /// key は FP8 のまま。
    kv4: bool = false,
    /// Positions p live on group member p % dcp at local row p / dcp.
    /// 位置 p はグループのメンバー p % dcp のローカル行 p / dcp に置く。
    dcp: u32 = 1,
    drank: u32 = 0,
    /// TF_GLM53_PREFIX=pool: the device PoolMap (block table, write range, error flag) every KV kernel maps rows
    /// through; 0: the contiguous cache.
    /// TF_GLM53_PREFIX=pool: KV カーネルが行を引くデバイスの PoolMap (ブロック表、書き込み範囲、異常フラグ)。0 なら
    /// 連続のキャッシュ。
    pmap: u64 = 0,

    /// out = RMSNorm(x) * w over rows of `d`.
    /// `d` 個ずつの行で out = RMSNorm(x) * w。
    pub fn rmsnorm(o: Ops, x: u64, x_stride: u32, w: u64, out: u64, o_stride: u32, rows: u32, d: u32, eps: f32) !void {
        return o.k.rmsnorm(o.stream, x, x_stride, w, out, o_stride, rows, d, eps);
    }

    /// Every gathered query head (G members x H) over this rank's keys of each row -> chunk partials PO / PM / PL.
    /// 集めた全 query head (G メンバー x H) を各行のこのランクの key に当て、チャンクごとの部分結果 PO / PM / PL へ。
    pub fn attnDcpAll(o: Ops, qall: u64, lc: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, r: u32, h: u32, g: u32, nch: u32, chk: u32, kt: u32) !void {
        if (!o.kv8 or !native.Kernels.attnOk(g * h, kt)) return unsupported("attn_dcp");
        return o.k.attn(o.stream, qall, 0, lc, tok, cnt, pos, po, pm, pl, r, h, g, nch, chk, kt, o.dcp, o.drank, false, f(scale_256), o.kv4, o.pmap);
    }

    /// Each (row, head) chunk partials merged in chunk order -> osend bf16 (normalized) and lsend fp32 (log-sum-exp).
    /// (row, head) ごとのチャンク部分結果をチャンク順に merge -> osend bf16 (正規化済み) と lsend fp32 (log-sum-exp)。
    pub fn mergeLse(o: Ops, po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, r: u32, ht: u32, h: u32, nch: u32) !void {
        return o.k.mergeLse(o.stream, po, pm, pl, osend, lsend, r, ht, h, nch);
    }

    /// Every member's partial of this rank's heads merged in rank order by their log-sum-exps -> ol [R, H, 512] bf16.
    /// このランクの head についての各メンバーの部分結果を、log-sum-exp を使いランク順に merge -> ol [R, H, 512] bf16。
    pub fn dcpCombine(o: Ops, orecv: u64, lrecv: u64, out: u64, r: u32, ss: u32, ssl: u32, h: u32, world: u32) !void {
        return o.k.dcpCombine(o.stream, orecv, lrecv, out, r, ss, ssl, h, world);
    }

    /// RMSNorm with the 64-group sums of the output (xs fp32 [rows, d / 64]) for the 4-bit matmul.
    /// RMSNorm に加え、4 ビット matmul 用に出力の 64 個ごとのグループ和 (xs fp32 [rows, d / 64]) を出す。
    pub fn rmsnormSums(o: Ops, x: u64, x_stride: u32, w: u64, out: u64, o_stride: u32, xs: u64, rows: u32, d: u32, eps: f32) !void {
        if (d % 2048 != 0) return unsupported("rmsnorm_sums");
        return o.k.rmsnormSums(o.stream, x, x_stride, w, out, o_stride, xs, rows, d, eps);
    }

    /// xs fp32 [m, k / 64] = the 64-input sums of bf16 rows.
    /// xs fp32 [m, k / 64] = bf16 の行の 64 入力ごとの和。
    pub fn groupSums(o: Ops, x: u64, x_stride: u32, xs: u64, m: u32, k: u32) !void {
        return o.k.groupSums(o.stream, x, x_stride, xs, m, k);
    }

    /// The drafter's two-tap grouped dynamic convolution over the block rows (+ residual), [rows, d] bf16.
    /// drafter の 2 タップのグループ化した動的畳み込みをブロックの行に適用する (+ residual)。[rows, d] bf16。
    pub fn dconv(o: Ops, x: u64, dyn: u64, base: u64, res: ?u64, out: u64, rows: u32, d: u32, gs: u32, branch: u32) !void {
        return o.k.dconv(o.stream, x, dyn, base, res, out, rows, d, gs, branch);
    }

    /// The drafter's q / k RMSNorm + RoPE -> q [heads, rows, hd], k / v [kvh, rows, hd] (heads 0: k and v only).
    /// drafter の q / k の RMSNorm + RoPE -> q [heads, rows, hd]、k / v [kvh, rows, hd] (heads 0 なら k と v のみ)。
    pub fn prep(o: Ops, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: u32, stride: u32, eps: f32, heads: u32, kvh: u32, half: u32) !void {
        if (half != 64) return unsupported("prep");
        return o.k.prep(o.stream, qkv, qn, kn, cos, sin, qo, ko, vo, rows, stride, eps, heads, kvh);
    }

    /// The drafter's attention: each KV head's queries over the sliding window of the ring and the block.
    /// drafter の attention: KV head ごとに、query をリングの sliding window とブロックの key に当てる。
    pub fn dattnRing(o: Ops, q: u64, k: u64, v: u64, out: u64, pos: u64, window: u32, scale: f32, n: u32, g: u32, nh: u32, hd: u32, kvh: u32, ring: u32, causal: bool) !void {
        if (!native.Kernels.dattnOk(n, g, hd, ring, causal)) return unsupported("dattn_ring");
        return o.k.dattnRing(o.stream, q, k, v, out, pos, window, scale, nh, kvh);
    }

    /// Rows of a bf16 table by token id (int32) into out [rows, d].
    /// token id (int32) で bf16 の表から行を引き、out [rows, d] に書く。
    pub fn embedB16(o: Ops, ids: u64, w: u64, out: u64, rows: u32, d: u32) !void {
        return o.k.embedB16(o.stream, ids, w, out, rows, d);
    }

    /// [gate | up] bf16 -> act bf16 [rows, w] and its 64-group sums xs fp32.
    /// [gate | up] bf16 -> act bf16 [rows, w] と、その 64 個ごとのグループ和 xs fp32。
    pub fn swiglu(o: Ops, gu: u64, out: u64, xs: u64, rows: u32, w: u32, limit: f32) !void {
        if (w % 512 != 0) return unsupported("swiglu");
        return o.k.swiglu(o.stream, gu, out, xs, rows, w, limit);
    }

    /// out fp32 [m, ne] = x bf16 [m, d] @ w^T, K split in 8 slices summed in order through `part`.
    /// out fp32 [m, ne] = x bf16 [m, d] @ w^T。K を 8 つに切り、`part` を経由して順に足す。
    pub fn router(o: Ops, x: u64, x_stride: u32, w: u64, part: u64, out: u64, m: u32, d: u32, ne: u32) !void {
        const ks: u32 = 8;
        if (d % (ks * 64) != 0) return unsupported("router");
        try o.k.routerPart(o.stream, x, x_stride, w, part, m, d, ne, ks);
        return o.k.routerSum(o.stream, part, out, m * ne, ks);
    }

    /// The window's latent rows into the cache at pos .. pos + R - 1.
    /// window の latent 行をキャッシュの pos .. pos + R - 1 に書く。
    pub fn kvWrite(o: Ops, kva: u64, kva_stride: u32, nw: u64, lc: u64, pos: u64, inv: u64, eps: f32, r: u32) !void {
        return o.k.kvWrite(o.stream, kva, kva_stride, nw, lc, pos, inv, eps, r, o.dcp, o.drank, if (o.kv4) 2 else @intFromBool(o.kv8), o.pmap);
    }

    /// LayerNorm + RoPE of the indexer's key rows into its cache.
    /// indexer の key 行に LayerNorm + RoPE を掛け、indexer のキャッシュに書く。
    pub fn ikWrite(o: Ops, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, r: u32) !void {
        return o.k.ikWrite(o.stream, ik, nw, nb, inv, ic, pos, r, o.dcp, o.drank, o.kv8, o.pmap);
    }

    /// The q rows' RoPE part into qrot [R, H, 64].
    /// q 行の RoPE 部分を qrot [R, H, 64] に書く。
    pub fn qrope(o: Ops, q: u64, inv: u64, qr: u64, pos: u64, r: u32, h: u32) !void {
        return o.k.qrope(o.stream, q, inv, qr, pos, r, h, 64);
    }

    /// Sparse MLA over the rows' key lists; `direct` (one chunk): straight into OL (bf16).
    /// 各行の key リスト上の sparse MLA。`direct` (1 チャンク) なら OL (bf16) へ直接書く。
    pub fn attnChunks(o: Ops, qa: u64, qr: u64, lc: u64, tok: u64, pos: u64, po: u64, pm: u64, pl: u64, r: u32, h: u32, nch: u32, chk: u32, kt: u32, direct: bool) !void {
        if (!o.kv8 or !native.Kernels.attnOk(h, kt)) return unsupported("attn_chunks");
        return o.k.attn(o.stream, qa, qr, lc, tok, 0, pos, po, pm, pl, r, h, 0, nch, chk, kt, 1, 0, direct, f(scale_256), o.kv4, o.pmap);
    }

    /// xout = x + the gathered partials summed in rank order ([world, rows, d] fp32).
    /// xout = x + 集めた部分結果をランク順に足したもの ([world, rows, d] fp32)。
    pub fn residualAdd(o: Ops, x: u64, xout: u64, g: u64, rows: u32, d: u32, world: u32) !void {
        return o.k.residualAdd(o.stream, x, xout, g, rows, d, world, false);
    }

    /// `residualAdd` of one bf16 partial (a ring reduce-scatter's share).
    /// bf16 の部分結果 1 つ (リング reduce-scatter の取り分) の `residualAdd`。
    pub fn residualAddBf16(o: Ops, x: u64, xout: u64, g: u64, rows: u32, d: u32) !void {
        return o.k.residualAdd(o.stream, x, xout, g, rows, d, 1, true);
    }

    /// act = silu(g) * u over rows of `width`.
    /// `width` 個ずつの行で act = silu(g) * u。
    pub fn swiglu2(o: Ops, g: u64, u: u64, out: u64, r: u32, width: u32) !void {
        return o.k.swiglu2(o.stream, g, u, out, r, width);
    }

    /// Sigmoid + bias top-8 of the router logits -> pick int32 / wts fp32 (normalized, scaled).
    /// router logits の sigmoid + bias で top-8 を選ぶ -> pick int32 / wts fp32 (正規化してスケール済み)。
    pub fn topk(o: Ops, logits: u64, bias: u64, pick: u64, wts: u64, n: u32, scale: f32, log: u64) !void {
        return o.k.topk(o.stream, logits, bias, pick, wts, n, scale, log);
    }

    /// The picks of `rows` rows (int32 [rows, 8]) counted into `hist` (u32 [experts]).
    /// `rows` 行の選択 (int32 [rows, 8]) を `hist` (u32 [experts]) に数える。
    pub fn pickHist(o: Ops, pick: u64, rows: u32, hist: u64, experts: u32) !void {
        return o.k.pickHist(o.stream, pick, rows, hist, experts);
    }

    /// RoPE of the indexer's query rows in place.
    /// indexer の query 行にその場で RoPE を掛ける。
    pub fn iqRope(o: Ops, q: u64, inv: u64, pos: u64, r: u32) !void {
        return o.k.iqRope(o.stream, q, inv, pos, r);
    }

    /// Rows r0 .. r0 + n - 1 against T keys; `pack`: int64 keys (score | position), else int32 order words.
    /// 行 r0 .. r0 + n - 1 対 T 個の key。`pack` なら int64 の key (score | position)、でなければ int32 の順序語。
    pub fn indexScores(o: Ops, q: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32, pack: bool) !void {
        if (!o.kv8) return unsupported("index_scores");
        return o.k.indexScores(o.stream, q, w, ic, pos, out, n, t, r0, o.dcp, o.drank, @intFromBool(pack), f(scale_32), f(scale_128), o.pmap);
    }

    /// Whether `indexScoresWords` can run (FP8 key rows).
    /// `indexScoresWords` を実行できるか (FP8 の key 行のとき)。
    pub fn wordScores(o: Ops) bool {
        return o.kv8;
    }

    /// `indexScores` as int32 order words (a later key 0), for the radix select.
    /// radix select 用に int32 の順序語で出す `indexScores` (後ろの key は 0)。
    pub fn indexScoresWords(o: Ops, q: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32) !void {
        return o.k.indexScores(o.stream, q, w, ic, pos, out, n, t, r0, o.dcp, o.drank, 0, f(scale_32), f(scale_128), o.pmap);
    }

    /// `indexScoresWords` with the words' top 16 bits after them in `out` (from prefix16Offset(n, t) bytes), for
    /// Glue.topkWordsAny16.
    /// 順序語の上位 16 ビットを `out` の後ろ (prefix16Offset(n, t) バイトから) にも書く `indexScoresWords`
    /// (Glue.topkWordsAny16 用)。
    pub fn indexScoresWords16(o: Ops, q: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32) !void {
        return o.k.indexScores(o.stream, q, w, ic, pos, out, n, t, r0, o.dcp, o.drank, 2, f(scale_32), f(scale_128), o.pmap);
    }

    /// Where indexScoresWords16 puts the top 16 bits: past the n t order words, on a 16-byte boundary.
    /// indexScoresWords16 が上位 16 ビットを置く位置: n t 個の順序語の先、16 バイト境界。
    pub fn prefix16Offset(n: u32, t: u32) u64 {
        return (@as(u64, n) * t * 4 + 15) / 16 * 16;
    }

    /// The index queries' e4m3 codes and scales (TF_GLM53_INDEX_FP8).
    /// index の query の e4m3 コードとスケール (TF_GLM53_INDEX_FP8)。
    pub fn iqFp8(o: Ops, q: u64, q8: u64, qs: u64, rows: u32) !void {
        return o.k.iqFp8(o.stream, q, q8, qs, rows);
    }

    /// `indexScores` on the FP8 tensor-core product (TF_GLM53_INDEX_FP8; other scores than the bf16 product's).
    /// FP8 の tensor core の積で求める `indexScores` (TF_GLM53_INDEX_FP8。bf16 の積とは点数が違う)。
    pub fn indexScores8(o: Ops, q8: u64, qs: u64, w: u64, ic: u64, pos: u64, out: u64, n: u32, t: u32, r0: u32, pack: bool) !void {
        return o.k.indexScores8(o.stream, q8, qs, w, ic, pos, out, n, t, r0, o.dcp, o.drank, pack, f(scale_32), f(scale_128), o.pmap);
    }

    /// Each row's 2048 best columns of int32 order words, ascending (radix select).
    /// int32 の順序語の各行で上位 2048 列を昇順で (radix select)。
    pub fn selectKeys(o: Ops, sc: u64, out: u64, n: u32, t: u32) !void {
        return o.k.selectKeys(o.stream, sc, out, n, t, 2048);
    }

    /// q_nope x W_UK into qlat and the q RoPE into qrot.
    /// q_nope x W_UK を qlat に、q の RoPE を qrot に書く。
    pub fn absorb(o: Ops, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, r: u32, h: u32) !void {
        return o.k.absorb(o.stream, q, wk, inv, qa, qr, pos, r, h, 512, 64);
    }

    /// `absorb` straight into the packed queries [R, H, 576] (latent 512, then the RoPE 64).
    /// pack した query [R, H, 576] (latent 512、続いて RoPE 64) へ直接書く `absorb`。
    pub fn absorbPacked(o: Ops, q: u64, wk: u64, inv: u64, qpack: u64, pos: u64, r: u32, h: u32) !void {
        return o.k.absorb(o.stream, q, wk, inv, qpack, qpack + 512 * 2, pos, r, h, 576, 576);
    }

    /// `qrope` into the packed queries' RoPE part.
    /// pack した query の RoPE 部分へ書く `qrope`。
    pub fn qropePacked(o: Ops, q: u64, inv: u64, qpack: u64, pos: u64, r: u32, h: u32) !void {
        return o.k.qrope(o.stream, q, inv, qpack + 512 * 2, pos, r, h, 576);
    }

    /// ol x W_UV into o [R, H * v].
    /// ol x W_UV を o [R, H * v] に書く。
    pub fn expand(o: Ops, ol: u64, wv: u64, out: u64, r: u32, h: u32) !void {
        return o.k.expand(o.stream, ol, wv, out, r, h);
    }

    /// The dense chunks' partials of each row and head into ol (bf16).
    /// 各行・各 head の dense チャンクの部分結果をまとめて ol (bf16) に書く。
    pub fn merge(o: Ops, po: u64, pm: u64, pl: u64, out: u64, cnt: u64, r: u32, h: u32, nch: u32) !void {
        _ = cnt;
        return o.k.merge(o.stream, po, pm, pl, out, r, h, nch);
    }

    /// Whether `attnDcpDirect` covers a prompt chunk of g members x h heads (FP8 or FP4 cache, 16 or 32 heads).
    /// g メンバー x h head の prompt チャンクを `attnDcpDirect` が扱えるか (FP8 か FP4 のキャッシュ、16 か 32 head)。
    pub fn canAttnDirect(o: Ops, g: u32, h: u32) bool {
        return o.kv8 and (g * h == 16 or g * h == 32);
    }

    /// `attnDcpAll` of one chunk (2,048 keys, tiles of 64) and `mergeLse` in one launch, without the fp32 partials.
    /// 1 チャンク (key 2,048 個、64 のタイル) の `attnDcpAll` と `mergeLse` を 1 回の起動で。fp32 の部分結果なし。
    pub fn attnDcpDirect(o: Ops, qall: u64, lc: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, r: u32, h: u32, g: u32) !void {
        _ = .{ po, pm, pl };
        return o.k.attnDl(o.stream, qall, lc, tok, cnt, pos, osend, lsend, r, h, g, o.dcp, o.drank, f(scale_256), o.kv4, o.pmap);
    }

    /// out (fp32 or bf16) [m, n] = (xh fp16 @ W fp16) @ H * svh.
    /// out (fp32 か bf16) [m, n] = (xh fp16 @ W fp16) @ H * svh。
    pub fn gemm(o: Ops, xh: u64, w: u64, had: u64, svh: u64, out: u64, out_bf16: bool, m: u32, o_stride: u32, k: u32, n: u32) !void {
        return o.k.gemm(o.stream, xh, w, had, svh, out, out_bf16, m, o_stride, k, n, f(scale_128));
    }
};
