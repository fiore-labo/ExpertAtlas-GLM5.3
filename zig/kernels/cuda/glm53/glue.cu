// Helper kernels of the GLM-5.3 engine: row gathers, strided copies, rank-order fp32 sums, bf16 rounding, the indexer's
// top-k selection, argmax, the drafter's helpers and the slot cache's copies and hashes. Copies move bits; every sum is
// one IEEE add an element (built with --fmad=false).
// GLM-5.3 エンジンの補助カーネル: 行の gather、strided コピー、ランク順の fp32 和、bf16 への丸め、indexer の top-k
// 選択、argmax、drafter の補助、スロットキャッシュのコピーとハッシュ。コピーはビットをそのまま運び、和は要素ごとに
// IEEE の加算 1 回 (--fmad=false でビルド)。
#include <cstdint>

extern "C" {

// dst[r, :] = src[ids[r], :] for rows of `width` 2-byte elements (bf16 embedding rows); ids are int64.
// `width` 個の 2 バイト要素の行 (bf16 の埋め込み行) で dst[r, :] = src[ids[r], :]。ids は int64。
__global__ void tf_glm53_gather_rows16(const uint16_t* __restrict__ src, const int64_t* __restrict__ ids,
                                       uint16_t* __restrict__ dst, int width) {
    const int r = blockIdx.y;
    const int64_t row = ids[r];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += gridDim.x * blockDim.x)
        dst[(int64_t)r * width + c] = src[row * width + c];
}

// dst[i, j, k] = src[i * si + j * sj + k] for a [n0, n1, n2] view (element strides si, sj; the last axis contiguous),
// dst contiguous; 2-byte (bf16) and 4-byte (fp32) elements.
// [n0, n1, n2] のビュー (要素単位のストライド si・sj、最後の軸は連続) から連続な dst へ
// dst[i, j, k] = src[i * si + j * sj + k]。要素は 2 バイト (bf16) と 4 バイト (fp32)。
__global__ void tf_glm53_copy3_16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int n1, int n2,
                                  long long si, long long sj) {
    const int i = blockIdx.z, j = blockIdx.y;   // grid (x, n1, n0) / グリッド (x, n1, n0)
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n2; k += gridDim.x * blockDim.x)
        dst[((long long)i * n1 + j) * n2 + k] = src[i * si + j * sj + k];
}

__global__ void tf_glm53_copy3_32(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int n1, int n2,
                                  long long si, long long sj) {
    const int i = blockIdx.z, j = blockIdx.y;   // grid (x, n1, n0) / グリッド (x, n1, n0)
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n2; k += gridDim.x * blockDim.x)
        dst[((long long)i * n1 + j) * n2 + k] = src[i * si + j * sj + k];
}

// The same copies with a strided destination: dst[i * di + j * dj + k] from a contiguous source.
// 同じコピーで書き込み先が strided なもの: 連続なコピー元から dst[i * di + j * dj + k] へ。
__global__ void tf_glm53_scatter3_16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int n1, int n2,
                                     long long di, long long dj) {
    const int i = blockIdx.z, j = blockIdx.y;   // grid (x, n1, n0) / グリッド (x, n1, n0)
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n2; k += gridDim.x * blockDim.x)
        dst[i * di + j * dj + k] = src[((long long)i * n1 + j) * n2 + k];
}

__global__ void tf_glm53_scatter3_32(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int n1, int n2,
                                     long long di, long long dj) {
    const int i = blockIdx.z, j = blockIdx.y;   // grid (x, n1, n0) / グリッド (x, n1, n0)
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n2; k += gridDim.x * blockDim.x)
        dst[i * di + j * dj + k] = src[((long long)i * n1 + j) * n2 + k];
}

// acc[i] += x[i] (fp32): one rank's share, added in rank order for the exact reduce-scatter.
// acc[i] += x[i] (fp32): 1 ランクぶんを足す。厳密な reduce-scatter のため、ランク順に足していく。
__global__ void tf_glm53_add_f32(float* __restrict__ acc, const float* __restrict__ x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        acc[i] = acc[i] + x[i];
}


// The indexer's top-k on one rank off the radix path (a block of fewer than 32 rows, or fewer keys than
// index_topk): the k largest packed int64 scores of a row (the packed keys are unique, so the set is exact) as int32
// 0x7FFFFFFF - (key & 0xFFFFFFFF), ascending. One block a row: a descending bitonic sort of the row in shared memory
// (padded to `npow2` with INT64_MIN), then a second bitonic pass sorts the first k keys ascending.
// radix の経路を使わないとき (32 行未満のブロック、またはキーが index_topk 未満) の 1 ランクでの indexer の top-k:
// 行のパック済み int64 スコアの上位 k 個 (パック済みキーは一意なので集合は厳密) を int32 の
// 0x7FFFFFFF - (key & 0xFFFFFFFF) として昇順に出す。1 行 1 ブロック: 共有メモリで行を降順に bitonic ソートし
// (`npow2` まで INT64_MIN で埋める)、2 回目の bitonic で先頭 k 個のキーを昇順に並べる。
__global__ void tf_glm53_topk_keys(const long long* __restrict__ sc, int* __restrict__ out, int T, int k, int npow2) {
    extern __shared__ long long buf[];
    const long long* row = sc + (long long)blockIdx.x * T;
    for (int i = threadIdx.x; i < npow2; i += blockDim.x) buf[i] = i < T ? row[i] : (long long)0x8000000000000000LL;
    __syncthreads();
    for (int size = 2; size <= npow2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < npow2; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const long long a = buf[i], b = buf[j];
                    if (desc ? (a < b) : (a > b)) { buf[i] = b; buf[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    // the k largest now lead; their int32 keys, then an ascending sort of those k (k a power of two)
    // 上位 k 個が先頭に来た。その int32 キーを作り、k 個を昇順に並べる (k は 2 のべき)
    int* keys = reinterpret_cast<int*>(buf + npow2);
    for (int i = threadIdx.x; i < k; i += blockDim.x) keys[i] = (int)(0x7FFFFFFFLL - (buf[i] & 0xFFFFFFFFLL));
    __syncthreads();
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < k; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool asc = (i & size) == 0;
                    const int a = keys[i], b = keys[j];
                    if (asc ? (a > b) : (a < b)) { keys[i] = b; keys[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[(long long)blockIdx.x * k + i] = keys[i];
}

// Whether (a, ia) comes before (b, ib) among a row's values: a NaN is the maximum, equal values by the lower index.
// 行の値の順で (a, ia) が (b, ib) より前か: NaN は最大、等しい値は index の小さいほうが前。
__device__ __forceinline__ bool tf_glm53_before(float a, int ia, float b, int ib) {
    const bool na = a != a, nb = b != b, lower = ia < ib;
    // a NaN: before unless b is a NaN at a lower index; else b not a NaN and a above it, or equal at the lower index
    // a が NaN なら、b が index の小さい NaN でない限り前。そうでなければ、b が NaN でなく a が大きいか、等しくて
    // index が小さいとき前
    return (na & (!nb | lower)) | (!na & !nb & ((a > b) | ((a == b) & lower)));
}

// The block's first (value, index) in that order over every thread's own (a thread without one: -inf, INT_MAX): each
// warp by shuffles, then the warps' winners by the first warp; 2 barriers. bv / bi: 33 shared slots, a warp's winner
// at its index and the block's at 32 (a following call writes only the warps' slots before its second barrier, so slow
// threads can still read slot 32). Blocks of whole warps.
// 各スレッドの候補 (無ければ -inf, INT_MAX) から、その順でブロックの先頭の (値, index) を求める: warp 内は shuffle、
// 各 warp の勝者は最初の warp でまとめる。バリアは 2 回。bv / bi は共有メモリの 33 枠で、warp の勝者は warp 番号の枠、
// ブロックの勝者は 32 番 (続く呼び出しは 2 回目のバリアまで warp の枠しか書かないので、遅いスレッドもまだ 32 番を
// 読める)。ブロックは warp 単位の大きさ。
__device__ void tf_glm53_block_first(float* v, int* at, float* bv, int* bi) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float x = *v;
    int ix = *at;
    for (int o = 16; o > 0; o >>= 1) {
        const float y = __shfl_xor_sync(0xFFFFFFFFu, x, o);
        const int iy = __shfl_xor_sync(0xFFFFFFFFu, ix, o);
        if (tf_glm53_before(y, iy, x, ix)) { x = y; ix = iy; }
    }
    if (lane == 0) { bv[wid] = x; bi[wid] = ix; }
    __syncthreads();
    if (wid == 0) {
        x = lane < nw ? bv[lane] : -__int_as_float(0x7f800000);
        ix = lane < nw ? bi[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float y = __shfl_xor_sync(0xFFFFFFFFu, x, o);
            const int iy = __shfl_xor_sync(0xFFFFFFFFu, ix, o);
            if (tf_glm53_before(y, iy, x, ix)) { x = y; ix = iy; }
        }
        if (lane == 0) { bv[32] = x; bi[32] = ix; }
    }
    __syncthreads();
    *v = bv[32];
    *at = bi[32];
}

// The head's argmax on the device (a graph replays it without the host): each row's first maximum over this rank's
// vocabulary share (a NaN is the maximum, the first one wins; a row of only -inf gives its first element) as a 16-byte
// row [value, float(id + vocab_off), 0, 0]. One block a row.
// デバイス上での head の argmax (グラフはホストなしで再生する): このランクが受け持つ語彙で各行の最初の最大値 (NaN は
// 最大、最初のものが勝つ。-inf だけの行は先頭の要素) を 16 バイトの行 [value, float(id + vocab_off), 0, 0] で出す。
// 1 行 1 ブロック。
__global__ void tf_glm53_row_argmax(const float* __restrict__ logits, float* __restrict__ amax, int V, int vocab_off) {
    __shared__ float bv[33];
    __shared__ int bi[33];
    const float* row = logits + (long long)blockIdx.x * V;
    float v = -__int_as_float(0x7f800000);
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float x = row[i];
        if (tf_glm53_before(x, i, v, at)) { v = x; at = i; }
    }
    tf_glm53_block_first(&v, &at, bv, bi);
    if (threadIdx.x == 0) {
        float* out = amax + 4 * blockIdx.x;
        out[0] = v;
        out[1] = (float)(at + vocab_off);
        out[2] = 0.0f;
        out[3] = 0.0f;
    }
}

// The gathered rows [world, n, 4]: per row, the lowest rank among equal maxima (a NaN is the maximum, the first one
// wins); its id goes into ids (int64, the next window's input) and tok (int32, for the host).
// 集めた行 [world, n, 4] から、行ごとに等しい最大値のうち最も小さいランクを選ぶ (NaN は最大、最初のものが勝つ)。
// その id を ids (int64、次のウィンドウの入力) と tok (int32、ホスト用) に書く。
__global__ void tf_glm53_pick(const float* __restrict__ all, int world, int n, long long* __restrict__ ids,
                              int* __restrict__ tok) {
    const int r = threadIdx.x;
    if (r >= n) return;
    int best = 0;
    float bv = all[r * 4];
    for (int w = 1; w < world; ++w) {
        const float x = all[(w * n + r) * 4];
        if (bv == bv && (x != x || x > bv)) { bv = x; best = w; }
    }
    const int id = (int)all[(best * n + r) * 4 + 1];
    ids[r] = id;
    tok[r] = id;
}

}  // extern "C"

// One int32 written by a kernel (the decode position before a graph replay; a memset there is a copy-engine op).
// カーネルで int32 を 1 つ書く (グラフ再生前の decode 位置。そこでの memset はコピーエンジンの処理になる)。
extern "C" __global__ void tf_glm53_set_i32(int* __restrict__ p, int v) {
    if (threadIdx.x == 0) *p = v;
}

// tf_glm53_topk_keys' result without sorting the row: a radix select (8 passes of 8 bits) finds the k-th largest value
// v; the values above v and the first (k - above) equal to v are compacted in index order, and their keys are sorted
// only when that order is not already ascending. Equal packed values carry equal keys, so which equal ones are kept
// does not matter.
// 行をソートせずに tf_glm53_topk_keys と同じ結果を出す: radix select (8 ビットずつ 8 パス) で k 番目に大きい値 v を
// 求め、v より大きい値と、v に等しい値の先頭 (k - above) 個を index 順に詰める。そのキーは昇順になっていないときだけ
// ソートする。等しいパック値は等しいキーを持つので、等しいもののどれを残しても結果は同じ。
__device__ int tf_glm53_block_scan(int v, int* warp_sums, int* total);

extern "C" __global__ void tf_glm53_topk_keys_select(const long long* __restrict__ sc, int* __restrict__ out, int T, int k) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_need;
    __shared__ int s_scan[1024];
    extern __shared__ int keys[];  // k ints / int が k 個
    const long long* row = sc + (long long)blockIdx.x * T;
    const int tid = threadIdx.x, nt = blockDim.x;
    unsigned long long prefix = 0, mask = 0;
    // how many of the values matching prefix (from the top) are still to take
    // prefix に一致する値のうち (上から) まだ取る数
    int need = k;
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < T; i += nt) {
            const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ULL;
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, d = 255;
            for (; d > 0; --d) {
                if (acc + (int)hist[d] >= need) break;
                acc += hist[d];
            }
            s_need = need - acc;
            s_prefix = prefix | ((unsigned long long)d << shift);
        }
        __syncthreads();
        need = s_need;
        prefix = s_prefix;
        mask |= 255ULL << shift;
        __syncthreads();
    }
    // prefix: the k-th largest value (unsigned order); need: how many equal to it to keep
    // prefix: k 番目に大きい値 (符号なしの順)。need: それに等しい値をいくつ残すか
    const int per = (T + nt - 1) / nt, lo = min(T, tid * per), hi = min(T, lo + per);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ULL;
        gt += u > prefix;
        eq += u == prefix;
    }
    // exclusive scans of the equal counts (to keep the first `need` in index order), then of the kept counts: by
    // shuffles inside the warps for the 1,024-thread block the launcher uses, else step by step through shared memory
    // 等しい値の数の排他 scan (index 順で先頭の `need` 個を残すため)、続いて残す数の排他 scan。起動側が使う 1,024
    // スレッドのブロックでは warp 内の shuffle、それ以外は共有メモリで 1 段ずつ
    int eq_take, mine, at;
    if (nt == 1024) {
        const int eq_before = tf_glm53_block_scan(eq, s_scan, 0);
        eq_take = max(0, min(eq, need - eq_before));
        mine = gt + eq_take;
        at = tf_glm53_block_scan(mine, s_scan, 0);
    } else {
        s_scan[tid] = eq;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) {
            const int v = tid >= o ? s_scan[tid - o] : 0;
            __syncthreads();
            s_scan[tid] += v;
            __syncthreads();
        }
        const int eq_before = s_scan[tid] - eq;
        eq_take = max(0, min(eq, need - eq_before));
        __syncthreads();
        mine = gt + eq_take;
        s_scan[tid] = mine;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) {
            const int v = tid >= o ? s_scan[tid - o] : 0;
            __syncthreads();
            s_scan[tid] += v;
            __syncthreads();
        }
        at = s_scan[tid] - mine;
    }
    int eq_left = eq_take;
    for (int i = lo; i < hi; ++i) {
        const long long v = row[i];
        const unsigned long long u = (unsigned long long)v ^ 0x8000000000000000ULL;
        if (u > prefix || (u == prefix && eq_left-- > 0)) keys[at++] = (int)(0x7FFFFFFFLL - (v & 0xFFFFFFFFLL));
    }
    __syncthreads();
    int unsorted = 0;
    for (int i = tid + 1; i < k; i += nt) unsorted |= keys[i - 1] > keys[i];
    if (__syncthreads_or(unsorted)) {
        for (int size = 2; size <= k; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < k; i += nt) {
                    const int j = i ^ stride;
                    if (j > i) {
                        const bool asc = (i & size) == 0;
                        const int a = keys[i], b = keys[j];
                        if (asc ? (a > b) : (a < b)) { keys[i] = b; keys[j] = a; }
                    }
                }
                __syncthreads();
            }
        }
    }
    for (int i = tid; i < k; i += nt) out[(long long)blockIdx.x * k + i] = keys[i];
}

// ---- DFlash2 drafter helpers (drafts need not be bit-exact; every rank runs these alike) ----
// ---- DFlash2 drafter の補助 (draft はビット一致を要しない。全ランクが同じように実行する) ----

// The drafter's rotary: cos / sin [n, half] of (pos + i) * inv_freq[j], inv_freq = 1 / theta ** (2 j / hd) in fp32
// (one IEEE step each; powf / cosf / sinf of libdevice). The frequencies do not depend on the position:
// tf_glm53_draft_invfreq makes the table once (one block of `half` threads) and the rope kernel reads it.
// drafter の rotary: (pos + i) * inv_freq[j] の cos / sin [n, half]、inv_freq = 1 / theta ** (2 j / hd) を fp32 で
// (各演算は IEEE 1 回、powf / cosf / sinf は libdevice)。周波数は位置によらないので、tf_glm53_draft_invfreq が表を
// 1 度だけ作り (`half` スレッドの 1 ブロック)、rope カーネルはそれを読む。
extern "C" __global__ void tf_glm53_draft_invfreq(float* __restrict__ inv, float theta, int half, int hd) {
    const int j = threadIdx.x;
    if (j >= half) return;
    const float e = (float)(2 * j) / (float)hd;
    inv[j] = 1.0f / powf(theta, e);
}

extern "C" __global__ void tf_glm53_draft_rope(const long long* __restrict__ pos, const float* __restrict__ inv,
                                               float* __restrict__ cos_out, float* __restrict__ sin_out, int half) {
    const int i = blockIdx.x, j = threadIdx.x;
    if (j >= half) return;
    const float p = (float)(pos[0] + i) * inv[j];
    cos_out[i * half + j] = cosf(p);
    sin_out[i * half + j] = sinf(p);
}

// Rows into the drafter's ring: src [kvh, n, hd] -> ring [kvh, RING, hd] at slot (pos + i) % RING.
// drafter のリングへ行を書く: src [kvh, n, hd] -> ring [kvh, RING, hd] のスロット (pos + i) % RING。
extern "C" __global__ void tf_glm53_draft_ring(const uint16_t* __restrict__ src, uint16_t* __restrict__ ring,
                                               const long long* __restrict__ pos, int n, int hd, int ring_rows) {
    const int h = blockIdx.y, i = blockIdx.x;
    const long long at = pos[0] + i;
    const long long slot = (ring_rows & (ring_rows - 1)) == 0 && at >= 0 ? (at & (ring_rows - 1)) : at % ring_rows;
    for (int d = threadIdx.x; d < hd; d += blockDim.x)
        ring[((long long)h * ring_rows + slot) * hd + d] = src[((long long)h * n + i) * hd + d];
}

// pos += n (the context update's last step).
// pos += n (コンテキスト更新の最後の手順)。
extern "C" __global__ void tf_glm53_draft_advance(long long* __restrict__ pos, int n) {
    if (threadIdx.x == 0) pos[0] += n;
}

// Every rank's fp32 partial [W, n] added in rank order (acc = g[0]; acc += g[i]), then rounded to bf16 (RNE).
// 全ランクの fp32 部分和 [W, n] をランク順に足し (acc = g[0]; acc += g[i])、bf16 へ丸める (RNE)。
extern "C" __global__ void tf_glm53_rank_sum_bf16(const float* __restrict__ g, uint16_t* __restrict__ out, int n,
                                                   int world) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float acc = g[i];
        for (int w = 1; w < world; ++w) acc = acc + g[(long long)w * n + i];
        const uint32_t b = __float_as_uint(acc);
        out[i] = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    }
}

// The top k of each row (k <= 32), deterministic: values descending (a NaN above everything), equal values by the
// lower index; out row = [k values | k global ids (vocab_off + index) as fp32 bits]. Every thread holds the first of
// its own elements not yet taken; a round takes the block's first of those, and only the thread it came from reads its
// elements again for its next one (the other threads' candidates stand).
// 各行の上位 k 個 (k <= 32) を決定的に: 値の降順 (NaN は何よりも上)、等しい値は index の小さいほうが前。出力の行は
// [k 個の値 | k 個のグローバル id (vocab_off + index) を fp32 のビットで]。各スレッドはまだ取られていない自分の要素の
// 先頭を持ち、1 ラウンドでそのうちブロックの先頭を取る。次の候補を探して要素を読み直すのは、取られたスレッドだけ
// (ほかのスレッドの候補はそのまま)。
extern "C" __global__ void tf_glm53_draft_topk(const float* __restrict__ logits, float* __restrict__ out, int V, int k,
                                               int vocab_off) {
    __shared__ float bv[33];
    __shared__ int bi[33];
    const float* row = logits + (long long)blockIdx.x * V;
    float* o = out + (long long)blockIdx.x * 2 * k;
    float v = -__int_as_float(0x7f800000);
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float x = row[i];
        if (tf_glm53_before(x, i, v, at)) { v = x; at = i; }
    }
    for (int t = 0; t < k; ++t) {
        float top_v = v;
        int top_i = at;
        tf_glm53_block_first(&top_v, &top_i, bv, bi);
        if (threadIdx.x == 0) {
            o[t] = top_v;
            o[k + t] = __int_as_float(top_i + vocab_off);
        }
        // this thread's candidate was the one taken: find its next
        // このスレッドの候補が取られた: 次の候補を探す
        if (t + 1 < k && top_i == at && at != 0x7fffffff) {
            const float lv = v;
            const int li = at;
            v = -__int_as_float(0x7f800000);
            at = 0x7fffffff;
            for (int i = threadIdx.x; i < V; i += blockDim.x) {
                const float x = row[i];
                if (tf_glm53_before(lv, li, x, i) && tf_glm53_before(x, i, v, at)) { v = x; at = i; }
            }
        }
    }
}

// The digit of a radix pass holding the need-th largest value (hist: 256 counts) and the need left inside it, by the
// first warp (every lane returns the same); the same d and need as the serial scan
//   acc = 0; for (d = 255; d > 0; --d) { if (acc + hist[d] >= need) break; acc += hist[d]; }  need -= acc
// radix の 1 パスで need 番目に大きい値を含む桁 (hist: 256 個の数) と、その桁の中で残る need を最初の warp で求める
// (全 lane が同じ値を返す)。d と need は上の逐次走査と同じになる。
__device__ __forceinline__ void tf_glm53_digit_pick(const unsigned int* hist, int need, int* d_out, int* need_out) {
    const int lane = threadIdx.x & 31;
    int c[8], sum = 0;
#pragma unroll
    // lane l: digits 255 - 8l .. 248 - 8l, descending
    // lane l: 桁 255 - 8l .. 248 - 8l を降順で
    for (int q = 0; q < 8; q++) {
        c[q] = (int)hist[255 - 8 * lane - q];
        sum += c[q];
    }
    // inclusive scan over lanes = counts in digits >= 248 - 8l
    // lane 方向の包含 scan = 桁 248 - 8l 以上の数
    int x = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o) x += y;
    }
    const int before = x - sum;
    const unsigned hit = __ballot_sync(0xFFFFFFFFu, need > 0 && before < need && x >= need);
    int d = 0, left = 0;
    if (need <= 0) {                             // the serial scan stops at once / 逐次走査ならすぐ止まる
        d = 255;
        left = need;
    // fewer than need in all: d = 0 after taking digits 255..1
    // 全体で need 未満: 桁 255..1 を取った後の d = 0
    } else if (hit == 0) {
        const int total = __shfl_sync(0xFFFFFFFFu, x, 31);
        d = 0;
        left = need - (total - (int)hist[0]);
    } else {
        const int L = __ffs(hit) - 1;
        int dd = 0, ll = 0;
        if (lane == L) {
            int acc = before;
            int q = 0;
            for (; q < 8; q++) {
                const int dig = 255 - 8 * lane - q;
                if (dig == 0 || acc + c[q] >= need) break;
                acc += c[q];
            }
            dd = 255 - 8 * lane - q;
            ll = need - acc;
        }
        d = __shfl_sync(0xFFFFFFFFu, dd, L);
        left = __shfl_sync(0xFFFFFFFFu, ll, L);
    }
    *d_out = d;
    *need_out = left;
}

// ---- decode context parallelism (DCP > 1): the indexer's choice over the group ----
// ---- decode のコンテキスト並列 (DCP > 1): グループ全体での indexer の選択 ----

// The radix select of tf_glm53_topk_keys_select over `n` packed int64 values of a row: the k-th largest in prefix,
// and how many equal to it to keep in need (block-wide; every thread returns the same).
// `top` bytes already known (prefix0 holds them, the rest 0): the passes start below them.
// 行の `n` 個のパック済み int64 値に tf_glm53_topk_keys_select と同じ radix select をかける: k 番目に大きい値を
// prefix に、それに等しい値をいくつ残すかを need に返す (ブロック全体で。全スレッドが同じ値を返す)。
// 上位 `top` バイトが既知なら (prefix0 に入れ、残りは 0)、パスはその下から始める。
__device__ void tf_glm53_radix_kth_from(const long long* __restrict__ row, int n, int k, unsigned long long prefix0,
                                        int top, unsigned long long* s_prefix, int* s_need, unsigned int* hist,
                                        unsigned long long* prefix_out, int* need_out) {
    const int tid = threadIdx.x, nt = blockDim.x;
    unsigned long long prefix = prefix0, mask = top ? ~0ULL << (64 - 8 * top) : 0;
    int need = k;
    for (int shift = 56 - 8 * top; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += nt) {
            const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ULL;
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, need, &d, &left);
            if (tid == 0) {
                *s_need = left;
                *s_prefix = prefix | ((unsigned long long)d << shift);
            }
        }
        __syncthreads();
        need = *s_need;
        prefix = *s_prefix;
        mask |= 255ULL << shift;
        __syncthreads();
    }
    *prefix_out = prefix;
    *need_out = need;
}

__device__ void tf_glm53_radix_kth(const long long* __restrict__ row, int n, int k, unsigned long long* s_prefix,
                                   int* s_need, unsigned int* hist, unsigned long long* prefix_out, int* need_out) {
    tf_glm53_radix_kth_from(row, n, k, 0, 0, s_prefix, s_need, hist, prefix_out, need_out);
}

// Exclusive prefix sum of v over a block of 1024 threads (in thread order); *total gets the sum. Every thread calls it.
// 1024 スレッドのブロックでの v の排他 prefix sum (スレッド順)。*total に総和が入る。全スレッドが呼ぶ。
__device__ int tf_glm53_block_scan(int v, int* warp_sums, int* total) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int w = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFFu, w, o);
            if (lane >= o) w += y;
        }
        warp_sums[lane] = w;                     // inclusive over warps / warp 方向の包含和
    }
    __syncthreads();
    const int before = (wid ? warp_sums[wid - 1] : 0) + x - v;
    if (total) *total = warp_sums[31];
    __syncthreads();
    return before;
}

// The threshold of the radix path's selectKeys: the k-th largest of n uint32 order words (four 8-bit passes) -> its
// value and how many of the values equal to it are taken (need).
// radix の経路の selectKeys と同じしきい値: n 個の uint32 の順序語のうち k 番目に大きいもの (8 ビットずつ 4 パス)。
// その値と、それに等しい値をいくつ取るか (need)。
__device__ void tf_glm53_radix_kth32(const unsigned* __restrict__ v, int n, int k, unsigned* s_prefix, int* s_need,
                                     unsigned int* hist, unsigned* prefix_out, int* need_out) {
    const int tid = threadIdx.x, nt = blockDim.x;
    unsigned prefix = 0, mask = 0;
    int need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += nt)
            if ((v[i] & mask) == prefix) atomicAdd(&hist[(v[i] >> shift) & 255], 1u);
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, need, &d, &left);
            if (tid == 0) {
                *s_need = left;
                *s_prefix = prefix | ((unsigned)d << shift);
            }
        }
        __syncthreads();
        need = *s_need;
        prefix = *s_prefix;
        mask |= 255u << shift;
        __syncthreads();
    }
    *prefix_out = prefix;
    *need_out = need;
}

// The kk = min(k, T) largest packed values of each row of T, into out [rows, k] (the reader takes any order); the
// slots past kk are filled with -9223372036854775807.
// 各行 (長さ T) のパック値の上位 kk = min(k, T) 個を out [rows, k] へ (読む側は順を問わない)。kk より後ろの枠は
// -9223372036854775807 で埋める。
extern "C" __global__ void tf_glm53_topk_values(const long long* __restrict__ sc, long long* __restrict__ out, int T,
                                                int k) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_need;
    const long long* row = sc + (long long)blockIdx.x * T;
    long long* o = out + (long long)blockIdx.x * k;
    const int kk = min(k, T);
    unsigned long long th;
    int need;
    tf_glm53_radix_kth(row, T, kk, &s_prefix, &s_need, hist, &th, &need);
    // the kk best in column order: all above the k-th, and the first `need` columns equal to it
    // 上位 kk 個を列の順に: k 番目より大きいものは全部、等しいものは先頭の `need` 列
    __shared__ int warp_sums[32];
    __shared__ int s_tot;
    int written = 0, eq_seen = 0;
    for (int c = 0; c < T; c += blockDim.x) {
        const int i = c + threadIdx.x;
        const unsigned long long u = i < T ? ((unsigned long long)row[i] ^ 0x8000000000000000ULL) : 0ULL;
        const int eq = i < T && u == th;
        const int eb = tf_glm53_block_scan(eq, warp_sums, &s_tot);
        const int eqn = s_tot;
        const int take = i < T && (u > th || (eq && eq_seen + eb < need));
        const int tb = tf_glm53_block_scan(take, warp_sums, &s_tot);
        if (take) o[written + tb] = row[i];
        written += s_tot;
        eq_seen += eqn;
    }
    for (int i = kk + threadIdx.x; i < k; i += blockDim.x) o[i] = -9223372036854775807LL;
}

// tf_glm53_topk_values' output from the scores' int32 order words (s [rows, T], 0 for a later key): the packed key of
// column t is (word ^ 0x80000000) << 32 | 0x7FFFFFFF - (t dcp + rank), and a later key (t dcp + rank > pos[0] + r0 +
// row) is the -9223372036854775807 sentinel. The kk best packed keys are the columns of the kk largest words, ties to
// the lowest columns (a lower column is the larger low word; later keys are the highest columns), so the set and its
// column order are tf_glm53_topk_values', from 4 + 1 passes over 4 bytes a key instead of 8 + 1 over 8.
// One block a row; a thread takes 4 words a step by one aligned 16-byte load. hist [256], s_prefix, s_need,
// warp_sums [32] and s_tot are the block's shared scratch.
// The loads take whole groups of 4: up to 3 words past the last row's end are read (and dropped by the column test),
// so `s` must hold them; the engine's score buffer has 8 bytes a word, twice what is read.
// スコアの int32 順序語 (s [rows, T]、後方のキーは 0) から tf_glm53_topk_values と同じ出力を作る: 列 t のパック済み
// キーは (word ^ 0x80000000) << 32 | 0x7FFFFFFF - (t dcp + rank)、後方のキー (t dcp + rank > pos[0] + r0 + row) は
// 番兵 -9223372036854775807。上位 kk 個のパック済みキーは語の上位 kk 個の列で、同点は小さい列が勝つ (小さい列ほど
// 下位語が大きく、後方のキーは最も大きい列)。そのため集合も列の順も tf_glm53_topk_values と同じになり、8 バイトの
// キーで 8 + 1 パスのところを 4 バイトで 4 + 1 パスで済む。
// 1 行 1 ブロック。スレッドは 1 ステップで 4 語を、整列した 16 バイトの load 1 回で読む。hist [256]・s_prefix・
// s_need・warp_sums [32]・s_tot はブロックの共有作業領域。
// load は 4 語単位なので、最後の行の末尾から最大 3 語先まで読む (列の判定で捨てる)。そのため `s` はその分を含む
// 必要がある。エンジンのスコアバッファは 1 語 8 バイトで、読む量の 2 倍ある。
__device__ void tf_glm53_topk_words_row(const unsigned* __restrict__ s, long long* __restrict__ out,
                                        const int* __restrict__ pos, int T, int k, int r0, int dcp, int rank,
                                        unsigned int* hist, unsigned* s_prefix_p, int* s_need_p, int* warp_sums, int* s_tot_p,
                                        // `top` bytes known: prefix0, need0 left
                                        // 上位 `top` バイトが既知: prefix0、残りは need0
                                        unsigned prefix0, int top, int need0) {
    const int tid = threadIdx.x, nt = blockDim.x;
    // the row's first word; groups of 4 words from a0 (aligned)
    // 行の最初の語。4 語のグループは a0 (整列済み) から
    const long long base = (long long)blockIdx.x * T;
    const long long a0 = base & ~3LL;
    const int lead = (int)(base - a0), groups = (lead + T + 3) / 4;
    long long* o = out + (long long)blockIdx.x * k;
    const int kk = min(k, T);
    unsigned prefix = prefix0, mask = top ? ~0u << (32 - 8 * top) : 0;
    int need = top ? need0 : kk;
    for (int shift = 24 - 8 * top; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        // a thread's run of one digit goes to the histogram in one add
        // スレッドで同じ桁が続く分は 1 回の加算でヒストグラムへ
        int bin = -1, cnt = 0;
        for (int g = tid; g < groups; g += nt) {
            const uint4 v = *(const uint4*)(s + a0 + 4LL * g);
            const unsigned u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = 4 * g + e - lead;
                if (col >= 0 && col < T && (u[e] & mask) == prefix) {
                    const int d = (u[e] >> shift) & 255;
                    if (d != bin) {
                        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
                        bin = d;
                        cnt = 0;
                    }
                    cnt++;
                }
            }
        }
        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, need, &d, &left);
            if (tid == 0) {
                *s_need_p = left;
                *s_prefix_p = prefix | ((unsigned)d << shift);
            }
        }
        __syncthreads();
        need = *s_need_p;
        prefix = *s_prefix_p;
        mask |= 255u << shift;
        __syncthreads();
    }
    // column order: the words above the k-th all, the ones equal to it while `need` lasts
    // 列の順に: k 番目より大きい語は全部、等しい語は `need` が尽きるまで
    const int p = pos[0] + r0 + (int)blockIdx.x;
    int written = 0, eq_seen = 0;
    for (int g0 = 0; g0 < groups; g0 += nt) {
        const int g = g0 + tid;
        unsigned u[4] = { 0, 0, 0, 0 };
        int gtm = 0, eqm = 0;
        if (g < groups) {
            const uint4 v = *(const uint4*)(s + a0 + 4LL * g);
            u[0] = v.x; u[1] = v.y; u[2] = v.z; u[3] = v.w;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = 4 * g + e - lead;
                if (col >= 0 && col < T) {
                    if (u[e] > prefix) gtm |= 1 << e;
                    if (u[e] == prefix) eqm |= 1 << e;
                }
            }
        }
        const int before = tf_glm53_block_scan(__popc(eqm) | (__popc(gtm) << 16), warp_sums, s_tot_p);
        const int total = *s_tot_p;
        int room = need - eq_seen;
        if (room < 0) room = 0;
        // words equal to the k-th before this thread in the step
        // このステップでこのスレッドより前の、k 番目に等しい語の数
        int eb = before & 0xffff;
        int at = written + (before >> 16) + (eb < room ? eb : room);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int take = ((gtm >> e) & 1) || (((eqm >> e) & 1) && eb < room);
            if (take) {
                const int gp = (4 * g + e - lead) * dcp + rank;
                o[at++] = gp <= p ? (long long)(((unsigned long long)(u[e] ^ 0x80000000u) << 32) | (unsigned)(0x7fffffff - gp))
                                  : -9223372036854775807LL;
            }
            eb += (eqm >> e) & 1;
        }
        const int eq_all = total & 0xffff;
        written += (total >> 16) + (eq_all < room ? eq_all : room);
        eq_seen += eq_all;
    }
    for (int i = kk + tid; i < k; i += nt) o[i] = -9223372036854775807LL;
}

extern "C" __global__ void tf_glm53_topk_words(const unsigned* __restrict__ s, long long* __restrict__ out,
                                               const int* __restrict__ pos, int T, int k, int r0, int dcp, int rank) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    __shared__ int warp_sums[32];
    __shared__ int s_tot;
    tf_glm53_topk_words_row(s, out, pos, T, k, r0, dcp, rank, hist, &s_prefix, &s_need, warp_sums, &s_tot, 0, 0, 0);
}

// tf_glm53_topk_words' picks in any order, from two passes over the row instead of five (for the sequence-parallel
// DCP select, whose merge takes the packed keys' exact order, so only the set matters):
//   a strided sample (16K words) guesses the top byte of the kk-th word;
//   pass 1 counts the top byte of every word and, for the three top bytes around the guess, the second byte: the
//          16-bit prefix of the kk-th word, how many words of its bucket are wanted and how many it holds;
//   pass 2 writes the words above the bucket as packed keys and gathers the bucket's in shared memory, where a radix
//          select over their packed keys (distinct) takes the wanted ones.
// A guess that missed, a bucket past TW_CAP words, a row of fewer valid keys than kk or a short row: the five passes
// (tf_glm53_topk_words_row), less the one or two whose byte pass 1 already fixed. One block of 1,024 threads a row.
// tf_glm53_topk_words と同じ選択を順不同で、行を 5 回でなく 2 回走査して求める (sequence-parallel の DCP 選択用。
// その merge がパック済みキーの厳密な順に並べるので、集合だけが問題になる):
//   間引いた標本 (16K 語) で kk 番目の語の最上位バイトを推定する。
//   パス 1 は全語の最上位バイトを数え、推定の周りの 3 つの最上位バイトについては 2 番目のバイトも数える。これで
//          kk 番目の語の 16 ビットの prefix、そのバケットから取る語の数、バケットの語数が決まる。
//   パス 2 はバケットより上の語をパック済みキーとして書き、バケットの語を共有メモリに集める。そこで互いに異なる
//          パック済みキーへの radix select をかけ、必要な分を取る。
// 推定が外れた、バケットが TW_CAP 語を超えた、有効なキーが kk 未満、行が短い、のどれかなら 5 パス
// (tf_glm53_topk_words_row) に戻り、パス 1 ですでに決まった 1〜2 バイトのパスは省く。1 行に 1,024 スレッドの
// ブロック 1 つ。
#define TW_CAP 4096
extern "C" __global__ void tf_glm53_topk_words_any(const unsigned* __restrict__ s, long long* __restrict__ out,
                                                   const int* __restrict__ pos, int T, int k, int r0, int dcp, int rank) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int hist2[3 * 256];
    __shared__ unsigned long long bk[TW_CAP];
    __shared__ unsigned long long s_p64;
    __shared__ unsigned s_prefix;
    __shared__ int s_need, s_cnt, s_bc, s_tot;
    __shared__ int warp_sums[32];
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long base = (long long)blockIdx.x * T, a0 = base & ~3LL;
    const int lead = (int)(base - a0), groups = (lead + T + 3) / 4;
    long long* o = out + (long long)blockIdx.x * k;
    const int kk = min(k, T);
    bool fast = T >= 8 * k && T >= 16 * nt;
    unsigned p16 = 0;
    int need16 = 0, bc = 0;
    // The fallback starts below the bytes the passes here fixed exactly: s_top of them (0 .. 2), still in s_prefix with
    // their need in s_need. Kept in shared memory: in registers they would slow every row for the few that fall back.
    // フォールバックは、ここでのパスが確定させたバイトの下から始める: 確定したのは s_top バイト (0〜2) で、値は
    // s_prefix、その need は s_need に残る。共有メモリに置くのは、レジスタだと少数のフォールバックのために全行が
    // 遅くなるから。
    __shared__ int s_top;
    const bool fast0 = fast;
    if (fast) {
        // the guess: the top byte of the kk-th largest of a strided sample (4 groups a thread)
        // 推定: 間引いた標本 (スレッドあたり 4 グループ) で kk 番目に大きい語の最上位バイト
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const int stride = groups / (4 * nt);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int g = (tid + j * nt) * stride;
            const uint4 v = *(const uint4*)(s + a0 + 4LL * g);
            const unsigned u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = 4 * g + e - lead;
                if (col >= 0 && col < T) atomicAdd(&hist[u[e] >> 24], 1u);
            }
        }
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, max(1, (int)((long long)kk * (16 * nt) / T)), &d, &left);
            if (tid == 0) s_prefix = (unsigned)d;
        }
        __syncthreads();
        // the second byte is counted for the top bytes lo3 .. lo3 + 2
        // 最上位バイトが lo3 .. lo3 + 2 の語は 2 番目のバイトも数える
        const int lo3 = (int)s_prefix - 1;
        __syncthreads();
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        for (int i = tid; i < 3 * 256; i += nt) hist2[i] = 0;
        __syncthreads();
        // a thread's run of one top byte goes to the histogram in one add
        // スレッドで同じ最上位バイトが続く分は 1 回の加算でヒストグラムへ
        int bin = -1, cnt = 0;
        for (int g = tid; g < groups; g += nt) {
            const uint4 v = *(const uint4*)(s + a0 + 4LL * g);
            const unsigned u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = 4 * g + e - lead;
                if (col >= 0 && col < T) {
                    const int d = (int)(u[e] >> 24);
                    if (d != bin) {
                        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
                        bin = d;
                        cnt = 0;
                    }
                    cnt++;
                    const unsigned j = (unsigned)(d - lo3);
                    if (j < 3u) atomicAdd(&hist2[j * 256 + ((u[e] >> 16) & 255)], 1u);
                }
            }
        }
        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, kk, &d, &left);
            if (tid == 0) {
                s_prefix = (unsigned)d;
                s_need = left;
                s_top = 1;
            }
        }
        __syncthreads();
        const int d3 = (int)s_prefix, need3 = s_need;
        const unsigned j3 = (unsigned)(d3 - lo3);
        __syncthreads();
        fast = d3 > 0 && j3 < 3u && need3 > 0;
        if (fast) {
            if (tid < 32) {
                int d, left;
                tf_glm53_digit_pick(hist2 + j3 * 256, need3, &d, &left);
                if (tid == 0) {
                    s_prefix = ((unsigned)d3 << 8) | (unsigned)d;
                    s_need = left;
                    s_bc = (int)hist2[j3 * 256 + d];
                    s_top = 2;
                }
            }
            __syncthreads();
            p16 = s_prefix;
            need16 = s_need;
            bc = s_bc;
            __syncthreads();
            fast = need16 > 0 && need16 <= bc && bc <= TW_CAP;
        }
    }
    if (!fast) {
        // a row too short for the passes above (fast0 false) wrote none of the three: nothing is read then
        // 上のパスには短すぎる行 (fast0 が false) は 3 つのどれも書いていないので、そのときは何も読まない
        const int top = fast0 ? s_top : 0;
        tf_glm53_topk_words_row(s, out, pos, T, k, r0, dcp, rank, hist, &s_prefix, &s_need, warp_sums, &s_tot,
                                top == 2 ? s_prefix << 16 : top == 1 ? s_prefix << 24 : 0u, top, top ? s_need : 0);
        return;
    }
    // pass 2: every word here is above 0, so its key is a scored one (not a later key)
    // パス 2: ここでの語はすべて 0 より大きいので、そのキーはスコアの付いたもの (後方のキーではない)
    if (tid == 0) {
        s_cnt = 0;
        s_bc = 0;
    }
    __syncthreads();
    for (int g = tid; g < groups; g += nt) {
        const uint4 v = *(const uint4*)(s + a0 + 4LL * g);
        const unsigned u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int col = 4 * g + e - lead;
            if (col >= 0 && col < T) {
                const unsigned t16 = u[e] >> 16;
                if (t16 >= p16) {
                    const unsigned long long key = ((unsigned long long)u[e] << 32) | (unsigned)(0x7fffffff - (col * dcp + rank));
                    if (t16 > p16) o[atomicAdd(&s_cnt, 1)] = (long long)(key ^ 0x8000000000000000ULL);
                    else bk[atomicAdd(&s_bc, 1)] = key;
                }
            }
        }
    }
    __syncthreads();
    // the need16 largest of the bucket's bc keys (their top 16 bits are p16): the radix select's remaining 6 bytes
    // バケットの bc 個のキー (上位 16 ビットは p16) のうち上位 need16 個: radix select の残り 6 バイト
    unsigned long long prefix = (unsigned long long)p16 << 48, mask = 0xffffULL << 48;
    int need = need16;
    if (need16 < bc) {
        for (int shift = 40; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < bc; i += nt)
                if ((bk[i] & mask) == prefix) atomicAdd(&hist[(bk[i] >> shift) & 255], 1u);
            __syncthreads();
            if (tid < 32) {
                int d, left;
                tf_glm53_digit_pick(hist, need, &d, &left);
                if (tid == 0) {
                    s_need = left;
                    s_p64 = prefix | ((unsigned long long)d << shift);
                }
            }
            __syncthreads();
            need = s_need;
            prefix = s_p64;
            mask |= 255ULL << shift;
            __syncthreads();
        }
    }
    for (int i = tid; i < bc; i += nt)
        if (need16 == bc || bk[i] >= prefix) o[atomicAdd(&s_cnt, 1)] = (long long)(bk[i] ^ 0x8000000000000000ULL);
    for (int i = kk + tid; i < k; i += nt) o[i] = -9223372036854775807LL;
}

// tf_glm53_topk_words_any's picks from the order words' top 16 bits p [rows, T] u16 (written beside them by
// tf_glm53_n_index_scores pack 2, 16-byte aligned): the sample and pass 1 read only those (8 a 16-byte load, half the
// bytes), pass 2 reads them too and fetches the full word of a column only when its top 16 bits are at or above the
// bucket's (the row's few thousand candidates instead of all its words). The same buckets, counts and keys as
// tf_glm53_topk_words_any; the fallback (tf_glm53_topk_words_row) reads the words as before. The loads take whole
// groups of 8: up to 7 values past the last row's end are read (and dropped by the column test).
// tf_glm53_topk_words_any と同じ選択を、順序語の上位 16 ビット p [rows, T] u16 (tf_glm53_n_index_scores の pack 2 が
// 隣に書く、16 バイト境界) から: 標本とパス 1 はそれだけを読み (16 バイトの load で 8 個、バイト数は半分)、パス 2 も
// それを読んで、上位 16 ビットがバケット以上の列だけ語全体を読む (行の数千個の候補だけで、全部の語ではない)。
// バケット・件数・キーは tf_glm53_topk_words_any と同じ。フォールバック (tf_glm53_topk_words_row) は従来どおり語を
// 読む。load は 8 個単位なので、最後の行の末尾から最大 7 個先まで読む (列の判定で捨てる)。
extern "C" __global__ void tf_glm53_topk_words_any16(const unsigned* __restrict__ s, const unsigned short* __restrict__ p,
                                                     long long* __restrict__ out, const int* __restrict__ pos, int T,
                                                     int k, int r0, int dcp, int rank) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int hist2[3 * 256];
    __shared__ unsigned long long bk[TW_CAP];
    __shared__ unsigned long long s_p64;
    __shared__ unsigned s_prefix;
    __shared__ int s_need, s_cnt, s_bc, s_tot;
    __shared__ int warp_sums[32];
    __shared__ int s_top;
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long base = (long long)blockIdx.x * T, b0 = base & ~7LL;
    // columns past the row's position hold word 0 (top 16 bits 0, not written by the scores): they are counted into
    // byte 0 without being read, so the histograms are those of every column
    // 行の位置より後の列は語 0 (上位 16 ビットも 0。点数の側は書かない): 読まずにバイト 0 に数えるので、ヒストグラムは
    // 全列のものと同じ
    const int pr = pos[0] + r0 + (int)blockIdx.x;
    const int cv = pr < rank ? 0 : min(T, (pr - rank) / dcp + 1);
    const int lead = (int)(base - b0), groups = (lead + cv + 7) / 8;
    const unsigned* srow = s + base;
    long long* o = out + (long long)blockIdx.x * k;
    const int kk = min(k, T);
    bool fast = cv >= 8 * k && cv >= 16 * nt;
    unsigned p16 = 0;
    int need16 = 0, bc = 0;
    const bool fast0 = fast;
    if (fast) {
        // the guess: the top byte of the kk-th largest of a strided sample (2 groups of 8 a thread)
        // 推定: 間引いた標本 (スレッドあたり 8 個のグループ 2 つ) で kk 番目に大きい語の最上位バイト
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const int stride = groups / (2 * nt);
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int g = (tid + j * nt) * stride;
            const uint4 v = *(const uint4*)(p + b0 + 8LL * g);
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int col = 8 * g + e - lead;
                if (col >= 0 && col < cv) atomicAdd(&hist[(w[e >> 1] >> (16 * (e & 1) + 8)) & 255], 1u);
            }
        }
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, max(1, (int)((long long)kk * (16 * nt) / cv)), &d, &left);
            if (tid == 0) s_prefix = (unsigned)d;
        }
        __syncthreads();
        const int lo3 = (int)s_prefix - 1;
        __syncthreads();
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        for (int i = tid; i < 3 * 256; i += nt) hist2[i] = 0;
        __syncthreads();
        // pass 1: every column's top byte; the second byte for the top bytes lo3 .. lo3 + 2
        // パス 1: 全列の最上位バイト。最上位バイトが lo3 .. lo3 + 2 の列は 2 番目のバイトも
        int bin = -1, cnt = 0;
        for (int g = tid; g < groups; g += nt) {
            const uint4 v = *(const uint4*)(p + b0 + 8LL * g);
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int col = 8 * g + e - lead;
                if (col >= 0 && col < cv) {
                    const unsigned t16 = (w[e >> 1] >> (16 * (e & 1))) & 0xffffu;
                    const int d = (int)(t16 >> 8);
                    if (d != bin) {
                        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
                        bin = d;
                        cnt = 0;
                    }
                    cnt++;
                    const unsigned j = (unsigned)(d - lo3);
                    if (j < 3u) atomicAdd(&hist2[j * 256 + (t16 & 255)], 1u);
                }
            }
        }
        if (cnt) atomicAdd(&hist[bin], (unsigned)cnt);
        if (tid == 0 && cv < T) {   // the later columns: top byte 0, second byte 0 / 後の列: 最上位バイト 0、2 番目 0
            atomicAdd(&hist[0], (unsigned)(T - cv));
            if ((unsigned)(0 - lo3) < 3u) atomicAdd(&hist2[(0 - lo3) * 256], (unsigned)(T - cv));
        }
        __syncthreads();
        if (tid < 32) {
            int d, left;
            tf_glm53_digit_pick(hist, kk, &d, &left);
            if (tid == 0) {
                s_prefix = (unsigned)d;
                s_need = left;
                s_top = 1;
            }
        }
        __syncthreads();
        const int d3 = (int)s_prefix, need3 = s_need;
        const unsigned j3 = (unsigned)(d3 - lo3);
        __syncthreads();
        fast = d3 > 0 && j3 < 3u && need3 > 0;
        if (fast) {
            if (tid < 32) {
                int d, left;
                tf_glm53_digit_pick(hist2 + j3 * 256, need3, &d, &left);
                if (tid == 0) {
                    s_prefix = ((unsigned)d3 << 8) | (unsigned)d;
                    s_need = left;
                    s_bc = (int)hist2[j3 * 256 + d];
                    s_top = 2;
                }
            }
            __syncthreads();
            p16 = s_prefix;
            need16 = s_need;
            bc = s_bc;
            __syncthreads();
            fast = need16 > 0 && need16 <= bc && bc <= TW_CAP;
        }
    }
    if (!fast) {
        const int top = fast0 ? s_top : 0;
        tf_glm53_topk_words_row(s, out, pos, T, k, r0, dcp, rank, hist, &s_prefix, &s_need, warp_sums, &s_tot,
                                top == 2 ? s_prefix << 16 : top == 1 ? s_prefix << 24 : 0u, top, top ? s_need : 0);
        return;
    }
    // pass 2: the columns whose top 16 bits reach the bucket's, their full words fetched (every such word is above 0:
    // a scored key)
    // パス 2: 上位 16 ビットがバケット以上の列だけ、語全体を読む (そうした語はすべて 0 より大きい: スコアの付いたキー)
    if (tid == 0) {
        s_cnt = 0;
        s_bc = 0;
    }
    __syncthreads();
    for (int g = tid; g < groups; g += nt) {
        const uint4 v = *(const uint4*)(p + b0 + 8LL * g);
        const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int col = 8 * g + e - lead;
            const unsigned t16 = (w[e >> 1] >> (16 * (e & 1))) & 0xffffu;
            if (col >= 0 && col < cv && t16 >= p16) {
                const unsigned long long key = ((unsigned long long)srow[col] << 32) | (unsigned)(0x7fffffff - (col * dcp + rank));
                if (t16 > p16) o[atomicAdd(&s_cnt, 1)] = (long long)(key ^ 0x8000000000000000ULL);
                else bk[atomicAdd(&s_bc, 1)] = key;
            }
        }
    }
    __syncthreads();
    // the need16 largest of the bucket's bc keys (their top 16 bits are p16): the radix select's remaining 6 bytes
    // バケットの bc 個のキー (上位 16 ビットは p16) のうち上位 need16 個: radix select の残り 6 バイト
    unsigned long long prefix = (unsigned long long)p16 << 48, mask = 0xffffULL << 48;
    int need = need16;
    if (need16 < bc) {
        for (int shift = 40; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < bc; i += nt)
                if ((bk[i] & mask) == prefix) atomicAdd(&hist[(bk[i] >> shift) & 255], 1u);
            __syncthreads();
            if (tid < 32) {
                int d, left;
                tf_glm53_digit_pick(hist, need, &d, &left);
                if (tid == 0) {
                    s_need = left;
                    s_p64 = prefix | ((unsigned long long)d << shift);
                }
            }
            __syncthreads();
            need = s_need;
            prefix = s_p64;
            mask |= 255ULL << shift;
            __syncthreads();
        }
    }
    for (int i = tid; i < bc; i += nt)
        if (need16 == bc || bk[i] >= prefix) o[atomicAdd(&s_cnt, 1)] = (long long)(bk[i] ^ 0x8000000000000000ULL);
    for (int i = kk + tid; i < k; i += nt) o[i] = -9223372036854775807LL;
}

// A global position g of a group of G members: whether `member` owns it (g % G) and its local slot there (g / G); a
// mask and a shift for a pair of members (G 2) and a position that is not negative, as every one is.
// G メンバーのグループでのグローバル位置 g: `member` がそれを持つか (g % G) と、そこでのローカルスロット (g / G)。
// メンバーがペア (G 2) で位置が負でない (どの位置もそう) ときはマスクとシフトで求める。
__device__ __forceinline__ bool tf_glm53_own_slot(long long g, int G, int member, int* slot) {
    if (G == 2 && g >= 0) {
        *slot = (int)(g >> 1);
        return (int)(g & 1) == member;
    }
    *slot = (int)(g / G);
    return g % G == member;
}

// The group's candidates of a row (cand [G, n, k] gathered in member order, row r at [g][r]): the k largest of the
// G k, their global positions (0x7FFFFFFF - low word), this member's own (gpos % G == member) as local slots
// gpos / G sorted ascending into tok [r, k] (the rest past them), and their count into cnt[r].
// `hiword` (the radix path, blocks of 32 rows or more): the k-th best by the score word alone, its ties taken in
// column order (member 0's candidates first, each member's in its own order); else the exact order of the packed keys.
// グループから集めた行の候補 (cand [G, n, k] はメンバー順、行 r は [g][r]): G k 個のうち上位 k 個とそのグローバル
// 位置 (0x7FFFFFFF - 下位語) を求め、このメンバーの分 (gpos % G == member) をローカルスロット gpos / G として昇順に
// tok [r, k] へ (残りはその後ろ)、その数を cnt[r] へ書く。
// `hiword` (radix の経路、32 行以上のブロック): k 番目はスコア語だけで決め、同点は列の順に取る (メンバー 0 の候補が先、
// 各メンバーの中はその順)。そうでなければパック済みキーの厳密な順。
extern "C" __global__ void tf_glm53_dcp_select(const long long* __restrict__ cand, int* __restrict__ tok,
                                               int* __restrict__ cnt, int n, int k, int G, int member, int hiword) {
    // hiword: G * k score words (the packed values' high words), G * k low words, then k int keys; else G * k packed
    // values, then k int keys. 8 G k + 4 k bytes either way (G 4: 73,728, under GB10's 101,376 a block).
    // hiword: G * k 個のスコア語 (パック値の上位語)、G * k 個の下位語、続いて k 個の int キー。それ以外は G * k 個の
    // パック値、続いて k 個の int キー。どちらも 8 G k + 4 k バイト (G 4 で 73,728、GB10 の 1 ブロック 101,376 以下)。
    extern __shared__ long long row[];
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ unsigned s_prefix32;
    __shared__ int s_need, s_at, s_eq, s_tot;
    __shared__ int warp_sums[32];
    const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x, m = G * k;
    int* keys = reinterpret_cast<int*>(row + m);
    unsigned* hi = reinterpret_cast<unsigned*>(row);
    unsigned* lo = hi + m;
    // k a power of two (the sort below): i / k and i % k without a division
    // k は 2 のべき (下のソートのため): i / k と i % k を除算なしで
    const int ks = 31 - __clz(k);
    for (int i = tid; i < m; i += nt) {
        const long long v = cand[((long long)(i >> ks) * n + r) * k + (i & (k - 1))];
        if (hiword) {
            hi[i] = (unsigned)(((unsigned long long)v ^ 0x8000000000000000ULL) >> 32);
            lo[i] = (unsigned)(v & 0xFFFFFFFFLL);
        } else {
            row[i] = v;
        }
    }
    if (tid == 0) { s_at = 0; s_eq = 0; }
    __syncthreads();
    if (hiword) {
        unsigned th;
        int need;
        tf_glm53_radix_kth32(hi, m, k, &s_prefix32, &s_need, hist, &th, &need);
        int eq_seen = 0;
        for (int c = 0; c < m; c += nt) {
            const int i = c + tid;
            const int eq = i < m && hi[i] == th;
            const int eb = tf_glm53_block_scan(eq, warp_sums, &s_tot);
            const int eqn = s_tot;
            if (i < m && (hi[i] > th || (eq && eq_seen + eb < need))) {
                const long long g = 0x7FFFFFFFLL - (long long)lo[i];
                int slot;
                if (member < 0) keys[atomicAdd(&s_at, 1)] = (int)g;
                else if (tf_glm53_own_slot(g, G, member, &slot)) keys[atomicAdd(&s_at, 1)] = slot;
            }
            eq_seen += eqn;
        }
    } else {
        unsigned long long th;
        int need;
        tf_glm53_radix_kth(row, m, k, &s_prefix, &s_need, hist, &th, &need);
        for (int i = tid; i < m; i += nt) {
            const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ULL;
            bool take = u > th;
            if (u == th) take = atomicAdd(&s_eq, 1) < need;
            if (!take) continue;
            const long long g = 0x7FFFFFFFLL - (row[i] & 0xFFFFFFFFLL);
            int slot;
            // the owner step of the sequence-parallel select: every global pick
            // sequence-parallel 選択の owner の手順: グローバルな選択をすべて
            if (member < 0) keys[atomicAdd(&s_at, 1)] = (int)g;
            else if (tf_glm53_own_slot(g, G, member, &slot)) keys[atomicAdd(&s_at, 1)] = slot;
        }
    }
    __syncthreads();
    const int own = s_at;
    // past the own ones: padding that sorts last (never read)
    // 自分の分より後ろ: 最後に並ぶ埋め草 (読まれない)
    for (int i = own + tid; i < k; i += nt) keys[i] = 0x7FFFFFFF;
    __syncthreads();
    if (k == 2 * nt) {
        // the same bitonic network (ascending, k a power of two), two keys a thread (elements 2t, 2t + 1): strides
        // of 32 and less inside the warp (shuffles, registers), the wider ones through shared memory, for fewer block
        // barriers. The keys are distinct below the 0x7FFFFFFF padding, so any correct sort gives the same output.
        // 同じ bitonic ネットワーク (昇順、k は 2 のべき) をスレッドあたり 2 キー (要素 2t、2t + 1) で: stride 32
        // 以下は warp 内 (shuffle とレジスタ)、それより広いものは共有メモリで行い、ブロックのバリアを減らす。
        // 0x7FFFFFFF の埋め草より下のキーは互いに異なるので、正しいソートならどれでも同じ出力になる。
        const int lane = tid & 31;
        int v0 = keys[2 * tid], v1 = keys[2 * tid + 1];
        for (int size = 2; size <= k; size <<= 1) {
            if ((size >> 1) >= 64) {
                __syncthreads();
                keys[2 * tid] = v0;
                keys[2 * tid + 1] = v1;
                __syncthreads();
                for (int stride = size >> 1; stride >= 64; stride >>= 1) {
                    for (int i = tid; i < k; i += nt) {
                        const int j = i ^ stride;
                        if (j > i) {
                            const bool asc = (i & size) == 0;
                            const int a = keys[i], b = keys[j];
                            if (asc ? (a > b) : (a < b)) { keys[i] = b; keys[j] = a; }
                        }
                    }
                    __syncthreads();
                }
                v0 = keys[2 * tid];
                v1 = keys[2 * tid + 1];
            }
            // the same for 2t and 2t + 1 (size >= 2)
            // 2t と 2t + 1 で同じ (size >= 2)
            const bool asc = ((2 * tid) & size) == 0;
            for (int stride = min(size >> 1, 32); stride >= 2; stride >>= 1) {
                // element 2t + j pairs with 2 (t ^ m) + j: lane ^ m
                // 要素 2t + j は 2 (t ^ m) + j と組む: lane ^ m
                const int m = stride >> 1;
                const bool lower = (lane & m) == 0;
                const int b0 = __shfl_xor_sync(0xFFFFFFFFu, v0, m), b1 = __shfl_xor_sync(0xFFFFFFFFu, v1, m);
                const bool keep_min = asc == lower;
                v0 = keep_min ? min(v0, b0) : max(v0, b0);
                v1 = keep_min ? min(v1, b1) : max(v1, b1);
            }
            if (asc ? (v0 > v1) : (v0 < v1)) { const int t = v0; v0 = v1; v1 = t; }   // stride 1 / stride 1 の段
        }
        // past the own ones: 0; the owner step (member < 0) fills all k
        // 自分の分より後ろは 0。owner の手順 (member < 0) では k 個すべてが埋まる
        tok[(long long)r * k + 2 * tid] = 2 * tid < own ? v0 : 0;
        tok[(long long)r * k + 2 * tid + 1] = 2 * tid + 1 < own ? v1 : 0;
        if (tid == 0 && cnt) cnt[r] = own;
        return;
    }
    for (int size = 2; size <= k; size <<= 1) {   // ascending (k a power of two) / 昇順 (k は 2 のべき)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < k; i += nt) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool asc = (i & size) == 0;
                    const int a = keys[i], b = keys[j];
                    if (asc ? (a > b) : (a < b)) { keys[i] = b; keys[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    // past the own ones: 0; the owner step (member < 0) fills all k
    // 自分の分より後ろは 0。owner の手順 (member < 0) では k 個すべてが埋まる
    for (int i = tid; i < k; i += nt) tok[(long long)r * k + i] = i < own ? keys[i] : 0;
    if (tid == 0 && cnt) cnt[r] = own;
}

// The sequence-parallel DCP select's last step: each row's k global picks (ascending, gathered from their owners) ->
// this member's own (gpos % G == member) as local slots gpos / G, in place, still ascending (tok [n, k]); their count
// into cnt[r]. One warp a row: a ballot compacts each 32 in order (writes never pass the reads still to come).
// sequence-parallel の DCP 選択の最後の手順: 各行の k 個のグローバルな選択 (昇順、各 owner から集めたもの) から、
// このメンバーの分 (gpos % G == member) をローカルスロット gpos / G として、その場で昇順のまま詰める (tok [n, k])。
// その数は cnt[r] へ。1 行 1 warp: ballot で 32 個ずつ順に詰める (書き込みがこれから読む位置を追い越すことはない)。
extern "C" __global__ void tf_glm53_dcp_own(int* __restrict__ tok, int* __restrict__ cnt, int n, int k, int G,
                                            int member) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= n) return;
    int* row = tok + (long long)r * k;
    int at = 0;
    // the pair of members: the same loop with the divisor a constant (shifts, no division)
    // メンバーがペアのとき: 除数を定数にした同じループ (シフトで、除算なし)
    if (G == 2) {
        for (int i0 = 0; i0 < k; i0 += 32) {
            const int i = i0 + lane;
            const int g = i < k ? row[i] : 0;
            const bool own = i < k && g % 2 == member;
            const unsigned m = __ballot_sync(0xFFFFFFFFu, own);
            __syncwarp();
            if (own) row[at + __popc(m & ((1u << lane) - 1u))] = g / 2;
            at += __popc(m);
            __syncwarp();
        }
    } else {
        for (int i0 = 0; i0 < k; i0 += 32) {
            const int i = i0 + lane;
            const int g = i < k ? row[i] : 0;
            const bool own = i < k && g % G == member;
            const unsigned m = __ballot_sync(0xFFFFFFFFu, own);
            __syncwarp();
            if (own) row[at + __popc(m & ((1u << lane) - 1u))] = g / G;
            at += __popc(m);
            __syncwarp();
        }
    }
    for (int i = at + lane; i < k; i += 32) row[i] = 0;          // past the own ones / 自分の分より後ろ
    if (lane == 0) cnt[r] = at;
}

// x = float(bf16(x)) in place, round to nearest even: an fp32 partial holding the prompt experts' bf16 output.
// x = float(bf16(x)) をその場で (最近接偶数丸め): prompt 用 expert の bf16 出力を fp32 の部分和に入れた値になる。
extern "C" __global__ void tf_glm53_round_bf16(float* __restrict__ x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    {
        unsigned u = __float_as_uint(x[i]);
        // __float2bfloat16_rn (NaN / inf kept)
        // __float2bfloat16_rn と同じ (NaN / inf はそのまま)
        if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
        x[i] = __uint_as_float(u & 0xFFFF0000u);
    }
}

// dst[:n] = bf16(src[:n]) from fp32, round to nearest even.
// fp32 から dst[:n] = bf16(src[:n])、最近接偶数丸め。
extern "C" __global__ void tf_glm53_to_bf16(const float* __restrict__ src, unsigned short* __restrict__ dst, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        unsigned u = __float_as_uint(src[i]);
        if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
        // NaN stays a (quiet) NaN
        // NaN は (quiet) NaN のまま
        else if (u & 0x007FFFFFu) u |= 0x00400000u;
        dst[i] = (unsigned short)(u >> 16);
    }
}

// dst[:n] = bf16(a[:n] + x[:n]): tf_glm53_add_f32 (a += x, one IEEE add) and tf_glm53_to_bf16 of the sum in one pass,
// for a decode window's expert partial + shared expert on its way to the bf16 reduce.
// dst[:n] = bf16(a[:n] + x[:n]): tf_glm53_add_f32 (a += x、IEEE 加算 1 回) と和の tf_glm53_to_bf16 を 1 パスで行う。
// decode ウィンドウの expert 部分和 + shared expert を bf16 の reduce へ渡すときに使う。
extern "C" __global__ void tf_glm53_add_to_bf16(const float* __restrict__ a, const float* __restrict__ x,
                                                unsigned short* __restrict__ dst, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        unsigned u = __float_as_uint(a[i] + x[i]);
        if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
        // NaN stays a (quiet) NaN
        // NaN は (quiet) NaN のまま
        else if (u & 0x007FFFFFu) u |= 0x00400000u;
        dst[i] = (unsigned short)(u >> 16);
    }
}

// dst[:n] = bf16(r(a[:n]) + x[:n]) with r = tf_glm53_round_bf16 when `round`, else the identity: a sequence-parallel
// part's routed-expert sum (fp32, rounded through bf16 as prompt_experts hands it over) plus the shared expert, on its
// way to the bf16 reduce-scatter; tf_glm53_round_bf16, tf_glm53_add_f32 and tf_glm53_to_bf16 in one pass over [n].
// It stays out of the down projection's gemm, which it would push past 128 registers a thread.
// dst[:n] = bf16(r(a[:n]) + x[:n])。r は `round` のとき tf_glm53_round_bf16、そうでなければ恒等: sequence-parallel
// の part の routed expert の和 (fp32。prompt_experts の渡し方に合わせて bf16 を通して丸める) に shared expert を
// 足し、bf16 の reduce-scatter へ渡す。tf_glm53_round_bf16・tf_glm53_add_f32・tf_glm53_to_bf16 を [n] の 1 パスで
// 行う。down projection の gemm に入れないのは、そのカーネルがスレッドあたり 128 レジスタを超えるから。
// round 2: a holds fp16 values (the prompt experts' fp16 sums, TF_GLM53_PE_F16), widened exactly, then as round 1.
// round 2: a は fp16 の値 (prompt expert の fp16 の和、TF_GLM53_PE_F16)。誤差なしに広げてから round 1 と同じ。
extern "C" __global__ void tf_glm53_radd_to_bf16(const float* __restrict__ a, const float* __restrict__ x,
                                                 unsigned short* __restrict__ dst, long long n, int round) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        unsigned p;
        if (round == 2) {
            float f;
            asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(reinterpret_cast<const unsigned short*>(a)[i]));
            p = __float_as_uint(f);
        } else p = __float_as_uint(a[i]);
        if (round) {
            // as tf_glm53_round_bf16
            // tf_glm53_round_bf16 と同じ
            if ((p & 0x7F800000u) != 0x7F800000u) p += 0x7FFFu + ((p >> 16) & 1u);
            p &= 0xFFFF0000u;
        }
        unsigned u = __float_as_uint(__uint_as_float(p) + x[i]);
        if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
        // NaN stays a (quiet) NaN
        // NaN は (quiet) NaN のまま
        else if (u & 0x007FFFFFu) u |= 0x00400000u;
        dst[i] = (unsigned short)(u >> 16);
    }
}

// dst[:n] = float(src[:n]) from bf16 (exact).
// bf16 から dst[:n] = float(src[:n]) (誤差なし)。
extern "C" __global__ void tf_glm53_from_bf16(const unsigned short* __restrict__ src, float* __restrict__ dst, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        dst[i] = __uint_as_float((unsigned)src[i] << 16);
}

// ---- the decode windows' top-k at long context (the exact set of the k largest packed int64 keys, a row spread over
// many blocks): two 8-bit histogram passes over the whole row fix the top 16 bits of the k-th key (tk_hist /
// tk_digit), one pass writes the keys above it to out and gathers the ones equal to it (tk_split), then one block a
// row picks the `need` best of those exactly (tk_finish). State a row: hist[256], prefix16, need, n_out, n_cand.
// ---- 長いコンテキストでの decode ウィンドウの top-k (パック済み int64 キーの上位 k 個の厳密な集合。1 行を多数の
// ブロックで処理): 行全体への 8 ビットのヒストグラム 2 パスで k 番目のキーの上位 16 ビットを決め (tk_hist /
// tk_digit)、1 パスでそれより大きいキーを out に書き、等しいものを集める (tk_split)。最後に 1 行 1 ブロックで、
// その中から上位 `need` 個を厳密に選ぶ (tk_finish)。行ごとの状態: hist[256]、prefix16、need、n_out、n_cand。
struct TkState {
    unsigned int hist[256];
    // the top bits fixed so far (pass 0: bits 63..56, pass 1: 63..48), in the flipped order
    // ここまでに決まった上位ビット (パス 0: 63..56、パス 1: 63..48)。符号ビットを反転した順
    unsigned int prefix;
    int need;              // keys still to take among those matching prefix / prefix に一致するキーのうちまだ取る数
    int n_out;
    int n_cand;
};

__device__ __forceinline__ unsigned long long tk_flip(long long v) { return (unsigned long long)v ^ 0x8000000000000000ULL; }

extern "C" __global__ void tf_glm53_tk_hist(const long long* __restrict__ sc, int T, TkState* __restrict__ st, int pass) {
    __shared__ unsigned int h[256];
    const int r = blockIdx.y;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const long long* row = sc + (long long)r * T;
    const unsigned int pre = st[r].prefix;
    const int shift = 56 - 8 * pass;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T; i += gridDim.x * blockDim.x) {
        const unsigned long long u = tk_flip(row[i]);
        if (pass == 0 || (unsigned int)(u >> 56) == pre) atomicAdd(&h[(u >> shift) & 255], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (h[i]) atomicAdd(&st[r].hist[i], h[i]);
}

// One block a row: the digit holding the need-th best key; prefix and need updated, the histogram cleared.
// 1 行 1 ブロック: need 番目のキーを含む桁を求め、prefix と need を更新し、ヒストグラムを消す。
extern "C" __global__ void tf_glm53_tk_digit(TkState* __restrict__ st, int T, int k, int pass) {
    TkState& s = st[blockIdx.x];
    __shared__ unsigned int h[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = s.hist[i];
    __syncthreads();
    if (threadIdx.x < 32) {
        const int need = pass == 0 ? min(k, T) : s.need;
        int d, left;
        tf_glm53_digit_pick(h, need, &d, &left);
        if (threadIdx.x == 0) {
        s.need = left;
        s.prefix = pass == 0 ? (unsigned)d : ((s.prefix << 8) | (unsigned)d);
        if (pass == 1) { s.n_out = 0; s.n_cand = 0; }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s.hist[i] = 0;
}

// Keys above the 16-bit prefix to out (any order), keys equal to it to cand.
// 16 ビットの prefix より大きいキーは out へ (順不同)、等しいキーは cand へ。
extern "C" __global__ void tf_glm53_tk_split(const long long* __restrict__ sc, int T, TkState* __restrict__ st,
                                             long long* __restrict__ out, long long* __restrict__ cand, int k) {
    const int r = blockIdx.y;
    const long long* row = sc + (long long)r * T;
    const unsigned int p16 = st[r].prefix;
    const int lane = threadIdx.x & 31;
    const unsigned lt = (1u << lane) - 1u;
    // whole warps iterate together (warp-aggregated atomics: one add a warp per kind)
    // warp 単位でそろって回る (warp で集約した atomic: 種類ごとに warp あたり加算 1 回)
    for (int i0 = blockIdx.x * blockDim.x; i0 < T; i0 += gridDim.x * blockDim.x) {
        const int i = i0 + threadIdx.x;
        const long long v = i < T ? row[i] : 0;
        const unsigned int top = i < T ? (unsigned int)(tk_flip(v) >> 48) : 0u;
        const bool up = i < T && top > p16, eq = i < T && top == p16;
        const unsigned mu = __ballot_sync(0xFFFFFFFFu, up), me = __ballot_sync(0xFFFFFFFFu, eq);
        int bu = 0, be = 0;
        if (lane == 0) {
            if (mu) bu = atomicAdd(&st[r].n_out, __popc(mu));
            if (me) be = atomicAdd(&st[r].n_cand, __popc(me));
        }
        bu = __shfl_sync(0xFFFFFFFFu, bu, 0);
        be = __shfl_sync(0xFFFFFFFFu, be, 0);
        if (up) out[(long long)r * k + bu + __popc(mu & lt)] = v;
        if (eq) cand[(long long)r * T + be + __popc(me & lt)] = v;
    }
}

// One block a row: the `need` best of the gathered candidates (exact), after the n_out keys already out; the slots past
// min(k, T) are filled with -9223372036854775807.
// 1 行 1 ブロック: 集めた候補から上位 `need` 個を厳密に選び、すでに出した n_out 個の後ろに書く。min(k, T) より後ろの
// 枠は -9223372036854775807 で埋める。
extern "C" __global__ void tf_glm53_tk_finish(const long long* __restrict__ cand, int T, TkState* __restrict__ st,
                                              long long* __restrict__ out, int k) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_need, s_at;
    const int r = blockIdx.x;
    const int n = st[r].n_cand, need0 = st[r].need, base = st[r].n_out;
    const long long* row = cand + (long long)r * T;
    long long* o = out + (long long)r * k;
    unsigned long long th;
    int need;
    // every candidate carries the 16 bits tk_digit fixed: six passes, not eight
    // どの候補も tk_digit が決めた 16 ビットを持つので、8 パスでなく 6 パス
    tf_glm53_radix_kth_from(row, n, need0, (unsigned long long)st[r].prefix << 48, 2, &s_prefix, &s_need, hist, &th, &need);
    if (threadIdx.x == 0) s_at = base;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (tk_flip(row[i]) > th) o[atomicAdd(&s_at, 1)] = row[i];
    __syncthreads();
    const int lim = base + need0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (tk_flip(row[i]) == th) {
            const int at = atomicAdd(&s_at, 1);
            if (at < lim) o[at] = row[i];
        }
    __syncthreads();
    for (int i = min(k, T) + threadIdx.x; i < k; i += blockDim.x) o[i] = -9223372036854775807LL;
}

// ---- the slot cache: KV rows between the compute cache and the store, and their row hash H, bit for bit
// slotcache/hashing.zig's. A "side" is a device table of L lane base pointers (every layer's latent rows, then every
// full-indexer layer's index-key rows) with words[l] 8-byte words a row.
// ---- スロットキャッシュ: 計算キャッシュとストアの間の KV の行と、その行ハッシュ H (slotcache/hashing.zig とビット
// 一致)。"side" は L 個の lane の先頭ポインタのデバイス上の表 (全層の latent の行、続いて full indexer の層の
// index-key の行) で、1 行は words[l] 個の 8 バイト語。
#define TF_SLOT_ROWS 128
#define TF_SLOT_CHUNK 32
__device__ __forceinline__ unsigned long long tf_slot_mix64(unsigned long long x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// lanes[lane * n + i] = lane hash of row r0 + i of lane blockIdx.y; with dst, row r0 + i is also copied to dst row
// d0 + i (the bytes hashed are the bytes written).
// lanes[lane * n + i] = lane blockIdx.y の行 r0 + i の lane ハッシュ。dst があれば行 r0 + i を dst の行 d0 + i にも
// コピーする (ハッシュしたバイトがそのまま書いたバイト)。
extern "C" __global__ void __launch_bounds__(TF_SLOT_ROWS) tf_glm53_slot_lanes(
    const long long* __restrict__ src, const long long* __restrict__ dst, const int* __restrict__ words, long long r0,
    long long d0, int n, unsigned long long* __restrict__ lanes) {
    __shared__ unsigned long long sh[TF_SLOT_ROWS][TF_SLOT_CHUNK + 1];
    const int lane = blockIdx.y, t = threadIdx.x, nw = words[lane];
    const int i0 = blockIdx.x * TF_SLOT_ROWS, rows = min(TF_SLOT_ROWS, n - i0);
    const unsigned long long* s = reinterpret_cast<const unsigned long long*>(src[lane]) + (r0 + i0) * nw;
    unsigned long long* d = dst != nullptr ? reinterpret_cast<unsigned long long*>(dst[lane]) + (d0 + i0) * nw : nullptr;
    unsigned long long h = tf_slot_mix64(static_cast<unsigned long long>(lane + 1) * 0xBB67AE8584CAA73Bull);
    for (int j0 = 0; j0 < nw; j0 += TF_SLOT_CHUNK) {
        const int k = min(TF_SLOT_CHUNK, nw - j0), pairs = k >> 1;
        for (int q = t; q < rows * pairs; q += TF_SLOT_ROWS) {
            const int r = q / pairs, c = (q - r * pairs) << 1;
            const long long off = static_cast<long long>(r) * nw + j0 + c;
            const ulonglong2 v = *reinterpret_cast<const ulonglong2*>(s + off);
            if (d != nullptr) *reinterpret_cast<ulonglong2*>(d + off) = v;
            sh[r][c] = v.x;
            sh[r][c + 1] = v.y;
        }
        __syncthreads();
        if (t < rows) {
#pragma unroll 4
            for (int c = 0; c < k; ++c)
                h = tf_slot_mix64(h ^ sh[t][c] ^ static_cast<unsigned long long>(j0 + c + 1) * 0xD6E8FEB86659FD93ull);
        }
        __syncthreads();
    }
    if (t < rows) lanes[static_cast<long long>(lane) * n + i0 + t] = h;
}

// H of position p0 + i * step (prefix identity pid[i], generation gen[i]) from its L lane hashes: into out[i], or,
// with exp, compared with exp[i] and the mismatches counted into bad[0].
// 位置 p0 + i * step (prefix の識別子 pid[i]、世代 gen[i]) の H を L 個の lane ハッシュから求めて out[i] へ。exp が
// あれば exp[i] と比べ、食い違いの数を bad[0] に足す。
extern "C" __global__ void tf_glm53_slot_fold(const unsigned long long* __restrict__ lanes, int L, int n, long long p0,
                                              long long step, const long long* __restrict__ pid,
                                              const long long* __restrict__ gen, long long* __restrict__ out,
                                              const long long* __restrict__ exp, unsigned long long* __restrict__ bad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long p = static_cast<unsigned long long>(p0 + i * step);
    unsigned long long acc = tf_slot_mix64(0x6A09E667F3BCC908ull ^ tf_slot_mix64(p) ^
                                           tf_slot_mix64(static_cast<unsigned long long>(pid[i]) ^ 0xA54FF53A5F1D36F1ull) ^
                                           tf_slot_mix64(static_cast<unsigned long long>(gen[i]) ^ 0x510E527FADE682D1ull));
    for (int l = 0; l < L; ++l)
        acc = tf_slot_mix64(acc ^ lanes[static_cast<long long>(l) * n + i] ^
                            static_cast<unsigned long long>(l + 1) * 0x3C6EF372FE94F82Bull);
    if (exp != nullptr) {
        if (static_cast<long long>(acc) != exp[i]) atomicAdd(bad, 1ull);
    } else out[i] = static_cast<long long>(acc);
}

// Rows [r0, r0 + n) of lane blockIdx.y to dst rows [d0, d0 + n): one contiguous run a lane.
// lane blockIdx.y の行 [r0, r0 + n) を dst の行 [d0, d0 + n) へ: lane ごとに連続した範囲 1 つ。
extern "C" __global__ void tf_glm53_slot_copy(const long long* __restrict__ src, const long long* __restrict__ dst,
                                              const int* __restrict__ words, long long r0, long long d0, long long n) {
    const int lane = blockIdx.y;
    const long long nw = words[lane], total = n * nw / 2;
    const ulonglong2* s = reinterpret_cast<const ulonglong2*>(reinterpret_cast<const unsigned long long*>(src[lane]) + r0 * nw);
    ulonglong2* d = reinterpret_cast<ulonglong2*>(reinterpret_cast<unsigned long long*>(dst[lane]) + d0 * nw);
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; q < total; q += step) d[q] = s[q];
}

// p[0 .. n) = 0 (int64): a released extent's expected values.
// p[0 .. n) = 0 (int64): 解放した extent の期待値。
extern "C" __global__ void tf_glm53_slot_zero(long long* __restrict__ p, long long n) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x)
        p[i] = 0;
}
