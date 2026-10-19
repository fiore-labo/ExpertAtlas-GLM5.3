// The GLM-5.3 engine's kernels, rewritten in plain CUDA from the Triton kernels of TensorFold's Python engine for full
// GLM-5.3. C style: POD argument structs by value, free functions, no classes or templates. Each kernel does its
// original's IEEE operations in the same order (--fmad=false --ftz=false), so its output bytes are the original's.
// GLM-5.3 エンジンのカーネル。TensorFold の Python 版フル GLM-5.3 エンジンの Triton カーネルを素の
// CUDA に書き直したもの。C の書き方: 引数は POD 構造体の値渡し、自由関数だけで、クラスもテンプレートも使わない。
// 各カーネルは元と同じ IEEE 演算を同じ順に行う (--fmad=false --ftz=false) ので、出力のバイトは元と一致する。
#include <cstdint>
#include <cuda_fp16.h>
#include <math.h>

typedef unsigned long long Dev;

// bf16 <-> fp32 as Triton casts: widening is exact, narrowing rounds to nearest even (cvt.rn.bf16.f32).
// Triton のキャストと同じ bf16 <-> fp32 変換: 拡大は厳密、縮小は最近接偶数への丸め (cvt.rn.bf16.f32)。
static __device__ __forceinline__ float bf16_f32(uint16_t v) { return __uint_as_float((unsigned)v << 16); }
static __device__ __forceinline__ uint16_t f32_bf16(float f) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}
// two values into one word, the first in the low half
// 2 つの値を 1 ワードに詰める。1 つ目を下位の半分に置く
static __device__ __forceinline__ unsigned f32_bf16x2(float lo, float hi) {
    unsigned r;
    asm("{ .reg .b16 a, b; cvt.rn.bf16.f32 a, %1; cvt.rn.bf16.f32 b, %2; mov.b32 %0, {a, b}; }" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

// Triton's math as its compiled kernels do it: exp(x) = ex2.approx(x log2(e)), sqrt = sqrt.approx.ftz,
// a / b = div.full, x / constant = x * (the constant's fp32 reciprocal). cos / sin are libdevice's (cosf / sinf).
// Triton のコンパイル済みカーネルと同じ計算: exp(x) = ex2.approx(x log2(e))、sqrt = sqrt.approx.ftz、a / b = div.full、
// x / 定数 = x * (定数の fp32 の逆数)。cos / sin は libdevice のもの (cosf / sinf)。
static __device__ __forceinline__ float t_exp(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950216293334961f));
    return r;
}
static __device__ __forceinline__ float t_div(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static __device__ __forceinline__ float t_sqrt(float x) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// 4 / 8 bf16 at p (8- / 16-byte aligned) as fp32, in one wide load (the same values as scalar loads)
// p にある 4 / 8 個の bf16 (8 / 16 バイト境界) を 1 回の幅広ロードで fp32 にする (値はスカラーロードと同じ)
static __device__ __forceinline__ void ld4_bf16v(const uint16_t* p, float* f) {
    const unsigned long long v = *(const unsigned long long*)p;
    f[0] = __uint_as_float((unsigned)v << 16); f[1] = __uint_as_float((unsigned)v & 0xffff0000u);
    f[2] = __uint_as_float((unsigned)(v >> 32) << 16); f[3] = __uint_as_float((unsigned)(v >> 32) & 0xffff0000u);
}
static __device__ __forceinline__ void ld8_bf16v(const uint16_t* p, float* f) {
    const uint4 v = *(const uint4*)p;
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}

// tl.sum over a block's threads as Triton reduces it: each warp by butterfly shuffles (16, 8, 4, 2, 1), then the warps'
// sums through shared memory by one more butterfly over the first `warps` lanes (warps / 2 .. 1). `v`: the thread's
// partial; returns the block's sum on every thread. `sh`: `warps` floats of shared memory.
// Triton と同じ順で、ブロックのスレッド全体の tl.sum を取る: 各 warp を butterfly shuffle (16, 8, 4, 2, 1) で足し、
// 続いて warp ごとの和を共有メモリ経由で、先頭 `warps` レーンのもう 1 段の butterfly (warps / 2 .. 1) で足す。
// `v` はスレッドの部分和で、ブロックの和を全スレッドに返す。`sh` は共有メモリの float `warps` 個。
// t_block_sum after its first butterfly level (`v`: the lane's value after the offset-16 exchange)
// 最初の butterfly 段より後の t_block_sum (`v` はオフセット 16 の交換を済ませたレーンの値)
static __device__ __forceinline__ float t_block_sum_tail(float v, float* sh, int warps);
static __device__ __forceinline__ float t_block_sum(float v, float* sh, int warps) {
    v = v + __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    return t_block_sum_tail(v, sh, warps);
}
static __device__ __forceinline__ float t_block_sum_tail(float v, float* sh, int warps) {
    for (int o = 8; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < warps ? sh[lane] : 0.0f;
        for (int o = warps >> 1; o > 0; o >>= 1) w = w + __shfl_xor_sync(0xFFFFFFFFu, w, o);
        if (lane == 0) sh[0] = w;
    }
    __syncthreads();
    const float total = sh[0];
    __syncthreads();   // all threads have read it before `sh` is reused / `sh` を再利用する前に全スレッドが読み終える
    return total;
}

// t_block_sum for a caller that never writes `sh` again: no closing barrier
// `sh` を二度と書かない呼び出し元用の t_block_sum。最後のバリアがない
static __device__ __forceinline__ float t_block_sum_last(float v, float* sh, int warps) {
    for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < warps ? sh[lane] : 0.0f;
        for (int o = warps >> 1; o > 0; o >>= 1) w = w + __shfl_xor_sync(0xFFFFFFFFu, w, o);
        if (lane == 0) sh[0] = w;
    }
    __syncthreads();
    return sh[0];
}

// tl.max over a block's threads: t_block_sum's butterflies with max (fmaxf).
// ブロックのスレッド全体の tl.max: t_block_sum の butterfly を max (fmaxf) で行う。
static __device__ __forceinline__ float t_block_max(float v, float* sh, int warps) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < warps ? sh[lane] : 0.0f;
        for (int o = warps >> 1; o > 0; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xFFFFFFFFu, w, o));
        if (lane == 0) sh[0] = w;
    }
    __syncthreads();
    const float total = sh[0];
    __syncthreads();   // all threads have read it before `sh` is reused / `sh` を再利用する前に全スレッドが読み終える
    return total;
}
// tl.exp2 (ex2.approx.ftz) and the fp32 -> e4m3 cast (cvt.rn.satfinite.e4m3x2.f32, one value)
// tl.exp2 (ex2.approx.ftz) と fp32 -> e4m3 のキャスト (cvt.rn.satfinite.e4m3x2.f32、1 値)
static __device__ __forceinline__ float t_exp2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
static __device__ __forceinline__ uint8_t f32_e4m3(float v) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(v), "f"(v));
    return (uint8_t)(r & 0xFF);
}
// the FP8 caches' power-of-two row scale: 2^(ceil(log2(max(amax, 1e-30))) - 8), with libdevice's log2
// FP8 キャッシュの行ごとの 2 のべき乗スケール: 2^(ceil(log2(max(amax, 1e-30))) - 8)。log2 は libdevice のもの
static __device__ __forceinline__ float t_kv8_scale(float amax) {
    return t_exp2(ceilf(log2f(fmaxf(amax, 1e-30f))) - 8.0f);
}
// TF_GLM53_KV=fp4 (the layout of MiaAI-Lab's fp4 cache): latent rows of 432 bytes - 512 e2m1 codes (two a byte, the
// even element low), an e4m3 scale for each 16 (byte 256 + block), the FP8 power-of-two row scale at byte 288, the
// bf16 rope at byte 304. Value = code x block scale x row scale. A code and a block scale are each an exact e4m3
// value and their product is exact in bf16, so the attention multiplies them where its tiles become MMA operands.
// TF_GLM53_KV=fp4 (MiaAI-Lab の fp4 キャッシュと同じ形): 432 バイトの latent 行 - e2m1 コード 512 個 (1 バイトに
// 2 つ、偶数番目が下位)、16 個ごとの e4m3 スケール (バイト 256 + ブロック)、バイト 288 に FP8 と同じ 2 のべき乗の
// 行スケール、バイト 304 に bf16 の rope。値 = コード x ブロックのスケール x 行スケール。コードもブロックの
// スケールも e4m3 でちょうど表せ、その積は bf16 でちょうど表せるので、attention はタイルを MMA の入力にする所で
// 両者を掛ける。
#define KV4_ROW 432
// TF_KV4P (family-specific targets, with the attention's transposed byte loads): each 16-byte piece of a row's codes is
// stored with its bytes in the order 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15, so that an ldmatrix row hands lane qq
// the bytes of codes 4 qq .. 4 qq + 3 and 16 + 4 qq .. + 3 of the piece - the attention keeps FP4 rows as they are
// in shared memory (432 bytes, by cp.async) and turns them into e4m3 where they become MMA operands. Writer and reader
// are this file's: the layout only lives in the cache.
// TF_KV4P (family-specific の target、attention の転置の 1 バイト読みと一緒): 行のコードの 16 バイトの各片を
// バイトの順 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15 で置く。ldmatrix の行がレーン qq に片のコード 4 qq .. 4 qq + 3 と
// 16 + 4 qq .. + 3 のバイトを渡すので、attention は FP4 の行をそのまま共有メモリに置き (432 バイト、cp.async)、MMA の
// 入力にする所で e4m3 にする。書く側も読む側もこのファイルで、この並びはキャッシュの中だけにある。
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
#define TF_KV4P 1
#define KV4P_ON 1
#else
#define KV4P_ON 0
#endif
static __device__ __forceinline__ unsigned kv4_nibble(float q) {   // nearest e2m1 of q (|q| <= 6) / q に最も近い e2m1
    const float a = fabsf(q);
    const unsigned i = a < 0.25f ? 0u : a < 0.75f ? 1u : a < 1.25f ? 2u : a < 1.75f ? 3u : a < 2.5f ? 4u : a < 3.5f ? 5u : a < 5.0f ? 6u : 7u;
    return (q < 0.0f && i ? 8u : 0u) | i;
}
// an e2m1 code as the e4m3 code of the same value (0, 0.5, 1, 1.5, 2, 3, 4, 6)
// e2m1 のコードを同じ値の e4m3 コードに (0、0.5、1、1.5、2、3、4、6)
static __device__ __forceinline__ unsigned kv4_e4m3(unsigned nib) {
    const unsigned lut = (nib & 7u) < 4u ? (0x3C383000u >> (8u * (nib & 3u))) & 0xFFu : (0x4C484440u >> (8u * (nib & 3u))) & 0xFFu;
    return lut | ((nib & 8u) << 4);
}
// four e2m1 codes (the low 16 bits of v, the first in the low nibble) as their e4m3 codes (the first in the low byte),
// kv4_e4m3 four at a time: the magnitudes pick bytes of the same 8-entry table, the signs 0x00 or 0x80
// e2m1 コード 4 つ (v の下位 16 ビット、先頭が下位のニブル) を e4m3 コードに (先頭が下位のバイト)。kv4_e4m3 を 4 つ
// まとめて: 絶対値は同じ 8 項の表からバイトを選び、符号は 0x00 か 0x80 を選ぶ
static __device__ __forceinline__ unsigned kv4_e4m3x4(unsigned v) {
    return __byte_perm(0x3C383000u, 0x4C484440u, v & 0x7777u) | __byte_perm(0u, 0x80808080u, (v >> 1) & 0x4444u);
}
static __device__ __forceinline__ float e4m3_f32(unsigned b) {
    unsigned short h;
    asm("{ .reg .b32 t; .reg .b16 s, hi; cvt.u16.u32 s, %1; cvt.rn.f16x2.e4m3x2 t, s; mov.b32 {%0, hi}, t; }" : "=h"(h) : "r"(b));
    return __half2float(__ushort_as_half(h));
}

extern "C" {

// RMSNorm: OUT[r] = bf16(W * bf16(X[r] * rinv)), rinv = 1 / sqrt(sum(x^2) / D + eps), rows of D bf16
// (`x_stride` / `o_stride` elements apart). A block a row, 128 threads up to D 2048, else 256; thread t holds elements
// [rep * 8 * threads + 8 t, + 8) of each repetition and adds their squares in that order
// (the first a product, the rest fused), then t_block_sum.
// xs != 0 (d a multiple of 64): XS[r, i] = the fp32 sum of y's 64-value group i; a thread adds its 8 values in order,
// the group's 8 threads by a butterfly (4, 2, 1).
// RMSNorm: OUT[r] = bf16(W * bf16(X[r] * rinv))、rinv = 1 / sqrt(sum(x^2) / D + eps)。行は D 個の bf16
// (間隔は `x_stride` / `o_stride` 要素)。1 行 1 ブロックで、D 2048 までは 128 スレッド、それより大きければ 256。
// スレッド t は各繰り返しの要素 [rep * 8 * threads + 8 t, + 8) を持ち、その順に二乗を足し
// (最初は積、残りは fma で融合)、最後に t_block_sum。
// xs != 0 (d は 64 の倍数): XS[r, i] = y の 64 値グループ i の fp32 和。スレッドは自分の 8 値を順に足し、グループの
// 8 スレッドは butterfly (4, 2, 1) で足す。
typedef struct { Dev x, w, out, xs; int rows, d, x_stride, o_stride, vec; float eps, rd; } RmsArgs;
__global__ void tf_glm53_n_rmsnorm(RmsArgs a) {
    __shared__ float sh[8];
    const uint16_t* x = (const uint16_t*)a.x + (long long)blockIdx.x * a.x_stride;
    const uint16_t* w = (const uint16_t*)a.w;
    uint16_t* out = (uint16_t*)a.out + (long long)blockIdx.x * a.o_stride;
    const int tile = 8 * blockDim.x;
    // 16-byte loads and stores, x read once (d a multiple of 8, at most 4 repetitions, pointers aligned)
    // 16 バイト単位でロードとストア、x は 1 回だけ読む (d は 8 の倍数、繰り返しは 4 回まで、ポインタはアライン済み)
    if (a.vec) {
        float xv[4][8], acc = 0.0f;
        bool first = true;
#pragma unroll
        for (int rep = 0; rep < 4; rep++) {
            const int base = 8 * threadIdx.x + rep * tile;
            if (base < a.d) {
                ld8_bf16v(x + base, xv[rep]);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc = first ? xv[rep][j] * xv[rep][j] : fmaf(xv[rep][j], xv[rep][j], acc);
                    first = false;
                }
            }
        }
        const float sum = t_block_sum_last(acc, sh, blockDim.x >> 5);
        const float rinv = t_div(1.0f, t_sqrt(fmaf(sum, a.rd, a.eps)));
#pragma unroll
        for (int rep = 0; rep < 4; rep++) {
            const int base = 8 * threadIdx.x + rep * tile;
            if (base < a.d) {
                float wv[8];
                unsigned yp[4];
                ld8_bf16v(w + base, wv);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned xp = f32_bf16x2(xv[rep][2 * j] * rinv, xv[rep][2 * j + 1] * rinv);
                    yp[j] = f32_bf16x2(wv[2 * j] * __uint_as_float(xp << 16), wv[2 * j + 1] * __uint_as_float(xp & 0xffff0000u));
                }
                *(uint4*)(out + base) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
                if (a.xs) {
                    float s = __uint_as_float(yp[0] << 16) + __uint_as_float(yp[0] & 0xffff0000u);
#pragma unroll
                    for (int j = 1; j < 4; j++) s = (s + __uint_as_float(yp[j] << 16)) + __uint_as_float(yp[j] & 0xffff0000u);
                    s = s + __shfl_xor_sync(0xffffffffu, s, 4);
                    s = s + __shfl_xor_sync(0xffffffffu, s, 2);
                    s = s + __shfl_xor_sync(0xffffffffu, s, 1);
                    if ((threadIdx.x & 7) == 0) ((float*)a.xs)[(long long)blockIdx.x * (a.d / 64) + base / 64] = s;
                }
            }
        }
        return;
    }
    float acc = 0.0f;
    bool first = true;
    for (int base = 8 * threadIdx.x; base < a.d; base += tile)
        for (int j = 0; j < 8; j++) {
            const float v = base + j < a.d ? bf16_f32(x[base + j]) : 0.0f;
            acc = first ? v * v : fmaf(v, v, acc);
            first = false;
        }
    const float sum = t_block_sum(acc, sh, blockDim.x >> 5);
    const float rinv = t_div(1.0f, t_sqrt(fmaf(sum, a.rd, a.eps)));
    if (a.xs == 0) {
        for (int base = 8 * threadIdx.x; base < a.d; base += tile)
            for (int j = 0; j < 8 && base + j < a.d; j++) {
                const float y = bf16_f32(f32_bf16(bf16_f32(x[base + j]) * rinv));
                out[base + j] = f32_bf16(bf16_f32(w[base + j]) * y);
            }
        return;
    }
    float* xs = (float*)a.xs + (long long)blockIdx.x * (a.d / 64);
    // d is a multiple of 64 * the block's warps here
    // ここでは d は 64 * (ブロックの warp 数) の倍数
    for (int base = 8 * threadIdx.x; base < a.d; base += tile) {
        float s = 0.0f;
        for (int j = 0; j < 8; j++) {
            const float y = bf16_f32(f32_bf16(bf16_f32(x[base + j]) * rinv));
            const uint16_t yo = f32_bf16(bf16_f32(w[base + j]) * y);
            out[base + j] = yo;
            s = j == 0 ? bf16_f32(yo) : s + bf16_f32(yo);
        }
        s = s + __shfl_xor_sync(0xffffffffu, s, 4);
        s = s + __shfl_xor_sync(0xffffffffu, s, 2);
        s = s + __shfl_xor_sync(0xffffffffu, s, 1);
        if ((threadIdx.x & 7) == 0) xs[base / 64] = s;
    }
}

// tf_glm53_n_rmsnorm's 16-byte path for one shape (d = 256 WARPS REPS, no sums) as an entry of its own, the same
// arithmetic: with the warp count and the repetitions constant, the cross-warp sum and the steps unroll into
// straight-line code.
// tf_glm53_n_rmsnorm の 16 バイト経路を 1 つの形状 (d = 256 WARPS REPS、和なし) 専用の入口にしたもの。計算は同じ。warp
// 数と繰り返し数が定数なので、warp をまたぐ和と各段が直線のコードに展開される。
static __device__ __forceinline__ void rms_fixed(RmsArgs a, float* sh, const int WARPS, const int REPS) {
    const uint16_t* x = (const uint16_t*)a.x + (long long)blockIdx.x * a.x_stride;
    const uint16_t* w = (const uint16_t*)a.w;
    uint16_t* out = (uint16_t*)a.out + (long long)blockIdx.x * a.o_stride;
    const int tile = 256 * WARPS;
    float xv[4][8], acc = 0.0f;
#pragma unroll
    for (int rep = 0; rep < REPS; rep++) {
        ld8_bf16v(x + 8 * threadIdx.x + rep * tile, xv[rep]);
#pragma unroll
        for (int j = 0; j < 8; j++) acc = (rep == 0 && j == 0) ? xv[rep][j] * xv[rep][j] : fmaf(xv[rep][j], xv[rep][j], acc);
    }
    const float sum = t_block_sum_last(acc, sh, WARPS);
    const float rinv = t_div(1.0f, t_sqrt(fmaf(sum, a.rd, a.eps)));
#pragma unroll
    for (int rep = 0; rep < REPS; rep++) {
        float wv[8];
        unsigned yp[4];
        ld8_bf16v(w + 8 * threadIdx.x + rep * tile, wv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned xp = f32_bf16x2(xv[rep][2 * j] * rinv, xv[rep][2 * j + 1] * rinv);
            yp[j] = f32_bf16x2(wv[2 * j] * __uint_as_float(xp << 16), wv[2 * j + 1] * __uint_as_float(xp & 0xffff0000u));
        }
        *(uint4*)(out + 8 * threadIdx.x + rep * tile) = make_uint4(yp[0], yp[1], yp[2], yp[3]);
    }
}
__global__ void tf_glm53_n_rmsnorm_d6144(RmsArgs a) {   // 256 threads, 3 repetitions / 256 スレッド、繰り返し 3 回
    __shared__ float sh[8];
    rms_fixed(a, sh, 8, 3);
}
__global__ void tf_glm53_n_rmsnorm_d2048(RmsArgs a) {   // 128 threads, 2 repetitions / 128 スレッド、繰り返し 2 回
    __shared__ float sh[8];
    rms_fixed(a, sh, 4, 2);
}

// OUT[i] = PART[0][i] + PART[1][i] + ... + PART[ks - 1][i]: the router's slice partials added in slice order (fp32).
// grid (ceil(total / 1024)); a block owns 1024 elements.
// OUT[i] = PART[0][i] + PART[1][i] + ... + PART[ks - 1][i]: ルーターのスライスごとの部分和をスライス順に fp32 で足す。
// grid (ceil(total / 1024))。1 ブロックが 1024 要素を受け持つ。
typedef struct { Dev part, out; int total, ks, vec; } RouterSumArgs;
__global__ void tf_glm53_n_router_sum(RouterSumArgs a) {
    const float* part = (const float*)a.part;
    float* out = (float*)a.out;
    const int end = min(a.total, ((int)blockIdx.x + 1) * 1024);
    // total a multiple of 4, pointers 16-byte aligned: 4 values a thread
    // total は 4 の倍数、ポインタは 16 バイト境界: 1 スレッド 4 値
    if (a.vec) {
        const int i = blockIdx.x * 1024 + 4 * threadIdx.x;
        if (i >= end) return;
        float4 acc = *(const float4*)(part + i);
        for (int s = 1; s < a.ks; s++) {
            const float4 v = *(const float4*)(part + (long long)s * a.total + i);
            acc.x = acc.x + v.x; acc.y = acc.y + v.y; acc.z = acc.z + v.z; acc.w = acc.w + v.w;
        }
        *(float4*)(out + i) = acc;
        return;
    }
    for (int i = blockIdx.x * 1024 + threadIdx.x; i < end; i += blockDim.x) {
        float acc = part[i];
        for (int s = 1; s < a.ks; s++) acc = acc + part[(long long)s * a.total + i];
        out[i] = acc;
    }
}

// Residual add: XOUT = bf16(X + bf16(the partials summed rank 0 first)). G: fp32 [world, rows, d]
// (slices `rs` elements apart), or one bf16 partial [1, rows, d] (g_bf16: a reduce-scatter's share). X may be XOUT.
// grid (rows, ceil(d / 1024)).
// 残差加算: XOUT = bf16(X + bf16(rank 0 から順に足した部分和))。G は fp32 [world, rows, d]
// (スライスの間隔は `rs` 要素)、または bf16 の部分和 1 つ [1, rows, d] (g_bf16: reduce-scatter の取り分)。X は
// XOUT と同じでもよい。grid (rows, ceil(d / 1024))。
typedef struct { Dev x, xout, g; int rows, d, world, rs, g_bf16, vec; } ResidualArgs;
__global__ void tf_glm53_n_residual_add(ResidualArgs a) {
    const uint16_t* x = (const uint16_t*)a.x;
    uint16_t* xout = (uint16_t*)a.xout;
    const long long row = (long long)blockIdx.y * a.d;
    const int end = min(a.d, ((int)blockIdx.x + 1) * 1024);
    // d a multiple of 4, pointers aligned: 4 values a thread
    // d は 4 の倍数、ポインタはアライン済み: 1 スレッド 4 値
    if (a.vec) {
        const int c = blockIdx.x * 1024 + 4 * threadIdx.x;
        if (c >= end) return;
        float xv[4], br[4];
        ld4_bf16v(x + row + c, xv);
        if (a.g_bf16) ld4_bf16v((const uint16_t*)a.g + row + c, br);
        else {
            const float* g = (const float*)a.g;
            float4 acc = *(const float4*)(g + row + c);
            for (int k = 1; k < a.world; k++) {
                const float4 v = *(const float4*)(g + (long long)k * a.rs + row + c);
                acc.x = acc.x + v.x; acc.y = acc.y + v.y; acc.z = acc.z + v.z; acc.w = acc.w + v.w;
            }
            br[0] = bf16_f32(f32_bf16(acc.x)); br[1] = bf16_f32(f32_bf16(acc.y)); br[2] = bf16_f32(f32_bf16(acc.z)); br[3] = bf16_f32(f32_bf16(acc.w));
        }
        unsigned long long o = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) o |= (unsigned long long)f32_bf16(xv[j] + br[j]) << (16 * j);
        *(unsigned long long*)(xout + row + c) = o;
        return;
    }
    for (int c = blockIdx.x * 1024 + threadIdx.x; c < end; c += blockDim.x) {
        float branch;
        if (a.g_bf16) {
            branch = bf16_f32(((const uint16_t*)a.g)[row + c]);
        } else {
            const float* g = (const float*)a.g;
            float acc = g[row + c];
            for (int k = 1; k < a.world; k++) acc = acc + g[(long long)k * a.rs + row + c];
            branch = bf16_f32(f32_bf16(acc));
        }
        xout[row + c] = f32_bf16(bf16_f32(x[row + c]) + branch);
    }
}

// Embedding: OUT[r] = W[IDS[r]], rows of d bf16. grid (rows, ceil(d / 1024)).
// 埋め込み: OUT[r] = W[IDS[r]]、行は d 個の bf16。grid (rows, ceil(d / 1024))。
typedef struct { Dev ids, w, out; int rows, d; } EmbedB16Args;
__global__ void tf_glm53_n_embed_b16(EmbedB16Args a) {
    const long long tok = ((const int*)a.ids)[blockIdx.y];
    const uint16_t* src = (const uint16_t*)a.w + tok * a.d;
    uint16_t* dst = (uint16_t*)a.out + (long long)blockIdx.y * a.d;
    const int end = min(a.d, ((int)blockIdx.x + 1) * 1024);
    for (int c = blockIdx.x * 1024 + threadIdx.x; c < end; c += blockDim.x) dst[c] = src[c];
}

// OUT = bf16(bf16(silu(G)) * U), silu(g) = g / (1 + exp(-g)) with t_exp and t_div, rows of `width` bf16.
// grid (rows, ceil(width / 1024)).
// OUT = bf16(bf16(silu(G)) * U)、silu(g) = g / (1 + exp(-g)) は t_exp と t_div で計算する。行は `width` 個の bf16。
// grid (rows, ceil(width / 1024))。
typedef struct { Dev g, u, out; int rows, width; } Swiglu2Args;
__global__ void tf_glm53_n_swiglu2(Swiglu2Args a) {
    const long long row = (long long)blockIdx.y * a.width;
    const uint16_t* gp = (const uint16_t*)a.g + row;
    const uint16_t* up = (const uint16_t*)a.u + row;
    uint16_t* out = (uint16_t*)a.out + row;
    const int end = min(a.width, ((int)blockIdx.x + 1) * 1024);
    for (int c = blockIdx.x * 1024 + threadIdx.x; c < end; c += blockDim.x) {
        const float g = bf16_f32(gp[c]);
        const float s = bf16_f32(f32_bf16(t_div(g, t_exp(0.0f - g) + 1.0f)));
        out[c] = f32_bf16(s * bf16_f32(up[c]));
    }
}

// A RoPE pair rotated as the original does it: ra = fma(a, cos, -(b sin)), rb = fma(b, cos, a sin) - the sin product
// rounded, the cos product fused. cos / sin are libdevice's.
// 元と同じ RoPE ペアの回転: ra = fma(a, cos, -(b sin))、rb = fma(b, cos, a sin)。sin との積は丸め、cos との積は
// fma に融合する。cos / sin は libdevice のもの。
static __device__ __forceinline__ void rope_pair(float a, float b, float ang, float* ra, float* rb) {
    const float c = cosf(ang), s = sinf(ang);
    *ra = fmaf(a, c, -(b * s));
    *rb = fmaf(b, c, a * s);
}

// Query RoPE: QR[r, h, :32] / [32:] = the rotated pairs of Q[r, h, 192:256] at position POS[0] + r. grid (rows, h),
// 32 threads (a pair a thread). qr_stride: elements between QR's (row, head) records
// (64, or 576 into the DCP exchange's packed queries).
// クエリの RoPE: QR[r, h, :32] / [32:] = 位置 POS[0] + r で回転した Q[r, h, 192:256] のペア。grid (rows, h)、
// 32 スレッド (1 スレッド 1 ペア)。qr_stride: QR の (row, head) レコード間の要素数
// (64、DCP 交換用に詰めたクエリへ書くときは 576)。
typedef struct { Dev q, inv, qr, pos; int rows, h, qr_stride; } QropeArgs;
__global__ void tf_glm53_n_qrope(QropeArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const uint16_t* q = (const uint16_t*)a.q + ((long long)r * a.h + h) * 256 + 192;
    uint16_t* qr = (uint16_t*)a.qr + ((long long)r * a.h + h) * a.qr_stride;
    const float ang = (float)(((const int*)a.pos)[0] + r) * ((const float*)a.inv)[i];
    float ra, rb;
    rope_pair(bf16_f32(q[2 * i]), bf16_f32(q[2 * i + 1]), ang, &ra, &rb);
    qr[i] = f32_bf16(ra);
    qr[32 + i] = f32_bf16(rb);
}

// The index query's first 64 dims of each head rotated in place at position POS[0] + r
// (interleaved pairs in, the two halves out: every pair is read before any is written). grid (rows, 32 heads),
// 32 threads.
// インデックス用クエリの各ヘッドの先頭 64 次元を、位置 POS[0] + r でその場で回転する
// (入力は交互に並んだペア、出力は前後の半分。書く前にすべてのペアを読む)。grid (rows, 32 heads)、32 スレッド。
typedef struct { Dev q, inv, pos; int rows; } IqRopeArgs;
__global__ void tf_glm53_n_iq_rope(IqRopeArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    uint16_t* base = (uint16_t*)a.q + ((long long)r * 32 + h) * 128;
    const float ang = (float)(((const int*)a.pos)[0] + r) * ((const float*)a.inv)[i];
    float ra, rb;
    rope_pair(bf16_f32(base[2 * i]), bf16_f32(base[2 * i + 1]), ang, &ra, &rb);
    __syncthreads();
    base[i] = f32_bf16(ra);
    base[32 + i] = f32_bf16(rb);
}

// x / d for x >= 0 and d > 0: a shift when d is a power of two, else the division. Only the router's slice width uses
// it; in the attention, merge and dconv kernels the shift was slower, so they divide.
// x >= 0、d > 0 での x / d: d が 2 のべき乗ならシフト、それ以外は除算。使うのはルーターのスライス幅だけ。
// attention・merge・dconv のカーネルではシフトのほうが遅かったので除算のままにする。
__device__ __forceinline__ int t_quot(int x, int d) { return (d & (d - 1)) == 0 ? x >> (31 - __clz(d)) : x / d; }

// The row a block of a cache write takes (r, at position pg = pos0 + r) and its slot p. DCP 2: the grid holds the
// owner's rows only ((rows + 1) / 2 blocks, block b takes row 2 b + the owner's first row, 0 or 1) and the slot is a
// shift (positions are not negative); otherwise a block a row, and a block whose row another member owns returns at
// once.
// キャッシュ書き込みで 1 ブロックが受け持つ行 (r、位置 pg = pos0 + r) とそのスロット p。DCP 2 では
// grid は所有する行だけを持ち ((rows + 1) / 2 ブロック、ブロック b は行 2 b + 所有する最初の行 (0 か 1))、
// スロットはシフトで求める (位置は負にならない)。それ以外は 1 行 1 ブロックで、
// 他のメンバーが所有する行のブロックはすぐ戻る。
// The KV pool's map (TF_GLM53_PREFIX=pool, glm53/POOL.md): positions in blocks of 256, a rank's local row u = p / dcp
// at physical row tbl[u >> s] << s | (u & (2^s - 1)), s = log2(256 / dcp) (dcp 1 or 2). A null map: the row is u (the
// contiguous cache). Writes outside [write_lo, write_hi) or into block z (the zero block) or past it are not done and
// set err (1: position, 2: block); readers never write.
// KV pool の表 (TF_GLM53_PREFIX=pool、glm53/POOL.md): 位置は 256 ずつのブロック。ランクのローカル行 u = p / dcp は
// 物理行 tbl[u >> s] << s | (u & (2^s - 1)) にある (s = log2(256 / dcp)、dcp は 1 か 2)。表が null なら行は u (連続の
// キャッシュ)。[write_lo, write_hi) の外や、ブロック z (ゼロのブロック) かそれ以降への書き込みはせず、err を立てる
// (1: 位置、2: ブロック)。読む側は書かない。
typedef struct { int write_lo, write_hi, err, z; unsigned tbl[1]; } PoolMap;
static __device__ __forceinline__ int pool_shift(int dcp) { return dcp == 2 ? 7 : 8; }
static __device__ __forceinline__ long long pool_row(const PoolMap* m, long long u, int sh) {
    return m ? ((long long)m->tbl[u >> sh] << sh) | (u & ((1LL << sh) - 1)) : u;
}
// A write's row: false (the block's thread 0 sets err) when position pg may not be written; *p from u to the row.
// 書き込みの行: 位置 pg に書けないときは false (ブロックのスレッド 0 が err を立てる)。*p を u から行にする。
static __device__ __forceinline__ bool pool_wrow(Dev map, int pg, int dcp, long long* p) {
    PoolMap* m = (PoolMap*)map;
    if (!m) return true;
    const int sh = pool_shift(dcp);
    const bool in = pg >= m->write_lo && pg < m->write_hi;
    const unsigned b = in ? m->tbl[*p >> sh] : 0u;
    if (!in || b >= (unsigned)m->z) {
        if (threadIdx.x == 0) atomicOr(&m->err, in ? 2 : 1);
        return false;
    }
    *p = ((long long)b << sh) | (*p & ((1LL << sh) - 1));
    return true;
}

__device__ __forceinline__ bool t_own_row(int pos0, int rows, int dcp, int rank, int* r, int* pg, long long* p) {
    if (dcp == 2) {
        *r = 2 * (int)blockIdx.x + ((rank - pos0) & 1);
        *pg = pos0 + *r;
        *p = *pg >> 1;
        return *r < rows;
    }
    *r = blockIdx.x;
    *pg = pos0 + *r;
    if (dcp > 1 && *pg % dcp != rank) return false;
    *p = *pg / dcp;
    return true;
}

// The KV cache write: row r at position pg = POS[0] + r; DCP > 1: only the owner (pg % dcp == rank) stores it, at slot
// pg / dcp. cache[:512] = W * bf16(x rinv) (RMSNorm of kva[:512]; thread t holds elements [4t, 4t + 4)), cache
// rope = the rotated pairs of kva[512:576] at pg. kv8: 656-byte rows - 512 e4m3 codes of y / scale, the fp32 scale at
// byte 512, the bf16 rope at byte 528; else 576 bf16. grid (rows; DCP 2: (rows + 1) / 2), 128 threads.
// KV キャッシュへの書き込み: 行 r は位置 pg = POS[0] + r。DCP > 1 では所有者 (pg % dcp == rank) だけがスロット
// pg / dcp に書く。cache[:512] = W * bf16(x rinv) (kva[:512] の RMSNorm、スレッド t は要素 [4t, 4t + 4) を持つ)、
// cache の rope = pg で回転した kva[512:576] のペア。kv8 では 656 バイトの行 (y / scale の e4m3 コード 512 個、
// バイト 512 に fp32 のスケール、バイト 528 に bf16 の rope)、それ以外は 576 個の bf16。
// grid (rows、DCP 2 では (rows + 1) / 2)、128 スレッド。
// kv8: 0 bf16 rows, 1 FP8 (656 bytes), 2 FP4 (KV4_ROW bytes)
// kv8: 0 は bf16 の行、1 は FP8 (656 バイト)、2 は FP4 (KV4_ROW バイト)
typedef struct { Dev kva, nw, lc, pos, inv; int rows, kva_stride, dcp, rank, kv8; float eps; Dev pmap; } KvWriteArgs;
__global__ void tf_glm53_n_kv_write(KvWriteArgs a) {
    __shared__ float sh[8];
    const int t = threadIdx.x;
    int r, pg;
    long long p;
    if (!t_own_row(((const int*)a.pos)[0], a.rows, a.dcp, a.rank, &r, &pg, &p)) return;
    if (!pool_wrow(a.pmap, pg, a.dcp, &p)) return;
    const uint16_t* x = (const uint16_t*)a.kva + (long long)r * a.kva_stride;
    const uint16_t* nw = (const uint16_t*)a.nw;
    float xv[4], acc = 0.0f;
    for (int j = 0; j < 4; j++) {
        xv[j] = bf16_f32(x[4 * t + j]);
        acc = j == 0 ? xv[j] * xv[j] : fmaf(xv[j], xv[j], acc);
    }
    const float sum = t_block_sum(acc, sh, 4);
    const float rinv = t_div(1.0f, t_sqrt(fmaf(sum, 0.001953125f, a.eps)));
    float y[4];
    for (int j = 0; j < 4; j++) y[j] = bf16_f32(nw[4 * t + j]) * bf16_f32(f32_bf16(xv[j] * rinv));
    if (a.kv8 == 2) {
        float m = fmaxf(fabsf(y[0]), fabsf(y[1]));
        m = fmaxf(m, fabsf(y[2]));
        m = fmaxf(m, fabsf(y[3]));
        const float sc = t_kv8_scale(t_block_max(m, sh, 4));
        uint8_t* row = (uint8_t*)a.lc + p * KV4_ROW;
        float v[4], bm = 0.0f;
        for (int j = 0; j < 4; j++) { v[j] = t_div(y[j], sc); bm = fmaxf(bm, fabsf(v[j])); }
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));   // the 16 of threads 4b .. 4b + 3 / スレッド 4b .. 4b + 3 の 16 個
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        // the block scale: bm / 6 as e4m3, rounded up so that no code passes 6
        // ブロックのスケール: bm / 6 を e4m3 に。コードが 6 を超えないよう切り上げる
        unsigned bs = f32_e4m3(bm / 6.0f);
        if (e4m3_f32(bs) < bm / 6.0f) bs += 1u;
        const float bv = e4m3_f32(bs), inv = bv > 0.0f ? 1.0f / bv : 0.0f;
        // codes 4t .. 4t + 3 are bytes 2t, 2t + 1 (TF_KV4P: at 16 (t >> 3) + 4 (t & 3) + 2 ((t >> 2) & 1) of the piece order)
        // コード 4t .. 4t + 3 はバイト 2t, 2t + 1 (TF_KV4P では片の並びで 16 (t >> 3) + 4 (t & 3) + 2 ((t >> 2) & 1))
        uint8_t* cb = row + (KV4P_ON ? 16 * (t >> 3) + 4 * (t & 3) + 2 * ((t >> 2) & 1) : 2 * t);
        cb[0] = (uint8_t)(kv4_nibble(v[0] * inv) | (kv4_nibble(v[1] * inv) << 4));
        cb[1] = (uint8_t)(kv4_nibble(v[2] * inv) | (kv4_nibble(v[3] * inv) << 4));
        if ((t & 3) == 0) row[256 + (t >> 2)] = (uint8_t)bs;
        if (t == 0) *(float*)(row + 288) = sc;
    } else if (a.kv8) {
        float m = fmaxf(fabsf(y[0]), fabsf(y[1]));
        m = fmaxf(m, fabsf(y[2]));
        m = fmaxf(m, fabsf(y[3]));
        const float sc = t_kv8_scale(t_block_max(m, sh, 4));
        uint8_t* row = (uint8_t*)a.lc + p * 656;
        for (int j = 0; j < 4; j++) row[4 * t + j] = f32_e4m3(t_div(y[j], sc));
        if (t == 0) *(float*)(row + 512) = sc;
    } else {
        uint16_t* row = (uint16_t*)a.lc + p * 576;
        for (int j = 0; j < 4; j++) row[4 * t + j] = f32_bf16(y[j]);
    }
    if (t < 32) {
        float ra, rb;
        rope_pair(bf16_f32(x[512 + 2 * t]), bf16_f32(x[512 + 2 * t + 1]), (float)pg * ((const float*)a.inv)[t], &ra, &rb);
        uint16_t* rope = a.kv8 == 2 ? (uint16_t*)((uint8_t*)a.lc + p * KV4_ROW + 304) : a.kv8 ? (uint16_t*)((uint8_t*)a.lc + p * 656 + 528) : (uint16_t*)a.lc + p * 576 + 512;
        rope[t] = f32_bf16(ra);
        rope[32 + t] = f32_bf16(rb);
    }
}

// The index key write: row r at position pg = POS[0] + r (DCP > 1: the owner's slot pg / dcp only).
// y = bf16((x - mean) rs W + B) (LayerNorm of the fp32 key, 128 dims, one a thread; rs = 1 / sqrt(var + eps));
// dims 64.. keep y, dims 0..63 hold the rotated (even, odd) pairs of y as the two halves. kv8: 144-byte rows - 128 e4m3
// codes of the bf16 values / scale, the fp32 scale at byte 128; else 128 bf16. grid (rows), 128 threads.
// As the original: x - mean = fma(sum, -1/128, x); the squares' first butterfly level is fused
// (fma(xc, xc, the partner's square)); (xc rs) W + B is one fma.
// インデックス用キーの書き込み: 行 r は位置 pg = POS[0] + r (DCP > 1 では所有者のスロット pg / dcp だけ)。
// y = bf16((x - mean) rs W + B) (fp32 のキー 128 次元の LayerNorm、1 スレッド 1 次元。rs = 1 / sqrt(var + eps))。
// 次元 64 以降は y のまま、次元 0..63 には y の (偶, 奇) ペアを回転して前後の半分に置く。kv8 では 144 バイトの行
// (bf16 値 / scale の e4m3 コード 128 個、バイト 128 に fp32 のスケール)、それ以外は 128 個の bf16。grid (rows)、
// 128 スレッド。
// 元と同じく、x - mean = fma(sum, -1/128, x)、二乗の最初の butterfly 段は融合 (fma(xc, xc, 相手の二乗))、(xc rs)
// W + B は 1 回の fma。
typedef struct { Dev ik, nw, nb, inv, ic, pos; int rows, dcp, rank, kv8; float eps; Dev pmap; } IkWriteArgs;
__global__ void tf_glm53_n_ik_write(IkWriteArgs a) {
    __shared__ float sh[8];
    const int t = threadIdx.x;
    int r, pg;
    long long p;
    if (!t_own_row(((const int*)a.pos)[0], a.rows, a.dcp, a.rank, &r, &pg, &p)) return;
    if (!pool_wrow(a.pmap, pg, a.dcp, &p)) return;
    const float* x = (const float*)a.ik + (long long)r * 128;
    const uint16_t* nw = (const uint16_t*)a.nw;
    const uint16_t* nb = (const uint16_t*)a.nb;
    const float sum = t_block_sum(x[t], sh, 4);
    const float xc = fmaf(sum, -0.0078125f, x[t]);
    const float sq = xc * xc;
    const float ss = t_block_sum_tail(fmaf(xc, xc, __shfl_xor_sync(0xFFFFFFFFu, sq, 16)), sh, 4);
    const float rs = t_div(1.0f, t_sqrt(fmaf(ss, 0.0078125f, a.eps)));
    const uint16_t y = f32_bf16(fmaf(xc * rs, bf16_f32(nw[t]), bf16_f32(nb[t])));
    float ra = 0.0f, rb = 0.0f;
    if (t < 32) {
        const int e = 2 * t, o = 2 * t + 1;
        const float ye = bf16_f32(f32_bf16(fmaf(fmaf(sum, -0.0078125f, x[e]) * rs, bf16_f32(nw[e]), bf16_f32(nb[e]))));
        const float yo = bf16_f32(f32_bf16(fmaf(fmaf(sum, -0.0078125f, x[o]) * rs, bf16_f32(nw[o]), bf16_f32(nb[o]))));
        rope_pair(ye, yo, (float)pg * ((const float*)a.inv)[t], &ra, &rb);
    }
    if (a.kv8) {
        const float yf = bf16_f32(y);
        ra = bf16_f32(f32_bf16(ra));
        rb = bf16_f32(f32_bf16(rb));
        float m = t >= 64 ? fabsf(yf) : 0.0f;
        if (t < 32) m = fmaxf(m, fmaxf(fabsf(ra), fabsf(rb)));
        const float sc = t_kv8_scale(t_block_max(m, sh, 4));
        uint8_t* row = (uint8_t*)a.ic + p * 144;
        if (t >= 64) row[t] = f32_e4m3(t_div(yf, sc));
        if (t < 32) {
            row[t] = f32_e4m3(t_div(ra, sc));
            row[32 + t] = f32_e4m3(t_div(rb, sc));
        }
        if (t == 0) *(float*)(row + 128) = sc;
    } else {
        uint16_t* row = (uint16_t*)a.ic + p * 128;
        if (t >= 64) row[t] = y;
        if (t < 32) {
            row[t] = f32_bf16(ra);
            row[32 + t] = f32_bf16(rb);
        }
    }
}

// Chunk merge for the DCP exchange; program (row, head of all `ht`): this rank's `nch` chunk partials
// (PO [nch, R, ht, 512] fp32, PM / PL [nch, R, ht]) merged in chunk order by the online log-sum-exp update
// (o = fma(o, a, co b), l = fma(l, a, cl b)) -> the normalized output bf16 and its log-sum-exp m + log(l), laid out
// [destination = head / h, row, head % h] for the exchange; no keys: 0 and -inf. exp is t_exp, log libdevice's.
// grid (rows, ht), 128 threads.
// DCP 交換用のチャンク統合。program は (row, 全 `ht` ヘッドの 1 つ): このランクの `nch` 個のチャンク部分結果
// (PO [nch, R, ht, 512] fp32、PM / PL [nch, R, ht]) をオンライン log-sum-exp 更新
// (o = fma(o, a, co b)、l = fma(l, a, cl b)) でチャンク順にまとめ、正規化した bf16 出力とその log-sum-exp
// m + log(l) を交換用の並び [送り先 = head / h, row, head % h] で書く。キーがなければ 0 と -inf。exp は t_exp、log は
// libdevice のもの。grid (rows, ht)、128 スレッド。
typedef struct { Dev po, pm, pl, osend, lsend; int rows, ht, h, nch; } MergeLseArgs;
__global__ void tf_glm53_n_merge_lse(MergeLseArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const float* po = (const float*)a.po;
    const float* pm = (const float*)a.pm;
    const float* pl = (const float*)a.pl;
    const float ninf = -INFINITY;
    float m = ninf, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
    for (int c = 0; c < a.nch; c++) {
        const long long base = ((long long)c * a.rows + r) * a.ht + h;
        const float cm = pm[base], cl = pl[base];
        const bool active = cl > 0.0f;
        const float next_m = active ? fmaxf(m, cm) : m;
        const float fa = active ? (m == ninf ? 0.0f : t_exp(m - next_m)) : 1.0f;
        const float fb = active ? t_exp(cm - next_m) : 0.0f;
        const float4 co = *(const float4*)(po + base * 512 + 4 * t);
        o[0] = fmaf(o[0], fa, co.x * fb); o[1] = fmaf(o[1], fa, co.y * fb); o[2] = fmaf(o[2], fa, co.z * fb); o[3] = fmaf(o[3], fa, co.w * fb);
        l = fmaf(l, fa, cl * fb);
        m = next_m;
    }
    const long long dst = ((long long)(h / a.h) * a.rows + r) * a.h + h % a.h;
    const bool has = l > 0.0f;
    const float ld = has ? l : 1.0f;
    unsigned long long ov = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) ov |= (unsigned long long)f32_bf16(has ? t_div(o[j], ld) : 0.0f) << (16 * j);
    *(unsigned long long*)((uint16_t*)a.osend + dst * 512 + 4 * t) = ov;
    if (t == 0) ((float*)a.lsend)[dst] = has ? logf(ld) + m : ninf;
}

// tf_glm53_n_merge_lse for exactly 8 chunks (a decode window's partials): the chunks unrolled and the "has this chunk
// keys" selects without branches (both exponentials computed, one kept: the same values).
// head / h and head % h stay divisions: a shift and a mask behind a power-of-two test were slower.
// チャンクがちょうど 8 個 (decode ウィンドウの部分結果) の場合の tf_glm53_n_merge_lse: チャンクを展開し、
// 「このチャンクにキーがあるか」の選択を分岐なしで行う (指数は両方計算して片方を使う。値は同じ)。
// head / h と head % h は除算のまま。2 のべき乗判定つきのシフトとマスクは遅かった。
__global__ void tf_glm53_n_merge_lse_c8(MergeLseArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const float* po = (const float*)a.po;
    const float* pm = (const float*)a.pm;
    const float* pl = (const float*)a.pl;
    const float ninf = -INFINITY;
    float m = ninf, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int c = 0; c < 8; c++) {
        const long long base = ((long long)c * a.rows + r) * a.ht + h;
        const float cm = pm[base], cl = pl[base];
        const bool active = cl > 0.0f;
        const float next_m = active ? fmaxf(m, cm) : m;
        const float ea = t_exp(m - next_m), eb = t_exp(cm - next_m);
        const float fa = active ? (m == ninf ? 0.0f : ea) : 1.0f;
        const float fb = active ? eb : 0.0f;
        const float4 co = *(const float4*)(po + base * 512 + 4 * t);
        o[0] = fmaf(o[0], fa, co.x * fb); o[1] = fmaf(o[1], fa, co.y * fb); o[2] = fmaf(o[2], fa, co.z * fb); o[3] = fmaf(o[3], fa, co.w * fb);
        l = fmaf(l, fa, cl * fb);
        m = next_m;
    }
    const long long dst = ((long long)(h / a.h) * a.rows + r) * a.h + h % a.h;
    const bool has = l > 0.0f;
    const float ld = has ? l : 1.0f;
    unsigned long long ov = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) ov |= (unsigned long long)f32_bf16(has ? t_div(o[j], ld) : 0.0f) << (16 * j);
    *(unsigned long long*)((uint16_t*)a.osend + dst * 512 + 4 * t) = ov;
    if (t == 0) ((float*)a.lsend)[dst] = has ? logf(ld) + m : ninf;
}

// DCP combine; program (row, own head): every member's normalized partial of this head
// (ORECV[src] bf16 at element stride `ss`, its log-sum-exp LRECV[src] at stride `ssl`) merged in rank order: weights
// exp(ls - max), acc = fma(w, x, acc), OUT = bf16(acc / sum of weights). grid (rows, h), 128 threads.
// DCP の結合。program は (row, 自分のヘッド): 全メンバーのこのヘッドの正規化済み部分結果
// (ORECV[src] は要素間隔 `ss` の bf16、その log-sum-exp LRECV[src] は間隔 `ssl`) をランク順にまとめる: 重み
// exp(ls - max)、acc = fma(w, x, acc)、OUT = bf16(acc / 重みの和)。grid (rows, h)、128 スレッド。
typedef struct { Dev orecv, lrecv, out; int rows, ss, ssl, h, world; } DcpCombineArgs;
// One body for any member count; straight-line code for two members was no faster.
// メンバー数によらず本体は 1 つ。2 メンバー専用の直線コードは速くならなかった。
__global__ void tf_glm53_n_dcp_combine(DcpCombineArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const float* lrecv = (const float*)a.lrecv;
    const uint16_t* orecv = (const uint16_t*)a.orecv;
    const long long rh = (long long)r * a.h + h;
    const float ninf = -INFINITY;
    float mx = ninf, lsv[8];   // world <= 8 (the launcher checks) / world <= 8 (起動側で確認する)
    for (int s = 0; s < a.world; s++) {
        lsv[s] = lrecv[(long long)s * a.ssl + rh];
        mx = fmaxf(mx, lsv[s]);
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, den = 0.0f;
    for (int s = 0; s < a.world; s++) {
        const float ls = lsv[s];
        const float w = ls == ninf ? 0.0f : t_exp(ls - mx);
        float xv[4];
        ld4_bf16v(orecv + (long long)s * a.ss + rh * 512 + 4 * t, xv);
#pragma unroll
        for (int j = 0; j < 4; j++) acc[j] = fmaf(w, xv[j], acc[j]);
        den = den + w;
    }
    unsigned long long ov = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) ov |= (unsigned long long)f32_bf16(t_div(acc[j], den)) << (16 * j);
    *(unsigned long long*)((uint16_t*)a.out + rh * 512 + 4 * t) = ov;
}

// Expert choice: the row's 8 of 256 experts by sigmoid(logit) + bias, the larger first, the lower id on a tie;
// WTS = sigmoid of each pick / (their sum in pick order + 1e-20) * scale, sigmoid = 1 / (1 + exp(-x)) with t_exp and
// t_div. The picks are exact comparisons, so only the sigmoid, the sum's order and the final division round.
// grid (rows), 256 threads (an expert a thread).
// expert の選択: 行ごとに 256 個から sigmoid(logit) + bias の大きい順に 8 個選ぶ (同値なら番号の小さいほう)。
// WTS = 各選択の sigmoid / (選択順の和 + 1e-20) * scale、sigmoid = 1 / (1 + exp(-x)) は t_exp と t_div で計算する。
// 選択は厳密な比較なので、丸めが入るのは sigmoid、和の順序、最後の除算だけ。grid (rows)、256 スレッド
// (1 スレッド 1 expert)。
typedef struct { Dev logits, bias, pick, wts; int rows; float scale; } TopkArgs;
// ONE: the block's first thread compares the warps' winners and hands the result to the others - fewer instructions,
// faster when the blocks fill the GPU, slower when a block's own critical path counts. Else every thread compares. The
// same comparisons either way; the launcher takes ONE from 128 rows.
// ONE: warp ごとの勝者をブロックの先頭スレッドが比べ、結果を他のスレッドに渡す。命令は少なく、ブロックが
// GPU を埋めるときは速いが、1 ブロックのクリティカルパスが効くときは遅い。ONE でなければ全スレッドが比べる。どちらでも
// 比較は同じ。起動側は 128 行以上で ONE を使う。
static __device__ __forceinline__ void topk_body(TopkArgs a, const int ONE, float* s_val, int* s_idx, float* s_wts, int* s_pick, int* s_win) {
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float score = t_div(1.0f, t_exp(0.0f - ((const float*)a.logits)[(long long)r * 256 + t]) + 1.0f);
    float choice = score + ((const float*)a.bias)[t];
    for (int k = 0; k < 8; k++) {
        float v = choice;
        int i = t;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, o);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, i, o);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { s_val[warp] = v; s_idx[warp] = i; }
        __syncthreads();
        int bi;
        if (!ONE || t == 0) {   // the warps' winners in warp order / warp ごとの勝者を warp 順に
            float bv = s_val[0];
            bi = s_idx[0];
            for (int w = 1; w < 8; w++)
                if (s_val[w] > bv || (s_val[w] == bv && s_idx[w] < bi)) { bv = s_val[w]; bi = s_idx[w]; }
            if (ONE) *s_win = bi;
        }
        // every thread has read the warps' winners before the next round overwrites them
        // 次の回が上書きする前に全スレッドが warp ごとの勝者を読み終える
        __syncthreads();
        // s_win is rewritten after the next round's first barrier, which every reader has passed
        // s_win が書き換わるのは次の回の最初のバリアの後で、読む側は全員そこを通過している
        if (ONE) bi = *s_win;
        if (t == bi) {
            s_pick[k] = bi;
            s_wts[k] = score;
            choice = -INFINITY;
        }
    }
    __syncthreads();
    if (t < 8) {
        float total = 0.0f;
        for (int k = 0; k < 8; k++) total = total + s_wts[k];
        ((int*)a.pick)[(long long)r * 8 + t] = s_pick[t];
        ((float*)a.wts)[(long long)r * 8 + t] = t_div(s_wts[t], total + 1e-20f) * a.scale;
    }
}
#define TOPK_ENTRY(name, ONE) \
__global__ void name(TopkArgs a) { \
    __shared__ float s_val[8]; \
    __shared__ int s_idx[8]; \
    __shared__ float s_wts[8]; \
    __shared__ int s_pick[8]; \
    __shared__ int s_win; \
    topk_body(a, ONE, s_val, s_idx, s_wts, s_pick, &s_win); \
}
TOPK_ENTRY(tf_glm53_n_topk, 0)
TOPK_ENTRY(tf_glm53_n_topk_one, 1)


// 8 bf16 at p (16-byte aligned) as fp32
// p にある 8 個の bf16 (16 バイト境界) を fp32 にする
static __device__ __forceinline__ void ld8_bf16(const uint16_t* p, float* f) {
    const uint4 v = *(const uint4*)p;
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}

// QA[r, h, n] = q[r, h, :192] . WK[h, :, n] for the 512 latent columns n, and on latent block 0 the rotated pairs of
// q[r, h, 192:256] at position POS[0] + r (as tf_glm53_n_qrope).
// grid (h, 16 latent blocks, row blocks of rbk), 128 threads. The order of an output's 192 products: thread
// (k0 = t >> 2, column group t & 3) chains k = k0 + 32 j for j = 0..5 (then + 0 twice: the padded rows 192..255); the
// warp sums its 8 chains (k0 & 7) by a butterfly (16, 8, 4 on the lane); the 4 warps add as (0 + 2) + (1 + 3). A
// product of two bf16 is exact in fp32, so fusing changes nothing.
// qa_stride / qr_stride: elements between the (row, head) records of QA and QR (512 / 64, or 576 / 576 with
// QR = QA + 512 elements: the DCP exchange's packed queries written in place)
// QA[r, h, n] = q[r, h, :192] . WK[h, :, n] (潜在列 n は 512 個)。潜在ブロック 0 では、位置 POS[0] + r で回転した
// q[r, h, 192:256] のペアも書く (tf_glm53_n_qrope と同じ)。
// grid (h, 潜在ブロック 16 個, rbk 行ずつの行ブロック)、128 スレッド。1 出力の 192 個の積の順序: スレッド
// (k0 = t >> 2、列グループ t & 3) が j = 0..5 の k = k0 + 32 j を順に連ねる (続けて + 0 を 2 回:
// パディングの行 192..255)。warp は 8 本の連鎖 (k0 & 7) を butterfly (レーンの 16, 8, 4) で足し、4 つの warp は
// (0 + 2) + (1 + 3) で足す。bf16 どうしの積は fp32 で厳密なので、融合しても値は変わらない。
// qa_stride / qr_stride: QA と QR の (row, head) レコード間の要素数 (512 / 64。DCP
// 交換用に詰めたクエリへその場で書くときは 576 / 576 で、QR = QA + 512 要素)
typedef struct { Dev q, wk, inv, qa, qr, pos; int rows, h, rbk, qa_stride, qr_stride; } AbsorbArgs;
__global__ void tf_glm53_n_absorb(AbsorbArgs a) {
    __shared__ float sh[128];
    const int h = blockIdx.x, nb = blockIdx.y, r0 = blockIdx.z * a.rbk, t = threadIdx.x;
    const int cg = t & 3, k0 = t >> 2, warp = t >> 5;
    const uint16_t* wk = (const uint16_t*)a.wk + ((long long)h * 192 + k0) * 512 + nb * 32 + cg * 8;
    float w[6][8];
    for (int j = 0; j < 6; j++) ld8_bf16(wk + (long long)j * 32 * 512, w[j]);
    for (int jr = 0; jr < a.rbk; jr++) {
        const int r = r0 + jr;
        if (r >= a.rows) break;
        const uint16_t* q = (const uint16_t*)a.q + ((long long)r * a.h + h) * 256;
        float qv[6], acc[8];
        for (int j = 0; j < 6; j++) qv[j] = bf16_f32(q[k0 + 32 * j]);
        for (int c = 0; c < 8; c++) {
            float s = fmaf(w[0][c], qv[0], w[1][c] * qv[1]);
            for (int j = 2; j < 6; j++) s = fmaf(w[j][c], qv[j], s);
            s = s + 0.0f;
            s = s + 0.0f;
            s = s + __shfl_xor_sync(0xffffffffu, s, 16);
            s = s + __shfl_xor_sync(0xffffffffu, s, 8);
            s = s + __shfl_xor_sync(0xffffffffu, s, 4);
            acc[c] = s;
        }
        if ((t & 0x1c) == 0)
            for (int c = 0; c < 8; c++) sh[cg * 32 + c * 4 + warp] = acc[c];
        __syncthreads();
        if (t < 32) {
            // column n = t: group t >> 3, c = t & 7
            // 列 n = t: グループ t >> 3、c = t & 7
            const float* v = sh + t * 4;
            ((uint16_t*)a.qa)[((long long)r * a.h + h) * a.qa_stride + nb * 32 + t] = f32_bf16((v[0] + v[2]) + (v[1] + v[3]));
            if (nb == 0) {
                uint16_t* qr = (uint16_t*)a.qr + ((long long)r * a.h + h) * a.qr_stride;
                const float ang = (float)(((const int*)a.pos)[0] + r) * ((const float*)a.inv)[t];
                float ra, rb;
                rope_pair(bf16_f32(q[192 + 2 * t]), bf16_f32(q[192 + 2 * t + 1]), ang, &ra, &rb);
                qr[t] = f32_bf16(ra);
                qr[32 + t] = f32_bf16(rb);
            }
        }
        __syncthreads();
    }
}

// OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] over the 512 latent k, for 256 output columns n.
// grid (h, 16 output blocks, row blocks of rbk), 128 threads. The order of an output's 512 products: thread
// (column group g = t & 63, row parity t >> 6) chains k = 8 g .. 8 g + 7; a warp sums its 32 chains by the full
// butterfly (16, 8, 4, 2, 1); the two warps of a parity add.
// OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] (潜在の k は 512 個、出力列 n は 256 個)。
// grid (h, 出力ブロック 16 個, rbk 行ずつの行ブロック)、128 スレッド。1 出力の 512 個の積の順序: スレッド
// (列グループ g = t & 63、行の偶奇 t >> 6) が k = 8 g .. 8 g + 7 を順に連ね、warp は 32 本の連鎖を完全な butterfly
// (16, 8, 4, 2, 1) で足し、同じ偶奇の 2 つの warp を足す。
typedef struct { Dev ol, wv, out; int rows, h, rbk; } ExpandArgs;
__global__ void tf_glm53_n_expand(ExpandArgs a) {
    __shared__ float sh[32];
    const int h = blockIdx.x, nb = blockIdx.y, r0 = blockIdx.z * a.rbk, t = threadIdx.x;
    const int b = t >> 6, g = t & 63, warp = t >> 5;
    const uint16_t* wv = (const uint16_t*)a.wv + ((long long)h * 256 + nb * 16 + b) * 512 + g * 8;
    float w[8][8];
    for (int j = 0; j < 8; j++) ld8_bf16(wv + (long long)j * 2 * 512, w[j]);
    for (int jr = 0; jr < a.rbk; jr++) {
        const int r = r0 + jr;
        if (r >= a.rows) break;
        float o[8], acc[8];
        ld8_bf16((const uint16_t*)a.ol + ((long long)r * a.h + h) * 512 + g * 8, o);
        for (int j = 0; j < 8; j++) {
            float s = fmaf(w[j][0], o[0], w[j][1] * o[1]);
            for (int c = 2; c < 8; c++) s = fmaf(w[j][c], o[c], s);
            s = s + __shfl_xor_sync(0xffffffffu, s, 16);
            s = s + __shfl_xor_sync(0xffffffffu, s, 8);
            s = s + __shfl_xor_sync(0xffffffffu, s, 4);
            s = s + __shfl_xor_sync(0xffffffffu, s, 2);
            s = s + __shfl_xor_sync(0xffffffffu, s, 1);
            acc[j] = s;
        }
        if ((t & 31) == 0)
            for (int j = 0; j < 8; j++) sh[j * 4 + warp] = acc[j];
        __syncthreads();
        // output n = t of the block: row pair t >> 1, parity t & 1 (warps 2 parity, 2 parity + 1)
        // ブロックの出力 n = t: 行ペア t >> 1、偶奇 t & 1 (warp 2 parity と 2 parity + 1)
        if (t < 16)
            ((uint16_t*)a.out)[((long long)r * a.h + h) * 256 + nb * 16 + t] = f32_bf16(sh[(t >> 1) * 4 + (t & 1) * 2] + sh[(t >> 1) * 4 + (t & 1) * 2 + 1]);
        __syncthreads();
    }
}

// The tensor-core pieces, as the original's: LDGSTS 16-byte copies into shared memory, LDSM, HMMA.16816.F32.BF16.
// テンソルコア用の部品。元と同じく、共有メモリへの 16 バイトの LDGSTS コピー、LDSM、HMMA.16816.F32.BF16。
static __device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
// 16 bytes global -> shared, or zeros without reading when `read` is 0
// グローバルから共有メモリへ 16 バイト。`read` が 0 なら読まずにゼロを書く
static __device__ __forceinline__ void cp16z(void* dst, const void* src, int read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(read ? 16 : 0));
}
static __device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
static __device__ __forceinline__ void cp_commit(void) { asm volatile("cp.async.commit_group;\n" ::); }
static __device__ __forceinline__ void cp_wait1(void) { asm volatile("cp.async.wait_group 1;\n" ::); }
static __device__ __forceinline__ void cp_wait0(void) { asm volatile("cp.async.wait_group 0;\n" ::); }
static __device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
// Two 16 x 16 byte matrices transposed (rows at the addresses of lanes 0..15 and 16..31): lane (g, qq) gets rows
// 4 qq .. 4 qq + 3 of column g in r[0] (lowest byte = row 4 qq) and of column g + 8 in r[1]; the second matrix's in
// r[2], r[3]. sm_100f / sm_120f and later only (family-specific targets).
// 16 x 16 バイトの行列 2 つを転置で読む (行の番地はレーン 0..15 と 16..31): レーン (g, qq) は列 g の行
// 4 qq .. 4 qq + 3 を r[0] に (最下位バイトが行 4 qq)、列 g + 8 の分を r[1] に、2 つ目の行列の分を r[2]、r[3] に
// 受け取る。sm_100f / sm_120f 以降 (family-specific の target) だけ。
#define TF_LDSM_B8 1
static __device__ __forceinline__ void ldsm_b8t_x2(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
// Present in builds with these loads: the engine then runs FP4's 32 heads as two programs along z (with the PV
// operand loaded this way they beat one block of 512 threads, bench_pair attn4).
// この読み込みを持つビルドにだけある: エンジンは FP4 の 32 head を z 方向の program 2 つで走らせる (PV の入力を
// この形で読むと 512 スレッドの 1 ブロックより速い、bench_pair attn4)。
__global__ void tf_glm53_n_pv_b8(void) {}
#endif
// c[16 x 8] += a[16 x 16] b[16 x 8] (bf16 in, fp32 accumulate)
// c[16 x 8] += a[16 x 16] b[16 x 8] (入力は bf16、fp32 で累積)
static __device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The router's matmul by K slice: PART[s, m, e] = x[m, slice s of K] . w[e, slice s] (fp32), the slice's k16 steps in
// ascending order through one accumulator (an element's bits do not depend on where it sits in a tile).
// grid (16-row blocks, 32-expert blocks, ks slices), 128 threads: warp w holds experts 8 w .. 8 w + 7 of the block.
// 64-wide K stages, 2 in flight, one barrier a stage; a staged row's 16-byte pieces sit at piece ^ (row & 7)
// (the 8 rows of an ldmatrix then fall on distinct banks).
// ルーターの行列積を K のスライスごとに: PART[s, m, e] = x[m, K のスライス s] . w[e, スライス s] (fp32)。スライスの k16
// 段を昇順に 1 つのアキュムレータで足す (要素のビットはタイル内の位置によらない)。
// grid (16 行ブロック, 32 expert ブロック, ks スライス)、128 スレッド: warp w はブロックの
// expert 8 w .. 8 w + 7 を持つ。K は 64 幅のステージで 2 つ先読みし、ステージごとにバリア 1 回。ステージした行の
// 16 バイト片は piece ^ (row & 7) に置く (ldmatrix の 8 行が別々のバンクに載る)。
typedef struct { Dev x, w, part; int m, x_stride, d, ne, ks; } RouterPartArgs;
// st = step % 3
// st は step % 3
static __device__ __forceinline__ void rp_load(RouterPartArgs a, uint8_t* sx, uint8_t* sw, int step, int st, int m0, int e0, int lo) {
    // this thread's staging: one 16-byte piece of x (row t >> 3, piece t & 7), two of w
    // (rows t >> 3 and 16 + (t >> 3)); a row's pieces are stored at piece ^ (row & 7)
    // このスレッドが運ぶ分: x の 16 バイト片 1 つ (行 t >> 3、片 t & 7) と w の片 2 つ (行 t >> 3 と 16 + (t >> 3))。
    // 行の片は piece ^ (row & 7) に置く
    const int t = threadIdx.x, xr = t >> 3, pc = t & 7, dp = (pc ^ (xr & 7)) * 16;
    const long long k0 = lo + step * 64 + pc * 8;
    cp16z(sx + st * 2048 + xr * 128 + dp, (const uint16_t*)a.x + (long long)(m0 + xr) * a.x_stride + k0, m0 + xr < a.m);
    cp16z(sw + st * 4096 + xr * 128 + dp, (const uint16_t*)a.w + (long long)(e0 + xr) * a.d + k0, e0 + xr < a.ne);
    cp16z(sw + st * 4096 + (16 + xr) * 128 + dp, (const uint16_t*)a.w + (long long)(e0 + 16 + xr) * a.d + k0, e0 + 16 + xr < a.ne);
}
__global__ void tf_glm53_n_router_part(RouterPartArgs a) {
    __shared__ __align__(16) uint8_t sx[3 * 16 * 128];
    __shared__ __align__(16) uint8_t sw[3 * 32 * 128];
    const int m0 = blockIdx.x * 16, e0 = blockIdx.y * 32, sl = blockIdx.z, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5, sw7 = lane & 7;
    const int kw = t_quot(a.d, a.ks), steps = kw / 64, lo = sl * kw;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int s0 = 0; s0 < 2; s0++) {
        if (s0 < steps) rp_load(a, sx, sw, s0, s0, m0, e0, lo);
        cp_commit();
    }
    const int arow = ((lane & 7) + ((lane >> 3) & 1) * 8) * 128, brow = (warp * 8 + (lane & 7)) * 128;
    int st = 0, st2 = 2;   // i % 3 and (i + 2) % 3, stepped as a ring / i % 3 と (i + 2) % 3 を環状に進める
    for (int i = 0; i < steps; i++) {
        cp_wait1();
        __syncthreads();
        // into the stage step i - 1 used
        // ステップ i - 1 が使ったステージへ
        if (i + 2 < steps) rp_load(a, sx, sw, i + 2, st2, m0, e0, lo);
        cp_commit();
        const uint8_t* px = sx + st * 2048 + arow;
        const uint8_t* pw = sw + st * 4096 + brow;
        st = st == 2 ? 0 : st + 1;
        st2 = st2 == 2 ? 0 : st2 + 1;
#pragma unroll
        for (int h = 0; h < 2; h++) {   // 32 columns a load of w / w の 1 回のロードで 32 列
            unsigned af[4], bf[4];
            ldsm4(bf, pw + ((4 * h + (lane >> 3)) ^ sw7) * 16);
            ldsm4(af, px + ((4 * h + (lane >> 4)) ^ sw7) * 16);
            mma_bf16(acc, af, bf[0], bf[1]);
            ldsm4(af, px + ((4 * h + 2 + (lane >> 4)) ^ sw7) * 16);
            mma_bf16(acc, af, bf[2], bf[3]);
        }
    }
    cp_wait0();
    const int row = m0 + (lane >> 2), col = e0 + warp * 8 + (lane & 3) * 2;
    float* part = (float*)a.part + (long long)sl * a.m * a.ne;
    // columns col and col + 1 of a row: one 8-byte store when both are inside and the pair is aligned
    // 行の列 col と col + 1: 両方が範囲内でペアがアライン済みなら 8 バイトのストア 1 回
    float* p0 = part + (long long)row * a.ne + col;
    float* p1 = part + (long long)(row + 8) * a.ne + col;
    const bool pair = col + 1 < a.ne;
    if (row < a.m) {
        if (pair && ((size_t)p0 & 7) == 0) *(float2*)p0 = make_float2(acc[0], acc[1]);
        else {
            if (col < a.ne) p0[0] = acc[0];
            if (pair) p0[1] = acc[1];
        }
    }
    if (row + 8 < a.m) {
        if (pair && ((size_t)p1 & 7) == 0) *(float2*)p1 = make_float2(acc[2], acc[3]);
        else {
            if (col < a.ne) p1[0] = acc[2];
            if (pair) p1[1] = acc[3];
        }
    }
}

// two e4m3 bytes (the low 16 bits of v) as two bf16, exactly: e4m3 -> f16 -> bf16 as the original
// 2 つの e4m3 バイト (v の下位 16 ビット) を 2 つの bf16 に厳密に変換する。元と同じく e4m3 -> f16 -> bf16
static __device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned v) {
    unsigned r;
    asm("{ .reg .b16 s, lo, hi, blo, bhi; .reg .b32 t;\n"
        "  cvt.u16.u32 s, %1;\n"
        "  cvt.rn.f16x2.e4m3x2 t, s;\n"
        "  mov.b32 {lo, hi}, t;\n"
        "  cvt.rn.bf16.f16 blo, lo;\n"
        "  cvt.rn.bf16.f16 bhi, hi;\n"
        "  mov.b32 %0, {blo, bhi}; }\n" : "=r"(r) : "r"(v));
    return r;
}
// two e4m3 codes (low 16 bits of v) times two e4m3 scales (low 16 bits of s), as bf16x2: exact (the products have at
// most 5 significant bits)
// e4m3 コード 2 つ (v の下位 16 ビット) x e4m3 スケール 2 つ (s の下位 16 ビット) を bf16x2 に: 積の有効桁は 5 ビット
// 以下なので誤差はない
static __device__ __forceinline__ unsigned e4m3x2_scaled_bf16x2(unsigned v, unsigned s) {
    unsigned r;
    asm("{ .reg .b16 c, d, lo, hi, blo, bhi; .reg .b32 t, u, w;\n"
        "  cvt.u16.u32 c, %1;\n"
        "  cvt.u16.u32 d, %2;\n"
        "  cvt.rn.f16x2.e4m3x2 t, c;\n"
        "  cvt.rn.f16x2.e4m3x2 u, d;\n"
        "  mul.rn.f16x2 w, t, u;\n"
        "  mov.b32 {lo, hi}, w;\n"
        "  cvt.rn.bf16.f16 blo, lo;\n"
        "  cvt.rn.bf16.f16 bhi, hi;\n"
        "  mov.b32 %0, {blo, bhi}; }\n" : "=r"(r) : "r"(v), "r"(s));
    return r;
}
// two e4m3 scales (low 16 bits of s) as f16x2, exact; with e4m3x2_mul_bf16x2 the same bits as e4m3x2_scaled_bf16x2,
// the scales converted once for every operand that uses them
// e4m3 スケール 2 つ (s の下位 16 ビット) を f16x2 に (誤差なし)。e4m3x2_mul_bf16x2 と合わせて e4m3x2_scaled_bf16x2 と
// 同じビットになり、スケールの変換はそれを使う入力すべてに対して 1 回で済む
static __device__ __forceinline__ unsigned e4m3x2_f16x2(unsigned s) {
    unsigned r;
    asm("{ .reg .b16 c; cvt.u16.u32 c, %1; cvt.rn.f16x2.e4m3x2 %0, c; }\n" : "=r"(r) : "r"(s));
    return r;
}
static __device__ __forceinline__ unsigned e4m3x2_mul_bf16x2(unsigned v, unsigned f) {
    unsigned r;
    asm("{ .reg .b16 c, lo, hi, blo, bhi; .reg .b32 t, w;\n"
        "  cvt.u16.u32 c, %1;\n"
        "  cvt.rn.f16x2.e4m3x2 t, c;\n"
        "  mul.rn.f16x2 w, t, %2;\n"
        "  mov.b32 {lo, hi}, w;\n"
        "  cvt.rn.bf16.f16 blo, lo;\n"
        "  cvt.rn.bf16.f16 bhi, hi;\n"
        "  mov.b32 %0, {blo, bhi}; }\n" : "=r"(r) : "r"(v), "r"(f));
    return r;
}

// Indexer scores: for window row R0 + r at position p = POS[0] + R0 + r and this rank's cache slots t
// (global position g = t dcp + rank),
//   score[t] = sum_h (W[r, h] wscale) max((q[r, h] . key[t]) qscale ks[t], 0)   over keys g <= p, t < T,
// stored as int64 (monotone int of the score << 32 | 0x7fffffff - g; later keys the minimum) or, pack 0, the score's
// unsigned-order int32 (later keys 0). Key rows are 144 bytes: 128 e4m3 codes, the fp32 scale at
// byte 128. grid (rows, 128-key blocks), 128 threads: warp w holds keys 32 w .. 32 w + 31 against all 32 heads.
// The original's arithmetic: the e4m3 codes become bf16 and the dot runs on HMMA.16816.F32.BF16, but Triton lays the K
// axis out 4 wide a thread (the 8-bit source), so one HMMA takes dims {32 rep + 16 b + 4 j + 2 half + e} into its k
// slots 8 b + 2 j + e (b, e in 0..1, j in 0..3) - 8 HMMAs an element in the order (rep, half). A thread then holds
// heads g, g + 8, g + 16, g + 24 (g = lane >> 2) of a key: fma(s[g], w[g], s[g + 8] w[g + 8]),
// then + g + 16, + g + 24 fused; the warp sums the 8 threads of a key by a butterfly (16, 8, 4).
// インデクサのスコア: ウィンドウの行 R0 + r (位置 p = POS[0] + R0 + r) と、このランクのキャッシュスロット t
// (全体の位置 g = t dcp + rank) について
//   score[t] = sum_h (W[r, h] wscale) max((q[r, h] . key[t]) qscale ks[t], 0)   (キー g <= p、t < T)
// を int64 (スコアを単調な整数にして << 32 | 0x7fffffff - g。p より後のキーは最小値) で、
// pack 0 ならスコアの符号なし順序の int32 (p より後のキーは 0) で書く。キーの行は 144 バイト: e4m3 コード 128 個、
// バイト 128 に fp32 のスケール。grid (rows, 128 キーのブロック)、128 スレッド: warp
// w はキー 32 w .. 32 w + 31 を全 32 ヘッドについて受け持つ。
// 元の計算のまま: e4m3 コードを bf16 にして内積を HMMA.16816.F32.BF16 で計算する。ただし Triton は 8 ビットの入力のため
// K 軸を 1 スレッド 4 幅で並べるので、1 回の HMMA は次元 {32 rep + 16 b + 4 j + 2 half + e} を k スロット 8 b + 2 j + e
// (b, e は 0..1、j は 0..3) に取り込み、1 要素あたり 8 回の HMMA を (rep, half) の順に行う。
// スレッドはキー 1 つについてヘッド g, g + 8, g + 16, g + 24 (g = lane >> 2) を持ち、
// fma(s[g], w[g], s[g + 8] w[g + 8]) に続けて + g + 16、+ g + 24 を融合して足す。warp はキーの 8 スレッドを butterfly
// (16, 8, 4) で足す。
#define IS_ROW 144
typedef struct { Dev q, w, ic, pos, out; int n, t, r0, dcp, rank, pack, rb; float wscale, qscale; Dev pmap; } IndexScoresArgs;
// One kernel for both outputs (a.pack): a separate entry for the order words was slower.
// 両方の出力形式 (a.pack) を 1 つのカーネルで扱う。順序ワード専用の入口は遅かった。
// IS_QROW: a query row in shared memory, 32 heads of 288 bytes (256 + 32: a half-warp's 8-byte reads of the A fragments
// hit 32 different banks); two of them fill the key tile's bytes once the keys sit in registers.
// IS_QROW: 共有メモリ上のクエリ 1 行。32 ヘッド x 288 バイト (256 + 32: A フラグメントの 8 バイト読みが half-warp で
// 32 個の別々のバンクに当たる)。キーをレジスタに移した後、キーのタイルの場所にちょうど 2 行入る。
#define IS_QH 288
#define IS_QROW (32 * IS_QH)
__global__ void __launch_bounds__(128, 4) tf_glm53_n_index_scores(IndexScoresArgs a) {
    // A block scores `rb` rows against its 128 keys: the keys staged once, converted to bf16 B fragments once (they
    // stay in registers for all the rows), then each row's query staged into shared memory by cp.async while the
    // row before computes (the four warps share it; read from global memory they each fetched the whole row).
    // ブロックは `rb` 行を自分の 128 キーに対して採点する: キーは 1 回置いて 1 回だけ bf16 の B フラグメントに変換し
    // (全行のあいだレジスタに置いたまま)、各行のクエリは前の行の計算中に cp.async で共有メモリに置く (4 warp で
    // 共有する。global から読むと warp ごとに行全体を読んでいた)。
    __shared__ __align__(16) uint8_t sk[128 * IS_ROW];
    __shared__ float sks[128];
    static_assert(2 * IS_QROW <= 128 * IS_ROW, "two query rows in the key tile's bytes");
    const int rg = blockIdx.x * a.rb, t0 = blockIdx.y * 128, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int g = lane >> 2, qq = lane & 3;
    const int p0 = ((const int*)a.pos)[0] + a.r0;
    const int re = min(rg + a.rb, a.n);
    // rows rg .. rs - 1: the block is later than them (positions grow with the row); rows rs .. re - 1 are scored
    // 行 rg .. rs - 1 はブロックがその行より後 (位置は行とともに増える)。行 rs .. re - 1 を採点する
    const int rs = max(rg, t0 * a.dcp + a.rank - p0);
    // pack 2: the order words and, after them in the same buffer (from the 16-byte boundary past n t words), their top
    // 16 bits [n, t] u16 (tf_glm53_topk_words_any16 reads those first), not in blocks later than the row (the reader
    // knows those columns are 0)
    // pack 2: 順序語と、同じバッファでその後ろ (n t 語の先の 16 バイト境界から) に上位 16 ビット [n, t] u16
    // (tf_glm53_topk_words_any16 はそちらを先に読む)。行より後のブロックには書かない (読む側はその列が 0 と知っている)
    uint16_t* p16 = (uint16_t*)((uint8_t*)a.out + ((long long)a.n * a.t * 4 + 15) / 16 * 16);
    for (int r = rg; r < min(rs, re); r++)
        if (t0 + t < a.t) {
            if (a.pack == 1) ((long long*)a.out)[(long long)r * a.t + t0 + t] = -9223372036854775807LL;
            else ((int*)a.out)[(long long)r * a.t + t0 + t] = 0;
        }
    if (rs >= re) return;
    // the block's 128 local rows lie in one pool block (256 / dcp >= 128 rows, t0 a multiple of 128)
    // ブロックの 128 ローカル行は pool の 1 ブロックに入る (256 / dcp >= 128 行、t0 は 128 の倍数)
    const uint8_t* ic = (const uint8_t*)a.ic + (pool_row((const PoolMap*)a.pmap, t0, pool_shift(a.dcp)) - t0) * IS_ROW;
#pragma unroll
    for (int j = 0; j < 8; j++) {   // a warp's copy: 4 whole key rows / 1 warp のコピー: キーの行 4 本分
        const int kk = (t >> 3) + 16 * j, kt = t0 + kk;
        cp16z(sk + kk * IS_ROW + (t & 7) * 16, ic + (long long)kt * IS_ROW + (t & 7) * 16, kt < a.t);
    }
    cp_commit();
    sks[t] = t0 + t < a.t ? *(const float*)(ic + (long long)(t0 + t) * IS_ROW + 128) : 0.0f;
    cp_wait0();
    __syncthreads();
    // the warp's 32 keys as bf16 B fragments: bk[nt][c] = the 8 keys nt of the warp, dims
    // {32 rep + 16 b + 4 qq + 2 half + e} (c = 2 rep + half) as (b0, b1)
    // warp の 32 キーの bf16 の B フラグメント: bk[nt][c] = warp の 8 キー nt、次元
    // {32 rep + 16 b + 4 qq + 2 half + e} (c = 2 rep + half) を (b0, b1) で
    unsigned bk[4][8][2];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        // the 8 keys' bytes 16 seg + 4 qq .. + 3, seg 0..7
        // 8 キーのバイト 16 seg + 4 qq .. + 3 (seg は 0..7)
        unsigned raw[8];
        const uint8_t* krow = sk + (warp * 32 + nt * 8 + (lane & 7)) * IS_ROW + (lane >> 3) * 16;
        ldsm4(raw, krow);
        ldsm4(raw + 4, krow + 64);
#pragma unroll
        for (int c = 0; c < 8; c++) {
            const int sh = (c & 1) * 16;
            bk[nt][c][0] = e4m3x2_bf16x2(raw[c & 6] >> sh);
            bk[nt][c][1] = e4m3x2_bf16x2(raw[(c & 6) + 1] >> sh);
        }
    }
    float ks[4][2];
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int j = 0; j < 2; j++) ks[nt][j] = sks[warp * 32 + nt * 8 + 2 * qq + j];
    __syncthreads();   // the key tile is free for the query rows / キーのタイルの場所をクエリの行に使える
    // row r's query (8,192 bytes) into buffer b: 512 chunks of 16 bytes, chunk ch = head ch >> 4, bytes 16 (ch & 15)
    // 行 r のクエリ (8,192 バイト) をバッファ b へ: 16 バイトの塊 512 個。塊 ch = ヘッド ch >> 4、バイト 16 (ch & 15)
    const uint8_t* qg = (const uint8_t*)a.q;
#define IS_STAGE_Q(r, b) _Pragma("unroll") for (int i = 0; i < 4; i++) { const int ch = t + 128 * i; \
        cp16(sk + (b) * IS_QROW + (ch >> 4) * IS_QH + (ch & 15) * 16, qg + (long long)(r) * 8192 + ch * 16); }
    IS_STAGE_Q(rs, 0)
    cp_commit();
    // the key this lane stores: the warp's 32 scores go out as one 128-byte (256-byte packed) store, lane (g, qq)
    // taking key 8 (g >> 1) + 2 qq + (g & 1) - after the butterfly every g holds the key's sum
    // このレーンが書くキー: warp の 32 個の点数を 1 回の 128 バイト (pack なら 256 バイト) の書き込みで出す。レーン
    // (g, qq) はキー 8 (g >> 1) + 2 qq + (g & 1) を書く (butterfly の後はどの g もそのキーの和を持つ)
    const int kmine = warp * 32 + (g >> 1) * 8 + 2 * qq + (g & 1), ktm = t0 + kmine, gpm = ktm * a.dcp + a.rank;
    for (int r = rs; r < re; r++) {
        const int p = p0 + r, buf = (r - rs) & 1;
        if (r + 1 < re) { IS_STAGE_Q(r + 1, buf ^ 1) cp_commit(); cp_wait1(); } else cp_wait0();
        float wv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) wv[i] = ((const float*)a.w)[r * 32 + g + 8 * i] * a.wscale;
        __syncthreads();
        // the query's A fragments, rows (heads) 16 m + g (+ 8), dims 32 rep + 4 qq + 2 half (+ 16), a rep at a time:
        // the products into each accumulator in the order c = 2 rep + half, as before
        // クエリの A フラグメント: 行 (ヘッド) 16 m + g (+ 8)、次元 32 rep + 4 qq + 2 half (+ 16) を rep ごとに。
        // 各アキュムレータへの積の順は c = 2 rep + half で従来どおり
        const uint8_t* qs = sk + buf * IS_QROW;
        float acc[4][2][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][m][i] = 0.0f;
#pragma unroll
        for (int rep = 0; rep < 4; rep++) {
            unsigned af[2][2][4];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const uint8_t* q0 = qs + (16 * m + g) * IS_QH + 64 * rep + 8 * qq;
                const uint2 v00 = *(const uint2*)q0, v01 = *(const uint2*)(q0 + 32);
                const uint2 v10 = *(const uint2*)(q0 + 8 * IS_QH), v11 = *(const uint2*)(q0 + 8 * IS_QH + 32);
                af[m][0][0] = v00.x; af[m][1][0] = v00.y;
                af[m][0][1] = v10.x; af[m][1][1] = v10.y;
                af[m][0][2] = v01.x; af[m][1][2] = v01.y;
                af[m][0][3] = v11.x; af[m][1][3] = v11.y;
            }
#pragma unroll
            for (int half = 0; half < 2; half++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    mma_bf16(acc[nt][0], af[0][half], bk[nt][2 * rep + half][0], bk[nt][2 * rep + half][1]);
                    mma_bf16(acc[nt][1], af[1][half], bk[nt][2 * rep + half][0], bk[nt][2 * rep + half][1]);
                }
        }
        float sv[8];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const float k = ks[nt][j];
                const float s0 = fmaxf(0.0f, (acc[nt][0][j] * a.qscale) * k), s1 = fmaxf(0.0f, (acc[nt][0][2 + j] * a.qscale) * k);
                const float s2 = fmaxf(0.0f, (acc[nt][1][j] * a.qscale) * k), s3 = fmaxf(0.0f, (acc[nt][1][2 + j] * a.qscale) * k);
                float sc = fmaf(s0, wv[0], s1 * wv[1]);
                sc = fmaf(s2, wv[2], sc);
                sc = fmaf(s3, wv[3], sc);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 16);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 8);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 4);
                sv[2 * nt + j] = sc;
            }
        float sc = sv[0];
#pragma unroll
        for (int i = 1; i < 8; i++) sc = g == i ? sv[i] : sc;
        if (ktm < a.t) {
            const int bits = __float_as_int(sc), key = bits >= 0 ? bits : bits ^ 0x7fffffff;
            const int ok = gpm <= p;
            if (a.pack == 1) ((long long*)a.out)[(long long)r * a.t + ktm] = ok ? (long long)(((unsigned long long)(unsigned)key << 32) | (unsigned)(0x7fffffff - gpm)) : -9223372036854775807LL;
            else {
                const unsigned word = ok ? (unsigned)key ^ 0x80000000u : 0u;
                ((unsigned*)a.out)[(long long)r * a.t + ktm] = word;
                if (a.pack == 2) p16[(long long)r * a.t + ktm] = (uint16_t)(word >> 16);
            }
        }
        __syncthreads();   // buffer buf is free for row r + 2 / バッファ buf を行 r + 2 に使える
    }
#undef IS_STAGE_Q
}

// ---- The indexer's scores on the FP8 tensor-core product (TF_GLM53_INDEX_FP8). Not bit-exact with
// tf_glm53_n_index_scores: the query too becomes e4m3 codes with a power-of-two scale a (row, head), as
// tf_glm53_n_ik_write makes the keys, and the dot runs on mma m16n8k32 (e4m3): 4 instructions an element in place
// of 8 HMMAs and 8 code conversions, 32 fragment registers a row in place of 64. The query's rounding (3 mantissa bits)
// changes the scores, so the picks can differ at the margin. ----
// ---- FP8 テンソルコアの積によるインデクサのスコア (TF_GLM53_INDEX_FP8)。tf_glm53_n_index_scores とはビット一致しない:
// クエリも (row, head) ごとの 2 のべき乗スケールで e4m3 コードにし (tf_glm53_n_ik_write がキーを作るのと同じ)、内積を
// mma m16n8k32 (e4m3) で計算する。1 要素あたり HMMA 8 回とコード変換 8 回の代わりに命令 4 つ、フラグメントのレジスタは
// 1 行 64 個の代わりに 32 個。クエリの丸め (仮数 3 ビット) でスコアが変わるので、境界付近では選択が変わることがある。
// ----

// c[16 x 8] += a[16 x 32] b[32 x 8] (e4m3 in, fp32 accumulate). A thread's registers: a0 = (row g, k 4 q .. + 3),
// a1 = (row g + 8, the same k), a2 / a3 = k + 16; b0 = (k 4 q .. + 3, column g), b1 = k + 16; c as m16n8k16.
// c[16 x 8] += a[16 x 32] b[32 x 8] (入力は e4m3、fp32 で累積)。スレッドのレジスタ: a0 = (行 g, k 4 q .. + 3)、
// a1 = (行 g + 8, 同じ k)、a2 / a3 = k + 16。b0 = (k 4 q .. + 3, 列 g)、b1 = k + 16。c は m16n8k16 と同じ。
static __device__ __forceinline__ void mma_e4m3(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Q [rows, 32, 128] bf16 (after tf_glm53_n_iq_rope) -> Q8 [rows, 4096] e4m3 codes of q / scale in fragment order and QS
// [rows, 32] the scales (2^(ceil(log2 amax) - 8), as the key rows'). Fragment order: lane 4 g + qq of a warp reads
// its 128 bytes at lane 128; word (M 4 + s) 4 + i of them = head 16 M + g + 8 (i & 1),
// dims 32 s + 16 (i >> 1) + 4 qq .. + 3 - the A fragment's register i of head group M, k step s. grid (rows, 32 heads),
// 128 threads (a dim each).
// Q [rows, 32, 128] bf16 (tf_glm53_n_iq_rope の後) -> Q8 [rows, 4096] (q / scale の e4m3 コード、フラグメント順) と QS
// [rows, 32] (スケール。キーの行と同じ 2^(ceil(log2 amax) - 8))。フラグメント順: warp のレーン 4 g + qq は
// lane 128 にある自分の 128 バイトを読む。そのうちワード (M 4 + s) 4 + i = ヘッド 16 M + g + 8 (i & 1)、
// 次元 32 s + 16 (i >> 1) + 4 qq .. + 3 で、ヘッドグループ M・k 段 s の A フラグメントのレジスタ i にあたる。
// grid (rows, 32 heads)、128 スレッド (1 スレッド 1 次元)。
typedef struct { Dev q, q8, qs; int rows; } IqFp8Args;
__global__ void tf_glm53_n_iq_fp8(IqFp8Args a) {
    __shared__ float sh[4];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float v = bf16_f32(((const uint16_t*)a.q)[((long long)r * 32 + h) * 128 + d]);
    const float amax = t_block_max(fabsf(v), sh, 4);
    const float sc = t_exp2(ceilf(log2f(fmaxf(amax, 1e-30f))) - 8.0f);
    const int M = h >> 4, g = h & 7, i0 = (h >> 3) & 1, s = d >> 5, i1 = (d >> 4) & 1, qq = (d >> 2) & 3, e = d & 3;
    ((uint8_t*)a.q8)[(long long)r * 4096 + (4 * g + qq) * 128 + ((M * 4 + s) * 4 + i0 + 2 * i1) * 4 + e] = f32_e4m3(t_div(v, sc));
    if (d == 0) ((float*)a.qs)[(long long)r * 32 + h] = sc;
}

// tf_glm53_n_index_scores' layout and outputs with the scores of the FP8 product:
//   score[t] = sum_h (W[r, h] wscale QS[r, h]) max((q8[r, h] . key8[t]) qscale ks[t], 0)
// A block scores `rb` rows against its 128 staged keys, as tf_glm53_n_index_scores does; a row's fragments are 8 loads
// of 16 bytes (a warp instruction: 512 contiguous bytes of Q8).
// tf_glm53_n_index_scores と同じ並びと出力で、FP8 の積のスコアを書く:
//   score[t] = sum_h (W[r, h] wscale QS[r, h]) max((q8[r, h] . key8[t]) qscale ks[t], 0)
// tf_glm53_n_index_scores と同じく、ブロックは `rb` 行を共有メモリに置いた 128 キーに対して採点する。
// 1 行のフラグメントは 16 バイトのロード 8 回 (warp の 1 命令で Q8 の連続 512 バイト)。
typedef struct { Dev q8, qs, w, ic, pos, out; int n, t, r0, dcp, rank, pack, rb; float wscale, qscale; Dev pmap; } IndexScores8Args;
__global__ void tf_glm53_n_index_scores8(IndexScores8Args a) {
    __shared__ __align__(16) uint8_t sk[128 * IS_ROW];
    __shared__ float sks[128];
    const int rg = blockIdx.x * a.rb, t0 = blockIdx.y * 128, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int g = lane >> 2, qq = lane & 3;
    const int p0 = ((const int*)a.pos)[0] + a.r0;
    // the whole block is later than every row of it
    // ブロック全体がどの行よりも後
    if (t0 * a.dcp + a.rank > p0 + rg + a.rb - 1) {
        for (int r = rg; r < rg + a.rb && r < a.n; r++)
            if (t0 + t < a.t) {
                if (a.pack) ((long long*)a.out)[(long long)r * a.t + t0 + t] = -9223372036854775807LL;
                else ((int*)a.out)[(long long)r * a.t + t0 + t] = 0;
            }
        return;
    }
    const uint8_t* ic = (const uint8_t*)a.ic + (pool_row((const PoolMap*)a.pmap, t0, pool_shift(a.dcp)) - t0) * IS_ROW;
#pragma unroll
    for (int j = 0; j < 8; j++) {   // a warp's copy: 4 whole key rows / 1 warp のコピー: キーの行 4 本分
        const int kk = (t >> 3) + 16 * j, kt = t0 + kk;
        cp16z(sk + kk * IS_ROW + (t & 7) * 16, ic + (long long)kt * IS_ROW + (t & 7) * 16, kt < a.t);
    }
    cp_commit();
    sks[t] = t0 + t < a.t ? *(const float*)(ic + (long long)(t0 + t) * IS_ROW + 128) : 0.0f;
    cp_wait0();
    __syncthreads();
    for (int r = rg; r < rg + a.rb && r < a.n; r++) {
        const int p = p0 + r;
        long long* out64 = (long long*)a.out + (long long)r * a.t;
        int* out32 = (int*)a.out + (long long)r * a.t;
        if (t0 * a.dcp + a.rank > p) {   // the block is later than this row / ブロックがこの行より後
            if (t0 + t < a.t) {
                if (a.pack) out64[t0 + t] = -9223372036854775807LL;
                else out32[t0 + t] = 0;
            }
            continue;
        }
        const uint4* qf = (const uint4*)((const uint8_t*)a.q8 + (long long)r * 4096 + lane * 128);
        unsigned af[2][4][4];
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int s = 0; s < 4; s++) {
                const uint4 v = qf[m * 4 + s];
                af[m][s][0] = v.x; af[m][s][1] = v.y; af[m][s][2] = v.z; af[m][s][3] = v.w;
            }
        float wv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) wv[i] = (((const float*)a.w)[r * 32 + g + 8 * i] * a.wscale) * ((const float*)a.qs)[r * 32 + g + 8 * i];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            // the 8 keys' bytes 16 seg + 4 qq .. + 3, seg 0..7: B's registers as they stand
            // 8 キーのバイト 16 seg + 4 qq .. + 3 (seg は 0..7)。そのまま B のレジスタになる
            unsigned raw[8];
            const uint8_t* krow = sk + (warp * 32 + nt * 8 + (lane & 7)) * IS_ROW + (lane >> 3) * 16;
            ldsm4(raw, krow);
            ldsm4(raw + 4, krow + 64);
            float acc[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#pragma unroll
            for (int s = 0; s < 4; s++) {
                mma_e4m3(acc[0], af[0][s], raw[2 * s], raw[2 * s + 1]);
                mma_e4m3(acc[1], af[1][s], raw[2 * s], raw[2 * s + 1]);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int kk = warp * 32 + nt * 8 + 2 * qq + j, kt = t0 + kk;
                const float ks = sks[kk];
                const float s0 = fmaxf(0.0f, (acc[0][j] * a.qscale) * ks), s1 = fmaxf(0.0f, (acc[0][2 + j] * a.qscale) * ks);
                const float s2 = fmaxf(0.0f, (acc[1][j] * a.qscale) * ks), s3 = fmaxf(0.0f, (acc[1][2 + j] * a.qscale) * ks);
                float sc = fmaf(s0, wv[0], s1 * wv[1]);
                sc = fmaf(s2, wv[2], sc);
                sc = fmaf(s3, wv[3], sc);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 16);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 8);
                sc = sc + __shfl_xor_sync(0xffffffffu, sc, 4);
                if (g == 0 && kt < a.t) {
                    const int bits = __float_as_int(sc), key = bits >= 0 ? bits : bits ^ 0x7fffffff;
                    const int gp = kt * a.dcp + a.rank, ok = gp <= p;
                    if (a.pack) out64[kt] = ok ? (long long)(((unsigned long long)(unsigned)key << 32) | (unsigned)(0x7fffffff - gp)) : -9223372036854775807LL;
                    else out32[kt] = ok ? (int)((unsigned)key ^ 0x80000000u) : 0;
                }
            }
        }
    }
}

// exclusive prefix sum of v over the block's threads (in thread order) and the block's total; sh: a slot a warp
// ブロックのスレッドにわたる v の排他的プレフィックス和 (スレッド順) とブロックの合計。sh は warp ごとに 1 スロット
static __device__ __forceinline__ int t_block_excl_scan(int v, int* sh, int warps, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int d = 1; d < 32; d <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) sh[warp] = x;
    __syncthreads();
    int base = 0, tot = 0;
    for (int w = 0; w < warps; w++) {
        const int sw = sh[w];
        if (w < warp) base += sw;
        tot += sw;
    }
    __syncthreads();
    *total = tot;
    return base + x - v;
}

// Key selection: row r of S [rows, nc] uint32 order words -> OUT[r, :k] = the columns of the k largest words, ascending
// (ties at the k-th word go to the lowest columns). Integers only, so any correct select gives the same bits; the
// method follows the original: four 8-bit histogram passes find the k-th word, one pass in column order writes the
// columns. grid (rows), 256 threads (4 columns a thread in each 1,024-column step).
// キーの選択: S [rows, nc] (uint32 の順序ワード) の行 r -> OUT[r, :k] = 大きい k 個のワードの列番号を昇順に
// (k 番目のワードと同値なら列番号の小さいほうを取る)。整数だけなので、正しい選択ならどれも同じビットになる。
// 方法は元と同じ: 8 ビットのヒストグラムを 4 回作って k 番目のワードを求め、列順の 1 パスで列番号を書く。grid (rows)、
// 256 スレッド (1,024 列の各段で 1 スレッド 4 列)。
typedef struct { Dev s, out; int rows, nc, k; } SelectKeysArgs;
__global__ void tf_glm53_n_select_keys(SelectKeysArgs a) {
    __shared__ int hist[256];
    __shared__ int sh[8];
    __shared__ int s_digit, s_sub;
    const int r = blockIdx.x, t = threadIdx.x;
    const unsigned* row = (const unsigned*)a.s + (long long)r * a.nc;
    int* out = (int*)a.out + (long long)r * a.k;
    unsigned prefix = 0, fixed = 0;
    int need = a.k;
    for (int p = 0; p < 4; p++) {
        const int shift = 24 - 8 * p;
        hist[t] = 0;
        __syncthreads();
        for (int c = 0; c < a.nc; c += 1024)
            for (int e = 0; e < 4; e++) {
                const int i = c + 4 * t + e;
                if (i < a.nc) {
                    const unsigned u = row[i];
                    if ((u & fixed) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
                }
            }
        __syncthreads();
        // the largest digit with `need` words at or above it (0 if none), and the words above it
        // その桁以上に `need` 個以上のワードがある最大の桁 (なければ 0) と、それより上のワードの数
        if (t == 0) {
            int above = 0, digit = 0, sub = 0, b = 255;
            for (; b >= 0; b--) {
                above += hist[b];
                if (above >= need) break;
            }
            if (b >= 0) { digit = b; sub = above - hist[b]; }
            else sub = above - hist[0];
            s_digit = digit;
            s_sub = sub;
        }
        __syncthreads();
        need -= s_sub;
        prefix |= (unsigned)s_digit << shift;
        fixed |= 0xffu << shift;
        __syncthreads();
    }
    int written = 0, equal_seen = 0;
    for (int c = 0; c < a.nc; c += 1024) {
        int eqm = 0, gtm = 0;
        for (int e = 0; e < 4; e++) {
            const int i = c + 4 * t + e;
            if (i < a.nc) {
                const unsigned u = row[i];
                if (u == prefix) eqm |= 1 << e;
                if (u > prefix) gtm |= 1 << e;
            }
        }
        int total;
        const int before = t_block_excl_scan(__popc(eqm) | (__popc(gtm) << 16), sh, 8, &total);
        // words equal to the k-th before this thread in the step
        // この段でこのスレッドより前にある、k 番目と等しいワードの数
        int eb = before & 0xffff;
        int room = need - equal_seen;
        if (room < 0) room = 0;
        int at = written + (before >> 16) + (eb < room ? eb : room);
        for (int e = 0; e < 4; e++) {
            if ((gtm >> e) & 1) out[at++] = c + 4 * t + e;
            else if ((eqm >> e) & 1) {
                if (eb < room) out[at++] = c + 4 * t + e;
                eb++;
            }
        }
        const int eq_all = total & 0xffff;
        written += (total >> 16) + (eq_all < room ? eq_all : room);
        equal_seen += eq_all;
    }
}

// Chunk merge; program (row, head): the row's `nch` chunk partials (PO [nch, R, h, 512] fp32, PM / PL [nch, R, h])
// merged in chunk order as tf_glm53_n_merge_lse does -> OUT[r, h] = bf16(o / l). grid (rows, h), 128 threads.
// チャンク統合。program は (row, head): 行の `nch` 個のチャンク部分結果 (PO [nch, R, h, 512] fp32、PM / PL
// [nch, R, h]) を tf_glm53_n_merge_lse と同じくチャンク順にまとめる -> OUT[r, h] = bf16(o / l)。grid (rows, h)、
// 128 スレッド。
typedef struct { Dev po, pm, pl, out; int rows, h, nch; } MergeArgs;
__global__ void tf_glm53_n_merge(MergeArgs a) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const float* po = (const float*)a.po;
    const float* pm = (const float*)a.pm;
    const float* pl = (const float*)a.pl;
    const float ninf = -INFINITY;
    float m = ninf, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
    for (int c = 0; c < a.nch; c++) {
        const long long base = ((long long)c * a.rows + r) * a.h + h;
        const float cm = pm[base], cl = pl[base];
        const bool active = cl > 0.0f;
        const float next_m = active ? fmaxf(m, cm) : m;
        const float fa = active ? (m == ninf ? 0.0f : t_exp(m - next_m)) : 1.0f;
        const float fb = active ? t_exp(cm - next_m) : 0.0f;
        const float4 co = *(const float4*)(po + base * 512 + 4 * t);
        o[0] = fmaf(o[0], fa, co.x * fb); o[1] = fmaf(o[1], fa, co.y * fb); o[2] = fmaf(o[2], fa, co.z * fb); o[3] = fmaf(o[3], fa, co.w * fb);
        l = fmaf(l, fa, cl * fb);
        m = next_m;
    }
    unsigned long long ov = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) ov |= (unsigned long long)f32_bf16(t_div(o[j], l)) << (16 * j);
    *(unsigned long long*)((uint16_t*)a.out + ((long long)r * a.h + h) * 512 + 4 * t) = ov;
}

// Sparse MLA attention; program (row, chunk): NH query heads of row r over entries [c chk, (c + 1) chk) of its key list
// (keys 0..n - 1 themselves while p < 2048, else TOK[r]), in tiles of KTT = 8 W keys
// (W warps: warp w holds keys 8 w .. 8 w + 7 of the tile for the scores and 512 / W latent columns for the output).
// Cache rows are 656 bytes: 512 e4m3 codes, the fp32 scale at byte 512, the bf16 rope at byte 528. A tile, as the
// original computes it (all on HMMA.16816.F32.BF16, codes -> bf16 exactly):
//   x = q_lat . codes       32 HMMAs; the 8-bit source makes Triton lay K out 4 wide a thread, so one HMMA takes dims
//                           {32 rep + 16 b + 4 j + 2 half + e} into its k slots 8 b + 2 j + e, in the order (rep, half)
//   x = x ks, then 4 more HMMAs add q_rope . rope (plain 16-dim steps) into the same accumulator; s = x scale
//   tile max, next_m, alpha = exp(m - next_m), pr = exp(s - next_m) (ex2.approx of x log2 e)
//   o = o alpha, then KTT / 16 HMMAs add bf16(pr ks) . codes with the tile's keys laid out the same 4-wide way
//   l = fma(alpha, l, sum pr): a thread's 2 keys, the warp's 4 threads by a butterfly (2, 1), the warps by a
//   butterfly (W / 2 .. 1)
// g = 0: QA [R, NH, 512], QR [R, NH, 64]; n = min(p + 1, 2048); direct: PO = bf16(o / l) [R, NH, 512].
// g > 0 (DCP): QA = QALL [g, R, h, 576], head j = member j / h, head j % h; n = this rank's slots or CNT[r].
// Partials go to PO [(c R + r) HT + j, 512] fp32, PM, PL (HT = ht, else NH). NH 8 uses half of the 16-row tile. The key tiles sit in
// dynamic shared memory (AT_SMEM(W) bytes); with W 4 the next tile is staged while the current one computes.
// 疎な MLA の attention。program は (row, chunk): 行 r の NH 個のクエリヘッドを、キー一覧の項目 [c chk, (c + 1) chk)
// (p < 2048 の間はキー 0..n - 1 そのもの、それ以降は TOK[r]) に対して、KTT = 8 W キーのタイルごとに計算する
// (W は warp 数。warp w はスコアではタイルのキー 8 w .. 8 w + 7、出力では潜在列 512 / W 個を持つ)。キャッシュの行は
// 656 バイト: e4m3 コード 512 個、バイト 512 に fp32 のスケール、バイト 528 に bf16 の rope。1 タイルの計算は元と同じ
// (すべて HMMA.16816.F32.BF16、コード -> bf16 は厳密):
//   x = q_lat . codes       HMMA 32 回。8 ビットの入力のため Triton は K を 1 スレッド 4 幅で並べ、1 回の HMMA は次元
//                           {32 rep + 16 b + 4 j + 2 half + e} を k スロット 8 b + 2 j + e に (rep, half) の順に入れる
//   x = x ks、続けて HMMA 4 回で q_rope . rope (普通の 16 次元ずつ) を同じアキュムレータに足す。s = x scale
//   タイルの max、next_m、alpha = exp(m - next_m)、pr = exp(s - next_m) (x log2 e の ex2.approx)
//   o = o alpha、続けて HMMA KTT / 16 回で bf16(pr ks) . codes を足す (タイルのキーは同じ 4 幅の並び)
//   l = fma(alpha, l, sum pr): スレッドの 2 キー、warp の 4 スレッドを butterfly (2, 1)、warp 間を butterfly
//   (W / 2 .. 1) で足す
// g = 0: QA [R, NH, 512]、QR [R, NH, 64]。n = min(p + 1, 2048)。direct なら PO = bf16(o / l) [R, NH, 512]。
// g > 0 (DCP): QA = QALL [g, R, h, 576]、ヘッド j = メンバー j / h のヘッド j % h。n はこのランクのスロット数か
// CNT[r]。
// 部分結果は PO [(c R + r) NH + j, 512] fp32、PM、PL に書く。NH 8 では 16 行タイルの半分を使う。
// キーのタイルは動的共有メモリ (AT_SMEM(W) バイト) に置き、W 4 では今のタイルの計算中に次のタイルを読み込む。
#define AT_ROW 656
// ht: the arrangement's head count (QALL's g h, or h), when a launch splits it into programs of NH heads along z (head
// j0 = z NH onward); 0 means NH.
// ht: 並び全体のヘッド数 (QALL の g h か h)。1 回の起動を z 方向に NH ヘッドずつの program に分けるとき (ヘッドは
// j0 = z NH から)。0 なら NH。
typedef struct { Dev qa, qr, lc, tok, cnt, pos, po, pm, pl; int rows, h, g, chk, dcp, rank, direct; float scale; int ht; Dev pmap; } AttnArgs;
// A head's member and index in it (j / h, j % h) and a short row's slot count stay plain divisions: with a shift and a
// mask behind a power-of-two test (h and dcp are powers of two) every program was slower.
// ヘッドのメンバーとその中の番号 (j / h、j % h) と、短い行のスロット数は普通の除算のまま。
// 2 のべき乗判定つきのシフトとマスク (h と dcp は 2 のべき乗) ではどの program も遅かった。
// HG head groups a block (1, or 2: two programs of NH heads in one block of 2 x 32 W threads that stage each key tile
// once for both; each group computes exactly as its own program would, in its own bf16(p) / max / sum partials).
// HG: ブロックあたりのヘッドの組の数 (1、または 2: NH ヘッドの program 2 つを 2 x 32 W スレッドの 1 ブロックにし、
// key のタイルを両方のために 1 回だけ置く。各組は自分の bf16(p) / max / sum の部分結果で、単独の program と同じ
// 計算をする)。
static __device__ __forceinline__ void attn_body(AttnArgs a, const int NH, const int W, const int DL, const int KV4, const int HG, uint8_t* sk, uint16_t* spr, float* smax, float* ssum, uint8_t* sbs) {
    const int KTT = 8 * W, NT = 64 / W, THREADS = 32 * W * HG;   // THREADS: the block's, every one stages / ブロック全体 (全員でタイルを置く)
    // P4 (TF_KV4P, FP4): tiles of the cache's 432-byte rows, codes turned into e4m3 at the MMA operands; else 656-byte
    // rows of e4m3 codes (FP8 as cached, FP4 converted while staged)
    // P4 (TF_KV4P、FP4): キャッシュの 432 バイトの行のままのタイル、コードは MMA の入力の所で e4m3 に。それ以外は
    // e4m3 コードの 656 バイトの行 (FP8 はキャッシュのまま、FP4 は置く時に変換)
    const int P4 = KV4 ? KV4P_ON : 0, AR = P4 ? KV4_ROW : AT_ROW;
    const int r = blockIdx.x, c = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = (t >> 5) % W, grp = (t >> 5) / W, g = lane >> 2, qq = lane & 3;
    spr += grp * 16 * KTT;
    smax += grp * 16 * W;
    ssum += grp * 16 * W;
    const int RS = NH == 16 ? 2 : 1;   // head rows a thread: g (and g + 8) / 1 スレッドのヘッド行: g (と g + 8)
    const int j0 = (blockIdx.z * HG + grp) * NH, HT = a.ht > 0 ? a.ht : NH;   // this program's first head, all heads / この program の先頭ヘッド、全ヘッド数
    const float ninf = -INFINITY;
    const int p = ((const int*)a.pos)[0] + r;
    int n;
    if (a.g == 0) n = p + 1 < 2048 ? p + 1 : 2048;
    else if (p < 2048) n = p >= a.rank ? (p - a.rank) / a.dcp + 1 : 0;
    else n = ((const int*)a.cnt)[r];
    const int start = c * a.chk;
    float m[2] = { ninf, ninf }, l[2] = { 0.0f, 0.0f };
    float o[16][4];
    for (int i = 0; i < NT; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    if (start < n) {
        const uint16_t* ql[2];
        const uint16_t* qr[2];
        for (int rs = 0; rs < 2; rs++) {
            const int j = j0 + (NH == 16 ? g + 8 * rs : g);
            if (a.g == 0) {
                ql[rs] = (const uint16_t*)a.qa + ((long long)r * HT + j) * 512;
                qr[rs] = (const uint16_t*)a.qr + ((long long)r * HT + j) * 64;
            } else {
                ql[rs] = (const uint16_t*)a.qa + (((long long)(j / a.h) * a.rows + r) * a.h + j % a.h) * 576;
                qr[rs] = ql[rs] + 512;
            }
        }
        const uint8_t* lc = (const uint8_t*)a.lc;
        const int* tok = (const int*)a.tok + (long long)r * 2048;
        // keys are local rows u (p < 2048: j0 + kk; else tok); the pool's map turns each into its row (a sparse
        // tile's keys lie in many blocks)
        // キーはローカル行 u (p < 2048 は j0 + kk、それ以外は tok)。pool の表で各キーを行にする (疎のタイルのキーは
        // あちこちのブロックにある)
        const PoolMap* pmap = (const PoolMap*)a.pmap;
        const int psh = pool_shift(a.g == 0 ? 1 : a.dcp);
        const int left = n - start < a.chk ? n - start : a.chk;
        // W 4 (the tiles of 32 keys that windows of 16 rows or fewer take, forward.zig: decode windows, few blocks,
        // latency-bound): each tile's key rows once into shared memory (two buffers, tile i in buffer i & 1), not per
        // 16-byte piece: tiles 0 and 1 here, tile i + 1 while tile i - PIPE computes (after its first barrier, so the
        // barriers before the next staging publish it; the buffer it takes was last read staging tile i - 1). W 8 (prompt
        // chunks) maps each piece's key in place: staged rows were 9 % slower there, and both loops in one kernel made the
        // prompt attention 6 % slower in the engine (W 8 fixed at compile time: one loop, no staging array)
        // W 4 (16 行以下のウィンドウが使う 32 キーのタイル、forward.zig: decode ウィンドウ、ブロックが少なくレイテンシ
        // 律速): 各タイルのキーの行を 16 バイトの片ごとでなく 1 回だけ共有メモリへ (バッファ 2 つ、タイル i はバッファ
        // i & 1)。タイル 0 と 1 はここで、タイル i + 1 はタイル i - PIPE の計算中に (その最初のバリアの後に書くので、
        // 次のステージングの前のバリアで見える。使うバッファを最後に読んだのはタイル i - 1 のステージング)。W 8
        // (prompt チャンク) は片ごとにその場で引く: 並べるとそこでは 9 % 遅く、1 つのカーネルに両方のループを持たせると
        // エンジンの prompt の attention が 6 % 遅かった (W 8 はコンパイル時に固定: ループ 1 つ、並べる配列なし)
        __shared__ long long s_key[2][64];
        const bool staged = W == 4 && a.rows <= 16;   // W 4 keeps the runtime test: its code came out 5 % faster with it
        const int ntile = (left + KTT - 1) / KTT;
        auto key_rows = [&](int ti) {
            const int jb = start + ti * KTT;
            for (int kk = t; kk < KTT; kk += THREADS)
                s_key[ti & 1][kk] = jb + kk < n ? pool_row(pmap, p < 2048 ? jb + kk : tok[jb + kk], psh) : 0;
        };
        if (staged) {
            key_rows(0);
            if (ntile > 1) key_rows(1);
            __syncthreads();
        }
        const int tiles = (left + KTT - 1) / KTT;
        uint8_t* const sk0 = sk;
        // W 4 (decode windows): two tile buffers, tile ti staged while tile ti - 1 computes
        // (the buffer it takes was last read by tile ti - 2, and every thread is past that tile's closing barrier).
        // W 8 (prompt chunks): one buffer - a second would take the block past 48 KiB of shared memory, and fewer
        // blocks would fit an SM.
        // W 4 (decode ウィンドウ): タイル用バッファ 2 つ。タイル ti - 1 の計算中にタイル ti を読み込む
        // (使うバッファを最後に読んだのはタイル ti - 2 で、全スレッドがそのタイルの最後のバリアを越えている)。
        // W 8 (prompt チャンク): バッファ 1 つ。2 つにするとブロックの共有メモリが 48 KiB を超え、
        // 1 SM に載るブロックが減る。
        // HG 2 (a block of 512 threads, one a SM already): two tile buffers as for W 4, tile ti staged by cp.async while
        // tile ti - 1 computes (the same tiles in the same order: the same bytes)
        // HG 2 (512 スレッドのブロック、もともと 1 SM に 1 つ): W 4 と同じくタイル用バッファ 2 つで、タイル ti - 1 の
        // 計算中にタイル ti を cp.async で読み込む (同じタイルを同じ順に計算するので同じバイト)
        const int PIPE = (W == 4 || HG == 2) ? 1 : 0;
        for (int ti = 0; ti < tiles + PIPE; ti++) {
            if (ti < tiles) {
                uint8_t* dst = sk0 + (PIPE ? (ti & 1) * KTT * AR : 0);
                const int j0 = start + ti * KTT;
                // the pieces' loops made twice: with the staged rows (decode windows) and with the rows mapped in place
                // 片のループを 2 通りに作る: 並べた行を読むもの (decode ウィンドウ) と、その場で行を引くもの
                auto stage = [&](auto keyf) {
                    if (P4) {
                        // the rows as cached, 27 pieces of 16 bytes / キャッシュの行のまま、16 バイトの片 27 個
                        for (int x = t; x < KTT * 27; x += THREADS) {
                            const int kk = x / 27, pc = x - kk * 27, ok = j0 + kk < n;
                            const long long key = !ok ? 0 : keyf(kk);
                            cp16z(dst + kk * AR + pc * 16, lc + key * KV4_ROW + pc * 16, ok);
                        }
                    } else if (KV4) {
                        // FP4 rows: 16 pieces of 32 codes as e4m3 codes of the same values, the 32 block scales to sbs, then
                        // the row scale and the rope as they are (16 bytes a piece, 288 .. 432 -> 512 .. 656)
                        // FP4 の行: コード 32 個ずつ 16 片を同じ値の e4m3 コードに、ブロックのスケール 32 個を sbs に、続いて
                        // 行スケールと rope をそのまま (16 バイトずつ、288 .. 432 -> 512 .. 656)
                        uint8_t* sdst = sbs + (PIPE ? (ti & 1) * KTT * 32 : 0);
                        for (int x = t; x < KTT * 27; x += THREADS) {
                            const int kk = x / 27, pc = x - kk * 27, ok = j0 + kk < n;
                            const long long key = !ok ? 0 : keyf(kk);
                            const uint8_t* src = lc + key * KV4_ROW;
                            uint8_t* drow = dst + kk * AT_ROW;
                            if (pc < 16) {
                                const uint4 w = ok ? *(const uint4*)(src + pc * 16) : make_uint4(0u, 0u, 0u, 0u);
                                const unsigned ws[4] = { w.x, w.y, w.z, w.w };
                                unsigned out[8];
                                for (int q = 0; q < 4; q++) {
                                    out[2 * q] = kv4_e4m3x4(ws[q]);
                                    out[2 * q + 1] = kv4_e4m3x4(ws[q] >> 16);
                                }
                                *(uint4*)(drow + pc * 32) = make_uint4(out[0], out[1], out[2], out[3]);
                                *(uint4*)(drow + pc * 32 + 16) = make_uint4(out[4], out[5], out[6], out[7]);
                            } else if (pc < 18) {
                                cp16z(sdst + kk * 32 + (pc - 16) * 16, src + 256 + (pc - 16) * 16, ok);
                            } else {
                                cp16z(drow + 512 + (pc - 18) * 16, src + 288 + (pc - 18) * 16, ok);
                            }
                        }
                    } else {
                        // the tile's rows, 16 bytes a piece
                        // タイルの行を 16 バイトずつ
                        for (int x = t; x < KTT * 41; x += THREADS) {
                            const int kk = x / 41, pc = x - kk * 41, ok = j0 + kk < n;
                            const long long key = !ok ? 0 : keyf(kk);
                            cp16z(dst + kk * AT_ROW + pc * 16, lc + key * AT_ROW + pc * 16, ok);
                        }
                    }
                };
                if (staged) stage([&](int kk) -> long long { return s_key[ti & 1][kk]; });
                else stage([&](int kk) -> long long { return pool_row(pmap, p < 2048 ? j0 + kk : tok[j0 + kk], psh); });
            }
            cp_commit();
            if (PIPE && ti == 0) continue;
            const int i0 = start + (ti - PIPE) * KTT;
            sk = sk0 + (PIPE ? ((ti - 1) & 1) * KTT * AR : 0);
            const uint8_t* sbt = sbs + (PIPE ? ((ti - 1) & 1) * KTT * 32 : 0);   // this tile's block scales (KV4) / このタイルのブロックのスケール (KV4)
            if (PIPE) cp_wait1();
            else cp_wait0();
            __syncthreads();
            if (staged && ti + 1 >= 2 && ti + 1 < ntile) key_rows(ti + 1);
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            unsigned af[4];
            const uint8_t* krow = sk + (8 * warp + (lane & 7)) * AR + (lane >> 3) * 16;
            if (P4) {
                // 128 latent dims a load (4 pieces of 32 codes); piece r gives this thread codes 32 r + 4 qq .. + 3 in its
                // low 16 bits and 32 r + 16 + 4 qq .. + 3 in its high 16: the e4m3 words the 656-byte rows held, and the
                // same MMAs in the same order (r ascending, half inner)
                // 1 回のロードで潜在次元 128 (コード 32 個の片 4 つ)。片 r はこのスレッドにコード 32 r + 4 qq .. + 3 を
                // 下位 16 ビットに、32 r + 16 + 4 qq .. + 3 を上位 16 ビットに渡す: 656 バイトの行にあった e4m3 の語と
                // 同じで、MMA も同じ順 (r の昇順、half が内側)
                const uint8_t* srow = sk + (8 * warp + g) * AR + 256;   // key 8 warp + g's block scales / ブロックのスケール
                for (int s8 = 0; s8 < 4; s8++) {
                    unsigned raw[4];
                    ldsm4(raw, krow + s8 * 64);
                    const unsigned sc8a = *(const unsigned*)(srow + 8 * s8), sc8b = *(const unsigned*)(srow + 8 * s8 + 4);
                    for (int i = 0; i < 4; i++) {
                        const int rr = 4 * s8 + i, d = 32 * rr + 4 * qq;
                        const unsigned long long v0 = *(const unsigned long long*)(ql[0] + d), v2 = *(const unsigned long long*)(ql[0] + d + 16);
                        const unsigned long long v1 = NH == 16 ? *(const unsigned long long*)(ql[1] + d) : v0;
                        const unsigned long long v3 = NH == 16 ? *(const unsigned long long*)(ql[1] + d + 16) : v2;
                        const unsigned scw = i < 2 ? sc8a : sc8b, sb2 = 2 * (i & 1);   // blocks 2 rr, 2 rr + 1
                        const unsigned f0 = e4m3x2_f16x2(__byte_perm(scw, 0u, 0x11u * sb2));
                        const unsigned f1 = e4m3x2_f16x2(__byte_perm(scw, 0u, 0x11u * (sb2 + 1)));
                        const unsigned wlo = kv4_e4m3x4(raw[i]), whi = kv4_e4m3x4(raw[i] >> 16);
                        for (int half = 0; half < 2; half++) {
                            const int sh = 32 * half;
                            af[0] = (unsigned)(v0 >> sh); af[1] = (unsigned)(v1 >> sh); af[2] = (unsigned)(v2 >> sh); af[3] = (unsigned)(v3 >> sh);
                            mma_bf16(acc, af, e4m3x2_mul_bf16x2(wlo >> (16 * half), f0), e4m3x2_mul_bf16x2(whi >> (16 * half), f1));
                        }
                    }
                }
            } else
            for (int s4 = 0; s4 < 8; s4++) {   // 64 latent dims a load / 1 回のロードで潜在次元 64
                unsigned raw[4];
                ldsm4(raw, krow + s4 * 64);
                // KV4: key 8 warp + g's block scales 4 s4 .. 4 s4 + 3 in one load
                // KV4: key 8 warp + g のブロックのスケール 4 s4 .. 4 s4 + 3 を 1 回で読む
                const unsigned sc4 = KV4 ? *(const unsigned*)(sbt + (8 * warp + g) * 32 + 4 * s4) : 0u;
                for (int hp = 0; hp < 2; hp++) {
                    const int d = 32 * (2 * s4 + hp) + 4 * qq;
                    const unsigned long long v0 = *(const unsigned long long*)(ql[0] + d), v2 = *(const unsigned long long*)(ql[0] + d + 16);
                    const unsigned long long v1 = NH == 16 ? *(const unsigned long long*)(ql[1] + d) : v0;
                    const unsigned long long v3 = NH == 16 ? *(const unsigned long long*)(ql[1] + d + 16) : v2;
                    // KV4: blocks 4 s4 + 2 hp and + 1, each scale twice, as f16x2
                    // KV4: ブロック 4 s4 + 2 hp と + 1。スケールを 2 つ並べて f16x2 に
                    const unsigned f0 = KV4 ? e4m3x2_f16x2(__byte_perm(sc4, 0u, 0x11u * (2 * hp))) : 0u;
                    const unsigned f1 = KV4 ? e4m3x2_f16x2(__byte_perm(sc4, 0u, 0x11u * (2 * hp + 1))) : 0u;
                    for (int half = 0; half < 2; half++) {
                        const int sh = 32 * half;
                        af[0] = (unsigned)(v0 >> sh); af[1] = (unsigned)(v1 >> sh); af[2] = (unsigned)(v2 >> sh); af[3] = (unsigned)(v3 >> sh);
                        if (KV4) {
                            mma_bf16(acc, af, e4m3x2_mul_bf16x2(raw[2 * hp] >> (16 * half), f0), e4m3x2_mul_bf16x2(raw[2 * hp + 1] >> (16 * half), f1));
                        } else {
                            mma_bf16(acc, af, e4m3x2_bf16x2(raw[2 * hp] >> (16 * half)), e4m3x2_bf16x2(raw[2 * hp + 1] >> (16 * half)));
                        }
                    }
                }
            }
            const int k0 = 8 * warp + 2 * qq;   // this thread's 2 keys of the tile / このスレッドが持つタイルの 2 キー
            const float ks0 = *(const float*)(sk + k0 * AR + (P4 ? 288 : 512)), ks1 = *(const float*)(sk + (k0 + 1) * AR + (P4 ? 288 : 512));
            acc[0] = acc[0] * ks0; acc[1] = acc[1] * ks1; acc[2] = acc[2] * ks0; acc[3] = acc[3] * ks1;
            for (int c2 = 0; c2 < 2; c2++) {   // 32 rope dims a load / 1 回のロードで rope 次元 32
                unsigned rb[4];
                ldsm4(rb, krow + (P4 ? 304 : 528) + c2 * 64);
                for (int cc = 0; cc < 2; cc++) {
                    const int d = 16 * (2 * c2 + cc) + 2 * qq;
                    af[0] = *(const unsigned*)(qr[0] + d); af[2] = *(const unsigned*)(qr[0] + d + 8);
                    af[1] = NH == 16 ? *(const unsigned*)(qr[1] + d) : af[0];
                    af[3] = NH == 16 ? *(const unsigned*)(qr[1] + d + 8) : af[2];
                    mma_bf16(acc, af, rb[2 * cc], rb[2 * cc + 1]);
                }
            }
            const int ok0 = i0 + k0 < n, ok1 = i0 + k0 + 1 < n;
            float sc[2][2], alpha[2], nm[2];
            for (int rs = 0; rs < RS; rs++) {
                sc[rs][0] = ok0 ? acc[2 * rs] * a.scale : ninf;
                sc[rs][1] = ok1 ? acc[2 * rs + 1] * a.scale : ninf;
                float mx = fmaxf(sc[rs][0], sc[rs][1]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                if (qq == 0) smax[(g + 8 * rs) * W + warp] = mx;
            }
            __syncthreads();
            for (int rs = 0; rs < RS; rs++) {
                float tm = smax[(g + 8 * rs) * W];
                for (int w = 1; w < W; w++) tm = fmaxf(tm, smax[(g + 8 * rs) * W + w]);
                const bool active = tm != ninf;
                nm[rs] = active ? fmaxf(tm, m[rs]) : m[rs];
                alpha[rs] = active ? (m[rs] == ninf ? 0.0f : t_exp(m[rs] - nm[rs])) : 1.0f;
                const float p0 = ok0 && active ? t_exp(sc[rs][0] - nm[rs]) : 0.0f;
                const float p1 = ok1 && active ? t_exp(sc[rs][1] - nm[rs]) : 0.0f;
                *(unsigned*)(spr + (g + 8 * rs) * KTT + k0) = f32_bf16x2(p0 * ks0, p1 * ks1);
                float sm = p0 + p1;
                sm = sm + __shfl_xor_sync(0xffffffffu, sm, 2);
                sm = sm + __shfl_xor_sync(0xffffffffu, sm, 1);
                if (qq == 0) ssum[(g + 8 * rs) * W + warp] = sm;
            }
            __syncthreads();
            for (int rs = 0; rs < RS; rs++) {
                const float* v = ssum + (g + 8 * rs) * W;
                const float tot = W == 4 ? (v[0] + v[2]) + (v[1] + v[3]) : ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
                l[rs] = fmaf(alpha[rs], l[rs], tot);
                m[rs] = nm[rs];
            }
            const float al1 = NH == 16 ? alpha[1] : alpha[0];
            for (int i = 0; i < NT; i++) {
                o[i][0] = o[i][0] * alpha[0]; o[i][1] = o[i][1] * alpha[0];
                o[i][2] = o[i][2] * al1; o[i][3] = o[i][3] * al1;
            }
            for (int rep = 0; rep < KTT / 32; rep++) {
                // this thread's 4 keys of the tile, and + 16
                // このスレッドが持つタイルの 4 キーと、その + 16
                const int kq = 32 * rep + 4 * qq;
                const unsigned long long v0 = *(const unsigned long long*)(spr + g * KTT + kq), v2 = *(const unsigned long long*)(spr + g * KTT + kq + 16);
                const unsigned long long v1 = NH == 16 ? *(const unsigned long long*)(spr + (g + 8) * KTT + kq) : v0;
                const unsigned long long v3 = NH == 16 ? *(const unsigned long long*)(spr + (g + 8) * KTT + kq + 16) : v2;
#ifdef TF_LDSM_B8
                // the 16-column blocks jj of the warp's columns (columns g and g + 8 of a block are o[2 jj], o[2 jj + 1]):
                // keys 32 rep .. + 15 and + 16 .. + 31 by one transposed byte load, this thread's keys kq .. kq + 3
                // (half 0: kq, kq + 1; half 1: kq + 2, kq + 3) a register; each accumulator still takes half 0 then 1
                // warp の列の 16 列ブロック jj (ブロックの列 g と g + 8 が o[2 jj]、o[2 jj + 1]): key 32 rep .. + 15 と
                // + 16 .. + 31 を転置の 1 バイト読み 1 回で、このスレッドの key kq .. kq + 3 (half 0: kq, kq + 1、
                // half 1: kq + 2, kq + 3) を 1 レジスタに。各アキュムレータは従来どおり half 0、1 の順に足す
                unsigned ah[2][4];
                for (int half = 0; half < 2; half++) {
                    const int sh = 32 * half;
                    ah[half][0] = (unsigned)(v0 >> sh); ah[half][1] = (unsigned)(v1 >> sh); ah[half][2] = (unsigned)(v2 >> sh); ah[half][3] = (unsigned)(v3 >> sh);
                }
                const int bo = warp * (32 / W);
                unsigned sa[2] = { 0u, 0u }, sb[2] = { 0u, 0u }, sc[2] = { 0u, 0u }, sd[2] = { 0u, 0u };
                if (P4) {
                    // the warp's 32-column pieces s (16 bytes of codes): byte g of keys 4 qq .. + 3 holds columns 2 L,
                    // 2 L + 1 and byte g + 8 columns 2 L + 8, + 9 (L = 8 ((g >> 1) & 1) + 2 (g >> 2) + (g & 1), the
                    // piece order), one accumulator each: o[4 s + set], set 0 / 1 / 2 / 3 = 2 L / + 1 / + 8 / + 9 (the
                    // epilogue writes them back); every column's sum in the same order as the 656-byte path
                    // warp の 32 列の片 s (コード 16 バイト): key 4 qq .. + 3 のバイト g が列 2 L, 2 L + 1、バイト g + 8 が
                    // 列 2 L + 8, + 9 を持つ (L = 8 ((g >> 1) & 1) + 2 (g >> 2) + (g & 1)、片の並び)。それぞれ 1 つの
                    // アキュムレータ o[4 s + set] (set 0 / 1 / 2 / 3 = 2 L / + 1 / + 8 / + 9、epilogue が書き戻す)。
                    // どの列の和も 656 バイトの経路と同じ順
                    const uint8_t* sbk = sk + 256 + warp * (32 / W) + ((g >> 1) & 1);
                    for (int s = 0; s < NT / 4; s++) {
                        unsigned kv[4];
                        ldsm_b8t_x2(kv, sk + (32 * rep + lane) * AR + warp * (256 / W) + 16 * s);
                        unsigned e4[4][2];   // [set][keys 0 .. 15 / 16 .. 31]: e4m3 codes of keys 4 qq .. 4 qq + 3
                        for (int mm = 0; mm < 2; mm++)
                            for (int b = 0; b < 2; b++)
                                for (int nb = 0; nb < 2; nb++) {
                                    const unsigned x = (kv[2 * mm + b] >> (4 * nb)) & 0x0F0F0F0Fu, y = x | (x >> 4);
                                    e4[2 * b + nb][mm] = kv4_e4m3x4(__byte_perm(y, 0u, 0x4420));
                                }
                        for (int half = 0; half < 2; half++) {
                            const int ka = kq + 2 * half;
                            const uint8_t* sbb = sbk + 2 * s;
                            const unsigned f0 = e4m3x2_f16x2((unsigned)sbb[ka * AR] | ((unsigned)sbb[(ka + 1) * AR] << 8));
                            const unsigned f1 = e4m3x2_f16x2((unsigned)sbb[(ka + 16) * AR] | ((unsigned)sbb[(ka + 17) * AR] << 8));
                            for (int set = 0; set < 4; set++)
                                mma_bf16(o[4 * s + set], ah[half], e4m3x2_mul_bf16x2(e4[set][0] >> (16 * half), f0), e4m3x2_mul_bf16x2(e4[set][1] >> (16 * half), f1));
                        }
                    }
                } else
                for (int jj = 0; jj < NT / 2; jj++) {
                    unsigned kv[4];
                    ldsm_b8t_x2(kv, sk + (32 * rep + lane) * AT_ROW + warp * (512 / W) + 16 * jj);
                    // KV4: block warp (32 / W) + jj of keys kq + 2 half (+1) and + 16 (+17), 4 blocks a word
                    // KV4: key kq + 2 half (+1) と + 16 (+17) のブロック warp (32 / W) + jj、1 語に 4 ブロック
                    if (KV4 && (jj & 3) == 0)
                        for (int half = 0; half < 2; half++) {
                            const int ka = kq + 2 * half, u = jj >> 2;
                            sa[half] = *(const unsigned*)(sbt + ka * 32 + bo + 4 * u);
                            sb[half] = *(const unsigned*)(sbt + (ka + 1) * 32 + bo + 4 * u);
                            sc[half] = *(const unsigned*)(sbt + (ka + 16) * 32 + bo + 4 * u);
                            sd[half] = *(const unsigned*)(sbt + (ka + 17) * 32 + bo + 4 * u);
                        }
                    for (int half = 0; half < 2; half++) {
                        const int sh = 16 * half;
                        if (KV4) {
                            const int bj = jj & 3;
                            const unsigned fs0 = e4m3x2_f16x2(__byte_perm(sa[half], sb[half], bj | ((bj + 4) << 4)));
                            const unsigned fs1 = e4m3x2_f16x2(__byte_perm(sc[half], sd[half], bj | ((bj + 4) << 4)));
                            mma_bf16(o[2 * jj], ah[half], e4m3x2_mul_bf16x2(kv[0] >> sh, fs0), e4m3x2_mul_bf16x2(kv[2] >> sh, fs1));
                            mma_bf16(o[2 * jj + 1], ah[half], e4m3x2_mul_bf16x2(kv[1] >> sh, fs0), e4m3x2_mul_bf16x2(kv[3] >> sh, fs1));
                        } else {
                            mma_bf16(o[2 * jj], ah[half], e4m3x2_bf16x2(kv[0] >> sh), e4m3x2_bf16x2(kv[2] >> sh));
                            mma_bf16(o[2 * jj + 1], ah[half], e4m3x2_bf16x2(kv[1] >> sh), e4m3x2_bf16x2(kv[3] >> sh));
                        }
                    }
                }
#else
                for (int half = 0; half < 2; half++) {
                    const int sh = 32 * half;
                    af[0] = (unsigned)(v0 >> sh); af[1] = (unsigned)(v1 >> sh); af[2] = (unsigned)(v2 >> sh); af[3] = (unsigned)(v3 >> sh);
                    const uint8_t* kb = sk + (kq + 2 * half) * AT_ROW + warp * (512 / W) + g;
                    // KV4: keys kq + 2 half (+1) and + 16 (+17); column warp (512 / W) + g + 8 i is in block
                    // warp (32 / W) + (i >> 1), so the warp's 32 / W blocks of each key come in 32 / W / 4 words
                    // KV4: key kq + 2 half (+1) と + 16 (+17)。列 warp (512 / W) + g + 8 i はブロック
                    // warp (32 / W) + (i >> 1) にあるので、warp の 32 / W 個のブロックを key ごとに 32 / W / 4 語で読む
                    const int ka = kq + 2 * half, bo = warp * (32 / W);
                    unsigned sa[2] = { 0u, 0u }, sb[2] = { 0u, 0u }, sc[2] = { 0u, 0u }, sd[2] = { 0u, 0u };
                    if (KV4) {
                        for (int u = 0; u < 32 / W / 4; u++) {
                            sa[u] = *(const unsigned*)(sbt + ka * 32 + bo + 4 * u);
                            sb[u] = *(const unsigned*)(sbt + (ka + 1) * 32 + bo + 4 * u);
                            sc[u] = *(const unsigned*)(sbt + (ka + 16) * 32 + bo + 4 * u);
                            sd[u] = *(const unsigned*)(sbt + (ka + 17) * 32 + bo + 4 * u);
                        }
                    }
                    unsigned fs0 = 0u, fs1 = 0u;
                    for (int i = 0; i < NT; i++) {
                        const unsigned b0 = (unsigned)kb[i * 8] | ((unsigned)kb[AT_ROW + i * 8] << 8);
                        const unsigned b1 = (unsigned)kb[16 * AT_ROW + i * 8] | ((unsigned)kb[17 * AT_ROW + i * 8] << 8);
                        if (KV4) {
                            if ((i & 1) == 0) {   // block j for columns i and i + 1 / 列 i と i + 1 のブロック j
                                const int j = i >> 1, u = j >> 2, bj = j & 3;
                                fs0 = e4m3x2_f16x2(__byte_perm(sa[u], sb[u], bj | ((bj + 4) << 4)));
                                fs1 = e4m3x2_f16x2(__byte_perm(sc[u], sd[u], bj | ((bj + 4) << 4)));
                            }
                            mma_bf16(o[i], af, e4m3x2_mul_bf16x2(b0, fs0), e4m3x2_mul_bf16x2(b1, fs1));
                        } else {
                            mma_bf16(o[i], af, e4m3x2_bf16x2(b0), e4m3x2_bf16x2(b1));
                        }
                    }
                }
#endif
            }
            __syncthreads();
        }
    }
    const int col0 = warp * (512 / W) + 2 * qq;
    for (int rs = 0; rs < RS; rs++) {
        const int j = j0 + g + 8 * rs;
        if (DL) {
            // The DCP path's one chunk and tf_glm53_n_merge_lse in one: PO = OSEND
            // [destination = j / h, row, j % h, 512] bf16 and PL = LSEND fp32. The merge's arithmetic for a single
            // chunk is kept as it stands (its running maximum starts at -inf, its sums at 0): o = fma(0, fa, co fb),
            // l = fma(0, fa, cl fb) with fa = 0 and fb = exp(cm - cm) for a chunk with keys.
            // DCP 経路のチャンク 1 つと tf_glm53_n_merge_lse を 1 つにしたもの: PO = OSEND
            // [送り先 = j / h, row, j % h, 512] bf16、PL = LSEND fp32。チャンク 1 つに対する統合の計算はそのまま残す
            // (最大値は -inf から、和は 0 から始まる): キーのあるチャンクでは fa = 0、fb = exp(cm - cm) として
            // o = fma(0, fa, co fb)、l = fma(0, fa, cl fb)。
            const float cm = m[rs], cl = l[rs];
            const bool active = cl > 0.0f;
            const float next_m = active ? fmaxf(ninf, cm) : ninf;
            const float fa = active ? 0.0f : 1.0f;
            const float fb = active ? t_exp(cm - next_m) : 0.0f;
            const float ll = fmaf(0.0f, fa, cl * fb);
            const bool has = ll > 0.0f;
            const float ld = has ? ll : 1.0f;
            const long long dst = ((long long)(j / a.h) * a.rows + r) * a.h + j % a.h;
            uint16_t* out = (uint16_t*)a.po + dst * 512 + col0;
            if (P4) {   // the P4 accumulators' columns (see the PV loop) / P4 のアキュムレータの列 (PV のループを参照)
                uint16_t* ob = (uint16_t*)a.po + dst * 512 + warp * (512 / W) + 16 * (qq & 1) + 4 * (qq >> 1);
                for (int s4 = 0; s4 < NT / 4; s4++)
                    for (int pp = 0; pp < 2; pp++)
                        for (int cc = 0; cc < 2; cc++) {
                            const float o0 = has ? fmaf(0.0f, fa, o[4 * s4 + 2 * pp][2 * rs + cc] * fb) : 0.0f, o1 = has ? fmaf(0.0f, fa, o[4 * s4 + 2 * pp + 1][2 * rs + cc] * fb) : 0.0f;
                            *(unsigned*)(ob + 32 * s4 + 8 * pp + 2 * cc) = has ? f32_bf16x2(t_div(o0, ld), t_div(o1, ld)) : f32_bf16x2(0.0f, 0.0f);
                        }
            } else if (has) {
                for (int i = 0; i < NT; i++) {
                    const float o0 = fmaf(0.0f, fa, o[i][2 * rs] * fb), o1 = fmaf(0.0f, fa, o[i][2 * rs + 1] * fb);
                    *(unsigned*)(out + i * 8) = f32_bf16x2(t_div(o0, ld), t_div(o1, ld));
                }
            } else {   // a head row without a key: zeros / キーのないヘッド行: ゼロ
                for (int i = 0; i < NT; i++) *(unsigned*)(out + i * 8) = f32_bf16x2(0.0f, 0.0f);
            }
            if (warp == 0 && qq == 0) ((float*)a.pl)[dst] = has ? logf(ld) + next_m : ninf;
        } else if (a.direct) {
            uint16_t* out = (uint16_t*)a.po + ((long long)r * HT + j) * 512 + col0;
            if (P4) {
                uint16_t* ob = (uint16_t*)a.po + ((long long)r * HT + j) * 512 + warp * (512 / W) + 16 * (qq & 1) + 4 * (qq >> 1);
                for (int s4 = 0; s4 < NT / 4; s4++)
                    for (int pp = 0; pp < 2; pp++)
                        for (int cc = 0; cc < 2; cc++)
                            *(unsigned*)(ob + 32 * s4 + 8 * pp + 2 * cc) = f32_bf16x2(t_div(o[4 * s4 + 2 * pp][2 * rs + cc], l[rs]), t_div(o[4 * s4 + 2 * pp + 1][2 * rs + cc], l[rs]));
            } else
            for (int i = 0; i < NT; i++) *(unsigned*)(out + i * 8) = f32_bf16x2(t_div(o[i][2 * rs], l[rs]), t_div(o[i][2 * rs + 1], l[rs]));
        } else {
            const long long base = ((long long)c * a.rows + r) * HT + j;
            float* out = (float*)a.po + base * 512 + col0;
            if (P4) {
                float* ob = (float*)a.po + base * 512 + warp * (512 / W) + 16 * (qq & 1) + 4 * (qq >> 1);
                for (int s4 = 0; s4 < NT / 4; s4++)
                    for (int pp = 0; pp < 2; pp++)
                        for (int cc = 0; cc < 2; cc++)
                            *(float2*)(ob + 32 * s4 + 8 * pp + 2 * cc) = make_float2(o[4 * s4 + 2 * pp][2 * rs + cc], o[4 * s4 + 2 * pp + 1][2 * rs + cc]);
            } else
            for (int i = 0; i < NT; i++) *(float2*)(out + i * 8) = make_float2(o[i][2 * rs], o[i][2 * rs + 1]);
            if (warp == 0 && qq == 0) {
                ((float*)a.pm)[base] = m[rs];
                ((float*)a.pl)[base] = l[rs];
            }
        }
    }
}
#define AT_TILES(W) ((W) == 4 ? 2 : 1)   // key tile buffers / キーのタイル用バッファの数
// key tiles, bf16(p), max / sum partials
// キーのタイル、bf16(p)、max / sum の部分結果
#define AT_SMEM(W) (AT_TILES(W) * 8 * (W) * AT_ROW + 16 * 8 * (W) * 2 + 2 * 16 * (W) * 4)
// KV4 entries add the tiles' block scales (32 bytes a key) past AT_SMEM(W)
// KV4 の実体は AT_SMEM(W) の後ろにタイルのブロックのスケール (key あたり 32 バイト) を足す
// AT_RW(KV4): a tile row's bytes (P4: the cache's 432) / タイルの行のバイト数 (P4 ではキャッシュの 432)
#define AT_RW(KV4) (((KV4) && KV4P_ON) ? KV4_ROW : AT_ROW)
#define AT_SMEM_K(W, KV4) (AT_TILES(W) * 8 * (W) * AT_RW(KV4) + 16 * 8 * (W) * 2 + 2 * 16 * (W) * 4)
// P4 with W 8 (prompt chunks): 30,720 bytes a block, so three fit an SM when the registers let them (<= 85)
// P4 で W 8 (prompt チャンク): ブロックあたり 30,720 バイトなので、レジスタが許せば (85 以下) 1 SM に 3 つ載る
#if KV4P_ON
#define AT_LB3 __launch_bounds__(256, 3)
#else
#define AT_LB3
#endif
#define AT_ENTRY_K(name, NH, W, DL, KV4) AT_ENTRY_KB(name, NH, W, DL, KV4, )
#define AT_ENTRY_KB(name, NH, W, DL, KV4, LB) \
__global__ void LB name(AttnArgs a) { \
    extern __shared__ __align__(16) uint8_t at[]; \
    uint8_t* sp = at + AT_TILES(W) * 8 * (W) * AT_RW(KV4); \
    attn_body(a, NH, W, DL, KV4, 1, at, (uint16_t*)sp, (float*)(sp + 16 * 8 * (W) * 2), (float*)(sp + 16 * 8 * (W) * 2 + 16 * (W) * 4), at + AT_SMEM_K(W, KV4)); \
}
// HG head groups a block: the key tiles once, then each group's bf16(p) and max / sum partials (then the KV4 scales)
// HG 組のブロック: key のタイルは 1 回分、続いて組ごとの bf16(p) と max / sum の部分結果 (その後ろに KV4 のスケール)
#define AT_TILES_G(W, HG) (((W) == 4 || (HG) == 2) ? 2 : 1)   // as attn_body's PIPE / attn_body の PIPE と同じ
#define AT_SMEM_G(W, HG) (AT_TILES_G(W, HG) * 8 * (W) * AT_ROW + (HG) * (16 * 8 * (W) * 2 + 2 * 16 * (W) * 4))
#define AT_ENTRY_G(name, NH, W, DL, KV4, HG) \
__global__ void __launch_bounds__(32 * (W) * (HG)) name(AttnArgs a) { \
    extern __shared__ __align__(16) uint8_t at[]; \
    uint8_t* sp = at + AT_TILES_G(W, HG) * 8 * (W) * AT_RW(KV4); \
    attn_body(a, NH, W, DL, KV4, HG, at, (uint16_t*)sp, (float*)(sp + (HG) * 16 * 8 * (W) * 2), (float*)(sp + (HG) * (16 * 8 * (W) * 2 + 16 * (W) * 4)), at + AT_TILES_G(W, HG) * 8 * (W) * AT_RW(KV4) + (HG) * (16 * 8 * (W) * 2 + 2 * 16 * (W) * 4)); \
}
#define AT_ENTRY(name, NH, W, DL) AT_ENTRY_K(name, NH, W, DL, 0)
AT_ENTRY(tf_glm53_n_attn_w4h8, 8, 4, 0)
AT_ENTRY(tf_glm53_n_attn_w4h16, 16, 4, 0)
AT_ENTRY(tf_glm53_n_attn_w8h8, 8, 8, 0)
AT_ENTRY(tf_glm53_n_attn_w8h16, 16, 8, 0)
// a prompt chunk under DCP 2 (one chunk of 2,048 keys): the normalized partials and their log-sum-exps for the exchange
// DCP 2 での prompt チャンク (2,048 キーのチャンク 1 つ): 交換用に正規化した部分結果とその log-sum-exp
AT_ENTRY(tf_glm53_n_attn_w8h16_dl, 16, 8, 1)
// the same over FP4 caches (TF_GLM53_KV=fp4)
// FP4 キャッシュ上の同じもの (TF_GLM53_KV=fp4)
AT_ENTRY_K(tf_glm53_n_attn_w4h8_k4, 8, 4, 0, 1)
AT_ENTRY_K(tf_glm53_n_attn_w4h16_k4, 16, 4, 0, 1)
AT_ENTRY_KB(tf_glm53_n_attn_w8h8_k4, 8, 8, 0, 1, AT_LB3)
AT_ENTRY_KB(tf_glm53_n_attn_w8h16_k4, 16, 8, 0, 1, AT_LB3)
AT_ENTRY_KB(tf_glm53_n_attn_w8h16_dl_k4, 16, 8, 1, 1, AT_LB3)
// TP=4 prompt chunks under DCP 2 (32 gathered heads): the two programs of 16 heads in one block
// TP=4 の DCP 2 の prompt チャンク (集めたヘッド 32): 16 ヘッドの program 2 つを 1 ブロックで
AT_ENTRY_G(tf_glm53_n_attn_w8h16x2, 16, 8, 0, 0, 2)
AT_ENTRY_G(tf_glm53_n_attn_w8h16x2_k4, 16, 8, 0, 1, 2)
AT_ENTRY_G(tf_glm53_n_attn_w8h16x2_dl, 16, 8, 1, 0, 2)
AT_ENTRY_G(tf_glm53_n_attn_w8h16x2_dl_k4, 16, 8, 1, 1, 2)

// four 8 x 8 matrices of 16-bit values, transposed: a thread gets (rows 2 (lane & 3), + 1; column lane >> 2) of each
// 16 ビット値の 8 x 8 行列 4 つを転置で読む: スレッドは各行列の (行 2 (lane & 3) と + 1、列 lane >> 2) を受け取る
static __device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
static __device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ float f16_f32(uint16_t v) {
    float f;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(v));
    return f;
}

// EXL3 prefill GEMM: OUT[m, block] = ((xh[m] @ W[:, block]) @ H) scale svh, a block = 128 columns. xh fp16 [M, K], W
// fp16 [K, N] (the decoded W_q), H bf16 [128, 128], svh fp16 [N]; OUT fp32 or bf16, rows o_stride.
// An element, as the original computes it: acc = the K products in 16-wide steps, ascending, on HMMA.16816.F32
// (fp16 in); top = bf16(acc), rest = bf16(acc - top); y = top . H then + rest . H, 8 + 8 HMMA.16816.F32.BF16 in one
// accumulator; y scale, then svh. The original's tiles only move the work around, so one layout serves every shape:
// grid (128-row blocks * 128-column blocks), 256 threads, warp w = rows 16 w .. 16 w + 15 across all 128 columns - its
// accumulator fragments are then exactly the A fragments of the Hadamard product (k = the column), with no trip through
// shared memory. K in GM_BK-wide stages, GM_STAGES - 1 in flight; the Hadamard matrix takes the stage area afterwards.
// Dynamic shared memory: GM_SMEM bytes.
// Two blocks an SM: a GB10 SM has 65,536 registers and 102,400 bytes of shared memory, so a 256-thread block shares an
// SM with a second one only at <= 128 registers a thread and <= 50,176 bytes. This kernel is built to fit
// (128 registers with GM_NTG 2 / GM_ETG 1, 49,152 bytes with GM_STAGES 3); with one block an SM the warps idle while
// their copies arrive. Check `nvcc -Xptxas -v` after any change here.
// Every counted loop is unrolled by pragma: left to the optimizer, the accumulator array stays in local memory.
// EXL3 prefill の GEMM: OUT[m, block] = ((xh[m] @ W[:, block]) @ H) scale svh (block は 128 列)。xh は fp16 [M, K]、
// W は fp16 [K, N] (復号した W_q)、H は bf16 [128, 128]、svh は fp16 [N]。OUT は fp32 か bf16 で、行の間隔は o_stride。
// 1 要素の計算は元と同じ: acc = K 個の積を 16 幅の段で昇順に HMMA.16816.F32 (入力 fp16) で足す。top = bf16(acc)、
// rest = bf16(acc - top)。y = top . H に続けて + rest . H を、1 つのアキュムレータで HMMA.16816.F32.BF16 を 8 + 8 回。
// y に scale、続けて svh を掛ける。元のタイル分割は仕事の配置を変えるだけなので、1 つの配置ですべての形状をまかなう:
// grid (128 行ブロック * 128 列ブロック)、256 スレッド、warp w は全 128 列にわたって行 16 w .. 16 w + 15 を持つ。すると
// アキュムレータのフラグメントがそのまま Hadamard 積の A フラグメント (k = 列) になり、共有メモリを経由しない。K は
// GM_BK 幅のステージで GM_STAGES - 1 個を先読みし、その後 Hadamard 行列がステージ領域を使う。動的共有メモリは GM_SMEM
// バイト。
// 1 SM に 2 ブロック: GB10 の SM はレジスタ 65,536 個と共有メモリ 102,400 バイトを持つので、256 スレッドのブロックが
// 2 つ目と SM を共有できるのは 1 スレッド 128 レジスタ以下かつ 50,176 バイト以下のときだけ。このカーネルはそこに
// 収まるように作ってある (GM_NTG 2 / GM_ETG 1 で 128 レジスタ、GM_STAGES 3 で 49,152 バイト)。
// 1 SM 1 ブロックだとコピーの到着を待つ間 warp が遊ぶ。ここを変えたら `nvcc -Xptxas -v` で確かめる。
// 数の決まったループはすべて pragma で展開する。最適化に任せるとアキュムレータの配列がローカルメモリに残る。
#ifndef GM_BK
#define GM_BK 32       // K columns a stage / 1 ステージの K 列数
#endif
#ifndef GM_STAGES
// stages of shared memory, GM_STAGES - 1 in flight (3 x 16,384 = 49,152 bytes, within the two-block limit)
// 共有メモリのステージ数。GM_STAGES - 1 個を先読みする (3 x 16,384 = 49,152 バイトで 2 ブロックの上限内)
#define GM_STAGES 3
#endif
#ifndef GM_GROUP
#define GM_GROUP 8    // row blocks a raster group / ラスタグループ 1 つの行ブロック数
#endif
#define GM_XROW (GM_BK * 2)
#define GM_WROW 256
// 16-byte pieces of a staged row sit at piece ^ (row & mask), so the 8 rows of an ldmatrix fall on distinct banks
// without padding (mask GM_XSW for x, 7 for w and the Hadamard matrix)
// ステージした行の 16 バイト片は piece ^ (row & mask) に置く。これで ldmatrix の
// 8 行がパディングなしで別々のバンクに載る (mask は x では GM_XSW、w と Hadamard 行列では 7)
#define GM_XSW (GM_BK / 8 - 1)
#define GM_STAGE (128 * GM_XROW + GM_BK * GM_WROW)
#define GM_SMEM (GM_STAGES * GM_STAGE)
#ifndef GM_OROW
#define GM_OROW 128    // a staged output row: 32 fp32 / ステージした出力の行: fp32 32 個
#endif
#ifndef GM_ETG
#define GM_ETG 1       // column tiles whose Hadamard chains interleave / Hadamard の連鎖を交互に進める列タイルの数
#endif
#ifndef GM_NTG
#define GM_NTG 2       // column tiles whose HMMAs interleave in the K loop / K ループで HMMA を交互に進める列タイルの数
#endif
#define GM_STR2(x) #x
#define GM_STR(x) GM_STR2(x)
static __device__ __forceinline__ void gm_wait(void) { asm volatile("cp.async.wait_group " GM_STR(GM_STAGES - 2) ";\n" ::); }
typedef struct { Dev x, w, h, svh, out; int m, o_stride, k, n, out_bf16; float scale; } GemmArgs;
// A thread's staging of one K step: GM_BK / 16 copies of x and of w. A copy instruction of a warp
// takes 32 consecutive 16-byte pieces - whole rows of the tile, whole cache lines. The sources of step 0 and the
// offsets inside a stage are worked out once (GmPlan); a step only adds its distance to the sources.
// 1 つの K 段でスレッドが運ぶ分: x と w をそれぞれ GM_BK / 16 回コピーする。warp のコピー命令 1 つは連続する
// 16 バイト片 32 個、つまりタイルの行全体、キャッシュライン全体を運ぶ。段 0 のコピー元とステージ内のオフセットは
// 1 回だけ求め (GmPlan)、各段はコピー元に自分の距離を足すだけ。
typedef struct { const uint8_t* xs; const uint8_t* ws; int xr, xd, wr, wp; } GmPlan;
static __device__ __forceinline__ void gm_plan(GemmArgs a, int m0, int n0, GmPlan* pl) {
    // copy j of a thread is piece (warp GM_BK / 16 + j) 32 + lane of the tile: j + 1 is 32 pieces on - 32 / (GM_BK / 8)
    // rows of x, 2 rows of w, the same piece of the row
    // スレッドのコピー j はタイルの片 (warp GM_BK / 16 + j) 32 + lane。j + 1 は 32 片先で、x では 32 / (GM_BK / 8)
    // 行先、w では 2 行先の、行内の同じ片
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, idx = warp * (GM_BK / 16) * 32 + lane;
    const int xr = idx / (GM_BK / 8), xp = idx % (GM_BK / 8), wr = idx >> 4, wp = idx & 15;
    pl->xs = (const uint8_t*)a.x + ((long long)(m0 + xr) * a.k) * 2 + xp * 16;
    pl->ws = (const uint8_t*)a.w + ((long long)wr * a.n + n0) * 2 + wp * 16;
    pl->xr = xr;
    // rows 32 / (GM_BK / 8) apart share the swizzle (a multiple of 4)
    // 32 / (GM_BK / 8) 行 (4 の倍数) 離れた行は同じスウィズルになる
    pl->xd = xr * GM_XROW + (xp ^ (xr & GM_XSW)) * 16;
    pl->wr = wr;
    pl->wp = wp;
}
static __device__ __forceinline__ void gm_load(GemmArgs a, const GmPlan* pl, uint8_t* st, int step, int m0) {
    const int xrows = 32 / (GM_BK / 8);
#pragma unroll
    for (int j = 0; j < GM_BK / 16; j++) {
        const int wr = pl->wr + 2 * j;
        cp16z(st + pl->xd + j * xrows * GM_XROW, pl->xs + ((long long)j * xrows * a.k + (long long)step * GM_BK) * 2, m0 + pl->xr + j * xrows < a.m);
        cp16(st + 128 * GM_XROW + wr * GM_WROW + (pl->wp ^ (wr & 7)) * 16, pl->ws + ((long long)(2 * j + step * GM_BK) * a.n) * 2);
    }
}
__global__ void tf_glm53_n_gemm(GemmArgs a) {
    extern __shared__ __align__(16) uint8_t gm[];
    // block order: GM_GROUP row blocks at a time across all column blocks, so their x tiles stay in the cache while the
    // column blocks go by (grid x = row blocks * column blocks)
    // ブロックの順序: GM_GROUP 個の行ブロックずつ全列ブロックを回り、列ブロックが進む間その x タイルをキャッシュに残す
    // (grid x = 行ブロック数 * 列ブロック数)
    const int nm = (a.m + 127) / 128, per = GM_GROUP * (a.n / 128), pid = blockIdx.x;
    const int first = pid / per * GM_GROUP, grows = nm - first < GM_GROUP ? nm - first : GM_GROUP;
    const int m0 = (first + pid % per % grows) * 128, n0 = (pid % per / grows) * 128;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int g = lane >> 2, qq = lane & 3, steps = a.k / GM_BK;
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    GmPlan pl;
    gm_plan(a, m0, n0, &pl);
#pragma unroll
    for (int s = 0; s < GM_STAGES - 1; s++) {
        if (s < steps) gm_load(a, &pl, gm + s * GM_STAGE, s, m0);
        cp_commit();
    }
    const int arow = (16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * GM_XROW, apc = lane >> 4, sw = lane & 7;
    const int brow = 128 * GM_XROW + ((lane & 7) + 8 * (lane >> 3)) * GM_WROW;
    // the stage step i computes from, the stage step i + GM_STAGES - 1 loads into
    // 段 i が計算に使うステージと、段 i + GM_STAGES - 1 が読み込むステージ
    int sc = 0, sn = GM_STAGES - 1;
    for (int i = 0; i < steps; i++) {
        gm_wait();
        __syncthreads();
        // the stage step i - 1 used: every thread is past it
        // 段 i - 1 が使ったステージ。全スレッドがそこを過ぎている
        if (i + GM_STAGES - 1 < steps) gm_load(a, &pl, gm + sn * GM_STAGE, i + GM_STAGES - 1, m0);
        cp_commit();
        const uint8_t* st = gm + sc * GM_STAGE;
        sc = sc + 1 == GM_STAGES ? 0 : sc + 1;
        sn = sn + 1 == GM_STAGES ? 0 : sn + 1;
        unsigned af[GM_BK / 16][4];
#pragma unroll
        for (int c = 0; c < GM_BK / 16; c++) ldsm4(af[c], st + arow + ((2 * c + apc) ^ (sw & GM_XSW)) * 16);
        // GM_NTG column tiles at a time: their k16 steps interleave (neighbouring HMMAs do not wait for each other) and
        // few B fragments are live at once, so the kernel stays within 128 registers a thread
        // (two blocks an SM, see above)
        // GM_NTG 個の列タイルずつ: k16 段が交互に進み (隣の HMMA が互いを待たない)、同時に生きる B
        // フラグメントも少ないので、1 スレッド 128 レジスタ以内に収まる (1 SM に 2 ブロック、上を参照)
#pragma unroll
        for (int n0t = 0; n0t < 16; n0t += GM_NTG) {
            unsigned b[GM_NTG][GM_BK / 8];
#pragma unroll
            for (int i = 0; i < GM_NTG; i++)
#pragma unroll
                for (int c2 = 0; c2 < GM_BK / 32; c2++) ldsm4t(b[i] + 4 * c2, st + brow + c2 * 32 * GM_WROW + ((n0t + i) ^ sw) * 16);
#pragma unroll
            for (int c = 0; c < GM_BK / 16; c++)
#pragma unroll
                for (int i = 0; i < GM_NTG; i++) mma_f16(acc[n0t + i], af[c], b[i][2 * c], b[i][2 * c + 1]);
        }
    }
    cp_wait0();
    __syncthreads();
    {   // the Hadamard matrix: 128 rows of 16 pieces, 8 a thread / Hadamard 行列: 16 片の行が 128 本、1 スレッド 8 片
#pragma unroll
        for (int j = 0; j < 8; j++) {   // a warp's instruction: 2 whole rows / warp の 1 命令で行 2 本分
            const int idx = (warp * 8 + j) * 32 + lane, hr = idx >> 4, hp = idx & 15;
            cp16(gm + hr * GM_WROW + (hp ^ (hr & 7)) * 16, (const uint8_t*)a.h + hr * 256 + hp * 16);
        }
        cp_commit();
    }
    // top / rest as A fragments: k16 step c = column tiles 2 c, 2 c + 1
    // A フラグメントとしての top / rest: k16 段 c = 列タイル 2 c と 2 c + 1
    unsigned tf[8][4], rf[8][4];
#pragma unroll
    for (int c = 0; c < 8; c++)
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int rs = 0; rs < 2; rs++) {
                const float e0 = acc[2 * c + i][2 * rs], e1 = acc[2 * c + i][2 * rs + 1];
                const unsigned tp = f32_bf16x2(e0, e1);
                tf[c][2 * i + rs] = tp;
                rf[c][2 * i + rs] = f32_bf16x2(e0 - __uint_as_float(tp << 16), e1 - __uint_as_float(tp & 0xffff0000u));
            }
    cp_wait0();
    __syncthreads();
    const uint16_t* svh = (const uint16_t*)a.svh + n0;
    const int hrow = ((lane & 7) + 8 * (lane >> 3)) * GM_WROW;
    // past the Hadamard rows: a 16-row tile a warp
    // Hadamard の行の後ろ: warp ごとに 16 行のタイル
    uint8_t* ow = gm + 128 * GM_WROW + warp * 16 * GM_OROW;
#pragma unroll
    // 4 column tiles = one 64-byte run of fp32 a row
    // 列タイル 4 つ = 行ごとに fp32 の 64 バイトの並び 1 つ
    for (int n4 = 0; n4 < 16; n4 += 4) {
#pragma unroll
        // GM_ETG tiles at a time: their 16-step chains interleave
        // GM_ETG 個のタイルずつ: 16 段の連鎖を交互に進める
        for (int e0 = 0; e0 < 4; e0 += GM_ETG) {
            unsigned b[GM_ETG][16];
            float y[GM_ETG][4];
#pragma unroll
            for (int i = 0; i < GM_ETG; i++) {
                y[i][0] = y[i][1] = y[i][2] = y[i][3] = 0.0f;
#pragma unroll
                for (int cp = 0; cp < 4; cp++) ldsm4t(b[i] + 4 * cp, gm + hrow + cp * 32 * GM_WROW + ((n4 + e0 + i) ^ sw) * 16);
            }
#pragma unroll
            for (int c = 0; c < 8; c++)
#pragma unroll
                for (int i = 0; i < GM_ETG; i++) mma_bf16(y[i], tf[c], b[i][2 * c], b[i][2 * c + 1]);
#pragma unroll
            for (int c = 0; c < 8; c++)
#pragma unroll
                for (int i = 0; i < GM_ETG; i++) mma_bf16(y[i], rf[c], b[i][2 * c], b[i][2 * c + 1]);
            // (y scale) svh into this warp's 16 x 32 staging tile, then out of it a row at a time in 16-byte pieces
            // (y scale) svh をこの warp の 16 x 32 のステージ用タイルに置き、そこから 1 行ずつ 16 バイト片で書き出す
#pragma unroll
            for (int i = 0; i < GM_ETG; i++) {
                const int ti = e0 + i, col = (n4 + ti) * 8 + 2 * qq;
                // two 2-byte loads: one 4-byte load of the pair (with its alignment test) takes the kernel
                // past 128 registers
                // 2 バイトのロード 2 回: ペアを 4 バイトで 1 回読むと (アライン判定つき) 128 レジスタを超える
                const float s0 = f16_f32(svh[col]), s1 = f16_f32(svh[col + 1]);
#pragma unroll
                for (int rs = 0; rs < 2; rs++) {
                    const float y0 = (y[i][2 * rs] * a.scale) * s0, y1 = (y[i][2 * rs + 1] * a.scale) * s1;
                    uint8_t* at = ow + (g + 8 * rs) * GM_OROW;
                    if (a.out_bf16) *(unsigned*)(at + (ti * 8 + 2 * qq) * 2) = f32_bf16x2(y0, y1);
                    else *(float2*)(at + (ti * 8 + 2 * qq) * 4) = make_float2(y0, y1);
                }
            }
        }
        // a store instruction writes whole 64-byte runs (4 lanes a run, 8 rows), so a cache line is never updated in
        // parts
        // ストア命令 1 つで 64 バイトの並び全体を書く (1 並びに 4 レーン、8 行) ので、
        // キャッシュラインが部分的に更新されることはない
        if (a.out_bf16) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int orow = g + 8 * j, row = m0 + 16 * warp + orow;
                if (row < a.m) *(uint4*)((uint16_t*)a.out + (long long)row * a.o_stride + n0 + n4 * 8 + 8 * qq) = *(const uint4*)(ow + orow * GM_OROW + 16 * qq);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int orow = g + 8 * (j >> 1), oc = 16 * (j & 1) + 4 * qq, row = m0 + 16 * warp + orow;
                if (row < a.m) *(uint4*)((float*)a.out + (long long)row * a.o_stride + n0 + n4 * 8 + oc) = *(const uint4*)(ow + orow * GM_OROW + oc * 4);
            }
        }
    }
}

// ---- The DFlash2 drafter's kernels. Its drafts are verified by the target, so they need only give the same result on
// every rank; they still follow the original's arithmetic. ----
// ---- DFlash2 drafter のカーネル。ドラフトはターゲットが検証するので、全ランクで同じ結果になりさえすればよい。それでも
// 元の計算に従う。 ----

// 64-value group sums: XS[m, g] = the fp32 sum of x[m, 64 g .. 64 g + 63] (bf16 rows x_stride apart): a thread
// adds 8 values in order, the group's 8 threads by a butterfly (4, 2, 1). grid (rows, ceil(k / 1024)), 128 threads.
// The 8 values stay eight 2-byte loads: one 16-byte load behind an alignment test was slower.
// 64 値グループの和: XS[m, g] = x[m, 64 g .. 64 g + 63] の fp32 和 (bf16 の行、間隔 x_stride): スレッドは
// 8 値を順に足し、グループの 8 スレッドは butterfly (4, 2, 1) で足す。grid (rows, ceil(k / 1024))、128 スレッド。8 値は
// 2 バイトのロード 8 回のまま。アライン判定つきの 16 バイトロード 1 回は遅かった。
typedef struct { Dev x, xs; int rows, x_stride, k; } GroupSumsArgs;
__global__ void tf_glm53_n_group_sums(GroupSumsArgs a) {
    const int m = blockIdx.x, t = threadIdx.x, g = blockIdx.y * 16 + (t >> 3), kg = a.k / 64;
    const uint16_t* x = (const uint16_t*)a.x + (long long)m * a.x_stride + g * 64 + (t & 7) * 8;
    float s = 0.0f;
    if (g < kg) {
        s = bf16_f32(x[0]) + bf16_f32(x[1]);
#pragma unroll
        for (int j = 2; j < 8; j++) s = s + bf16_f32(x[j]);
    }
    s = s + __shfl_xor_sync(0xffffffffu, s, 4);
    s = s + __shfl_xor_sync(0xffffffffu, s, 2);
    s = s + __shfl_xor_sync(0xffffffffu, s, 1);
    if ((t & 7) == 0 && g < kg) ((float*)a.xs)[(long long)m * kg + g] = s;
}

// SwiGLU with clipping: [gate | up] bf16 rows of 2 w -> OUT = bf16(bf16(silu(min(g, L))) clip(u, -L, L)) [rows, w] and
// XS = its 64-value group sums (fp32): a thread's 4 values in order, the group's 16 threads by a butterfly
// (8, 4, 2, 1). grid (rows, w / 512), 128 threads.
// クリップつき SwiGLU: 2 w 個の bf16 の行 [gate | up] -> OUT = bf16(bf16(silu(min(g, L))) clip(u, -L, L))
// [rows, w] と、その 64 値グループごとの和 XS (fp32): スレッドの 4 値を順に足し、グループの 16 スレッドは butterfly
// (8, 4, 2, 1) で足す。grid (rows, w / 512)、128 スレッド。
typedef struct { Dev gu, out, xs; int rows, w; float limit; } SwigluArgs;
__global__ void tf_glm53_n_swiglu(SwigluArgs a) {
    const int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x, d0 = cb * 512 + 4 * t;
    const uint16_t* gp = (const uint16_t*)a.gu + (long long)r * 2 * a.w + d0;
    uint16_t* out = (uint16_t*)a.out + (long long)r * a.w + d0;
    // a thread's 4 gates, 4 ups and 4 outputs: one 8-byte access each when the three are aligned (the buffers are)
    // スレッドの gate 4 つ、up 4 つ、出力 4 つ: 3 つともアライン済みなら (バッファはそうなっている) それぞれ
    // 8 バイトのアクセス 1 回
    const bool al = ((((size_t)gp) | ((size_t)(gp + a.w)) | (size_t)out) & 7) == 0;
    float av[4], gv[4], uv[4];
    if (al) {
        ld4_bf16v(gp, gv);
        ld4_bf16v(gp + a.w, uv);
    } else {
#pragma unroll
        for (int j = 0; j < 4; j++) { gv[j] = bf16_f32(gp[j]); uv[j] = bf16_f32(gp[a.w + j]); }
    }
    unsigned long long ov = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float g = fminf(gv[j], a.limit);
        const float u = fminf(fmaxf(uv[j], 0.0f - a.limit), a.limit);
        const uint16_t y = f32_bf16(bf16_f32(f32_bf16(t_div(g, t_exp(0.0f - g) + 1.0f))) * u);
        ov |= (unsigned long long)y << (16 * j);
        av[j] = bf16_f32(y);
    }
    if (al) *(unsigned long long*)out = ov;
    else {
#pragma unroll
        for (int j = 0; j < 4; j++) out[j] = (uint16_t)(ov >> (16 * j));
    }
    float s = ((av[0] + av[1]) + av[2]) + av[3];
    s = s + __shfl_xor_sync(0xffffffffu, s, 8);
    s = s + __shfl_xor_sync(0xffffffffu, s, 4);
    s = s + __shfl_xor_sync(0xffffffffu, s, 2);
    s = s + __shfl_xor_sync(0xffffffffu, s, 1);
    if ((t & 15) == 0) ((float*)a.xs)[(long long)r * (a.w / 64) + cb * 8 + (t >> 4)] = s;
}

// The drafter's two-tap grouped dynamic convolution over the block rows: y = bf16(x k0 + prev k1) with
// k = bf16(base + dyn[group]) (prev: the row above, 0 for row 0), then with a residual bf16(res + y). Elementwise: the
// products of two bf16 are exact, so only the roundings to bf16 matter. grid (rows, ceil(d / 1024)), 128 threads. A
// column's group is c / gs by division, written at both uses; a shift behind a power-of-two test (t_quot), once into a
// local, was slower.
// drafter の 2 タップのグループ別動的畳み込み (ブロックの行方向): y = bf16(x k0 + prev k1)、k = bf16(base + dyn[group])
// (prev は 1 つ上の行、行 0 では 0)。残差があれば bf16(res + y)。要素ごとの計算で、bf16 どうしの積は厳密なので、
// 効くのは bf16 への丸めだけ。grid (rows, ceil(d / 1024))、128 スレッド。列のグループは c / gs を除算で、使う
// 2 か所それぞれに書く。2 のべき乗判定つきのシフト (t_quot) を 1 回だけローカルに求める形は遅かった。
typedef struct { Dev x, dyn, base, res, out; int rows, d, gs, branch, has_res, vec; } DconvArgs;
__global__ void tf_glm53_n_dconv(DconvArgs a) {
    const int row = blockIdx.x, G = a.d / a.gs;
    const uint16_t* x = (const uint16_t*)a.x + (long long)row * a.d;
    const uint16_t* dyn = (const uint16_t*)a.dyn + (long long)(row * 2 + a.branch) * 2 * G;
    const uint16_t* base = (const uint16_t*)a.base + (long long)a.branch * 2 * a.d;
    const int end = min(a.d, ((int)blockIdx.y + 1) * 1024);
    // d and gs multiples of 8, pointers aligned: 8 values (one group) a thread
    // d と gs は 8 の倍数、ポインタはアライン済み: 1 スレッド 8 値 (1 グループ)
    if (a.vec) {
        const int c = blockIdx.y * 1024 + 8 * threadIdx.x;
        if (c >= end) return;
        float xv[8], pv[8], b0[8], b1[8], rv[8];
        ld8_bf16v(x + c, xv);
        if (row > 0) ld8_bf16v(x + c - a.d, pv);
        ld8_bf16v(base + c, b0);
        ld8_bf16v(base + a.d + c, b1);
        if (a.has_res) ld8_bf16v((const uint16_t*)a.res + (long long)row * a.d + c, rv);
        const float d0 = bf16_f32(dyn[c / a.gs]), d1 = bf16_f32(dyn[G + c / a.gs]);
        unsigned yo[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float k0 = bf16_f32(f32_bf16(b0[j] + d0)), k1 = bf16_f32(f32_bf16(b1[j] + d1));
            uint16_t y = f32_bf16(fmaf(xv[j], k0, (row > 0 ? pv[j] : 0.0f) * k1));
            if (a.has_res) y = f32_bf16(rv[j] + bf16_f32(y));
            yo[j] = y;
        }
        *(uint4*)((uint16_t*)a.out + (long long)row * a.d + c) = make_uint4(yo[0] | (yo[1] << 16), yo[2] | (yo[3] << 16), yo[4] | (yo[5] << 16), yo[6] | (yo[7] << 16));
        return;
    }
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < end; c += blockDim.x) {
        const float xv = bf16_f32(x[c]), prev = row > 0 ? bf16_f32(x[c - a.d]) : 0.0f;
        const float k0 = bf16_f32(f32_bf16(bf16_f32(base[c]) + bf16_f32(dyn[c / a.gs])));
        const float k1 = bf16_f32(f32_bf16(bf16_f32(base[a.d + c]) + bf16_f32(dyn[G + c / a.gs])));
        uint16_t y = f32_bf16(fmaf(xv, k0, prev * k1));
        if (a.has_res) y = f32_bf16(bf16_f32(((const uint16_t*)a.res)[(long long)row * a.d + c]) + bf16_f32(y));
        ((uint16_t*)a.out)[(long long)row * a.d + c] = y;
    }
}

static __device__ __forceinline__ float t_rsqrt(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Drafter q / k / v preparation; program (row, head of [h q heads | hkv k heads | hkv v heads]). q and k heads: RMSNorm
// over the head's 128 dims (sum a^2 + sum b^2 of the two halves, each a thread's pair fused, then the warp's butterfly;
// rsqrt.approx of sum / 128 + eps), times the norm weight -> bf16, then RoPE with the row's cos / sin
// (fma(a, cos, -(b sin)) and fma(a, sin, b cos): here the a sin product is fused, unlike rope_pair). v heads are
// copied. Outputs [heads, L, 128]. grid (rows, h + 2 hkv), 32 threads.
// drafter の q / k / v の準備。program は (row, [q ヘッド h 個 | k ヘッド hkv 個 | v ヘッド hkv 個] の 1 つ)。q と
// k のヘッド: ヘッドの 128 次元で RMSNorm (前後の半分それぞれの sum a^2 + sum b^2。どちらもスレッドのペアを融合し、
// warp の butterfly で足す。rsqrt.approx(sum / 128 + eps))、ノルムの重みを掛けて bf16 にし、行の cos / sin で RoPE
// (fma(a, cos, -(b sin)) と fma(a, sin, b cos)。rope_pair と違い、ここでは a sin の積を融合する)。
// v のヘッドはコピーする。出力は [heads, L, 128]。grid (rows, h + 2 hkv)、32 スレッド。
typedef struct { Dev qkv, qn, kn, cos, sin, qo, ko, vo; int rows, stride, h, hkv; float eps; } PrepArgs;
__global__ void tf_glm53_n_prep(PrepArgs a) {
    const int row = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
    const uint16_t* src = (const uint16_t*)a.qkv + (long long)row * a.stride + head * 128;
    if (head >= a.h + a.hkv) {
        uint16_t* dst = (uint16_t*)a.vo + ((long long)(head - a.h - a.hkv) * a.rows + row) * 128;
        // the two pairs as two 4-byte copies
        // 2 つのペアを 4 バイトのコピー 2 回で
        if (((((size_t)src) | (size_t)dst) & 3) == 0) {
            *(uint32_t*)(dst + 2 * t) = *(const uint32_t*)(src + 2 * t);
            *(uint32_t*)(dst + 64 + 2 * t) = *(const uint32_t*)(src + 64 + 2 * t);
        } else {
            dst[2 * t] = src[2 * t]; dst[2 * t + 1] = src[2 * t + 1];
            dst[64 + 2 * t] = src[64 + 2 * t]; dst[64 + 2 * t + 1] = src[64 + 2 * t + 1];
        }
        return;
    }
    float av[2] = { bf16_f32(src[2 * t]), bf16_f32(src[2 * t + 1]) }, bv[2] = { bf16_f32(src[64 + 2 * t]), bf16_f32(src[64 + 2 * t + 1]) };
    float sa = fmaf(av[0], av[0], av[1] * av[1]), sb = fmaf(bv[0], bv[0], bv[1] * bv[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sa = sa + __shfl_xor_sync(0xffffffffu, sa, o);
        sb = sb + __shfl_xor_sync(0xffffffffu, sb, o);
    }
    const float rstd = t_rsqrt(fmaf(sa + sb, 0.0078125f, a.eps));
    const uint16_t* wn = (const uint16_t*)(head < a.h ? a.qn : a.kn);
    uint16_t* dst = head < a.h ? (uint16_t*)a.qo + ((long long)head * a.rows + row) * 128 : (uint16_t*)a.ko + ((long long)(head - a.h) * a.rows + row) * 128;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const float an = bf16_f32(f32_bf16((av[j] * rstd) * bf16_f32(wn[2 * t + j])));
        const float bn = bf16_f32(f32_bf16((bv[j] * rstd) * bf16_f32(wn[64 + 2 * t + j])));
        const float c = ((const float*)a.cos)[(long long)row * 64 + 2 * t + j], sn = ((const float*)a.sin)[(long long)row * 64 + 2 * t + j];
        dst[2 * t + j] = f32_bf16(fmaf(an, c, -(bn * sn)));
        dst[64 + 2 * t + j] = f32_bf16(fmaf(an, sn, bn * c));
    }
}

// Drafter attention over the KV ring (8 query heads a KV head, 128 dims); program (KV head): its 64 query rows m
// (head kvh 8 + m / 8, block row m % 8) attend the keys [s - window, s + 8) of the ring
// (slot = position % 4096; s = POS[0], the block's first position), in tiles of 64 keys from lo = max(s - window, 0)
// rounded down to 64. A context key (kk < s) is seen by row m while (s + m % 8) - kk <= window; the block's keys by
// all.
//   sc = (q . k) scale: 8 HMMA.16816.F32.BF16 an element (16 dims a step, ascending), masked -> -inf
//   m' = max(m, tile max), alpha = exp(m - m'), p = exp(sc - m') (ex2.approx of x log2 e)
//   l = fma(alpha, l, sum p): a thread's 4 keys in order, the warp's 4 threads by a butterfly (2, 1), the 4 warps as
//   (0 + 2) + (1 + 3)
//   acc = acc alpha, then 4 HMMAs add bf16(p) . v (16 keys a step)
// OUT[m % 8, head, :] = bf16(acc / l). 128 threads: warp w holds keys 8 w .. 8 w + 7 and 32 + 8 w .. of the tile
// (the original's layout: 4 warps tile 32 keys, twice, which fixes the grouping of the sum of p) for the scores and
// dims 32 w .. 32 w + 31 for the output, all 64 rows. Shared memory (dynamic, DR_SMEM bytes): q, the tile's k and v
// (rows of 256 bytes, 16-byte pieces at piece ^ (row & 7)), bf16(p) (rows of 128 bytes, likewise), the reductions.
// KV リング上の drafter の attention (KV ヘッド 1 つにクエリヘッド 8 つ、128 次元)。program は KV ヘッド 1 つ: その
// 64 本のクエリ行 m (ヘッド kvh 8 + m / 8、ブロック行 m % 8) が、リング (スロット = 位置 % 4096。
// s = POS[0] はブロックの最初の位置) のキー [s - window, s + 8) を見る。タイルは 64 キーずつで、
// lo = max(s - window, 0) を 64 に切り下げた位置から始める。文脈のキー (kk < s) は
// (s + m % 8) - kk <= window の間だけ行 m から見え、ブロックのキーは全行から見える。
//   sc = (q . k) scale: 1 要素に HMMA.16816.F32.BF16 を 8 回 (16 次元ずつ昇順)、マスクは -inf
//   m' = max(m, タイルの max)、alpha = exp(m - m')、p = exp(sc - m') (x log2 e の ex2.approx)
//   l = fma(alpha, l, sum p): スレッドの 4 キーを順に、warp の 4 スレッドを butterfly (2, 1) で、4 つの warp を
//   (0 + 2) + (1 + 3) で足す
//   acc = acc alpha、続けて HMMA 4 回で bf16(p) . v を足す (16 キーずつ)
// OUT[m % 8, head, :] = bf16(acc / l)。128 スレッド: warp w はスコアではタイルのキー 8 w .. 8 w + 7 と 32 + 8 w .. を
// (元の配置: 4 つの warp で 32 キーを 2 回敷き詰める。これで p の和のまとめ方が決まる)、
// 出力では次元 32 w .. 32 w + 31 を、全 64 行について持つ。共有メモリ (動的、DR_SMEM バイト): q、タイルの k と v
// (256 バイトの行、16 バイト片は piece ^ (row & 7))、bf16(p) (128 バイトの行、同様)、リダクション用。
#define DR_SMEM (5 * 16384 + 8192 + 2 * 1024)
typedef struct { Dev q, k, v, out, pos; int window, nh; float scale; } DattnArgs;
__global__ void tf_glm53_n_dattn_ring(DattnArgs a) {
    extern __shared__ __align__(16) uint8_t dr[];
    uint8_t* sq = dr;
    // two tiles of [k | v]: the next one is staged while the current one computes
    // [k | v] のタイル 2 つ: 今のタイルの計算中に次を読み込む
    uint8_t* sk0 = dr + 16384;
    uint8_t* sp = dr + 81920;
    float* smax = (float*)(dr + 90112);
    float* ssum = (float*)(dr + 91136);
    const int kvh = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, qq = lane & 3, sw = lane & 7;
    const int s = (int)((const long long*)a.pos)[0], klen = s + 8;
    // the compiled code is the same as with a.window / a.scale at each use: the compiler rereads them from the constant
    // bank rather than hold a register
    // a.window / a.scale を使う箇所ごとに書いても同じコードになる。コンパイラはレジスタに保持せず定数バンクから読み直す
    const int win = a.window;
    const float scl = a.scale;
    int lo = s - win > 0 ? s - win : 0;
    lo -= lo % 64;
    // the queries: rows m of this KV head are contiguous ((kvh 8 + m / 8) 8 + m % 8 = kvh 64 + m)
    // クエリ: この KV ヘッドの行 m は連続している ((kvh 8 + m / 8) 8 + m % 8 = kvh 64 + m)
    {
        const int row = t >> 1, p0 = (t & 1) * 8;
        const uint8_t* src = (const uint8_t*)a.q + ((long long)kvh * 64 + row) * 256;
#pragma unroll
        for (int j = 0; j < 8; j++) cp16(sq + row * 256 + ((p0 + j) ^ (row & 7)) * 16, src + (p0 + j) * 16);
        cp_commit();
    }
    float mi[8], li[8], acc[16][4];
#pragma unroll
    for (int i = 0; i < 8; i++) { mi[i] = -1e30f; li[i] = 0.0f; }
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    const int arow = ((lane & 7) + ((lane >> 3) & 1) * 8) * 256, apc = lane >> 4;
    const int tiles = (klen - lo + 63) / 64;
    for (int ti = 0; ti <= tiles; ti++) {
        // stage tile ti (while tile ti - 1 computes below): 16 pieces of a row each, 2 threads a key
        // タイル ti を読み込む (その間に下でタイル ti - 1 を計算する)。行ごとに 16 片、1 キーに 2 スレッド
        if (ti < tiles) {
            const int key = t >> 1, p0 = (t & 1) * 8, kk = lo + ti * 64 + key, ok = kk < klen;
            const long long src = ((long long)kvh * 4096 + (kk & 4095)) * 256;
            uint8_t* dk = sk0 + (ti & 1) * 32768;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                cp16z(dk + key * 256 + ((p0 + j) ^ (key & 7)) * 16, (const uint8_t*)a.k + src + (p0 + j) * 16, ok);
                cp16z(dk + 16384 + key * 256 + ((p0 + j) ^ (key & 7)) * 16, (const uint8_t*)a.v + src + (p0 + j) * 16, ok);
            }
        }
        cp_commit();
        if (ti == 0) continue;
        const int start = lo + (ti - 1) * 64;
        const uint8_t* sk = sk0 + ((ti - 1) & 1) * 32768;
        const uint8_t* sv = sk + 16384;
        cp_wait1();
        __syncthreads();
        // scores: 4 row tiles x 2 key tiles, the 8 chains interleaved
        // スコア: 行タイル 4 x キータイル 2、8 本の連鎖を交互に進める
        float sc[4][2][4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.0f;
#pragma unroll
        for (int c2 = 0; c2 < 4; c2++) {
            unsigned b[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; nt++) ldsm4(b[nt], sk + (8 * warp + 32 * nt + sw) * 256 + ((4 * c2 + (lane >> 3)) ^ sw) * 16);
#pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                unsigned af[4][4];
#pragma unroll
                for (int mt = 0; mt < 4; mt++) ldsm4(af[mt], sq + mt * 16 * 256 + arow + ((2 * (2 * c2 + cc) + apc) ^ sw) * 16);
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
#pragma unroll
                    for (int nt = 0; nt < 2; nt++) mma_bf16(sc[mt][nt], af[mt], b[nt][2 * cc], b[nt][2 * cc + 1]);
            }
        }
        // rows 16 mt + g (+ 8) = slot 2 mt (+ 1); this thread's 4 keys of a row: 8 warp + 32 nt + 2 qq (+ 1)
        // 行 16 mt + g (+ 8) = スロット 2 mt (+ 1)。このスレッドが持つ行の 4 キー: 8 warp + 32 nt + 2 qq (+ 1)
        float pv[8][4], alpha[8], mn[8];
#pragma unroll
        for (int rw = 0; rw < 8; rw++) {
            const int row = 16 * (rw >> 1) + g + 8 * (rw & 1), qpos = s + (row & 7);
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int kk = start + 8 * warp + 32 * (i >> 1) + 2 * qq + (i & 1);
                const bool ok = kk < klen && ((kk < s && qpos - kk <= win) || kk >= s);
                const float v = ok ? sc[rw >> 1][i >> 1][2 * (rw & 1) + (i & 1)] * scl : -INFINITY;
                pv[rw][i] = v;
                mx = fmaxf(mx, v);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            if (qq == 0) smax[row * 4 + warp] = mx;
        }
        __syncthreads();
#pragma unroll
        for (int rw = 0; rw < 8; rw++) {
            const int row = 16 * (rw >> 1) + g + 8 * (rw & 1);
            const float* v = smax + row * 4;
            mn[rw] = fmaxf(mi[rw], fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])));
            alpha[rw] = t_exp(mi[rw] - mn[rw]);
            unsigned pk[2];
#pragma unroll
            for (int i = 0; i < 4; i++) pv[rw][i] = t_exp(pv[rw][i] - mn[rw]);
            pk[0] = f32_bf16x2(pv[rw][0], pv[rw][1]);
            pk[1] = f32_bf16x2(pv[rw][2], pv[rw][3]);
            // bf16(p) of keys 8 warp + 32 nt + 2 qq (+ 1): piece warp + 4 nt of the row, 4 bytes at 4 qq
            // キー 8 warp + 32 nt + 2 qq (+ 1) の bf16(p): 行の片 warp + 4 nt の、4 qq から 4 バイト
#pragma unroll
            for (int nt = 0; nt < 2; nt++) *(unsigned*)(sp + row * 128 + ((warp + 4 * nt) ^ (row & 7)) * 16 + 4 * qq) = pk[nt];
            float sm = ((pv[rw][0] + pv[rw][1]) + pv[rw][2]) + pv[rw][3];
            sm = sm + __shfl_xor_sync(0xffffffffu, sm, 2);
            sm = sm + __shfl_xor_sync(0xffffffffu, sm, 1);
            if (qq == 0) ssum[row * 4 + warp] = sm;
        }
        __syncthreads();
#pragma unroll
        for (int rw = 0; rw < 8; rw++) {
            const float* v = ssum + (16 * (rw >> 1) + g + 8 * (rw & 1)) * 4;
            li[rw] = fmaf(alpha[rw], li[rw], (v[0] + v[2]) + (v[1] + v[3]));
            mi[rw] = mn[rw];
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nd = 0; nd < 4; nd++) {
                float* o = acc[4 * mt + nd];
                o[0] = o[0] * alpha[2 * mt]; o[1] = o[1] * alpha[2 * mt];
                o[2] = o[2] * alpha[2 * mt + 1]; o[3] = o[3] * alpha[2 * mt + 1];
            }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {   // 32 keys a load of v / v の 1 回のロードで 32 キー
            unsigned b[4][4];
#pragma unroll
            for (int nd = 0; nd < 4; nd++) ldsm4t(b[nd], sv + (32 * k2 + sw + 8 * (lane >> 3)) * 256 + ((4 * warp + nd) ^ sw) * 16);
#pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                unsigned af[4][4];
#pragma unroll
                for (int mt = 0; mt < 4; mt++) ldsm4(af[mt], sp + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * 128 + ((2 * (2 * k2 + cc) + apc) ^ sw) * 16);
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
#pragma unroll
                    for (int nd = 0; nd < 4; nd++) mma_bf16(acc[4 * mt + nd], af[mt], b[nd][2 * cc], b[nd][2 * cc + 1]);
            }
        }
        __syncthreads();
    }
    cp_wait0();
#pragma unroll
    for (int rw = 0; rw < 8; rw++) {
        const int row = 16 * (rw >> 1) + g + 8 * (rw & 1);
        uint16_t* out = (uint16_t*)a.out + (long long)(row & 7) * a.nh * 128 + (kvh * 8 + (row >> 3)) * 128 + 32 * warp + 2 * qq;
#pragma unroll
        for (int nd = 0; nd < 4; nd++) *(unsigned*)(out + nd * 8) = f32_bf16x2(t_div(acc[4 * (rw >> 1) + nd][2 * (rw & 1)], li[rw]), t_div(acc[4 * (rw >> 1) + nd][2 * (rw & 1) + 1], li[rw]));
    }
}

}  // end of extern "C" / extern "C" の終わり
