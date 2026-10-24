// From TensorFold's Python engine (src/tensorfold/cuda/exl3/prompt_experts.cu), Apache-2.0, Copyright 2026 TensorFold
// contributors; modified by Fiore-lab: host launchers compiled out for the Zig build, kernels in a named namespace, the
// expert-order table built only when ordering by count.
// TensorFold の Python エンジン由来 (Apache-2.0)、Fiore-lab が Zig ビルド向けに改変。
// EXL3 routed experts for prompt chunks (hundreds to thousands of rows): a width per expert (any codebook), tensor
// cores, trellis decoded on the fly, no dequantized weights and no rotated inputs in memory.
// prompt チャンク (数百〜数千行) 向けの EXL3 routed expert。幅は expert ごと (codebook は任意)、tensor core を使い、
// trellis はその場で復号する。復号した重みも回転した入力もメモリに置かない。
//
// Design adapted from MiaAI-Lab's GLM-5.3-Flash prompt kernels (Apache-2.0; patches "glm-prompt-kernels",
// "glm-prompt-experts-order", "glm-exl3-prompt-experts": prompt_gateup_kernel / prompt_down_kernel).
// 設計は MiaAI-Lab の GLM-5.3-Flash 用 prompt カーネル (Apache-2.0) を元にしている。
// Y^T = W^T X^T: a decoded trellis tile is the m16n8k16 A fragment against up to IPM member rows staged in shared
// memory; one fp32 chain an output (no split K); the epilogues fuse the FWHT, svh, SwiGLU and the next rotation. The
// trellis decode is experts_grouped.cuh's (Fmt / LaneMap / decode_tile, after ExLlamaV3, MIT, Copyright (c) 2025
// Turboderp). Every expert has its own suh, so the input rotation moves onto the weights (see gate | up).
// Y^T = W^T X^T: 復号した trellis タイルを m16n8k16 の A fragment とし、共有メモリに置いた最大 IPM 行のメンバー
// に掛ける。出力 1 つにつき fp32 の加算列は 1 本 (K を分割しない)。FWHT・svh・SwiGLU・次の回転は epilogue で行う。
// trellis の復号は experts_grouped.cuh のもの。expert ごとに suh が違うので、入力の回転を重みの側へ移す (gate | up)。
//
// Launches: route (one block: histogram, scan, items of <= IPM pairs an expert, scatter), gate|up (+ SwiGLU and the
// down input's rotation: xd [pairs, I] fp16 in routed order; also zeroes out), down (+ svh, routing weight, red.add
// into an fp32 out).
// 起動するカーネル: route (1 ブロック: ヒストグラム、scan、expert ごとに IPM 組以下の item、scatter)、gate|up
// (SwiGLU と down 入力の回転まで行い、xd [pairs, I] fp16 を routed 順に書く。out のゼロ埋めも行う)、down (svh、
// routing 重み、fp32 の out への red.add)。

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifndef TF_ZIG
#include <ATen/ATen.h>
#endif
#ifndef TF_ZIG
#include <ATen/cuda/CUDAContext.h>
#endif
#ifndef TF_ZIG
#include <c10/cuda/CUDAGuard.h>
#endif

#include "experts_grouped.cuh"

#ifdef TF_ZIG
namespace tf_exl3p {
#else
namespace {
#endif

using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::decode_tile;
// load_words: word[j] = lane + 32*j < TW ? load(base + 32*j) : 0.
// K2=10 keeps TW=40 (no 64-word padding); decode_tile shuffles on every lane.
using tf_exl3x::load_words;

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
#ifndef TF_PE_GPM
// gate|up members an item (a multiple of 16, <= 128)
// item あたりの gate|up のメンバー数 (16 の倍数、128 以下)
#define TF_PE_GPM 112
#endif
// routed pairs an item (route, both GEMMs)
// item あたりの routed pair 数 (route と両方の GEMM)
constexpr int IPM = TF_PE_GPM;
constexpr int NW = 16;                              // warps a block / ブロックあたりの warp 数
constexpr int NTHREADS = NW * 32;
// k values a chunk (one Hadamard block)
// チャンクあたりの k の数 (Hadamard ブロック 1 つ)
constexpr int CK = 128;

// mma.m16n8k16 fp16 -> fp32, d += a b; not volatile (registers only), so the compiler may interleave the chains
// mma.m16n8k16 fp16 -> fp32、d += a b。volatile にしない (レジスタのみ) ので、コンパイラが加算列を交互に並べられる
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// out as fp16 (TF_EXL3_PROMPT_F16_OUT): two f16x2 adds a call; dst indexes the fp16 buffer like the fp32 one
// out を fp16 にする場合 (TF_EXL3_PROMPT_F16_OUT): 1 回で f16x2 の加算 2 つ。dst は fp32 と同じ添字で fp16 を指す
__device__ __forceinline__ void red_add_f16x4(half* dst, float a, float b, float c, float d) {
    const half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
    asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1, %2};\n" ::"l"(dst), "r"(*reinterpret_cast<const uint32_t*>(&lo)),
                 "r"(*reinterpret_cast<const uint32_t*>(&hi)) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(p));
}
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d)
                 : "memory");
}

__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        // lower lane: v + o, upper: o - v (one fma)
        // 下位の lane は v + o、上位は o - v (fma 1 回)
        const float sg = (lane & m) ? -1.f : 1.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = fmaf(sg, v[j], o);
        }
    }
}

// fwht128 of N vectors at once (independent chains interleaved)
// N 本のベクトルの fwht128 を一度に行う (独立な加算列を交互に並べる)
template <int N>
__device__ __forceinline__ void fwht128_n(float (&v)[N][4], int lane) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
        const float a = v[n][0] + v[n][1], b = v[n][0] - v[n][1], c = v[n][2] + v[n][3], d = v[n][2] - v[n][3];
        v[n][0] = a + c; v[n][1] = b + d; v[n][2] = a - c; v[n][3] = b - d;
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float sg = (lane & m) ? -1.f : 1.f;
#pragma unroll
        for (int n = 0; n < N; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float o = __shfl_xor_sync(0xffffffffu, v[n][j], m);
                v[n][j] = fmaf(sg, v[n][j], o);
            }
    }
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// H16[i][j] sign as a half's sign bit
// H16[i][j] の符号を half の符号ビットで返す
__device__ __forceinline__ uint32_t hsign(int i, int j) { return (__popc(i & j) & 1) ? 0x8000u : 0u; }

// ---- routing ------------------------------------------------------------------------------------------------------
// One block: per expert counts, offsets and items (e, first, count <= IPM) in expert order; sorted[] = pair ids
// (row * slots + slot) grouped by expert (order within an expert: arrival).
// 1 ブロックで expert ごとの件数、オフセット、item (e, first, count <= IPM) を expert 順に作る。sorted[] は pair id
// (row * slots + slot) を expert ごとにまとめたもの (expert 内の順は到着順)。
constexpr int ROUTE_THREADS = 1024;

__global__ void __launch_bounds__(ROUTE_THREADS) route_kernel(const int* __restrict__ pick, int n, int E,
                                                             int* __restrict__ sorted, int* __restrict__ items,
                                                             int* __restrict__ item_count, int max_items,
                                                             int by_count) {
    extern __shared__ int sh[];
    int* cnt = sh;              // [E]
    int* cur = sh + E;          // [E]
    // experts in item order; by_count: most pairs first, so the longest blocks start first and the last wave is
    // short (an expert's items stay adjacent)
    // item 順の expert。by_count では組の多い順にし、長いブロックが先に始まり最後の wave が短くなる (expert の
    // item は隣り合ったまま)
    int* ord = sh + 2 * E;      // [E]
    __shared__ int wsum[ROUTE_THREADS / 32][2];
    for (int e = threadIdx.x; e < E; e += ROUTE_THREADS) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += ROUTE_THREADS) {
        const int e = pick[i];
        if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    // without by_count the item order is the expert order: no table, no barrier (ORD below)
    // by_count でなければ item 順は expert 順: 表もバリアも要らない (下の ORD)
    if (by_count) {
        for (int e = threadIdx.x; e < E; e += ROUTE_THREADS) {
            const int c = cnt[e];
            int r = 0;
            for (int f = 0; f < E; ++f) r += cnt[f] > c || (cnt[f] == c && f < e);
            ord[r] = e;
        }
        __syncthreads();
    }
#define ORD(x) (by_count ? ord[x] : (x))
    // exclusive scans of counts and of items over experts; each thread owns a contiguous run of experts
    // expert をまたいだ件数と item 数の排他 scan。各スレッドは連続した expert の範囲を受け持つ
    const int per = (E + ROUTE_THREADS - 1) / ROUTE_THREADS;
    const int e0 = threadIdx.x * per;
    int c_sum = 0, i_sum = 0;
    for (int q = 0; q < per; ++q) {
        const int e = e0 + q;
        if (e < E) { c_sum += cnt[ORD(e)]; i_sum += (cnt[ORD(e)] + IPM - 1) / IPM; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int ci = c_sum, ii = i_sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, ci, o), b = __shfl_up_sync(0xffffffffu, ii, o);
        if (lane >= o) { ci += a; ii += b; }
    }
    if (lane == 31) { wsum[warp][0] = ci; wsum[warp][1] = ii; }
    __syncthreads();
    if (warp == 0) {
        int a = wsum[lane][0], b = wsum[lane][1];
        int sa = a, sb = b;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, sa, o), y = __shfl_up_sync(0xffffffffu, sb, o);
            if (lane >= o) { sa += x; sb += y; }
        }
        wsum[lane][0] = sa - a;
        wsum[lane][1] = sb - b;
        if (lane == 31) item_count[0] = min(sb, max_items);
    }
    __syncthreads();
    int off = wsum[warp][0] + ci - c_sum, it = wsum[warp][1] + ii - i_sum;
    for (int q = 0; q < per; ++q) {
        if (e0 + q >= E) break;
        const int e = ORD(e0 + q);
        const int c = cnt[e];
        cur[e] = off;
        for (int f = 0; f < c; f += IPM, ++it)
            if (it < max_items) {
                items[3 * it] = e;
                items[3 * it + 1] = off + f;
                items[3 * it + 2] = min(IPM, c - f);
            }
        off += c;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += ROUTE_THREADS) {
        const int e = pick[i];
        if (e >= 0 && e < E) sorted[atomicAdd(&cur[e], 1)] = i;
    }
}
#undef ORD

// ---- gate | up ----------------------------------------------------------------------------------------------------
// The input rotation moves onto the weights, so the members' rows are staged once (fp16) for gate and up alike:
//
//   (x suh H) W_q = x W_eff,   W_eff[k] = suh[k] (H W_q)[k],   H = H8 (x) H16 on a 128 block of k
//
// Each warp turns its m tile's 8 decoded tiles of a 128-k chunk into W_eff^T A fragments: W_q^T H16 a tile (two mma
// against constant H16 fragments, whose fp32 outputs are already in A-fragment layout), the H8 butterfly across the
// 8 tiles and the suh scaling in half2.
// Block (column block cb of 128 gate + 128 up columns, item of up to GPM members): 16 warps; warp w owns m tile
// (w & 7) of matrix w >> 3 for all members. Rows (bf16) and suh land by cp.async three stages deep; a stage is
// converted to fp16 in place the chunk before it is used; one barrier a chunk.
// Rows of 128 halves are XOR-swizzled in 16-byte pieces (piece ^ (row & 7)) so ldmatrix hits distinct banks.
// 入力の回転を重みの側へ移す (上の式)。これでメンバーの行は gate と up に共通で 1 回だけ (fp16 で) 置けばよい。
// 各 warp は 128-k チャンクにある自分の m タイルの復号済みタイル 8 枚を W_eff^T の A fragment にする: タイルごとに
// W_q^T H16 (定数の H16 fragment との mma 2 回。fp32 の出力はそのまま A fragment の配置)、8 枚にまたがる H8
// butterfly、half2 での suh の掛け算。
// ブロック (gate 128 列 + up 128 列の列ブロック cb、最大 GPM メンバーの item): 16 warp。warp w は行列 w >> 3 の
// m タイル (w & 7) を全メンバー分受け持つ。行 (bf16) と suh は cp.async で 3 段先まで読み込み、各段は使う 1 チャンク
// 前にその場で fp16 にする。バリアはチャンクごとに 1 回。
// 128 half の行は 16 バイト単位で XOR swizzle し (piece ^ (row & 7))、ldmatrix が別々のバンクに当たるようにする。
constexpr int GPM = TF_PE_GPM;
#ifndef TF_PE_GU_MT
#define TF_PE_GU_MT 1
#endif
// m tiles a warp (independent mma / decode chains)
// warp あたりの m タイル数 (独立した mma・復号の列)
constexpr int GU_MT = TF_PE_GU_MT;
// warps a block (16 m tiles: 8 gate, 8 up)
// ブロックあたりの warp 数 (m タイル 16 枚: gate 8、up 8)
constexpr int GU_NW = 16 / GU_MT;
constexpr int GU_THREADS = GU_NW * 32;
constexpr int GU_WPM = GU_NW / 2;                             // warps a matrix / 行列あたりの warp 数
constexpr int GU_STAGES = 3;
constexpr int GU_ES = 128 + 4;                                // epilogue row stride (fp32) / epilogue の行 stride
constexpr int GU_EROWS = 64;                                  // epilogue rows a pass / 1 パスの epilogue 行数
constexpr size_t GU_XS = (size_t)GPM * CK * 2;                // a stage's rows / 1 段分の行
constexpr size_t GU_SU = (size_t)2 * CK * 2;                  // a stage's suh (gate, up) / 1 段分の suh (gate, up)
constexpr size_t GU_STAGE = GU_XS + GU_SU;
constexpr size_t GU_SMEM_MAIN = GU_STAGES * GU_STAGE;
constexpr size_t GU_SMEM_EPI = (size_t)2 * GU_EROWS * GU_ES * 4;
constexpr size_t GU_SMEM = GU_SMEM_MAIN > GU_SMEM_EPI ? GU_SMEM_MAIN : GU_SMEM_EPI;

// half offset of (row, k) in a swizzled [rows][128] buffer
// swizzle した [rows][128] バッファでの (row, k) の位置 (half 単位)
__device__ __forceinline__ int swz(int row, int k) {
    return row * CK + ((((k >> 3) ^ row) & 7) | ((k >> 3) & 8)) * 8 + (k & 7);
}

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

// W_q^T H16 of one decoded tile (two mma against the constant H16 fragments; their fp32 outputs are in A-fragment
// layout: a0 / a1 from the first n8 half, a2 / a3 from the second).
// 復号したタイル 1 枚の W_q^T H16 (定数の H16 fragment との mma 2 回。fp32 の出力はそのまま A fragment の配置で、
// a0 / a1 は前半の n8、a2 / a3 は後半から)。
template <int CB, int K2>
__device__ __forceinline__ void tile_h16(const uint32_t (&word)[Fmt<K2>::LW], const LaneMap<K2>& map,
                                         const uint32_t (&hB)[2][2], int lane, half2 (&w)[4]) {
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(word, map, lane, b0, b1);
    const uint32_t A[4] = {b0[0], b1[0], b0[1], b1[1]};            // W_q^T (m = column, k) / W_q^T (m は列, k)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float f[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(f, A, hB[h]);
        w[2 * h] = __floats2half2_rn(f[0], f[1]);
        w[2 * h + 1] = __floats2half2_rn(f[2], f[3]);
    }
}

// The H8 butterfly across a chunk's 8 tiles (in place).
// チャンクの 8 タイルにまたがる H8 butterfly (その場で行う)。
__device__ __forceinline__ void tiles_h8(half2 (&W)[8][4]) {
#pragma unroll
    for (int s = 1; s < 8; s <<= 1)
#pragma unroll
        for (int a = 0; a < 8; ++a)
            if (!(a & s))
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const half2 p = W[a][q], m = W[a + s][q];
                    W[a][q] = __hadd2(p, m);
                    W[a + s][q] = __hsub2(p, m);
                }
}

// One chunk of a warp, software-pipelined across chunks: W holds this chunk's (H W_q)^T tiles; step kk runs tile kk
// (times suh) over the staged members (2 NB mma), then decodes tile kk of the next chunk into W[kk] (its words in
// wbuf since a chunk ago) and fetches the chunk after's word in its place. The next chunk's H8 closes the chunk.
// warp の 1 チャンク分。チャンクをまたいでソフトウェアパイプライン化している: W はこのチャンクの (H W_q)^T タイル。
// ステップ kk はタイル kk (suh を掛けたもの) を置かれたメンバーに掛け (mma 2 NB 回)、次のチャンクのタイル kk を
// W[kk] へ復号し (その語は 1 チャンク前から wbuf にある)、空いた所へ次の次のチャンクの語を読む。次のチャンクの H8
// で終わる。
template <int CB, int K2, int NB>
__device__ __forceinline__ void gu_chunk(const half* xs, const half* su, int moff, half2 (&W)[GU_MT][8][4],
                                         uint32_t (&wbuf)[GU_MT][8][Fmt<K2>::LW], bool has_next,
                                         const uint32_t* __restrict__ wnext2, size_t kstride, size_t tstride,
                                         const LaneMap<K2>& map, const uint32_t (&hB)[2][2], int lane,
                                         float (&acc)[GU_MT][2 * NB][4]) {
    const int t = lane & 3;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const half2 s0 = *reinterpret_cast<const half2*>(su + kk * 16 + 2 * t);
        const half2 s1 = *reinterpret_cast<const half2*>(su + kk * 16 + 8 + 2 * t);
        uint32_t A[GU_MT][4];
#pragma unroll
        for (int i = 0; i < GU_MT; ++i) {
            A[i][0] = h2u(__hmul2(W[i][kk][0], s0));
            A[i][1] = h2u(__hmul2(W[i][kk][1], s0));
            A[i][2] = h2u(__hmul2(W[i][kk][2], s1));
            A[i][3] = h2u(__hmul2(W[i][kk][3], s1));
        }
        const half* xk = xs + (moff ^ ((2 * (kk & 3)) << 3)) + (kk >= 4 ? 64 : 0);
#pragma unroll
        for (int jb = 0; jb < NB; ++jb) {
            uint32_t r[4];
            ldsm_x4(r, xk + jb * 16 * CK);
            const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
            for (int i = 0; i < GU_MT; ++i) {
                mma16816(acc[i][2 * jb], A[i], bl);
                mma16816(acc[i][2 * jb + 1], A[i], bh);
            }
        }
        if (has_next) {
#pragma unroll
            for (int i = 0; i < GU_MT; ++i) {
                tile_h16<CB, K2>(wbuf[i][kk], map, hB, lane, W[i][kk]);
                if (wnext2) load_words<K2>(wbuf[i][kk], wnext2 + (size_t)kk * kstride + i * tstride, lane);
            }
        }
    }
    if (has_next)
#pragma unroll
        for (int i = 0; i < GU_MT; ++i) tiles_h8(W[i]);
}

template <int CB, int K2, int NB>
__device__ __forceinline__ void gu_mainloop(const __nv_bfloat16* __restrict__ x, int xstride,
                                            const uint32_t* __restrict__ Tm, const uint32_t* __restrict__ Tother,
                                            int NS, const half* __restrict__ suh_g,
                                            const half* __restrict__ suh_u, const int* rows_sh, int cnt, int cb,
                                            int K, unsigned char* smem, float (&acc)[GU_MT][2 * NB][4]) {
    constexpr int TW = Fmt<K2>::TW;
    static_assert(K2 == 4 || K2 == 6 || K2 == 8 || K2 == 10);
    auto xs_of = [&](int b) { return reinterpret_cast<half*>(smem + b * GU_STAGE); };
    auto su_of = [&](int b) { return reinterpret_cast<half*>(smem + b * GU_STAGE + GU_XS); };
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int chunks = K / CK;
    const LaneMap<K2> map(lane);
    const int mat = warp / GU_WPM, nt = cb * 8 + (warp % GU_WPM) * GU_MT;
    const int l7 = lane & 7;
    // the lane's ldmatrix row and piece for k tile 0 (k tile q: piece ^ 2q; + 64 halves from tile 4 on)
    // k タイル 0 での lane の ldmatrix の行と piece (k タイル q では piece ^ 2q、タイル 4 からは + 64 half)
    const int moff = (l7 + ((lane >> 4) & 1) * 8) * CK + ((((lane >> 3) & 1) ^ l7) & 7) * 8;
    // H16 as the B fragments of its two n8 halves (k = 2t.., n = 8h + g), fp16 +-1
    // H16 を 2 つの n8 半分の B fragment にする (k = 2t.., n = 8h + g)。fp16 の +-1
    uint32_t hB[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int n = 8 * h + g;
        hB[h][0] = (0x3C00u | hsign(2 * t, n)) | ((0x3C00u | hsign(2 * t + 1, n)) << 16);
        hB[h][1] = (0x3C00u | hsign(2 * t + 8, n)) | ((0x3C00u | hsign(2 * t + 9, n)) << 16);
    }

#pragma unroll
    for (int i = 0; i < GU_MT; ++i)
#pragma unroll
        for (int j = 0; j < 2 * NB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    auto load = [&](int chunk) {
        if (chunk >= chunks) return;
        half* dst = xs_of(chunk % GU_STAGES);
        const int k0 = chunk * CK;
#pragma unroll
        for (int q = 0; q < (GPM * (CK / 8) + GU_THREADS - 1) / GU_THREADS; ++q) {
            const int idx = threadIdx.x + q * GU_THREADS;
            if (GPM * (CK / 8) % GU_THREADS && idx >= GPM * (CK / 8)) break;
            const int i = idx >> 4, j = idx & 15;
            // re-read: no live row pointers
            // 毎回読み直す: 行のポインタを保持し続けない
            const int r = reinterpret_cast<const volatile int*>(rows_sh)[i];
            const __nv_bfloat16* src = x + (size_t)(r < 0 ? 0 : r) * xstride + k0 + j * 8;
            cp_async16(dst + swz(i, j * 8), src, r >= 0);
        }
        if (threadIdx.x < 32) {
            const int m = threadIdx.x >> 4, j = threadIdx.x & 15;
            cp_async16(su_of(chunk % GU_STAGES) + m * CK + j * 8, (m ? suh_u : suh_g) + k0 + j * 8, true);
        }
    };
    // bf16 -> fp16 in place, a stage (rows only)
    // 1 段分を bf16 から fp16 へその場で変換する (行のみ)
    auto convert = [&](int chunk) {
        if (chunk >= chunks) return;
        uint4* p = reinterpret_cast<uint4*>(xs_of(chunk % GU_STAGES));
#pragma unroll
        for (int q = 0; q < (GPM * CK / 8 + GU_THREADS - 1) / GU_THREADS; ++q) {
            if (GPM * CK / 8 % GU_THREADS && threadIdx.x + q * GU_THREADS >= GPM * CK / 8) break;
            uint4 v = p[threadIdx.x + q * GU_THREADS];
            uint32_t* w = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w[j]);
                w[j] = h2u(__float22half2_rn(__bfloat1622float2(b)));
            }
            p[threadIdx.x + q * GU_THREADS] = v;
        }
    };
    const uint32_t* tbase = Tm + (size_t)nt * TW + lane;
    const size_t kstride = (size_t)NS * TW;
    // L2 prefetch of a chunk's trellis for the whole block: per (matrix, k tile) its 8 m tiles are 8 TW contiguous
    // words (8 TW * 4 bytes = TW / 4 lines of 128 bytes); thread q < 2 * 8 * (TW / 4) takes one line
    // ブロック全体のためにチャンクの trellis を L2 へ先読みする: (行列, k タイル) ごとに m タイル 8 枚は連続した
    // 8 TW 語 (8 TW * 4 バイト = 128 バイトの行 TW / 4 本)。スレッド q < 2 * 8 * (TW / 4) が 1 行ずつ受け持つ
    constexpr int PF_LINES = TW / 4;
    constexpr int PF_PER_THREAD = (2 * 8 * PF_LINES + GU_THREADS - 1) / GU_THREADS;
    const uint32_t* pf_base[PF_PER_THREAD];
#pragma unroll
    for (int i = 0; i < PF_PER_THREAD; ++i) {
        const int q = threadIdx.x + i * GU_THREADS;
        pf_base[i] = nullptr;
        if (q < 2 * 8 * PF_LINES) {
            const int pm = q / (8 * PF_LINES), pk = (q / PF_LINES) % 8, pl = q % PF_LINES;
            const uint32_t* T = pm == mat ? Tm : Tother;
            pf_base[i] = T + (size_t)pk * kstride + (size_t)cb * 8 * TW + pl * 32;
        }
    }
    auto prefetch = [&](int chunk) {
#pragma unroll
        for (int i = 0; i < PF_PER_THREAD; ++i)
            if (pf_base[i] && chunk < chunks) prefetch_l2(pf_base[i] + (size_t)chunk * 8 * kstride);
    };
    prefetch(1);
    prefetch(2);

    load(0);
    cp_async_commit();
    load(1);
    cp_async_commit();
    // chunk 0's tiles decoded and rotated up front; wbuf: chunk 1's words
    // チャンク 0 のタイルは先に復号して回転しておく。wbuf にはチャンク 1 の語
    half2 W[GU_MT][8][4];
    uint32_t wbuf[GU_MT][8][Fmt<K2>::LW];
#pragma unroll
    for (int i = 0; i < GU_MT; ++i)
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
            load_words<K2>(wbuf[i][kk], tbase + (size_t)kk * kstride + i * TW, lane);
#pragma unroll
    for (int i = 0; i < GU_MT; ++i) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            tile_h16<CB, K2>(wbuf[i][kk], map, hB, lane, W[i][kk]);
            if (chunks > 1)
                load_words<K2>(wbuf[i][kk], tbase + (size_t)(8 + kk) * kstride + i * TW, lane);
            else
#pragma unroll
                for (int j = 0; j < Fmt<K2>::LW; ++j) wbuf[i][kk][j] = 0u;
        }
        tiles_h8(W[i]);
    }
    cp_async_wait<1>();
    __syncthreads();
    convert(0);

    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<0>();
        // stage c converted; stage c + 1 landed; stage c + 2 (= c - 1) free
        // 段 c は変換済み、段 c + 1 は到着済み、段 c + 2 (= c - 1) は空き
        __syncthreads();
        load(c + 2);
        cp_async_commit();
        prefetch(c + 3);
        convert(c + 1);
        const half* xs = xs_of(c % GU_STAGES);
        const half* su = su_of(c % GU_STAGES) + mat * CK;
        const uint32_t* wnext2 = c + 2 < chunks ? tbase + (size_t)(c + 2) * 8 * kstride : nullptr;
        gu_chunk<CB, K2, NB>(xs, su, moff, W, wbuf, c + 1 < chunks, wnext2, kstride, (size_t)TW, map, hB, lane,
                             acc);
    }
    __syncthreads();
}

struct GuEpi {
    const half* svh_g;
    const half* svh_u;
    const half* suh_d;
    half* xd;
    int e, cb, N, first, cnt;
    float limit;
    int act_mode;
};

// acc -> E[mat][member][col] (fp32), 64 members a pass, then a warp a member: FWHT, svh, SwiGLU, suh_d, FWHT -> xd
// acc -> E[mat][member][col] (fp32) を 1 パスで 64 メンバー。続いて 1 メンバーを 1 warp で: FWHT、svh、SwiGLU、suh_d、
// FWHT -> xd
template <int NB>
__device__ __forceinline__ void gu_epilogue(const float (&acc)[GU_MT][2 * NB][4], unsigned char* smem,
                                            const GuEpi& p) {
    float* Eg = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int mat = warp / GU_WPM;
    const int col = (warp % GU_WPM) * GU_MT * 16 + g;
    const int n = p.cb * 128 + 4 * lane;
    const size_t eo = (size_t)p.e * p.N + n;
    float svg[4], svu[4], sud[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // W_eff leaves out the input rotation's 1/sqrt(128): with the output's, 1/128
        // W_eff には入力の回転の 1/sqrt(128) が入っていない: 出力側の分と合わせて 1/128
        svg[j] = __half2float(p.svh_g[eo + j]) * (1.f / 128.f);
        svu[j] = __half2float(p.svh_u[eo + j]) * (1.f / 128.f);
        sud[j] = __half2float(p.suh_d[eo + j]);
    }
    constexpr int PASSES = (2 * NB * 8 + GU_EROWS - 1) / GU_EROWS;
#pragma unroll
    for (int hf = 0; hf < PASSES; ++hf) {
        if (hf) __syncthreads();
#pragma unroll
        for (int q = 0; q < GU_EROWS / 8; ++q) {
            const int jn = hf * (GU_EROWS / 8) + q;
            if (jn < 2 * NB) {
                const int r = q * 8 + 2 * t;
#pragma unroll
                for (int i = 0; i < GU_MT; ++i) {
                    float* E0 = Eg + ((size_t)mat * GU_EROWS + r) * GU_ES + col + 16 * i;
                    E0[0] = acc[i][jn][0];
                    E0[GU_ES] = acc[i][jn][1];
                    E0[8] = acc[i][jn][2];
                    E0[GU_ES + 8] = acc[i][jn][3];
                }
            }
        }
        __syncthreads();
        for (int i = warp; i < GU_EROWS && hf * GU_EROWS + i < p.cnt; i += GU_NW) {
            float gv[4], uv[4];
            const float4 g4 = *reinterpret_cast<const float4*>(Eg + (size_t)i * GU_ES + 4 * lane);
            const float4 u4 = *reinterpret_cast<const float4*>(Eg + (size_t)(GU_EROWS + i) * GU_ES + 4 * lane);
            gv[0] = g4.x; gv[1] = g4.y; gv[2] = g4.z; gv[3] = g4.w;
            uv[0] = u4.x; uv[1] = u4.y; uv[2] = u4.z; uv[3] = u4.w;
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float act;
                if (p.act_mode == 0) {
                    const float gg = fminf(bf16r(gv[j] * svg[j]), p.limit);
                    const float uu = fminf(fmaxf(bf16r(uv[j] * svu[j]), -p.limit), p.limit);
                    act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                } else {
                    const float gg = fminf(gv[j] * svg[j], p.limit);
                    const float uu = fminf(fmaxf(uv[j] * svu[j], -p.limit), p.limit);
                    act = gg / (1.f + expf(-gg)) * uu;
                }
                v[j] = act * sud[j];
            }
            fwht128(v, lane);
            const uint2 o = make_uint2(pack_h2(v[0] * HAD_SCALE, v[1] * HAD_SCALE),
                                       pack_h2(v[2] * HAD_SCALE, v[3] * HAD_SCALE));
            *reinterpret_cast<uint2*>(p.xd + (size_t)(p.first + hf * GU_EROWS + i) * p.N + n) = o;
        }
    }
}

template <int CB, bool K10>
__device__ __forceinline__ void prompt_gateup_body(
    const __nv_bfloat16* __restrict__ x, int xstride, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ gate_ptr, const int64_t* __restrict__ up_ptr,
    const int* __restrict__ k2s, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
    const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
    half* __restrict__ xd, int K, int N, int NS, int slots, float limit, int act_mode, float4* __restrict__ zero,
    long long zero_n) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[GPM];
    const int cb = blockIdx.x, item = blockIdx.y;
    {
        // a slice of out to zero for the down kernel's red.add (instead of a memset launch)
        // down カーネルの red.add のために out の一部をゼロにする (memset の起動の代わり)
        const long long blocks = (long long)gridDim.x * gridDim.y, b = (long long)blockIdx.y * gridDim.x + blockIdx.x;
        const long long per = (zero_n + blocks - 1) / blocks, z0 = b * per, z1 = min(zero_n, z0 + per);
        for (long long i = z0 + threadIdx.x; i < z1; i += GU_THREADS) zero[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (item >= item_count[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < GPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? sorted[first + threadIdx.x] / slots : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / GU_WPM;
    const uint32_t* Tm = reinterpret_cast<const uint32_t*>(mat ? up_ptr[e] : gate_ptr[e]);
    const uint32_t* To = reinterpret_cast<const uint32_t*>(mat ? gate_ptr[e] : up_ptr[e]);
    const half* sg = suh_g + (size_t)e * K;
    const half* su = suh_u + (size_t)e * K;
    const int nb = (cnt + 15) >> 4;
    const GuEpi ep{svh_g, svh_u, suh_d, xd, e, cb, N, first, cnt, limit, act_mode};
    switch (k2s[e] * 16 + nb) {
#define TF_PE_GU(K2_, NB_)                                                                                         \
    case K2_ * 16 + NB_: {                                                                                        \
        if constexpr (K2_ != 10 || K10) { \
        float acc[GU_MT][2 * NB_][4];                                                                             \
        gu_mainloop<CB, K2_, NB_>(x, xstride, Tm, To, NS, sg, su, rows_sh, cnt, cb, K, smem_raw, acc);            \
        gu_epilogue<NB_>(acc, smem_raw, ep);                                                                      \
        } else { __trap(); } \
        break;                                                                                                    \
    }
#if TF_PE_GPM > 112
#define TF_PE_GU8(K2_) TF_PE_GU(K2_, 8)
#else
#define TF_PE_GU8(K2_)
#endif
#define TF_PE_GU_K(K2_) TF_PE_GU(K2_, 1) TF_PE_GU(K2_, 2) TF_PE_GU(K2_, 3) TF_PE_GU(K2_, 4) TF_PE_GU(K2_, 5) \
                        TF_PE_GU(K2_, 6) TF_PE_GU(K2_, 7) TF_PE_GU8(K2_)
        TF_PE_GU_K(4)
        TF_PE_GU_K(6)
        TF_PE_GU_K(8)
        TF_PE_GU_K(10)
#undef TF_PE_GU_K
#undef TF_PE_GU8
#undef TF_PE_GU
        default: __trap();
    }
}

// Keep the legacy entry free of the LW=2 mainloops. K10 entry also accepts mixed widths.
template <int CB>
__global__ void __launch_bounds__(GU_THREADS, 1) prompt_gateup_kernel(
    const __nv_bfloat16* __restrict__ x, int xstride, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ gate_ptr, const int64_t* __restrict__ up_ptr,
    const int* __restrict__ k2s, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
    const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
    half* __restrict__ xd, int K, int N, int NS, int slots, float limit, int act_mode, float4* __restrict__ zero,
    long long zero_n) {
    prompt_gateup_body<CB, false>(x, xstride, sorted, items, item_count, gate_ptr, up_ptr, k2s, suh_g, suh_u,
                                  svh_g, svh_u, suh_d, xd, K, N, NS, slots, limit, act_mode, zero, zero_n);
}

extern "C" __global__ void __launch_bounds__(GU_THREADS, 1) tf_pe_gu_k10(
    const __nv_bfloat16* __restrict__ x, int xstride, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ gate_ptr, const int64_t* __restrict__ up_ptr,
    const int* __restrict__ k2s, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
    const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
    half* __restrict__ xd, int K, int N, int NS, int slots, float limit, int act_mode, float4* __restrict__ zero,
    long long zero_n) {
    prompt_gateup_body<1, true>(x, xstride, sorted, items, item_count, gate_ptr, up_ptr, k2s, suh_g, suh_u,
                                  svh_g, svh_u, suh_d, xd, K, N, NS, slots, limit, act_mode, zero, zero_n);
}

// ---- down ---------------------------------------------------------------------------------------------------------
// Block (item of up to IPM members, group of ncb column blocks of 256): 16 warps; warp w owns m tile w of each
// column block. Steps (column block, 128-k chunk of the I-wide input) run in order: the item's xd chunk lands by
// cp.async a step ahead (two stages), trellis words roll a step ahead in registers. After a column block's chunks the
// epilogue (32 members a pass): output FWHT, svh_d, routing weight, red.add into out. Grid: items fastest, so the
// blocks in flight share a column slab of out (it stays in L2 for the atomics).
// ブロック (最大 IPM メンバーの item、256 列の列ブロック ncb 個の組): 16 warp。warp w は各列ブロックの m タイル w を
// 受け持つ。ステップ (列ブロック、幅 I の入力の 128-k チャンク) は順に進む: item の xd チャンクは cp.async で
// 1 ステップ先に届き (2 段)、trellis の語はレジスタで 1 ステップ先に回る。列ブロックのチャンクを終えたら epilogue
// (1 パス 32 メンバー): 出力の FWHT、svh_d、routing 重み、out への red.add。grid は item が最速なので、実行中の
// ブロックは out の同じ列の slab を共有する (atomic のために L2 に残る)。
constexpr int DN_STAGES = 2;
constexpr size_t DN_XS = (size_t)IPM * CK * 2;                // a stage / 1 段分
constexpr int DN_ES = 256 + 4;
constexpr int DN_EROWS = 32;
constexpr size_t DN_SMEM = DN_STAGES * DN_XS + (size_t)DN_EROWS * DN_ES * 4;

struct DnEpi {
    const half* svh_d;
    const float* wts_sh;      // the item's routing weights (shared memory) / item の routing 重み (共有メモリ)
    float* out;
    const int* pair_sh;       // the item's rows of out (row * D), shared memory / item の out 行 (row * D)、共有メモリ
    bool f16;                 // out is the fp16 accumulation buffer / out は fp16 の加算用バッファ
    int e, D, slots, cnt;
};

// svh_d of the lane's 4 + 4 columns of a column block, as two uint2 of half4 (fetched a column block ahead)
// 列ブロックのうち lane の 4 + 4 列の svh_d を、half4 の uint2 2 つで返す (1 列ブロック先に読む)
__device__ __forceinline__ void dn_svh_fetch(const DnEpi& p, int cb, uint2 (&sv)[2]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk)
        sv[blk] = __ldg(reinterpret_cast<const uint2*>(p.svh_d + (size_t)p.e * p.D + cb * 256 + blk * 128 + 4 * lane));
}

template <int NB>
__device__ __forceinline__ void dn_epilogue(const float (&acc)[2 * NB][4], float* Ed, int cb, const DnEpi& p,
                                            const uint2 (&svh)[2]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    float sv[2][4];
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&svh[blk].x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&svh[blk].y));
        sv[blk][0] = a.x * HAD_SCALE; sv[blk][1] = a.y * HAD_SCALE;
        sv[blk][2] = b.x * HAD_SCALE; sv[blk][3] = b.y * HAD_SCALE;
    }
    constexpr int PASSES = (2 * NB * 8 + DN_EROWS - 1) / DN_EROWS;
#pragma unroll
    for (int hf = 0; hf < PASSES; ++hf) {
        __syncthreads();                                       // Ed free / Ed が空いた
#pragma unroll
        for (int q = 0; q < DN_EROWS / 8; ++q) {
            const int jn = hf * (DN_EROWS / 8) + q;
            if (jn < 2 * NB) {
                const int r = q * 8 + 2 * t, c = warp * 16 + g;
                float* E0 = Ed + (size_t)r * DN_ES + c;
                E0[0] = acc[jn][0];
                E0[DN_ES] = acc[jn][1];
                E0[8] = acc[jn][2];
                E0[DN_ES + 8] = acc[jn][3];
            }
        }
        __syncthreads();
        // a warp's DN_EROWS / NW members x both 128 blocks at once: independent butterflies hide the shuffle latency
        // warp の DN_EROWS / NW メンバー x 両方の 128 ブロックを一度に: 独立な butterfly で shuffle の待ちを隠す
        constexpr int RW = DN_EROWS / NW;
        float v[RW * 2][4];
#pragma unroll
        for (int ri = 0; ri < RW; ++ri)
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const float4 v4 =
                    *reinterpret_cast<const float4*>(Ed + (size_t)(warp + ri * NW) * DN_ES + blk * 128 + 4 * lane);
                v[ri * 2 + blk][0] = v4.x; v[ri * 2 + blk][1] = v4.y; v[ri * 2 + blk][2] = v4.z; v[ri * 2 + blk][3] = v4.w;
            }
        fwht128_n<RW * 2>(v, lane);
#pragma unroll
        for (int ri = 0; ri < RW; ++ri) {
            const int i = hf * DN_EROWS + warp + ri * NW;
            if (i < p.cnt) {
                const float w = p.wts_sh[i];
                const size_t orow = (size_t)p.pair_sh[i] + cb * 256 + 4 * lane;
#pragma unroll
                for (int blk = 0; blk < 2; ++blk) {
                    const float* u = v[ri * 2 + blk];
                    if (p.f16)
                        red_add_f16x4(reinterpret_cast<half*>(p.out) + orow + blk * 128, u[0] * sv[blk][0] * w, u[1] * sv[blk][1] * w,
                               u[2] * sv[blk][2] * w, u[3] * sv[blk][3] * w);
                    else
                        red_add_v4(p.out + orow + blk * 128, u[0] * sv[blk][0] * w, u[1] * sv[blk][1] * w,
                               u[2] * sv[blk][2] * w, u[3] * sv[blk][3] * w);
                }
            }
        }
    }
}

template <int CB, int K2, int NB>
__device__ __forceinline__ void dn_item(const half* __restrict__ xd, const uint32_t* __restrict__ Td, int NS,
                                        int first, int cb0, int ncb, int K, unsigned char* smem, const DnEpi& ep) {
    constexpr int TW = Fmt<K2>::TW;
    static_assert(K2 == 4 || K2 == 6 || K2 == 8 || K2 == 10);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const LaneMap<K2> map(lane);
    const int l7 = lane & 7;
    const int moff = (l7 + ((lane >> 4) & 1) * 8) * CK + ((((lane >> 3) & 1) ^ l7) & 7) * 8;
    float* Ed = reinterpret_cast<float*>(smem + DN_STAGES * DN_XS);
    const int kchunks = K / CK, steps = ncb * kchunks;
    const int cnt = ep.cnt;
    auto xs_of = [&](int b) { return reinterpret_cast<half*>(smem + b * DN_XS); };
    auto load = [&](int step) {
        if (step >= steps) return;
        half* dst = xs_of(step % DN_STAGES);
        const int k0 = (step % kchunks) * CK;
        for (int idx = threadIdx.x; idx < NB * 16 * (CK / 8); idx += NTHREADS) {
            const int i = idx >> 4, j = idx & 15;
            const bool ok = i < cnt;
            cp_async16(dst + swz(i, j * 8), xd + (size_t)(first + (ok ? i : 0)) * K + k0 + j * 8, ok);
        }
    };
    const size_t kstride = (size_t)NS * TW;
    auto tile_ptr = [&](int step) {
        const int cb = cb0 + step / kchunks;
        return Td + (size_t)(cb * 16 + warp) * TW + (size_t)(step % kchunks) * 8 * kstride + lane;
    };
    // L2 prefetch of a step's trellis for the block: per k tile its 16 m tiles are 16 TW contiguous words (TW / 2
    // lines of 128 bytes); thread q < 8 * (TW / 2) takes one line
    // ブロックのために 1 ステップ分の trellis を L2 へ先読みする: k タイルごとに m タイル 16 枚は連続した 16 TW 語
    // (128 バイトの行 TW / 2 本)。スレッド q < 8 * (TW / 2) が 1 行ずつ受け持つ
    constexpr int PF_LINES = TW / 2;
    auto prefetch = [&](int step) {
        if (threadIdx.x < 8 * PF_LINES && step < steps) {
            const int pk = threadIdx.x / PF_LINES, pl = threadIdx.x % PF_LINES;
            const int cb = cb0 + step / kchunks;
            prefetch_l2(Td + (size_t)(cb * 16) * TW + (size_t)((step % kchunks) * 8 + pk) * kstride + pl * 32);
        }
    };
    prefetch(1);
    prefetch(2);
    uint32_t wbuf[8][Fmt<K2>::LW];
    {
        const uint32_t* tp = tile_ptr(0);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) load_words<K2>(wbuf[kk], tp + (size_t)kk * kstride, lane);
    }
    load(0);
    cp_async_commit();
    float acc[2 * NB][4];
    uint2 svh[2];
    for (int step = 0; step < steps; ++step) {
        const int kc = step % kchunks;
        if (kc == 0) {
            dn_svh_fetch(ep, cb0 + step / kchunks, svh);
#pragma unroll
            for (int j = 0; j < 2 * NB; ++j)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
        }
        cp_async_wait<0>();
        // stage step landed; stage step + 1 free
        // 段 step は到着済み、段 step + 1 は空き
        __syncthreads();
        load(step + 1);
        cp_async_commit();
        prefetch(step + 3);
        const half* xs = xs_of(step % DN_STAGES);
        const uint32_t* wnext = step + 1 < steps ? tile_ptr(step + 1) : nullptr;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t w1[Fmt<K2>::LW];
#pragma unroll
            for (int j = 0; j < Fmt<K2>::LW; ++j) w1[j] = wbuf[kk][j];
            if (wnext) load_words<K2>(wbuf[kk], wnext + (size_t)kk * kstride, lane);
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w1, map, lane, b0, b1);
            const uint32_t A[4] = {b0[0], b1[0], b0[1], b1[1]};
            const half* xk = xs + (moff ^ ((2 * (kk & 3)) << 3)) + (kk >= 4 ? 64 : 0);
#pragma unroll
            for (int jb = 0; jb < NB; ++jb) {
                uint32_t r[4];
                ldsm_x4(r, xk + jb * 16 * CK);
                const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                mma16816(acc[2 * jb], A, bl);
                mma16816(acc[2 * jb + 1], A, bh);
            }
        }
        if (kc == kchunks - 1) dn_epilogue<NB>(acc, Ed, cb0 + step / kchunks, ep, svh);
    }
}

template <int CB, bool K10>
__device__ __forceinline__ void prompt_down_body(
    const half* __restrict__ xd, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ down_ptr, const int* __restrict__ k2s,
    const half* __restrict__ svh_d, const float* __restrict__ wts, float* __restrict__ out, int K, int D, int slots,
    int ncb, int f16) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int pair_sh[IPM];
    __shared__ float wts_sh[IPM];
    // items fastest: a column slab of out stays in L2
    // item が最速: out の列の slab が L2 に残る
    const int item = blockIdx.x;
    if (item >= item_count[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < IPM) {
        const int pr = (int)threadIdx.x < cnt ? sorted[first + threadIdx.x] : 0;
        pair_sh[threadIdx.x] = (pr / slots) * D;
        wts_sh[threadIdx.x] = (int)threadIdx.x < cnt ? wts[pr] : 0.f;
    }
    const uint32_t* Td = reinterpret_cast<const uint32_t*>(down_ptr[e]);
    const DnEpi ep{svh_d, wts_sh, out, pair_sh, f16 != 0, e, D, slots, cnt};
    const int nb = (cnt + 15) >> 4;
    switch (k2s[e] * 16 + nb) {
#define TF_PE_DN(K2_, NB_) \
    case K2_ * 16 + NB_: \
        if constexpr (K2_ != 10 || K10) \
            dn_item<CB, K2_, NB_>(xd, Td, D >> 4, first, blockIdx.y * ncb, ncb, K, smem_raw, ep); \
        else { __trap(); } \
        break;
#if TF_PE_GPM > 112
#define TF_PE_DN8(K2_) TF_PE_DN(K2_, 8)
#else
#define TF_PE_DN8(K2_)
#endif
#define TF_PE_DN_K(K2_) TF_PE_DN(K2_, 1) TF_PE_DN(K2_, 2) TF_PE_DN(K2_, 3) TF_PE_DN(K2_, 4) TF_PE_DN(K2_, 5) \
                        TF_PE_DN(K2_, 6) TF_PE_DN(K2_, 7) TF_PE_DN8(K2_)
        TF_PE_DN_K(4)
        TF_PE_DN_K(6)
        TF_PE_DN_K(8)
        TF_PE_DN_K(10)
#undef TF_PE_DN_K
#undef TF_PE_DN8
#undef TF_PE_DN
        default: __trap();
    }
}

// Keep the legacy entry free of the LW=2 mainloops. K10 entry also accepts mixed widths.
template <int CB>
__global__ void __launch_bounds__(NTHREADS, 1) prompt_down_kernel(
    const half* __restrict__ xd, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ down_ptr, const int* __restrict__ k2s,
    const half* __restrict__ svh_d, const float* __restrict__ wts, float* __restrict__ out, int K, int D, int slots,
    int ncb, int f16) {
    prompt_down_body<CB, false>(xd, sorted, items, item_count, down_ptr, k2s, svh_d, wts, out, K, D, slots, ncb, f16);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1) tf_pe_dn_k10(
    const half* __restrict__ xd, const int* __restrict__ sorted, const int* __restrict__ items,
    const int* __restrict__ item_count, const int64_t* __restrict__ down_ptr, const int* __restrict__ k2s,
    const half* __restrict__ svh_d, const float* __restrict__ wts, float* __restrict__ out, int K, int D, int slots,
    int ncb, int f16) {
    prompt_down_body<1, true>(xd, sorted, items, item_count, down_ptr, k2s, svh_d, wts, out, K, D, slots, ncb, f16);
}

#ifndef TF_ZIG  // host section / ホスト側
template <int CB>
void launch_cb(const at::Tensor& x, const at::Tensor& sorted, const at::Tensor& items, const at::Tensor& item_count,
               const at::Tensor& gate_ptr, const at::Tensor& up_ptr, const at::Tensor& down_ptr,
               const at::Tensor& gu_k2, const at::Tensor& d_k2, const at::Tensor& suh_g, const at::Tensor& suh_u,
               const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
               const at::Tensor& wts, at::Tensor& xd, at::Tensor& out, int D, int I, int NS, int slots, int max_items,
               float limit, int act_mode, int ncb, int which, int f16, cudaStream_t stream) {
    static bool once = [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(prompt_gateup_kernel<CB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)GU_SMEM));
        C10_CUDA_CHECK(cudaFuncSetAttribute(prompt_down_kernel<CB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)DN_SMEM));
        return true;
    }();
    (void)once;
    auto h = [](const at::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); };
    if (which & 1)
    prompt_gateup_kernel<CB><<<dim3((unsigned)(I / 128), (unsigned)max_items), GU_THREADS, GU_SMEM, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0), sorted.data_ptr<int>(),
        items.data_ptr<int>(), item_count.data_ptr<int>(), gate_ptr.data_ptr<int64_t>(), up_ptr.data_ptr<int64_t>(),
        gu_k2.data_ptr<int>(), h(suh_g), h(suh_u), h(svh_g), h(svh_u), h(suh_d), reinterpret_cast<half*>(xd.data_ptr()),
        D, I, NS, slots, limit, act_mode, reinterpret_cast<float4*>(out.data_ptr()),
        (long long)(out.numel() * out.element_size() / 16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (which & 2)
    prompt_down_kernel<CB><<<dim3((unsigned)max_items, (unsigned)(D / 256 / ncb)), NTHREADS, DN_SMEM, stream>>>(
        reinterpret_cast<const half*>(xd.data_ptr()), sorted.data_ptr<int>(), items.data_ptr<int>(),
        item_count.data_ptr<int>(), down_ptr.data_ptr<int64_t>(), d_k2.data_ptr<int>(), h(svh_d),
        wts.data_ptr<float>(), reinterpret_cast<float*>(out.data_ptr()), I, D, slots, ncb, f16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif  // TF_ZIG host section / TF_ZIG のホスト側
}  // namespace
#ifndef TF_ZIG  // host section / ホスト側

int64_t exl3p_item_rows() { return IPM; }

void exl3p_route_cuda(const at::Tensor& pick, int64_t n, int64_t E, at::Tensor& sorted, at::Tensor& items,
                      at::Tensor& item_count, int64_t max_items, int64_t by_count) {
    TORCH_CHECK(E <= 4096, "route: at most 4096 experts");
    const size_t smem = (size_t)3 * E * sizeof(int);
    route_kernel<<<1, ROUTE_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), (int)n, (int)E, sorted.data_ptr<int>(), items.data_ptr<int>(),
        item_count.data_ptr<int>(), (int)max_items, (int)by_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3p_experts_cuda(const at::Tensor& x, const at::Tensor& sorted, const at::Tensor& items,
                        const at::Tensor& item_count, const at::Tensor& gate_ptr, const at::Tensor& up_ptr,
                        const at::Tensor& down_ptr, const at::Tensor& gu_k2, const at::Tensor& d_k2,
                        const at::Tensor& suh_g, const at::Tensor& suh_u, const at::Tensor& svh_g,
                        const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
                        const at::Tensor& wts, at::Tensor& xd, at::Tensor& out, int64_t D, int64_t I, int64_t NS,
                        int64_t slots, int64_t max_items, double limit, int64_t act_mode, int64_t cb, int64_t ncb,
                        int64_t which, int64_t f16) {
    TORCH_CHECK(D % 256 == 0 && I % 128 == 0, "prompt experts: D % 256, I % 128");
    TORCH_CHECK((D / 256) % ncb == 0, "prompt experts: ncb must divide D / 256");
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_PE_ARGS x, sorted, items, item_count, gate_ptr, up_ptr, down_ptr, gu_k2, d_k2, suh_g, suh_u, svh_g, svh_u, \
                   suh_d, svh_d, wts, xd, out, (int)D, (int)I, (int)NS, (int)slots, (int)max_items, (float)limit,   \
                   (int)act_mode, (int)ncb, (int)which, (int)f16, stream
    if (cb == 0) launch_cb<0>(TF_PE_ARGS);
    else if (cb == 1) launch_cb<1>(TF_PE_ARGS);
    else if (cb == 2) launch_cb<2>(TF_PE_ARGS);
    else TORCH_CHECK(false, "codebook must be 0, 1 or 2");
#undef TF_PE_ARGS
}
#endif  // TF_ZIG host section / TF_ZIG のホスト側

#ifdef TF_ZIG
// The instantiations the GLM-5.3 engine launches: codebook 1 (mcg).
// GLM-5.3 エンジンが起動するインスタンス: codebook 1 (mcg)。
namespace tf_exl3p {
template __global__ void prompt_gateup_kernel<1>(const __nv_bfloat16*, int, const int*, const int*, const int*,
                                                 const int64_t*, const int64_t*, const int*, const half*, const half*,
                                                 const half*, const half*, const half*, half*, int, int, int, int,
                                                 float, int, float4*, long long);
template __global__ void prompt_down_kernel<1>(const half*, const int*, const int*, const int*, const int64_t*,
                                               const int*, const half*, const float*, float*, int, int, int, int, int);
}  // namespace tf_exl3p
#endif
