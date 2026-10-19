// RoCE one-shot all-reduce and all-gather, written with reference to b12x's
// (https://github.com/local-inference-lab/b12x, Apache-2.0).
// b12x (Apache-2.0) を参考に書いた RoCE の one-shot all-reduce と all-gather。
// Over the RDMA proxy of roce_proxy.c; the all-reduce sums in fixed rank order, the all-gather copies in rank order.
// ctrl: u32 seq, nbytes, error seq, missing peer; u32 nbytes[2] per slot at 16; u32 missing hca at 24.
// roce_proxy.c の RDMA proxy の上で動く。all-reduce はランク順に固定して足し、all-gather はランク順にコピーする。
// ctrl: u32 seq, nbytes, error seq, missing peer。16 から slot ごとの u32 nbytes[2]、24 に u32 missing hca。

#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_v4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// 0 on a match, 1 after `limit` polls
// 一致したら 0、`limit` 回読んでも一致しなければ 1
__device__ __forceinline__ uint32_t spin_eq(const uint32_t* p, uint32_t want, uint32_t limit) {
    for (uint32_t polls = 0;;) {
        uint32_t seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(p) : "memory");
        if (seen == want) return 0;
        if (++polls >= limit) return 1;
    }
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
// two round-to-nearest bf16 conversions, low half first
// bf16 への最近接丸めを 2 回、下位の半分が先
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    unsigned short a, b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(a) : "f"(lo));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(hi));
    return (uint32_t)a | ((uint32_t)b << 16);
}

struct Region {
    uint64_t recv, flag, send, ctrl, slot_bytes;
    uint32_t* epoch;
    uint32_t* stage_ctr;
    uint32_t* tail_ctr;
    uint32_t* poison;
    uint32_t spin_limit;
    int world, rank, slots, hcas, flag_stride;
};

// stage into send[seq & 1], ring the doorbell (nbytes, then seq), wait on every peer's per-HCA flag (a timeout
// poisons the runtime); returns seq and whether the data phase may run
// send[seq & 1] に書き込み、doorbell を鳴らし (nbytes、次に seq)、各 peer の HCA ごとのフラグを待つ (タイムアウトで
// ランタイムを poison する)。seq と、データの段に進んでよいかを返す
__device__ __forceinline__ bool stage_and_wait(const Region& g, const uint8_t* in, int packs, uint32_t nbytes,
                                               uint32_t& seq) {
    const uint32_t epoch = ld_relaxed_gpu(g.epoch);
    seq = epoch + 1;
    const uint64_t slot = seq & 1u;
    uint8_t* send_slot = (uint8_t*)(g.send + slot * g.slot_bytes);
    const int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    if (ld_relaxed_gpu(g.poison) != 0) return false;
    for (int i = index; i < packs; i += stride) st_v4(send_slot + (uint64_t)i * 16, ld_v4(in + (uint64_t)i * 16));
    __syncthreads();
    uint32_t* ctrl = (uint32_t*)g.ctrl;
    if (threadIdx.x == 0) {
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(g.stage_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0) {
            st_relaxed_sys(ctrl + 1, nbytes);
            st_relaxed_sys(ctrl + 4 + slot, nbytes);
            fence_sc_sys();
            st_relaxed_sys(ctrl, seq);
        }
    }
    if ((int)threadIdx.x < g.world * g.hcas) {
        const int peer = threadIdx.x / g.hcas, hca = threadIdx.x - peer * g.hcas;
        if (peer != g.rank) {
            const uint32_t* flag = (const uint32_t*)(g.flag + (((uint64_t)peer * g.slots + slot) * g.hcas + hca) * g.flag_stride);
            if (spin_eq(flag, seq, g.spin_limit)) {
                st_relaxed_sys(ctrl + 3, (uint32_t)peer);
                st_relaxed_sys(ctrl + 6, (uint32_t)hca);
                st_relaxed_sys(ctrl + 2, seq);
                st_release_gpu(g.poison, seq);
            }
        }
    }
    __syncthreads();
    return ld_relaxed_gpu(g.poison) == 0;
}

// the last block to finish publishes the next epoch, unless a wait failed
// 最後に終えたブロックが次の epoch を公開する。待ちが失敗したときは公開しない
__device__ __forceinline__ void tail(const Region& g, uint32_t seq) {
    fence_sc_gpu();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(g.tail_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0) {
            fence_sc_gpu();
            if (ld_relaxed_sys((const uint32_t*)g.ctrl + 2) == 0) st_release_gpu(g.epoch, seq);
        }
    }
}

template <bool BF16>
__device__ void allreduce(const uint8_t* in, uint8_t* out, int packs, uint32_t nbytes, Region g) {
    uint32_t seq;
    if (ld_relaxed_gpu(g.poison) != 0) return;    // poisoned: no stage, no tail / poison 済み: stage も tail もしない
    const bool ok = stage_and_wait(g, in, packs, nbytes, seq);
    if (ok) {
        const uint64_t slot = seq & 1u;
        const int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
        for (int i = index; i < packs; i += stride) {
            const uint64_t off = (uint64_t)i * 16;
            float acc[8];
            for (int s = 0; s < g.world; s++) {
                const uint4 w = s == g.rank ? ld_v4(in + off)
                                            : ld_relaxed_sys_v4((const uint8_t*)(g.recv + ((uint64_t)s * g.slots + slot) * g.slot_bytes) + off);
                const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
                if (BF16) {
                    for (int q = 0; q < 4; q++) {
                        const float lo = bf16_lo(ws[q]), hi = bf16_hi(ws[q]);
                        if (s == 0) { acc[2 * q] = lo; acc[2 * q + 1] = hi; }
                        else { acc[2 * q] = acc[2 * q] + lo; acc[2 * q + 1] = acc[2 * q + 1] + hi; }
                    }
                } else {
                    for (int q = 0; q < 4; q++) {
                        const float v = __uint_as_float(ws[q]);
                        acc[q] = s == 0 ? v : acc[q] + v;
                    }
                }
            }
            uint4 o;
            if (BF16) {
                o.x = pack_bf16x2(acc[0], acc[1]);
                o.y = pack_bf16x2(acc[2], acc[3]);
                o.z = pack_bf16x2(acc[4], acc[5]);
                o.w = pack_bf16x2(acc[6], acc[7]);
            } else {
                o.x = __float_as_uint(acc[0]);
                o.y = __float_as_uint(acc[1]);
                o.z = __float_as_uint(acc[2]);
                o.w = __float_as_uint(acc[3]);
            }
            st_v4(out + off, o);
        }
    }
    tail(g, seq);
}

}  // namespace

extern "C" __global__ void tf_glm53_roce_allreduce_f32(const uint8_t* in, uint8_t* out, int packs, uint32_t nbytes,
                                                        Region g) {
    allreduce<false>(in, out, packs, nbytes, g);
}

extern "C" __global__ void tf_glm53_roce_allreduce_bf16(const uint8_t* in, uint8_t* out, int packs, uint32_t nbytes,
                                                         Region g) {
    allreduce<true>(in, out, packs, nbytes, g);
}

// Shard s lands at column block s of every output row (rows of row_packs 16-byte packs).
// shard s は各出力行の列ブロック s に入る (行は 16 バイトの pack が row_packs 個)。
extern "C" __global__ void tf_glm53_roce_allgather(const uint8_t* in, uint8_t* out, int packs, uint32_t nbytes,
                                                    int row_packs, Region g) {
    uint32_t seq;
    if (ld_relaxed_gpu(g.poison) != 0) return;
    const bool ok = stage_and_wait(g, in, packs, nbytes, seq);
    if (ok) {
        const uint64_t slot = seq & 1u;
        const int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
        const int out_row = g.world * row_packs;
        for (int s = 0; s < g.world; s++) {
            const uint8_t* src = s == g.rank ? in : (const uint8_t*)(g.recv + ((uint64_t)s * g.slots + slot) * g.slot_bytes);
            for (int i = index; i < packs; i += stride) {
                const int row = i / row_packs, col = i - row * row_packs;
                const uint4 w = s == g.rank ? ld_v4(src + (uint64_t)i * 16) : ld_relaxed_sys_v4(src + (uint64_t)i * 16);
                st_v4(out + ((uint64_t)row * out_row + (uint64_t)s * row_packs + col) * 16, w);
            }
        }
    }
    tail(g, seq);
}
