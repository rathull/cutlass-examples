#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ==================================================
//                 TMA UTILITIES
// ==================================================
// TODO: which of the below assmelby utils have a CUDA wrapped version like in cuda::barrier?
// First num_consumer_threads threads issue WGMMAs, the last num_producer_threads threads issue TMAs
// TODO: need to figure out how to handle case where dimensions are not divisble by 64
constexpr int num_consumer_groups = (BM + 63) / 64;
constexpr int num_consumer_threads = 128 * num_consumer_groups;
constexpr int num_producer_threads = 32;
constexpr int wgmma_m = 64;
constexpr int wgmma_n = BN;
constexpr int wgmma_k = 16;

static_assert(BM % 64 == 0, "BM must be divisible by 64");
static_assert(BN % 8 == 0, "BN must be divisible by 8");
static_assert(NUM_THREADS <= 1024, "CTA cannot have more than 1024 threads");
static_assert(NUM_STAGES >= 2, "Need at least 2 pipeline stages.");
static_assert(2 * NUM_STAGES * (BM * BK + BN * BK) + 2 * NUM_STAGES <= 227 * 1024, "SMEM per CTA is too large.");
static_assert(BM * BN / 4 <= 256 * 1024, "Accumulator registers per CTA is too large.");

// Generic pointer -> SMEM 32-bit offset
__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile("cvta.to.shared.u64 %0, %1;\n\t"
                 "cvt.u32.u64 %0, %0;"
                 : "=r"(addr) : "l"(ptr));
    return addr;
}

// Initialize mbarrier in SMEM with given arrival count
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(bar_addr), "r"(count));
}

// Make mbarrier visible to async proxy
// TODO: why do we do this instead of fence.proxy.async.shared::cta? And is the below PTX cluster-scoped instead of CTA-scoped?
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Arrive and set expected transaction bytes
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_count) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar_addr), "r"(tx_count));
}

// Arrive without tx-count
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(bar_addr));
}

// Spin wait on a barrier with specified phase bit
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"  // TODO: should I try_wait or test_wait?
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(bar_addr), "r"(phase));
}

// Issue a 2D TMA load
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* mbar,
    void* smem_ptr, int32_t coord_0, int32_t coord_1
) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(desc),
           "r"(coord_0), "r"(coord_1), "r"(mbar_addr)
        : "memory");
}

// ==================================================
//                WGMMA UTILITIES
// ==================================================   


// ==================================================
//                 LD/ST UTILITIES
// ==================================================
namespace gemm_hopper_bf16::native_cuda {

constexpr int max_bf16_vec_elems(int contiguous_elems) {
    return contiguous_elems % 8 == 0 ? 8 :
           contiguous_elems % 4 == 0 ? 4 :
           contiguous_elems % 2 == 0 ? 2 : 1;
}

template <int VecElems>
struct Bf16Vec;

template <>
struct Bf16Vec<1> {
    using Type = nv_bfloat16;
};

template <>
struct Bf16Vec<2> {
    using Type = unsigned int;
};

template <>
struct Bf16Vec<4> {
    using Type = uint2;
};

template <>
struct Bf16Vec<8> {
    using Type = uint4;
};

template <int VecElems>
__device__ __forceinline__ void copy_bf16_vec(
    nv_bfloat16* __restrict__ dst,
    const nv_bfloat16* __restrict__ src
) {
    using VecType = typename Bf16Vec<VecElems>::Type;
    *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);
}

template <int TileRows, int TileCols, int VecElems, int NumThreads>
__device__ __forceinline__ void load_row_major_tile(
    nv_bfloat16* __restrict__ smem,
    const nv_bfloat16* __restrict__ gmem,
    const int global_row_start,
    const int global_col_start,
    const int global_ld,
    const int valid_rows,
    const int valid_cols,
    const int tid
) {
    static_assert(VecElems == 1 || VecElems == 2 || VecElems == 4 || VecElems == 8);
    static_assert(TileCols % VecElems == 0, "vector copy cannot cross tile rows");

    constexpr int TILE_ELEMENTS = TileRows * TileCols;
    constexpr int THREAD_STRIDE = NumThreads * VecElems;

    for (int linear = tid * VecElems; linear < TILE_ELEMENTS; linear += THREAD_STRIDE) {
        const int row = linear / TileCols;
        const int col = linear % TileCols;

        const int global_offset = (global_row_start + row) * global_ld + global_col_start + col;
        const bool full_vector_in_bounds = row < valid_rows && col + VecElems <= valid_cols;
        const bool global_vector_aligned = global_offset % VecElems == 0;

        if (full_vector_in_bounds && global_vector_aligned) {
            copy_bf16_vec<VecElems>(
                &smem[linear],
                &gmem[global_offset]
            );
        } else {
            #pragma unroll
            for (int i = 0; i < VecElems; ++i) {
                const bool element_in_bounds = row < valid_rows && col + i < valid_cols;
                smem[linear + i] = element_in_bounds
                    ? gmem[global_offset + i]
                    : __float2bfloat16(0.0f);
            }
        }
    }
}

}  // namespace gemm_hopper_bf16::native_cuda
