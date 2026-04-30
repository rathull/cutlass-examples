#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ==================================================
//                 TMA UTILITIES
// ==================================================
// TODO: which of the below assmelby utils have a CUDA wrapped version like in cuda::barrier?

// Generic pointer -> SMEM 32-bit offset
__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint64_t addr;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(addr) : "l"(ptr));
    return static_cast<uint32_t>(addr);
}

// Initialize mbarrier in SMEM with given arrival count
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(bar_addr), "r"(count)
                 : "memory");
}

// Make mbarrier visible to async proxy
// TODO: why do we do this instead of fence.proxy.async.shared::cta? And is the below PTX cluster-scoped instead of CTA-scoped?
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Arrive and set expected transaction bytes
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_count) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar_addr), "r"(tx_count)
                 : "memory");
}

// Arrive without tx-count
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(bar_addr)
                 : "memory");
}

// Spin wait on a barrier with specified phase bit
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_addr = cvta_to_shared_u32(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"  // TODO: should I try_wait or test_wait?
        "@P1 bra DONE_%=;\n"
        "bra LAB_WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "r"(bar_addr), "r"(phase)
        : "memory");
}

// Issue a 2D TMA load
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* mbar,
    void* smem_ptr, int32_t coord_0, int32_t coord_1
) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.tile"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(desc),
           "r"(coord_0), "r"(coord_1), "r"(mbar_addr)
        : "memory");
}

// ==================================================
//                WGMMA UTILITIES
// ==================================================   
template <int NumRegs>
__device__ __forceinline__ void warpgroup_reg_deallocate() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(NumRegs));
}

template <int NumRegs>
__device__ __forceinline__ void warpgroup_reg_allocate() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(NumRegs));
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7);
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
// 32 fp32 accumulators per thread for n64
// TODO: template this over N
template <bool ScaleD>
__device__ __forceinline__ void wgmma_m64n64k16_bf16(
    uint64_t desc_a, uint64_t desc_b,
    float* d
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, "
        "%34, 1, 1, 0, 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b),
          "n"(ScaleD ? 1 : 0)
        : "memory");
}

enum class Sm90GmmaSwizzleMode : uint64_t {
    None = 0,
    // PTX WGMMA descriptor encoding: 1=128B, 2=64B, 3=32B.
    // This is intentionally different from CUtensorMapSwizzle enum ordering.
    Swizzle128B = 1,
    Swizzle64B = 2,
    Swizzle32B = 3,
};

struct WgmmaOperandDescriptors {
    uint64_t A;
    uint64_t B;
};

// PTX "Matrix Descriptor Format" for wgmma.mma_async:
// matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4.
__device__ __forceinline__ uint32_t wgmma_matrix_descriptor_encode(uint32_t byte_offset) {
    return (byte_offset & 0x3FFFFu) >> 4;
}

// For swizzled layouts, PTX defines base_offset from the start address of the
// swizzle repeating pattern. This is zero when a 128B-swizzle pattern starts on
// a 1024B boundary.
__device__ __forceinline__ uint32_t wgmma_swizzle_base_offset(
    uint32_t pattern_start_addr,
    Sm90GmmaSwizzleMode swizzle
) {
    return swizzle == Sm90GmmaSwizzleMode::None ? 0u : ((pattern_start_addr >> 7) & 0x7u);
}

template <int KStrideElems>
__device__ __forceinline__ uint64_t make_wgmma_bf16_k_major_smem_desc(
    const nv_bfloat16* matrix_start,
    const nv_bfloat16* swizzle_pattern_start
) {
    static_assert(KStrideElems > 0, "K stride must be positive.");
    static_assert(KStrideElems == 64,
                  "BF16 K-major 128B TMA/WGMMA layout requires exactly 64 contiguous K elements.");
    static_assert((8ull * KStrideElems * sizeof(nv_bfloat16)) <= 0x3FFFFull,
                  "WGMMA descriptor stride byte offset exceeds the encodable range.");

    constexpr Sm90GmmaSwizzleMode swizzle = Sm90GmmaSwizzleMode::Swizzle128B;

    // PTX SM90 GMMA K-major 128B canonical layout:
    //   Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1,T))
    // The descriptor is independent of the WGMMA N dimension. Larger N tiles
    // reuse the same row-major/K-contiguous B descriptor with more accumulator
    // registers and a different wgmma.m64nNk16 opcode.
    // For swizzled K-major layouts, PTX specifies LBO as unused/assumed 1.
    // Descriptor fields are stored in 16-byte units by PTX's encoder.
    constexpr uint32_t leading_byte_offset = 16;
    constexpr uint32_t stride_byte_offset = 8u * KStrideElems * sizeof(nv_bfloat16);

    const uint32_t matrix_start_addr = cvta_to_shared_u32(matrix_start);
    const uint32_t pattern_start_addr = cvta_to_shared_u32(swizzle_pattern_start);

    uint64_t desc = 0;
    desc |= static_cast<uint64_t>(wgmma_matrix_descriptor_encode(matrix_start_addr));
    desc |= static_cast<uint64_t>(wgmma_matrix_descriptor_encode(leading_byte_offset)) << 16;
    desc |= static_cast<uint64_t>(wgmma_matrix_descriptor_encode(stride_byte_offset)) << 32;
    desc |= static_cast<uint64_t>(wgmma_swizzle_base_offset(pattern_start_addr, swizzle)) << 49;
    desc |= static_cast<uint64_t>(swizzle) << 62;
    return desc;
}

template <int KStrideElems>
__device__ __forceinline__ WgmmaOperandDescriptors make_wgmma_bf16_k_major_operand_descs(
    const nv_bfloat16* a_matrix_start,
    const nv_bfloat16* a_swizzle_pattern_start,
    const nv_bfloat16* b_matrix_start,
    const nv_bfloat16* b_swizzle_pattern_start
) {
    return {
        make_wgmma_bf16_k_major_smem_desc<KStrideElems>(a_matrix_start, a_swizzle_pattern_start),
        make_wgmma_bf16_k_major_smem_desc<KStrideElems>(b_matrix_start, b_swizzle_pattern_start),
    };
}

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
