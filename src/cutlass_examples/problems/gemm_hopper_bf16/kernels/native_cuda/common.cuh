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

// PTX supports BF16 WGMMA f32 accumulation for m64n{8..256 step 8}k16.
// Keep wrappers only for GEMM tile widths this kernel is expected to sweep.
#define WGMMA_REG_LIST_16 "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
#define WGMMA_REG_LIST_32 WGMMA_REG_LIST_16 ", %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
#define WGMMA_REG_LIST_48 WGMMA_REG_LIST_32 ", %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47"
#define WGMMA_REG_LIST_64 WGMMA_REG_LIST_48 ", %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
#define WGMMA_REG_LIST_96 WGMMA_REG_LIST_64 ", %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"
#define WGMMA_REG_LIST_128 WGMMA_REG_LIST_96 ", %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"

#define WGMMA_INPUT_LIST_16 "%16, %17, %18"
#define WGMMA_INPUT_LIST_32 "%32, %33, %34"
#define WGMMA_INPUT_LIST_48 "%48, %49, %50"
#define WGMMA_INPUT_LIST_64 "%64, %65, %66"
#define WGMMA_INPUT_LIST_96 "%96, %97, %98"
#define WGMMA_INPUT_LIST_128 "%128, %129, %130"

#define WGMMA_OUT_LIST_16(d) \
    "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), "+f"((d)[4]), "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), \
    "+f"((d)[8]), "+f"((d)[9]), "+f"((d)[10]), "+f"((d)[11]), "+f"((d)[12]), "+f"((d)[13]), "+f"((d)[14]), "+f"((d)[15])
#define WGMMA_OUT_LIST_32(d) WGMMA_OUT_LIST_16(d), \
    "+f"((d)[16]), "+f"((d)[17]), "+f"((d)[18]), "+f"((d)[19]), "+f"((d)[20]), "+f"((d)[21]), "+f"((d)[22]), "+f"((d)[23]), \
    "+f"((d)[24]), "+f"((d)[25]), "+f"((d)[26]), "+f"((d)[27]), "+f"((d)[28]), "+f"((d)[29]), "+f"((d)[30]), "+f"((d)[31])
#define WGMMA_OUT_LIST_48(d) WGMMA_OUT_LIST_32(d), \
    "+f"((d)[32]), "+f"((d)[33]), "+f"((d)[34]), "+f"((d)[35]), "+f"((d)[36]), "+f"((d)[37]), "+f"((d)[38]), "+f"((d)[39]), \
    "+f"((d)[40]), "+f"((d)[41]), "+f"((d)[42]), "+f"((d)[43]), "+f"((d)[44]), "+f"((d)[45]), "+f"((d)[46]), "+f"((d)[47])
#define WGMMA_OUT_LIST_64(d) WGMMA_OUT_LIST_48(d), \
    "+f"((d)[48]), "+f"((d)[49]), "+f"((d)[50]), "+f"((d)[51]), "+f"((d)[52]), "+f"((d)[53]), "+f"((d)[54]), "+f"((d)[55]), \
    "+f"((d)[56]), "+f"((d)[57]), "+f"((d)[58]), "+f"((d)[59]), "+f"((d)[60]), "+f"((d)[61]), "+f"((d)[62]), "+f"((d)[63])
#define WGMMA_OUT_LIST_96(d) WGMMA_OUT_LIST_64(d), \
    "+f"((d)[64]), "+f"((d)[65]), "+f"((d)[66]), "+f"((d)[67]), "+f"((d)[68]), "+f"((d)[69]), "+f"((d)[70]), "+f"((d)[71]), \
    "+f"((d)[72]), "+f"((d)[73]), "+f"((d)[74]), "+f"((d)[75]), "+f"((d)[76]), "+f"((d)[77]), "+f"((d)[78]), "+f"((d)[79]), \
    "+f"((d)[80]), "+f"((d)[81]), "+f"((d)[82]), "+f"((d)[83]), "+f"((d)[84]), "+f"((d)[85]), "+f"((d)[86]), "+f"((d)[87]), \
    "+f"((d)[88]), "+f"((d)[89]), "+f"((d)[90]), "+f"((d)[91]), "+f"((d)[92]), "+f"((d)[93]), "+f"((d)[94]), "+f"((d)[95])
#define WGMMA_OUT_LIST_128(d) WGMMA_OUT_LIST_96(d), \
    "+f"((d)[96]), "+f"((d)[97]), "+f"((d)[98]), "+f"((d)[99]), "+f"((d)[100]), "+f"((d)[101]), "+f"((d)[102]), "+f"((d)[103]), \
    "+f"((d)[104]), "+f"((d)[105]), "+f"((d)[106]), "+f"((d)[107]), "+f"((d)[108]), "+f"((d)[109]), "+f"((d)[110]), "+f"((d)[111]), \
    "+f"((d)[112]), "+f"((d)[113]), "+f"((d)[114]), "+f"((d)[115]), "+f"((d)[116]), "+f"((d)[117]), "+f"((d)[118]), "+f"((d)[119]), \
    "+f"((d)[120]), "+f"((d)[121]), "+f"((d)[122]), "+f"((d)[123]), "+f"((d)[124]), "+f"((d)[125]), "+f"((d)[126]), "+f"((d)[127])

template <int N, bool ScaleD>
struct Wgmma_m64nNk16_bf16;

#define DEFINE_WGMMA_M64NNK16_BF16(N, NUM_REGS)                                                 \
template <bool ScaleD>                                                                          \
struct Wgmma_m64nNk16_bf16<N, ScaleD> {                                                         \
    __device__ __forceinline__ static void run(uint64_t desc_a, uint64_t desc_b, float* d) {    \
        asm volatile(                                                                           \
            "wgmma.mma_async.sync.aligned.m64n" #N "k16.f32.bf16.bf16 "                         \
            "{" WGMMA_REG_LIST_##NUM_REGS "}, "                                                 \
            WGMMA_INPUT_LIST_##NUM_REGS ", 1, 1, 0, 0;\n"                                       \
            : WGMMA_OUT_LIST_##NUM_REGS(d)                                                      \
            : "l"(desc_a), "l"(desc_b), "n"(ScaleD ? 1 : 0)                                     \
            : "memory");                                                                        \
    }                                                                                           \
};

DEFINE_WGMMA_M64NNK16_BF16(32, 16)
DEFINE_WGMMA_M64NNK16_BF16(64, 32)
DEFINE_WGMMA_M64NNK16_BF16(96, 48)
DEFINE_WGMMA_M64NNK16_BF16(128, 64)
DEFINE_WGMMA_M64NNK16_BF16(192, 96)
DEFINE_WGMMA_M64NNK16_BF16(256, 128)

#undef DEFINE_WGMMA_M64NNK16_BF16

template <int N, bool ScaleD>
__device__ __forceinline__ void wgmma_m64nNk16_bf16(
    uint64_t desc_a, uint64_t desc_b,
    float* d
) {
    static_assert(N == 32 || N == 64 || N == 96 || N == 128 || N == 192 || N == 256,
                  "This kernel wraps WGMMA only for BN in {32, 64, 96, 128, 192, 256}.");
    Wgmma_m64nNk16_bf16<N, ScaleD>::run(desc_a, desc_b, d);
}

enum class Sm90GmmaSwizzleMode : uint64_t {
    None = 0,
    // PTX WGMMA descriptor encoding: 1=128B, 2=64B, 3=32B.
    // This is intentionally different from CUtensorMapSwizzle enum ordering.
    Swizzle128B = 1,
    Swizzle64B = 2,
    Swizzle32B = 3,
};

template <int KStrideElems>
__host__ __device__ constexpr Sm90GmmaSwizzleMode sm90_bf16_k_major_swizzle() {
    constexpr int row_bytes = KStrideElems * static_cast<int>(sizeof(nv_bfloat16));
    static_assert(row_bytes == 32 || row_bytes == 64 || row_bytes == 128,
                  "BF16 K-major WGMMA supports BK in {16, 32, 64}.");
    return row_bytes == 128 ? Sm90GmmaSwizzleMode::Swizzle128B :
           row_bytes == 64 ? Sm90GmmaSwizzleMode::Swizzle64B :
                             Sm90GmmaSwizzleMode::Swizzle32B;
}

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
    static_assert((8ull * KStrideElems * sizeof(nv_bfloat16)) <= 0x3FFFFull,
                  "WGMMA descriptor stride byte offset exceeds the encodable range.");

    constexpr Sm90GmmaSwizzleMode swizzle = sm90_bf16_k_major_swizzle<KStrideElems>();

    // PTX SM90 GMMA K-major canonical layouts:
    //   Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1,T))
    // The descriptor is independent of the WGMMA N dimension and row stride,
    // as long as the row remains K-contiguous and matches the TMA swizzle.
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
