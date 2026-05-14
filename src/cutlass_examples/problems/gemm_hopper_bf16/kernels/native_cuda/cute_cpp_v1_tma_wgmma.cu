#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#ifndef NUM_STAGES
#define NUM_STAGES 4
#endif
#ifndef ENTRYPOINT
#define ENTRYPOINT cute_cpp_v1_tma_wgmma
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME cute_cpp_v1_tma_wgmma_kernel
#endif

constexpr bool is_supported_wgmma_n(int n) {
    return n == 32 || n == 64 || n == 96 || n == 128 || n == 192 || n == 256;
}

constexpr CUtensorMapSwizzle tma_swizzle_for_bf16_k(int bk) {
    return bk == 64 ? CU_TENSOR_MAP_SWIZZLE_128B :
           bk == 32 ? CU_TENSOR_MAP_SWIZZLE_64B :
                      CU_TENSOR_MAP_SWIZZLE_32B;
}

constexpr int round_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

using bf16 = cutlass::bfloat16_t;  // bitwise equivalent to __nv_bfloat16
constexpr int num_consumer_groups = BM / 64;
constexpr int num_consumer_threads = 128 * num_consumer_groups;
constexpr int num_producer_threads = 128;
constexpr int num_threads = num_consumer_threads + num_producer_threads;
constexpr int regs_per_consumer_thread = 232;
constexpr int regs_per_producer_thread = 40;

constexpr int wgmma_m = 64;
constexpr int wgmma_n = BN;
constexpr int wgmma_k = 16;

// Select SMEM layout based on swizzle atom + tile_to_shape
template<int Bk> struct KMajorAtomBf16;
// 128B: composes Swizzle<3, 4, 3> and inner layout (_8, _64) : (_64, _1) 
// K-major mapping (m, k) -> offset, swizzle re-permutes addresses in every 128B chunk so
// 8 consecutive ldmatrix/WGMMA reads hit 8 different SMEM banks
// (_8, _64) : (_64, _1)  is the smallest tile this swizzle is defined over, so we need to
// tile this up to our block size
template<> struct KMajorAtomBf16<64> { using type = GMMA::Layout_K_SW128_Atom<bf16>; };
template<> struct KMajorAtomBf16<32> { using type = GMMA::Layout_K_SW64_Atom<bf16>;  };
template<> struct KMajorAtomBf16<16> { using type = GMMA::Layout_K_SW32_Atom<bf16>;  };
using SmemLayoutAtom = typename KMajorAtomBf16<BK>::type;

// (BM, BK, NUM_STAGES) -> offset
using SmemLayoutA = decltype(cute::tile_to_shape(
    SmemLayoutAtom{},
    cute::make_shape(Int<BM>{}, Int<BK>{}, Int<NUM_STAGES>{})
));
// (BN, BK, NUM_STAGES) -> offset
using SmemLayoutB = decltype(cute::tile_to_shape(
    SmemLayoutAtom{},
    cute::make_shape(Int<BN>{}, Int<BK>{}, Int<NUM_STAGES>{})
));

constexpr int size_smemA = sizeof(bf16) * cute::cosize_v<SmemLayoutA>;
constexpr int size_smemB = sizeof(bf16) * cute::cosize_v<SmemLayoutB>;
constexpr int size_smem_bar = sizeof(uint64_t) * 2 * NUM_STAGES;
constexpr int size_smem = round_up(size_smemA + size_smemB + size_smem_bar, 1024);  // Align stage bases



// Select MMA atom and TiledMMA
template<int BN_T> struct WgmmaSS;
template<> struct WgmmaSS<32>  { using type = SM90_64x32x16_F32BF16BF16_SS< GMMA::Major::K, GMMA::Major::K>; };
template<> struct WgmmaSS<64>  { using type = SM90_64x64x16_F32BF16BF16_SS< GMMA::Major::K, GMMA::Major::K>; };
template<> struct WgmmaSS<96>  { using type = SM90_64x96x16_F32BF16BF16_SS< GMMA::Major::K, GMMA::Major::K>; };
template<> struct WgmmaSS<128> { using type = SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>; };
template<> struct WgmmaSS<192> { using type = SM90_64x192x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>; };
template<> struct WgmmaSS<256> { using type = SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>; };
using MMAAtom = typename WgmmaSS<BN>::type;
constexpr int max_wgmma_groups_in_flight = NUM_STAGES - 1;

// TODO: need to figure out how to handle case where dimensions are not divisble by 64
static_assert(BM > 0 && BN > 0 && BK > 0, "Tile dimensions must be positive.");
static_assert(BM % 64 == 0, "BM must be divisible by 64");
static_assert(BM <= 128,
              "BM=256 needs lower consumer register allocation; 4 consumer warpgroups at 232 regs can deadlock setmaxnreg.");
static_assert(num_consumer_groups >= 1 && num_consumer_groups <= 2,
              "Current setmaxnreg configuration supports at most 2 consumer warpgroups.");
static_assert(is_supported_wgmma_n(BN),
              "This kernel wraps WGMMA only for BN in {32, 64, 96, 128, 192, 256}.");
static_assert(BK == 16 || BK == 32 || BK == 64,
              "This kernel supports BK in {16, 32, 64}.");
static_assert(BN <= 256 && BK <= 256, "TMA tiled box dimensions must be <= 256.");
static_assert(num_threads <= 1024, "CTA cannot have more than 1024 threads");
static_assert(NUM_STAGES >= 1, "Need at least 1 pipeline stage.");
static_assert(NUM_STAGES <= 8, "wgmma.wait_group supports immediates in [0, 7].");
static_assert(size_smem <= 227 * 1024, "SMEM per CTA is too large.");
static_assert(BM * BN / 4 <= 256 * 1024, "Accumulator registers per CTA is too large.");

// TODO: use this
// struct PipelineState {
//     int stage = 0;  // stage slot
//     int phase = 0;  // mbarrier parity
//     inline __device__ void advance() {
//         ++stage;
//         if (stage == NUM_STAGES) {
//             stage = 0;
//             phase ^= 1;
//         }
//     }
// };

template <class TmaA, class TmaB, class ASmemLayout, class BSmemLayout, class TiledMMA_T>
__global__ __launch_bounds__(num_threads)
void KERNEL_NAME(
    const int M, const int N, const int K, float alpha, float beta,
    const __grid_constant__ TmaA tma_A,
    const __grid_constant__ TmaB tma_B,
    bf16* __restrict__ C,
    ASmemLayout sA_layout,
    BSmemLayout sB_layout,
    TiledMMA_T tiled_mma
) {
    const int tid = threadIdx.x;

    // Allocate SMEM tensors
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    bf16* smemA = reinterpret_cast<bf16*>(smem_buf);
    bf16* smemB = reinterpret_cast<bf16*>(smem_buf + size_smemA);
    uint64_t* full_barrier = reinterpret_cast<uint64_t*>(smem_buf + size_smemA + size_smemB);
    uint64_t* empty_barrier = full_barrier + NUM_STAGES;

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // Virtual (M, K) and (N, K) tensors
    // TODO: why are these virtual?
    Tensor mA = tma_A.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_B.get_tma_tensor(make_shape(N, K));
    Tensor mC = make_tensor(
        make_gmem_ptr(C),
        make_shape(M, N), make_stride(N, _1{})
    );

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);  // (m_blk, n_blk, _)

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BM, BK, k_tiles)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BM, BK, k_tiles)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BM, BN)

    // TODO: what are the below? What is the TMA mode? 
    // tXgX has shape (TMA, k). Leading TMA mode is a single value the TMA instruction consumes.
    // tXsX has shape (TMA, STAGES) iterates over stages to write into.
    auto [tAgA, tAsA] = tma_partition(
        tma_A,
        _0{},  // No multicast (TODO: what is this param?)
        Layout<_1>{},  // No multicast (TODO: what is this param?)
        group_modes<0,2>(sA),
        group_modes<0,2>(gA)
    );
    auto [tBgB, tBsB] = tma_partition(
        tma_B,
        _0{},  // No multicast (TODO: what is this param?)
        Layout<_1>{},  // No multicast (TODO: what is this param?)
        group_modes<0,2>(sB),
        group_modes<0,2>(gB)
    );

    // Initialize mbarriers for TMA loads
    if (tid == 0) {
        CUTE_UNROLL
        for (int s = 0; s < NUM_STAGES; ++s) {
            // "full" arrives when TMA load completes, so expected count is 1
            cutlass::arch::ClusterTransactionBarrier::init(&full_barrier[s], 1);
            // "empty" needs all consumer WGs to release, so we track one arrival per WG
            cutlass::arch::ClusterBarrier::init(&empty_barrier[s], num_consumer_groups);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();  // Ensure all threads see initialized mbarriers

    const int k_tile_count = size<2>(gA);

    // Producer thread
    if (tid >= num_consumer_threads) {
        cutlass::arch::warpgroup_reg_dealloc<regs_per_producer_thread>();  // setmaxnreg.dec 40
        
        const int producer_tid = tid - num_consumer_threads;
        // TODO: do both of the below work? Why?
        // bool is_tma_thread = producer_tid == 0;
        bool is_tma_thread = (producer_tid / 32 == 0) && cute::elect_one_sync();
        if (is_tma_thread) {
            int phase = 0;
            // const int num_k_tiles = (K + BK - 1) / BK;
            CUTE_NO_UNROLL // TODO: do we need this? Why?
            for (int kt = 0; kt < k_tile_count; ++kt) {
                int s = kt % NUM_STAGES;

                if (kt >= NUM_STAGES) {
                    // Empty barriers start at phase 0. The first producer wait
                    // after wrapping must wait for consumers to complete phase 0.
                    cutlass::arch::ClusterBarrier::wait(&empty_barrier[s], phase ^ 1);
                }

                constexpr uint32_t tx_count = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&full_barrier[s], tx_count);

                copy(tma_A.with(full_barrier[s]), tAgA(_, kt), tAsA(_, s));
                copy(tma_B.with(full_barrier[s]), tBgB(_, kt), tBsB(_, s));
                
                if (s == NUM_STAGES - 1) phase ^= 1;
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<regs_per_consumer_thread>();  // setmaxnreg.inc 232
        const int lane_in_wg = tid % 128;

        // TODO: what is happening below? What are each of the shapes and modes? What is each object?
        // TiledMMA partitioning
        auto thr_mma = tiled_mma.get_thread_slice(tid);
        Tensor tCsA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K, STAGES)
        Tensor tCsB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K, STAGES)
        Tensor tCsC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
        
        // There's no actual data in registers. tCrA walks 64-bit matrix descriptors, each pointing to an SMEM
        // region, as the gemm function iterates over MMA_K. Indexing tCrA(_, _, k_block, stage) constructors
        // the descriptor for that particular SMEM sub-tile on the fly. Only one descriptor is live in regs
        // at a time, so this is essentially free.
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // DesciptorIterator over SMEM
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // DesciptorIterator over SMEM
        Tensor tCrC = thr_mma.make_fragment_C(tCsC);  // RMEM accumulator
        clear(tCrC);

        int phase = 0;
        // TODO: what is this release_stage mechanism?
        int release_stage = 0;
        CUTE_NO_UNROLL // TODO: do we need this? Why?
        for (int kt = 0; kt < k_tile_count; ++kt) {
            int s = kt % NUM_STAGES;
            cutlass::arch::ClusterTransactionBarrier::wait(&full_barrier[s], phase);
            if (s == NUM_STAGES - 1) phase ^= 1;
            
            warpgroup_arrive();
            cute::gemm(tiled_mma, tCrA(_, _, _, s), tCrB(_, _, _, s), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<max_wgmma_groups_in_flight>();

            // Release the oldest stage when its WGMMA is proven done
            if (kt >= max_wgmma_groups_in_flight) {
                if (lane_in_wg == 0) {
                    cutlass::arch::ClusterBarrier::arrive(&empty_barrier[release_stage]);
                }
                release_stage = (release_stage + 1) % NUM_STAGES;
            }
        }

        // TODO: what does the below do?
        warpgroup_wait<0>();  // Drain remaining in-flight WGMMA-batches
        const int remaining_stages = 
            k_tile_count < max_wgmma_groups_in_flight ? k_tile_count : max_wgmma_groups_in_flight;
        for (int i = 0; i < remaining_stages; ++i) {
            if (lane_in_wg == 0) {
                cutlass::arch::ClusterBarrier::arrive(&empty_barrier[release_stage]);
            }
            release_stage = (release_stage + 1) % NUM_STAGES;
        }

        // Epilgue
        Tensor cC = make_identity_tensor(make_shape(M, N));
        Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{});
        Tensor tCcC = thr_mma.partition_C(cta_cC);

        CUTE_NO_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            if (elem_less(tCcC(i), make_coord(M, N))) {
                float a = alpha * tCrC(i);
                float out = (beta == 0.0f) ? a : a + beta * float(tCsC(i));
                tCsC(i) = bf16(out);
            }
        }
    }
}

extern "C" void ENTRYPOINT(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {

    // Tile MMA atoms across CTA-level tile via an AtomLayoutMNK
    // Extends the MMA atom (per warpgroup) across all consumer groups
    // get_thread_slice's index range now grows to 128..127+ncg*128
    // The accumulator layout's domain is the full CTA output tile
    TiledMMA tiled_mma = make_tiled_mma(
        MMAAtom{},
        Layout<Shape<Int<num_consumer_groups>, _1, _1>>{}  // AtomLayoutMNK
        // Optional PermutationMNK when we want to redistributed resulting TV layotus after tiling
    );

    // Make TMA copy
    Tensor mA = make_tensor(
        make_gmem_ptr(reinterpret_cast<bf16 const*>(A)),
        make_shape(M, K),
        make_stride(K, _1{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(reinterpret_cast<bf16 const*>(BT)),
        make_shape(N, K),
        make_stride(K, _1{})
    );
    SmemLayoutA sA_layout;
    SmemLayoutB sB_layout;

    // Get the rank-2 (BM, BK) -> offset layout for one stage of SMEM
    // Since the TMA descriptor knows how to copy one (BM, BK) tile from GMEM -> SMEM
    auto sA_one_stage = sA_layout(_, _, Int<0>{});
    auto sB_one_stage = sB_layout(_, _, Int<0>{});

    auto tma_A = make_tma_copy(SM90_TMA_LOAD{}, mA, sA_one_stage);
    auto tma_B = make_tma_copy(SM90_TMA_LOAD{}, mB, sB_one_stage);

    static bool attr_configured = false;
    if (!attr_configured) {
        cudaError_t err = cudaFuncSetAttribute(
            (const void*)KERNEL_NAME<decltype(tma_A), decltype(tma_B), SmemLayoutA, SmemLayoutB, decltype(tiled_mma)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            size_smem
        );
        assert(err == cudaSuccess);
        attr_configured = true;
    }

    dim3 block(num_threads);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    KERNEL_NAME<decltype(tma_A), decltype(tma_B), SmemLayoutA, SmemLayoutB, decltype(tiled_mma)><<<grid, block, size_smem>>>(
        M, N, K, alpha, beta,
        tma_A, tma_B,
        reinterpret_cast<bf16*>(C),
        sA_layout, sB_layout, tiled_mma
    );
}
