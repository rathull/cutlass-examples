#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cassert>
#include <cstdint>

#include "common.cuh"

#ifndef BM
#define BM 128
#endif

#ifndef BN
#define BN 64
#endif

#ifndef BK
#define BK 64
#endif

#ifndef NUM_STAGES
#define NUM_STAGES 4
#endif

#ifndef ENTRYPOINT
#define ENTRYPOINT cuda_v2_tma_wgmma
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME cuda_v2_tma_wgmma_kernel
#endif

constexpr int num_consumer_groups = (BM + 63) / 64;
constexpr int num_consumer_threads = 128 * num_consumer_groups;
constexpr int num_producer_threads = 128;
constexpr int num_threads = num_consumer_threads + num_producer_threads;
constexpr int wgmma_m = 64;
constexpr int wgmma_n = BN;
constexpr int wgmma_k = 16;

template <int BM_T, int BN_T, int BK_T, int NUM_STAGES_T>
struct alignas(1024) SharedStorage {
    alignas(1024) nv_bfloat16 sA[NUM_STAGES_T][BM_T * BK_T];
    alignas(1024) nv_bfloat16 sBT[NUM_STAGES_T][BN_T * BK_T];
    alignas(8) uint64_t full_barrier[NUM_STAGES_T];
    alignas(8) uint64_t empty_barrier[NUM_STAGES_T];
};
using SmemT = SharedStorage<BM, BN, BK, NUM_STAGES>;
constexpr int smem_size = sizeof(SmemT);

constexpr int regs_per_consumer_thread = 232;
constexpr int regs_per_producer_thread = 40;

// TODO: need to figure out how to handle case where dimensions are not divisble by 64
static_assert(BM % 64 == 0, "BM must be divisible by 64");
static_assert(BN == 64,
              "Only the WGMMA wrapper/epilogue are fixed to n64; the SMEM descriptor is BN-agnostic.");
static_assert(BK == 64, "CU_TENSOR_MAP_SWIZZLE_128B requires the BF16 inner box to be <= 128 bytes.");
static_assert(BM <= 256 && BN <= 256 && BK <= 256, "TMA tiled box dimensions must be <= 256.");
static_assert((BK * static_cast<int>(sizeof(nv_bfloat16))) == 128,
              "WGMMA K-major 128B swizzle expects a 128-byte contiguous K row.");
static_assert(num_threads <= 1024, "CTA cannot have more than 1024 threads");
static_assert(NUM_STAGES >= 2, "Need at least 2 pipeline stages.");
static_assert(NUM_STAGES <= 8, "wgmma.wait_group supports immediates in [0, 7].");
static_assert(smem_size <= 227 * 1024, "SMEM per CTA is too large.");
static_assert(BM * BN / 4 <= 256 * 1024, "Accumulator registers per CTA is too large.");

// TODO: CUTLASS uses a separate "consumer phase" and "producer phase" that each track their own barrier. Implement this.
struct PipelineState {
    int stage = 0;  // stage slot
    int phase = 0;  // mbarrier parity
    inline __device__ void advance() {
        ++stage;
        if (stage == NUM_STAGES) {
            stage = 0;
            phase ^= 1;
        }
    }
};

__launch_bounds__(num_threads)
__global__ void KERNEL_NAME(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const __grid_constant__ CUtensorMap tma_A_desc,
    const __grid_constant__ CUtensorMap tma_B_desc
) {
    A = (const nv_bfloat16*)__builtin_assume_aligned(A, 16);
    BT = (const nv_bfloat16*)__builtin_assume_aligned(BT, 16);
    C = (nv_bfloat16*)__builtin_assume_aligned(C, 16);

    // This CTA computes C[block_offset_m:+BM][block_offset_n:+BN]
    const int block_offset_m = blockIdx.y * BM;
    const int block_offset_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const bool is_wgmma_thread = tid < num_consumer_threads;
    const bool is_tma_thread = tid >= num_consumer_threads;

    // 128B TMA swizzle repeats every 1024 bytes; aligning the stage bases keeps
    // WGMMA's descriptor base-offset at zero for the default tile shape.
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    auto& smem = *reinterpret_cast<SharedStorage<BM, BN, BK, NUM_STAGES>*>(smem_raw);
    auto* sA = smem.sA;
    auto* sBT = smem.sBT;
    auto* full_barrier = smem.full_barrier;
    auto* empty_barrier = smem.empty_barrier;

    // Initialize mbarriers for TMA loads
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; ++s) {
            // "full" arrives when TMA load completes, so expected count is 1
            mbarrier_init(&full_barrier[s], 1);
            // "empty" needs all consumer WGs to release, so we track one arrival per WG
            mbarrier_init(&empty_barrier[s], num_consumer_groups);
        }
        fence_barrier_init();
    }
    __syncthreads();  // Ensure all threads see initialized mbarriers

    if (is_tma_thread) {
        warpgroup_reg_deallocate<regs_per_producer_thread>();
        const int producer_tid = tid - num_consumer_threads;
        if (producer_tid == 0) {
            PipelineState state;
            const int num_k_tiles = (K + BK - 1) / BK;
            for (int kt = 0; kt < num_k_tiles; ++kt) {
                if (kt >= NUM_STAGES) {
                    // Empty barriers start at phase 0. The first producer wait
                    // after wrapping must wait for consumers to complete phase 0.
                    mbarrier_wait(&empty_barrier[state.stage], state.phase ^ 1);
                }

                // At this point, consumer WG has signaled that it is done with the current stage
                const int k_start = kt * BK;

                // If requested tile extends past matrix bounds, TMA can avoid writing OOB elements but we need to
                // zero the stale data in SMEM. This is currently not the fastest implementation as we do not
                // recruit every preoducer thread, but 
                const bool partial_a_tile = block_offset_m + BM > M || k_start + BK > K;
                const bool partial_b_tile = block_offset_n + BN > N || k_start + BK > K;
                if (partial_a_tile) {
                    for (int i = 0; i < BM * BK; ++i) {
                        sA[state.stage][i] = __float2bfloat16(0.0f);
                    }
                }
                if (partial_b_tile) {
                    for (int i = 0; i < BN * BK; ++i) {
                        sBT[state.stage][i] = __float2bfloat16(0.0f);
                    }
                }
                if (partial_a_tile || partial_b_tile) {
                    fence_proxy_async_shared_cta();
                }

                uint32_t tx_count = sizeof(nv_bfloat16) * (BM * BK + BN * BK);
                mbarrier_arrive_expect_tx(&full_barrier[state.stage], tx_count);

                // Issue TMA loads and signal full_barrier once load is complete
                tma_load_2d(&tma_A_desc, &full_barrier[state.stage], sA[state.stage], k_start, block_offset_m);
                tma_load_2d(&tma_B_desc, &full_barrier[state.stage], sBT[state.stage], k_start, block_offset_n);
                
                state.advance();
            }
        }
    } else if (is_wgmma_thread) {
        warpgroup_reg_allocate<regs_per_consumer_thread>();

        const int wg_idx = tid / 128;
        const int lane_in_wg = tid % 128;
        const int warp_in_wg = lane_in_wg / 32;
        const int lane_in_warp = lane_in_wg % 32;
        const int frag_row_a = warp_in_wg * 16 + lane_in_warp / 4;
        const int frag_col_base = (lane_in_warp % 4) * 2;

        PipelineState state;
        constexpr int num_accum_regs_per_thread = wgmma_n / 2;
        constexpr int max_wgmma_groups_in_flight = NUM_STAGES - 1;
        float tCrC[num_accum_regs_per_thread] = {0.0f};

        int release_stage = 0;
        const int num_k_tiles = (K + BK - 1) / BK;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            // Wait for this stage's data to arrive
            mbarrier_wait(&full_barrier[state.stage], state.phase);

            // Build descriptors: each WG has same sB but different 64-row sA tile
            nv_bfloat16* sA_stage = &sA[state.stage][0];
            nv_bfloat16* sA_stage_wg = sA_stage + wg_idx * wgmma_m * BK;
            nv_bfloat16* sBT_stage = &sBT[state.stage][0];

            warpgroup_arrive();  // wgmma.fence.sync.aligned

            // Issue BK/16 WGMMAs along K within this stage
            #pragma unroll
            for (int k_in_stage = 0; k_in_stage < BK; k_in_stage += wgmma_k) {
                WgmmaOperandDescriptors desc = make_wgmma_bf16_k_major_operand_descs<BK>(
                    sA_stage_wg + k_in_stage,
                    sA_stage,
                    sBT_stage + k_in_stage,
                    sBT_stage
                );

                wgmma_m64n64k16_bf16<true>(
                    desc.A, desc.B,
                    tCrC
                );
            }
            warpgroup_commit_batch();  // wgmma.commit_group.sync.aligned

            // Keep up to NUM_STAGES-1 committed WGMMA groups in flight. Only
            // release the oldest stage after wait_group proves that group done.
            warpgroup_wait<max_wgmma_groups_in_flight>();
            if (kt >= max_wgmma_groups_in_flight) {
                if (lane_in_wg == 0) {
                    mbarrier_arrive(&empty_barrier[release_stage]);
                }
                release_stage = (release_stage + 1) % NUM_STAGES;
            }

            state.advance();
        }

        warpgroup_wait<0>(); // Drain remaining in-flight WGMMA-batches

        const int remaining_stages =
            num_k_tiles < max_wgmma_groups_in_flight ? num_k_tiles : max_wgmma_groups_in_flight;
        for (int i = 0; i < remaining_stages; ++i) {
            if (lane_in_wg == 0) {
                mbarrier_arrive(&empty_barrier[release_stage]);
            }
            release_stage = (release_stage + 1) % NUM_STAGES;
        }

        // CUTLASS SM90 WGMMA CLayout_64xN maps each thread's f32 registers as:
        // d[4k+0/1] -> row_a, cols 8k+2c/8k+2c+1; d[4k+2/3] -> row_a+8.
        constexpr int col_pair_groups = BN / 8;
        const int wg_m_offset = wg_idx * wgmma_m;
        const int row_a = block_offset_m + wg_m_offset + frag_row_a;
        const int row_b = row_a + 8;

        #pragma unroll
        for (int group = 0; group < col_pair_groups; ++group) {
            const int col0 = block_offset_n + 8 * group + frag_col_base;

            if (row_a < M) {
                if (col0 < N) {
                    float value = alpha * tCrC[4 * group + 0];
                    if (beta != 0.0f) {
                        value += beta * __bfloat162float(C[row_a * N + col0]);
                    }
                    C[row_a * N + col0] = __float2bfloat16(value);
                }
                if (col0 + 1 < N) {
                    float value = alpha * tCrC[4 * group + 1];
                    if (beta != 0.0f) {
                        value += beta * __bfloat162float(C[row_a * N + col0 + 1]);
                    }
                    C[row_a * N + col0 + 1] = __float2bfloat16(value);
                }
            }

            if (row_b < M) {
                if (col0 < N) {
                    float value = alpha * tCrC[4 * group + 2];
                    if (beta != 0.0f) {
                        value += beta * __bfloat162float(C[row_b * N + col0]);
                    }
                    C[row_b * N + col0] = __float2bfloat16(value);
                }
                if (col0 + 1 < N) {
                    float value = alpha * tCrC[4 * group + 3];
                    if (beta != 0.0f) {
                        value += beta * __bfloat162float(C[row_b * N + col0 + 1]);
                    }
                    C[row_b * N + col0 + 1] = __float2bfloat16(value);
                }
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
    // TODO: move TMA descriptor encoding to utils
    alignas(64) CUtensorMap tma_A_desc;
    alignas(64) CUtensorMap tma_B_desc;
    CUresult res;
    assert(M > 0 && N > 0 && K > 0);
    assert(reinterpret_cast<uintptr_t>(A) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(BT) % 16 == 0);
    assert((K * static_cast<int>(sizeof(nv_bfloat16))) % 16 == 0);
    // Hopper's 128B swizzle operates on 16B chunks within a 128B row. The
    // WGMMA descriptors below use PTX's matching K-major 128B swizzle encoding.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    // TODO: do I need to ensure that this returns CUDA_SUCCESS? Will that create overhead?
    cuuint64_t globalDim_A[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t globalStrides_A[1] = {(cuuint64_t)(K * sizeof(nv_bfloat16))};
    cuuint32_t boxDim_A[2] = {BK, BM};
    cuuint32_t elementStrides_A[2] = {1, 1};
    cuuint64_t globalDim_B[2] = {(cuuint64_t)K, (cuuint64_t)N};
    cuuint64_t globalStrides_B[1] = {(cuuint64_t)(K * sizeof(nv_bfloat16))};
    cuuint32_t boxDim_B[2] = {BK, BN};
    cuuint32_t elementStrides_B[2] = {1, 1};
    res = cuTensorMapEncodeTiled(
        &tma_A_desc,                        // tensorMap 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // tensorDataType
        2,                                  // tensorRank
        (void*)A,                           // globalAddress
        globalDim_A,                        // globalDim, fastest changing dimension first
        globalStrides_A,                    // globalStrides, innermost stride is implicitly sizeof(element)
        boxDim_A,                           // boxDim, fastest changing dimension first
                                            // TODO: do we need BK, BM <= 256? Does this change max num_consumer_groups?
        elementStrides_A,                   // elementStrides
        CU_TENSOR_MAP_INTERLEAVE_NONE,      // interleave
        CU_TENSOR_MAP_SWIZZLE_128B,         // swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, // l2Promotion TODO: hyperparameter, I think this is right
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // oobFill
    );
    assert(res == CUDA_SUCCESS);
    res = cuTensorMapEncodeTiled(
        &tma_B_desc,                        // tensorMap 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // tensorDataType
        2,                                  // tensorRank
        (void*) BT,                         // globalAddress
        globalDim_B,                        // globalDim, fastest changing dimension first
        globalStrides_B,                    // globalStrides, innermost stride is implicitly sizeof(element)
        boxDim_B,                           // boxDim, fastest changing dimension first
                                            // TODO: do we need BK, BM <= 256? Does this change max num_consumer_groups?
        elementStrides_B,                   // elementStrides
        CU_TENSOR_MAP_INTERLEAVE_NONE,      // interleave
        CU_TENSOR_MAP_SWIZZLE_128B,         // swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, // l2Promotion TODO: hyperparameter, I think this is right
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // oobFill
    );
    assert(res == CUDA_SUCCESS);
    static bool attr_configured = false;
    if (!attr_configured) {
        cudaError_t err = cudaFuncSetAttribute(
            (const void*)KERNEL_NAME,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        assert(err == cudaSuccess);
        attr_configured = true;
    }
    dim3 block(num_threads);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    KERNEL_NAME<<<grid, block, smem_size>>>(
        A, BT, C, M, N, K, alpha, beta,
        tma_A_desc, tma_B_desc
    );
}
