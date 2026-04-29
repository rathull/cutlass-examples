#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

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

__launch_bounds__(num_consumer_threads + num_producer_threads) // TODO: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
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
    const int is_wgmma_thread = tid < num_consumer_threads;
    const int is_tma_thread = tid >= num_consumer_threads;

    // SMEM tiles
    __align__(16) __shared__ nv_bfloat16 sA[NUM_STAGES][BM * BK];
    __align__(16) __shared__ nv_bfloat16 sB[NUM_STAGES][BK * BN];

    // Initialize thread-partitioned accumulator registers for WGMMA
    constexpr int num_accum_regs_per_thread = (wgmma_m * wgmma_n) / 128;
    float tCrC[num_accum_regs_per_thread] = {0.0f};

    // Initialize mbarriers for TMA loads
    uint64_t empty_barrier[num_stages], full_barrier[num_stages];
    // TODO: initialize mbarriers

    // Prologue: issue first num_stages TMAs to load A[] and B[] into SMEM

    
    for (int k_start = 0; k_start < K; k_start += BK) {
        // TMA load A[]
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
    CUtensorMap tma_A_desc, tma_B_desc;
    CUresult err;
    // Hopper's swizzle pattern is on 128-byte chunks, so 64 elements
    // We configure TMA descriptor to swizzle the data, specify that B is transposed
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
    err = cuTensorMapEncodeTiled(
        &tma_A_desc,                        // tensorMap 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // tensorDataType
        2,                                  // tensorRank
        A,                                  // globalAddress
        globalDim_A,                        // globalDim, fastest changing dimension first
        globalStrides_A,                    // globalStrides, innermost stride is implicitly sizeof(element)
        boxDim_A,                           // boxDim, fastest changing dimension first
                                            // TODO: do we need BK, BM <= 256? Does this change max num_consumer_groups?
        elementStrides_A,                   // elementStrides
        CU_TENSOR_MAP_INTERLEAVE_NONE,      // interleave
        CU_TENSOR_MAP_SWIZZLE_128B,         // swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, // l2Promotion TODO: hyperparameter, I think this is right
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,  // oobFill
    );
    assert(err == CUDA_SUCCESS);
    cuTensorMapEncodeTiled(
        &tma_B_desc,                        // tensorMap 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // tensorDataType
        2,                                  // tensorRank
        BT,                                 // globalAddress
        globalDim_B,                        // globalDim, fastest changing dimension first
        globalStrides_B,                    // globalStrides, innermost stride is implicitly sizeof(element)
        boxDim_B,                           // boxDim, fastest changing dimension first
                                            // TODO: do we need BK, BM <= 256? Does this change max num_consumer_groups?
        elementStrides_B,                   // elementStrides
        CU_TENSOR_MAP_INTERLEAVE_NONE,      // interleave
        CU_TENSOR_MAP_SWIZZLE_128B,         // swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, // l2Promotion TODO: hyperparameter, I think this is right
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,  // oobFill
    );
    assert(err == CUDA_SUCCESS);
    dim3 block(num_consumer_threads + num_producer_threads);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    KERNEL_NAME<<<grid, block>>>(
        A, BT, C, M, N, K, alpha, beta,
        tma_A_desc, tma_B_desc
    );
}
