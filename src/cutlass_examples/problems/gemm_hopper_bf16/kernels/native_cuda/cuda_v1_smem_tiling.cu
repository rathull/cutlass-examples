#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

#ifndef TM
#define TM 8
#endif

#ifndef TN
#define TN 8
#endif

#ifndef ENTRYPOINT
#define ENTRYPOINT cuda_v1_smem_tiling
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME cuda_v1_smem_tiling_kernel
#endif

// CTA sizes
constexpr int CTA_M = (BM + TM - 1) / TM;
constexpr int CTA_N = (BN + TN - 1) / TN;
constexpr int NUM_THREADS = CTA_M * CTA_N;

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");
static_assert(NUM_THREADS <= 1024, "CTA cannot have more than 1024 threads");
constexpr int A_VEC_ELEMS = gemm_hopper_bf16::native_cuda::max_bf16_vec_elems(BK);
constexpr int BT_VEC_ELEMS = gemm_hopper_bf16::native_cuda::max_bf16_vec_elems(BK);

__launch_bounds__(NUM_THREADS) // TODO: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__global__ void KERNEL_NAME(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // This thread block computes an TM*TN tile of the output matrix C

    // 16 byte alignment allows for vectorized 128-bit load/stores
    A = (const nv_bfloat16*)__builtin_assume_aligned(A, 16);
    BT = (const nv_bfloat16*)__builtin_assume_aligned(BT, 16);
    C = (nv_bfloat16*)__builtin_assume_aligned(C, 16);

    const int cta_m_start = blockIdx.y * BM;  // row of start of C tile
    const int cta_n_start = blockIdx.x * BN;  // col of start of C tile
    const int thread_m_start = threadIdx.y * TM;
    const int thread_n_start = threadIdx.x * TN;
    const int tid = threadIdx.y * CTA_N + threadIdx.x;

    // Each thread block will compute a tile of the output matrix C
    // TODO: what is the canonical way to name these? Like tRsA or just sA or what is that notation for?
    // TODO: do I need to align these to some byte boundary? Would I need to for vectorized loads or TMA or WGMMA?
    __align__(16) __shared__ nv_bfloat16 sA[BM * BK];
    __align__(16) __shared__ nv_bfloat16 sBT[BN * BK];

    // TODO: move this somewhere else since these threads may still have to perform HBM load
    // if (row >= M || col >= N) {
    //     return;
    // }

    // Initialize output register tile
    float out_reg_tile[TM * TN] = {0.0f};

    // TODO: where is everywhere I need to guard to ensure we don't have OOB access at boundary tiles?
    for (int k_start = 0; k_start < K; k_start += BK) {
        // Load A[cta_m_start:+BM, k_start:+BK] into row-major sA[BM, BK].
        gemm_hopper_bf16::native_cuda::load_row_major_tile<BM, BK, A_VEC_ELEMS, NUM_THREADS>(
            sA,
            A,
            cta_m_start,
            k_start,
            K,
            M - cta_m_start < BM ? M - cta_m_start : BM,
            K - k_start < BK ? K - k_start : BK,
            tid
        );

        // Load BT[cta_n_start:+BN, k_start:+BK] into row-major sBT[BN, BK].
        gemm_hopper_bf16::native_cuda::load_row_major_tile<BN, BK, BT_VEC_ELEMS, NUM_THREADS>(
            sBT,
            BT,
            cta_n_start,
            k_start,
            K,
            N - cta_n_start < BN ? N - cta_n_start : BN,
            K - k_start < BK ? K - k_start : BK,
            tid
        );

        // Synchronize so all threads have produced tiles
        __syncthreads();

        // Compute output tile C[cta_m_start + m_in_sA_tile][cta_n_start + n_in_sB_tile].
        // Keep K outermost so the per-k operands can feed the whole per-thread TMxTN tile.
        #if BK <= 16
            #pragma unroll
        #else
            #pragma unroll 4
        #endif
        for (int k = 0; k < BK; ++k) {
            float b_frag[TN];

            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                b_frag[c] = __bfloat162float(sBT[(thread_n_start + c) * BK + k]);
            }

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                const float a = __bfloat162float(sA[(thread_m_start + r) * BK + k]);

                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    out_reg_tile[r * TN + c] += a * b_frag[c];
                }
            }
        }

        // Synchronize so all threads have consumed tile
        __syncthreads();
    }

    if (beta == 0.0f) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                const int row = cta_m_start + thread_m_start + r;
                const int col = cta_n_start + thread_n_start + c;
                if (row < M && col < N) {
                    C[row * N + col] =
                        __float2bfloat16(alpha * out_reg_tile[r * TN + c]);
                }
            }
        }
    } else {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                const int row = cta_m_start + thread_m_start + r;
                const int col = cta_n_start + thread_n_start + c;
                if (row < M && col < N) {
                    C[row * N + col] =
                        __float2bfloat16(alpha * out_reg_tile[r * TN + c] +
                            beta * __bfloat162float(C[row * N + col]));
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
    dim3 block(CTA_N, CTA_M);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    KERNEL_NAME<<<grid, block>>>(
        A, BT, C, M, N, K, alpha, beta
    );
}
