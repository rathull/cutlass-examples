// Tensor-core BF16 GEMM for Hopper (sm_90) using mma.m16n8k16 + cp.async + ldmatrix.
//
// Layout: NT mode (A row-major MxK, BT row-major NxK; both K-contiguous).
// Tile:   CTA = BM x BN (default 128x128), inner BK (default 64).
// Warps:  4 warps (128 threads), 2x2 over the CTA tile (each warp owns 64x64).
// Math:   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32, FP32 accumulate.
// Loads:  cp.async.cg.shared.global, 16B vectors (8 bf16), NUM_STAGES-deep pipeline.
// Frags:  ldmatrix.x4 for A (16x16), ldmatrix.x2.trans for B (16x8), per K=16 step.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common.cuh"

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NUM_STAGES
#define NUM_STAGES 3
#endif

#ifndef ENTRYPOINT
#define ENTRYPOINT cuda_v2_mma_async
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME cuda_v2_mma_async_kernel
#endif

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 16;

constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int WARPS_M = BM / WARP_M;
constexpr int WARPS_N = BN / WARP_N;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_TILES = WARP_M / WMMA_M;
constexpr int N_TILES = WARP_N / WMMA_N;
constexpr int K_TILES = BK / WMMA_K;

constexpr int VEC_BF16 = 8;  // 16-byte cp.async = 8 bf16

static_assert(BM % WARP_M == 0 && BN % WARP_N == 0, "Warp tile must divide CTA tile.");
static_assert(BK % WMMA_K == 0, "BK must be a multiple of 16.");
static_assert(BK % VEC_BF16 == 0, "BK must be a multiple of 8.");
static_assert(NUM_STAGES >= 2, "Need at least 2 pipeline stages.");
static_assert(NUM_THREADS == 128, "v2 hard-codes 4 warps / 128 threads.");

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float* d,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

}  // namespace

extern "C" __launch_bounds__(NUM_THREADS, 2)
__global__ void KERNEL_NAME(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    A  = (const nv_bfloat16*)__builtin_assume_aligned(A,  16);
    BT = (const nv_bfloat16*)__builtin_assume_aligned(BT, 16);
    C  = (nv_bfloat16*)__builtin_assume_aligned(C,  16);

    const int cta_m = blockIdx.y * BM;
    const int cta_n = blockIdx.x * BN;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int warp_m = (warp / WARPS_N) * WARP_M;
    const int warp_n = (warp % WARPS_N) * WARP_N;

    __align__(16) __shared__ nv_bfloat16 sA [NUM_STAGES][BM][BK];
    __align__(16) __shared__ nv_bfloat16 sBT[NUM_STAGES][BN][BK];

    float acc[M_TILES][N_TILES][4] = {};

    constexpr int VECS_PER_A_ROW  = BK / VEC_BF16;
    constexpr int VECS_PER_BT_ROW = BK / VEC_BF16;
    constexpr int A_TILE_VECS  = (BM * BK) / VEC_BF16;
    constexpr int BT_TILE_VECS = (BN * BK) / VEC_BF16;

    auto issue_load = [&](int stage, int k_tile) {
        // A[cta_m..cta_m+BM][k_tile..k_tile+BK] -> sA[stage]
        #pragma unroll
        for (int i = tid; i < A_TILE_VECS; i += NUM_THREADS) {
            const int row = i / VECS_PER_A_ROW;
            const int col = (i % VECS_PER_A_ROW) * VEC_BF16;
            cp_async_16(
                to_smem(&sA[stage][row][col]),
                &A[(cta_m + row) * K + k_tile + col]
            );
        }
        // BT[cta_n..cta_n+BN][k_tile..k_tile+BK] -> sBT[stage]
        #pragma unroll
        for (int i = tid; i < BT_TILE_VECS; i += NUM_THREADS) {
            const int row = i / VECS_PER_BT_ROW;
            const int col = (i % VECS_PER_BT_ROW) * VEC_BF16;
            cp_async_16(
                to_smem(&sBT[stage][row][col]),
                &BT[(cta_n + row) * K + k_tile + col]
            );
        }
        cp_async_commit();
    };

    const int num_k_tiles = K / BK;

    // Prefetch the first NUM_STAGES-1 stages.
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            issue_load(s, s * BK);
        } else {
            cp_async_commit();  // empty group, keeps wait counts consistent
        }
    }

    // Main loop.
    for (int outer = 0; outer < num_k_tiles; ++outer) {
        cp_async_wait_group<NUM_STAGES - 2>();
        __syncthreads();

        const int read_stage = outer % NUM_STAGES;

        #pragma unroll
        for (int kt = 0; kt < K_TILES; ++kt) {
            // ---- Fragment loads ----
            // A frag (16x16) per warp_m_block, packed into 4 .b32 regs per thread via ldmatrix.x4.
            // ldmatrix.x4 uses all 32 lanes; lanes [0..7]:[8..15]:[16..23]:[24..31] supply the 8
            // smem-row pointers for tiles (0,0),(0,1),(1,0),(1,1) of the 16x16 region.
            uint32_t a_frag[M_TILES][4];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int row = warp_m + mt * WMMA_M + ((lane / 16) * 8) + (lane % 8);
                const int col = kt * WMMA_K + ((lane / 8) % 2) * 8;
                ldmatrix_x4(a_frag[mt], to_smem(&sA[read_stage][row][col]));
            }

            // B frag (K=16, N=8) per warp_n_block, .trans because smem is N-major (BT[N][K]).
            uint32_t b_frag[N_TILES][2];
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int row = warp_n + nt * WMMA_N + (lane % 8);
                const int col = kt * WMMA_K + ((lane / 8) % 2) * 8;
                ldmatrix_x2_trans(b_frag[nt], to_smem(&sBT[read_stage][row][col]));
            }

            // ---- mma issues ----
            // ldmatrix.x4 returns regs in the order (M=0..7,K=0..7), (M=0..7,K=8..15),
            // (M=8..15,K=0..7), (M=8..15,K=8..15). mma.m16n8k16 expects A in the order
            // (M=0..7,K=0..7), (M=8..15,K=0..7), (M=0..7,K=8..15), (M=8..15,K=8..15) —
            // hence the 0,2,1,3 swap below.
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                #pragma unroll
                for (int nt = 0; nt < N_TILES; ++nt) {
                    mma_m16n8k16(
                        acc[mt][nt],
                        a_frag[mt][0], a_frag[mt][2], a_frag[mt][1], a_frag[mt][3],
                        b_frag[nt][0], b_frag[nt][1]
                    );
                }
            }
        }

        // Issue the load that becomes available NUM_STAGES-1 iterations ahead.
        const int prefetch_iter = outer + (NUM_STAGES - 1);
        if (prefetch_iter < num_k_tiles) {
            issue_load(prefetch_iter % NUM_STAGES, prefetch_iter * BK);
        } else {
            cp_async_commit();
        }
    }

    cp_async_wait_group<0>();
    __syncthreads();

    // ---- Epilogue: write fp32 accumulator -> bf16 C ----
    // mma.m16n8k16 D-fragment per thread (16x8 fp32 tile):
    //   row_a = (lane / 4),  row_b = row_a + 8
    //   col_0 = (lane % 4) * 2, col_1 = col_0 + 1
    //   d[0]=C[row_a][col_0] d[1]=C[row_a][col_1] d[2]=C[row_b][col_0] d[3]=C[row_b][col_1]
    const int row_a_base = cta_m + warp_m + (lane / 4);
    const int col_base   = cta_n + warp_n + (lane % 4) * 2;

    if (beta == 0.0f) {
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int row_a = row_a_base + mt * WMMA_M;
            const int row_b = row_a + 8;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int col0 = col_base + nt * WMMA_N;
                const float* d = acc[mt][nt];
                if (row_a < M) {
                    const __nv_bfloat162 packed = __floats2bfloat162_rn(alpha * d[0], alpha * d[1]);
                    if (col0 + 1 < N) {
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_a * N + col0]) = packed;
                    } else if (col0 < N) {
                        C[row_a * N + col0] = __low2bfloat16(packed);
                    }
                }
                if (row_b < M) {
                    const __nv_bfloat162 packed = __floats2bfloat162_rn(alpha * d[2], alpha * d[3]);
                    if (col0 + 1 < N) {
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_b * N + col0]) = packed;
                    } else if (col0 < N) {
                        C[row_b * N + col0] = __low2bfloat16(packed);
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int row_a = row_a_base + mt * WMMA_M;
            const int row_b = row_a + 8;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int col0 = col_base + nt * WMMA_N;
                const float* d = acc[mt][nt];
                #define STORE_RB(R, IDX0, IDX1)                                                  \
                    if ((R) < M) {                                                                \
                        if (col0 + 1 < N) {                                                       \
                            const float p0 = beta * __bfloat162float(C[(R) * N + col0]);          \
                            const float p1 = beta * __bfloat162float(C[(R) * N + col0 + 1]);      \
                            *reinterpret_cast<__nv_bfloat162*>(&C[(R) * N + col0]) =              \
                                __floats2bfloat162_rn(alpha * d[IDX0] + p0, alpha * d[IDX1] + p1);\
                        } else if (col0 < N) {                                                    \
                            const float p0 = beta * __bfloat162float(C[(R) * N + col0]);          \
                            C[(R) * N + col0] = __float2bfloat16(alpha * d[IDX0] + p0);           \
                        }                                                                         \
                    }
                STORE_RB(row_a, 0, 1)
                STORE_RB(row_b, 2, 3)
                #undef STORE_RB
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
    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    KERNEL_NAME<<<grid, block>>>(A, BT, C, M, N, K, alpha, beta);
}
