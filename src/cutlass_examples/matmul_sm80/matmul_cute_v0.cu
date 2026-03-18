#include <cuda_bf16.h>

#include <cute/tensor.hpp>

// CuTe tutorial GEMM kernel (sgemm_1 style).
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout>
__global__ static
__launch_bounds__(decltype(cute::size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);
    static_assert(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
    CUTE_STATIC_ASSERT_V(size(tC) == size(tA));

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

    // Full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Per-CTA tiles
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    // Partition gmem->smem copies across threads
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

    // Partition smem for compute (via tC projections)
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

    // fp32 accumulators for numerical precision with bf16 inputs
    Tensor tCrC = make_tensor<float>(shape(tCgC));                       // (THR_M,THR_N)
    clear(tCrC);

    // Main loop: load tiles into smem, compute on them
    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        copy(tAgA(_,_,k_tile), tAsA);
        copy(tBgB(_,_,k_tile), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Manual FMA loop with explicit bf16->fp32 casts.
        // PyTorch defines __CUDA_NO_BFLOAT16_CONVERSIONS__ which disables
        // implicit bf16<->float conversion, so CuTe's gemm() can't be used
        // directly with mixed accumulator types.
        CUTE_UNROLL
        for (int k = 0; k < size<1>(tCsA); ++k) {
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tCrC); ++m) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tCrC); ++n) {
                    tCrC(m,n) += __bfloat162float(tCsA(m,k)) * __bfloat162float(tCsB(n,k));
                }
            }
        }

        __syncthreads();
    }

    // Epilogue: convert fp32 accumulators back to bf16
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = __float2bfloat16(tCrC(i));
    }
}

// Wrapper matching MatmulFn: void(const bf16*, const bf16*, bf16*, int, int, int)
// A is MxK row-major, B is NxK row-major, C is MxN row-major.
void matmul_cute_v0(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    using namespace cute;

    auto prob_shape = make_shape(M, N, K);

    // Row-major strides for all three matrices
    auto dA = make_stride(K, Int<1>{});                                   // (dM, dK)
    auto dB = make_stride(K, Int<1>{});                                   // (dN, dK)
    auto dC = make_stride(N, Int<1>{});                                   // (dM, dN)

    // CTA tile sizes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // k-major smem layouts (LayoutRight) to match row-major global memory
    auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
    auto sB = make_layout(make_shape(bN, bK), LayoutRight{});
    auto sC = make_layout(make_shape(bM, bN));

    // k-major thread layouts for coalesced global memory access
    auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});
    auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock, 0, 0>>>(
        prob_shape, cta_tiler,
        A, dA, sA, tA,
        B, dB, sB, tB,
        C, dC, sC, tC);
}
