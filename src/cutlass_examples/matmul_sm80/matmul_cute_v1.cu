// CuTe GEMM with SM80 tensor core MMA (m16n8k16) and cp.async global→shared loads.
// Single-buffered (no software pipelining), no shared memory swizzling.
// A: M×K row-major, B: N×K row-major (≡ K×N col-major), C: M×N row-major.

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

template <int BLK_M, int BLK_N, int BLK_K>
__global__ __launch_bounds__(128)
void gemm_cute_v1_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
          nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    auto cta_tiler = make_shape(Int<BLK_M>{}, Int<BLK_N>{}, Int<BLK_K>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N, BLK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M, BLK_N)

    // Shared memory — K-major (LayoutRight) to match row-major global memory
    auto sA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{});
    auto sB_layout = make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}), LayoutRight{});

    __shared__ nv_bfloat16 smemA[cosize_v<decltype(sA_layout)>];
    __shared__ nv_bfloat16 smemB[cosize_v<decltype(sB_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // -- TiledMMA: 4 warps, SM80 bf16 tensor cores (mma.sync.m16n8k16.row.col) --
    // Atom covers 16×8×16, tiled 2×2×1 → 32×16 per MMA step, 128 threads
    TiledMMA tiled_mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_2, _2, _1>>{}
    );

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);          // (MMA, MMA_M, MMA_K)
    auto tCsB = thr_mma.partition_B(sB);          // (MMA, MMA_N, MMA_K)
    auto tCgC = thr_mma.partition_C(gC);          // (MMA, MMA_M, MMA_N)
    auto tCrC = thr_mma.make_fragment_C(tCgC);    // fp32 register accumulators
    clear(tCrC);

    // -- TiledCopy: cp.async 128-bit (8 bf16) loads from global → shared --
    // 32×4 thread layout × 1×8 val layout → each step copies a 32×32 tile
    auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, nv_bfloat16>{};

    auto tiled_copy_a = make_tiled_copy(copy_atom,
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape< _1, _8>>{});
    auto tiled_copy_b = make_tiled_copy(copy_atom,
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},
        Layout<Shape< _1, _8>>{});

    auto thr_copy_a = tiled_copy_a.get_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)
    auto tAsA = thr_copy_a.partition_D(sA);   // (CPY, CPY_M, CPY_K)

    auto thr_copy_b = tiled_copy_b.get_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB);   // (CPY, CPY_N, CPY_K, k)
    auto tBsB = thr_copy_b.partition_D(sB);   // (CPY, CPY_N, CPY_K)

    // -- Main loop --
    auto num_k_tiles = size<2>(gA);

    for (int k = 0; k < num_k_tiles; ++k) {
        copy(tiled_copy_a, tAgA(_, _, _, k), tAsA);
        copy(tiled_copy_b, tBgB(_, _, _, k), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tiled_mma, tCsA, tCsB, tCrC);

        __syncthreads();
    }

    // -- Epilogue: fp32 accumulators → bf16 global memory --
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = __float2bfloat16(tCrC(i));
    }
}

void matmul_cute_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    using namespace cute;
    constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;
    dim3 block(128);
    dim3 grid(ceil_div(M, BLK_M), ceil_div(N, BLK_N));
    gemm_cute_v1_kernel<BLK_M, BLK_N, BLK_K><<<grid, block>>>(A, B, C, M, N, K);
}
