#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#ifndef BM
#define BM 16
#endif

#ifndef BN
#define BN 16
#endif

#ifndef ENTRYPOINT
#define ENTRYPOINT cute_cpp_v0
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME cute_cpp_v0_kernel
#endif

static_assert(BM > 0 && BN > 0, "Tile dimensions must be positive.");
static_assert(BM * BN <= 1024, "CTA cannot have more than 1024 threads.");

__launch_bounds__(BM * BN)
__global__ void KERNEL_NAME(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    A = (const nv_bfloat16*)__builtin_assume_aligned(A, 16);
    BT = (const nv_bfloat16*)__builtin_assume_aligned(BT, 16);
    C = (nv_bfloat16*)__builtin_assume_aligned(C, 16);

    const int row = blockIdx.y * BM + threadIdx.y;
    const int col = blockIdx.x * BN + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    auto gA = cute::make_tensor(
        cute::make_gmem_ptr(A),
        cute::make_shape(M, K),
        cute::make_stride(K, 1)
    );
    auto gBT = cute::make_tensor(
        cute::make_gmem_ptr(BT),
        cute::make_shape(N, K),
        cute::make_stride(K, 1)
    );
    auto gC = cute::make_tensor(
        cute::make_gmem_ptr(C),
        cute::make_shape(M, N),
        cute::make_stride(N, 1)
    );

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(gA(row, k)) * __bfloat162float(gBT(col, k));
    }

    const float out = beta == 0.0f
        ? alpha * acc
        : alpha * acc + beta * __bfloat162float(gC(row, col));
    gC(row, col) = __float2bfloat16(out);
}

extern "C" void ENTRYPOINT(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ BT,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    dim3 block(BN, BM);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    KERNEL_NAME<<<grid, block>>>(
        A, BT, C, M, N, K, alpha, beta
    );
}
