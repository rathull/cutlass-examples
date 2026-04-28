#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE_M = 16;
constexpr int BLOCK_SIZE_N = 16;

__launch_bounds__(BLOCK_SIZE_M * BLOCK_SIZE_N) // TODO: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__global__ void cuda_v0_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // 16 byte alignment allows for vectorized 128-bit load/stores
    A = (const nv_bfloat16*)__builtin_assume_aligned(A, 16);
    B = (const nv_bfloat16*)__builtin_assume_aligned(B, 16);
    C = (nv_bfloat16*)__builtin_assume_aligned(C, 16);

    const int row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }   
    
    // This thread will compute C[row][col]
    float out = 0.0f;
    for (int k = 0; k < K; ++k) {
        out += __bfloat162float(A[row * K + k]) * __bfloat162float(B[col * K + k]);
    }

    if (beta == 0.0f) {
        C[row * N + col] = __float2bfloat16(alpha * out);
    } else {
        C[row * N + col] = __float2bfloat16(alpha * out + beta * __bfloat162float(C[row * N + col]));
    }
}

extern "C" void cuda_v0(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    dim3 block(BLOCK_SIZE_M, BLOCK_SIZE_N);
    dim3 grid(
        (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    cuda_v0_kernel<<<grid, block>>>(
        A, B, C, M, N, K, alpha, beta
    );
}
