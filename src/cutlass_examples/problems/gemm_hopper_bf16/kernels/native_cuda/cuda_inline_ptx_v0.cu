#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void cuda_inline_ptx_v0_kernel(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M,
    int N,
    int K,
    long long stride_am,
    long long stride_ak,
    long long stride_bk,
    long long stride_bn
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __bfloat162float(A[row * stride_am + k * stride_ak]);
        float b = __bfloat162float(B[k * stride_bk + col * stride_bn]);
        acc += a * b;
    }

    // Minimal inline PTX marker so this example exercises inline assembly too.
    asm volatile("// cuda_inline_ptx_v0 store");
    C[row * N + col] = __float2bfloat16(acc);
}

extern "C" void cuda_inline_ptx_v0(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M,
    int N,
    int K,
    long long stride_am,
    long long stride_ak,
    long long stride_bk,
    long long stride_bn
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cuda_inline_ptx_v0_kernel<<<grid, block>>>(
        A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn
    );
}
