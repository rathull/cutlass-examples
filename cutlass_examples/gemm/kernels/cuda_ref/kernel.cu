// CUDA matmul kernel implementation
// Performs C = A @ B where A is (M, K), B is (K, N), C is (M, N)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simple tiled matrix multiplication kernel
// Each thread block computes a tile of the output matrix
__global__ void matmul_kernel(
    const float* A,  // M x K matrix
    const float* B,  // K x N matrix
    float* C,        // M x N output matrix
    int M, int N, int K
) {
    // Block size for tiling (can be tuned)
    const int TILE_SIZE = 16;
    
    // Calculate which tile this thread block is responsible for
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Shared memory for caching tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Use double precision for accumulation to improve numerical accuracy
    // This matches the precision used by many optimized BLAS implementations
    double sum = 0.0;
    
    // Iterate over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        
        // Load tile of B into shared memory
        int bRow = tile * TILE_SIZE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute partial dot product
        // Accumulate in double precision for better accuracy
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += (double)As[threadIdx.y][k] * (double)Bs[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory (cast back to float)
    if (row < M && col < N) {
        C[row * N + col] = (float)sum;
    }
}

// Wrapper function to launch the kernel
extern "C" void launch_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    const int TILE_SIZE = 16;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K);
}
