// Python bindings for CUDA matmul kernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>  // for at::cuda::getCurrentCUDAStream()

// Forward declaration of the CUDA kernel launch function
extern "C" void launch_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
);

// Python-callable wrapper
void launch_matmul_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int M, int N, int K
) { 
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_matmul(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_matmul", &launch_matmul_torch, "Launch custom CUDA matmul kernel");
}
