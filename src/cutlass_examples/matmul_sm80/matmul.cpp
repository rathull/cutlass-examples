#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v0;
MatmulFn matmul_v1;
MatmulFn matmul_cute_v0;

template <MatmulFn matmul_fn>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    //auto C = at::zeros({M, N}, A.options());    // for correctness check, use this
    matmul_fn(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K
    );
    return C;
}

typedef void ProfileMatmulFn(
    const nv_bfloat16 *A,
    const nv_bfloat16 *B,
          nv_bfloat16 *C,
    int M, int N, int K,
    int64_t *profiler,
    int num_entries
);

ProfileMatmulFn profile_matmul_v5;
ProfileMatmulFn profile_matmul_v6;

template <ProfileMatmulFn profile_matmul_fn>
at::Tensor profile_matmul(
    const at::Tensor& A,
    const at::Tensor& B,
          at::Tensor& profiler,
    int64_t num_entries
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    //auto C = at::zeros({M, N}, A.options());  // for correctness check, use this
    profile_matmul_fn(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        num_entries
    );
    return C;
}

TORCH_LIBRARY(my_matmul, m) {
    m.def("matmul_v0(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_v0", &matmul<matmul_v0>);
    m.def("matmul_v1(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_v1", &matmul<matmul_v1>);
    m.def("matmul_cute_v0(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_cute_v0", &matmul<matmul_cute_v0>);
}