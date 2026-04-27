from __future__ import annotations

from ...common.kernel_registry import KernelRegistry, KernelSpec

PROBLEM = "gemm_hopper"


def get_registry() -> KernelRegistry:
    return KernelRegistry(_kernel_specs())


def _kernel_specs() -> list[KernelSpec]:
    return [
        KernelSpec(
            name="cuBLAS",
            problem=PROBLEM,
            kind="reference",
            target="cutlass_examples.problems.gemm_hopper.backends.reference.kernels:cublas",
            supported_gpus=("any",),
            tags=("reference", "baseline"),
            description="torch.matmul/cuBLAS reference baseline",
        ),
        KernelSpec(
            name="sm80_v0",
            problem=PROBLEM,
            kind="native_cuda",
            target="cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:sm80_v0",
            supported_gpus=("A100", "A100-40GB", "A100-80GB"),
            tags=("sm80", "native_cuda"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:prepare_sm80",
                "ptxas": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:inspect_sm80_ptxas",
            },
        ),
        KernelSpec(
            name="sm80_v1",
            problem=PROBLEM,
            kind="native_cuda",
            target="cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:sm80_v1",
            supported_gpus=("A100", "A100-40GB", "A100-80GB"),
            tags=("sm80", "native_cuda"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:prepare_sm80",
                "ptxas": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:inspect_sm80_ptxas",
            },
        ),
        KernelSpec(
            name="sm80_cute_v0",
            problem=PROBLEM,
            kind="native_cuda",
            target="cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:sm80_cute_v0",
            supported_gpus=("A100", "A100-40GB", "A100-80GB"),
            tags=("sm80", "native_cuda", "cute_cpp"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:prepare_sm80",
                "ptxas": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:inspect_sm80_ptxas",
            },
        ),
        KernelSpec(
            name="sm80_cute_v1",
            problem=PROBLEM,
            kind="native_cuda",
            target="cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:sm80_cute_v1",
            supported_gpus=("A100", "A100-40GB", "A100-80GB"),
            tags=("sm80", "native_cuda", "cute_cpp"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:prepare_sm80",
                "ptxas": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:inspect_sm80_ptxas",
            },
        ),
        KernelSpec(
            name="sm80_cutlass_v0",
            problem=PROBLEM,
            kind="native_cuda",
            target="cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:sm80_cutlass_v0",
            supported_gpus=("A100", "A100-40GB", "A100-80GB"),
            tags=("sm80", "native_cuda", "cutlass_cpp"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:prepare_sm80",
                "ptxas": "cutlass_examples.problems.gemm_hopper.backends.native_cuda.kernels:inspect_sm80_ptxas",
            },
        ),
        KernelSpec(
            name="hopper_triton_v0",
            problem=PROBLEM,
            kind="triton",
            target="cutlass_examples.problems.gemm_hopper.backends.triton.kernels:matmul_v0",
            supported_gpus=("H100", "H200", "B200"),
            tags=("hopper", "blackwell", "triton", "starter"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.triton.kernels:prepare_triton",
            },
            description="Starter Triton GEMM kernel for Hopper/Blackwell",
        ),
        KernelSpec(
            name="hopper_gluon_smoke",
            problem=PROBLEM,
            kind="gluon",
            target="cutlass_examples.problems.gemm_hopper.backends.gluon.kernels:smoke_matmul",
            supported_gpus=("H100", "H200", "B200"),
            tags=("hopper", "blackwell", "gluon", "smoke"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.gluon.kernels:prepare_gluon",
            },
            description="Gluon dependency smoke kernel using cuBLAS fallback",
        ),
        KernelSpec(
            name="hopper_cute_dsl_smoke",
            problem=PROBLEM,
            kind="cute_dsl",
            target="cutlass_examples.problems.gemm_hopper.backends.cute_dsl.kernels:smoke_matmul",
            supported_gpus=("H100", "H200", "B200"),
            tags=("hopper", "blackwell", "cute_dsl", "smoke"),
            metadata={
                "prepare": "cutlass_examples.problems.gemm_hopper.backends.cute_dsl.kernels:prepare_cute_dsl",
            },
            description="CuTe DSL dependency smoke kernel using cuBLAS fallback",
        ),
    ]
