import pytest

from cutlass_examples.common.kernel_discovery import discover_problem_kernels
from cutlass_examples.common.kernel_registry import KernelRegistry, KernelSpec
from cutlass_examples.problems.gemm_hopper_bf16.registry import get_registry


def test_registry_rejects_duplicate_names():
    spec = KernelSpec(name="k0", problem="gemm_hopper_bf16", kind="reference", target="mod:fn")
    registry = KernelRegistry([spec])

    with pytest.raises(ValueError):
        registry.register(spec)


def test_resolve_by_kernel_name_and_gpu():
    specs = get_registry().resolve(
        problem="gemm_hopper_bf16",
        kernels="triton_v0",
        gpu="H100!",
        dtype="bf16",
    )

    assert [spec.name for spec in specs] == ["triton_v0"]


def test_resolve_by_kind_filter():
    registry = KernelRegistry(
        [
            KernelSpec(
                name="triton_kernel",
                problem="gemm_hopper_bf16",
                kind="triton",
                target="mod:triton_kernel",
            ),
            KernelSpec(
                name="gluon_kernel",
                problem="gemm_hopper_bf16",
                kind="gluon",
                target="mod:gluon_kernel",
            ),
        ]
    )

    specs = registry.resolve(
        problem="gemm_hopper_bf16",
        kernels="all",
        kinds="triton",
    )

    assert [spec.name for spec in specs] == ["triton_kernel"]


def test_list_all_gemm_kernels_has_reference():
    names = [spec.name for spec in get_registry().all(problem="gemm_hopper_bf16")]

    assert "cublas" in names


def test_discovery_infers_name_kind_problem_and_target(tmp_path):
    problem_dir = tmp_path / "toy_problem"
    kernel = problem_dir / "kernels" / "triton" / "my_kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("def run(inputs): return inputs\n")

    specs = discover_problem_kernels(problem="toy_problem", problem_dir=problem_dir)

    assert len(specs) == 1
    assert specs[0].name == "my_kernel"
    assert specs[0].problem == "toy_problem"
    assert specs[0].kind == "triton"
    assert specs[0].target == "cutlass_examples.problems.toy_problem.kernels.triton.my_kernel:run"
    assert specs[0].source == "kernels/triton/my_kernel.py"


def test_discovery_infers_native_source(tmp_path):
    problem_dir = tmp_path / "toy_problem"
    kernel = problem_dir / "kernels" / "native_cuda" / "cuda_inline_ptx_v0.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("__global__ void kernel() {}\n")

    specs = discover_problem_kernels(problem="toy_problem", problem_dir=problem_dir)

    assert len(specs) == 1
    assert specs[0].name == "cuda_inline_ptx_v0"
    assert specs[0].kind == "native_cuda"
    assert specs[0].source == "kernels/native_cuda/cuda_inline_ptx_v0.cu"
    assert specs[0].metadata["source"] == "kernels/native_cuda/cuda_inline_ptx_v0.cu"
