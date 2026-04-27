import pytest

from cutlass_examples.common.kernel_registry import KernelRegistry, KernelSpec
from cutlass_examples.problems.gemm_hopper.registry import get_registry


def test_registry_rejects_duplicate_names():
    spec = KernelSpec(name="k0", problem="gemm_hopper", kind="reference", target="mod:fn")
    registry = KernelRegistry([spec])

    with pytest.raises(ValueError):
        registry.register(spec)


def test_resolve_by_kernel_name_and_gpu():
    specs = get_registry().resolve(
        problem="gemm_hopper",
        kernels="hopper_triton_v0",
        gpu="H100!",
        dtype="bf16",
    )

    assert [spec.name for spec in specs] == ["hopper_triton_v0"]


def test_resolve_by_tags_excludes_broken_tags():
    registry = KernelRegistry(
        [
            KernelSpec(
                name="good",
                problem="gemm_hopper",
                kind="triton",
                target="mod:good",
                tags=("hopper",),
            ),
            KernelSpec(
                name="bad",
                problem="gemm_hopper",
                kind="triton",
                target="mod:bad",
                tags=("hopper", "broken"),
            ),
        ]
    )

    specs = registry.resolve(
        problem="gemm_hopper",
        kernels="all",
        tags="hopper",
        exclude_tags="broken",
    )

    assert [spec.name for spec in specs] == ["good"]


def test_list_all_gemm_kernels_has_reference():
    names = [spec.name for spec in get_registry().all(problem="gemm_hopper")]

    assert "cuBLAS" in names
