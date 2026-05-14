import json

from cutlass_examples.common.kernel_registry import (
    ALLOW_FAILURE_METADATA_KEY,
    EXTRA_CUDA_CFLAGS_METADATA_KEY,
    EXTRA_LDFLAGS_METADATA_KEY,
    KERNEL_PARAMS_METADATA_KEY,
    PREPARE_FACTORY_METADATA_KEY,
    PTXAS_FACTORY_METADATA_KEY,
    RUNNER_FACTORY_METADATA_KEY,
    KernelSpec,
)
from cutlass_examples.common.kernel_params import expand_parameterized_specs


def test_cuda_v1_expand_specs_replaces_kernel_with_variants():
    base = KernelSpec(
        name="cuda_v1_smem_tiling",
        problem="gemm_hopper_bf16",
        kind="native_cuda",
        target=(
            "cutlass_examples.problems.gemm_hopper_bf16.kernels.native_cuda."
            "cuda_v1_smem_tiling:run"
        ),
        source="kernels/native_cuda/cuda_v1_smem_tiling.cu",
        metadata={
            "source": "kernels/native_cuda/cuda_v1_smem_tiling.cu",
            "prepare": "base:prepare",
            "ptxas": "base:inspect_ptxas",
        },
    )

    variants = expand_parameterized_specs(
        [base],
        {"BM": "64,128", "BN": "128", "BK": "16", "TM": "8", "TN": "8"},
    )

    assert [variant.name for variant in variants] == [
        "cuda_v1_smem_tiling__bm64_bn128_bk16_tm8_tn8",
        "cuda_v1_smem_tiling__bm128_bn128_bk16_tm8_tn8",
    ]
    assert variants[0].target is None
    assert variants[0].metadata[KERNEL_PARAMS_METADATA_KEY] == (
        "BM=64,BN=128,BK=16,TM=8,TN=8"
    )
    assert "prepare" not in variants[0].metadata
    assert "ptxas" not in variants[0].metadata
    assert RUNNER_FACTORY_METADATA_KEY in variants[0].metadata
    assert PREPARE_FACTORY_METADATA_KEY in variants[0].metadata
    assert PTXAS_FACTORY_METADATA_KEY in variants[0].metadata
    assert variants[0].metadata[ALLOW_FAILURE_METADATA_KEY] == "true"

    cuda_cflags = json.loads(variants[0].metadata[EXTRA_CUDA_CFLAGS_METADATA_KEY])
    assert "-DENTRYPOINT=cuda_v1_smem_tiling__bm64_bn128_bk16_tm8_tn8" in cuda_cflags
    assert "-DKERNEL_NAME=cuda_v1_smem_tiling__bm64_bn128_bk16_tm8_tn8_kernel" in cuda_cflags
    assert "-DBM=64" in cuda_cflags
    assert "-DBN=128" in cuda_cflags
    assert json.loads(variants[0].metadata[EXTRA_LDFLAGS_METADATA_KEY]) == []


def test_cuda_v2_expand_specs_includes_num_stages_and_ldflags():
    base = KernelSpec(
        name="cuda_v2_tma_wgmma",
        problem="gemm_hopper_bf16",
        kind="native_cuda",
        target=(
            "cutlass_examples.problems.gemm_hopper_bf16.kernels.native_cuda."
            "cuda_v2_tma_wgmma:run"
        ),
        source="kernels/native_cuda/cuda_v2_tma_wgmma.cu",
        metadata={"source": "kernels/native_cuda/cuda_v2_tma_wgmma.cu"},
    )

    variants = expand_parameterized_specs(
        [base],
        {"BM": "64", "BN": "64", "BK": "64", "NUM_STAGES": "2,4"},
    )

    assert [variant.name for variant in variants] == [
        "cuda_v2_tma_wgmma__bm64_bn64_bk64_num_stages2",
        "cuda_v2_tma_wgmma__bm64_bn64_bk64_num_stages4",
    ]
    assert variants[0].metadata[KERNEL_PARAMS_METADATA_KEY] == (
        "BM=64,BN=64,BK=64,NUM_STAGES=2"
    )
    cuda_cflags = json.loads(variants[0].metadata[EXTRA_CUDA_CFLAGS_METADATA_KEY])
    assert "-DNUM_STAGES=2" in cuda_cflags
    assert json.loads(variants[0].metadata[EXTRA_LDFLAGS_METADATA_KEY]) == ["-lcuda"]


def test_cuda_v1_expand_specs_uses_defaults_for_unspecified_values():
    base = KernelSpec(
        name="cuda_v1_smem_tiling",
        problem="gemm_hopper_bf16",
        kind="native_cuda",
        target=(
            "cutlass_examples.problems.gemm_hopper_bf16.kernels.native_cuda."
            "cuda_v1_smem_tiling:run"
        ),
        source="kernels/native_cuda/cuda_v1_smem_tiling.cu",
    )

    variants = expand_parameterized_specs([base], {"BM": "64"})

    assert [variant.name for variant in variants] == [
        "cuda_v1_smem_tiling__bm64_bn128_bk16_tm8_tn8"
    ]


def test_cuda_v1_expand_specs_keeps_base_kernel_without_args():
    base = KernelSpec(
        name="cuda_v1_smem_tiling",
        problem="gemm_hopper_bf16",
        kind="native_cuda",
        target=(
            "cutlass_examples.problems.gemm_hopper_bf16.kernels.native_cuda."
            "cuda_v1_smem_tiling:run"
        ),
        source="kernels/native_cuda/cuda_v1_smem_tiling.cu",
    )

    assert expand_parameterized_specs([base], {}) == [base]


def test_sweep_jobs_split_parameter_grid():
    from cutlass_examples.cli import _kernel_args, _make_sweep_jobs

    kernel_args = _kernel_args(BM="64,128", BN="", BK="64", NUM_STAGES="2,4")
    jobs = _make_sweep_jobs(
        problem="gemm_hopper_bf16",
        gpu="h100",
        kernels="cuda_v2_tma_wgmma",
        kernel_args=kernel_args,
        kinds="",
        shapes="4096,4096,4096",
        dtype="bf16",
        warmup_runs=1,
        bench_runs=1,
        quantiles="0.5",
        repetitions=1,
        seed=0,
        check_correctness=True,
        benchmark_ref=True,
        force_prepare=False,
    )

    assert [job["kernel_args"] for job in jobs] == [
        {"BM": "64", "BK": "64", "NUM_STAGES": "2"},
        {"BM": "64", "BK": "64", "NUM_STAGES": "4"},
        {"BM": "128", "BK": "64", "NUM_STAGES": "2"},
        {"BM": "128", "BK": "64", "NUM_STAGES": "4"},
    ]
    assert [job["benchmark_ref"] for job in jobs] == [True, False, False, False]


def test_sweep_remote_exception_becomes_failed_record():
    from cutlass_examples.cli import _coerce_sweep_record, _kernel_args, _make_sweep_jobs

    [job] = _make_sweep_jobs(
        problem="gemm_hopper_bf16",
        gpu="h100",
        kernels="cuda_v2_tma_wgmma",
        kernel_args=_kernel_args(BM="64", BN="64", BK="64", NUM_STAGES="2"),
        kinds="",
        shapes="4096,4096,4096",
        dtype="bf16",
        warmup_runs=1,
        bench_runs=1,
        quantiles="0.5",
        repetitions=1,
        seed=0,
        check_correctness=True,
        benchmark_ref=False,
        force_prepare=False,
    )

    record = _coerce_sweep_record(job, RuntimeError("boom"))

    assert record["results"][0]["correctness"]["passed"] is False
    assert record["results"][0]["correctness"]["error"].startswith(
        "remote job failed: RuntimeError: boom"
    )
    assert record["results"][0]["kernel_params"] == "BM=64,BN=64,BK=64,NUM_STAGES=2"
