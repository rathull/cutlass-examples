import json

from cutlass_examples.common.benchmarking import (
    BenchmarkConfig,
    BenchmarkStats,
    CorrectnessResult,
    KernelResult,
    RunRecord,
    ShapeSpec,
    write_artifacts,
)


def test_write_artifacts(tmp_path):
    shape = ShapeSpec(m=128, n=128, k=128)
    record = RunRecord(
        config=BenchmarkConfig(
            problem="gemm_hopper_bf16",
            gpu="H100!",
            kernels=("cublas",),
            shapes=(shape,),
        ),
        results=[
            KernelResult(
                problem="gemm_hopper_bf16",
                kernel="cublas",
                kind="reference",
                gpu="H100!",
                shape=shape,
                dtype="bf16",
                stats=BenchmarkStats(
                    samples=3,
                    mean_ms=1.0,
                    std_ms=0.1,
                    min_ms=0.9,
                    median_ms=1.0,
                    max_ms=1.1,
                    quantiles_ms={"p50": 1.0},
                    m=128,
                    n=128,
                    k=128,
                ),
                correctness=CorrectnessResult(passed=True, max_abs=0.0, max_rel=0.0),
                kernel_path="kernels/reference/cublas.py",
            )
        ],
        metadata={"device_name": "test-gpu"},
    )

    write_artifacts(record, tmp_path)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "results.csv").exists()

    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["metadata"]["device_name"] == "test-gpu"
    assert payload["results"][0]["kernel"] == "cublas"
    assert payload["results"][0]["kernel_path"] == "kernels/reference/cublas.py"
