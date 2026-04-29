from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import modal
from modal.exception import FunctionTimeoutError

from .common.benchmarking import (
    DEFAULT_BENCH_RUNS,
    DEFAULT_QUANTILES,
    DEFAULT_WARMUP_RUNS,
    RunRecord,
    ShapeSpec,
    print_comparison,
    write_artifacts,
)
from .common.modal_utils import (
    ARTIFACTS_DIR,
    BUILD_CACHE_DIR,
    REMOTE_PACKAGE_DIR,
    build_kernel_image,
    gpu_arch,
    normalize_gpu,
)
from .common.kernel_registry import (
    KERNEL_PARAMS_METADATA_KEY,
    PTXAS_FACTORY_METADATA_KEY,
    load_target,
)
from .common.kernel_params import expand_parameterized_specs
from .common.ptxas import parse_ptxas_log, print_ptxas_report, write_ptxas_artifacts
from .common.runner import config_from_parts, run_gemm_benchmark
from .common.utils import CUTLASS_VERSION, parse_bool, parse_quantiles, parse_shapes
from .problems.gemm_hopper_bf16.registry import get_registry

LOCAL_PACKAGE_DIR = Path(__file__).resolve().parent
REMOTE_CUTLASS_PATH = Path("/opt/cutlass")
DEFAULT_OUTPUT_DIR = "artifacts/runs/latest"
BENCHMARK_TIMEOUT_SECONDS = 30 * 60
PTXAS_TIMEOUT_SECONDS = 20 * 60

image = build_kernel_image(
    local_mounts=[(LOCAL_PACKAGE_DIR, REMOTE_PACKAGE_DIR)],
    extra_commands=(
        f"git clone --depth 1 --branch v{CUTLASS_VERSION} "
        f"https://github.com/NVIDIA/cutlass.git {REMOTE_CUTLASS_PATH}",
    ),
)
app = modal.App(name="kernel-benchmarks", image=image)

build_cache_sm90 = modal.Volume.from_name("kernel-bench-build-sm90", create_if_missing=True)
build_cache_sm100 = modal.Volume.from_name("kernel-bench-build-sm100", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("kernel-bench-artifacts", create_if_missing=True)


@app.function(
    gpu="H100!",
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    volumes={str(BUILD_CACHE_DIR): build_cache_sm90, str(ARTIFACTS_DIR): artifacts_volume},
)
def _doctor_h100() -> dict[str, object]:
    return _doctor()

@app.function(
    gpu="B200",
    volumes={str(BUILD_CACHE_DIR): build_cache_sm100, str(ARTIFACTS_DIR): artifacts_volume},
)
def _doctor_b200() -> dict[str, object]:
    return _doctor()


@app.function(
    gpu="H100!",
    timeout=PTXAS_TIMEOUT_SECONDS,
    volumes={str(BUILD_CACHE_DIR): build_cache_sm90, str(ARTIFACTS_DIR): artifacts_volume},
)
def _benchmark_h100(job: dict[str, Any]) -> dict[str, object]:
    return _run_remote_job(job)


@app.function(
    gpu="H100!",
    volumes={str(BUILD_CACHE_DIR): build_cache_sm90, str(ARTIFACTS_DIR): artifacts_volume},
)
def _ptxas_h100(job: dict[str, Any]) -> dict[str, object]:
    return _run_remote_ptxas(job)


@app.function(
    gpu="B200",
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    volumes={str(BUILD_CACHE_DIR): build_cache_sm100, str(ARTIFACTS_DIR): artifacts_volume},
)
def _benchmark_b200(job: dict[str, Any]) -> dict[str, object]:
    return _run_remote_job(job)


@app.function(
    gpu="B200",
    timeout=PTXAS_TIMEOUT_SECONDS,
    volumes={str(BUILD_CACHE_DIR): build_cache_sm100, str(ARTIFACTS_DIR): artifacts_volume},
)
def _ptxas_b200(job: dict[str, Any]) -> dict[str, object]:
    return _run_remote_ptxas(job)


@app.local_entrypoint()
def main(
    command: str = "benchmark",
    problem: str = "gemm_hopper_bf16",
    gpu: str = "h100",
    kernels: str = "all",
    kinds: str = "",
    shapes: str = "4096,4096,4096",
    dtype: str = "bf16",
    bm: str = "",
    bn: str = "",
    bk: str = "",
    tm: str = "",
    tn: str = "",
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    quantiles: str = DEFAULT_QUANTILES,
    repetitions: int = 1,
    seed: int = 0,
    check_correctness: str = "true",
    benchmark_ref: str = "true",
    force_prepare: str = "false",
    parallel: str = "false",
    out: str = DEFAULT_OUTPUT_DIR,
) -> None:
    normalized_gpu = normalize_gpu(gpu)
    kernel_args = _kernel_args(BM=bm, BN=bn, BK=bk, TM=tm, TN=tn)

    if command == "doctor":
        _print_doctor(_doctor_fn(normalized_gpu).remote())
        return
    if command == "list-kernels":
        _print_kernels(
            problem=problem,
            gpu=normalized_gpu,
            kernels=kernels,
            kernel_args=kernel_args,
            kinds=kinds,
            dtype=dtype,
        )
        return
    if command == "benchmark":
        job = _make_job(
            problem=problem,
            gpu=normalized_gpu,
            kernels=kernels,
            kernel_args=kernel_args,
            kinds=kinds,
            shapes=shapes,
            dtype=dtype,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
            quantiles=quantiles,
            repetitions=repetitions,
            seed=seed,
            check_correctness=parse_bool(check_correctness),
            benchmark_ref=parse_bool(benchmark_ref),
            force_prepare=parse_bool(force_prepare),
            print_results=True,
        )
        try:
            record = _benchmark_fn(normalized_gpu).remote(job)
        except FunctionTimeoutError as exc:
            _write_timeout_artifact(
                error=exc,
                output_dir=Path(out),
                job=job,
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
            return
        _write_returned_record(record, Path(out))
        return
    if command == "ptxas":
        job = _make_job(
            problem=problem,
            gpu=normalized_gpu,
            kernels=kernels,
            kernel_args=kernel_args,
            kinds=kinds,
            dtype=dtype,
            force_prepare=parse_bool(force_prepare),
        )
        try:
            report = _ptxas_fn(normalized_gpu).remote(job)
        except FunctionTimeoutError as exc:
            _write_timeout_artifact(
                error=exc,
                output_dir=Path(out),
                job=job,
                timeout_seconds=PTXAS_TIMEOUT_SECONDS,
            )
            return
        _write_returned_ptxas(report, Path(out))
        return
    if command == "sweep":
        jobs = _make_sweep_jobs(
            problem=problem,
            gpu=normalized_gpu,
            kernels=kernels,
            kernel_args=kernel_args,
            kinds=kinds,
            shapes=shapes,
            dtype=dtype,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
            quantiles=quantiles,
            repetitions=repetitions,
            seed=seed,
            check_correctness=parse_bool(check_correctness),
            benchmark_ref=parse_bool(benchmark_ref),
            force_prepare=parse_bool(force_prepare),
        )
        fn = _benchmark_fn(normalized_gpu)
        try:
            records = (
                list(fn.map(jobs))
                if parse_bool(parallel)
                else [fn.remote(job) for job in jobs]
            )
        except FunctionTimeoutError as exc:
            _write_timeout_artifact(
                error=exc,
                output_dir=Path(out),
                job={"jobs": jobs},
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
            return
        _write_aggregate(records, Path(out))
        return

    raise ValueError(f"Unknown command: {command}")


def _doctor() -> dict[str, object]:
    import importlib
    import subprocess

    import torch

    major, minor = torch.cuda.get_device_capability()
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"sm_{major}{minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    result["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()

    for name, module in (
        ("triton", "triton"),
        ("gluon", "triton.experimental.gluon"),
        ("cute_dsl", "cutlass"),
    ):
        try:
            imported = importlib.import_module(module)
        except Exception as exc:
            result[name] = {"ok": False, "error": str(exc)}
        else:
            result[name] = {
                "ok": True,
                "version": getattr(imported, "__version__", "unknown"),
            }
    return result


def _run_remote_job(job: dict[str, Any]) -> dict[str, object]:
    registry = get_registry()
    specs = _resolve_specs(registry, job)
    if job["benchmark_ref"] and all(spec.name != "cublas" for spec in specs):
        specs.insert(0, registry.get("cublas"))

    shapes = tuple(ShapeSpec.from_tuple(shape) for shape in parse_shapes(job["shapes"]))
    config = config_from_parts(
        problem=job["problem"],
        gpu=job["gpu"],
        kernels=[spec.name for spec in specs],
        shapes=shapes,
        dtype=job["dtype"],
        warmup_runs=job["warmup_runs"],
        bench_runs=job["bench_runs"],
        quantiles=parse_quantiles(job["quantiles"]),
        repetitions=job["repetitions"],
        seed=job["seed"],
        check_correctness=job["check_correctness"],
        benchmark_ref=job["benchmark_ref"],
    )
    record = run_gemm_benchmark(
        config=config,
        specs=specs,
        output_dir=None,
        force_prepare=job["force_prepare"],
        print_results=bool(job.get("print_results", True)),
    )
    return record.to_dict()


def _run_remote_ptxas(job: dict[str, Any]) -> dict[str, object]:
    registry = get_registry()
    specs = _resolve_specs(registry, job)
    ptxas_inputs = [
        spec
        for spec in specs
        if "ptxas" in spec.metadata or PTXAS_FACTORY_METADATA_KEY in spec.metadata
    ]
    if not ptxas_inputs:
        names = ", ".join(spec.name for spec in specs)
        raise ValueError(f"No selected kernels expose ptxas inspection hooks: {names}")

    logs = []
    for spec in sorted(ptxas_inputs, key=lambda item: item.name):
        factory_target = spec.metadata.get(PTXAS_FACTORY_METADATA_KEY)
        if factory_target is not None:
            inspect = load_target(factory_target)
            logs.append(inspect(spec, force_prepare=job["force_prepare"]))
            continue
        inspect = load_target(spec.metadata["ptxas"])
        logs.append(inspect(force_prepare=job["force_prepare"]))

    report = parse_ptxas_log("\n".join(logs))
    print_ptxas_report(report)
    return report.to_dict()


def _make_job(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _make_sweep_jobs(**kwargs: Any) -> list[dict[str, Any]]:
    shapes = parse_shapes(kwargs["shapes"])
    jobs = []
    for idx, shape in enumerate(shapes):
        job = dict(kwargs)
        job["shapes"] = ",".join(str(dim) for dim in shape)
        job["seed"] = kwargs["seed"] + idx
        job["print_results"] = False
        jobs.append(job)
    return jobs


def _resolve_specs(registry, job: dict[str, Any]):
    specs = registry.resolve(
        problem=job["problem"],
        kernels=job["kernels"],
        kinds=job["kinds"],
        gpu=job["gpu"],
        dtype=job["dtype"],
    )
    return expand_parameterized_specs(
        specs,
        cast(dict[str, str], job.get("kernel_args", {})),
    )


def _kernel_args(**kwargs: str) -> dict[str, str]:
    return {key: value for key, value in kwargs.items() if value.strip()}


def _benchmark_fn(gpu: str):
    arch = gpu_arch(gpu)
    if arch == "sm90":
        return _benchmark_h100
    if arch == "sm100":
        return _benchmark_b200
    raise ValueError(f"Unsupported GPU: {gpu}")


def _ptxas_fn(gpu: str):
    arch = gpu_arch(gpu)
    if arch == "sm90":
        return _ptxas_h100
    if arch == "sm100":
        return _ptxas_b200
    raise ValueError(f"Unsupported GPU: {gpu}")


def _doctor_fn(gpu: str):
    arch = gpu_arch(gpu)
    if arch == "sm90":
        return _doctor_h100
    if arch == "sm100":
        return _doctor_b200
    raise ValueError(f"Unsupported GPU: {gpu}")


def _print_doctor(result: dict[str, object]) -> None:
    print("Modal kernel environment:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def _print_kernels(
    *,
    problem: str,
    gpu: str,
    kernels: str,
    kernel_args: dict[str, str],
    kinds: str,
    dtype: str,
) -> None:
    specs = expand_parameterized_specs(
        get_registry().resolve(
            problem=problem,
            kernels=kernels,
            kinds=kinds,
            gpu=gpu,
            dtype=dtype,
        ),
        kernel_args,
    )
    print(f"Available kernels for problem={problem!r}, gpu={gpu!r}, dtype={dtype!r}:")
    for spec in specs:
        path = spec.source or spec.metadata.get("path", "")
        params = spec.metadata.get(KERNEL_PARAMS_METADATA_KEY, "")
        print(f"  {spec.name:48s} {spec.kind:12s} {path} {params}")


def _write_returned_record(record: dict[str, object], output_dir: Path) -> None:
    run_record = _record_from_dict(record)
    write_artifacts(run_record, output_dir)
    print(f"\nWrote artifacts to {output_dir}")


def _write_timeout_artifact(
    *,
    error: FunctionTimeoutError,
    output_dir: Path,
    job: dict[str, object],
    timeout_seconds: int,
) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "timeout",
        "timeout_seconds": timeout_seconds,
        "error": str(error),
        "job": job,
    }
    (output_dir / "timeout.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "timeout.txt").write_text(
        f"Benchmark timed out after {timeout_seconds} seconds.\n{error}\n"
    )
    print(
        f"\nBenchmark timed out after {timeout_seconds} seconds. "
        f"Wrote timeout details to {output_dir}"
    )


def _write_returned_ptxas(report: dict[str, object], output_dir: Path) -> None:
    from .common.ptxas import PtxasRecord, PtxasReport

    records = []
    for raw in cast(list[dict[str, object]], report["records"]):
        records.append(
            PtxasRecord(
                function=str(raw["function"]),
                arch=None if raw["arch"] is None else str(raw["arch"]),
                registers=(
                    None if raw["registers"] is None
                    else _as_int(raw["registers"])
                ),
                smem_bytes=_as_int(raw["smem_bytes"]),
                stack_frame_bytes=_as_int(raw["stack_frame_bytes"]),
                spill_stores_bytes=_as_int(raw["spill_stores_bytes"]),
                spill_loads_bytes=_as_int(raw["spill_loads_bytes"]),
                cmem_bytes={
                    str(key): _as_int(value)
                    for key, value in cast(dict[str, object], raw["cmem_bytes"]).items()
                },
                raw=[str(line) for line in cast(list[object], raw["raw"])],
            )
        )
    ptxas_report = PtxasReport(
        records=records,
        warnings=[str(item) for item in cast(list[object], report["warnings"])],
        raw_log=str(report["raw_log"]),
    )
    write_ptxas_artifacts(ptxas_report, output_dir)
    print(f"\nWrote ptxas artifacts to {output_dir}")


def _write_aggregate(records: list[dict[str, object]], output_dir: Path) -> None:
    if not records:
        raise ValueError("No records returned from sweep.")

    run_records = [_record_from_dict(record) for record in records]
    first = run_records[0]
    aggregate = RunRecord(
        config=first.config,
        results=[result for record in run_records for result in record.results],
        metadata={"records": len(run_records), **first.metadata},
    )
    write_artifacts(aggregate, output_dir)
    print_comparison(aggregate.results)
    print(f"\nWrote aggregate artifacts to {output_dir}")


def _record_from_dict(payload: dict[str, object]) -> RunRecord:
    from .common.benchmarking import (
        BenchmarkConfig,
        BenchmarkStats,
        CorrectnessResult,
        KernelResult,
    )

    config_raw = payload["config"]
    assert isinstance(config_raw, dict)
    shapes_raw = cast(list[dict[str, object]], config_raw["shapes"])
    quantiles_raw = cast(list[object], config_raw["quantiles"])
    config = BenchmarkConfig(
        problem=str(config_raw["problem"]),
        gpu=str(config_raw["gpu"]),
        kernels=tuple(cast(list[str], config_raw["kernels"])),
        shapes=tuple(
            ShapeSpec(
                m=_as_int(shape["m"]),
                n=_as_int(shape["n"]),
                k=_as_int(shape["k"]),
            )
            for shape in shapes_raw
        ),
        dtype=str(config_raw["dtype"]),
        warmup_runs=_as_int(config_raw["warmup_runs"]),
        bench_runs=_as_int(config_raw["bench_runs"]),
        quantiles=tuple(_as_float(item) for item in quantiles_raw),
        repetitions=_as_int(config_raw["repetitions"]),
        seed=_as_int(config_raw["seed"]),
        check_correctness=bool(config_raw["check_correctness"]),
        benchmark_ref=bool(config_raw["benchmark_ref"]),
    )

    results = []
    for raw in cast(list[dict[str, object]], payload["results"]):
        stats_raw = cast(dict[str, object], raw["stats"])
        correctness_raw = cast(dict[str, object], raw["correctness"])
        shape_raw = cast(dict[str, object], raw["shape"])
        results.append(
            KernelResult(
                problem=str(raw["problem"]),
                kernel=str(raw["kernel"]),
                kind=str(raw["kind"]),
                gpu=str(raw["gpu"]),
                shape=ShapeSpec(
                    m=_as_int(shape_raw["m"]),
                    n=_as_int(shape_raw["n"]),
                    k=_as_int(shape_raw["k"]),
                ),
                dtype=str(raw["dtype"]),
                stats=BenchmarkStats(
                    samples=_as_int(stats_raw["samples"]),
                    mean_ms=_as_float(stats_raw["mean_ms"]),
                    std_ms=_as_float(stats_raw["std_ms"]),
                    min_ms=_as_float(stats_raw["min_ms"]),
                    median_ms=_as_float(stats_raw["median_ms"]),
                    max_ms=_as_float(stats_raw["max_ms"]),
                    quantiles_ms={
                        str(key): _as_float(value)
                        for key, value in cast(
                            dict[str, object],
                            stats_raw["quantiles_ms"],
                        ).items()
                    },
                    m=_as_int(stats_raw["m"]),
                    n=_as_int(stats_raw["n"]),
                    k=_as_int(stats_raw["k"]),
                ),
                correctness=CorrectnessResult(
                    passed=bool(correctness_raw["passed"]),
                    max_abs=(
                        None if correctness_raw["max_abs"] is None
                        else _as_float(correctness_raw["max_abs"])
                    ),
                    max_rel=(
                        None if correctness_raw["max_rel"] is None
                        else _as_float(correctness_raw["max_rel"])
                    ),
                    error=(
                        None if correctness_raw["error"] is None
                        else str(correctness_raw["error"])
                    ),
                ),
                repetition=_as_int(raw["repetition"]),
                kernel_path=(
                    None if raw.get("kernel_path") is None
                    else str(raw["kernel_path"])
                ),
                kernel_params=(
                    None if raw.get("kernel_params") is None
                    else str(raw["kernel_params"])
                ),
            )
        )
    metadata = cast(dict[str, object], payload["metadata"])

    return RunRecord(
        config=config,
        results=results,
        metadata=metadata,
        created_at=str(payload["created_at"]),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-like value, got {type(value).__name__}")
