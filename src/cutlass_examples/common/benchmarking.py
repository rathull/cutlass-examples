from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .utils import get_tflops, quantile_label

DEFAULT_WARMUP_RUNS = 50
DEFAULT_BENCH_RUNS = 500
DEFAULT_QUANTILES = "0.20,0.50,0.80,0.90,0.95,0.99"


@dataclass(frozen=True)
class ShapeSpec:
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"

    @classmethod
    def from_tuple(cls, shape: tuple[int, int, int]) -> ShapeSpec:
        return cls(m=shape[0], n=shape[1], k=shape[2])


@dataclass(frozen=True)
class BenchmarkConfig:
    problem: str
    gpu: str
    kernels: tuple[str, ...]
    shapes: tuple[ShapeSpec, ...]
    dtype: str = "bf16"
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    bench_runs: int = DEFAULT_BENCH_RUNS
    quantiles: tuple[float, ...] = (0.20, 0.50, 0.80, 0.90, 0.95, 0.99)
    repetitions: int = 1
    seed: int = 0
    check_correctness: bool = True
    benchmark_ref: bool = True


@dataclass(frozen=True)
class BenchmarkStats:
    samples: int
    mean_ms: float
    std_ms: float
    min_ms: float
    median_ms: float
    max_ms: float
    quantiles_ms: dict[str, float]
    m: int
    n: int
    k: int

    def tflops(self, latency_ms: float) -> float:
        return get_tflops(self.m, self.n, self.k, latency_ms)

    @property
    def median_tflops(self) -> float:
        return self.tflops(self.median_ms)


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs: float | None = None
    max_rel: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class KernelResult:
    problem: str
    kernel: str
    kind: str
    gpu: str
    shape: ShapeSpec
    dtype: str
    stats: BenchmarkStats
    correctness: CorrectnessResult
    repetition: int = 0
    kernel_path: str | None = None
    kernel_params: str | None = None

    @property
    def row(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "kernel": self.kernel,
            "kind": self.kind,
            "gpu": self.gpu,
            "shape": self.shape.label,
            "m": self.shape.m,
            "n": self.shape.n,
            "k": self.shape.k,
            "dtype": self.dtype,
            "repetition": self.repetition,
            "samples": self.stats.samples,
            "median_ms": self.stats.median_ms,
            "median_tflops": self.stats.median_tflops,
            "mean_ms": self.stats.mean_ms,
            "std_ms": self.stats.std_ms,
            "min_ms": self.stats.min_ms,
            "max_ms": self.stats.max_ms,
            "correct": self.correctness.passed,
            "max_abs": self.correctness.max_abs,
            "max_rel": self.correctness.max_rel,
            "kernel_path": self.kernel_path,
            "kernel_params": self.kernel_params,
        }


@dataclass(frozen=True)
class RunRecord:
    config: BenchmarkConfig
    results: list[KernelResult]
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "metadata": self.metadata,
            "config": _jsonify(asdict(self.config)),
            "results": [_jsonify(asdict(result)) for result in self.results],
        }


def do_bench(
    fn: Callable[[], object],
    *,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
) -> list[float]:
    import torch

    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got: {warmup_runs}")
    if bench_runs <= 0:
        raise ValueError(f"bench_runs must be > 0, got: {bench_runs}")

    torch.cuda.synchronize()
    for _ in range(warmup_runs):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_runs)]

    for idx in range(bench_runs):
        start_events[idx].record()
        fn()
        end_events[idx].record()

    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(start_events, end_events)]


def summarize_latency(
    latency_ms_list: list[float],
    *,
    shape: ShapeSpec,
    quantiles: Iterable[float],
) -> BenchmarkStats:
    if not latency_ms_list:
        raise ValueError("Expected at least one latency sample.")

    latency_ms_array = np.asarray(latency_ms_list, dtype=float)
    return BenchmarkStats(
        samples=len(latency_ms_array),
        mean_ms=float(np.mean(latency_ms_array)),
        std_ms=float(np.std(latency_ms_array)),
        min_ms=float(np.min(latency_ms_array)),
        median_ms=float(np.median(latency_ms_array)),
        max_ms=float(np.max(latency_ms_array)),
        quantiles_ms={
            quantile_label(quantile): float(np.quantile(latency_ms_array, quantile))
            for quantile in quantiles
        },
        m=shape.m,
        n=shape.n,
        k=shape.k,
    )


def print_result(result: KernelResult) -> None:
    correctness = "ok" if result.correctness.passed else "failed"
    print(
        f"{result.kernel:24s} {result.shape.label:>16s} "
        f"median={result.stats.median_ms:9.4f} ms "
        f"mean={result.stats.mean_ms:9.4f} ms "
        f"std={result.stats.std_ms:9.4f} ms "
        f"{result.stats.median_tflops:10.3f} TFLOPS "
        f"correct={correctness}"
    )


def print_comparison(results: list[KernelResult], *, baseline_kernel: str = "cublas") -> None:
    if not results:
        return

    by_shape: dict[str, list[KernelResult]] = {}
    for result in results:
        by_shape.setdefault(result.shape.label, []).append(result)

    for shape_label, shape_results in by_shape.items():
        baseline = next(
            (result for result in shape_results if result.kernel == baseline_kernel),
            None,
        )
        baseline_ms = baseline.stats.median_ms if baseline else None
        sorted_results = sorted(shape_results, key=lambda item: item.stats.median_ms)

        # Pre-format all values so we can measure their widths
        rows_data = []
        for result in sorted_results:
            speedup = baseline_ms / result.stats.median_ms if baseline_ms is not None else None
            rows_data.append({
                "kernel":  result.kernel,
                "kind":    result.kind,
                "median":  f"{result.stats.median_ms:.4f}",
                "mean":    f"{result.stats.mean_ms:.4f}",
                "std":     f"{result.stats.std_ms:.4f}",
                "tflops":  f"{result.stats.median_tflops:.3f}",
                "speedup": f"{speedup:.2f}x" if speedup is not None else None,
            })

        # Compute column widths from headers and all row values
        columns = ["kernel", "kind", "median", "mean", "std", "tflops"]
        headers = ["kernel", "kind", "median", "mean", "std", "TFLOPS"]
        if baseline_ms is not None:
            columns.append("speedup")
            headers.append("speedup")

        widths = {
            col: max(len(hdr), *(len(row[col] or "") for row in rows_data))
            for col, hdr in zip(columns, headers)
        }

        header = "  ".join(
            hdr.ljust(widths[col]) if col in ("kernel", "kind") else hdr.rjust(widths[col])
            for col, hdr in zip(columns, headers)
        )

        print(f"\nComparison for {shape_label} (latency in ms):")
        print(header)
        print("-" * len(header))

        for row in rows_data:
            line = "  ".join(
                (row[col] or "").ljust(widths[col]) if col in ("kernel", "kind")
                else (row[col] or "").rjust(widths[col])
                for col in columns
            )
            print(line)


def write_artifacts(record: RunRecord, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = record.to_dict()

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w") as fp:
        for result in record.results:
            row: dict[str, object] = {
                "created_at": record.created_at,
                "metadata": record.metadata,
            }
            row.update(result.row)
            fp.write(json.dumps(_jsonify(row)) + "\n")

    csv_path = output_dir / "results.csv"
    rows = [result.row for result in record.results]
    if rows:
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def default_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            metadata.update(
                {
                    "device_name": torch.cuda.get_device_name(),
                    "compute_capability": f"sm_{major}{minor}",
                }
            )
    except Exception as exc:
        metadata["torch_error"] = str(exc)

    return metadata


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _jsonify(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value
