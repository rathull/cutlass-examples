from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..common.benchmarking import (
    DEFAULT_BENCH_RUNS,
    DEFAULT_QUANTILES,
    DEFAULT_WARMUP_RUNS,
    ShapeSpec,
    do_bench as _do_bench,
    summarize_latency as _summarize_common_latency,
)
from ..common.utils import quantile_label


@dataclass(frozen=True)
class BenchmarkStats:
    name: str
    samples: int
    mean_ms: float
    std_ms: float
    min_ms: float
    median_ms: float
    max_ms: float
    quantiles_ms: dict[float, float]
    m: int
    n: int
    k: int

    def tflops(self, latency_ms: float) -> float:
        from ..common.utils import get_tflops

        return get_tflops(self.m, self.n, self.k, latency_ms)

    @property
    def median_tflops(self) -> float:
        return self.tflops(self.median_ms)


do_bench: Callable[..., list[float]] = _do_bench

def summarize_latency(
    name: str,
    latency_ms_list: list[float],
    *,
    m: int,
    n: int,
    k: int,
    quantiles: list[float],
) -> BenchmarkStats:
    stats = _summarize_common_latency(
        latency_ms_list,
        shape=ShapeSpec(m=m, n=n, k=k),
        quantiles=quantiles,
    )
    return BenchmarkStats(
        name=name,
        samples=stats.samples,
        mean_ms=stats.mean_ms,
        std_ms=stats.std_ms,
        min_ms=stats.min_ms,
        median_ms=stats.median_ms,
        max_ms=stats.max_ms,
        quantiles_ms={
            quantile: stats.quantiles_ms[quantile_label(quantile)]
            for quantile in quantiles
        },
        m=stats.m,
        n=stats.n,
        k=stats.k,
    )


def print_summary(
    name: str,
    latency_ms_list: list[float],
    *,
    m: int,
    n: int,
    k: int,
    quantiles: list[float],
) -> BenchmarkStats:
    stats = summarize_latency(
        name,
        latency_ms_list,
        m=m,
        n=n,
        k=k,
        quantiles=quantiles,
    )
    print_stats(stats)
    return stats


def print_stats(stats: BenchmarkStats) -> None:
    print(f"{stats.name:16s} stats ({stats.samples} samples):")
    for quantile, latency_ms in stats.quantiles_ms.items():
        print(
            f"  {quantile_label(quantile):>6s}: "
            f"{latency_ms:.4f} ms\t{stats.tflops(latency_ms):.3f} TFLOPS"
        )

    print()

    for label, latency_ms in (
        ("min", stats.min_ms),
        ("max", stats.max_ms),
        ("median", stats.median_ms),
    ):
        print(
            f"  {label:>6s}: "
            f"{latency_ms:.4f} ms\t{stats.tflops(latency_ms):.3f} TFLOPS"
        )

    print()

    print(
        f"  {'mean':>6s}: "
        f"{stats.mean_ms:.4f} ms\t{stats.tflops(stats.mean_ms):.3f} TFLOPS"
    )
    print(f"  {'std':>6s}: {stats.std_ms:.4f} ms\n")


def print_comparison(
    results: list[BenchmarkStats],
    *,
    baseline_name: str | None = None,
) -> None:
    if not results:
        return

    baseline = next((result for result in results if result.name == baseline_name), None)
    baseline_ms = baseline.median_ms if baseline else None

    print("Comparison (median latency):")
    header = f"{'kernel':16s} {'ms':>10s} {'TFLOPS':>10s}"
    if baseline_ms is not None:
        header += f" {'speedup':>10s}"
    print(header)
    print("-" * len(header))

    for result in results:
        row = (
            f"{result.name:16s} "
            f"{result.median_ms:10.4f} "
            f"{result.median_tflops:10.3f}"
        )
        if baseline_ms is not None:
            row += f" {baseline_ms / result.median_ms:9.2f}x"
        print(row)

    print()
