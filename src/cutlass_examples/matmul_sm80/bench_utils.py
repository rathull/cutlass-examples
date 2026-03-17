from __future__ import annotations

from typing import Callable

import numpy as np

from ..common.utils import get_tflops, quantile_label

DEFAULT_WARMUP_RUNS = 50
DEFAULT_BENCH_RUNS = 500
DEFAULT_QUANTILES = "0.20,0.50,0.80,0.90,0.95,0.99"


def do_bench(
    fn: Callable,
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


def print_summary(
    name: str,
    latency_ms_list: list[float],
    *,
    m: int,
    n: int,
    k: int,
    quantiles: list[float],
) -> None:
    latency_ms_array = np.asarray(latency_ms_list, dtype=float)

    print(f"{name:12s} stats ({len(latency_ms_array)} samples):")
    for quantile in quantiles:
        latency_ms = np.quantile(latency_ms_array, quantile)
        tflops = get_tflops(m, n, k, latency_ms)
        print(f"  {quantile_label(quantile):>6s}: {latency_ms:.4f} ms\t{tflops:.3f} TFLOPS")

    print()

    for label, latency_ms in (
        ("min", np.min(latency_ms_array)),
        ("max", np.max(latency_ms_array)),
        ("median", np.median(latency_ms_array)),
    ):
        print(f"  {label:>6s}: {latency_ms:.4f} ms\t{get_tflops(m, n, k, latency_ms):.3f} TFLOPS")

    print()

    print(f"  {'mean':>6s}: {np.mean(latency_ms_array):.4f} ms\t{get_tflops(m, n, k, np.mean(latency_ms_array)):.3f} TFLOPS")
    print(f"  {'std':>6s}: {np.std(latency_ms_array):.4f} ms\n")
