from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .benchmarking import (
    BenchmarkConfig,
    CorrectnessResult,
    KernelResult,
    RunRecord,
    ShapeSpec,
    default_metadata,
    do_bench,
    print_comparison,
    print_result,
    summarize_latency,
    write_artifacts,
)
from .kernel_registry import KernelSpec, load_target


def run_gemm_benchmark(
    *,
    config: BenchmarkConfig,
    specs: Iterable[KernelSpec],
    output_dir: Path | None = None,
    force_prepare: bool = False,
    print_results: bool = True,
) -> RunRecord:
    import torch

    from ..problems.gemm_hopper.problem import check_correctness, make_inputs, reference

    selected_specs = list(specs)
    prepare_kernels(selected_specs, force_prepare=force_prepare)
    loaded = {spec.name: load_target(spec.target) for spec in selected_specs}
    results: list[KernelResult] = []

    if print_results:
        print(
            f"Running {len(selected_specs)} kernels on {config.gpu} "
            f"for {len(config.shapes)} shape(s)"
        )
    for repetition in range(config.repetitions):
        for shape_idx, shape in enumerate(config.shapes):
            inputs = make_inputs(
                shape,
                dtype=config.dtype,
                seed=config.seed + shape_idx + (repetition * len(config.shapes)),
            )
            expected = reference(inputs) if config.check_correctness else None
            torch.cuda.synchronize()

            for spec in selected_specs:
                fn = loaded[spec.name]
                actual = fn(inputs)
                torch.cuda.synchronize()
                correctness = (
                    check_correctness(actual, expected, dtype=config.dtype)
                    if expected is not None
                    else CorrectnessResult(passed=True)
                )
                if print_results and not correctness.passed:
                    print(f"{spec.name} correctness failed: {correctness.error}")

                def timed_call() -> object:
                    return fn(inputs)

                latency_ms = do_bench(
                    timed_call,
                    warmup_runs=config.warmup_runs,
                    bench_runs=config.bench_runs,
                )
                result = KernelResult(
                    problem=config.problem,
                    kernel=spec.name,
                    kind=spec.kind,
                    gpu=config.gpu,
                    shape=shape,
                    dtype=config.dtype,
                    stats=summarize_latency(
                        latency_ms,
                        shape=shape,
                        quantiles=config.quantiles,
                    ),
                    correctness=correctness,
                    repetition=repetition,
                    tags=spec.tags,
                )
                results.append(result)
                if print_results:
                    print_result(result)

    record = RunRecord(
        config=config,
        results=results,
        metadata=default_metadata(),
    )
    if print_results:
        print_comparison(results)
    if output_dir is not None:
        write_artifacts(record, output_dir)
        if print_results:
            print(f"\nWrote artifacts to {output_dir}")
    return record


def prepare_kernels(specs: Iterable[KernelSpec], *, force_prepare: bool = False) -> None:
    prepare_targets = {
        spec.metadata["prepare"]
        for spec in specs
        if "prepare" in spec.metadata
    }
    for target in sorted(prepare_targets):
        prepare = load_target(target)
        prepare(force_prepare=force_prepare)


def config_from_parts(
    *,
    problem: str,
    gpu: str,
    kernels: Iterable[str],
    shapes: Iterable[ShapeSpec],
    dtype: str,
    warmup_runs: int,
    bench_runs: int,
    quantiles: Iterable[float],
    repetitions: int,
    seed: int,
    check_correctness: bool,
    benchmark_ref: bool,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        problem=problem,
        gpu=gpu,
        kernels=tuple(kernels),
        shapes=tuple(shapes),
        dtype=dtype,
        warmup_runs=warmup_runs,
        bench_runs=bench_runs,
        quantiles=tuple(quantiles),
        repetitions=repetitions,
        seed=seed,
        check_correctness=check_correctness,
        benchmark_ref=benchmark_ref,
    )
