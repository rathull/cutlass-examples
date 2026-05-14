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
from .kernel_registry import (
    ALLOW_FAILURE_METADATA_KEY,
    KERNEL_PARAMS_METADATA_KEY,
    PREPARE_FACTORY_METADATA_KEY,
    RUNNER_FACTORY_METADATA_KEY,
)


def run_gemm_benchmark(
    *,
    config: BenchmarkConfig,
    specs: Iterable[KernelSpec],
    output_dir: Path | None = None,
    force_prepare: bool = False,
    print_results: bool = True,
) -> RunRecord:
    import torch

    from ..problems.gemm_hopper_bf16.problem import check_correctness, make_inputs, make_outputs, reference

    selected_specs = list(specs)
    prepare_errors = prepare_kernels(selected_specs, force_prepare=force_prepare)
    loaded = {spec.name: _load_runner(spec) for spec in selected_specs}
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
            torch.cuda.synchronize()

            for spec in selected_specs:
                if spec.name in prepare_errors:
                    result = _failed_result(
                        spec=spec,
                        config=config,
                        shape=shape,
                        repetition=repetition,
                        error=f"prepare failed: {prepare_errors[spec.name]}",
                    )
                    results.append(result)
                    if print_results:
                        print(f"{spec.name} prepare failed: {prepare_errors[spec.name]}")
                    continue

                fn = loaded[spec.name]
                correctness_outputs = make_outputs(shape, dtype=config.dtype)
                expected = (
                    reference(inputs, correctness_outputs)
                    if config.check_correctness
                    else None
                )
                try:
                    actual = fn(inputs, correctness_outputs)
                    if actual is None:
                        actual = correctness_outputs.c
                    torch.cuda.synchronize()
                except Exception as exc:
                    if not _allows_failure(spec):
                        raise
                    result = _failed_result(
                        spec=spec,
                        config=config,
                        shape=shape,
                        repetition=repetition,
                        error=f"run failed: {exc}",
                    )
                    results.append(result)
                    if print_results:
                        print(f"{spec.name} run failed: {exc}")
                    continue
                correctness = (
                    check_correctness(actual, expected, dtype=config.dtype)
                    if expected is not None
                    else CorrectnessResult(passed=True)
                )
                if print_results and not correctness.passed:
                    print(f"{spec.name} correctness failed: {correctness.error}")

                timed_outputs = make_outputs(shape, dtype=config.dtype)

                def timed_call() -> object:
                    result = fn(inputs, timed_outputs)
                    return timed_outputs.c if result is None else result

                try:
                    latency_ms = do_bench(
                        timed_call,
                        warmup_runs=config.warmup_runs,
                        bench_runs=config.bench_runs,
                    )
                except Exception as exc:
                    if not _allows_failure(spec):
                        raise
                    result = _failed_result(
                        spec=spec,
                        config=config,
                        shape=shape,
                        repetition=repetition,
                        error=f"benchmark failed: {exc}",
                    )
                    results.append(result)
                    if print_results:
                        print(f"{spec.name} benchmark failed: {exc}")
                    continue
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
                    kernel_path=spec.source or spec.metadata.get("path"),
                )
                _set_kernel_params(result, spec)
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


def prepare_kernels(specs: Iterable[KernelSpec], *, force_prepare: bool = False) -> dict[str, str]:
    errors: dict[str, str] = {}
    prepare_targets = sorted({
        spec.metadata["prepare"]
        for spec in specs
        if "prepare" in spec.metadata
    })
    for target in prepare_targets:
        prepare = load_target(target)
        prepare(force_prepare=force_prepare)

    for spec in sorted(specs, key=lambda item: item.name):
        prepare_factory_target = spec.metadata.get(PREPARE_FACTORY_METADATA_KEY)
        if prepare_factory_target is None:
            continue
        prepare_factory = load_target(prepare_factory_target)
        try:
            prepare_factory(spec, force_prepare=force_prepare)
        except Exception as exc:
            if _allows_failure(spec):
                errors[spec.name] = str(exc)
                continue
            raise
    return errors


def _load_runner(spec: KernelSpec):
    runner_factory_target = spec.metadata.get(RUNNER_FACTORY_METADATA_KEY)
    if runner_factory_target is not None:
        runner_factory = load_target(runner_factory_target)
        return runner_factory(spec)
    return load_target(_require_target(spec))


def _set_kernel_params(result: KernelResult, spec: KernelSpec) -> None:
    kernel_params = spec.metadata.get(KERNEL_PARAMS_METADATA_KEY)
    if kernel_params is not None:
        object.__setattr__(result, "kernel_params", kernel_params)


def _allows_failure(spec: KernelSpec) -> bool:
    return spec.metadata.get(ALLOW_FAILURE_METADATA_KEY) == "true"


def _failed_result(
    *,
    spec: KernelSpec,
    config: BenchmarkConfig,
    shape: ShapeSpec,
    repetition: int,
    error: str,
) -> KernelResult:
    result = KernelResult(
        problem=config.problem,
        kernel=spec.name,
        kind=spec.kind,
        gpu=config.gpu,
        shape=shape,
        dtype=config.dtype,
        stats=summarize_latency([float("nan")], shape=shape, quantiles=config.quantiles),
        correctness=CorrectnessResult(passed=False, error=error),
        repetition=repetition,
        kernel_path=spec.source or spec.metadata.get("path"),
    )
    object.__setattr__(result.stats, "samples", 0)
    _set_kernel_params(result, spec)
    return result


def _require_target(spec: KernelSpec) -> str:
    if spec.target is None:
        raise ValueError(f"Kernel {spec.name!r} does not have a callable target.")
    return spec.target


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
