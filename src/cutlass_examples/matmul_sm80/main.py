from pathlib import Path

import modal

from ..common.modal_utils import REMOTE_PACKAGE_DIR, build_cuda_image
from ..common.utils import (
    CUTLASS_VERSION,
    GPU_TO_ARCH,
    parse_bool,
    parse_quantiles,
    parse_shapes,
)
from .bench_utils import (
    BenchmarkStats,
    DEFAULT_BENCH_RUNS,
    DEFAULT_QUANTILES,
    DEFAULT_WARMUP_RUNS,
    do_bench,
    print_comparison,
    print_summary,
)
from .matmul_utils import (
    DEFAULT_SHAPE,
    DEFAULT_VERSIONS,
    KERNEL_SOURCES,
    parse_versions,
)

GPU = "A100"
CUDA_VERSION = "13.2.0"
CUDA_FLAVOR = "devel"  # includes full CUDA toolkit
OPERATING_SYSTEM = "ubuntu24.04"
MODULE_NAME = "matmul_sm80_module"

LOCAL_BENCH_DIR = Path(__file__).resolve().parent
LOCAL_COMMON_DIR = LOCAL_BENCH_DIR.parent / "common"
REMOTE_BENCH_DIR = REMOTE_PACKAGE_DIR / "matmul_sm80"
REMOTE_COMMON_DIR = REMOTE_PACKAGE_DIR / "common"
REMOTE_CUTLASS_PATH = Path("/opt/cutlass")
REMOTE_CUTLASS_INCLUDE = REMOTE_CUTLASS_PATH / "include"
BUILD_CACHE_DIR = Path("/cache/build")

image = build_cuda_image(
    cuda_version=CUDA_VERSION,
    flavor=CUDA_FLAVOR,
    operating_system=OPERATING_SYSTEM,
    local_mounts=[
        (LOCAL_BENCH_DIR, REMOTE_BENCH_DIR),
        (LOCAL_COMMON_DIR, REMOTE_COMMON_DIR),
    ],
    extra_pip_packages=("numpy", "ninja"),
    extra_apt_packages=("git",),
    extra_commands=(
        f"git clone --depth 1 --branch v{CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git {REMOTE_CUTLASS_PATH}",
    ),
)
app = modal.App(name="sm80-matmul", image=image)
build_cache_volume = modal.Volume.from_name("sm80-build-cache", create_if_missing=True)


def get_module(*, force_recompile: bool = False, return_build_log: bool = False):
    import shutil
    import time

    import torch
    from torch.utils.cpp_extension import load as load_cpp_extension

    from ..common.build_cache import compute_source_hash, get_cached_so

    assert torch.version.cuda
    # TODO: configure properly so torch CUDA version 13.2
    # assert torch.version.cuda == CUDA_VERSION, f"torch.version.cuda ({torch.version.cuda}) != CUDA_VERSION ({CUDA_VERSION})"

    arch = GPU_TO_ARCH[GPU]
    sources = [REMOTE_BENCH_DIR / "matmul.cpp"]  # bindings
    sources.extend(REMOTE_BENCH_DIR / source_name for source_name in KERNEL_SOURCES.values())  # kernels

    include_files = sorted(REMOTE_COMMON_DIR.glob("*.h"))
    cuda_cflags = [
        "-O3",
        "-lineinfo",          # line numbers for device code
        "-Xptxas=-v",         # print register, smem, and constant memory usage
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    ldflags: list[str] = [
        # "-lcuda",    # Link against CUDA Driver API library, for TMA on >=sm100
        # "-lcudart",  # Functions that start with cuda, e.g. cudaMalloc
    ]

    source_hash = compute_source_hash(
        sources=sources,
        include_files=include_files,
        cuda_cflags=cuda_cflags,
        ldflags=ldflags,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cutlass_version=CUTLASS_VERSION,
    )
    cached_so = get_cached_so(BUILD_CACHE_DIR, source_hash, MODULE_NAME)

    build_log = ""

    def load_or_compile() -> None:
        if not force_recompile and cached_so.exists():
            print(f"Build cache hit [{source_hash}], skipping compilation")
            torch.ops.load_library(str(cached_so))
        else:
            reason = "force recompile" if force_recompile else "cache miss"
            print(f"Compiling ({reason}) [{source_hash}]")

            if force_recompile and cached_so.parent.exists():
                shutil.rmtree(cached_so.parent)
            cached_so.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            load_cpp_extension(
                name=MODULE_NAME,
                sources=[str(source) for source in sources],
                build_directory=str(cached_so.parent),
                extra_include_paths=[str(REMOTE_COMMON_DIR), str(REMOTE_CUTLASS_INCLUDE)],
                extra_cuda_cflags=cuda_cflags,
                extra_ldflags=ldflags,
                verbose=True,
                is_python_module=False,
            )
            elapsed = time.perf_counter() - t0
            print(f"Compilation took {elapsed:.1f}s")

            try:
                build_cache_volume.commit()
            except RuntimeError as exc:
                print(f"Skipping build cache volume commit: {exc}")

    if return_build_log:
        build_log = _capture_stdout_stderr(load_or_compile)
    else:
        load_or_compile()

    module = torch.ops.my_matmul
    if return_build_log:
        return module, build_log
    return module


def _capture_stdout_stderr(fn) -> str:
    import os
    import sys
    import tempfile

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as fp:
            os.dup2(fp.fileno(), 1)
            os.dup2(fp.fileno(), 2)
            try:
                fn()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_fd, 1)
                os.dup2(stderr_fd, 2)
            fp.seek(0)
            return fp.read().decode(errors="replace")
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)


@app.function(gpu=GPU, volumes={str(BUILD_CACHE_DIR): build_cache_volume})
def profile(shape: str = DEFAULT_SHAPE):
    raise NotImplementedError(f"Profiling is not implemented yet for shape={shape!r}")


def _print_runtime_info() -> None:
    import torch

    device = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"device = {device} (sm_{major}{minor})")
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")


def _make_inputs(m: int, n: int, k: int, *, seed: int):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=generator)
    # NT kernel path: B is K-major in memory but N-major conceptually.
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=generator).T
    return a, b


def _check_correctness(actual, expected, *, name: str) -> None:
    import torch

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-5)).max().item()
    try:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1.6e-2)
    except Exception as exc:
        raise AssertionError(
            f"{name} failed correctness check: max_abs={max_abs:.6g}, "
            f"max_rel={max_rel:.6g}"
        ) from exc
    print(
        f"{name:16s} correctness: ok "
        f"(max_abs={max_abs:.6g}, max_rel={max_rel:.6g})"
    )


@app.function(gpu=GPU, volumes={str(BUILD_CACHE_DIR): build_cache_volume})
def benchmark(
    shape: str = DEFAULT_SHAPE,
    versions: str = DEFAULT_VERSIONS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    quantiles: str = DEFAULT_QUANTILES,
    benchmark_ref: bool = True,
    check_correctness: bool = True,
    force_recompile: bool = False,
    seed: int = 0,
):
    import torch

    selected_versions = parse_versions(versions)
    quantile_values = parse_quantiles(quantiles)
    my_matmul = get_module(force_recompile=force_recompile)
    shapes = parse_shapes(shape)

    _print_runtime_info()
    print(
        f"shapes = {shapes}, "
        f"selected_versions = {selected_versions}, "
        f"{warmup_runs = }, {bench_runs = }, {quantile_values = }, "
        f"{benchmark_ref = }, {check_correctness = }, {seed = }"
    )

    for shape_idx, (m, n, k) in enumerate(shapes):
        print("\n" + "=" * 34 + f" SHAPE {m}x{n}x{k} " + "=" * 34 + "\n")
        a, b = _make_inputs(m, n, k, seed=seed + shape_idx)

        def bench_and_print(op, name: str) -> BenchmarkStats:
            latency_ms_list = do_bench(
                lambda: op(a, b),
                warmup_runs=warmup_runs,
                bench_runs=bench_runs,
            )
            return print_summary(
                name,
                latency_ms_list,
                m=m,
                n=n,
                k=k,
                quantiles=quantile_values,
            )

        output_ref = torch.matmul(a, b) if (benchmark_ref or check_correctness) else None
        results: list[BenchmarkStats] = []
        if benchmark_ref:
            results.append(bench_and_print(torch.matmul, "cuBLAS (ref)"))

        for version in selected_versions:
            kernel = getattr(my_matmul, f"matmul_{version}")
            if check_correctness:
                if output_ref is None:
                    raise RuntimeError(
                        "Internal error: correctness requested without a reference output."
                    )
                out = kernel(a, b)
                torch.cuda.synchronize()
                _check_correctness(out, output_ref, name=version)
            results.append(bench_and_print(kernel, version))

        print_comparison(results, baseline_name="cuBLAS (ref)" if benchmark_ref else None)


@app.local_entrypoint()
def main(
    action: str = "benchmark",
    shape: str = DEFAULT_SHAPE,
    versions: str = DEFAULT_VERSIONS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    quantiles: str = DEFAULT_QUANTILES,
    benchmark_ref: str = "true",
    check_correctness: str = "true",
    force_recompile: str = "false",
    seed: int = 0,
):
    if action == "benchmark":
        benchmark.remote(
            shape=shape,
            versions=versions,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
            quantiles=quantiles,
            benchmark_ref=parse_bool(benchmark_ref),
            check_correctness=parse_bool(check_correctness),
            force_recompile=parse_bool(force_recompile),
            seed=seed,
        )
    elif action == "profile":
        profile.remote(shape=shape)
    else:
        raise NotImplementedError(f"Action not implemented: {action}")
