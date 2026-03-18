from pathlib import Path

import modal

from ..common.modal_utils import REMOTE_PACKAGE_DIR, build_cuda_image
from ..common.utils import CUTLASS_VERSION, GPU_TO_ARCH, parse_bool, parse_quantiles, parse_shape
from .bench_utils import (
    DEFAULT_BENCH_RUNS,
    DEFAULT_QUANTILES,
    DEFAULT_WARMUP_RUNS,
    do_bench,
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


def get_module(*, force_recompile: bool = False):
    import time

    import torch
    import torch.utils.cpp_extension

    from ..common.build_cache import compute_source_hash, get_cached_so

    print(f"{torch.__version__ = }")
    print(f"{torch.version.cuda = }")
    
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

    if not force_recompile and cached_so.exists():
        print(f"Build cache hit [{source_hash}], skipping compilation")
        torch.ops.load_library(str(cached_so))
    else:
        reason = "force recompile" if force_recompile else "cache miss"
        print(f"Compiling ({reason}) [{source_hash}]")

        cached_so.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        torch.utils.cpp_extension.load(
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

        build_cache_volume.commit()

    return torch.ops.my_matmul


@app.function(gpu=GPU, volumes={str(BUILD_CACHE_DIR): build_cache_volume})
def profile(shape: str):
    ...

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
):
    import torch
    
    selected_versions = parse_versions(versions)
    quantile_values = parse_quantiles(quantiles)
    my_matmul = get_module(force_recompile=force_recompile)
    
    M, N, K = parse_shape(shape)
    print(
        f"{M = }, {N = }, {K = }, "
        f"selected_versions = {selected_versions}, "
        f"{warmup_runs = }, {bench_runs = }, {quantile_values = }, "
        f"{benchmark_ref = }, {check_correctness = }"
    )
    
    # To use NT kernel path, B will be in K-major order in memory but N-major conceptually
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T

    def bench_and_print(op, name: str):
        latency_ms_list = do_bench(lambda: op(A, B), warmup_runs=warmup_runs, bench_runs=bench_runs)
        print_summary(name, latency_ms_list, m=M, n=N, k=K, quantiles=quantile_values)
    
    print("\n" + "=" * 40 + " RESULTS " + "=" * 40 + "\n")
    output_ref = torch.matmul(A, B) if (benchmark_ref or check_correctness) else None
    if benchmark_ref:
        bench_and_print(torch.matmul, "cuBLAS (ref)")
    
    for version in selected_versions:
        f = getattr(my_matmul, f"matmul_{version}")
        if check_correctness:
            out = f(A, B)
            torch.cuda.synchronize()
            try:
                torch.testing.assert_close(out, output_ref, atol=1e-3, rtol=1.6e-2)
            except Exception:
                print(output_ref)
                print(out)
                raise
        bench_and_print(f, version)

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
        )
    else:
        raise NotImplementedError(f"Action not implemented: {action}")
