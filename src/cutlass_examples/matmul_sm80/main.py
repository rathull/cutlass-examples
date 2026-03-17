from pathlib import Path

import modal

from ..common.modal_utils import REMOTE_PACKAGE_DIR, build_cuda_image
from ..common.utils import GPU_TO_ARCH, parse_bool, parse_quantiles, parse_shape
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
CUDA_VERSION = "13.1.1"
CUDA_FLAVOR = "devel"  # includes full CUDA toolkit
OPERATING_SYSTEM = "ubuntu24.04"

LOCAL_BENCH_DIR = Path(__file__).resolve().parent
LOCAL_COMMON_DIR = LOCAL_BENCH_DIR.parent / "common"
REMOTE_BENCH_DIR = REMOTE_PACKAGE_DIR / "matmul_sm80"
REMOTE_COMMON_DIR = REMOTE_PACKAGE_DIR / "common"

image = build_cuda_image(
    cuda_version=CUDA_VERSION,
    flavor=CUDA_FLAVOR,
    operating_system=OPERATING_SYSTEM,
    local_mounts=[
        (LOCAL_BENCH_DIR, REMOTE_BENCH_DIR),
        (LOCAL_COMMON_DIR, REMOTE_COMMON_DIR),
    ],
    extra_pip_packages=("numpy", "ninja"),
)
app = modal.App(name="sm80-matmul", image=image)


def get_module():
    import torch
    import torch.utils.cpp_extension
    
    print(f"{torch.__version__ = }")
    print(f"{torch.version.cuda = }")
    
    arch = GPU_TO_ARCH[GPU]
    sources = [REMOTE_BENCH_DIR / "matmul.cpp"]  # bindings
    sources.extend(REMOTE_BENCH_DIR / source_name for source_name in KERNEL_SOURCES.values())  # kernels
    
    torch.utils.cpp_extension.load(
        name="matmul_sm80_module",
        sources=[str(source) for source in sources],
        extra_include_paths=[str(REMOTE_COMMON_DIR)],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",   # line numbers for device code
            "-Xptxas=-v",  # print register, smem, and constant memory usage
            f"-gencode=arch=compute_{arch},code=sm_{arch}",  # compile for Ampere
            # "-arch=sm_80",
        ],
        extra_ldflags=[
            # "-lcuda",    # Link against CUDA Driver API library, for TMA on >=sm100
            # "-lcudart",  # Functions that start with cuda, e.g. cudaMalloc
        ],  
        verbose=True,
        is_python_module=False,
    )
    return torch.ops.my_matmul


@app.function(gpu=GPU)
def profile(shape: str):
    ...

@app.function(gpu=GPU)
def benchmark(
    shape: str = DEFAULT_SHAPE,
    versions: str = DEFAULT_VERSIONS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    bench_runs: int = DEFAULT_BENCH_RUNS,
    quantiles: str = DEFAULT_QUANTILES,
    benchmark_ref: bool = True,
    check_correctness: bool = True,
):
    import torch
    
    selected_versions = parse_versions(versions)
    quantile_values = parse_quantiles(quantiles)
    my_matmul = get_module()
    
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
                torch.testing.assert_close(out, output_ref)
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
        )
    else:
        raise NotImplementedError(f"Action not implemented: {action}")
