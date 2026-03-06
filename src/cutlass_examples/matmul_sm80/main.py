# main.py

from pathlib import Path
from typing import Callable

import modal

gpu = "A100"

cuda_version = "13.1.1"
flavor = "devel"  # includes full CUDA toolkit
operating_system = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"
HF_CACHE_PATH = "/cache"

GPU_TO_ARCH = {
    "A100": "80",
    "A100-40GB": "80",
    "A100-80GB": "80",
    "H100": "90",
    "H200": "90",
    "B200": "100",
}

def get_tflops(M, N, K, latency_ms):
    return 2 * M * N * K * 1e-9 / latency_ms

LOCAL_BENCH_DIR = Path(__file__).resolve().parent     # src/cutlass_examples/matmul_sm80
LOCAL_COMMON_DIR = LOCAL_BENCH_DIR.parent / "common"  # src/cutlass_examples/common
REMOTE_BENCH_DIR = Path("/opt/cutlass_examples/matmul_sm80")
REMOTE_COMMON_DIR = Path("/opt/cutlass_examples/common")

KERNEL_SOURCES = {
    "v0": "matmul_v0.cu",
    "v1": "matmul_v1.cu",
}

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])  # Remove verblose logging by base image on entry
    .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install(
        "ninja",
        "triton>=3.6.0",
    )
    .add_local_dir(str(LOCAL_BENCH_DIR), remote_path=str(REMOTE_BENCH_DIR))
    .add_local_dir(str(LOCAL_COMMON_DIR), remote_path=str(REMOTE_COMMON_DIR))
)
app = modal.App(name="sm80-matmul", image=image)

def parse_versions(versions: str) -> list[str]:
    selected_versions = [version.strip() for version in versions.split(",") if version.strip()]
    if not selected_versions:
        raise ValueError("Expected at least one kernel version, e.g. 'v0' or 'v0,v1'.")

    unknown_versions = [version for version in selected_versions if version not in KERNEL_SOURCES]
    if unknown_versions:
        raise ValueError(
            f"Unsupported kernel versions: {unknown_versions}. "
            f"Supported versions: {sorted(KERNEL_SOURCES)}"
        )

    return selected_versions

def get_module():
    import torch
    import torch.utils.cpp_extension
    
    print(f"{torch.__version__ = }")
    print(f"{torch.version.cuda = }")
    
    arch = GPU_TO_ARCH[gpu]
    sources = [REMOTE_BENCH_DIR / "matmul.cpp"]  # bindings
    sources.extend(REMOTE_BENCH_DIR / source_name for source_name in KERNEL_SOURCES.values())  # kernels
    
    torch.utils.cpp_extension.load(
        name="module",
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

# TODO
@app.function(gpu=gpu)
def profile(shape: str):
    ...

@app.function(gpu=gpu)
def benchmark(shape: str, versions: str = "v0,v1"):
    import torch
    from triton.testing import do_bench
    
    selected_versions = parse_versions(versions)
    my_matmul = get_module()
    
    M, N, K = map(int, shape.split(","))
    print(f"{M = }, {N = }, {K = }, selected_versions = {selected_versions}")
    
    # To use NT kernel path, B will be in K-major order in memory but N-major conceptually
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    
    def bench_and_print(f: Callable, name: str):
        # Sleep to stabilize thermal
        torch.cuda.synchronize()
        
        # TODO: use mode "all"
        latency_ms = do_bench(lambda: f(A, B), warmup=10, rep=100, return_mode="median")
        tflops = get_tflops(M, N, K, latency_ms)
        print(f"{name:12s}:\t{latency_ms:.4f} ms\t{tflops:.3f} TFLOPS")
    
    output_ref = torch.matmul(A, B)
    bench_and_print(torch.matmul, "cuBLAS (ref)")
    
    for version in selected_versions:
        f = getattr(my_matmul, f"matmul_{version}")
        out = f(A, B)
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out, output_ref)
        except:
            print(output_ref)
            print(out)
            raise
        bench_and_print(f, version)

@app.local_entrypoint()
def main(action: str, shape: str = "4096,4096,4096", versions: str = "v0,v1"):
    if action == "benchmark":
        benchmark.remote(shape, versions)
    else:
        raise NotImplementedError(f"Action not implemented: {action}")
