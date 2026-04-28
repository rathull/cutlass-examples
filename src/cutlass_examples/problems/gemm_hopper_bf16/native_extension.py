from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

MODULE_NAME = "gemm_hopper_bf16_native"
TORCH_NAMESPACE = "gemm_hopper_bf16_native"
REMOTE_PROBLEM_DIR = Path("/opt/cutlass_examples/problems/gemm_hopper_bf16")
REMOTE_KERNEL_DIR = REMOTE_PROBLEM_DIR / "kernels" / "native_cuda"
BUILD_ROOT = Path("/cache/build") / MODULE_NAME
_ops_cache = None


def prepare(*, force_prepare: bool = False) -> None:
    _get_ops(force_prepare=force_prepare)


def inspect_ptxas(*, force_prepare: bool = False) -> str:
    _, build_log = _get_ops(force_prepare=force_prepare, return_build_log=True)
    return build_log


def __getattr__(name: str) -> Callable[[object], object]:
    def _run(inputs):
        ops = _get_ops()
        return getattr(ops, name)(inputs.a, inputs.b)

    return _run


def _get_ops(*, force_prepare: bool = False, return_build_log: bool = False):
    global _ops_cache

    import io
    import shutil
    import time

    import torch
    from torch.utils.cpp_extension import load as load_cpp_extension

    from ...common.build_cache import compute_source_hash, get_cached_so
    from ...common.utils import CUTLASS_VERSION, GPU_TO_ARCH

    assert torch.version.cuda
    gpu_name = torch.cuda.get_device_name()
    arch = "90" if "H100" in gpu_name or "H200" in gpu_name else GPU_TO_ARCH.get("B200", "100")

    cuda_sources = sorted(REMOTE_KERNEL_DIR.glob("*.cu"))
    if not cuda_sources:
        raise ValueError(f"No native CUDA kernels found in {REMOTE_KERNEL_DIR}")

    if _ops_cache is not None and not force_prepare and not return_build_log:
        return _ops_cache

    binding_source = BUILD_ROOT / "generated_bindings.cpp"
    binding_source.parent.mkdir(parents=True, exist_ok=True)
    binding_source.write_text(_render_bindings([source.stem for source in cuda_sources]))

    cuda_cflags = [
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    ldflags: list[str] = []
    sources = [binding_source, *cuda_sources]
    source_hash = compute_source_hash(
        sources=sources,
        include_files=[],
        cuda_cflags=cuda_cflags,
        ldflags=ldflags,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cutlass_version=CUTLASS_VERSION,
    )
    cached_so = get_cached_so(BUILD_ROOT, source_hash, MODULE_NAME)
    build_log = io.StringIO()

    def compile_or_load() -> None:
        if not force_prepare and cached_so.exists():
            print(f"Native build cache hit [{source_hash}], loading {cached_so}")
            torch.ops.load_library(str(cached_so))
            return

        if force_prepare and cached_so.parent.exists():
            shutil.rmtree(cached_so.parent)
        cached_so.parent.mkdir(parents=True, exist_ok=True)
        local_binding_source = cached_so.parent / "generated_bindings.cpp"
        local_binding_source.write_text(binding_source.read_text())
        t0 = time.perf_counter()
        load_cpp_extension(
            name=MODULE_NAME,
            sources=[str(local_binding_source), *[str(source) for source in cuda_sources]],
            build_directory=str(cached_so.parent),
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=ldflags,
            verbose=True,
            is_python_module=False,
        )
        print(f"Native compilation took {time.perf_counter() - t0:.1f}s")

    if return_build_log:
        build_log.write(_capture_stdout_stderr(compile_or_load))
    else:
        compile_or_load()

    ops = getattr(torch.ops, TORCH_NAMESPACE)
    if not return_build_log:
        _ops_cache = ops
    if return_build_log:
        return ops, build_log.getvalue()
    return ops


def _render_bindings(kernel_names: list[str]) -> str:
    declarations = "\n".join(
        f"extern \"C\" void {name}(const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*, "
        "int, int, int, long long, long long, long long, long long);"
        for name in kernel_names
    )
    wrappers = "\n\n".join(_render_wrapper(name) for name in kernel_names)
    registrations = "\n".join(
        f'    m.def("{name}(Tensor A, Tensor B) -> Tensor"); m.impl("{name}", &{name}_torch);'
        for name in kernel_names
    )
    return f"""#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <torch/library.h>

{declarations}

{wrappers}

TORCH_LIBRARY({TORCH_NAMESPACE}, m) {{
{registrations}
}}
"""


def _render_wrapper(name: str) -> str:
    return f"""at::Tensor {name}_torch(const at::Tensor& A, const at::Tensor& B) {{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({{M, N}}, A.options());
    {name}(
        reinterpret_cast<const nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(C.data_ptr()),
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1)
    );
    return C;
}}"""


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
