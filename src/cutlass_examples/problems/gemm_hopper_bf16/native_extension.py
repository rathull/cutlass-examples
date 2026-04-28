from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

BASE_MODULE_NAME = "gemm_hopper_bf16_native"
REMOTE_PROBLEM_DIR = Path("/opt/cutlass_examples/problems/gemm_hopper_bf16")
REMOTE_KERNEL_DIR = REMOTE_PROBLEM_DIR / "kernels" / "native_cuda"
REMOTE_CUTLASS_DIR = Path("/opt/cutlass")
BUILD_ROOT = Path("/cache/build") / BASE_MODULE_NAME
_ops_cache: dict[tuple[str, str], object] = {}


class NativeKernel(NamedTuple):
    ops: object
    build_log: str


def load_kernel(
    name: str,
    *,
    force_prepare: bool = False,
    return_build_log: bool = False,
    source: str | Path | None = None,
    extra_sources: Iterable[str | Path] = (),
    extra_cuda_cflags: Iterable[str] = (),
    extra_ldflags: Iterable[str] = (),
    extra_include_paths: Iterable[str | Path] = (),
) -> object | NativeKernel:
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

    source_path = _resolve_source(name, source)
    sources = [source_path, *[_resolve_path(path) for path in extra_sources]]
    module_name = _module_name(name)
    torch_namespace = _torch_namespace(name)
    build_root = BUILD_ROOT / name
    binding_source = build_root / "generated_bindings.cpp"
    binding_source.parent.mkdir(parents=True, exist_ok=True)
    binding_source.write_text(_render_bindings(name=name, torch_namespace=torch_namespace))

    cuda_cflags = [
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        *extra_cuda_cflags,
    ]
    ldflags = list(extra_ldflags)
    include_paths = _default_include_paths() + [_resolve_path(path) for path in extra_include_paths]
    cache_cuda_cflags = [
        *cuda_cflags,
        *[f"-I{path}" for path in include_paths],
    ]
    source_hash = compute_source_hash(
        sources=[binding_source, *sources],
        include_files=[path for path in include_paths if path.is_file()],
        cuda_cflags=cache_cuda_cflags,
        ldflags=ldflags,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cutlass_version=CUTLASS_VERSION,
    )
    cache_key = (name, source_hash)
    if cache_key in _ops_cache and not force_prepare and not return_build_log:
        return _ops_cache[cache_key]

    cached_so = get_cached_so(build_root, source_hash, module_name)
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
            name=module_name,
            sources=[str(local_binding_source), *[str(path) for path in sources]],
            build_directory=str(cached_so.parent),
            extra_cuda_cflags=cuda_cflags,
            extra_include_paths=[str(path) for path in include_paths if path.exists()],
            extra_ldflags=ldflags,
            verbose=True,
            is_python_module=False,
        )
        print(f"Native compilation took {time.perf_counter() - t0:.1f}s")

    if return_build_log:
        build_log.write(_capture_stdout_stderr(compile_or_load))
    else:
        compile_or_load()

    ops = getattr(torch.ops, torch_namespace)
    if not return_build_log:
        _ops_cache[cache_key] = ops
    if return_build_log:
        return NativeKernel(ops=ops, build_log=build_log.getvalue())
    return ops


def _resolve_source(name: str, source: str | Path | None) -> Path:
    if source is not None:
        return _resolve_path(source)
    source_path = REMOTE_KERNEL_DIR / f"{name}.cu"
    if not source_path.exists():
        raise ValueError(f"Native kernel source not found: {source_path}")
    return source_path


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return REMOTE_PROBLEM_DIR / raw


def _default_include_paths() -> list[Path]:
    return [
        REMOTE_PROBLEM_DIR,
        REMOTE_CUTLASS_DIR / "include",
        REMOTE_CUTLASS_DIR / "tools" / "util" / "include",
        REMOTE_CUTLASS_DIR / "examples" / "common",
    ]


def _module_name(name: str) -> str:
    return f"{BASE_MODULE_NAME}_{_sanitize_identifier(name)}"


def _torch_namespace(name: str) -> str:
    return f"{BASE_MODULE_NAME}_{_sanitize_identifier(name)}"


def _sanitize_identifier(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _render_bindings(*, name: str, torch_namespace: str) -> str:
    declaration = (
        f"extern \"C\" void {name}(const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*, "
        "int, int, int, float, float);"
    )
    wrapper = _render_wrapper(name)
    return f"""#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <torch/library.h>

{declaration}

{wrapper}

TORCH_LIBRARY({torch_namespace}, m) {{
    m.def("{name}(Tensor A, Tensor B, Tensor C, float alpha, float beta) -> Tensor");
    m.impl("{name}", &{name}_torch);
}}
"""


def _render_wrapper(name: str) -> str:
    return f"""at::Tensor {name}_torch(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    double alpha,
    double beta
) {{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    {name}(
        reinterpret_cast<const nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(C.data_ptr()),
        M, N, K,
        static_cast<float>(alpha),
        static_cast<float>(beta)
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
