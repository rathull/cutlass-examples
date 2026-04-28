from __future__ import annotations

from typing import cast

from ... import native_extension

KERNEL_NAME = "cuda_v1_smem_tiling"
SOURCE = "kernels/native_cuda/cuda_v1_smem_tiling.cu"
EXTRA_CUDA_CFLAGS: tuple[str, ...] = ()
EXTRA_LDFLAGS: tuple[str, ...] = ()
EXTRA_INCLUDE_PATHS: tuple[str, ...] = ()
kernel = None


def prepare(*, force_prepare: bool = False) -> None:
    global kernel

    ops = native_extension.load_kernel(
        KERNEL_NAME,
        force_prepare=force_prepare,
        source=SOURCE,
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_ldflags=EXTRA_LDFLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
    )
    kernel = getattr(ops, KERNEL_NAME)


def inspect_ptxas(*, force_prepare: bool = False) -> str:
    compiled = native_extension.load_kernel(
        KERNEL_NAME,
        force_prepare=force_prepare,
        return_build_log=True,
        source=SOURCE,
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_ldflags=EXTRA_LDFLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
    )
    return cast(native_extension.NativeKernel, compiled).build_log


def run(inputs, outputs):
    # Ensure prepare() is called before run()
    return kernel(inputs.a, inputs.b, outputs.c, inputs.alpha, inputs.beta)
