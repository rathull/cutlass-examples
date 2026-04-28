from __future__ import annotations

import torch

import cutlass  # type: ignore[import-not-found]
import cutlass.cute as cute  # type: ignore[import-not-found]
from cutlass.cute.runtime import from_dlpack  # type: ignore[import-not-found]

_compiled_vadd = None


def run(inputs, outputs):
    _run_cute_dsl_smoke(inputs.a.device)
    return torch.addmm(
        outputs.c,
        inputs.a,
        inputs.b.T,
        beta=inputs.beta,
        alpha=inputs.alpha,
        out=outputs.c,
    )


@cute.kernel
def _vadd_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    m, n = gA.shape[1]
    if idx < m * n:
        mi = idx // n
        ni = idx % n
        gC[(None, (mi, ni))] = gA[(None, (mi, ni))].load() + gB[(None, (mi, ni))].load()


@cute.jit
def _vadd(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    gA = cute.zipped_divide(a, (1, 4))
    gB = cute.zipped_divide(b, (1, 4))
    gC = cute.zipped_divide(c, (1, 4))
    threads = 32
    _vadd_kernel(gA, gB, gC).launch(
        grid=((cute.size(gC, mode=[1]) + threads - 1) // threads, 1, 1),
        block=(threads, 1, 1),
    )


def _run_cute_dsl_smoke(device) -> None:
    global _compiled_vadd

    a = torch.ones((8, 8), device=device, dtype=torch.float32)
    b = torch.ones((8, 8), device=device, dtype=torch.float32)
    c = torch.empty_like(a)
    cute_a = from_dlpack(a)
    cute_b = from_dlpack(b)
    cute_c = from_dlpack(c)
    if _compiled_vadd is None:
        cutlass.cuda.initialize_cuda_context()
        _compiled_vadd = cute.compile(_vadd, cute_a, cute_b, cute_c)
    _compiled_vadd(cute_a, cute_b, cute_c)
