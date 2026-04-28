from __future__ import annotations

from dataclasses import dataclass

from ...common.benchmarking import CorrectnessResult, ShapeSpec

import torch

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.0


@dataclass(frozen=True)
class GemmInputs:
    a: torch.Tensor
    b: torch.Tensor
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA


@dataclass(frozen=True)
class GemmOutputs:
    c: torch.Tensor


def make_inputs(shape: ShapeSpec, *, dtype: str, seed: int) -> GemmInputs:
    torch_dtype = _torch_dtype(dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn(shape.m, shape.k, dtype=torch_dtype, device="cuda", generator=generator)
    # NT mode: A is MxK, B is stored as NxK and consumed as B.T.
    b = torch.randn(shape.n, shape.k, dtype=torch_dtype, device="cuda", generator=generator)
    return GemmInputs(a=a, b=b)


def make_outputs(shape: ShapeSpec, *, dtype: str) -> GemmOutputs:
    torch_dtype = _torch_dtype(dtype)
    c = torch.empty(shape.m, shape.n, dtype=torch_dtype, device="cuda")
    return GemmOutputs(c=c)


def reference(inputs: GemmInputs, outputs: GemmOutputs | None = None) -> torch.Tensor:
    c = torch.empty(inputs.a.shape[0], inputs.b.shape[0], dtype=inputs.a.dtype, device=inputs.a.device)
    if outputs is not None:
        c.copy_(outputs.c)
    return torch.addmm(c, inputs.a, inputs.b.T, beta=inputs.beta, alpha=inputs.alpha)


def check_correctness(actual, expected, *, dtype: str) -> CorrectnessResult:
    import torch

    atol, rtol = _tolerances(dtype)
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected.float().abs().clamp_min(1e-5)).max().item())
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except Exception as exc:
        return CorrectnessResult(
            passed=False,
            max_abs=max_abs,
            max_rel=max_rel,
            error=str(exc),
        )
    return CorrectnessResult(passed=True, max_abs=max_abs, max_rel=max_rel)


def _torch_dtype(dtype: str):
    import torch

    match dtype:
        case "bf16":
            return torch.bfloat16
        case "fp16":
            return torch.float16
        case "fp32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported GEMM dtype: {dtype}")


def _tolerances(dtype: str) -> tuple[float, float]:
    match dtype:
        case "bf16":
            return 1e-3, 1.6e-2
        case "fp16":
            return 1e-3, 1e-2
        case "fp32":
            return 1e-4, 1e-4
        case _:
            raise ValueError(f"Unsupported GEMM dtype: {dtype}")
