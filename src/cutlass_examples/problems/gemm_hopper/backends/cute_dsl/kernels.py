from __future__ import annotations


def prepare_cute_dsl(*, force_prepare: bool = False) -> None:
    _ = force_prepare
    import cutlass  # noqa: F401


def smoke_matmul(inputs):
    prepare_cute_dsl()

    # Starter placeholder: this validates CuTe DSL is installed while the
    # Hopper GEMM skeleton is filled in with a real CuTe DSL kernel.
    import torch

    return torch.matmul(inputs.a, inputs.b)
