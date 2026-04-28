from __future__ import annotations


def run(inputs):
    import cutlass  # noqa: F401

    # Smoke placeholder: validates CuTe DSL while real kernels are added.
    import torch

    return torch.matmul(inputs.a, inputs.b)
