from __future__ import annotations


def run(inputs):
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401

    # Smoke placeholder: validates the Gluon runtime while real kernels are added.
    import torch

    return torch.matmul(inputs.a, inputs.b)
