from __future__ import annotations


def prepare_gluon(*, force_prepare: bool = False) -> None:
    _ = force_prepare
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401


def smoke_matmul(inputs):
    prepare_gluon()

    # Starter placeholder: this validates the Gluon runtime is installed while
    # leaving room to replace the body with a real Gluon kernel.
    import torch

    return torch.matmul(inputs.a, inputs.b)
