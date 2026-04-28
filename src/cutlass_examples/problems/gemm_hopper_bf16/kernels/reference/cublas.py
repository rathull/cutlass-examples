from __future__ import annotations


def run(inputs, outputs):
    import torch

    return torch.addmm(
        outputs.c,
        inputs.a,
        inputs.b.T,
        beta=inputs.beta,
        alpha=inputs.alpha,
        out=outputs.c,
    )
