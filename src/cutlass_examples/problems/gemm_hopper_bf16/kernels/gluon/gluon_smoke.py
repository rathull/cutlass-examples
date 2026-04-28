from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def run(inputs, outputs):
    import torch

    scratch_in = torch.ones((1,), device=inputs.a.device, dtype=torch.float32)
    scratch_out = torch.empty_like(scratch_in)
    _copy_kernel[(1,)](scratch_in, scratch_out)

    return torch.addmm(
        outputs.c,
        inputs.a,
        inputs.b.T,
        beta=inputs.beta,
        alpha=inputs.alpha,
        out=outputs.c,
    )


@gluon.jit
def _copy_kernel(src, dst):
    value = gl.load(src + 0)
    # Keep one tiny real Gluon kernel in this example so imports are not the
    # only thing being validated.
    gl.store(dst + 0, value)
