from __future__ import annotations


def run(inputs):
    import torch

    return torch.matmul(inputs.a, inputs.b)
