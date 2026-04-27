from __future__ import annotations


def cublas(inputs):
    import torch

    return torch.matmul(inputs.a, inputs.b)
