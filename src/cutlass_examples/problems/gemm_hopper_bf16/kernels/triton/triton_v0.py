from __future__ import annotations

import triton
import triton.language as tl


def run(inputs, outputs, *, block_m: int = 32, block_n: int = 32, block_k: int = 32):
    a = inputs.a
    b = inputs.b
    c = outputs.c
    m, k = a.shape
    n, _ = b.shape
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        inputs.alpha,
        inputs.beta,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_idx = k_start + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * k + k_idx[None, :],
            mask=(offs_m[:, None] < m) & (k_idx[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * k + k_idx[:, None],
            mask=(k_idx[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c_offsets = c_ptr + offs_m[:, None] * n + offs_n[None, :]
    if beta != 0.0:
        c_prev = tl.load(
            c_offsets,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + c_prev * beta
    else:
        acc = acc * alpha

    tl.store(
        c_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )
