# cutlass_examples/gemm/kernels/kernels.py
import torch
from .cuda_ref.wrapper import matmul_cuda_ref
from .cuda_kernel.wrapper import matmul_cuda


def make_args(
    shape: tuple[int, int, int] = (1024, 1024, 1024),
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    M, N, K = shape
    # TODO: is this requires_grad=False sufficient to avoid backprop overhead or
    # do I also need a no_grad context manager? For most faithful and accurate benchmarking?
    A = torch.randn((M, K), device=device, dtype=dtype, requires_grad=False)
    B = torch.randn((K, N), device=device, dtype=dtype, requires_grad=False)
    C = torch.empty((M, N), device=device, dtype=dtype, requires_grad=False)
    return (A, B), C

def get_flop_count(shape: tuple[int, int, int]) -> int:
    M, N, K = shape
    return 2 * M * N * K

def torch_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    torch.matmul(A, B, out=C)

# def cuda_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
#     # TODO: this is a placeholder for the actual CUDA kernel I'm going to
#     # implement and pybind
#     torch.matmul(A, B, out=C)

kernels_map = {
    "ref": matmul_cuda_ref,
    "torch_ref": torch_matmul,
    "cuda": matmul_cuda,
}