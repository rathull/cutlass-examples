"""
Custom CUDA matmul kernel wrapper using PyTorch JIT compilation.
"""
import torch
from pathlib import Path

# Get the directory where this file is located
_KERNEL_DIR = Path(__file__).parent

def _load_cuda_kernel():
    """Load and compile the CUDA kernel using PyTorch JIT."""
    sources = [
        str(_KERNEL_DIR / "kernel.cu"),
        str(_KERNEL_DIR / "bindings.cpp"),
    ]
    
    extra_include_paths = []
    # TODO: potentially remove --use_fast_math for correctness?
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    extra_cflags = ["-O3"]
    
    # Load and compile the extension
    from torch.utils.cpp_extension import load
    print(f"Loading CUDA kernel from {sources}...")
    ext = load(
        name="custom_matmul_kernel",
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    print(f"Finished loading CUDA kernel from {sources}")
    
    return ext


# Lazy loading - compile on first use
_kernel_module = None

def _get_kernel_module():
    """Get the compiled kernel module, compiling if necessary."""
    global _kernel_module
    if _kernel_module is None:
        _kernel_module = _load_cuda_kernel()
    return _kernel_module


def matmul_cuda_ref(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    """
    Custom CUDA matmul kernel wrapper.
    
    Performs C = A @ B where:
    - A: (M, K) tensor
    - B: (K, N) tensor  
    - C: (M, N) output tensor (pre-allocated)
    
    Args:
        A: Input matrix A
        B: Input matrix B
        C: Output matrix C (written in-place)
    """
    # Validate inputs
    # TODO: how much performance penalty is there for validating these inputs in the hot loop?
    # Should I leave this validation in the main kernel? How would it be done in production level
    # code? Would it be better to only validate in the correctness check and not the perf bench?
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors"
    assert A.device.type == "cuda" and B.device.type == "cuda" and C.device.type == "cuda", \
        "All tensors must be on CUDA device"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "Currently only float32 is supported"
    
    M, K_A = A.shape
    K_B, N = B.shape
    assert K_A == K_B, f"Dimension mismatch: A is (M={M}, K={K_A}), B is (K={K_B}, N={N})"
    
    M_C, N_C = C.shape
    assert M_C == M and N_C == N, \
        f"Output tensor shape mismatch: C is ({M_C}, {N_C}), expected ({M}, {N})"
    
    # Get the compiled kernel module
    kernel_module = _get_kernel_module()
    
    # Launch kernel (the C++ wrapper handles stream management)
    kernel_module.launch_matmul(A, B, C, M, N, K_A)
    
    # Synchronize to ensure kernel completes
    torch.cuda.synchronize()

