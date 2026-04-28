from __future__ import annotations

from pathlib import Path

from ...common.kernel_discovery import discover_problem_kernels
from ...common.kernel_registry import KernelRegistry, KernelSpec

PROBLEM = "gemm_hopper_bf16"
SUPPORTED_GPUS = ("H100", "H200", "B200")
PROBLEM_DIR = Path(__file__).resolve().parent


def get_registry() -> KernelRegistry:
    return KernelRegistry(_kernel_specs())


def _kernel_specs() -> list[KernelSpec]:
    return discover_problem_kernels(
        problem=PROBLEM,
        problem_dir=PROBLEM_DIR,
        default_supported_gpus=SUPPORTED_GPUS,
    )
