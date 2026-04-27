from __future__ import annotations

from pathlib import Path
from typing import Iterable

import modal

REMOTE_PACKAGE_DIR = Path("/opt/cutlass_examples")
BUILD_CACHE_DIR = Path("/cache/build")
ARTIFACTS_DIR = Path("/artifacts")

DEFAULT_CUDA_VERSION = "13.0.2"
DEFAULT_CUDA_FLAVOR = "devel"
DEFAULT_OPERATING_SYSTEM = "ubuntu24.04"
DEFAULT_TORCH_SPEC = "torch==2.10.0"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
DEFAULT_KERNEL_PACKAGES = (
    "numpy",
    "ninja",
    "triton>=3.6.0",
    "nvidia-cutlass-dsl[cu13]",
)

GPU_ALIASES = {
    "a100": "A100",
    "h100": "H100!",
    "h100!": "H100!",
    "h200": "H200",
    "b200": "B200",
    "b200+": "B200+",
}


def build_cuda_image(
    *,
    cuda_version: str,
    flavor: str,
    operating_system: str,
    local_mounts: list[tuple[Path, Path]],
    extra_pip_packages: Iterable[str] = (),
    extra_apt_packages: Iterable[str] = (),
    extra_commands: Iterable[str] = (),
    python_version: str = "3.12",
    torch_spec: str = "torch==2.10.0",
    torch_index_url: str = "https://download.pytorch.org/whl/cu130",
) -> modal.Image:
    tag = f"{cuda_version}-{flavor}-{operating_system}"

    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python=python_version)
        .entrypoint([])
        .uv_pip_install(torch_spec, index_url=torch_index_url)
    )

    extra_pip_packages = tuple(extra_pip_packages)
    if extra_pip_packages:
        image = image.uv_pip_install(*extra_pip_packages)

    extra_apt_packages = tuple(extra_apt_packages)
    if extra_apt_packages:
        image = image.apt_install(*extra_apt_packages)

    extra_commands = list(extra_commands)
    if extra_commands:
        image = image.run_commands(*extra_commands)

    # local mounts must be last -- Modal doesn't allow build steps after add_local_dir
    for local_dir, remote_dir in local_mounts:
        image = image.add_local_dir(str(local_dir), remote_path=str(remote_dir))

    return image


def normalize_gpu(gpu: str) -> str:
    normalized = gpu.strip().lower()
    return GPU_ALIASES.get(normalized, gpu)


def gpu_arch(gpu: str) -> str:
    normalized = normalize_gpu(gpu).upper().rstrip("!+")
    if normalized.startswith("A100"):
        return "sm80"
    if normalized in {"H100", "H200"}:
        return "sm90"
    if normalized in {"B200", "B300"}:
        return "sm100"
    raise ValueError(f"Unsupported GPU for kernel benchmarks: {gpu}")


def build_kernel_image(
    *,
    local_mounts: list[tuple[Path, Path]],
    cuda_version: str = DEFAULT_CUDA_VERSION,
    flavor: str = DEFAULT_CUDA_FLAVOR,
    operating_system: str = DEFAULT_OPERATING_SYSTEM,
    extra_pip_packages: Iterable[str] = (),
    extra_apt_packages: Iterable[str] = (),
    extra_commands: Iterable[str] = (),
) -> modal.Image:
    packages = (*DEFAULT_KERNEL_PACKAGES, *tuple(extra_pip_packages))
    return build_cuda_image(
        cuda_version=cuda_version,
        flavor=flavor,
        operating_system=operating_system,
        local_mounts=local_mounts,
        extra_pip_packages=packages,
        extra_apt_packages=("git", *tuple(extra_apt_packages)),
        extra_commands=extra_commands,
        torch_spec=DEFAULT_TORCH_SPEC,
        torch_index_url=DEFAULT_TORCH_INDEX_URL,
    )
