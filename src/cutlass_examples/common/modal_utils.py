from __future__ import annotations

from pathlib import Path
from typing import Iterable

import modal

REMOTE_PACKAGE_DIR = Path("/opt/cutlass_examples")


def build_cuda_image(
    *,
    cuda_version: str,
    flavor: str,
    operating_system: str,
    local_mounts: list[tuple[Path, Path]],
    extra_pip_packages: Iterable[str] = (),
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

    for local_dir, remote_dir in local_mounts:
        image = image.add_local_dir(str(local_dir), remote_path=str(remote_dir))

    return image
