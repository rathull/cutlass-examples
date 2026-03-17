from __future__ import annotations

from ..common.utils import parse_csv_list

KERNEL_SOURCES = {
    "v0": "matmul_v0.cu",
    "v1": "matmul_v1.cu",
}

DEFAULT_SHAPE = "4096,4096,4096"
DEFAULT_VERSIONS = ",".join(KERNEL_SOURCES)


def parse_versions(versions: str) -> list[str]:
    selected_versions = parse_csv_list(versions)
    unknown_versions = [version for version in selected_versions if version not in KERNEL_SOURCES]
    if unknown_versions:
        raise ValueError(
            f"Unsupported kernel versions: {unknown_versions}. "
            f"Supported versions: {sorted(KERNEL_SOURCES)}"
        )
    return selected_versions

