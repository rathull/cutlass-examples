from __future__ import annotations

import hashlib
from pathlib import Path


def compute_source_hash(
    *,
    sources: list[Path],
    include_files: list[Path],
    cuda_cflags: list[str],
    ldflags: list[str],
    torch_version: str,
    cuda_version: str,
    cutlass_version: str,
) -> str:
    """Deterministic hash of everything that affects the compiled .so binary.

    Inputs hashed:
      - contents of every source file (.cu, .cpp)
      - contents of every tracked include file (e.g. common.h)
      - CUDA compiler flags and linker flags
      - torch and CUDA runtime versions (affect ABI)
      - CUTLASS/CuTe version (headers are not hashed individually)

    Why not rely on torch.utils.cpp_extension's or ninja's built-in caching:
      - ninja uses file modification timestamps, not content hashes. Modal re-mounts source files
        each run with fresh timestamps, which can lead ninja into unnecessary full rebuilds.
      - PyTorch's JIT versioner has a known bug where it sometimes triggers recompilation even when
        sources haven't changed (https://github.com/pytorch/pytorch/issues/124454).
    This content-hash approach is immune to both issues and bypasses
    torch.utils.cpp_extension.load() entirely on cache hits.
    
    This currently does not implement per-kernel hashing, and all .cu files are compiled into the
    same .so. This is because torch.utils.cpp_extension.load() links all sources into one .so and
    we can't partially relink into the .so with its own hash. To implement this we would need to 
    have separate load() calls and TORCH_LIBRARY assignments for each .so with its own hash which
    would be a larger refactor.
    """
    h = hashlib.sha256()

    for src in sorted(sources, key=lambda p: p.name):
        h.update(f"source:{src.name}\n".encode())
        h.update(src.read_bytes())

    for inc in sorted(include_files, key=lambda p: p.name):
        h.update(f"include:{inc.name}\n".encode())
        h.update(inc.read_bytes())

    for flag in cuda_cflags:
        h.update(f"cuda_cflag:{flag}\n".encode())

    for flag in ldflags:
        h.update(f"ldflag:{flag}\n".encode())

    h.update(f"torch:{torch_version}\n".encode())
    h.update(f"cuda:{cuda_version}\n".encode())
    h.update(f"cutlass:{cutlass_version}\n".encode())

    return h.hexdigest()[:16]


def get_cached_so(cache_dir: Path, source_hash: str, module_name: str) -> Path:
    """Return the path where a cached .so would live for a given hash."""
    return cache_dir / source_hash / f"{module_name}.so"
