"""Tests for the build cache hashing and lookup logic.

All tests run locally without GPU, Modal, or CUDA -- they exercise the pure
hashing functions and path helpers using temporary files.
"""

from pathlib import Path

import pytest

from cutlass_examples.common.build_cache import compute_source_hash, get_cached_so

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def source_files(tmp_path: Path) -> list[Path]:
    """Create a minimal set of fake .cu / .cpp source files."""
    cpp = tmp_path / "matmul.cpp"
    cpp.write_text('#include "common.h"\nvoid bind() {}')
    cu0 = tmp_path / "matmul_v0.cu"
    cu0.write_text("__global__ void kernel_v0() {}")
    cu1 = tmp_path / "matmul_v1.cu"
    cu1.write_text("__global__ void kernel_v1() {}")
    return [cpp, cu0, cu1]


@pytest.fixture()
def include_files(tmp_path: Path) -> list[Path]:
    header = tmp_path / "common.h"
    header.write_text("#pragma once\nint cdiv(int a, int b);")
    return [header]


@pytest.fixture()
def default_flags() -> dict:
    return {
        "cuda_cflags": ["-O3", "-lineinfo", "-gencode=arch=compute_80,code=sm_80"],
        "ldflags": [],
        "torch_version": "2.10.0+cu130",
        "cuda_version": "13.0",
        "cutlass_version": "4.4.2",
    }


# ---------------------------------------------------------------------------
# compute_source_hash
# ---------------------------------------------------------------------------

class TestComputeSourceHash:
    def test_deterministic(self, source_files, include_files, default_flags):
        """Same inputs always produce the same hash."""
        h1 = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        h2 = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        assert h1 == h2

    def test_length(self, source_files, include_files, default_flags):
        h = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_source_order_independent(self, source_files, include_files, default_flags):
        """Hash should not change when sources are passed in a different order."""
        h_forward = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        h_reversed = compute_source_hash(
            sources=list(reversed(source_files)),
            include_files=include_files,
            **default_flags,
        )
        assert h_forward == h_reversed

    def test_changes_on_source_content_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        source_files[1].write_text("__global__ void kernel_v0_modified() {}")
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        assert h_before != h_after

    def test_changes_on_header_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        include_files[0].write_text("#pragma once\nint cdiv(int a, int b);\nint NEW_FUNC();")
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        assert h_before != h_after

    def test_changes_on_cuda_cflag_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        modified = {**default_flags, "cuda_cflags": ["-O2", "-lineinfo"]}
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **modified,
        )
        assert h_before != h_after

    def test_changes_on_ldflag_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        modified = {**default_flags, "ldflags": ["-lcuda"]}
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **modified,
        )
        assert h_before != h_after

    def test_changes_on_torch_version_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        modified = {**default_flags, "torch_version": "2.11.0+cu131"}
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **modified,
        )
        assert h_before != h_after

    def test_changes_on_cuda_version_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        modified = {**default_flags, "cuda_version": "13.1"}
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **modified,
        )
        assert h_before != h_after

    def test_changes_on_cutlass_version_change(self, source_files, include_files, default_flags):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        modified = {**default_flags, "cutlass_version": "4.5.0"}
        h_after = compute_source_hash(
            sources=source_files, include_files=include_files, **modified,
        )
        assert h_before != h_after

    def test_adding_source_changes_hash(self, source_files, include_files, default_flags, tmp_path):
        h_before = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        new_src = tmp_path / "matmul_v2.cu"
        new_src.write_text("__global__ void kernel_v2() {}")
        h_after = compute_source_hash(
            sources=source_files + [new_src],
            include_files=include_files,
            **default_flags,
        )
        assert h_before != h_after

    def test_empty_include_files(self, source_files, default_flags):
        """Works with no tracked include files."""
        h = compute_source_hash(
            sources=source_files, include_files=[], **default_flags,
        )
        assert len(h) == 16


# ---------------------------------------------------------------------------
# get_cached_so
# ---------------------------------------------------------------------------

class TestGetCachedSo:
    def test_path_structure(self, tmp_path):
        so = get_cached_so(tmp_path, "abc123", "my_module")
        assert so == tmp_path / "abc123" / "my_module.so"

    def test_different_hashes_different_paths(self, tmp_path):
        so1 = get_cached_so(tmp_path, "hash_a", "mod")
        so2 = get_cached_so(tmp_path, "hash_b", "mod")
        assert so1 != so2
        assert so1.parent != so2.parent

    def test_different_modules_different_paths(self, tmp_path):
        so1 = get_cached_so(tmp_path, "same_hash", "mod_a")
        so2 = get_cached_so(tmp_path, "same_hash", "mod_b")
        assert so1 != so2


# ---------------------------------------------------------------------------
# Cache hit / miss integration (filesystem-level, no GPU)
# ---------------------------------------------------------------------------

class TestCacheLookup:
    def test_cache_miss_when_empty(self, tmp_path, source_files, include_files, default_flags):
        h = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so = get_cached_so(tmp_path, h, "test_module")
        assert not so.exists()

    def test_cache_hit_after_store(self, tmp_path, source_files, include_files, default_flags):
        h = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so = get_cached_so(tmp_path, h, "test_module")
        so.parent.mkdir(parents=True, exist_ok=True)
        so.write_bytes(b"\x7fELF_fake_so")  # simulate a compiled .so

        # look it up again with the same hash
        so2 = get_cached_so(tmp_path, h, "test_module")
        assert so2.exists()
        assert so2.read_bytes() == b"\x7fELF_fake_so"

    def test_cache_miss_after_source_change(self, tmp_path, source_files, include_files, default_flags):
        """Modifying a source produces a new hash that has no cached .so."""
        h_old = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so_old = get_cached_so(tmp_path, h_old, "mod")
        so_old.parent.mkdir(parents=True, exist_ok=True)
        so_old.write_bytes(b"old_binary")

        # mutate a source
        source_files[0].write_text("// changed source")

        h_new = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so_new = get_cached_so(tmp_path, h_new, "mod")
        assert not so_new.exists(), "Modified source should not hit the old cache"
        assert so_old.exists(), "Old cached .so should still be on disk"

    def test_multiple_cached_builds_coexist(self, tmp_path, source_files, include_files, default_flags):
        """Different source versions each get their own cached .so."""
        h1 = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so1 = get_cached_so(tmp_path, h1, "mod")
        so1.parent.mkdir(parents=True, exist_ok=True)
        so1.write_bytes(b"binary_v1")

        # change a kernel
        source_files[1].write_text("__global__ void kernel_v0_v2() {}")
        h2 = compute_source_hash(
            sources=source_files, include_files=include_files, **default_flags,
        )
        so2 = get_cached_so(tmp_path, h2, "mod")
        so2.parent.mkdir(parents=True, exist_ok=True)
        so2.write_bytes(b"binary_v2")

        assert h1 != h2
        assert so1.exists() and so2.exists()
        assert so1.read_bytes() == b"binary_v1"
        assert so2.read_bytes() == b"binary_v2"

    def test_correct_binary_retrieved(self, tmp_path, source_files, include_files, default_flags):
        """After multiple builds, looking up a specific hash returns the right binary."""
        binaries: dict[str, bytes] = {}
        for i in range(3):
            source_files[1].write_text(f"__global__ void kernel_v0_iter{i}() {{}}")
            h = compute_source_hash(
                sources=source_files, include_files=include_files, **default_flags,
            )
            so = get_cached_so(tmp_path, h, "mod")
            so.parent.mkdir(parents=True, exist_ok=True)
            payload = f"binary_iter{i}".encode()
            so.write_bytes(payload)
            binaries[h] = payload

        # verify each hash still maps to the correct binary
        for h, expected in binaries.items():
            so = get_cached_so(tmp_path, h, "mod")
            assert so.exists()
            assert so.read_bytes() == expected
