from __future__ import annotations

from pathlib import Path

from .kernel_registry import KernelSpec

PYTHON_KERNEL_KINDS = {"reference", "triton", "gluon", "cute_dsl"}
NATIVE_KERNEL_KINDS = {"native_cuda"}
NATIVE_SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".cuh")


def discover_problem_kernels(
    *,
    problem: str,
    problem_dir: Path,
    default_supported_gpus: tuple[str, ...] = ("any",),
    default_supported_dtypes: tuple[str, ...] = ("bf16",),
) -> list[KernelSpec]:
    kernels_dir = problem_dir / "kernels"
    if not kernels_dir.exists():
        return []

    specs: list[KernelSpec] = []
    for kind_dir in sorted(path for path in kernels_dir.iterdir() if path.is_dir()):
        kind = kind_dir.name
        for path in sorted(kind_dir.iterdir()):
            if path.name.startswith("_"):
                continue
            if path.suffix == ".py" and kind in PYTHON_KERNEL_KINDS:
                specs.append(
                    _python_kernel_spec(
                        problem=problem,
                        problem_dir=problem_dir,
                        kind=kind,
                        path=path,
                        default_supported_gpus=default_supported_gpus,
                        default_supported_dtypes=default_supported_dtypes,
                    )
                )
            elif path.suffix == ".py" and kind in NATIVE_KERNEL_KINDS:
                source_path = _native_source_for_runner(path)
                if source_path is None:
                    continue
                specs.append(
                    _native_kernel_spec(
                        problem=problem,
                        problem_dir=problem_dir,
                        kind=kind,
                        path=path,
                        source_path=source_path,
                        default_supported_gpus=default_supported_gpus,
                        default_supported_dtypes=default_supported_dtypes,
                    )
                )

    return specs


def _python_kernel_spec(
    *,
    problem: str,
    problem_dir: Path,
    kind: str,
    path: Path,
    default_supported_gpus: tuple[str, ...],
    default_supported_dtypes: tuple[str, ...],
) -> KernelSpec:
    module = _module_for_kernel(problem, kind, path.stem)
    supported_gpus = ("any",) if kind == "reference" else default_supported_gpus
    return KernelSpec(
        name=path.stem,
        problem=problem,
        kind=kind,
        target=f"{module}:run",
        source=_relative_to_problem(problem_dir, path),
        supported_gpus=supported_gpus,
        supported_dtypes=default_supported_dtypes,
        metadata={
            "path": _relative_to_problem(problem_dir, path),
        },
    )


def _native_kernel_spec(
    *,
    problem: str,
    problem_dir: Path,
    kind: str,
    path: Path,
    source_path: Path,
    default_supported_gpus: tuple[str, ...],
    default_supported_dtypes: tuple[str, ...],
) -> KernelSpec:
    source = _relative_to_problem(problem_dir, source_path)
    module = _module_for_kernel(problem, kind, path.stem)
    return KernelSpec(
        name=path.stem,
        problem=problem,
        kind=kind,
        target=f"{module}:run",
        source=source,
        supported_gpus=default_supported_gpus,
        supported_dtypes=default_supported_dtypes,
        metadata={
            "path": _relative_to_problem(problem_dir, path),
            "source": source,
            "prepare": f"{module}:prepare",
            "ptxas": f"{module}:inspect_ptxas",
        },
    )


def _native_source_for_runner(path: Path) -> Path | None:
    for suffix in NATIVE_SOURCE_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _module_for_kernel(problem: str, kind: str, stem: str) -> str:
    return f"cutlass_examples.problems.{problem}.kernels.{kind}.{stem}"


def _relative_to_problem(problem_dir: Path, path: Path) -> str:
    return str(path.relative_to(problem_dir))
