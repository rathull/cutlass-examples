from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

KERNEL_PARAMS_METADATA_KEY = "kernel_params"
EXTRA_CUDA_CFLAGS_METADATA_KEY = "extra_cuda_cflags"
EXTRA_LDFLAGS_METADATA_KEY = "extra_ldflags"
EXTRA_INCLUDE_PATHS_METADATA_KEY = "extra_include_paths"
RUNNER_FACTORY_METADATA_KEY = "runner_factory"
PREPARE_FACTORY_METADATA_KEY = "prepare_factory"
PTXAS_FACTORY_METADATA_KEY = "ptxas_factory"
ALLOW_FAILURE_METADATA_KEY = "allow_failure"


@dataclass(frozen=True)
class KernelSpec:
    name: str
    problem: str
    kind: str
    target: str | None = None
    source: str | None = None
    supported_gpus: tuple[str, ...] = ("any",)
    supported_dtypes: tuple[str, ...] = ("bf16",)
    supported_layouts: tuple[str, ...] = ("nt",)
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def supports_gpu(self, gpu: str) -> bool:
        normalized = gpu.upper().rstrip("!")
        supported = {item.upper().rstrip("!") for item in self.supported_gpus}
        return "ANY" in supported or normalized in supported


class KernelRegistry:
    def __init__(self, specs: Iterable[KernelSpec] = ()) -> None:
        self._specs: dict[str, KernelSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: KernelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate kernel name: {spec.name}")
        self._specs[spec.name] = spec

    def all(self, *, problem: str | None = None) -> list[KernelSpec]:
        specs = list(self._specs.values())
        if problem is not None:
            specs = [spec for spec in specs if spec.problem == problem]
        return sorted(specs, key=lambda spec: spec.name)

    def get(self, name: str) -> KernelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown kernel: {name}") from exc

    def resolve(
        self,
        *,
        problem: str,
        kernels: str,
        kinds: str = "",
        gpu: str | None = None,
        dtype: str | None = None,
    ) -> list[KernelSpec]:
        selected: list[KernelSpec]
        if kernels.strip().lower() == "all":
            selected = self.all(problem=problem)
        else:
            names = _parse_csv(kernels)
            selected = [self.get(name) for name in names]
            wrong_problem = [spec.name for spec in selected if spec.problem != problem]
            if wrong_problem:
                raise ValueError(f"Kernels do not belong to problem {problem!r}: {wrong_problem}")

        selected_kinds = set(_parse_csv(kinds, allow_empty=True))
        if selected_kinds:
            selected = [spec for spec in selected if spec.kind in selected_kinds]

        if gpu is not None:
            selected = [spec for spec in selected if spec.supports_gpu(gpu)]

        if dtype is not None:
            selected = [spec for spec in selected if dtype in spec.supported_dtypes]

        if not selected:
            raise ValueError("No kernels matched the requested filters.")

        return selected


def load_target(target: str) -> Callable[..., Any]:
    module_name, symbol_name = target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    symbol: Any = module
    for part in symbol_name.split("."):
        symbol = getattr(symbol, part)
    if not callable(symbol):
        raise TypeError(f"Kernel target is not callable: {target}")
    return symbol


def _parse_csv(raw: str, *, allow_empty: bool = False) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values and not allow_empty:
        raise ValueError("Expected at least one comma-separated value.")
    return values
