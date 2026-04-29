from __future__ import annotations

import importlib
import itertools
import json
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Iterable, cast

from .kernel_registry import (
    ALLOW_FAILURE_METADATA_KEY,
    EXTRA_CUDA_CFLAGS_METADATA_KEY,
    KERNEL_PARAMS_METADATA_KEY,
    KernelSpec,
    PREPARE_FACTORY_METADATA_KEY,
    PTXAS_FACTORY_METADATA_KEY,
    RUNNER_FACTORY_METADATA_KEY,
)


def expand_parameterized_specs(
    specs: Iterable[KernelSpec],
    args: dict[str, str],
) -> list[KernelSpec]:
    if not args:
        return list(specs)

    expanded: list[KernelSpec] = []
    for spec in specs:
        expanded.extend(_expand_spec(spec, args))
    return expanded


def _expand_spec(spec: KernelSpec, args: dict[str, str]) -> list[KernelSpec]:
    if spec.target is None:
        return [spec]

    module_name, _ = spec.target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    params_type = getattr(module, "Params", None)
    if params_type is None:
        return [spec]
    if not is_dataclass(params_type):
        raise TypeError(f"{module_name}.Params must be a dataclass")
    params_class = cast(Any, params_type)

    param_names = tuple(field.name for field in fields(params_class))
    if not any(args.get(name, "").strip() for name in param_names):
        return [spec]

    defaults = params_class()
    values_by_name = {
        name: _parse_values(args.get(name, ""), getattr(defaults, name))
        for name in param_names
    }
    variants: list[KernelSpec] = []
    for values in itertools.product(*(values_by_name[name] for name in param_names)):
        params = params_class(**dict(zip(param_names, values, strict=True)))
        variant_name = _variant_name(spec.name, params)
        variants.append(
            KernelSpec(
                name=variant_name,
                problem=spec.problem,
                kind=spec.kind,
                target=None,
                source=spec.source,
                supported_gpus=spec.supported_gpus,
                supported_dtypes=spec.supported_dtypes,
                supported_layouts=spec.supported_layouts,
                description=spec.description,
                metadata={
                    "path": spec.metadata.get("path", ""),
                    "source": spec.metadata.get("source", spec.source or ""),
                    KERNEL_PARAMS_METADATA_KEY: _params_label(params),
                    EXTRA_CUDA_CFLAGS_METADATA_KEY: json.dumps(
                        _cuda_cflags(params, variant_name)
                    ),
                    RUNNER_FACTORY_METADATA_KEY: (
                        f"{_native_extension_module(spec)}:make_variant_runner"
                    ),
                    PREPARE_FACTORY_METADATA_KEY: (
                        f"{_native_extension_module(spec)}:prepare_variant"
                    ),
                    PTXAS_FACTORY_METADATA_KEY: (
                        f"{_native_extension_module(spec)}:inspect_ptxas_variant"
                    ),
                    ALLOW_FAILURE_METADATA_KEY: "true",
                },
            )
        )
    return variants


def _parse_values(raw: str, default: int) -> tuple[int, ...]:
    if not raw.strip():
        return (default,)
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    min_value = 0 if default == 0 else 1
    if not values or any(value < min_value for value in values):
        expected = "nonnegative" if min_value == 0 else "positive"
        raise ValueError(f"Expected {expected} comma-separated integers, got: {raw!r}")
    return values


def _variant_name(base_name: str, params: Any) -> str:
    suffix = "_".join(f"{name.lower()}{value}" for name, value in asdict(params).items())
    return f"{base_name}__{suffix}"


def _params_label(params: Any) -> str:
    return ",".join(f"{name}={value}" for name, value in asdict(params).items())


def _cuda_cflags(params: Any, variant_name: str) -> tuple[str, ...]:
    return (
        f"-DENTRYPOINT={variant_name}",
        f"-DKERNEL_NAME={variant_name}_kernel",
        *[f"-D{name}={value}" for name, value in asdict(params).items()],
    )


def _native_extension_module(spec: KernelSpec) -> str:
    return f"cutlass_examples.problems.{spec.problem}.native_extension"
