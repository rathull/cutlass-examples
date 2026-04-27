from __future__ import annotations


def prepare_sm80(*, force_prepare: bool = False) -> None:
    from .....matmul_sm80.main import get_module

    get_module(force_recompile=force_prepare)


def inspect_sm80_ptxas(*, force_prepare: bool = True) -> str:
    from .....matmul_sm80.main import get_module

    _, build_log = get_module(force_recompile=force_prepare, return_build_log=True)
    return build_log


def sm80_v0(inputs):
    return _run_sm80("v0", inputs)


def sm80_v1(inputs):
    return _run_sm80("v1", inputs)


def sm80_cute_v0(inputs):
    return _run_sm80("cute_v0", inputs)


def sm80_cute_v1(inputs):
    return _run_sm80("cute_v1", inputs)


def sm80_cutlass_v0(inputs):
    return _run_sm80("cutlass_v0", inputs)


def _run_sm80(version: str, inputs):
    from .....matmul_sm80.main import get_module

    module = get_module()
    kernel = getattr(module, f"matmul_{version}")
    return kernel(inputs.a, inputs.b)
