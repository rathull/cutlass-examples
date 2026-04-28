from __future__ import annotations

from collections.abc import Callable


def prepare(*, force_prepare: bool = False) -> None:
    _ = force_prepare
    # Native CUDA/CuTe C++ kernels are discovered by filename. The generated
    # binding/compiler path belongs here, outside the kernel source files.


def inspect_ptxas(*, force_prepare: bool = False) -> str:
    _ = force_prepare
    raise NotImplementedError("ptxas inspection for convention-based native kernels is not implemented yet.")


def __getattr__(name: str) -> Callable[[object], object]:
    def _missing_native_kernel(inputs):
        _ = inputs
        raise NotImplementedError(
            f"Native kernel {name!r} was discovered, but no generated binding "
            "has been implemented in native_extension.py yet."
        )

    return _missing_native_kernel
