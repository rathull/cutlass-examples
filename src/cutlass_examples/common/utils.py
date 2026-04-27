from __future__ import annotations

GPU_TO_ARCH = {
    "A100": "80",
    "A100-40GB": "80",
    "A100-80GB": "80",
    "H100": "90",
    "H200": "90",
    "B200": "100",
}

CUTLASS_VERSION = "4.4.2"


def get_tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return float("inf")
    return 2 * m * n * k * 1e-9 / latency_ms


def parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_shape(shape: str) -> tuple[int, int, int]:
    parts = parse_csv_list(shape)
    if len(parts) != 3:
        raise ValueError(f"Expected shape as M,N,K but got: {shape!r}")
    parsed = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in parsed):
        raise ValueError(f"Shape dimensions must be positive, got: {shape!r}")
    return parsed  # type: ignore[return-value]


def parse_shapes(shapes: str) -> list[tuple[int, int, int]]:
    """Parse one or more M,N,K shapes separated by semicolons or whitespace."""
    if ";" in shapes:
        raw_shapes = [item.strip() for item in shapes.split(";") if item.strip()]
    else:
        whitespace_parts = shapes.split()
        if len(whitespace_parts) > 1 and all(part.count(",") == 2 for part in whitespace_parts):
            raw_shapes = whitespace_parts
        else:
            raw_shapes = [shapes.strip()] if shapes.strip() else []

    parsed_shapes = [parse_shape(item) for item in raw_shapes]
    if not parsed_shapes:
        raise ValueError("Expected at least one shape.")
    return parsed_shapes


def parse_quantiles(raw: str) -> list[float]:
    quantiles = [float(item) for item in parse_csv_list(raw)]
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantiles must be between 0 and 1, got: {quantile}")
    return quantiles


def quantile_label(quantile: float) -> str:
    percent = quantile * 100
    if percent.is_integer():
        return f"p{int(percent)}"
    return f"p{percent:g}"


def parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {raw!r}")