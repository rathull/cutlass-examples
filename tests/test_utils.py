import pytest

from cutlass_examples.common.utils import (  # type: ignore[import-untyped]
    get_tflops,
    parse_bool,
    parse_quantiles,
    parse_shape,
    parse_shapes,
    quantile_label,
)


def test_get_tflops():
    assert get_tflops(1024, 1024, 1024, 1.0) == pytest.approx(2.147483648)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4096,4096,4096", (4096, 4096, 4096)),
        (" 128, 256, 512 ", (128, 256, 512)),
    ],
)
def test_parse_shape(raw, expected):
    assert parse_shape(raw) == expected


@pytest.mark.parametrize("raw", ["", "1,2", "1,2,3,4", "1,0,3", "1,-2,3"])
def test_parse_shape_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_shape(raw)


def test_parse_shapes():
    assert parse_shapes("128,128,128;256,256,256") == [
        (128, 128, 128),
        (256, 256, 256),
    ]
    assert parse_shapes("128,128,128 256,256,256") == [
        (128, 128, 128),
        (256, 256, 256),
    ]


def test_parse_quantiles():
    assert parse_quantiles("0.5,0.95") == [0.5, 0.95]
    with pytest.raises(ValueError):
        parse_quantiles("1.2")


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("off", False)])
def test_parse_bool(raw, expected):
    assert parse_bool(raw) is expected


@pytest.mark.parametrize(("quantile", "expected"), [(0.5, "p50"), (0.995, "p99.5")])
def test_quantile_label(quantile, expected):
    assert quantile_label(quantile) == expected
