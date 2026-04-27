import pytest

from cutlass_examples.matmul_sm80.bench_utils import (  # type: ignore[import-untyped]
    summarize_latency,
)


def test_summarize_latency_computes_core_stats():
    stats = summarize_latency(
        "kernel",
        [1.0, 2.0, 3.0],
        m=1024,
        n=1024,
        k=1024,
        quantiles=[0.5, 1.0],
    )

    assert stats.name == "kernel"
    assert stats.samples == 3
    assert stats.mean_ms == pytest.approx(2.0)
    assert stats.median_ms == pytest.approx(2.0)
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(3.0)
    assert stats.quantiles_ms[0.5] == pytest.approx(2.0)
    assert stats.quantiles_ms[1.0] == pytest.approx(3.0)
    assert stats.median_tflops == pytest.approx(1.073741824)


def test_summarize_latency_rejects_empty_samples():
    with pytest.raises(ValueError):
        summarize_latency("kernel", [], m=1, n=1, k=1, quantiles=[0.5])
