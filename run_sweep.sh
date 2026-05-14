#!/bin/bash
set -euo pipefail

PROBLEM=gemm_hopper_bf16
GPU=h100
KERNELS=cuda_v2_tma_wgmma
SHAPES="4096,4096,4096"
DTYPE=bf16
WARMUP_RUNS=50
BENCH_RUNS=200
REPETITIONS=1
CHECK_CORRECTNESS=true
BENCHMARK_REF=true
FORCE_PREPARE=false
PARALLEL=true
OUT=artifacts/runs/hopper-cuda-v2-tma-wgmma-sweep

# TODO: see if BN=256 can be supported at all
BM=128,192,256
BN=256
BK=64
NUM_STAGES=2,4

# Want to try (128, 128, 64), (256, 128, 64), (128, 256, 64)

uv run modal run -m cutlass_examples.cli \
  --command sweep \
  --problem "$PROBLEM" \
  --gpu "$GPU" \
  --kernels "$KERNELS" \
  --shapes "$SHAPES" \
  --dtype "$DTYPE" \
  --bm "$BM" \
  --bn "$BN" \
  --bk "$BK" \
  --num-stages "$NUM_STAGES" \
  --warmup-runs "$WARMUP_RUNS" \
  --bench-runs "$BENCH_RUNS" \
  --repetitions "$REPETITIONS" \
  --check-correctness "$CHECK_CORRECTNESS" \
  --benchmark-ref "$BENCHMARK_REF" \
  --force-prepare "$FORCE_PREPARE" \
  --parallel "$PARALLEL" \
  --out "$OUT"
