#!/bin/bash
set -euo pipefail

PROBLEM=gemm_hopper_bf16
GPU=h100
KERNELS=cuda_v1_smem_tiling
SHAPES="4096,4096,4096"
DTYPE=bf16
WARMUP_RUNS=20
BENCH_RUNS=50
REPETITIONS=1
CHECK_CORRECTNESS=true
BENCHMARK_REF=true
FORCE_PREPARE=false
OUT=artifacts/runs/hopper-native-tile-sweep-stage2

# Stage 2: local search around the best point from the previous run:
#   bm128_bn64_bk32_tm8_tn8
#
# The previous sweep showed:
# - TM=8,TN=8 dominated TM=4,TN=8.
# - BN=64 dominated BN=128.
# - BK=32 was slightly better than BK=16.
#
# This sweep asks:
# - Does a larger M tile improve reuse/occupancy? 128 -> 192/256.
# - Is BN=64 actually optimal, or do 32/96 win?
# - Does BK=64 improve K reuse enough to offset smem/load cost?
#
# 18 variants total, plus cublas when BENCHMARK_REF=true.
BM=128,192,256
BN=32,64,96
BK=32,64
TM=8
TN=8

uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem "$PROBLEM" \
  --gpu "$GPU" \
  --kernels "$KERNELS" \
  --shapes "$SHAPES" \
  --dtype "$DTYPE" \
  --bm "$BM" \
  --bn "$BN" \
  --bk "$BK" \
  --tm "$TM" \
  --tn "$TN" \
  --warmup-runs "$WARMUP_RUNS" \
  --bench-runs "$BENCH_RUNS" \
  --repetitions "$REPETITIONS" \
  --check-correctness "$CHECK_CORRECTNESS" \
  --benchmark-ref "$BENCHMARK_REF" \
  --force-prepare "$FORCE_PREPARE" \
  --out "$OUT"
