#!/bin/bash

PROBLEM=gemm_hopper_bf16
GPU=h100
SHAPES="1024,1024,1024;2048,2048,2048;4096,4096,4096"
DTYPE=bf16
WARMUP_RUNS=1000
BENCH_RUNS=10000
OUT=artifacts/runs/hopper-cuda-v2

KERNELS=cublas,cuda_v2_tma_wgmma

uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem $PROBLEM \
  --gpu $GPU \
  --kernels $KERNELS \
  --shapes $SHAPES \
  --dtype $DTYPE \
  --warmup-runs $WARMUP_RUNS \
  --bench-runs $BENCH_RUNS \
  --out $OUT