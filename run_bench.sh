#!/bin/bash

PROBLEM=gemm_hopper_bf16
GPU=h100
SHAPES="1024,1024,1024;2048,2048,2048;4096,4096,4096"
DTYPE=bf16
WARMUP_RUNS=10
BENCH_RUNS=10
OUT=artifacts/runs/hopper-cute-cpp

KERNELS=cublas,cute_cpp_v0,cute_cpp_v1_tma_wgmma

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