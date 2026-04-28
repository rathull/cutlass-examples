#!/bin/bash

PROBLEM=gemm_hopper_bf16
GPU=h100
KERNELS=all
KINDS=triton,cute_dsl
SHAPES='1024,1024,1024;2048,2048,2048;4096,4096,4096'
REPETITIONS=3
PARALLEL=true
OUT=artifacts/runs/hopper-sweep

uv run modal run -m cutlass_examples.cli \
  --command sweep \
  --problem $PROBLEM \
  --gpu $GPU \
  --kernels $KERNELS \
  --kinds $KINDS \
  --shapes $SHAPES \
  --repetitions $REPETITIONS \
  --parallel $PARALLEL \
  --out $OUT
