#!/bin/bash

PROBLEM=gemm_hopper_bf16
GPU=h100
KERNELS=cutlass_wgmma_v0
FORCE_PREPARE=true
OUT=artifacts/ptxas/hopper-sweep

uv run modal run -m cutlass_examples.cli \
  --command ptxas \
  --problem $PROBLEM \
  --gpu $GPU \
  --kernels $KERNELS \
  --force-prepare $FORCE_PREPARE \
  --out $OUT