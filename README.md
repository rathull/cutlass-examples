# CUTLASS Examples

## Getting Started

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Kernel Benchmark CLI

Check the Modal GPU environment:

```bash
uv run modal run -m cutlass_examples.cli --command doctor --gpu h100
uv run modal run -m cutlass_examples.cli --command doctor --gpu b200
```

List kernels runnable on a GPU:

```bash
uv run modal run -m cutlass_examples.cli \
  --command list-kernels \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels all
```

Run a single benchmark:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels cublas,triton_v0 \
  --shapes 4096,4096,4096 \
  --dtype bf16 \
  --warmup-runs 50 \
  --bench-runs 500 \
  --out artifacts/runs/hopper-smoke
```

Run a sweep. Use semicolons or whitespace to pass multiple shapes:

```bash
uv run modal run -m cutlass_examples.cli \
  --command sweep \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels all \
  --kinds triton,cute_dsl \
  --shapes '1024,1024,1024;2048,2048,2048;4096,4096,4096' \
  --repetitions 3 \
  --parallel true \
  --out artifacts/runs/hopper-sweep
```

Sweep native CUDA/PTX compile-time parameters with direct kernel parameter
flags. The base kernel's `.py` runner declares which parameters it supports;
for `cuda_v1_smem_tiling`, each cartesian-product configuration is compiled as
its own ephemeral kernel variant:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels cuda_v1_smem_tiling \
  --BM 64,128 \
  --BN 128 \
  --BK 16,32 \
  --TM 8 \
  --TN 8 \
  --shapes 4096,4096,4096 \
  --force-prepare true \
  --out artifacts/runs/hopper-native-param-sweep
```

For C++ kernels, keep tile sizes, register tile sizes, pipeline stages, PTX
instruction variants, and CUTLASS/CuTe template shapes as compile-time values.
Native sweep variants pass values as `nvcc` defines like `-DBM=128` and
`-DTM=8`; the `.cu` file provides defaults for those macros. Runtime arguments
should remain problem data only: pointers, `M`, `N`, `K`, `alpha`, and `beta`.

Inspect native CUDA resource usage from `ptxas -v` for native kernels:

```bash
uv run modal run -m cutlass_examples.cli \
  --command ptxas \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels cutlass_wgmma_v0 \
  --force-prepare true \
  --out artifacts/ptxas/<run-name>
```

This force-compiles native CUDA/CuTe C++ kernels with `-Xptxas=-v`, parses the
compiler log, and writes:

- `ptxas.html`: browser-friendly resource report
- `ptxas.json`: structured function records and raw log
- `ptxas.csv`: spreadsheet-friendly rows
- `ptxas.log`: raw compiler output

The report includes registers/thread, static shared memory, stack frame bytes,
spill stores/loads, constant memory, warnings, and the raw `ptxas` log.

Each run writes:

- `results.json`: full structured run metadata and stats
- `results.jsonl`: append-friendly per-kernel rows
- `results.csv`: spreadsheet-friendly per-kernel rows

## Adding Kernels

Add kernels under `src/cutlass_examples/problems/gemm_hopper_bf16/kernels/<kind>/`.
The benchmark runner discovers them by convention:

- `problem` comes from the problem directory, e.g. `gemm_hopper_bf16`.
- `kind` comes from the folder under `kernels/`, e.g. `triton`, `gluon`,
  `cute_dsl`, `native_cuda`, or `reference`.
- `name` comes from the filename without its extension.

The GEMM problem is NT mode:

```text
C = alpha * A @ B.T + beta * C
A: M x K, contiguous
B: N x K, contiguous and logically transposed
C: M x N, contiguous and preallocated by the benchmark runner
```

The default coefficients are `alpha=1.0` and `beta=0.0`, so empty output
tensors are valid for the default benchmark. Native CUDA/CUTLASS/CuTe C++
kernels can assume the contiguous NT layout and do not need generic stride
parameters.

For Python DSL kernels, export a `run(inputs, outputs)` function. The benchmark
runner allocates `outputs.c` before timing, and the kernel writes into it:

```python
# src/cutlass_examples/problems/gemm_hopper_bf16/kernels/triton/triton_v1.py
def run(inputs, outputs):
    # Write the MxN GEMM result into outputs.c.
    return outputs.c
```

For native CUDA, CUTLASS C++, or CuTe C++ kernels, use a paired source and
runner file under `kernels/native_cuda/`:

```text
src/cutlass_examples/problems/gemm_hopper_bf16/kernels/native_cuda/cutlass_wgmma_v0.cu
src/cutlass_examples/problems/gemm_hopper_bf16/kernels/native_cuda/cutlass_wgmma_v0.py
```

The `.py` file is your explicit user-controlled benchmark entrypoint:

```python
from ... import native_extension

KERNEL_NAME = "cutlass_wgmma_v0"
SOURCE = "kernels/native_cuda/cutlass_wgmma_v0.cu"
EXTRA_CUDA_CFLAGS = ()
EXTRA_LDFLAGS = ()
EXTRA_INCLUDE_PATHS = ()


def prepare(*, force_prepare: bool = False) -> None:
    global kernel
    ops = native_extension.load_kernel(KERNEL_NAME, force_prepare=force_prepare, source=SOURCE)
    kernel = getattr(ops, KERNEL_NAME)


def run(inputs, outputs):
    return kernel(inputs.a, inputs.b, outputs.c, inputs.alpha, inputs.beta)
```

Shared native wrapping/compilation lives in
`src/cutlass_examples/problems/gemm_hopper_bf16/native_extension.py`. It compiles
the matching `.cu` file before timing, registers a `torch.ops` binding, adds the
CUTLASS include paths, and includes per-kernel flags from the runner file. Native
bindings receive `A`, `B`, preallocated `C`, `alpha`, and `beta`, so output
allocation is not part of the timed kernel call.
The repo includes minimal H100 examples for each supported kernel kind:

- `triton_v0`: a simple Triton BF16 GEMM.
- `gluon_smoke`: launches a tiny Gluon kernel, then participates in the GEMM benchmark path.
- `cute_dsl_smoke`: launches a tiny CuTe DSL kernel, then participates in the GEMM benchmark path.
- `cuda_v0`: a naive native CUDA BF16 GEMM with an inline PTX marker.

To compile and compare all examples on H100:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels cublas,triton_v0,gluon_smoke,cute_dsl_smoke,cuda_v0 \
  --shapes 64,64,64 \
  --warmup-runs 1 \
  --bench-runs 2 \
  --force-prepare true \
  --out artifacts/runs/h100-language-smoke
```

After that, run it by name:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels triton_v1 \
  --shapes 4096,4096,4096
```

The CLI handles implementation details from `kind`: native CUDA/CuTe C++
extensions, Triton, Gluon, CuTe DSL, and reference kernels all use the same
benchmark path. Select groups of kernels with `--kinds`.

```bash
# Run all Triton kernels.
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels all \
  --kinds triton

# See all Triton kernels available for H100.
uv run modal run -m cutlass_examples.cli \
  --command list-kernels \
  --problem gemm_hopper_bf16 \
  --gpu h100 \
  --kernels all \
  --kinds triton
```

## Modal Notes

- Use `--gpu h100` for `gpu="H100!"`, which avoids Modal auto-upgrading H100
  benchmark runs to H200.
- Use `--gpu b200` for Blackwell, "--gpu b200+" may upgrade to B300.
- The Modal image installs Torch, Triton, Gluon via Triton, and
  `nvidia-cutlass-dsl[cu13]`.
- Native build caches are split by architecture: `sm90` and `sm100` and first runs may compile or
  JIT kernels; later runs should hit Modal volumes and
  framework caches.
