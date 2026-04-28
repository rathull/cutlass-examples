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

For Python DSL kernels, export a `run(inputs)` function:

```python
# src/cutlass_examples/problems/gemm_hopper_bf16/kernels/triton/triton_v1.py
def run(inputs):
    ...
```

For native CUDA/CuTe C++ kernels, put the source file under `kernels/native_cuda/`:

```text
src/cutlass_examples/problems/gemm_hopper_bf16/kernels/native_cuda/cutlass_wgmma_v0.cu
```

Native wrapping/compilation lives outside the kernel files in
`src/cutlass_examples/problems/gemm_hopper_bf16/native_extension.py`.

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
- Use `--gpu b200` for Blackwell.
- The Modal image installs Torch, Triton, Gluon via Triton, and
  `nvidia-cutlass-dsl[cu13]`.
- Native build caches are split by architecture: `sm90` and `sm100`.
- First runs may compile or JIT kernels; later runs should hit Modal volumes and
  framework caches.

## Project Structure

```
cutlass-examples/
├── pyproject.toml
├── uv.lock
└── src/
    └── cutlass_examples/
        ├── __init__.py
        ├── common/           # Your shared C++/CuTe headers
        │   ├── __init__.py
        │   ├── build_cache.py
        │   ├── common.h
        │   ├── benchmarking.py
        │   ├── kernel_registry.py
        │   ├── modal_utils.py
        │   ├── runner.py
        │   └── utils.py
        ├── problems/
        │   └── gemm_hopper_bf16/
        │       ├── problem.py
        │       ├── registry.py
        │       ├── native_extension.py
        │       └── kernels/
        │           ├── reference/
        │           ├── triton/
        │           ├── gluon/
        │           ├── cute_dsl/
        │           └── native_cuda/
        └── ...
```