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
  --problem gemm_hopper \
  --gpu h100 \
  --kernels all
```

Run a single benchmark:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper \
  --gpu h100 \
  --kernels cuBLAS,hopper_triton_v0 \
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
  --problem gemm_hopper \
  --gpu h100 \
  --kernels all \
  --tags hopper \
  --exclude-tags smoke \
  --shapes '1024,1024,1024;2048,2048,2048;4096,4096,4096' \
  --repetitions 3 \
  --parallel true \
  --out artifacts/runs/hopper-sweep
```

Inspect native CUDA resource usage from `ptxas -v` for kernels that expose a
`ptxas` inspection hook:

```bash
uv run modal run -m cutlass_examples.cli \
  --command ptxas \
  --problem gemm_hopper \
  --gpu h100 \
  --kernels <native_kernel_name> \
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
spill stores/loads, constant memory, warnings, and the raw `ptxas` log. This
only applies to native CUDA/CuTe C++ kernels with a registered `ptxas` hook.

Each run writes:

- `results.json`: full structured run metadata and stats
- `results.jsonl`: append-friendly per-kernel rows
- `results.csv`: spreadsheet-friendly per-kernel rows

## Adding Kernels

Add kernels under `src/cutlass_examples/problems/gemm_hopper/backends/<kind>/`,
then register them in `src/cutlass_examples/problems/gemm_hopper/registry.py`.

The registry entry is the user-facing handle:

```python
KernelSpec(
    name="hopper_triton_v1",
    problem="gemm_hopper",
    kind="triton",
    target="cutlass_examples.problems.gemm_hopper.backends.triton.kernels:matmul_v1",
    supported_gpus=("H100", "H200", "B200"),
    tags=("hopper", "triton"),
)
```

After that, run it by name:

```bash
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper \
  --gpu h100 \
  --kernels hopper_triton_v1 \
  --shapes 4096,4096,4096
```

The CLI handles implementation details from `KernelSpec.kind`: native CUDA/CuTe
C++ extensions, Triton, Gluon, CuTe DSL, and reference kernels all use the same
benchmark path.

## Tags

Tags are labels attached to each `KernelSpec`. They let you select groups of
kernels without remembering every kernel name.

Examples:

- `hopper`: kernels intended for Hopper/H100 or H200.
- `blackwell`: kernels expected to run on B200.
- `triton`, `gluon`, `cute_dsl`, `native_cuda`: implementation families.
- `smoke`: dependency/runtime smoke kernels, useful for checking setup but not
  meaningful for performance comparison.
- `baseline` or `reference`: cuBLAS/reference kernels.

Selection rules:

- `--kernels all --tags hopper` selects kernels that have the `hopper` tag.
- `--kernels all --tags hopper,triton` selects kernels that have both tags.
- `--exclude-tags smoke` removes smoke kernels from the selected set.
- Explicit names still work: `--kernels cuBLAS,hopper_triton_v0`.

Useful examples:

```bash
# Run real Hopper kernels, but skip smoke placeholders.
uv run modal run -m cutlass_examples.cli \
  --command benchmark \
  --problem gemm_hopper \
  --gpu h100 \
  --kernels all \
  --tags hopper \
  --exclude-tags smoke

# See all Triton kernels available for H100.
uv run modal run -m cutlass_examples.cli \
  --command list-kernels \
  --problem gemm_hopper \
  --gpu h100 \
  --kernels all \
  --tags triton
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
        │   └── gemm_hopper/
        │       ├── problem.py
        │       ├── registry.py
        │       └── backends/
        │           ├── triton/
        │           ├── gluon/
        │           ├── cute_dsl/
        │           └── reference/
        └── ...
```