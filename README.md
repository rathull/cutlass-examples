# CUTLASS Examples

## Getting Started

Intstall `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Running `matmul_sm80`

The `matmul_sm80` benchmark uses a minimal JIT setup:

- Note on Modal: Python modules under `src/cutlass_examples/` use package-relative imports such as `from ..common.utils import ...`
- Run Modal in module mode so those imports stay valid: `modal run -m cutlass_examples.matmul_sm80.main`
- Modal will include the Python package for the function automatically, but native sources still need to be mounted explicitly for `torch.utils.cpp_extension.load()`
- Shared native headers are mounted in the remote container at `/opt/cutlass_examples/common`

Run the benchmark from the repo root:

```bash
uv run modal run -m cutlass_examples.matmul_sm80.main --action benchmark --shape 4096,4096,4096 --versions v0,v1

# Adjust parameters as needed
uv run modal run -m cutlass_examples.matmul_sm80.main \
  --action benchmark \
  --shape 4096,4096,4096 \
  --versions v0,v1 \
  --warmup-runs 50 \
  --bench-runs 300 \
  --quantiles 0.20,0.50,0.80,0.90,0.95,0.99 \
  --benchmark-ref true \
  --check-correctness true
```

## Project Structure

Benchmarking setup from [gau-nernst's learn-cuda](https://github.com/gau-nernst/learn-cuda/blob/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100/main.py)

```
cutlass-examples/
├── pyproject.toml
├── uv.lock
└── src/
    └── cutlass_examples/
        ├── __init__.py
        ├── common/           # Your shared C++/CuTe headers
        │   ├── __init__.py
        │   ├── common.h
        │   └── profiler.h
        ├── matmul_sm100/
        │   ├── __init__.py
        │   ├── main.py       # Modal App & orchestrator
        │   ├── matmul.cpp    # PyTorch bindings
        │   ├── matmul_v0.cu  # Baseline
        │   └── matmul_v7.cu  # Final
        └── attention/
            └── ...
```