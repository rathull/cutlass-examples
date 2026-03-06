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

- Modal copies `src/cutlass_examples/` into the remote container so shared headers like `common/common.h` are available
- `torch.utils.cpp_extension.load()` only compiles `matmul.cpp` and the selected kernel `.cu` files for that run
- Shared headers are available in the remote container at `/opt/cutlass_examples/common`

Run the benchmark from the repo root:

```bash
uv run modal run src/cutlass_examples/matmul_sm80/main.py --action benchmark --shape 4096,4096,4096 --versions v0,v1
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