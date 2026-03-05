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

## Project Structure

I've tested this project on CUDA 12.6.

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