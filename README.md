# lets_route

LLM-routing experimentation repo (hackathon project, `2026-11-02`).

## Project structure

```
setup_route_f.sh          # builds the uv-managed "route_f" environment
data/load_xroutebench.py  # lazy loader for the xRouteBench dataset (no bulk download)
route_f_playground.ipynb  # notebook on the route_f kernel
.gitignore                # ignores environments / caches
```

## Environment — `route_f`

A **uv**-managed Python 3.12 virtual environment, identical on **Linux (CUDA)**
and **macOS (MPS/CPU)**. This is the single env used by both the terminal and
the VS Code notebook kernel.

**Verified stack (on this Linux box):**
- torch **2.15.0+cu132** — CUDA available (RTX 3060)
- datasets **5.1.0** · transformers **5.22.1** · peft **0.21.2** · accelerate **1.15.0**
- huggingface_hub, safetensors, tokenizers, numpy, pandas, pyarrow, ipykernel

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (v0.11.x)
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - or `brew install uv` (macOS)

### Create the environment

```bash
chmod +x setup_route_f.sh
./setup_route_f.sh
```

The script:
1. Creates `.route_f/` (Python 3.12).
2. Installs **torch 2.15.0** from PyPI — CUDA is bundled on Linux,
   MPS/CPU on macOS; no extra index needed.
3. Installs the core ML stack.
4. Registers the Jupyter kernel **route_f**.

### Use it (same kernel everywhere)

- **Terminal:** `source .route_f/bin/activate`
- **VS Code notebook:** select the **`route_f (py3.12)`** kernel in the picker
  (registered per-user, so it appears on both machines).

### Notes

- `.route_f/` lives only locally (already in `.gitignore`) — never commit it.
- On macOS the backend is `device="mps"`; on Linux it's `device="cuda"`.
- **macOS build difference:** torch on macOS is CPU/MPS, so the wheels are
  smaller and no NVIDIA libs are installed.

## Dataset — xRouteBench

See [ulab-ai/xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench)
(LLM-routing benchmark). `data/load_xroutebench.py` exposes:

- `list_configs()` / `schema(config, split)` — via the HF API, no download.
- `load(config, split, stream=True)` — **lazy** streaming (default); set
  `stream=False` only when you actually want the full local download.
- `peek(config, split, n)` — stream the first `n` rows only.
- `row_cost(row, prices)` — price-aware routing cost helper.

```python
import sys; sys.path.insert(0, ".")
from data.load_xroutebench import load

stream = load("llmrouter_generic", "train", stream=True)  # lazy, no bulk download
ds     = load("llmrouter_generic", "train", stream=False) # full download (training)
```

Main configs: `llmrouter_generic`, `memory_locomo`, `memory_longmemeval`,
`timeseries`, `video`, `multimodal_geometry3k`, `multimodal_mathvista`,
`personalized`, plus raw-query `*_queries` variants and `llm_candidates`
(pricing pool).

