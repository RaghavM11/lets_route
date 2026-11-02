# lets_route

LLM-routing experimentation repo (hackathon project, `2026-11-02`).

## Project structure

```
lets_route/
├── setup_route_f.sh              # builds the uv-managed "route_f" env (Linux+macOS)
├── .gitignore                    # envs, secrets, data/raw parquet content
├── README.md
├── data/
│   ├── load_xroutebench.py       # lazy loader for xRouteBench (no bulk download)
│   ├── download_to_raw.py        # download all parquet -> data/raw
│   └── raw/                      # dataset skeleton only (.gitkeep per config)
│       ├── llmrouter_generic/
│       ├── llm_candidates/
│       └── … (17 config folders; parquet content is gitignored)
├── docs/
│   ├── README.md                 # docs index
│   ├── xroutebench.md            # dataset reference (configs, schema, usage)
│   └── dataset-directory-structure.md  # where/how dataset files are organized
├── experiments/
│   ├── xroutebench_official_baseline/  # official-matrix router baseline
│   └── qwen3_5_27b_modal/        # Qwen3.5-27B xRouteBench run on Modal H200
├── route_f_playground.ipynb      # env sanity-check notebook (route_f kernel)
└── study_xroutebench.ipynb       # dataset study notebook
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
(LLM-routing benchmark). Full reference and on-disk storage layout live in
`docs/xroutebench.md` and `docs/dataset-directory-structure.md`.
`data/load_xroutebench.py` exposes:

- `list_configs()` / `schema(config, split)` — via the HF API, no download.
- `load(config, split, stream=True)` — **lazy** streaming (default); set
  `stream=False` only when you actually want the full local download.
- `peek(config, split, n)` — stream the first `n` rows only.
- `build_price_table(candidates)` / `row_cost(row, prices)` — price-aware
  routing-cost helpers for the `llm_candidates` pool.

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

## Modal H200 experiment

The reproducible Qwen3.5-27B launcher is documented in
[`experiments/qwen3_5_27b_modal/README.md`](experiments/qwen3_5_27b_modal/README.md).
It uses the existing Modal profile in `~/.modal.toml`, caches the pinned model in
a Modal Volume, runs the six-task multiple-choice suite on one H200, and saves
complete local and remote result artifacts.

