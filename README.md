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
├── models/mercury.py             # Inception Mercury 3 API client
├── experiments/
│   ├── xroutebench_official_baseline/  # official-matrix router baseline
│   └── qwen3_5_27b_modal/        # Qwen3.5-27B xRouteBench run on Modal H200
├── route_f_playground.ipynb      # env sanity-check notebook (route_f kernel)
└── study_xroutebench.ipynb       # dataset study notebook
```

## Environment — `route_f`

A **uv**-managed Python 3.12 virtual environment for **Linux (CUDA)**
and **macOS (MPS/CPU)**. Intel Macs use a compatible older ML stack, described
below. This is the single env used by both the terminal and
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
2. Installs **torch 2.15.0** from PyPI on Linux and Apple Silicon Macs.
   Intel Macs use **torch 2.2.2**, **NumPy 1.x**, **transformers 4.44.2**,
   **peft 0.13.2**, **datasets 3.6.0**, and **pyarrow 18.1.0** because newer
   PyTorch releases lack Intel macOS wheels. The Arrow pin preserves NumPy 1.x
   compatibility.
3. Installs the core ML stack.
4. Registers the Jupyter kernel **route_f**.

### Use it (same kernel everywhere)

- **Terminal:** `source .route_f/bin/activate`
- **VS Code notebook:** select the **`route_f (py3.12)`** kernel in the picker
  (registered per-user, so it appears on both machines).

### Notes

- `.route_f/` lives only locally (already in `.gitignore`) — never commit it.
- Choose `device="cuda"` if `torch.cuda.is_available()`, otherwise
  `device="mps"` if `torch.backends.mps.is_available()`, otherwise `device="cpu"`.
- The setup can be rerun to finish an interrupted installation; it reuses
  the existing environment.
- **macOS build difference:** torch on macOS is CPU/MPS, so the wheels are
  smaller and no NVIDIA libs are installed.

## Model — Mercury 3

The client reads `INCEPTION_API_KEY` from your environment or a local `.env`
file (ignored by Git). See `.env.example` for the variable name.

```bash
source .route_f/bin/activate
python -m models.mercury "What is a diffusion model?"
```

Use it from Python or the notebook:

```python
from models.mercury import chat

response = chat([{"role": "user", "content": "What is a diffusion model?"}])
print(response["choices"][0]["message"]["content"])
```

Requests use `mercury-3`, default to `reasoning_effort="low"`, time out after
60 seconds, and raise on HTTP errors. Override reasoning effort or timeout
with the matching `chat()` keyword argument.

## Fixed English dataset examples

Start with `fixtures/model_examples50/all_50_results.txt` to read five models'
original responses to the same ten fixed prompts. Each model also has its own
plain-text report in that folder. `fixtures/smoke10/tasks.json` is the fixed
input manifest. These are small curated examples; full downloaded datasets
under `data/raw/` remain local and are ignored by Git.

All saved documentation, generated report labels, and annotations use English.
Original dataset prompts, references, responses, and scores are preserved.
See `AGENTS.md` for the persistent repository convention.

For SGLang serving on Modal, see `experiments/sglang_modal/README.md`.
The experimental workflow is recorded in `docs/experiment-plan.md`.

The completed four-tier Qwen smoke test contains 40 live executions on those
same ten prompts. Read the English report at
[`results/sglang_fixed_ten/qwen-four-fixed-ten-20261102/all_results.txt`](results/sglang_fixed_ten/qwen-four-fixed-ten-20261102/all_results.txt).
Its local scores, token usage, latency matrices, and raw completion responses
are included alongside the report. Scores use the documented local evaluator.

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
It uses SGLang, caches the pinned model in a Modal Volume, runs the six-task
multiple-choice suite on one H200, and saves complete local and remote result
artifacts.

