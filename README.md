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
├── hf/
│   ├── push_release.py           # publish a release to Hugging Face Hub
│   └── README.md                 # HF publishing guide
├── experiments/
│   ├── xroutebench_official_baseline/  # official-matrix router baseline
│   ├── qwen3_5_27b_modal/        # Qwen3.5-27B xRouteBench run on Modal H200
│   └── system1_router_full_h200/  # full-dataset System-1 router (Qwen3.5-9B, Modal H200)
├── notebooks/archive/
│   ├── route_f_playground.ipynb  # env sanity-check notebook (route_f kernel)
│   └── study_xroutebench.ipynb   # dataset study notebook
├── router/
│   ├── build_oracle_matrix.py    # counterfactual matrix + oracle labels
│   ├── prepare_dataset.py        # stratified cost/perf splits
│   ├── train_lora.py             # LoRA training script
│   └── metrics.py                # shared aggregate helpers
└── training/
    └── system1_router_baseline.ipynb  # Qwen LoRA System-1 baseline
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

## Counterfactual Matrix (Ground Truth)

The router's training signal is a **counterfactual query × model matrix** derived
from the benchmark's own output structure (see `docs/dataset-directory-structure.md`):

```
data/raw/llmrouter_generic/<split>-000.parquet   # one row per (query, model) execution
data/raw/llm_candidates/train-000.parquet        # 18-model pricing pool
```

### 1. Build the matrix

For each query `q` (unique `embedding_id`/`task_id`) and each model `m`, pull:

| Field | Where | Meaning |
|-------|-------|---------|
| `performance` `y_{q,m}` | routing parquet | task score of `m` on `q` (0–1) |
| `cost` `c_{q,m}` | `llm_candidates` × token counts | realized per-row cost |
| `task_name`, `metric` | routing parquet | group queries for eval splits |

`performance` values are normalized **per metric** (e.g. `em_mc`, `GSM8K`,
`code_eval`, `MATH`, `f1_score`) before use as labels.

### 2. Oracle target (ε-optimal cheapest-tier)

For each query, the acceptable set keeps models within `ε` of the best score:

```
A_ε(q) = { m : y_{q,m} ≥ max_k y_{q,k} − ε }
target(q) = argmin_{m ∈ A_ε(q)} c_{q,m}      # cheapest acceptable model
```

If `max_k y_{q,k} = 0` (no model solves it), default to the cheapest model.

### 3. Labels for training

- **Multi-label BCE** on the raw matrix `y_{q,m}` (every model that solves the
  query is a valid target — avoids penalizing a bigger model that also got it right).
- **Oracle class** `target(q)` for a hard/soft decision head (cost-aware routing).

### 4. Pipeline order

1. Materialize the matrix + oracle labels (`data/` or a notebook).
2. **Lock evals & metrics** — a fixed held-out evaluation recipe
   (task-stratified sample, exact-score MC extraction, cost/latency accounting).
3. Train the System-1 router head against the oracle labels.
4. Evaluate vs fixed baselines (always-cheapest, always-best-category, official baseline).

## Initial results — System-1 router baseline

Trained a **Qwen3.5-0.8B** backbone (frozen) + LoRA + joint choice-head router
against the xRouteBench counterfactual oracle (`llmrouter_generic`, ε-optimal
cheapest-tier). See `training/system1_router_baseline.ipynb`.

**Setup:** 1,200 train queries · 200 held-out test queries · 400 steps ×
batch 4 (1,600 presentations) · ~19 min on one RTX 3060.

| Metric | Training (200 test queries) |
|---|---|
| **Exact oracle match** (`pred == oracle_target`) | **0.73** |
| **Within acceptable set** (`pred ∈ acceptable_models`) | **0.795** |
| Batched decision latency | ~201 ms/query |
| Generative baseline (oracle match) | 0.00 (192/200 invalid JSON) |
| Generative baseline (acceptable hit) | 0.02 |

Realized cost/performance vs the oracle on the same test split
(`router/eval_router.py`):

| Quantity | Ground truth model (oracle) | Our router |
|---|---|---|
| Mean per-query performance | 0.922 | **0.717** |
| Mean per-query cost | $28.8e-6 | **$11.9e-6** |

**Takeaways**
- The router matches the oracle's exact cheapest-acceptable pick 73% of the
  time and lands inside the acceptable set 79.5% — far above the generative
  baseline, which emitted invalid JSON on 192/200 queries.
- It runs at roughly **half the oracle's per-query cost** while giving up some
  peak accuracy — the intended cost/accuracy tradeoff.
- The decision head currently collapses onto cheap small models (gpt-oss-20b,
  gemma-2-9b-it); distinguishing *when* to escalate to strong/expensive
  candidates is the main open improvement.

The published release (adapter + head + tokenizer + manifest) lives on HF:
`RaghavM11/Fireworks_Hack_JEV_Style_System1_router_base_qwen3.5_0.8B`
(see `hf/` for the publishing tooling).

**Modal run tracking:** all training runs persist their releases into the
`route-training` Modal Volume (`/releases/router_<uuid>/`); the run log
(volume path ↔ run details ↔ metrics) is maintained in
[`experiments/system1_router_full_h200/README.md`](experiments/system1_router_full_h200/README.md#run-log--volume-vs-run-details).

## Modal H200 experiment

The reproducible Qwen3.5-27B launcher is documented in
[`experiments/qwen3_5_27b_modal/README.md`](experiments/qwen3_5_27b_modal/README.md).
It uses SGLang, caches the pinned model in a Modal Volume, runs the six-task
multiple-choice suite on one H200, and saves complete local and remote result
artifacts.

## Checkpoint & rollback

| Ref | Hash | What |
|-----|------|------|
| tag `checkpoint-baseline` | `d04dbb3` (`d04dbb3329f33987087e61c311e2aa4e10d398cd`) | dataset + router + training code baseline, before training experiments |

To restore if later changes break something:

```bash
git reset --hard d04dbb3        # or: git checkout checkpoint-baseline
```

