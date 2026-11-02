# Qwen3.5-27B on Modal H200

This experiment runs the deterministic xRouteBench multiple-choice suite from
`route_f_playground.ipynb` with the official `Qwen/Qwen3.5-27B` checkpoint on
one Modal H200. It evaluates two queries from each of six tasks by default, then
compares Qwen with all 18 recorded xRouteBench candidates on the same query IDs.

The model is pinned to Hugging Face revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The first invocation downloads the
approximately 55.6 GB BF16 checkpoint using a CPU function and stores it in the
`fireworks-qwen35-models-v2` Modal Volume. Later runs reuse that copy. Benchmark
artifacts are persisted in `fireworks-xroute-results-v2` and mirrored to the
local `results/qwen3_5_27b_modal/` directory.

## Prerequisites

1. Keep the working Modal profile in `~/.modal.toml` (or authenticate with
   `modal setup`).
2. Add a valid payment method to the Modal workspace; H200 functions require it.
3. Download the repository data with `data/download_to_raw.py`. The launcher
   uploads the local test matrix and candidate price table so the exact input
   bytes are fingerprinted in every result.

No local Modal install is required when `uvx` is available.

## Run

From the repository root:

```bash
uvx --from modal modal run experiments/qwen3_5_27b_modal/modal_app.py
```

Useful overrides:

```bash
uvx --from modal modal run experiments/qwen3_5_27b_modal/modal_app.py \
  --per-task 10 \
  --max-new-tokens 64 \
  --run-id qwen27b-mc-60
```

To attach your current H200 rate to per-query cost measurements, pass
`--gpu-usd-per-hour RATE`. Leave it unset to avoid recording a stale or
incorrect price. This estimate measures generation time only; Modal also bills
for model loading and container startup.

The default run uses non-thinking mode. Qwen3.5 thinks by default, which is a
bad fit for the notebook's short answer budget and can truncate before the
multiple-choice answer appears.

## Results

Each run writes:

- `summary.json` — configuration, model/data revisions, hardware, accuracy,
  latency, throughput, memory, and timing;
- `query_results.csv` — response and score for every selected query;
- `candidate_comparison.csv` — Qwen and the 18 recorded candidates evaluated
  on identical query IDs.

If the local process is interrupted after the remote job finishes, recover the
persisted artifacts with:

```bash
modal volume ls fireworks-xroute-results-v2
modal volume get fireworks-xroute-results-v2 RUN_ID results/recovered-RUN_ID
```

Run the inexpensive local checks before launching paid compute:

```bash
.route_f/bin/python -m unittest discover \
  -s experiments/qwen3_5_27b_modal -p 'test_*.py'
```
