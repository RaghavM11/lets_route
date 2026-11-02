# xRouteBench — Dataset Reference

Source: [ulab-ai/xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench)

**What it is:** a benchmark for building/evaluating **LLM routers** — systems that
pick the best model from a candidate pool for each query, trading off
**performance vs price cost**.

Every query was executed against all **18 candidate LLMs**, recording each model's
response, score, token usage, and latency.

---

## Configs (scenarios)

| Config | What | Splits |
|--------|------|--------|
| `llmrouter_generic` | 13 classic NLP benchmarks (MMLU, GSM8K, MATH, MBPP, ARC, …) | train / test |
| `memory_locomo` | Long-conversation memory QA (RAG top-k=5) | train / test |
| `memory_longmemeval` | Long-term memory eval (RAG top-k=5) | train / test |
| `timeseries` | Time-series understanding (7 sub-tasks) | train / test |
| `video` | Egocentric video QA (Charades-Ego) | train / test |
| `multimodal_geometry3k` | Geometry math (multimodal) | train / test |
| `multimodal_mathvista` | Visual math reasoning | train / test |
| `personalized` | Personalized preference (LLM-judge, chat-format) | train / test |
| `llm_candidates` | The 18-model candidate pool with pricing | train |

Each scenario also ships a raw-query config `<scenario>_queries`
(no model executions) with **train / valid / test** splits — use these to run
your own candidate models from scratch.

---

## Row schema (routing data: `llmrouter_generic` etc.)

Each row = one `(query, candidate model)` pair.

| Field | Type | Meaning |
|-------|------|---------|
| `task_name` | str | Sub-task (e.g. `gsm8k`, `mbpp`) |
| `query` | str | Full input prompt (personalized: JSON-encoded chat messages) |
| `ground_truth` | list/str | Reference answer(s) |
| `metric` | str | Scoring metric for this row |
| `choices` | str | JSON-encoded options for multiple-choice items |
| `task_id` | str | ID within the sub-task |
| `model_name` | str | Candidate LLM this row was executed with |
| `response` | str | The model's actual response |
| `token_num` | int | Total tokens |
| `input_tokens` / `output_tokens` | int | Token breakdown (for price computation) |
| `response_time` | float | Latency in seconds |
| `performance` | float | Task score of this model on this query (0–1) |
| `embedding_id` | int | Index into precomputed query-embedding files (not included) |

### Candidate pool (`llm_candidates`)
Columns: `model_name`, `size`, `input_price_per_1m`, `output_price_per_1m`,
`service`, `api_model_id`, `description`. Prices span $0.05–$1.25 per 1M input
tokens (25× spread). Row cost = `input_tokens × input_price/1e6 + output_tokens × output_price/1e6`.

---

## Metrics seen in `llmrouter_generic`
`em_mc`, `GSM8K`, `code_eval`, `MATH`, `f1_score`.

---

## Usage

```python
import sys; sys.path.insert(0, ".")
from data.load_xroutebench import load, build_price_table, row_cost

# Lazy stream (default, no bulk download):
stream = load("llmrouter_generic", "train", stream=True)

# Full download (for training):
ds = load("llmrouter_generic", "train", stream=False)

# Pricing (for cost-aware routing):
cand = load("llm_candidates", "train", stream=False)
price = build_price_table(cand)
cost = row_cost(ds[0], price)
```

---

## Study highlights (from `notebooks/archive/study_xroutebench.ipynb`)

- `llmrouter_generic` train: **80,802 rows = 4,488 queries × 18 models**.
- **Best model per query:** `qwen2.5-7b-instruct` 58.6%, `gemma-2-9b-it` 23.3%.
- **Cheapest per query:** `gpt-oss-20b` ($0.05/M) 86.1%.
- Mean performance of best model 0.907 vs cheapest 0.680 → a router should
  **default to cheap, escalate to strong** only when needed.
- Query embeddings are NOT shipped — embed `query` yourself with any sentence encoder.

---

## Note
- The full dataset is only ~47 MB — cheap to load fully.
- The memory scenarios use a RAG top-k=5 turn-pair context variant.
