# xRouteBench official-matrix baseline

This experiment starts from the complete `llmrouter_generic` matrix and keeps
all 18 candidate models. It is intentionally conservative: establish a clean,
repeatable baseline close to xRouteBench before changing candidate pools,
objectives, encoders, or routing policies.

## Files

- `xroutebench_official_baseline.ipynb` — the experiment walkthrough.
- `baseline.py` — tested data, feature, router, and evaluation helpers used by
  the notebook.

## Data policy

- Preserve the official xRouteBench `train` and `test` splits.
- Create validation only from official training queries.
- Use split-local `embedding_id` as the query-matrix key. The dataset's
  `task_id` is missing or repeated for multiple tasks and is not a safe key.
- Group identical prompt hashes when creating validation so exact duplicate
  prompts cannot cross train and validation.
- Keep all 18 recorded model outcomes for every query.
- Never use response, ground truth, performance, cost, latency, model name, or
  task label as router input.

## Primary objective

The notebook follows the benchmark's quality-cost framing:

`utility = alpha * performance - beta * normalized_cost`

It evaluates a sweep of `(alpha, beta)` values. Latency is reported but is not
part of the initial decision rule. Cost is normalized by the training-only 95th
percentile, and that choice is recorded explicitly so later experiments can
replace it without ambiguity.

## Baselines

For each quality-cost setting the notebook reports:

1. cheapest fixed model;
2. best fixed model selected on validation;
3. seeded uniform random routing;
4. task lookup learned from the training subset;
5. a lightweight learned System-1 router;
6. the per-query test oracle upper bound.

## Run

Use the repository's `route_f` kernel and open
`xroutebench_official_baseline.ipynb`. The experiment uses only packages already
installed by `setup_route_f.sh` and the local Parquet files under
`data/raw/llmrouter_generic` and `data/raw/llm_candidates`.

