# Counterfactual utility-driven router: agreed experimental context

Source: the user's whiteboard formulation and
`/Users/a511/Downloads/message-2.txt`, supplied November 2, 2026.

## Objective and order

Immediate checkpoint requested by the user: freeze `fixtures/smoke10/tasks.json`
and use these exact ten prompts before expanding to larger calibration runs.
Its labels come from the 18 recorded candidates and are not live-tier labels.

Route each query to the cheapest acceptable model. Start with a small scored
serving smoke test, then run a shared task-stratified sample (500–1,000 initially,
2,000 as the calibration target) through every candidate tier. Produce a
query-by-model score and cost matrix before training a router.

The desired capacity tiers are approximately 0.8B, 7B, 27B, and 70B. The exact
70B-class checkpoint remains to be selected. Model generation, revision,
precision, GPU type, decoding settings, and thinking mode must be recorded.
Existing xRouteBench outputs for 18 candidates cannot supply supervision for
new model checkpoints that were not evaluated in that table.

## Data and identity

The downloaded raw `llmrouter_generic_queries` config contains 4,489 train,
523 validation, and 3,729 test rows (8,741 total). Counts quoted in earlier
discussion were approximate and sometimes refer to a different config.
Keep official splits. Define `query_uid` from a stable hash of the prompt,
audit overlap across splits, and use the same ordered manifest for every tier.
Do not train on held-out test outcomes. Exclude ground truth from model input.

Store prompt, output, task/category, reference, evaluation metric, input/output
tokens, latency, model revision, and run settings for every query/model pair.
Persist errors separately from incorrect answers so missing runs do not become
false negatives. Score each task with its actual metric; code evaluation needs
an isolated execution environment and is not implemented by MC answer parsing.

## Cost and supervision

Read recorded performance/token usage and model-specific prices for the
offline baseline. Live GPU cost needs measured billable wall time and an
explicit rate. Token counts are exact usage, not an exact dollar cost. Track
generation-only estimates separately from startup/idle/CPU/memory overhead.

Train a multi-output router using the raw correctness/score vector and
multi-label BCE. Support a linear/MLP (`lm_head`) head and a cosine-prototype
(`cosine`) head on prompt representations. No router training is implemented
by the serving scaffold.

For each query, define the acceptable set as models within epsilon of its best
score and choose its lowest-cost member. If all models score zero, use the
predeclared cheapest tier as fallback. Freeze epsilon, cost units, and
tie-breaking before comparing runs. A new cosine prototype still requires
calibration and validation; extensibility is not automatic accuracy.

## Evaluation

Sweep the utility cost weight lambda or confidence threshold tau. Report mean
accuracy, mean cost, and p95 latency/Pareto frontiers on held-out queries.
Compare always-cheapest, strongest-model, best-per-category learned using
training data, and the repository's xRouteBench baseline. Keep offline and live
candidate pools explicit; do not compare them as though they were identical.

## Operational status

The local branch was fast-forwarded to the fetched upstream baseline; prior
Mercury and Intel Mac environment changes were reapplied without committing.
The original recovery stash is retained. The dataset configs needed for the
first experiment are downloaded under `data/raw/`. Modal login and GPU runtime
validation are tracked separately from code preparation; a scaffold is not a
verified running endpoint.
