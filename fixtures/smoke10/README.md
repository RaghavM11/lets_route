# Fixed ten-task smoke set

Use `tasks.json` as the input manifest for every model. Do not resample between
tiers. `tasks.txt` is the readable version, `labels.csv` the label summary, and
`recorded_matrix.csv` the complete 10-by-18 recorded outcome table.

Each task comes from the training execution table, has all 18 candidate
records, has at least one fully correct response, and has no exact-prompt hash
overlap with the raw validation or test splits. Ten task sources are represented
with one query each, selected deterministically with seed 42. This is a
success-conditioned smoke set, not a representative benchmark or held-out test.

The label minimizes recorded token cost among models with performance 1.
Ties are broken lexicographically by model name. Costs use the dataset's
candidate prices, not Modal GPU costs. A cheapest-correct label need not name
the smallest-parameter model. These labels describe the recorded 18 candidates;
they do not establish performance for new SGLang checkpoints, revisions, or
decoding settings.

The manifest retains full SHA-256 query identifiers and input-file fingerprints.
`data/freeze_smoke_tasks.py` refuses to overwrite the frozen set.

Only `query` should be sent as the model prompt. Keep `ground_truth`, labels,
and recorded answers in the offline scorer; never include them in the request.
