# Five models, ten fixed tasks each: 50 recorded executions

Open `all_50_results.txt` for the complete plain-text report, grouped from
largest to smallest model. Each `01_...txt` through `05_...txt` contains one
model's ten original prompts, references, responses, scores, token counts,
recorded latency, and token-price cost estimates. No responses are truncated
or rewritten. `all_50_results.csv` contains the same records programmatically.

These are **10 unique prompts and 50 model-query executions**, not 50 distinct
questions. They reuse the frozen `../smoke10/tasks.json` manifest. No new GPU
or API inference was run. `data/export_model_examples.py` builds the report
from the local parquet files and verifies the original dataset fingerprint.

`score_matrix.csv` shows the 10-by-5 scores. `model_summary.csv` includes the
ten-task full/zero/partial score counts and each model's mean score over all
4,489 training records. The latter is a dataset-specific strength indicator,
not a universal capability ranking. Scores combine different task metrics;
they are not uniformly binary accuracy. This small, success-conditioned set
must not be used to estimate population accuracy or pick the final router.

The selected scale ladder is Cogito v2.1 (671B MoE total parameters), Llama 3.3
(70B), Mistral Small 3 (24B), Gemma 2 (9B), and Mistral v0.3 (7B). Parameter
count, active compute, price, and empirical score are different quantities.
The selected five have publicly available model weights. Mistral models use
Apache 2.0; Cogito uses MIT; Llama and Gemma have their own model licenses.
Open weights do not imply identical licensing or fully open training data.

Official model cards:

- https://huggingface.co/deepcogito/cogito-671b-v2.1
- https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
- https://huggingface.co/google/gemma-2-9b-it
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

Recorded service variants such as Together's Turbo use provider-specific API
names; their underlying public model card does not guarantee that a self-hosted
checkpoint reproduces the recorded serving precision or decoding behavior.
The dataset's Cogito API ID differs from its official Hugging Face repo name;
both are retained without claiming a pinned checkpoint identity.

Scoring caveats visible in these examples:

- `smoke-02`: Mistral 7B returns `I. Ransom Trojan`, matching the reference
  letter semantically, but its recorded score is zero. A format/extraction issue
  is plausible; the scoring implementation was not audited here.
- `smoke-10`: SQuAD uses token F1. All five outputs mention the reference
  `ringers` but have partial scores, so partial is kept separate from failure.
- Large models can score zero on queries where smaller models receive one;
  model size is not a per-query correctness oracle.

`all_18_model_inventory.csv` lists the full recorded candidate pool, its
provider metadata, prices, and training mean scores. Its original `size` field
is dataset metadata, not a separately verified parameter audit.
