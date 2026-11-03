# System-1 Router — Full Dataset Run on Modal H200

Full-scale training of the System-1 router on a **Qwen3.5-9B** backbone using
an **H200** GPU on Modal. This is the "big sibling" of the 0.8B baseline
(`training/system1_router_baseline.ipynb`), run over the **entire** dataset to
test the hypothesis that a larger backbone makes a better router.

## Scope

| | Baseline (0.8B) | This full run |
|---|---|---|
| Backbone | Qwen3.5-0.8B | **Qwen3.5-9B** |
| Training data | 1,200 sampled queries | **train + valid (4,488 queries)** |
| Held-out test | 200 | **full test (3,729 queries)** |
| Steps | 400 | **2,244 (~2 epochs, batch 4)** |
| GPU | RTX 3060 (local) | **Modal H200** |

## Files

- `modal_train.py` — the Modal app (adapted from
  `training/modal_train.py`): trains on H200, saves the release **inside a
  Modal Volume**, and can re-evaluate a saved release.
- `system1_router_full_qwen35_9b.ipynb` — the replicated/renamed baseline
  notebook, updated for the 9B full-run config (useful for local inspection;
  the heavy run happens on Modal).

## Run on Modal

One-time setup (unchanged from the baseline Modal app):

```bash
# 1. HF token secret (uses your local HF token)
modal secret create hf-token HF_TOKEN=$(cat ~/.cache/huggingface/token)

# 2. sync the prepared splits into the route-training Volume
modal volume create route-training
modal volume put route-training data/processed /processed
```

Launch the training:

```bash
cd experiments/system1_router_full_h200

# interactive (shows progress)
modal run modal_train.py

# detached (survives terminal close)
modal run modal_train.py --detach
```

## Where results live (Volume persistence)

The trained release is written **inside the Modal Volume** at
`/data/releases/router_<uuid>/` containing:

- `adapter/` — the LoRA adapter (PEFT format)
- `tokenizer/` — matching tokenizer
- `head.pt` — the trained choice-head weights
- `manifest.json` — model id, hyperparameters, training seconds, held-out
  oracle_acc / acceptable_hit

The volume (`route-training`) persists across runs — the artifact is not lost
if the container dies or the terminal closes.

## Get the results back

List releases and pull the newest one:

```bash
modal volume ls route-training /releases          # find the newest router_xxxx
modal volume get route-training /releases/router_<uuid> ./downloaded_release
```

Then load/evaluate it locally or push to HF:

```python
from router.eval_router import load_release, router_vs_gt
router, tok, manifest = load_release("./downloaded_release/router_<uuid>")
results = router_vs_gt("./downloaded_release/router_<uuid>")   # vs ground-truth oracle
```

## Run log — volume vs run details

Every training run writes its release into the **`route-training` Volume**
under `/releases/router_<uuid>/`. The manifest captures the run config and
held-out metrics. We track runs here.

### Volume layout

| Path (in `route-training` Volume) | Contents |
|---|---|
| `/processed/routing_train.jsonl` | training split (train+valid source) |
| `/processed/routing_valid.jsonl` | validation split |
| `/processed/routing_test.jsonl` | held-out test split (3,729 queries) |
| `/releases/router_<uuid>/adapter` | LoRA adapter (PEFT) |
| `/releases/router_<uuid>/head.pt` | trained choice-head weights |
| `/releases/router_<uuid>/tokenizer` | matching tokenizer |
| `/releases/router_<uuid>/manifest.json` | run config + metrics |

### Run log

| # | Date | Backbone | Scope | Steps | Volume release | oracle_acc | acceptable_hit | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-02 | Qwen3.5-0.8B | 1,200 train / 200 test | 400 | `releases/router_0c98a5eb3d` | 0.730 | 0.795 | ✅ done (baseline) |
| 2 | 2026-11-02 | Qwen3.5-9B | fast-test: 100 train / 32 test | 12 | `releases/router_7fda15b1ca` | 0.688* | 0.875* | ✅ done (pipeline smoke) |
| 3 | 2026-11-02 | Qwen3.5-9B | 1,000 train / full test 3,729 | 125 | `releases/router_563c668b78` | 0.706 | 0.773 | ✅ done (9B / 1k prompts) |
| 4 | — | Qwen3.5-9B | full: 4,488 train / 3,729 test | 2,244 | _to be filled_ | — | — | ⏳ planned |
| 5 | 2026-11-02 | Qwen3.5-27B | fast-test: 1,000 train / 200 test | 250 | _picking_ | — | — | ⏳ running |

\* Fast-test run is a 12-step smoke test on a tiny subset — numbers are not
conclusive; it only validates the H200 pipeline end-to-end.

\** Row 3 metrics are the full held-out evaluation on the complete test split
(3,729 queries). The training-time subset eval (200 queries) for the same run
recorded oracle_acc 0.740 / acceptable_hit 0.815.

**How to keep this log current after a new run:** after training finishes,
run:

```bash
modal volume ls route-training /releases        # find newest router_<uuid>
modal volume get route-training /releases/router_<uuid>/manifest.json ./manifest.json
cat manifest.json
```

Then add a row above with the manifest's `steps`, `train_size`/`test_size`,
and `held_out.oracle_acc` / `held_out.acceptable_hit`.

## Expected outcome

Larger backbone (9B vs 0.8B) should improve exact oracle match beyond the
0.8B baseline (0.73) by learning better query-difficulty discrimination —
i.e., knowing *when* to escalate to strong/expensive candidates instead of
always defaulting to the cheap small models (gpt-oss-20b / gemma-2-9b-it).
