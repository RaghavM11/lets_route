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

## Expected outcome

Larger backbone (9B vs 0.8B) should improve exact oracle match beyond the
0.8B baseline (0.73) by learning better query-difficulty discrimination —
i.e., knowing *when* to escalate to strong/expensive candidates instead of
always defaulting to the cheap small models (gpt-oss-20b / gemma-2-9b-it).
