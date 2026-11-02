# `hf/` — Hugging Face publishing

This folder manages publishing router releases to the Hugging Face Hub so
others can download and reproduce the exact router.

## Contents

- `push_release.py` — uploads a release dir (adapter + head + tokenizer +
  manifest) to an HF model repo, generating a README model card from the
  release manifest's metrics.

## Publish a release

Login once (token is cached by `huggingface_hub`):

```bash
source .route_f/bin/activate
huggingface-cli login       # or set HF_TOKEN
```

Then push:

```bash
python hf/push_release.py \
    --release training/router_releases/router_8ac97088cc \
    --repo RaghavM11/Fireworks_Hack_JEV_Style_System1_router_base_qwen3.5_0.8B
```

Add `--private` to create the repo as private (default is public). The script
creates the repo if it does not exist.

## What is uploaded

From the release directory:

| Local path | Uploaded to | Purpose |
|---|---|---|
| `adapter/` | `adapter/` | LoRA adapter (PEFT format) |
| `tokenizer/` | `tokenizer/` | matching fast tokenizer |
| `head.pt` | `head.pt` | trained choice-head weights |
| `manifest.json` | `manifest.json` | full config + metrics |
| — (generated) | `README.md` | model card |

## Reproducing from HF

```python
from huggingface_hub import snapshot_download
from router.eval_router import load_release

local = snapshot_download("RaghavM11/Fireworks_Hack_JEV_Style_System1_router_base_qwen3.5_0.8B")
router, tokenizer, manifest = load_release(local)
```
