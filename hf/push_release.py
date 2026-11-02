#!/usr/bin/env python
"""Publish a System-1 router release to Hugging Face Hub.

Uploads the full release directory (LoRA adapter, choice head, tokenizer,
manifest) to a HF model repo so others can download and reproduce the router.

Usage:
    python hf/push_release.py --release training/router_releases/router_8ac97088cc \
        --repo RaghavM11/Fireworks_Hack_JEV_Style_System1_router_base_qwen3.5_0.8B \
        [--private] [--token <hf_token>]

Requires:
    - huggingface_hub installed (in the route_f env)
    - HF authentication (HF_TOKEN env var OR cached `huggingface-cli login`)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


def build_model_card(manifest: dict, results: dict) -> str:
    """Generate a README.md model card for the router release."""
    head = manifest["head_config"]
    pool = manifest["model_pool"]
    gen = results.get("gen_baseline", {})
    tuned = results.get("trained_decision_model", {})
    n_pool = len(pool)
    lines = [
        "---",
        "language:",
        "  - en",
        "license: apache-2.0",
        "tags:",
        "  - llm-router",
        "  - xroutebench",
        "  - lora",
        "  - system-1",
        "  - choice-head",
        "base_model: " + manifest["model_id"],
        "---",
        "",
        "# System-1 Router — " + manifest["model_id"],
        "",
        "A compact **System-1 router** that picks which candidate LLM should "
        "answer a query in a single forward pass. Trained against the "
        "counterfactual oracle labels from xRouteBench (`llmrouter_generic`).",
        "",
        "## What is stored here",
        "",
        f"- **Base model:** `{manifest['model_id']}` (frozen backbone, {manifest['model_revision']})",
        f"- **LoRA adapter:** [`adapter/`](./adapter) — target modules: the language-model "
        f"linear layers (r=16, alpha=32)",
        f"- **Choice head:** [`head.pt`](./head.pt) — `ChoiceHead` "
        f"(width={head['width']}, heads={head['num_heads']}, joint_layers={head['joint_layers']})",
        f"- **Tokenizer:** [`tokenizer/`](./tokenizer)",
        f"- **Manifest:** [`manifest.json`](./manifest.json) — full config + metrics",
        "",
        f"The router chooses among **{n_pool} candidate models**. Backbone + adapter "
        "encode the query; the choice head scores each candidate option.",
        "",
        "## Metrics (held-out test)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Exact oracle match | **{tuned.get('oracle_acc'):.3f}** |",
        f"| Within acceptable set | **{tuned.get('acceptable_hit'):.3f}** |",
        f"| Batched decision latency | {tuned.get('batched_ms_per_query', 0):.1f} ms/query |",
        f"| Generative baseline (oracle match) | {gen.get('oracle_acc', 0):.2f} |",
        f"| Generative baseline (acceptable hit) | {gen.get('acceptable_hit', 0):.2f} |",
        "",
        "## Reproduction",
        "",
        "1. Clone the repo and set up the `route_f` env (see root README).",
        "2. Load the release:",
        "",
        "```python",
        "import sys; sys.path.insert(0, '.')",
        "from router.eval_router import load_release",
        "router, tokenizer, manifest = load_release('training/router_releases/router_8ac97088cc')",
        "```",
        "",
        "3. Or download this HF repo and point `load_release` at the local copy.",
        "",
        "## Files",
        "",
        "- `manifest.json` — training config, hyperparameters, and metrics",
        "- `head.pt` — the trained choice-head weights (torch state_dict)",
        "- `adapter/` — LoRA adapter (PEFT format)",
        "- `tokenizer/` — matching fast tokenizer",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path, help="Path to the release directory")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. owner/repo")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--token", default=None, help="HF token (defaults to cached env)")
    args = ap.parse_args()

    release = args.release.resolve()
    if not (release / "manifest.json").exists():
        sys.exit(f"release dir missing manifest.json: {release}")

    manifest = json.loads((release / "manifest.json").read_text())
    results = manifest.get("metrics", {})
    card = build_model_card(manifest, results)

    api = HfApi(token=args.token)

    try:
        api.repo_info(args.repo, repo_type="model")
        print(f"repo exists: {args.repo}")
    except RepositoryNotFoundError:
        print(f"creating repo: {args.repo} (private={args.private})")
        api.create_repo(args.repo, repo_type="model", private=args.private)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for sub in ("adapter", "tokenizer"):
            shutil.copytree(release / sub, staging / sub)
        shutil.copy2(release / "head.pt", staging / "head.pt")
        shutil.copy2(release / "manifest.json", staging / "manifest.json")
        (staging / "README.md").write_text(card)

        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Publish System-1 router release (LoRA + choice head + tokenizer)",
        )

    print(f"published → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
