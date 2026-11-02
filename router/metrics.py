#!/usr/bin/env python
"""Shared metrics/aggregation for router analysis and notebooks.

Single source of truth for:
- loading the prepared JSONL splits,
- computing per-candidate accuracy/cost aggregates from the raw parquet.

Notebooks should import these instead of re-implementing the same logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIG = "llmrouter_generic"
CANDIDATES_CFG = "llm_candidates"
SPLIT_FILES = {
    "train": "routing_train.jsonl",
    "valid": "routing_valid.jsonl",
    "test": "routing_test.jsonl",
}


def repo_root(cwd: Path | None = None) -> Path:
    """Find the repo root by searching upward for the data/raw marker."""
    cwd = cwd or Path.cwd()
    return next(
        (p for p in [cwd, *cwd.parents] if (p / "data" / "raw").exists()), cwd
    )


def load_splits(root: Path | None = None, split: str | None = None):
    """Load prepared JSONL splits from data/processed.

    Returns dict {train, valid, test} or a single list if `split` is given.
    """
    root = root or repo_root()
    proc = root / "data" / "processed"
    out = {}
    for key, fname in SPLIT_FILES.items():
        path = proc / fname
        with open(path) as fh:
            out[key] = [json.loads(line) for line in fh if line.strip()]
    return out[split] if split else out


def routing_aggregate(splits: tuple[str, ...] = ("train", "test")) -> pd.DataFrame:
    """Per-candidate accuracy/cost aggregate from raw routing parquet.

    Returns DataFrame indexed by model_name with columns:
      accuracy, mean_cost, rows
    (accuracy = mean performance; mean_cost = mean per-row USD cost).
    """
    root = repo_root()
    raw = root / "data" / "raw"

    frames = []
    for split in splits:
        path = raw / CONFIG / f"{split}-000.parquet"
        frames.append(pd.read_parquet(path))
    routing = pd.concat(frames, ignore_index=True)

    cand = pd.read_parquet(raw / CANDIDATES_CFG / "train-000.parquet")
    price = cand.set_index("model_name")[
        ["input_price_per_1m", "output_price_per_1m"]
    ]
    df = routing.merge(price, left_on="model_name", right_index=True, how="left")
    df["cost_usd"] = (
        df["input_tokens"] * df["input_price_per_1m"]
        + df["output_tokens"] * df["output_price_per_1m"]
    ) / 1e6

    return (
        df.groupby("model_name")
        .agg(
            accuracy=("performance", "mean"),
            mean_cost=("cost_usd", "mean"),
            rows=("performance", "size"),
        )
        .reset_index()
    )
