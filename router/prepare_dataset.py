#!/usr/bin/env python
"""Prepare xRouteBench for router training, in the right order.

Order:
  1. Materialize the counterfactual query x model matrix (performance + cost).
  2. Annotate each query: best_perf, spread, difficulty, oracle target, cost gap.
  3. Stratified train / valid split from the OFFICIAL train (kept separate from
     the official test). Official test stays untouched as the final eval set.
  4. Write JSONL: data/processed/routing_{train,valid,test}.jsonl

Splitting is stratified by `task | difficulty | cost_tier` so every slice of
the cost/performance space is represented in train and valid.

Usage:
    python router/prepare_dataset.py [--eps 0.0] [--val 0.12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
from collections import defaultdict

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CONFIG = "llmrouter_generic"
CANDIDATES_CFG = "llm_candidates"

TASK_COLUMNS = ["task_name", "query", "ground_truth", "metric", "embedding_id"]


def _query_key(row) -> tuple[str, str]:
    h = hashlib.md5(str(row["query"]).encode("utf-8")).hexdigest()
    return (row["task_name"], h)


def _difficulty(best_perf: float) -> str:
    if best_perf >= 0.9:
        return "easy"
    if best_perf >= 0.5:
        return "medium"
    return "hard"


def _cost_tier(cost: float, quartiles: tuple[float, float]) -> str:
    q1, q3 = quartiles
    if cost <= q1:
        return "budget"
    if cost >= q3:
        return "premium"
    return "mid"


def load_labeled(split: str, eps: float) -> list[dict]:
    """Load + label every query of a split (train|test)."""
    routing_path = RAW / CONFIG / f"{split}-000.parquet"
    cand_path = RAW / CANDIDATES_CFG / "train-000.parquet"
    frame = pd.read_parquet(routing_path)
    cand = pd.read_parquet(cand_path)
    price = {
        r["model_name"]: (r["input_price_per_1m"], r["output_price_per_1m"])
        for _, r in cand.iterrows()
    }

    def cost(r) -> float:
        ip, op = price.get(r["model_name"], (0.0, 0.0))
        return (r.get("input_tokens", 0) or 0) * ip / 1e6 + (
            r.get("output_tokens", 0) or 0
        ) * op / 1e6

    groups: dict = defaultdict(list)
    for r in frame.to_dict("records"):
        groups[_query_key(r)].append(r)

    # candidate-level cost quartiles (across all rows) for tiering
    all_costs = sorted(cost(r) for r in frame.to_dict("records"))
    q1 = all_costs[len(all_costs) // 4]
    q3 = all_costs[3 * len(all_costs) // 4]

    records = []
    for (task, _), execs in groups.items():
        cands = sorted(
            ({"model": e["model_name"], "perf": float(e["performance"]),
              "cost_usd": float(cost(e))} for e in execs),
            key=lambda c: c["cost_usd"],
        )
        best_perf = max(c["perf"] for c in cands)
        accept = [c for c in cands if c["perf"] >= best_perf - eps]
        target = min(accept, key=lambda c: c["cost_usd"]) if accept else cands[0]
        cheapest = cands[0]
        records.append(
            {
                "task": task,
                "query": execs[0]["query"],
                "ground_truth": execs[0].get("ground_truth"),
                "metric": execs[0].get("metric"),
                "embedding_id": execs[0].get("embedding_id"),
                "candidates": cands,
                "accept_epsilon": eps,
                "acceptable_models": [c["model"] for c in accept],
                "oracle_target": target["model"],
                "best_perf": best_perf,
                "spread": best_perf - min(c["perf"] for c in cands),
                "target_cost_usd": target["cost_usd"],
                "cheapest_cost_usd": cheapest["cost_usd"],
                "cost_gap_usd": target["cost_usd"] - cheapest["cost_usd"],
                "difficulty": _difficulty(best_perf),
                "cost_tier": _cost_tier(target["cost_usd"], (q1, q3)),
            }
        )
    return records


def stratified_split(records: list[dict], val_frac: float, seed: int = 42):
    """Split official train into train/valid, stratified by task|difficulty|cost_tier."""
    rng = random.Random(seed)
    train, valid = [], []
    df = pd.DataFrame.from_records(records)
    df["stratum"] = df["task"] + "|" + df["difficulty"] + "|" + df["cost_tier"]
    for _, group in df.groupby("stratum", sort=False):
        idx = list(group.index)
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_frac)) if len(idx) > 1 else 0
        valid_idx = set(idx[:n_val])
        for i in idx:
            (valid if i in valid_idx else train).append(records[i])
    return train, valid


def write_records(path: pathlib.Path, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def summarize(records: list[dict], label: str) -> None:
    from collections import Counter
    print(f"--- {label}: {len(records)} queries ---")
    print("  difficulty:", dict(Counter(r["difficulty"] for r in records)))
    print("  cost_tier:", dict(Counter(r["cost_tier"] for r in records)))
    avg_gap = sum(r["cost_gap_usd"] for r in records) / max(1, len(records))
    print(f"  avg target-vs-cheapest cost gap: ${avg_gap:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--val", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train_all = load_labeled("train", args.eps)
    test = load_labeled("test", args.eps)

    train, valid = stratified_split(train_all, args.val, args.seed)
    write_records(OUT / "routing_train.jsonl", train)
    write_records(OUT / "routing_valid.jsonl", valid)
    write_records(OUT / "routing_test.jsonl", test)

    print(f"Official train -> {len(train)} train / {len(valid)} valid; official test kept ({len(test)}).")
    summarize(train, "train")
    summarize(valid, "valid")
    summarize(test, "test (held-out)")


if __name__ == "__main__":
    main()
