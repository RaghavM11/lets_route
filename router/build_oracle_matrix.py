#!/usr/bin/env python
"""Build the counterfactual query x model matrix and oracle labels for xRouteBench.

Reads the project-local parquet (data/raw), computes for each query the
epsilon-optimal cheapest-tier oracle target, and writes a training-ready JSONL
(for the LoRA System-1 router).

Output per line:
  {
    "task", "query", "ground_truth", "metric", "embedding_id",
    "candidates": [{"model", "perf", "cost_usd"}...],
    "accept_epsilon": float,
    "acceptable_models": [model...],     # within epsilon of best perf
    "oracle_target": model                # cheapest acceptable (or cheapest if all fail)
  }

Usage:
    python router/build_oracle_matrix.py --eps 0.0 --split train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CONFIG = "llmrouter_generic"
CANDIDATES_CFG = "llm_candidates"


def _query_key(row: dict) -> tuple[str, str]:
    """Robust grouping key: (task_name, full-query md5)."""
    h = hashlib.md5(str(row["query"]).encode("utf-8")).hexdigest()
    return (row["task_name"], h)


def build_matrix(split: str, eps: float) -> list[dict]:
    import pandas as pd

    routing_path = RAW / CONFIG / f"{split}-000.parquet"
    cand_path = RAW / CANDIDATES_CFG / "train-000.parquet"
    if not routing_path.exists() or not cand_path.exists():
        raise FileNotFoundError(f"Missing raw parquet: {routing_path} / {cand_path}")

    frame = pd.read_parquet(routing_path)
    cand = pd.read_parquet(cand_path)
    price = {
        r["model_name"]: (r["input_price_per_1m"], r["output_price_per_1m"])
        for _, r in cand.iterrows()
    }

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in frame.to_dict("records"):
        groups[_query_key(r)].append(r)

    def cost(r) -> float:
        ip, op = price.get(r["model_name"], (0.0, 0.0))
        return (r.get("input_tokens", 0) or 0) * ip / 1e6 + (
            r.get("output_tokens", 0) or 0
        ) * op / 1e6

    rows = []
    for (task, _), execs in groups.items():
        best = max(e["performance"] for e in execs)
        accept = [e for e in execs if e["performance"] >= best - eps]
        target = min(accept, key=cost) if accept else min(execs, key=cost)
        rows.append(
            {
                "task": task,
                "query": execs[0]["query"],
                "ground_truth": execs[0].get("ground_truth"),
                "metric": execs[0].get("metric"),
                "embedding_id": execs[0].get("embedding_id"),
                "candidates": [
                    {
                        "model": e["model_name"],
                        "perf": float(e["performance"]),
                        "cost_usd": float(cost(e)),
                    }
                    for e in sorted(execs, key=cost)
                ],
                "accept_epsilon": eps,
                "acceptable_models": [e["model_name"] for e in accept],
                "oracle_target": target["model_name"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = build_matrix(args.split, args.eps)
    if args.limit:
        rows = rows[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"routing_{args.split}_eps{args.eps:.2f}.jsonl"
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_acc_set = sum(len(r["acceptable_models"]) for r in rows)
    print(f"wrote {len(rows)} queries -> {out_path}")
    print(f"avg acceptable models/query: {n_acc_set / max(1, len(rows)):.2f}")
    from collections import Counter
    print("oracle target distribution:")
    for m, c in Counter(r["oracle_target"] for r in rows).most_common(8):
        print(f"  {m:28s} {c:6d}  {100*c/len(rows):5.1f}%")


if __name__ == "__main__":
    main()
