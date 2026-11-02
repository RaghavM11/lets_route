"""Freeze ten task-diverse training queries with recorded cheapest-correct labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "mmlu", "mmlu_pro", "arc_challenge", "openbook_qa", "commonsense_qa",
    "boolq", "hellaswag", "gsm8k", "math", "squad",
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if (ROOT / "fixtures/smoke10/tasks.json").exists():
        raise SystemExit("The 10 tasks are already frozen. Reuse fixtures/smoke10/tasks.json; refusing to overwrite.")
    raw = ROOT / "data/raw"
    source = raw / "llmrouter_generic/train-000.parquet"
    price_path = raw / "llm_candidates/train-000.parquet"
    frame = pd.read_parquet(source)
    prices = pd.read_parquet(price_path)
    frame = frame.merge(
        prices[["model_name", "input_price_per_1m", "output_price_per_1m"]],
        on="model_name", validate="many_to_one",
    )
    frame["query_uid"] = frame["query"].map(sha)
    frame["cost_usd"] = (
        frame.input_tokens * frame.input_price_per_1m
        + frame.output_tokens * frame.output_price_per_1m
    ) / 1e6
    assert frame.cost_usd.notna().all() and (frame.cost_usd >= 0).all()
    held_out = set()
    for split in ("valid", "test"):
        data = pd.read_parquet(raw / "llmrouter_generic_queries" / f"{split}-000.parquet")
        held_out.update(data["query"].map(sha))

    records, matrix_rows = [], []
    for task in TASKS:
        subset = frame[frame.task_name.eq(task)]
        candidates = sorted(set(subset.query_uid) - held_out, key=lambda uid: sha(f"42:{uid}"))
        for uid in candidates:
            group = subset[subset.query_uid.eq(uid)].copy()
            # Require a complete counterfactual row and at least one fully correct model.
            if len(group) != 18 or group.model_name.nunique() != 18 or group.performance.max() != 1:
                continue
            winner = group[group.performance.eq(1)].sort_values(
                ["cost_usd", "model_name"], kind="stable"
            ).iloc[0]
            example = group.iloc[0]
            index = len(records) + 1
            outcomes = json.loads(group.sort_values("model_name")[[
                "model_name", "performance", "cost_usd", "input_tokens",
                "output_tokens", "input_price_per_1m", "output_price_per_1m", "response",
            ]].to_json(orient="records", double_precision=15))
            records.append({
                "fixed_id": f"smoke-{index:02d}", "query_uid": uid,
                "source_split": "train", "embedding_id": int(example.embedding_id),
                "task_name": task, "metric": example.metric,
                "query": example.query, "ground_truth": example.ground_truth,
                "choices": example.choices,
                "cheap_model_label": winner.model_name,
                "label_performance": float(winner.performance),
                "label_cost_usd": float(winner.cost_usd),
                "label_response": winner.response, "recorded_outcomes": outcomes,
            })
            matrix_rows.extend({"fixed_id": f"smoke-{index:02d}", "query_uid": uid, **row} for row in outcomes)
            break
        else:
            raise ValueError(f"No complete successful training query found for {task}")
    assert len(records) == len({r["query_uid"] for r in records}) == 10
    out = ROOT / "fixtures/smoke10"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "ulab-ai/xRouteBench", "config": "llmrouter_generic",
        "split": "train", "seed": 42, "count": 10,
        "label_rule": "Among all 18 recorded models with performance == 1, minimize recorded token cost; break ties by model_name. epsilon=0. These are offline recorded labels, not new SGLang measurements.",
        "cost_rule": "(input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1e6",
        "source_sha256": sha_bytes(source), "prices_sha256": sha_bytes(price_path),
        "held_out_prompt_overlap": 0, "tasks": records,
    }
    (out / "tasks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame([{k: r[k] for k in ("fixed_id", "query_uid", "task_name", "cheap_model_label", "label_performance", "label_cost_usd")} for r in records]).to_csv(out / "labels.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(out / "recorded_matrix.csv", index=False)
    text = ["Ten fixed training tasks (offline labels; not new model measurements)", payload["label_rule"], ""]
    for r in records:
        text.extend(["=" * 80, f"{r['fixed_id']} | {r['task_name']} | {r['metric']}",
                     f"Cheapest correct model: {r['cheap_model_label']}",
                     f"Recorded score: {r['label_performance']} | token cost: ${r['label_cost_usd']:.9f}",
                     f"query_uid: {r['query_uid']}", "", "TASK / PROMPT", r["query"],
                     "", "REFERENCE ANSWER / GROUND TRUTH", str(r["ground_truth"]),
                     "", "Recorded response from the labeled model", r["label_response"], ""])
    (out / "tasks.txt").write_text("\n".join(text), encoding="utf-8")
    print(pd.read_csv(out / "labels.csv").to_string(index=False))
    print(f"Saved fixed tasks to {out}")


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
