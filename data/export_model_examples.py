"""Export five size tiers on the same frozen ten queries: 50 recorded runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("cogito-v2-1-671b", "671B (MoE total)", "https://huggingface.co/deepcogito/cogito-671b-v2.1"),
    ("llama-3.3-70b-instruct-turbo", "70B", "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"),
    ("mistral-small-3-24b-instruct", "24B", "https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501"),
    ("gemma-2-9b-it", "9B", "https://huggingface.co/google/gemma-2-9b-it"),
    ("mistral-7b-instruct-v0.3", "7B", "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"),
)


def status(score: float) -> str:
    return "Full score / score=1" if score == 1 else "Zero score / score=0" if score == 0 else "Partial score / 0<score<1"


def main() -> None:
    manifest_path = ROOT / "fixtures/smoke10/tasks.json"
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / "data/raw/llmrouter_generic/train-000.parquet"
    frame = pd.read_parquet(source)
    prices = pd.read_parquet(ROOT / "data/raw/llm_candidates/train-000.parquet")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    out = ROOT / "fixtures/model_examples50"
    out.mkdir(parents=True, exist_ok=True)
    records, summaries, sections = [], [], []
    header = (
        "5 models x the same 10 fixed tasks = 50 recorded executions\n"
        "Source: xRouteBench / llmrouter_generic / train; no new model calls were made.\n"
        "Ordered by model size, largest first; strength is contextualized by full-training-set mean scores, not these ten tasks.\n"
        "A zero score does not always imply a semantic error; formatting or scoring rules may affect it. Partial F1 scores are preserved.\n"
        "The fixed set requires at least one successful candidate per task and is not an unbiased evaluation sample.\n"
    )
    for rank, (name, size, card) in enumerate(MODELS, 1):
        pool = frame[frame.model_name.eq(name)]
        price = prices[prices.model_name.eq(name)].iloc[0]
        parts = [header, "=" * 90, f"Model {rank}: {name} | Parameter count {size}",
                 f"Mean score over all {len(pool)} training executions: {pool.performance.mean():.6f}",
                 f"Recorded provider: {price.service} | API model ID: {price.api_model_id}",
                 f"Open-weight model card: {card}", ""]
        model_rows = []
        for task in manifest["tasks"]:
            match = pool[pool.embedding_id.eq(task["embedding_id"])]
            assert len(match) == 1
            row = match.iloc[0]
            assert row.query == task["query"]
            score = float(row.performance)
            note = ""
            if task["fixed_id"] == "smoke-02" and name == "mistral-7b-instruct-v0.3":
                note = "Review note: the response is 'I. Ransom Trojan' and the reference is I, but the recorded score is zero. Formatting/extraction may explain this; the original score is unchanged."
            if task["metric"] == "f1_score":
                note = "This is token F1, not binary accuracy. An answer containing the correct phrase may still receive a low score because of extra text."
            record = {
                "fixed_id": task["fixed_id"], "query_uid": task["query_uid"],
                "embedding_id": task["embedding_id"], "source_split": "train",
                "task_name": task["task_name"], "metric": task["metric"],
                "model_name": name, "size": size, "service": price.service,
                "api_model_id": price.api_model_id, "query": row.query,
                "ground_truth": row.ground_truth, "response": row.response,
                "performance": score, "score_status": status(score),
                "input_tokens": int(row.input_tokens), "output_tokens": int(row.output_tokens),
                "response_time_s": float(row.response_time),
                "recorded_cost_usd": float((row.input_tokens * price.input_price_per_1m
                                             + row.output_tokens * price.output_price_per_1m) / 1e6),
                "review_note": note,
            }
            records.append(record)
            model_rows.append(record)
            parts.extend(["-" * 90, f"{task['fixed_id']} | {task['task_name']} | metric={task['metric']}",
                          f"Recorded score: {score:.15g} | {status(score)}",
                          f"Input/output tokens: {record['input_tokens']}/{record['output_tokens']} | Recorded latency: {record['response_time_s']:.6f}s | Estimated token cost: ${record['recorded_cost_usd']:.9f}",
                          f"query_uid: {task['query_uid']}", "", "[ORIGINAL PROMPT]", row.query,
                          "", "[REFERENCE ANSWER / GROUND TRUTH]", str(row.ground_truth),
                          "", "[ORIGINAL MODEL RESPONSE]", row.response,
                          "", "[REVIEW NOTES]", note or "No additional notes; the original dataset score is preserved.", ""])
        text = "\n".join(parts)
        filename = f"{rank:02d}_{name}.txt"
        (out / filename).write_text(text, encoding="utf-8")
        sections.append(text)
        summaries.append({
            "model_name": name, "size": size, "model_card": card,
            "full_train_mean_score": float(pool.performance.mean()),
            "sample_mean_score": sum(r["performance"] for r in model_rows) / 10,
            "full_score_count": sum(r["performance"] == 1 for r in model_rows),
            "zero_score_count": sum(r["performance"] == 0 for r in model_rows),
            "partial_score_count": sum(0 < r["performance"] < 1 for r in model_rows),
            "text_file": filename,
        })
    assert len(records) == 50
    assert all(sum(r["model_name"] == m[0] for r in records) == 10 for m in MODELS)
    (out / "all_50_results.txt").write_text("\n\n".join(sections), encoding="utf-8")
    result = pd.DataFrame(records)
    result.to_csv(out / "all_50_results.csv", index=False)
    matrix = result.pivot(index="fixed_id", columns="model_name", values="performance")
    matrix = matrix[[m[0] for m in MODELS]]
    matrix.to_csv(out / "score_matrix.csv")
    summary = {"count": 50, "unique_queries": 10, "model_count": 5,
               "execution_source": "recorded dataset results; not live inference",
               "frozen_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_sha256": manifest["source_sha256"], "models": summaries}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(summaries).to_csv(out / "model_summary.csv", index=False)
    inventory = prices.merge(frame.groupby("model_name").performance.mean().rename("train_mean_score"), on="model_name")
    inventory.sort_values("train_mean_score", ascending=False).to_csv(out / "all_18_model_inventory.csv", index=False)
    print(pd.DataFrame(summaries).drop(columns=["model_card", "text_file"]).to_string(index=False))
    print("\n", matrix.to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
