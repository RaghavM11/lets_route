"""Reusable xRouteBench evaluation logic for the Modal Qwen experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


ROUTING_COLUMNS = (
    "embedding_id",
    "task_name",
    "query",
    "ground_truth",
    "metric",
    "model_name",
    "performance",
    "input_tokens",
    "output_tokens",
    "response_time",
)

DEFAULT_TASKS = (
    "mmlu",
    "arc_challenge",
    "openbook_qa",
    "commonsense_qa",
    "boolq",
    "hellaswag",
)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_xroute_queries(
    data_root: Path, split: str = "test"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and validate one complete xRouteBench candidate matrix."""

    path = data_root / "llmrouter_generic" / f"{split}-000.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing xRouteBench Parquet: {path}")

    frame = pd.read_parquet(path, columns=list(ROUTING_COLUMNS))
    missing = set(ROUTING_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["embedding_id", "model_name"]).any():
        raise ValueError("Duplicate (embedding_id, model_name) executions found")

    candidate_count = frame["model_name"].nunique()
    per_query = frame.groupby("embedding_id", sort=False).agg(
        rows=("model_name", "size"),
        models=("model_name", "nunique"),
        queries=("query", "nunique"),
        tasks=("task_name", "nunique"),
        metrics=("metric", "nunique"),
    )
    invalid = per_query[
        (per_query["rows"] != candidate_count)
        | (per_query["models"] != candidate_count)
        | (per_query[["queries", "tasks", "metrics"]] != 1).any(axis=1)
    ]
    if len(invalid):
        raise ValueError(f"{len(invalid)} incomplete or inconsistent query matrices")

    query_table = (
        frame.sort_values("embedding_id")
        .drop_duplicates("embedding_id")
        [["embedding_id", "task_name", "query", "ground_truth", "metric"]]
        .reset_index(drop=True)
    )
    query_table["query_hash"] = query_table["query"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    audit = {
        "split": split,
        "execution_rows": int(len(frame)),
        "query_instances": int(len(query_table)),
        "unique_prompt_texts": int(query_table["query_hash"].nunique()),
        "candidate_models": int(candidate_count),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    return query_table, frame, audit


def select_mc_queries(
    query_table: pd.DataFrame,
    tasks: Iterable[str] = DEFAULT_TASKS,
    *,
    per_task: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministically select exact-score multiple-choice queries by task."""

    if per_task < 1:
        raise ValueError("per_task must be at least 1")

    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("At least one task is required")

    selections = []
    for task in tasks:
        task_rows = query_table[
            query_table["task_name"].eq(task) & query_table["metric"].eq("em_mc")
        ].copy()
        if len(task_rows) < per_task:
            raise ValueError(
                f"Task {task!r} has only {len(task_rows)} em_mc queries; "
                f"requested {per_task}"
            )
        task_rows["sample_order"] = task_rows["query_hash"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
        )
        selections.append(task_rows.sort_values("sample_order").head(per_task))

    return pd.concat(selections, ignore_index=True).drop(columns="sample_order")


def xroute_messages(raw_query: Any) -> list[dict[str, str]]:
    """Convert xRouteBench's combined prompt into proper chat roles."""

    raw_query = str(raw_query)
    system_marker, user_marker = "[System Instruction]", "[User Query]"
    if raw_query.startswith(system_marker) and user_marker in raw_query:
        system_text, user_text = raw_query[len(system_marker) :].split(user_marker, 1)
        return [
            {"role": "system", "content": system_text.strip()},
            {"role": "user", "content": user_text.strip()},
        ]
    return [{"role": "user", "content": raw_query}]


def extract_mc_choice(response: Any) -> str | None:
    """Extract the final A-D choice without rewarding letters in prose."""

    text = str(response).strip().upper()
    parenthesized = re.findall(r"\(([A-D])\)", text)
    if parenthesized:
        return parenthesized[-1]
    final_answer = re.findall(
        r"FINAL\s+ANSWER(?:\s+IS)?[:\s]+([A-D])(?:\b|$)", text
    )
    if final_answer:
        return final_answer[-1]
    standalone = re.fullmatch(r"\s*([A-D])(?:[.)])?\s*", text)
    return standalone.group(1) if standalone else None


def evaluate_generator(
    queries: pd.DataFrame,
    generate_fn: Callable[[list[dict[str, str]]], dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate an adapter returning response/token/latency/cost fields."""

    rows: list[dict[str, Any]] = []
    for position, row in enumerate(queries.itertuples(index=False), start=1):
        try:
            result = generate_fn(xroute_messages(row.query))
            parsed = extract_mc_choice(result["response"])
            error = None
        except Exception as exc:  # Preserve the rest of a paid benchmark run.
            result = {
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_s": math.nan,
                "cost_usd": math.nan,
            }
            parsed = None
            error = f"{type(exc).__name__}: {exc}"

        expected = str(row.ground_truth).strip().upper()
        rows.append(
            {
                "embedding_id": int(row.embedding_id),
                "task_name": row.task_name,
                "expected": expected,
                "parsed_answer": parsed,
                "correct": bool(parsed == expected),
                "response": result.get("response", ""),
                "input_tokens": result.get("input_tokens", math.nan),
                "output_tokens": result.get("output_tokens", math.nan),
                "latency_s": result.get("latency_s", math.nan),
                "cost_usd": result.get("cost_usd", math.nan),
                "error": error,
            }
        )
        print(
            f"[{position:03d}/{len(queries)}] {row.task_name}: "
            f"expected={expected}, predicted={parsed}",
            flush=True,
        )
    return pd.DataFrame(rows)


def candidate_comparison(
    results: pd.DataFrame,
    selected_queries: pd.DataFrame,
    test_long: pd.DataFrame,
    candidates_path: Path,
    model_label: str,
) -> pd.DataFrame:
    """Compare the measured Qwen run with recorded candidates on identical IDs."""

    candidates = pd.read_parquet(candidates_path)
    price_columns = ["model_name", "input_price_per_1m", "output_price_per_1m"]
    missing = set(price_columns).difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate price table is missing: {sorted(missing)}")
    if candidates["model_name"].duplicated().any():
        raise ValueError("Candidate price table contains duplicate model names")

    selected_ids = set(selected_queries["embedding_id"])
    candidate_eval = test_long[test_long["embedding_id"].isin(selected_ids)].merge(
        candidates[price_columns], on="model_name", how="left", validate="many_to_one"
    )
    if candidate_eval[price_columns[1:]].isna().any().any():
        raise ValueError("At least one recorded candidate is missing price data")
    candidate_eval["cost_usd"] = (
        candidate_eval["input_tokens"] * candidate_eval["input_price_per_1m"]
        + candidate_eval["output_tokens"] * candidate_eval["output_price_per_1m"]
    ) / 1_000_000.0
    candidate_summary = (
        candidate_eval.groupby("model_name")
        .agg(
            accuracy=("performance", "mean"),
            mean_cost_per_1k_usd=("cost_usd", lambda values: values.mean() * 1000),
            mean_latency_s=("response_time", "mean"),
        )
        .reset_index()
    )
    qwen_summary = pd.DataFrame(
        [
            {
                "model_name": model_label,
                "accuracy": results["correct"].mean(),
                "mean_cost_per_1k_usd": results["cost_usd"].mean() * 1000,
                "mean_latency_s": results["latency_s"].mean(),
            }
        ]
    )
    return (
        pd.concat([qwen_summary, candidate_summary], ignore_index=True)
        .sort_values(["accuracy", "mean_cost_per_1k_usd"], ascending=[False, True])
        .reset_index(drop=True)
    )


def safe_run_id(run_id: str | None = None) -> str:
    """Return a path-safe run id, generating one when absent."""

    value = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(
            "run_id must be 1-80 path-safe characters: letters, digits, _, ., or -"
        )
    return value


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records, including NaN -> null."""

    return json.loads(frame.to_json(orient="records"))


def run_transformers_benchmark(
    *,
    model_path: Path,
    model_id: str,
    model_revision: str,
    data_root: Path,
    results_root: Path,
    tasks: Iterable[str] = DEFAULT_TASKS,
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 32,
    gpu_usd_per_hour: float = 0.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Load Qwen on one GPU, run the suite, and persist complete artifacts."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if gpu_usd_per_hour < 0:
        raise ValueError("gpu_usd_per_hour cannot be negative")

    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen3.5-27B is configured for BF16, but this GPU lacks support")

    run_id = safe_run_id(run_id)
    tasks = tuple(tasks)
    query_table, test_long, loader_audit = load_xroute_queries(data_root, "test")
    selected = select_mc_queries(
        query_table, tasks, per_task=per_task, seed=seed
    )
    candidates_path = data_root / "llm_candidates" / "train-000.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidate price Parquet: {candidates_path}")

    print(f"Loading {model_id}@{model_revision} from {model_path}", flush=True)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started

    def make_inputs(messages: list[dict[str, str]]):
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to("cuda")

    # Exclude one-time CUDA/kernel initialization from measured query latency.
    warmup_inputs = make_inputs(xroute_messages(selected.iloc[0]["query"]))
    with torch.inference_mode():
        model.generate(**warmup_inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    del warmup_inputs

    def generate(messages: list[dict[str, str]]) -> dict[str, Any]:
        inputs = make_inputs(messages)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - started
        response_ids = output[0, inputs["input_ids"].shape[-1] :]
        response = processor.decode(response_ids, skip_special_tokens=True).strip()
        cost_usd = (
            latency_s * gpu_usd_per_hour / 3600.0
            if gpu_usd_per_hour > 0
            else math.nan
        )
        return {
            "response": response,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(response_ids.numel()),
            "latency_s": latency_s,
            "cost_usd": cost_usd,
        }

    benchmark_started = time.perf_counter()
    results = evaluate_generator(selected, generate)
    benchmark_s = time.perf_counter() - benchmark_started
    comparison = candidate_comparison(
        results,
        selected,
        test_long,
        candidates_path,
        f"{model_id} (Modal H200)",
    )

    total_output_tokens = int(results["output_tokens"].sum())
    total_query_latency = float(results["latency_s"].sum())
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_path": str(model_path),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "thinking_enabled": False,
        "seed": seed,
        "tasks": list(tasks),
        "per_task": per_task,
        "query_count": int(len(results)),
        "max_new_tokens": max_new_tokens,
        "accuracy": float(results["correct"].mean()),
        "correct": int(results["correct"].sum()),
        "errors": int(results["error"].notna().sum()),
        "mean_latency_s": float(results["latency_s"].mean()),
        "output_tokens_per_s": (
            total_output_tokens / total_query_latency if total_query_latency else None
        ),
        "gpu_usd_per_hour": gpu_usd_per_hour or None,
        "estimated_cost_per_1k_queries_usd": (
            float(results["cost_usd"].mean() * 1000)
            if gpu_usd_per_hour > 0
            else None
        ),
        "model_load_s": model_load_s,
        "benchmark_s": benchmark_s,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "data": {
            "test": loader_audit,
            "candidates": {
                "path": str(candidates_path),
                "bytes": candidates_path.stat().st_size,
                "sha256": file_sha256(candidates_path),
            },
        },
    }

    output_dir = results_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "query_results.csv", index=False)
    comparison.to_csv(output_dir / "candidate_comparison.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    return {
        "summary": summary,
        "query_results": json_records(results),
        "candidate_comparison": json_records(comparison),
        "remote_result_dir": str(output_dir),
    }
