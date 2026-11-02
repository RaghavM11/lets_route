"""xRouteBench evaluation through a local SGLang server."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_TASKS = (
    "mmlu",
    "arc_challenge",
    "openbook_qa",
    "commonsense_qa",
    "boolq",
    "hellaswag",
)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_queries(
    data_root: Path, split: str = "test"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = data_root / "llmrouter_generic" / f"{split}-000.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing xRouteBench data: {path}")

    frame = pd.read_parquet(path, columns=list(ROUTING_COLUMNS))
    if frame.duplicated(["embedding_id", "model_name"]).any():
        raise ValueError("Duplicate (embedding_id, model_name) rows")

    candidate_count = frame["model_name"].nunique()
    grouped = frame.groupby("embedding_id", sort=False).agg(
        rows=("model_name", "size"),
        models=("model_name", "nunique"),
        prompts=("query", "nunique"),
        tasks=("task_name", "nunique"),
        metrics=("metric", "nunique"),
    )
    invalid = grouped[
        (grouped["rows"] != candidate_count)
        | (grouped["models"] != candidate_count)
        | (grouped[["prompts", "tasks", "metrics"]] != 1).any(axis=1)
    ]
    if len(invalid):
        raise ValueError(f"{len(invalid)} incomplete query matrices")

    queries = (
        frame.sort_values("embedding_id")
        .drop_duplicates("embedding_id")
        [["embedding_id", "task_name", "query", "ground_truth", "metric"]]
        .reset_index(drop=True)
    )
    queries["query_hash"] = queries["query"].map(
        lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    )
    audit = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "split": split,
        "execution_rows": int(len(frame)),
        "query_instances": int(len(queries)),
        "candidate_models": int(candidate_count),
    }
    return queries, frame, audit


def select_queries(
    queries: pd.DataFrame,
    tasks: Iterable[str] = DEFAULT_TASKS,
    *,
    per_task: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    if per_task < 1:
        raise ValueError("per_task must be at least 1")

    selected = []
    for task in tasks:
        subset = queries[
            queries["task_name"].eq(task) & queries["metric"].eq("em_mc")
        ].copy()
        if len(subset) < per_task:
            raise ValueError(
                f"Task {task!r} has {len(subset)} em_mc rows; requested {per_task}"
            )
        subset["sample_order"] = subset["query_hash"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        selected.append(subset.sort_values("sample_order").head(per_task))
    if not selected:
        raise ValueError("At least one task is required")
    return pd.concat(selected, ignore_index=True).drop(columns="sample_order")


def xroute_messages(raw_query: Any) -> list[dict[str, str]]:
    text = str(raw_query)
    system_marker, user_marker = "[System Instruction]", "[User Query]"
    if text.startswith(system_marker) and user_marker in text:
        system, user = text[len(system_marker) :].split(user_marker, 1)
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
    return [{"role": "user", "content": text}]


def extract_mc_choice(response: Any) -> str | None:
    """Accept A-E only when the model presents it as a final answer."""

    text = str(response).strip().upper()
    marked = re.findall(
        r"(?:FINAL\s+)?(?:ANSWER|CHOICE)(?:\s+IS)?\s*[:=-]?\s*"
        r"(?:\*\*)?\(?([A-E])\)?(?:\*\*)?(?:[.)])?",
        text,
    )
    if marked:
        return marked[-1]
    trailing = re.search(
        r"(?:^|\n)\s*(?:\*\*)?\(?([A-E])\)?(?:\*\*)?(?:[.)])?\s*$", text
    )
    return trailing.group(1) if trailing else None


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if data else {}


def _wait_for_server(process: subprocess.Popen[bytes], base_url: str) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < 15 * 60:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with code {process.returncode}")
        try:
            _http_json("GET", f"{base_url}/health_generate", timeout=30)
            return time.perf_counter() - started
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError("SGLang did not become ready within 15 minutes")


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _gpu_info() -> tuple[str, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, memory_mib = (part.strip() for part in output.split(",", 1))
    return name, round(float(memory_mib) / 1024, 2)


def _candidate_comparison(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    test_long: pd.DataFrame,
    candidates_path: Path,
    model_label: str,
) -> pd.DataFrame:
    prices = pd.read_parquet(candidates_path)[
        ["model_name", "input_price_per_1m", "output_price_per_1m"]
    ]
    if prices["model_name"].duplicated().any():
        raise ValueError("Candidate price table has duplicate models")

    recorded = test_long[test_long["embedding_id"].isin(selected["embedding_id"])].merge(
        prices, on="model_name", how="left", validate="many_to_one"
    )
    if recorded[["input_price_per_1m", "output_price_per_1m"]].isna().any().any():
        raise ValueError("Candidate price data is incomplete")
    recorded["cost_usd"] = (
        recorded["input_tokens"] * recorded["input_price_per_1m"]
        + recorded["output_tokens"] * recorded["output_price_per_1m"]
    ) / 1_000_000
    baseline = (
        recorded.groupby("model_name")
        .agg(
            accuracy=("performance", "mean"),
            mean_cost_per_1k_usd=("cost_usd", lambda values: values.mean() * 1000),
            mean_latency_s=("response_time", "mean"),
        )
        .reset_index()
    )
    qwen = pd.DataFrame(
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
        pd.concat([qwen, baseline], ignore_index=True)
        .sort_values(["accuracy", "mean_cost_per_1k_usd"], ascending=[False, True])
        .reset_index(drop=True)
    )


def safe_run_id(value: str | None) -> str:
    value = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("run_id must be 1-80 path-safe characters")
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def run_sglang_benchmark(
    *,
    model_path: Path,
    model_id: str,
    model_revision: str,
    data_root: Path,
    results_root: Path,
    tasks: Iterable[str] = DEFAULT_TASKS,
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 256,
    gpu_usd_per_hour: float = 0,
    run_id: str | None = None,
) -> dict[str, Any]:
    if max_new_tokens < 1 or gpu_usd_per_hour < 0:
        raise ValueError("Token limit must be positive and GPU rate cannot be negative")

    run_id = safe_run_id(run_id)
    tasks = tuple(tasks)
    queries, test_long, data_audit = load_queries(data_root)
    selected = select_queries(queries, tasks, per_task=per_task, seed=seed)
    candidates_path = data_root / "llm_candidates" / "train-000.parquet"
    if not model_path.exists() or not candidates_path.exists():
        raise FileNotFoundError("Model checkpoint or candidate price data is missing")

    base_url = "http://127.0.0.1:30000"
    command = [
        "sglang",
        "serve",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.80",
        "--max-running-requests",
        "1",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-bs-decode",
        "1",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
    ]
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    print("Starting SGLang:", " ".join(command), flush=True)
    server = subprocess.Popen(command, env=environment, start_new_session=True)

    try:
        startup_s = _wait_for_server(server, base_url)
        rows = []
        benchmark_started = time.perf_counter()
        for position, row in enumerate(selected.itertuples(index=False), start=1):
            started = time.perf_counter()
            error = None
            try:
                response = _http_json(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    {
                        "model": str(model_path),
                        "messages": xroute_messages(row.query),
                        "temperature": 0,
                        "max_tokens": max_new_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=10 * 60,
                )
                latency_s = time.perf_counter() - started
                text = response["choices"][0]["message"]["content"].strip()
                usage = response.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                parsed = extract_mc_choice(text)
            except Exception as exc:
                latency_s = time.perf_counter() - started
                text, parsed, prompt_tokens, output_tokens = "", None, 0, 0
                error = f"{type(exc).__name__}: {exc}"

            expected = str(row.ground_truth).strip().upper()
            rows.append(
                {
                    "embedding_id": int(row.embedding_id),
                    "task_name": row.task_name,
                    "expected": expected,
                    "parsed_answer": parsed,
                    "correct": parsed == expected,
                    "response": text,
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "latency_s": latency_s,
                    "cost_usd": (
                        latency_s * gpu_usd_per_hour / 3600
                        if gpu_usd_per_hour
                        else math.nan
                    ),
                    "truncated": output_tokens >= max_new_tokens,
                    "error": error,
                }
            )
            print(
                f"[{position:03d}/{len(selected)}] {row.task_name}: "
                f"expected={expected}, predicted={parsed}",
                flush=True,
            )
        benchmark_s = time.perf_counter() - benchmark_started
    finally:
        _stop_server(server)

    results = pd.DataFrame(rows)
    comparison = _candidate_comparison(
        results,
        selected,
        test_long,
        candidates_path,
        f"{model_id} (SGLang / Modal H200)",
    )
    gpu_name, gpu_memory_gib = _gpu_info()
    latency_total = results["latency_s"].sum()
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "sglang",
        "sglang_version": importlib.metadata.version("sglang"),
        "model_id": model_id,
        "model_revision": model_revision,
        "gpu": gpu_name,
        "gpu_memory_gib": gpu_memory_gib,
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
        "unparsed_responses": int(results["parsed_answer"].isna().sum()),
        "truncated_responses": int(results["truncated"].sum()),
        "mean_latency_s": float(results["latency_s"].mean()),
        "output_tokens_per_s": (
            float(results["output_tokens"].sum() / latency_total)
            if latency_total
            else None
        ),
        "gpu_usd_per_hour": gpu_usd_per_hour or None,
        "estimated_cost_per_1k_queries_usd": (
            float(results["cost_usd"].mean() * 1000) if gpu_usd_per_hour else None
        ),
        "engine_startup_s": startup_s,
        "benchmark_s": benchmark_s,
        "data": {
            "test": data_audit,
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return {
        "summary": summary,
        "query_results": _records(results),
        "candidate_comparison": _records(comparison),
        "remote_result_dir": str(output_dir),
    }
