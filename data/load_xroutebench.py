"""Loaders for the xRouteBench LLM-routing benchmark (ulab-ai/xRouteBench).

Design principle: no bulk download by default. All metadata (configs, splits,
schema, sample rows) comes from the Hugging Face datasets-server HTTP API in a
single lightweight request. Full data is only fetched if you explicitly call
load(stream=False) or iterate a stream far enough to pull the parquet chunks.

Configs (scenarios) — each is a routing-data set with train/test splits:
    llmrouter_generic, memory_locomo, memory_longmemeval, timeseries, video,
    multimodal_geometry3k, multimodal_mathvista, personalized
Raw query configs (no model executions) ship as ``<scenario>_queries`` with
train/valid/test splits. ``llm_candidates`` holds the 18-model pricing pool.

Row schema (routing configs):
    task_name, query, ground_truth, metric, choices, task_id, model_name,
    response, token_num, input_tokens, output_tokens, response_time,
    performance, embedding_id
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

DATASET = "ulab-ai/xRouteBench"
_API = "https://datasets-server.huggingface.co"

# Scenarios (routing-data configs) from the dataset card.
SCENARIOS = [
    "llmrouter_generic",
    "memory_locomo",
    "memory_longmemeval",
    "timeseries",
    "video",
    "multimodal_geometry3k",
    "multimodal_mathvista",
    "personalized",
]
CANDIDATES_CONFIG = "llm_candidates"

ROUTING_COLUMNS = [
    "task_name", "query", "ground_truth", "metric", "choices", "task_id",
    "model_name", "response", "token_num", "input_tokens", "output_tokens",
    "response_time", "performance", "embedding_id",
]


def _get(url: str, params: dict | None = None) -> dict:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "route_f-loader"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def list_splits() -> dict[str, list[str]]:
    """Return {config: [splits...]} across all configs. No data download."""
    data = _get(f"{_API}/splits", {"dataset": DATASET})
    out: dict[str, list[str]] = {}
    for s in data["splits"]:
        out.setdefault(s["config"], []).append(s["split"])
    return out


def list_configs() -> list[str]:
    return sorted(list_splits().keys())


def schema(config: str, split: str = "train") -> tuple[list[str], list[dict]]:
    """Return (column_names, sample_rows) from the server. No data download."""
    data = _get(f"{_API}/first-rows", {
        "dataset": DATASET, "config": config, "split": split,
    })
    cols = [f["name"] for f in data["features"]]
    rows = [r["row"] for r in data["rows"]]
    return cols, rows


def load(config: str, split: str = "train", stream: bool = True, **kwargs):
    """Load a config/split.

    stream=True (default) -> lazy `datasets` iterable; parquet is pulled
    chunk-by-chunk only as you iterate, so nothing is bulk-downloaded.
    stream=False -> fully download and cache locally (use when actually training).
    """
    from datasets import load_dataset
    return load_dataset(DATASET, config, split=split, streaming=stream, **kwargs)


def peek(config: str, split: str = "train", n: int = 5) -> list[dict]:
    """Return the first `n` rows via streaming (reads only the first chunk)."""
    ds = load(config, split, stream=True)
    out, seen = [], 0
    for _, row in zip(range(n), ds):
        out.append(row)
    return out


def row_cost(row: dict, prices: dict[str, dict[str, float]]) -> float:
    """Cost of one executed row in $, given a pricing table keyed by model."""
    model = row.get("model_name")
    p = prices.get(model)
    if p is None:
        return float("nan")
    return (
        row.get("input_tokens", 0) * p["input_price"] / 1e6
        + row.get("output_tokens", 0) * p["output_price"] / 1e6
    )
