"""Serve four public Qwen checkpoints on Modal and run the frozen ten tasks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import modal

SPECS = {
    "0.8b": {"id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "gpu": "H100", "tp": 1, "gpu_usd_per_s": 0.001097},
    "7b": {"id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28", "gpu": "H100", "tp": 1, "gpu_usd_per_s": 0.001097},
    "27b": {"id": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654", "gpu": "H200", "tp": 1, "gpu_usd_per_s": 0.001261},
    "72b": {"id": "Qwen/Qwen2.5-72B-Instruct", "revision": "495f39366efef23836d0cfae4fbe635880d2be31", "gpu": "H200:2", "tp": 2, "gpu_usd_per_s": 0.002522},
}
app = modal.App("lets-route-fixed-ten-qwen")
weights = modal.Volume.from_name("lets-route-sglang-weights-v2", create_if_missing=True, version=2)
results = modal.Volume.from_name("lets-route-fixed-ten-results-v2", create_if_missing=True, version=2)
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.36.2", "hf_xet")
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.23").entrypoint([])
if modal.is_local():
    image = image.add_local_dir(Path(__file__).resolve().parent, "/root/smoke_run")


def checkpoint(key: str) -> Path:
    spec = SPECS[key]
    return Path("/models") / spec["id"].replace("/", "--") / spec["revision"]


def resume_runs(folder: Path, metadata: dict, keys: list[str]) -> tuple[list[dict], dict]:
    previous_path = folder / "runs.json"
    if not previous_path.exists():
        return [], metadata
    previous = json.loads(previous_path.read_text())
    for field in ("manifest_sha256", "max_new_tokens", "temperature", "seed", "thinking_enabled"):
        if previous["metadata"][field] != metadata[field]:
            raise ValueError(f"Cannot resume: {field} changed")
    runs = previous["runs"]
    if any(r["server_error"] or len(r["rows"]) != 10 for r in runs):
        raise ValueError("A previous tier is incomplete; use a new run ID for recovery")
    all_keys = list(dict.fromkeys(previous["metadata"]["tiers"] + keys))
    return runs, {**previous["metadata"], "tiers": all_keys, "expected_results": 10*len(all_keys)}


@app.function(image=download_image, cpu=4, memory=8192, timeout=7200,
              volumes={"/models": weights}, max_containers=1, scaledown_window=2)
def cache_model(key: str) -> str:
    from huggingface_hub import snapshot_download

    path = checkpoint(key)
    complete = path / "route_download_complete.json"
    if not complete.exists():
        spec = SPECS[key]
        snapshot_download(spec["id"], revision=spec["revision"], local_dir=str(path),
                          allow_patterns=["*.json", "*.jinja", "*.model", "*.tiktoken", "model*.safetensors", "*.txt"],
                          ignore_patterns=["original/*"], max_workers=8)
        complete.write_text(json.dumps(spec))
        weights.commit()
    print(f"Cached {key}: {sum(p.stat().st_size for p in path.rglob('*.safetensors'))/1e9:.2f} GB of weights", flush=True)
    return str(path)


@app.function(image=image, cpu=4, memory=65536, timeout=2400,
              volumes={"/models": weights.with_mount_options(read_only=True), "/results": results},
              max_containers=1, scaledown_window=2)
def run_tier(key: str, prompts: list[dict], max_tokens: int, run_id: str) -> dict:
    import sys
    sys.path.insert(0, "/root/smoke_run")
    from smoke_worker import run_server

    return run_server(SPECS[key], str(checkpoint(key)), prompts, max_tokens,
                      Path("/results") / run_id / key, results.commit)


def write_reports(folder: Path, runs: list[dict], tasks: list[dict], metadata: dict) -> None:
    import csv
    from experiments.sglang_modal.smoke_worker import score_response

    task_by_id = {t["fixed_id"]: t for t in tasks}
    rows = []
    text = ["LIVE MODAL / SGLANG RESULTS ON TEN FIXED TASKS",
            "New inference outputs, distinct from the dataset's recorded responses.",
            "Local scoring rules are transparent approximations, not the original benchmark evaluator.", ""]
    for run in runs:
        spec = run["model"]
        text.extend(["="*90, f"MODEL: {spec['id']} @ {spec['revision']}",
                     f"GPU: {run['gpu_info']}", f"Server error: {run['server_error']}", ""])
        for record in run["rows"]:
            task = task_by_id[record["fixed_id"]]
            response = record["api_response"]
            choice = response["choices"][0] if response else {}
            answer = (choice.get("message", {}).get("content") or "")
            usage = response.get("usage", {}) if response else {}
            score = score_response(task, answer) if not record["error"] else {"score": None, "prediction": None, "scorer": "request_error"}
            row = {"fixed_id": task["fixed_id"], "query_uid": task["query_uid"],
                   "task_name": task["task_name"], "metric": task["metric"],
                   "model_id": spec["id"], "model_revision": spec["revision"],
                   "query": task["query"], "ground_truth": task["ground_truth"], "response": answer,
                   **score, "input_tokens": usage.get("prompt_tokens"),
                   "output_tokens": usage.get("completion_tokens"), "finish_reason": choice.get("finish_reason"),
                   "latency_s": record["latency_s"], "error": record["error"],
                   "generation_gpu_cost_estimate_usd": record["latency_s"]*spec["gpu_usd_per_s"]}
            rows.append(row)
            text.extend(["-"*90, f"{task['fixed_id']} | {task['task_name']} | {task['metric']}",
                         f"Local score: {score['score']} ({score['scorer']}) | finish_reason={row['finish_reason']}",
                         f"Input/output tokens: {row['input_tokens']}/{row['output_tokens']} | latency: {record['latency_s']:.3f}s",
                         f"Error: {record['error']}", "", "[ORIGINAL PROMPT]", task["query"],
                         "", "[REFERENCE ANSWER]", str(task["ground_truth"]), "", "[LIVE MODEL OUTPUT]", answer, ""])
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"runs.json").write_text(json.dumps({"metadata": metadata, "runs": runs}, ensure_ascii=False, indent=2))
    (folder/"all_results.txt").write_text("\n".join(text))
    (folder/"scored_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if rows:
        with (folder/"scored_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        model_ids = [run["model"]["id"] for run in runs]
        for filename, field in (("score_matrix.csv", "score"), ("latency_matrix.csv", "latency_s")):
            indexed = {(r["fixed_id"], r["model_id"]): r[field] for r in rows}
            with (folder/filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["fixed_id", "query_uid", *model_ids])
                writer.writeheader()
                for task in tasks:
                    writer.writerow({"fixed_id": task["fixed_id"], "query_uid": task["query_uid"],
                                     **{model_id: indexed.get((task["fixed_id"], model_id)) for model_id in model_ids}})
    summary = {**metadata, "completed_results": len(rows), "model_summaries": []}
    for run in runs:
        subset = [r for r in rows if r["model_id"] == run["model"]["id"]]
        scored = [r["score"] for r in subset if r["score"] is not None]
        summary["model_summaries"].append({"model_id": run["model"]["id"], "results": len(subset),
            "request_errors": sum(r["error"] is not None for r in subset),
            "length_limited_count": sum(r["finish_reason"] == "length" for r in subset),
            "mean_local_score": sum(scored)/len(scored) if scored else None,
            "full_score_count": sum(r["score"] == 1 for r in subset),
            "mean_latency_s": sum(r["latency_s"] for r in subset)/len(subset) if subset else None,
            "server_error": run["server_error"], "worker_wall_s": run["worker_wall_s"],
            "worker_gpu_cost_estimate_usd": run["worker_wall_s"]*run["model"]["gpu_usd_per_s"]})
    (folder/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


@app.local_entrypoint()
def main(tiers: str = "0.8b,7b,27b,72b", max_new_tokens: int = 1024,
         run_id: str = "", cache_only: bool = False) -> None:
    keys = [key.strip() for key in tiers.split(",")]
    if not keys or any(key not in SPECS for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("Select distinct tiers from 0.8b,7b,27b,72b")
    if not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens must be between 1 and 2048")
    run_id = run_id or datetime.now(timezone.utc).strftime("qwen-four-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Unsafe run ID")
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "fixtures/smoke10/tasks.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    assert len(tasks) == len({t["query_uid"] for t in tasks}) == 10
    # Only prompts and identifiers are sent to the serving worker; references stay local.
    prompts = [{k: t[k] for k in ("fixed_id", "query_uid", "query", "task_name", "metric")} for t in tasks]
    folder = root / "results/sglang_fixed_ten" / run_id
    metadata = {"run_id": run_id, "tiers": keys, "max_new_tokens": max_new_tokens,
                "temperature": 0, "seed": 42, "thinking_enabled": False,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "unique_prompts": 10, "expected_results": 10*len(keys),
                "price_source": "https://modal.com/pricing (2026-11-02); GPU estimates exclude CPU, RAM and image/caching overhead"}
    runs, metadata = resume_runs(folder, metadata, keys)
    completed_models = {r["model"]["id"] for r in runs}
    print(f"Run: {run_id}; local artifacts: {folder}", flush=True)
    for key in keys:
        if SPECS[key]["id"] in completed_models:
            print(f"Reusing saved ten outputs for {SPECS[key]['id']}", flush=True)
            continue
        print(f"Caching {SPECS[key]['id']} on CPU", flush=True)
        print(cache_model.remote(key), flush=True)
        if cache_only:
            continue
        spec = SPECS[key]
        print(f"Allocating {spec['gpu']} and serving the same ten prompts", flush=True)
        run = run_tier.with_options(gpu=spec["gpu"], memory=196608 if key == "72b" else 65536).remote(key, prompts, max_new_tokens, run_id)
        runs.append(run)
        write_reports(folder, runs, tasks, metadata)
        if run["server_error"]:
            raise RuntimeError(f"Serving failed for {key}; diagnostics saved in {folder}")
    if not cache_only:
        print((folder/"summary.json").read_text(), flush=True)
