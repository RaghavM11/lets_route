"""Evaluate four Fireworks-hosted baselines and Nexus Auto on ten fixed tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

from experiments.sglang_modal.smoke_worker import messages, score_response
from .client import FireworksClient, FireworksError, ROOT

ARMS = {
    "auto": {"model": "auto", "effort": "low", "description": "Nexus Auto managed candidate pool"},
    "nemotron30b": {"model": "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b", "effort": "none", "description": "Nemotron 3.5 Lightning 30B-A3B"},
    "gptoss120b": {"model": "accounts/fireworks/models/gpt-oss-120b", "effort": "low", "description": "GPT-OSS 120B; reasoning cannot be disabled"},
    "glmflash": {"model": "accounts/fireworks/models/glm-5p3-flash", "effort": "none", "description": "GLM 5.3 Flash"},
    "qwenmax": {"model": "accounts/fireworks/models/qwen3p8-max", "effort": "none", "description": "Qwen 3.8 Max"},
}
# USD per million uncached input / cached input / output tokens, Standard mode.
RATES = {
    "nemotron-lightning-3p5-30b-a3b": (0.05, 0.01, 0.20),
    "gpt-oss-120b": (0.15, 0.015, 0.60),
    "glm-5p3-flash": (0.15, 0.03, 0.50),
    "qwen3p8-max": (2.00, 0.25, 6.00),
    "qwen3p8-2p4t-a95b": (2.00, 0.25, 6.00),
    "glm-5p3": (1.40, 0.26, 4.40),
    "kimi-k3": (3.00, 0.30, 15.00),
    "deepseek-v4p1-flash": (0.30, 0.006, 1.20),
    "deepseek-v4p1-flash-1017": (0.30, 0.006, 1.20),
}
PRICE_SOURCE = "https://docs.fireworks.ai/serverless/pricing"
# Verified against the authenticated catalog's displayName, preserving raw model fields.
PRICE_MODEL_ALIASES = {"Qwen 3.8 Max": "qwen3p8-max"}


def estimate_cost(usage: dict, served_model: str) -> float | None:
    key = served_model.rsplit("/", 1)[-1]
    rates = RATES.get(PRICE_MODEL_ALIASES.get(key,key))
    if rates is None or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cached = min(max(cached, 0), usage["prompt_tokens"])
    return ((usage["prompt_tokens"]-cached)*rates[0] + cached*rates[1] + usage["completion_tokens"]*rates[2]) / 1e6


def invoke(client: FireworksClient, payload: dict) -> tuple[dict | None, list[dict], str | None, dict]:
    attempts = []
    effective = dict(payload)
    adjusted = False
    for attempt in range(3):
        before = time.perf_counter()
        try:
            response = client.chat(effective)
            attempts.append({"status": 200, "latency_s": time.perf_counter()-before,
                             "reasoning_effort": effective.get("reasoning_effort")})
            return response, attempts, None, effective
        except FireworksError as exc:
            attempts.append({"status": exc.status, "latency_s": time.perf_counter()-before,
                             "error": str(exc), "reasoning_effort": effective.get("reasoning_effort")})
            if exc.status in {401, 402, 403}:
                raise
            if (exc.status == 400 and not adjusted and
                any(word in str(exc).lower() for word in ("reasoning", "thinking"))):
                if effective.get("reasoning_effort") == "none":
                    effective["reasoning_effort"] = "low"
                else:
                    effective.pop("reasoning_effort", None)
                adjusted = True
                continue
            if exc.status == 429 and attempt < 2:
                time.sleep(2*(attempt+1))
                continue
            return None, attempts, str(exc), effective
        except requests.RequestException as exc:
            # Do not retry ambiguous POST timeouts: generation may already have billed.
            error = client.sanitize(str(exc))
            attempts.append({"status": None, "latency_s": time.perf_counter()-before, "error": error})
            return None, attempts, error, effective
    return None, attempts, attempts[-1].get("error", "Retry limit exceeded"), effective


def record_result(task: dict, arm: str, response: dict | None, attempts: list[dict],
                  error: str | None, effective: dict) -> dict:
    response = response or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    answer = (choice.get("message") or {}).get("content") or ""
    if not error and not choices:
        error = "Successful HTTP response contained no completion choice"
    usage = response.get("usage") or {}
    served = response.get("model") or ""
    local = score_response(task, answer) if not error else {"score": None, "prediction": None, "scorer": "request_error"}
    return {
        "arm": arm, "requested_model": ARMS[arm]["model"], "served_model": served,
        "auto_model_resolved": bool(served and served != "auto") if arm == "auto" else None,
        "fixed_id": task["fixed_id"], "query_uid": task["query_uid"], "task_name": task["task_name"],
        "metric": task["metric"], "query": task["query"], "ground_truth": task["ground_truth"],
        "response": answer, **local, "input_tokens": usage.get("prompt_tokens"),
        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "output_tokens": usage.get("completion_tokens"), "finish_reason": choice.get("finish_reason"),
        "reasoning_effort": effective.get("reasoning_effort"),
        "latency_s": sum(a["latency_s"] for a in attempts),
        "estimated_token_cost_usd": estimate_cost(usage, served), "error": error,
        "empty_answer": not bool(answer.strip()), "attempts": attempts,
        "effective_request": effective, "api_response": response,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def reports(folder: Path, rows: list[dict], metadata: dict, tasks: list[dict], state="running", failure=None):
    (folder/"results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    flat = [{k: v for k,v in row.items() if k not in {"attempts", "effective_request", "api_response"}} for row in rows]
    write_csv(folder/"results.csv", flat)
    text = ["FIREWORKS-HOSTED MODEL AND NEXUS AUTO EVALUATION", "Same ten fixed prompts; live Fireworks API responses.",
            "The four hosted baselines are not size-matched equivalents of the Modal Qwen checkpoints.",
            "Local scoring is shared with the Modal run and is not the official xRouteBench evaluator.", ""]
    for row in rows:
        text.extend(["="*90, f"ARM: {row['arm']} | REQUESTED MODEL: {row['requested_model']}",
                     f"SERVED MODEL: {row['served_model']}", f"{row['fixed_id']} | {row['task_name']} | {row['metric']}",
                     f"Local score: {row['score']} | scorer: {row['scorer']}",
                     f"Input/cached/output tokens: {row['input_tokens']}/{row['cached_input_tokens']}/{row['output_tokens']}",
                     f"Request latency: {row['latency_s']:.3f}s | finish reason: {row['finish_reason']} | effort: {row['reasoning_effort']}",
                     f"Estimated token cost (not invoice): {row['estimated_token_cost_usd']}",
                     f"Request error: {row['error']}", "", "[ORIGINAL PROMPT]", row['query'],
                     "", "[REFERENCE ANSWER]", str(row['ground_truth']), "", "[LIVE MODEL OUTPUT]", row['response'], ""])
    (folder/"all_results.txt").write_text("\n".join(text))
    arms = metadata["arms"]
    for filename,field in (("score_matrix.csv","score"),("latency_matrix.csv","latency_s"),
                           ("served_model_matrix.csv","served_model"),("cost_matrix.csv","estimated_token_cost_usd")):
        lookup={(r['fixed_id'],r['arm']):r[field] for r in rows}
        write_csv(folder/filename,[{"fixed_id":t['fixed_id'],"query_uid":t['query_uid'],
                                   **{arm:lookup.get((t['fixed_id'],arm)) for arm in arms}} for t in tasks])
    summaries=[]
    for arm in arms:
        subset=[r for r in rows if r['arm']==arm]
        scored=[r['score'] for r in subset if r['score'] is not None]
        summaries.append({"arm":arm,"requested_model":ARMS[arm]['model'],"completed_requests":len(subset),
            "request_errors":sum(r['error'] is not None for r in subset),
            "full_score_count":sum(r['score']==1 for r in subset),
            "mean_local_score":sum(scored)/len(scored) if scored else None,
            "mean_request_latency_s":sum(r['latency_s'] for r in subset)/len(subset) if subset else None,
            "length_limited_count":sum(r['finish_reason']=='length' for r in subset),
            "empty_answer_count":sum(r['empty_answer'] for r in subset),
            "priced_request_count":sum(r['estimated_token_cost_usd'] is not None for r in subset),
            "estimated_token_cost_usd":sum(r['estimated_token_cost_usd'] or 0 for r in subset),
            "served_model_counts":dict(Counter(r['served_model'] for r in subset if r['served_model']))})
    summary={**metadata,"state":state,"failure":failure,"recorded_requests":len(rows),"summaries":summaries}
    (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    comparison=[]
    modal=ROOT/"results/sglang_fixed_ten/qwen-four-fixed-ten-20261102/summary.json"
    if modal.exists():
        prior=json.loads(modal.read_text())
        if prior['manifest_sha256']==metadata['manifest_sha256']:
            comparison.extend({"provider":"Modal/SGLang","model":r['model_id'],"results":r['results'],
                               "full_score_count":r['full_score_count'],"mean_local_score":r['mean_local_score'],
                               "latency_s":r['mean_latency_s'],"latency_scope":"worker-local HTTP; excludes internet round trip"} for r in prior['model_summaries'])
    comparison.extend({"provider":"Fireworks","model":r['requested_model'],"results":r['completed_requests'],
                       "full_score_count":r['full_score_count'],"mean_local_score":r['mean_local_score'],
                       "latency_s":r['mean_request_latency_s'],"latency_scope":"client API request including internet round trip"} for r in summaries)
    write_csv(folder/"modal_fireworks_comparison.csv",comparison)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',default=','.join(ARMS))
    parser.add_argument('--run-id',default='fireworks-eval-20261102')
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--retry-errors',action='store_true',help='Retry only saved request errors and preserve their history')
    args=parser.parse_args()
    if not args.run_id.replace('-','').replace('_','').isalnum():
        raise ValueError('Unsafe run ID')
    arms=args.arms.split(',')
    if any(arm not in ARMS for arm in arms) or len(arms)!=len(set(arms)):
        raise ValueError('Unknown or repeated arm')
    if not 1<=args.max_tokens<=4096:
        raise ValueError('Token limit must be 1..4096')
    folder=ROOT/'results/fireworks_fixed_ten'/args.run_id
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'fixtures/smoke10/tasks.json'
    tasks=json.loads(manifest_path.read_text())['tasks']
    assert len(tasks)==len({t['query_uid'] for t in tasks})==10
    metadata={"run_id":args.run_id,"arms":arms,"max_tokens":args.max_tokens,"temperature":0,"seed":42,
              "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "created_at":datetime.now(timezone.utc).isoformat(),"expected_requests":10*len(arms),
              "price_source":PRICE_SOURCE,"rates_usd_per_million":RATES,
              "price_model_aliases":PRICE_MODEL_ALIASES,
              "candidate_pool_note":"Nexus Auto manages its own pool; it is not constrained to the four pinned baselines.",
              "model_match_note":"Four serverless alternatives, not the same checkpoints or parameter tiers as Modal.",
              "provider_credentials_forwarded":False}
    previous=folder/'results.json'
    rows=json.loads(previous.read_text()) if previous.exists() else []
    old={}
    if rows:
        old=json.loads((folder/'summary.json').read_text())
        if set(old['arms'])!=set(arms):
            raise ValueError('Cannot resume a different set of arms')
        for field in ('max_tokens','manifest_sha256','temperature','seed'):
            if old[field]!=metadata[field]: raise ValueError(f'Cannot resume changed {field}')
        metadata['created_at']=old['created_at']
        metadata['arms']=old['arms']
    metadata['execution_order']=arms
    metadata['retried_error_records']=old.get('retried_error_records',0)
    if args.retry_errors:
        failed=[r for r in rows if r['error'] is not None]
        if failed:
            history_path=folder/'failed_request_history.json'
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            history.extend(failed)
            history_path.write_text(json.dumps(history,ensure_ascii=False,indent=2))
            rows=[r for r in rows if r['error'] is None]
            metadata['retried_error_records']=len(history)
    done={(r['arm'],r['fixed_id']) for r in rows}
    client=FireworksClient()
    for arm in arms:
        for task in tasks:
            if (arm,task['fixed_id']) in done: continue
            payload={"model":ARMS[arm]['model'],"messages":messages(task['query']),"temperature":0,
                     "seed":42,"max_tokens":args.max_tokens,"reasoning_effort":ARMS[arm]['effort']}
            try:
                response,attempts,error,effective=invoke(client,payload)
            except FireworksError as exc:
                reports(folder,rows,metadata,tasks,'blocked',str(exc))
                raise SystemExit(str(exc))
            row=record_result(task,arm,response,attempts,error,effective)
            rows.append(client.sanitize(row))
            reports(folder,rows,metadata,tasks)
            print(f"{arm} {task['fixed_id']}: served={row['served_model']} score={row['score']} finish={row['finish_reason']} error={row['error']}",flush=True)
    reports(folder,rows,metadata,tasks,'complete')
    print(f"Saved {len(rows)} Fireworks request records to {folder}",flush=True)


if __name__=='__main__':
    main()
