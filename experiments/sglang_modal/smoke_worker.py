"""Run fixed prompts against a temporary SGLang HTTP server; score locally."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import signal
import string
import subprocess
import time
import urllib.request


def messages(query: str) -> list[dict]:
    system, user = "[System Instruction]", "[User Query]"
    if query.startswith(system) and user in query:
        a, b = query[len(system):].split(user, 1)
        return [{"role": "system", "content": a.strip()}, {"role": "user", "content": b.strip()}]
    return [{"role": "user", "content": query}]


def http_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if data else {}


def run_server(spec: dict, checkpoint: str, prompts: list[dict], max_tokens: int,
               output: Path, persist) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    command = ["python", "-m", "sglang.launch_server", "--model-path", checkpoint,
               "--served-model-name", spec["id"], "--host", "127.0.0.1", "--port", "30000",
               "--dtype", "bfloat16", "--context-length", "8192", "--mem-fraction-static", "0.80",
               "--tp", str(spec["tp"]), "--max-running-requests", "4"]
    url = "http://127.0.0.1:30000"
    rows = []
    started = time.perf_counter()
    failure = None
    startup_s = None
    with (output / "sglang.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                                   start_new_session=True)
        try:
            while time.perf_counter() - started < 900:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang exited with status {process.returncode}: " + (output / "sglang.log").read_text()[-6000:])
                try:
                    http_json(url + "/health", timeout=5)
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise TimeoutError("SGLang startup exceeded 900 seconds")
            startup_s = time.perf_counter() - started
            print(f"SGLang ready for {spec['id']} after {startup_s:.1f}s", flush=True)
            for prompt in prompts:
                before = time.perf_counter()
                response, error = None, None
                try:
                    response = http_json(url + "/v1/chat/completions", {
                        "model": spec["id"], "messages": messages(prompt["query"]),
                        "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                latency = time.perf_counter() - before
                record = {**prompt, "model_id": spec["id"], "model_revision": spec["revision"],
                          "latency_s": latency, "error": error, "api_response": response}
                rows.append(record)
                with (output / "outputs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                persist()
                print(f"{spec['id']} {prompt['fixed_id']}: {'ERROR '+error if error else 'completed'} in {latency:.2f}s", flush=True)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(failure, flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    result = {"model": spec, "rows": rows, "startup_s": startup_s,
              "worker_wall_s": time.perf_counter()-started, "gpu_info": info,
              "server_error": failure, "server_log_tail": (output / "sglang.log").read_text()[-12000:]}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    persist()
    return result


def score_response(task: dict, text: str) -> dict:
    """Transparent local scoring, not a claim of exact xRouteBench reproduction."""
    metric = task["metric"]
    reference = str(task["ground_truth"])
    if metric == "em_mc":
        # Include A-J for MMLU-Pro. Prefer an explicit final answer, then parentheses.
        matches = re.findall(r"(?:final\s+)?(?:answer|choice)(?:\s+is)?\s*[:=]?\s*\*{0,2}\(?([A-J])\)?(?:[.\s]|$)", text, re.I)
        if not matches:
            matches = re.findall(r"\(([A-J])\)", text, re.I)
        if not matches:
            matches = re.findall(r"(?:^|\n)\s*\*{0,2}([A-J])(?:[.)]|\s*$)", text, re.I)
        prediction = matches[-1].upper() if matches else None
        return {"score": float(prediction == reference.strip().upper()), "prediction": prediction, "scorer": "local_mc_A_to_J"}
    if metric == "f1_score":
        def normalize(value):
            value = value.lower().translate(str.maketrans('', '', string.punctuation))
            return re.sub(r"\b(a|an|the)\b", " ", value).split()
        a, b = normalize(text), normalize(reference)
        overlap = sum((Counter(a) & Counter(b)).values())
        score = (2 * overlap / (len(a) + len(b))) if a and b else float(a == b)
        return {"score": score, "prediction": text, "scorer": "local_normalized_token_f1"}
    if metric in {"GSM8K", "MATH"}:
        def number(value):
            if "####" in value:
                value = value.rsplit("####", 1)[1]
            boxed = re.findall(r"\\boxed\{([^{}]+)\}", value)
            if boxed:
                value = boxed[-1]
            values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
            if not values:
                return None
            try:
                return Decimal(values[-1].replace(',', ''))
            except InvalidOperation:
                return None
        predicted, expected = number(text), number(reference)
        return {"score": float(predicted is not None and predicted == expected),
                "prediction": str(predicted) if predicted is not None else None,
                "scorer": "local_last_numeric_answer_fixed_smoke_tasks"}
    return {"score": None, "prediction": None, "scorer": "unsupported"}
