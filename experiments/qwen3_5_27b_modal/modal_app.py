"""Run Qwen3.5-27B xRouteBench experiments with SGLang on Modal."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_MOUNT = Path("/models")
RESULTS_MOUNT = Path("/results")
DATA_ROOT = Path("/workspace/data/raw")
DEFAULT_TASKS = "mmlu,arc_challenge,openbook_qa,commonsense_qa,boolq,hellaswag"

app = modal.App("fireworks-qwen35-27b-xroutebench")
model_volume = modal.Volume.from_name(
    "fireworks-qwen35-models-v2", create_if_missing=True, version=2
)
results_volume = modal.Volume.from_name(
    "fireworks-xroute-results-v2", create_if_missing=True, version=2
)

download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub==2.4.0"
)
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.23")
    .entrypoint([])
    .uv_pip_install("pandas==3.1.0", "pyarrow==26.0.0")
)

if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    experiment_root = Path(__file__).resolve().parent
    sglang_image = (
        sglang_image
        .add_local_dir(experiment_root, "/root/qwen3_5_27b_modal")
        .add_local_file(
            project_root / "data/raw/llmrouter_generic/test-000.parquet",
            str(DATA_ROOT / "llmrouter_generic/test-000.parquet"),
        )
        .add_local_file(
            project_root / "data/raw/llm_candidates/train-000.parquet",
            str(DATA_ROOT / "llm_candidates/train-000.parquet"),
        )
    )


def model_path() -> Path:
    return MODEL_MOUNT / MODEL_ID.replace("/", "--") / MODEL_REVISION


@app.function(
    image=download_image,
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={MODEL_MOUNT: model_volume},
)
def cache_model() -> str:
    from huggingface_hub import snapshot_download

    destination = model_path()
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=destination)
    model_volume.commit()
    return str(destination)


@app.function(
    image=sglang_image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        MODEL_MOUNT: model_volume.with_mount_options(read_only=True),
        RESULTS_MOUNT: results_volume,
    },
)
def run_benchmark(
    tasks: str = DEFAULT_TASKS,
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 256,
    gpu_usd_per_hour: float = 0,
    run_id: str = "",
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from qwen3_5_27b_modal.experiment import run_sglang_benchmark

    result = run_sglang_benchmark(
        model_path=model_path(),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        data_root=DATA_ROOT,
        results_root=RESULTS_MOUNT,
        tasks=tuple(value.strip() for value in tasks.split(",") if value.strip()),
        per_task=per_task,
        seed=seed,
        max_new_tokens=max_new_tokens,
        gpu_usd_per_hour=gpu_usd_per_hour,
        run_id=run_id or None,
    )
    results_volume.commit()
    return result


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@app.local_entrypoint()
def main(
    tasks: str = DEFAULT_TASKS,
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 256,
    gpu_usd_per_hour: float = 0,
    run_id: str = "",
    output_dir: str = "results/qwen3_5_27b_modal",
) -> None:
    print(f"Cached checkpoint: {cache_model.remote()}")
    result = run_benchmark.remote(
        tasks, per_task, seed, max_new_tokens, gpu_usd_per_hour, run_id
    )

    summary = result["summary"]
    destination = Path(output_dir) / summary["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(destination / "query_results.csv", result["query_results"])
    _write_csv(
        destination / "candidate_comparison.csv", result["candidate_comparison"]
    )
    print(f"Local artifacts: {destination}")
    print(f"Modal artifacts: {result['remote_result_dir']}")
    print(
        f"Accuracy: {summary['accuracy']:.1%} "
        f"({summary['correct']}/{summary['query_count']}); "
        f"mean latency: {summary['mean_latency_s']:.3f}s"
    )
