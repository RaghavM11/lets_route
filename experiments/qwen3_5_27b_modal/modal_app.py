"""Modal launcher for the Qwen3.5-27B xRouteBench experiment.

Run from the repository root:

    uvx --from modal modal run experiments/qwen3_5_27b_modal/modal_app.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "fireworks-qwen35-27b-xroutebench"
MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_MOUNT = Path("/models")
RESULTS_MOUNT = Path("/results")
REMOTE_DATA_ROOT = Path("/workspace/data/raw")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent

model_volume = modal.Volume.from_name(
    "fireworks-qwen35-models-v2", create_if_missing=True, version=2
)
results_volume = modal.Volume.from_name(
    "fireworks-xroute-results-v2", create_if_missing=True, version=2
)

download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub==2.4.0",
)

experiment_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0",
        "transformers==5.22.1",
        "accelerate==1.15.0",
        "huggingface_hub==2.4.0",
        "safetensors==0.8.0",
        "numpy==2.5.4",
        "pandas==3.1.0",
        "pyarrow==26.0.0",
    )
    .add_local_dir(EXPERIMENT_ROOT, remote_path="/root/qwen3_5_27b_modal")
    .add_local_file(
        PROJECT_ROOT / "data/raw/llmrouter_generic/test-000.parquet",
        remote_path=str(REMOTE_DATA_ROOT / "llmrouter_generic/test-000.parquet"),
    )
    .add_local_file(
        PROJECT_ROOT / "data/raw/llm_candidates/train-000.parquet",
        remote_path=str(REMOTE_DATA_ROOT / "llm_candidates/train-000.parquet"),
    )
)

app = modal.App(APP_NAME)


def model_storage_path(model_id: str, revision: str) -> Path:
    """Map a Hub model/revision to a stable path inside the model Volume."""

    return MODEL_MOUNT / model_id.replace("/", "--") / revision


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT: model_volume},
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
)
def download_model(model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> str:
    """Download the pinned checkpoint once without paying for GPU time."""

    from huggingface_hub import snapshot_download

    destination = model_storage_path(model_id, revision)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
    )
    model_volume.commit()
    print(f"Model ready at {destination}", flush=True)
    return str(destination)


@app.function(
    image=experiment_image,
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
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    tasks: str = "mmlu,arc_challenge,openbook_qa,commonsense_qa,boolq,hellaswag",
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 32,
    gpu_usd_per_hour: float = 0.0,
    run_id: str = "",
) -> dict[str, Any]:
    """Run the reproducible Qwen benchmark on one H200."""

    import sys

    sys.path.insert(0, "/root")
    from qwen3_5_27b_modal.experiment import run_transformers_benchmark

    parsed_tasks = tuple(value.strip() for value in tasks.split(",") if value.strip())
    result = run_transformers_benchmark(
        model_path=model_storage_path(model_id, revision),
        model_id=model_id,
        model_revision=revision,
        data_root=REMOTE_DATA_ROOT,
        results_root=RESULTS_MOUNT,
        tasks=parsed_tasks,
        per_task=per_task,
        seed=seed,
        max_new_tokens=max_new_tokens,
        gpu_usd_per_hour=gpu_usd_per_hour,
        run_id=run_id or None,
    )
    results_volume.commit()
    return result


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)


@app.local_entrypoint()
def main(
    tasks: str = "mmlu,arc_challenge,openbook_qa,commonsense_qa,boolq,hellaswag",
    per_task: int = 2,
    seed: int = 42,
    max_new_tokens: int = 32,
    gpu_usd_per_hour: float = 0.0,
    run_id: str = "",
    output_dir: str = "results/qwen3_5_27b_modal",
) -> None:
    """Cache the model, run the H200 job, and mirror artifacts locally."""

    model_path = download_model.remote(MODEL_ID, MODEL_REVISION)
    print(f"Using cached model: {model_path}")
    result = run_benchmark.remote(
        MODEL_ID,
        MODEL_REVISION,
        tasks,
        per_task,
        seed,
        max_new_tokens,
        gpu_usd_per_hour,
        run_id,
    )

    summary = result["summary"]
    local_result_dir = Path(output_dir) / summary["run_id"]
    local_result_dir.mkdir(parents=True, exist_ok=False)
    (local_result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(local_result_dir / "query_results.csv", result["query_results"])
    write_csv(
        local_result_dir / "candidate_comparison.csv",
        result["candidate_comparison"],
    )

    print(f"Local artifacts: {local_result_dir}")
    print(f"Modal artifacts: {result['remote_result_dir']}")
    print(
        f"Accuracy: {summary['accuracy']:.1%} "
        f"({summary['correct']}/{summary['query_count']}); "
        f"mean latency: {summary['mean_latency_s']:.3f}s"
    )
