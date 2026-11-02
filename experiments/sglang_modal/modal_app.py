"""Authenticated, scale-to-zero SGLang serving. Select a tier with ROUTE_TIER."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

TIERS = {
    "0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "L4"),
    "7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "L4"),
    "27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "H200"),
}
TIER = os.environ.get("ROUTE_TIER", "0.8b").lower()
if TIER not in TIERS:
    raise ValueError(f"Unknown ROUTE_TIER {TIER!r}; choose {', '.join(TIERS)}")
MODEL_ID, REVISION, GPU = TIERS[TIER]
MODEL_PATH = Path("/models") / MODEL_ID.replace("/", "--") / REVISION

app = modal.App(f"lets-route-sglang-{TIER.replace('.', '-')}")
weights = modal.Volume.from_name("lets-route-sglang-weights", create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub==2.4.0"
)
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.23").entrypoint([])


@app.function(
    image=download_image, cpu=2, memory=8192, timeout=7200,
    volumes={"/models": weights},
)
def cache_weights() -> str:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(MODEL_PATH))
    weights.commit()
    return str(MODEL_PATH)


def server_command() -> list[str]:
    return [
        "python", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_PATH),
        "--served-model-name", MODEL_ID,
        "--host", "0.0.0.0", "--port", "30000",
        "--dtype", "bfloat16", "--context-length", "8192",
        "--mem-fraction-static", "0.80", "--max-running-requests", "16",
    ]


@app.function(
    image=image, gpu=GPU, cpu=4, memory=32768, timeout=1800,
    min_containers=0, max_containers=1, scaledown_window=60,
    volumes={"/models": weights.with_mount_options(read_only=True)},
    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)
@modal.concurrent(max_inputs=16)
@modal.web_server(30000, startup_timeout=900, requires_proxy_auth=True)
def serve() -> None:
    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError("Cache this tier first with modal run ...::cache")
    subprocess.Popen(server_command())


@app.local_entrypoint()
def cache() -> None:
    print(f"Cached {MODEL_ID}@{REVISION}: {cache_weights.remote()}")
