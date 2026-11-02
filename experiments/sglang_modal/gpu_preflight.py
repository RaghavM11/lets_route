"""Check Modal GPU allocation before downloading large model checkpoints."""
import subprocess

import modal

app = modal.App("lets-route-gpu-preflight")


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              gpu="H100", cpu=1, memory=1024, timeout=60, scaledown_window=2)
def probe() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    )


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
