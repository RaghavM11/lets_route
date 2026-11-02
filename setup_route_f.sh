#!/usr/bin/env bash
# Create a "route_f" uv-managed environment for this project.
# Cross-platform: works on Linux (CUDA) and macOS (MPS / CPU).
# Usage:
#   chmod +x setup_route_f.sh
#   ./setup_route_f.sh
set -euo pipefail

ENV_DIR=".route_f"
PY_VERSION="3.12"

# ---------------------------------------------------------------------------
# Detect OS. 'uv' works on both Linux and macOS.
# ---------------------------------------------------------------------------
OS="$(uname -s)"
echo "== Detected OS: ${OS} =="

echo "== Creating uv venv: ${ENV_DIR} (py${PY_VERSION}) =="
uv venv "${ENV_DIR}" --python "${PY_VERSION}"
# uv creates a .venv-style layout; the interpreter is at .route_f/bin/python

# ---------------------------------------------------------------------------
# Install torch from PyPI. On Linux, PyPI wheels already bundle CUDA support;
# on macOS they use the MPS/CPU backend. No special index needed.
# ---------------------------------------------------------------------------
echo "== Installing torch 2.15.0 from PyPI (CUDA on Linux / MPS on macOS) =="
uv pip install --python "${ENV_DIR}/bin/python" "torch==2.15.0"

# ---------------------------------------------------------------------------
# Core ML/training stack
# ---------------------------------------------------------------------------
echo "== Installing core ML/training stack =="
uv pip install --python "${ENV_DIR}/bin/python" \
  "transformers>=4.57" \
  peft==0.21.2 \
  datasets==5.1.0 \
  huggingface_hub==2.4.0 \
  accelerate==1.15.0 \
  safetensors==0.8.0 \
  tokenizers==0.23.3 \
  numpy==2.5.4 \
  pandas==3.1.0 \
  tqdm==4.70.1 \
  pyarrow==26.0.0

# ---------------------------------------------------------------------------
# Jupyter kernel packages + register kernel
# ---------------------------------------------------------------------------
echo "== Jupyter kernel packages =="
uv pip install --python "${ENV_DIR}/bin/python" \
  ipykernel ipython jupyter_client

echo "== Register Jupyter kernel 'route_f' =="
"${ENV_DIR}/bin/python" -m ipykernel install --user \
  --name route_f --display-name "route_f (py3.12)"

# ---------------------------------------------------------------------------
# Optional: verl / FSDP2 notebook stack
# Uncomment if you run notebooks that need the verl training path.
# ---------------------------------------------------------------------------
# echo "== Optional verl stack =="
# uv pip install --python "${ENV_DIR}/bin/python" verl tensordict

echo
echo "Done."
echo "  Local activate : source ${ENV_DIR}/bin/activate"
echo "  VS Code        : open the 'route_f (py3.12)' kernel in any notebook."
