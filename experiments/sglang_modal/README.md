# Open-model SGLang endpoints on Modal

The three explicit tiers are Qwen3.5-0.8B (L4), Qwen2.5-7B-Instruct (L4),
and Qwen3.5-27B (H200). These are different model generations; record the full
model ID rather than treating tier size as an architecture-controlled ablation.
All revisions are pinned. A 70B-class model is intentionally still undecided.

Activate `.route_f`, install the Modal client, then authenticate once:

```bash
source .route_f/bin/activate
uv pip install --python .route_f/bin/python modal 'cbor2<6'
modal setup
```

The cbor2 constraint allows installation on Intel Macs without a Rust compiler.

Cache weights with a CPU function before starting a GPU server:

```bash
source .route_f/bin/activate
ROUTE_TIER=0.8b modal run experiments/sglang_modal/modal_app.py::cache
ROUTE_TIER=0.8b modal serve experiments/sglang_modal/modal_app.py
```

The second command prints the temporary endpoint URL and runs until interrupted.
Use `ROUTE_TIER=7b` or `ROUTE_TIER=27b` in **both** commands for another tier.
For a persistent deployment, replace `modal serve` with `modal deploy`.

Create a Modal Proxy Token in workspace settings. Put `MODAL_KEY`,
`MODAL_SECRET`, and `SGLANG_BASE_URL` (the printed URL, no trailing slash) in
your local `.env`. These are distinct from the Inception API key and Modal's
CLI login credentials. Never commit them.

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()
response = requests.post(
    os.environ["SGLANG_BASE_URL"] + "/v1/chat/completions",
    headers={"Modal-Key": os.environ["MODAL_KEY"],
             "Modal-Secret": os.environ["MODAL_SECRET"]},
    json={"model": "Qwen/Qwen3.5-0.8B",
          "messages": [{"role": "user", "content": "What is 2 + 2?"}],
          "temperature": 0, "max_tokens": 128,
          "chat_template_kwargs": {"enable_thinking": False}},
    timeout=1200,
)
response.raise_for_status()
print(response.json())
```

Each tier scales to zero after 60 idle seconds and allows at most one GPU
container. Startup, requests, CPU, memory, and the idle window are billable;
the 30-minute request timeout is not a dollar budget. Stop a temporary server
with Ctrl-C, or a deployment with `modal app stop APP_NAME`.

The server uses an 8,192-token context limit for an inexpensive smoke test.
Inspect prompt lengths before a full benchmark. Use identical prompts,
decoding limits, and thinking settings across tiers. This endpoint alone does
not score answers or build the counterfactual matrix.

The existing `../qwen3_5_27b_modal/` experiment provides the 12-query scored
multiple-choice smoke test and comparison against the recorded candidates.
