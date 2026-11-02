# Fireworks hosted-model and Nexus Auto evaluation

This evaluation calls the Fireworks Chat Completions API directly. It does not
start Modal jobs, deploy GPUs, install Nexus CLI, or change coding-harness
settings. The API key comes only from `FIREWORKS_API_KEY` in the environment or
the ignored local `.env`. Requests never forward Anthropic, OpenAI, or AWS
credentials. Fireworks may still have account-level Provider Keys configured;
the actual serving model is recorded on every response.

## Models selected for this run

The authenticated catalog snapshot contains 319 model records. The exact
Qwen2.5-7B-Instruct, Qwen3.5-27B, and Qwen2.5-72B-Instruct entries are marked
`supportsServerless: false`; there is no matching Qwen3.5-0.8B entry. Therefore,
the user approved these four available Fireworks-hosted serverless baselines:

| Arm | Fireworks model ID |
| --- | --- |
| Nemotron 3.5 Lightning 30B-A3B | `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` |
| GPT-OSS 120B | `accounts/fireworks/models/gpt-oss-120b` |
| GLM 5.3 Flash | `accounts/fireworks/models/glm-5p3-flash` |
| Qwen 3.8 Max | `accounts/fireworks/models/qwen3p8-max` |
| Nexus Auto | `auto` |

The four models are alternatives, not size-matched or checkpoint-equivalent
replacements for the four Modal Qwen tiers. Nexus Auto manages its own model
pool and is not constrained to these four baselines. Do not interpret differences
as router-only gains or regret over a common candidate set.

## Run

From the repository root:

```bash
source .route_f/bin/activate
python -m experiments.fireworks_eval.discover
python -m experiments.fireworks_eval.evaluate
```

The default run has five arms on the unchanged ten-query fixture: 50 requests.
Run only the four pinned Fireworks models with:

```bash
python -m experiments.fireworks_eval.evaluate \
  --arms nemotron30b,gptoss120b,glmflash,qwenmax --run-id fireworks-pinned-only
```

Use `--arms auto --run-id fireworks-auto-only` to evaluate only Nexus Auto.
Each request uses the original prompt roles, temperature 0, seed 42, and a
1,024-token limit. Ground truth is used only by the local scorer. An interrupted
run resumes saved request records after verifying its prompt hash and settings.
To retry saved request errors, rerun the same command with `--retry-errors`.
Previous failed records are retained in `failed_request_history.json`; successful
model/task pairs are not repeated. Their successful request latencies and costs
remain separate from the archived failed requests, whose billing is unknown.
If a particular prompt repeatedly stalls in non-thinking mode, an explicit
`--retry-errors --retry-reasoning-effort low` recovery run changes effort only
for those failed pairs. The change is saved in `retry_settings_history.json`
and in each result's effective request; it is not a strictly identical-settings
rerun of that query.

Nexus Auto and GPT-OSS request low reasoning effort. GPT-OSS does not support
disabling reasoning. Other pinned arms request `none`; if a model explicitly
rejects that setting, the runner retries with `low` and records the changed
effective request and both attempts. This means reasoning settings are not
strictly identical to the non-thinking Modal run. Output limits may include
reasoning tokens, and a length-limited empty final answer is preserved and scored
as an empty response rather than fabricated from the reasoning trace.

## Artifacts

Results stay under the ignored `results/fireworks_fixed_ten/RUN_ID/` directory
unless the user explicitly requests publication:

- `all_results.txt`: English prompts, references, returned models, and answers.
- `results.json`: raw API responses, effective requests, attempts, local scores,
  token usage, and request latency; no authentication headers or API keys.
- `results.csv`: flattened response and scoring records.
- `summary.json`: results per arm and the actual model distribution for Auto.
- `score_matrix.csv`, `served_model_matrix.csv`, `latency_matrix.csv`, and
  `cost_matrix.csv`: ten-prompt comparison matrices.
- `modal_fireworks_comparison.csv`: a descriptive comparison with the prior
  Modal run when its prompt-manifest fingerprint matches.
- `catalog.json`: authenticated public-catalog snapshot from discovery.

Scoring reuses the local Modal evaluator: A-J multiple-choice extraction,
numeric matching for the fixed math questions, and normalized token F1. It is
not claimed to reproduce the official xRouteBench scorer. Fireworks latency
includes the client's internet round trip; Modal latency measured worker-local
HTTP calls, so those latency columns are not directly interchangeable.

Token costs use the documented Standard serverless rates and reported cached
tokens. Unknown serving-model rates remain null. Estimates are not invoices;
provider-specific account rates, pricing updates, missing cache details, and
failed attempts may change billed totals. The original xRouteBench prices are
not used to price Fireworks requests.

## Checks and sources

```bash
python -m unittest discover -s experiments/fireworks_eval -p 'test_*.py'
```

- https://docs.fireworks.ai/llms.txt
- https://docs.fireworks.ai/nexus/auto
- https://docs.fireworks.ai/api-reference/list-models
- https://docs.fireworks.ai/api-reference/post-chatcompletions
- https://docs.fireworks.ai/serverless/pricing
- https://docs.fireworks.ai/guides/reasoning
