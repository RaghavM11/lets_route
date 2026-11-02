# Fireworks evaluation results

All 50 planned model/task pairs were attempted using Fireworks APIs: four pinned
hosted baselines plus Nexus Auto on the same ten fixed prompts. There are 49
valid responses and one unresolved service failure, Nemotron `smoke-07`. Its
score is null, not an incorrect-answer label. A final low-reasoning-effort
recovery attempt also timed out. Other successful responses were not rerun.

Read `all_results.txt` for English outputs and `summary.json` for per-arm counts.
`failed_request_history.json`, `retry_settings_history.json`, and
`operational_notes.json` preserve prior service failures, the explicit setting
change, and an interrupted request. Unknown billing for failed/interrupted
requests is excluded from successful-token cost estimates.

All ten successful Nexus Auto responses reported GLM 5.3 Flash as their serving
model. Its managed pool is not restricted to the four pinned baselines. These
Fireworks models are not the original four Modal Qwen checkpoints.

The complete public catalog cache and all credentials remain local and ignored;
`catalog_selection.json` contains only the relevant catalog evidence.
