# Repository conventions

- Use English for all saved files: code comments, documentation, report headings,
  annotations, generated summaries, and artifact filenames. Chat may use the
  user's preferred language.
- Preserve original dataset prompts, reference answers, model responses, and
  scores verbatim. English report formatting must not rewrite source records.
- Keep full downloaded datasets under `data/raw/`, ignored by Git. Small curated
  fixtures and their readable reports may be committed under `fixtures/`.
- Never commit API keys, `.env`, cloud credentials, virtual environments, or
  model checkpoints. Keep generated experiment results local unless the user
  explicitly requests publication; scope any ignore exception to that run.
- Reuse `fixtures/smoke10/tasks.json` for the fixed ten-task comparison; do not
  silently resample or replace it.
