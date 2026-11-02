"""Inception Mercury 3 chat client.

Run a prompt with: python -m models.mercury "What is a diffusion model?"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

MODEL = "mercury-3"
ENDPOINT = "https://api.inceptionlabs.ai/v1/chat/completions"


def chat(
    messages: list[dict[str, str]],
    *,
    reasoning_effort: str = "low",
    timeout: float = 60,
) -> dict:
    """Return the full chat-completion response; raise on HTTP failures.

    Credentials come from INCEPTION_API_KEY or the project's local .env file.
    An existing environment variable takes precedence over .env.
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("INCEPTION_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set INCEPTION_API_KEY in your environment or project .env file.")
    response = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to Mercury 3.")
    parser.add_argument("prompt", nargs="?", default="What is a diffusion model?")
    args = parser.parse_args()
    print(json.dumps(chat([{"role": "user", "content": args.prompt}]), indent=2))


if __name__ == "__main__":
    main()
