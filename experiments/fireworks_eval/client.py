"""Minimal Fireworks-only client. Credentials are never included in artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CHAT_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
CATALOG_ENDPOINT = "https://api.fireworks.ai/v1/accounts/fireworks/models"


class FireworksError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Fireworks HTTP {status}: {message}")
        self.status = status


class FireworksClient:
    def __init__(self, key: str | None = None):
        self._key = key or os.environ.get("FIREWORKS_API_KEY") or dotenv_values(ROOT/".env").get("FIREWORKS_API_KEY")
        if not self._key:
            raise ValueError("Set FIREWORKS_API_KEY in the ignored local .env or environment.")
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Connection": "close"})

    def sanitize(self, value):
        return json.loads(json.dumps(value, ensure_ascii=False).replace(self._key, "[REDACTED]"))

    def request(self, method: str, url: str, *, payload=None, params=None, timeout=180):
        if url not in {CHAT_ENDPOINT, CATALOG_ENDPOINT}:
            raise ValueError("This client sends credentials only to the configured Fireworks endpoints.")
        response = self._session.request(method, url, json=payload, params=params, timeout=timeout, allow_redirects=False)
        try:
            data = response.json()
        except ValueError:
            data = {"message": response.text[:2000]}
        data = self.sanitize(data)
        if not 200 <= response.status_code < 300:
            raise FireworksError(response.status_code, json.dumps(data, ensure_ascii=False)[:2000])
        return data

    def catalog(self) -> list[dict]:
        records, token = [], None
        for _ in range(30):
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            data = self.request("GET", CATALOG_ENDPOINT, params=params, timeout=60)
            records.extend(data.get("models", []))
            token = data.get("nextPageToken")
            if not token:
                return records
        raise RuntimeError("Catalog pagination exceeded the safety limit.")

    def chat(self, payload: dict):
        return self.request("POST", CHAT_ENDPOINT, payload=payload)
