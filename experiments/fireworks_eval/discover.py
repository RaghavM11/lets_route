"""Save the public Fireworks catalog and show candidate Qwen matches."""
import json
from pathlib import Path

from .client import FireworksClient, FireworksError, ROOT


def main():
    folder = ROOT / "results/fireworks_fixed_ten/fireworks-eval-20261102"
    folder.mkdir(parents=True, exist_ok=True)
    client = FireworksClient()
    try:
        models = client.catalog()
    except FireworksError as exc:
        (folder / "preflight.json").write_text(json.dumps({"status": "failed", "http_status": exc.status, "error": str(exc)}, indent=2))
        print(str(exc))
        raise SystemExit(1)
    (folder / "catalog.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))
    print(f"Catalog models: {len(models)}")
    for model in models:
        if "qwen" in (model.get("name", "") + model.get("displayName", "")).lower():
            fields = {k: v for k, v in model.items() if k in {"name", "displayName", "huggingFaceUrl", "supportsServerless", "state"}}
            fields["parameterCount"] = model.get("baseModelDetails", {}).get("parameterCount")
            print(json.dumps(fields))


if __name__ == "__main__":
    main()
