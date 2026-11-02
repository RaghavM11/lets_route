#!/usr/bin/env python
"""Download all xRouteBench parquet files into ./data/raw/<config>/.

Mirrors the HF cache layout into a project-local folder so the team shares
a pinned copy of the dataset. Uses only the standard library.

Usage:
    python data/download_to_raw.py [--config CONFIG]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

DATASET = "ulab-ai/xRouteBench"
PARQUET_API = f"https://huggingface.co/api/datasets/{DATASET}/parquet"
RAW = pathlib.Path(__file__).resolve().parent / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="only download this config (optional)")
    args = ap.parse_args()

    with urllib.request.urlopen(PARQUET_API, timeout=60) as resp:
        index = json.load(resp)  # {config: {split: [url,...]}}

    if args.config:
        if args.config not in index:
            raise SystemExit(f"config not found: {args.config}")
        index = {args.config: index[args.config]}

    total = 0
    for config, splits in sorted(index.items()):
        for split, urls in sorted(splits.items()):
            for i, url in enumerate(urls):
                dest = RAW / config / f"{split}-{i:03d}.parquet"
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"downloading {config}/{split} -> {dest}")
                urllib.request.urlretrieve(url, dest)
                total += dest.stat().st_size

    print(f"\nDone. Wrote {total/1e6:.1f} MB under {RAW}")


if __name__ == "__main__":
    main()
