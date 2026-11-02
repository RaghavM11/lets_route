# Dataset — Directory Structure

Where `xRouteBench` data lives and how it is organized.

---

## 1. This repo's data layer

```
lets_route/
├── data/
│   ├── load_xroutebench.py    # lazy loader + helpers (no bulk download by default)
│   ├── download_to_raw.py     # download all parquet files into ./data/raw
│   └── raw/                   # project-local copy of the dataset (pinned)
│       ├── <config>/<split>-000.parquet   # e.g. llmrouter_generic/train-000.parquet
│       └── ... (41 parquet files, ~46 MB total)
├── docs/
│   ├── README.md
│   ├── xroutebench.md         # dataset reference (configs, schema, usage)
│   └── dataset-directory-structure.md  # this file
├── study_xroutebench.ipynb    # dataset study notebook (route_f kernel)
└── (the HF cache still holds a separate per-machine copy)
```

We keep a **project-local pinned copy** in `data/raw/` (mirroring the HF cache
layout) so the whole team shares the exact same files. Load it locally with:

```python
from datasets import load_dataset
ds = load_dataset("parquet",
                  data_files="data/raw/llmrouter_generic/train-000.parquet",
                  split="train")
```

Re-download (or fetch a single config) with:

```bash
python data/download_to_raw.py                # everything
python data/download_to_raw.py --config llmrouter_generic
```

---

## 2. Flow

```
query text ──> data/load_xroutebench.py  ──>  data/raw/ (or HF cache)  ──>  study/notebooks
```
