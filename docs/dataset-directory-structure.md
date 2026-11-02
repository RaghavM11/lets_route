# Dataset — Directory Structure

Where `xRouteBench` data lives and how it is organized.

---

## 1. Hugging Face cache (on-disk, per machine)

When you load a config with `datasets` (`stream=False`), `parquet` files are
downloaded and cached under Hugging Face's cache root:

```
~/.cache/huggingface/
└── datasets/
    ├── <owner>___<dataset>/
    │   ├── <config>/
    │   │   └── <version>/
    │   │       ├── <split>-00000-of-00001.parquet
    │   │       ├── dataset_info.json
    │   │       └── state.json
    │   └── ...
    └── <...>.lock            # lock files during download
```

Real example for this repo (owner `ulab-ai`, dataset `xRouteBench`):

```
~/.cache/huggingface/datasets/
└── ulab-ai___x_route_bench/
    ├── llmrouter_generic/0.0.0/   # train.parquet + test.parquet
    ├── llm_candidates/0.0.0/      # train.parquet (18 rows)
    └── ... (one dir per config)
```

Notes:
- Dataset names are normalized in the path (`xRouteBench` → `x_route_bench`).
- The cache is machine-local and is **never** committed to git.
- `stream=True` downloads parquet lazily in chunks and does not keep a full copy
  (it still uses the cache for fetched chunks).

---

## 2. This repo's data layer

```
lets_route/
├── data/
│   └── load_xroutebench.py    # lazy loader + helpers (no bulk download by default)
├── docs/
│   ├── README.md
│   ├── xroutebench.md         # dataset reference (configs, schema, usage)
│   └── dataset-directory-structure.md  # this file
├── study_xroutebench.ipynb    # dataset study notebook (README_f kernel)
└── (downloaded parquet files are NOT stored here — they live in the HF cache)
```

By design, raw dataset files are **not** copied into the repo; we always load
via `datasets` from the HF cache. If you later want a pinned local copy, add a
`data/raw/` folder and document it here.

---

## 3. Flow

```
query text ──> load("llmrouter_generic", split, stream=True/False)
                    │
                    ▼
               HF cache (~/.cache/huggingface/datasets/...)
                    │
                    ▼
        data/load_xroutebench.py  ->  study/notebooks
```
