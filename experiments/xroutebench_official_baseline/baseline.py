"""Reusable helpers for the xRouteBench all-candidate baseline experiment."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REQUIRED_ROUTING_COLUMNS = {
    "task_name",
    "query",
    "metric",
    "embedding_id",
    "model_name",
    "input_tokens",
    "output_tokens",
    "response_time",
    "performance",
}

TRADEOFFS = (
    (1.0, 0.0),
    (0.8, 0.2),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.2, 0.8),
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class QueryMatrix:
    """One row per query with aligned outcomes for every candidate model."""

    meta: pd.DataFrame
    performance: np.ndarray
    cost_usd: np.ndarray
    latency_s: np.ndarray
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        n_queries = len(self.meta)
        n_models = len(self.models)
        expected = (n_queries, n_models)
        for name, value in (
            ("performance", self.performance),
            ("cost_usd", self.cost_usd),
            ("latency_s", self.latency_s),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains missing or non-finite values")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load local Parquet files and return data plus immutable fingerprints."""

    paths = {
        "train": repo_root / "data/raw/llmrouter_generic/train-000.parquet",
        "test": repo_root / "data/raw/llmrouter_generic/test-000.parquet",
        "candidates": repo_root / "data/raw/llm_candidates/train-000.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local xRouteBench files:\n- " + "\n- ".join(missing)
        )

    fingerprints = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    return (
        pd.read_parquet(paths["train"]),
        pd.read_parquet(paths["test"]),
        pd.read_parquet(paths["candidates"]),
        fingerprints,
    )


def prepare_long_table(
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Validate one official split and attach recorded per-execution USD cost."""

    missing = REQUIRED_ROUTING_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")

    price_columns = ["model_name", "input_price_per_1m", "output_price_per_1m"]
    if candidates["model_name"].duplicated().any():
        raise ValueError("Candidate price table contains duplicate model names")

    frame = raw.merge(
        candidates[price_columns],
        on="model_name",
        how="left",
        validate="many_to_one",
    ).copy()
    if frame[price_columns[1:]].isna().any().any():
        missing_prices = sorted(
            frame.loc[frame[price_columns[1]].isna(), "model_name"].unique()
        )
        raise ValueError(f"Missing candidate prices for: {missing_prices}")

    frame["cost_usd"] = (
        frame["input_tokens"] * frame["input_price_per_1m"]
        + frame["output_tokens"] * frame["output_price_per_1m"]
    ) / 1_000_000.0
    frame["query_hash"] = frame["query"].map(
        lambda text: hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    )
    frame["query_uid"] = split_name + ":" + frame["embedding_id"].astype(str)

    duplicate_pairs = frame.duplicated(["embedding_id", "model_name"]).sum()
    if duplicate_pairs:
        raise ValueError(
            f"{split_name} contains {duplicate_pairs} duplicate query-model rows"
        )
    return frame


def audit_long_table(frame: pd.DataFrame, split_name: str) -> pd.Series:
    per_query = frame.groupby("embedding_id", sort=False).agg(
        rows=("model_name", "size"),
        models=("model_name", "nunique"),
        task_names=("task_name", "nunique"),
        query_texts=("query", "nunique"),
    )
    return pd.Series(
        {
            "split": split_name,
            "execution_rows": len(frame),
            "query_instances": frame["embedding_id"].nunique(),
            "unique_prompt_texts": frame["query_hash"].nunique(),
            "candidate_models": frame["model_name"].nunique(),
            "min_rows_per_query": int(per_query["rows"].min()),
            "max_rows_per_query": int(per_query["rows"].max()),
            "min_models_per_query": int(per_query["models"].min()),
            "max_models_per_query": int(per_query["models"].max()),
            "duplicate_query_model_rows": int(
                frame.duplicated(["embedding_id", "model_name"]).sum()
            ),
            "inconsistent_query_metadata": int(
                ((per_query["task_names"] != 1) | (per_query["query_texts"] != 1)).sum()
            ),
        }
    )


def build_query_matrix(frame: pd.DataFrame, models: Iterable[str]) -> QueryMatrix:
    """Pivot a validated long table to one row per split-local embedding_id."""

    models = tuple(models)
    expected_models = set(models)
    observed_models = set(frame["model_name"].unique())
    if observed_models != expected_models:
        raise ValueError(
            f"Candidate mismatch. Missing={sorted(expected_models-observed_models)}, "
            f"extra={sorted(observed_models-expected_models)}"
        )

    counts = frame.groupby("embedding_id")["model_name"].nunique()
    incomplete = counts[counts != len(models)]
    if len(incomplete):
        raise ValueError(f"Found {len(incomplete)} incomplete candidate matrices")

    meta = (
        frame.sort_values("embedding_id")
        .drop_duplicates("embedding_id")
        [["embedding_id", "query_uid", "query_hash", "task_name", "metric", "query"]]
        .reset_index(drop=True)
    )
    query_order = meta["embedding_id"].tolist()

    def pivot(column: str) -> np.ndarray:
        wide = frame.pivot(index="embedding_id", columns="model_name", values=column)
        wide = wide.reindex(index=query_order, columns=models)
        # Pandas may expose a read-only Arrow-backed view. The router converts
        # these arrays to torch tensors, so own a writable NumPy copy here.
        return wide.to_numpy(dtype=np.float32, copy=True)

    return QueryMatrix(
        meta=meta,
        performance=pivot("performance"),
        cost_usd=pivot("cost_usd"),
        latency_s=pivot("response_time"),
        models=models,
    )


def deterministic_grouped_validation_split(
    meta: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify approximately by task while grouping identical prompt hashes."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    selected_hashes: set[str] = set()
    unique_groups = meta[["task_name", "query_hash"]].drop_duplicates()
    for task_name, task_groups in unique_groups.groupby("task_name", sort=True):
        hashes = task_groups["query_hash"].tolist()
        hashes.sort(
            key=lambda value: hashlib.sha256(
                f"{seed}:{task_name}:{value}".encode("utf-8")
            ).hexdigest()
        )
        n_validation = min(
            len(hashes) - 1,
            max(1, int(round(validation_fraction * len(hashes)))),
        )
        selected_hashes.update(hashes[:n_validation])

    validation_mask = meta["query_hash"].isin(selected_hashes).to_numpy()
    train_idx = np.flatnonzero(~validation_mask)
    validation_idx = np.flatnonzero(validation_mask)

    train_hashes = set(meta.iloc[train_idx]["query_hash"])
    validation_hashes = set(meta.iloc[validation_idx]["query_hash"])
    if train_hashes & validation_hashes:
        raise AssertionError("Exact prompt leakage between train and validation")
    return train_idx, validation_idx


def assert_no_prompt_overlap(left: pd.DataFrame, right: pd.DataFrame) -> None:
    overlap = set(left["query_hash"]) & set(right["query_hash"])
    if overlap:
        raise AssertionError(f"Found {len(overlap)} exact prompt hashes across splits")


def _stable_bucket(feature: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    index = value % dimension
    sign = 1.0 if (value >> 63) == 0 else -1.0
    return index, sign


def hashed_text_features(texts: Iterable[str], dimension: int = 4096) -> np.ndarray:
    """Deterministic unigram/bigram hashing plus inference-safe surface features."""

    texts = list(texts)
    features = np.zeros((len(texts), dimension + 6), dtype=np.float32)
    for row, raw_text in enumerate(texts):
        text = str(raw_text)
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        terms = [f"u:{token}" for token in tokens]
        terms.extend(f"b:{a}|{b}" for a, b in zip(tokens, tokens[1:]))
        for term in terms:
            index, sign = _stable_bucket(term, dimension)
            features[row, index] += sign

        char_count = len(text)
        token_count = len(tokens)
        digit_count = sum(character.isdigit() for character in text)
        features[row, dimension:] = (
            math.log1p(char_count) / 10.0,
            math.log1p(token_count) / 8.0,
            min(text.count("\n") / 50.0, 1.0),
            float("```" in text or "def " in text or "class " in text),
            float(any(marker in text for marker in ("\\frac", "\\sum", "equation", "solve"))),
            digit_count / max(char_count, 1),
        )

    norms = np.linalg.norm(features[:, :dimension], axis=1, keepdims=True)
    features[:, :dimension] /= np.maximum(norms, 1.0)
    return features


class OutcomeRouter(nn.Module):
    """Fast query encoder predicting quality, normalized cost, and latency."""

    def __init__(self, input_dim: int, n_models: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.quality_head = nn.Linear(hidden_dim, n_models)
        self.cost_head = nn.Linear(hidden_dim, n_models)
        self.latency_head = nn.Linear(hidden_dim, n_models)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(inputs)
        quality = torch.sigmoid(self.quality_head(hidden))
        cost = torch.nn.functional.softplus(self.cost_head(hidden))
        latency = torch.nn.functional.softplus(self.latency_head(hidden))
        return quality, cost, latency


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_router(
    features: np.ndarray,
    matrix: QueryMatrix,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    cost_scale: float,
    latency_scale: float,
    *,
    seed: int = 42,
    hidden_dim: int = 256,
    batch_size: int = 128,
    max_epochs: int = 60,
    patience: int = 8,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> tuple[OutcomeRouter, pd.DataFrame]:
    """Fit the inference-safe outcome predictor with validation early stopping."""

    set_reproducible_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = OutcomeRouter(features.shape[1], len(matrix.models), hidden_dim).to(device)

    x = torch.from_numpy(features)
    quality = torch.from_numpy(matrix.performance)
    cost = torch.from_numpy(matrix.cost_usd / cost_scale)
    latency = torch.from_numpy(matrix.latency_s / latency_scale)
    dataset = TensorDataset(x[train_idx], quality[train_idx], cost[train_idx], latency[train_idx])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-2
    )
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    validation_tensors = (
        x[validation_idx].to(device),
        quality[validation_idx].to(device),
        cost[validation_idx].to(device),
        latency[validation_idx].to(device),
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        training_losses = []
        for batch_x, batch_q, batch_c, batch_l in loader:
            batch_x = batch_x.to(device)
            batch_q = batch_q.to(device)
            batch_c = batch_c.to(device)
            batch_l = batch_l.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_q, pred_c, pred_l = model(batch_x)
            loss = (
                mse(pred_q, batch_q)
                + 0.25 * huber(pred_c, batch_c)
                + 0.10 * huber(pred_l, batch_l)
            )
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            pred_q, pred_c, pred_l = model(validation_tensors[0])
            validation_loss = (
                mse(pred_q, validation_tensors[1])
                + 0.25 * huber(pred_c, validation_tensors[2])
                + 0.10 * huber(pred_l, validation_tensors[3])
            )
        validation_value = float(validation_loss.cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(training_losses)),
                "validation_loss": validation_value,
            }
        )

        if validation_value < best_validation - 1e-6:
            best_validation = validation_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def predict_outcomes(
    model: OutcomeRouter,
    features: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = device or next(model.parameters()).device.type
    batches = DataLoader(torch.from_numpy(features), batch_size=batch_size, shuffle=False)
    outputs = [[], [], []]
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            predictions = model(batch.to(device))
            for destination, value in zip(outputs, predictions):
                destination.append(value.cpu().numpy())
    return tuple(np.concatenate(parts, axis=0) for parts in outputs)  # type: ignore[return-value]


def utility_matrix(
    matrix: QueryMatrix,
    alpha: float,
    beta: float,
    cost_scale: float,
) -> np.ndarray:
    return alpha * matrix.performance - beta * (matrix.cost_usd / cost_scale)


def fixed_model_choice(
    matrix: QueryMatrix,
    selection_idx: np.ndarray,
    alpha: float,
    beta: float,
    cost_scale: float,
) -> int:
    mean_utility = utility_matrix(matrix, alpha, beta, cost_scale)[selection_idx].mean(axis=0)
    return int(np.argmax(mean_utility))


def cheapest_fixed_choice(matrix: QueryMatrix, selection_idx: np.ndarray) -> int:
    return int(np.argmin(matrix.cost_usd[selection_idx].mean(axis=0)))


def fit_task_lookup(
    matrix: QueryMatrix,
    selection_idx: np.ndarray,
    alpha: float,
    beta: float,
    cost_scale: float,
) -> dict[str, int]:
    utilities = utility_matrix(matrix, alpha, beta, cost_scale)
    mapping: dict[str, int] = {}
    tasks = matrix.meta["task_name"].to_numpy()
    for task in sorted(set(tasks[selection_idx])):
        task_idx = selection_idx[tasks[selection_idx] == task]
        mapping[str(task)] = int(np.argmax(utilities[task_idx].mean(axis=0)))
    return mapping


def evaluate_choices(
    name: str,
    matrix: QueryMatrix,
    choices: np.ndarray,
    alpha: float,
    beta: float,
    cost_scale: float,
) -> dict:
    choices = np.asarray(choices, dtype=np.int64)
    rows = np.arange(len(matrix.meta))
    quality = matrix.performance[rows, choices]
    cost = matrix.cost_usd[rows, choices]
    latency = matrix.latency_s[rows, choices]
    utility = alpha * quality - beta * (cost / cost_scale)
    oracle_utility = utility_matrix(matrix, alpha, beta, cost_scale).max(axis=1)

    task_frame = pd.DataFrame(
        {"task_name": matrix.meta["task_name"].to_numpy(), "quality": quality}
    )
    shares = np.bincount(choices, minlength=len(matrix.models)) / len(choices)
    return {
        "method": name,
        "alpha": alpha,
        "beta": beta,
        "mean_quality": float(quality.mean()),
        "macro_task_quality": float(task_frame.groupby("task_name")["quality"].mean().mean()),
        "mean_cost_per_1k_queries_usd": float(cost.mean() * 1000.0),
        "p95_latency_s": float(np.percentile(latency, 95)),
        "mean_utility": float(utility.mean()),
        "oracle_regret": float((oracle_utility - utility).mean()),
        "most_routed_model": matrix.models[int(np.argmax(shares))],
        "most_routed_share": float(shares.max()),
    }


def task_breakdown(
    matrix: QueryMatrix,
    choices: np.ndarray,
    alpha: float,
    beta: float,
    cost_scale: float,
) -> pd.DataFrame:
    rows = np.arange(len(matrix.meta))
    quality = matrix.performance[rows, choices]
    cost = matrix.cost_usd[rows, choices]
    latency = matrix.latency_s[rows, choices]
    frame = pd.DataFrame(
        {
            "task_name": matrix.meta["task_name"].to_numpy(),
            "quality": quality,
            "cost_usd": cost,
            "latency_s": latency,
            "utility": alpha * quality - beta * (cost / cost_scale),
        }
    )
    return (
        frame.groupby("task_name", sort=True)
        .agg(
            queries=("quality", "size"),
            mean_quality=("quality", "mean"),
            mean_cost_per_1k_queries_usd=("cost_usd", lambda values: values.mean() * 1000.0),
            p95_latency_s=("latency_s", lambda values: np.percentile(values, 95)),
            mean_utility=("utility", "mean"),
        )
        .reset_index()
    )

