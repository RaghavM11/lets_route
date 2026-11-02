#!/usr/bin/env python
"""Evaluate a released System-1 router against the ground-truth oracle.

Single source of truth for loading a router release and computing, on a
held-out split:
  - oracle accuracy (exact match with the oracle target),
  - acceptable-model hit rate,
  - realized per-query cost/performance vs the oracle's realized values.

Notebooks and CLI scripts import these instead of duplicating the model code.
"""
from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from router.metrics import load_splits

torch.backends.mha.set_fastpath_enabled(False)

MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "main"
SEED = 7
TRAIN_SIZE, TEST_SIZE = 1200, 200
MAX_STATE_TOKENS, MAX_LENGTH = 128, 768
EVAL_BATCH = 2

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = (
    (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    if DEVICE == "cuda"
    else torch.float32
)

rng = random.Random(SEED)


def make_pool():
    import pandas as pd

    from router.metrics import repo_root

    cand = pd.read_parquet(repo_root() / "data/raw/llm_candidates/train-000.parquet")
    return {r["model_name"]: f"{r['model_name']} ({r['size']})" for _, r in cand.iterrows()}


MODEL_POOL = make_pool()


_SPLITS_CACHE: dict | None = None


def sample_split(name: str) -> list[dict]:
    """Reproduce the exact split used at training time (same seed/order).

    The training notebook draws train, then valid, then test from ONE seeded
    RNG. To reproduce the identical test rows, all three draws must be made
    in that order on the shared RNG.
    """
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        splits = load_splits()
        sizes = {"train": TRAIN_SIZE, "valid": TEST_SIZE, "test": TEST_SIZE}
        _SPLITS_CACHE = {
            k: rng.sample(splits[k], min(sizes[k], len(splits[k])))
            for k in ("train", "valid", "test")
        }
    return _SPLITS_CACHE[name]


def request(query, pool=None):
    pool = MODEL_POOL if pool is None else pool
    return {
        "state": query,
        "questions": {
            "model": {
                "type": "choice",
                "question": "Which candidate model should answer this query?",
                "criteria": dict(pool),
            }
        },
    }


def clipped_state(tokenizer, text, max_state_tokens=MAX_STATE_TOKENS):
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_state_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


@dataclass
class EncodedQuestion:
    key: str
    question_span: tuple
    option_ids: list
    option_spans: list


@dataclass
class EncodedRecord:
    input_ids: list
    state_span: tuple
    questions: list


def token_span(offsets, start, end):
    positions = [
        i
        for i, (left, right) in enumerate(offsets)
        if right > left and right > start and left < end
    ]
    if not positions:
        raise ValueError("A schema/state span has no tokens.")
    return positions[0], positions[-1] + 1


def encode_record(tokenizer, record, max_state_tokens=MAX_STATE_TOKENS, max_length=MAX_LENGTH):
    key, spec = next(iter(record["questions"].items()))
    criteria = spec["criteria"]
    state = clipped_state(tokenizer, record["state"], max_state_tokens)
    payload = "STATE:\n" + state
    state_chars = (len("STATE:\n"), len(payload))
    payload += "\n\nSCHEMA:\nQuestion: "
    question_start = len(payload)
    payload += spec["question"]
    question_chars = (question_start, len(payload))
    option_chars = []
    for option_id, description in criteria.items():
        payload += "\nOption: "
        start = len(payload)
        payload += option_id + " — " + description
        option_chars.append((start, len(payload)))
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Read the state and the schema. Compare the allowed options."},
            {"role": "user", "content": payload},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    prompt += "JOINT SCHEMA DECISIONS:\n"
    origin = prompt.index(payload)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError(f"Request needs {len(ids)} tokens; shorten state/schema or increase max_length.")

    def span(chars):
        return token_span(offsets, origin + chars[0], origin + chars[1])

    return EncodedRecord(
        ids,
        span(state_chars),
        [EncodedQuestion(key, span(question_chars), list(criteria), [span(c) for c in option_chars])],
    )


def collate(encs, pad_token_id, device):
    size, length = len(encs), max(len(e.input_ids) for e in encs)
    slots = max(1 + len(e.questions[0].option_ids) for e in encs)
    ids = torch.full((size, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros(size, length, dtype=torch.long)
    state_mask = torch.zeros(size, length, dtype=torch.bool)
    pool = torch.zeros(size, slots, length, dtype=torch.float32)
    slot_mask = torch.zeros(size, slots, dtype=torch.bool)
    for i, enc in enumerate(encs):
        n = len(enc.input_ids)
        ids[i, :n] = torch.tensor(enc.input_ids)
        attention[i, :n] = 1
        a, b = enc.state_span
        state_mask[i, a:b] = True
        q = enc.questions[0]
        spans = [q.question_span, *q.option_spans]
        for j, (a, b) in enumerate(spans):
            pool[i, j, a:b] = 1.0 / (b - a)
            slot_mask[i, j] = True
    t = {"input_ids": ids, "attention_mask": attention, "state_mask": state_mask,
         "pool": pool, "slot_mask": slot_mask}
    return {**{name: value.to(device) for name, value in t.items()}, "encs": encs}


class ChoiceHead(nn.Module):
    def __init__(self, hidden_size, width=128, num_heads=4, joint_layers=2, dropout=0.1):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.lexical = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.memory = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.roles = nn.Embedding(2, width)
        self.evidence = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.evidence_norm = nn.LayerNorm(width)
        self.joint = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, num_heads, dim_feedforward=2 * width,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
                for _ in range(joint_layers)
            ]
        )
        self.score = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width),
            nn.GELU(), nn.Linear(width, 1),
        )

    def forward(self, hidden, lexical, batch):
        hidden, lexical = hidden.float(), lexical.float()
        pool = batch["pool"]
        schema = self.context(pool @ hidden) + self.lexical(pool @ lexical)
        roles = torch.ones(schema.shape[1], dtype=torch.long, device=schema.device)
        roles[0] = 0
        schema = schema + self.roles(roles)[None]
        memory = self.memory(hidden)
        evidence, _ = self.evidence(
            schema, memory, memory, key_padding_mask=~batch["state_mask"], need_weights=False
        )
        schema = self.evidence_norm(schema + evidence)
        for layer in self.joint:
            schema = layer(schema, src_key_padding_mask=~batch["slot_mask"])
        question = schema[:, :1].expand(-1, schema.shape[1] - 1, -1)
        scores = self.score(torch.cat([schema[:, 1:], question], dim=-1)).squeeze(-1)
        return [
            [scores[i, :len(enc.questions[0].option_ids)]]
            for i, enc in enumerate(batch["encs"])
        ]


class SystemOneRouter(nn.Module):
    def __init__(self, model, head, encoding_config=None):
        super().__init__()
        self.model = model
        self.head = head
        self.encoding_config = encoding_config or {
            "max_state_tokens": MAX_STATE_TOKENS,
            "max_length": MAX_LENGTH,
        }

    def forward(self, batch):
        base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        text_model = base.model.language_model
        dtype = next(text_model.parameters()).dtype
        amp = (
            torch.autocast("cuda", dtype=dtype)
            if batch["input_ids"].device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with amp:
            hidden = text_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                use_cache=False, return_dict=True,
            ).last_hidden_state
        lexical = text_model.get_input_embeddings()(batch["input_ids"])
        return self.head(hidden, lexical, batch)


def load_release(directory, device=DEVICE, dtype=DTYPE):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", use_fast=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        manifest["model_id"], revision=manifest["model_revision"], dtype=dtype,
        attn_implementation="eager", trust_remote_code=False,
    )
    base.requires_grad_(False)
    adapted = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False)
    restored_head = ChoiceHead(base.config.text_config.hidden_size, **manifest["head_config"])
    restored_head.load_state_dict(
        torch.load(directory / "head.pt", map_location="cpu", weights_only=True), strict=True
    )
    router = SystemOneRouter(adapted, restored_head, manifest["encoding_config"]).to(device).eval()
    return router, tokenizer, manifest


@torch.no_grad()
def evaluate(clef, tok, records, batch=EVAL_BATCH):
    """Return (oracle_acc, acceptable_hit, preds)."""
    clef.eval()
    right = hit = 0
    preds = []
    for offset in range(0, len(records), batch):
        part = records[offset : offset + batch]
        encs = [encode_record(tok, request(row["query"])) for row in part]
        for row, enc, lg in zip(part, encs, clef(collate(encs, tok.pad_token_id, DEVICE))):
            pred = enc.questions[0].option_ids[lg[0].argmax().item()]
            right += pred == row["oracle_target"]
            hit += pred in row["acceptable_models"]
            preds.append(pred)
    n = len(records)
    return right / n, hit / n, preds


def router_vs_gt(release_dir: str | Path, records: list[dict] | None = None) -> dict:
    """Run the released router on records and compare against the oracle."""
    clef, tok, manifest = load_release(release_dir)
    records = records if records is not None else sample_split("test")
    n = len(records)
    oracle_acc, hit, preds = evaluate(clef, tok, records)

    oracle_cost = oracle_perf = router_cost = router_perf = 0.0
    for r, pred in zip(records, preds):
        tgt = r["oracle_target"]
        tgt_perf = next(c["perf"] for c in r["candidates"] if c["model"] == tgt)
        tgt_cost = next(c["cost_usd"] for c in r["candidates"] if c["model"] == tgt)
        oracle_cost += tgt_cost
        oracle_perf += tgt_perf
        for c in r["candidates"]:
            if c["model"] == pred:
                router_cost += c["cost_usd"]
                router_perf += c["perf"]
                break
    oracle_cost /= n
    oracle_perf /= n
    router_cost /= n
    router_perf /= n

    return {
        "release": str(Path(release_dir).resolve()),
        "test_examples": n,
        "oracle_acc": oracle_acc,
        "acceptable_hit": hit,
        "router_realized_perf": router_perf,
        "router_realized_cost": router_cost,
        "oracle_realized_perf": oracle_perf,
        "oracle_realized_cost": oracle_cost,
        "preds": preds,
        "records": records,
        "gen_baseline": manifest["metrics"].get("generative_baseline"),
        "trained_decision_model": manifest["metrics"].get("trained_decision_model"),
    }
