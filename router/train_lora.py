#!/usr/bin/env python
"""LoRA training starting point for the System-1 router (mini-clef style).

Trains Qwen3.5-0.8B (local, models/qwen3.5-0.8b) with LoRA to pick, in one
forward pass, which candidate model should answer a query. Labels come from the
counterfactual oracle (data/processed/routing_<split>_eps*.jsonl).

Only the completion (the oracle target) is supervised; the prompt is masked.
No text generation at routing time — this is a System-1 classifier.

Usage:
    python router/train_lora.py --data data/processed/routing_train_eps0.00.jsonl \
        --eval-data data/processed/routing_test_eps0.00.jsonl \
        --base models/qwen3.5-0.8b --out models/router-lora
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "models" / "qwen3.5-0.8b"


# ---------------------------------------------------------------------------
# Prompt format (mini-clef style: state + options in one prompt, answer picks)
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are a router. Given a user query and a list of candidate models, "
    "select the single best model to answer it. Reply with only the model id."
)


def format_prompt(rec: dict) -> str:
    lines = [rec["query"].strip()]
    lines.append("\nCandidate models:")
    for i, c in enumerate(rec["candidates"], start=1):
        lines.append(f"{i}. {c['model']}")
    lines.append("\nSelected model id:")
    return "\n".join(lines)


def format_answer(rec: dict) -> str:
    return rec["oracle_target"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RoutingDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.recs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.recs.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx: int):
        rec = self.recs[idx]
        prompt = format_prompt(rec)
        answer = format_answer(rec)
        full = prompt + "\n" + answer
        # tokenize prompt separately so we can mask it
        prompt_ids = self.tok(
            prompt, add_special_tokens=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )["input_ids"][0].tolist()
        full_ids = self.tok(
            full, add_special_tokens=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )["input_ids"][0].tolist()
        if self.tok.eos_token_id is not None and full_ids[-1] != self.tok.eos_token_id:
            full_ids.append(self.tok.eos_token_id)
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(0)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--out", default="models/router-lora")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, trust_remote_code=True
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = RoutingDataset(args.data, tok, args.max_len)
    eval_ds = RoutingDataset(args.eval_data, tok, args.max_len) if args.eval_data else None

    eval_kwargs = {}
    if eval_ds is not None:
        eval_kwargs = {"eval_strategy": "steps", "eval_steps": 200}

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=(args.dtype == "bf16"),
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=[],
        **eval_kwargs,
    )

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                print("LOG", {k: round(v, 5) if isinstance(v, float) else v for k, v in logs.items()})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        callbacks=[LogCallback()],
    )
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
