"""Run the System-1 router baseline on a Modal H200 (mirrors the notebook).

Local training continues undisturbed — this is a separate Modal app.

Setup (one-time):
    # 1. create HF-token secret (uses your local HF token)
    modal secret create hf-token HF_TOKEN=$(cat ~/.cache/huggingface/token)
    # 2. sync prepared splits into a Modal Volume
    modal volume create route-training
    modal volume put route-training data/processed /processed

Run:
    modal run training/modal_train.py            # interactive, shows progress
    # or detached:
    modal run training/modal_train.py --detach
"""
from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import modal

APP = modal.App("router-system1-h200")
VOLUME = modal.Volume.from_name("route-training", create_if_missing=True)
DATA = Path("/data")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "transformers>=4.57",
        "peft>=0.14",
        "datasets",
        "huggingface_hub",
        "tqdm",
        "pandas",
        "pyarrow",
    )
)

MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "main"
SEED = 7
STEPS, BATCH = 400, 4
EVAL_BATCH = 2
TRAIN_SIZE, TEST_SIZE = 1200, 200
MAX_STATE_TOKENS, MAX_LENGTH = 128, 768
HEAD_CONFIG = {"width": 128, "num_heads": 4, "joint_layers": 2, "dropout": 0.1}
DTYPE_NAME = "bfloat16"


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


def _clipped_state(tok, text):
    return tok.decode(tok.encode(text, add_special_tokens=False)[:MAX_STATE_TOKENS], skip_special_tokens=True)


def _request(query, pool):
    return {"state": query, "questions": {
        "model": {"type": "choice", "question": "Which candidate model should answer this query?",
                  "criteria": dict(pool)}
    }}


def _token_span(offsets, start, end):
    pos = [i for i, (l, r) in enumerate(offsets) if r > l and r > start and l < end]
    return pos[0], pos[-1] + 1


def _encode_record(tok, record):
    spec = record["questions"]["model"]
    payload = "STATE:\n" + record["state"]
    state_chars = (len("STATE:\n"), len(payload))
    payload += "\n\nSCHEMA:\nQuestion: "
    qs = len(payload); payload += spec["question"]; q_chars = (qs, len(payload))
    opts = []
    for k, v in spec["criteria"].items():
        payload += "\nOption: "; s = len(payload); payload += k + " — " + v; opts.append((s, len(payload)))
    prompt = tok.apply_chat_template([
        {"role": "system", "content": "Read the state and the schema. Compare the allowed options."},
        {"role": "user", "content": payload},
    ], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    prompt += "JOINT SCHEMA DECISIONS:\n"
    origin = prompt.index(payload)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    def span(ch): return _token_span(offs, origin + ch[0], origin + ch[1])
    return EncodedRecord(ids, span(state_chars),
                         [EncodedQuestion("model", span(q_chars), list(spec["criteria"]), [span(c) for c in opts])])


def _collate(encs, pad_id, device):
    size, length = len(encs), max(len(e.input_ids) for e in encs)
    slots = max(1 + len(e.questions[0].option_ids) for e in encs)
    ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention = torch.zeros(size, length, dtype=torch.long)
    state_mask = torch.zeros(size, length, dtype=torch.bool)
    pool = torch.zeros(size, slots, length, dtype=torch.float32)
    slot_mask = torch.zeros(size, slots, dtype=torch.bool)
    for i, enc in enumerate(encs):
        n = len(enc.input_ids); ids[i, :n] = torch.tensor(enc.input_ids); attention[i, :n] = 1
        a, b = enc.state_span; state_mask[i, a:b] = True
        spans = [enc.questions[0].question_span, *enc.questions[0].option_spans]
        for j, (a, b) in enumerate(spans):
            pool[i, j, a:b] = 1.0 / (b - a); slot_mask[i, j] = True
    t = {"input_ids": ids, "attention_mask": attention, "state_mask": state_mask,
         "pool": pool, "slot_mask": slot_mask}
    return {**{k: v.to(device) for k, v in t.items()}, "encs": encs}


@APP.function(
    image=IMAGE,
    gpu="H200",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/data": VOLUME},
)
def train_on_h200():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if DTYPE_NAME == "bfloat16" else torch.float16
    torch.manual_seed(SEED)
    random.seed(SEED)

    splits = {}
    for key in ("train", "valid", "test"):
        with open(DATA / f"processed/routing_{key}.jsonl") as fh:
            splits[key] = [json.loads(line) for line in fh if line.strip()]

    rng = random.Random(SEED)
    train = rng.sample(splits["train"], min(TRAIN_SIZE, len(splits["train"])))
    test = rng.sample(splits["test"], min(TEST_SIZE, len(splits["test"])))
    pool = {m: f"{m} (candidate)" for m in sorted({c["model"] for r in train for c in r["candidates"]})}

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=DTYPE, attn_implementation="eager", trust_remote_code=False
    ).to(DEVICE).eval()

    # --- head ---
    class ChoiceHead(nn.Module):
        def __init__(self, hidden_size, width=128, num_heads=4, joint_layers=2, dropout=0.1):
            super().__init__()
            self.context = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.lexical = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.memory = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.roles = nn.Embedding(2, width)
            self.evidence = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
            self.evidence_norm = nn.LayerNorm(width)
            self.joint = nn.ModuleList([
                nn.TransformerEncoderLayer(width, num_heads, dim_feedforward=2 * width, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(joint_layers)])
            self.score = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, hidden, lexical, batch):
            hidden, lexical = hidden.float(), lexical.float()
            p = batch["pool"]
            schema = self.context(p @ hidden) + self.lexical(p @ lexical)
            roles = torch.ones(schema.shape[1], dtype=torch.long, device=schema.device); roles[0] = 0
            schema = schema + self.roles(roles)[None]
            memory = self.memory(hidden)
            evidence, _ = self.evidence(schema, memory, memory, key_padding_mask=~batch["state_mask"], need_weights=False)
            schema = self.evidence_norm(schema + evidence)
            for layer in self.joint:
                schema = layer(schema, src_key_padding_mask=~batch["slot_mask"])
            question = schema[:, :1].expand(-1, schema.shape[1] - 1, -1)
            scores = self.score(torch.cat([schema[:, 1:], question], dim=-1)).squeeze(-1)
            return [[scores[i, :len(enc.questions[0].option_ids)]] for i, enc in enumerate(batch["encs"])]

    class SystemOneRouter(nn.Module):
        def __init__(self, model, head):
            super().__init__(); self.model = model; self.head = head
        def forward(self, batch):
            base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
            tm = base.model.language_model
            hidden = tm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
            lexical = tm.get_input_embeddings()(batch["input_ids"])
            return self.head(hidden, lexical, batch)

    head = ChoiceHead(model.config.text_config.hidden_size, **HEAD_CONFIG).to(DEVICE)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    LORA = [name for name, layer in model.named_modules()
            if ".language_model." in f".{name}." and isinstance(layer, nn.Linear)]
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA, bias="none"))
    router = SystemOneRouter(model, head)

    def train_request(row):
        others = [k for k in pool if k != row["oracle_target"]]
        keep = list(pool)
        if rng.random() < 0.5:
            keep = [row["oracle_target"], *rng.sample(others, rng.randint(1, len(others)))]
        rng.shuffle(keep)
        return _request(row["query"], {k: pool[k] for k in keep})

    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 2e-4},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    warmup = min(20, max(1, STEPS // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) if s < warmup
                                              else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, STEPS - warmup))))
    trainable = [p for p in router.parameters() if p.requires_grad]
    rows = []
    while len(rows) < STEPS * BATCH:
        order = list(train); rng.shuffle(order); rows.extend(order)
    rows = rows[:STEPS * BATCH]

    router.train(); opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for step in range(STEPS):
        part = rows[step * BATCH:(step + 1) * BATCH]
        encs = [_encode_record(tok, train_request(r)) for r in part]
        logits = router(_collate(encs, tok.pad_token_id, DEVICE))
        loss = torch.stack([
            F.cross_entropy(lg[0].float()[None], torch.tensor(
                [enc.questions[0].option_ids.index(row["oracle_target"])], device=DEVICE))
            for row, enc, lg in zip(part, encs, logits)
        ]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step}/{STEPS} loss={loss.item():.3f}", flush=True)
    train_secs = time.perf_counter() - t0
    print(f"training took {train_secs/60:.2f} min", flush=True)

    # --- held-out evaluation (decision head) ---
    router.eval()
    right = hit = 0
    with torch.no_grad():
        for off in range(0, len(test), EVAL_BATCH):
            part = test[off:off + EVAL_BATCH]
            encs = [_encode_record(tok, _request(r["query"], pool)) for r in part]
            logits = router(_collate(encs, tok.pad_token_id, DEVICE))
            for row, enc, lg in zip(part, encs, logits):
                pred = enc.questions[0].option_ids[lg[0].argmax().item()]
                right += pred == row["oracle_target"]
                hit += pred in row["acceptable_models"]
    n = len(test)
    oracle_acc = right / n
    acc_hit = hit / n
    print(f"held-out: oracle_acc={oracle_acc:.3f} acceptable_hit={acc_hit:.3f}", flush=True)

    # save release back to volume
    release = DATA / "releases" / ("router_" + uuid.uuid4().hex[:10])
    release.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(release / "adapter", safe_serialization=True, save_embedding_layers=False)
    tok.save_pretrained(release / "tokenizer")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, release / "head.pt")
    (release / "manifest.json").write_text(json.dumps({
        "format": "router-adapter-v1", "model_id": MODEL, "steps": STEPS, "batch": BATCH,
        "seed": SEED, "train_size": len(train), "test_size": len(test),
        "training_seconds": train_secs, "loss_at_end": loss.item(),
        "held_out": {"oracle_acc": oracle_acc, "acceptable_hit": acc_hit},
    }, indent=2))
    VOLUME.commit()
    print(f"Saved release: {release}", flush=True)
    return str(release)


@APP.function(
    image=IMAGE,
    gpu="H200",
    timeout=1800,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/data": VOLUME},
)
def evaluate_release(release_dir: str):
    """Load a trained release and run held-out eval (no re-training)."""
    import json

    import torch
    from torch import nn
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    rel = DATA / release_dir
    manifest = json.loads((rel / "manifest.json").read_text())

    class ChoiceHead(nn.Module):
        def __init__(self, hidden_size, width=128, num_heads=4, joint_layers=2, dropout=0.1):
            super().__init__()
            self.context = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.lexical = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.memory = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
            self.roles = nn.Embedding(2, width)
            self.evidence = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
            self.evidence_norm = nn.LayerNorm(width)
            self.joint = nn.ModuleList([
                nn.TransformerEncoderLayer(width, num_heads, dim_feedforward=2 * width, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(joint_layers)])
            self.score = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, hidden, lexical, batch):
            hidden, lexical = hidden.float(), lexical.float()
            p = batch["pool"]
            schema = self.context(p @ hidden) + self.lexical(p @ lexical)
            roles = torch.ones(schema.shape[1], dtype=torch.long, device=schema.device); roles[0] = 0
            schema = schema + self.roles(roles)[None]
            memory = self.memory(hidden)
            evidence, _ = self.evidence(schema, memory, memory, key_padding_mask=~batch["state_mask"], need_weights=False)
            schema = self.evidence_norm(schema + evidence)
            for layer in self.joint:
                schema = layer(schema, src_key_padding_mask=~batch["slot_mask"])
            question = schema[:, :1].expand(-1, schema.shape[1] - 1, -1)
            scores = self.score(torch.cat([schema[:, 1:], question], dim=-1)).squeeze(-1)
            return [[scores[i, :len(enc.questions[0].option_ids)]] for i, enc in enumerate(batch["encs"])]

    class SystemOneRouter(nn.Module):
        def __init__(self, model, head):
            super().__init__(); self.model = model; self.head = head
        def forward(self, batch):
            base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
            tm = base.model.language_model
            hidden = tm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
            lexical = tm.get_input_embeddings()(batch["input_ids"])
            return self.head(hidden, lexical, batch)

    tok = AutoTokenizer.from_pretrained(rel / "tokenizer", use_fast=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        manifest["model_id"], dtype=DTYPE, attn_implementation="eager", trust_remote_code=False
    ).to(DEVICE).eval()
    base.requires_grad_(False)
    adapted = PeftModel.from_pretrained(base, rel / "adapter", is_trainable=False)
    head = ChoiceHead(base.config.text_config.hidden_size, **HEAD_CONFIG)
    head.load_state_dict(torch.load(rel / "head.pt", map_location="cpu", weights_only=True), strict=True)
    router = SystemOneRouter(adapted, head).to(DEVICE).eval()

    # load test split from volume
    with open(DATA / "processed/routing_test.jsonl") as fh:
        test = [json.loads(line) for line in fh if line.strip()]
    pool = {m: f"{m} (candidate)" for m in sorted({c["model"] for r in test for c in r["candidates"]})}

    right = hit = 0
    with torch.no_grad():
        for off in range(0, len(test), EVAL_BATCH):
            part = test[off:off + EVAL_BATCH]
            encs = [_encode_record(tok, _request(r["query"], pool)) for r in part]
            logits = router(_collate(encs, tok.pad_token_id, DEVICE))
            for row, enc, lg in zip(part, encs, logits):
                pred = enc.questions[0].option_ids[lg[0].argmax().item()]
                right += pred == row["oracle_target"]
                hit += pred in row["acceptable_models"]
    n = len(test)
    print(f"held-out (release {release_dir}): oracle_acc={right/n:.3f} acceptable_hit={hit/n:.3f} on {n} queries", flush=True)
    return {"oracle_acc": right / n, "acceptable_hit": hit / n, "n": n}
