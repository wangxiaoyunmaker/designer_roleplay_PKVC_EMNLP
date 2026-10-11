"""train_v2.py — Qwen2.5-7B + LoRA SFT, L_lm only (framework v1.1 / Path D).

Loss design (user 2026-05-25 拍): L_lm only.
  - Completion = "[认知路由标签]\n{routing}\n\n[NL 回答]\n{nl}"
  - Token CE on full completion, prompt 段 mask -100
  - 不拆 L_pkvc / L_answer 段 (统一 schema 已让 routing+NL 在同一 sequence)
  - 不加 L_cc (v3.7 砍 cluster 后 cluster-contrastive 失去定义)

Sampler: trait-stratified (按 F1 quartile, v3.7 §4.1 替代 cluster-balanced).
Fold:    5-fold designer-level CV.

CLI:
    # 主训练 (单 fold)
    python -m sft_pipeline.train_v2 \\
        --config sft_pipeline/config/config_v2.yaml \\
        --fold 0

    # 5-fold loop
    python -m sft_pipeline.train_v2 \\
        --config sft_pipeline/config/config_v2.yaml \\
        --all-folds
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("train_v2")

# Defer heavy imports (torch / transformers / peft) into main() so --help works
# without GPU stack installed.


# ─────────────────────────────────────────────────────────────────────
# Dataset (L_lm only, prompt mask -100)
# ─────────────────────────────────────────────────────────────────────

def make_dataset_class(class_weights: dict | None = None):
    """class_weights: T-W ablation. dict like {'P': 2.0, 'K': 1.0, 'V': 1.0, 'C': 0.8}.
    None = unweighted (默认, 跟 cloud 一致)."""
    import torch
    from torch.utils.data import Dataset

    IGNORE_INDEX = -100

    class SFTDataset(Dataset):
        """jsonl 加载 (system + prompt + completion) → token CE.

        T-W: 当 class_weights 非空, 对 [认知路由标签] 段的 P/K/V/C 单字母 token
             apply per-class 权重, 其他 token 权重 = 1.0. WeightedTrainer 用这个
             weights tensor 算 weighted CE.
        """

        def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 4096):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.samples: list[dict] = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    self.samples.append(json.loads(line))
            # 预先 cache PKVC 单字母 token id (T-W 用)
            self.class_weights = class_weights or {}
            self.pkvc_token_weights: dict[int, float] = {}
            if self.class_weights:
                for letter, w in self.class_weights.items():
                    if letter not in ("P", "K", "V", "C"):
                        continue
                    ids = tokenizer(letter, add_special_tokens=False).input_ids
                    if len(ids) == 1:
                        self.pkvc_token_weights[ids[0]] = float(w)
                    else:
                        # Qwen tokenizer 单字母 ASCII 应该 1 token; 出 multi-token 警告
                        import logging
                        logging.warning("PKVC letter %s 在 tokenizer 上变 %d tokens %s, T-W 不生效",
                                        letter, len(ids), ids)

        def __len__(self) -> int:
            return len(self.samples)

        def _build_messages(self, s: dict) -> list[dict]:
            return [
                {"role": "system",    "content": s["system"]},
                {"role": "user",      "content": s["prompt"]},
                {"role": "assistant", "content": s["completion"]},
            ]

        def __getitem__(self, idx: int) -> dict:
            import torch
            s = self.samples[idx]
            msgs = self._build_messages(s)

            full_text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                msgs[:2], tokenize=False, add_generation_prompt=True,
            )

            full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                       truncation=True, max_length=self.max_length).input_ids
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                         truncation=True, max_length=self.max_length).input_ids

            input_ids = full_ids
            labels = list(full_ids)
            n_mask = min(len(prompt_ids), len(labels))
            for i in range(n_mask):
                labels[i] = IGNORE_INDEX

            # T-W: per-token loss weight tensor
            # default 1.0; PKVC token (where label != -100) 用 class weight
            loss_weights = [1.0] * len(input_ids)
            if self.pkvc_token_weights:
                for i in range(n_mask, len(input_ids)):
                    tok = input_ids[i]
                    if tok in self.pkvc_token_weights:
                        loss_weights[i] = self.pkvc_token_weights[tok]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels":    torch.tensor(labels,    dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
            }

    return SFTDataset, IGNORE_INDEX


# ─────────────────────────────────────────────────────────────────────
# Trait-stratified sampler (v3.7 §4.1 替代 cluster-balanced)
# ─────────────────────────────────────────────────────────────────────

def make_sampler_class():
    import torch
    from torch.utils.data import Sampler

    class TraitStratifiedSampler(Sampler):
        """每 batch 尽量来自 4 个 factor quartile 各 batch_size/4 samples.

        实现: 按 user_id → F1 quartile 分桶, 每 batch 从 4 桶轮询取.
        """

        def __init__(self, samples: list[dict], factor_df_dict: dict[str, int],
                     batch_size: int, seed: int = 42):
            self.batch_size = batch_size
            self.seed = seed
            # bucket by F1 quartile of user
            self.buckets: dict[int, list[int]] = defaultdict(list)
            for i, s in enumerate(samples):
                q = factor_df_dict.get(s["user_id"], 0)
                self.buckets[q].append(i)
            self.n = len(samples)

        def __iter__(self):
            rng = np.random.default_rng(self.seed)
            # shuffle 每个桶
            buckets = {q: list(idxs) for q, idxs in self.buckets.items()}
            for q in buckets:
                rng.shuffle(buckets[q])
            # round-robin pull
            pointers = {q: 0 for q in buckets}
            quartile_order = sorted(buckets.keys())
            remaining = sum(len(v) for v in buckets.values())
            while remaining > 0:
                for q in quartile_order:
                    if pointers[q] < len(buckets[q]):
                        yield buckets[q][pointers[q]]
                        pointers[q] += 1
                        remaining -= 1
                        if remaining == 0:
                            return

        def __len__(self):
            return self.n

    return TraitStratifiedSampler


# ─────────────────────────────────────────────────────────────────────
# Collator (pad to longest in batch)
# ─────────────────────────────────────────────────────────────────────

def make_collate_fn(tokenizer):
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch: list[dict]) -> dict:
        max_len = max(item["input_ids"].size(0) for item in batch)
        bs = len(batch)
        input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
        labels    = torch.full((bs, max_len), -100,   dtype=torch.long)
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long)
        has_lw = "loss_weights" in batch[0]
        loss_weights = torch.zeros((bs, max_len), dtype=torch.float) if has_lw else None
        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L] = item["input_ids"]
            labels[i, :L]    = item["labels"]
            attn_mask[i, :L] = 1
            if has_lw:
                loss_weights[i, :L] = item["loss_weights"]
        out = {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
        if has_lw:
            out["loss_weights"] = loss_weights
        return out

    return collate


# ─────────────────────────────────────────────────────────────────────
# Model load + LoRA wrap
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    base = model_cfg.get("remote_base_model") if Path(model_cfg.get("remote_base_model", "/nonexistent")).exists() \
        else model_cfg["base_model"]

    logger.info("Loading tokenizer + model from %s ...", base)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if train_cfg["bf16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, trust_remote_code=True,
    )
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        target_modules=model_cfg["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if device:
        model = model.to(device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Train loop (HF Trainer wrapper)
# ─────────────────────────────────────────────────────────────────────

def factor_quartile_dict(factor_path: Path) -> dict[str, int]:
    """user_id → F1 quartile (0-3)."""
    import pandas as pd
    df = pd.read_csv(factor_path)
    df["user_id"] = df["user_id"].astype(str)
    q = np.quantile(df["F1"].values, [0.25, 0.5, 0.75])
    df["quartile"] = np.digitize(df["F1"].values, q)
    return dict(zip(df["user_id"], df["quartile"].astype(int)))


def train_one_fold(cfg: dict, fold_idx: int, override_for_phase0: dict | None = None) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import Trainer, TrainingArguments

    train_cfg = dict(cfg["train"])  # copy
    if override_for_phase0:
        train_cfg.update(override_for_phase0)
    paths = cfg["paths"]

    model, tokenizer = load_model_and_tokenizer(cfg)

    # T-W: class_weights from config (None / dict)
    class_weights = train_cfg.get("class_weights")
    if class_weights:
        logger.info("T-W ablation: class_weights = %s", class_weights)

    SFTDataset, IGNORE_INDEX = make_dataset_class(class_weights=class_weights)
    TraitStratifiedSampler = make_sampler_class()
    collate_fn = make_collate_fn(tokenizer)

    fold_dir = Path(paths["sft_v2_out_dir"]) / f"fold_{fold_idx}"
    train_ds = SFTDataset(fold_dir / "train.jsonl", tokenizer,
                          max_length=train_cfg["max_seq_len"])
    eval_ds = None
    test_path = fold_dir / "test.jsonl"
    if test_path.exists() and test_path.stat().st_size > 0:
        eval_ds = SFTDataset(test_path, tokenizer, max_length=train_cfg["max_seq_len"])

    logger.info("fold %d: train=%d  eval=%s", fold_idx, len(train_ds),
                 len(eval_ds) if eval_ds else "none")

    # build sampler (trait-stratified)
    quartile_dict = factor_quartile_dict(Path(paths["factor_scores_out"]))
    sampler = TraitStratifiedSampler(train_ds.samples, quartile_dict,
                                      batch_size=train_cfg["batch_size"],
                                      seed=train_cfg["seed"])

    out_dir = Path(paths["ckpt_dir"]) / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=max(1, train_cfg["batch_size"] // 2),
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        learning_rate=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["scheduler"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch" if eval_ds else "no",
        report_to=[],
        remove_unused_columns=False,
        seed=train_cfg["seed"],
        dataloader_num_workers=2,
    )

    class _WeightedTrainerWithSampler(Trainer):
        """Custom Trainer:
          1. trait-stratified sampler (override _get_train_sampler)
          2. weighted CE if 'loss_weights' in batch (T-W ablation)
        """
        def _get_train_sampler(self):
            return sampler

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            loss_weights = inputs.pop("loss_weights", None)
            outputs = model(**inputs)
            logits = outputs.logits
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            mask = (shift_labels.view(-1) != -100)
            if loss_weights is not None:
                shift_w = loss_weights[..., 1:].contiguous().view(-1)
                # weighted mean over non-ignored positions
                num = (per_token_loss * shift_w)[mask].sum()
                den = shift_w[mask].sum().clamp_min(1e-8)
                loss = num / den
            else:
                loss = per_token_loss[mask].mean()
            return (loss, outputs) if return_outputs else loss

    trainer = _WeightedTrainerWithSampler(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate_fn, tokenizer=tokenizer,
    )

    logger.info("Starting training (fold %d) ...", fold_idx)
    trainer.train()
    logger.info("Saving adapter → %s", out_dir)
    trainer.save_model(str(out_dir))
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_pipeline/config/config_v2.yaml")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_folds = cfg["data"]["n_folds"]
    if args.all_folds:
        for k in range(n_folds):
            train_one_fold(cfg, k)
    else:
        fold = args.fold if args.fold is not None else 0
        train_one_fold(cfg, fold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
