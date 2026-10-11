"""train_v3.py — Qwen2.5-7B + LoRA SFT (sft_data_format_v1.md v1.1).

Curriculum (spec §7.2):
  Stage 1 warmup  (1 ep, L_lm only)            ← 本版本支持
  Stage 2 main    (3 ep, L_lm + λ_cls·L_cls)   ← TODO (需 K-way head)
  Stage 3 refine  (1 ep, +λ_cc·L_cc)            ← TODO (需 consistency head)

跟 train_v2 区别:
  - 砍 fold (用 80/20 split jsonl 直接)
  - 砍 trait-stratified sampler (default random; trait-strat 可选 future)
  - 砍 T-W class_weights (默认 None)
  - completion = target JSON 字符串 (按 spec 3 task_type schema)

CLI:
    python -m sft_pipeline.train_v3 \\
        --config sft_pipeline/config/config_v3.yaml \\
        --stage warmup
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("train_v3")


# ─────────────────────────────────────────────────────────────────────
# Dataset (L_lm only, prompt mask -100; JSON completion)
# ─────────────────────────────────────────────────────────────────────

def make_dataset_class():
    import torch
    from torch.utils.data import Dataset

    IGNORE_INDEX = -100

    class SFTDataset(Dataset):
        """jsonl 加载 (system + prompt + completion) → token CE on completion only.

        Sample fields (build_sft_data_q1q6.py 产):
          user_id, qid, target_q, system, prompt, target (dict), target_str, completion
        """

        def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 4096):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.samples: list[dict] = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    self.samples.append(json.loads(line))

        def __len__(self) -> int:
            return len(self.samples)

        def _build_messages(self, s: dict) -> list[dict]:
            return [
                {"role": "system",    "content": s["system"]},
                {"role": "user",      "content": s["prompt"]},
                {"role": "assistant", "content": s["completion"]},
            ]

        def __getitem__(self, idx: int) -> dict:
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

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels":    torch.tensor(labels,    dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            }

    return SFTDataset, IGNORE_INDEX


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
        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L] = item["input_ids"]
            labels[i, :L]    = item["labels"]
            attn_mask[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    return collate


# ─────────────────────────────────────────────────────────────────────
# Model load + LoRA wrap
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    base = model_cfg.get("remote_base_model")
    if not base or not Path(base).exists():
        base = model_cfg["base_model"]

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
        r=cfg.get("_lora_r_override") or model_cfg["lora"]["r"],
        lora_alpha=cfg.get("_lora_alpha_override") or model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        target_modules=model_cfg["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────

STAGE_EPOCHS = {
    "warmup": 1,
    "main":   3,
    "refine": 1,
}


def train_stage(cfg: dict, stage: str, resume_from: str | None = None,
                 epochs_override: int | None = None) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    if stage in ("main", "refine"):
        raise NotImplementedError(
            f"stage={stage} 需 K-way L_cls / consistency L_cc head, 留 TODO. "
            "Stage 1 warmup (L_lm only) 先跑通."
        )

    train_cfg = dict(cfg["train"])
    paths = cfg["paths"]

    model, tokenizer = load_model_and_tokenizer(cfg)

    SFTDataset, _ = make_dataset_class()
    collate_fn = make_collate_fn(tokenizer)

    train_path = Path(paths["sft_train_jsonl"])
    val_path = Path(paths["sft_val_jsonl"])
    train_ds = SFTDataset(train_path, tokenizer, max_length=train_cfg["max_seq_len"])
    eval_ds = SFTDataset(val_path, tokenizer, max_length=train_cfg["max_seq_len"]) if val_path.exists() else None
    logger.info("Stage=%s | train=%d eval=%s", stage, len(train_ds),
                len(eval_ds) if eval_ds else "none")

    epochs = epochs_override if epochs_override is not None else STAGE_EPOCHS[stage]
    # 加 epoch 后缀, 避免覆盖单 ep baseline
    suffix = f"_ep{epochs}" if epochs != STAGE_EPOCHS[stage] else ""
    out_suffix_extra = cfg.get("_out_suffix_extra", "")
    out_dir = Path(paths["ckpt_dir"]) / f"stage_{stage}{suffix}{out_suffix_extra}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=max(1, train_cfg["batch_size"] // 2),
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        learning_rate=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=epochs,
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["scheduler"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds else "no",
        report_to=[],
        remove_unused_columns=False,
        seed=train_cfg["seed"],
        dataloader_num_workers=2,
    )

    kl_lambda = float(cfg.get("_kl_lambda") or 0.0)
    if kl_lambda > 0:
        logger.info("KL-to-base loss enabled, lambda=%s", kl_lambda)
        import torch.nn.functional as F

        class KLAwareTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.get("labels")
                model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                outputs = model(**model_inputs)
                sft_logits = outputs.logits

                shift_sft = sft_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce = F.cross_entropy(shift_sft.reshape(-1, shift_sft.size(-1)),
                                      shift_labels.reshape(-1), ignore_index=-100)

                # KL(base || SFT) on completion tokens (where label != -100)
                with model.disable_adapter():
                    with torch.no_grad():
                        base_logits = model(**model_inputs).logits
                shift_base = base_logits[..., :-1, :].contiguous()
                mask = (shift_labels != -100)
                sft_logp = F.log_softmax(shift_sft, dim=-1)
                base_logp = F.log_softmax(shift_base, dim=-1)
                base_p = base_logp.exp()
                kl_per_token = (base_p * (base_logp - sft_logp)).sum(dim=-1)
                kl_loss = kl_per_token[mask].mean() if mask.any() else torch.tensor(0.0, device=ce.device)

                total = ce + kl_lambda * kl_loss
                return (total, outputs) if return_outputs else total
        trainer = KLAwareTrainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collate_fn, tokenizer=tokenizer,
        )
    else:
        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collate_fn, tokenizer=tokenizer,
        )

    logger.info("Stage 1 warmup (L_lm only, %d ep) starting...", epochs)
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    logger.info("Saving adapter → %s", out_dir)
    trainer.save_model(str(out_dir))
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_pipeline/config/config_v3.yaml")
    p.add_argument("--stage", choices=["warmup", "main", "refine"], default="warmup")
    p.add_argument("--resume", default=None, help="resume_from_checkpoint 路径")
    p.add_argument("--epochs", type=int, default=None, help="override STAGE_EPOCHS[stage]")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument("--lora-r", type=int, default=None, help="override LoRA r (alpha 2x r)")
    p.add_argument("--kl-lambda", type=float, default=0.0,
                   help="KL(base||SFT) coefficient on completion tokens. 0 = pure LM loss")
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--grad-accum", type=int, default=None, help="override train.grad_accum_steps")
    p.add_argument("--out-suffix", default="", help="append to adapter dir name (避免覆盖)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_suffix:
        cfg["_out_suffix_extra"] = args.out_suffix
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
        logger.info("Override train.lr = %s", args.lr)
    if args.lora_r is not None:
        cfg["_lora_r_override"] = args.lora_r
        cfg["_lora_alpha_override"] = args.lora_r * 2
        logger.info("Override LoRA r=%s alpha=%s", args.lora_r, args.lora_r * 2)
    if args.kl_lambda > 0:
        cfg["_kl_lambda"] = args.kl_lambda
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        logger.info("Override batch_size=%s", args.batch_size)
    if args.grad_accum is not None:
        cfg["train"]["grad_accum_steps"] = args.grad_accum
        logger.info("Override grad_accum_steps=%s", args.grad_accum)

    train_stage(cfg, args.stage, resume_from=args.resume, epochs_override=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
