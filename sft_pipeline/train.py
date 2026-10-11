"""SFT 训练 loop.

多任务 loss:
  L_total = L_sft + λ_cc · L_cc + λ_cls · L_cls
  (方向一额外: curriculum schedule + L_cs)

骨架结构, 真实跑通需:
  - 数据 jsonl 已生成 (data_prep.py)
  - GPU 可用 (CUDA + bf16)
  - flash-attn 安装

uvx run:
  uvx --python 3.11 --with torch --with transformers --with peft --with accelerate \
      --with pyyaml --with bitsandbytes python train.py \
      --config config.yaml --direction q1q6 --fold 0
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from sft_pipeline.data.dataset import SFTDataset, ClusterBalancedSampler, collate_fn
from sft_pipeline.model import (
    QwenSFTModel,
    compute_cluster_contrastive_loss,
    compute_cluster_classification_loss,
    load_tokenizer,
)


def compute_3track_loss(
    lm_logits: torch.Tensor,      # [B, T, V]
    labels: torch.Tensor,          # [B, T], IGNORE_INDEX on prompt
    segment_ids: torch.Tensor,     # [B, T], -1=prompt, 0=PKVC, 1=decision, 2=answer
    loss_weights: dict[str, float],
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """β+++ 三轨 loss = α·L_PKVC + β·L_decision + γ·L_answer.

    Returns (total_loss, per_segment_loss_dict for logging).
    """
    # Shift for next-token prediction: logits[..., t] predicts labels[..., t+1]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_segments = segment_ids[..., 1:].contiguous()

    # 3 段对应 weight key + segment id
    seg_map = {
        0: ("pkvc_analysis", loss_weights.get("pkvc_analysis", 0.5)),
        1: ("decision_signature", loss_weights.get("decision_signature", 0.3)),
        2: ("answer_letter", loss_weights.get("answer_letter", 0.2)),
    }

    total_loss = torch.tensor(0.0, device=lm_logits.device, dtype=lm_logits.dtype)
    per_seg: dict[str, float] = {}
    for seg_id, (name, weight) in seg_map.items():
        mask = (shift_segments == seg_id) & (shift_labels != ignore_index)
        if mask.sum() == 0:
            per_seg[name] = 0.0
            continue
        seg_logits = shift_logits[mask]    # [N, V]
        seg_labels = shift_labels[mask]    # [N]
        seg_loss = F.cross_entropy(seg_logits, seg_labels, reduction="mean")
        per_seg[name] = float(seg_loss.item())
        total_loss = total_loss + weight * seg_loss
    return total_loss, per_seg


def find_prompt_response_positions(input_ids: torch.Tensor, labels: torch.Tensor,
                                    ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """从 labels 中找 prompt_end (最后一个 ignored token) 和 response_end (last token).

    Returns: (prompt_end_pos [B], response_end_pos [B])
    """
    B, T = labels.shape
    prompt_end = torch.zeros(B, dtype=torch.long, device=labels.device)
    response_end = torch.zeros(B, dtype=torch.long, device=labels.device)
    for i in range(B):
        valid = labels[i] != ignore_index
        if valid.any():
            first_resp = valid.nonzero(as_tuple=True)[0][0].item()
            prompt_end[i] = max(0, first_resp - 1)
            response_end[i] = valid.nonzero(as_tuple=True)[0][-1].item()
        else:
            prompt_end[i] = T - 1
            response_end[i] = T - 1
    return prompt_end, response_end


def get_curriculum_weights(direction: str, step: int, total_steps: int,
                             cfg: dict, default_loss: dict) -> dict:
    """方向一 curriculum: stage 1 only L_B1, stage 2 +L_B2+L_cc, stage 3 +L_B3."""
    if direction != "q7" or not cfg["curriculum"]["enable_for_direction_q7"]:
        return default_loss
    r1 = cfg["curriculum"]["stage_1_ratio"]
    r2 = r1 + cfg["curriculum"]["stage_2_ratio"]
    progress = step / max(total_steps, 1)
    weights = dict(default_loss)
    if progress < r1:
        # stage 1: only L_B1 + L_cls
        weights["L_B2"] = 0.0
        weights["L_B3"] = 0.0
        weights["L_cc"] = 0.0
    elif progress < r2:
        # stage 2: + L_B2 + L_cc (full)
        weights["L_B3"] = 0.0
    # else: stage 3, 全开
    return weights


def task_id_to_loss_key(task_id: str, direction: str) -> str | None:
    """方向一: B1/B2/B3 → L_B1/L_B2/L_B3; 方向二: 全部到 L_pkvc + L_answer 混合."""
    if direction == "q7":
        return f"L_{task_id}"
    return None  # 方向二 SFT loss 直接由 token-level CE 给出


def _resolve_n_clusters(cfg: dict) -> int:
    """Resolve n_clusters from the cluster CSV (config.yaml says "auto" — a string
    that would crash the model init if passed through). Single source of truth for
    K (brief v3.5.1: K must be re-determined after 200-designer EFA).
    """
    import pandas as pd  # local import to keep top of module light
    cluster_file = Path(cfg["data"]["cluster_file"])
    if not cluster_file.exists():
        raise FileNotFoundError(
            f"cluster_file not found: {cluster_file}. "
            "Run scripts/cluster_pilot or equivalent to produce it first."
        )
    df = pd.read_csv(cluster_file)
    n = int(df["cluster_id"].nunique())
    if n < 2:
        raise ValueError(f"cluster_file has only {n} unique cluster_id — invalid")
    return n


def train(cfg_path: str, direction: str, fold: int) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "SFT 需 GPU"

    # ---- Resolve K (cluster count) from the actual cluster file ----
    # config.yaml says "auto" for n_clusters; do not trust it. Read from the file
    # so K=6 (current 79 EFA pilot) or K=? (post-200 EFA) is auto-picked up.
    n_clusters = _resolve_n_clusters(cfg)
    samples_per_cluster = max(1, cfg["training"]["per_device_batch_size"])
    effective_batch = n_clusters * samples_per_cluster
    # config.yaml `effective_batch_size: 24` is a placeholder; overwrite with the
    # K-resolved value so the ClusterBalancedSampler assertion (batch_size == K × spc)
    # holds for any K.
    cfg["training"]["effective_batch_size"] = effective_batch
    cfg.setdefault("auxiliary", {}).setdefault("cls_head", {})["n_clusters"] = n_clusters

    # ---- Tokenizer & Model ----
    base_name = cfg["base_model"]["name"]
    tokenizer = load_tokenizer(base_name)
    model = QwenSFTModel(
        base_model_name=base_name,
        lora_config=cfg["lora"],
        n_clusters=n_clusters,
    )

    # ---- Data ----
    data_path = Path(cfg["data"]["processed_dir"]) / f"{direction}_fold{fold}_train.jsonl"
    assert data_path.exists(), f"数据 jsonl 不存在: {data_path}, 先跑 data_prep.py"

    train_ds = SFTDataset(data_path, tokenizer, max_length=cfg["training"]["max_seq_length"])
    sampler = ClusterBalancedSampler(
        train_ds,
        batch_size=effective_batch,
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        seed=cfg["training"]["seed"],
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    # ---- Optimizer ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
    )
    n_epochs = cfg["training"]["num_epochs"]
    total_steps = len(sampler) * n_epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg["training"]["warmup_steps"], num_training_steps=total_steps,
    )

    default_loss = cfg["loss"][f"direction_{direction}"]

    # ---- Training loop ----
    step = 0
    # β+++ 三轨 loss weights (config.yaml loss_weights section, 默认 α=0.5/β=0.3/γ=0.2)
    track_weights = cfg.get("loss_weights", {
        "pkvc_analysis": 0.5,
        "decision_signature": 0.3,
        "answer_letter": 0.2,
    })

    for epoch in range(n_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            segment_ids = batch["segment_ids"].to(device)  # β+++
            cluster_ids = batch["cluster_ids"].to(device)

            prompt_end, response_end = find_prompt_response_positions(input_ids, labels)

            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                prompt_end_pos=prompt_end, response_end_pos=response_end,
            )

            weights = get_curriculum_weights(direction, step, total_steps, cfg, default_loss)

            # ---- β+++ 三轨 loss ----
            # 用 model 输出的 lm_logits + segment_ids 算 α·L_PKVC + β·L_dec + γ·L_ans
            # 替代 out["loss"] 整段 CE (后者不区分三段)
            sft_loss, per_seg = compute_3track_loss(
                out["lm_logits"], labels, segment_ids, track_weights,
            )
            total_loss = sft_loss

            if out["cc_anchor"] is not None and weights.get("L_cc", 0) > 0:
                l_cc = compute_cluster_contrastive_loss(out["cc_anchor"], cluster_ids)
                total_loss = total_loss + weights["L_cc"] * l_cc

            if out["cls_logits"] is not None and weights.get("L_cls", 0) > 0:
                l_cls = compute_cluster_classification_loss(out["cls_logits"], cluster_ids)
                total_loss = total_loss + weights["L_cls"] * l_cls

            total_loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

            if step % cfg["training"]["logging_steps"] == 0:
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"sft={sft_loss.item():.4f} "
                      f"pkvc={per_seg.get('pkvc_analysis', 0):.4f} "
                      f"dec={per_seg.get('decision_signature', 0):.4f} "
                      f"ans={per_seg.get('answer_letter', 0):.4f} "
                      f"cc={l_cc.item() if 'l_cc' in locals() else 0:.4f} "
                      f"cls={l_cls.item() if 'l_cls' in locals() else 0:.4f}")
            step += 1

            if step % cfg["training"]["save_steps"] == 0:
                ckpt_dir = Path(cfg["paths"]["ckpt_dir"]) / f"{direction}_fold{fold}_step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.lm.save_pretrained(ckpt_dir)
                if hasattr(model, "cls_head") and model.cls_head is not None:
                    torch.save(model.cls_head.state_dict(), ckpt_dir / "cls_head.pt")
                print(f"Saved ckpt: {ckpt_dir}")

    # Final save — small datasets (e.g. q1q6 fold = 25 step) may never hit save_steps=100;
    # without this the run leaves no ckpt on disk.
    final_dir = Path(cfg["paths"]["ckpt_dir"]) / f"{direction}_fold{fold}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(final_dir)
    if hasattr(model, "cls_head") and model.cls_head is not None:
        torch.save(model.cls_head.state_dict(), final_dir / "cls_head.pt")
    print(f"Saved final ckpt: {final_dir}")
    print(f"Training done, total_steps = {step}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_pipeline/config/config.yaml")
    parser.add_argument("--direction", choices=["q7", "q1q6"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    train(args.config, args.direction, args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
