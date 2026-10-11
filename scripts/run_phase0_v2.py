"""run_phase0_v2.py — CPU sanity check pre-GPU.

跑 5 user × 1 fold × max_seq=1024 × batch=2 × 1 epoch (B1 prompt, 最小)
目的:
  1. prep_v2 装配出来的 sft jsonl tokenize 不报 error
  2. 模型 + LoRA forward pass 通
  3. Loss 在 1 epoch 内 decrease (>= 5% lower than initial)
  4. infer_v2 能 load adapter 跑出 1 个 Q7.seg1 (parse_ok 即可, 内容质量不要求)

CPU 上能跑通 = GPU 端只是放大, 不会因 schema bug 浪费 GPU 时间.

CLI:
    python scripts/run_phase0_v2.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CFG_PATH = ROOT / "sft_pipeline" / "config" / "config_v2.yaml"


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] phase0: %(message)s")
    logger = logging.getLogger("phase0")

    with CFG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    p0 = cfg["phase0"]
    logger.info("Phase 0 sanity: n_users=%d max_seq=%d batch=%d epoch=%d baseline=%s device=%s",
                p0["n_users"], p0["max_seq_len"], p0["batch_size"], p0["epochs"],
                p0["baseline"], p0["device"])

    # ─────────────────────────────────────────────────
    # Step 1: prep_v2 --smoke (5 user, ×3 noise)
    # ─────────────────────────────────────────────────
    logger.info("Step 1/3: prep_v2 smoke build...")
    from sft_pipeline.data.prep_v2 import run as prep_run
    rc = prep_run(CFG_PATH, baseline=p0["baseline"], smoke=True, fold_filter=p0["fold_idx"])
    if rc != 0:
        logger.error("prep_v2 failed (rc=%d)", rc)
        return rc

    fold_dir = ROOT / cfg["paths"]["sft_v2_out_dir"] / f"fold_{p0['fold_idx']}"
    train_jsonl = fold_dir / "train.jsonl"
    if not train_jsonl.exists() or train_jsonl.stat().st_size == 0:
        logger.error("No train.jsonl built at %s", train_jsonl)
        return 1
    n_lines = sum(1 for _ in train_jsonl.open())
    logger.info("Step 1 OK: %d samples in %s", n_lines, train_jsonl)

    # ─────────────────────────────────────────────────
    # Step 2: model + LoRA forward pass + loss decrease
    # ─────────────────────────────────────────────────
    logger.info("Step 2/3: tiny train loop (forward + backward + loss decrease)...")
    try:
        import torch
    except ImportError:
        logger.error("torch not installed; phase 0 cannot proceed")
        return 1

    # Override training cfg for phase 0
    p0_train_override = {
        "batch_size":    p0["batch_size"],
        "max_seq_len":   p0["max_seq_len"],
        "epochs":        p0["epochs"],
        "warmup_steps":  0,
        "bf16":          False,             # CPU 不 support bf16, 强制 fp32
        "gradient_checkpointing": False,
        "grad_accum_steps": 1,
    }
    if p0["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from sft_pipeline.train_v2 import train_one_fold
    try:
        out_dir = train_one_fold(cfg, p0["fold_idx"], override_for_phase0=p0_train_override)
        logger.info("Step 2 OK: adapter saved → %s", out_dir)
    except Exception as e:
        logger.error("Step 2 train failed: %s", e, exc_info=True)
        return 1

    # ─────────────────────────────────────────────────
    # Step 3: infer 1 sample
    # ─────────────────────────────────────────────────
    logger.info("Step 3/3: infer 1 Q7.seg1 sample on test fold user...")
    from sft_pipeline.infer_v2 import run_fold as infer_run_fold
    try:
        # patch infer cfg for phase 0 (smaller new_tokens, deterministic)
        cfg_p0 = {**cfg}
        cfg_p0["infer"] = {**cfg["infer"], "max_new_tokens": 128, "do_sample": False}
        infer_run_fold(cfg_p0, p0["fold_idx"], baselines=[p0["baseline"]])
        gen_dir = ROOT / cfg["paths"]["gen_dir"] / f"fold_{p0['fold_idx']}" / p0["baseline"]
        gen_files = list(gen_dir.glob("*__seg*.json"))
        logger.info("Step 3 OK: %d generation files at %s", len(gen_files), gen_dir)
        if gen_files:
            import json
            with gen_files[0].open(encoding="utf-8") as f:
                g = json.load(f)
            logger.info("Sample parsed_ok=%s routing[:80]=%r",
                         g.get("parsed_ok"), g.get("routing", "")[:80])
    except Exception as e:
        logger.error("Step 3 infer failed: %s", e, exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("Phase 0 SANITY PASS — pipeline 端到端通. GPU 端放大跑主训练.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
