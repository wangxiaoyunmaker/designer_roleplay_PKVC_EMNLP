"""build_sft_data_q1q6.py — Build SFT v3 train/val jsonl (sft_data_format_v1.md v1.1).

Pipeline:
  1. Load: users.csv / scale_responses.csv / task_responses.csv
           + q1q6_option_labels.jsonl / q2_training_labels.jsonl
           + q1q6_expansion_outputs.jsonl / quad_extracted_163_fixed.jsonl
  2. Per (designer, target_q) in {Q1, Q2, Q3, Q4, Q5, Q6}:
       build_sft_example → {prompt, target}
  3. 80/20 split by (designer, Q) pair (random, seed=42, designer 全部见过)
  4. Output:
       data_v3/processed/sft_train.jsonl  (~770)
       data_v3/processed/sft_val.jsonl    (~200)

Usage:
    .venv/bin/python -m scripts.build_sft_data_q1q6 \\
        [--smoke]                 # 只跑 5 designer 验证
        [--out-dir data_v3/processed]
        [--scenario-from-config sft_pipeline/config/config_v2.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

# Reuse prep_v2 loaders (Q2 labels, expansion, option_labels, parse_task_responses, polarity)
from sft_pipeline.data.prep_v2 import (
    LIKERT_TO_POLARITY,
    POLARITY_ORDER,
    build_profile_matrix,
    load_q1q6_expansion,
    load_q1q6_option_labels,
    load_q2_training_labels,
    numerical_to_polarity,
    parse_task_responses,
)
from sft_pipeline.data.sft_format import (
    _SCENARIO_TRAINING_FALLBACK,
    build_sft_example,
)

logger = logging.getLogger("build_sft_data_q1q6")


# ────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────

def load_quads(p: Path) -> dict[str, list[dict]]:
    """designer_id → list of quad rows."""
    out: dict[str, list[dict]] = {}
    if not p.exists():
        logger.warning("quad file not found: %s", p)
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(str(r["designer_id"]), []).append(r)
    return out


def load_scenario_text(config_path: Path | None) -> str:
    """从 config_v2.yaml 读 scenario_text.training, fallback 用 sft_format 内嵌全感知效率空间."""
    if config_path is None or not config_path.exists():
        return _SCENARIO_TRAINING_FALLBACK
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return (cfg.get("scenario_text") or {}).get("training") or _SCENARIO_TRAINING_FALLBACK


# ────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────

def run(
    root: Path,
    out_dir: Path,
    config_path: Path | None,
    smoke: bool,
    seed: int,
    val_ratio: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_text = load_scenario_text(config_path)

    logger.info("Loading raw CSV...")
    users = pd.read_csv(root / "data_v3/users.csv")
    users["user_id"] = users["wechat_id"].astype(str)
    users_indexed = users.set_index("user_id")
    scales = pd.read_csv(root / "data_v3/scale_responses.csv")
    tasks = pd.read_csv(root / "data_v3/task_responses.csv")

    logger.info("Loading jsonl resources...")
    option_labels = load_q1q6_option_labels(root / "data_v3/processed/q1q6_option_labels.jsonl")
    q2_labels = load_q2_training_labels(root / "data_v3/processed/q2_training_labels.jsonl")
    expansion = load_q1q6_expansion(root / "data_v3/processed/q1q6_expansion_outputs.jsonl")
    quads = load_quads(root / "data_v3/processed/quad_extracted_163_fixed.jsonl")
    parsed_resp = parse_task_responses(tasks)

    logger.info("Building profile (30 维 Likert → polarity)...")
    profile_matrix = build_profile_matrix(scales)
    profile_matrix.index = profile_matrix.index.astype(str)

    designer_ids = sorted(set(profile_matrix.index) & set(users_indexed.index))
    if smoke:
        designer_ids = designer_ids[:5]
        logger.info("[SMOKE] limit to %d designer", len(designer_ids))
    logger.info("Total designer: %d", len(designer_ids))

    examples: list[dict] = []
    skipped: dict[tuple[str, str], int] = {}

    for did in designer_ids:
        if did not in profile_matrix.index:
            continue
        user_row = users_indexed.loc[did].to_dict() if did in users_indexed.index else {}
        polarity = numerical_to_polarity(profile_matrix.loc[did])
        q2_entry = q2_labels.get(did)
        responses_for_user = parsed_resp.get(did, {})
        quads_for_did = quads.get(did, [])

        for target_q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
            ex = build_sft_example(
                designer_id=did,
                target_q=target_q,
                user_row=user_row,
                polarity=polarity,
                parsed_responses=responses_for_user,
                q2_entry=q2_entry,
                expansion=expansion,
                option_labels=option_labels,
                quads_for_designer=quads_for_did,
                scenario_text=scenario_text,
            )
            if ex is None:
                skipped[(did, target_q)] = skipped.get((did, target_q), 0) + 1
                continue
            examples.append(ex)

    logger.info("Built %d examples (skipped %d (designer,Q) pairs)",
                len(examples), len(skipped))
    by_q: dict[str, int] = {}
    for ex in examples:
        by_q[ex["target_q"]] = by_q.get(ex["target_q"], 0) + 1
    logger.info("by target_q: %s", by_q)

    # 80/20 split by (designer, Q) pair, deterministic
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = int(len(examples) * val_ratio)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    for path, rows in ((train_path, train_examples), (val_path, val_examples)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows → %s", len(rows), path)

    # 分布报告
    print()
    print(f"[OK] train={len(train_examples)}  val={len(val_examples)}")
    print("Train by target_q:")
    tq: dict[str, int] = {}
    for ex in train_examples:
        tq[ex["target_q"]] = tq.get(ex["target_q"], 0) + 1
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        print(f"  {q}: {tq.get(q, 0)}")
    print("Val by target_q:")
    vq: dict[str, int] = {}
    for ex in val_examples:
        vq[ex["target_q"]] = vq.get(ex["target_q"], 0) + 1
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        print(f"  {q}: {vq.get(q, 0)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Repo root (default: cwd)")
    p.add_argument("--out-dir", default="data_v3/processed")
    p.add_argument("--config", default="sft_pipeline/config/config_v2.yaml",
                   help="读 scenario_text.training; 没有就用 fallback")
    p.add_argument("--smoke", action="store_true", help="只跑 5 designer 验证")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return run(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config) if args.config else None,
        smoke=args.smoke,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    sys.exit(main())
