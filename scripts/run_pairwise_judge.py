"""Pairwise LLM Judge on Q7 baselines (B1..B5).

从已有推理结果 jsonl 抽取 generation pairs, 跑 GPT-4o pairwise judge.
默认: B1 (pi-only) 作 baseline, B5 (pi+30维+self+CAPS+peer) 作 model.
也支持 adjacent pairs (B1→B2→B3→B4→B5) 逐级比较.

Usage:
  # 跑 B1 vs B5 (需要 OPENAI_API_KEY)
  .venv/bin/python scripts/run_pairwise_judge.py --mode b1-vs-b5

  # 跑 adjacent pairs (B1vsB2, B2vsB3, B3vsB4, B4vsB5)
  .venv/bin/python scripts/run_pairwise_judge.py --mode adjacent

  # dry run (no API)
  .venv/bin/python scripts/run_pairwise_judge.py --mode b1-vs-b5 --dry-run

  # 指定 key
  .venv/bin/python scripts/run_pairwise_judge.py --api-key sk-xxx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.eval.judge import (
    judge_single_pair,
    judge_batch_async,
    compute_win_rate,
    render_report,
    JudgePairResult,
    LIKERT_TO_POLARITY,
)

logger = logging.getLogger("run_pairwise_judge")

REPORTS_DIR = Path("reports/datav3")
DATA_DIR = Path("data_v3")

# Source files
STRUCTURED_INFERENCE = REPORTS_DIR / "baselines_q7_structured_inference.jsonl"
B2TO5_INFERENCE = REPORTS_DIR / "baselines_q7_structured_b2to5_inference.jsonl"


# ---------------------------------------------------------------------------
# Load baseline generations
# ---------------------------------------------------------------------------

def load_baseline_generations(
    baseline_name: str,
) -> dict[tuple[str, str], str]:
    """Load {(designer_id, scenario): simulated_nl} for a given baseline."""
    out: dict[tuple[str, str], str] = {}

    sources = []
    if baseline_name in ("B1", "B1d"):
        sources = [STRUCTURED_INFERENCE]
    elif baseline_name in ("B2", "B3", "B4"):
        sources = [STRUCTURED_INFERENCE, B2TO5_INFERENCE]
    elif baseline_name == "B5":
        sources = [B2TO5_INFERENCE]
    else:
        sources = [STRUCTURED_INFERENCE, B2TO5_INFERENCE]

    for src in sources:
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("baseline") != baseline_name:
                    continue
                if not d.get("ok", True):
                    continue
                did = d["designer_id"]
                scenario = d["scenario"]
                nl = d.get("simulated_nl", "")
                if nl:
                    out[(did, scenario)] = nl
    return out


# ---------------------------------------------------------------------------
# Build profiles from scale_responses
# ---------------------------------------------------------------------------

def build_profiles(data_dir: Path) -> dict[str, dict[str, str]]:
    """Build designer profiles from scale_responses.csv."""
    import pandas as pd

    scales_path = data_dir / "scale_responses.csv"
    if not scales_path.exists():
        logger.warning("scale_responses.csv not found at %s", scales_path)
        return {}

    scales = pd.read_csv(scales_path)
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    scales["user_id"] = scales["user_id"].astype(str)

    pivot = scales.pivot_table(
        index="user_id", columns="question_node", values="score", aggfunc="mean",
    )
    pivot = pivot.fillna(pivot.mean(numeric_only=True))

    profiles: dict[str, dict[str, str]] = {}
    for uid in pivot.index:
        row = pivot.loc[uid]
        parts = []
        for col in sorted(row.index):
            score = row[col]
            polarity = LIKERT_TO_POLARITY.get(round(score), "弱+")
            parts.append(f"{col}={polarity}")
        profiles[uid] = {
            "trait_summary": "; ".join(parts),
            "factor_scores": "",
        }

    # Try to load factor scores
    fs_path = data_dir / "user_factor_scores.csv"
    if fs_path.exists():
        fs = pd.read_csv(fs_path)
        fs.columns = [c.strip().lstrip("﻿") for c in fs.columns]
        fs["user_id"] = fs["user_id"].astype(str)
        factor_cols = [c for c in fs.columns if c.startswith("F") or c.startswith("factor")]
        for _, r in fs.iterrows():
            uid = str(r["user_id"])
            if uid in profiles:
                scores = [f"{c}={r[c]:.2f}" for c in factor_cols if c in r.index]
                profiles[uid]["factor_scores"] = "; ".join(scores)

    return profiles


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def run_comparison(
    model_name: str,
    baseline_name: str,
    *,
    profiles: dict[str, dict[str, str]],
    judge_model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "https://yunwu.ai/v1",
    concurrency: int = 4,
    dry_run: bool = False,
    max_pairs: int | None = None,
) -> tuple[list[JudgePairResult], str]:
    """Run pairwise judge for model_name vs baseline_name."""
    logger.info("Loading %s generations ...", model_name)
    model_gens = load_baseline_generations(model_name)
    logger.info("  %d records", len(model_gens))

    logger.info("Loading %s generations ...", baseline_name)
    baseline_gens = load_baseline_generations(baseline_name)
    logger.info("  %d records", len(baseline_gens))

    common = sorted(set(model_gens) & set(baseline_gens))
    logger.info("  %d common (designer, scenario) pairs", len(common))

    pairs = []
    skipped = 0
    for did, scenario in common:
        if did not in profiles:
            skipped += 1
            continue
        mr = model_gens[(did, scenario)]
        br = baseline_gens[(did, scenario)]
        if not mr.strip() or not br.strip():
            skipped += 1
            continue
        pairs.append({
            "profile": profiles[did],
            "user_id": did,
            "task_id": scenario,
            "model_response": mr,
            "baseline_response": br,
        })

    if skipped:
        logger.warning("  skipped %d (no profile or empty)", skipped)

    if max_pairs and len(pairs) > max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)
        logger.info("  sampled down to %d pairs", max_pairs)

    logger.info("Judging %d pairs (%s vs %s) ...", len(pairs), model_name, baseline_name)

    t0 = time.time()
    results = asyncio.run(judge_batch_async(
        pairs,
        judge_model=judge_model,
        api_key=api_key,
        base_url=base_url,
        concurrency=concurrency,
        dry_run=dry_run,
    ))
    elapsed = time.time() - t0
    logger.info("  done in %.1fs", elapsed)

    report = compute_win_rate(results)
    label = f"{model_name} vs {baseline_name}"
    md = render_report(report, label=label)

    return results, md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pairwise LLM Judge on Q7 baselines")
    parser.add_argument("--mode", choices=["b1-vs-b5", "adjacent", "custom"],
                        default="b1-vs-b5",
                        help="b1-vs-b5: B5 model vs B1 baseline; "
                             "adjacent: B2vsB1, B3vsB2, B4vsB3, B5vsB4")
    parser.add_argument("--model-baseline", default=None,
                        help="custom mode: model baseline name (e.g., B5)")
    parser.add_argument("--ref-baseline", default=None,
                        help="custom mode: reference baseline name (e.g., B1)")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://yunwu.ai/v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="cap number of pairs (for cost control)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "judge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error("No API key. Set OPENAI_API_KEY or pass --api-key, or use --dry-run.")
        sys.exit(1)

    logger.info("Building designer profiles from %s ...", args.data_dir)
    profiles = build_profiles(args.data_dir)
    logger.info("  %d profiles", len(profiles))

    # Define comparison pairs
    if args.mode == "b1-vs-b5":
        comparisons = [("B5", "B1")]
    elif args.mode == "adjacent":
        comparisons = [("B2", "B1"), ("B3", "B2"), ("B4", "B3"), ("B5", "B4")]
    elif args.mode == "custom":
        if not args.model_baseline or not args.ref_baseline:
            logger.error("--model-baseline and --ref-baseline required for custom mode")
            sys.exit(1)
        comparisons = [(args.model_baseline, args.ref_baseline)]
    else:
        comparisons = [("B5", "B1")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_md = ["# Pairwise LLM Judge Report", ""]

    for model_name, baseline_name in comparisons:
        results, md = run_comparison(
            model_name, baseline_name,
            profiles=profiles,
            judge_model=args.judge_model,
            api_key=api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
            max_pairs=args.max_pairs,
        )
        all_md.append(md)

        # Save per-comparison detail
        detail_path = args.output_dir / f"judge_{model_name}_vs_{baseline_name}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name,
                "baseline": baseline_name,
                "judge_model": args.judge_model,
                "dry_run": args.dry_run,
                "n_pairs": len(results),
                "report": compute_win_rate(results).to_dict(),
                "judgments": [r.to_dict() for r in results],
            }, f, ensure_ascii=False, indent=2)
        logger.info("  saved %s", detail_path)

    # Save combined markdown
    combined_md_path = args.output_dir / "judge_report.md"
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_md))
    logger.info("Combined report: %s", combined_md_path)

    # Print summary
    print("\n" + "\n".join(all_md))


if __name__ == "__main__":
    main()
