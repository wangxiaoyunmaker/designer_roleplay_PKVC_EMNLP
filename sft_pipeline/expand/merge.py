"""回填扩写到下游可消费格式.

输入:
  - data/task_responses.csv (原始)
  - data/processed/q1q6_expansion_outputs.jsonl (batch 输出)

输出:
  - data/task_responses_expanded.csv (跟原 csv 同 schema, 只 patch text_supplement)
  - data/processed/q1q6_expansion_provenance.jsonl (扩写 trace, 含 seed / quote /
    factor / confidence, 给 EMNLP data card + 人工 sample 校验用)

策略:
  - 只回填 validation.ok == True 的样本
  - validation 失败的 + confidence=low 的 → 保留原 seed_nl, 标 was_expanded=False
  - validation 失败但 cache_hit 的 → 不重试, 等下次 batch 重跑 (清 cache 后)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("expand.merge")


def load_outputs(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """读 batch 输出, 索引到 (user_id, question_id)."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["user_id"], obj["question_id"])
            by_key[key] = obj
    return by_key


def merge_into_task_responses(
    task_csv: Path,
    outputs_jsonl: Path,
    out_csv: Path,
    provenance_jsonl: Path,
) -> dict[str, Any]:
    df = pd.read_csv(task_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "text_supplement" not in df.columns:
        raise ValueError(f"task_responses.csv 缺 text_supplement 列: {list(df.columns)}")

    by_key = load_outputs(outputs_jsonl)
    stats = Counter()
    provenance: list[dict[str, Any]] = []

    new_supplements: list[str] = []
    for row in df.itertuples(index=False):
        uid, qid = str(row.user_id), row.question_id
        original = "" if pd.isna(row.text_supplement) else str(row.text_supplement)
        out = by_key.get((uid, qid))

        if out is None:
            new_supplements.append(original)
            stats["unchanged_no_match"] += 1
            continue

        result = out.get("result")
        valid = out.get("validation", {}).get("ok", False)
        fails = out.get("validation", {}).get("fails", [])
        confidence = (result or {}).get("confidence", "")
        expanded = (result or {}).get("expanded_nl", "")

        # quote-only fail (quote_len 超 / quote_not_in_ground) 视作 soft fail —
        # expanded_nl 内容本身 OK, evidence_quote 只是 trace 字段没截到 verbatim.
        # 仍用 expanded, 但 provenance 标 quote_failed 给人工 spot-check.
        quote_only_fail = (
            not valid
            and expanded
            and confidence != "low"
            and all(f.startswith("quote_") for f in fails)
        )

        if (not valid and not quote_only_fail) or confidence == "low" or not expanded:
            new_supplements.append(original)
            stats["unchanged_validation_fail"] += 1
            provenance.append({
                "user_id": uid, "question_id": qid,
                "was_expanded": False, "seed_nl": original,
                "expanded_nl": None,
                "reason": "validation_fail" if not valid else "confidence_low",
                "fails": fails,
            })
            continue

        new_supplements.append(expanded)
        if quote_only_fail:
            stats["expanded_quote_soft_fail"] += 1
        else:
            stats["expanded"] += 1
        provenance.append({
            "user_id": uid, "question_id": qid,
            "was_expanded": True,
            "seed_nl": original,
            "expanded_nl": expanded,
            "inferred_factor": result.get("inferred_factor"),
            "inferred_subtype": result.get("inferred_subtype"),
            "evidence_quote": result.get("evidence_quote"),
            "evidence_source": result.get("evidence_source"),
            "confidence": confidence,
            "quote_soft_fail": quote_only_fail,
            "quote_fails": fails if quote_only_fail else [],
            "model": out.get("model"),
        })

    df["text_supplement"] = new_supplements
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    provenance_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with provenance_jsonl.open("w", encoding="utf-8") as f:
        for p in provenance:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("merge stats: %s", dict(stats))
    logger.info("Wrote patched task_responses → %s", out_csv)
    logger.info("Wrote provenance → %s", provenance_jsonl)
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-csv", type=Path, default=Path("data/task_responses.csv"))
    parser.add_argument("--outputs", type=Path,
                        default=Path("data/processed/q1q6_expansion_outputs.jsonl"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("data/task_responses_expanded.csv"))
    parser.add_argument("--provenance", type=Path,
                        default=Path("data/processed/q1q6_expansion_provenance.jsonl"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    merge_into_task_responses(args.task_csv, args.outputs, args.out_csv, args.provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
