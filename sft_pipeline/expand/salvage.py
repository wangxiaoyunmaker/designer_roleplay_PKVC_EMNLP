"""Salvage step: 把机械越界的 fail 用后处理 trim 救回来.

只处理两种 fail:
  - len_out_of_range (expanded_nl > 130 字): 截到 130 字, 优先在句号/标点处截断
  - quote_len (evidence_quote > 40 字): 截 quote 到 40 字, 不影响 expanded_nl

不处理:
  - quote_len < 8 字 (太短): 真信号缺失, 需 retry
  - quote_not_in_ground: 真 hallucination, 需 retry 或人工
  - bad_factor / bad_confidence: schema 问题, 需 retry

重 parse + 重 validate, 写回 outputs.jsonl. 0 API 成本.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from sft_pipeline.expand.parser import (
    LEN_MAX, QUOTE_MAX_LEN, ExpansionResult, parse_and_validate, result_to_jsonl_dict,
)
from sft_pipeline.expand.screen import load_candidates

logger = logging.getLogger("expand.salvage")

# 中文句末标点 (按优先级)
SENTENCE_END = "。.!?！？；;"


def trim_to_sentence(text: str, max_len: int) -> str:
    """截到 max_len, 优先在句末标点处截断, 否则在逗号/顿号处, 最差直接截."""
    if len(text) <= max_len:
        return text
    # 在 [max_len-15, max_len] 找最后一个句末标点
    window = text[max(0, max_len - 15):max_len]
    best = -1
    for i, ch in enumerate(window):
        if ch in SENTENCE_END:
            best = max(0, max_len - 15) + i + 1  # +1 包含标点本身
    if best > 0:
        return text[:best]
    # 找逗号/顿号
    for i, ch in enumerate(window):
        if ch in "，,、":
            best = max(0, max_len - 15) + i  # 不包含逗号本身, 改用句号
    if best > 0:
        return text[:best] + "。"
    # 直接截 + 句号
    return text[:max_len - 1] + "。"


def trim_quote(quote: str, max_len: int) -> str:
    """quote 截到 max_len, 优先在标点处截."""
    if len(quote) <= max_len:
        return quote
    window = quote[:max_len]
    best = -1
    for i, ch in enumerate(window):
        if ch in SENTENCE_END + "，,、:：":
            best = i  # 不含标点 (quote 是节选, 标点截掉也 OK)
    if best > 4:  # 不要截得太短
        return quote[:best]
    return quote[:max_len]


def salvage_one(rec: dict[str, Any], candidate_map: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """对 1 条 fail 记录尝试 trim. 返回 (新 rec, 是否被 salvage 修过)."""
    fails = rec["validation"]["fails"]
    fail_types = set(f.split(":")[0] for f in fails)
    result = rec.get("result")
    if not result:
        return rec, False
    if not fail_types.issubset({"len_out_of_range", "quote_len"}):
        return rec, False  # 只处理这两种, 其他不动

    # 不处理 quote 太短的情形 (quote_len < 8): 留给 retry
    if "quote_len" in fail_types and len(result["evidence_quote"]) < 8:
        return rec, False

    new_expanded = result["expanded_nl"]
    new_quote = result["evidence_quote"]
    salvaged = False
    if "len_out_of_range" in fail_types and len(new_expanded) > LEN_MAX:
        new_expanded = trim_to_sentence(new_expanded, LEN_MAX)
        salvaged = True
    if "quote_len" in fail_types and len(new_quote) > QUOTE_MAX_LEN:
        new_quote = trim_quote(new_quote, QUOTE_MAX_LEN)
        salvaged = True
    if not salvaged:
        return rec, False

    # 装回 raw_json 风格, 重跑 validator (要 candidate 做 quote_in_ground 检查)
    candidate = candidate_map.get(rec["candidate_id"])
    if candidate is None:
        return rec, False  # 没法重 validate
    new_obj = {
        "expanded_nl": new_expanded,
        "inferred_factor": result["inferred_factor"],
        "inferred_subtype": result["inferred_subtype"],
        "evidence_quote": new_quote,
        "evidence_source": result["evidence_source"],
        "confidence": result["confidence"],
        "reason_if_low": result["reason_if_low"],
    }
    new_raw = json.dumps(new_obj, ensure_ascii=False)
    new_result, new_report = parse_and_validate(new_raw, candidate, cache_hit=True)
    new_rec = result_to_jsonl_dict(candidate, new_result, new_report, rec["model"])
    new_rec["api_error"] = False
    new_rec["salvaged"] = True
    new_rec["original_fails"] = fails
    return new_rec, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path,
                        default=Path("data/processed/q1q6_expansion_outputs.jsonl"))
    parser.add_argument("--candidates", type=Path,
                        default=Path("data/processed/q1q6_expansion_candidates.jsonl"))
    parser.add_argument("--in-place", action="store_true",
                        help="直接覆写 outputs.jsonl (默认: 同 dir 写 .salvaged.jsonl)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cands = load_candidates(args.candidates)
    cand_map = {c.candidate_id: c for c in cands}

    rows = [json.loads(l) for l in args.outputs.open(encoding="utf-8") if l.strip()]
    new_rows: list[dict] = []
    n_salvaged = 0
    n_now_ok = 0
    for r in rows:
        if r["validation"]["ok"]:
            new_rows.append(r)
            continue
        new_r, did = salvage_one(r, cand_map)
        new_rows.append(new_r)
        if did:
            n_salvaged += 1
            if new_r["validation"]["ok"]:
                n_now_ok += 1
                logger.info(
                    "salvaged ok: %s (orig fails: %s)",
                    r["candidate_id"], r["validation"]["fails"],
                )
            else:
                logger.warning(
                    "salvaged but still fail: %s now: %s",
                    r["candidate_id"], new_r["validation"]["fails"],
                )

    out_path = args.outputs if args.in_place else args.outputs.with_suffix(".salvaged.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ok = sum(1 for r in new_rows if r["validation"]["ok"])
    total = len(new_rows)
    logger.info(
        "Salvage: attempted %d → recovered %d → final ok %d/%d (%.1f%%)",
        n_salvaged, n_now_ok, total_ok, total, 100 * total_ok / max(total, 1),
    )
    logger.info("Wrote → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
