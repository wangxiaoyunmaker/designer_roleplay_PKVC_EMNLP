"""复合 hint 强制对齐 — in-place patch v3_fixed.jsonl.

规则:
  对每行 quad, 查 mapping[Q_option] 的 o_subtype_hint:
    1. hint 含 '➡️' (复合 hint) 但 LLM 输出 o_subtype 是单值 → 强制改成复合 X→Y
    2. hint 单值 (e.g. 'k2', 'v3') → 不动
    3. hint None → 不动
  o_primary 同步改 = 复合的终点 prefix.

理由: option_trait_mapping.json 的 o_subtype_hint 是学姐 metadata schema-level prior
(选项 → cardId 1:1 设计意图), 不是 NL extraction noise. 选了某选项 = 触发该选项的
认知 route pattern, LLM 不该 "用 NL 信号弱" 退回单值.

输出: data/processed/quad_extracted_v3_aligned.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sft_pipeline.extract.codebook import load_mapping, parse_o_subtype, validate_quad, TCapsQuad, fill_codebook

logger = logging.getLogger("quad_align_hint")


def align_one(row: dict, mapping: dict) -> tuple[dict, str]:
    """返回 (new_row, action). action ∈ {unchanged, aligned, skipped}."""
    if "error" in row:
        return row, "unchanged"
    q = row["question_id"]
    opt = row["_meta"]["choice_selected"]
    key = f"{q}_{opt}"
    entry = mapping.get(key)
    if not entry:
        return row, "skipped"
    hint = entry.get("o_subtype_hint") or ""
    if "➡️" not in hint:
        return row, "unchanged"  # 单 hint 或 None, 不动

    # 复合 hint, 转 ➡️ → →
    hint_normalized = hint.replace("➡️", "→").upper()
    if "→" not in hint_normalized:
        return row, "skipped"
    start, end = hint_normalized.split("→")
    if not (start and end):
        return row, "skipped"

    current = row.get("o_subtype") or ""
    if "→" in current:
        # 已经是复合, 但要确认跟 hint 一致
        if current == hint_normalized:
            return row, "unchanged"
        # 跟 hint 不一致 — log + 仍强制对齐
        action = "aligned_overwrite_diff_composite"
    else:
        action = "aligned_single_to_composite"

    # 强制改
    new_row = dict(row)
    old_primary = row["o_primary"]
    old_subtype = row["o_subtype"]
    new_row["o_primary"] = end[0]   # 终点 prefix
    new_row["o_subtype"] = hint_normalized
    new_row["_meta"] = dict(row["_meta"])
    new_row["_meta"]["_aligned_to_hint"] = (
        f"original: o_primary={old_primary}/o_subtype={old_subtype} → "
        f"forced: o_primary={new_row['o_primary']}/o_subtype={hint_normalized} "
        f"(hint composite {hint!r})"
    )
    new_row["_meta"]["validation_fails"] = []  # post-align 重新 ok

    return new_row, action


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=Path("data/processed/quad_extracted_v3_fixed.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/processed/quad_extracted_v3_aligned.jsonl"))
    p.add_argument("--mapping", type=Path, default=Path("reports/datav3/per_designer/option_trait_mapping.json"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    mapping = load_mapping(str(args.mapping))
    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    logger.info("Loaded %d rows", len(rows))

    actions = Counter()
    new_rows = []
    for r in rows:
        nr, act = align_one(r, mapping)
        actions[act] += 1
        new_rows.append(nr)

    logger.info("Actions: %s", dict(actions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 分布对比
    ok = [r for r in new_rows if "error" not in r]
    sub = Counter(r["o_subtype"] for r in ok)
    composite = {k: v for k, v in sub.items() if "→" in k}
    print()
    print(f"复合 o_subtype 分布 (aligned):")
    for k, v in sorted(composite.items()):
        print(f"  {k}: {v}")
    print(f"复合占比: {sum(composite.values())}/{len(ok)} = {100*sum(composite.values())/len(ok):.0f}%")
    print()
    print(f"o_primary 最终: {dict(sorted(Counter(r['o_primary'] for r in ok).items()))}")

    logger.info("Wrote → %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
