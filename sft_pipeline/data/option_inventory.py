"""Q1-Q6 option content + designer selection frequency report.

Produces a Markdown table per question that you can send to 学姐 as a starting
draft for PKVC factor mapping:

* Per option (字母 / cardId): the option's textual content and how many of the
  79 designers chose it.
* For sort question Q2: aggregate ranking position (mean order) and times each
  cardId appeared in the top-3.

Output:

* ``data/processed/q1q6_option_inventory.md`` — paper-ready markdown tables.
* ``data/processed/q1q6_option_inventory.json`` — same data in JSON for tooling.

CLI
---
::

    python q1q6_option_inventory.py
    python q1q6_option_inventory.py --csv data/task_responses.csv \\
        --output-md data/processed/inventory.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("q1q6_inventory")

MCQ_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6", "Q8")
SORT_QUESTION = "Q2"

_LETTER_RE = re.compile(r'^"?"?([A-E])\.\s*"?(.*?)"?"?$', re.DOTALL)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def parse_mcq_option(answer_value: str) -> tuple[str | None, str]:
    """Extract ``(letter, option_text)`` from raw ``"\"A. xxx\""`` field."""
    if not answer_value:
        return None, ""
    s = answer_value.strip()
    # CSV adds extra layers of escaping: ""A. xxx""
    s = s.strip('"').replace('""', '"').strip().strip('"')
    m = re.match(r"([A-E])\.\s*(.*)", s, re.DOTALL)
    if not m:
        return None, s
    return m.group(1), m.group(2).strip()


def parse_q2_cards(answer_value: str) -> list[dict[str, Any]]:
    if not answer_value:
        return []
    try:
        obj = json.loads(answer_value)
    except json.JSONDecodeError:
        return []
    return obj if isinstance(obj, list) else []


def build_mcq_inventory(rows: list[dict[str, Any]], question_id: str) -> dict[str, Any]:
    """For Q1, Q3-Q6, Q8: enumerate options + counts + sample explanations."""
    # letter → {"text": <canonical option text>, "count": N, "explanations": [...]}
    by_letter: dict[str, dict[str, Any]] = {}
    total = 0
    for row in rows:
        if row.get("question_id") != question_id:
            continue
        letter, opt_text = parse_mcq_option(row.get("answer_value", ""))
        if letter is None:
            continue
        total += 1
        entry = by_letter.setdefault(letter, {"text": opt_text,
                                                "count": 0,
                                                "explanations": []})
        # Preserve longest variant of the option text (some answers truncated)
        if len(opt_text) > len(entry["text"]):
            entry["text"] = opt_text
        entry["count"] += 1
        explanation = (row.get("text_supplement") or "").strip()
        if explanation:
            entry["explanations"].append(explanation)
    return {"question_id": question_id, "total_responses": total, "options": by_letter}


def build_q2_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For Q2 (sort): count cardId mentions + average order + top-3 frequency."""
    card_total: Counter[str] = Counter()
    card_orders: dict[str, list[int]] = defaultdict(list)
    card_top3: Counter[str] = Counter()
    card_text: dict[str, str] = {}

    n_designers = 0
    for row in rows:
        if row.get("question_id") != SORT_QUESTION:
            continue
        cards = parse_q2_cards(row.get("answer_value", ""))
        if not cards:
            continue
        n_designers += 1
        for c in cards:
            cid = c.get("cardId")
            if not cid:
                continue
            card_total[cid] += 1
            order = c.get("order", -1)
            if isinstance(order, int) and order >= 0:
                card_orders[cid].append(order)
                if order < 3:
                    card_top3[cid] += 1
            text = c.get("text", "")
            if text and len(text) > len(card_text.get(cid, "")):
                card_text[cid] = text

    cards_summary: dict[str, dict[str, Any]] = {}
    for cid in sorted(card_total):
        orders = card_orders[cid]
        cards_summary[cid] = {
            "text": card_text.get(cid, ""),
            "mentions": card_total[cid],
            "mention_pct": card_total[cid] / max(n_designers, 1),
            "mean_order": sum(orders) / len(orders) if orders else None,
            "top3_count": card_top3[cid],
        }
    return {"question_id": SORT_QUESTION,
            "n_designers_responding": n_designers,
            "cards": cards_summary}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_mcq_section(inv: dict[str, Any], n_total: int) -> str:
    lines = [f"## {inv['question_id']} (单选, N={inv['total_responses']}/{n_total})", ""]
    lines.append("| 选项 | 选择人数 | % | 选项文本 | 待学姐填: PKVC 因子 |")
    lines.append("|:---:|---:|---:|:---|:---|")
    options = inv["options"]
    for letter in sorted(options):
        opt = options[letter]
        pct = opt["count"] / max(inv["total_responses"], 1) * 100
        text = opt["text"][:180].replace("\n", " ").replace("|", "/")
        if len(opt["text"]) > 180:
            text += "…"
        lines.append(f"| **{letter}** | {opt['count']} | {pct:.0f}% | {text} |  |")
    lines.append("")
    lines.append("**学姐请填**: 每个选项的主导 PKVC 因子(P/K/V/C, 也可写路径如 V→K)及对应的 SADR 节点 ID。")
    # Show sample text_supplement explanations (truncated)
    samples_total = sum(len(o["explanations"]) for o in options.values())
    if samples_total:
        lines.append("")
        lines.append("<details><summary>展开: 用户 `text_supplement` 解释示例 (各选项前 3 条)</summary>")
        lines.append("")
        for letter in sorted(options):
            opt = options[letter]
            if not opt["explanations"]:
                continue
            lines.append(f"- **{letter}**: " + " / ".join(f"`{e[:30]}`" for e in opt["explanations"][:3]))
        lines.append("")
        lines.append("</details>")
    return "\n".join(lines)


def render_q2_section(inv: dict[str, Any]) -> str:
    n = inv["n_designers_responding"]
    lines = [f"## {inv['question_id']} (排序, N={n})", ""]
    lines.append("| cardId | 出现次数 | % | 平均排名 | 进入 top-3 次数 | 卡片文本 | 待学姐填: PKVC 因子 |")
    lines.append("|:---:|---:|---:|---:|---:|:---|:---|")
    for cid in sorted(inv["cards"]):
        d = inv["cards"][cid]
        text = d["text"][:120].replace("\n", " ").replace("|", "/")
        if len(d["text"]) > 120:
            text += "…"
        mean_order = f"{d['mean_order']:.2f}" if d["mean_order"] is not None else "—"
        lines.append(
            f"| **{cid}** | {d['mentions']} | {d['mention_pct']*100:.0f}% | "
            f"{mean_order} | {d['top3_count']} | {text} |  |"
        )
    lines.append("")
    lines.append(f"> Q2 卡片体系: 4 因子 × 3 子节点 = **12 cardId** 总数. ")
    lines.append("> 12 cardId 的完整定义 (含极性默认 / 触发关键词) 见 `extractor_prompt_v2.md` 占位.")
    return "\n".join(lines)


def render_report(rows: list[dict[str, Any]]) -> str:
    n_total = len({r.get("user_id") for r in rows if r.get("user_id")})
    lines = ["# Q1-Q6 选项内容 + 选择频次清单",
             "",
             f"自动从 `data/task_responses.csv` 抽取, N = {n_total} designers。",
             "",
             "**目的**: 给学姐当起草输入, 填每个选项对应的 PKVC 因子,",
             "省去她从零写 metadata 的工作。",
             "",
             "---"]
    for qid in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q8"):
        if qid == "Q2":
            lines.append(render_q2_section(build_q2_inventory(rows)))
        else:
            inv = build_mcq_inventory(rows, qid)
            if inv["total_responses"] == 0:
                continue
            lines.append(render_mcq_section(inv, n_total))
        lines.append("\n---\n")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("0512技术方案/data/task_responses.csv"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("0512技术方案/data/processed/q1q6_option_inventory.md"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("0512技术方案/data/processed/q1q6_option_inventory.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from sft_pipeline.data.prep import EXCLUDED_USERS
    rows = read_csv_rows(args.csv)
    n_raw = len(rows)
    rows = [r for r in rows if r.get("user_id") not in EXCLUDED_USERS]
    n_excluded = n_raw - len(rows)
    if n_excluded:
        logger.warning("Dropped %d rows from excluded users: %s",
                       n_excluded, sorted(EXCLUDED_USERS))
    logger.info("Loaded %d rows from %s (post-exclusion)", len(rows), args.csv)

    md = render_report(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    logger.info("Wrote %s (%d chars)", args.output_md, len(md))

    bundle = {
        "mcq": {qid: build_mcq_inventory(rows, qid) for qid in MCQ_QUESTIONS},
        "sort": build_q2_inventory(rows),
    }
    args.output_json.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
