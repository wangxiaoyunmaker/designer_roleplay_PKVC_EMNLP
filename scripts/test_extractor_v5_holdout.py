"""Holdout 测试 extractor_v5 (只标 PKVC + T_CAPS, 不抽 30 SADR).

评测维度:
  1. dominant_factor 跟专家标注的"主导序列"是否一致 (per-seg)
  2. criterion 是否能从 INPUT verbatim 抽到 (反 hallucination)
  3. o_primary 跟 dominant_factor 一致性
  4. confidence 分布

Usage: set -a; source .env; set +a; unset all_proxy ALL_PROXY
       .venv/bin/python scripts/test_extractor_v5_holdout.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.expand.llm import call_chat

PROMPT_PATH = Path("sft_pipeline/config/prompts/extractor_v5.md")
GOLD_DOC = Path("docs/notes/被试回答专家标注.md")
OUT_PATH = Path("data/processed/extractor_v5_holdout.md")


# 专家 ground truth — 每个附录 per-seg dominant_factor (从卡片序列推断)
HOLDOUT_GT = {
    "F": {
        "label": "C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)",
        "seg_expected": {
            1: ["C", "C→P"],          # seg1: C1→C1→P3 (C 主导, 末尾 P)
            2: ["C", "K", "C→K"],     # seg2: C2→K2→K1→K2 (C 起头但 K 深化)
            3: ["V", "V→C", "C"],     # seg3: V1→V3→V1→P3 (V 主导, 收束 P/C)
        },
        "criterion_keyword": ["人类福祉", "幸福", "公平"],
    },
    "G": {
        "label": "P (情境矛盾) → K (双重) → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)",
        "seg_expected": {
            1: ["P", "P→K", "P→V"],   # seg1: P2→K1→K1→V2→C1 (P 先启)
            2: ["V", "V→K", "V→C"],   # seg2: V1→V1→K1→V3→C2 (V 宣言型主导)
            3: ["C", "C→V", "V→C"],   # seg3: K1→V2→C2 (C 综合收束)
        },
        "criterion_keyword": ["设计存在本质", "用户选择权", "更好的体验"],
    },
}


def extract_segs_for(letter: str) -> dict[int, str]:
    """提取附录的 seg1/seg2/seg3 文本.

    bug fix: 原 regex 用 \"...\" 单字符引号匹配, 中文内嵌引号会提前截断 (G seg1 应
    ~400 字截到 19). 改成从 **seg(N) 标记后读到下一个 seg/horizontal-rule.
    """
    text = GOLD_DOC.read_text(encoding="utf-8")
    m = re.search(rf"## \*\*附录 {letter}：(.+?)(?=\n## \*\*附录 |\Z)", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"附录 {letter} 没找到")
    body = m.group(1)
    # 找每个 seg 起点位置 (3 个: seg1/seg2/seg3) 跟终点 (下一个 seg / "---")
    starts = list(re.finditer(r"\*\*seg(\d).{0,50}?\*\*", body))
    if not starts:
        return {}
    out: dict[int, str] = {}
    for i, m in enumerate(starts):
        n = int(m.group(1))
        s = m.end()
        e = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        seg_block = body[s:e]
        # 截到第一个 "---" 之前
        sep = seg_block.find("\n---")
        if sep >= 0:
            seg_block = seg_block[:sep]
        # 去开头的引号 / 空白 / > 符号
        cleaned = seg_block.strip().lstrip("“”\"\"").rstrip("“”\"\"")
        cleaned = cleaned.strip().strip(">").strip()
        out[n] = cleaned
    return out


def load_v5_prompt() -> tuple[str, str]:
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    if len(fences) < 2:
        raise RuntimeError(f"v5 prompt 缺 fenced block (got {len(fences)})")
    return fences[0], fences[1]


def render_user_prompt(template: str, seg_role: str, seg_text: str) -> str:
    placeholders = {
        "{SEGMENT_ROLE}": seg_role,
        "{SCENARIO}": "Q7_老城区超市重构",
        "{C_LOCAL_TAG}": "(holdout 测试, 无 metadata)",
        "{TASK_BACKGROUND}": "(holdout 测试, 无 metadata)",
        "{S_PRIMARY_TAGS}": "(holdout, 无 metadata)",
        "{S_SECONDARY_FACTOR}": "(holdout, 无 metadata)",
        "{INPUT_TEXT}": seg_text,
    }
    out = template
    for k, v in placeholders.items():
        out = out.replace(k, v)
    return out


SEG_ROLE_MAP = {1: "第一反应", 2: "决策链路", 3: "核心 tradeoff"}


def run_one(seg_text: str, seg_role: str, system: str, user_tmpl: str) -> dict:
    user_msg = render_user_prompt(user_tmpl, seg_role, seg_text)
    # temperature=0 + no cache: 之前 0.2 + cache 出现过 null 污染
    raw, _ = call_chat(system, user_msg, model="gpt-4o", temperature=0.0, use_cache=False)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        obj = json.loads(raw[s:e + 1]) if s >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        obj = {"_parse_fail": True, "_raw": raw[:200]}
    return obj


def check_dominant(actual: str | None, expected_list: list[str]) -> str:
    """判 LLM dominant 跟专家任一可接受 dominant 一致.

    路径终点也算命中 (e.g. 专家 "C→P", LLM "P" 算 partial 命中)."""
    if not actual:
        return "MISS (null)"
    actual = actual.strip()
    if actual in expected_list:
        return f"HIT ({actual})"
    # path 终点
    for exp in expected_list:
        if "→" in exp and actual == exp.split("→")[-1]:
            return f"PARTIAL ({actual} = {exp} 路径终点)"
        if "→" in actual and actual.split("→")[-1] == exp:
            return f"PARTIAL ({actual} 路径终点 = {exp})"
    return f"MISS ({actual}; expected one of {expected_list})"


def check_criterion(criterion: str | None, keywords: list[str], seg_text: str) -> str:
    if not criterion:
        return "null"
    in_input = criterion in seg_text or any(c in seg_text for c in criterion if len(c) > 1)
    keyword_hit = any(kw in criterion for kw in keywords)
    status = "GROUNDED" if in_input else "NOT_IN_INPUT"
    if keyword_hit:
        status += "+KEYWORD"
    return f"{status}: {criterion!r}"


def main() -> int:
    if not PROMPT_PATH.exists():
        print(f"v5 prompt 没找到: {PROMPT_PATH}"); return 1
    system, user_tmpl = load_v5_prompt()

    lines = ["# Extractor v5 — Holdout F/G 测试 (只标 PKVC, 不抽 SADR)", "",
             f"- prompt: {PROMPT_PATH}",
             "- model: gpt-4o, temperature=0.2",
             "- 评估: dominant_factor 一致性 + criterion grounded 检查", ""]

    summary = []
    for letter in ("F", "G"):
        gt = HOLDOUT_GT[letter]
        lines += ["---", f"## 附录 {letter}",
                  f"- **专家结论**: {gt['label']}", ""]
        segs = extract_segs_for(letter)
        per_seg_status = []
        for n in (1, 2, 3):
            seg_text = segs.get(n, "")
            if not seg_text:
                lines.append(f"### seg{n}: (缺失)")
                continue
            obj = run_one(seg_text, SEG_ROLE_MAP[n], system, user_tmpl)
            if "_parse_fail" in obj:
                lines += [f"### seg{n} ⚠ parse 失败",
                          f"raw: {obj['_raw']}", ""]
                per_seg_status.append("PARSE_FAIL")
                continue
            df = obj.get("dominant_factor")
            crit = obj.get("criterion")
            expected = gt["seg_expected"][n]
            dom_status = check_dominant(df, expected)
            crit_status = check_criterion(crit, gt["criterion_keyword"], seg_text)

            lines += [f"### seg{n} ({SEG_ROLE_MAP[n]})",
                      f"- 原文 ({len(seg_text)}字): {seg_text[:120]}{'...' if len(seg_text)>120 else ''}",
                      f"- LLM dominant_factor: **{df}** vs 专家 {expected} → {dom_status}",
                      f"- LLM criterion: {crit_status}",
                      f"- LLM o_primary: {obj.get('o_primary')}",
                      f"- LLM p_surface: {obj.get('p_surface')}",
                      f"- LLM confidence: {obj.get('confidence')}",
                      f"- LLM notes: {obj.get('extraction_notes', '')[:80]}",
                      ""]
            per_seg_status.append(dom_status.split(" ")[0])  # HIT/PARTIAL/MISS

        hits = per_seg_status.count("HIT")
        partials = per_seg_status.count("PARTIAL")
        misses = per_seg_status.count("MISS")
        lines += [f"### {letter} 评估", "",
                  f"- HIT: {hits}/3, PARTIAL: {partials}/3, MISS: {misses}/3",
                  ""]
        summary.append((letter, hits, partials, misses))

    lines += ["---", "## 总评", "",
              "| letter | HIT | PARTIAL | MISS |",
              "|---|---|---|---|"]
    for l, h, p, m in summary:
        lines.append(f"| {l} | {h}/3 | {p}/3 | {m}/3 |")
    lines.append("")
    lines += ["**通过门槛**: 实际 234 段 Q7 大部分是单一 K/V/P 主导, 比 F/G 简单. ",
              "若 holdout HIT+PARTIAL ≥ 4/6 → 可用于 Q7 全量; ≥ 2/6 → 用于 Q7 但需人工 sample 校.",
              "≤ 1/6 → 不可用, prompt 继续 iterate."]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
