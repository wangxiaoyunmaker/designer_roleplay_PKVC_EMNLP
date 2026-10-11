"""Holdout 测试 extractor_v6 (cleaned 被试原文 + few-shot C/D/E).

读 docs/notes/被试回答专家标注_cleaned.md (清洗版), 用 F/G 的 cleaned 3 segs
跑 v6 prompt, 跟专家标注对比 dominant_factor / criterion grounded.

Usage: set -a; source .env; set +a; unset all_proxy ALL_PROXY
       .venv/bin/python scripts/test_extractor_v6_holdout.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.expand.llm import call_chat

PROMPT_PATH = Path("sft_pipeline/config/prompts/extractor_v6.md")
GOLD_DOC = Path("docs/notes/被试回答专家标注_cleaned.md")
OUT_PATH = Path("data/processed/extractor_v6_holdout.md")


HOLDOUT_GT = {
    "F": {
        "label": "C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)",
        "seg_expected": {
            1: ["C", "C→P", "P"],
            2: ["C", "K", "C→K", "K→V"],
            3: ["V", "V→C", "C", "V→P"],
        },
        "criterion_keyword": ["人类福祉", "幸福", "公平"],
    },
    "G": {
        "label": "P (情境矛盾) → K → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)",
        "seg_expected": {
            1: ["P", "P→K", "P→V", "P→C"],
            2: ["V", "V→K", "V→C", "K→V"],
            3: ["C", "C→V", "V→C", "V"],
        },
        "criterion_keyword": ["设计存在", "更好的体验", "选择权", "尊重用户"],
    },
}


def extract_segs_for(letter: str) -> dict[int, str]:
    """从 cleaned .md 提取 seg1/seg2/seg3."""
    text = GOLD_DOC.read_text(encoding="utf-8")
    m = re.search(rf"## 附录 {letter} —(.+?)(?=\n## 附录 |\n## 清洗|\Z)", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"附录 {letter} 没找到 (cleaned doc)")
    body = m.group(1)
    # 每个 seg 用 "### seg(N) (...)" header 开始, 段落用空行分割
    starts = list(re.finditer(r"### seg(\d) \([^)]+\)\n\n", body))
    out: dict[int, str] = {}
    for i, sm in enumerate(starts):
        n = int(sm.group(1))
        s = sm.end()
        e = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        seg = body[s:e].strip()
        # 截到下一个 ### 或 --- 之前 (已经被 e 控制)
        out[n] = seg
    return out


def load_v6_prompt() -> tuple[str, str]:
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    if len(fences) < 2:
        raise RuntimeError(f"v6 prompt 缺 fenced block (got {len(fences)})")
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
    raw, _ = call_chat(system, user_msg, model="gpt-4o", temperature=0.0, use_cache=False)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        obj = json.loads(raw[s:e + 1]) if s >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        obj = {"_parse_fail": True, "_raw": raw[:200]}
    return obj


def check_dominant(actual: str | None, expected_list: list[str]) -> str:
    if not actual:
        return "MISS (null)"
    actual = actual.strip()
    if actual in expected_list:
        return f"HIT ({actual})"
    # 路径终点匹配
    for exp in expected_list:
        if "→" in exp and actual == exp.split("→")[-1]:
            return f"PARTIAL ({actual} = {exp} 路径终点)"
        if "→" in actual and actual.split("→")[-1] == exp:
            return f"PARTIAL ({actual} 路径终点 = {exp})"
    return f"MISS ({actual}; expected {expected_list})"


def check_criterion(criterion: str | None, keywords: list[str], seg_text: str) -> str:
    if not criterion:
        return "null"
    in_input = criterion in seg_text
    if not in_input:
        # try sub-string match (5+ chars)
        in_input = any(criterion[i:i+5] in seg_text for i in range(0, max(1, len(criterion)-4)))
    keyword_hit = any(kw in criterion for kw in keywords)
    status = "GROUNDED" if in_input else "NOT_IN_INPUT"
    if keyword_hit:
        status += "+KW"
    return f"{status}: {criterion!r}"


def main() -> int:
    if not PROMPT_PATH.exists():
        print(f"v6 prompt 没找到: {PROMPT_PATH}"); return 1
    system, user_tmpl = load_v6_prompt()

    lines = ["# Extractor v6 — Holdout F/G 测试 (cleaned 输入 + 3 few-shot)", "",
             f"- prompt: {PROMPT_PATH}",
             f"- input: {GOLD_DOC} (cleaned 版)",
             "- model: gpt-4o, temperature=0.0, no cache",
             "- 评估: dominant_factor 一致性 + criterion grounded", ""]

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
                lines.append(f"### seg{n}: (缺失)"); continue
            obj = run_one(seg_text, SEG_ROLE_MAP[n], system, user_tmpl)
            if "_parse_fail" in obj:
                lines += [f"### seg{n} ⚠ parse 失败", f"raw: {obj['_raw']}", ""]
                per_seg_status.append("PARSE_FAIL"); continue
            df = obj.get("dominant_factor")
            crit = obj.get("criterion")
            expected = gt["seg_expected"][n]
            dom_status = check_dominant(df, expected)
            crit_status = check_criterion(crit, gt["criterion_keyword"], seg_text)
            lines += [f"### seg{n} ({SEG_ROLE_MAP[n]})",
                      f"- INPUT ({len(seg_text)}字): {seg_text[:100]}...",
                      f"- LLM dominant_factor: **{df}** vs 专家 {expected} → {dom_status}",
                      f"- LLM criterion: {crit_status}",
                      f"- LLM o_primary: {obj.get('o_primary')}",
                      f"- LLM p_surface: {obj.get('p_surface')}",
                      f"- LLM confidence: {obj.get('confidence')}",
                      f"- LLM notes: {(obj.get('extraction_notes') or '')[:80]}",
                      ""]
            per_seg_status.append(dom_status.split(" ")[0])
        hits = per_seg_status.count("HIT")
        partials = per_seg_status.count("PARTIAL")
        misses = per_seg_status.count("MISS")
        lines += [f"### {letter} 评估",
                  f"- HIT: {hits}/3, PARTIAL: {partials}/3, MISS: {misses}/3", ""]
        summary.append((letter, hits, partials, misses))

    lines += ["---", "## 总评", "",
              "| letter | HIT | PARTIAL | MISS |",
              "|---|---|---|---|"]
    for l, h, p, m in summary:
        lines.append(f"| {l} | {h}/3 | {p}/3 | {m}/3 |")
    lines += ["",
              "**通过门槛**: HIT+PARTIAL ≥ 4/6 = 可用; ≥ 2/6 = 用但 sample 人工校;",
              "≤ 1/6 = prompt 还要 iterate."]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
