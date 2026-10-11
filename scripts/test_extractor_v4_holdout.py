"""一次性脚本: 用 extractor_v4.md 跑附录 F + G holdout, 跟专家标注对照.

不改 extract/llm.py (它 hardcoded v1), 直接复用 expand/llm.py 的 generic chat call.

Usage:
    set -a; source .env; set +a; unset all_proxy ALL_PROXY
    .venv/bin/python scripts/test_extractor_v4_holdout.py

输出: data/processed/extractor_v4_holdout.md (人工对照看)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.expand.llm import call_chat

PROMPT_PATH = Path("sft_pipeline/config/prompts/extractor_v4.md")
GOLD_DOC = Path("docs/notes/被试回答专家标注.md")
OUT_PATH = Path("data/processed/extractor_v4_holdout.md")


# 专家标注的 ground truth (从 GOLD_DOC 手抽出关键字段)
HOLDOUT_GT = {
    "F": {
        "card_sequence": "C1 → C1 → P3 → C2 → K2 → K1 → K2 → V1 → V3 → V1 → P3",
        "label": "C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)",
        "strong_sadr": ["#30+", "#15+", "#26+", "#24-", "#10+", "#12+"],
        "criterion": "关注人类福祉, 为人们提供幸福、自由和公平",
    },
    "G": {
        "card_sequence": "P2 → K1 → K1 → V2 → C1 → V1 → V1 → K1 → V3 → C2 → K1 → V2 → C2",
        "label": "P (情境矛盾) → K (双重) → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)",
        "strong_sadr": ["#15+", "#24-", "#26+", "#10-", "#12+", "#30+", "#27-", "#29+"],
        "criterion": "设计存在本质是为人们提供更好的体验, 给用户选择权",
    },
}


def extract_segs_for(letter: str) -> dict[int, str]:
    text = GOLD_DOC.read_text(encoding="utf-8")
    m = re.search(rf"## \*\*附录 {letter}：(.+?)(?=\n## \*\*附录 |\Z)", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"附录 {letter} 没找到")
    body = m.group(1)
    segs = re.findall(r"\*\*seg(\d).+?\*\*\s*[\"\"](.+?)[\"\"]", body, re.DOTALL)
    return {int(n): t.strip() for n, t in segs}


def load_v4_prompt() -> tuple[str, str]:
    """提取 v4 的 system + user template. 跟 expand/prompt.py 同方法."""
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    if len(fences) < 2:
        raise RuntimeError(f"v4 prompt 缺 fenced block (got {len(fences)})")
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


def run_one(letter: str, seg_num: int, seg_text: str, system: str, user_tmpl: str) -> dict:
    user_msg = render_user_prompt(user_tmpl, SEG_ROLE_MAP[seg_num], seg_text)
    raw, cache_hit = call_chat(system, user_msg, model="gpt-4o", temperature=0.2)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        obj = json.loads(raw[s:e + 1]) if s >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        obj = {"_parse_fail": True, "_raw": raw[:200]}
    return {"raw": raw, "parsed": obj, "cache_hit": cache_hit}


def fmt_sadr(items: list) -> str:
    if not items:
        return "(空)"
    parts = []
    for it in items:
        if isinstance(it, dict):
            parts.append(f"{it.get('id','?')}={it.get('label','?')}: {it.get('evidence','')[:30]}")
    return "\n  - " + "\n  - ".join(parts) if parts else "(空)"


def main() -> int:
    if not PROMPT_PATH.exists():
        print(f"v4 prompt 没找到: {PROMPT_PATH}"); return 1
    system, user_tmpl = load_v4_prompt()

    lines = ["# Extractor v4 — Holdout F/G 测试结果", "",
             f"- prompt: {PROMPT_PATH}",
             "- model: gpt-4o, temperature=0.2",
             "- holdout: 附录 F (C 先启 复杂模式), 附录 G (V 双形态)",
             "- 评估角度: dominant_factor / SADR 命中率 / criterion 一致性",
             ""]

    summary_rows = []
    for letter in ("F", "G"):
        gt = HOLDOUT_GT[letter]
        lines += [f"---", f"## 附录 {letter}",
                  f"- **专家结论**: {gt['label']}",
                  f"- **专家 strong SADR**: {', '.join(gt['strong_sadr'])}",
                  f"- **专家 criterion**: {gt['criterion']}",
                  ""]
        segs = extract_segs_for(letter)
        # 抽取后 LLM 命中的 strong sadr 集合 (跨 3 段并集)
        union_sadr: set[str] = set()
        dominant_seq = []
        for n in (1, 2, 3):
            seg_text = segs.get(n, "")
            if not seg_text:
                lines.append(f"### seg{n}: (缺失)")
                continue
            res = run_one(letter, n, seg_text, system, user_tmpl)
            p = res["parsed"]
            lines += [f"### seg{n} ({SEG_ROLE_MAP[n]})",
                      f"原文: {seg_text[:160]}{'...' if len(seg_text)>160 else ''}", ""]
            if "_parse_fail" in p:
                lines.append(f"⚠ parse 失败: {p['_raw']}")
                continue
            df = p.get("dominant_factor") or "?"
            dominant_seq.append(df)
            sadr = p.get("sadr_activations") or []
            for it in sadr:
                if isinstance(it, dict) and it.get("id"):
                    union_sadr.add(it["id"])
            lines += [f"- LLM dominant_factor: **{df}**",
                      f"- LLM p_surface: {p.get('p_surface')}",
                      f"- LLM o_primary: {p.get('o_primary')}",
                      f"- LLM confidence: {p.get('confidence')}",
                      f"- LLM criterion: {p.get('criterion')}",
                      f"- LLM SADR ({len(sadr)} 条):{fmt_sadr(sadr)}", ""]

        # Hit-rate
        hits = [s for s in gt["strong_sadr"] if s in union_sadr]
        miss = [s for s in gt["strong_sadr"] if s not in union_sadr]
        extra = [s for s in union_sadr if s not in gt["strong_sadr"]]
        rate = len(hits) / max(len(gt["strong_sadr"]), 1)
        lines += [f"### {letter} 评估", "",
                  f"- LLM dominant 序列: {' → '.join(dominant_seq)}",
                  f"- LLM SADR 并集 (跨 3 段): {sorted(union_sadr)}",
                  f"- 命中专家 strong SADR: **{len(hits)}/{len(gt['strong_sadr'])}** = {100*rate:.0f}%",
                  f"  - hit: {hits}",
                  f"  - miss: {miss}",
                  f"  - extra (LLM 多抽的, 可能 noise 也可能漏掉的真信号): {extra[:10]}{'...' if len(extra)>10 else ''}",
                  ""]
        summary_rows.append((letter, len(hits), len(gt["strong_sadr"]), rate, dominant_seq))

    lines += ["---", "## 总评", "", "| letter | strong SADR hit | rate | dominant 序列 |",
              "|---|---|---|---|"]
    for letter, h, t, r, seq in summary_rows:
        lines.append(f"| {letter} | {h}/{t} | {100*r:.0f}% | {' → '.join(seq)} |")
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
