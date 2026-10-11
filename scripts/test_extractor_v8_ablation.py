"""v8 ablation: v8 vs v8_nomarker vs v8_nosc 在 20 sample 上.

测哪些 prompt 元素真正起效:
  - v8_nomarker 砍 12 子类典型句式 marker 表 (~30 行)
  - v8_nosc 砍 self-check Q1-Q7 (~10 行)

若 _nomarker 表现退化 → marker 表关键; 若不退化 → marker 没起作用, 可砍.
若 _nosc 表现退化 → self-check 真校准; 若不退化 → LLM 不真做自查, 可砍.

输出: data/processed/extractor_v8_ablation.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.expand.llm import call_chat

PROMPT_DIR = Path("sft_pipeline/config/prompts")
SEGMENTS_PATH = Path("data/processed/q7_segments.jsonl")
OUT_PATH = Path("data/processed/extractor_v8_ablation.md")
SEG_ROLE_MAP = {"Q7_seg1": "第一反应", "Q7_seg2": "决策链路", "Q7_seg3": "核心 tradeoff"}

VARIANTS = [
    ("v8", "extractor_v8.md"),
    ("v8_nomarker", "extractor_v8_nomarker.md"),
    ("v8_nosc", "extractor_v8_nosc.md"),
]


def load_prompt(filename):
    p = PROMPT_DIR / filename
    content = p.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    return fences[0], fences[1]


def render(template, seg_text):
    return template.replace("{INPUT_TEXT}", seg_text)


def run_one(system, user):
    raw, _ = call_chat(system, user, model="gpt-4o", temperature=0.0, use_cache=False)
    try:
        s = raw.find("{"); e = raw.rfind("}")
        return json.loads(raw[s:e+1]) if s >= 0 else {"_raw": raw[:200]}
    except (json.JSONDecodeError, ValueError):
        return {"_parse_fail": True, "_raw": raw[:200]}


def crit_kind(crit, txt):
    if crit is None: return "null"
    if crit in txt: return "GROUNDED"
    for j in range(0, max(1, len(crit) - 4)):
        if crit[j:j+5] in txt: return "PARTIAL"
    return "NOT_IN_INPUT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    loaded = [(name, load_prompt(fn)) for name, fn in VARIANTS]

    segs = [json.loads(l) for l in SEGMENTS_PATH.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(segs)
    sample = segs[: args.n]

    lines = [f"# v8 Ablation — gpt-4o, n={args.n}, seed={args.seed}", "",
             "测哪些 v8 prompt 元素真正起效:",
             "- **v8** (baseline): 完整 prompt 含 12 子类 marker + self-check",
             "- **v8_nomarker**: 砍 12 子类典型句式 marker 表",
             "- **v8_nosc**: 砍 self-check Q1-Q7",
             "", "若 ablation 跟 baseline 接近 → 该部分没起作用, 可砍.",
             ""]

    dists = {v: {"factor": {}, "crit": {}, "conf": {}} for v, _ in VARIANTS}

    for i, seg in enumerate(sample, 1):
        row_results = []
        for name, (sys_p, user_tmpl) in loaded:
            r = run_one(sys_p, render(user_tmpl, seg["text"]))
            df = r.get("dominant_factor") or "(null)"
            cr = r.get("criterion")
            conf = r.get("confidence") or "(null)"
            g = crit_kind(cr, seg["text"])
            dists[name]["factor"][df] = dists[name]["factor"].get(df, 0) + 1
            dists[name]["crit"][g] = dists[name]["crit"].get(g, 0) + 1
            dists[name]["conf"][conf] = dists[name]["conf"].get(conf, 0) + 1
            row_results.append((name, df, cr, g, conf))

        # all-agreement check
        factors = [r[1] for r in row_results]
        crit_strs = [r[2] or "" for r in row_results]
        agree = "✓" if len(set(factors)) == 1 else "✗"
        lines += [f"## {i}/{args.n} {agree} | {seg['user_id']} | {seg['scenario_id']} ({SEG_ROLE_MAP[seg['scenario_id']]})",
                  f"原文 ({len(seg['text'])}字): {seg['text'][:140]}...", "",
                  "| variant | dominant_factor | criterion | grounded | conf |",
                  "|---|---|---|---|---|"]
        for name, df, cr, g, conf in row_results:
            lines.append(f"| {name} | **{df}** | `{cr}` | {g} | {conf} |")
        lines.append("")

    lines += ["---", "## 总评", "",
              "### dominant_factor 分布",
              "| factor | " + " | ".join(n for n, _ in VARIANTS) + " |",
              "|" + "---|" * (len(VARIANTS) + 1)]
    all_factors = set()
    for v, _ in VARIANTS:
        all_factors.update(dists[v]["factor"].keys())
    for f in sorted(all_factors):
        row = f"| {f} |"
        for v, _ in VARIANTS:
            row += f" {dists[v]['factor'].get(f, 0)} |"
        lines.append(row)

    lines += ["", "### criterion grounded 状态",
              "| 状态 | " + " | ".join(n for n, _ in VARIANTS) + " |",
              "|" + "---|" * (len(VARIANTS) + 1)]
    for s in ("GROUNDED", "PARTIAL", "null", "NOT_IN_INPUT"):
        row = f"| {s} |"
        for v, _ in VARIANTS:
            row += f" {dists[v]['crit'].get(s, 0)} |"
        lines.append(row)

    lines += ["", "### confidence 分布",
              "| level | " + " | ".join(n for n, _ in VARIANTS) + " |",
              "|" + "---|" * (len(VARIANTS) + 1)]
    for c in ("high", "medium", "low", "(null)"):
        row = f"| {c} |"
        for v, _ in VARIANTS:
            row += f" {dists[v]['conf'].get(c, 0)} |"
        lines.append(row)

    lines += ["",
              "**判断**:",
              "- 跟 v8 完全一致 → 砍掉的部分无用 (可 ship 简化版)",
              "- 退化 (V 偏增 / criterion grounded 降 / confidence 全 high) → 砍掉的部分关键",
              "- 偏移其他方向 (e.g. C 识别强了) → 复杂 tradeoff, 看场景"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
