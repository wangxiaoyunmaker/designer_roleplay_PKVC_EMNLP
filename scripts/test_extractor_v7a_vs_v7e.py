"""v7a vs v7e 直接对比, 同 20 段 (seed=42) 跑 gpt-4o.

观察 v7e (加 C example + 12 cardId 细分) 是否:
  1. 破除 v7a 的 V 偏 (v7a 20 段 13 个判 V)
  2. 增加 C 主导识别 (v7a 只 2 个 C)
  3. criterion grounded 率维持 90%

输出: data/processed/extractor_v7a_vs_v7e.md
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
OUT_PATH = Path("data/processed/extractor_v7a_vs_v7e.md")

SEG_ROLE_MAP = {"Q7_seg1": "第一反应", "Q7_seg2": "决策链路", "Q7_seg3": "核心 tradeoff"}


def load_prompt(filename):
    p = PROMPT_DIR / filename
    content = p.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    return fences[0], fences[1]


def render(template, seg_role, seg_text):
    out = template
    for k, v in {"{SEGMENT_ROLE}": seg_role, "{SCENARIO}": "Q7_老城区超市重构",
                 "{INPUT_TEXT}": seg_text}.items():
        out = out.replace(k, v)
    return out


def run_one(system, user, model="gpt-4o"):
    raw, _ = call_chat(system, user, model=model, temperature=0.0, use_cache=False)
    try:
        s = raw.find("{"); e = raw.rfind("}")
        return json.loads(raw[s:e+1]) if s >= 0 else {"_raw": raw[:200]}
    except (json.JSONDecodeError, ValueError):
        return {"_parse_fail": True, "_raw": raw[:200]}


def crit_kind(crit, txt):
    if crit is None:
        return "null"
    if crit in txt:
        return "GROUNDED"
    for j in range(0, max(1, len(crit) - 4)):
        if crit[j:j+5] in txt:
            return "PARTIAL"
    return "NOT_IN_INPUT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sys_a, user_a = load_prompt("extractor_v7a.md")
    sys_e, user_e = load_prompt("extractor_v7e.md")

    segs = [json.loads(l) for l in SEGMENTS_PATH.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(segs)
    sample = segs[: args.n]

    lines = [f"# v7a vs v7e — gpt-4o, n={args.n}, seed={args.seed}", "",
             "v7e 加 C example (附录 F seg1) + 12 cardId 细分指引,",
             "看是否破除 v7a 的 V 偏 + 增加 C 识别. criterion grounded 率应维持 90%.",
             ""]

    a_dist, e_dist = {}, {}
    a_crit, e_crit = {}, {}
    factor_agreement = 0
    crit_agreement = 0

    for i, seg in enumerate(sample, 1):
        srole = SEG_ROLE_MAP[seg["scenario_id"]]
        r_a = run_one(sys_a, render(user_a, srole, seg["text"]))
        r_e = run_one(sys_e, render(user_e, srole, seg["text"]))

        a_df = r_a.get("dominant_factor") or "(null)"
        e_df = r_e.get("dominant_factor") or "(null)"
        a_cr = r_a.get("criterion")
        e_cr = r_e.get("criterion")
        a_g = crit_kind(a_cr, seg["text"])
        e_g = crit_kind(e_cr, seg["text"])

        a_dist[a_df] = a_dist.get(a_df, 0) + 1
        e_dist[e_df] = e_dist.get(e_df, 0) + 1
        a_crit[a_g] = a_crit.get(a_g, 0) + 1
        e_crit[e_g] = e_crit.get(e_g, 0) + 1

        # factor 一致 (含路径终点)
        ta = a_df.split("→")[-1] if "→" in a_df else a_df
        te = e_df.split("→")[-1] if "→" in e_df else e_df
        fm = (a_df == e_df) or (ta == te)
        # crit 一致
        cm = (a_cr is None and e_cr is None) or (a_cr and e_cr and
              any(a_cr[j:j+5] in (e_cr or "") for j in range(max(1, len(a_cr)-4))))
        if fm: factor_agreement += 1
        if cm: crit_agreement += 1

        marker = "✓" if (fm and cm) else "⚠" if (fm or cm) else "✗"
        lines += [f"## {i}/{args.n} {marker} | {seg['user_id']} | {seg['scenario_id']} ({srole})",
                  f"原文 ({len(seg['text'])}字): {seg['text'][:160]}...", "",
                  "| variant | dominant_factor | criterion | grounded | confidence |",
                  "|---|---|---|---|---|",
                  f"| v7a | **{a_df}** | `{a_cr}` | {a_g} | {r_a.get('confidence')} |",
                  f"| v7e | **{e_df}** | `{e_cr}` | {e_g} | {r_e.get('confidence')} |",
                  f"- factor 一致: {'YES' if fm else 'NO'}, criterion 一致: {'YES' if cm else 'NO'}",
                  ""]

    n = args.n
    lines += ["---", "## 总评", "",
              f"- factor 跨 prompt 一致率: {factor_agreement}/{n} = {100*factor_agreement/n:.0f}%",
              f"- criterion 跨 prompt 一致率: {crit_agreement}/{n} = {100*crit_agreement/n:.0f}%",
              "", "### dominant_factor 分布",
              "| factor | v7a | v7e |", "|---|---|---|"]
    for f in sorted(set(a_dist) | set(e_dist)):
        lines.append(f"| {f} | {a_dist.get(f, 0)} | {e_dist.get(f, 0)} |")
    lines += ["", "### criterion grounded 状态",
              "| 状态 | v7a | v7e |", "|---|---|---|"]
    for s in ("GROUNDED", "PARTIAL", "null", "NOT_IN_INPUT"):
        lines.append(f"| {s} | {a_crit.get(s, 0)} | {e_crit.get(s, 0)} |")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
