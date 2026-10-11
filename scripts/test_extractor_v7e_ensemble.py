"""v7e gpt-4o + claude-3-5-sonnet 跨家族 ensemble 测试.

同 20 段 (seed=42) 跑两个 model 用 v7e prompt, 算 agreement rate:
  - factor_agreement: dominant_factor 完全一致 (或路径终点一致)
  - criterion_agreement: 都 null 或文字相似 (5+ char overlap)

输出: data/processed/extractor_v7e_ensemble.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.expand.llm import call_chat

PROMPT_PATH = Path("sft_pipeline/config/prompts/extractor_v7e.md")
SEGMENTS_PATH = Path("data/processed/q7_segments.jsonl")
OUT_PATH = Path("data/processed/extractor_v7e_ensemble.md")
SEG_ROLE_MAP = {"Q7_seg1": "第一反应", "Q7_seg2": "决策链路", "Q7_seg3": "核心 tradeoff"}


def call_via_yunwu(system: str, user: str, model: str, temperature: float = 0.0) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def load_prompt():
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    return fences[0], fences[1]


def render(template, seg_role, seg_text):
    out = template
    for k, v in {"{SEGMENT_ROLE}": seg_role, "{SCENARIO}": "Q7_老城区超市重构",
                 "{INPUT_TEXT}": seg_text}.items():
        out = out.replace(k, v)
    return out


def run_one(system, user, model):
    if model.startswith("gpt-"):
        raw, _ = call_chat(system, user, model=model, temperature=0.0, use_cache=False)
    else:
        raw = call_via_yunwu(system, user, model)
    try:
        s = raw.find("{"); e = raw.rfind("}")
        return json.loads(raw[s:e+1]) if s >= 0 else {"_raw": raw[:200]}
    except (json.JSONDecodeError, ValueError):
        return {"_parse_fail": True, "_raw": raw[:200]}


def factor_match(a, b):
    """两个 dominant_factor 是否算一致 (完全相同 或 路径终点相同)."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta = a.split("→")[-1] if "→" in a else a
    tb = b.split("→")[-1] if "→" in b else b
    return ta == tb


def crit_overlap(a, b, threshold=5):
    """两个 criterion: 都 null 算 match; 否则 5+ 字 substring overlap."""
    if not a and not b:
        return True
    if (not a) != (not b):
        return False
    for i in range(0, max(1, len(a) - threshold + 1)):
        if a[i:i+threshold] in b:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    system, user_tmpl = load_prompt()
    segs = [json.loads(l) for l in SEGMENTS_PATH.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(segs)
    sample = segs[: args.n]

    lines = [f"# Extractor v7e Ensemble — gpt-4o vs claude-3-5-sonnet (n={args.n})",
             "",
             f"- prompt: {PROMPT_PATH}",
             f"- models: gpt-4o + claude-3-5-sonnet-20241022 (via yunwu.ai)",
             f"- sample seed={args.seed}, temperature=0.0", ""]

    factor_match_count = 0
    crit_match_count = 0
    g_factor_dist = {}
    c_factor_dist = {}
    g_crit_status = {"GROUNDED": 0, "NOT_IN_INPUT": 0, "null": 0, "PARTIAL": 0}
    c_crit_status = {"GROUNDED": 0, "NOT_IN_INPUT": 0, "null": 0, "PARTIAL": 0}

    for i, seg in enumerate(sample, 1):
        srole = SEG_ROLE_MAP[seg["scenario_id"]]
        user = render(user_tmpl, srole, seg["text"])
        r_g = run_one(system, user, "gpt-4o")
        r_c = run_one(system, user, "claude-3-5-sonnet-20241022")

        g_df = r_g.get("dominant_factor") or "(null)"
        c_df = r_c.get("dominant_factor") or "(null)"
        g_crit = r_g.get("criterion")
        c_crit = r_c.get("criterion")

        # tally
        g_factor_dist[g_df] = g_factor_dist.get(g_df, 0) + 1
        c_factor_dist[c_df] = c_factor_dist.get(c_df, 0) + 1
        for src, crit, dist in ((g_crit, "g"), (c_crit, "c")):
            pass  # placeholder

        def crit_kind(crit, txt):
            if crit is None:
                return "null"
            if crit in txt:
                return "GROUNDED"
            for j in range(0, max(1, len(crit) - 4)):
                if crit[j:j+5] in txt:
                    return "PARTIAL"
            return "NOT_IN_INPUT"

        g_crit_status[crit_kind(g_crit, seg["text"])] += 1
        c_crit_status[crit_kind(c_crit, seg["text"])] += 1

        fm = factor_match(g_df, c_df)
        cm = crit_overlap(g_crit or "", c_crit or "")
        if fm: factor_match_count += 1
        if cm: crit_match_count += 1
        marker = "✓" if (fm and cm) else "⚠" if (fm or cm) else "✗"

        lines += [f"## {i}/{args.n} {marker} | {seg['user_id']} | {seg['scenario_id']} ({srole})",
                  f"**原文** ({len(seg['text'])}字): {seg['text'][:160]}...",
                  "",
                  "| model | dominant_factor | criterion | confidence |",
                  "|---|---|---|---|",
                  f"| gpt-4o | **{g_df}** | `{g_crit}` | {r_g.get('confidence')} |",
                  f"| claude | **{c_df}** | `{c_crit}` | {r_c.get('confidence')} |",
                  f"- factor_match: {'YES' if fm else 'NO'}, crit_match: {'YES' if cm else 'NO'}", ""]

    n = args.n
    lines += ["---", "## 总评", "",
              f"- **factor 一致率**: {factor_match_count}/{n} = {100*factor_match_count/n:.0f}%",
              f"- **criterion 一致率**: {crit_match_count}/{n} = {100*crit_match_count/n:.0f}%",
              "", "### dominant_factor 分布",
              "| factor | gpt-4o | claude |", "|---|---|---|"]
    all_f = set(g_factor_dist) | set(c_factor_dist)
    for f in sorted(all_f):
        lines.append(f"| {f} | {g_factor_dist.get(f, 0)} | {c_factor_dist.get(f, 0)} |")
    lines += ["", "### criterion grounded 状态",
              "| 状态 | gpt-4o | claude |", "|---|---|---|"]
    for s in ("GROUNDED", "PARTIAL", "null", "NOT_IN_INPUT"):
        lines.append(f"| {s} | {g_crit_status.get(s, 0)} | {c_crit_status.get(s, 0)} |")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
