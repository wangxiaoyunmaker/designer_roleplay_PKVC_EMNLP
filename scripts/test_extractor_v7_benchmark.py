"""4 个 v7 variant 在 20 段 Q7 sample 上的对比 benchmark.

Variants:
  v7a — minimal 4 字段 (砍 T_CAPS), 单 prompt
  v7b — conditional segment-aware, 单 prompt 加段类型条件指令
  v7c — 拆 3 个 prompt (按 segment_role: B1/B2/B3)
  v7d — 拆 2 个 prompt (按字段: factor + criterion), 每段调 2 次 LLM

输出: data/processed/extractor_v7_benchmark.md (对照表)

Usage:
    set -a; source .env; set +a; unset all_proxy ALL_PROXY
    .venv/bin/python scripts/test_extractor_v7_benchmark.py [--n 20]
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
OUT_PATH = Path("data/processed/extractor_v7_benchmark.md")

SEG_ROLE_MAP = {"Q7_seg1": "第一反应", "Q7_seg2": "决策链路", "Q7_seg3": "核心 tradeoff"}
SEG_TO_V7C_PROMPT = {
    "Q7_seg1": "extractor_v7c_b1.md",
    "Q7_seg2": "extractor_v7c_b2.md",
    "Q7_seg3": "extractor_v7c_b3.md",
}


def load_prompt(filename: str) -> tuple[str, str]:
    path = PROMPT_DIR / filename
    content = path.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    if len(fences) < 2:
        raise RuntimeError(f"{filename} 缺 fenced block (got {len(fences)})")
    return fences[0], fences[1]


def render(template: str, seg_role: str, seg_text: str) -> str:
    out = template
    for k, v in {
        "{SEGMENT_ROLE}": seg_role,
        "{SCENARIO}": "Q7_老城区超市重构",
        "{INPUT_TEXT}": seg_text,
    }.items():
        out = out.replace(k, v)
    return out


def call_one(system: str, user: str, model: str = "gpt-4o") -> dict:
    raw, _ = call_chat(system, user, model=model, temperature=0.0, use_cache=False)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        return json.loads(raw[s:e + 1]) if s >= 0 else {"_raw": raw[:200]}
    except (json.JSONDecodeError, ValueError):
        return {"_parse_fail": True, "_raw": raw[:200]}


def run_v7a(seg, sys_a, user_tmpl_a) -> dict:
    user_msg = render(user_tmpl_a, SEG_ROLE_MAP[seg["scenario_id"]], seg["text"])
    return call_one(sys_a, user_msg)


def run_v7b(seg, sys_b, user_tmpl_b) -> dict:
    user_msg = render(user_tmpl_b, SEG_ROLE_MAP[seg["scenario_id"]], seg["text"])
    return call_one(sys_b, user_msg)


def run_v7c(seg, c_prompts) -> dict:
    pf = SEG_TO_V7C_PROMPT[seg["scenario_id"]]
    sys_c, user_tmpl_c = c_prompts[pf]
    user_msg = render(user_tmpl_c, SEG_ROLE_MAP[seg["scenario_id"]], seg["text"])
    return call_one(sys_c, user_msg)


def run_v7d(seg, sys_f, user_tmpl_f, sys_c, user_tmpl_c) -> dict:
    """v7d 两步: factor + criterion 各调 1 次, 合并 result."""
    user_msg_f = render(user_tmpl_f, SEG_ROLE_MAP[seg["scenario_id"]], seg["text"])
    r_f = call_one(sys_f, user_msg_f)
    user_msg_c = render(user_tmpl_c, SEG_ROLE_MAP[seg["scenario_id"]], seg["text"])
    r_c = call_one(sys_c, user_msg_c)
    return {
        "dominant_factor": r_f.get("dominant_factor"),
        "criterion":       r_c.get("criterion"),
        "confidence":      r_f.get("confidence"),  # use factor's confidence as primary
        "_factor_raw":     r_f,
        "_criterion_raw":  r_c,
    }


def check_grounded(crit, seg_text):
    if not crit:
        return "null"
    if crit in seg_text:
        return "GROUNDED"
    for i in range(0, max(1, len(crit) - 4)):
        if crit[i:i+5] in seg_text:
            return "PARTIAL (5+ 字)"
    return "NOT_IN_INPUT"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load all prompts
    sys_a, user_tmpl_a = load_prompt("extractor_v7a.md")
    sys_b, user_tmpl_b = load_prompt("extractor_v7b.md")
    c_prompts = {fn: load_prompt(fn) for fn in SEG_TO_V7C_PROMPT.values()}
    sys_d_f, user_tmpl_d_f = load_prompt("extractor_v7d_factor.md")
    sys_d_c, user_tmpl_d_c = load_prompt("extractor_v7d_criterion.md")

    # Load + sample segments
    segs = [json.loads(l) for l in SEGMENTS_PATH.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(segs)
    sample = segs[: args.n]

    lines = [f"# Extractor v7 Benchmark — 4 variants × {args.n} 段 Q7",
             "",
             "Variants:",
             "- **v7a** minimal 4 字段 (砍 T_CAPS), 单 prompt",
             "- **v7b** conditional segment-aware, 单 prompt",
             "- **v7c** 拆 3 prompt (按 segment B1/B2/B3)",
             "- **v7d** 拆 2 prompt (按字段 factor + criterion, 每段调 2 次)",
             "",
             f"sample seed={args.seed}, model=gpt-4o, temperature=0.0, no cache",
             ""]

    # Aggregate stats
    stats: dict[str, dict] = {v: {"factor_dist": {}, "crit_status": {}}
                              for v in ("v7a", "v7b", "v7c", "v7d")}

    for i, seg in enumerate(sample, 1):
        srole = SEG_ROLE_MAP[seg["scenario_id"]]
        r_a = run_v7a(seg, sys_a, user_tmpl_a)
        r_b = run_v7b(seg, sys_b, user_tmpl_b)
        r_c = run_v7c(seg, c_prompts)
        r_d = run_v7d(seg, sys_d_f, user_tmpl_d_f, sys_d_c, user_tmpl_d_c)

        lines += [f"## {i}/{args.n} | {seg['user_id']} | {seg['scenario_id']} ({srole})",
                  f"**原文** ({len(seg['text'])}字):",
                  f"> {seg['text'][:200]}{'...' if len(seg['text'])>200 else ''}",
                  "",
                  "| variant | dominant_factor | criterion | grounded | confidence |",
                  "|---|---|---|---|---|"]

        for v, r in (("v7a", r_a), ("v7b", r_b), ("v7c", r_c), ("v7d", r_d)):
            df = r.get("dominant_factor") or "(null)"
            crit = r.get("criterion")
            grounded = check_grounded(crit, seg["text"])
            conf = r.get("confidence") or "(null)"
            lines.append(f"| {v} | **{df}** | `{crit}` | {grounded} | {conf} |")
            # tally
            stats[v]["factor_dist"][df] = stats[v]["factor_dist"].get(df, 0) + 1
            stats[v]["crit_status"][grounded] = stats[v]["crit_status"].get(grounded, 0) + 1
        lines.append("")

    lines += ["---", "## 总评 (跨 variant 对比)", ""]
    lines += ["### dominant_factor 分布",
              "| factor | v7a | v7b | v7c | v7d |",
              "|---|---|---|---|---|"]
    all_factors = set()
    for v in ("v7a", "v7b", "v7c", "v7d"):
        all_factors.update(stats[v]["factor_dist"].keys())
    for f in sorted(all_factors):
        row = f"| {f} |"
        for v in ("v7a", "v7b", "v7c", "v7d"):
            row += f" {stats[v]['factor_dist'].get(f, 0)} |"
        lines.append(row)

    lines += ["", "### criterion grounded 状态",
              "| 状态 | v7a | v7b | v7c | v7d |",
              "|---|---|---|---|---|"]
    all_gs = set()
    for v in ("v7a", "v7b", "v7c", "v7d"):
        all_gs.update(stats[v]["crit_status"].keys())
    for s in sorted(all_gs):
        row = f"| {s} |"
        for v in ("v7a", "v7b", "v7c", "v7d"):
            row += f" {stats[v]['crit_status'].get(s, 0)} |"
        lines.append(row)

    lines += ["",
              "### 判断标准 (无 ground truth, 抽样人眼判)",
              "- factor 分布: P/K/V/C 不应该过度偏向某 1 类 (v6.1 的 P=13/20 = 65% 过度抽)",
              "- criterion 应有更多 GROUNDED 或 null (NOT_IN_INPUT 越少越好)",
              "- 跨 variant 一致性: 多数 variant 判同一 factor 高 confidence"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
