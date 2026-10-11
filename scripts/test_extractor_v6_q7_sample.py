"""抽 N 条 q7_segments 真实数据用 v6.1 prompt 跑, 输出对照表.

跟 holdout 不同: 没有专家 ground truth, 只 dump LLM 输出让人眼判断.

抽样: 78 designer × 3 seg = 234 段, 固定 seed shuffle 后取前 N. 保证 segment 类型
均衡 (大致 seg1 / seg2 / seg3 各 1/3).

Usage:
    set -a; source .env; set +a; unset all_proxy ALL_PROXY
    .venv/bin/python scripts/test_extractor_v6_q7_sample.py [--n 20]

输出: data/processed/extractor_v6_q7_sample{N}.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

from sft_pipeline.expand.llm import call_chat

PROMPT_PATH = Path("sft_pipeline/config/prompts/extractor_v6.md")
SEGMENTS_PATH = Path("data/processed/q7_segments.jsonl")


def call_chat_via_openai_router(system: str, user: str, model: str, temperature: float = 0.0) -> str:
    """走 OPENAI_BASE_URL (yunwu.ai) 的 openai-compatible 接口调任意 model (含 claude/deepseek).

    yunwu.ai 这种中转网关把 anthropic / deepseek / openai 都包装成同一个
    chat/completions 接口, 区别只在 model name. 比 anthropic SDK 直连方便.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


SEG_ROLE_MAP = {"Q7_seg1": "第一反应", "Q7_seg2": "决策链路", "Q7_seg3": "核心 tradeoff"}


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
        "{C_LOCAL_TAG}": "(sample 测试, 无 metadata)",
        "{TASK_BACKGROUND}": "(sample 测试, 无 metadata)",
        "{S_PRIMARY_TAGS}": "(sample, 无 metadata)",
        "{S_SECONDARY_FACTOR}": "(sample, 无 metadata)",
        "{INPUT_TEXT}": seg_text,
    }
    out = template
    for k, v in placeholders.items():
        out = out.replace(k, v)
    return out


def run_one(seg_text: str, seg_role: str, system: str, user_tmpl: str, model: str) -> dict:
    user_msg = render_user_prompt(user_tmpl, seg_role, seg_text)
    if model.startswith("gpt-"):
        raw, _ = call_chat(system, user_msg, model=model, temperature=0.0, use_cache=False)
    else:
        # claude / deepseek 走 yunwu.ai openai-compat 接口
        raw = call_chat_via_openai_router(system, user_msg, model, temperature=0.0)
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        obj = json.loads(raw[s:e + 1]) if s >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        obj = {"_parse_fail": True, "_raw": raw[:200]}
    return obj


def check_criterion_grounded(crit: str | None, seg_text: str) -> str:
    if not crit:
        return "null"
    if crit in seg_text:
        return "GROUNDED (verbatim)"
    # fuzzy: 任意 5+ char substring 是否在 input
    for i in range(0, max(1, len(crit) - 4)):
        if crit[i:i+5] in seg_text:
            return "PARTIAL (5+ 字符子串在 INPUT)"
    return "NOT_IN_INPUT"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="gpt-4o",
                   help="gpt-4o | claude-3-5-sonnet-20241022 | deepseek-chat | ...")
    args = p.parse_args()

    if not PROMPT_PATH.exists():
        print(f"v6 prompt 没找到: {PROMPT_PATH}"); return 1
    system, user_tmpl = load_v6_prompt()

    all_segs = [json.loads(l) for l in SEGMENTS_PATH.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(all_segs)
    sample = all_segs[: args.n]

    lines = [f"# Extractor v6.1 — Q7 实际数据 sample (n={args.n}, model={args.model})", "",
             f"- prompt: {PROMPT_PATH}",
             f"- input: {SEGMENTS_PATH}",
             f"- model: {args.model}, temperature=0.0, no cache, seed={args.seed}",
             "- 评估: dominant_factor (人眼判) + criterion grounded (脚本判)", ""]

    summary = {"P": 0, "K": 0, "V": 0, "C": 0, "path": 0, "null": 0}
    crit_status = {"GROUNDED (verbatim)": 0, "PARTIAL (5+ 字符子串在 INPUT)": 0,
                   "NOT_IN_INPUT": 0, "null": 0}
    confidence_dist = {"high": 0, "medium": 0, "low": 0, None: 0}

    for i, seg in enumerate(sample, 1):
        seg_role = SEG_ROLE_MAP[seg["scenario_id"]]
        obj = run_one(seg["text"], seg_role, system, user_tmpl, args.model)
        if "_parse_fail" in obj:
            lines += [f"## {i}/{args.n} ⚠ parse fail | {seg['user_id']} {seg['scenario_id']}",
                      f"raw: {obj['_raw']}", ""]
            continue

        df = obj.get("dominant_factor") or "(null)"
        crit = obj.get("criterion")
        crit_st = check_criterion_grounded(crit, seg["text"])
        crit_status[crit_st] = crit_status.get(crit_st, 0) + 1

        if df == "(null)": summary["null"] += 1
        elif "→" in df: summary["path"] += 1
        else: summary[df] = summary.get(df, 0) + 1

        conf = obj.get("confidence")
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        lines += [f"## {i}/{args.n} | designer {seg['user_id']} | {seg['scenario_id']} ({seg_role})",
                  f"**原文** ({len(seg['text'])}字):",
                  f"> {seg['text'][:300]}{'...' if len(seg['text'])>300 else ''}",
                  "",
                  f"**LLM 输出**:",
                  f"- dominant_factor: **{df}**",
                  f"- criterion: `{crit}` → {crit_st}",
                  f"- criterion_strength: {obj.get('criterion_strength')}",
                  f"- confidence: {conf}",
                  f"- p_surface: {obj.get('p_surface')}",
                  f"- o_primary: {obj.get('o_primary')}",
                  f"- extraction_notes: {(obj.get('extraction_notes') or '')[:80]}",
                  ""]

    lines += ["---", "## 总评", "", "### dominant_factor 分布",
              "| factor | count |", "|---|---|"]
    for k, v in summary.items():
        if v: lines.append(f"| {k} | {v} |")
    lines += ["", "### criterion grounded 状态",
              "| 状态 | count |", "|---|---|"]
    for k, v in crit_status.items():
        if v: lines.append(f"| {k} | {v} |")
    lines += ["", "### confidence 分布",
              "| level | count |", "|---|---|"]
    for k, v in confidence_dist.items():
        if v: lines.append(f"| {k} | {v} |")

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = Path(f"data/processed/extractor_v6_q7_sample{args.n}_{safe_model}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
