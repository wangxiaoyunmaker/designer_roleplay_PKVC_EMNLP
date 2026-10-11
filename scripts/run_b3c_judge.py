"""Generate B3c (pi+30维+self+CAPS, NO peer) for 10 sampled designers, then judge vs B1.

B3c = B5 minus peer block. Tests whether CAPS routing alone (without peer injection)
produces better persona fidelity than pi-only.

Usage:
  OPENAI_API_KEY=sk-xxx .venv/bin/python scripts/run_b3c_judge.py
  # or:
  .venv/bin/python scripts/run_b3c_judge.py --api-key sk-xxx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

# Reuse existing helpers from run_3_baselines_q7
from scripts.run_3_baselines_q7 import (
    Q7_BACKGROUND, SEG_TASKS, SYSTEM_PROMPT,
    _SEG_OUTPUT_INSTR, build_seg_schema,
    load_designer_demographics,
    load_designer_personas, polarity_to_nl,
    load_self_q1q6, format_self_q1q6,
    load_caps_quadruples, load_designer_scores,
    get_activated_quads, format_activated_quads_nl,
    load_caps_pool,
    call_one,
)
from sft_pipeline.eval.judge import (
    judge_batch_async, compute_win_rate, render_report,
    LIKERT_TO_POLARITY,
)

DATA_DIR = Path("data_v3")
REPORTS_DIR = Path("reports/datav3")


def build_b3c_prompt(
    scenario_key: str, demo_profile: str,
    persona_pol: dict[str, list[str]], self_block: str,
    activated_quads_nl: str,
) -> str:
    """B3c = pi + 30维 + self + CAPS (NO peer)."""
    seg = scenario_key.split("_")[1]
    persona_block = polarity_to_nl(persona_pol)
    parts = [
        f"# 你的人口统计学信息\n{demo_profile}",
        f"# 你的设计信念档案 (30 维 polarity-aware)\n{persona_block}",
        f"# 你的认知路由模式 (CAPS 四元组)\n\n"
        f"以下是基于你的量表得分激活的认知路由模式 — 描述你在不同设计冲突情境下的思维转向倾向:\n\n"
        f"{activated_quads_nl}",
    ]
    if self_block:
        parts.append(f"# 你在其他设计场景的选择与回答\n\n{self_block}")
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASKS[seg]}")
    parts.append(_SEG_OUTPUT_INSTR[seg])
    return "\n\n".join(parts)


def load_b1_generations() -> dict[tuple[str, str], str]:
    src = REPORTS_DIR / "baselines_q7_structured_inference.jsonl"
    out = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("baseline") == "B1" and d.get("ok", True):
                out[(d["designer_id"], d["scenario"])] = d.get("simulated_nl", "")
    return out


def build_profiles(data_dir: Path) -> dict[str, dict[str, str]]:
    import pandas as pd
    scales = pd.read_csv(data_dir / "scale_responses.csv")
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    scales["user_id"] = scales["user_id"].astype(str)
    pivot = scales.pivot_table(index="user_id", columns="question_node", values="score", aggfunc="mean")
    pivot = pivot.fillna(pivot.mean(numeric_only=True))
    profiles = {}
    for uid in pivot.index:
        row = pivot.loc[uid]
        parts = [f"{col}={LIKERT_TO_POLARITY.get(round(row[col]), '弱+')}" for col in sorted(row.index)]
        profiles[uid] = {"trait_summary": "; ".join(parts), "factor_scores": ""}
    return profiles


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default="https://yunwu.ai/v1")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: need API key"); sys.exit(1)

    # ---- Load data ----
    print("[load] CAPSPool + persona + demographics + self + CAPS quads ...")
    pool = load_caps_pool()
    scale_csv = DATA_DIR / "scale_responses.csv"
    persona = load_designer_personas(scale_csv)
    demo = load_designer_demographics(DATA_DIR / "users.csv")
    self_q1q6 = load_self_q1q6(DATA_DIR / "task_responses.csv")
    caps_quads = load_caps_quadruples(Path("data/processed/caps_quadruples.json"))
    designer_scores = load_designer_scores(scale_csv)
    print(f"  {len(pool.designers)} pool, {len(persona)} persona, {len(caps_quads)} quads")

    # ---- Pick the same 10 designers as the previous judge run ----
    import random
    eligible = [did for did in pool.designers
                if persona.get(did)
                and demo.get(did)
                and any(pool.designers[did].has_scenario(f"Q7_{s}") for s in ("seg1","seg2","seg3"))]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    # We need to match the previous 10 pairs; load B1 to get common pool
    b1_gens = load_b1_generations()
    b1_designers = {uid for uid, _ in b1_gens}
    eligible = [d for d in eligible if d in b1_designers]

    # Build (designer, scenario) pairs, sample N
    all_pairs = []
    for did in eligible:
        for seg in ("seg1", "seg2", "seg3"):
            scenario = f"Q7_{seg}"
            if pool.designers[did].has_scenario(scenario) and (did, scenario) in b1_gens:
                all_pairs.append((did, scenario))
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[:args.n_pairs]
    print(f"[select] {len(selected_pairs)} pairs to generate B3c for")

    # ---- Generate B3c ----
    print("[generate] B3c (pi+30维+self+CAPS, no peer) ...")
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)

    async def gen_one(did, scenario):
        async with sem:
            pol = persona.get(did, {})
            demo_profile = demo.get(did, "")
            self_data = self_q1q6.get(did, {})
            seg = scenario.split("_")[1]
            self_block = format_self_q1q6(self_data, seg) if self_data else ""
            scores_d = designer_scores.get(did, {})
            activated = get_activated_quads(scores_d, caps_quads, top_k=5)
            quads_nl = format_activated_quads_nl(activated)
            prompt = build_b3c_prompt(scenario, demo_profile, pol, self_block, quads_nl)
            return await call_one(client, "B3c", did, scenario, prompt, args.model)

    tasks = [gen_one(did, sc) for did, sc in selected_pairs]
    t0 = time.time()
    results_gen = await asyncio.gather(*tasks)
    gen_elapsed = time.time() - t0
    ok_count = sum(1 for r in results_gen if r.get("ok"))
    print(f"  generated {ok_count}/{len(results_gen)} in {gen_elapsed:.1f}s")

    # Save B3c generations
    out_path = REPORTS_DIR / "judge" / "b3c_inference.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results_gen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved {out_path}")

    # ---- Build judge pairs ----
    profiles = build_profiles(DATA_DIR)
    b3c_gens = {}
    for r in results_gen:
        if r.get("ok"):
            b3c_gens[(r["designer_id"], r["scenario"])] = r["simulated_nl"]

    pairs = []
    for did, scenario in selected_pairs:
        if (did, scenario) not in b3c_gens:
            continue
        if did not in profiles:
            continue
        b1_nl = b1_gens.get((did, scenario), "")
        b3c_nl = b3c_gens[(did, scenario)]
        if not b1_nl or not b3c_nl:
            continue
        pairs.append({
            "profile": profiles[did],
            "user_id": did,
            "task_id": scenario,
            "model_response": b3c_nl,
            "baseline_response": b1_nl,
        })
    print(f"[judge] {len(pairs)} pairs (B3c vs B1) ...")

    # ---- Run judge ----
    t0 = time.time()
    judge_results = await judge_batch_async(
        pairs,
        judge_model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
    )
    judge_elapsed = time.time() - t0
    print(f"  judged in {judge_elapsed:.1f}s")

    report = compute_win_rate(judge_results)
    md = render_report(report, label="B3c (pi+30维+self+CAPS) vs B1 (pi)")

    # Save
    detail_path = REPORTS_DIR / "judge" / "judge_B3c_vs_B1.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "B3c",
            "baseline": "B1",
            "description": "B3c = pi + 30维 + self + CAPS (no peer)",
            "judge_model": args.model,
            "n_pairs": len(pairs),
            "gen_elapsed": round(gen_elapsed, 1),
            "judge_elapsed": round(judge_elapsed, 1),
            "report": report.to_dict(),
            "judgments": [j.to_dict() for j in judge_results],
        }, f, ensure_ascii=False, indent=2)

    md_path = REPORTS_DIR / "judge" / "judge_B3c_vs_B1.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n{md}")
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
