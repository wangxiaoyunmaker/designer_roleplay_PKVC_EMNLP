"""CAPS-anchored role-play inference demo.

给定一组 (target_designer, target_scenario), 跑:
  1. retrieve top-3 CAPS-anchored peers (caps_anchored.retrieve_caps_anchored_top3)
  2. 构造 role-play prompt: target 的 typology signature + 3 个 peer 的 NL/PKVC label
  3. 调 GPT-4o, 让它扮演 target 设计师, 输出 simulated answer
  4. 跟 target 的真实 NL 比 (held-out ground truth)
  5. 输出结果 jsonl 给 spot-check

Usage:
  .venv/bin/python scripts/caps_role_play_inference.py [--n 10] [--scenarios Q7_seg2,Q5]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

from sft_pipeline.rag.caps_anchored import (
    CAPSPool, load_caps_pool, retrieve_caps_anchored_top3, Peer, Designer,
)


SYSTEM_PROMPT = """你是一个设计师认知模拟器. 任务: 给你 1 个目标设计师的 CAPS 类型档案
(per-question 类型分布) + 3 个同类型 peer 设计师在同一场景下的真实回答, 你需要模拟
**目标设计师在指定场景下会怎么回答**.

输出格式:
1. 用第一人称 ("我" / "我觉得" / "我会..."), 中文
2. 长度 200-400 字, 跟 peer NL 风格一致 (口语化 + 推理过程)
3. 体现目标设计师的 CAPS 类型 (从 typology dist 推, 不要 hard-copy peer 的 NL)
4. 不要解释你为什么这么模拟, 只输出 simulated NL"""


def format_typology_dist(typology_dist: dict[str, dict[str, float]]) -> str:
    """Format per-qid typology distribution as compact string for prompt."""
    lines = []
    for qid in ("Q3", "Q4", "Q5", "Q6"):
        if qid not in typology_dist:
            continue
        top3 = sorted(typology_dist[qid].items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{tid}({p:.2f})" for tid, p in top3)
        lines.append(f"  {qid}: {top_str}")
    return "\n".join(lines)


def format_scenario_brief(scenario_key: str) -> str:
    """Brief description of what each scenario asks."""
    BRIEF = {
        "Q7_seg1": "Q7 第一反应: 你接到老城区超市无人化改造项目, 看到 3 个矛盾点后你的第一感受是什么",
        "Q7_seg2": "Q7 决策链路: 在数据/商业要求/用户伦理之间, 你按什么优先级权衡, 思考逻辑是什么",
        "Q7_seg3": "Q7 核心 tradeoff: 面对管理层硬性推进全数字化, 你的核心设计底线 + 妥协方案是什么",
        "Q1": "Q1 设计第一反应: 接到设计任务时你的第一反应模式 (直觉/经验/概念/价值)",
        "Q2": "Q2 设计中心: 设计核心要素的优先排序",
        "Q3": "Q3 数据 vs 美感: 99% 准确率工程方案 vs 高美感设计, 你怎么决策",
        "Q4": "Q4 商业 vs 用户隐私: 黑盒商业模型 vs 用户数据知情权, 你的处理",
        "Q5": "Q5 AI 工具用法: AI 工具在设计流程中扮演什么角色 (主导/辅助/拒绝/...)",
        "Q6": "Q6 用户责任边界: 用户行为后果, 设计师承担多大责任",
    }
    return BRIEF.get(scenario_key, scenario_key)


def build_user_prompt(target: Designer, target_seg, scenario_key: str, peers: list[Peer]) -> str:
    typology_str = format_typology_dist(target.typology_dist)
    scenario_brief = format_scenario_brief(scenario_key)

    peer_blocks = []
    for i, p in enumerate(peers, 1):
        seg = p.segment
        peer_typology = format_typology_dist(target.typology_dist)  # placeholder; will replace
        # use peer's actual typology
        peer_typo_str = ""  # we don't have peer's Designer object passed in
        peer_blocks.append(
            f"### Peer {i} (similarity={p.similarity:.2f}, dom={seg.dominant_factor})\n"
            f"PKVC sequence: {seg.state_sequence}\n"
            f"criterion: {seg.criterion or '(无)'}\n"
            f"NL ({len(seg.text)}字):\n{seg.text}"
        )
    peer_section = "\n\n".join(peer_blocks)

    return f"""# 目标设计师 CAPS 档案

designer_id: {target.designer_id}

Per-qid typology distribution (Q3-Q6 各 top-3):
{typology_str}

# 目标场景

scenario: {scenario_key}
任务: {scenario_brief}

# 3 个 CAPS-anchored peer 真实回答

{peer_section}

# 现在请模拟目标设计师在 {scenario_key} 上的回答

要求:
- 第一人称, 中文, 200-400 字
- 体现 target 的 typology dist 主导特征 (例: 若 Q3 主导是 Q3B-1 P3→K3 type, 模拟时偏向"质疑数据→评估技术栈"风格)
- 风格跟 peer NL 一致 (口语化 / 设计师内部讨论感)
- 不要解释模拟过程, 直接输出 simulated NL
"""


async def infer_one(client, target: Designer, scenario_key: str, peers: list[Peer], model: str):
    target_seg = target.get_scenario(scenario_key)
    user_prompt = build_user_prompt(target, target_seg, scenario_key, peers)
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,                  # 模拟生成允许多样性
            max_tokens=600,
        )
        simulated = r.choices[0].message.content.strip()
        return {
            "ok": True,
            "target_designer": target.designer_id,
            "scenario": scenario_key,
            "ground_truth_nl": target_seg.text,
            "ground_truth_pkvc": {
                "dominant": target_seg.dominant_factor,
                "sequence": target_seg.state_sequence,
                "criterion": target_seg.criterion,
                "confidence": target_seg.confidence,
            },
            "retrieved_peers": [
                {
                    "peer_id": p.designer_id,
                    "similarity": p.similarity,
                    "peer_dom": p.segment.dominant_factor,
                    "peer_seq": p.segment.state_sequence,
                    "peer_nl_preview": p.segment.text[:200],
                }
                for p in peers
            ],
            "simulated_nl": simulated,
            "user_prompt_chars": len(user_prompt),
        }
    except Exception as e:
        return {
            "ok": False,
            "target_designer": target.designer_id,
            "scenario": scenario_key,
            "error": str(e),
        }


async def run_demo(targets: list[tuple[str, str]], pool: CAPSPool, model: str, concurrency: int):
    """targets = list of (designer_id, scenario_key)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return []
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(idx, did, scenario):
        async with sem:
            target = pool.designers.get(did)
            if not target or not target.has_scenario(scenario):
                return {"ok": False, "target_designer": did, "scenario": scenario,
                        "error": "missing target or scenario"}
            peers = retrieve_caps_anchored_top3(did, scenario, pool, k=3)
            if len(peers) < 3:
                print(f"[{idx}] WARN: only {len(peers)} peers for {did} {scenario}", flush=True)
            print(f"[{idx}/{len(targets)}] running {did[:10]} {scenario} ...", flush=True)
            result = await infer_one(client, target, scenario, peers, model)
            return result

    tasks = [asyncio.create_task(_wrap(i + 1, did, sc)) for i, (did, sc) in enumerate(targets)]
    return await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of target samples")
    ap.add_argument("--scenarios", default="Q7_seg1,Q7_seg2,Q7_seg3,Q5",
                    help="comma-separated scenario keys")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", default="reports/datav3/caps_roleplay_demo.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[load] CAPSPool...")
    pool = load_caps_pool()
    print(f"  {len(pool.designers)} designers")

    scenarios = args.scenarios.split(",")
    # pick n random (designer, scenario) pairs that have ground truth + typology dist
    random.seed(args.seed)
    eligible = []
    for did, d in pool.designers.items():
        if not d.typology_dist:
            continue
        for sc in scenarios:
            if d.has_scenario(sc):
                # skip very short / vacuous NL (< 100 字 likely not informative)
                seg = d.get_scenario(sc)
                if len(seg.text) >= 100:
                    eligible.append((did, sc))
    random.shuffle(eligible)
    targets = eligible[:args.n]
    print(f"  {len(eligible)} eligible (designer, scenario) pairs; sampling {len(targets)}")

    if not targets:
        print("ERROR: no eligible targets")
        return 1

    t0 = time.time()
    results = asyncio.run(run_demo(targets, pool, args.model, args.concurrency))
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
