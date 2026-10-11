"""跑 v8.7 PKVC extractor on simulated NL, 算 dominant 一致率 per baseline.

输入: reports/datav3/baselines_q7_inference.jsonl (1458 simulated NL)
对比: 每条的 ground_truth_pkvc.dominant (来自 v8.7 标注 q7_extracted_v8.7_full.jsonl)
输出: reports/datav3/baselines_q7_pkvc_eval.jsonl + reports/datav3/baselines_q7_pkvc_eval.md

跑法: ~1457 GPT-4o 调用, concurrency=12, ~15min, ~$22
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

V8_7_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "state_sequence": {
            "type": "array", "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 7,
        },
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": ["dominant_factor", "state_sequence", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def load_prompt(path: Path):
    pt = path.read_text(encoding="utf-8")
    sys_section = pt.find("## System Prompt")
    sys_open = pt.find("```", sys_section)
    sys_close = pt.find("```", sys_open + 3)
    sys_block = pt[sys_open + 3:sys_close].strip()
    usr_section = pt.find("## User Prompt Template")
    usr_open = pt.find("```", usr_section)
    usr_close = pt.find("```", usr_open + 3)
    usr_template = pt[usr_open + 3:usr_close].strip()
    return sys_block, usr_template


async def extract_one(client, model, simulated_nl, sys_b, usr_t, segment_role, sem, idx, total):
    async with sem:
        msg = usr_t.replace("{INPUT_TEXT}", simulated_nl).replace("{SEGMENT_ROLE}", segment_role)
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_b}, {"role": "user", "content": msg}],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "pkvc_extraction", "schema": V8_7_SCHEMA, "strict": True},
                },
            )
            pred = json.loads(r.choices[0].message.content)
            if idx % 100 == 0:
                print(f"[{idx}/{total}] ok", flush=True)
            return {"ok": True, "pred": pred}
        except Exception as e:
            if idx % 100 == 0 or idx <= 5:
                print(f"[{idx}/{total}] FAIL: {e}", flush=True)
            return {"ok": False, "error": str(e)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/datav3/baselines_q7_inference.jsonl")
    ap.add_argument("--output", default="reports/datav3/baselines_q7_pkvc_eval.jsonl")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ok") and r.get("simulated_nl")]
    print(f"[load] {len(rows)} rows to extract")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    sys_b, usr_t = load_prompt(Path("sft_pipeline/config/prompts/extractor_v8.7.md"))
    sem = asyncio.Semaphore(args.concurrency)

    async def _wrap(idx, r):
        seg_role = r["scenario"].split("_")[1] if "_" in r["scenario"] else "single"
        out = await extract_one(client, args.model, r["simulated_nl"], sys_b, usr_t,
                                 seg_role, sem, idx, len(rows))
        return {
            "baseline": r["baseline"],
            "designer_id": r["designer_id"],
            "scenario": r["scenario"],
            "ground_truth_dominant": r.get("ground_truth_pkvc", {}).get("dominant"),
            "ground_truth_sequence": r.get("ground_truth_pkvc", {}).get("sequence"),
            "simulated_nl_chars": len(r["simulated_nl"]),
            **out,
        }

    t0 = time.time()
    tasks = [asyncio.create_task(_wrap(i + 1, r)) for i, r in enumerate(rows)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} extractions ok in {elapsed:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    # quick summary
    print("\n=== per-baseline PKVC dominant accuracy ===")
    by_b = defaultdict(list)
    for r in results:
        if r["ok"] and r["ground_truth_dominant"]:
            pred_head = r["pred"]["dominant_factor"][0]
            gt_head = r["ground_truth_dominant"][0]
            by_b[r["baseline"]].append((gt_head, pred_head))
    for b in ("B1", "B2", "B3"):
        pairs = by_b.get(b, [])
        if not pairs: continue
        acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        print(f"  {b}: {sum(1 for g, p in pairs if g == p)}/{len(pairs)} = {acc:.1%}")
        # confusion
        conf = Counter(pairs)
        print(f"    confusion (gt→pred):")
        for (g, p), c in sorted(conf.items()):
            mark = "✓" if g == p else "✗"
            print(f"      {mark} {g}→{p}: {c}")

    # per-seg breakdown
    print("\n=== per-baseline × per-seg accuracy ===")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        print(f"  {sc}:")
        for b in ("B1", "B2", "B3"):
            pairs = [(g, p) for r in results
                     if r["ok"] and r["baseline"] == b and r["scenario"] == sc
                        and r["ground_truth_dominant"]
                     for g, p in [(r["ground_truth_dominant"][0], r["pred"]["dominant_factor"][0])]]
            if pairs:
                acc = sum(1 for g, p in pairs if g == p) / len(pairs)
                print(f"    {b}: {sum(1 for g, p in pairs if g == p)}/{len(pairs)} = {acc:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
