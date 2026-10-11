"""Test v8.7 on (1) 5-gold + (2) the 25 v8.6 seg2 single-letter cases.

Goal: verify v8.7 (a) doesn't break 5-gold accuracy, (b) recovers multi-state on
real multi-state seg2 that v8.6 compressed.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
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


def load_prompt(path):
    pt = Path(path).read_text(encoding="utf-8")
    sys_section = pt.find("## System Prompt")
    sys_open = pt.find("```", sys_section)
    sys_close = pt.find("```", sys_open + 3)
    sys_block = pt[sys_open + 3:sys_close].strip()
    usr_section = pt.find("## User Prompt Template")
    usr_open = pt.find("```", usr_section)
    usr_close = pt.find("```", usr_open + 3)
    usr_template = pt[usr_open + 3:usr_close].strip()
    return sys_block, usr_template


async def extract(client, entry, sys_b, usr_t, sem, idx, total):
    async with sem:
        msg = usr_t.replace("{INPUT_TEXT}", entry["text"]).replace("{SEGMENT_ROLE}", entry["segment_role"])
        try:
            r = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": sys_b}, {"role": "user", "content": msg}],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "pkvc_extraction", "schema": V8_7_SCHEMA, "strict": True},
                },
            )
            pred = json.loads(r.choices[0].message.content)
            print(f"[{idx:3d}/{total}] {entry['user_id'][:12]} {entry['segment_role']:6s} "
                  f"dom={pred['dominant_factor']:5s} seq={pred['state_sequence']}", flush=True)
            return {**entry, "pred_v8_7": pred, "ok": True}
        except Exception as e:
            print(f"[{idx:3d}/{total}] FAIL: {e}")
            return {**entry, "pred_v8_7": None, "ok": False, "error": str(e)}


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # Build test set
    entries = []
    # 1. 5-gold (15 segments)
    gold = json.load(open("data/processed/expert_gold/parsed.json"))
    for u in gold["users"]:
        for seg in ("seg1", "seg2", "seg3"):
            entries.append({
                "id": f"GOLD_{u['user_id']}_{seg}",
                "user_id": u["user_id"], "qid": "Q7", "segment_role": seg,
                "text": u[seg]["text"], "source": "5-gold",
                "gold_dominant": u[seg]["dominant"],
                "gold_sequence": u[seg]["sequence"],
            })
    # 2. 25 v8.6 seg2 single-letter cases from data_v3 full run
    v86 = [json.loads(l) for l in Path("data_v3/processed/q7_extracted_v8.6_full.jsonl").open()]
    single_seg2 = [r for r in v86 if r["segment_role"] == "seg2" and len(r["pred"]["state_sequence"]) == 1]
    for r in single_seg2:
        entries.append({
            "id": f"V86SINGLE_{r['user_id']}_seg2",
            "user_id": r["user_id"], "qid": "Q7", "segment_role": "seg2",
            "text": r["text"], "source": "v8.6_single_seg2",
            "v8_6_pred": r["pred"],
        })
    print(f"[total] {len(entries)} test segments: 15 gold + {len(single_seg2)} v8.6 single seg2")

    sys_b, usr_t = load_prompt("sft_pipeline/config/prompts/extractor_v8.7.md")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [extract(client, e, sys_b, usr_t, sem, i + 1, len(entries)) for i, e in enumerate(entries)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    # Analyze
    print("\n=== 5-gold accuracy (v8.7) ===")
    gold_r = [r for r in results if r["source"] == "5-gold" and r["ok"]]
    n_g = sum(1 for r in gold_r if r["pred_v8_7"]["dominant_factor"][0] == r["gold_dominant"])
    print(f"  dominant: {n_g}/{len(gold_r)} = {n_g/len(gold_r):.0%}")
    # show all 15
    print("  per-seg detail:")
    for r in gold_r:
        h = r["pred_v8_7"]["dominant_factor"][0]
        match = "✓" if h == r["gold_dominant"] else "✗"
        print(f"    {match} {r['user_id'][:12]} {r['segment_role']}: v8.7={r['pred_v8_7']['dominant_factor']} (seq={r['pred_v8_7']['state_sequence']}) gold={r['gold_dominant']} (seq={r['gold_sequence']})")

    print("\n=== v8.6 single-letter seg2 → v8.7 sequence diversity ===")
    sing_r = [r for r in results if r["source"] == "v8.6_single_seg2" and r["ok"]]
    n_now_multi = sum(1 for r in sing_r if len(r["pred_v8_7"]["state_sequence"]) >= 2)
    print(f"  v8.6 → v8.7: {n_now_multi}/{len(sing_r)} 转成 multi-state = {n_now_multi/len(sing_r):.0%}")
    print(f"  v8.6 → v8.7 sample changes:")
    for r in sing_r[:15]:
        v86 = r["v8_6_pred"]
        v87 = r["pred_v8_7"]
        change = "↑" if len(v87["state_sequence"]) > 1 else "="
        print(f"    {change} {r['user_id'][:12]}: v8.6 dom={v86['dominant_factor']} seq={v86['state_sequence']} → v8.7 dom={v87['dominant_factor']} seq={v87['state_sequence']}")

    # save
    out = Path("reports/datav3/v8_7_test.json")
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
