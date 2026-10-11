"""Run v8.6 on the 48 hand-labeled segments + same 200 val + 5 expert gold.

Compute v8.6 hand-gold accuracy + compare to v8.5.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import AsyncOpenAI

V8_X_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "state_sequence": {
            "type": "array", "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 5,
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


async def extract_one(client, model, entry, sys_b, usr_t, sem, idx, total):
    async with sem:
        msg = (
            usr_t.replace("{INPUT_TEXT}", entry["text"])
                 .replace("{SEGMENT_ROLE}", entry["segment_role"])
        )
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_b}, {"role": "user", "content": msg}],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "pkvc_extraction", "schema": V8_X_SCHEMA, "strict": True},
                },
            )
            pred = json.loads(r.choices[0].message.content)
            print(f"[{idx:3d}/{total}] {entry['user_id'][:10]} {entry['segment_role']:6s} "
                  f"dom={pred['dominant_factor']:5s} seq={pred['state_sequence']} "
                  f"conf={pred['confidence']}", flush=True)
            return {**entry, "pred": pred, "ok": True}
        except Exception as e:
            print(f"[{idx:3d}/{total}] FAIL: {e}", flush=True)
            return {**entry, "pred": None, "ok": False, "error": str(e)}


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # Same 200 val set
    entries = [json.loads(l) for l in Path("data/processed/extractor_val_200.jsonl").open()]

    sys_b, usr_t = load_prompt(Path("sft_pipeline/config/prompts/extractor_v8.6.md"))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [
        extract_one(client, "gpt-4o", e, sys_b, usr_t, sem, i + 1, len(entries))
        for i, e in enumerate(entries)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    out = Path("reports/datav3/val_200_v8.6.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compare against hand-labels (48)
    hand = json.load(open("data/processed/hand_labels.json"))["labels"]
    by_id = {(l["user"], l["qid"], l["seg"]): l for l in hand}
    # match val results to hand labels by (user_id, qid, segment_role)
    matched = []
    for r in results:
        if not r["ok"]:
            continue
        key = (r["user_id"], r["qid"], r["segment_role"])
        h = by_id.get(key)
        if h is None:
            continue
        v86_head = r["pred"]["dominant_factor"][0]
        matched.append({
            "i": h["i"], "group": h["group"], "user": r["user_id"], "qid": r["qid"], "seg": r["segment_role"],
            "v8.5": h["v8.5"], "v8.6": v86_head, "hand": h["hand"],
            "v8.5_match": h["v8.5"] == h["hand"], "v8.6_match": v86_head == h["hand"],
            "v8.6_pred_full": r["pred"]["dominant_factor"], "v8.6_seq": r["pred"]["state_sequence"],
        })
    if matched:
        n5 = sum(1 for m in matched if m["v8.5_match"])
        n6 = sum(1 for m in matched if m["v8.6_match"])
        print(f"\n=== v8.6 vs hand-gold on {len(matched)} segments ===")
        print(f"  v8.5: {n5}/{len(matched)} = {n5/len(matched):.0%}")
        print(f"  v8.6: {n6}/{len(matched)} = {n6/len(matched):.0%}")
        print(f"  delta: {n6 - n5:+d}")
        # changes
        print("\n=== changes (v8.5 → v8.6) ===")
        for m in matched:
            if m["v8.5"] != m["v8.6"]:
                fix = "→FIX" if m["v8.6_match"] and not m["v8.5_match"] else "→BROKE" if not m["v8.6_match"] and m["v8.5_match"] else "→TIE"
                print(f"  [{m['i']:2d}] {m['user'][:10]} {m['seg']}: v8.5={m['v8.5']} → v8.6={m['v8.6']}  hand={m['hand']}  {fix}")
        # all errors in v8.6
        print("\n=== v8.6 errors ===")
        for m in matched:
            if not m["v8.6_match"]:
                print(f"  [{m['i']:2d}] {m['user'][:10]} {m['seg']}: v8.6={m['v8.6']} hand={m['hand']}")

    # global distribution v8.6 vs v8.5
    print(f"\n=== v8.6 distribution on 200 ===")
    for src in ("Q7", "Q1-Q6_expanded"):
        srs = [r for r in results if r["ok"] and r["source"] == src]
        if src == "Q7":
            for seg in ("seg1", "seg2", "seg3"):
                sr = [r for r in srs if r["segment_role"] == seg]
                if not sr: continue
                dom = Counter(r["pred"]["dominant_factor"][0] for r in sr)
                multi = sum(1 for r in sr if len(r["pred"]["state_sequence"]) >= 2)
                print(f"  Q7 {seg}: dom={dict(dom)}  multi={multi}/{len(sr)}")
        else:
            dom = Counter(r["pred"]["dominant_factor"][0] for r in srs)
            print(f"  Q1-Q6: dom={dict(dom)}")

    print(f"\n[saved] reports/datav3/val_200_v8.6.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
