"""Run v8.5 extractor on 200-seg val set with concurrency.

Parallel via asyncio + AsyncOpenAI, concurrency=8. Output written incrementally.
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

V8_5_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
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


async def extract_one(client, model, entry, sys_block, usr_template, sem, idx, total):
    async with sem:
        msg = (
            usr_template
            .replace("{INPUT_TEXT}", entry["text"])
            .replace("{SEGMENT_ROLE}", entry["segment_role"])
        )
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_block},
                    {"role": "user", "content": msg},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "pkvc_extraction", "schema": V8_5_SCHEMA, "strict": True},
                },
            )
            pred = json.loads(r.choices[0].message.content)
            print(f"[{idx:3d}/{total}] {entry['user_id'][:10]} {entry['segment_role']:6s} "
                  f"dom={pred['dominant_factor']:5s} seq={pred['state_sequence']} "
                  f"conf={pred['confidence']}", flush=True)
            return {**entry, "pred": pred, "ok": True}
        except Exception as e:
            print(f"[{idx:3d}/{total}] FAIL {entry['user_id'][:10]} {entry['segment_role']}: {e}", flush=True)
            return {**entry, "pred": None, "ok": False, "error": str(e)}


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    entries = [json.loads(l) for l in Path("data/processed/extractor_val_200.jsonl").open()]
    print(f"[data] {len(entries)} entries")

    sys_block, usr_template = load_prompt(Path("sft_pipeline/config/prompts/extractor_v8.5.md"))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [
        extract_one(client, "gpt-4o", e, sys_block, usr_template, sem, i + 1, len(entries))
        for i, e in enumerate(entries)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    out = Path("reports/datav3/val_200_v8.5.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out}")

    # summary
    print(f"\n=== distribution (n={n_ok}) ===")
    for src in ("Q7", "Q1-Q6_expanded"):
        srs = [r for r in results if r["ok"] and r["source"] == src]
        if not srs:
            continue
        if src == "Q7":
            for seg in ("seg1", "seg2", "seg3"):
                sr = [r for r in srs if r["segment_role"] == seg]
                if not sr:
                    continue
                dom = Counter(r["pred"]["dominant_factor"][0] for r in sr)
                seq_len = Counter(len(r["pred"]["state_sequence"]) for r in sr)
                conf = Counter(r["pred"]["confidence"] for r in sr)
                multi = sum(1 for r in sr if len(r["pred"]["state_sequence"]) >= 2)
                print(f"\n[{src} {seg}] n={len(sr)}")
                print(f"  dom: {dict(dom)}")
                print(f"  seq_len: {dict(sorted(seq_len.items()))}; multi {multi/len(sr):.0%}")
                print(f"  conf: {dict(conf)}")
        else:
            dom = Counter(r["pred"]["dominant_factor"][0] for r in srs)
            seq_len = Counter(len(r["pred"]["state_sequence"]) for r in srs)
            conf = Counter(r["pred"]["confidence"] for r in srs)
            print(f"\n[{src}] n={len(srs)}")
            print(f"  dom: {dict(dom)}")
            print(f"  seq_len: {dict(sorted(seq_len.items()))}")
            print(f"  conf: {dict(conf)}")
            # vs expansion_inferred_factor
            match = sum(1 for r in srs if r["pred"]["dominant_factor"][0] == r.get("expansion_inferred_factor"))
            print(f"  vs expansion inferred_factor (canonical): {match}/{len(srs)} = {match/len(srs):.0%}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
