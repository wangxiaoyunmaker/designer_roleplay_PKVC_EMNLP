"""Run v8.6 on full data_v3 corpus: 163 users × 3 Q7 seg + 309 Q1-Q6 expansion = 798 segments.

Diff from run_full_corpus.py: uses data_v3/ not data/, and reads Q7 directly from
task_responses.csv (no pre-built q7_segments.jsonl in data_v3).
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
import pandas as pd
from openai import AsyncOpenAI

DV3 = Path(__file__).resolve().parent.parent / "data_v3"

V8_X_SCHEMA = {
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


async def extract_one(client, model, entry, sys_b, usr_t, sem, idx, total):
    async with sem:
        msg = (
            usr_t.replace("{INPUT_TEXT}", entry["text"])
                 .replace("{SEGMENT_ROLE}", entry["segment_role"])
        )
        for attempt in range(3):
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
                if idx % 50 == 0:
                    print(f"[{idx:3d}/{total}] {entry['user_id'][:10]} {entry['segment_role']:6s} dom={pred['dominant_factor']:5s}", flush=True)
                return {**entry, "pred": pred, "ok": True}
            except Exception as e:
                if attempt == 2:
                    print(f"[{idx:3d}/{total}] FAIL after 3 retries: {e}", flush=True)
                    return {**entry, "pred": None, "ok": False, "error": str(e)}
                await asyncio.sleep(1 + attempt)


def build_entries():
    entries = []
    # Q7 from data_v3/task_responses.csv
    tr = pd.read_csv(DV3 / "task_responses.csv")
    q7 = tr[tr["question_id"] == "Q7"]
    print(f"[data_v3] {len(q7)} Q7 rows")
    n_q7_segs = 0
    for _, row in q7.iterrows():
        try:
            blob = json.loads(row["answer_value"])
        except (json.JSONDecodeError, TypeError):
            # try cleaning control chars
            try:
                import re
                s = re.sub(r"[\x00-\x1f]+", " ", row["answer_value"])
                blob = json.loads(s)
            except Exception:
                print(f"  skip {row['user_id']} (bad JSON)")
                continue
        for i, seg in enumerate(("seg1", "seg2", "seg3"), 1):
            if seg not in blob:
                continue
            text = blob[seg].get("text", "") if isinstance(blob[seg], dict) else str(blob[seg])
            text = text.strip()
            if not text:
                continue
            entries.append({
                "id": f"{row['user_id']}_Q7_{seg}",
                "user_id": row["user_id"], "qid": "Q7", "segment_role": seg,
                "text": text, "source": "Q7",
            })
            n_q7_segs += 1
    print(f"[Q7] built {n_q7_segs} segments")

    # Q1-Q6 expansion from data_v3
    exp_raw = [json.loads(l) for l in (DV3 / "processed/q1q6_expansion_outputs.jsonl").open()]
    n_q16 = 0
    for r in exp_raw:
        result = r.get("result")
        if not isinstance(result, dict):
            continue
        text = result.get("expanded_nl", "").strip()
        if len(text) < 30:
            continue
        entries.append({
            "id": r.get("candidate_id") or f"{r['user_id']}_{r['question_id']}",
            "user_id": r["user_id"], "qid": r["question_id"], "segment_role": "single",
            "text": text, "source": "Q1-Q6_expanded",
            "expansion_inferred_factor": result.get("inferred_factor"),
            "expansion_inferred_subtype": result.get("inferred_subtype"),
        })
        n_q16 += 1
    print(f"[Q1-Q6] built {n_q16} segments")
    return entries


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    entries = build_entries()
    print(f"[total] {len(entries)} segments\n")

    sys_b, usr_t = load_prompt(Path("sft_pipeline/config/prompts/extractor_v8.7.md"))
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
    n_fail = len(results) - n_ok
    print(f"\n[done] {n_ok}/{len(results)} ok, {n_fail} fail in {elapsed:.1f}s")

    # save split — to data_v3/processed/
    q7_out = DV3 / "processed/q7_extracted_v8.7_full.jsonl"
    q16_out = DV3 / "processed/q1q6_extracted_v8.7_full.jsonl"
    with q7_out.open("w", encoding="utf-8") as f:
        for r in results:
            if r["source"] == "Q7":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with q16_out.open("w", encoding="utf-8") as f:
        for r in results:
            if r["source"] == "Q1-Q6_expanded":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {q7_out}\n        {q16_out}")

    # distribution
    print(f"\n=== full data_v3 distribution ===")
    for src in ("Q7", "Q1-Q6_expanded"):
        srs = [r for r in results if r["ok"] and r["source"] == src]
        if src == "Q7":
            for seg in ("seg1", "seg2", "seg3"):
                sr = [r for r in srs if r["segment_role"] == seg]
                if not sr: continue
                dom = Counter(r["pred"]["dominant_factor"][0] for r in sr)
                seq_len = Counter(len(r["pred"]["state_sequence"]) for r in sr)
                conf = Counter(r["pred"]["confidence"] for r in sr)
                multi = sum(1 for r in sr if len(r["pred"]["state_sequence"]) >= 2)
                crit_null = sum(1 for r in sr if r["pred"]["criterion"] is None)
                print(f"\n  Q7 {seg} (n={len(sr)}):")
                print(f"    dom: {dict(dom)}")
                print(f"    seq_len: {dict(sorted(seq_len.items()))}; multi={multi}/{len(sr)}={multi/len(sr):.0%}")
                print(f"    conf: {dict(conf)}; high% = {conf.get('high',0)/len(sr):.0%}")
                print(f"    criterion null: {crit_null}/{len(sr)} = {crit_null/len(sr):.0%}")
        else:
            dom = Counter(r["pred"]["dominant_factor"][0] for r in srs)
            seq_len = Counter(len(r["pred"]["state_sequence"]) for r in srs)
            conf = Counter(r["pred"]["confidence"] for r in srs)
            print(f"\n  Q1-Q6_expanded (n={len(srs)}):")
            print(f"    dom: {dict(dom)}")
            print(f"    seq_len: {dict(sorted(seq_len.items()))}")
            print(f"    conf: {dict(conf)}")

    # unique users covered
    users_q7 = {r["user_id"] for r in results if r["ok"] and r["source"] == "Q7"}
    users_q16 = {r["user_id"] for r in results if r["ok"] and r["source"] == "Q1-Q6_expanded"}
    print(f"\n=== coverage ===")
    print(f"  Q7 unique users: {len(users_q7)}")
    print(f"  Q1-Q6 unique users: {len(users_q16)}")
    print(f"  union: {len(users_q7 | users_q16)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
