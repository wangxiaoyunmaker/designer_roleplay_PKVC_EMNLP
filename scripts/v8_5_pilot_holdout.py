"""v8.5 sanity check on 10-user × 3-seg pilot (holdout, no gold).

跟 v8.2 同 10 user 跑 v8.5, 看 label 分布是否仍合理 (不全 V vacuous), confidence 分布,
state_sequence 长度分布. 这不是 accuracy 测试, 是 sanity check.
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI
from sft_pipeline.eval.datav3_stats import loaders

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


def load_v8_5_prompt():
    pt = Path("sft_pipeline/config/prompts/extractor_v8.5.md").read_text(encoding="utf-8")
    sys_section = pt.find("## System Prompt")
    sys_open = pt.find("```", sys_section)
    sys_close = pt.find("```", sys_open + 3)
    sys_block = pt[sys_open + 3:sys_close].strip()
    usr_section = pt.find("## User Prompt Template")
    usr_open = pt.find("```", usr_section)
    usr_close = pt.find("```", usr_open + 3)
    usr_template = pt[usr_open + 3:usr_close].strip()
    return sys_block, usr_template


def extract(client, model, nl, sys_block, usr_template, segment_role):
    msg = usr_template.replace("{INPUT_TEXT}", nl).replace("{SEGMENT_ROLE}", segment_role)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_block}, {"role": "user", "content": msg}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pkvc_extraction", "schema": V8_5_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    pilot = [json.loads(l) for l in Path("reports/datav3/roleplay_pilot/results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    target_uids = [r["user_id"] for r in pilot]
    raw_task = loaders.load_task_long(expanded=False)
    q7 = raw_task[raw_task["question_id"] == "Q7"]
    user_segs = {}
    for _, r in q7.iterrows():
        if r["user_id"] not in target_uids:
            continue
        try:
            blob = json.loads(r["answer_value"])
            user_segs[r["user_id"]] = {k: blob[k]["text"] for k in ("seg1","seg2","seg3")}
        except Exception:
            continue
    print(f"[data] {len(user_segs)} users, {3*len(user_segs)} segs")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_b, usr_t = load_v8_5_prompt()

    results = []
    for i, (uid, segs) in enumerate(user_segs.items(), 1):
        for seg in ("seg1","seg2","seg3"):
            nl = segs[seg]
            print(f"[{i:2d}/10] {uid[:10]} {seg} ({len(nl)}字)", end=" ", flush=True)
            try:
                pred = extract(client, "gpt-4o", nl, sys_b, usr_t, seg)
                print(f"dom={pred['dominant_factor']:5s} seq={pred['state_sequence']} conf={pred['confidence']}")
                results.append({"user_id": uid, "seg": seg, "nl_len": len(nl), **pred})
            except Exception as e:
                print(f"FAIL {e}")

    # summary
    print(f"\n=== v8.5 sanity (n={len(results)}) ===")
    for seg in ("seg1","seg2","seg3"):
        sr = [r for r in results if r["seg"] == seg]
        if not sr: continue
        dom = Counter(r["dominant_factor"][0] for r in sr)
        seq_len = Counter(len(r["state_sequence"]) for r in sr)
        seq_letters = Counter(c for r in sr for c in r["state_sequence"])
        conf = Counter(r["confidence"] for r in sr)
        multi = sum(1 for r in sr if len(r["state_sequence"]) >= 2)
        print(f"\n=== {seg} (n={len(sr)}) ===")
        print(f"  dominant head 分布: {dict(dom)}")
        print(f"  sequence 长度分布: {dict(sorted(seq_len.items()))}; multi-state {multi}/{len(sr)} = {multi/len(sr):.0%}")
        print(f"  sequence 字母分布: {dict(seq_letters)}")
        print(f"  confidence: {dict(conf)}")

    out = Path("reports/datav3/v8_5_pilot_holdout.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
