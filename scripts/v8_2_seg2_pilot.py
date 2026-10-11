"""v8.2 extractor pilot — 验证 state_sequence 字段是否真在 seg2 出 N-state chain.

10 user × {seg1, seg2, seg3} = 30 段 (跟 v8_seg23_diagnostic 1:1 可比).
跑 v8.2 prompt + GPT-4o + json_schema strict.

看:
  - state_sequence 长度分布 (期望 seg2 >> seg1/seg3 in multi-state %)
  - sequence PKVC 字母分布 (脱出全 V default)
  - dominant_factor 跟 sequence 的关系一致性
  - schema enforce 是否真起作用
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


V8_2_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {
            "type": "string",
            "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$",
        },
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1,
            "maxItems": 5,
        },
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": [
        "dominant_factor",
        "state_sequence",
        "criterion",
        "confidence",
        "extraction_notes",
    ],
    "additionalProperties": False,
}


def load_v8_2_prompt():
    pt = Path("sft_pipeline/config/prompts/extractor_v8.2.md").read_text(encoding="utf-8")
    sys_section_start = pt.find("## System Prompt")
    sys_fence_open = pt.find("```", sys_section_start)
    sys_fence_close = pt.find("```", sys_fence_open + 3)
    sys_block = pt[sys_fence_open + 3 : sys_fence_close].strip()
    usr_section_start = pt.find("## User Prompt Template")
    usr_fence_open = pt.find("```", usr_section_start)
    usr_fence_close = pt.find("```", usr_fence_open + 3)
    usr_template = pt[usr_fence_open + 3 : usr_fence_close].strip()
    return sys_block, usr_template


def extract_pkvc(client, model, nl, sys_block, usr_template):
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_block},
            {"role": "user", "content": usr_template.replace("{INPUT_TEXT}", nl)},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pkvc_extraction_v8_2",
                "schema": V8_2_SCHEMA,
                "strict": True,
            },
        },
    )
    return json.loads(r.choices[0].message.content)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY (yunwu.ai)")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # 同 v8_seg23_diagnostic 的 10 user, seg1 + seg2 + seg3 全跑
    pilot_path = Path("reports/datav3/roleplay_pilot/results.jsonl")
    pilot = [json.loads(l) for l in pilot_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    target_uids = [r["user_id"] for r in pilot]  # 10 users

    raw_task = loaders.load_task_long(expanded=False)
    q7 = raw_task[raw_task["question_id"] == "Q7"]
    user_segs = {}
    for _, r in q7.iterrows():
        if r["user_id"] not in target_uids:
            continue
        try:
            blob = json.loads(r["answer_value"])
            user_segs[r["user_id"]] = {
                "seg1": blob["seg1"]["text"],
                "seg2": blob["seg2"]["text"],
                "seg3": blob["seg3"]["text"],
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    print(f"[data] {len(user_segs)} users have Q7 seg1/2/3 ({3*len(user_segs)} segments to extract)")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_v8_2_prompt()

    results = []
    for i, (uid, segs) in enumerate(user_segs.items(), 1):
        for seg_name in ("seg1", "seg2", "seg3"):
            nl = segs[seg_name]
            print(f"[{i:2d}/{len(user_segs)}] {uid[:10]} {seg_name} ({len(nl)}字)", end=" ", flush=True)
            try:
                res = extract_pkvc(client, "gpt-4o", nl, sys_block, usr_template)
                seq_str = "→".join(res["state_sequence"])
                print(f"dom={res['dominant_factor']:5s}  seq=[{seq_str}]  conf={res['confidence']}")
                results.append({"user_id": uid, "seg": seg_name, "nl_len": len(nl), "nl": nl, **res})
            except Exception as e:
                print(f"FAIL: {e}")

    if not results:
        return 1

    print(f"\n=== summary (n={len(results)}) ===\n")
    for seg_name in ("seg1", "seg2", "seg3"):
        seg_res = [r for r in results if r["seg"] == seg_name]
        if not seg_res:
            continue
        seq_lens = Counter(len(r["state_sequence"]) for r in seg_res)
        dom_dist = Counter(r["dominant_factor"] for r in seg_res)
        seq_letters = Counter(c for r in seg_res for c in r["state_sequence"])
        confs = Counter(r["confidence"] for r in seg_res)
        multi = sum(1 for r in seg_res if len(r["state_sequence"]) >= 2)
        print(f"=== {seg_name} (n={len(seg_res)}) ===")
        print(f"  sequence 长度分布: {dict(sorted(seq_lens.items()))}")
        print(f"  多 state (≥2) 比例: {multi}/{len(seg_res)} = {multi/len(seg_res):.0%}")
        print(f"  dominant_factor 分布: {dict(dom_dist)}")
        print(f"  sequence 字母总分布 (n={sum(seq_letters.values())}): {dict(seq_letters)}")
        print(f"  confidence: {dict(confs)}\n")

    # cross-seg verdict
    seg1_multi = sum(1 for r in results if r["seg"] == "seg1" and len(r["state_sequence"]) >= 2)
    seg2_multi = sum(1 for r in results if r["seg"] == "seg2" and len(r["state_sequence"]) >= 2)
    seg3_multi = sum(1 for r in results if r["seg"] == "seg3" and len(r["state_sequence"]) >= 2)
    print("=== VERDICT ===")
    print(f"  seg1 multi-state %: {seg1_multi/10:.0%}  (期望低, seg1 是第一反应)")
    print(f"  seg2 multi-state %: {seg2_multi/10:.0%}  (期望高 ≥50%, seg2 是决策链路)")
    print(f"  seg3 multi-state %: {seg3_multi/10:.0%}  (期望中, seg3 是底线/折中)")
    if seg2_multi >= 5:
        print(f"  ✅ seg2 multi-state ≥ 50%: state_sequence 字段 deliver — N-state chain 真出来了")
    else:
        print(f"  ❌ seg2 multi-state < 50%: state_sequence 字段 under-trigger — prompt 还要加强")

    out_path = Path("reports/datav3/v8_2_seg123_pilot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"n_users": len(user_segs), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
