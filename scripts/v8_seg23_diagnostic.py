"""B: v8 extractor on Q7 seg2 + seg3 — 看 label diversity 是否比 seg1 高.

seg1 (第一反应) 经验偏 V (10/10 V in pilot).
seg2 (决策链路) 应有 K2/P3 markers (weighing data + problem reframe).
seg3 (核心 tradeoff) 应有 V3/P 底线 markers.

抽 10 user (从 pilot 一致), 对每 user 跑 seg2 + seg3 共 20 NL.
看 label 分布 (K/P/V/C 占比) + confidence 跟 seg1 比.
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


V8_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": ["dominant_factor", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def load_v8_prompt():
    """Fix: 用 ``` 围栏边界, 不靠 '---' (markdown table 误中)."""
    pt = Path("sft_pipeline/config/prompts/extractor_v8.md").read_text(encoding="utf-8")
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
            "json_schema": {"name": "pkvc_extraction", "schema": V8_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # load 10 users from previous pilot
    pilot_path = Path("reports/datav3/roleplay_pilot/results.jsonl")
    pilot = [json.loads(l) for l in pilot_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    target_uids = [r["user_id"] for r in pilot]

    # load Q7 seg2 + seg3 for those users
    raw_task = loaders.load_task_long(expanded=False)
    q7 = raw_task[raw_task["question_id"] == "Q7"]
    user_segs: dict[str, dict] = {}
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
    print(f"[data] {len(user_segs)} users have Q7 seg1/2/3")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_v8_prompt()

    results = []
    for i, (uid, segs) in enumerate(user_segs.items(), 1):
        for seg_name in ["seg2", "seg3"]:
            nl = segs[seg_name]
            print(f"[{i:2d}] {uid[:8]} {seg_name} ({len(nl)}字)", end=" ", flush=True)
            try:
                res = extract_pkvc(client, "gpt-4o", nl, sys_block, usr_template)
                print(f"→ {res['dominant_factor']} ({res['confidence']}) — {res['extraction_notes'][:40]}")
                results.append({
                    "user_id": uid,
                    "seg": seg_name,
                    "nl_len": len(nl),
                    "predicted": res["dominant_factor"],
                    "head": res["dominant_factor"][0],
                    "confidence": res["confidence"],
                    "notes": res["extraction_notes"],
                })
            except Exception as e:
                print(f"FAIL: {e}")

    if not results:
        return 1

    # summary
    print(f"\n=== RESULTS ({len(results)} extractions) ===\n")
    for seg_name in ["seg2", "seg3"]:
        seg_res = [r for r in results if r["seg"] == seg_name]
        if not seg_res:
            continue
        labels = Counter(r["head"] for r in seg_res)
        confs = Counter(r["confidence"] for r in seg_res)
        avg_len = sum(r["nl_len"] for r in seg_res) / len(seg_res)
        weak = sum(1 for r in seg_res if "信号弱" in r.get("notes", ""))
        print(f"=== {seg_name} ===")
        print(f"  n = {len(seg_res)}, avg len = {avg_len:.0f} 字")
        print(f"  label 分布: {dict(labels)}")
        print(f"  confidence:  {dict(confs)}")
        print(f"  信号弱 notes: {weak}/{len(seg_res)} = {weak/len(seg_res):.0%}\n")

    # 跟 seg1 (来自 roleplay_pilot real_pkvc) 对比
    print("=== 跟 seg1 对比 (来自 roleplay_pilot real_pkvc) ===")
    seg1_labels = Counter(r["real_pkvc"]["dominant_factor"][0] for r in pilot)
    seg1_confs = Counter(r["real_pkvc"]["confidence"] for r in pilot)
    print(f"  seg1 label 分布: {dict(seg1_labels)}  (10/10 都 V, 完全无 diversity)")
    print(f"  seg1 confidence: {dict(seg1_confs)}")

    seg2_labels = Counter(r["head"] for r in results if r["seg"] == "seg2")
    seg3_labels = Counter(r["head"] for r in results if r["seg"] == "seg3")

    # verdict
    print("\n=== VERDICT ===")
    seg2_diverse = len([k for k in seg2_labels if seg2_labels[k] > 0]) >= 2
    seg3_diverse = len([k for k in seg3_labels if seg3_labels[k] > 0]) >= 2
    if seg2_diverse or seg3_diverse:
        print(f"✅ seg2/seg3 比 seg1 更 diverse (seg2 unique labels: {len(seg2_labels)}, seg3: {len(seg3_labels)}, seg1: 1)")
        print(f"   建议: role-play eval 用 seg2 或 seg3 作 target, 不要 seg1")
    else:
        print(f"❌ seg2/seg3 仍 single-label dominant → v8 extractor 真的全 V default 不分输入")

    # save
    out_path = Path("reports/datav3/v8_seg23_diagnostic.json")
    out = {
        "n_users": len(user_segs),
        "results": results,
        "seg2_label_distribution": dict(seg2_labels),
        "seg3_label_distribution": dict(seg3_labels),
        "seg1_label_distribution": dict(seg1_labels),
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
