"""Sample 200 segments from Q7 (150) + Q1-Q6 expanded (50) for mid-scale validation.

- Q7: 234 segs (78 users × 3). Exclude 5 gold users → 73 users × 3 = 219 candidates.
  Sample 50 users → 150 segs. Each user contributes seg1/seg2/seg3 (positional).
- Q1-Q6 expanded: 171 entries. Sample 50 deterministically.
- Output: data/processed/extractor_val_200.jsonl
"""
from __future__ import annotations
import json
import random
from pathlib import Path

GOLD_USERS = {"CYXcyx289", "BuzhichunD", "wtimesc", "h18717957422", "Yyd3144117584"}
SEED = 42


def main():
    random.seed(SEED)

    # ---- Q7 ----
    q7 = [json.loads(l) for l in Path("data/processed/q7_segments.jsonl").open()]
    by_user = {}
    for e in q7:
        by_user.setdefault(e["user_id"], []).append(e)
    # exclude gold users
    candidates = [u for u in by_user if u not in GOLD_USERS]
    random.shuffle(candidates)
    pick_users = candidates[:50]
    q7_picked = []
    for u in pick_users:
        segs = by_user[u]
        # assume positional seg1/seg2/seg3
        for i, e in enumerate(segs, 1):
            q7_picked.append({
                "id": e["id"],
                "user_id": u,
                "qid": "Q7",
                "segment_role": f"seg{i}",
                "text": e["text"],
                "source": "Q7",
            })
    print(f"Q7 sampled: {len(q7_picked)} (50 users × 3 seg)")

    # ---- Q1-Q6 expanded ----
    raw = [json.loads(l) for l in Path("data/processed/q1q6_expansion_outputs.jsonl").open()]
    candidates_q16 = []
    for r in raw:
        # extract expanded text + inferred factor
        result = r.get("result")
        if not isinstance(result, dict):
            continue
        text = result.get("expanded_nl", "").strip()
        if len(text) < 30:
            continue
        candidates_q16.append({
            "id": r.get("candidate_id") or f"{r['user_id']}_{r['question_id']}",
            "user_id": r["user_id"],
            "qid": r["question_id"],
            "segment_role": "single",
            "text": text,
            "source": "Q1-Q6_expanded",
            "expansion_inferred_factor": result.get("inferred_factor"),
            "expansion_inferred_subtype": result.get("inferred_subtype"),
            "expansion_confidence": result.get("confidence"),
        })
    print(f"Q1-Q6 expanded usable: {len(candidates_q16)}")
    random.shuffle(candidates_q16)
    q16_picked = candidates_q16[:50]

    all_picked = q7_picked + q16_picked
    print(f"Total: {len(all_picked)} = 150 Q7 + 50 Q1-Q6")

    out = Path("data/processed/extractor_val_200.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for e in all_picked:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[saved] {out}")

    # length stats
    lens = [len(e["text"]) for e in all_picked]
    print(f"\ntext length: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)} avg={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
