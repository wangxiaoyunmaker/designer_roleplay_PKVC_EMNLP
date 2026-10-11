"""Q2 v8.7 ranking extraction eval.

用 v8.7 extractor 从 refined Q2 NL 抽 state_sequence (factor-level PKVC 时序),
跟 user 真实 ranking (cardId order) factor-level 化后比相似度.

target: 80% top-3 factor match (5 个用户里 4 个 top-3 一致).

run:
    .venv/bin/python scripts/q2_v8_7_ranking_eval.py \
        --refined-jsonl data/processed/q2_nl_refined.jsonl \
        --output data/processed/q2_v8_7_ranking_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI


V8_7_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1,
            "maxItems": 7,
        },
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": [
        "dominant_factor", "state_sequence", "criterion", "confidence", "extraction_notes",
    ],
    "additionalProperties": False,
}


def load_v8_7_prompt() -> tuple[str, str]:
    """Parse extractor_v8.7.md → (system_prompt, user_template)."""
    pt = Path("sft_pipeline/config/prompts/extractor_v8.7.md").read_text(encoding="utf-8")
    sys_start = pt.find("## System Prompt")
    sys_fence_open = pt.find("```", sys_start)
    sys_fence_close = pt.find("```", sys_fence_open + 3)
    sys_block = pt[sys_fence_open + 3 : sys_fence_close].strip()

    usr_start = pt.find("## User Prompt Template")
    usr_fence_open = pt.find("```", usr_start)
    usr_fence_close = pt.find("```", usr_fence_open + 3)
    usr_template = pt[usr_fence_open + 3 : usr_fence_close].strip()
    return sys_block, usr_template


def extract_pkvc_sequence(
    client: OpenAI,
    model: str,
    nl: str,
    sys_block: str,
    usr_template: str,
    segment_role: str = "single",
) -> dict:
    user_msg = usr_template.replace("{INPUT_TEXT}", nl).replace("{SEGMENT_ROLE}", segment_role)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_block},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pkvc_extraction", "schema": V8_7_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def cardid_ranking_to_factor_sequence(cardids: list[str]) -> list[str]:
    """[P1, V3, K2, C2, P2, K1, ...] → factor-level sequence by first occurrence.

    保留每个 factor 第一次出现的相对顺序; 重复 factor 跳过 (因为已经记过).
    e.g. [P1, V3, K2, C2, P2, K1] → [P, V, K, C] (each factor first occurrence rank)
    """
    seen = set()
    seq = []
    for cid in cardids:
        if not cid:
            continue
        f = cid[0].upper()
        if f not in ("P", "K", "V", "C"):
            continue
        if f not in seen:
            seq.append(f)
            seen.add(f)
    return seq


def compute_similarity(extracted_seq: list[str], real_seq: list[str]) -> dict:
    """跟 user 真实 factor sequence 比, 多 metric."""
    # 1. top-1 match (用 dominant / 第一个 state)
    top1_match = extracted_seq[0] == real_seq[0] if extracted_seq and real_seq else False

    # 2. top-3 set match (factor 集合, ignore 顺序)
    top3_extract = set(extracted_seq[:3])
    top3_real = set(real_seq[:3])
    top3_overlap = len(top3_extract & top3_real)
    top3_match_pct = top3_overlap / max(len(top3_real), 1)

    # 3. top-3 ordered match (Kendall-like: 是否完全同序)
    top3_ordered_match = extracted_seq[:3] == real_seq[:3]

    # 4. set similarity (全 factor 集合 Jaccard)
    ext_set, real_set = set(extracted_seq), set(real_seq)
    if not ext_set and not real_set:
        jaccard = 0.0
    else:
        jaccard = len(ext_set & real_set) / max(len(ext_set | real_set), 1)

    # 5. order-aware: kendall tau on common factors
    common = [f for f in real_seq if f in ext_set]
    if len(common) >= 2:
        ext_positions = {f: i for i, f in enumerate(extracted_seq) if f in common}
        real_positions = {f: i for i, f in enumerate(real_seq) if f in common}
        concordant = 0
        discordant = 0
        for i, fi in enumerate(common):
            for fj in common[i + 1:]:
                if (ext_positions[fi] < ext_positions[fj]) == (real_positions[fi] < real_positions[fj]):
                    concordant += 1
                else:
                    discordant += 1
        kendall = (concordant - discordant) / max(concordant + discordant, 1)
    else:
        kendall = float("nan")

    return {
        "top1_match": bool(top1_match),
        "top3_overlap_count": top3_overlap,
        "top3_match_pct": float(top3_match_pct),
        "top3_ordered_match": bool(top3_ordered_match),
        "jaccard": float(jaccard),
        "kendall_tau": float(kendall) if kendall == kendall else None,  # NaN handling
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined-jsonl", default="data/processed/q2_nl_refined.jsonl")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--segment-role", default="single",
                    help="seg1 / seg2 / seg3 / single (single = 非 Q7 chain)")
    ap.add_argument("--output", default="data/processed/q2_v8_7_ranking_eval.json")
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    refined_path = Path(args.refined_jsonl)
    if not refined_path.exists():
        print(f"ERROR: {refined_path} not found", file=sys.stderr)
        return 1

    refined = [json.loads(l) for l in refined_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[data] loaded {len(refined)} refined Q2 NLs")
    if args.max_users > 0:
        refined = refined[: args.max_users]
        print(f"[run] capping to {len(refined)} users")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_v8_7_prompt()
    print(f"[v8.7] system {len(sys_block)} chars, user template {len(usr_template)} chars\n")

    results = []
    t0 = time.time()
    for i, r in enumerate(refined, 1):
        uid = r["user_id"]
        refined_nl = r["refined_nl"]
        real_cardids = r["ranking_cardids"]
        real_factor_seq = cardid_ranking_to_factor_sequence(real_cardids)

        try:
            extracted = extract_pkvc_sequence(client, args.model, refined_nl, sys_block, usr_template,
                                              segment_role=args.segment_role)
        except Exception as e:
            print(f"[{i:3d}/{len(refined)}] {uid[:8]} FAIL: {e}")
            continue

        sim = compute_similarity(extracted["state_sequence"], real_factor_seq)
        flag = "✓" if sim["top1_match"] else "✗"
        flag3 = f"{sim['top3_overlap_count']}/3"
        print(
            f"[{i:3d}/{len(refined)}] {uid[:8]} {flag} top1 (real={real_factor_seq[0]} ext={extracted['state_sequence'][0]}) | "
            f"top3 {flag3} ({sim['top3_match_pct']:.0%}) | "
            f"real_seq={real_factor_seq} ext_seq={extracted['state_sequence']} | "
            f"elapsed={time.time()-t0:.0f}s"
        )

        results.append({
            "user_id": uid,
            "real_cardids": real_cardids,
            "real_factor_seq": real_factor_seq,
            "extracted_dominant": extracted["dominant_factor"],
            "extracted_state_sequence": extracted["state_sequence"],
            "extracted_confidence": extracted["confidence"],
            "extracted_notes": extracted["extraction_notes"],
            "similarity": sim,
        })

        # incremental save
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"n": len(results), "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 汇总
    print(f"\n=== SUMMARY ({len(results)} users) ===\n")
    if not results:
        print("no results")
        return 1

    n = len(results)
    top1_acc = sum(1 for r in results if r["similarity"]["top1_match"]) / n
    top3_avg = sum(r["similarity"]["top3_match_pct"] for r in results) / n
    top3_ordered_acc = sum(1 for r in results if r["similarity"]["top3_ordered_match"]) / n
    jaccard_avg = sum(r["similarity"]["jaccard"] for r in results) / n
    kendalls = [r["similarity"]["kendall_tau"] for r in results if r["similarity"]["kendall_tau"] is not None]
    kendall_avg = sum(kendalls) / max(len(kendalls), 1)

    print(f"top-1 dominant 一致率 (4-way acc, random=25%):   {top1_acc:.1%}")
    print(f"top-3 set overlap avg (3/3=100%):                {top3_avg:.1%}")
    print(f"top-3 ordered exact match:                       {top3_ordered_acc:.1%}")
    print(f"全 factor jaccard avg:                           {jaccard_avg:.1%}")
    print(f"Kendall tau (order on common factors) avg:       {kendall_avg:+.3f}")

    # confusion matrix top-1 (extracted dominant 第一字母 vs real top-1)
    print("\nConfusion matrix (rows = real top-1, cols = extracted top-1 head):")
    classes = ["P", "K", "V", "C"]
    cm = {gt: {p: 0 for p in classes} for gt in classes}
    for r in results:
        gt = r["real_factor_seq"][0]
        ext = r["extracted_state_sequence"][0]
        if gt in classes and ext in classes:
            cm[gt][ext] += 1
    print(f"  GT\\Pred |  P  |  K  |  V  |  C  | total")
    for gt in classes:
        row = [str(cm[gt][p]).rjust(3) for p in classes]
        total = sum(cm[gt].values())
        correct = cm[gt].get(gt, 0)
        print(f"  {gt}        | {' | '.join(row)} | {correct}/{total}={correct/max(total,1):.0%}")

    print(f"\n=== VERDICT (target 80% top-3 overlap) ===")
    if top3_avg >= 0.8:
        print(f"✅ top-3 overlap {top3_avg:.1%} ≥ 80% target — v8.7 extractor 能 recover Q2 ranking")
    elif top3_avg >= 0.6:
        print(f"⚠ top-3 overlap {top3_avg:.1%} ∈ [60%, 80%) — 接近 target, 可考虑优化 prompt or refined NL")
    else:
        print(f"❌ top-3 overlap {top3_avg:.1%} < 60% — 大差距, refined NL 跟 ranking 不够 aligned, 或 v8.7 不适用 Q2-style 排序题")

    summary = {
        "n_users": n,
        "top1_dominant_acc": top1_acc,
        "top3_set_overlap_avg": top3_avg,
        "top3_ordered_match_acc": top3_ordered_acc,
        "jaccard_avg": jaccard_avg,
        "kendall_tau_avg": kendall_avg,
        "confusion_matrix": cm,
        "n_results": n,
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] full results: {args.output}")
    print(f"[saved] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
