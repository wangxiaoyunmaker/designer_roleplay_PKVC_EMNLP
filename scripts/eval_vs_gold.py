"""Eval a v8.x prompt against 5-user expert gold (parsed.json).

Run extractor on 5 users × 3 segs = 15 calls, compare to gold per seg:
  - dominant accuracy (predicted_head == gold_dominant)
  - sequence Jaccard (set-overlap)
  - sequence edit distance (Levenshtein, capped at 5)
  - criterion substring match (if gold criterion has verbatim phrase in NL)

Output: reports/datav3/eval_vs_gold_{version}.json + concise stdout summary.

Usage:
  .venv/bin/python scripts/eval_vs_gold.py --prompt sft_pipeline/config/prompts/extractor_v8.2.md --version v8.2
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI


V8_X_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
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
    "required": ["dominant_factor", "state_sequence", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def load_prompt(prompt_path: Path):
    pt = prompt_path.read_text(encoding="utf-8")
    sys_section = pt.find("## System Prompt")
    sys_open = pt.find("```", sys_section)
    sys_close = pt.find("```", sys_open + 3)
    sys_block = pt[sys_open + 3:sys_close].strip()
    usr_section = pt.find("## User Prompt Template")
    usr_open = pt.find("```", usr_section)
    usr_close = pt.find("```", usr_open + 3)
    usr_template = pt[usr_open + 3:usr_close].strip()
    return sys_block, usr_template


def extract(client, model, nl, sys_block, usr_template, segment_role="single"):
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
            "json_schema": {"name": "pkvc_extraction", "schema": V8_X_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(a)][len(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="sft_pipeline/config/prompts/extractor_v8.2.md")
    ap.add_argument("--version", required=True, help="label for output file, e.g. v8.2")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return 1

    gold = json.loads(Path("data/processed/expert_gold/parsed.json").read_text(encoding="utf-8"))
    users = gold["users"]
    print(f"[gold] {len(users)} users × 3 segs = {3 * len(users)} extractions")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_prompt(Path(args.prompt))

    results = []
    for i, u in enumerate(users, 1):
        for seg_name in ("seg1", "seg2", "seg3"):
            g = u[seg_name]
            nl = g["text"]
            print(f"[{i}/{len(users)}] {u['user_id'][:12]} {seg_name} ({len(nl)}字)", flush=True)
            try:
                pred = extract(client, args.model, nl, sys_block, usr_template, segment_role=seg_name)
            except Exception as e:
                print(f"  FAIL: {e}")
                continue
            # diagnostics
            pred_head = pred["dominant_factor"][0]
            gold_dom = g["dominant"] or "?"
            dom_match = (pred_head == gold_dom)
            pred_seq = pred["state_sequence"]
            gold_seq = g["sequence"]
            edit = levenshtein(pred_seq, gold_seq)
            jaccard = (len(set(pred_seq) & set(gold_seq)) / len(set(pred_seq) | set(gold_seq))
                       if (pred_seq or gold_seq) else 1.0)
            crit_match = bool(pred["criterion"]) and pred["criterion"] in nl

            print(f"  pred dom={pred['dominant_factor']:5s}  seq={pred_seq}  gold dom={gold_dom}  gold_seq={gold_seq}")
            print(f"  → dom_match={dom_match}  edit={edit}  jaccard={jaccard:.2f}")

            results.append({
                "user_id": u["user_id"],
                "seg": seg_name,
                "nl_len": len(nl),
                "gold_dominant": gold_dom,
                "gold_sequence": gold_seq,
                "gold_criterion": u.get("criterion", "")[:200],
                "pred_dominant": pred["dominant_factor"],
                "pred_head": pred_head,
                "pred_sequence": pred_seq,
                "pred_criterion": pred["criterion"],
                "pred_confidence": pred["confidence"],
                "pred_notes": pred["extraction_notes"],
                "dom_match": dom_match,
                "seq_edit": edit,
                "seq_jaccard": jaccard,
                "crit_in_nl": crit_match,
                "nl": nl,
            })

    n = len(results)
    if not n:
        return 1

    # aggregate
    dom_acc = sum(r["dom_match"] for r in results) / n
    avg_edit = sum(r["seq_edit"] for r in results) / n
    avg_jacc = sum(r["seq_jaccard"] for r in results) / n
    per_seg = {}
    for seg in ("seg1", "seg2", "seg3"):
        seg_r = [r for r in results if r["seg"] == seg]
        if not seg_r:
            continue
        per_seg[seg] = {
            "n": len(seg_r),
            "dom_acc": sum(r["dom_match"] for r in seg_r) / len(seg_r),
            "avg_edit": sum(r["seq_edit"] for r in seg_r) / len(seg_r),
            "avg_jacc": sum(r["seq_jaccard"] for r in seg_r) / len(seg_r),
        }
    confusion = Counter((r["gold_dominant"], r["pred_head"]) for r in results)

    print(f"\n=== {args.version} vs expert gold (n={n}) ===")
    print(f"  overall dominant accuracy: {dom_acc:.0%}")
    print(f"  overall avg sequence edit distance: {avg_edit:.2f}")
    print(f"  overall avg sequence Jaccard: {avg_jacc:.2f}")
    print(f"  per-seg:")
    for seg, s in per_seg.items():
        print(f"    {seg}: dom_acc={s['dom_acc']:.0%}  avg_edit={s['avg_edit']:.2f}  avg_jacc={s['avg_jacc']:.2f}  (n={s['n']})")
    print(f"  confusion (gold → pred):")
    for (g, p), c in sorted(confusion.items()):
        mark = "✓" if g == p else "✗"
        print(f"    {mark} {g} → {p}: {c}")

    out_path = Path(f"reports/datav3/eval_vs_gold_{args.version}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "version": args.version,
            "prompt": str(args.prompt),
            "n": n,
            "overall_dom_acc": dom_acc,
            "overall_avg_edit": avg_edit,
            "overall_avg_jacc": avg_jacc,
            "per_seg": per_seg,
            "confusion": {f"{g}→{p}": c for (g, p), c in confusion.items()},
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
