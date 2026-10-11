"""Quick chain diversity per inference jsonl (no extract needed).

Reads inference jsonl directly, computes:
  - seg2 chain diversity (1 - modal_chain_freq, 162 designer)
  - seg2 non-V recall (跳出 V-default 比例)
  - seg2 first-letter distribution

Used to quickly compare v3 variants before paying for extract.
"""
import json, sys, glob
from collections import Counter


def stats_for_file(path):
    rows = [json.loads(l) for l in open(path)]
    ok_rows = [r for r in rows if r.get("ok")]

    # seg2 chains (from inference's state_sequence field)
    seg2 = [r for r in ok_rows if r.get("scenario") == "Q7_seg2"]
    chains = [tuple(r.get("state_sequence") or []) for r in seg2]
    chains = [c for c in chains if c]

    # GT seg2 first letter from ground_truth_pkvc
    gt_seg2 = [r.get("ground_truth_pkvc", {}).get("dominant", "") for r in seg2]
    pred_seg2_first = [c[0] if c else "" for c in chains]

    # Chain diversity = 1 - modal chain freq
    cnt = Counter(chains)
    chain_div = 1 - cnt.most_common(1)[0][1] / len(chains) if chains else 0

    # non-V recall: among GT non-V chains, how many pred match non-V starting
    non_v_gt_idx = [i for i, g in enumerate(gt_seg2) if g and g != "V"]
    if non_v_gt_idx and len(pred_seg2_first) >= max(non_v_gt_idx) + 1:
        non_v_hit = sum(
            1 for i in non_v_gt_idx
            if i < len(pred_seg2_first) and pred_seg2_first[i] == gt_seg2[i]
        )
        non_v_recall = non_v_hit / len(non_v_gt_idx)
    else:
        non_v_recall = 0

    # First-letter distribution
    first_dist = Counter(pred_seg2_first)

    # Top-3 modal chains
    top3_chains = cnt.most_common(3)

    return {
        "file": path.split("/")[-1],
        "n_seg2_ok": len(seg2),
        "n_chains_with_state": len(chains),
        "chain_diversity": round(chain_div, 3),
        "non_v_recall": round(non_v_recall, 3),
        "first_letter_dist": dict(first_dist),
        "top3_modal_chains": [(list(c), n) for c, n in top3_chains],
    }


def main():
    patterns = sys.argv[1:] if len(sys.argv) > 1 else [
        "reports/datav3/sft_v3*_inference.jsonl"
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    files = list(dict.fromkeys(files))
    print(f"{'file':45s} | seg2_ok | chain_div | non_V | top1_chain (freq)")
    print("-" * 110)
    for path in files:
        s = stats_for_file(path)
        top1 = s["top3_modal_chains"][0] if s["top3_modal_chains"] else (None, 0)
        top1_str = f"{top1[0]} ({top1[1]})"
        print(f"{s['file']:45s} | {s['n_seg2_ok']:7d} | {s['chain_diversity']:.3f}    | {s['non_v_recall']:.3f} | {top1_str}")
    print()
    print("Full details:")
    for path in files:
        s = stats_for_file(path)
        print(f"  {s['file']}: first_dist={s['first_letter_dist']}, top3={s['top3_modal_chains'][:3]}")


if __name__ == "__main__":
    main()
