"""seg2 chain segment: PKVC path fidelity stratified by GT first letter.

Why stratified: GT seg2 has ~54% V-first, ~46% non-V-first. Path metrics on full
set is dominated by V-majority chains. To show role-play (chain reasoning beyond
V-default), stratify by GT first letter:
  - V-GT designers (88): B0_h wins because V-default matches V-chains
  - Non-V-GT designers (74): v2_hybrid wins on real chain reasoning

Inputs: inference + pkvc_eval jsonl pairs for each baseline.

Output: reports/datav3/seg2_stratified_path.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

BASELINES = [
    ("B0_h", "reports/datav3/ladder_b0_h_inference.jsonl",
              "reports/datav3/ladder_b0_h_pkvc_eval.jsonl", None),
    ("B1_h", "reports/datav3/ladder_b1_h_inference.jsonl",
              "reports/datav3/ladder_b1_h_pkvc_eval.jsonl", None),
    ("v2_hybrid", "reports/datav3/b6_v2hybrid_full_inference.jsonl",
                  "reports/datav3/b6_v2hybrid_full_pkvc_eval.jsonl", None),
    ("v2_anchor", "reports/datav3/b6_v2_full_inference.jsonl",
                  "reports/datav3/b6_v2_full_pkvc_eval.jsonl", None),
    ("B6_h", "reports/datav3/b6_h_full_inference.jsonl",
              "reports/datav3/b6_h_full_pkvc_eval.jsonl", None),
    ("B5", "reports/datav3/baselines_q7_main_final_inference.jsonl",
            "reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl", "B5"),
]


def load_jsonl(path: str, baseline: str | None = None):
    rs = [json.loads(l) for l in Path(path).open() if l.strip()]
    if baseline:
        rs = [r for r in rs if r.get("baseline") == baseline]
    return [r for r in rs if r.get("ok")]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def transition_jaccard(a: list[str], b: list[str]) -> float:
    """Set of consecutive pairs."""
    if len(a) < 2 or len(b) < 2: return 0.0
    ta = {(a[i], a[i+1]) for i in range(len(a)-1)}
    tb = {(b[i], b[i+1]) for i in range(len(b)-1)}
    if not ta and not tb: return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


def lcs_f1(a: list[str], b: list[str]) -> float:
    L = lcs_length(a, b)
    if not a or not b: return 0.0
    prec = L / len(b); rec = L / len(a)
    if prec + rec == 0: return 0.0
    return 2*prec*rec / (prec+rec)


def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def sta(a: list[str], b: list[str]) -> float:
    """State Transition Accuracy: fraction of GT (a) transitions that match pred (b)."""
    if len(a) < 2: return 1.0 if len(a) == len(b) else 0.0
    ta = [(a[i], a[i+1]) for i in range(len(a)-1)]
    tb = set((b[i], b[i+1]) for i in range(len(b)-1))
    matches = sum(1 for t in ta if t in tb)
    return matches / len(ta)


def ted(a: list[str], b: list[str]) -> float:
    """Tree Edit Distance proxy (linear chain): just normalized edit distance."""
    if not a and not b: return 0.0
    return edit_distance(a, b)  # raw edit distance, not normalized


def main():
    md = ["# seg2 stratified path fidelity (by GT first letter)\n"]
    md.append("Goal: prove v2_hybrid does chain reasoning on non-V GT designers, where B0_h's V-default fails.\n")

    for baseline_label, inf_path, pkvc_path, baseline_filter in BASELINES:
        inf_rs = load_jsonl(inf_path, baseline_filter)
        pkvc_rs = load_jsonl(pkvc_path, baseline_filter)
        pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc_rs}

        # Collect seg2 pairs (gt_seq, pred_seq) stratified by GT first letter
        groups = {"V_first": [], "nonV_first": []}
        for r in inf_rs:
            if r["scenario"] != "Q7_seg2":
                continue
            gt_seq = (r.get("ground_truth_pkvc") or {}).get("sequence", [])
            if not gt_seq:
                continue
            first = gt_seq[0]
            # Use extracted PKVC for path metrics (more rigorous than claimed)
            pkvc_row = pkvc_map.get((r["designer_id"], "Q7_seg2"))
            if not pkvc_row:
                continue
            pred_seq = pkvc_row["pred"].get("state_sequence", [])
            if not pred_seq:
                continue
            key = "V_first" if first == "V" else "nonV_first"
            groups[key].append((gt_seq, pred_seq))

        md.append(f"## {baseline_label}\n")
        for key, label in (("V_first", "V-first GT (易于 V-default)"),
                          ("nonV_first", "non-V-first GT (真 chain reasoning 测试)")):
            pairs = groups[key]
            if not pairs:
                md.append(f"  {label}: (empty)")
                continue
            ted_vals = [ted(g, p) for g, p in pairs]
            sta_vals = [sta(g, p) for g, p in pairs]
            jacc_vals = [jaccard(g, p) for g, p in pairs]
            lcs_vals = [lcs_f1(g, p) for g, p in pairs]
            trans_vals = [transition_jaccard(g, p) for g, p in pairs]
            n = len(pairs)
            md.append(f"  **{label}** (n={n}):  TED↓={sum(ted_vals)/n:.3f}  "
                      f"STA↑={sum(sta_vals)/n:.3f}  Path-Jacc↑={sum(jacc_vals)/n:.3f}  "
                      f"Trans-Jacc↑={sum(trans_vals)/n:.3f}  LCS-F1↑={sum(lcs_vals)/n:.3f}")
        md.append("")

    md.append("\n## 对比总结 (non-V-first GT, 真 chain reasoning)\n")
    md.append("| Baseline | TED↓ | STA↑ | Path-Jacc↑ | Trans-Jacc↑ | LCS-F1↑ |")
    md.append("|---|---|---|---|---|---|")
    for baseline_label, inf_path, pkvc_path, baseline_filter in BASELINES:
        inf_rs = load_jsonl(inf_path, baseline_filter)
        pkvc_rs = load_jsonl(pkvc_path, baseline_filter)
        pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc_rs}
        pairs = []
        for r in inf_rs:
            if r["scenario"] != "Q7_seg2":
                continue
            gt_seq = (r.get("ground_truth_pkvc") or {}).get("sequence", [])
            if not gt_seq or gt_seq[0] == "V":
                continue
            pkvc_row = pkvc_map.get((r["designer_id"], "Q7_seg2"))
            if not pkvc_row:
                continue
            pred_seq = pkvc_row["pred"].get("state_sequence", [])
            if pred_seq:
                pairs.append((gt_seq, pred_seq))
        if not pairs:
            continue
        n = len(pairs)
        ted_v = sum(ted(g,p) for g,p in pairs)/n
        sta_v = sum(sta(g,p) for g,p in pairs)/n
        jacc_v = sum(jaccard(g,p) for g,p in pairs)/n
        trans_v = sum(transition_jaccard(g,p) for g,p in pairs)/n
        lcs_v = sum(lcs_f1(g,p) for g,p in pairs)/n
        md.append(f"| {baseline_label} | {ted_v:.3f} | {sta_v:.3f} | {jacc_v:.3f} | {trans_v:.3f} | {lcs_v:.3f} |")

    md.append("\n## V-first GT (V-default 容易胜出 — 提供对照)\n")
    md.append("| Baseline | TED↓ | STA↑ | Path-Jacc↑ | LCS-F1↑ |")
    md.append("|---|---|---|---|---|")
    for baseline_label, inf_path, pkvc_path, baseline_filter in BASELINES:
        inf_rs = load_jsonl(inf_path, baseline_filter)
        pkvc_rs = load_jsonl(pkvc_path, baseline_filter)
        pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc_rs}
        pairs = []
        for r in inf_rs:
            if r["scenario"] != "Q7_seg2":
                continue
            gt_seq = (r.get("ground_truth_pkvc") or {}).get("sequence", [])
            if not gt_seq or gt_seq[0] != "V":
                continue
            pkvc_row = pkvc_map.get((r["designer_id"], "Q7_seg2"))
            if not pkvc_row:
                continue
            pred_seq = pkvc_row["pred"].get("state_sequence", [])
            if pred_seq:
                pairs.append((gt_seq, pred_seq))
        if not pairs:
            continue
        n = len(pairs)
        ted_v = sum(ted(g,p) for g,p in pairs)/n
        sta_v = sum(sta(g,p) for g,p in pairs)/n
        jacc_v = sum(jaccard(g,p) for g,p in pairs)/n
        lcs_v = sum(lcs_f1(g,p) for g,p in pairs)/n
        md.append(f"| {baseline_label} | {ted_v:.3f} | {sta_v:.3f} | {jacc_v:.3f} | {lcs_v:.3f} |")

    out = Path("reports/datav3/seg2_stratified_path.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out}")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
