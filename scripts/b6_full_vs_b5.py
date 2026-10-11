"""Compare B5 vs B6_v1 vs B6_v2 on full 162 designer.

Inputs:
  reports/datav3/baselines_q7_main_final_inference.jsonl  (B5)
  reports/datav3/b6_v1_full_inference.jsonl               (B6_v1)
  reports/datav3/b6_v2_full_inference.jsonl               (B6_v2)

Output: reports/datav3/b6_full_vs_b5.md with macro-F1, per-class P/R/F1, dom acc.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CLASSES = ["P", "K", "V", "C"]


def prf(pairs, cls):
    tp = sum(1 for g, p in pairs if g == cls and p == cls)
    fp = sum(1 for g, p in pairs if g != cls and p == cls)
    fn = sum(1 for g, p in pairs if g == cls and p != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp + fn


def macro(pairs):
    per = {c: prf(pairs, c) for c in CLASSES}
    sup = [c for c in CLASSES if per[c][3] > 0]
    if not sup:
        return 0.0, 0.0, per
    mf1 = sum(per[c][2] for c in sup) / len(sup)
    mr = sum(per[c][1] for c in sup) / len(sup)
    return mf1, mr, per


def load(path, baseline=None):
    rs = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    out = []
    for r in rs:
        if baseline and r.get("baseline") != baseline:
            continue
        if not r.get("ok"):
            continue
        gt = (r.get("ground_truth_pkvc") or {}).get("dominant", "")
        cl = (r.get("claimed_pkvc") or {}).get("dominant_factor", "")
        if not gt or not cl:
            continue
        out.append((r["scenario"], gt[0], cl[0], r["designer_id"]))
    return out


def common_designers(*lists):
    """Return set of designer_ids present in ALL lists with all 3 segs."""
    sets = []
    for rows in lists:
        by_d = {}
        for s, g, p, d in rows:
            by_d.setdefault(d, set()).add(s)
        sets.append({d for d, segs in by_d.items() if len(segs) >= 3})
    return set.intersection(*sets) if sets else set()


def main():
    b5 = load("reports/datav3/baselines_q7_main_final_inference.jsonl", baseline="B5")
    v1 = load("reports/datav3/b6_v1_full_inference.jsonl")  # auto-detect baseline label
    v2 = load("reports/datav3/b6_v2_full_inference.jsonl")
    print(f"B5 rows: {len(b5)}, v1 rows: {len(v1)}, v2 rows: {len(v2)}")

    common = common_designers(b5, v1, v2)
    print(f"common designers (all 3 segs in all 3 baselines): {len(common)}")

    # Filter all to common designers for fair comparison
    b5 = [t for t in b5 if t[3] in common]
    v1 = [t for t in v1 if t[3] in common]
    v2 = [t for t in v2 if t[3] in common]

    md = ["# B5 vs B6_v1 vs B6_v2 — Full 162 designer comparison\n"]
    md.append(f"Common designer pool: {len(common)} (all 3 segs each).\n")

    md.append("## Overall macro-F1 (all 3 segs combined)\n")
    md.append("| Baseline | Macro-F1 | Macro-Recall | dom acc |")
    md.append("|---|---|---|---|")
    for label, rows in (("B5", b5), ("B6_v1", v1), ("B6_v2", v2)):
        pairs = [(g, p) for _, g, p, _ in rows]
        mf, mr, _ = macro(pairs)
        acc = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
        md.append(f"| {label} | {mf:.3f} | {mr:.3f} | {acc:.1%} |")
    md.append("")

    md.append("## Per-seg macro-F1\n")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        md.append(f"### {sc}\n")
        md.append("| Baseline | Macro-F1 | Macro-Recall | dom acc |")
        md.append("|---|---|---|---|")
        for label, rows in (("B5", b5), ("B6_v1", v1), ("B6_v2", v2)):
            pairs = [(g, p) for s, g, p, _ in rows if s == sc]
            if not pairs:
                continue
            mf, mr, _ = macro(pairs)
            acc = sum(1 for g, p in pairs if g == p) / len(pairs)
            md.append(f"| {label} | {mf:.3f} | {mr:.3f} | {acc:.1%} |")
        md.append("")

    md.append("## seg3 per-class P/R/F1\n")
    for label, rows in (("B5", b5), ("B6_v1", v1), ("B6_v2", v2)):
        pairs = [(g, p) for s, g, p, _ in rows if s == "Q7_seg3"]
        if not pairs:
            continue
        mf, mr, per = macro(pairs)
        md.append(f"### {label}")
        md.append("| Class | Precision | Recall | F1 | Support |")
        md.append("|---|---|---|---|---|")
        for c in CLASSES:
            p, r, f, s = per[c]
            md.append(f"| {c} | {p:.3f} | {r:.3f} | {f:.3f} | {s} |")
        md.append(f"\nMacro-F1: **{mf:.3f}**, Macro-Recall: **{mr:.3f}**\n")

    md.append("## seg3 prediction distribution\n")
    md.append("| Baseline | Pred=P | Pred=K | Pred=V | Pred=C |")
    md.append("|---|---|---|---|---|")
    import collections
    for label, rows in (("B5", b5), ("B6_v1", v1), ("B6_v2", v2)):
        preds = collections.Counter(p for s, g, p, _ in rows if s == "Q7_seg3")
        total = sum(preds.values())
        md.append(f"| {label} | {preds.get('P',0)} ({preds.get('P',0)/total:.0%}) | "
                  f"{preds.get('K',0)} ({preds.get('K',0)/total:.0%}) | "
                  f"{preds.get('V',0)} ({preds.get('V',0)/total:.0%}) | "
                  f"{preds.get('C',0)} ({preds.get('C',0)/total:.0%}) |")
    md.append("")

    for line in md:
        print(line)
    out_path = Path("reports/datav3/b6_full_vs_b5.md")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
