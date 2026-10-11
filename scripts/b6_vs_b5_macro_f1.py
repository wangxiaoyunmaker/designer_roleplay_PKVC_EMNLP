"""Compare B6 (sample) vs B5 (same 20 designer subset) macro-F1.

B5 source: reports/datav3/baselines_q7_main_final_inference.jsonl (filter to SAMPLE_DESIGNERS)
B6 source: reports/datav3/b6_sample_inference.jsonl

Output: per-seg + overall macro-F1 + per-class P/R/F1 for both, side-by-side.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_b6_sample import SAMPLE_DESIGNERS

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
        if r["designer_id"] not in SAMPLE_DESIGNERS:
            continue
        gt = (r.get("ground_truth_pkvc") or {}).get("dominant", "")
        cl = (r.get("claimed_pkvc") or {}).get("dominant_factor", "")
        if not gt or not cl:
            continue
        out.append((r["scenario"], gt[0], cl[0], r["designer_id"]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--b6-path", default="reports/datav3/b6_sample_inference.jsonl")
    ap.add_argument("--label", default="B6")
    ap.add_argument("--output", default="reports/datav3/b6_vs_b5_sample_macro_f1.md")
    a = ap.parse_args()
    b5 = load("reports/datav3/baselines_q7_main_final_inference.jsonl", baseline="B5")
    # Auto-detect B6 baseline label in file (B6 / B6_v1 / B6_v2 / B6_v3 etc)
    b6 = load(a.b6_path, baseline=None)
    print(f"B5: {len(b5)} rows, {a.label}: {len(b6)} rows (filtered to {len(SAMPLE_DESIGNERS)} designers)\n")

    md = [f"# {a.label} sample vs B5 (same 20 designer) macro-F1\n"]
    md.append(f"Sample size: 20 designer × 3 seg = 60 calls per baseline. B5 source: main_final.jsonl.\n")
    md.append("## Overall macro-F1\n")
    md.append("| Baseline | Macro-F1 | Macro-Recall | dom acc |")
    md.append("|---|---|---|---|")
    for label, rows in (("B5", b5), (a.label, b6)):
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
        for label, rows in (("B5", b5), (a.label, b6)):
            pairs = [(g, p) for s, g, p, _ in rows if s == sc]
            if not pairs:
                continue
            mf, mr, _ = macro(pairs)
            acc = sum(1 for g, p in pairs if g == p) / len(pairs)
            md.append(f"| {label} | {mf:.3f} | {mr:.3f} | {acc:.1%} |")
        md.append("")

    md.append("## seg3 per-class P/R/F1 (关键: 看 B6 能否拉起 P/K/C recall)\n")
    md.append("GT in sample: 7 P + 4 K + 1 C + 8 V = 20\n")
    for label, rows in (("B5", b5), (a.label, b6)):
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

    # Per-designer seg3 prediction breakdown (whose did B6 get right / wrong vs B5)
    md.append("## Per-designer seg3 prediction (sorted by GT class)\n")
    md.append(f"| Designer | GT | B5 pred | {a.label} pred | B5 ✓? | {a.label} ✓? |")
    md.append("|---|---|---|---|---|---|")
    b5_seg3 = {(d, "Q7_seg3"): (g, p) for s, g, p, d in b5 if s == "Q7_seg3"}
    b6_seg3 = {(d, "Q7_seg3"): (g, p) for s, g, p, d in b6 if s == "Q7_seg3"}
    by_gt = {}
    for (d, sc), (g, p) in b5_seg3.items():
        by_gt.setdefault(g, []).append(d)
    order = []
    for cls in ("P", "K", "C", "V"):
        order.extend(sorted(by_gt.get(cls, [])))
    for d in order:
        b5_v = b5_seg3.get((d, "Q7_seg3"), ("?", "?"))
        b6_v = b6_seg3.get((d, "Q7_seg3"), ("?", "?"))
        b5_ok = "✓" if b5_v[0] == b5_v[1] else "✗"
        b6_ok = "✓" if b6_v[0] == b6_v[1] else "✗"
        md.append(f"| {d[:14]} | {b5_v[0]} | {b5_v[1]} | {b6_v[1]} | {b5_ok} | {b6_ok} |")
    md.append("")

    # Print to stdout
    for line in md:
        print(line)

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
