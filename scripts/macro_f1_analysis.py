"""Macro-F1 + class-aware analysis for B0-B5 baselines.

Replaces dom-acc (which has majority-class artifact in seg3 GT V=93%).

For each (baseline, scenario), compute:
  - per-class precision / recall / F1 (P/K/V/C)
  - macro F1 (mean over 4 classes)
  - balanced accuracy (mean recall over 4 classes)
  - "all-V" baseline F1 for comparison

Output:
  reports/datav3/baselines_q7_macro_f1.md
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ["P", "K", "V", "C"]


def prf_per_class(pairs: list[tuple[str, str]], cls: str) -> dict:
    """Compute precision/recall/F1 for a single class."""
    tp = sum(1 for g, p in pairs if g == cls and p == cls)
    fp = sum(1 for g, p in pairs if g != cls and p == cls)
    fn = sum(1 for g, p in pairs if g == cls and p != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}


def macro_f1(pairs: list[tuple[str, str]]) -> dict:
    """Macro F1 + per-class breakdown."""
    per_class = {c: prf_per_class(pairs, c) for c in CLASSES}
    # Macro F1: average over classes that have support
    classes_with_support = [c for c in CLASSES if per_class[c]["support"] > 0]
    if not classes_with_support:
        return {"macro_f1": 0.0, "macro_recall": 0.0, "per_class": per_class}
    mf1 = sum(per_class[c]["f1"] for c in classes_with_support) / len(classes_with_support)
    mr = sum(per_class[c]["recall"] for c in classes_with_support) / len(classes_with_support)
    return {"macro_f1": mf1, "macro_recall": mr, "per_class": per_class}


def all_v_baseline_f1(pairs: list[tuple[str, str]]) -> dict:
    """What if we just claim V for everyone?"""
    pseudo = [(g, "V") for g, _ in pairs]
    return macro_f1(pseudo)


def main():
    rs = [json.loads(l) for l in
          Path("reports/datav3/baselines_q7_main_final_inference.jsonl").open(encoding="utf-8")
          if l.strip()]

    md = ["# Q7 6-Baseline Macro-F1 + Class-Aware 分析\n"]
    md.append("> seg3 GT V=93% 极度不平衡, dom acc 有 majority-class artifact. 用 macro-F1 替代.\n")

    md.append("## Overall Macro-F1 (mean over P/K/V/C with support)\n")
    md.append("| Baseline | Macro-F1 | Macro-Recall | dom acc | all-V baseline F1 |")
    md.append("|---|---|---|---|---|")

    # Overall (all seg combined)
    for b in ("B0","B1","B2","B3","B4","B5"):
        pairs = []
        for r in rs:
            if r["baseline"] != b or not r["ok"]: continue
            gt = r.get("ground_truth_pkvc",{}).get("dominant","")
            cl = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
            if not gt or not cl: continue
            pairs.append((gt[0], cl[0]))
        if not pairs: continue
        mf = macro_f1(pairs)
        all_v_mf = all_v_baseline_f1(pairs)
        dom_acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        md.append(f"| {b} | {mf['macro_f1']:.3f} | {mf['macro_recall']:.3f} | "
                  f"{dom_acc:.1%} | {all_v_mf['macro_f1']:.3f} |")
    md.append("")

    # Per-seg macro-F1
    md.append("## Per-seg Macro-F1\n")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        md.append(f"### {sc}\n")
        md.append("| Baseline | Macro-F1 | Macro-Recall | dom acc | all-V F1 |")
        md.append("|---|---|---|---|---|")
        for b in ("B0","B1","B2","B3","B4","B5"):
            pairs = []
            for r in rs:
                if r["baseline"] != b or r["scenario"] != sc or not r["ok"]: continue
                gt = r.get("ground_truth_pkvc",{}).get("dominant","")
                cl = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
                if not gt or not cl: continue
                pairs.append((gt[0], cl[0]))
            if not pairs: continue
            mf = macro_f1(pairs)
            all_v_mf = all_v_baseline_f1(pairs)
            dom_acc = sum(1 for g, p in pairs if g == p) / len(pairs)
            md.append(f"| {b} | {mf['macro_f1']:.3f} | {mf['macro_recall']:.3f} | "
                      f"{dom_acc:.1%} | {all_v_mf['macro_f1']:.3f} |")
        md.append("")

    # Per-class detail (most interesting: seg3 per-class)
    md.append("## Seg3 per-class P/R/F1 (暴露 majority-class artifact)\n")
    md.append("GT distribution: V=150, P=7, K=4, C=1 (V 占 93%)\n")
    md.append("**警告: dom acc 看不见的 — B5 把 12 个非 V 个体全 claim V (recall 非 V = 0%)**\n")
    for b in ("B0","B1","B2","B3","B4","B5"):
        pairs = []
        for r in rs:
            if r["baseline"] != b or r["scenario"] != "Q7_seg3" or not r["ok"]: continue
            gt = r.get("ground_truth_pkvc",{}).get("dominant","")
            cl = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
            if not gt or not cl: continue
            pairs.append((gt[0], cl[0]))
        if not pairs: continue
        mf = macro_f1(pairs)
        md.append(f"### {b}")
        md.append("| Class | Precision | Recall | F1 | Support |")
        md.append("|---|---|---|---|---|")
        for c in CLASSES:
            pc = mf["per_class"][c]
            md.append(f"| {c} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                      f"{pc['f1']:.3f} | {pc['support']} |")
        md.append(f"\nMacro-F1: **{mf['macro_f1']:.3f}**, Macro-Recall: **{mf['macro_recall']:.3f}**\n")

    # Also print to stdout
    print("=== Overall Macro-F1 ===")
    print(f"{'Baseline':<5} {'Macro-F1':<10} {'Macro-Recall':<14} {'dom acc':<10} {'all-V F1':<10}")
    for b in ("B0","B1","B2","B3","B4","B5"):
        pairs = []
        for r in rs:
            if r["baseline"] != b or not r["ok"]: continue
            gt = r.get("ground_truth_pkvc",{}).get("dominant","")
            cl = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
            if not gt or not cl: continue
            pairs.append((gt[0], cl[0]))
        if not pairs: continue
        mf = macro_f1(pairs)
        all_v_mf = all_v_baseline_f1(pairs)
        dom_acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        print(f"{b:<5} {mf['macro_f1']:<10.3f} {mf['macro_recall']:<14.3f} "
              f"{dom_acc:<10.1%} {all_v_mf['macro_f1']:<10.3f}")

    print("\n=== Per-seg Macro-F1 ===")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        print(f"\n{sc}:")
        print(f"{'B':<5} {'Macro-F1':<10} {'Macro-Recall':<14} {'dom acc':<10} {'all-V F1':<10}")
        for b in ("B0","B1","B2","B3","B4","B5"):
            pairs = []
            for r in rs:
                if r["baseline"] != b or r["scenario"] != sc or not r["ok"]: continue
                gt = r.get("ground_truth_pkvc",{}).get("dominant","")
                cl = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
                if not gt or not cl: continue
                pairs.append((gt[0], cl[0]))
            if not pairs: continue
            mf = macro_f1(pairs)
            all_v_mf = all_v_baseline_f1(pairs)
            dom_acc = sum(1 for g, p in pairs if g == p) / len(pairs)
            print(f"{b:<5} {mf['macro_f1']:<10.3f} {mf['macro_recall']:<14.3f} "
                  f"{dom_acc:<10.1%} {all_v_mf['macro_f1']:<10.3f}")

    out_path = Path("reports/datav3/baselines_q7_macro_f1.md")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
