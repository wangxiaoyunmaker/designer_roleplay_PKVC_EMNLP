"""Merge + relabel inference & extract jsonl for unified new B0-B5 naming.

bg1 (OLD code, structured): B1/B1d/B2/B3/B4 — only keep B1→B0, B1d→B1 (rest were CoT/no-pi, discard)
bg2 (NEW code, structured): B2/B3/B4/B5 (new accumulative, no CoT) — keep as-is

Output:
  - reports/datav3/baselines_q7_main_inference.jsonl  (B0-B5 unified)
  - reports/datav3/baselines_q7_main_pkvc_eval.jsonl  (B0-B5 unified extract)
"""
from __future__ import annotations
import json
from pathlib import Path


RELABEL = {"B1": "B0", "B1d": "B1"}  # bg1 OLD → NEW naming
KEEP_FROM_BG1 = {"B1", "B1d"}        # discard bg1 B2/B3/B4 (CoT)
KEEP_FROM_BG2 = {"B2", "B3", "B4", "B5"}


def relabel_row(r: dict, mapping: dict[str, str]) -> dict:
    """Return new dict with baseline field relabeled."""
    new_r = dict(r)
    if r["baseline"] in mapping:
        new_r["baseline"] = mapping[r["baseline"]]
    return new_r


def main():
    # ============ INFERENCE ============
    bg1_inf = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_inference.jsonl").open(encoding="utf-8")
               if l.strip()]
    bg2_inf = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_b2to5_inference.jsonl").open(encoding="utf-8")
               if l.strip()]

    bg1_keep = [relabel_row(r, RELABEL) for r in bg1_inf if r["baseline"] in KEEP_FROM_BG1]
    bg2_keep = [r for r in bg2_inf if r["baseline"] in KEEP_FROM_BG2]

    merged_inf = bg1_keep + bg2_keep
    print(f"[inference]")
    print(f"  bg1 kept (B1→B0, B1d→B1): {len(bg1_keep)}")
    print(f"  bg2 kept (B2-B5):          {len(bg2_keep)}")
    print(f"  total merged:              {len(merged_inf)}")

    out_inf = Path("reports/datav3/baselines_q7_main_inference.jsonl")
    with out_inf.open("w", encoding="utf-8") as f:
        for r in merged_inf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [saved] {out_inf}")

    # ============ EXTRACT (PKVC v8.7) ============
    bg1_ext = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_pkvc_eval.jsonl").open(encoding="utf-8")
               if l.strip()]
    bg2_ext = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_b2to5_pkvc_eval.jsonl").open(encoding="utf-8")
               if l.strip()]

    bg1_ext_keep = [relabel_row(r, RELABEL) for r in bg1_ext if r["baseline"] in KEEP_FROM_BG1]
    bg2_ext_keep = [r for r in bg2_ext if r["baseline"] in KEEP_FROM_BG2]

    merged_ext = bg1_ext_keep + bg2_ext_keep
    print(f"\n[extract]")
    print(f"  bg1 kept (B1→B0, B1d→B1): {len(bg1_ext_keep)}")
    print(f"  bg2 kept (B2-B5):          {len(bg2_ext_keep)}")
    print(f"  total merged:              {len(merged_ext)}")
    n_ok = sum(1 for r in merged_ext if r["ok"])
    print(f"  ok: {n_ok}/{len(merged_ext)} ({n_ok/len(merged_ext):.1%})")

    out_ext = Path("reports/datav3/baselines_q7_main_pkvc_eval.jsonl")
    with out_ext.open("w", encoding="utf-8") as f:
        for r in merged_ext:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [saved] {out_ext}")

    # Per-baseline summary
    from collections import Counter
    print(f"\n[main inference per-baseline]")
    for b, n in sorted(Counter(r["baseline"] for r in merged_inf).items()):
        print(f"  {b}: {n} rows")
    print(f"\n[main extract per-baseline (ok only)]")
    for b, n in sorted(Counter(r["baseline"] for r in merged_ext if r["ok"]).items()):
        print(f"  {b}: {n} ok rows")


if __name__ == "__main__":
    main()
