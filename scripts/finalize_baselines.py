"""Finalize baselines: merge retry results into main + ablation jsonl, then ready for eval.

Inputs:
  Main: baselines_q7_main_{inference,pkvc_eval}.jsonl (含 bg1+bg2 merged)
        + baselines_q7_b2to5_retry{,_pkvc_eval}.jsonl (65 retry rows for B2-B5)
  Ablation: baselines_q7_ablation{,_pkvc_eval}.jsonl + retry counterparts

Output:
  baselines_q7_main_final_{inference,pkvc_eval}.jsonl (B0-B5 100% coverage)
  baselines_q7_ablation_final_{inference,pkvc_eval}.jsonl (B5 variants 100% coverage)
"""
from __future__ import annotations
import json
from pathlib import Path


def merge_replace(base_path: Path, retry_path: Path, out_path: Path) -> tuple[int, int]:
    """Load base + retry. Replace base's failed rows with retry's successful rows. APPEND retry rows
    whose key is not in base. Save."""
    base_rows = [json.loads(l) for l in base_path.open(encoding="utf-8") if l.strip()]
    retry_rows = [json.loads(l) for l in retry_path.open(encoding="utf-8") if l.strip()]
    retry_map = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in retry_rows if r["ok"]}
    n_replaced = 0
    final = []
    base_keys = set()
    for r in base_rows:
        key = (r["baseline"], r["designer_id"], r["scenario"])
        base_keys.add(key)
        if not r["ok"] and key in retry_map:
            final.append(retry_map[key])
            n_replaced += 1
        else:
            final.append(r)
    # Append retry rows whose key not in base (newly added)
    n_appended = 0
    for r in retry_rows:
        if not r["ok"]: continue
        key = (r["baseline"], r["designer_id"], r["scenario"])
        if key not in base_keys:
            final.append(r)
            n_appended += 1

    n_ok = sum(1 for r in final if r["ok"])
    with out_path.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {out_path.name}: {n_ok}/{len(final)} ok ({n_ok/len(final):.1%}); "
          f"{n_replaced} replaced + {n_appended} appended")
    return n_ok, len(final)


def main():
    # Main inference (B0-B5)
    print("[main inference] — merging bg2 retry into bg2+bg1 merged main...")
    # The merge_relabel_baselines.py already merged bg1 (B1→B0, B1d→B1) + bg2.
    # bg2 retry covers B2-B5 fails. Re-merge with the same pipeline + retry.
    # Strategy: re-read bg1, bg2, bg2_retry, then assemble.
    bg1 = [json.loads(l) for l in
           Path("reports/datav3/baselines_q7_structured_inference.jsonl").open(encoding="utf-8")
           if l.strip()]
    bg2 = [json.loads(l) for l in
           Path("reports/datav3/baselines_q7_structured_b2to5_inference.jsonl").open(encoding="utf-8")
           if l.strip()]
    bg2_retry = [json.loads(l) for l in
                 Path("reports/datav3/baselines_q7_b2to5_retry.jsonl").open(encoding="utf-8")
                 if l.strip()]
    bg2_retry_map = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in bg2_retry if r["ok"]}

    # bg1: keep B1 (→ B0), B1d (→ B1) only
    RELABEL = {"B1": "B0", "B1d": "B1"}
    bg1_keep = []
    for r in bg1:
        if r["baseline"] not in RELABEL:
            continue
        new_r = dict(r)
        new_r["baseline"] = RELABEL[r["baseline"]]
        bg1_keep.append(new_r)

    # bg2: B2-B5, replace failed with retry
    bg2_final = []
    n_replaced_bg2 = 0
    for r in bg2:
        if r["baseline"] not in ("B2","B3","B4","B5"):
            continue
        if not r["ok"]:
            key = (r["baseline"], r["designer_id"], r["scenario"])
            if key in bg2_retry_map:
                bg2_final.append(bg2_retry_map[key])
                n_replaced_bg2 += 1
                continue
        bg2_final.append(r)

    main_final = bg1_keep + bg2_final
    n_ok = sum(1 for r in main_final if r["ok"])
    out_path = Path("reports/datav3/baselines_q7_main_final_inference.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in main_final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  baselines_q7_main_final_inference.jsonl: {n_ok}/{len(main_final)} ok; {n_replaced_bg2} retries replaced")

    # Main extract: same merge
    print("\n[main extract]")
    bg1_ext = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_pkvc_eval.jsonl").open(encoding="utf-8")
               if l.strip()]
    bg2_ext = [json.loads(l) for l in
               Path("reports/datav3/baselines_q7_structured_b2to5_pkvc_eval.jsonl").open(encoding="utf-8")
               if l.strip()]
    bg2_ext_retry = [json.loads(l) for l in
                     Path("reports/datav3/baselines_q7_b2to5_retry_pkvc_eval.jsonl").open(encoding="utf-8")
                     if l.strip()]
    bg2_ext_retry_map = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in bg2_ext_retry if r["ok"]}

    bg1_ext_keep = []
    for r in bg1_ext:
        if r["baseline"] not in RELABEL:
            continue
        new_r = dict(r)
        new_r["baseline"] = RELABEL[r["baseline"]]
        bg1_ext_keep.append(new_r)

    bg2_ext_final = []
    for r in bg2_ext:
        if r["baseline"] not in ("B2","B3","B4","B5"):
            continue
        if not r["ok"]:
            key = (r["baseline"], r["designer_id"], r["scenario"])
            if key in bg2_ext_retry_map:
                bg2_ext_final.append(bg2_ext_retry_map[key])
                continue
        bg2_ext_final.append(r)

    # Append any retry rows that weren't in original (in case)
    main_ext_final = bg1_ext_keep + bg2_ext_final
    # Also append bg2_ext_retry rows whose key doesn't appear in bg2_ext (newly added)
    existing_keys = {(r["baseline"], r["designer_id"], r["scenario"]) for r in main_ext_final}
    for r in bg2_ext_retry:
        if r["ok"] and (r["baseline"], r["designer_id"], r["scenario"]) not in existing_keys:
            main_ext_final.append(r)

    n_ok = sum(1 for r in main_ext_final if r["ok"])
    out_path = Path("reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in main_ext_final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  baselines_q7_main_final_pkvc_eval.jsonl: {n_ok}/{len(main_ext_final)} ok")

    # ============ ABLATION ============
    print("\n[ablation inference]")
    merge_replace(
        Path("reports/datav3/baselines_q7_ablation.jsonl"),
        Path("reports/datav3/baselines_q7_ablation_retry.jsonl"),
        Path("reports/datav3/baselines_q7_ablation_final_inference.jsonl"),
    )
    print("\n[ablation extract]")
    merge_replace(
        Path("reports/datav3/baselines_q7_ablation_pkvc_eval.jsonl"),
        Path("reports/datav3/baselines_q7_ablation_retry_pkvc_eval.jsonl"),
        Path("reports/datav3/baselines_q7_ablation_final_pkvc_eval.jsonl"),
    )

    # Per-baseline summary
    print("\n=== final main inference per-baseline ===")
    from collections import Counter
    for b, n in sorted(Counter(r["baseline"] for r in main_final if r["ok"]).items()):
        print(f"  {b}: {n} rows")

    print("\n=== final main extract per-baseline (ok) ===")
    for b, n in sorted(Counter(r["baseline"] for r in main_ext_final if r["ok"]).items()):
        print(f"  {b}: {n} rows")


if __name__ == "__main__":
    main()
