"""Adapter: convert inference jsonl's claimed_pkvc → pkvc_eval-like jsonl.

Fallback when GPT-4o extraction not available (quota exhausted).
Uses model's self-reported state_sequence + dominant + criterion.

Note: less rigorous than v8.7 extractor (which re-parses from NL),
but provides preliminary metrics when extract unavailable.

Usage:
  python scripts/claimed_to_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
  # Or batch:
  python scripts/claimed_to_pkvc.py --batch reports/datav3/crossmodel/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def convert_one(inf_path: Path, out_path: Path):
    n_ok = 0
    n_total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for line in inf_path.open(encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            n_total += 1
            if not r.get("ok"): continue
            cp = r.get("claimed_pkvc", {})
            seq = cp.get("state_sequence", [])
            dom = cp.get("dominant_factor", "")
            if not seq and not dom:
                continue
            out_row = {
                "baseline": r.get("baseline"),
                "designer_id": r["designer_id"],
                "scenario": r["scenario"],
                "ok": True,
                "pred": {
                    "dominant_factor": dom,
                    "state_sequence": seq if seq else ([dom[0]] if dom else []),
                    "criterion": cp.get("criterion"),
                    "confidence": cp.get("confidence", "high"),
                    "extraction_notes": "from claimed_pkvc (no v8.7 extract)",
                }
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_ok += 1
    return n_ok, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="single inference jsonl input")
    ap.add_argument("--output", help="single output pkvc-like jsonl")
    ap.add_argument("--batch", help="batch process all *_inference.jsonl in dir (output to same dir as *_pkvc_eval.jsonl)")
    args = ap.parse_args()

    if args.batch:
        d = Path(args.batch)
        files = sorted(d.glob("*_inference.jsonl"))
        print(f"Found {len(files)} inference files")
        for inf in files:
            out_name = inf.name.replace("_inference.jsonl", "_pkvc_eval.jsonl")
            out_path = d / out_name
            n_ok, n_total = convert_one(inf, out_path)
            print(f"  {inf.name}: {n_ok}/{n_total} → {out_name}")
    else:
        if not args.input or not args.output:
            print("Need --input + --output, or --batch"); return
        n_ok, n_total = convert_one(Path(args.input), Path(args.output))
        print(f"{n_ok}/{n_total} → {args.output}")


if __name__ == "__main__":
    main()
