"""Retry v8.7 PKVC extract on rows that failed previously (quota/network).

Input:
  - eval jsonl with some rows ok=False (containing baseline/designer_id/scenario)
  - inference jsonl with original simulated_nl

Output:
  - same eval jsonl path with failed rows replaced (or fresh output)

Usage:
  .venv/bin/python scripts/retry_failed_extract.py \
      --eval reports/datav3/baselines_q7_structured_pkvc_eval.jsonl \
      --inference reports/datav3/baselines_q7_structured_inference.jsonl \
      --concurrency 12
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import AsyncOpenAI

from scripts.eval_baselines_pkvc import V8_7_SCHEMA, load_prompt, extract_one


async def main_async(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # Load existing eval
    eval_rows = [json.loads(l) for l in Path(args.eval).open(encoding="utf-8") if l.strip()]
    failed = [r for r in eval_rows if not r["ok"]]
    print(f"[load] eval: {len(eval_rows)} total, {len(failed)} failed, {len(eval_rows) - len(failed)} ok")

    if not failed:
        print("nothing to retry"); return 0

    # Load inference jsonl to get simulated_nl for failed rows
    inf_rows = [json.loads(l) for l in Path(args.inference).open(encoding="utf-8") if l.strip()]
    inf_map = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in inf_rows
               if r.get("ok") and r.get("simulated_nl")}
    print(f"[load] inference: {len(inf_map)} ok rows")

    # Build retry tasks (only for failed rows that have inference data)
    retry_tasks = []
    for r in failed:
        key = (r["baseline"], r["designer_id"], r["scenario"])
        inf = inf_map.get(key)
        if not inf:
            continue
        retry_tasks.append({"failed_row": r, "simulated_nl": inf["simulated_nl"]})
    print(f"[retry] {len(retry_tasks)} rows to retry")

    sys_b, usr_t = load_prompt(Path("sft_pipeline/config/prompts/extractor_v8.7.md"))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(args.concurrency)

    async def _wrap(idx, task):
        seg_role = task["failed_row"]["scenario"].split("_")[1] if "_" in task["failed_row"]["scenario"] else "single"
        out = await extract_one(client, args.model, task["simulated_nl"], sys_b, usr_t,
                                 seg_role, sem, idx, len(retry_tasks))
        return {**task["failed_row"], **out, "_retried": True}

    t0 = time.time()
    tasks = [asyncio.create_task(_wrap(i + 1, t)) for i, t in enumerate(retry_tasks)]
    new_results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    n_ok_new = sum(1 for r in new_results if r["ok"])
    print(f"\n[done] retry: {n_ok_new}/{len(new_results)} ok in {elapsed:.1f}s")

    # Merge: keep originally ok rows + replace failed rows with retried
    by_key_retried = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in new_results}
    merged = []
    for r in eval_rows:
        if r["ok"]:
            merged.append(r)
        else:
            key = (r["baseline"], r["designer_id"], r["scenario"])
            merged.append(by_key_retried.get(key, r))

    # Save back (overwrite same file)
    n_ok_final = sum(1 for r in merged if r["ok"])
    print(f"[merge] final: {n_ok_final}/{len(merged)} ok ({n_ok_final/len(merged):.1%})")
    Path(args.eval).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged), encoding="utf-8"
    )
    print(f"[saved] {args.eval} (overwritten)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    help="eval jsonl path with ok=False rows to retry")
    ap.add_argument("--inference", required=True,
                    help="inference jsonl with simulated_nl source")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
