"""v8 extractor diagnostic: PKVC label recovery accuracy on Q1-Q6 expanded NL.

Ground truth = option_trait_mapping.json 的 o_primary_hint (项目 metadata).
样本 = 每 PKVC class 抽 10 段, 共 40 (从 q1q6_expansion_outputs.jsonl).
跑 v8 extractor on each, 算:
  - head accuracy (predicted_head == ground_truth_letter)
  - confusion matrix (4×4)
  - per-class accuracy
  - per-confidence accuracy
Verdict:
  - > 70%: extractor OK, pilot 'all V' 是 Q7 数据 + LLM bias 真问题
  - 40-70%: extractor 弱, 需 v8-short prompt 或 calibration
  - < 40%: extractor 坏, 不能用
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from sft_pipeline.eval.datav3_stats import loaders


V8_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": ["dominant_factor", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def load_v8_prompt(prompt_path: Path | None = None) -> tuple[str, str]:
    """Fix: 用 ``` 围栏边界, 不靠 '---' (会误中 markdown table '|---|---|')."""
    if prompt_path is None:
        prompt_path = Path("sft_pipeline/config/prompts/extractor_v8.md")
    pt = prompt_path.read_text(encoding="utf-8")
    sys_section_start = pt.find("## System Prompt")
    sys_fence_open = pt.find("```", sys_section_start)
    sys_fence_close = pt.find("```", sys_fence_open + 3)
    sys_block = pt[sys_fence_open + 3 : sys_fence_close].strip()
    usr_section_start = pt.find("## User Prompt Template")
    usr_fence_open = pt.find("```", usr_section_start)
    usr_fence_close = pt.find("```", usr_fence_open + 3)
    usr_template = pt[usr_fence_open + 3 : usr_fence_close].strip()
    return sys_block, usr_template


def extract_pkvc(client: OpenAI, model: str, nl: str, sys_block: str, usr_template: str) -> dict:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_block},
            {"role": "user", "content": usr_template.replace("{INPUT_TEXT}", nl)},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pkvc_extraction", "schema": V8_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="reports/datav3/v8_diagnostic.json")
    ap.add_argument("--prompt", default="sft_pipeline/config/prompts/extractor_v8.md",
                    help="path to v8.x prompt md")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY env")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    # load
    print("[data] loading expansion + option_trait_mapping + task...")
    expansion_df = loaders.load_expansion()
    option_trait_map = json.loads(
        Path("reports/datav3/per_designer/option_trait_mapping.json").read_text(encoding="utf-8")
    )
    task_df = loaders.add_canonical_option(loaders.load_task_long(expanded=True))

    # (user_id, qid) → option lookup
    option_lookup: dict[tuple[str, str], str] = {}
    for _, r in task_df.iterrows():
        opt = r.get("option")
        if isinstance(opt, str):
            option_lookup[(r["user_id"], r["question_id"])] = opt

    # 拼 samples
    samples = []
    for _, r in expansion_df.iterrows():
        uid = r["user_id"]
        qid = r["question_id"]
        opt = option_lookup.get((uid, qid))
        if not opt:
            continue
        key = f"{qid}_{opt}"
        if key not in option_trait_map:
            continue
        gt_pkvc = option_trait_map[key].get("o_primary_hint")
        if not gt_pkvc or gt_pkvc not in ("K", "P", "V", "C"):
            continue
        nl = r.get("expanded_nl")
        if not nl or not isinstance(nl, str):
            continue
        samples.append({
            "user_id": uid,
            "qid": qid,
            "option": opt,
            "ground_truth": gt_pkvc,
            "nl": nl,
            "nl_len": len(nl),
        })

    print(f"[data] eligible samples (with gt + NL): {len(samples)}")

    # stratify by ground truth
    by_gt: dict[str, list] = defaultdict(list)
    for s in samples:
        by_gt[s["ground_truth"]].append(s)
    print(f"[data] per-class available: {dict((k, len(v)) for k, v in by_gt.items())}")

    rng = np.random.default_rng(args.seed)
    selected: list[dict] = []
    for cls in ["K", "P", "V", "C"]:
        pool = by_gt.get(cls, [])
        rng.shuffle(pool)
        taken = pool[: args.n_per_class]
        print(f"  class {cls}: {len(pool)} available → sampled {len(taken)}")
        selected.extend(taken)

    print(f"\n[run] total samples to extract: {len(selected)}")
    print(f"[run] model: {args.model}\n")

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_v8_prompt(Path(args.prompt))
    print(f"[prompt] {args.prompt}: system {len(sys_block)} chars, user template {len(usr_template)} chars")

    results = []
    t_start = time.time()
    for i, s in enumerate(selected, 1):
        try:
            res = extract_pkvc(client, args.model, s["nl"], sys_block, usr_template)
            pred = res["dominant_factor"]
            pred_head = pred[0] if pred else "?"
            match = pred_head == s["ground_truth"]
            elapsed = time.time() - t_start
            print(
                f"[{i:2d}/{len(selected)}] {s['user_id'][:6]} {s['qid']}.{s['option']} "
                f"({s['nl_len']:>3d}字) gt={s['ground_truth']} → {pred} "
                f"({res['confidence']}) {'✓' if match else '✗'}  [{elapsed:.0f}s]"
            )
            results.append({
                **s,
                "predicted": pred,
                "predicted_head": pred_head,
                "confidence": res["confidence"],
                "notes": res["extraction_notes"],
                "match": match,
            })
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:2d}/{len(selected)}] FAIL: {e}")

    if not results:
        print("no results")
        return 1

    # accuracy + confusion
    classes = ["K", "P", "V", "C"]
    n_match = sum(1 for r in results if r["match"])
    acc = n_match / len(results)
    cm: dict[str, dict[str, int]] = {gt: {p: 0 for p in classes} for gt in classes}
    for r in results:
        if r["predicted_head"] in classes:
            cm[r["ground_truth"]][r["predicted_head"]] += 1

    print(f"\n=== RESULTS ({len(results)} samples) ===\n")
    print(f"Overall head-accuracy: {n_match}/{len(results)} = {acc:.1%}\n")

    print("Confusion matrix (rows = ground truth, cols = predicted head):")
    print(f"  GT\\Pred |  K  |  P  |  V  |  C  | total")
    for gt in classes:
        row = [str(cm[gt].get(p, 0)).rjust(3) for p in classes]
        total = sum(cm[gt].values())
        correct = cm[gt].get(gt, 0)
        acc_str = f"{correct}/{total}={correct/max(total,1):.0%}"
        print(f"  {gt:7s} | {' | '.join(row)} | {acc_str}")

    print("\nPer-confidence accuracy:")
    conf_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"match": 0, "total": 0})
    for r in results:
        conf_stats[r["confidence"]]["total"] += 1
        if r["match"]:
            conf_stats[r["confidence"]]["match"] += 1
    for c in ["high", "medium", "low"]:
        d = conf_stats.get(c, {"match": 0, "total": 0})
        if d["total"]:
            print(f"  {c}: {d['match']}/{d['total']} = {d['match']/d['total']:.1%}")
        else:
            print(f"  {c}: 0/0 = —")

    # extraction_notes 里 "信号弱" 比例
    n_weak = sum(1 for r in results if "信号弱" in r.get("notes", ""))
    print(f"\n'信号弱' in notes: {n_weak}/{len(results)} = {n_weak/len(results):.1%}")

    # verdict
    print("\n=== VERDICT ===")
    if acc > 0.7:
        verdict = f"✅ extractor OK (head-acc {acc:.0%} > 70%) — '全 V' in pilot 是 Q7 数据 + LLM gen bias 真问题"
    elif acc > 0.4:
        verdict = f"⚠ extractor 弱 (head-acc {acc:.0%} ∈ [40-70]) — 需 v8-short prompt or calibration"
    else:
        verdict = f"❌ extractor 坏 (head-acc {acc:.0%} < 40%) — v8 在短 NL 上不能用, 必修"
    print(verdict)

    # save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_samples": len(results),
        "head_accuracy": acc,
        "confusion_matrix": {gt: dict(cm[gt]) for gt in classes},
        "per_class_acc": {
            gt: cm[gt].get(gt, 0) / max(sum(cm[gt].values()), 1) for gt in classes
        },
        "per_confidence_acc": {
            c: (d["match"] / d["total"] if d["total"] else 0) for c, d in conf_stats.items()
        },
        "n_weak_notes": n_weak,
        "verdict": verdict,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
