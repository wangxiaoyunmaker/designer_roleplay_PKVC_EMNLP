"""Aggregate saved final-paper conditions across 5 models and merge
into cross_model_full_metrics_v2.json.

Original cross_model_full_metrics.json only has 6 chain-structure metrics.
This script computes the remaining 5 NL-embedding personalization metrics
(seg1_nl_specificity, seg1_persona_align, seg1_trait_pred_rho,
seg2_nl_specificity, seg3_nl_specificity) using cached embeddings.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scripts.final_eval_with_bootstrap import (
    build_baseline_data, METRICS, load_trait_vectors, load_supplements,
    load_cache, get_emb, load_jsonl,
)


CONDITIONS = [
    "b0", "b1", "b2", "b3", "v2",
    "v2_no_caps", "v2_no_seg3_rule", "v2_no_beta_summary",
    "v2_plus_peer", "v2_hybrid",
]


def gpt4o_path(cond: str) -> tuple[str, str]:
    """Map cross-model condition name to GPT-4o file paths (verified by matching
    original cross_model.json TED/non-V values)."""
    base = Path("reports/datav3")
    mapping = {
        "b0":                  "ladder_b0_h",
        "b1":                  "ladder_b1_h",
        "b2":                  "ladder_b2_h",
        "b3":                  "ladder_b3_h",
        "v2":                  "b6_v2_full",
        "v2_no_caps":          "ablation_v2_no_caps",
        "v2_no_seg3_rule":     "ablation_v2_no_seg3_rule",
        "v2_no_beta_summary":  "ablation_v2_no_beta_summary",
        "v2_plus_peer":        "ablation_v2_plus_peer",
        "v2_hybrid":           "b6_v2hybrid_full",
    }
    stem = mapping[cond]
    return str(base / f"{stem}_inference.jsonl"), str(base / f"{stem}_pkvc_eval.jsonl")


def crossmodel_path(slug: str, cond: str) -> tuple[str, str]:
    base = Path("reports/datav3/crossmodel")
    return str(base / f"{slug}_{cond}_inference.jsonl"), str(base / f"{slug}_{cond}_pkvc_eval.jsonl")


def cond_to_label(cond: str) -> str:
    """Map condition slug to baseline label used in JSONL."""
    return {
        "b0": "B0_h", "b1": "B1_h", "b2": "B2_h", "b3": "B3_h",
        "v2": "v2_anchor",
        "v2_no_caps": "v2_no_caps",
        "v2_no_seg3_rule": "v2_no_seg3_rule",
        "v2_no_beta_summary": "v2_no_beta_summary",
        "v2_plus_peer": "v2_plus_peer",
        "v2_hybrid": "v2_hybrid",
    }[cond]


def main():
    models = [
        ("gpt4o", "GPT-4o"),
        ("gemini31flashlite", "Gemini 3.1 Flash Lite"),
        ("claudehaiku45", "Claude Haiku 4.5"),
        ("deepseekv4flash", "DeepSeek V4 Flash"),
        ("qwen35b", "Qwen 3.6 35B A3B"),
    ]

    print("[load] shared data...")
    trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    supplements = load_supplements(Path("data_v3/task_responses.csv"))
    cache = load_cache()
    if not cache:
        raise RuntimeError("Embedding cache is absent. Regenerate it with evaluate_paper before aggregation.")
    print(f"  cache: {len(cache)} embeddings")

    q16_rows = [json.loads(l) for l in
                Path("data_v3/processed/q1q6_extracted_v8.7_full.jsonl").open() if l.strip()]
    designer_q16_nls = defaultdict(list)
    for r in q16_rows:
        if r.get("ok") and r.get("text"):
            designer_q16_nls[r["user_id"]].append(r["text"])
    persona_embs = {}
    for did, nls in designer_q16_nls.items():
        embs = [get_emb(nl, cache) for nl in nls]
        embs = [e for e in embs if e is not None]
        if embs: persona_embs[did] = np.mean(embs, axis=0)
    print(f"  persona_embs: {len(persona_embs)} designers")

    out = {}
    for slug, display in models:
        print(f"\n== {display} ({slug}) ==")
        per_cond = {}
        for cond in CONDITIONS:
            if slug == "gpt4o":
                inf_p, pkvc_p = gpt4o_path(cond)
            else:
                inf_p, pkvc_p = crossmodel_path(slug, cond)
            if not Path(inf_p).exists() or not Path(pkvc_p).exists():
                print(f"  {cond}: MISSING ({inf_p} or pkvc)")
                continue
            label = cond_to_label(cond)
            # GPT-4o files don't tag baseline matching cross-model label scheme,
            # so we skip baseline filter (these files are single-condition runs).
            d = build_baseline_data(label, inf_p, pkvc_p, None, cache,
                                    trait_vectors, supplements, persona_embs)
            if d is None or d.get("n", 0) == 0:
                print(f"  {cond}: NO DATA (label={label}, file={inf_p})")
                continue
            n = d["n"]
            metrics = {"n": n}
            full_sample = list(range(n))
            for name, _, fn in METRICS:
                try:
                    v = fn(d, full_sample)
                    metrics[name] = float(v)
                except Exception as e:
                    metrics[name] = None
                    print(f"    [warn] {cond} {name}: {type(e).__name__}: {str(e)[:80]}")
            per_cond[cond] = metrics
            print(f"  {cond}: n={n}, {len([k for k,v in metrics.items() if v is not None])-1}/11 ok")
        out[slug] = {"name": display, "metrics": per_cond}

    out_path = Path("reports/datav3/cross_model_full_metrics_v2.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
