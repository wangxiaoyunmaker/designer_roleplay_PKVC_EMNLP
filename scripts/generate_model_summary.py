"""Per-model summary md generator — runs 11-metric bootstrap eval on 7 conditions of
one model and writes paper-ready markdown.

Usage:
  python scripts/generate_model_summary.py --model-slug gemini31flashlite \
      --display-name "Gemini 3.1 Flash Lite" --model-id gemini-3.1-flash-lite \
      --output reports/datav3/gemini31flashlite_experiments_summary.md

For each cross-model: takes 7 inference jsonl + 7 pkvc_eval jsonl files in
reports/datav3/crossmodel/{slug}_{cond}_inference.jsonl etc.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse from final_eval_with_bootstrap
from scripts.final_eval_with_bootstrap import (
    build_baseline_data, run_bootstrap, METRICS, LOWER_BETTER,
    load_trait_vectors, load_supplements, load_cache, get_emb, embed_missing,
    load_jsonl,
)
from collections import defaultdict
import numpy as np


CONDITIONS = [
    # (suffix, display, baseline_label_in_jsonl_pattern, variant_for_label)
    ("b0",                "B0_h (zero-shot)",     "B0_h"),
    ("b1",                "B1_h (+pi)",           "B1_h"),
    ("b2",                "B2_h (+stance)",       "B2_h"),
    ("b3",                "B3_h (+self)",         "B3_h"),
    ("v2",                "v2_anchor (B4_h main)", "v2_anchor"),
    ("v2_no_caps",        "v2 - CAPS module",     "v2_no_caps"),
    ("v2_no_seg3_rule",   "v2 - seg3 rule",       "v2_no_seg3_rule"),
    ("v2_no_beta_summary","v2 - β 速览",          "v2_no_beta_summary"),
    ("v2_plus_peer",      "v2 + peer",            "v2_plus_peer"),
    ("v2_hybrid",         "v2_hybrid (seg3 v2_clean)", "v2_hybrid"),
]


def build_baselines_for_model(slug: str):
    """Return BASELINES-like list for one model's 7 conditions."""
    base = Path(f"reports/datav3/crossmodel")
    out = []
    for suffix, _, label in CONDITIONS:
        inf = base / f"{slug}_{suffix}_inference.jsonl"
        pkvc = base / f"{slug}_{suffix}_pkvc_eval.jsonl"
        if not inf.exists():
            print(f"  WARN: missing {inf}, skip")
            continue
        if not pkvc.exists():
            print(f"  WARN: missing {pkvc}, skip")
            continue
        out.append((label, str(inf), str(pkvc), None))
    return out


def fmt_ci(s, key="point"):
    """Format point [CI lo, hi]."""
    return f"{s['point']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", required=True, help="e.g. gemini31flashlite")
    ap.add_argument("--display-name", required=True, help="e.g. Gemini 3.1 Flash Lite")
    ap.add_argument("--model-id", required=True, help="actual API model ID (for reproducibility)")
    ap.add_argument("--output", required=True, help="output md path")
    args = ap.parse_args()

    BASELINES = build_baselines_for_model(args.model_slug)
    if not BASELINES:
        print("ERROR: no baseline data found"); sys.exit(1)
    print(f"Loaded {len(BASELINES)} baselines for {args.display_name}")

    # ============ Load shared data ============
    print("[load] trait + supplements + persona + embedding cache...")
    trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    supplements = load_supplements(Path("data_v3/task_responses.csv"))
    cache = load_cache()
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

    # Pre-embed any new NL texts (skip gracefully if quota/key missing)
    texts_to_embed = []
    for label, inf_p, _, bl in BASELINES:
        rs = load_jsonl(inf_p, bl)
        for r in rs:
            if r.get("simulated_nl"): texts_to_embed.append(r["simulated_nl"])
            crit = (r.get("claimed_pkvc") or {}).get("criterion","")
            if crit: texts_to_embed.append(crit)
    embedding_available = True
    try:
        embed_missing(texts_to_embed, cache)
    except Exception as e:
        print(f"  WARN: embedding skipped ({type(e).__name__}: {str(e)[:100]})")
        print(f"  NL-embedding metrics will be N/A (zero vectors)")
        embedding_available = False
    print(f"  cache: {len(cache)}, NL-embed available: {embedding_available}")

    # ============ Build + bootstrap each baseline ============
    print("[build] per-baseline data + bootstrap...")
    baseline_data = {}
    summaries = {}
    for label, inf_p, pkvc_p, bl in BASELINES:
        d = build_baseline_data(label, inf_p, pkvc_p, bl, cache,
                                trait_vectors, supplements, persona_embs)
        if d is None:
            print(f"  WARN: build_baseline_data returned None for {label}")
            continue
        baseline_data[label] = d
        print(f"  {label}: n={d['n']}, bootstrapping...")
        summaries[label] = run_bootstrap(d)

    # ============ Generate markdown ============
    print("[generate] markdown...")
    md = []
    md.append(f"# {args.display_name} 阶段实验结果 (paper-ready)\n")
    md.append(f"> 整理日期: 2026-05-26 (cross-model validation)")
    md.append(f"> 数据集: 162 designer × 3 Q7 seg")
    md.append(f"> 模型: {args.display_name} (id: `{args.model_id}`)")
    md.append(f"> 输入构造: v2_anchor 系列 prompt (跟 GPT-4o 同 prompt 套件)")
    md.append(f"> Evaluation: 11-metric set + 1000-bootstrap 95% CI (designer-level resampling)\n")
    if not embedding_available:
        md.append("> ⚠ **NL-embedding metrics 可能受 quota 限制** — 部分新 text 未 embed, NL specificity / persona-NL / criterion-persona 可能用 zero vector fallback (degrade). 等 quota refill 后重跑可恢复.\n")
    # Detect PKVC source by inspecting first pkvc_eval entry
    pkvc_source = "v8.7 extracted (real)"
    for _, _, pkvc_p, _ in BASELINES[:1]:
        try:
            with open(pkvc_p, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r0 = json.loads(line)
                        notes = (r0.get("pred") or {}).get("extraction_notes", "")
                        if "from claimed_pkvc" in notes:
                            pkvc_source = "claimed_pkvc fallback (model self-reported, not v8.7 extracted)"
                        break
        except Exception:
            pass
    if "claimed_pkvc" in pkvc_source:
        md.append(f"> ⚠ **PKVC source: {pkvc_source}** — quota 不够, 未跑 v8.7 extract. 等 quota 后重跑可补全.\n")
    else:
        md.append(f"> PKVC source: {pkvc_source}\n")
    md.append("---\n")

    # Section 0
    md.append("## 0. 实验总览\n")
    md.append(f"### 0.1 总规模\n")
    md.append(f"- **{len(BASELINES)} 个 condition × 162 designer × 3 seg = ~{len(BASELINES)*486} inference calls**")
    md.append(f"- + PKVC extraction calls (v8.7 extractor via GPT-4o)\n")

    md.append("### 0.2 7 个 condition 分类\n")
    md.append("| Group | Condition | 用途 |")
    md.append("|---|---|---|")
    md.append("| **Main** | B0 (zero-shot) | floor baseline |")
    md.append("|  | **v2_anchor** ★ | paper main prompt (PKVC def + seg3 rule + β 速览 + 4 类 CAPS) |")
    md.append("| **组件 ablation** | v2_no_caps (删 ⑤ CAPS 整块) | **CAPS module 整体必要性** |")
    md.append("|  | v2_no_seg3_rule (删 SYSTEM seg3 rule) | seg3 决策规则必要性 |")
    md.append("|  | v2_no_beta_summary (删 β 速览 header) | β 速览价值 |")
    md.append("| **架构 ablation** | v2_plus_peer (+ ④ peer) | peer 加上去 net 价值 |")
    md.append("|  | v2_hybrid (seg3 用 v2_clean prompt) | per-seg dispatch 价值 |\n")
    md.append("---\n")

    # Section 1: metric set (compact reference)
    md.append("## 1. Final 11-Metric Set (reference, locked across all models)\n")
    md.append("3 大维度 × 11 metric:\n")
    md.append("- **Fidelity (路径准度, seg2)**: macro-TED ↓, macro-LCS-F1 ↑, non-V recall ↑")
    md.append("- **Personalization (个性化)**: NL specificity ratio (3 segs), Persona-NL alignment (seg1), Trait-pred Spearman ρ (seg1), Criterion-persona alignment (seg3)")
    md.append("- **Anti-collapse (差异化)**: Chain diversity (seg2), Criterion uniqueness (seg3)")
    md.append("")
    md.append("详见 `reports/datav3/gpt4o_experiments_summary.md §1`.\n")
    md.append("---\n")

    # Section 2: Per-segment main result tables
    md.append("## 2. 主结果表 (1000-bootstrap 95% CI)\n")

    # seg1
    md.append("### 2.1 seg1 (第一反应 anchor 段)\n")
    md.append("| Baseline | NL specificity ↑ | Persona-NL ↑ | Trait-pred ρ ↑ |")
    md.append("|---|---|---|---|")
    for label, _, _, _ in BASELINES:
        if label not in summaries: continue
        s = summaries[label]
        md.append(f"| {label} | {fmt_ci(s['seg1_nl_specificity'])} | "
                  f"{fmt_ci(s['seg1_persona_align'])} | "
                  f"{fmt_ci(s['seg1_trait_pred_rho'])} |")
    md.append("")

    # seg2
    md.append("### 2.2 seg2 (chain reasoning, multi-state)\n")
    md.append("| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |")
    md.append("|---|---|---|---|---|---|")
    for label, _, _, _ in BASELINES:
        if label not in summaries: continue
        s = summaries[label]
        md.append(f"| {label} | {fmt_ci(s['seg2_macro_TED'])} | "
                  f"{fmt_ci(s['seg2_macro_LCS_F1'])} | "
                  f"{fmt_ci(s['seg2_non_V_recall'])} | "
                  f"{fmt_ci(s['seg2_chain_diversity'])} | "
                  f"{fmt_ci(s['seg2_nl_specificity'])} |")
    md.append("")

    # seg3
    md.append("### 2.3 seg3 (anchor + criterion 段)\n")
    md.append("| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |")
    md.append("|---|---|---|---|")
    for label, _, _, _ in BASELINES:
        if label not in summaries: continue
        s = summaries[label]
        md.append(f"| {label} | {fmt_ci(s['seg3_nl_specificity'])} | "
                  f"{fmt_ci(s['seg3_criterion_uniqueness'])} | "
                  f"{fmt_ci(s['seg3_criterion_persona'])} |")
    md.append("")

    # Section 3: v2_anchor vs B0 significance
    if "v2_anchor" in summaries and "B0_h" in summaries:
        md.append("## 3. v2_anchor vs B0 (zero-shot) 显著性测试\n")
        md.append("★ = 95% CI 不重叠 (likely significant).\n")
        md.append("| Metric | B0 CI | v2_anchor CI | Δ | Sig? |")
        md.append("|---|---|---|---|---|")
        for name, seg, _ in METRICS:
            b0 = summaries["B0_h"][name]
            v2 = summaries["v2_anchor"][name]
            delta = v2["point"] - b0["point"]
            if name in LOWER_BETTER:
                sig = "★" if v2["ci_hi"] < b0["ci_lo"] else ""
            else:
                sig = "★" if v2["ci_lo"] > b0["ci_hi"] else ""
            md.append(f"| {name} | [{b0['ci_lo']:.3f}, {b0['ci_hi']:.3f}] | "
                      f"[{v2['ci_lo']:.3f}, {v2['ci_hi']:.3f}] | "
                      f"{delta:+.3f} | {sig} |")
        md.append("")

    # Section 4: CAPS module ablation
    if "v2_anchor" in summaries and "v2_no_caps" in summaries:
        md.append("## 4. CAPS module 必要性 (paper main claim 关键 ablation)\n")
        md.append("| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |")
        md.append("|---|---|---|---|---|")
        for name, seg, _ in METRICS:
            v2 = summaries["v2_anchor"][name]
            nc = summaries["v2_no_caps"][name]
            delta = nc["point"] - v2["point"]
            if name in LOWER_BETTER:
                sig = "★" if nc["ci_lo"] > v2["ci_hi"] else ""
            else:
                sig = "★" if nc["ci_hi"] < v2["ci_lo"] else ""
            md.append(f"| {name} | [{v2['ci_lo']:.3f}, {v2['ci_hi']:.3f}] | "
                      f"[{nc['ci_lo']:.3f}, {nc['ci_hi']:.3f}] | "
                      f"{delta:+.3f} (Δ towards no_caps) | {sig} |")
        md.append("")

    # Section 5: Winner per metric
    md.append("## 5. Winner per metric (within this model's 7 conditions)\n")
    md.append("| Metric | Winner | Point | CI |")
    md.append("|---|---|---|---|")
    for name, seg, _ in METRICS:
        valid = [b for b, _, _, _ in BASELINES if b in summaries]
        if not valid: continue
        if name in LOWER_BETTER:
            best = min(valid, key=lambda b: summaries[b][name]["point"])
        else:
            best = max(valid, key=lambda b: summaries[b][name]["point"])
        s = summaries[best][name]
        md.append(f"| {name} | **{best}** | {s['point']:.3f} | "
                  f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] |")
    md.append("")

    # Section 6: Paper claims (template)
    md.append("## 6. Paper Main Claims for this model\n")
    md.append("Place numerical claims here based on §3-§5 above. Template:\n")
    if "v2_anchor" in summaries and "B0_h" in summaries:
        v2_nv = summaries["v2_anchor"]["seg2_non_V_recall"]
        b0_nv = summaries["B0_h"]["seg2_non_V_recall"]
        v2_div = summaries["v2_anchor"]["seg2_chain_diversity"]
        b0_div = summaries["B0_h"]["seg2_chain_diversity"]
        v2_cu = summaries["v2_anchor"]["seg3_criterion_uniqueness"]
        b0_cu = summaries["B0_h"]["seg3_criterion_uniqueness"]
        md.append(f"```")
        md.append(f"On {args.display_name}, our v2_anchor prompt achieves vs zero-shot:")
        md.append(f"  - seg2 non-V recall: {v2_nv['point']:.1%} [{v2_nv['ci_lo']:.1%}, {v2_nv['ci_hi']:.1%}] vs {b0_nv['point']:.1%} [{b0_nv['ci_lo']:.1%}, {b0_nv['ci_hi']:.1%}]")
        md.append(f"  - seg2 chain diversity: {v2_div['point']:.3f} [{v2_div['ci_lo']:.3f}, {v2_div['ci_hi']:.3f}] vs {b0_div['point']:.3f}")
        md.append(f"  - seg3 criterion uniqueness: {v2_cu['point']:.3f} [{v2_cu['ci_lo']:.3f}, {v2_cu['ci_hi']:.3f}] vs {b0_cu['point']:.3f}")
        md.append(f"```")
    md.append("")

    # Section 7: Reproducibility
    md.append("## 7. Reproducibility\n")
    md.append(f"- Model ID: `{args.model_id}`")
    md.append(f"- Run command:")
    md.append(f"```")
    md.append(f"# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):")
    md.append(f"python scripts/run_b6_sample.py --variant <V> --n 0 \\")
    md.append(f"    --model {args.model_id} --output reports/datav3/crossmodel/{args.model_slug}_<COND>_inference.jsonl")
    md.append(f"# Then PKVC extract via GPT-4o:")
    md.append(f"python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>")
    md.append(f"# Then bootstrap eval:")
    md.append(f"python scripts/generate_model_summary.py --model-slug {args.model_slug} \\")
    md.append(f"    --display-name '{args.display_name}' --model-id {args.model_id} \\")
    md.append(f"    --output {args.output}")
    md.append(f"```")
    md.append("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
