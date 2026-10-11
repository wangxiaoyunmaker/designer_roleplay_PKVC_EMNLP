"""seg2 chain segment: 重新设计 path fidelity metrics, 去除 V-majority + mode collapse artifact.

4 个新 metric:
  1. Macro-averaged path metrics: mean over (P-GT, K-GT, V-GT, C-GT) groups
  2. Chain diversity: 1 - modal_freq, entropy
  3. Per-designer specificity: sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j))
  4. Trait-chain Spearman correlation

Output: reports/datav3/seg2_role_play_metrics.md
"""
from __future__ import annotations
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASELINES = [
    ("B0_h", "reports/datav3/ladder_b0_h_inference.jsonl",
              "reports/datav3/ladder_b0_h_pkvc_eval.jsonl", None),
    ("B1_h", "reports/datav3/ladder_b1_h_inference.jsonl",
              "reports/datav3/ladder_b1_h_pkvc_eval.jsonl", None),
    ("B2_h", "reports/datav3/ladder_b2_h_inference.jsonl",
              "reports/datav3/ladder_b2_h_pkvc_eval.jsonl", None),
    ("B3_h", "reports/datav3/ladder_b3_h_inference.jsonl",
              "reports/datav3/ladder_b3_h_pkvc_eval.jsonl", None),
    ("v2_hybrid", "reports/datav3/b6_v2hybrid_full_inference.jsonl",
                  "reports/datav3/b6_v2hybrid_full_pkvc_eval.jsonl", None),
    ("v2_anchor", "reports/datav3/b6_v2_full_inference.jsonl",
                  "reports/datav3/b6_v2_full_pkvc_eval.jsonl", None),
    ("B6_h", "reports/datav3/b6_h_full_inference.jsonl",
              "reports/datav3/b6_h_full_pkvc_eval.jsonl", None),
    ("B5", "reports/datav3/baselines_q7_main_final_inference.jsonl",
            "reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl", "B5"),
]


def load_jsonl(path, baseline=None):
    rs = [json.loads(l) for l in Path(path).open() if l.strip()]
    if baseline:
        rs = [r for r in rs if r.get("baseline") == baseline]
    return [r for r in rs if r.get("ok")]


def load_trait_vectors(path: Path):
    raw = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw[r["user_id"]][r["question_node"]] = float(r["score"])
    traits = sorted(next(iter(raw.values())).keys())
    return traits, {uid: np.array([s.get(t, 3.5) for t in traits]) for uid, s in raw.items()}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na*nb)) if (na*nb) else 0.0


# ============================================================
# Path metric primitives
# ============================================================

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


def lcs_f1(a, b):
    if not a or not b: return 0.0
    L = lcs_length(a, b)
    prec, rec = L/len(b), L/len(a)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def sta(gt, pred):
    if len(gt) < 2:
        return 1.0 if len(pred) == len(gt) else 0.0
    tgt = [(gt[i], gt[i+1]) for i in range(len(gt)-1)]
    tpred = set((pred[i], pred[i+1]) for i in range(len(pred)-1))
    return sum(1 for t in tgt if t in tpred) / len(tgt)


def collect_pairs(inf_path, pkvc_path, baseline_filter, seg="Q7_seg2"):
    inf = load_jsonl(inf_path, baseline_filter)
    pkvc = load_jsonl(pkvc_path, baseline_filter)
    pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc}

    pairs = []
    for r in inf:
        if r["scenario"] != seg: continue
        gt = r.get("ground_truth_pkvc",{}).get("sequence", [])
        if not gt: continue
        pkvc_row = pkvc_map.get((r["designer_id"], seg))
        if not pkvc_row: continue
        pred = pkvc_row["pred"].get("state_sequence", [])
        if not pred: continue
        pairs.append({"designer_id": r["designer_id"], "gt": gt, "pred": pred, "gt_first": gt[0]})
    return pairs


# ============================================================
# Metric 1: Macro-averaged path metrics
# ============================================================

def macro_path_metrics(pairs):
    """Mean over (P-first, K-first, V-first, C-first) groups."""
    groups = defaultdict(list)
    for p in pairs:
        groups[p["gt_first"]].append(p)
    out = {}
    for cls in "PKVC":
        gp = groups.get(cls, [])
        if not gp:
            out[cls] = {"n": 0, "TED": None, "STA": None, "PathJacc": None, "LCS_F1": None}
            continue
        n = len(gp)
        out[cls] = {
            "n": n,
            "TED": sum(edit_distance(p["gt"], p["pred"]) for p in gp) / n,
            "STA": sum(sta(p["gt"], p["pred"]) for p in gp) / n,
            "PathJacc": sum(jaccard(p["gt"], p["pred"]) for p in gp) / n,
            "LCS_F1": sum(lcs_f1(p["gt"], p["pred"]) for p in gp) / n,
        }
    # Macro: mean over classes with support
    sup_classes = [c for c in "PKVC" if out[c]["n"] > 0]
    out["macro"] = {
        "TED": sum(out[c]["TED"] for c in sup_classes) / len(sup_classes),
        "STA": sum(out[c]["STA"] for c in sup_classes) / len(sup_classes),
        "PathJacc": sum(out[c]["PathJacc"] for c in sup_classes) / len(sup_classes),
        "LCS_F1": sum(out[c]["LCS_F1"] for c in sup_classes) / len(sup_classes),
        "n_classes": len(sup_classes),
    }
    return out


# ============================================================
# Metric 2: Chain diversity
# ============================================================

def chain_diversity(pairs):
    chain_counts = Counter(tuple(p["pred"]) for p in pairs)
    n = sum(chain_counts.values())
    modal_freq = chain_counts.most_common(1)[0][1] / n
    # Entropy
    probs = np.array([c/n for c in chain_counts.values()])
    ent = float(-(probs * np.log2(probs + 1e-12)).sum())
    return {
        "modal_freq": modal_freq,
        "diversity": 1.0 - modal_freq,
        "entropy_bits": ent,
        "n_unique_chains": len(chain_counts),
        "n_total": n,
    }


# ============================================================
# Metric 3: Per-designer specificity (jaccard-based)
# ============================================================

def per_designer_specificity(pairs):
    """For each designer i: sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)).
    Use Jaccard of state sets as similarity.
    spec_i > 1: pred_i specifically matches GT_i more than generic.
    spec_i ≈ 1: no specificity (mode collapse).
    """
    if len(pairs) < 2:
        return {"mean_spec": 0.0, "median_spec": 0.0, "n_above_1": 0}
    gts = [p["gt"] for p in pairs]
    specs = []
    for i, p in enumerate(pairs):
        own_sim = jaccard(p["pred"], gts[i])
        other_sims = [jaccard(p["pred"], gts[j]) for j in range(len(pairs)) if j != i]
        avg_other = sum(other_sims) / len(other_sims) if other_sims else 1e-9
        if avg_other == 0:
            spec = own_sim / 1e-9 if own_sim > 0 else 1.0
        else:
            spec = own_sim / avg_other
        specs.append(spec)
    specs_arr = np.array(specs)
    return {
        "mean_spec": float(specs_arr.mean()),
        "median_spec": float(np.median(specs_arr)),
        "n_above_1": int((specs_arr > 1.0).sum()),
        "n_total": len(specs),
    }


# ============================================================
# Metric 4: Trait-chain similarity correlation
# ============================================================

def trait_chain_correlation(pairs, trait_vectors):
    """For pairs of designers, compute trait_sim and chain_sim.
    Spearman correlation across all pairs."""
    valid = [p for p in pairs if p["designer_id"] in trait_vectors]
    n = len(valid)
    if n < 2: return {"rho": 0.0, "n_pairs": 0}
    trait_sims = []
    chain_sims = []
    for i in range(n):
        for j in range(i+1, n):
            d1, d2 = valid[i]["designer_id"], valid[j]["designer_id"]
            t_sim = cosine(trait_vectors[d1], trait_vectors[d2])
            c_sim = jaccard(valid[i]["pred"], valid[j]["pred"])
            trait_sims.append(t_sim); chain_sims.append(c_sim)
    trait_sims = np.array(trait_sims); chain_sims = np.array(chain_sims)
    if trait_sims.std() == 0 or chain_sims.std() == 0:
        return {"rho": 0.0, "n_pairs": len(trait_sims)}
    rho, _ = spearmanr(trait_sims, chain_sims)
    return {
        "rho": float(rho if not np.isnan(rho) else 0.0),
        "n_pairs": len(trait_sims),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("[load] trait vectors...")
    _, trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))

    md = ["# seg2 chain segment: 重新设计 path fidelity metrics\n"]
    md.append("4 个新 metric (替代 V-majority + mode collapse biased 旧 metric):\n")
    md.append("1. **Macro-averaged path metrics**: 按 GT first letter 分 P/K/V/C 4 组, 各组均值, 再 macro 平均")
    md.append("2. **Chain diversity**: 1 - modal_freq + entropy. 量化是否真在 differentiate.")
    md.append("3. **Per-designer specificity**: sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)). > 1 = role-play.")
    md.append("4. **Trait-chain Spearman ρ**: 相似 trait → 相似 chain pred. 直接 role-play 实证.\n")

    # ============================================================
    # Metric 1
    # ============================================================
    md.append("## 1. Macro-averaged path metrics (按 GT first letter 分 P/K/V/C)\n")
    md.append("| Baseline | macro-TED ↓ | macro-STA ↑ | macro-PathJacc ↑ | macro-LCS-F1 ↑ |")
    md.append("|---|---|---|---|---|")
    results = {}
    for label, inf, pkvc, bl in BASELINES:
        pairs = collect_pairs(inf, pkvc, bl)
        if not pairs:
            continue
        m = macro_path_metrics(pairs)
        results[label] = (pairs, m)
        macro = m["macro"]
        md.append(f"| {label} | {macro['TED']:.3f} | {macro['STA']:.3f} | "
                  f"{macro['PathJacc']:.3f} | {macro['LCS_F1']:.3f} |")
    md.append("")

    # Per-class table
    md.append("### Per-class breakdown (TED ↓)\n")
    md.append("| Baseline | P (n) | K (n) | V (n) | C (n) |")
    md.append("|---|---|---|---|---|")
    for label, (pairs, m) in results.items():
        cells = []
        for c in "PKVC":
            v = m[c]
            if v["n"] == 0:
                cells.append("—")
            else:
                cells.append(f"{v['TED']:.2f} ({v['n']})")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("### Per-class breakdown (STA ↑)\n")
    md.append("| Baseline | P | K | V | C |")
    md.append("|---|---|---|---|---|")
    for label, (pairs, m) in results.items():
        cells = []
        for c in "PKVC":
            v = m[c]
            cells.append(f"{v['STA']:.3f}" if v["n"] > 0 else "—")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    # ============================================================
    # Metric 2: Chain diversity
    # ============================================================
    md.append("## 2. Chain diversity (mode collapse 指标)\n")
    md.append("| Baseline | Modal freq ↓ | Diversity (1-modal) ↑ | Entropy bits ↑ | #unique chains |")
    md.append("|---|---|---|---|---|")
    for label, (pairs, m) in results.items():
        d = chain_diversity(pairs)
        md.append(f"| {label} | {d['modal_freq']:.2%} | {d['diversity']:.3f} | "
                  f"{d['entropy_bits']:.2f} | {d['n_unique_chains']} / {d['n_total']} |")
    md.append("")

    # ============================================================
    # Metric 3: Per-designer specificity
    # ============================================================
    md.append("## 3. Per-designer specificity (sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)))\n")
    md.append("Mean / Median > 1 表示 model 输出 specifically 给 designer i (role-play). ≈ 1 表示 generic.\n")
    md.append("| Baseline | Mean spec | Median spec | #(spec > 1) / total |")
    md.append("|---|---|---|---|")
    for label, (pairs, m) in results.items():
        s = per_designer_specificity(pairs)
        md.append(f"| {label} | {s['mean_spec']:.3f} | {s['median_spec']:.3f} | "
                  f"{s['n_above_1']} / {s['n_total']} ({s['n_above_1']/s['n_total']:.0%}) |")
    md.append("")

    # ============================================================
    # Metric 4: Trait-chain correlation
    # ============================================================
    md.append("## 4. Trait-chain Spearman correlation\n")
    md.append("相似 30维 trait → 相似 chain pred. 直接 role-play 实证.\n")
    md.append("| Baseline | Spearman ρ | n_pairs |")
    md.append("|---|---|---|")
    for label, (pairs, m) in results.items():
        c = trait_chain_correlation(pairs, trait_vectors)
        md.append(f"| {label} | {c['rho']:+.4f} | {c['n_pairs']} |")
    md.append("")

    out = Path("reports/datav3/seg2_role_play_metrics.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out}\n")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
