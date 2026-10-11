"""seg2 final metric suite — PKVC sequence + NL.

PKVC sequence (6):
  1. macro-TED ↓
  2. macro-STA ↑
  3. macro-PathJacc ↑
  4. macro-LCS-F1 ↑
  5. Chain diversity ↑ (1 - modal_freq)
  6. Trait-chain Spearman ρ ↑

NL (4):
  7. Macro-semantic cosine ↑ (按 GT class)
  8. NL specificity ratio ↑ (sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)))
  9. Persona-NL alignment ↑ (cosine(pred_NL, designer_Q1-Q6_NL_center))
  10. Supplement reference cosine ↑ (cosine(pred_NL, designer_Q3-Q6_supplements))

Output: reports/datav3/seg2_final_metrics.md
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EMBED_CACHE = Path("data_v3/processed/embedding_cache_text_embed_3_small.jsonl")


def text_key(t): return hashlib.md5(t.encode("utf-8")).hexdigest()


def load_cache():
    cache = {}
    if EMBED_CACHE.exists():
        for line in EMBED_CACHE.open():
            r = json.loads(line)
            cache[r["key"]] = r["vec"]
    return cache


def get_emb(text, cache):
    v = cache.get(text_key(text))
    return np.array(v) if v else None


def embed_missing(texts, cache):
    from openai import OpenAI
    unique = list(dict.fromkeys(texts))
    missing = [t for t in unique if text_key(t) not in cache]
    if not missing:
        return
    print(f"  embedding {len(missing)} new texts...", flush=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"))
    new_records = []
    BATCH = 200
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i+BATCH]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for txt, item in zip(batch, resp.data):
            cache[text_key(txt)] = item.embedding
            new_records.append({"key": text_key(txt), "vec": item.embedding})
    with EMBED_CACHE.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cos(a, b):
    if a is None or b is None: return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na*nb)) if (na*nb) else 0.0


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


def lcs_f1(a, b):
    if not a or not b: return 0.0
    L = lcs_length(a, b)
    prec, rec = L/len(b), L/len(a)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0


def sta(gt, pred):
    if len(gt) < 2: return 1.0 if len(pred) == len(gt) else 0.0
    tgt = [(gt[i], gt[i+1]) for i in range(len(gt)-1)]
    tpred = set((pred[i], pred[i+1]) for i in range(len(pred)-1))
    return sum(1 for t in tgt if t in tpred) / len(tgt)


BASELINES = [
    ("B0_h",        "reports/datav3/ladder_b0_h_inference.jsonl",        "reports/datav3/ladder_b0_h_pkvc_eval.jsonl",        None),
    ("B1_h",        "reports/datav3/ladder_b1_h_inference.jsonl",        "reports/datav3/ladder_b1_h_pkvc_eval.jsonl",        None),
    ("B2_h",        "reports/datav3/ladder_b2_h_inference.jsonl",        "reports/datav3/ladder_b2_h_pkvc_eval.jsonl",        None),
    ("B3_h",        "reports/datav3/ladder_b3_h_inference.jsonl",        "reports/datav3/ladder_b3_h_pkvc_eval.jsonl",        None),
    ("v2_hybrid",   "reports/datav3/b6_v2hybrid_full_inference.jsonl",   "reports/datav3/b6_v2hybrid_full_pkvc_eval.jsonl",   None),
    ("v2_anchor",   "reports/datav3/b6_v2_full_inference.jsonl",         "reports/datav3/b6_v2_full_pkvc_eval.jsonl",         None),
    ("B6_h",        "reports/datav3/b6_h_full_inference.jsonl",          "reports/datav3/b6_h_full_pkvc_eval.jsonl",          None),
    ("B5",          "reports/datav3/baselines_q7_main_final_inference.jsonl", "reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl", "B5"),
]


def load_jsonl(path, baseline=None):
    rs = [json.loads(l) for l in Path(path).open() if l.strip()]
    if baseline:
        rs = [r for r in rs if r.get("baseline") == baseline]
    return [r for r in rs if r.get("ok")]


def load_trait_vectors(path):
    raw = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw[r["user_id"]][r["question_node"]] = float(r["score"])
    traits = sorted(next(iter(raw.values())).keys())
    return {uid: np.array([s.get(t, 3.5) for t in traits]) for uid, s in raw.items()}


def load_supplements(task_csv):
    """Load designer Q3-Q6 supplements concatenated per designer."""
    out = defaultdict(list)
    with task_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["question_id"] in ("Q3","Q4","Q5","Q6") and r["text_supplement"].strip():
                out[r["user_id"]].append(r["text_supplement"].strip())
    return {uid: "\n".join(s) for uid, s in out.items()}


def collect_seg2(inf_path, pkvc_path, bl):
    inf = load_jsonl(inf_path, bl)
    pkvc = load_jsonl(pkvc_path, bl)
    pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc}
    pairs = []
    for r in inf:
        if r["scenario"] != "Q7_seg2": continue
        gt_seq = r.get("ground_truth_pkvc",{}).get("sequence", [])
        gt_nl = r.get("ground_truth_nl", "")
        pkvc_row = pkvc_map.get((r["designer_id"], "Q7_seg2"))
        if not (gt_seq and gt_nl and pkvc_row): continue
        pred_seq = pkvc_row["pred"].get("state_sequence", [])
        if not pred_seq: continue
        pairs.append({
            "designer_id": r["designer_id"],
            "gt_seq": gt_seq, "pred_seq": pred_seq,
            "gt_nl": gt_nl, "pred_nl": r["simulated_nl"],
            "gt_first": gt_seq[0],
        })
    return pairs


def main():
    print("[load] data sources...")
    trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    supplements = load_supplements(Path("data_v3/task_responses.csv"))
    print(f"  {len(trait_vectors)} trait, {len(supplements)} supplements")

    # Q1-Q6 NL for persona center
    q16_rows = [json.loads(l) for l in
                Path("data_v3/processed/q1q6_extracted_v8.7_full.jsonl").open() if l.strip()]
    designer_q16_nls = defaultdict(list)
    for r in q16_rows:
        if r.get("ok") and r.get("text"):
            designer_q16_nls[r["user_id"]].append(r["text"])

    cache = load_cache()
    print(f"  cache: {len(cache)} entries")

    # Embed missing (supplements + pred_NL + GT_NL — most should be cached)
    texts_to_embed = list(supplements.values())
    for nls in designer_q16_nls.values():
        texts_to_embed.extend(nls)
    print(f"[embed] checking {len(set(texts_to_embed))} unique texts (supplements + Q1-Q6 NL)...")
    embed_missing(texts_to_embed, cache)

    # Collect all pred_NL + GT_NL across baselines too
    all_baseline_texts = []
    all_pairs_by_baseline = {}
    for label, inf_p, pkvc_p, bl in BASELINES:
        pairs = collect_seg2(inf_p, pkvc_p, bl)
        all_pairs_by_baseline[label] = pairs
        for p in pairs:
            all_baseline_texts.append(p["pred_nl"])
            all_baseline_texts.append(p["gt_nl"])
    print(f"[embed] checking {len(set(all_baseline_texts))} unique NL texts...")
    embed_missing(all_baseline_texts, cache)
    print(f"  cache: {len(cache)} entries (after embed)")

    # Persona centers
    persona_centers = {}
    for did, nls in designer_q16_nls.items():
        embs = [get_emb(nl, cache) for nl in nls]
        embs = [e for e in embs if e is not None]
        if embs:
            persona_centers[did] = np.mean(embs, axis=0)
    print(f"  {len(persona_centers)} persona centers")

    # Supplement embeddings
    supplement_embs = {did: get_emb(s, cache) for did, s in supplements.items()}
    supplement_embs = {k: v for k, v in supplement_embs.items() if v is not None}
    print(f"  {len(supplement_embs)} supplement embeddings")

    # ============================================================
    # Compute all metrics per baseline
    # ============================================================
    md = ["# seg2 final metric suite (PKVC sequence + NL)\n"]
    md.append(f"Baselines compared: {len(BASELINES)} | Designer pairs: 162 (with GT seq + NL).\n")

    # Build flat row per baseline
    rows = []
    for label, _, _, _ in BASELINES:
        pairs = all_pairs_by_baseline[label]
        if not pairs:
            continue

        # ========== PKVC sequence ==========
        # 1-4. Macro path metrics
        groups = defaultdict(list)
        for p in pairs:
            groups[p["gt_first"]].append(p)
        per_class = {}
        for c in "PKVC":
            gp = groups.get(c, [])
            if not gp:
                per_class[c] = None; continue
            n = len(gp)
            per_class[c] = {
                "n": n,
                "TED": sum(edit_distance(p["gt_seq"], p["pred_seq"]) for p in gp) / n,
                "STA": sum(sta(p["gt_seq"], p["pred_seq"]) for p in gp) / n,
                "Jacc": sum(jaccard(p["gt_seq"], p["pred_seq"]) for p in gp) / n,
                "LCS": sum(lcs_f1(p["gt_seq"], p["pred_seq"]) for p in gp) / n,
            }
        sup = [c for c in "PKVC" if per_class[c]]
        macro_ted = sum(per_class[c]["TED"] for c in sup) / len(sup)
        macro_sta = sum(per_class[c]["STA"] for c in sup) / len(sup)
        macro_jacc = sum(per_class[c]["Jacc"] for c in sup) / len(sup)
        macro_lcs = sum(per_class[c]["LCS"] for c in sup) / len(sup)

        # 5. Chain diversity
        chain_counter = Counter(tuple(p["pred_seq"]) for p in pairs)
        n_total = sum(chain_counter.values())
        modal_freq = chain_counter.most_common(1)[0][1] / n_total
        diversity = 1 - modal_freq
        probs = np.array([c/n_total for c in chain_counter.values()])
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())

        # 6. Trait-chain Spearman ρ
        valid = [p for p in pairs if p["designer_id"] in trait_vectors]
        t_sims, c_sims = [], []
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                t_sims.append(cos(trait_vectors[valid[i]["designer_id"]],
                                 trait_vectors[valid[j]["designer_id"]]))
                c_sims.append(jaccard(valid[i]["pred_seq"], valid[j]["pred_seq"]))
        t_sims = np.array(t_sims); c_sims = np.array(c_sims)
        if t_sims.std() == 0 or c_sims.std() == 0:
            trait_chain_rho = 0.0
        else:
            rho, _ = spearmanr(t_sims, c_sims)
            trait_chain_rho = float(rho if not np.isnan(rho) else 0.0)

        # ========== NL metrics ==========
        # 7. Macro-semantic cosine (按 GT class)
        per_class_sem = {}
        for c in "PKVC":
            gp = groups.get(c, [])
            if not gp:
                per_class_sem[c] = None; continue
            sims = [cos(get_emb(p["pred_nl"], cache), get_emb(p["gt_nl"], cache)) for p in gp]
            per_class_sem[c] = {"n": len(sims), "sem": sum(sims)/len(sims)}
        macro_sem = sum(per_class_sem[c]["sem"] for c in sup) / len(sup)

        # 8. NL specificity ratio
        gt_nl_embs = [get_emb(p["gt_nl"], cache) for p in pairs]
        pred_nl_embs = [get_emb(p["pred_nl"], cache) for p in pairs]
        specs = []
        for i, p in enumerate(pairs):
            if pred_nl_embs[i] is None: continue
            own_sim = cos(pred_nl_embs[i], gt_nl_embs[i])
            other_sims = [cos(pred_nl_embs[i], gt_nl_embs[j]) for j in range(len(pairs)) if j != i and gt_nl_embs[j] is not None]
            if not other_sims:
                continue
            avg_other = sum(other_sims) / len(other_sims)
            if avg_other > 0:
                specs.append(own_sim / avg_other)
        mean_spec_nl = sum(specs) / len(specs) if specs else 0.0
        n_above_1 = sum(1 for s in specs if s > 1)

        # 9. Persona-NL alignment
        persona_aligns = []
        for p in pairs:
            pne = get_emb(p["pred_nl"], cache)
            pc = persona_centers.get(p["designer_id"])
            if pne is None or pc is None: continue
            persona_aligns.append(cos(pne, pc))
        persona_align = sum(persona_aligns)/len(persona_aligns) if persona_aligns else 0.0

        # 10. Supplement reference cosine
        sup_refs = []
        for p in pairs:
            pne = get_emb(p["pred_nl"], cache)
            se = supplement_embs.get(p["designer_id"])
            if pne is None or se is None: continue
            sup_refs.append(cos(pne, se))
        sup_ref = sum(sup_refs)/len(sup_refs) if sup_refs else 0.0

        rows.append({
            "label": label,
            "macro_TED": macro_ted, "macro_STA": macro_sta,
            "macro_Jacc": macro_jacc, "macro_LCS": macro_lcs,
            "diversity": diversity, "entropy": entropy, "modal_freq": modal_freq,
            "trait_chain_rho": trait_chain_rho,
            "macro_sem": macro_sem,
            "nl_specificity": mean_spec_nl, "n_above_1": n_above_1, "n_pairs": len(specs),
            "persona_align": persona_align,
            "supplement_ref": sup_ref,
            "per_class": per_class, "per_class_sem": per_class_sem,
        })

    # ============================================================
    # Output
    # ============================================================
    md.append("## PKVC sequence metrics (按 GT class 分组取均值)\n")
    md.append("| Baseline | macro-TED ↓ | macro-STA ↑ | macro-Jacc ↑ | macro-LCS ↑ |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['label']} | {r['macro_TED']:.3f} | {r['macro_STA']:.3f} | "
                  f"{r['macro_Jacc']:.3f} | {r['macro_LCS']:.3f} |")

    md.append("\n## Differentiation (PKVC sequence)\n")
    md.append("| Baseline | Chain diversity ↑ | Modal freq ↓ | Entropy bits ↑ | Trait-chain ρ ↑ |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['label']} | {r['diversity']:.3f} | {r['modal_freq']:.2%} | "
                  f"{r['entropy']:.2f} | {r['trait_chain_rho']:+.4f} |")

    md.append("\n## NL metrics (4 个)\n")
    md.append("| Baseline | macro-semantic ↑ | NL specificity ↑ | Persona-NL ↑ | Supplement ref ↑ |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['label']} | {r['macro_sem']:.3f} | {r['nl_specificity']:.3f} ({r['n_above_1']}/{r['n_pairs']}) | "
                  f"{r['persona_align']:.3f} | {r['supplement_ref']:.3f} |")

    md.append("\n## Per-class NL semantic (按 GT first letter)\n")
    md.append("| Baseline | P | K | V | C |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        cells = []
        for c in "PKVC":
            v = r["per_class_sem"].get(c)
            cells.append(f"{v['sem']:.3f} (n={v['n']})" if v else "—")
        md.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    md.append("\n## Summary — winner per metric\n")
    metrics_to_rank = [
        ("macro-TED ↓", "macro_TED", "lower"),
        ("macro-STA ↑", "macro_STA", "higher"),
        ("macro-Jacc ↑", "macro_Jacc", "higher"),
        ("macro-LCS ↑", "macro_LCS", "higher"),
        ("Chain diversity ↑", "diversity", "higher"),
        ("Trait-chain ρ ↑", "trait_chain_rho", "higher"),
        ("macro-semantic ↑", "macro_sem", "higher"),
        ("NL specificity ↑", "nl_specificity", "higher"),
        ("Persona-NL ↑", "persona_align", "higher"),
        ("Supplement ref ↑", "supplement_ref", "higher"),
    ]
    md.append("| Metric | Winner |")
    md.append("|---|---|")
    for name, key, direction in metrics_to_rank:
        if direction == "lower":
            best = min(rows, key=lambda r: r[key])
        else:
            best = max(rows, key=lambda r: r[key])
        md.append(f"| {name} | **{best['label']}** ({best[key]:.4f}) |")

    out = Path("reports/datav3/seg2_final_metrics.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out}")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
