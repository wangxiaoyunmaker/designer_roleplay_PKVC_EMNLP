"""Apply 4 NL metrics + role-play metrics to seg1 and seg3.

Common metrics (per segment):
  NL-1. Macro-semantic cosine (by GT class)
  NL-2. NL specificity ratio
  NL-3. Persona-NL alignment
  NL-4. Supplement reference cosine

seg1 extras:
  - Distribution KL vs GT (lower = better match)
  - Prediction entropy (higher = differentiation)
  - Trait-prediction Spearman ρ

seg3 extras:
  - Criterion uniqueness (pairwise distance)
  - Criterion-persona alignment

Output: reports/datav3/seg1_seg3_nl_metrics.md
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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
    unique = list(dict.fromkeys([t for t in texts if t]))
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


BASELINES = [
    ("B0_h",        "reports/datav3/ladder_b0_h_inference.jsonl",        None),
    ("B1_h",        "reports/datav3/ladder_b1_h_inference.jsonl",        None),
    ("B2_h",        "reports/datav3/ladder_b2_h_inference.jsonl",        None),
    ("B3_h",        "reports/datav3/ladder_b3_h_inference.jsonl",        None),
    ("v2_hybrid",   "reports/datav3/b6_v2hybrid_full_inference.jsonl",   None),
    ("v2_anchor",   "reports/datav3/b6_v2_full_inference.jsonl",         None),
    ("B6_h",        "reports/datav3/b6_h_full_inference.jsonl",          None),
    ("B5",          "reports/datav3/baselines_q7_main_final_inference.jsonl", "B5"),
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
    out = defaultdict(list)
    with task_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["question_id"] in ("Q3","Q4","Q5","Q6") and r["text_supplement"].strip():
                out[r["user_id"]].append(r["text_supplement"].strip())
    return {uid: "\n".join(s) for uid, s in out.items()}


def collect_seg(inf_path, bl, seg_name):
    inf = load_jsonl(inf_path, bl)
    pairs = []
    for r in inf:
        if r["scenario"] != seg_name: continue
        gt_dom = (r.get("ground_truth_pkvc") or {}).get("dominant", "")
        gt_nl = r.get("ground_truth_nl", "")
        pred_dom = (r.get("claimed_pkvc") or {}).get("dominant_factor", "")
        pred_nl = r.get("simulated_nl", "")
        criterion = (r.get("claimed_pkvc") or {}).get("criterion", "")
        if not (gt_dom and gt_nl and pred_dom and pred_nl): continue
        pairs.append({
            "designer_id": r["designer_id"],
            "gt_dom": gt_dom[0], "pred_dom": pred_dom[0],
            "gt_nl": gt_nl, "pred_nl": pred_nl,
            "criterion": criterion,
        })
    return pairs


def class_distribution(letters, smooth=0.01):
    counter = Counter(letters)
    dist = np.array([counter.get(c, 0) for c in "PKVC"], dtype=float) + smooth
    return dist / dist.sum()


def kl(p, q): return float(np.sum(p * np.log(p / q)))
def entropy(p): return float(-np.sum(p * np.log2(p + 1e-12)))


def main():
    print("[load] data sources...")
    trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    supplements = load_supplements(Path("data_v3/task_responses.csv"))
    cache = load_cache()
    print(f"  trait={len(trait_vectors)}, supplements={len(supplements)}, cache={len(cache)}")

    # Q1-Q6 NL for persona center
    q16_rows = [json.loads(l) for l in
                Path("data_v3/processed/q1q6_extracted_v8.7_full.jsonl").open() if l.strip()]
    designer_q16_nls = defaultdict(list)
    for r in q16_rows:
        if r.get("ok") and r.get("text"):
            designer_q16_nls[r["user_id"]].append(r["text"])

    # Embed (NL, supplements, Q1-Q6 NL)
    all_texts = list(supplements.values())
    for nls in designer_q16_nls.values():
        all_texts.extend(nls)
    for label, inf_p, bl in BASELINES:
        rs = load_jsonl(inf_p, bl)
        for r in rs:
            if r.get("simulated_nl"): all_texts.append(r["simulated_nl"])
            if r.get("ground_truth_nl"): all_texts.append(r["ground_truth_nl"])
            crit = (r.get("claimed_pkvc") or {}).get("criterion", "")
            if crit: all_texts.append(crit)
    print(f"[embed] checking {len(set(all_texts))} unique texts...")
    embed_missing(all_texts, cache)
    print(f"  cache: {len(cache)}")

    # Persona centers
    persona = {}
    for did, nls in designer_q16_nls.items():
        embs = [get_emb(nl, cache) for nl in nls]
        embs = [e for e in embs if e is not None]
        if embs: persona[did] = np.mean(embs, axis=0)
    supplement_embs = {did: get_emb(s, cache) for did, s in supplements.items()}
    supplement_embs = {k: v for k, v in supplement_embs.items() if v is not None}
    print(f"  persona={len(persona)}, supplement_embs={len(supplement_embs)}")

    md = ["# seg1 + seg3 NL metric eval\n"]

    for seg_name in ("Q7_seg1", "Q7_seg3"):
        md.append(f"\n## ============ {seg_name} ============\n")

        # Collect per baseline
        baseline_data = {}
        for label, inf_p, bl in BASELINES:
            baseline_data[label] = collect_seg(inf_p, bl, seg_name)

        # GT distribution (use any baseline's GT)
        gt_letters = [p["gt_dom"] for p in baseline_data["B0_h"]]
        gt_dist = class_distribution(gt_letters)
        md.append(f"\nGT distribution: P={gt_dist[0]:.1%} K={gt_dist[1]:.1%} V={gt_dist[2]:.1%} C={gt_dist[3]:.1%} | "
                  f"entropy={entropy(gt_dist):.3f}\n")

        # NL metrics + dist KL + entropy + trait-pred ρ
        rows = []
        for label in baseline_data:
            pairs = baseline_data[label]
            if not pairs: continue

            # Distribution KL + entropy
            preds = [p["pred_dom"] for p in pairs]
            pred_dist = class_distribution(preds)
            d_kl = kl(pred_dist, gt_dist)
            d_ent = entropy(pred_dist)

            # Trait-pred ρ
            valid = [p for p in pairs if p["designer_id"] in trait_vectors]
            t_sims, p_match = [], []
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    t_sims.append(cos(trait_vectors[valid[i]["designer_id"]],
                                     trait_vectors[valid[j]["designer_id"]]))
                    p_match.append(1 if valid[i]["pred_dom"] == valid[j]["pred_dom"] else 0)
            t_sims, p_match = np.array(t_sims), np.array(p_match)
            if t_sims.std() == 0 or p_match.std() == 0:
                trait_rho = 0.0
            else:
                rho, _ = spearmanr(t_sims, p_match)
                trait_rho = float(rho if not np.isnan(rho) else 0.0)

            # Group by GT class for macro-semantic
            groups = defaultdict(list)
            for p in pairs:
                groups[p["gt_dom"]].append(p)
            per_class_sem = {}
            for c in "PKVC":
                gp = groups.get(c, [])
                if not gp:
                    per_class_sem[c] = None; continue
                sims = [cos(get_emb(p["pred_nl"], cache), get_emb(p["gt_nl"], cache)) for p in gp]
                per_class_sem[c] = {"n": len(sims), "sem": sum(sims)/len(sims)}
            sup_classes = [c for c in "PKVC" if per_class_sem[c]]
            macro_sem = sum(per_class_sem[c]["sem"] for c in sup_classes) / len(sup_classes)

            # NL specificity ratio
            gt_nl_embs = [get_emb(p["gt_nl"], cache) for p in pairs]
            pred_nl_embs = [get_emb(p["pred_nl"], cache) for p in pairs]
            specs = []
            for i, p in enumerate(pairs):
                if pred_nl_embs[i] is None: continue
                own_sim = cos(pred_nl_embs[i], gt_nl_embs[i])
                other_sims = [cos(pred_nl_embs[i], gt_nl_embs[j]) for j in range(len(pairs))
                              if j != i and gt_nl_embs[j] is not None]
                if not other_sims: continue
                avg_other = sum(other_sims) / len(other_sims)
                if avg_other > 0:
                    specs.append(own_sim / avg_other)
            mean_spec = sum(specs)/len(specs) if specs else 0.0
            n_above_1 = sum(1 for s in specs if s > 1)

            # Persona-NL alignment
            persona_aligns = []
            for p in pairs:
                pne = get_emb(p["pred_nl"], cache)
                pc = persona.get(p["designer_id"])
                if pne is not None and pc is not None:
                    persona_aligns.append(cos(pne, pc))
            persona_align = sum(persona_aligns)/len(persona_aligns) if persona_aligns else 0.0

            # Supplement reference cosine
            sup_refs = []
            for p in pairs:
                pne = get_emb(p["pred_nl"], cache)
                se = supplement_embs.get(p["designer_id"])
                if pne is not None and se is not None:
                    sup_refs.append(cos(pne, se))
            sup_ref = sum(sup_refs)/len(sup_refs) if sup_refs else 0.0

            row = {
                "label": label,
                "kl": d_kl, "entropy": d_ent, "trait_rho": trait_rho,
                "macro_sem": macro_sem,
                "nl_spec": mean_spec, "n_above_1": n_above_1, "n_pairs": len(specs),
                "persona": persona_align, "sup_ref": sup_ref,
                "per_class_sem": per_class_sem,
            }

            # seg3 only: criterion uniqueness + criterion-persona
            if seg_name == "Q7_seg3":
                crits = [p["criterion"] for p in pairs if p["criterion"]]
                if crits and len(crits) >= 2:
                    crit_embs = [get_emb(c, cache) for c in crits]
                    crit_embs = [e for e in crit_embs if e is not None]
                    if len(crit_embs) >= 2:
                        embs = np.array(crit_embs)
                        norms = np.linalg.norm(embs, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        embs_n = embs / norms
                        cos_matrix = embs_n @ embs_n.T
                        iu = np.triu_indices(len(embs), k=1)
                        mean_cos = float(cos_matrix[iu].mean())
                        row["crit_uniq"] = 1 - mean_cos
                # Criterion-persona alignment
                crit_persona = []
                for p in pairs:
                    if not p["criterion"]: continue
                    ce = get_emb(p["criterion"], cache)
                    pc = persona.get(p["designer_id"])
                    if ce is not None and pc is not None:
                        crit_persona.append(cos(ce, pc))
                row["crit_persona"] = sum(crit_persona)/len(crit_persona) if crit_persona else 0.0
                # Criterion-supplement ref
                crit_sup = []
                for p in pairs:
                    if not p["criterion"]: continue
                    ce = get_emb(p["criterion"], cache)
                    se = supplement_embs.get(p["designer_id"])
                    if ce is not None and se is not None:
                        crit_sup.append(cos(ce, se))
                row["crit_sup_ref"] = sum(crit_sup)/len(crit_sup) if crit_sup else 0.0

            rows.append(row)

        # Output table for this seg
        md.append(f"### {seg_name} — 4 NL metrics + dist KL + entropy + trait-pred ρ\n")
        md.append("| Baseline | Dist KL ↓ | Entropy ↑ | Trait-pred ρ ↑ | macro-sem ↑ | NL spec ↑ | Persona-NL ↑ | Supplement ref ↑ |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['label']} | {r['kl']:.4f} | {r['entropy']:.3f} | {r['trait_rho']:+.4f} | "
                      f"{r['macro_sem']:.3f} | {r['nl_spec']:.3f} ({r['n_above_1']}/{r['n_pairs']}) | "
                      f"{r['persona']:.3f} | {r['sup_ref']:.3f} |")

        if seg_name == "Q7_seg3":
            md.append(f"\n### {seg_name} — criterion-specific metrics\n")
            md.append("| Baseline | Criterion uniqueness ↑ | Criterion-persona ↑ | Criterion-supplement ref ↑ |")
            md.append("|---|---|---|---|")
            for r in rows:
                cu = r.get("crit_uniq", 0)
                cp = r.get("crit_persona", 0)
                cs = r.get("crit_sup_ref", 0)
                md.append(f"| {r['label']} | {cu:.4f} | {cp:.4f} | {cs:.4f} |")

        # Per-class breakdown
        md.append(f"\n### {seg_name} — per-class macro-semantic (by GT class)\n")
        md.append("| Baseline | P | K | V | C |")
        md.append("|---|---|---|---|---|")
        for r in rows:
            cells = []
            for c in "PKVC":
                v = r["per_class_sem"].get(c)
                cells.append(f"{v['sem']:.3f}" if v else "—")
            md.append(f"| {r['label']} | " + " | ".join(cells) + " |")

        # Winner per metric
        md.append(f"\n### {seg_name} — winner per metric\n")
        metric_specs = [
            ("Dist KL ↓", "kl", "lower"),
            ("Entropy ↑", "entropy", "higher"),
            ("Trait-pred ρ ↑", "trait_rho", "higher"),
            ("macro-sem ↑", "macro_sem", "higher"),
            ("NL specificity ↑", "nl_spec", "higher"),
            ("Persona-NL ↑", "persona", "higher"),
            ("Supplement ref ↑", "sup_ref", "higher"),
        ]
        if seg_name == "Q7_seg3":
            metric_specs.extend([
                ("Criterion uniqueness ↑", "crit_uniq", "higher"),
                ("Criterion-persona ↑", "crit_persona", "higher"),
                ("Criterion-supplement ↑", "crit_sup_ref", "higher"),
            ])
        md.append("| Metric | Winner |")
        md.append("|---|---|")
        for name, key, direction in metric_specs:
            valid = [r for r in rows if r.get(key) is not None]
            if not valid: continue
            if direction == "lower":
                best = min(valid, key=lambda r: r[key])
            else:
                best = max(valid, key=lambda r: r[key])
            md.append(f"| {name} | **{best['label']}** ({best[key]:.4f}) |")

    out = Path("reports/datav3/seg1_seg3_nl_metrics.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out}")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
