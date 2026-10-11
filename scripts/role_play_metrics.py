"""Role-play metric eval: validate v2_hybrid has role-play signal on seg1+seg3.

Beyond GT-match metrics (TED/Path/STA) — these are seg2-biased and trivially
favor V-default on seg3 (V-majority artifact).

New metrics designed for designer-DIFFERENTIATION:
  A. PKVC distribution KL — model pred distribution vs GT distribution per seg
  B. Prediction entropy — high = differentiated, low = collapsed to default
  C. Trait-prediction correlation — Spearman(trait_sim, pred_match) across designer pairs
  D. Criterion uniqueness (seg3) — mean pairwise cosine distance of criteria
  E. Criterion-persona alignment (seg3) — cosine(criterion_emb, designer_Q1Q6_emb)

Inputs:
  - Inference jsonls for each baseline
  - data_v3/scale_responses.csv (30维 trait)
  - data_v3/processed/embedding_cache_text_embed_3_small.jsonl
  - data_v3/processed/q1q6_extracted_v8.7_full.jsonl

Output: reports/datav3/role_play_metrics.md
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EMBED_CACHE = Path("data_v3/processed/embedding_cache_text_embed_3_small.jsonl")


def text_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    cache = {}
    if EMBED_CACHE.exists():
        for line in EMBED_CACHE.open():
            r = json.loads(line)
            cache[r["key"]] = r["vec"]
    return cache


def get_emb(text: str, cache: dict[str, list[float]]) -> np.ndarray | None:
    v = cache.get(text_key(text))
    if v is None:
        return None
    return np.array(v)


def embed_missing(texts: list[str], cache: dict[str, list[float]]) -> None:
    """Embed missing texts via OpenAI API; update cache + append to file."""
    from openai import OpenAI
    unique = list(dict.fromkeys(texts))
    missing = [t for t in unique if text_key(t) not in cache]
    if not missing:
        return
    print(f"  embedding {len(missing)} criteria...", flush=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"))
    new_records = []
    BATCH = 200
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i+BATCH]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        for txt, item in zip(batch, resp.data):
            cache[text_key(txt)] = item.embedding
            new_records.append({"key": text_key(txt), "vec": item.embedding})
    # Append to cache file
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with EMBED_CACHE.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 30维 trait loading
# ============================================================

def load_trait_vectors(scale_csv: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    with scale_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw[r["user_id"]][r["question_node"]] = float(r["score"])
    traits = sorted(next(iter(raw.values())).keys())
    vectors = {uid: np.array([scores.get(t, 3.5) for t in traits]) for uid, scores in raw.items()}
    return traits, vectors


# ============================================================
# A. PKVC distribution KL
# ============================================================

def class_distribution(pred_letters: list[str], smoothing: float = 0.01) -> np.ndarray:
    counter = Counter(pred_letters)
    n = sum(counter.values())
    dist = np.array([counter.get(c, 0) for c in "PKVC"], dtype=float) + smoothing
    return dist / dist.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q). Lower is better when q is GT."""
    return float(np.sum(p * np.log(p / q)))


def entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits. Max = log2(4) = 2 for uniform 4-class."""
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ============================================================
# C. Trait-prediction correlation
# ============================================================

def trait_pred_correlation(
    designer_ids: list[str],
    trait_vectors: dict[str, np.ndarray],
    predictions: dict[str, str],
) -> tuple[float, float, int]:
    """For all pairs (i, j) of designers, compute:
       trait_sim = cosine(trait_i, trait_j)
       pred_match = 1 if pred_i == pred_j else 0
    Return Spearman(trait_sim, pred_match), and mean trait_sim for matched vs unmatched.

    Returns (rho, mean_sim_matched_minus_unmatched, n_pairs).
    """
    from scipy.stats import spearmanr
    valid_ids = [d for d in designer_ids if d in trait_vectors and d in predictions]
    sims = []
    matches = []
    for i in range(len(valid_ids)):
        for j in range(i+1, len(valid_ids)):
            di, dj = valid_ids[i], valid_ids[j]
            s = cosine(trait_vectors[di], trait_vectors[dj])
            m = 1 if predictions[di] == predictions[dj] else 0
            sims.append(s); matches.append(m)
    sims = np.array(sims); matches = np.array(matches)
    if sims.std() == 0 or matches.std() == 0:
        rho = 0.0
    else:
        rho, _ = spearmanr(sims, matches)
    mean_matched = sims[matches == 1].mean() if (matches == 1).any() else 0.0
    mean_unmatched = sims[matches == 0].mean() if (matches == 0).any() else 0.0
    return float(rho if not np.isnan(rho) else 0.0), float(mean_matched - mean_unmatched), len(sims)


# ============================================================
# D. Criterion uniqueness (seg3)
# ============================================================

def criterion_uniqueness(criteria: list[str], emb_cache: dict) -> float:
    """Mean pairwise cosine distance (1 - cos) across criteria embeddings.
    High = designers say different things, Low = everyone says same.
    """
    embs = [get_emb(c, emb_cache) for c in criteria]
    embs = [e for e in embs if e is not None]
    if len(embs) < 2:
        return 0.0
    embs = np.array(embs)
    # Normalize for cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embs_n = embs / norms
    # Pairwise cos = embs_n @ embs_n.T
    cos_matrix = embs_n @ embs_n.T
    # Take upper triangle (exclude diagonal)
    n = len(embs)
    iu = np.triu_indices(n, k=1)
    mean_cos = float(cos_matrix[iu].mean())
    return 1.0 - mean_cos


# ============================================================
# E. Criterion-persona alignment (seg3)
# ============================================================

def criterion_persona_alignment(
    designer_criteria: dict[str, str],
    designer_personas: dict[str, np.ndarray],
    emb_cache: dict,
) -> float:
    """For each designer: cosine(criterion_emb, designer_persona_center).
    Mean over designers. Higher = criterion matches designer style.
    """
    sims = []
    for did, crit in designer_criteria.items():
        crit_emb = get_emb(crit, emb_cache)
        if crit_emb is None or did not in designer_personas:
            continue
        sims.append(cosine(crit_emb, designer_personas[did]))
    return float(np.mean(sims)) if sims else 0.0


# ============================================================
# Main
# ============================================================

BASELINES = {
    "B5 (orig)": ("reports/datav3/baselines_q7_main_final_inference.jsonl", "B5"),
    "B0_h": ("reports/datav3/ladder_b0_h_inference.jsonl", None),
    "B1_h": ("reports/datav3/ladder_b1_h_inference.jsonl", None),
    "B2_h": ("reports/datav3/ladder_b2_h_inference.jsonl", None),
    "B3_h": ("reports/datav3/ladder_b3_h_inference.jsonl", None),
    "v2_hybrid": ("reports/datav3/b6_v2hybrid_full_inference.jsonl", None),
    "v2 (anchor)": ("reports/datav3/b6_v2_full_inference.jsonl", None),
    "B6_h (+peer)": ("reports/datav3/b6_h_full_inference.jsonl", None),
}


def load_inference(path: str, baseline_filter: str | None):
    rs = [json.loads(l) for l in Path(path).open() if l.strip()]
    out = []
    for r in rs:
        if baseline_filter and r.get("baseline") != baseline_filter:
            continue
        if not r.get("ok"):
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/datav3/role_play_metrics.md")
    args = ap.parse_args()

    print("[load] trait vectors...")
    traits, trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    print(f"  {len(trait_vectors)} designers, {len(traits)} traits")

    print("[load] embedding cache...")
    emb_cache = load_cache()
    print(f"  {len(emb_cache)} cached embeddings")

    print("[load] designer persona centers (from Q1-Q6 NL)...")
    q16_rows = [json.loads(l) for l in
                Path("data_v3/processed/q1q6_extracted_v8.7_full.jsonl").open() if l.strip()]
    designer_q16_nls: dict[str, list[str]] = defaultdict(list)
    for r in q16_rows:
        if r.get("ok") and r.get("text"):
            designer_q16_nls[r["user_id"]].append(r["text"])
    personas: dict[str, np.ndarray] = {}
    for did, nls in designer_q16_nls.items():
        embs = [get_emb(nl, emb_cache) for nl in nls]
        embs = [e for e in embs if e is not None]
        if embs:
            personas[did] = np.mean(embs, axis=0)
    print(f"  {len(personas)} designer persona centers")

    md = ["# Role-play metric eval (seg1/seg3 differentiation)\n"]
    md.append("Compares baselines on designer-DIFFERENTIATION metrics, beyond GT-match.\n")

    # ============================================================
    # A + B: seg1/seg3 distribution KL + entropy
    # ============================================================
    md.append("## A+B. PKVC dom 分布 KL ↓ + 预测 entropy ↑ (seg1, seg3)\n")
    md.append("- KL: model 预测分布 vs GT 分布. Low = match population GT distribution.")
    md.append("- Entropy: model 预测的多样性. High = differentiation, Low = V-default collapse.\n")

    for seg in ("Q7_seg1", "Q7_seg3"):
        md.append(f"### {seg}\n")
        # First, compute GT distribution
        gt_letters_by_did = {}
        for label, (path, bl) in BASELINES.items():
            rs = load_inference(path, bl)
            for r in rs:
                if r["scenario"] != seg:
                    continue
                gt = (r.get("ground_truth_pkvc") or {}).get("dominant", "")
                if gt:
                    gt_letters_by_did[r["designer_id"]] = gt[0]
        gt_letters = list(gt_letters_by_did.values())
        gt_dist = class_distribution(gt_letters)
        md.append(f"GT distribution (n={len(gt_letters)}): "
                  f"P={gt_dist[0]:.1%} K={gt_dist[1]:.1%} V={gt_dist[2]:.1%} C={gt_dist[3]:.1%}, "
                  f"entropy={entropy_bits(gt_dist):.3f}\n")
        md.append("| Baseline | Pred dist (P/K/V/C) | KL(pred‖GT) ↓ | Entropy ↑ |")
        md.append("|---|---|---|---|")
        for label, (path, bl) in BASELINES.items():
            rs = load_inference(path, bl)
            preds = []
            for r in rs:
                if r["scenario"] != seg:
                    continue
                cl = (r.get("claimed_pkvc") or {}).get("dominant_factor", "")
                if cl:
                    preds.append(cl[0])
            if not preds:
                continue
            pred_dist = class_distribution(preds)
            kl = kl_divergence(pred_dist, gt_dist)
            ent = entropy_bits(pred_dist)
            md.append(f"| {label} | "
                      f"P={pred_dist[0]:.1%} K={pred_dist[1]:.1%} V={pred_dist[2]:.1%} C={pred_dist[3]:.1%} | "
                      f"{kl:.4f} | {ent:.3f} |")
        md.append("")

    # ============================================================
    # C: Trait-prediction correlation (seg1, seg3)
    # ============================================================
    md.append("## C. Trait-prediction correlation (seg1, seg3)\n")
    md.append("Spearman(trait cosine sim, pred match) across designer pairs. "
              "Positive = similar 30维 traits → similar predictions (role-play 信号).\n")

    for seg in ("Q7_seg1", "Q7_seg3"):
        md.append(f"### {seg}\n")
        md.append("| Baseline | Spearman ρ | Δ mean sim (matched − unmatched) | N pairs |")
        md.append("|---|---|---|---|")
        for label, (path, bl) in BASELINES.items():
            rs = load_inference(path, bl)
            preds = {}
            for r in rs:
                if r["scenario"] != seg:
                    continue
                cl = (r.get("claimed_pkvc") or {}).get("dominant_factor", "")
                if cl:
                    preds[r["designer_id"]] = cl[0]
            if not preds:
                continue
            all_ids = list(preds.keys())
            rho, delta, n_pairs = trait_pred_correlation(all_ids, trait_vectors, preds)
            md.append(f"| {label} | {rho:+.4f} | {delta:+.4f} | {n_pairs} |")
        md.append("")

    # ============================================================
    # D + E: Criterion uniqueness + persona alignment (seg3 only)
    # ============================================================
    md.append("## D + E. Criterion uniqueness + persona alignment (seg3)\n")
    md.append("- Uniqueness: 162 designer criteria 的 mean pairwise cosine distance. "
              "High = each designer has unique criterion (role-play). Low = everyone says same.\n")
    md.append("- Persona align: cosine(criterion_emb, designer_Q1-Q6_emb_center). "
              "High = criterion matches designer style.\n")
    md.append("| Baseline | Criterion uniqueness ↑ | Persona alignment ↑ | n_criteria |")
    md.append("|---|---|---|---|")

    # First gather ALL criteria across baselines, then batch embed missing
    all_criteria: list[str] = []
    for label, (path, bl) in BASELINES.items():
        rs = load_inference(path, bl)
        for r in rs:
            if r["scenario"] != "Q7_seg3":
                continue
            crit = (r.get("claimed_pkvc") or {}).get("criterion", "")
            if crit:
                all_criteria.append(crit)
    print(f"  total criteria across baselines: {len(all_criteria)}, unique: {len(set(all_criteria))}")
    embed_missing(all_criteria, emb_cache)
    print(f"  cache now: {len(emb_cache)} entries")

    for label, (path, bl) in BASELINES.items():
        rs = load_inference(path, bl)
        designer_criteria: dict[str, str] = {}
        for r in rs:
            if r["scenario"] != "Q7_seg3":
                continue
            crit = (r.get("claimed_pkvc") or {}).get("criterion", "")
            if crit:
                designer_criteria[r["designer_id"]] = crit
        if not designer_criteria:
            continue
        uniqueness = criterion_uniqueness(list(designer_criteria.values()), emb_cache)
        align = criterion_persona_alignment(designer_criteria, personas, emb_cache)
        n_cached = sum(1 for c in designer_criteria.values() if get_emb(c, emb_cache) is not None)
        md.append(f"| {label} | {uniqueness:.4f} | {align:.4f} | {n_cached}/{len(designer_criteria)} cached |")
    md.append("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
