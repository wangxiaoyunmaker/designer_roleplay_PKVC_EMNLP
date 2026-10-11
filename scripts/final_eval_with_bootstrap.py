"""Final 11-metric eval with 1000-bootstrap 95% CI on 162 designer.

Metric set (locked):
  seg1 (anchor, single dom): NL specificity, Persona-NL alignment, Trait-pred Spearman ρ
  seg2 (chain): macro-TED↓, macro-LCS-F1↑, non-V recall↑, Chain diversity↑, NL specificity↑
  seg3 (anchor + criterion): NL specificity, Criterion uniqueness, Criterion-persona alignment

Bootstrap: 1000 resamples of 162 designers (with replacement, position-based).
Report mean [95% CI percentile method], no Bonferroni (orthogonal dimensions).

Output:
  reports/datav3/final_eval_bootstrap.md
  reports/datav3/final_eval_bootstrap.jsonl
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EMBED_CACHE = Path("data_v3/processed/embedding_cache_text_embed_3_small.jsonl")
N_BOOTSTRAP = 1000
SEED = 42

BASELINES = [
    ("B0_h",      "reports/datav3/ladder_b0_h_inference.jsonl",      "reports/datav3/ladder_b0_h_pkvc_eval.jsonl",      None),
    ("B1_h",      "reports/datav3/ladder_b1_h_inference.jsonl",      "reports/datav3/ladder_b1_h_pkvc_eval.jsonl",      None),
    ("B2_h",      "reports/datav3/ladder_b2_h_inference.jsonl",      "reports/datav3/ladder_b2_h_pkvc_eval.jsonl",      None),
    ("B3_h",      "reports/datav3/ladder_b3_h_inference.jsonl",      "reports/datav3/ladder_b3_h_pkvc_eval.jsonl",      None),
    ("v2_hybrid", "reports/datav3/b6_v2hybrid_full_inference.jsonl", "reports/datav3/b6_v2hybrid_full_pkvc_eval.jsonl", None),
    ("v2_anchor", "reports/datav3/b6_v2_full_inference.jsonl",       "reports/datav3/b6_v2_full_pkvc_eval.jsonl",       None),
    ("v2_no_pkvc_def",     "reports/datav3/ablation_v2_no_pkvc_def_inference.jsonl",      "reports/datav3/ablation_v2_no_pkvc_def_pkvc_eval.jsonl",      None),
    ("v2_no_seg3_rule",    "reports/datav3/ablation_v2_no_seg3_rule_inference.jsonl",     "reports/datav3/ablation_v2_no_seg3_rule_pkvc_eval.jsonl",     None),
    ("v2_no_beta_summary", "reports/datav3/ablation_v2_no_beta_summary_inference.jsonl",  "reports/datav3/ablation_v2_no_beta_summary_pkvc_eval.jsonl",  None),
    # 干净 ablation (variant=v2, 跟 v2_anchor SYSTEM 严格一致)
    ("v2_no_caps",         "reports/datav3/ablation_v2_no_caps_inference.jsonl",          "reports/datav3/ablation_v2_no_caps_pkvc_eval.jsonl",          None),
    ("v2_plus_peer",       "reports/datav3/ablation_v2_plus_peer_inference.jsonl",        "reports/datav3/ablation_v2_plus_peer_pkvc_eval.jsonl",        None),
    ("B6_h",      "reports/datav3/b6_h_full_inference.jsonl",        "reports/datav3/b6_h_full_pkvc_eval.jsonl",        None),
    ("B5",        "reports/datav3/baselines_q7_main_final_inference.jsonl", "reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl", "B5"),
]


# ============================================================
# Loading
# ============================================================

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


def load_supplements(path):
    out = defaultdict(list)
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["question_id"] in ("Q3","Q4","Q5","Q6") and r["text_supplement"].strip():
                out[r["user_id"]].append(r["text_supplement"].strip())
    return {uid: "\n".join(s) for uid, s in out.items()}


# ============================================================
# Path metric primitives
# ============================================================

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


# ============================================================
# Build baseline data — per-designer fields + precomputed matrices
# ============================================================

def build_baseline_data(label, inf_path, pkvc_path, baseline_filter, cache,
                       trait_vectors, supplements, persona_embs):
    """Return dict with:
      - designer_ids list (162)
      - per-seg per-designer fields (pred_dom, pred_seq, pred_nl, gt_*, criterion)
      - precomputed cosine matrices (162×162) for pred_NL vs GT_NL etc.
    """
    inf = load_jsonl(inf_path, baseline_filter)
    pkvc = load_jsonl(pkvc_path, baseline_filter)
    pkvc_map = {(r["designer_id"], r["scenario"]): r for r in pkvc}

    # Per designer per seg
    by_designer = defaultdict(dict)
    for r in inf:
        sc = r["scenario"]
        if sc not in ("Q7_seg1","Q7_seg2","Q7_seg3"): continue
        did = r["designer_id"]
        gt_dom = (r.get("ground_truth_pkvc") or {}).get("dominant","")
        gt_seq = (r.get("ground_truth_pkvc") or {}).get("sequence", [])
        gt_nl = r.get("ground_truth_nl","")
        pred_dom = (r.get("claimed_pkvc") or {}).get("dominant_factor","")
        pred_nl = r.get("simulated_nl","")
        criterion = (r.get("claimed_pkvc") or {}).get("criterion","")
        pkvc_row = pkvc_map.get((did, sc))
        pred_seq = pkvc_row["pred"].get("state_sequence",[]) if pkvc_row else []
        by_designer[did][sc] = {
            "gt_dom": gt_dom[0] if gt_dom else "",
            "gt_seq": gt_seq, "gt_nl": gt_nl,
            "pred_dom": pred_dom[0] if pred_dom else "",
            "pred_seq": pred_seq, "pred_nl": pred_nl,
            "criterion": criterion,
        }

    # Filter: only designer with all 3 seg + trait_vector
    designer_ids = [d for d in by_designer
                    if all(s in by_designer[d] for s in ("Q7_seg1","Q7_seg2","Q7_seg3"))
                    and d in trait_vectors]
    designer_ids.sort()
    n = len(designer_ids)
    if n == 0:
        return None

    def emb_or_zero(text):
        if not text:
            return np.zeros(1536)
        e = get_emb(text, cache)
        return e if e is not None else np.zeros(1536)

    # Precompute per-designer arrays
    data = {"label": label, "designer_ids": designer_ids, "n": n, "by_designer": by_designer}

    # Trait similarity matrix (n×n)
    trait_mat = np.array([trait_vectors[d] for d in designer_ids])  # (n, 30)
    trait_norm = trait_mat / np.linalg.norm(trait_mat, axis=1, keepdims=True).clip(1e-9)
    data["trait_cos"] = trait_norm @ trait_norm.T  # (n, n)

    # Persona center embedding (Q1-Q6 NL) per designer
    persona_arr = np.array([persona_embs.get(d, np.zeros(1536)) for d in designer_ids])
    data["persona"] = persona_arr  # (n, 1536)

    # Supplement embedding per designer
    sup_arr = np.array([emb_or_zero(supplements.get(d,"")) for d in designer_ids])
    data["sup"] = sup_arr

    # Per-seg embeddings + meta
    for sc in ("Q7_seg1","Q7_seg2","Q7_seg3"):
        pred_nl_arr = np.array([emb_or_zero(by_designer[d][sc]["pred_nl"]) for d in designer_ids])
        gt_nl_arr = np.array([emb_or_zero(by_designer[d][sc]["gt_nl"]) for d in designer_ids])
        # Normalize for cosine
        pred_nl_n = pred_nl_arr / np.linalg.norm(pred_nl_arr, axis=1, keepdims=True).clip(1e-9)
        gt_nl_n = gt_nl_arr / np.linalg.norm(gt_nl_arr, axis=1, keepdims=True).clip(1e-9)
        persona_n = persona_arr / np.linalg.norm(persona_arr, axis=1, keepdims=True).clip(1e-9)
        sup_n = sup_arr / np.linalg.norm(sup_arr, axis=1, keepdims=True).clip(1e-9)
        data[f"{sc}_pred_nl_n"] = pred_nl_n
        data[f"{sc}_gt_nl_n"] = gt_nl_n
        data[f"{sc}_pred_gt_cos"] = pred_nl_n @ gt_nl_n.T  # (n, n)
        data[f"{sc}_pred_persona_cos"] = np.array([pred_nl_n[i] @ persona_n[i] for i in range(n)])  # (n,)
        data[f"{sc}_pred_sup_cos"] = np.array([pred_nl_n[i] @ sup_n[i] for i in range(n)])
        data[f"{sc}_pred_dom"] = np.array([by_designer[d][sc]["pred_dom"] for d in designer_ids])
        data[f"{sc}_gt_dom"] = np.array([by_designer[d][sc]["gt_dom"] for d in designer_ids])
        data[f"{sc}_pred_seq"] = [by_designer[d][sc]["pred_seq"] for d in designer_ids]
        data[f"{sc}_gt_seq"] = [by_designer[d][sc]["gt_seq"] for d in designer_ids]
        data[f"{sc}_gt_first"] = np.array([gs[0] if gs else "" for gs in data[f"{sc}_gt_seq"]])

    # Pre-compute path metric values per designer for seg2
    data["seg2_ted"] = np.array([edit_distance(data["Q7_seg2_gt_seq"][i],
                                                data["Q7_seg2_pred_seq"][i]) for i in range(n)])
    data["seg2_lcs"] = np.array([lcs_f1(data["Q7_seg2_gt_seq"][i],
                                         data["Q7_seg2_pred_seq"][i]) for i in range(n)])
    data["seg2_pred_first"] = np.array([s[0] if s else "" for s in data["Q7_seg2_pred_seq"]])

    # Criterion embeddings (seg3)
    crit_arr = np.array([emb_or_zero(by_designer[d]["Q7_seg3"]["criterion"]) for d in designer_ids])
    crit_n = crit_arr / np.linalg.norm(crit_arr, axis=1, keepdims=True).clip(1e-9)
    data["crit_n"] = crit_n
    data["crit_cos"] = crit_n @ crit_n.T  # (n, n) pairwise
    data["crit_persona_cos"] = np.array([crit_n[i] @ persona_n[i] for i in range(n)])
    data["crit_sup_cos"] = np.array([crit_n[i] @ sup_n[i] for i in range(n)])

    return data


# ============================================================
# Metric functions (operate on bootstrap sample = list of designer indices)
# ============================================================

def m_nl_specificity(d, sample, seg):
    """Mean over positions k: cos(pred_k, GT_k) / mean_{l≠k}(cos(pred_k, GT_l)).
    sample[k] is index into baseline's designer list."""
    pred_gt_cos = d[f"{seg}_pred_gt_cos"]
    n = len(sample)
    if n < 2: return 0.0
    specs = []
    for k, idx_k in enumerate(sample):
        own = pred_gt_cos[idx_k, idx_k]
        others_sum = sum(pred_gt_cos[idx_k, sample[l]] for l in range(n) if l != k)
        others_mean = others_sum / (n - 1)
        if others_mean > 0:
            specs.append(own / others_mean)
    return float(np.mean(specs)) if specs else 0.0


def m_persona_align(d, sample, seg):
    """Mean cos(pred_NL_i, designer_persona_center) over sample positions."""
    arr = d[f"{seg}_pred_persona_cos"]  # (n,)
    return float(np.mean([arr[idx] for idx in sample]))


def m_trait_pred_rho(d, sample, seg):
    """Spearman(trait_cos_pair, pred_dom_match_pair) across i<j in sample."""
    trait_cos = d["trait_cos"]
    pred_dom = d[f"{seg}_pred_dom"]
    n = len(sample)
    if n < 2: return 0.0
    t_sims = []
    matches = []
    for k in range(n):
        for l in range(k+1, n):
            i, j = sample[k], sample[l]
            t_sims.append(trait_cos[i, j])
            matches.append(1.0 if pred_dom[i] == pred_dom[j] else 0.0)
    t_sims = np.array(t_sims); matches = np.array(matches)
    if t_sims.std() == 0 or matches.std() == 0:
        return 0.0
    rho, _ = spearmanr(t_sims, matches)
    return float(rho) if not np.isnan(rho) else 0.0


def m_macro_ted(d, sample):
    """Mean over GT_first ∈ {P,K,V,C} of mean TED per group."""
    gt_first = d["Q7_seg2_gt_first"]
    ted = d["seg2_ted"]
    groups = defaultdict(list)
    for idx in sample:
        groups[gt_first[idx]].append(ted[idx])
    classes = [c for c in "PKVC" if c in groups and groups[c]]
    if not classes: return 0.0
    return float(np.mean([np.mean(groups[c]) for c in classes]))


def m_macro_lcs(d, sample):
    gt_first = d["Q7_seg2_gt_first"]
    lcs = d["seg2_lcs"]
    groups = defaultdict(list)
    for idx in sample:
        groups[gt_first[idx]].append(lcs[idx])
    classes = [c for c in "PKVC" if c in groups and groups[c]]
    if not classes: return 0.0
    return float(np.mean([np.mean(groups[c]) for c in classes]))


def m_non_v_recall_seg2(d, sample):
    """# correctly predicted non-V first letter / # non-V GT in sample."""
    gt_first = d["Q7_seg2_gt_first"]
    pred_first = d["seg2_pred_first"]
    num = 0; den = 0
    for idx in sample:
        if gt_first[idx] != "V":
            den += 1
            if pred_first[idx] == gt_first[idx]:
                num += 1
    return num / den if den > 0 else 0.0


def m_chain_diversity(d, sample):
    """1 - modal_chain_freq in sample."""
    pred_seq = d["Q7_seg2_pred_seq"]
    chain_counter = Counter(tuple(pred_seq[idx]) for idx in sample)
    n = sum(chain_counter.values())
    if n == 0: return 0.0
    modal = chain_counter.most_common(1)[0][1] / n
    return 1.0 - modal


def m_criterion_uniqueness(d, sample):
    """1 - mean pairwise cosine over criterion embeddings in sample."""
    crit_cos = d["crit_cos"]
    n = len(sample)
    if n < 2: return 0.0
    sims = []
    for k in range(n):
        for l in range(k+1, n):
            sims.append(crit_cos[sample[k], sample[l]])
    return 1.0 - float(np.mean(sims))


def m_criterion_persona(d, sample):
    arr = d["crit_persona_cos"]
    return float(np.mean([arr[idx] for idx in sample]))


# ============================================================
# Bootstrap loop
# ============================================================

METRICS = [
    # (display name, seg, function, args)
    ("seg1_nl_specificity",      "Q7_seg1", lambda d,s: m_nl_specificity(d, s, "Q7_seg1")),
    ("seg1_persona_align",       "Q7_seg1", lambda d,s: m_persona_align(d, s, "Q7_seg1")),
    ("seg1_trait_pred_rho",      "Q7_seg1", lambda d,s: m_trait_pred_rho(d, s, "Q7_seg1")),
    ("seg2_macro_TED",           "Q7_seg2", lambda d,s: m_macro_ted(d, s)),
    ("seg2_macro_LCS_F1",        "Q7_seg2", lambda d,s: m_macro_lcs(d, s)),
    ("seg2_non_V_recall",        "Q7_seg2", lambda d,s: m_non_v_recall_seg2(d, s)),
    ("seg2_chain_diversity",     "Q7_seg2", lambda d,s: m_chain_diversity(d, s)),
    ("seg2_nl_specificity",      "Q7_seg2", lambda d,s: m_nl_specificity(d, s, "Q7_seg2")),
    ("seg3_nl_specificity",      "Q7_seg3", lambda d,s: m_nl_specificity(d, s, "Q7_seg3")),
    ("seg3_criterion_uniqueness","Q7_seg3", lambda d,s: m_criterion_uniqueness(d, s)),
    ("seg3_criterion_persona",   "Q7_seg3", lambda d,s: m_criterion_persona(d, s)),
]

LOWER_BETTER = {"seg2_macro_TED"}


def run_bootstrap(d, n_iter=N_BOOTSTRAP, seed=SEED):
    """Return {metric_name: (point_estimate, ci_low, ci_high, all_values)}"""
    n = d["n"]
    rng = np.random.default_rng(seed)
    results = {name: [] for name, _, _ in METRICS}

    # Point estimate (no resample, full sample)
    full_sample = list(range(n))
    point_estimates = {}
    for name, seg, fn in METRICS:
        point_estimates[name] = fn(d, full_sample)

    # Bootstrap iterations
    for it in range(n_iter):
        sample = rng.integers(0, n, size=n).tolist()  # with replacement
        for name, seg, fn in METRICS:
            try:
                v = fn(d, sample)
                results[name].append(v)
            except Exception:
                pass

    summary = {}
    for name in results:
        vals = np.array(results[name])
        if len(vals) == 0:
            summary[name] = {"point": point_estimates[name], "ci_lo": None, "ci_hi": None, "n_iter": 0}
        else:
            summary[name] = {
                "point": point_estimates[name],
                "mean_boot": float(vals.mean()),
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5)),
                "n_iter": len(vals),
            }
    return summary


# ============================================================
# Main
# ============================================================

def embed_missing(texts, cache):
    """Embed any texts not in cache via OpenAI API."""
    from openai import OpenAI
    unique = list(dict.fromkeys([t for t in texts if t]))
    missing = [t for t in unique if text_key(t) not in cache]
    if not missing:
        return
    print(f"  embedding {len(missing)} new texts via API...", flush=True)
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


def main():
    print("[load] trait + supplements + persona centers + embedding cache...")
    trait_vectors = load_trait_vectors(Path("data_v3/scale_responses.csv"))
    supplements = load_supplements(Path("data_v3/task_responses.csv"))
    cache = load_cache()
    print(f"  trait={len(trait_vectors)}, supplements={len(supplements)}, cache={len(cache)}")

    # Pre-embed all texts needed across baselines (handles new ablations)
    print("[pre-embed] collecting all texts (simulated_nl, criterion, gt_nl)...")
    texts_to_embed = []
    for label, inf_p, _, bl in BASELINES:
        rs = load_jsonl(inf_p, bl)
        for r in rs:
            if r.get("simulated_nl"): texts_to_embed.append(r["simulated_nl"])
            if r.get("ground_truth_nl"): texts_to_embed.append(r["ground_truth_nl"])
            crit = (r.get("claimed_pkvc") or {}).get("criterion","")
            if crit: texts_to_embed.append(crit)
    embed_missing(texts_to_embed, cache)
    print(f"  cache: {len(cache)} entries after embed")

    # Persona centers
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
        if embs:
            persona_embs[did] = np.mean(embs, axis=0)
    print(f"  persona centers: {len(persona_embs)}")

    print("\n[build] per-baseline data...")
    baseline_data = {}
    for label, inf_p, pkvc_p, bl in BASELINES:
        print(f"  {label}...", end="", flush=True)
        d = build_baseline_data(label, inf_p, pkvc_p, bl, cache,
                                trait_vectors, supplements, persona_embs)
        if d:
            baseline_data[label] = d
            print(f" n={d['n']}")
        else:
            print(" SKIP (no data)")

    print(f"\n[bootstrap] {N_BOOTSTRAP} iterations × {len(METRICS)} metrics × {len(baseline_data)} baselines...")
    all_summaries = {}
    import time
    t0 = time.time()
    for label, d in baseline_data.items():
        print(f"  {label}...", end="", flush=True)
        all_summaries[label] = run_bootstrap(d)
        print(f" done ({time.time()-t0:.1f}s elapsed)")

    # ============================================================
    # Output markdown
    # ============================================================
    md = ["# Final 11-metric eval with 1000-bootstrap 95% CI\n"]
    md.append(f"Baselines: {list(baseline_data.keys())}")
    md.append(f"Bootstrap: {N_BOOTSTRAP} resamples × 162 designers (with replacement)")
    md.append("CI: percentile method (2.5%–97.5%). No Bonferroni correction (orthogonal dimensions).\n")

    # Per-segment tables
    for seg, seg_label in [("Q7_seg1","seg1 (第一反应 anchor)"),
                           ("Q7_seg2","seg2 (chain reasoning)"),
                           ("Q7_seg3","seg3 (anchor + criterion)")]:
        md.append(f"\n## {seg_label}\n")
        seg_metrics = [(name, fn) for name, s, fn in METRICS if s == seg]
        # Header
        md.append("| Baseline | " + " | ".join(n for n,_ in seg_metrics) + " |")
        md.append("|" + "|".join("---" for _ in range(1 + len(seg_metrics))) + "|")
        for label in baseline_data:
            cells = [label]
            for name, _ in seg_metrics:
                s = all_summaries[label][name]
                cells.append(f"{s['point']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
        # Annotate direction
        directions = {
            "seg2_macro_TED": "↓",
        }
        for name, _ in seg_metrics:
            direction = "↓" if name in LOWER_BETTER else "↑"
            md.append(f"- {name}: {direction}")

    # ============================================================
    # Winner per metric (CI-aware)
    # ============================================================
    md.append("\n## Winner per metric (point estimate, CI annotation)\n")
    md.append("| Metric | Winner | Point | CI |")
    md.append("|---|---|---|---|")
    for name, seg, _ in METRICS:
        direction = "lower" if name in LOWER_BETTER else "higher"
        if direction == "lower":
            best = min(baseline_data.keys(), key=lambda b: all_summaries[b][name]["point"])
        else:
            best = max(baseline_data.keys(), key=lambda b: all_summaries[b][name]["point"])
        s = all_summaries[best][name]
        md.append(f"| {name} | **{best}** | {s['point']:.4f} | [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] |")
    md.append("")

    # ============================================================
    # Pairwise CI comparison vs B0_h (zero-shot baseline)
    # ============================================================
    md.append("\n## Δ vs B0_h (zero-shot baseline), 95% CI overlap test\n")
    md.append("Marked ★ if CI of v2_anchor and B0_h do not overlap (likely significant).\n")
    md.append("| Metric | B0_h point [CI] | v2_anchor point [CI] | Δ | Sig? |")
    md.append("|---|---|---|---|---|")
    for name, seg, _ in METRICS:
        if "B0_h" not in baseline_data or "v2_anchor" not in baseline_data: continue
        b0 = all_summaries["B0_h"][name]
        v2 = all_summaries["v2_anchor"][name]
        delta = v2["point"] - b0["point"]
        # CI overlap test
        if name in LOWER_BETTER:
            sig = "★" if v2["ci_hi"] < b0["ci_lo"] else ""  # v2 strictly better
        else:
            sig = "★" if v2["ci_lo"] > b0["ci_hi"] else ""
        md.append(f"| {name} | {b0['point']:.3f} [{b0['ci_lo']:.3f}, {b0['ci_hi']:.3f}] | "
                  f"{v2['point']:.3f} [{v2['ci_lo']:.3f}, {v2['ci_hi']:.3f}] | "
                  f"{delta:+.3f} | {sig} |")
    md.append("")

    # Save
    out_md = Path("reports/datav3/final_eval_bootstrap.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_md}")

    out_jsonl = Path("reports/datav3/final_eval_bootstrap.jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for label in baseline_data:
            for name, _, _ in METRICS:
                s = all_summaries[label][name]
                f.write(json.dumps({
                    "baseline": label, "metric": name,
                    "point": s["point"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                    "n_iter": s["n_iter"],
                }, ensure_ascii=False) + "\n")
    print(f"[saved] {out_jsonl}")

    # Print main winner table to stdout
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
