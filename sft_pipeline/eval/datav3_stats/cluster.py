"""L3: K-means cluster sweep on factor scores, silhouette/DB/gap, per-cluster Q-option χ²."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def gap_statistic(X: np.ndarray, k: int, n_ref: int = 10, seed: int = 42) -> float:
    """Tibshirani gap statistic with uniform reference."""
    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    Wk = np.log(km.inertia_)
    Wkb = []
    mins, maxs = X.min(axis=0), X.max(axis=0)
    for _ in range(n_ref):
        ref = rng.uniform(mins, maxs, size=X.shape)
        km_ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(ref)
        Wkb.append(np.log(km_ref.inertia_))
    return float(np.mean(Wkb) - Wk)


def pick_best_k(
    sweep_rows: list[dict],
    chi_results: dict[int, dict[str, dict]],
    min_per_cluster_n: int = 20,
    sil_tol: float = 0.02,
    min_sig_q: int = 2,
) -> tuple[int, str]:
    """挑 K 的规则 (任务导向):

    1. min_cluster_n >= min_per_cluster_n  (mapping miner 不饿死)
    2. χ² 至少 min_sig_q 个 Q 在 p<.05  (cluster 行为可区分)
    3. silhouette 在 (max_sil - sil_tol) 之上  (cluster 结构 OK)
    候选里取 **最小 K** (Occam).
    """
    df = pd.DataFrame(sweep_rows)
    if df.empty:
        return 0, "no sweep rows"
    max_sil = df["silhouette"].max()
    candidates: list[tuple[int, int]] = []
    for row in sweep_rows:
        k = row["K"]
        if row["min_cluster_n"] < min_per_cluster_n:
            continue
        if row["silhouette"] < max_sil - sil_tol:
            continue
        n_sig = sum(1 for v in chi_results.get(k, {}).values() if v.get("p", 1.0) < 0.05)
        if n_sig < min_sig_q:
            continue
        candidates.append((k, n_sig))
    if not candidates:
        best_k = int(df.loc[df["silhouette"].idxmax(), "K"])
        return best_k, (
            f"fallback: 无 K 同时满足 (min_n>={min_per_cluster_n}, sig_Q>={min_sig_q}, "
            f"sil knee within {sil_tol}); 取 silhouette 最优 K={best_k}"
        )
    candidates.sort(key=lambda x: (x[0], -x[1]))
    best_k, n_sig = candidates[0]
    return best_k, (
        f"min K with per-cluster N>={min_per_cluster_n}, sig on {n_sig} Q, "
        f"silhouette within {sil_tol} of max ({max_sil:.3f})"
    )


def run(
    output_path: Path | None = None,
    k_candidates: tuple[int, ...] = (3, 4, 5, 6, 7),
    factor_scores_path: Path | None = None,
) -> dict:
    output_path = output_path or _REPORTS / "L3_cluster.md"
    factor_scores_path = factor_scores_path or loaders.get_default_factor_scores_path()
    if not factor_scores_path.exists():
        raise FileNotFoundError(f"factor scores not found: {factor_scores_path}. Run L2 EFA first.")

    scores = pd.read_csv(factor_scores_path, index_col=0)
    X = scores.values

    findings: dict = {
        "n": int(len(scores)),
        "n_factors": int(X.shape[1]),
        "factor_scores_path": str(factor_scores_path),
    }
    sections: list[str] = []
    sections.append(f"# L3 Cluster K-Sweep (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- factor scores: {factor_scores_path.name}, shape = {X.shape}\n")

    rows = []
    label_dict = {}
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        labels = km.labels_
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        gap = gap_statistic(X, k)
        per_cluster = pd.Series(labels).value_counts().sort_index().tolist()
        rows.append({
            "K": k,
            "silhouette": round(float(sil), 3),
            "davies_bouldin": round(float(db), 3),
            "gap": round(float(gap), 3),
            "min_cluster_n": int(min(per_cluster)),
            "per_cluster_n": str(per_cluster),
        })
        label_dict[k] = labels

    sweep_df = pd.DataFrame(rows)
    sections.append("## K sweep metrics\n")
    sections.append(sweep_df.to_markdown(index=False))
    findings["sweep"] = sweep_df.to_dict("records")

    # per-cluster Q-option χ²
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    sections.append("\n## per-cluster Q-option χ² (per K)\n")
    chi_results: dict[int, dict[str, dict]] = {}
    for k, labels in label_dict.items():
        cluster_map = pd.Series(labels, index=scores.index)
        q_k = q_single.copy()
        q_k["cluster"] = q_k["user_id"].map(cluster_map)
        q_k = q_k.dropna(subset=["cluster"])
        sections.append(f"\n### K = {k}")
        q_chi: dict[str, dict] = {}
        for qid in sorted(q_k["question_id"].unique()):
            sub = q_k[q_k["question_id"] == qid]
            ct = pd.crosstab(sub["cluster"], sub["option"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = chi2_contingency(ct)
            q_chi[qid] = {"chi2": round(float(chi2), 2), "p": round(float(p), 4), "dof": int(dof)}
            sections.append(f"- {qid}: χ²={chi2:.2f}, dof={dof}, p={p:.4f}  ({'sig' if p < 0.05 else 'ns'})")
        chi_results[k] = q_chi
    findings["chi_results"] = {str(k): v for k, v in chi_results.items()}

    # auto-pick
    picked_k, reason = pick_best_k(rows, chi_results)
    findings["picked_k"] = int(picked_k)
    findings["picked_reason"] = reason

    sections.append("\n## Verdict\n")
    best_sil = int(sweep_df.loc[sweep_df["silhouette"].idxmax(), "K"])
    best_gap = int(sweep_df.loc[sweep_df["gap"].idxmax(), "K"])
    sections.append(f"- silhouette 最优 K = {best_sil}")
    sections.append(f"- gap 最优 K = {best_gap}")
    sections.append(f"- **picked K (任务导向)** = {picked_k}")
    sections.append(f"  - rule: min K satisfy [per-cluster N>=20, χ² sig on >=2 Q, silhouette within 0.02 of max]")
    sections.append(f"  - 结论: {reason}")

    # save chosen K (and all) cluster assignment
    for k, labels in label_dict.items():
        out = pd.DataFrame({"user_id": scores.index, "cluster": labels})
        out.to_csv(_REPORTS / "per_designer" / f"user_clusters_k{k}.csv", index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
