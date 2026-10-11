"""EFA pilot: 30 量表降维到 4 因子, 然后在 4 因子空间聚类.

Caveat: N<150 (5×30 推荐下限) 时 EFA 稳定性受限, 仅作 pilot.

Steps:
  1. KMO + Bartlett 球形检验 (sample 是否适合 EFA)
  2. 特征值 / scree
  3. 4 factor EFA (PKVC 假设), varimax 旋转
  4. 因子载荷 + 累计方差
  5. user 4 因子 score
  6. K-means K=3..7 on 4 因子 score
  7. silhouette 对比原始 30 维
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_pipeline.data.prep import EXCLUDED_USERS  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_FACTORS = 4
K_SWEEP = [3, 4, 5, 6, 7]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"数据目录, 默认 {DEFAULT_DATA_DIR}")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_LOADING = DATA_DIR / "efa_loadings.csv"
    OUT_SCORES = DATA_DIR / "user_factor_scores.csv"
    OUT_CLUSTER = DATA_DIR / "user_cluster_efa.csv"

    scales = pd.read_csv(DATA_DIR / "scale_responses.csv")
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    n_before = scales["user_id"].nunique()
    scales = scales[~scales["user_id"].astype(str).isin(EXCLUDED_USERS)]
    n_after = scales["user_id"].nunique()
    if n_before != n_after:
        print(f"⚠ Excluded {n_before - n_after} user(s) from EFA: "
              f"{sorted(EXCLUDED_USERS)} → N={n_after}\n")

    mat = scales.pivot_table(
        index="user_id", columns="question_node", values="score", aggfunc="mean"
    ).fillna(0)
    print(f"Matrix shape: {mat.shape}")
    if mat.shape[0] < 150:
        print(f"⚠ N={mat.shape[0]} < 5×30=150 推荐下限, EFA 稳定性受限\n")
    else:
        print(f"✓ N={mat.shape[0]} ≥ 150 推荐下限\n")

    # KMO + Bartlett
    print("=== 适配性检验 ===")
    chi2, p = calculate_bartlett_sphericity(mat)
    print(f"Bartlett 球形检验:  chi2={chi2:.2f}, p={p:.4g}  (p<0.05 适合 EFA)")
    kmo_per_item, kmo_total = calculate_kmo(mat)
    print(f"KMO 总值: {kmo_total:.3f}  (>0.6 良好, >0.5 可接受, <0.5 不适合)")

    # 特征值 (用相关矩阵的特征值)
    X_std = StandardScaler().fit_transform(mat.values)
    corr = np.corrcoef(X_std.T)
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    print(f"\n=== 特征值 (前 10, 相关矩阵) ===")
    for i, e in enumerate(eigvals[:10], 1):
        marker = " ←" if e > 1 else ""
        print(f"  factor {i:>2}: eigenvalue={e:.3f}{marker}")
    print(f"  特征值 > 1 的因子数: {(eigvals > 1).sum()}  (Kaiser 准则)")

    # 4 factor EFA varimax (sklearn 自带)
    print(f"\n=== {N_FACTORS} 因子 EFA (sklearn FA + varimax) ===")
    fa = FactorAnalysis(n_components=N_FACTORS, rotation="varimax", random_state=SEED)
    fa.fit(X_std)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=mat.columns,
        columns=[f"F{i+1}" for i in range(N_FACTORS)],
    )

    var = (loadings.values ** 2).sum(axis=0)
    prop = var / X_std.shape[1]
    cum = prop.cumsum()
    print(f"  因子方差解释:")
    for i in range(N_FACTORS):
        print(f"    F{i+1}: var={var[i]:.2f}, prop={prop[i]:.3f}, cum={cum[i]:.3f}")

    print(f"\n=== 因子载荷 (|载荷| > 0.4 的题目, 每因子 top-5) ===")
    for f in range(N_FACTORS):
        col = f"F{f+1}"
        top = loadings[col].abs().sort_values(ascending=False).head(5)
        print(f"  {col}:")
        for q, abs_l in top.items():
            actual = loadings.loc[q, col]
            marker = "★" if abs(actual) > 0.4 else " "
            print(f"    {marker} {q:20s}  loading={actual:+.3f}")

    loadings.to_csv(OUT_LOADING)
    print(f"\nSaved loadings to: {OUT_LOADING}")

    # User 因子 score
    scores = fa.transform(X_std)
    score_df = pd.DataFrame(scores, index=mat.index, columns=[f"F{i+1}" for i in range(N_FACTORS)])
    score_df.to_csv(OUT_SCORES)
    print(f"Saved factor scores to: {OUT_SCORES}")

    # K-means on 4 因子空间
    Xs = StandardScaler().fit_transform(scores)
    print(f"\n=== K-means on 4 因子 score, K-sweep ===")
    print(f"{'K':>3}  {'silhouette':>12}  {'min_size':>8}  {'max_size':>8}")
    best_k, best_sil, best_labels = None, -np.inf, None
    for k in K_SWEEP:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        sizes = pd.Series(km.labels_).value_counts()
        print(f"{k:>3}  {sil:>12.4f}  {sizes.min():>8d}  {sizes.max():>8d}")
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, km.labels_

    print(f"\nBest K={best_k}, silhouette={best_sil:.4f}")

    # 对比原 30 维 K=5 silhouette = 0.047
    print(f"\n=== 对比 ===")
    print(f"  原 30 量表 K=5 silhouette ≈ 0.047")
    print(f"  EFA 4 因子 K=5 silhouette = "
          f"{silhouette_score(Xs, KMeans(n_clusters=5, random_state=SEED, n_init=20).fit_predict(Xs)):.4f}")
    print(f"  EFA 4 因子 best K={best_k} silhouette = {best_sil:.4f}")

    pd.DataFrame({"user_id": mat.index.tolist(), "cluster_id": best_labels}).to_csv(OUT_CLUSTER, index=False)
    print(f"\nSaved best cluster to: {OUT_CLUSTER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
