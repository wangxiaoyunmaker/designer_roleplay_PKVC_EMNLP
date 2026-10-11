"""K-means cluster pilot on 30-dim scale responses.

输入: scale_responses.csv (long format, N × 30 题)
输出:
  - N × 30 user-by-question 矩阵
  - K=3..7 多个 K 的 silhouette + inertia
  - K=5 的 cluster 标签 (按 framing 锁定) + per-cluster 大小
  - cluster 标签存到 user_cluster.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_pipeline.data.prep import EXCLUDED_USERS  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_CLUSTERS_FINAL = 5
N_CLUSTERS_SWEEP = [3, 4, 5, 6, 7]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"数据目录, 默认 {DEFAULT_DATA_DIR}")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_PATH = DATA_DIR / "user_cluster.csv"

    scales = pd.read_csv(DATA_DIR / "scale_responses.csv")
    # 注意 user_id 列可能带 BOM, 清一下
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    n_before = scales["user_id"].nunique()
    scales = scales[~scales["user_id"].astype(str).isin(EXCLUDED_USERS)]
    n_after = scales["user_id"].nunique()
    if n_before != n_after:
        print(f"⚠ Excluded {n_before - n_after} user(s): "
              f"{sorted(EXCLUDED_USERS)} → N={n_after}\n")

    # pivot 成 79 × 30 矩阵
    matrix = scales.pivot_table(
        index="user_id",
        columns="question_node",
        values="score",
        aggfunc="mean",  # 如果同 user 同 node 多次提交, 取均值
    )
    print(f"Matrix shape: {matrix.shape}")
    print(f"Question nodes ({matrix.shape[1]}): {list(matrix.columns)[:5]} ...")
    print(f"Missing cells: {matrix.isna().sum().sum()}")

    if matrix.isna().any().any():
        print("⚠ 有 NA, 用列均值填充")
        matrix = matrix.fillna(matrix.mean())

    X = matrix.values
    user_ids = matrix.index.tolist()

    # 标准化
    Xs = StandardScaler().fit_transform(X)

    # K-sweep
    print("\n=== K sweep ===")
    print(f"{'K':>3}  {'inertia':>10}  {'silhouette':>12}")
    for k in N_CLUSTERS_SWEEP:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"{k:>3}  {km.inertia_:>10.2f}  {sil:>12.4f}")

    # final K=5
    print(f"\n=== Final K={N_CLUSTERS_FINAL} ===")
    km = KMeans(n_clusters=N_CLUSTERS_FINAL, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    print(f"Silhouette: {sil:.4f}")

    df = pd.DataFrame({"user_id": user_ids, "cluster_id": labels})
    print("\nCluster size:")
    print(df["cluster_id"].value_counts().sort_index().to_string())

    # 每个 cluster 的 centroid (原始尺度反算)
    centroids_std = km.cluster_centers_
    print("\n=== Cluster centroid (标准化空间, top-3 维 abs 最大) ===")
    feature_names = matrix.columns.tolist()
    for c in range(N_CLUSTERS_FINAL):
        idx_sorted = np.argsort(-np.abs(centroids_std[c]))[:3]
        sig = ", ".join(f"{feature_names[i]}={centroids_std[c, i]:+.2f}" for i in idx_sorted)
        print(f"  cluster {c} (n={(labels==c).sum()}): {sig}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cluster labels to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
