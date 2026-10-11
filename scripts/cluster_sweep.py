"""Cluster sweep: 量表 + Q1-Q6 PKVC 选项 特征, 多算法对比.

Features (54 维):
  - 30 量表 (numerical, 6 点 Likert)
  - Q1 one-hot (4 维, A/B/C/D)
  - Q2 PKVC 因子平均排序位置 (4 维: P/K/V/C)
  - Q3-Q6 one-hot (4 题 × 4 维 = 16 维)

Algorithms:
  - K-means (K=3..7)
  - Gaussian Mixture (K=3..7)
  - Agglomerative (K=5, linkage=ward/complete/average)
  - HDBSCAN (auto-K, min_cluster_size=5/10)
  - Spectral (K=5)

输出: silhouette / cluster sizes, 最佳 silhouette 的 labels 存 user_cluster_v2.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

DATA_DIR = Path(r"D:\Designer_RolePlay\0512技术方案\data")
OUT_PATH = DATA_DIR / "user_cluster_v2.csv"
SEED = 42
PKVC_FACTORS = ["P", "K", "V", "C"]


def extract_letter(answer: str) -> str | None:
    # 从 answer_value 字符串里提取第一个 A./B./C./D. 的字母
    if pd.isna(answer):
        return None
    m = re.search(r'[A-E]\.', str(answer))
    return m.group(0)[0] if m else None


def parse_q2(json_str: str) -> dict:
    """Q2 JSON → 每个 PKVC 因子的平均排序位置 (没出现的因子 = 默认 11)."""
    try:
        items = json.loads(str(json_str).replace('""', '"'))
    except json.JSONDecodeError:
        return {f: 11 for f in PKVC_FACTORS}
    factor_orders = {f: [] for f in PKVC_FACTORS}
    for it in items:
        cid = it.get("cardId", "")
        if cid and cid[0] in PKVC_FACTORS:
            factor_orders[cid[0]].append(it.get("order", 11))
    return {f: (np.mean(v) if v else 11.0) for f, v in factor_orders.items()}


def build_features(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    scales = pd.read_csv(data_dir / "scale_responses.csv")
    tasks = pd.read_csv(data_dir / "task_responses.csv")
    for df in [scales, tasks]:
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # 量表 pivot
    scale_mat = scales.pivot_table(
        index="user_id", columns="question_node", values="score", aggfunc="mean"
    ).fillna(0)

    # 79 个完整 user (有 scale)
    valid_users = scale_mat.index.tolist()
    tasks = tasks[tasks["user_id"].isin(valid_users)].copy()

    # Q1, Q3-Q6 one-hot
    qs_to_onehot = ["Q1", "Q3", "Q4", "Q5", "Q6"]
    letter_cols = []
    for q in qs_to_onehot:
        q_df = tasks[tasks["question_id"] == q].copy()
        q_df["letter"] = q_df["answer_value"].apply(extract_letter)
        oh = pd.get_dummies(q_df.set_index("user_id")["letter"], prefix=f"{q}_letter")
        scale_mat = scale_mat.join(oh, how="left").fillna(0)
        letter_cols.extend(oh.columns.tolist())

    # Q2 PKVC 因子平均位置
    q2_df = tasks[tasks["question_id"] == "Q2"].copy()
    q2_features = q2_df.set_index("user_id")["answer_value"].apply(parse_q2).apply(pd.Series)
    q2_features.columns = [f"Q2_factor_pos_{c}" for c in q2_features.columns]
    scale_mat = scale_mat.join(q2_features, how="left").fillna(11)

    feature_cols = scale_mat.columns.tolist()
    return scale_mat, feature_cols


def evaluate(name: str, labels: np.ndarray, X: np.ndarray) -> dict:
    valid = labels != -1  # HDBSCAN 用 -1 表示噪点
    n_clusters = len(set(labels[valid]))
    n_noise = (labels == -1).sum()
    if n_clusters < 2 or valid.sum() < 2:
        return {"algo": name, "n_cluster": n_clusters, "n_noise": n_noise,
                "silhouette": np.nan, "min_size": np.nan, "max_size": np.nan}
    sil = silhouette_score(X[valid], labels[valid])
    sizes = pd.Series(labels[valid]).value_counts()
    return {
        "algo": name,
        "n_cluster": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
        "min_size": sizes.min(),
        "max_size": sizes.max(),
    }


def main() -> int:
    print("=== Loading features ===")
    feat_df, feature_cols = build_features(DATA_DIR)
    print(f"Shape: {feat_df.shape} (users × features)")
    print(f"Features: 量表 30 + Q1 oh + Q2 pkvc4 + Q3-Q6 oh = {len(feature_cols)} 维")

    X = StandardScaler().fit_transform(feat_df.values)
    user_ids = feat_df.index.tolist()

    results = []
    best_sil = -np.inf
    best_labels = None
    best_name = None

    print("\n=== K-means sweep ===")
    for k in [3, 4, 5, 6, 7]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X)
        r = evaluate(f"KMeans K={k}", km.labels_, X)
        results.append(r); print(r)
        if r["silhouette"] > best_sil:
            best_sil, best_labels, best_name = r["silhouette"], km.labels_, r["algo"]

    print("\n=== Gaussian Mixture sweep ===")
    for k in [3, 4, 5, 6, 7]:
        gmm = GaussianMixture(n_components=k, random_state=SEED, n_init=10).fit(X)
        labels = gmm.predict(X)
        r = evaluate(f"GMM K={k}", labels, X)
        results.append(r); print(r)
        if r["silhouette"] > best_sil:
            best_sil, best_labels, best_name = r["silhouette"], labels, r["algo"]

    print("\n=== Agglomerative (K=5) ===")
    for linkage in ["ward", "complete", "average"]:
        agg = AgglomerativeClustering(n_clusters=5, linkage=linkage).fit(X)
        r = evaluate(f"Agglo K=5 {linkage}", agg.labels_, X)
        results.append(r); print(r)
        if r["silhouette"] > best_sil:
            best_sil, best_labels, best_name = r["silhouette"], agg.labels_, r["algo"]

    print("\n=== Spectral (K=5) ===")
    try:
        spec = SpectralClustering(n_clusters=5, random_state=SEED, affinity="rbf",
                                  assign_labels="kmeans").fit(X)
        r = evaluate("Spectral K=5", spec.labels_, X)
        results.append(r); print(r)
        if r["silhouette"] > best_sil:
            best_sil, best_labels, best_name = r["silhouette"], spec.labels_, r["algo"]
    except Exception as e:
        print(f"Spectral failed: {e}")

    if HAS_HDBSCAN:
        print("\n=== HDBSCAN (auto-K) ===")
        for mcs in [5, 10]:
            hdb = HDBSCAN(min_cluster_size=mcs).fit(X)
            r = evaluate(f"HDBSCAN mcs={mcs}", hdb.labels_, X)
            results.append(r); print(r)
            if r["silhouette"] > best_sil and r["n_cluster"] >= 3:
                best_sil, best_labels, best_name = r["silhouette"], hdb.labels_, r["algo"]
    else:
        print("\n(HDBSCAN 不可用, sklearn < 1.3)")

    print("\n=== Ranking by silhouette ===")
    ranked = sorted(results, key=lambda r: r["silhouette"] if not np.isnan(r["silhouette"]) else -np.inf, reverse=True)
    for r in ranked[:10]:
        print(f"  {r['algo']:25s} sil={r['silhouette']:.4f}  K={r['n_cluster']}  "
              f"sizes=[{r['min_size']}, {r['max_size']}]  noise={r['n_noise']}")

    print(f"\nBest: {best_name}, silhouette={best_sil:.4f}")
    pd.DataFrame({"user_id": user_ids, "cluster_id": best_labels}).to_csv(OUT_PATH, index=False)
    print(f"Saved best cluster to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
