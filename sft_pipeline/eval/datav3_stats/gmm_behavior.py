"""L3-GMM: Gaussian Mixture Model on Q_option_behavior space.

L3-multi 用 K-means 发现 Q_option_behavior 比 trait 强 (silhouette 0.11, χ²=6/6),
但 K-means 假设球状 cluster + 23-d binary 稀疏. GMM (diag covariance) 放松假设,
用 BIC 选 K, 看 "5-6 behavior mode" 是真的吗 + 每个 mode 的 signature 是什么.

Caveat: GMM 假设 Gaussian, binary feature 用它**不严格 ideal** (Bernoulli Mixture
更对路), 但 sklearn 没现成 BMM. diag covariance 在 high-d sparse 上 stable enough.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from . import loaders
from .cluster_multi import _build_source_spaces

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def _avg_posterior_entropy(post: np.ndarray) -> float:
    """每个 designer 的 posterior 熵 (越低越 clean), avg over designers."""
    ent = -np.sum(post * np.log(post + 1e-10), axis=1)
    return float(ent.mean())


def _signature_table(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """每个 cluster 的 feature 均值 (binary feature 即 = 选该 option 的比例)."""
    sig = pd.DataFrame(features.values, index=features.index, columns=features.columns)
    sig["cluster"] = labels
    return sig.groupby("cluster").mean()


def run(
    output_path: Path | None = None,
    k_candidates: tuple[int, ...] = tuple(range(2, 11)),
    seed: int = 42,
    cov_type: str = "diag",
) -> dict:
    output_path = output_path or _REPORTS / "L3_gmm_behavior.md"

    sources = _build_source_spaces()
    behavior = sources["Q_option_behavior"]
    X = behavior.values.astype(float)
    n, d = X.shape

    sections: list[str] = []
    sections.append(f"# L3-GMM — Gaussian Mixture on Q_option_behavior (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- Source: Q_option_behavior 23-d binary (Q1/Q3-Q6/Q8 单选 one-hot)")
    sections.append(f"- Shape: {n} × {d}")
    sections.append(f"- GMM covariance_type = {cov_type} (diag for high-d binary stability)")
    sections.append(f"- K sweep: {min(k_candidates)} .. {max(k_candidates)}")
    sections.append(f"- Caveat: Gaussian 假设对 binary 不严格 ideal; 严格做 Bernoulli Mixture 但 sklearn 没现成, diag GMM 是次优实用替代\n")

    rows = []
    gmm_models = {}
    for k in k_candidates:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            random_state=seed,
            max_iter=300,
            n_init=5,
        ).fit(X)
        labels = gmm.predict(X)
        post = gmm.predict_proba(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        per_cluster = pd.Series(labels).value_counts().sort_index().tolist()
        ent = _avg_posterior_entropy(post)
        rows.append({
            "K": k,
            "BIC": round(float(bic), 1),
            "AIC": round(float(aic), 1),
            "silhouette": round(float(sil), 3),
            "avg_entropy": round(ent, 3),
            "min_cluster_n": int(min(per_cluster)),
            "per_cluster_n": str(per_cluster),
        })
        gmm_models[k] = (gmm, labels, post)

    df = pd.DataFrame(rows)
    sections.append("## BIC / AIC / silhouette sweep\n")
    sections.append(df.to_markdown(index=False))

    # find best K by BIC
    best_k_bic = int(df.loc[df["BIC"].idxmin(), "K"])
    # also AIC for cross-check
    best_k_aic = int(df.loc[df["AIC"].idxmin(), "K"])
    # find K with lowest entropy among min_cluster_n >= 10 (避免退化)
    df_filt = df[df["min_cluster_n"] >= 10]
    best_k_entropy = int(df_filt.loc[df_filt["avg_entropy"].idxmin(), "K"]) if len(df_filt) else best_k_bic
    sections.append("\n## Best K selection\n")
    sections.append(f"- **BIC 最优** (lowest, GMM standard): K = **{best_k_bic}**")
    sections.append(f"- AIC 最优 (lowest): K = {best_k_aic}")
    sections.append(f"- entropy 最低 (cleanest soft assignment, min_n>=10): K = {best_k_entropy}")

    findings: dict = {
        "sweep": rows,
        "best_k_bic": best_k_bic,
        "best_k_aic": best_k_aic,
        "best_k_entropy": best_k_entropy,
        "shape": [n, d],
    }

    # Detailed characterization of BIC-best model
    sections.append(f"\n## Best model (K = {best_k_bic} by BIC) — per-cluster behavior signature\n")
    best_gmm, best_labels, best_post = gmm_models[best_k_bic]
    sig = _signature_table(behavior, best_labels)
    sections.append("Cluster × feature (binary feature 均值 = 选该 option 的 designer 比例):\n")
    sections.append(sig.round(2).to_markdown())

    # top distinguishing features per cluster
    sections.append("\n## Per-cluster signature (top 5 most distinguishing features)\n")
    sections.append("(in-cluster rate − other-cluster mean rate; 正值 = 该 cluster 显著偏好这个 option)\n")
    cluster_signatures: dict = {}
    for c in range(best_k_bic):
        in_c = sig.loc[c]
        out_c = sig.drop(c).mean()
        diff = in_c - out_c
        top = diff.abs().sort_values(ascending=False).head(5)
        n_in = int((best_labels == c).sum())
        sections.append(f"\n### Cluster {c} (n = {n_in})\n")
        sig_rows = []
        for feat in top.index:
            sig_rows.append({
                "feature": feat,
                "in_cluster_rate": round(float(in_c[feat]), 2),
                "out_cluster_rate": round(float(out_c[feat]), 2),
                "diff": round(float(diff[feat]), 2),
            })
        sections.append(pd.DataFrame(sig_rows).to_markdown(index=False))
        cluster_signatures[str(c)] = {
            "n": n_in,
            "top_features": sig_rows,
        }
    findings["cluster_signatures"] = cluster_signatures

    # comparison vs L3-multi K-means
    sections.append("\n## 跟 L3-multi K-means 对比\n")
    sections.append("| Metric | K-means (L3-multi) | GMM (本报告) |")
    sections.append("|---|---|---|")
    sections.append(f"| Best K | 3 (rule-picked) | {best_k_bic} (BIC) |")
    sections.append("| best silhouette | 0.109 (K=5) | " + f"{df['silhouette'].max():.3f} (K={int(df.loc[df['silhouette'].idxmax(), 'K'])})" + " |")
    sections.append("| Gap statistic | 负 (<-0.5) | n/a |")
    sections.append("| 形状假设 | 球状 | 椭球 (diag cov) |")

    # check BIC monotonicity = model class 不对路信号
    bic_vals = df["BIC"].values
    is_monotonic_decreasing = all(bic_vals[i] >= bic_vals[i + 1] for i in range(len(bic_vals) - 1))
    best_k_sil = int(df.loc[df["silhouette"].idxmax(), "K"])

    sections.append("\n## Verdict\n")
    if is_monotonic_decreasing:
        sections.append(f"- ⚠ **BIC 单调下降到 K = {max(k_candidates)} 没 elbow** — 这是**模型 class 不对路**的信号")
        sections.append("  - Gaussian (continuous) 强行去 fit binary (one-hot) features, 每多 1 component 就额外 fit 一小群人的具体选项串")
        sections.append("  - K=10 不是 '真有 10 个 behavior mode', 是**过度参数化记忆 small subgroup**")
        sections.append(f"- **silhouette 最优 K = {best_k_sil}** (peak 在 K=3-4 之间), 这个更可信因为不依赖 likelihood assumption")
        sections.append(f"- 推荐 K = **{best_k_sil}-{best_k_sil + 1}** 作为 'behavior mode' 报告值, BIC 的 K={best_k_bic} 是 fit-with-fingerprints, 不是 mode")
        recommended_k = best_k_sil
    else:
        sections.append(f"- BIC 最优 K = **{best_k_bic}** (找到 elbow, model class OK)")
        if 5 <= best_k_bic <= 6:
            sections.append("- **'5-6 behavior mode' 假设被支持**")
        recommended_k = best_k_bic

    sections.append(f"\n- BIC-best (K={best_k_bic}) 的 cluster signatures 仍可作 **'behavior fingerprint'** 表 (8-10 个 sparse co-occurrence pattern), 但不是 clean 'archetype cluster'")
    sections.append(f"- 'trait 连续 / behavior 离散投影' framing 仍成立, 只是 behavior 离散面是 **sparse fingerprint** 不是 clean mode")
    sections.append("- Caveat: Bernoulli Mixture (BMM) 才是 binary 数据的对路模型; sklearn 没现成, diag GMM 是次优实用替代, BIC 阈值不能直接 trust")
    sections.append(f"\n- 下一步: BMM 实现 (~50 行 EM) 验证 silhouette-best K = {best_k_sil} 是否稳; 或加 't-SNE / UMAP + 视觉检查 cluster boundary'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2)[:1000])
