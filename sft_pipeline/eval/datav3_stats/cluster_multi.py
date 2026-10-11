"""L3-multi: cluster sweep on 3 source spaces (30-d trait / Q-option behavior / joint).

L3 主报告只在 4-d EFA factor 空间聚, 发现 cluster 弱区分.
这里换 3 个不同的 source space sanity check, 看是否 method artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import loaders
from .cluster import gap_statistic, pick_best_k

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def _build_source_spaces() -> dict[str, pd.DataFrame]:
    """3 source space DataFrames (index = user_id)."""
    scale = loaders.load_scale_wide().dropna()  # 163 × 30
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()]

    # source 1: 30-d standardized trait
    scale_std = pd.DataFrame(
        StandardScaler().fit_transform(scale.values),
        index=scale.index,
        columns=[f"trait_{c}" for c in scale.columns],
    )

    # source 2: Q-option one-hot behavior (Q1, Q3-Q6, Q8 单选)
    target_qs = ["Q1", "Q3", "Q4", "Q5", "Q6", "Q8"]
    q_sub = q_single[q_single["question_id"].isin(target_qs)].copy()
    q_sub["feat"] = q_sub["question_id"].astype(str) + "_" + q_sub["option"].astype(str)
    onehot = (
        pd.crosstab(q_sub["user_id"], q_sub["feat"])
        .pipe(lambda d: (d > 0).astype(int))
    )

    # source 3: joint (trait + behavior), aligned to common users
    common = scale_std.index.intersection(onehot.index)
    joint = pd.concat([scale_std.loc[common], onehot.loc[common].astype(float)], axis=1)

    return {
        "30d_trait": scale_std,
        "Q_option_behavior": onehot,
        "joint_trait_plus_behavior": joint,
    }


def _sweep_source(
    df: pd.DataFrame,
    q_single: pd.DataFrame,
    k_candidates: tuple[int, ...],
    seed: int,
) -> tuple[list[dict], dict, int, str]:
    X = df.values.astype(float)
    rows = []
    label_dict = {}
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = km.labels_
        per = pd.Series(labels).value_counts().sort_index().tolist()
        rows.append({
            "K": k,
            "silhouette": round(float(silhouette_score(X, labels)), 3),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 3),
            "gap": round(float(gap_statistic(X, k, seed=seed)), 3),
            "min_cluster_n": int(min(per)),
            "per_cluster_n": str(per),
        })
        label_dict[k] = labels

    # chi² on per-cluster Q-option distribution
    chi_results: dict[int, dict[str, dict]] = {}
    for k, labels in label_dict.items():
        cm = pd.Series(labels, index=df.index)
        q_k = q_single.copy()
        q_k["cluster"] = q_k["user_id"].map(cm)
        q_k = q_k.dropna(subset=["cluster"])
        q_chi: dict[str, dict] = {}
        for qid in sorted(q_k["question_id"].unique()):
            sub = q_k[q_k["question_id"] == qid]
            ct = pd.crosstab(sub["cluster"], sub["option"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = chi2_contingency(ct)
            q_chi[qid] = {"chi2": round(float(chi2), 2), "p": round(float(p), 4), "dof": int(dof)}
        chi_results[k] = q_chi

    picked_k, reason = pick_best_k(rows, chi_results)
    return rows, chi_results, picked_k, reason


def run(
    output_path: Path | None = None,
    k_candidates: tuple[int, ...] = (3, 4, 5, 6, 7),
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L3_multi_source.md"
    sources = _build_source_spaces()
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()

    findings: dict = {}
    sections: list[str] = []
    sections.append(f"# L3-multi — cluster sweep on 3 source spaces (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append("L3 主报告只在 **4-d EFA factor** 空间跑 K-means, picked K=3 fallback, cluster 弱区分.")
    sections.append("这里换 3 个不同 source space, 比较是否 method artifact (4-d EFA 太压缩) 还是真信号.")
    sections.append("方法: 同 K-means + silhouette/DB/gap + per-cluster Q-option χ² (跟 L3 主同).\n")

    for src_name, df in sources.items():
        sections.append(f"\n## Source: `{src_name}`  (shape = {df.shape})\n")
        rows, chi_results, picked_k, reason = _sweep_source(df, q_single, k_candidates, seed)

        sections.append("**Sweep metrics:**\n")
        sections.append(pd.DataFrame(rows).to_markdown(index=False))

        sections.append("\n**per-K χ² sig count (over Q1/Q3-Q6/Q8 单选题):**\n")
        for k, q_chi in chi_results.items():
            sigs = [qid for qid, v in q_chi.items() if v.get("p", 1.0) < 0.05]
            sections.append(f"- K={k}: **{len(sigs)} sig** ({sigs})")

        sections.append(f"\n**picked K (任务导向规则)**: {picked_k}")
        sections.append(f"- {reason}")

        findings[src_name] = {
            "shape": list(df.shape),
            "sweep": rows,
            "chi_results": {str(k): v for k, v in chi_results.items()},
            "picked_k": int(picked_k),
            "picked_reason": reason,
        }

    # 跨 source 总结
    sections.append("\n---\n\n## 跨 source 对照总结\n")
    sections.append("| Source | shape | best silhouette | picked K | max χ² sig Q (across K) |")
    sections.append("|---|---|---|---|---|")
    src_summaries = {}
    for src_name, f in findings.items():
        best_sil = max(r["silhouette"] for r in f["sweep"])
        max_sig = max(
            sum(1 for v in qc.values() if v.get("p", 1) < 0.05)
            for qc in f["chi_results"].values()
        )
        src_summaries[src_name] = (best_sil, max_sig)
        sections.append(f"| {src_name} | {tuple(f['shape'])} | {best_sil:.3f} | {f['picked_k']} | {max_sig} |")

    # L3 主结果对比 (4-d EFA)
    sections.append("\n参考: L3 主报告 4-d EFA factor 空间 best silhouette ≈ 0.199, max χ² sig = 1.\n")

    sections.append("\n## Verdict\n")
    max_sil_overall = max(s for s, _ in src_summaries.values())
    max_sig_overall = max(m for _, m in src_summaries.values())

    if max_sil_overall < 0.25 and max_sig_overall <= 2:
        sections.append(f"- **所有 source 的 silhouette < 0.25 (最大 {max_sil_overall:.3f}); max χ² sig {max_sig_overall} (在 6 Q 里)**")
        sections.append("- → cluster 弱区分**不是 4-d EFA 的 method artifact**, 多 source 一致, 是真信号")
        sections.append("- → 强支持 'trait-continuous personalization' framing")
        sections.append("- → brief v3.5.1 §3.5 '4 层 cluster conditioning' 经验基础不牢, 建议改 framing")
    elif max_sil_overall >= 0.25:
        best_src = max(src_summaries.items(), key=lambda kv: kv[1][0])
        sections.append(f"- **source `{best_src[0]}` 出现 silhouette ≥ 0.25**")
        sections.append("- → cluster 弱区分**可能是 4-d EFA artifact**, 在原始 space 仍有 structure")
        sections.append(f"- → 重新审视 L3 主结论, 考虑 source space 升级为 `{best_src[0]}`")
    else:
        best_sig_src = max(src_summaries.items(), key=lambda kv: kv[1][1])
        sections.append(f"- silhouette 都弱, 但 source `{best_sig_src[0]}` 在 χ² 上 sig 较多")
        sections.append("- → cluster 形状可能非球状 (K-means 假设错), 建议下一步加 GMM / DBSCAN")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2)[:600])
