"""统计学验证: B1 EFA + Cronbach's α + KMO + Bartlett, B2 multi-group CFA.

B1 量表 PKVC 4 因子验证 (paper Tier 1, EMNLP main 必跑).
B2 跨场景 (Q7 vs Q1-Q6) measurement invariance (Tier 2, 直接关联 R2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


@dataclass
class B1Result:
    n: int
    n_features: int
    kmo_total: float
    bartlett_chi2: float
    bartlett_p: float
    eigenvalues_top10: list[float]
    n_factors_kaiser: int
    loadings: pd.DataFrame
    explained_variance_cum: list[float]
    cronbach_alpha_per_factor: dict[str, float]
    cronbach_n_per_factor: dict[str, int]
    factor_method: str = "sklearn-FA (ML, varimax rotation)"
    common_variance_total: float = float("nan")
    rotation_orthogonality_residual: float = float("nan")


def calculate_kmo(X: np.ndarray) -> tuple[np.ndarray, float]:
    """KMO 测度 (Kaiser-Meyer-Olkin), > 0.6 良好, > 0.5 可接受."""
    corr = np.corrcoef(X.T)
    inv_corr = np.linalg.pinv(corr)
    n = corr.shape[0]
    partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial_corr, 0)
    np.fill_diagonal(corr, 0)
    kmo_per_item = np.sum(corr ** 2, axis=1) / (
        np.sum(corr ** 2, axis=1) + np.sum(partial_corr ** 2, axis=1) + 1e-9
    )
    kmo_total = float(np.sum(corr ** 2) / (np.sum(corr ** 2) + np.sum(partial_corr ** 2) + 1e-9))
    return kmo_per_item, kmo_total


def calculate_bartlett(X: np.ndarray) -> tuple[float, float]:
    """Bartlett 球形检验 (p < 0.05 适合 EFA).

    P2-3 fix: use ``chi2.sf`` instead of ``1 - chi2.cdf`` to avoid precision
    loss to 0 in the upper tail of the chi-square distribution.
    """
    from scipy.stats import chi2
    n, p = X.shape
    corr = np.corrcoef(X.T)
    det = np.linalg.det(corr)
    chi2_stat = -((n - 1) - (2 * p + 5) / 6) * np.log(max(det, 1e-300))
    df = p * (p - 1) / 2
    p_value = chi2.sf(chi2_stat, df)
    return float(chi2_stat), float(p_value)


def cronbach_alpha(items: pd.DataFrame) -> tuple[float, int]:
    """Cronbach's α: 内部一致性, > 0.7 可接受.

    P2-4 fix: 不再 silently dropna (旧逻辑下每个因子可能基于不同子样本算 α,
    数值不可比). 改成: 先用列均值填充缺失, 然后报告实际有效行数 ``n_used``.

    Returns:
        ``(alpha, n_used)``: alpha 为 Cronbach's α, n_used 为参与计算的行数
        (= items.shape[0], 因为我们填充而非丢弃).
    """
    if items.shape[1] < 2 or items.shape[0] < 2:
        return float("nan"), 0
    # Impute column mean for missing cells (preserve N).
    filled = items.copy()
    if filled.isna().any().any():
        filled = filled.fillna(filled.mean(numeric_only=True))
    item_var = filled.var(axis=0, ddof=1)
    total_var = filled.sum(axis=1).var(ddof=1)
    k = filled.shape[1]
    if total_var <= 0:
        return float("nan"), int(filled.shape[0])
    alpha = float(k / (k - 1) * (1 - item_var.sum() / total_var))
    return alpha, int(filled.shape[0])


def compute_b1(scale_matrix: pd.DataFrame, n_factors: int = 4) -> B1Result:
    """Run the B1 EFA pipeline: KMO + Bartlett + EFA + varimax + Cronbach α.

    Methodological notes
    --------------------
    * **EFA implementation** (P1-3): we use ``sklearn.decomposition.FactorAnalysis``
      with varimax rotation. This is an ML-based factor model, NOT Principal
      Axis Factoring (PAF) which is more common in psychometrics. If
      reviewers question this choice, swap to
      ``factor_analyzer.FactorAnalyzer(rotation='varimax', method='minres')``
      with a pinned sklearn version compatible with that library. The numeric
      results may shift by a few %% but factor structure should be similar.
      Method choice is recorded in ``B1Result.factor_method`` and surfaced
      in the rendered report.
    * **Variance explained** (P1-4): after varimax orthogonal rotation, the
      sum of squared loadings divided by the number of items equals the total
      *common* variance explained by all factors jointly. We report it as
      ``common_variance_total`` (not "cumulative variance" – the wording was
      ambiguous in the prior version). Orthogonality is asserted via the
      Frobenius-norm residual of ``L^T L - diag(L^T L)`` and recorded.
    """
    X = scale_matrix.values
    X_std = StandardScaler().fit_transform(X)

    _, kmo_total = calculate_kmo(X_std)
    chi2_stat, p_value = calculate_bartlett(X_std)

    eigvals = np.sort(np.linalg.eigvalsh(np.corrcoef(X_std.T)))[::-1]
    n_kaiser = int((eigvals > 1).sum())

    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=42)
    fa.fit(X_std)
    loadings_arr = fa.components_.T
    loadings = pd.DataFrame(
        loadings_arr, index=scale_matrix.columns,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )

    # Sanity: after varimax, factors should be ~orthogonal.
    # Residual = ||L^T L - diag(L^T L)||_F should be small relative to ||L^T L||_F.
    ltl = loadings_arr.T @ loadings_arr
    off_diag = ltl - np.diag(np.diag(ltl))
    orth_residual = float(np.linalg.norm(off_diag) / (np.linalg.norm(ltl) + 1e-12))

    # Per-factor variance explained (orthogonal rotation: factors are
    # uncorrelated → individual contributions sum to total common variance).
    var_per_factor = (loadings_arr ** 2).sum(axis=0) / X_std.shape[1]
    cum_var = list(np.cumsum(var_per_factor))
    common_variance_total = float(cum_var[-1]) if cum_var else float("nan")

    # Per-factor Cronbach α on items with |loading| > 0.4.
    alpha_per_factor: dict[str, float] = {}
    n_per_factor: dict[str, int] = {}
    for f in range(n_factors):
        col = f"F{f+1}"
        items_in_factor = loadings.index[loadings[col].abs() > 0.4].tolist()
        if len(items_in_factor) >= 2:
            a, n_used = cronbach_alpha(scale_matrix[items_in_factor])
            alpha_per_factor[col] = a
            n_per_factor[col] = n_used
        else:
            alpha_per_factor[col] = float("nan")
            n_per_factor[col] = 0

    return B1Result(
        n=X.shape[0], n_features=X.shape[1],
        kmo_total=kmo_total,
        bartlett_chi2=chi2_stat, bartlett_p=p_value,
        eigenvalues_top10=eigvals[:10].tolist(),
        n_factors_kaiser=n_kaiser,
        loadings=loadings,
        explained_variance_cum=cum_var,
        cronbach_alpha_per_factor=alpha_per_factor,
        cronbach_n_per_factor=n_per_factor,
        common_variance_total=common_variance_total,
        rotation_orthogonality_residual=orth_residual,
    )


def compute_b2_multigroup_cfa(
    scale_matrix_scenario1: pd.DataFrame,
    scale_matrix_scenario2: pd.DataFrame,
    factor_structure: dict[str, list[str]],
) -> dict:
    """B2 跨场景 measurement invariance.

    需要装 semopy 才能跑完整 CFA. 当前给出 placeholder.
    """
    try:
        import semopy  # noqa: F401
    except ImportError:
        return {
            "status": "skipped",
            "reason": "semopy 未装, 跳过 B2 CFA",
            "install": "uv pip install semopy",
        }

    # TODO: 实现 multi-group CFA
    # 步骤:
    # 1. 写 lavaan-style model spec from factor_structure
    # 2. fit configural model on group 1 + group 2
    # 3. add metric invariance (loadings equal)
    # 4. add scalar invariance (intercepts equal)
    # 5. 报告 chi-square diff test + CFI / RMSEA
    return {"status": "todo", "note": "需要 semopy + 双场景数据齐全后实现"}


def _kmo_grade(kmo: float) -> str:
    if kmo >= 0.7:
        return "良好"
    if kmo >= 0.6:
        return "可接受"
    if kmo >= 0.5:
        return "勉强"
    return "不适合"


def _alpha_grade(alpha: float) -> str:
    if alpha != alpha:  # NaN
        return "—"
    if alpha >= 0.9:
        return "极优"
    if alpha >= 0.8:
        return "良好"
    if alpha >= 0.7:
        return "可接受"
    if alpha >= 0.5:
        return "勉强"
    return "差"


def render_b1_report(result: B1Result) -> str:
    """Render a paper-ready Markdown report of B1 EFA validation."""
    lines = ["## B1 PKVC Validation (EFA + Cronbach's α)", ""]

    # Method note (P1-3 caveat surfaced for transparency)
    lines.append("### Methodology")
    lines.append("")
    lines.append(f"- Factor method: **{result.factor_method}**")
    if "sklearn" in result.factor_method.lower():
        lines.append(
            "  - 注: sklearn 的 FactorAnalysis 是 ML-based, 非心理统计常用的 PAF/minres. "
            "若审稿质疑可换 factor_analyzer.FactorAnalyzer(method='minres', rotation='varimax')."
        )
    lines.append(
        f"- Orthogonality residual after varimax: {result.rotation_orthogonality_residual:.3e} "
        f"({'✓ 正交' if result.rotation_orthogonality_residual < 0.05 else '⚠ 弱正交'})"
    )
    lines.append("")

    # Summary
    lines.append("### 总览")
    lines.append("")
    lines.append("| 指标 | 值 | 评级 |")
    lines.append("|---|---:|:---:|")
    lines.append(f"| Sample size (N) | {result.n} | {'⚠ 偏小 (建议 ≥ 150)' if result.n < 150 else '充足'} |")
    lines.append(f"| Features | {result.n_features} | — |")
    lines.append(f"| **KMO** | {result.kmo_total:.3f} | **{_kmo_grade(result.kmo_total)}** |")
    lines.append(f"| **Bartlett** χ² | {result.bartlett_chi2:.2f} | p = {result.bartlett_p:.3g} |")
    lines.append(f"| Kaiser (eigenvalue > 1) | {result.n_factors_kaiser} 因子 | — |")
    lines.append(
        f"| {len(result.explained_variance_cum)} 因子总共同方差 (varimax) | "
        f"{result.common_variance_total*100:.1f}% | "
        f"{'✓' if result.common_variance_total >= 0.5 else '⚠ 低'} |"
    )
    lines.append("")

    # Cronbach's α
    lines.append("### Cronbach's α (内部一致性, 缺失值用列均值填充)")
    lines.append("")
    lines.append("| Factor | α | n_used | 评级 |")
    lines.append("|:---:|---:|---:|:---:|")
    for f, a in result.cronbach_alpha_per_factor.items():
        a_disp = f"{a:.3f}" if a == a else "—"
        n_used = result.cronbach_n_per_factor.get(f, 0)
        lines.append(f"| {f} | {a_disp} | {n_used} | {_alpha_grade(a)} |")
    lines.append("")

    # Top loadings
    lines.append("### Top loadings per factor (按 |loading| 排序, 阈值 0.4)")
    lines.append("")
    for col in result.loadings.columns:
        top = result.loadings[col].abs().sort_values(ascending=False).head(8)
        lines.append(f"**{col}**:")
        lines.append("")
        lines.append("| 题目 | Loading | 强度 |")
        lines.append("|:---|---:|:---:|")
        for q in top.index:
            l = float(result.loadings.loc[q, col])
            mark = "★" if abs(l) > 0.5 else ("✓" if abs(l) > 0.4 else "·")
            lines.append(f"| {q} | {l:+.3f} | {mark} |")
        lines.append("")

    # Eigenvalues table
    lines.append("### 前 10 特征值 (Scree)")
    lines.append("")
    lines.append("| Factor # | Eigenvalue | > 1 ? |")
    lines.append("|:---:|---:|:---:|")
    for i, ev in enumerate(result.eigenvalues_top10, 1):
        lines.append(f"| {i} | {ev:.3f} | {'✓' if ev > 1 else '·'} |")
    lines.append("")

    # Diagnostic block
    lines.append("### 诊断与建议")
    lines.append("")
    if result.kmo_total < 0.6:
        lines.append(
            f"- ⚠ **KMO {result.kmo_total:.3f} < 0.6**, 数据规模或量表相关性偏弱; "
            f"建议数据加倍后重跑."
        )
    if result.bartlett_p > 0.05:
        lines.append(
            f"- ⚠ Bartlett p={result.bartlett_p:.3g} > 0.05, 量表内部相关不显著, "
            f"框架假设受质疑."
        )
    weak_alpha = [f for f, a in result.cronbach_alpha_per_factor.items() if a == a and a < 0.7]
    if weak_alpha:
        lines.append(
            f"- ⚠ Cronbach's α < 0.7 的因子: {', '.join(weak_alpha)}; "
            f"内部一致性偏弱, 因子题目可能需要重组."
        )
    if result.explained_variance_cum[-1] < 0.5:
        lines.append(
            f"- ⚠ 4 因子累计方差 {result.explained_variance_cum[-1]*100:.1f}% < 50%, "
            f"模型可能 under-fit 真实结构 (Kaiser 给 {result.n_factors_kaiser} 因子)."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale-csv", required=True)
    p.add_argument("--n-factors", type=int, default=4)
    args = p.parse_args()

    scales = pd.read_csv(args.scale_csv)
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    mat = scales.pivot_table(index="user_id", columns="question_node", values="score", aggfunc="mean").fillna(0)
    result = compute_b1(mat, n_factors=args.n_factors)
    print(render_b1_report(result))
