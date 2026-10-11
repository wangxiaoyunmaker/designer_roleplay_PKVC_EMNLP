"""L2: EFA pilot — KMO, Bartlett, parallel analysis, 3/4/5/6 factor sweep, Cronbach α."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Compat shim: factor_analyzer 0.5.1 uses sklearn's old `force_all_finite` kwarg,
# but sklearn ≥1.6 renamed it to `ensure_all_finite`. Translate before importing
# factor_analyzer.
import sklearn.utils.validation as _sk_validation  # noqa: E402

_orig_check_array = _sk_validation.check_array


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


_sk_validation.check_array = _check_array_compat  # type: ignore[assignment]

from factor_analyzer import FactorAnalyzer  # noqa: E402
from factor_analyzer.factor_analyzer import (  # noqa: E402
    calculate_bartlett_sphericity,
    calculate_kmo,
)

# factor_analyzer 在 import 时已经把 check_array 绑到本地命名空间,
# 必须再 patch 它自己的 module reference.
import factor_analyzer.factor_analyzer as _fa_mod  # noqa: E402

_fa_mod.check_array = _check_array_compat  # type: ignore[assignment]
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import loaders  # noqa: E402

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def cronbach_alpha(items: np.ndarray) -> float:
    """items: n × k matrix, k items. Return α."""
    k = items.shape[1]
    if k < 2:
        return float("nan")
    item_var = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def parallel_analysis(data: np.ndarray, n_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Horn parallel analysis: simulate random eigenvalues, return 95-th percentile per index."""
    rng = np.random.default_rng(seed)
    n, k = data.shape
    rand_eigs = np.zeros((n_iter, k))
    for i in range(n_iter):
        rand = rng.standard_normal((n, k))
        corr = np.corrcoef(rand, rowvar=False)
        rand_eigs[i] = np.sort(np.linalg.eigvalsh(corr))[::-1]
    return np.percentile(rand_eigs, 95, axis=0)


def run(output_path: Path | None = None, k_candidates: tuple[int, ...] = (3, 4, 5, 6)) -> dict:
    output_path = output_path or _REPORTS / "L2_efa.md"
    scale = loaders.load_scale_wide().dropna()
    n, p = scale.shape

    findings: dict = {"n": int(n), "p": int(p), "n_per_item": round(n / p, 2)}
    sections: list[str] = []
    sections.append(f"# L2 EFA Pilot (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- N = {n}, items = {p}, N/item = {n / p:.2f}  ({'临界可做' if n / p >= 5 else 'FAIL <5'})")

    # KMO + Bartlett
    chi2, p_bart = calculate_bartlett_sphericity(scale)
    kmo_all, kmo_model = calculate_kmo(scale)
    findings["kmo"] = float(kmo_model)
    findings["bartlett_chi2"] = float(chi2)
    findings["bartlett_p"] = float(p_bart)
    sections.append(f"- **KMO**: {kmo_model:.3f}  ({'PASS≥0.6' if kmo_model >= 0.6 else 'WARN<0.6'})")
    sections.append(f"- **Bartlett χ²**: {chi2:.1f}, p={p_bart:.2e}  ({'PASS' if p_bart < 0.05 else 'FAIL'})")

    # Eigenvalues + parallel analysis
    X = StandardScaler().fit_transform(scale.values)
    corr = np.corrcoef(X, rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    rand_eigs95 = parallel_analysis(X)
    pa_k = int((eigs > rand_eigs95).sum())
    findings["pa_suggested_k"] = pa_k
    sections.append(f"- **Parallel analysis 建议 K**: {pa_k}  (real eig > 95th pct random eig)")
    sections.append("\n## 特征值 (前 10) vs random 95-pct\n")
    pa_table = pd.DataFrame({"idx": range(1, 11), "real_eig": eigs[:10].round(3), "rand_95": rand_eigs95[:10].round(3)})
    sections.append(pa_table.to_markdown(index=False))

    # 3/4/5/6 factor sweep
    per_k: dict = {}
    for k in k_candidates:
        fa = FactorAnalyzer(n_factors=k, rotation="varimax", method="ml")
        fa.fit(scale)
        loadings = pd.DataFrame(fa.loadings_, index=scale.columns, columns=[f"F{i+1}" for i in range(k)])
        var_explained = fa.get_factor_variance()

        # α per factor: items with |loading| > 0.4 on this factor
        alphas: dict = {}
        for f_idx, f_col in enumerate(loadings.columns):
            items_in = loadings.index[loadings[f_col].abs() > 0.4]
            if len(items_in) >= 2:
                alphas[f_col] = round(cronbach_alpha(scale[items_in].values), 3)
            else:
                alphas[f_col] = None

        sections.append(f"\n## K = {k} factor solution\n")
        sections.append(f"- cumulative variance: {var_explained[2][-1]:.1%}")
        sections.append(f"- Cronbach α per factor (|loading|>0.4): {alphas}")
        sections.append(f"- top 5 items per factor (by |loading|):")
        for f_col in loadings.columns:
            top = loadings[f_col].abs().sort_values(ascending=False).head(5).index.tolist()
            sections.append(f"  - **{f_col}**: {top}")

        per_k[k] = {
            "cumulative_var": float(var_explained[2][-1]),
            "alphas": alphas,
        }

        # save factor scores for **all** sweep K
        scores = pd.DataFrame(fa.transform(scale), index=scale.index, columns=loadings.columns)
        scores_path = _REPORTS / "per_designer" / f"factor_scores_k{k}.csv"
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_path)
        loadings.to_csv(_REPORTS / "per_designer" / f"loadings_k{k}.csv")

    findings["per_k"] = per_k

    sections.append("\n## Verdict\n")
    pa_in_sweep = pa_k in k_candidates
    if pa_in_sweep:
        sections.append(f"- **EFA-K 推荐**: K = {pa_k} (parallel analysis)")
    else:
        clamped = min(k_candidates, key=lambda kk: abs(kk - pa_k))
        sections.append(f"- **EFA-K 推荐**: K = {pa_k} (out of sweep; 下游 clamp 到 K={clamped})")
    sections.append(f"- factor scores 已存 reports/datav3/per_designer/factor_scores_k{{3,4,5,6}}.csv (gitignored)")
    sections.append(f"- 这只决定 **EFA factor count (L_cls head 维度)**, K-means 的 K 由 L3 独立挑")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
