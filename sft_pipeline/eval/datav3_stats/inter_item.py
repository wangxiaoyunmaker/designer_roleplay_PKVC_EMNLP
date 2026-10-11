"""L6: 30×30 inter-item correlation — Pearson + Spearman + |r|>0.5 pair table + heatmap.

E1 of 30-dim relationship analysis. 答 "30 题彼此 zero-order 关系长什么样".
互补于 L2 EFA (latent factor) 和 L9 network (conditional partial correlation).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"
_FIGURES = _REPORTS / "figures"

# CJK font fallback (macOS / linux); axis label 仍用数字索引以保证可移植.
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _pearson_pvalues(X: np.ndarray) -> np.ndarray:
    """Pearson p-value matrix via scipy.stats.pearsonr (lazy, p=30 so cost OK)."""
    p = X.shape[1]
    P = np.ones((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            r, pv = stats.pearsonr(X[:, i], X[:, j])
            P[i, j] = P[j, i] = pv
    np.fill_diagonal(P, 0.0)
    return P


def _spearman_full(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman rank correlation + p-value (scipy)."""
    rho, pv = stats.spearmanr(X)
    return np.asarray(rho), np.asarray(pv)


def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR on flat p-values (returns reject mask + threshold)."""
    flat = pvals.ravel()
    n = len(flat)
    order = np.argsort(flat)
    ranked = flat[order]
    crit = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= crit
    if passed.any():
        k_max = np.where(passed)[0].max()
        thresh = ranked[k_max]
    else:
        thresh = 0.0
    reject = pvals <= thresh
    return reject, float(thresh)


def _plot_heatmaps(
    pearson_r: pd.DataFrame,
    spearman_r: pd.DataFrame,
    save_path: Path,
) -> None:
    """Side-by-side lower-triangle heatmap (Pearson left, Spearman right).
    Axis labels = 1..30 index; legend table 在 markdown.
    """
    n = pearson_r.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool))  # mask upper tri (keep lower)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, mat, title in [
        (axes[0], pearson_r.values.copy(), "Pearson r"),
        (axes[1], spearman_r.values.copy(), "Spearman ρ"),
    ]:
        mat[mask] = np.nan
        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=-0.6,
            vmax=0.6,
            interpolation="nearest",
        )
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=7)
        ax.set_yticklabels([str(i + 1) for i in range(n)], fontsize=7)
        ax.set_title(f"{title} (lower tri, N=163)", fontsize=11)
        ax.tick_params(axis="x", which="both", top=False, bottom=True)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_strength_bar(
    items: list[str],
    strength: np.ndarray,
    save_path: Path,
) -> None:
    """Sum |r| per item — 哪几题是 'hub'."""
    order = np.argsort(strength)[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(items))
    ax.barh(y, strength[order], color="steelblue", alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([items[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Σ |Pearson r| (excl. self)", fontsize=10)
    ax.set_title("Item connectivity strength (zero-order)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    output_path: Path | None = None,
    pair_thresh: float = 0.5,
) -> dict:
    output_path = output_path or _REPORTS / "L6_inter_item.md"
    _FIGURES.mkdir(parents=True, exist_ok=True)

    scale = loaders.load_scale_wide().dropna()
    n, p = scale.shape
    items = scale.columns.tolist()
    X = scale.values.astype(float)

    # Pearson
    pearson_r = pd.DataFrame(np.corrcoef(X, rowvar=False), index=items, columns=items)
    pearson_p = _pearson_pvalues(X)
    # Spearman
    spearman_r_arr, spearman_p_arr = _spearman_full(X)
    spearman_r = pd.DataFrame(spearman_r_arr, index=items, columns=items)

    # off-diag flat r distribution
    iu = np.triu_indices(p, k=1)
    pearson_off = pearson_r.values[iu]
    spearman_off = spearman_r.values[iu]
    pearson_p_off = pearson_p[iu]
    spearman_p_off = spearman_p_arr[iu]

    # FDR on Pearson + Spearman p-values (off-diag)
    _, fdr_thresh_p = _bh_fdr(pearson_p_off, alpha=0.05)
    _, fdr_thresh_s = _bh_fdr(spearman_p_off, alpha=0.05)

    # |r| > thresh pairs
    pairs: list[dict] = []
    for k_idx, (i, j) in enumerate(zip(*iu, strict=True)):
        r_p = pearson_r.values[i, j]
        r_s = spearman_r.values[i, j]
        if max(abs(r_p), abs(r_s)) < pair_thresh:
            continue
        pairs.append(
            {
                "i": int(i + 1),
                "j": int(j + 1),
                "item_i": items[i],
                "item_j": items[j],
                "pearson_r": round(float(r_p), 3),
                "pearson_p": float(pearson_p_off[k_idx]),
                "spearman_r": round(float(r_s), 3),
                "spearman_p": float(spearman_p_off[k_idx]),
                "sign_agree": bool((r_p * r_s) > 0),
            }
        )
    pairs.sort(key=lambda d: -abs(d["pearson_r"]))

    # per-item strength (Σ |r| over off-diag)
    strength = np.abs(pearson_r.values).sum(axis=1) - 1.0  # exclude self
    strength_df = pd.DataFrame({"item": items, "strength_sum_abs_r": np.round(strength, 3)})
    strength_df = strength_df.sort_values("strength_sum_abs_r", ascending=False)

    # determinant of correlation matrix (singularity check)
    sign_det, logdet = np.linalg.slogdet(pearson_r.values)
    det_val = float(sign_det * np.exp(logdet)) if sign_det != 0 else 0.0

    # save figures
    heatmap_path = _FIGURES / "L6_inter_item_heatmap.png"
    _plot_heatmaps(pearson_r, spearman_r, heatmap_path)
    strength_path = _FIGURES / "L6_inter_item_strength.png"
    _plot_strength_bar(items, strength, strength_path)

    # save full matrices (csv, gitignored under per_designer/)
    per_designer = _REPORTS / "per_designer"
    per_designer.mkdir(parents=True, exist_ok=True)
    pearson_r.round(4).to_csv(per_designer / "L6_pearson_r.csv")
    spearman_r.round(4).to_csv(per_designer / "L6_spearman_r.csv")
    pd.DataFrame(pearson_p, index=items, columns=items).to_csv(per_designer / "L6_pearson_p.csv")

    # ----------- markdown ----------
    s: list[str] = []
    s.append(f"# L6 — 30×30 Inter-Item Correlation (data_v3, {pd.Timestamp.now().date()})\n")
    s.append("E1 of 30-dim relationship analysis. Zero-order (Pearson + Spearman).")
    s.append("互补: L2 EFA (latent), L9 network (partial correlation / conditional).\n")

    s.append(f"- N = {n}, items = {p}")
    s.append(f"- Pearson r off-diag: median = {np.median(pearson_off):.3f}, "
             f"mean |r| = {np.mean(np.abs(pearson_off)):.3f}, "
             f"max |r| = {np.max(np.abs(pearson_off)):.3f}")
    s.append(f"- Spearman ρ off-diag: median = {np.median(spearman_off):.3f}, "
             f"mean |ρ| = {np.mean(np.abs(spearman_off)):.3f}")
    s.append(f"- corr matrix det = {det_val:.2e}  ({'PASS (non-singular)' if abs(det_val) > 1e-6 else 'WARN near-singular'})")
    s.append(f"- BH-FDR α=.05 threshold: Pearson p ≤ {fdr_thresh_p:.4f}, Spearman p ≤ {fdr_thresh_s:.4f}")
    n_sig_p = int((pearson_p_off <= fdr_thresh_p).sum())
    n_sig_s = int((spearman_p_off <= fdr_thresh_s).sum())
    total_pairs = len(pearson_off)
    s.append(f"- significant pairs (BH-FDR .05): Pearson {n_sig_p}/{total_pairs}, "
             f"Spearman {n_sig_s}/{total_pairs}")

    s.append("\n## Item index legend\n")
    legend = pd.DataFrame({"idx": range(1, p + 1), "item": items})
    s.append(legend.to_markdown(index=False))

    s.append("\n## Heatmap (Pearson + Spearman lower-tri)\n")
    s.append(f"![inter-item heatmap](figures/{heatmap_path.name})")

    s.append(f"\n## High-correlation pairs (|r| ≥ {pair_thresh}, sorted by |Pearson r|)\n")
    if pairs:
        pair_df = pd.DataFrame(pairs)
        s.append(pair_df.to_markdown(index=False))
    else:
        s.append(f"_无 pair 跨过 |r| ≥ {pair_thresh}._ (见下方 top-20 fallback)")

    # Top-20 absolute Pearson r (regardless of threshold) — 给负面发现以实证替代
    top20_rows: list[dict] = []
    flat_idx = np.argsort(-np.abs(pearson_r.values[iu]))[:20]
    for k_idx in flat_idx:
        i_, j_ = iu[0][k_idx], iu[1][k_idx]
        r_p = pearson_r.values[i_, j_]
        r_s = spearman_r.values[i_, j_]
        top20_rows.append(
            {
                "i": int(i_ + 1),
                "j": int(j_ + 1),
                "item_i": items[i_],
                "item_j": items[j_],
                "pearson_r": round(float(r_p), 3),
                "pearson_p": round(float(pearson_p_off[k_idx]), 4),
                "spearman_r": round(float(r_s), 3),
                "sign_agree": bool((r_p * r_s) > 0),
            }
        )
    s.append("\n## Top-20 pairs by |Pearson r| (fallback if 上表为空)\n")
    s.append(pd.DataFrame(top20_rows).to_markdown(index=False))

    s.append("\n## Item connectivity strength (Σ |Pearson r|, exclude self)\n")
    s.append(strength_df.to_markdown(index=False))
    s.append(f"\n![strength](figures/{strength_path.name})")

    s.append("\n## Verdict\n")
    if pairs:
        n_strong = sum(1 for d in pairs if abs(d["pearson_r"]) >= 0.5)
        n_signagree = sum(1 for d in pairs if d["sign_agree"])
        s.append(f"- {n_strong} pair |r|≥0.5, 其中 {n_signagree} pair Pearson/Spearman 同号 (一致性 check)")
    else:
        s.append("- 无 pair 跨过 |r|≥0.5 → zero-order 关联整体弱, 跟 L2 PA K=4 cumulative variance 23% 一致")
    top_hub = strength_df.iloc[0]["item"]
    bot_hub = strength_df.iloc[-1]["item"]
    s.append(f"- 最强 hub: **{top_hub}** (Σ|r|={strength_df.iloc[0]['strength_sum_abs_r']}), "
             f"最弱: **{bot_hub}** (Σ|r|={strength_df.iloc[-1]['strength_sum_abs_r']})")
    s.append("- 下游: L9 network 跑 partial correlation 看哪些 zero-order 关联是 spurious (third-variable)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(s), encoding="utf-8")

    findings = {
        "n": n,
        "p": p,
        "pearson": {
            "mean_abs_off": float(np.mean(np.abs(pearson_off))),
            "median_off": float(np.median(pearson_off)),
            "max_abs_off": float(np.max(np.abs(pearson_off))),
            "fdr_thresh": fdr_thresh_p,
            "n_sig_pairs": n_sig_p,
        },
        "spearman": {
            "mean_abs_off": float(np.mean(np.abs(spearman_off))),
            "median_off": float(np.median(spearman_off)),
            "fdr_thresh": fdr_thresh_s,
            "n_sig_pairs": n_sig_s,
        },
        "det_corr_matrix": det_val,
        "pair_threshold": pair_thresh,
        "n_high_pairs": len(pairs),
        "high_pairs": pairs,
        "top20_pairs": top20_rows,
        "strength_top5": strength_df.head(5).to_dict(orient="records"),
        "strength_bottom5": strength_df.tail(5).to_dict(orient="records"),
        "figures": [f"figures/{heatmap_path.name}", f"figures/{strength_path.name}"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
