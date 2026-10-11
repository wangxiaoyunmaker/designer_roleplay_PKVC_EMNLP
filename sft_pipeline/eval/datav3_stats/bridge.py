"""L5: bridge statistics — bootstrap stability, partial-out, inhibit/excite, bypass."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def _cohens_d_from_groups(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def bootstrap_sign_stability(
    scores: np.ndarray,
    chose: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> float:
    """Bootstrap resample N×; fraction with same sign as observed delta."""
    scores = np.asarray(scores, dtype=float)
    chose = np.asarray(chose, dtype=bool)
    n = len(scores)
    if n < 10:
        return float("nan")
    observed = scores[chose].mean() - scores[~chose].mean()
    if observed == 0:
        return float("nan")
    target_sign = np.sign(observed)
    rng = np.random.default_rng(seed)
    same_sign = 0
    valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_b = scores[idx]
        c_b = chose[idx]
        if c_b.sum() < 2 or (~c_b).sum() < 2:
            continue
        delta = s_b[c_b].mean() - s_b[~c_b].mean()
        valid += 1
        if np.sign(delta) == target_sign:
            same_sign += 1
    if valid == 0:
        return float("nan")
    return float(same_sign / valid)


def partial_out_cohens_d(
    factor: np.ndarray,
    chose: np.ndarray,
    covariates: np.ndarray,
) -> float:
    """Residualize factor against covariates, then Cohen's d on residuals."""
    factor = np.asarray(factor, dtype=float)
    chose = np.asarray(chose, dtype=bool)
    cov = np.asarray(covariates, dtype=float)
    X = np.column_stack([np.ones(len(factor)), cov])
    coef, *_ = np.linalg.lstsq(X, factor, rcond=None)
    resid = factor - X @ coef
    return _cohens_d_from_groups(resid[chose], resid[~chose])


def inhibit_excite_asymmetry(
    factor: np.ndarray,
    chose: np.ndarray,
    quartile: float = 0.25,
) -> dict:
    """Top vs bottom quartile factor → P(chose)."""
    factor = np.asarray(factor, dtype=float)
    chose = np.asarray(chose, dtype=bool)
    q_lo = np.quantile(factor, quartile)
    q_hi = np.quantile(factor, 1 - quartile)
    top_mask = factor >= q_hi
    bot_mask = factor <= q_lo
    if top_mask.sum() == 0 or bot_mask.sum() == 0:
        return {"top_rate": float("nan"), "bot_rate": float("nan"), "asym": float("nan")}
    top_rate = chose[top_mask].mean()
    bot_rate = chose[bot_mask].mean()
    return {"top_rate": float(top_rate), "bot_rate": float(bot_rate), "asym": float(top_rate - bot_rate)}


def _build_covariate_matrix(users: pd.DataFrame, target_ids: pd.Index) -> tuple[np.ndarray, list[str]]:
    """years_exp (numeric) + identity (one-hot) + education_stage_current (one-hot)."""
    sub = users.set_index("wechat_id").reindex(target_ids)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if "years_exp" in sub.columns:
        ye = pd.to_numeric(sub["years_exp"], errors="coerce").fillna(0).values
        parts.append(ye.reshape(-1, 1))
        names.append("years_exp")
    for col in ["identity", "education_stage_current"]:
        if col in sub.columns:
            dummies = pd.get_dummies(sub[col].fillna("unknown"), prefix=col, drop_first=True)
            parts.append(dummies.values.astype(float))
            names.extend(dummies.columns.tolist())
    if not parts:
        return np.zeros((len(target_ids), 0)), []
    return np.column_stack(parts), names


def run(
    output_path: Path | None = None,
    factor_scores_path: Path | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L5_bridge.md"
    factor_scores_path = factor_scores_path or loaders.get_default_factor_scores_path()
    scores = pd.read_csv(factor_scores_path, index_col=0)
    users = loaders.load_users()
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    q_single = q_single[q_single["user_id"].isin(scores.index)]

    assoc_path = _REPORTS / "per_designer" / "quad_s_secondary_associations.jsonl"
    if not assoc_path.exists():
        raise FileNotFoundError(f"{assoc_path}. Run L4 mapping first.")
    associations = [json.loads(line) for line in assoc_path.read_text().splitlines() if line.strip()]
    sig_assocs = [a for a in associations if a["significant"]]

    findings: dict = {"n_sig_associations": len(sig_assocs)}
    sections: list[str] = []
    sections.append(f"# L5 — Bridge stats for quad s_secondary associations (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- 输入: {len(sig_assocs)} 个 L4 显著 quad s_secondary association\n")
    sections.append("跟 quad pipeline 对应: 本表验证 quad s_secondary 候选的 robustness (bootstrap")
    sections.append("sign stability + demographic partial-out + inhibit/excite asymmetry + bypass")
    sections.append("threshold data-driven calibration). 通过的 association → quad s_secondary 主集.\n")

    rows = []
    for t in sig_assocs:
        sub = q_single[q_single["question_id"] == t["qid"]]
        factor_vec = scores.loc[sub["user_id"], t["factor"]].values
        chose_vec = (sub["option"] == t["option"]).values

        stab = bootstrap_sign_stability(factor_vec, chose_vec, n_boot=n_boot, seed=seed)
        cov_mat, cov_names = _build_covariate_matrix(users, sub["user_id"])
        partial_d = partial_out_cohens_d(factor_vec, chose_vec, cov_mat) if cov_mat.shape[1] > 0 else t["cohens_d"]
        asym = inhibit_excite_asymmetry(factor_vec, chose_vec)
        d_dropped = (1 - abs(partial_d) / max(abs(t["cohens_d"]), 1e-6)) * 100
        rows.append({
            "factor": t["factor"],
            "qid": t["qid"],
            "option": t["option"],
            "raw_d": t["cohens_d"],
            "partial_d": round(partial_d, 3),
            "d_dropped_pct": round(d_dropped, 1),
            "boot_stability": round(stab, 3),
            "top_q_rate": round(asym["top_rate"], 3),
            "bot_q_rate": round(asym["bot_rate"], 3),
            "asym": round(asym["asym"], 3),
        })

    df = pd.DataFrame(rows)
    sections.append("## 5.1 / 5.2 / 5.3 — per-triple bridge stats\n")
    sections.append(df.to_markdown(index=False))

    if not df.empty:
        n_robust = int(((df["boot_stability"] > 0.9) & (df["d_dropped_pct"] < 30)).sum())
        n_killed = int((df["d_dropped_pct"] > 50).sum())
        findings["robust_triples_after_L5"] = n_robust
        findings["covariate_attack_killed"] = n_killed

        sections.append("\n## Robustness summary\n")
        sections.append(f"- **robust 通过** (boot>0.9 且 covariate drop<30%): **{n_robust}** / {len(df)}")
        sections.append(f"- **被 covariate 几乎杀掉** (d_dropped >50%): {n_killed}")

    # 5.4 bypass calibration: |d| 分布
    all_d = np.array([abs(t["cohens_d"]) for t in associations if not np.isnan(t["cohens_d"])])
    if len(all_d) > 0:
        q25 = float(np.percentile(all_d, 25))
        q50 = float(np.percentile(all_d, 50))
        q75 = float(np.percentile(all_d, 75))
        q90 = float(np.percentile(all_d, 90))
        sections.append("\n## 5.4 Bypass / strength threshold calibration\n")
        sections.append(f"- |d| 分布 (across all {len(all_d)} 探查 triples): 25%={q25:.3f}, 50%={q50:.3f}, 75%={q75:.3f}, 90%={q90:.3f}")
        sections.append(f"- 当前 mapping.py 阈值: bypass<0.10, Weak<0.50, Strong>=0.50")
        sections.append(f"- **建议 (data-driven)**: bypass<{q25:.2f}, Weak<{q75:.2f}, Strong>={q75:.2f}")
        findings["d_quantiles"] = {"q25": q25, "q50": q50, "q75": q75, "q90": q90}

    sections.append("\n## Verdict\n")
    sections.append("- L4 sig 且 L5 robust 通过的 association = T_CAPS quad s_secondary 主集候选 (paper main)")
    sections.append("- 被 covariate 杀掉的 association 写 paper §4 caveat")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
