"""L4: T_CAPS quad — s_secondary 统计关联表 (factor → option) via permutation test.

跟 quad pipeline (docs/notes/CAPS四元组与簇内RAG_20260523.md) 的对应:
  - 输出 quad_s_secondary_associations.jsonl: 给 quad 的 s_secondary_factor +
    direction 字段提供统计证据 (在 individual-level 上, 哪个 4-d factor 对哪个
    (qid, option) 显著)
  - 互补文件: option_trait_mapping.json (Step 2a, 由 L4c lasso_30_items 输出)
    给 quad 的 s_primary (群体级 top-3 30-d trait) 提供数据
  - 跟 quad pipeline Step 2b (p_surface + p_modulation + o LLM 抽取) 互补,
    后者依赖 blocker A extractor 解锁

Tonight 用 Q1/Q3-Q6/Q8 raw answer_value (A/B/C/D, canonicalized) 作为 target,
作为 quad o_primary 的 proxy. blocker A 解后改为 PKVC state target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"

# Cohen's d cutoffs (L5 会 calibrate, 这里是 default)
_STRENGTH_CUTOFFS = {"bypass": 0.1, "weak": 0.5}


def compute_shift_and_pvalue(
    factor_scores: np.ndarray,
    chose_option: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """两组 factor score mean 差 (delta) + permutation p-value.

    delta = mean(scores[chose]) - mean(scores[~chose])
    """
    scores = np.asarray(factor_scores, dtype=float)
    chosen = np.asarray(chose_option, dtype=bool)
    if chosen.sum() < 2 or (~chosen).sum() < 2:
        return float("nan"), float("nan")

    observed_delta = scores[chosen].mean() - scores[~chosen].mean()

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(chosen)
        d = scores[perm].mean() - scores[~perm].mean()
        if abs(d) >= abs(observed_delta):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(observed_delta), float(p)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d, pooled sd."""
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    na, nb = len(group_a), len(group_b)
    va, vb = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled)


def classify_strength(d: float) -> str:
    """|d| < 0.1 → bypass, 0.1-0.5 → Weak, >0.5 → Strong, with sign."""
    if np.isnan(d):
        return "nan"
    if abs(d) < _STRENGTH_CUTOFFS["bypass"]:
        return "bypass"
    sign = "+" if d > 0 else "-"
    if abs(d) < _STRENGTH_CUTOFFS["weak"]:
        return f"Weak{sign}"
    return f"Strong{sign}"


def run(
    output_path: Path | None = None,
    factor_scores_path: Path | None = None,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L4_mapping.md"
    factor_scores_path = factor_scores_path or loaders.get_default_factor_scores_path()
    if not factor_scores_path.exists():
        raise FileNotFoundError(f"{factor_scores_path}. Run L2 EFA first.")

    scores = pd.read_csv(factor_scores_path, index_col=0)
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    q_single = q_single[q_single["user_id"].isin(scores.index)]

    findings: dict = {
        "n_factors": int(scores.shape[1]),
        "factor_scores_path": str(factor_scores_path),
    }
    associations = []
    rows = []

    for factor_col in scores.columns:
        for qid in sorted(q_single["question_id"].unique()):
            sub = q_single[q_single["question_id"] == qid]
            user_score = scores.loc[sub["user_id"], factor_col].values
            for opt in sorted(sub["option"].dropna().unique()):
                chose = (sub["option"] == opt).values
                n_chose = int(chose.sum())
                if n_chose < 3 or n_chose > len(chose) - 3:
                    continue
                delta, p = compute_shift_and_pvalue(user_score, chose, n_perm=n_perm, seed=seed)
                d = cohens_d(user_score[chose], user_score[~chose])
                strength = classify_strength(d)
                assoc = {
                    "factor": factor_col,
                    "qid": qid,
                    "option": opt,
                    "n_chose": n_chose,
                    "n_total": int(len(chose)),
                    "delta_mean": round(delta, 3),
                    "cohens_d": round(d, 3),
                    "p_perm": round(p, 4),
                    "strength": strength,
                    "significant": bool(p < 0.05),
                }
                associations.append(assoc)
                rows.append(assoc)

    df = pd.DataFrame(rows)

    sections: list[str] = []
    sections.append(f"# L4 — T_CAPS quad s_secondary 统计关联 (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append("跟 quad pipeline 对应: 本表给 quad schema 的 s_secondary_factor + direction")
    sections.append("字段提供 individual-level 统计证据 (factor 跟 option 的显著关联). 互补于 L4c")
    sections.append("option_trait_mapping (s_primary 群体级特质). 见 docs/notes/CAPS四元组与簇内RAG_20260523.md.\n")
    sections.append(f"- factor scores: K = {scores.shape[1]}, N designer = {len(scores)}")
    sections.append(f"- (factor, qid, option) 关联探查总数: {len(df)} (sample 量过低 cell 已 filter)")
    sections.append(f"- permutation 次数: {n_perm}")
    sections.append(f"- **significant (p<.05)**: {int(df['significant'].sum())} / {len(df)}\n")

    sections.append("## strength 分布 (4 档 modulation + bypass calibration 候选)\n")
    sections.append(df["strength"].value_counts().to_frame("count").to_markdown())

    sections.append("\n## top 20 显著关联 (按 |Cohen's d|)\n")
    sig_df = df[df["significant"]]
    if len(sig_df):
        top = sig_df.assign(abs_d=lambda x: x["cohens_d"].abs()).nlargest(20, "abs_d")
        sections.append(top.drop(columns=["abs_d"]).to_markdown(index=False))
    else:
        sections.append("(无显著关联)")

    sections.append("\n## per-factor 显著关联数\n")
    sections.append(df.groupby("factor")["significant"].sum().to_frame("n_sig").to_markdown())

    sections.append("\n## per-question 显著关联数\n")
    sections.append(df.groupby("qid")["significant"].sum().to_frame("n_sig").to_markdown())

    sections.append("\n## Verdict\n")
    sig_rate = df["significant"].mean() if len(df) else 0.0
    sections.append(f"- 显著率 = {sig_rate:.1%}  ({'rich signal' if sig_rate > 0.2 else ('moderate' if sig_rate > 0.1 else 'sparse')})")
    sections.append("- 推荐: 跑 L4b LASSO 跟本表做 intersect, 取共同显著 = robust quad s_secondary 候选")
    sections.append("- 下游: 这些关联当 quad schema s_secondary 的 prior, blocker A 解后再加 p/o 抽取")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")

    assoc_path = _REPORTS / "per_designer" / "quad_s_secondary_associations.jsonl"
    assoc_path.parent.mkdir(parents=True, exist_ok=True)
    with assoc_path.open("w", encoding="utf-8") as f:
        for a in associations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    findings["n_associations"] = int(len(associations))
    findings["n_significant"] = int(df["significant"].sum()) if len(df) else 0
    findings["association_file"] = str(assoc_path)
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
