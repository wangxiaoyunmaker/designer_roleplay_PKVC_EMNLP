"""L4b: multinomial LASSO logistic regression, cross-check L4 mapping triples."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def run(
    output_path: Path | None = None,
    factor_scores_path: Path | None = None,
    C: float = 0.5,
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L4b_lasso_check.md"
    factor_scores_path = factor_scores_path or loaders.get_default_factor_scores_path()
    scores = pd.read_csv(factor_scores_path, index_col=0)
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    q_single = q_single[q_single["user_id"].isin(scores.index)]

    findings: dict = {"factor_scores_path": str(factor_scores_path)}
    sections: list[str] = []
    sections.append(f"# L4b LASSO cross-check (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- C (1/λ) = {C}, multinomial L1 logistic")
    sections.append(f"- factor predictors: {list(scores.columns)}\n")

    all_coefs: dict[str, dict] = {}
    for qid in sorted(q_single["question_id"].unique()):
        sub = q_single[q_single["question_id"] == qid].copy()
        X = scores.loc[sub["user_id"]].values
        y = sub["option"].values
        vc = pd.Series(y).value_counts()
        keep_opts = vc[vc >= 5].index.tolist()
        mask = np.isin(y, keep_opts)
        if mask.sum() < 30 or len(keep_opts) < 2:
            sections.append(f"## {qid}: skipped (n={int(mask.sum())}, kept opts={keep_opts})")
            continue
        X = StandardScaler().fit_transform(X[mask])
        y = y[mask]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", FutureWarning)
            clf = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=C,
                max_iter=5000,
                random_state=seed,
            ).fit(X, y_enc)
        coef = pd.DataFrame(clf.coef_, index=le.classes_, columns=scores.columns)
        all_coefs[qid] = {str(opt): {fc: round(float(coef.loc[opt, fc]), 3) for fc in scores.columns} for opt in coef.index}

        sections.append(f"\n## {qid} (n={int(mask.sum())}, options={list(le.classes_)})\n")
        sections.append(coef.round(3).to_markdown())

        sections.append("\n**top-2 |coef| per option:**")
        for opt in le.classes_:
            top = coef.loc[opt].abs().sort_values(ascending=False).head(2).index.tolist()
            sections.append(f"- {opt}: {top}")

    findings["coefs_by_q"] = all_coefs

    # intersect with L4 quad s_secondary associations
    assoc_path = _REPORTS / "per_designer" / "quad_s_secondary_associations.jsonl"
    intersect_count = 0
    if assoc_path.exists():
        associations = [json.loads(line) for line in assoc_path.read_text().splitlines() if line.strip()]
        sig_assocs = [a for a in associations if a["significant"]]
        agree = []
        for a in sig_assocs:
            qid, opt, factor = a["qid"], a["option"], a["factor"]
            if qid in all_coefs and opt in all_coefs[qid]:
                lasso_coef = all_coefs[qid][opt].get(factor, 0.0)
                if abs(lasso_coef) > 0.05 and np.sign(lasso_coef) == np.sign(a["delta_mean"]):
                    agree.append({**a, "lasso_coef": round(lasso_coef, 3)})
        intersect_count = len(agree)
        sections.append("\n## Intersect with L4 quad s_secondary associations (sig 且 LASSO 同号 |coef|>0.05)\n")
        sections.append(f"- agree count: **{intersect_count}** / {len(sig_assocs)} sig associations")
        if agree:
            sections.append("\ntop 10 agreed associations:")
            agree_df = pd.DataFrame(agree)
            sections.append(agree_df.head(10).to_markdown(index=False))
        findings["intersect_with_L4"] = intersect_count
        findings["L4_sig_total"] = len(sig_assocs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
