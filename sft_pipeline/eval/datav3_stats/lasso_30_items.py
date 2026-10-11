"""L4c: T_CAPS quad — option_trait_mapping (Step 2a of quad pipeline).

直接对应 docs/notes/CAPS四元组与簇内RAG_20260523.md §2.2 Step 2a:
  - 30 维 trait → 每个 (Q, option) top-3 高分 item
  - = quad schema 的 **s_primary** (群体级特质画像, **反向推导**: 选了这个 option
    的所有人的共同 top trait, 不是个人 trait)
  - 反向推导建立 "trait → behavior" 因果链 (而不是从个人 trait 归因)

输出两份:
  - reports/datav3/L4c_lasso_30items.md: 人读版, 22 段 English Statistical
    Observation (可喂 quad pipeline 的 "CPO Compiler Prompt")
  - reports/datav3/per_designer/option_trait_mapping.json: quad pipeline Step 2a
    标准 schema, 直接 plug-in 后续 quad 生成 (blocker A 解后)

跟 L4 (factor → option 关联) 互补:
- L4 factor 级 (4-d EFA), 提供 quad s_secondary 证据
- L4c item 级 (30-d), 提供 quad s_primary 输入
"""
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


def _perm_p_value_for_item(
    scale_col: np.ndarray,
    chose: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> float:
    """Marginal permutation test: 单个 item 对 chose 的 mean-diff 显著性.

    注意: 这是 marginal p-value, 不是 'given other items 条件下' 的 p. LASSO 系数才是 joint.
    """
    s = np.asarray(scale_col, dtype=float)
    c = np.asarray(chose, dtype=bool)
    if c.sum() < 2 or (~c).sum() < 2:
        return float("nan")
    obs = s[c].mean() - s[~c].mean()
    if obs == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(c)
        d = s[perm].mean() - s[~perm].mean()
        if abs(d) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def _english_observation(
    qid: str,
    option: str,
    top_items: list[tuple],
    n_chose: int,
    n_total: int,
    polarity_map: dict | None = None,
    option_pkvc: dict | None = None,
) -> str:
    """CPO Compiler Prompt 格式, enriched 版:
    - higher/lower 标注 stance label (from polarity_map)
    - option 标注 PKVC main factor + chain (from option_pkvc)
    """
    polarity_map = polarity_map or {}
    option_pkvc = option_pkvc or {}
    if not top_items:
        return f"For {qid}.{option}: no item passed selection threshold."
    parts = []
    for item, coef, p in top_items:
        direction = "higher" if coef > 0 else "lower"
        pol = polarity_map.get(item, {})
        stance = pol.get("plus_label", "?") if coef > 0 else pol.get("minus_label", "?")
        p_str = f", marginal p={p:.3f}" if not np.isnan(p) else ""
        parts.append(
            f"{direction} scores on \"{item}\" (= {stance} stance; β={coef:+.3f}{p_str})"
        )
    parts_str = " and ".join(parts)

    opt_meta = option_pkvc.get(qid, {}).get(option, {})
    opt_factor = opt_meta.get("main_factor")
    opt_chain = opt_meta.get("chain")
    if opt_factor:
        chain_str = f", chain={opt_chain}" if opt_chain and opt_chain.lower() != opt_factor.lower() else ""
        opt_descr = f"option {option} (activates PKVC factor: {opt_factor}{chain_str})"
    else:
        opt_descr = f"option {option}"

    return (
        f"In question {qid}, LASSO multinomial logistic regression on 30 raw trait items "
        f"shows that designers with {parts_str} have a higher probability of selecting "
        f"{opt_descr} (n_chose={n_chose}/{n_total})."
    )


def run(
    output_path: Path | None = None,
    C: float = 0.5,
    top_k: int = 3,
    min_abs_coef: float = 0.05,
    n_perm: int = 500,
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L4c_lasso_30items.md"
    scale = loaders.load_scale_wide().dropna()  # 163 × 30
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    q_single = q_single[q_single["user_id"].isin(scale.index)]

    polarity_map = loaders.load_item_polarity()
    option_pkvc = loaders.load_q1q6_option_pkvc()
    q_context_c = loaders.load_q3_q6_context_c()

    sections: list[str] = []
    sections.append(f"# L4c — 30 raw items → Q-option LASSO (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append("回应 quad pipeline Step 2a 的精确需求: 30 维原始量表 (S1..S30) → 每个 (Q, option) top-2-3 predictor item.\n")
    sections.append(f"- predictors: 30 raw 量表题 (S{{1..30}}), source = scale_responses (per-designer wide)")
    sections.append(f"- model: multinomial L1 logistic (C={C}), per option top-{top_k} |coef|≥{min_abs_coef}")
    sections.append(f"- per-item p-value: marginal permutation ({n_perm}×), 非 conditional")
    sections.append("- LASSO 系数已 standardize 过 (StandardScaler on items), 系数大小可跨 item 比")
    sections.append(f"- 元数据: item 极性 (from docs/notes/PKVC_30节点本体.md, {len(polarity_map)} items 已 load)")
    sections.append(f"- 元数据: Q-option PKVC 因子 (from docs/notes/Q1-Q6每个选项的元数据.md, {sum(len(v) for v in option_pkvc.values())} options 已 load)\n")

    findings: dict = {"by_q": {}, "total_observations": 0}
    all_observations: list[str] = []

    for qid in sorted(q_single["question_id"].unique()):
        sub = q_single[q_single["question_id"] == qid].copy()
        y_raw = sub["option"].values
        vc = pd.Series(y_raw).value_counts()
        keep_opts = vc[vc >= 5].index.tolist()
        mask = np.isin(y_raw, keep_opts)
        if mask.sum() < 30 or len(keep_opts) < 2:
            sections.append(f"\n## {qid}: skipped (n={int(mask.sum())}, kept_opts={keep_opts})\n")
            continue
        sub_kept = sub[mask].reset_index(drop=True)
        X_raw = scale.loc[sub_kept["user_id"]].values
        X = StandardScaler().fit_transform(X_raw)
        y = sub_kept["option"].values
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

        coef = pd.DataFrame(clf.coef_, index=le.classes_, columns=scale.columns)
        sections.append(f"\n## {qid} (n={int(mask.sum())}, options={list(le.classes_)})\n")

        q_findings: dict = {}
        for opt in le.classes_:
            opt_coefs = coef.loc[opt]
            top = opt_coefs.abs().sort_values(ascending=False).head(top_k * 2).index
            selected = [(item, float(opt_coefs[item])) for item in top if abs(opt_coefs[item]) >= min_abs_coef][:top_k]

            if not selected:
                sections.append(f"\n### {qid}.{opt}: no item passed |coef|≥{min_abs_coef}\n")
                continue

            chose_mask = (y_enc == le.transform([opt])[0])
            n_chose = int(chose_mask.sum())

            top_with_p = []
            for item, c in selected:
                col_idx = list(scale.columns).index(item)
                p = _perm_p_value_for_item(X_raw[:, col_idx], chose_mask, n_perm=n_perm, seed=seed)
                top_with_p.append((item, c, p))

            sections.append(f"\n### {qid}.{opt} (n_chose={n_chose}/{int(mask.sum())})\n")
            sections.append("**Top items:**\n")
            sections.append("| item | β (standardized LASSO coef) | direction | marginal perm-p |")
            sections.append("|---|---|---|---|")
            for item, c, p in top_with_p:
                direction = "higher" if c > 0 else "lower"
                p_cell = f"{p:.3f}" if not np.isnan(p) else "nan"
                sections.append(f"| {item} | {c:+.3f} | {direction} | {p_cell} |")
            obs = _english_observation(
                qid, opt, top_with_p, n_chose, int(mask.sum()),
                polarity_map=polarity_map,
                option_pkvc=option_pkvc,
            )
            sections.append("\n**Statistical Observation (English, for paper narrative / CPO Compiler input):**\n")
            sections.append(f"> {obs}")
            all_observations.append(obs)

            q_findings[str(opt)] = [
                {"item": str(item), "coef": round(c, 3), "marginal_perm_p": round(p, 3) if not np.isnan(p) else None}
                for item, c, p in top_with_p
            ]

        findings["by_q"][qid] = q_findings

    # 汇总: 一段一段 paragraph, 可直接复制粘贴喂"本体编译器" (CPO Compiler Prompt)
    sections.append("\n---\n\n## 全部 Statistical Observation 段落 (for direct CPO Compiler input)\n")
    for i, obs in enumerate(all_observations, 1):
        sections.append(f"{i}. {obs}\n")
    findings["total_observations"] = len(all_observations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时输出 quad pipeline Step 2a 标准 schema
    option_trait_path = _REPORTS / "per_designer" / "option_trait_mapping.json"
    option_trait_path.parent.mkdir(parents=True, exist_ok=True)
    option_trait_mapping: dict = {}
    for qid, opts in findings["by_q"].items():
        for opt, items in opts.items():
            key = f"{qid}_{opt}"
            # 计算这个 (Q, option) 的 n_samples (跟 polarity 加 stance label)
            sub = q_single[(q_single["question_id"] == qid) & (q_single["option"] == opt)]
            n_samples = int(len(sub))
            s_primary = [it["item"] for it in items]
            s_primary_with_stance = []
            for it in items:
                pol = polarity_map.get(it["item"], {})
                stance = pol.get("plus_label") if it["coef"] > 0 else pol.get("minus_label")
                s_primary_with_stance.append({
                    "item": it["item"],
                    "stance": stance,
                    "direction": "+" if it["coef"] > 0 else "-",
                    "coef": it["coef"],
                    "marginal_perm_p": it.get("marginal_perm_p"),
                })
            opt_meta = option_pkvc.get(qid, {}).get(opt, {})
            c_meta = q_context_c.get(qid, {})
            option_trait_mapping[key] = {
                "qid": qid,
                "option": opt,
                "n_samples": n_samples,
                "C_raw": c_meta.get("C_raw"),
                "C_conflict_axes": c_meta.get("conflict_axes"),
                "C_formal_proposed": c_meta.get("C_formal_proposed"),
                "s_primary": s_primary,
                "s_primary_with_stance": s_primary_with_stance,
                "o_primary_hint": opt_meta.get("main_factor"),
                "o_subtype_hint": opt_meta.get("chain"),
            }
    option_trait_path.write_text(
        json.dumps(option_trait_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    findings["option_trait_mapping_file"] = str(option_trait_path)
    findings["n_option_trait_entries"] = len(option_trait_mapping)
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
