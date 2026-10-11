"""L7 leakage / completeness audit + L1b expansion subset selection bias."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def run_l7(output_path: Path | None = None) -> dict:
    output_path = output_path or _REPORTS / "L7_audit.md"
    users = loaders.load_users()
    scale = loaders.load_scale_wide()
    task = loaders.load_task_long(expanded=True)

    findings: dict = {}
    sections: list[str] = []
    sections.append(f"# L7 Audit — leakage / completeness (data_v3, {pd.Timestamp.now().date()})\n")

    # wechat_id duplicates
    dup = int(users["wechat_id"].duplicated().sum())
    findings["wechat_dup"] = dup
    sections.append(f"- **wechat_id 重复**: {dup}  ({'PASS' if dup == 0 else 'FAIL'})")

    # users in users.csv vs scale vs task
    u_set = set(users["wechat_id"])
    s_set = set(scale.index)
    t_set = set(task["user_id"])
    findings["users_missing_scale"] = len(u_set - s_set)
    findings["users_missing_task"] = len(u_set - t_set)
    findings["scale_orphans"] = len(s_set - u_set)
    findings["task_orphans"] = len(t_set - u_set)

    sections.append(f"- **有 user 没 scale**: {findings['users_missing_scale']}")
    sections.append(f"- **有 user 没 task**: {findings['users_missing_task']}")
    sections.append(f"- **scale 孤儿 (无 user)**: {findings['scale_orphans']}")
    sections.append(f"- **task 孤儿 (无 user)**: {findings['task_orphans']}")

    # per-user completeness
    scale_ok = scale.notna().sum(axis=1)
    full_scale = int((scale_ok == 30).sum())
    findings["users_full_scale_30"] = full_scale
    sections.append(f"- **完整 30 量表的 user**: {full_scale} / {len(users)}")

    q_per_user = task[task["question_id"].isin([f"Q{i}" for i in range(1, 7)])].groupby("user_id").size()
    full_q16 = int((q_per_user >= 6).sum())
    findings["users_full_q16"] = full_q16
    sections.append(f"- **Q1-Q6 全答的 user**: {full_q16} / {len(users)}")

    q_per_user_full = task.groupby("user_id").size()
    full_q18 = int((q_per_user_full >= 8).sum())
    findings["users_full_q18"] = full_q18
    sections.append(f"- **Q1-Q8 全答的 user**: {full_q18} / {len(users)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


def run_l1b(output_path: Path | None = None) -> dict:
    output_path = output_path or _REPORTS / "L1b_expansion_audit.md"
    users = loaders.load_users()
    task = loaders.load_task_long(expanded=False)  # RAW for selection bias 判断
    expansion = loaders.load_expansion()

    findings: dict = {"n_expanded": int(len(expansion))}
    sections: list[str] = []
    sections.append(f"# L1b Expansion Subset Audit (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append(f"- expansion ok rows: **{len(expansion)}**")
    sections.append(f"- 理论上限 163 × 6 = 978 cells, **coverage = {len(expansion) / (len(users) * 6) * 100:.1f}%**\n")

    if expansion.empty:
        sections.append("Expansion 空, 跳过 bias 分析.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(sections), encoding="utf-8")
        return findings

    # selection bias: 谁的 cell 被扩了 vs 没被扩
    if {"user_id", "question_id"}.issubset(expansion.columns):
        expanded_keys = set(zip(expansion["user_id"], expansion["question_id"]))
        q16 = task[task["question_id"].isin([f"Q{i}" for i in range(1, 7)])].copy()
        q16["was_expanded"] = q16.apply(lambda r: (r["user_id"], r["question_id"]) in expanded_keys, axis=1)

        # 1) 被扩 vs 没被扩 的 **RAW** supplement 长度 (用 task_responses.csv 非 expanded)
        if "text_supplement" in q16.columns:
            q16["supp_len"] = q16["text_supplement"].fillna("").astype(str).str.len()
            len_by_expanded = q16.groupby("was_expanded")["supp_len"].agg(["mean", "median", "max", "min"]).round(1)
            sections.append("## 被扩 vs 没被扩 的 raw supplement 长度 (selection bias)\n")
            sections.append("(预期: 被扩组的 raw 长度应明显短于未被扩组, 因为扩写挑短的)\n")
            sections.append(len_by_expanded.to_markdown())
            findings["len_expanded_vs_not"] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in len_by_expanded.to_dict("index").items()}
            # selection bias verdict
            try:
                expanded_med = float(len_by_expanded.loc[True, "median"])
                not_expanded_med = float(len_by_expanded.loc[False, "median"])
                verdict = "PASS (扩写选了短的, 符合预期)" if expanded_med < not_expanded_med else "WARN (扩写选了长的, 反预期, 查 pipeline)"
                sections.append(f"\n**Verdict**: {verdict} (被扩 median={expanded_med}, 未扩 median={not_expanded_med})")
                findings["selection_bias_verdict"] = verdict
            except (KeyError, ValueError):
                pass

        # 2) per-Q 扩写覆盖率
        cov_by_q = q16.groupby("question_id")["was_expanded"].agg(["sum", "count"])
        cov_by_q["coverage_pct"] = (cov_by_q["sum"] / cov_by_q["count"] * 100).round(1)
        sections.append("\n## per-Q 扩写覆盖率\n")
        sections.append(cov_by_q.to_markdown())
        findings["cov_by_q"] = cov_by_q.astype(object).to_dict()

    # PKVC factor 分布 (从 expansion result.inferred_factor)
    if "inferred_factor" in expansion.columns:
        pkvc = expansion["inferred_factor"].value_counts()
        sections.append("\n## inferred_factor 分布 (P/K/V/C)\n")
        for k in ["P", "K", "V", "C"]:
            sections.append(f"- {k}: {int(pkvc.get(k, 0))}")
        findings["pkvc"] = {k: int(pkvc.get(k, 0)) for k in ["P", "K", "V", "C"]}

    if "inferred_subtype" in expansion.columns:
        sub = expansion["inferred_subtype"].value_counts().head(15)
        sections.append("\n## inferred_subtype top 15\n")
        for k, v in sub.items():
            sections.append(f"- {k}: {int(v)}")

    # quality flags
    for col in ["confidence", "validation_ok"]:
        if col in expansion.columns:
            vc = expansion[col].value_counts()
            sections.append(f"\n## {col} 分布\n")
            for k, v in vc.items():
                sections.append(f"- {k}: {int(v)}")

    # expanded_nl 长度
    if "expanded_nl" in expansion.columns:
        nl_lens = expansion["expanded_nl"].fillna("").astype(str).str.len()
        sections.append("\n## expanded_nl 长度分布 (字符)\n")
        sections.append(f"- min={nl_lens.min()} median={nl_lens.median():.0f} max={nl_lens.max()} mean={nl_lens.mean():.1f}")
        findings["expanded_nl_median_len"] = float(nl_lens.median())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print("L7:", run_l7())
    print()
    print("L1b:", run_l1b())
