"""L1: data_v3 health audit.

Output: reports/datav3/L1_health.md (aggregate stats only, no PII).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import loaders

_REPORT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3" / "L1_health.md"


def _md_table(df: pd.DataFrame, headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run(output_path: Path | None = None) -> dict:
    output_path = output_path or _REPORT_PATH
    users = loaders.load_users()
    scale = loaders.load_scale_wide()
    task = loaders.load_task_long(expanded=True)
    expansion = loaders.load_expansion()

    findings: dict = {
        "n_users": int(len(users)),
        "scale_shape": list(scale.shape),
        "n_task_rows": int(len(task)),
        "n_expansion_rows": int(len(expansion)),
    }

    sections: list[str] = []
    sections.append(f"# L1 Health Audit — data_v3 (generated {pd.Timestamp.now().date()})\n")
    sections.append(f"- **N designers**: {len(users)}  (brief §1 旧值 200, 现以 163 为准)")
    sections.append(f"- **Scale matrix**: {scale.shape[0]} × {scale.shape[1]}  (期望 163 × 30)")
    sections.append(f"- **Task rows (expanded)**: {len(task)}")
    sections.append(f"- **Q1-Q6 expansion ok rows**: {len(expansion)}\n")

    # 30-dim describe
    desc = scale.describe().T[["mean", "std", "min", "max"]].round(2)
    desc["missing"] = scale.isna().sum().values
    desc["item"] = desc.index
    sections.append("## 30 量表题描述\n")
    sections.append(_md_table(desc.reset_index(drop=True), ["item", "mean", "std", "min", "max", "missing"]))

    # Q1-Q8 选项频率 (canonicalize 后, Q2 排序 / Q7 NL 跳过)
    task_canon = loaders.add_canonical_option(task)
    single_choice = task_canon[task_canon["option"].notna()]
    q_freq = single_choice.groupby(["question_id", "option"]).size().unstack(fill_value=0)
    sections.append("\n## Q1/Q3-Q6/Q8 单选题 选项频率 (canonicalized)\n")
    sections.append("(Q2 排序题 / Q7 NL 已跳过, 见后续 Q2-rank 段)\n")
    q_freq_md = q_freq.copy()
    q_freq_md["question_id"] = q_freq_md.index
    sections.append(_md_table(q_freq_md.reset_index(drop=True), ["question_id", *list(q_freq.columns)]))
    findings["q_option_min_freq"] = int(q_freq.values.min())
    findings["q_option_below_5pct"] = int(((q_freq / len(users)) < 0.05).sum().sum())

    # Q2 排序题: top-1 cardId 分布
    q2 = task[task["question_id"] == "Q2"].copy()
    if len(q2):
        q2["top1_card"] = q2["answer_value"].apply(lambda s: (loaders.parse_q2_ranking(s) or [None])[0])
        q2_top1 = q2["top1_card"].value_counts()
        sections.append("\n## Q2 排序题 top-1 cardId 分布\n")
        for k, v in q2_top1.items():
            sections.append(f"- {k}: {int(v)}")
        findings["q2_top1_counts"] = {str(k): int(v) for k, v in q2_top1.items()}

    # text_supplement 长度分布
    if "text_supplement" in task.columns:
        lens = task["text_supplement"].fillna("").astype(str).str.len()
        sections.append("\n## text_supplement 长度分布 (字符)\n")
        sections.append(f"- min={lens.min()} median={lens.median():.0f} max={lens.max()} mean={lens.mean():.1f}")
        findings["text_supp_median_len"] = float(lens.median())

    # PKVC imbalance (从 expansion 推断)
    if not expansion.empty and "inferred_factor" in expansion.columns:
        pkvc_counts = expansion["inferred_factor"].value_counts().to_dict()
        sections.append("\n## PKVC inferred factor 分布 (from expansion)\n")
        for k in ["P", "K", "V", "C"]:
            sections.append(f"- {k}: {int(pkvc_counts.get(k, 0))}")
        max_c, min_c = max(pkvc_counts.values()), min(pkvc_counts.values())
        ratio = max_c / max(min_c, 1)
        verdict = "FAIL (>3×)" if ratio > 3 else ("WARN (>2×)" if ratio > 2 else "PASS")
        sections.append(f"\n**Imbalance verdict**: {verdict}  (max/min = {ratio:.1f}×)")
        findings["pkvc_imbalance_ratio"] = float(ratio)
        findings["pkvc_counts"] = {k: int(v) for k, v in pkvc_counts.items()}

    # demographic skew
    sections.append("\n## Demographic 分布\n")
    for col in ["identity", "education_stage_current", "design_field", "years_exp"]:
        if col not in users.columns:
            continue
        vc = users[col].value_counts().head(8)
        sections.append(f"\n**{col}** (top 8):")
        for k, v in vc.items():
            sections.append(f"- {k}: {int(v)}")

    sections.append("\n## Verdicts\n")
    sections.append(f"- N = {len(users)} → brief §1 / __init__.py / 写稿 grep 全部 200→163")
    sections.append(f"- 量表 missing = {int(scale.isna().sum().sum())} cells")
    sections.append(f"- Q-option < 5% 频率单元: {findings['q_option_below_5pct']} 个 (mapping miner 要不要合并)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")

    findings_path = output_path.with_suffix(".json")
    findings_path.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    f = run()
    print(json.dumps(f, ensure_ascii=False, indent=2))
