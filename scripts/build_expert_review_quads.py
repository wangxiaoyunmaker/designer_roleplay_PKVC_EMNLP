"""生成专家手工挑选用的 quad 候选清单 (老师 2026-05-24 拍板).

不再 cluster, 直接全局列候选 quad, 专家挑.

输入:
  data/processed/quad_extracted_v4_full.jsonl       (316 个 designer-quad 实例)
  data/task_responses_expanded.csv                  (designer NL + answer_value)
  reports/datav3/per_designer/option_trait_mapping.json  (学姐 schema + LASSO s_primary)
  data/processed/quad_templates_v1.jsonl            (factor-level robust 标 ★)

策略:
  每 (Q, option) 按 (p_surface, o_subtype) 联合分布:
    - 单一型 (n>=1) → 1 条
    - 双峰 (top-2 n>=2) → 2 条
    - 多峰 (top-3 n>=2) → 3 条
  每条 attach 2 个 confidence=high 的 designer NL 例证 (designer_id 匿名为 D1/D2/...).

输出:
  docs/notes/expert_review_quad_candidates_v1.md   (markdown 表 + 勾选列)
  data/processed/expert_review_quad_candidates_v1.csv  (Excel 便用)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from sft_pipeline.extract.codebook import (
    O_PRIMARY_NL, O_SUBTYPE_NL, O_SUBTYPE_DEFINITIONS,
    P_SURFACE_NL, P_MODULATION_NL,
    P_SURFACE_TO_MODULATION,
    C_LOCAL_NL_ZH, C_CONFLICT_TENSION_NL,
)


QUAD_JSONL = ROOT / "data/processed/quad_extracted_v4_full.jsonl"
TASK_CSV = ROOT / "data/task_responses_expanded.csv"
MAPPING = ROOT / "reports/datav3/per_designer/option_trait_mapping.json"
TEMPLATES = ROOT / "data/processed/quad_templates_v1.jsonl"
OUT_MD = ROOT / "docs/notes/expert_review_quad_candidates_v1.md"
OUT_CSV = ROOT / "data/processed/expert_review_quad_candidates_v1.csv"


def format_p(p_surface: str) -> str:
    return f"{P_SURFACE_NL.get(p_surface, '?')} ({p_surface})"


def format_o(o_subtype: str) -> str:
    if "→" in o_subtype:
        s, e = o_subtype.split("→")
        return f"{O_SUBTYPE_NL.get(s, s)} → {O_SUBTYPE_NL.get(e, e)} ({o_subtype})"
    return f"{O_SUBTYPE_NL.get(o_subtype, o_subtype)} ({o_subtype})"


def format_s_primary(stance_list: list) -> str:
    if not stance_list:
        return "(无)"
    parts = []
    for it in stance_list:
        dir_ = it.get("direction", "?")
        stance = it.get("stance") or it.get("item", "?")
        coef = it.get("coef", 0)
        p = it.get("marginal_perm_p")
        sig = " ★" if p is not None and p < 0.05 else ""
        parts.append(f"{stance} ({dir_}β={coef:+.2f}{sig})")
    return "; ".join(parts)


def load_factor_robust_flag(templates_path: Path) -> dict:
    """返回 {(qid, option): [(factor, direction, p_perm, strength), ...]} 仅 robust."""
    out = defaultdict(list)
    for line in templates_path.open(encoding="utf-8"):
        row = json.loads(line)
        if not row["_stats"]["significant"]:
            continue
        if not row["_meta"]["in_quad_scope"]:
            continue
        out[(row["_meta"]["qid"], row["_meta"]["option"])].append((
            row["s_secondary"]["factor"],
            row["s_secondary"]["direction"],
            row["_stats"]["p_perm"],
            row["p_modulation"],
        ))
    return out


def select_candidates(rows: list[dict]) -> list[dict]:
    """对每 (Q, option) 应用 top-k 规则, 返回候选 quad 列表."""
    by_qopt = defaultdict(list)
    for r in rows:
        by_qopt[(r["question_id"], r["_meta"]["choice_selected"])].append(r)

    candidates = []
    for (qid, opt) in sorted(by_qopt.keys()):
        instances = by_qopt[(qid, opt)]
        pairs = Counter((r["p_surface"], r["o_subtype"]) for r in instances)
        chosen = []
        for (p, o), c in pairs.most_common():
            if not chosen:
                chosen.append(((p, o), c))
            elif len(chosen) < 3 and c >= 2:
                chosen.append(((p, o), c))
            else:
                break

        for idx, ((p, o), c) in enumerate(chosen, 1):
            matching = [r for r in instances
                        if r["p_surface"] == p and r["o_subtype"] == o]
            candidates.append({
                "quad_id": f"{qid}{opt}-{idx}",
                "qid": qid,
                "option": opt,
                "n_instances_total": len(instances),
                "n_match": c,
                "p_surface": p,
                "o_subtype": o,
                "matching_designers": [r["designer_id"] for r in matching],
                "matching_confidences": [r["confidence"] for r in matching],
            })
    return candidates


def attach_examples(candidates: list[dict], task_df: pd.DataFrame) -> None:
    """每条候选挂 2 个 confidence=high 的 designer NL."""
    task_df["user_id"] = task_df["user_id"].astype(str)
    nl_lookup = {}
    answer_lookup = {}
    for _, row in task_df.iterrows():
        key = (row["user_id"], row["question_id"])
        nl_lookup[key] = str(row["text_supplement"] or "").strip()
        answer_lookup[(row["user_id"], row["question_id"])] = str(row.get("answer_value", "")).strip()

    for c in candidates:
        # 优先 confidence=high, 其次 medium
        priority = [(d, conf) for d, conf in zip(c["matching_designers"], c["matching_confidences"])
                    if conf == "high"]
        if len(priority) < 2:
            extra = [(d, conf) for d, conf in zip(c["matching_designers"], c["matching_confidences"])
                     if conf == "medium" and (d, conf) not in priority]
            priority += extra
        examples = []
        for d, conf in priority[:2]:
            nl = nl_lookup.get((d, c["qid"]), "(NL 缺失)")
            examples.append({"designer_id": d, "confidence": conf, "nl": nl})
        c["examples"] = examples
        # 顺便存选项原文
        if c["matching_designers"]:
            d0 = c["matching_designers"][0]
            c["answer_value"] = answer_lookup.get((d0, c["qid"]), "")


def render_markdown(candidates: list[dict], mapping: dict, factor_robust: dict) -> str:
    lines = []
    lines.append("# T-CAPS Quad 候选清单 — 专家手工挑选 v1")
    lines.append("")
    lines.append("> 老师 2026-05-24 拍板: **不再聚类**, 直接全局列候选 quad 给专家挑.")
    lines.append("> 共 {} 条候选, 覆盖 Q3-Q6 × 4 option (16 个 cell), 按 (p_surface, o_subtype) 联合分布拆解.".format(len(candidates)))
    lines.append("")
    lines.append("## 怎么用")
    lines.append("")
    lines.append("- 每条 quad 一段, 含: **情境 C** / **特质画像 s** / **认知路由 p** / **目标状态 o** + 2 段 designer 真实 NL 例证.")
    lines.append("- 专家在每条末尾 `专家判断` 行勾选 `[x] 收 / [x] 不收 / [x] 待定`, 并在 `批注` 写理由.")
    lines.append("- ★ 标记 = 该 (Q, option) 在 EFA factor 层 mining 通过 permutation test (p<0.05).")
    lines.append("- 字段释义见末尾 §字段表.")
    lines.append("")
    lines.append("---")
    lines.append("")

    cur_q = None
    for c in candidates:
        if c["qid"] != cur_q:
            cur_q = c["qid"]
            lines.append("")
            lines.append(f"## {cur_q}")
            lines.append("")

        # 选项 header
        if c["quad_id"].endswith("-1"):
            mapping_key = f"{c['qid']}_{c['option']}"
            m = mapping.get(mapping_key, {})
            C_formal = m.get("C_formal_proposed") or "?"
            C_axes = m.get("C_conflict_axes") or "?"
            C_raw = m.get("C_raw") or ""
            C_nl = C_LOCAL_NL_ZH.get(C_formal, C_formal)
            C_tension_nl = C_CONFLICT_TENSION_NL.get(C_axes, C_axes)
            lines.append(f"### {c['qid']} 选项 {c['option']} (n={c['n_instances_total']})")
            lines.append("")
            lines.append(f"**选项原文**: {c.get('answer_value', '(待补)')}")
            lines.append("")
            lines.append(f"**情境 C**: {C_nl} ({C_formal})")
            lines.append("")
            lines.append(f"**冲突轴**: {C_tension_nl}")
            if C_raw:
                lines.append(f" — 学姐原文: {C_raw}")
            lines.append("")
            lines.append(f"**特质画像 s_primary** (LASSO 选这个选项的人共同 trait, ★ = marginal_perm_p<0.05):")
            lines.append("")
            lines.append(f"  {format_s_primary(m.get('s_primary_with_stance', []))}")
            lines.append("")
            robust_factors = factor_robust.get((c["qid"], c["option"]), [])
            if robust_factors:
                rf_str = "; ".join(f"({fac}, {dir_}) → {strn}, p={p:.3f}"
                                  for fac, dir_, p, strn in robust_factors)
                lines.append(f"**Factor-level robust ★**: {rf_str}")
                lines.append("")
            lines.append(f"**学姐 schema hint**: o_primary={m.get('o_primary_hint') or '?'}, "
                        f"o_subtype={m.get('o_subtype_hint') or '?'}")
            lines.append("")

        # 候选 quad
        p_mod = P_SURFACE_TO_MODULATION.get(c["p_surface"], "?")
        p_mod_nl = P_MODULATION_NL.get(p_mod, p_mod)
        # o_primary 推断
        o_end = c["o_subtype"].split("→")[-1] if "→" in c["o_subtype"] else c["o_subtype"]
        o_primary = o_end[0]
        o_primary_nl = O_PRIMARY_NL.get(o_primary, "?")
        sup_pct = 100 * c["n_match"] / c["n_instances_total"]

        lines.append(f"#### Quad `{c['quad_id']}` — 支持度 {c['n_match']}/{c['n_instances_total']} ({sup_pct:.0f}%)")
        lines.append("")
        lines.append(f"- **p_surface (认知路由表面)**: {format_p(c['p_surface'])}")
        lines.append(f"- **p_modulation (路由强度)**: {p_mod_nl} ({p_mod})")
        lines.append(f"- **o_primary (跳到哪个 PKVC 状态)**: {o_primary_nl} ({o_primary})")
        lines.append(f"- **o_subtype (cardId 细分)**: {format_o(c['o_subtype'])}")
        lines.append("")
        lines.append("**自然语言读法**:")
        lines.append("")
        mapping_key2 = f"{c['qid']}_{c['option']}"
        m2 = mapping.get(mapping_key2, {})
        c_short = C_CONFLICT_TENSION_NL.get(m2.get("C_conflict_axes", ""), "?")
        s_str = format_s_primary(m2.get("s_primary_with_stance", []))
        s_short = s_str[:80] + ("..." if len(s_str) > 80 else "")
        nl_reading = (
            f"  当设计师面临【{c_short}】的冲突情境时, "
            f"具有【{s_short}】这类特质的人, "
            f"会通过【{P_SURFACE_NL.get(c['p_surface'], '?')}】的方式 ({p_mod_nl}), "
            f"跳到【{format_o(c['o_subtype'])}】这个认知状态."
        )
        lines.append(nl_reading)
        lines.append("")

        # NL 例证
        lines.append("**Designer 真实 NL 例证** (anonymized):")
        lines.append("")
        for i, ex in enumerate(c.get("examples", []), 1):
            nl_clean = ex["nl"].replace("\n", " ").replace("\r", " ")
            lines.append(f"  - **D{i}** (confidence={ex['confidence']}): {nl_clean}")
        lines.append("")

        # 专家判断 + 批注
        lines.append("**专家判断**: `[ ] 收` `[ ] 不收` `[ ] 待定`")
        lines.append("")
        lines.append("**批注**: ")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 字段表
    lines.append("")
    lines.append("## 字段表 (附)")
    lines.append("")
    lines.append("### p_surface 8 词闭集")
    lines.append("")
    lines.append("| token | 中文 | p_modulation |")
    lines.append("|---|---|---|")
    for k, v in P_SURFACE_NL.items():
        lines.append(f"| {k} | {v} | {P_SURFACE_TO_MODULATION[k]} |")
    lines.append("")
    lines.append("### o_subtype 12 cardId 子类")
    lines.append("")
    lines.append("| cardId | 名称 | 一句话定义 |")
    lines.append("|---|---|---|")
    for k, v in O_SUBTYPE_DEFINITIONS.items():
        name, definition = v.split(" — ", 1)
        lines.append(f"| {k} | {name} | {definition} |")
    lines.append("")

    return "\n".join(lines)


def render_csv(candidates: list[dict], mapping: dict, factor_robust: dict) -> list[dict]:
    out = []
    for c in candidates:
        mapping_key = f"{c['qid']}_{c['option']}"
        m = mapping.get(mapping_key, {})
        o_end = c["o_subtype"].split("→")[-1] if "→" in c["o_subtype"] else c["o_subtype"]
        o_primary = o_end[0]
        p_mod = P_SURFACE_TO_MODULATION.get(c["p_surface"], "")
        robust_factors = factor_robust.get((c["qid"], c["option"]), [])
        rf_str = "; ".join(f"({fac},{dir_})p={p:.3f}" for fac, dir_, p, _ in robust_factors)
        ex_nls = [ex["nl"] for ex in c.get("examples", [])]
        out.append({
            "quad_id": c["quad_id"],
            "qid": c["qid"],
            "option": c["option"],
            "answer_value": c.get("answer_value", ""),
            "C_formal": m.get("C_formal_proposed", ""),
            "C_conflict_axes": m.get("C_conflict_axes", ""),
            "s_primary_summary": format_s_primary(m.get("s_primary_with_stance", [])),
            "p_surface": c["p_surface"],
            "p_modulation": p_mod,
            "o_primary": o_primary,
            "o_subtype": c["o_subtype"],
            "support": f"{c['n_match']}/{c['n_instances_total']}",
            "factor_robust": rf_str,
            "example_1": ex_nls[0] if len(ex_nls) > 0 else "",
            "example_2": ex_nls[1] if len(ex_nls) > 1 else "",
            "expert_decision": "",
            "expert_note": "",
        })
    return out


def main() -> None:
    rows = [json.loads(l) for l in QUAD_JSONL.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "error" not in r]
    print(f"Loaded {len(rows)} quad instances")

    mapping = json.load(MAPPING.open(encoding="utf-8"))
    print(f"Loaded {len(mapping)} option_trait_mapping entries")

    factor_robust = load_factor_robust_flag(TEMPLATES)
    print(f"Robust factor-level cells: {sum(len(v) for v in factor_robust.values())}")

    task_df = pd.read_csv(TASK_CSV)
    task_df.columns = [c.strip().lstrip("﻿") for c in task_df.columns]
    print(f"Loaded task_responses_expanded: {len(task_df)} rows")

    candidates = select_candidates(rows)
    print(f"Generated {len(candidates)} candidates")

    attach_examples(candidates, task_df)

    md = render_markdown(candidates, mapping, factor_robust)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\nWrote markdown → {OUT_MD.relative_to(ROOT)}")

    csv_rows = render_csv(candidates, mapping, factor_robust)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote csv → {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
