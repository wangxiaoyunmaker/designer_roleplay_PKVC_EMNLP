"""Join mining associations + option_trait_mapping → 完整 quad 模板.

输入:
  reports/datav3/per_designer/quad_s_secondary_associations.jsonl  (92 行 raw 统计)
  reports/datav3/per_designer/option_trait_mapping.json            (23 个 Q_option entry)

输出:
  data/processed/quad_templates_v1.jsonl          (92 行, 含 in_quad_scope flag)
  data/processed/quad_templates_v1_robust.jsonl   (Q3-Q6 + significant=true, 论文用)

每行 quad 槽位:
  C           ← C_formal_proposed (Q3-Q6) 或 None (Q1/Q8)
  s_secondary ← {factor: F1..F5, direction: + / -}
  p_modulation← strength (5 档)
  o_primary   ← o_primary_hint (Q3-Q6 一定有, Q1/Q8 多数 None)
  o_subtype   ← o_subtype_hint (可能复合 X→Y)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSOC = ROOT / "reports/datav3/per_designer/quad_s_secondary_associations.jsonl"
MAPPING = ROOT / "reports/datav3/per_designer/option_trait_mapping.json"
OUT_ALL = ROOT / "data/processed/quad_templates_v1.jsonl"
OUT_ROBUST = ROOT / "data/processed/quad_templates_v1_robust.jsonl"


def build_quad(assoc_row: dict, mapping: dict) -> dict:
    qid = assoc_row["qid"]
    opt = assoc_row["option"]
    key = f"{qid}_{opt}"
    m = mapping.get(key, {})

    direction = "+" if assoc_row["delta_mean"] >= 0 else "-"

    C = m.get("C_formal_proposed") or m.get("C_raw")
    o_primary = m.get("o_primary_hint")
    in_quad_scope = qid in {"Q3", "Q4", "Q5", "Q6"} and o_primary is not None

    return {
        "C": C,
        "C_conflict_axes": m.get("C_conflict_axes"),
        "s_secondary": {"factor": assoc_row["factor"], "direction": direction},
        "p_modulation": assoc_row["strength"],
        "o_primary": o_primary,
        "o_subtype": m.get("o_subtype_hint"),
        "_stats": {
            "p_perm": assoc_row["p_perm"],
            "cohens_d": assoc_row["cohens_d"],
            "delta_mean": assoc_row["delta_mean"],
            "n_chose": assoc_row["n_chose"],
            "n_total": assoc_row["n_total"],
            "significant": assoc_row["significant"],
        },
        "_meta": {
            "qid": qid,
            "option": opt,
            "in_quad_scope": in_quad_scope,
            "join_complete": C is not None and o_primary is not None,
        },
    }


def main() -> None:
    assocs = [json.loads(l) for l in ASSOC.open(encoding="utf-8") if l.strip()]
    mapping = json.load(MAPPING.open(encoding="utf-8"))
    print(f"输入: {len(assocs)} associations, {len(mapping)} mapping entries")

    quads = [build_quad(a, mapping) for a in assocs]

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ALL.open("w", encoding="utf-8") as f:
        for q in quads:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    robust = [q for q in quads
              if q["_meta"]["in_quad_scope"] and q["_stats"]["significant"]]
    with OUT_ROBUST.open("w", encoding="utf-8") as f:
        for q in robust:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 报告
    print()
    print(f"写盘:")
    print(f"  {OUT_ALL.relative_to(ROOT)}            {len(quads)} rows (全部)")
    print(f"  {OUT_ROBUST.relative_to(ROOT)}     {len(robust)} rows (Q3-Q6 + significant)")
    print()

    by_q = Counter(q["_meta"]["qid"] for q in quads)
    print(f"全部按 qid: {dict(sorted(by_q.items()))}")

    in_scope = [q for q in quads if q["_meta"]["in_quad_scope"]]
    sig_in_scope = [q for q in in_scope if q["_stats"]["significant"]]
    print(f"in_quad_scope (Q3-Q6, 有 o_primary): {len(in_scope)} / {len(quads)}")
    print(f"  └─ significant: {len(sig_in_scope)} (这是论文 main mining 的 robust cell)")

    print()
    print("Robust quad 模板 (论文用):")
    print(f"  {'qid':<4} {'opt':<4} {'C_axes':<25} {'s_secondary':<14} {'p_mod':<10} {'o_prim':<3} {'o_subtype':<10} {'p_perm':<8}")
    for q in robust:
        s = q["s_secondary"]
        st = q["_stats"]
        print(f"  {q['_meta']['qid']:<4} {q['_meta']['option']:<4} "
              f"{(q.get('C_conflict_axes') or ''):<25} "
              f"({s['factor']}, {s['direction']})       "
              f"{q['p_modulation']:<10} "
              f"{q['o_primary'] or '-':<3} "
              f"{q.get('o_subtype') or '-':<10} "
              f"{st['p_perm']:<8.3f}")


if __name__ == "__main__":
    main()
