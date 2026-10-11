"""CAPS typology matching: individual quad ↔ expert typology similarity + soft assignment.

依赖:
  - Individual quad (data_v3/processed/quad_extracted_163_fixed.jsonl): 163 designer × 4 Q (Q3-Q6) = 652 quad
  - Typology candidates (data/processed/expert_review_quad_candidates_v1.csv): 28 candidates (待专家收 ~20)
    专家 finalize 后, expert_decision=='收' 的 row 是 final typology pool

Similarity = α * s_jaccard + β * p_surface_match + γ * p_modulation_compat + δ * o_match
默认 α=0.40, β=0.25, γ=0.15, δ=0.20.

Soft assignment: designer 的 4 个 quad 各自跟所有 typology 算 sim, softmax 聚合.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Typology:
    quad_id: str            # "Q3A-1"
    qid: str                # "Q3"
    option: str             # "A"
    C_formal: str
    s_items: set[str]       # {"后天努力论", "包容失控派", "独立打磨派"}
    p_surface: str          # "forces_transition_to"
    p_modulation: str       # "Strong+"
    o_primary: str          # "K"
    o_subtype: str          # "K2" or "P3→K3"
    support: str            # "4/6"


@dataclass
class IndividualQuad:
    designer_id: str
    qid: str
    option: str
    s_items: set[str]
    p_surface: str
    p_modulation: str
    o_primary: str
    o_subtype: str
    confidence: str


# ============================================================
# Loaders
# ============================================================

S_ITEM_RE = re.compile(r"([^();,\s]+(?:[^();,]*?))\s*\([+\-]?β=[+\-]?[\d.]+(?:\s*★)?\)")


def parse_s_summary(s_summary: str | float | None) -> set[str]:
    """Parse 's_primary_summary' string into set of item names.
    e.g. '后天努力论 (-β=-0.35); 包容失控派 (+β=+0.28); 独立打磨派 (+β=+0.25)'
         → {'后天努力论', '包容失控派', '独立打磨派'}
    """
    if not isinstance(s_summary, str):
        return set()
    matches = S_ITEM_RE.findall(s_summary)
    return {m.strip() for m in matches if m.strip() and m.strip() != "(无)"}


def load_typology_candidates(
    csv_path: Path,
    only_accepted: bool = False,
) -> list[Typology]:
    """Load expert review candidates. If only_accepted=True, filter by expert_decision=='收'.

    Note: 专家 review 没完, only_accepted=False 返回全部 28 个候选 (作为 mock typology).
    专家 finalize 后, only_accepted=True 返回 ~20 final.
    """
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        if only_accepted and str(r.get("expert_decision") or "").strip() != "收":
            continue
        out.append(Typology(
            quad_id=str(r["quad_id"]),
            qid=str(r["qid"]),
            option=str(r["option"]),
            C_formal=str(r["C_formal"]),
            s_items=parse_s_summary(r["s_primary_summary"]),
            p_surface=str(r["p_surface"]),
            p_modulation=str(r["p_modulation"]),
            o_primary=str(r["o_primary"]),
            o_subtype=str(r["o_subtype"]),
            support=str(r.get("support", "")),
        ))
    return out


def load_individual_quads(jsonl_path: Path) -> list[IndividualQuad]:
    out = []
    for line in jsonl_path.open(encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            continue
        # Skip rows that still have validation fails (shouldn't be any post-fix)
        if r.get("_meta", {}).get("validation_fails"):
            continue
        # Parse s_primary: it's a list of strings already
        s_items = set(r.get("s_primary") or [])
        out.append(IndividualQuad(
            designer_id=str(r["designer_id"]),
            qid=str(r["question_id"]),
            option=str(r.get("_meta", {}).get("choice_selected", "")),
            s_items=s_items,
            p_surface=str(r["p_surface"]),
            p_modulation=str(r["p_modulation"]),
            o_primary=str(r["o_primary"]),
            o_subtype=str(r["o_subtype"]),
            confidence=str(r.get("confidence", "medium")),
        ))
    return out


# ============================================================
# Similarity components
# ============================================================

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def p_surface_match(s1: str, s2: str) -> float:
    """Exact match on 8-word closed set."""
    return 1.0 if s1 == s2 else 0.0


def p_modulation_compat(m1: str, m2: str) -> float:
    """Compatibility on Strong+/Weak+/Strong-/Weak-.

    Same: 1.0
    Same sign, different strength: 0.6
    Opposite sign: 0.0
    Unknown or '?': 0.5 (lenient)
    """
    if m1 == m2:
        return 1.0
    if not m1 or not m2 or "?" in (m1, m2):
        return 0.5
    sign1 = m1[-1] if m1[-1] in "+-" else None
    sign2 = m2[-1] if m2[-1] in "+-" else None
    if sign1 is None or sign2 is None:
        return 0.5
    if sign1 != sign2:
        return 0.0
    return 0.6  # same sign, different strength


def o_match(
    o_primary1: str, o_subtype1: str,
    o_primary2: str, o_subtype2: str,
) -> float:
    """Hierarchical match: o_primary level + subtype refinement.

    v2 (2026-05-25, after Q6 analysis): trajectory mismatch 更严, 帮区分
    "V2 (停在 V2)" vs "V2→P1 (从 V2 转 P1)" 等 transition pattern.

    Primary mismatch: 0.0
    Primary match, exact subtype match (含 transition path): 1.0
    Primary match, end-prefix subtype同 letter (e.g. K3 vs K3): 0.7
    Primary match, **trajectory pattern 不同 (一边有 X→Y 一边没)**: **0.3** (改严)
    Primary match, end primary 同但 subtype 不同 letter: 0.5
    Primary match, end primary 不同: 0.3
    """
    if o_primary1 != o_primary2:
        return 0.0
    if o_subtype1 == o_subtype2:
        return 1.0
    if not o_subtype1 or not o_subtype2:
        return 0.5
    has_arrow1 = "→" in o_subtype1
    has_arrow2 = "→" in o_subtype2
    # trajectory pattern mismatch: 一个 X→Y 一个单字母 → 行为模式根本不同
    if has_arrow1 != has_arrow2:
        return 0.3
    # Compare end-prefix (for X→Y, take Y; else take subtype)
    end1 = o_subtype1.split("→")[-1]
    end2 = o_subtype2.split("→")[-1]
    if end1[:2] == end2[:2]:
        # same end subtype letter (e.g., K3 vs K3, or P1 vs P1)
        # 若都是 X→Y 且 Y 同, 看起点 X 是否同
        if has_arrow1 and has_arrow2:
            start1 = o_subtype1.split("→")[0]
            start2 = o_subtype2.split("→")[0]
            if start1[:2] == start2[:2]:
                return 1.0  # 完整 path 同
            return 0.6  # Y 同 X 不同
        return 0.7
    if end1[0] == end2[0]:
        return 0.5
    return 0.3


# ============================================================
# Full similarity
# ============================================================

@dataclass
class SimWeights:
    """Default weights v3 (after Q6 平局 case analysis 2026-05-25):
    Q6 内 Q6A-1 跟 Q6C-3 仅 option 字段差 (other completely same), 不加 option match 就 50/50 平.
    所以引入 epsilon_option: designer 实际选 option 跟 typology option 一致才有 bonus.
    """
    alpha_s: float = 0.10        # s_jaccard (砍到 0.10, 同 option 内没区分)
    beta_p_surface: float = 0.25 # p_surface exact match
    gamma_p_mod: float = 0.10    # p_modulation compatibility
    delta_o: float = 0.25        # o_match (o_primary + o_subtype)
    epsilon_option: float = 0.30 # option match (designer 选的 option == typology.option)


def quad_similarity(
    individual: IndividualQuad,
    typology: Typology,
    weights: SimWeights = SimWeights(),
) -> float:
    """Compute sim(individual, typology) ∈ [0, 1].
    """
    s = jaccard(individual.s_items, typology.s_items)
    p = p_surface_match(individual.p_surface, typology.p_surface)
    g = p_modulation_compat(individual.p_modulation, typology.p_modulation)
    o = o_match(individual.o_primary, individual.o_subtype,
                typology.o_primary, typology.o_subtype)
    # v3: designer 实际选 option 跟 typology option 一致才得分
    opt_match = 1.0 if (individual.option and individual.option == typology.option) else 0.0
    return (
        weights.alpha_s * s
        + weights.beta_p_surface * p
        + weights.gamma_p_mod * g
        + weights.delta_o * o
        + weights.epsilon_option * opt_match
    )


# ============================================================
# Soft assignment
# ============================================================

def soft_assign_designer(
    designer_quads: list[IndividualQuad],
    typologies: list[Typology],
    weights: SimWeights = SimWeights(),
    temperature: float = 1.0,
    restrict_to_same_qid: bool = True,
) -> dict[str, float]:
    """对 1 个 designer (含 4 个 quads) → typology id 上的 probability distribution.

    策略:
      1. For each individual_quad q in designer_quads:
         For each typology t (optionally restrict to t.qid == q.qid):
            score(q, t) = quad_similarity(q, t)
      2. Aggregate across designer_quads:
           score(t) = mean over q of score(q, t)         (when restrict_to_same_qid:
                                                          只取 q.qid == t.qid 的那个 q)
      3. Softmax over typologies → probability distribution.

    Args:
      restrict_to_same_qid:
        True (default): typology t 跟 same-qid 的 individual quad 比 (4 quads × N_per_qid typologies)
        False: 每个 quad 跟所有 typology 比 (16 个 typology 比成 cross-qid 也比, 信号弱)

    Returns:
      dict[typology_id → prob], sums to 1
    """
    typo_scores: dict[str, list[float]] = defaultdict(list)
    for q in designer_quads:
        for t in typologies:
            if restrict_to_same_qid and t.qid != q.qid:
                continue
            s = quad_similarity(q, t, weights)
            typo_scores[t.quad_id].append(s)

    # Aggregate: mean (only positive scores count)
    raw: dict[str, float] = {}
    for tid, scores in typo_scores.items():
        if scores:
            raw[tid] = float(np.mean(scores))

    # Softmax
    if not raw:
        return {}
    keys = list(raw.keys())
    vals = np.array([raw[k] for k in keys])
    vals_scaled = vals / max(temperature, 1e-6)
    exp = np.exp(vals_scaled - vals_scaled.max())  # numerical stability
    probs = exp / exp.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def soft_assign_per_qid(
    designer_quads: list[IndividualQuad],
    typologies: list[Typology],
    weights: SimWeights = SimWeights(),
    temperature: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Per-qid soft assignment for a designer.

    Returns dict[qid → dict[typology_id → prob]], each per-qid dist sums to 1.
    Each qid 内的 quad 跟 same-qid typology 做 softmax. 4 qid = 4 个独立 distribution.

    跟 Mischel CAPS If-Then framing 对齐: 每个情境 (qid) 下 designer 走哪个 typology.
    """
    by_qid: dict[str, list[IndividualQuad]] = defaultdict(list)
    for q in designer_quads:
        by_qid[q.qid].append(q)

    typo_by_qid: dict[str, list[Typology]] = defaultdict(list)
    for t in typologies:
        typo_by_qid[t.qid].append(t)

    result: dict[str, dict[str, float]] = {}
    for qid, quads in by_qid.items():
        local_typos = typo_by_qid.get(qid, [])
        if not local_typos:
            continue
        sims = []
        for t in local_typos:
            # average sim across designer's quads for this qid (typically just 1)
            qs = [quad_similarity(q, t, weights) for q in quads]
            sims.append((t.quad_id, float(np.mean(qs))))
        keys, vals = zip(*sims) if sims else ([], [])
        if not vals:
            continue
        vals_arr = np.array(vals) / max(temperature, 1e-6)
        exp = np.exp(vals_arr - vals_arr.max())
        probs = exp / exp.sum()
        result[qid] = {k: float(p) for k, p in zip(keys, probs)}
    return result


def assign_all_designers(
    individual_quads_path: Path,
    typology_csv_path: Path,
    only_accepted_typologies: bool = False,
    weights: SimWeights = SimWeights(),
    temperature: float = 0.1,           # 低温让 distribution 更尖, 偏向 top-1/top-2
    restrict_to_same_qid: bool = True,
    per_qid: bool = False,              # True: per-qid 4 dists; False: global 28-dim
) -> dict[str, dict]:
    """Top-level: load + assign all designers.

    Returns:
      若 per_qid=False: {designer_id → {typology_id → prob}}, 全局 softmax over all typologies
      若 per_qid=True:  {designer_id → {qid → {typology_id → prob}}}, 每个 qid 单独 softmax
    """
    individual_quads = load_individual_quads(individual_quads_path)
    typologies = load_typology_candidates(typology_csv_path, only_accepted=only_accepted_typologies)

    by_designer: dict[str, list[IndividualQuad]] = defaultdict(list)
    for q in individual_quads:
        by_designer[q.designer_id].append(q)

    out: dict[str, dict] = {}
    for did, quads in by_designer.items():
        if per_qid:
            out[did] = soft_assign_per_qid(quads, typologies, weights=weights, temperature=temperature)
        else:
            out[did] = soft_assign_designer(
                quads, typologies, weights=weights,
                temperature=temperature, restrict_to_same_qid=restrict_to_same_qid,
            )
    return out


# ============================================================
# CLI / demo
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quads", default="data_v3/processed/quad_extracted_163_fixed.jsonl")
    ap.add_argument("--typology-csv", default="data/processed/expert_review_quad_candidates_v1.csv")
    ap.add_argument("--only-accepted", action="store_true",
                    help="仅用 expert_decision='收' 的 typology (待专家 finalize)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--per-qid", action="store_true",
                    help="True: 每 qid 一个 distribution (CAPS If-Then framing); False: 全局 28-dim")
    ap.add_argument("--output", default="data_v3/processed/typology_soft_assignment.json")
    ap.add_argument("--top-k-preview", type=int, default=3, help="每 designer 显示 top-k")
    args = ap.parse_args()

    print(f"[load] individual quads from {args.quads}")
    iq = load_individual_quads(Path(args.quads))
    print(f"  {len(iq)} valid individual quads across {len({q.designer_id for q in iq})} designers")

    print(f"[load] typology candidates from {args.typology_csv} "
          f"(only_accepted={args.only_accepted})")
    typo = load_typology_candidates(Path(args.typology_csv), only_accepted=args.only_accepted)
    print(f"  {len(typo)} typology candidates loaded")
    if not typo:
        print("  WARNING: 0 typologies! (likely expert hasn't marked any '收' yet)")
        return 1

    from collections import Counter
    qid_count = Counter(t.qid for t in typo)
    print(f"  per qid: {dict(qid_count)}")

    print(f"\n[assign] computing soft assignments (temperature={args.temperature}, "
          f"per_qid={args.per_qid})...")
    assignments = assign_all_designers(
        Path(args.quads), Path(args.typology_csv),
        only_accepted_typologies=args.only_accepted,
        temperature=args.temperature,
        per_qid=args.per_qid,
    )
    print(f"  assigned {len(assignments)} designers")

    # Preview
    if args.per_qid:
        print(f"\n=== per-qid top-{args.top_k_preview} preview (first 3 designers) ===")
        for did in list(assignments.keys())[:3]:
            print(f"\n  {did[:14]}:")
            for qid, dist in sorted(assignments[did].items()):
                top_k = sorted(dist.items(), key=lambda x: -x[1])[:args.top_k_preview]
                top_str = "  ".join(f"{tid}={p:.2f}" for tid, p in top_k)
                print(f"    {qid}: {top_str}")

        # Per-qid concentration stats
        print(f"\n=== per-qid distribution concentration ===")
        for qid in ("Q3", "Q4", "Q5", "Q6"):
            top1s = []
            for probs in assignments.values():
                if qid in probs and probs[qid]:
                    top1s.append(max(probs[qid].values()))
            if top1s:
                print(f"  {qid}: top-1 median={np.median(top1s):.2f}, "
                      f"sharp (>0.5): {sum(1 for t in top1s if t > 0.5)}/{len(top1s)}")
        # Per-qid 最 popular typology
        print(f"\n=== per-qid most popular typology (top-1 count) ===")
        for qid in ("Q3", "Q4", "Q5", "Q6"):
            top1_typo = Counter()
            for probs in assignments.values():
                if qid in probs and probs[qid]:
                    top1_typo[max(probs[qid], key=probs[qid].get)] += 1
            print(f"  {qid}: {top1_typo.most_common(5)}")
    else:
        print(f"\n=== top-{args.top_k_preview} typology preview (first 5 designers) ===")
        for did in list(assignments.keys())[:5]:
            sorted_probs = sorted(assignments[did].items(), key=lambda x: -x[1])
            top_k = sorted_probs[:args.top_k_preview]
            top_str = "  ".join(f"{tid}={p:.2f}" for tid, p in top_k)
            print(f"  {did[:14]}: {top_str}")

        top1_probs = [max(probs.values()) for probs in assignments.values() if probs]
        print(f"\n=== distribution concentration (across {len(top1_probs)} designers) ===")
        print(f"  top-1 prob: min={min(top1_probs):.2f} median={np.median(top1_probs):.2f} max={max(top1_probs):.2f}")
        top3_concentration = [sum(sorted(p.values(), reverse=True)[:3]) for p in assignments.values() if p]
        print(f"  top-3 cumulative prob: median={np.median(top3_concentration):.2f}")
        top1_typo = Counter(max(p.items(), key=lambda x: x[1])[0] for p in assignments.values() if p)
        print(f"  most popular typology (top-1 count): {top1_typo.most_common(5)}")

    # save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
