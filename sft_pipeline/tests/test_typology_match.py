"""Unit + sanity tests for typology_match.

Run: .venv/bin/python -m pytest sft_pipeline/tests/test_typology_match.py -v
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from sft_pipeline.rag.typology_match import (
    IndividualQuad, Typology, SimWeights,
    jaccard, p_surface_match, p_modulation_compat, o_match,
    quad_similarity, soft_assign_designer, parse_s_summary,
    load_individual_quads, load_typology_candidates,
)


# ============================================================
# component tests
# ============================================================

def test_jaccard():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1/3)


def test_p_surface_match():
    assert p_surface_match("forces_transition_to", "forces_transition_to") == 1.0
    assert p_surface_match("forces_transition_to", "compliantly_follows") == 0.0


def test_p_modulation_compat():
    assert p_modulation_compat("Strong+", "Strong+") == 1.0
    assert p_modulation_compat("Strong+", "Weak+") == 0.6   # same sign, diff strength
    assert p_modulation_compat("Strong+", "Strong-") == 0.0  # opposite sign
    assert p_modulation_compat("Weak+", "Weak-") == 0.0
    assert p_modulation_compat("?", "Strong+") == 0.5        # unknown lenient


def test_o_match():
    # primary mismatch
    assert o_match("K", "K2", "V", "V3") == 0.0
    # exact match
    assert o_match("K", "K2", "K", "K2") == 1.0
    # primary match, subtype letter same
    assert o_match("K", "K3", "K", "K3") == 1.0
    # v2: both X→Y, same Y, diff X (P3→K3 vs P2→K3)
    assert o_match("K", "P3→K3", "K", "P2→K3") == 0.6
    # v2: both X→Y exact match
    assert o_match("K", "P3→K3", "K", "P3→K3") == 1.0
    # v2: trajectory pattern mismatch (one has arrow, other doesn't)
    assert o_match("K", "K2", "K", "P3→K3") == 0.3
    # primary match, only primary same: K2 vs K3
    assert o_match("K", "K2", "K", "K3") == 0.5


def test_parse_s_summary():
    s = "后天努力论 (-β=-0.35); 包容失控派 (+β=+0.28); 独立打磨派 (+β=+0.25)"
    assert parse_s_summary(s) == {"后天努力论", "包容失控派", "独立打磨派"}
    assert parse_s_summary(None) == set()
    assert parse_s_summary("(无)") == set()


# ============================================================
# full sim test
# ============================================================

def test_quad_similarity_perfect_match():
    iq = IndividualQuad(
        designer_id="D1", qid="Q3", option="A",
        s_items={"a", "b"}, p_surface="forces_transition_to",
        p_modulation="Strong+", o_primary="K", o_subtype="K2",
        confidence="high",
    )
    t = Typology(
        quad_id="Q3A-1", qid="Q3", option="A", C_formal="C1",
        s_items={"a", "b"}, p_surface="forces_transition_to",
        p_modulation="Strong+", o_primary="K", o_subtype="K2",
        support="5/5",
    )
    sim = quad_similarity(iq, t)
    assert sim == pytest.approx(1.0)


def test_quad_similarity_full_mismatch():
    """所有 component 不同 + option 也不同 → 全 0."""
    iq = IndividualQuad(
        designer_id="D1", qid="Q3", option="A",
        s_items={"x"}, p_surface="forces_transition_to",
        p_modulation="Strong+", o_primary="K", o_subtype="K2",
        confidence="high",
    )
    t = Typology(
        quad_id="Q3B-1", qid="Q3", option="B", C_formal="C1",  # different option
        s_items={"y"}, p_surface="compliantly_follows",
        p_modulation="Weak-", o_primary="V", o_subtype="V3",
        support="3/3",
    )
    sim = quad_similarity(iq, t)
    assert sim == 0.0


def test_quad_similarity_partial():
    """Designer matches o (V) but not p, has half s overlap.
    Default SimWeights v3: α=0.10, β=0.25, γ=0.10, δ=0.25, ε=0.30
    """
    iq = IndividualQuad(
        designer_id="D1", qid="Q4", option="B",
        s_items={"a", "b"}, p_surface="forces_transition_to",
        p_modulation="Strong+", o_primary="V", o_subtype="V3",
        confidence="medium",
    )
    t = Typology(
        quad_id="Q4B-1", qid="Q4", option="B", C_formal="C1",
        s_items={"a", "c"}, p_surface="compliantly_follows",
        p_modulation="Strong-", o_primary="V", o_subtype="V3",
        support="5/5",
    )
    # s: 1/3 ≈ 0.333; p_surface: 0; p_mod: 0; o: 1; opt: 1 (B==B)
    expected = 0.10 * (1/3) + 0.25 * 0 + 0.10 * 0 + 0.25 * 1.0 + 0.30 * 1.0
    sim = quad_similarity(iq, t)
    assert sim == pytest.approx(expected)


def test_quad_similarity_option_mismatch():
    """Same p/o/s but different option → epsilon contribution = 0."""
    iq = IndividualQuad(
        designer_id="D1", qid="Q6", option="A",
        s_items={"x"}, p_surface="gradually_shifts_to",
        p_modulation="Weak+", o_primary="V", o_subtype="V2",
        confidence="high",
    )
    t_same_opt = Typology(
        quad_id="Q6A-1", qid="Q6", option="A", C_formal="C",
        s_items={"x"}, p_surface="gradually_shifts_to",
        p_modulation="Weak+", o_primary="V", o_subtype="V2",
        support="10/13",
    )
    t_diff_opt = Typology(
        quad_id="Q6C-3", qid="Q6", option="C", C_formal="C",
        s_items={"x"}, p_surface="gradually_shifts_to",
        p_modulation="Weak+", o_primary="V", o_subtype="V2",
        support="9/33",
    )
    sim_same = quad_similarity(iq, t_same_opt)
    sim_diff = quad_similarity(iq, t_diff_opt)
    # 同 option 跟不同 option, 应该差 epsilon = 0.30
    assert sim_same - sim_diff == pytest.approx(0.30)


# ============================================================
# soft assignment test
# ============================================================

def test_soft_assign_returns_distribution():
    """Soft assign 1 designer with 2 quads against 3 typologies → sums to 1."""
    quads = [
        IndividualQuad("D1", "Q3", "A", {"a"}, "forces_transition_to",
                       "Strong+", "K", "K2", "high"),
        IndividualQuad("D1", "Q4", "B", {"b"}, "compliantly_follows",
                       "Weak+", "V", "V3", "high"),
    ]
    typos = [
        Typology("Q3A-1", "Q3", "A", "C1", {"a"}, "forces_transition_to",
                 "Strong+", "K", "K2", "5/5"),
        Typology("Q3A-2", "Q3", "A", "C1", {"x"}, "compliantly_follows",
                 "Weak+", "P", "P1", "2/5"),
        Typology("Q4B-1", "Q4", "B", "C2", {"b"}, "compliantly_follows",
                 "Weak+", "V", "V3", "8/8"),
    ]
    probs = soft_assign_designer(quads, typos, temperature=0.5)
    assert sum(probs.values()) == pytest.approx(1.0)
    # Q3A-1 should rank high (perfect match for q1)
    # Q4B-1 should also rank high (perfect match for q2)
    # Q3A-2 should rank lower (mismatch with q1)
    top = max(probs, key=probs.get)
    assert top in ("Q3A-1", "Q4B-1")
    assert probs["Q3A-2"] < probs["Q3A-1"]


def test_soft_assign_restrict_same_qid():
    """When restrict_to_same_qid=True, a Q3 quad shouldn't influence Q4 typology score."""
    q3_quad = IndividualQuad("D1", "Q3", "A", {"a"}, "forces_transition_to",
                              "Strong+", "K", "K2", "high")
    typos = [
        Typology("Q3A-1", "Q3", "A", "C1", {"a"}, "forces_transition_to",
                 "Strong+", "K", "K2", "5/5"),
        Typology("Q4A-1", "Q4", "A", "C1", {"a"}, "forces_transition_to",
                 "Strong+", "K", "K2", "5/5"),  # identical content but Q4
    ]
    # Only Q3 quad provided → Q4A-1 typology should get score=0 (or not appear)
    probs = soft_assign_designer([q3_quad], typos, restrict_to_same_qid=True)
    # Q3A-1 gets full match, Q4A-1 gets nothing (filtered by qid)
    assert "Q3A-1" in probs
    assert "Q4A-1" not in probs or probs.get("Q4A-1", 0) == 0
    assert probs["Q3A-1"] == pytest.approx(1.0)


# ============================================================
# integration: real data
# ============================================================

REAL_QUADS = Path("data_v3/processed/quad_extracted_163_fixed.jsonl")
REAL_TYPO = Path("data/processed/expert_review_quad_candidates_v1.csv")


@pytest.mark.skipif(not REAL_QUADS.exists(), reason="real quad data missing")
def test_load_real_individual_quads():
    iq = load_individual_quads(REAL_QUADS)
    assert len(iq) > 600, f"expected ~648, got {len(iq)}"
    # 163 designers, 4 quads each ish
    designers = {q.designer_id for q in iq}
    assert 150 <= len(designers) <= 165
    qids = {q.qid for q in iq}
    assert qids == {"Q3", "Q4", "Q5", "Q6"}


@pytest.mark.skipif(not REAL_TYPO.exists(), reason="typology candidates missing")
def test_load_real_typology_candidates():
    typo = load_typology_candidates(REAL_TYPO, only_accepted=False)
    assert len(typo) >= 20  # 28 candidates
    # parse_s_summary should yield non-empty for most
    n_with_s = sum(1 for t in typo if t.s_items)
    assert n_with_s >= len(typo) * 0.8


@pytest.mark.skipif(
    not (REAL_QUADS.exists() and REAL_TYPO.exists()),
    reason="real data missing",
)
def test_full_assignment_sanity():
    """End-to-end: 163 designer → 28 typology candidate distribution."""
    iq = load_individual_quads(REAL_QUADS)
    typo = load_typology_candidates(REAL_TYPO, only_accepted=False)
    # group by designer + pick first 3 designers as smoke
    from collections import defaultdict
    by_d = defaultdict(list)
    for q in iq:
        by_d[q.designer_id].append(q)
    sampled = list(by_d.items())[:3]
    for did, quads in sampled:
        probs = soft_assign_designer(quads, typo, temperature=0.3)
        assert probs, f"designer {did} got empty distribution"
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
        top_prob = max(probs.values())
        assert 0 < top_prob <= 1.0
