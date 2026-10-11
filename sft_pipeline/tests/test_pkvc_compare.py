"""Tests for sft_pipeline.eval.pkvc_compare — 5 比较模式覆盖."""
from __future__ import annotations

import pytest

from sft_pipeline.eval.pkvc_compare import (
    COMPARATORS,
    exact_match,
    pkvc_jaccard,
    pkvc_sequence_sim,
    same_dominant_factor,
    same_tail_factor,
)


# =====================================================================
# exact_match
# =====================================================================

def test_exact_single():
    assert exact_match("P", "P")
    assert not exact_match("P", "K")


def test_exact_path():
    assert exact_match("K→V", "K→V")
    assert not exact_match("K→V", "V→K")     # 方向不同
    assert not exact_match("K", "K→V")        # 长度不同


def test_exact_ascii_arrow_tolerated():
    assert exact_match("K->V", "K→V")          # ASCII '->' = U+2192


def test_exact_edge_empty():
    assert not exact_match("", "P")
    assert not exact_match(None, "P")  # type: ignore[arg-type]


# =====================================================================
# same_dominant_factor (head)
# =====================================================================

def test_head_single_path_same_first():
    assert same_dominant_factor("P", "P→C")    # head P
    assert same_dominant_factor("K→V", "K")    # head K
    assert not same_dominant_factor("P", "K→P")  # head K, not P


def test_head_path_path_diff_first():
    assert same_dominant_factor("K→V", "K→P")
    assert not same_dominant_factor("K→V", "P→V")


# =====================================================================
# same_tail_factor (tail)
# =====================================================================

def test_tail_single_path_same_last():
    assert same_tail_factor("V", "K→V")        # tail V
    assert not same_tail_factor("V", "V→K")    # tail K


def test_tail_path_path():
    assert same_tail_factor("K→V", "P→V")
    assert not same_tail_factor("K→V", "K→P")


# =====================================================================
# pkvc_jaccard
# =====================================================================

def test_jaccard_identical():
    assert pkvc_jaccard("P", "P") == 1.0
    assert pkvc_jaccard("K→V", "V→K") == 1.0   # 同集合不同方向


def test_jaccard_disjoint():
    assert pkvc_jaccard("P", "K") == 0.0


def test_jaccard_partial():
    assert pkvc_jaccard("K→V", "K→P") == pytest.approx(1 / 3)
    assert pkvc_jaccard("K", "K→V") == 0.5


# =====================================================================
# pkvc_sequence_sim
# =====================================================================

def test_seq_sim_full_match():
    assert pkvc_sequence_sim("P", "P") == pytest.approx(1.0)
    assert pkvc_sequence_sim("K→V", "K→V") == pytest.approx(1.0)


def test_seq_sim_no_match():
    assert pkvc_sequence_sim("P", "K") == 0.0


def test_seq_sim_head_only():
    # K→V vs K→P: head match, tail miss, jaccard 1/3
    s = pkvc_sequence_sim("K→V", "K→P")
    expected = 0.5 * 0 + 0.2 * 1 + 0.2 * 0 + 0.1 * (1 / 3)
    assert s == pytest.approx(expected)


def test_seq_sim_tail_only():
    # K→V vs P→V: head miss, tail match, jaccard 1/3
    s = pkvc_sequence_sim("K→V", "P→V")
    expected = 0.5 * 0 + 0.2 * 0 + 0.2 * 1 + 0.1 * (1 / 3)
    assert s == pytest.approx(expected)


def test_seq_sim_same_set_reverse_direction():
    # K→V vs V→K: 集合同, 方向反, exact/head/tail 都失败, 仅 jaccard=1
    s = pkvc_sequence_sim("K→V", "V→K")
    assert s == pytest.approx(0.1)


def test_seq_sim_single_vs_path_head_match():
    # K vs K→V: head match (parsing K → ["K"], K→V → ["K","V"]; ["K"]==["K","V"]? No
    # so exact=0. head: K==K → 1. tail: K vs V → 0. jaccard: {K}∩{K,V}/{K,V}=1/2
    s = pkvc_sequence_sim("K", "K→V")
    expected = 0.5 * 0 + 0.2 * 1 + 0.2 * 0 + 0.1 * 0.5
    assert s == pytest.approx(expected)


def test_seq_sim_bounds():
    """全部组合 in [0, 1]."""
    factors = ["P", "K", "V", "C", "P→K", "K→V", "V→C", "C→P", "P→C", "K→P"]
    for a in factors:
        for b in factors:
            s = pkvc_sequence_sim(a, b)
            assert 0.0 <= s <= 1.0, f"{a} vs {b} = {s} 超界"


# =====================================================================
# COMPARATORS lookup dict
# =====================================================================

def test_comparators_dict_keys():
    assert set(COMPARATORS.keys()) == {"exact", "head", "tail", "jaccard", "sequence_sim"}


def test_comparators_dict_callable():
    # 每个 callable 接受 (str, str) 返回 bool 或 float
    for name, fn in COMPARATORS.items():
        result = fn("P", "K→P")
        assert isinstance(result, (bool, float, int))
