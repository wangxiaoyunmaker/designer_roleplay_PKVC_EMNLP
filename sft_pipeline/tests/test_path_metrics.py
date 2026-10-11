"""Unit tests for path fidelity 4 sub-metrics (jaccard / norm_edit / transition_jacc / lcs_f1).

Run: .venv/bin/python -m pytest sft_pipeline/tests/test_path_metrics.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import pytest
from eval_baselines_5metric import (
    transition_set, lcs_length, levenshtein, path_fidelity,
)


# --- transition_set ---

def test_transition_set_basic():
    assert transition_set(["V", "K", "P", "V"]) == {("V","K"), ("K","P"), ("P","V")}
    assert transition_set(["V"]) == set()  # single state, no transitions
    assert transition_set([]) == set()


def test_transition_set_repeated():
    # consecutive same letters
    assert transition_set(["V", "V", "K"]) == {("V","V"), ("V","K")}


# --- lcs_length ---

def test_lcs_length_basic():
    assert lcs_length(["V","K","P"], ["V","K","P"]) == 3
    assert lcs_length(["V","K","P"], ["V","P"]) == 2
    assert lcs_length(["V","K","P"], ["K"]) == 1
    assert lcs_length(["V","K"], ["P","C"]) == 0
    assert lcs_length([], ["V"]) == 0


def test_lcs_length_order_preserved():
    # LCS preserves order
    assert lcs_length(["V","K","P","V"], ["P","K","V"]) == 2  # K,V or V,V
    assert lcs_length(["A","B","C"], ["C","B","A"]) == 1


# --- path_fidelity full ---

def test_path_fidelity_exact_match():
    r = path_fidelity(["V","K","P","V"], ["V","K","P","V"])
    assert r["jaccard"] == 1.0
    assert r["norm_edit"] == 1.0
    assert r["transition_jacc"] == 1.0
    assert r["lcs_f1"] == 1.0


def test_path_fidelity_complete_mismatch():
    r = path_fidelity(["V","K"], ["P","C"])
    assert r["jaccard"] == 0.0
    assert r["lcs_f1"] == 0.0
    # edit: max=2, edit=2, norm = 0
    assert r["norm_edit"] == 0.0


def test_path_fidelity_order_reversal():
    """V→K→P 跟 P→K→V: state set 相同 (Jaccard=1), 但顺序逆 (transition_jacc=0, lcs<1)."""
    r = path_fidelity(["V","K","P"], ["P","K","V"])
    assert r["jaccard"] == 1.0       # set {V,K,P} same
    # transition: pred {(V,K),(K,P)}, gt {(P,K),(K,V)} — 完全不重合
    assert r["transition_jacc"] == 0.0
    # lcs: longest preserving order = "K" or any single, len=1; f1 = 2/6 = 0.333
    assert r["lcs_f1"] == pytest.approx(2 / 6)
    # edit: V→P, K=K, P→V = 2 substitutions; norm = 1 - 2/3 ≈ 0.333
    assert r["norm_edit"] == pytest.approx(1 - 2/3)


def test_path_fidelity_subset():
    """pred=[V] gt=[V,K,P,V]: pred 是 gt 子集."""
    r = path_fidelity(["V"], ["V","K","P","V"])
    # jaccard: {V} ∩ {V,K,P} = 1, ∪ = 3 → 1/3
    assert r["jaccard"] == pytest.approx(1/3)
    # transition: pred has 0 transitions, gt has 3 → 0/3 = 0
    assert r["transition_jacc"] == 0.0
    # lcs: V matches, len=1; f1 = 2 / (1+4) = 0.4
    assert r["lcs_f1"] == pytest.approx(2 / 5)
    # edit: insert 3 → norm = 1 - 3/4 = 0.25
    assert r["norm_edit"] == pytest.approx(0.25)


def test_path_fidelity_single_state_match():
    """Both length 1, same letter — transition fallback: trans_jacc=1.0."""
    r = path_fidelity(["V"], ["V"])
    assert r["jaccard"] == 1.0
    assert r["transition_jacc"] == 1.0  # fallback path
    assert r["lcs_f1"] == 1.0


def test_path_fidelity_single_state_diff():
    """Both length 1, different — transition fallback: trans_jacc=0.0."""
    r = path_fidelity(["V"], ["K"])
    assert r["jaccard"] == 0.0
    assert r["transition_jacc"] == 0.0  # fallback
    assert r["lcs_f1"] == 0.0


def test_path_fidelity_partial_transitions():
    """pred=[V,K,P,V] gt=[V,K,V,P]
    pred trans = {(V,K),(K,P),(P,V)}, gt trans = {(V,K),(K,V),(V,P)}
    intersect = {(V,K)} = 1, union = 5
    transition_jacc = 1/5 = 0.2
    STA (recall over gt bigrams) = 1/3 (只 (V,K) 命中 3 个 gt bigram 之一)
    """
    r = path_fidelity(["V","K","P","V"], ["V","K","V","P"])
    assert r["transition_jacc"] == pytest.approx(1/5)
    assert r["sta"] == pytest.approx(1/3)


# --- TED ---

def test_ted_exact_match():
    r = path_fidelity(["V","K","P"], ["V","K","P"])
    assert r["ted"] == 0


def test_ted_single_substitution():
    # V→K differs at pos 0
    r = path_fidelity(["V","K","P"], ["K","K","P"])
    assert r["ted"] == 1


def test_ted_insert_delete():
    # gt has 1 extra char
    r = path_fidelity(["V","K"], ["V","K","P"])
    assert r["ted"] == 1


def test_ted_complete_replace():
    r = path_fidelity(["V","K"], ["P","C"])
    assert r["ted"] == 2


# --- STA ---

def test_sta_perfect_recall():
    # gt bigrams all present in pred
    r = path_fidelity(["V","K","P"], ["V","K","P"])
    # gt bigrams = {(V,K),(K,P)}, pred bigrams = same → recall = 2/2 = 1.0
    assert r["sta"] == 1.0


def test_sta_zero_recall():
    # pred bigrams disjoint from gt bigrams
    r = path_fidelity(["P","C"], ["V","K","P"])
    # gt = {(V,K),(K,P)}, pred = {(P,C)} → 0 ∩ gt = 0 → 0/2
    assert r["sta"] == 0.0


def test_sta_partial_recall():
    # gt has 3 bigrams, pred recovers 1
    # gt = [V,K,P,V] → {(V,K),(K,P),(P,V)}
    # pred = [V,K,X,Y] → wait letters limited to PKVC
    # Let pred = [V,K,V] → {(V,K),(K,V)}; intersection w/ gt = {(V,K)} = 1, gt count = 3
    r = path_fidelity(["V","K","V"], ["V","K","P","V"])
    assert r["sta"] == pytest.approx(1/3)


def test_sta_gt_length_1():
    # gt = [V] (no bigrams). STA fall back to sequence-equality.
    r1 = path_fidelity(["V"], ["V"])
    assert r1["sta"] == 1.0
    r2 = path_fidelity(["K"], ["V"])
    assert r2["sta"] == 0.0
    r3 = path_fidelity(["V","K"], ["V"])
    assert r3["sta"] == 0.0  # gt empty bigrams → fall back, but sequences不等
