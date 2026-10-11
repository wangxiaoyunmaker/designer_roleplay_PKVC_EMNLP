"""Tests for sft_pipeline.eval.datav3_stats — math-critical functions only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sft_pipeline.eval.datav3_stats import loaders


# =====================================================================
# Loaders (Task 1)
# =====================================================================

def _write_mini_data(tmp_path: Path) -> Path:
    """Synthetic 3-designer × 30-dim × Q1-Q6 fixture."""
    root = tmp_path / "data_v3_fake"
    root.mkdir()

    users = pd.DataFrame({
        "wechat_id": ["u1", "u2", "u3"],
        "identity": ["pro", "student", "pro"],
        "years_exp": [5, 1, 10],
        "education_stage_current": ["master", "bachelor", "phd"],
        "design_field": ["UX", "graphic", "UX"],
    })
    users.to_csv(root / "users.csv", index=False)

    scale_rows = []
    for uid in ["u1", "u2", "u3"]:
        for q in range(1, 31):
            scale_rows.append({
                "user_id": uid,
                "question_node": f"S{q}",
                "score": (q % 6) + 1,
                "duration_seconds": 5,
                "device_type": "pc",
                "submitted_at": "2026-05-20",
            })
    pd.DataFrame(scale_rows).to_csv(root / "scale_responses.csv", index=False)

    task_rows = []
    for uid in ["u1", "u2", "u3"]:
        for qid in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
            task_rows.append({
                "user_id": uid,
                "question_id": qid,
                "answer_value": "A",
                "text_supplement": "short",
                "duration_seconds": 10,
                "device_type": "pc",
                "submitted_at": "2026-05-20",
            })
    pd.DataFrame(task_rows).to_csv(root / "task_responses.csv", index=False)
    pd.DataFrame(task_rows).to_csv(root / "task_responses_expanded.csv", index=False)
    return root


def test_load_users(tmp_path):
    root = _write_mini_data(tmp_path)
    df = loaders.load_users(root)
    assert len(df) == 3
    assert {"wechat_id", "years_exp", "identity"}.issubset(df.columns)


def test_load_scale_wide(tmp_path):
    root = _write_mini_data(tmp_path)
    df = loaders.load_scale_wide(root)
    assert df.shape == (3, 30), f"expected 3×30, got {df.shape}"
    assert df.index.name == "user_id"


def test_load_task_long(tmp_path):
    root = _write_mini_data(tmp_path)
    df = loaders.load_task_long(root, expanded=True)
    assert len(df) == 18  # 3 × 6
    assert {"user_id", "question_id", "answer_value"}.issubset(df.columns)


def test_load_item_polarity_smoke():
    """Real metadata: docs/notes/PKVC_30节点本体.md → 30 items with +/- labels."""
    pol = loaders.load_item_polarity()
    assert len(pol) == 30, f"expected 30 items, got {len(pol)}"
    # spot-check a known row
    assert "数据 vs 直觉" in pol
    assert pol["数据 vs 直觉"]["plus_label"] == "重直觉立场"
    assert pol["数据 vs 直觉"]["minus_label"] == "重数据立场"


def test_load_q1q6_option_pkvc_smoke():
    """Real metadata: each of Q1/Q3/Q4/Q5/Q6 has 4 options with main_factor in PKVC."""
    opm = loaders.load_q1q6_option_pkvc()
    for qid in ["Q1", "Q3", "Q4", "Q5", "Q6"]:
        assert qid in opm, f"{qid} missing"
        assert len(opm[qid]) == 4, f"{qid} has {len(opm[qid])} options, expected 4"
        for opt, meta in opm[qid].items():
            assert meta["main_factor"] in {"P", "K", "V", "C"}, f"{qid}.{opt} bad factor {meta}"


# =====================================================================
# Mapping (Task 6) — math-critical: permutation test sanity
# =====================================================================

from sft_pipeline.eval.datav3_stats import mapping


def test_permutation_no_effect():
    """Random data → 0 < p < 1 (not crashing, sane bounds)."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(100)
    chosen = rng.integers(0, 2, 100).astype(bool)
    delta, p = mapping.compute_shift_and_pvalue(scores, chosen, n_perm=500, seed=0)
    assert 0.0 < p <= 1.0


def test_permutation_strong_effect():
    """Group A factor=+2, group B factor=-2 → p ≈ 0, |delta| large."""
    rng = np.random.default_rng(1)
    chosen = np.array([True] * 50 + [False] * 50)
    scores = np.concatenate([rng.normal(2.0, 0.5, 50), rng.normal(-2.0, 0.5, 50)])
    delta, p = mapping.compute_shift_and_pvalue(scores, chosen, n_perm=500, seed=1)
    assert abs(delta) > 1.5
    assert p < 0.05


def test_classify_modulation_strength():
    assert mapping.classify_strength(0.05) == "bypass"
    assert mapping.classify_strength(0.3) == "Weak+"
    assert mapping.classify_strength(-0.3) == "Weak-"
    assert mapping.classify_strength(0.9) == "Strong+"
    assert mapping.classify_strength(-0.9) == "Strong-"


# =====================================================================
# Bridge (Task 8) — math-critical: bootstrap + partial-out
# =====================================================================

from sft_pipeline.eval.datav3_stats import bridge


def test_bootstrap_stable_sign():
    """Strong consistent-sign data → stability ≈ 100%."""
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(2.0, 0.5, 60), rng.normal(-2.0, 0.5, 60)])
    chose = np.array([True] * 60 + [False] * 60)
    stab = bridge.bootstrap_sign_stability(scores, chose, n_boot=200, seed=2)
    assert stab > 0.95


def test_bootstrap_strong_above_noise():
    """Strong-signal data 的 stability 必须严格高于 noise 数据.

    单一 noise sample 可能巧合有稳定 sign (sample 不是 'true 0 effect'),
    所以直接测 'noise < 0.5' 不可靠. 改测 'strong > noise + 0.1' 比较稳.
    """
    rng = np.random.default_rng(3)
    # noise
    n_scores = rng.standard_normal(120)
    n_chose = rng.integers(0, 2, 120).astype(bool)
    noise_stab = bridge.bootstrap_sign_stability(n_scores, n_chose, n_boot=200, seed=3)
    # strong signal
    s_scores = np.concatenate([rng.normal(2.0, 0.5, 60), rng.normal(-2.0, 0.5, 60)])
    s_chose = np.array([True] * 60 + [False] * 60)
    strong_stab = bridge.bootstrap_sign_stability(s_scores, s_chose, n_boot=200, seed=3)
    assert strong_stab > noise_stab + 0.1, f"strong={strong_stab}, noise={noise_stab}"
    assert strong_stab > 0.95


def test_partial_out_removes_covariate():
    """Outcome depends only on covariate → partial-out residual effect ≈ 0."""
    rng = np.random.default_rng(4)
    n = 200
    cov = rng.standard_normal(n)
    factor = 0.8 * cov + 0.2 * rng.standard_normal(n)
    chose = cov > 0
    raw_d = bridge._cohens_d_from_groups(factor[chose], factor[~chose])
    partial_d = bridge.partial_out_cohens_d(factor, chose, np.column_stack([cov]))
    assert abs(raw_d) > 0.5
    assert abs(partial_d) < abs(raw_d) * 0.5
