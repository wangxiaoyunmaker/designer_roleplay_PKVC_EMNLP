"""Eval pipeline smoke test (CPU, < 60s, no API key, no SFT ckpt).

跑通整套 Tier 1 投稿评测在 mock 模式下不挂:
  - predictions.py: mock 4 档 RAG predictions + regex 反推
  - ablation.py: phase0_test_a + phase0_test_b + rag_ablation_nway
  - baselines.py: B0_random + B0_prime fallback + B3 + B4
  - metrics.py: A1/A2/A3 + Q8 subgroup
  - report.py: Tier 1 markdown 渲染
  - runner.py: --full-tier1 端到端

跑法:
  .venv/bin/python -m sft_pipeline.tests.test_eval_smoke

不跑 GPU / 不调 API / 不需要 SFT ckpt; 只验工程接口正确.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test_eval_smoke")


# ---------------------------------------------------------------------------
# Mock data fabricators (不依赖 data/ 真 CSV, 让 smoke 在任何环境跑)
# ---------------------------------------------------------------------------

def fabricate_user_cluster(n_users: int = 60, n_clusters: int = 4) -> dict[str, int]:
    rng = np.random.default_rng(42)
    return {f"u{i:03d}": int(rng.integers(0, n_clusters)) for i in range(n_users)}


def fabricate_truth_records(user_cluster: dict[str, int],
                              questions=("Q1", "Q3", "Q4", "Q5", "Q6"),
                              options=("A", "B", "C", "D", "E")) -> list[dict]:
    """每 user 每题给一个 cluster-biased answer (簇内有偏好)."""
    rng = np.random.default_rng(42)
    cluster_pref = {c: rng.choice(options, size=len(questions)) for c in set(user_cluster.values())}
    out = []
    for uid, c in user_cluster.items():
        for qi, q in enumerate(questions):
            # 70% 跟簇偏好, 30% 随机
            if rng.random() < 0.7:
                ans = cluster_pref[c][qi]
            else:
                ans = rng.choice(options)
            out.append({"user_id": uid, "task_id": q, "answer": str(ans)})
    return out


# ---------------------------------------------------------------------------
# Test 1: predictions.py — mock + regex 反推
# ---------------------------------------------------------------------------

def test_predictions_module() -> None:
    log.info("=== (1) predictions.py: mock + regex 反推 ===")
    from sft_pipeline.eval.predictions import (
        build_mock_predictions, parse_response_regex, build_trajectory_vec,
        canonical_rag_mode, RAG_MODES_V351,
    )

    # alias 漂移防御
    assert canonical_rag_mode("trait_similarity") == "cosine"
    assert canonical_rag_mode("og_rag+modulation") == "og_rag+modulation"
    log.info("  ✓ canonical_rag_mode 别名映射 OK")

    user_cluster = fabricate_user_cluster()
    truth = fabricate_truth_records(user_cluster)
    preds = build_mock_predictions(user_cluster, truth, rag_mode="og_rag", seed=42, flip_rate=0.3)
    assert len(preds) == len(truth), f"mock 数量错: {len(preds)} vs {len(truth)}"
    assert all(p.answer in "ABCDE" for p in preds), "mock answer 不在 A-E"
    assert all(p.rag_mode == "og_rag" for p in preds), "rag_mode 标错"
    log.info("  ✓ mock predictions: n=%d, mode=og_rag", len(preds))

    # regex 反推
    ans, fac, sadrs = parse_response_regex(
        "[PKVC 分析] 主导因子: V → K → V | sadr: #13+, #14-, #服务保留-\n[答案]: B"
    )
    assert ans == "B", f"answer 解析错: {ans}"
    assert fac and "V" in fac, f"factor 解析错: {fac}"
    assert "#13+" in sadrs, f"sadr 解析错: {sadrs}"
    log.info("  ✓ regex 反推: answer=%s factor=%s sadr=%s", ans, fac, sadrs)

    # trajectory vec
    vec_recs = build_trajectory_vec(preds)
    assert len(vec_recs) == len(set(p.user_id for p in preds)), "vec 数 ≠ user 数"
    assert all("trajectory_vec" in r for r in vec_recs)
    log.info("  ✓ trajectory vec: %d users, dim=%d",
              len(vec_recs), len(vec_recs[0]["trajectory_vec"]))

    # 全 4 档 mode
    for mode in RAG_MODES_V351:
        ps = build_mock_predictions(user_cluster, truth, rag_mode=mode, seed=42, flip_rate=0.25)
        assert len(ps) > 0
    log.info("  ✓ 4 档 mode 全部 mockable")


# ---------------------------------------------------------------------------
# Test 2: ablation 4 档 RAG ablation
# ---------------------------------------------------------------------------

def test_rag_ablation_nway() -> None:
    log.info("=== (2) ablation.rag_ablation_nway 4 档 ===")
    from sft_pipeline.eval.ablation import rag_ablation_nway

    # 模拟 4 档 score (og_rag+modulation 比 og_rag 高 1.5%, 应该 PROMOTE)
    fake_preds_by_mode = {
        "no_rag": list(range(10)),
        "cosine": list(range(10)),
        "og_rag": list(range(10)),
        "og_rag+modulation": list(range(10)),
    }
    fake_scores = {
        "no_rag": 0.40,
        "cosine": 0.45,
        "og_rag": 0.50,
        "og_rag+modulation": 0.515,
    }
    res = rag_ablation_nway(
        fake_preds_by_mode,
        metric_fn=lambda preds: fake_scores[next(m for m, p in fake_preds_by_mode.items() if p is preds)],
    )
    assert res.rag_helps_at_all, "RAG 应该 helps (0.515 - 0.40 > 1%)"
    assert res.mapping_promotes_main_claim, "mapping 应该 PROMOTE (0.515 - 0.50 > 1%)"
    assert res.ranking[0][0] == "og_rag+modulation", f"排序错: {res.ranking}"
    log.info("  ✓ ranking: %s", [(m, f"{s:.3f}") for m, s in res.ranking])
    log.info("  ✓ mapping_delta=%.4f, promote=%s", res.mapping_delta, res.mapping_promotes_main_claim)

    # 边界: mapping 增益不到 1%
    fake_scores["og_rag+modulation"] = 0.505
    res2 = rag_ablation_nway(
        fake_preds_by_mode,
        metric_fn=lambda preds: fake_scores[next(m for m, p in fake_preds_by_mode.items() if p is preds)],
    )
    assert not res2.mapping_promotes_main_claim, "mapping 不应 PROMOTE (Δ=0.005 < 1%)"
    log.info("  ✓ 边界 case (Δ < 1%) 正确判 NOT PROMOTE")


# ---------------------------------------------------------------------------
# Test 3: baselines B0_random / B0_prime fallback / B3 / B4
# ---------------------------------------------------------------------------

def test_baselines_offline() -> None:
    log.info("=== (3) baselines: B0_random + B0_prime fallback + B3 + B4 ===")
    from sft_pipeline.eval.baselines import (
        baseline_b0_random_uniform, baseline_b0_strong_prompt,
        baseline_b3_knn, baseline_b4_gbm,
    )
    user_cluster = fabricate_user_cluster()
    truth = fabricate_truth_records(user_cluster)
    truth_df = pd.DataFrame(truth)

    # 假 30 维 profile
    rng = np.random.default_rng(42)
    profile_mat = pd.DataFrame(
        rng.normal(size=(len(user_cluster), 30)),
        index=list(user_cluster.keys()),
        columns=[f"f{i}" for i in range(30)],
    )

    # B0_random
    preds_b0r = baseline_b0_random_uniform(list(user_cluster.keys())[:10], truth_df)
    assert len(preds_b0r) > 0 and preds_b0r[0].method == "B0_random_uniform"
    log.info("  ✓ B0_random: %d preds", len(preds_b0r))

    # B0_prime fallback (no API key)
    preds_b0p = baseline_b0_strong_prompt(
        list(user_cluster.keys())[:10], truth_df,
        user_cluster=user_cluster, fallback_mode=True,
    )
    assert len(preds_b0p) > 0 and "cluster_majority" in preds_b0p[0].method
    log.info("  ✓ B0_prime fallback: %d preds (cluster majority)", len(preds_b0p))

    # B3 KNN
    train_uids = list(user_cluster.keys())[:40]
    test_uids = list(user_cluster.keys())[40:]
    train_df = truth_df[truth_df["user_id"].isin(train_uids)]
    preds_b3 = baseline_b3_knn(
        train_profile=profile_mat.loc[train_uids],
        train_answers=train_df,
        train_cluster=user_cluster,
        test_users=test_uids,
        test_profile=profile_mat.loc[test_uids],
        test_cluster=user_cluster, k=3,
    )
    assert len(preds_b3) > 0
    log.info("  ✓ B3 KNN: %d preds", len(preds_b3))

    # B4 GBM
    preds_b4 = baseline_b4_gbm(
        train_profile=profile_mat.loc[train_uids],
        train_answers=train_df,
        test_users=test_uids,
        test_profile=profile_mat.loc[test_uids],
    )
    assert len(preds_b4) > 0
    log.info("  ✓ B4 GBM: %d preds", len(preds_b4))


# ---------------------------------------------------------------------------
# Test 4: report.py 渲染不挂 (即使 missing 段全 None)
# ---------------------------------------------------------------------------

def test_report_render_resilient() -> None:
    log.info("=== (4) report.render_tier1_report 缺段安全降级 ===")
    from sft_pipeline.eval.report import render_tier1_report

    # 全空, 应该全部走 — / skipped 但不抛异常
    md_empty = render_tier1_report()
    assert "Tier 1 评测报告" in md_empty
    assert "skipped" in md_empty or "—" in md_empty
    log.info("  ✓ 空输入 render OK (%d 字符)", len(md_empty))

    # 主指标 + Phase 0 部分填
    md_partial = render_tier1_report(
        phase0={
            "test_a": {"mean_ratio": 1.15, "cramers_v_mean": 0.18,
                       "permutation_p_value": 0.003, "n_permutations": 1000,
                       "passes": True, "explanation": "PASS"},
            "test_b": {"accuracy": 0.62, "chance_level": 0.25,
                       "passes": True, "n_samples": 200, "n_clusters": 4,
                       "explanation": "PASS"},
        },
        metrics_by_mode={
            "no_rag": {"A1_weighted": 0.42, "A1_factor_hamming": 0.55,
                       "A1_sadr_jaccard": 0.41, "A2_js_mean": 0.18,
                       "A3_model_acc": 0.51, "A3_floor_acc": 0.48, "n": 195},
            "og_rag+modulation": {"A1_weighted": 0.58, "A1_factor_hamming": 0.42,
                                  "A1_sadr_jaccard": 0.55, "A2_js_mean": 0.12,
                                  "A3_model_acc": 0.62, "A3_floor_acc": 0.48, "n": 195},
        },
        direction="q1q6",
    )
    assert "no_rag" in md_partial and "og_rag+modulation" in md_partial
    assert "Phase 0" in md_partial
    log.info("  ✓ 部分填充 render OK (%d 字符)", len(md_partial))


# ---------------------------------------------------------------------------
# Test 5: 端到端 --full-tier1 mock 跑通 (用 fabricated data, 不读 data/)
# ---------------------------------------------------------------------------

def test_full_tier1_end_to_end_mock() -> None:
    """走完整 run_full_tier1 流程, 用 fabricated CSV 不依赖真数据."""
    log.info("=== (5) end-to-end: --full-tier1 mock 跑通 ===")
    from sft_pipeline.eval.runner import run_full_tier1

    user_cluster = fabricate_user_cluster(n_users=40, n_clusters=4)
    truth = fabricate_truth_records(user_cluster)

    # 临时 CSV 模拟 data/ 目录
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "raw").mkdir()
        users_df = pd.DataFrame({"wechat_id": list(user_cluster.keys()),
                                  "identity": ["设计师"] * len(user_cluster)})
        users_df.to_csv(tdp / "raw" / "users.csv", index=False)

        tasks_rows = []
        for t in truth:
            tasks_rows.append({"user_id": t["user_id"], "question_id": t["task_id"],
                                "answer_value": f"{t['answer']}. 选项文本",
                                "text_supplement": ""})
        # 加一些 Q8
        for uid in list(user_cluster.keys())[:30]:
            tasks_rows.append({"user_id": uid, "question_id": "Q8",
                                "answer_value": np.random.default_rng(hash(uid) % 99).choice(["A. 有差异", "B. 无差异", "C. 中立"]),
                                "text_supplement": ""})
        pd.DataFrame(tasks_rows).to_csv(tdp / "raw" / "task_responses.csv", index=False)

        # 30 维 scale (4 因子 × 7 题 + 2 填充)
        scale_rows = []
        for uid in user_cluster.keys():
            for q_idx in range(30):
                scale_rows.append({"user_id": uid,
                                    "question_node": f"q{q_idx:02d}",
                                    "score": int(np.random.default_rng(hash((uid, q_idx)) % 999).integers(1, 7))})
        pd.DataFrame(scale_rows).to_csv(tdp / "raw" / "scale_responses.csv", index=False)

        cluster_df = pd.DataFrame({"user_id": list(user_cluster.keys()),
                                    "cluster_id": list(user_cluster.values())})
        cluster_df.to_csv(tdp / "cluster.csv", index=False)

        cfg = {
            "data": {"raw_dir": str(tdp / "raw"),
                     "cluster_file": str(tdp / "cluster.csv"),
                     "rag": {"k": 3}},
            "cv": {"n_folds": 4, "fold_seed": 42},
        }

        from sft_pipeline.eval.runner import load_data
        data = load_data(cfg)

        report_md = run_full_tier1(
            cfg, data,
            pred_dir=None,                                    # 强制 mock
            direction="q1q6",
            mock_predictions=True,
            include_baselines=("B0_random", "B0_prime", "B3", "B4"),
        )
        assert "Tier 1 评测报告" in report_md
        assert "Phase 0" in report_md
        assert "RAG mode" in report_md
        assert "mapping" in report_md.lower()
        log.info("  ✓ Tier 1 markdown render OK (%d 字符)", len(report_md))

        # 落盘验证 IO 路径
        out_path = tdp / "report_smoke.md"
        out_path.write_text(report_md, encoding="utf-8")
        assert out_path.exists() and out_path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_predictions_module,
        test_rag_ablation_nway,
        test_baselines_offline,
        test_report_render_resilient,
        test_full_tier1_end_to_end_mock,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            log.exception("FAIL: %s — %s", t.__name__, e)
            failed.append(t.__name__)
    print("")
    if failed:
        print(f"=== {len(failed)} / {len(tests)} TESTS FAILED: {failed} ===")
        return 1
    print(f"=== ALL {len(tests)} EVAL SMOKE TESTS PASSED ✓ ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
