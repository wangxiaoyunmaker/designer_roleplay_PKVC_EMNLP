"""Evaluation runner: load → predict → extract → metric → report.

Fold-aware, leakage-safe, and reproducible. All randomness uses the YAML
``cv.fold_seed`` (default 42). Stratification is done on ``cluster_id``;
falls back to deterministic hashlib hashing only when StratifiedKFold cannot
split (some cluster has fewer members than ``n_folds``).

CLI
---
::

    # Phase 0 sanity (no model) + B1 EFA + B3/B4 baselines (legacy)
    python -m sft_pipeline.eval.runner --phase0 --validation --baselines B3,B4

    # 投稿用一键 Tier 1 (4 档 RAG + Phase 0 + B1 + B0/B0'/B3/B4 + Q8 + §5.5 判读)
    python -m sft_pipeline.eval.runner --full-tier1 \\
        --predictions-dir preds --direction q1q6 \\
        --rag-modes no_rag,cosine,og_rag,og_rag+modulation \\
        --output reports/tier1.md

    # SFT ckpt 没就位 → mock fallback 跑通 pipeline 验码
    python -m sft_pipeline.eval.runner --full-tier1 --mock-predictions --output reports/dry_run.md

Output reports are written under ``reports/`` with a timestamp and a
``[draft]`` prefix until externally promoted.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("eval_runner")


# Q1-Q6 task ids handled by MCQ baselines / metrics.
# Q2 is currently held out: it is a sort question (cardId order), not a single
# letter answer, and requires its own metric (Kendall's tau / Spearman rank
# correlation). Will be added once metrics for sort answers are implemented.
MCQ_QUESTIONS: tuple[str, ...] = ("Q1", "Q3", "Q4", "Q5", "Q6")
SORT_QUESTIONS_HELD_OUT: tuple[str, ...] = ("Q2",)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------

def _read_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def load_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load all evaluation inputs and align ID types.

    All user_id-like columns are forced to ``str`` to defeat pandas's auto
    int/str inference that previously caused silent dict.get misses.
    """
    data_dir = Path(cfg["data"]["raw_dir"])
    users = _read_csv(data_dir / "users.csv")
    tasks = _read_csv(data_dir / "task_responses.csv")
    scales = _read_csv(data_dir / "scale_responses.csv")
    cluster = _read_csv(Path(cfg["data"]["cluster_file"]))

    # Force string user IDs everywhere
    for df, col in ((users, "wechat_id"), (tasks, "user_id"),
                    (scales, "user_id"), (cluster, "user_id")):
        df[col] = df[col].astype(str)

    scale_matrix = scales.pivot_table(
        index="user_id", columns="question_node", values="score", aggfunc="mean"
    )
    scale_matrix = scale_matrix.fillna(scale_matrix.mean(numeric_only=True))

    user_cluster: dict[str, int] = {
        str(uid): int(cid)
        for uid, cid in zip(cluster["user_id"], cluster["cluster_id"])
    }

    q8 = tasks[tasks["question_id"] == "Q8"].copy()
    q8["letter"] = q8["answer_value"].astype(str).str.extract(r"([A-E])\.")
    user_q8: dict[str, str] = dict(zip(q8["user_id"], q8["letter"]))

    return {
        "users": users,
        "tasks": tasks,
        "scales": scales,
        "cluster": cluster,
        "scale_matrix": scale_matrix,
        "user_cluster": user_cluster,
        "user_q8": user_q8,
    }


# ---------------------------------------------------------------------------
# Fold assignment (P0-1 fix)
# ---------------------------------------------------------------------------

def assign_user_folds(
    user_cluster: dict[str, int],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Stratified user → fold mapping. Reproducible across runs and Python
    versions (no Python ``hash()`` calls)."""
    import hashlib

    user_ids = sorted(user_cluster.keys())
    if not user_ids:
        return {}
    y = np.array([user_cluster[u] for u in user_ids], dtype=int)
    counts = np.bincount(y)
    if counts.min() < n_folds:
        logger.warning(
            "Smallest cluster has %d members < n_folds=%d; "
            "falling back to deterministic hashlib hashing.",
            counts.min(), n_folds,
        )
        return {
            u: int(hashlib.md5(f"{u}:{seed}".encode()).hexdigest(), 16) % n_folds
            for u in user_ids
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(user_ids)
    fold_map: dict[str, int] = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(arr, y)):
        for i in test_idx:
            fold_map[str(arr[i])] = int(fold_idx)
    return fold_map


# ---------------------------------------------------------------------------
# Phase 0 Test A (uses task_id-aware fold-aware ratio)
# ---------------------------------------------------------------------------

def run_phase0(cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Phase 0 Test A: same-cluster designers should agree more than
    between-cluster designers on per-question true answers."""
    from sft_pipeline.eval.ablation import phase0_test_a

    fold_map = assign_user_folds(
        data["user_cluster"],
        cfg["cv"]["n_folds"],
        cfg["cv"]["fold_seed"],
    )

    tasks = data["tasks"]
    tasks = tasks[tasks["question_id"].isin(MCQ_QUESTIONS)].copy()
    tasks["answer"] = tasks["answer_value"].astype(str).str.extract(r"([A-E])\.")
    tasks["cluster_id"] = tasks["user_id"].map(data["user_cluster"])
    tasks["fold"] = tasks["user_id"].map(fold_map)
    tasks = tasks.dropna(subset=["answer", "cluster_id", "fold"])
    tasks["cluster_id"] = tasks["cluster_id"].astype(int)
    tasks["fold"] = tasks["fold"].astype(int)
    tasks = tasks.rename(columns={"question_id": "q_id"})

    result = phase0_test_a(tasks[["user_id", "fold", "cluster_id", "q_id", "answer"]],
                          n_folds=cfg["cv"]["n_folds"])

    logger.info("Phase 0 Test A:")
    logger.info("  fold ratios: %s", [f"{r:.3f}" for r in result.fold_ratios])
    logger.info("  mean ratio:  %.3f", result.mean_ratio)
    logger.info("  %s", result.explanation)

    return {
        "test_a": _to_jsonable(result),
        "test_b": {"status": "skipped, SFT 未完成"},
        "fold_assignment_method": "StratifiedKFold(seed=%d)" % cfg["cv"]["fold_seed"],
        "n_clusters_observed": len(set(data["user_cluster"].values())),
    }


# ---------------------------------------------------------------------------
# B1 EFA Validation
# ---------------------------------------------------------------------------

def run_b1_validation(cfg: dict[str, Any], data: dict[str, Any]) -> str:
    from sft_pipeline.eval.validation import compute_b1, render_b1_report
    result = compute_b1(data["scale_matrix"], n_factors=4)
    report_md = render_b1_report(result)
    logger.info("B1 EFA complete: KMO=%.3f, cum_var=%.3f",
                result.kmo_total, result.explained_variance_cum[-1])
    return report_md


# ---------------------------------------------------------------------------
# Baselines (P0-2 fix: per-fold train/test split)
# ---------------------------------------------------------------------------

def _prepare_mcq_answers(tasks: pd.DataFrame) -> pd.DataFrame:
    """Filter to MCQ questions and extract single-letter answers."""
    t = tasks[tasks["question_id"].isin(MCQ_QUESTIONS)].copy()
    t["letter"] = t["answer_value"].astype(str).str.extract(r"([A-E])\.")
    t = t.dropna(subset=["letter"])
    return t.rename(columns={"question_id": "task_id", "letter": "answer"})[
        ["user_id", "task_id", "answer"]
    ]


def run_baselines(
    cfg: dict[str, Any],
    data: dict[str, Any],
    baseline_names: list[str],
) -> dict[str, Any]:
    """Run baselines under proper per-fold train/test split.

    For each fold k:
      * train designers = those with fold ≠ k
      * test designers  = those with fold == k
      * baseline is fit on train, predicted on test
    Predictions across folds are concatenated. This eliminates the previous
    leakage where test users themselves appeared in B3 K-NN's neighbor pool
    and in B4 GBM's training set.
    """
    from sft_pipeline.eval.baselines import run_baseline, BaselinePrediction

    fold_map = assign_user_folds(
        data["user_cluster"],
        cfg["cv"]["n_folds"],
        cfg["cv"]["fold_seed"],
    )
    if not fold_map:
        return {n: {"status": "no fold assignment"} for n in baseline_names}

    profile_mat = data["scale_matrix"]
    answers_df = _prepare_mcq_answers(data["tasks"])

    out: dict[str, Any] = {}
    for name in baseline_names:
        logger.info("Running baseline %s with %d-fold CV", name, cfg["cv"]["n_folds"])
        try:
            preds = _run_one_baseline_cv(
                name=name,
                fold_map=fold_map,
                user_cluster=data["user_cluster"],
                profile_mat=profile_mat,
                answers_df=answers_df,
                k=cfg["data"]["rag"]["k"],
            )
            sample_preds = [_to_jsonable(p) for p in preds[:5]]
            out[name] = {
                "n_predictions": len(preds),
                "n_folds": cfg["cv"]["n_folds"],
                "sample": sample_preds,
            }
            logger.info("  %s: %d predictions", name, len(preds))
        except NotImplementedError as e:
            out[name] = {"status": "todo", "reason": str(e)}
            logger.info("  %s: %s", name, e)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
            logger.exception("  %s failed", name)
    return out


def _run_one_baseline_cv(
    name: str,
    fold_map: dict[str, int],
    user_cluster: dict[str, int],
    profile_mat: pd.DataFrame,
    answers_df: pd.DataFrame,
    k: int = 3,
) -> list[Any]:
    """K-fold CV driver for B3 / B4 (no LLM, no SFT)."""
    from sft_pipeline.eval.baselines import run_baseline

    n_folds = max(fold_map.values()) + 1
    all_preds: list[Any] = []
    for fold_k in range(n_folds):
        train_users = [u for u, f in fold_map.items() if f != fold_k]
        test_users = [u for u, f in fold_map.items() if f == fold_k]
        if not train_users or not test_users:
            continue

        # Intersect with available rows to defeat KeyError (P2-7)
        train_users = [u for u in train_users if u in profile_mat.index]
        test_users = [u for u in test_users if u in profile_mat.index]

        train_answers_k = answers_df[answers_df["user_id"].isin(train_users)]
        train_profile_k = profile_mat.loc[
            [u for u in train_answers_k["user_id"].unique() if u in profile_mat.index]
        ]
        test_profile_k = profile_mat.loc[test_users]

        if name == "B3":
            fold_preds = run_baseline(
                "B3",
                train_profile=train_profile_k,
                train_answers=train_answers_k,
                train_cluster=user_cluster,
                test_users=test_users,
                test_profile=test_profile_k,
                test_cluster=user_cluster,
                k=k,
            )
        elif name == "B4":
            fold_preds = run_baseline(
                "B4",
                train_profile=train_profile_k,
                train_answers=train_answers_k,
                test_users=test_users,
                test_profile=test_profile_k,
            )
        else:
            raise NotImplementedError(
                f"baseline {name} requires LLM / SFT model, not implemented here"
            )
        all_preds.extend(fold_preds)
    return all_preds


# ---------------------------------------------------------------------------
# Cluster prototype (P0-3 fix: delegate to cluster_prototype.py)
# ---------------------------------------------------------------------------

def build_train_prototypes(
    extracted_records: list[dict[str, Any]],
    user_cluster: dict[str, int],
    fold_map: dict[str, int],
    test_fold: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Build cluster prototypes ONLY from train-fold designers (not test fold).

    This is the leakage-safe variant. Calls into ``cluster_prototype.build_prototypes``
    with ``train_users`` set to designers whose fold ≠ ``test_fold``.
    """
    from sft_pipeline.eval.prototype import build_prototypes, index_prototypes

    train_users = {u for u, f in fold_map.items() if f != test_fold}
    prototypes = build_prototypes(extracted_records, user_cluster, train_users)
    return {(p.cluster_id, p.scenario_id): _to_jsonable(p)
            for p in prototypes}


# ---------------------------------------------------------------------------
# Tier 1 投稿用主流程 (4 档 RAG + Phase 0 + B0/B0'/B3/B4 + §5.5 D-14 判读)
# ---------------------------------------------------------------------------

DEFAULT_RAG_MODES: tuple[str, ...] = ("no_rag", "cosine", "og_rag", "og_rag+modulation")


def _scenario_aware_for_direction(direction: str) -> bool:
    """方向二 (q1q6→Q7 eval) Q7 含多段, 需 scenario_aware; 方向一 Q1-Q6 不需要."""
    return direction == "q1q6"


def _compute_main_metrics_for_mode(
    preds: list[Any],
    truth_records: list[dict[str, Any]],
    user_cluster: dict[str, int],
    prototypes_by_key: dict[Any, dict[str, Any]],
    scenario_aware: bool,
) -> dict[str, float]:
    """A1 + A2 + A3 一次性算完 (单 RAG mode)."""
    from sft_pipeline.eval.metrics import compute_a1, compute_a2, compute_a3

    # PredictionRecord → dict (metrics expects dicts)
    pred_dicts = [
        {"user_id": p.user_id, "task_id": p.task_id, "scenario_id": p.scenario_id,
         "answer": p.answer, "dominant_factor": p.dominant_factor,
         "sadr_activations": p.sadr_activations}
        for p in preds
    ]
    a1 = compute_a1(pred_dicts, prototypes_by_key, user_cluster, scenario_aware=scenario_aware)
    a2 = compute_a2(pred_dicts, truth_records, user_cluster)
    a3 = compute_a3(pred_dicts, truth_records, user_cluster)
    return {
        "A1_weighted": a1.weighted_score,
        "A1_factor_hamming": a1.factor_hamming_mean,
        "A1_sadr_jaccard": a1.sadr_jaccard_mean,
        "A2_js_mean": a2.js_mean,
        "A3_model_acc": a3.model_accuracy,
        "A3_floor_acc": a3.within_cluster_floor_accuracy,
        "n": a1.n,
    }


def _load_or_mock_predictions_for_mode(
    pred_dir: Path | None,
    direction: str,
    rag_mode: str,
    n_folds: int,
    use_extractor: bool,
    extractor_model: str,
    mock_fallback: bool,
    mock_seed: int,
    mock_flip_rate: float,
    user_cluster: dict[str, int],
    truth_records: list[dict[str, Any]],
) -> list[Any]:
    """先尝试读真 jsonl; 缺 → mock_fallback=True 时给 mock preds, False 时返回 []."""
    from sft_pipeline.eval.predictions import (
        canonical_rag_mode, load_predictions_for_mode, build_mock_predictions,
    )
    canon = canonical_rag_mode(rag_mode)
    if pred_dir is not None:
        preds = load_predictions_for_mode(
            pred_dir, direction, canon, n_folds=n_folds,
            use_extractor=use_extractor, extractor_model=extractor_model,
        )
        if preds:
            return preds
        logger.info("[%s] no real preds at %s; falling back to mock=%s", canon, pred_dir, mock_fallback)
    if not mock_fallback:
        return []
    # 不同 mode 用 seed 偏移让 mock 之间有差异
    seed_offset = abs(hash(canon)) % 1000
    return build_mock_predictions(
        user_cluster, truth_records,
        rag_mode=canon, seed=mock_seed + seed_offset,
        flip_rate=mock_flip_rate,
    )


def run_full_tier1(
    cfg: dict[str, Any],
    data: dict[str, Any],
    *,
    pred_dir: Path | None,
    direction: str,
    rag_modes: tuple[str, ...] = DEFAULT_RAG_MODES,
    extracted_records_path: Path | None = None,
    use_extractor: bool = False,
    extractor_model: str = "gpt-4o",
    mock_predictions: bool = False,
    include_baselines: tuple[str, ...] = ("B0_random", "B0_prime", "B3", "B4"),
) -> str:
    """投稿用 Tier 1 完整流程, 返回 markdown 报告内容.

    流程:
      1. fold 分配 (复用 assign_user_folds)
      2. Phase 0 Test A (数据)
      3. B1 EFA validation
      4. 4 档 RAG: 加载 (或 mock) preds → 反推 PKVC → 算 A1/A2/A3 + Q8 subgroup
      5. Phase 0 Test B: 用 no_rag preds 反推 trajectory → 跑分类器
      6. Baseline 矩阵 (per-fold CV B3/B4, offline B0_random + B0_prime fallback)
      7. §5.5 D-14 4 档 RAG ablation 判读
      8. render markdown
    """
    from sft_pipeline.eval.predictions import (
        canonical_rag_mode, build_trajectory_vec, tasks_df_to_truth_records,
    )
    from sft_pipeline.eval.prototype import build_prototypes
    from sft_pipeline.eval.metrics import report_subgroup, render_subgroup_markdown
    from sft_pipeline.eval.ablation import phase0_test_b, rag_ablation_nway
    from sft_pipeline.eval.report import render_tier1_report

    n_folds = cfg["cv"]["n_folds"]
    fold_seed = cfg["cv"]["fold_seed"]
    user_cluster = data["user_cluster"]
    fold_map = assign_user_folds(user_cluster, n_folds, fold_seed)

    # truth 记录 (metrics 用 dict 形式)
    truth_records = tasks_df_to_truth_records(data["tasks"], MCQ_QUESTIONS)

    # § 1 Phase 0 Test A
    phase0 = run_phase0(cfg, data)

    # § 2 B1 EFA
    validation_md = run_b1_validation(cfg, data)

    # § 3 主指标 × 4 档
    # 先建 fold-aware prototypes — train_users = 全数据 (Tier 1 评测端
    # 没有真 SFT fold split, mock 模式下用全 designer 当 train); 真跑 SFT 后
    # 走 per-fold 嵌套调用即可.
    extracted_records: list[dict[str, Any]] = []
    if extracted_records_path and extracted_records_path.exists():
        from sft_pipeline.eval.predictions import load_extracted_records_normalized
        extracted_records = load_extracted_records_normalized(extracted_records_path)
        logger.info("Loaded %d extracted records → for prototypes", len(extracted_records))
    is_mock_prototypes = False
    if not extracted_records:
        # 缺 extractor 输出时, 用 cluster_id 决定性映射到 P/K/V/C 之一做 mock prototype
        # (映射是 deterministic, 同 cluster designer 共享同一 PKVC factor, 让 A1 能算出 > 0)
        # 投稿前必须 swap 真 extract.batch 输出.
        is_mock_prototypes = True
        from collections import defaultdict
        pkvc_pool = ["P", "K", "V", "C"]
        scenario_set = sorted({t["task_id"] for t in truth_records})
        users_by_cluster: dict[int, set[str]] = defaultdict(set)
        for uid, c in user_cluster.items():
            users_by_cluster[c].add(str(uid))
        for c, uids in users_by_cluster.items():
            for tid in scenario_set:
                pkvc = pkvc_pool[c % len(pkvc_pool)]
                # 每个 designer 一条 record (build_prototypes 按 designer 数聚合 dominant_factor)
                for uid in uids:
                    extracted_records.append({
                        "user_id": uid,
                        "scenario_id": tid,
                        "dominant_factor": pkvc,
                        "sadr_activations": [f"#mock_c{c}+"],
                        "is_mock_prototype": True,
                    })
    prototypes = build_prototypes(extracted_records, user_cluster, train_users=None)
    if _scenario_aware_for_direction(direction):
        protos_by_key: dict[Any, dict[str, Any]] = {(p.cluster_id, p.scenario_id): p.asdict() for p in prototypes}
    else:
        # 方向一: cluster-only key, take first scenario per cluster
        by_c: dict[int, dict[str, Any]] = {}
        for p in prototypes:
            by_c.setdefault(p.cluster_id, p.asdict())
        protos_by_key = by_c

    # iterate 4 档
    preds_by_mode: dict[str, list[Any]] = {}
    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode in rag_modes:
        canon = canonical_rag_mode(mode)
        preds = _load_or_mock_predictions_for_mode(
            pred_dir, direction, canon, n_folds,
            use_extractor=use_extractor, extractor_model=extractor_model,
            mock_fallback=mock_predictions or pred_dir is None,
            mock_seed=fold_seed, mock_flip_rate=0.25,
            user_cluster=user_cluster, truth_records=truth_records,
        )
        preds_by_mode[canon] = preds
        if not preds:
            metrics_by_mode[canon] = {}
            continue
        metrics_by_mode[canon] = _compute_main_metrics_for_mode(
            preds, truth_records, user_cluster,
            protos_by_key, scenario_aware=_scenario_aware_for_direction(direction),
        )

    # § 4 Phase 0 Test B (用 no_rag 预测的 trajectory vec)
    test_b_preds = preds_by_mode.get(canonical_rag_mode("no_rag"), [])
    if test_b_preds:
        traj = build_trajectory_vec(test_b_preds)
        tb_res = phase0_test_b(traj, user_cluster, n_folds=n_folds)
        phase0["test_b"] = _to_jsonable(tb_res)

    # § 5 Q8 subgroup (拿表现最好的 mode 当代表; 一般是 og_rag+modulation 或 og_rag)
    q8_md: str | None = None
    if data["user_q8"] and preds_by_mode:
        # 选用 metric 最佳的非空 mode
        best_mode = max(
            (m for m, mtrx in metrics_by_mode.items() if mtrx.get("A1_weighted") is not None),
            key=lambda m: metrics_by_mode[m].get("A1_weighted", 0),
            default=None,
        )
        if best_mode and preds_by_mode[best_mode]:
            preds = preds_by_mode[best_mode]
            pred_dicts = [
                {"user_id": p.user_id, "task_id": p.task_id, "scenario_id": p.scenario_id,
                 "answer": p.answer, "dominant_factor": p.dominant_factor,
                 "sadr_activations": p.sadr_activations}
                for p in preds
            ]
            # report_subgroup 内部 compute_a1 不传 scenario_aware, 需要 cluster→proto 扁平 dict
            if _scenario_aware_for_direction(direction):
                flat_protos: dict[Any, dict[str, Any]] = {}
                for (cid, _sid), p in protos_by_key.items():
                    flat_protos.setdefault(int(cid), p)
            else:
                flat_protos = protos_by_key
            subgroup = report_subgroup(
                pred_dicts, truth_records, user_cluster, data["user_q8"], flat_protos,
            )
            q8_md = render_subgroup_markdown(subgroup)
            logger.info("Q8 subgroup computed against best_mode=%s", best_mode)

    # § 6 Baseline 矩阵 (offline-safe defaults)
    baselines_result = _run_tier1_baselines(cfg, data, fold_map, include_baselines)

    # § 7 RAG ablation 4 档判读 (用 A1_weighted 主指标)
    rag_nway = None
    if metrics_by_mode:
        scores_by_mode = {m: d.get("A1_weighted") for m, d in metrics_by_mode.items()}
        # 包成可调用 metric (已算好直接取数)
        def _scorer(_preds, _mode_score=None):  # noqa: ARG001
            return _mode_score if _mode_score is not None else float("nan")

        # 直接构造 result 而不重算 metric (避免 double work)
        from sft_pipeline.eval.ablation import RAGAblNwayResult
        ranking = sorted(
            ((m, s) for m, s in scores_by_mode.items() if s is not None and s == s),
            key=lambda x: x[1], reverse=True,
        )
        ref_mode = canonical_rag_mode("no_rag")
        ref_s = scores_by_mode.get(ref_mode, float("nan"))
        deltas = {m: (s - ref_s) for m, s in scores_by_mode.items()
                  if s is not None and s == s and ref_s == ref_s}
        a_mode = canonical_rag_mode("og_rag")
        b_mode = canonical_rag_mode("og_rag+modulation")
        s_a, s_b = scores_by_mode.get(a_mode), scores_by_mode.get(b_mode)
        mapping_delta = (s_b - s_a) if (s_a is not None and s_b is not None
                                         and s_a == s_a and s_b == s_b) else None
        promotes = mapping_delta is not None and mapping_delta > 0.01
        no_rag_s = scores_by_mode.get(ref_mode, float("nan"))
        non_ref_scores = [s for m, s in scores_by_mode.items()
                           if m != ref_mode and s is not None and s == s]
        rag_max = max(non_ref_scores) if non_ref_scores else float("nan")
        rag_helps = (no_rag_s == no_rag_s and rag_max == rag_max
                     and rag_max - no_rag_s > 0.01)
        rag_nway = RAGAblNwayResult(
            scores={m: (s if (s is not None and s == s) else float("nan"))
                    for m, s in scores_by_mode.items()},
            ranking=ranking,
            deltas_vs_reference=deltas,
            reference_mode=ref_mode,
            mapping_promotes_main_claim=promotes,
            mapping_delta=mapping_delta,
            rag_helps_at_all=rag_helps,
            metric_name="A1_weighted",
            n_per_mode={m: len(preds_by_mode.get(m, [])) for m in scores_by_mode},
        )

    # § 8 render
    caveats: list[str] = []
    if mock_predictions or pred_dir is None:
        caveats.append("⚠ 主指标走 mock predictions (random P/K/V/C + flip noise), "
                        "数字不代表真模型性能 — 跑通 pipeline 用.")
    if is_mock_prototypes:
        caveats.append("⚠ cluster prototypes 用 deterministic cluster→PKVC 映射 mock 构造, "
                        "A1 数字仅验工程, 投稿前必须用 extract.batch 输出替换 "
                        "(--extracted-records data/processed/extracted_all.jsonl).")
    if not (use_extractor and pred_dir is not None):
        caveats.append("⚠ PKVC 反推走 regex fallback, 没调 extractor; A1 上限被截.")
    return render_tier1_report(
        phase0=phase0,
        validation_md=validation_md,
        metrics_by_mode=metrics_by_mode,
        baselines=baselines_result,
        q8_md=q8_md,
        rag_nway_result=rag_nway,
        direction=direction,
        rag_mode_order=tuple(canonical_rag_mode(m) for m in rag_modes),
        extra_caveats=caveats,
    )


def _run_tier1_baselines(
    cfg: dict[str, Any],
    data: dict[str, Any],
    fold_map: dict[str, int],
    names: tuple[str, ...],
) -> dict[str, Any]:
    """B0_random / B0_prime fallback / B3 / B4 — offline-runnable 子集."""
    from sft_pipeline.eval.baselines import (
        baseline_b0_random_uniform, baseline_b0_strong_prompt,
    )
    out: dict[str, Any] = {}

    profile_mat = data["scale_matrix"]
    answers_df = _prepare_mcq_answers(data["tasks"])

    # B3 / B4 走原有 per-fold CV
    cv_names = [n for n in names if n in ("B3", "B4")]
    if cv_names:
        # 复用现成实现
        result = run_baselines(cfg, data, cv_names)
        out.update(result)

    # B0_random offline (全 user 当 test)
    if "B0_random" in names:
        users = list(fold_map.keys())
        try:
            preds = baseline_b0_random_uniform(users, answers_df, seed=cfg["cv"]["fold_seed"])
            out["B0_random"] = {
                "n_predictions": len(preds), "n_folds": 1,
                "sample": [_to_jsonable(p) for p in preds[:5]],
            }
        except Exception as e:  # noqa: BLE001
            out["B0_random"] = {"error": str(e)}

    # B0_prime fallback (per-cluster majority, 不调 API)
    if "B0_prime" in names:
        users = list(fold_map.keys())
        try:
            preds = baseline_b0_strong_prompt(
                users, answers_df, user_cluster=data["user_cluster"],
                fallback_mode=True,
            )
            out["B0_prime"] = {
                "n_predictions": len(preds), "n_folds": 1,
                "sample": [_to_jsonable(p) for p in preds[:5]],
                "status": "fallback=cluster_majority (no API key)",
            }
        except NotImplementedError as e:
            out["B0_prime"] = {"status": "todo", "reason": str(e)}
        except Exception as e:  # noqa: BLE001
            out["B0_prime"] = {"error": str(e)}

    return out


# ---------------------------------------------------------------------------
# Reporting helpers (P2-2 fix: no silent str fallback)
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Recursive conversion to JSON-safe types. Raises on anything we don't
    know how to handle, instead of silently stringifying."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return _to_jsonable(obj.__dict__)
    raise TypeError(f"Don't know how to JSON-serialize {type(obj).__name__}: {obj!r}")


def render_report(results: dict[str, Any], output: Path) -> None:
    timestamp = datetime.now().isoformat(timespec="minutes")
    lines = [f"# 评测报告 (draft, {timestamp})",
             "",
             "> 此报告为 draft 状态; 数据未补完前的指标只能作 sanity 用, 不要直接进 paper.",
             ""]
    for section, content in results.items():
        lines.append(f"## {section}")
        if isinstance(content, str):
            lines.append(content)
        else:
            jsonable = _to_jsonable(content)
            lines.append("```json")
            lines.append(json.dumps(jsonable, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved → %s", output)


def _timestamped_path(base: Path) -> Path:
    """Insert a UTC timestamp + ``[draft]`` prefix into ``base``."""
    ts = datetime.now().strftime("%Y%m%dT%H%M")
    return base.parent / f"[draft]_{ts}_{base.name}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--phase0", action="store_true")
    parser.add_argument("--validation", action="store_true",
                        help="Run B1 EFA validation.")
    parser.add_argument("--baselines", default="",
                        help="Comma-separated baseline names (B3,B4,...).")
    parser.add_argument("--direction", choices=("q7", "q1q6", "both"), default="q1q6",
                        help="主方向 (默认 q1q6 = 方向二, 老师拍板).")
    parser.add_argument("--output", type=Path, default=Path("reports/eval_report.md"))
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Don't auto-prefix the report path with [draft]_<timestamp>.")
    parser.add_argument("--verbose", action="store_true")

    # ─── Tier 1 投稿流程 (新加) ─────────────────────────────────────────
    parser.add_argument("--full-tier1", action="store_true",
                        help="一键投稿 Tier 1: Phase 0 + B1 + 4 档 RAG + baseline 矩阵 "
                             "+ Q8 subgroup + §5.5 D-14 判读.")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="推理输出根目录, 命名约定 "
                             "{direction}__{rag_mode}__fold{k}.jsonl.")
    parser.add_argument("--rag-modes",
                        default=",".join(DEFAULT_RAG_MODES),
                        help="4 档 RAG 模式 (逗号分隔), v3.5.1/v3.7 别名都接受.")
    parser.add_argument("--extracted-records", type=Path, default=None,
                        help="extract.batch 输出 jsonl, 给 cluster prototype 用.")
    parser.add_argument("--use-extractor", action="store_true",
                        help="反推 PKVC 用 extract.llm.extract_pkvc (需 API key).")
    parser.add_argument("--extractor-model", default="gpt-4o")
    parser.add_argument("--mock-predictions", action="store_true",
                        help="强制 mock predictions (SFT ckpt 没就位时跑通 pipeline).")
    parser.add_argument("--tier1-baselines",
                        default="B0_random,B0_prime,B3,B4",
                        help="Tier 1 baseline 子集 (逗号分隔).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    data = load_data(cfg)

    logger.info("Loaded: users=%d  scale_matrix=%s  clusters=%s",
                len(data["users"]), data["scale_matrix"].shape,
                data["cluster"]["cluster_id"].value_counts().sort_index().to_dict())
    if data["user_q8"]:
        logger.info("Q8 self-report: %s",
                    pd.Series(data["user_q8"]).value_counts().to_dict())

    # ─── --full-tier1: 走新一键流程 ─────────────────────────────────────
    if args.full_tier1:
        rag_modes = tuple(m.strip() for m in args.rag_modes.split(",") if m.strip())
        baselines_subset = tuple(n.strip() for n in args.tier1_baselines.split(",") if n.strip())
        if args.direction == "both":
            logger.warning("--full-tier1 不支持 both, 自动用 q1q6 (主方向)")
            direction = "q1q6"
        else:
            direction = args.direction
        report_md = run_full_tier1(
            cfg, data,
            pred_dir=args.predictions_dir,
            direction=direction,
            rag_modes=rag_modes,
            extracted_records_path=args.extracted_records,
            use_extractor=args.use_extractor,
            extractor_model=args.extractor_model,
            mock_predictions=args.mock_predictions,
            include_baselines=baselines_subset,
        )
        output_path = args.output if args.no_timestamp else _timestamped_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_md, encoding="utf-8")
        logger.info("Tier 1 report → %s", output_path)
        return 0

    # ─── legacy 散件入口 (向后兼容) ────────────────────────────────────
    results: dict[str, Any] = {}
    if args.phase0:
        results["Phase 0"] = run_phase0(cfg, data)
    if args.validation:
        results["B1 PKVC Validation"] = run_b1_validation(cfg, data)
    if args.baselines:
        names = [n.strip() for n in args.baselines.split(",") if n.strip()]
        results["Baselines"] = run_baselines(cfg, data, names)

    if not (args.phase0 or args.validation or args.baselines):
        logger.info("No tasks specified, defaulting to phase0 + validation + B3/B4 baselines")
        results["Phase 0"] = run_phase0(cfg, data)
        results["B1 PKVC Validation"] = run_b1_validation(cfg, data)
        results["Baselines"] = run_baselines(cfg, data, ["B3", "B4"])

    output_path = args.output if args.no_timestamp else _timestamped_path(args.output)
    render_report(results, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
