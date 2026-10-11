"""Ablation 指标 + Phase 0 sanity:
  D2 factor injection 置空 (主指标置空后输出坍塌测试)
  D4 memorization probe (输入扰动 → 输出变化)
  RAG-abl (有/无 peer RAG 对比)
  Phase 0 Test A (数据侧 cluster behavioral stability)
  Phase 0 Test B (模型侧 cross-scenario PKVC invariance)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ====================================
# Phase 0 Test A — 数据侧
# ====================================

@dataclass
class PhaseATestResult:
    fold_ratios: list[float]
    mean_ratio: float
    cramers_v_per_fold: list[float]     # Cramér's V per (q_id × cluster_id) per fold
    cramers_v_mean: float
    permutation_p_value: float          # null = cluster label is random
    n_permutations: int
    passes: bool                        # p < 0.05 AND ratio > 1
    n_clusters: int
    explanation: str


def _within_between_ratio(
    fold_data: pd.DataFrame,
) -> tuple[float, float]:
    """Return (within_mean, between_mean) of ``1 - majority_share`` heterogeneity.

    * within: per (cluster_id, q_id) majority-share complement, averaged.
    * between: per q_id, mode of each cluster → majority-share complement
      of those modes.
    """
    cluster_var: list[float] = []
    for _, grp in fold_data.groupby("cluster_id"):
        wv = grp.groupby("q_id")["answer"].apply(
            lambda s: 1 - s.value_counts(normalize=True).max()
        ).mean()
        cluster_var.append(float(wv))
    within_mean = float(np.mean(cluster_var)) if cluster_var else float("nan")

    if fold_data["cluster_id"].nunique() < 2:
        return within_mean, float("nan")
    modes = fold_data.groupby(["q_id", "cluster_id"])["answer"].agg(
        lambda s: s.value_counts().idxmax()
    ).reset_index()
    between_mean = float(
        modes.groupby("q_id")["answer"].apply(
            lambda s: 1 - s.value_counts(normalize=True).max()
        ).mean()
    )
    return within_mean, between_mean


def _cramers_v(fold_data: pd.DataFrame) -> float:
    """Cramér's V on the (cluster_id × answer) contingency table aggregated
    across all questions. Bounded in [0, 1]; 0 = independence.
    """
    from scipy.stats import chi2_contingency
    table = pd.crosstab(fold_data["cluster_id"], fold_data["answer"])
    if table.shape[0] < 2 or table.shape[1] < 2 or table.values.sum() == 0:
        return float("nan")
    try:
        chi2, _, _, _ = chi2_contingency(table.values, correction=False)
    except ValueError:
        return float("nan")
    n = table.values.sum()
    k = min(table.shape) - 1
    if k <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * k)))


def phase0_test_a(
    test_truth: pd.DataFrame,
    n_folds: int = 5,
    n_permutations: int = 1000,
    seed: int = 42,
) -> PhaseATestResult:
    """Phase 0 Test A — same-cluster designers should agree on a question
    more than between-cluster designers do.

    P1-7 fix: the previous version reported ``between_var / within_var`` with
    an arbitrary 1.0 PASS threshold. We now augment it with two more rigorous
    pieces of evidence:

    1. **Cramér's V** of (cluster_id × answer) per fold. A standard
       contingency-table effect size in [0, 1]; > 0.1 small, > 0.3 medium,
       > 0.5 large.
    2. **Permutation test** for the global ratio: shuffle cluster_id labels
       across designers ``n_permutations`` times, compute the same ratio
       under that null, and report ``p = P(ratio_null ≥ ratio_observed)``.

    PASS criterion (stricter): observed ratio > 1 AND permutation p < 0.05.
    """
    if test_truth.empty:
        return PhaseATestResult([], float("nan"), [], float("nan"),
                                float("nan"), 0, False,
                                0, "empty input")

    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    cramers_v_list: list[float] = []

    for fold in range(n_folds):
        fd = test_truth[test_truth["fold"] == fold]
        if fd.empty:
            continue
        within_mean, between_mean = _within_between_ratio(fd)
        if not np.isnan(within_mean) and not np.isnan(between_mean):
            ratios.append(between_mean / max(within_mean, 1e-9))
        cramers_v_list.append(_cramers_v(fd))

    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    cramers_v_mean = (
        float(np.nanmean(cramers_v_list)) if cramers_v_list else float("nan")
    )

    # Permutation test on the global (across-fold) ratio.
    user_cluster_table = test_truth.drop_duplicates("user_id")[
        ["user_id", "cluster_id"]
    ].reset_index(drop=True)
    user_ids = user_cluster_table["user_id"].to_numpy()
    cluster_labels = user_cluster_table["cluster_id"].to_numpy()

    base_within, base_between = _within_between_ratio(test_truth)
    base_ratio = base_between / max(base_within, 1e-9)

    null_ratios: list[float] = []
    base_table = test_truth[["user_id", "fold", "q_id", "answer"]].copy()
    for _ in range(n_permutations):
        shuffled = rng.permutation(cluster_labels)
        label_map = dict(zip(user_ids, shuffled))
        permuted = base_table.copy()
        permuted["cluster_id"] = permuted["user_id"].map(label_map)
        w, b = _within_between_ratio(permuted)
        if not (np.isnan(w) or np.isnan(b)):
            null_ratios.append(b / max(w, 1e-9))

    if null_ratios:
        p_value = float(
            (np.sum(np.array(null_ratios) >= base_ratio) + 1)
            / (len(null_ratios) + 1)
        )
    else:
        p_value = float("nan")

    passes = (
        not np.isnan(mean_ratio)
        and mean_ratio > 1
        and not np.isnan(p_value)
        and p_value < 0.05
    )
    expl = (
        f"ratio={mean_ratio:.3f}, Cramér's V={cramers_v_mean:.3f}, "
        f"permutation p={p_value:.4g} (n={len(null_ratios)}); "
        + ("PASS, cluster 在评测端有显著行为差异"
           if passes else "FAIL or borderline, 跨场景 framing 需 reconsider")
    )

    return PhaseATestResult(
        fold_ratios=ratios,
        mean_ratio=mean_ratio,
        cramers_v_per_fold=cramers_v_list,
        cramers_v_mean=cramers_v_mean,
        permutation_p_value=p_value,
        n_permutations=len(null_ratios),
        passes=passes,
        n_clusters=int(test_truth["cluster_id"].nunique()),
        explanation=expl,
    )


# ====================================
# Phase 0 Test B — 模型侧
# ====================================

@dataclass
class PhaseBTestResult:
    accuracy: float
    chance_level: float
    passes: bool                        # acc > 50% (chance 20% for 5 cluster)
    n_clusters: int
    n_samples: int
    explanation: str


def phase0_test_b(
    model_pred_records: list[dict],     # 模型在评测端 推 trajectory: {user_id, factor_path, sadr_vec}
    user_cluster: dict[str, int],
    n_folds: int = 5,
) -> PhaseBTestResult:
    """模型 trajectory 反推 cluster, 看 acc."""
    rows = []
    for r in model_pred_records:
        uid = r["user_id"]
        c = user_cluster.get(uid)
        if c is None:
            continue
        vec = r.get("trajectory_vec")     # 模型输出 trajectory embedding / sadr 频次
        if vec is None:
            continue
        rows.append((uid, c, vec))

    if len(rows) < 10:
        return PhaseBTestResult(np.nan, 0.0, False, 0, 0, "样本不足")

    X = np.array([v for _, _, v in rows])
    y = np.array([c for _, c, _ in rows])
    cv = StratifiedKFold(n_splits=min(n_folds, len(set(y))))
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        if len(set(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]))
    acc = float(np.mean(accs)) if accs else np.nan
    n_c = int(len(set(y)))
    chance = 1 / n_c
    passes = acc > 0.5
    expl = (f"trajectory → cluster classifier acc = {acc:.3f} "
             f"(chance {chance:.3f}); "
             + ("PASS, cross-scenario PKVC 有 invariance" if passes
                else "FAIL, 砍 cross-scenario claim"))
    return PhaseBTestResult(acc, chance, passes, n_c, len(rows), expl)


# ====================================
# D2 Factor Injection
# ====================================

@dataclass
class D2Result:
    flip_rate: float                    # 因子置空后输出反转比例
    passes: bool
    n: int


def d2_factor_injection(
    pred_normal: list[dict],            # 正常推理 {user_id, answer / factor}
    pred_factor_blanked: list[dict],    # PKVC 因子部分被置空 / 随机替换的推理
    threshold: float = 0.5,
) -> D2Result:
    """因子置空后, 输出应有 ≥ threshold 反转率 (说明模型真依赖 PKVC)."""
    normal_map = {r["user_id"]: r.get("answer") for r in pred_normal}
    flip_count = 0
    total = 0
    for r in pred_factor_blanked:
        n_ans = normal_map.get(r["user_id"])
        if n_ans is None or r.get("answer") is None:
            continue
        total += 1
        if r["answer"] != n_ans:
            flip_count += 1
    rate = flip_count / max(total, 1)
    return D2Result(flip_rate=rate, passes=rate >= threshold, n=total)


# ====================================
# D4 Memorization Probe
# ====================================

@dataclass
class D4Result:
    surface_perturb_change_rate: float  # 字面扰动 (无关词替换) 输出变化率
    semantic_perturb_change_rate: float # 语义扰动 (改 PKVC 关键词) 输出变化率
    passes: bool                        # surface 低, semantic 高 → 模型学到的是语义不是记忆
    n: int


def d4_memorization_probe(
    pred_normal: list[dict],
    pred_surface_perturb: list[dict],
    pred_semantic_perturb: list[dict],
) -> D4Result:
    n_map = {r["user_id"]: r.get("answer") for r in pred_normal}
    surf_change, sem_change = 0, 0
    total = 0
    for sp, mp in zip(pred_surface_perturb, pred_semantic_perturb):
        if sp["user_id"] != mp["user_id"]:
            continue
        n_ans = n_map.get(sp["user_id"])
        if n_ans is None:
            continue
        total += 1
        if sp.get("answer") != n_ans:
            surf_change += 1
        if mp.get("answer") != n_ans:
            sem_change += 1
    s_rate = surf_change / max(total, 1)
    m_rate = sem_change / max(total, 1)
    return D4Result(
        surface_perturb_change_rate=s_rate,
        semantic_perturb_change_rate=m_rate,
        passes=(s_rate < 0.2 and m_rate > 0.5),
        n=total,
    )


# ====================================
# RAG-abl
# ====================================

@dataclass
class RAGAblResult:
    with_rag_score: float
    no_rag_score: float
    delta: float                        # with - no
    passes: bool
    n: int


def rag_ablation(
    pred_with_rag: list[dict],
    pred_no_rag: list[dict],
    metric_fn,
) -> RAGAblResult:
    """对比有/无 peer RAG 的主指标 (任意可调用 metric)."""
    with_score = metric_fn(pred_with_rag)
    no_score = metric_fn(pred_no_rag)
    delta = with_score - no_score
    return RAGAblResult(
        with_rag_score=with_score, no_rag_score=no_score,
        delta=delta, passes=delta > 0.05,    # 经验阈值
        n=len(pred_with_rag),
    )


# ====================================
# 4 档 RAG ablation (v3.5.1 §5.5 D-14)
# ====================================

@dataclass
class RAGAblNwayResult:
    """N-mode RAG ablation 结果 (主 4 档对比 + judgment).

    judgment 字段对应 brief §5.5 main-claim 升级判据:
      mapping_promotes_main_claim = (og_rag+modulation) - og_rag > 1%
    """
    scores: dict[str, float]                   # mode_name → metric score
    ranking: list[tuple[str, float]]           # 降序 (高分优先)
    deltas_vs_reference: dict[str, float]      # mode → (mode_score - ref_score)
    reference_mode: str
    mapping_promotes_main_claim: bool          # §5.5 升级判据
    mapping_delta: float | None                # og_rag+mod - og_rag (None if 缺一档)
    rag_helps_at_all: bool                     # max(rag) > no_rag + 1%
    metric_name: str
    n_per_mode: dict[str, int]


def rag_ablation_nway(
    preds_by_mode: dict[str, list[object]],    # mode_name → predictions (任意 record 类型)
    metric_fn,                                  # callable(preds) → float
    metric_name: str = "A1_weighted",
    reference_mode: str = "no_rag",
    main_claim_pair: tuple[str, str] = ("og_rag", "og_rag+modulation"),
    upgrade_threshold: float = 0.01,            # 1% 涨点
) -> RAGAblNwayResult:
    """N 档 (默认 4 档) RAG mode 主指标对比 + §5.5 D-14 判读.

    示例:
        from functools import partial
        from sft_pipeline.eval.metrics import compute_a1
        scorer = lambda preds: compute_a1(preds, protos, user_cluster).weighted_score

        preds_by_mode = {"no_rag": [...], "cosine": [...], "og_rag": [...],
                         "og_rag+modulation": [...]}
        res = rag_ablation_nway(preds_by_mode, scorer, "A1_weighted")
        if res.mapping_promotes_main_claim:
            print("→ 升级 mapping 为 main contribution")
    """
    scores: dict[str, float] = {}
    n_per_mode: dict[str, int] = {}
    for mode, preds in preds_by_mode.items():
        n_per_mode[mode] = len(preds)
        if not preds:
            scores[mode] = float("nan")
            continue
        try:
            scores[mode] = float(metric_fn(preds))
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger("eval.ablation").warning(
                "metric_fn failed on mode=%s: %s", mode, e,
            )
            scores[mode] = float("nan")

    ranking = sorted(
        ((m, s) for m, s in scores.items() if not np.isnan(s)),
        key=lambda x: x[1], reverse=True,
    )
    ref_s = scores.get(reference_mode, float("nan"))
    deltas = {m: (s - ref_s) for m, s in scores.items() if not np.isnan(s) and not np.isnan(ref_s)}

    a, b = main_claim_pair
    s_a, s_b = scores.get(a, float("nan")), scores.get(b, float("nan"))
    mapping_delta = (s_b - s_a) if not (np.isnan(s_a) or np.isnan(s_b)) else None
    promotes = (mapping_delta is not None and mapping_delta > upgrade_threshold)

    no_rag_s = scores.get("no_rag", float("nan"))
    rag_max = max((s for m, s in scores.items() if m != "no_rag" and not np.isnan(s)),
                  default=float("nan"))
    rag_helps = (not np.isnan(no_rag_s) and not np.isnan(rag_max)
                  and rag_max - no_rag_s > upgrade_threshold)

    return RAGAblNwayResult(
        scores=scores,
        ranking=ranking,
        deltas_vs_reference=deltas,
        reference_mode=reference_mode,
        mapping_promotes_main_claim=promotes,
        mapping_delta=mapping_delta,
        rag_helps_at_all=rag_helps,
        metric_name=metric_name,
        n_per_mode=n_per_mode,
    )
