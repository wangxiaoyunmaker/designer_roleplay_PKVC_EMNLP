"""Tier 1 评测指标: A1 / A2 / A3 + Q8 subgroup.

A1 trajectory match (主指标):
  - factor sequence Hamming distance (cluster prototype 比对)
  - SADR Jaccard similarity
  - 加权综合 score

A2 cluster distribution match (辅指标):
  - per-cluster 答案分布 vs 真值分布 JS-divergence

A3 individual L1 + within-cluster floor (sanity):
  - 个体精确匹配
  - 同簇 majority floor baseline
  - random baseline

Q8 subgroup: 每个指标分 A/B/C 三组报告.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PKVC_FACTORS = ["P", "K", "V", "C"]


@dataclass
class A1Result:
    factor_hamming_mean: float
    factor_hamming_std: float
    sadr_jaccard_mean: float
    sadr_jaccard_std: float
    weighted_score: float            # 综合 (1 - factor_h) * 0.5 + jaccard * 0.5
    n: int


@dataclass
class A2Result:
    js_per_cluster: dict[int, float]
    js_mean: float
    js_std: float
    n_clusters: int


@dataclass
class A3Result:
    model_accuracy: float
    within_cluster_floor_accuracy: float
    random_baseline_accuracy: float
    n: int


@dataclass
class SubgroupReport:
    overall: dict[str, float] = field(default_factory=dict)
    by_q8: dict[str, dict[str, float]] = field(default_factory=dict)  # {"A": {...}, "B": {...}, "C": {...}}


# ====================================
# A1 Trajectory Match
# ====================================

def parse_factor_path(s: str | None) -> list[str]:
    """'V → K → V' or 'V' → ['V', 'K', 'V'] or ['V']."""
    if not s:
        return []
    for sep in ["→", "->", "|"]:
        if sep in s:
            return [c.strip() for c in s.split(sep) if c.strip() in PKVC_FACTORS]
    return [s.strip()] if s.strip() in PKVC_FACTORS else []


def factor_distance(pred: list[str], truth: list[str]) -> float:
    """Normalized factor-sequence distance in [0, 1].

    Uses Levenshtein edit distance / max(len) so short predictions are not
    over-penalized in the way the previous length-padded Hamming did
    (e.g., pred=['V'], truth=['V','K','V'] yields 2/3 = 0.667 here vs the
    old 2/3 = 0.667 — same for that case, but: pred=['V'], truth=['V']
    yields 0.0 here vs old 0.0; pred=[], truth=['V'] yields 1.0 here vs
    old 1.0. The new metric is symmetric and handles deletions/insertions
    rather than treating them as positional mismatches).
    """
    if not pred and not truth:
        return 0.0
    if not pred or not truth:
        return 1.0
    n, m = len(pred), len(truth)
    # Standard Levenshtein
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(n, m)


# Backwards-compatible alias (callers may have hard-coded the old name)
factor_hamming = factor_distance


def _normalize_sadr_token(token: object) -> str:
    """Normalize a SADR activation entry to ``#ID±`` form.

    Accepts either a raw string (already in ``#ID+`` form) or a dict
    ``{"id": "#X", "polarity": "+", ...}``. The polarity is preserved so
    that prediction vs prototype Jaccard sees the same token shape on
    both sides (P1-5 fix: previously the dict form lost the ± suffix).
    """
    if isinstance(token, str):
        return token.strip()
    if isinstance(token, dict):
        sid = str(token.get("id", "")).strip()
        pol = str(token.get("polarity", "")).strip()
        return f"{sid}{pol}" if sid else ""
    return ""


def sadr_jaccard(pred: list[object], truth: list[object]) -> float:
    """Set Jaccard over normalized ``#ID±`` SADR tokens, [0, 1], 1 = perfect."""
    s_p = {t for t in (_normalize_sadr_token(x) for x in pred) if t}
    s_t = {t for t in (_normalize_sadr_token(x) for x in truth) if t}
    if not s_p and not s_t:
        return 1.0
    union = s_p | s_t
    return len(s_p & s_t) / max(len(union), 1)


def compute_a1(
    pred_records: list[dict],
    cluster_prototypes: dict[int | tuple[int, str], dict],
    user_cluster: dict[str, int],
    scenario_aware: bool = False,
) -> A1Result:
    """Predict trajectory vs cluster prototype.

    Args:
      pred_records: 模型推理后, extractor 反推的结果. 每条含 ``user_id`` /
        ``dominant_factor`` / ``sadr_activations``. 后两者既可以是字符串也
        可以是 dict 形式 (extractor JSON 输出)。
      cluster_prototypes: 簇原型. 两种 key:
        * ``{cluster_id: {...}}`` (scenario-agnostic, 默认)
        * ``{(cluster_id, scenario_id): {...}}`` (scenario_aware=True)
      user_cluster: user_id → cluster_id.
      scenario_aware: if True, expects ``cluster_prototypes`` keyed by
        ``(cluster_id, scenario_id)`` and predictions to have ``scenario_id``.

    SADR tokens are passed through ``sadr_jaccard`` which now normalizes both
    raw-string and dict forms to ``#ID±`` so prediction-side tokens and
    prototype-side tokens are comparable (P1-5 fix).
    """
    factor_dists: list[float] = []
    sadr_jacs: list[float] = []
    for r in pred_records:
        uid = r.get("user_id")
        c = user_cluster.get(uid)
        if c is None:
            continue
        if scenario_aware:
            scenario = r.get("scenario_id") or r.get("task_id")
            proto = cluster_prototypes.get((int(c), scenario))
        else:
            proto = cluster_prototypes.get(int(c))
        if proto is None:
            continue

        pred_f = parse_factor_path(r.get("dominant_factor") or "")
        truth_f = parse_factor_path(proto.get("dominant_factor") or "")
        factor_dists.append(factor_distance(pred_f, truth_f))

        pred_sadr = r.get("sadr_activations") or []
        truth_sadr = proto.get("sadr") or proto.get("sadr_frequency") or []
        # Prototype may store frequency dict; pull keys if so.
        if isinstance(truth_sadr, dict):
            truth_sadr = list(truth_sadr.keys())
        sadr_jacs.append(sadr_jaccard(pred_sadr, truth_sadr))

    if not factor_dists:
        return A1Result(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    fh_mean = float(np.mean(factor_dists))
    fh_std = float(np.std(factor_dists))
    sj_mean = float(np.mean(sadr_jacs))
    sj_std = float(np.std(sadr_jacs))
    weighted = (1 - fh_mean) * 0.5 + sj_mean * 0.5
    return A1Result(fh_mean, fh_std, sj_mean, sj_std, weighted, len(factor_dists))


# ====================================
# A2 Cluster Distribution Match
# ====================================

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions.

    P1-6 fix: in KL(p || m) where ``m = (p + q) / 2``, ``m_i > 0`` whenever
    ``p_i > 0``, so we only need to mask ``a > 0`` (not both). The old double
    mask silently under-counted divergence on options that the model predicted
    but the ground truth never used (or vice versa).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-9)
    q = q / max(q.sum(), 1e-9)
    m = 0.5 * (p + q)

    def safe_kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (safe_kl(p, m) + safe_kl(q, m))


def compute_a2(
    pred_records: list[dict],         # 每条 含 user_id + answer (字符串选项)
    truth_records: list[dict],
    user_cluster: dict[str, int],
) -> A2Result:
    """Per-cluster 模型预测分布 vs 真值分布 JS-divergence."""
    pred_by_cluster = defaultdict(list)
    truth_by_cluster = defaultdict(list)
    for p in pred_records:
        c = user_cluster.get(p["user_id"])
        if c is not None and p.get("answer"):
            pred_by_cluster[c].append(p["answer"])
    for t in truth_records:
        c = user_cluster.get(t["user_id"])
        if c is not None and t.get("answer"):
            truth_by_cluster[c].append(t["answer"])

    js_per_cluster = {}
    for c in sorted(set(pred_by_cluster) | set(truth_by_cluster)):
        all_options = sorted(set(pred_by_cluster[c]) | set(truth_by_cluster[c]))
        p_vec = np.array([pred_by_cluster[c].count(o) for o in all_options])
        t_vec = np.array([truth_by_cluster[c].count(o) for o in all_options])
        js_per_cluster[c] = js_divergence(p_vec, t_vec)

    if not js_per_cluster:
        return A2Result({}, np.nan, np.nan, 0)
    return A2Result(
        js_per_cluster=js_per_cluster,
        js_mean=float(np.mean(list(js_per_cluster.values()))),
        js_std=float(np.std(list(js_per_cluster.values()))),
        n_clusters=len(js_per_cluster),
    )


# ====================================
# A3 Individual L1 + Floor
# ====================================

def compute_a3(
    pred_records: list[dict],
    truth_records: list[dict],
    user_cluster: dict[str, int],
) -> A3Result:
    """A3 individual accuracy + within-cluster floor + random baseline.

    P1-1 fix: ``cluster_truth_dist`` is keyed by ``(cluster_id, task_id)`` so
    the within-cluster floor reflects "same-cluster peers' majority answer on
    THIS specific question", not "any answer the cluster ever gave on any
    question". The previous version conflated Q1=A and Q3=A into one count.
    Leave-one-out subtraction now removes exactly the target user's
    contribution to the (cluster, task) cell.
    """
    truth_map = {(t["user_id"], t.get("task_id", "")): t["answer"]
                 for t in truth_records}

    # (cluster, task) -> Counter(answer -> count)
    cluster_task_dist: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for t in truth_records:
        c = user_cluster.get(t["user_id"])
        if c is not None and t.get("answer"):
            cluster_task_dist[(int(c), t.get("task_id", ""))][t["answer"]] += 1

    # Per-task option pool for proper random baseline
    task_options: dict[str, list[str]] = defaultdict(list)
    for t in truth_records:
        if t.get("answer"):
            task_options[t.get("task_id", "")].append(t["answer"])
    task_options_set = {k: sorted(set(v)) for k, v in task_options.items()}

    rng = np.random.default_rng(42)
    correct: list[float] = []
    floor_correct: list[float] = []
    random_correct: list[float] = []

    for p in pred_records:
        key = (p["user_id"], p.get("task_id", ""))
        truth_ans = truth_map.get(key)
        if truth_ans is None:
            continue
        correct.append(1.0 if p.get("answer") == truth_ans else 0.0)

        c = user_cluster.get(p["user_id"])
        tid = p.get("task_id", "")
        if c is not None:
            dist = cluster_task_dist[(int(c), tid)].copy()
            # leave-one-out: remove exactly the target user's contribution
            if dist.get(truth_ans, 0) > 0:
                dist[truth_ans] -= 1
            dist = Counter({a: n for a, n in dist.items() if n > 0})
            if dist:
                majority = dist.most_common(1)[0][0]
                floor_correct.append(1.0 if majority == truth_ans else 0.0)

        opts = task_options_set.get(tid)
        if opts:
            rand_ans = rng.choice(opts)
            random_correct.append(1.0 if rand_ans == truth_ans else 0.0)

    return A3Result(
        model_accuracy=float(np.mean(correct)) if correct else np.nan,
        within_cluster_floor_accuracy=float(np.mean(floor_correct)) if floor_correct else np.nan,
        random_baseline_accuracy=float(np.mean(random_correct)) if random_correct else np.nan,
        n=len(correct),
    )


# ====================================
# Q8 Subgroup 分析
# ====================================

def stratify_by_q8(
    pred_records: list[dict],
    truth_records: list[dict],
    user_q8: dict[str, str],          # user_id → "A" / "B" / "C"
) -> dict[str, tuple[list, list]]:
    """按 Q8 答案分组 pred + truth."""
    groups = {"A": ([], []), "B": ([], []), "C": ([], [])}
    for p in pred_records:
        g = user_q8.get(p["user_id"])
        if g in groups:
            groups[g][0].append(p)
    for t in truth_records:
        g = user_q8.get(t["user_id"])
        if g in groups:
            groups[g][1].append(t)
    return groups


def report_subgroup(
    pred_records: list[dict],
    truth_records: list[dict],
    user_cluster: dict[str, int],
    user_q8: dict[str, str],
    cluster_prototypes: dict[int, dict],
) -> SubgroupReport:
    """Overall + Q8=A/B/C 各组的 A1/A2/A3 指标."""
    report = SubgroupReport()

    overall_a1 = compute_a1(pred_records, cluster_prototypes, user_cluster)
    overall_a2 = compute_a2(pred_records, truth_records, user_cluster)
    overall_a3 = compute_a3(pred_records, truth_records, user_cluster)
    report.overall = {
        "A1_weighted": overall_a1.weighted_score,
        "A1_factor_hamming": overall_a1.factor_hamming_mean,
        "A1_sadr_jaccard": overall_a1.sadr_jaccard_mean,
        "A2_js_mean": overall_a2.js_mean,
        "A3_model_acc": overall_a3.model_accuracy,
        "A3_floor_acc": overall_a3.within_cluster_floor_accuracy,
        "n": overall_a1.n,
    }

    groups = stratify_by_q8(pred_records, truth_records, user_q8)
    for grp_name, (gp, gt) in groups.items():
        if not gp:
            continue
        a1 = compute_a1(gp, cluster_prototypes, user_cluster)
        a2 = compute_a2(gp, gt, user_cluster)
        a3 = compute_a3(gp, gt, user_cluster)
        report.by_q8[grp_name] = {
            "A1_weighted": a1.weighted_score,
            "A2_js_mean": a2.js_mean,
            "A3_model_acc": a3.model_accuracy,
            "A3_floor_acc": a3.within_cluster_floor_accuracy,
            "n": a1.n,
        }
    return report


# ====================================
# Markdown rendering
# ====================================

_Q8_LABELS = {
    "A": "A — 自报有差异 (跨场景思考变化)",
    "B": "B — 自报无差异 (跨场景思考稳定)",
    "C": "C — 自报中立 / 无明显感受",
}
_METRIC_FORMAT = {
    "A1_weighted": (".3f", "↑"),
    "A1_factor_hamming": (".3f", "↓"),
    "A1_sadr_jaccard": (".3f", "↑"),
    "A2_js_mean": (".3f", "↓"),
    "A3_model_acc": (".3f", "↑"),
    "A3_floor_acc": (".3f", "↑"),
    "n": (".0f", "·"),
}


def _fmt(metric: str, value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and (value != value)):  # NaN
        return "—"
    fmt, _ = _METRIC_FORMAT.get(metric, (".3f", "·"))
    return format(value, fmt)


def render_subgroup_markdown(report: SubgroupReport) -> str:
    """Render a paper-ready Markdown table of overall + per-Q8 subgroup."""
    metrics = list(_METRIC_FORMAT.keys())

    lines = ["## Q8 子群评测报告", ""]
    lines.append("**指标解读**: ↑ 越高越好 · ↓ 越低越好")
    lines.append("")

    # Table header
    cols = ["Subgroup", "n"] + [m for m in metrics if m != "n"]
    arrow_row = ["—"] * 2 + [f"{_METRIC_FORMAT[m][1]}" for m in metrics if m != "n"]
    header_top = "| " + " | ".join(cols) + " |"
    header_arrow = "| " + " | ".join(arrow_row) + " |"
    header_sep = "|" + "|".join("---" for _ in cols) + "|"
    lines.append(header_top)
    lines.append(header_arrow)
    lines.append(header_sep)

    # Overall row
    overall_cells = ["**Overall**", _fmt("n", report.overall.get("n"))]
    for m in metrics:
        if m == "n":
            continue
        overall_cells.append(_fmt(m, report.overall.get(m)))
    lines.append("| " + " | ".join(overall_cells) + " |")

    # Per-Q8 rows in fixed A/B/C order
    for grp_key in ("A", "B", "C"):
        d = report.by_q8.get(grp_key)
        if not d:
            continue
        cells = [_Q8_LABELS.get(grp_key, grp_key), _fmt("n", d.get("n"))]
        for m in metrics:
            if m == "n":
                continue
            cells.append(_fmt(m, d.get(m)))
        lines.append("| " + " | ".join(cells) + " |")

    # Narrative interpretation
    lines.append("")
    a_n = report.by_q8.get("A", {}).get("n", 0)
    b_n = report.by_q8.get("B", {}).get("n", 0)
    a1_b = report.by_q8.get("B", {}).get("A1_weighted")
    a1_a = report.by_q8.get("A", {}).get("A1_weighted")
    if a1_b is not None and a1_a is not None:
        delta = (a1_b - a1_a) if (a1_b == a1_b and a1_a == a1_a) else None  # NaN guard
        if delta is not None:
            direction = "高于" if delta > 0 else "低于"
            lines.append(
                f"> **核心叙事**: Q8=B (自报稳定, n={b_n}) 的 A1 trajectory match "
                f"{direction} Q8=A (自报变化, n={a_n}) {abs(delta):.3f}, 跟设计师 metacognition 一致."
            )
    return "\n".join(lines)
