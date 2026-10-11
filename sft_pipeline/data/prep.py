"""Data preparation pipeline: raw CSV -> SFT/eval JSONL.

Pipeline stages
---------------
1. Load raw CSVs (users / task_responses / scale_responses / user_cluster).
2. Filter to designers that have complete data (scale + task).
3. Build per-user profile and parse Q1-Q7 responses (Q8 is metadata only).
4. Convert numerical profile (6-point Likert) into 4-bucket polarity NL list.
5. Run 5-fold stratified CV split by ``cluster_id`` with a hash-based fallback
   when any cluster has fewer members than ``n_folds``.
6. Build per-fold same-cluster peer RAG pool. Two retrieval backends are
   supported: cosine top-k (default, no extra deps) or ontology-grounded
   set-cover (``og_rag``; requires extracted PKVC records).
7. Emit per-fold JSONL files for both training directions:
     - ``q7``     -> per-segment 3-task (B1 / B2 / B3) following brief v3.4 §4A
     - ``q1q6``   -> per-question single-step (Q1, Q3-Q6) following §4B
8. Optionally apply profile-noise and lexical-scenario augmentation.

CLI
---
::

    # Generate both directions for fold 0, default cosine peer RAG
    python data_prep.py --direction both --fold 0 \\
        --output-dir /data/wy/Designer_RolePlay/data/processed

    # Use ontology-grounded RAG (requires extracted_records.jsonl ready)
    python data_prep.py --direction q1q6 --fold 0 \\
        --use-og-rag --extracted-records data/processed/extracted.jsonl

    # Smoke test only (no I/O, prints summary)
    python data_prep.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("data_prep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from 6-point Likert score to 4-bucket polarity label.
LIKERT_TO_POLARITY: dict[int, str] = {
    1: "strong_minus",
    2: "weak_minus",
    3: "weak_minus",
    4: "weak_plus",
    5: "weak_plus",
    6: "strong_plus",
}
POLARITY_NL_LABEL: dict[str, str] = {
    "strong_plus":  "强 +",
    "weak_plus":    "弱 +",
    "weak_minus":   "弱 -",
    "strong_minus": "强 -",
}
POLARITY_ORDER = ("strong_plus", "weak_plus", "weak_minus", "strong_minus")

#: Q1, Q3-Q6 are MCQ; Q2 is sort; Q7 is open NL; Q8 is metadata.
MCQ_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6")
SORT_QUESTION = "Q2"
OPEN_QUESTION = "Q7"
META_QUESTION = "Q8"

#: Alpha-test / pre-launch users to exclude from training, EFA, and clustering.
#: wiwsx19940412 注册于 2026-05-18 08:20 (cohort 第 1 名, 比第 3 名早 2 天),
#: 用老问卷版本 (Q1/Q5 文案不同), Q7 用语音录音但 fileUrl=None 全部上传失败,
#: text_supplement 8 题全空. 整人无可用数据.
EXCLUDED_USERS: frozenset[str] = frozenset({"wiwsx19940412"})

#: Q7 JSON keys → internal task id (per brief v3.4 §4A).
Q7_SEG_TO_TASKID: dict[str, str] = {"seg1": "B1", "seg2": "B2", "seg3": "B3"}
SEG_NAME: dict[str, str] = {"B1": "INTUITION", "B2": "WEIGH", "B3": "BOTTOM_LINE"}
SEG_CN: dict[str, str] = {"B1": "第一反应", "B2": "决策链路", "B3": "核心 tradeoff"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DesignerProfile:
    """Per-designer profile bundle.

    v3.7: 加 factor_scores (4-d EFA factor) + factor_quartile (F1 4 分位) —
    替代 cluster_id 作 conditioning + sampler 主键. cluster_id 保留兼容.
    """
    user_id: str
    cluster_id: int                                       # [DEPRECATED v3.7] cluster 弱区分
    profile_30d_numerical: dict[str, float]
    profile_polarity: dict[str, list[str]]
    profile_polarity_nl: str
    fold: int = -1
    q8_self_report: str | None = None  # A / B / C
    factor_scores: list[float] = field(default_factory=list)   # v3.7: 4-d EFA factor
    factor_quartile: int = -1                                  # v3.7: F1 quartile (0..3)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Q1Q6Response:
    user_id: str
    question_id: str
    answer_letter: str | None
    answer_full_text: str
    explanation_nl: str
    cardid_order: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Q7Response:
    user_id: str
    segments: dict[str, str]  # B1/B2/B3 -> raw NL


@dataclass(slots=True)
class SFTSample:
    """Single training sample written as one JSONL line.

    v3.7: 加 factor_scores + factor_quartile, cluster_id 保留兼容旧 trainer.
    """
    user_id: str
    cluster_id: int                                       # [DEPRECATED v3.7]
    direction: Literal["q7", "q1q6"]
    task_id: str          # B1/B2/B3 or Q1/Q3/Q4/Q5/Q6
    fold: int
    input_text: str
    output_text: str
    meta: dict[str, Any] = field(default_factory=dict)
    factor_scores: list[float] = field(default_factory=list)   # v3.7: 4-d EFA factor
    factor_quartile: int = -1                                  # v3.7: F1 quartile

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_csv(p: Path) -> pd.DataFrame:
    """Read a CSV and strip UTF-8 BOM from column names."""
    df = pd.read_csv(p)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def load_raw_data(
    data_dir: Path,
    cluster_filename: str = "user_cluster_efa.csv",
) -> dict[str, Any]:
    """Load all four source CSVs and identify the set of complete designers.

    A designer is considered complete iff present in BOTH ``users.csv`` and
    ``scale_responses.csv`` (the ``task_responses.csv`` may contain extra IDs
    from incomplete sessions; those are dropped).

    cluster_filename defaults to the 79-pilot file. Switch to
    ``user_cluster_efa_v2_200.csv`` after 200 入袋 + EFA 重跑 (brief v3.5.1 §4C lock).
    """
    users = _read_csv(data_dir / "users.csv")
    tasks = _read_csv(data_dir / "task_responses.csv")
    scales = _read_csv(data_dir / "scale_responses.csv")
    cluster = _read_csv(data_dir / cluster_filename)

    scale_uids = set(scales["user_id"].astype(str).tolist())
    user_uids = set(users["wechat_id"].astype(str).tolist())
    complete_uids = sorted((scale_uids & user_uids) - EXCLUDED_USERS)
    excluded_present = sorted((scale_uids | user_uids) & EXCLUDED_USERS)

    if not complete_uids:
        raise RuntimeError(
            "No complete designers (intersection of users + scale_responses is empty). "
            "Check that user_id and wechat_id fields refer to the same IDs."
        )

    users = users[~users["wechat_id"].astype(str).isin(EXCLUDED_USERS)].copy()

    tasks = tasks.copy()
    tasks["user_id"] = tasks["user_id"].astype(str)
    tasks = tasks[tasks["user_id"].isin(complete_uids)]

    scales = scales.copy()
    scales["user_id"] = scales["user_id"].astype(str)
    scales = scales[scales["user_id"].isin(complete_uids)]

    cluster = cluster.copy()
    cluster["user_id"] = cluster["user_id"].astype(str)
    cluster = cluster[~cluster["user_id"].isin(EXCLUDED_USERS)]

    if excluded_present:
        logger.warning(
            "Excluded %d alpha-test user(s) from cohort: %s",
            len(excluded_present), excluded_present,
        )
    logger.info(
        "Loaded raw data: %d complete designers (scale ∩ users, post-exclusion), "
        "%d task rows, %d scale rows, %d cluster rows",
        len(complete_uids), len(tasks), len(scales), len(cluster),
    )

    return {
        "users": users,
        "tasks": tasks,
        "scales": scales,
        "cluster": cluster,
        "complete_uids": complete_uids,
    }


def build_profile_matrix(scales: pd.DataFrame) -> pd.DataFrame:
    """Pivot ``scale_responses`` long format -> wide ``user × question_node``."""
    matrix = scales.pivot_table(
        index="user_id",
        columns="question_node",
        values="score",
        aggfunc="mean",
    )
    if matrix.isna().any().any():
        fill = matrix.mean(numeric_only=True)
        matrix = matrix.fillna(fill)
        logger.warning("Profile matrix had NaN cells; filled with column means.")
    return matrix


# ---------------------------------------------------------------------------
# Profile polarity conversion
# ---------------------------------------------------------------------------

def numerical_to_polarity(profile_row: pd.Series) -> dict[str, list[str]]:
    """Convert one designer's 30-dim Likert into a 4-bucket polarity dict."""
    out: dict[str, list[str]] = {k: [] for k in POLARITY_ORDER}
    for q_node, score in profile_row.items():
        if pd.isna(score):
            continue
        try:
            bucket = LIKERT_TO_POLARITY[int(round(float(score)))]
        except (KeyError, ValueError):
            logger.warning("Out-of-range Likert score %s on %s", score, q_node)
            continue
        out[bucket].append(str(q_node))
    return out


def polarity_to_nl(polarity: dict[str, list[str]]) -> str:
    """Render polarity dict as the NL list block consumed by SFT prompts."""
    lines = ["[Profile] (4 档极性)"]
    for key in POLARITY_ORDER:
        items = polarity.get(key, [])
        if items:
            lines.append(f"{POLARITY_NL_LABEL[key]}: {', '.join(items)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q1-Q7 parsing
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"[A-E]\.")


def parse_answer_letter(answer: Any) -> str | None:
    """Extract the first single uppercase letter answer from a free-form string."""
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return None
    s = str(answer)
    m = _LETTER_RE.search(s)
    return m.group(0)[0] if m else None


def parse_q2_cardid_order(answer: Any) -> list[str]:
    """Parse Q2 JSON list and return cardId sorted by user-assigned order."""
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return []
    try:
        items = json.loads(str(answer))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    sorted_items = sorted(items, key=lambda x: x.get("order", 0))
    return [str(it.get("cardId", "")) for it in sorted_items if it.get("cardId")]


def parse_q7_segments(answer: Any) -> dict[str, str]:
    """Parse Q7 JSON ``{seg1, seg2, seg3}`` -> ``{B1, B2, B3}`` raw NL."""
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return {}
    try:
        obj = json.loads(str(answer))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str] = {}
    for seg_key, task_id in Q7_SEG_TO_TASKID.items():
        seg = obj.get(seg_key, {})
        if isinstance(seg, dict):
            text = seg.get("text", "")
            if isinstance(text, str) and text.strip():
                out[task_id] = text.strip()
    return out


def build_canonical_option_text(tasks: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Per (qid, letter), pick the most-frequent raw answer_value as canonical.

    Defends against question-text drift across问卷版本 (e.g. Q1 B 选项有 2 个变体
    "我会开始梳理项目中..." vs "我会立刻开始梳理：我们到底是要..." — 同 letter,
    不同 wording). Keying on letter and picking the modal text gives every B-picker
    the same prompt option text downstream, regardless of which问卷版本他们答的.
    """
    from collections import Counter
    counts: dict[tuple[str, str], Counter[str]] = {}
    for row in tasks.itertuples(index=False):
        if row.question_id not in MCQ_QUESTIONS:
            continue
        letter = parse_answer_letter(row.answer_value)
        if letter is None:
            continue
        raw = str(row.answer_value or "")
        counts.setdefault((row.question_id, letter), Counter())[raw] += 1
    return {key: c.most_common(1)[0][0] for key, c in counts.items()}


def collect_designer_responses(
    tasks: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Pivot per-row task_responses into per-designer structured dict."""
    canonical = build_canonical_option_text(tasks)
    by_user: dict[str, dict[str, Any]] = {}
    for row in tasks.itertuples(index=False):
        uid = str(row.user_id)
        qid = row.question_id
        entry = by_user.setdefault(uid, {"q1q6": {}, "q7": {}, "q8": None})

        if qid in MCQ_QUESTIONS:
            letter = parse_answer_letter(row.answer_value)
            canonical_text = canonical.get((qid, letter), str(row.answer_value or ""))
            entry["q1q6"][qid] = Q1Q6Response(
                user_id=uid, question_id=qid,
                answer_letter=letter,
                answer_full_text=canonical_text,
                explanation_nl=str(getattr(row, "text_supplement", "") or ""),
            )
        elif qid == SORT_QUESTION:
            entry["q1q6"][qid] = Q1Q6Response(
                user_id=uid, question_id=qid, answer_letter=None,
                answer_full_text=str(row.answer_value or ""),
                explanation_nl=str(getattr(row, "text_supplement", "") or ""),
                cardid_order=parse_q2_cardid_order(row.answer_value),
            )
        elif qid == OPEN_QUESTION:
            entry["q7"] = parse_q7_segments(row.answer_value)
        elif qid == META_QUESTION:
            entry["q8"] = parse_answer_letter(row.answer_value)
    return by_user


# ---------------------------------------------------------------------------
# 5-fold stratified split
# ---------------------------------------------------------------------------

def assign_folds(
    user_ids: Iterable[str],
    cluster_labels: dict[str, int],
    n_folds: int = 5,
    seed: int = 42,
    stratify_by: str = "trait_quartile",        # v3.7: 'trait_quartile' | 'cluster_id' (legacy) | 'none'
    factor_scores_dict: dict[str, dict[str, float]] | None = None,
) -> dict[str, int]:
    """5-fold stratified split.

    v3.7: 默认按 trait_quartile (F1 4 分位) stratify, 替代 cluster_id (cluster 弱区分).
    Falls back to hash split when stratify key 缺数据或 sample 不足.

    Args:
      stratify_by: 'trait_quartile' (v3.7) / 'cluster_id' (v3.5.1 legacy) / 'none' (hash)
      factor_scores_dict: 仅 stratify_by='trait_quartile' 时用, {uid: {F1: ..., ...}}
    """
    valid = list(user_ids)
    if not valid:
        return {}

    # 选 stratification label
    if stratify_by == "trait_quartile" and factor_scores_dict:
        f1_vals = [(u, float(factor_scores_dict.get(u, {}).get("F1", 0.0))) for u in valid]
        sorted_uids = sorted(f1_vals, key=lambda x: x[1])
        n = len(sorted_uids)
        bounds = [n // 4, n // 2, 3 * n // 4]
        uid_to_label = {}
        for i, (u, _) in enumerate(sorted_uids):
            q = 0
            if i >= bounds[0]: q = 1
            if i >= bounds[1]: q = 2
            if i >= bounds[2]: q = 3
            uid_to_label[u] = q
        y = np.array([uid_to_label[u] for u in valid], dtype=int)
        label_name = "trait_quartile"
    elif stratify_by == "cluster_id":
        # v3.5.1 legacy path
        valid = [u for u in valid if u in cluster_labels]
        if not valid:
            return {}
        y = np.array([cluster_labels[u] for u in valid], dtype=int)
        label_name = "cluster_id"
    else:
        return {u: (hash((u, seed)) & 0x7FFFFFFF) % n_folds for u in valid}

    counts = np.bincount(y)
    if counts.min() < n_folds:
        logger.warning(
            "Smallest %s stratum has %d members < n_folds=%d; using hash fallback.",
            label_name, counts.min(), n_folds,
        )
        return {u: (hash((u, seed)) & 0x7FFFFFFF) % n_folds for u in valid}

    fold_map: dict[str, int] = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(valid)
    for fold_idx, (_, test_idx) in enumerate(skf.split(arr, y)):
        for i in test_idx:
            fold_map[str(arr[i])] = fold_idx
    return fold_map


# ---------------------------------------------------------------------------
# Peer RAG
# ---------------------------------------------------------------------------

def cosine_topk_same_cluster_peers(
    target_uid: str,
    target_cluster: int,
    profile_matrix: pd.DataFrame,
    cluster_labels: dict[str, int],
    candidate_uids: list[str],
    k: int = 3,
) -> list[str]:
    """Cosine top-k same-cluster peers, excluding target itself."""
    same_cluster = [
        u for u in candidate_uids
        if cluster_labels.get(u) == target_cluster
        and u != target_uid
        and u in profile_matrix.index
    ]
    if not same_cluster:
        return []
    if target_uid not in profile_matrix.index:
        return same_cluster[:k]

    target_vec = profile_matrix.loc[target_uid].to_numpy(dtype=float)
    t_norm = np.linalg.norm(target_vec) + 1e-9

    scored: list[tuple[str, float]] = []
    for u in same_cluster:
        v = profile_matrix.loc[u].to_numpy(dtype=float)
        cos = float(np.dot(target_vec, v) / (t_norm * (np.linalg.norm(v) + 1e-9)))
        scored.append((u, cos))
    scored.sort(key=lambda x: -x[1])
    return [u for u, _ in scored[:k]]


def format_cosine_peer_block(
    peer_uids: list[str],
    designer_responses: dict[str, dict[str, Any]],
    profile_polarities: dict[str, dict[str, list[str]]],
    similarities: list[float] | None = None,
    direction: str = "q1q6",
) -> str:
    """Render retrieved peers as a prompt-ready block."""
    if not peer_uids:
        return "[Cluster_peer_examples] (无可用同簇 peer)"

    lines = ["[Cluster_peer_examples] (同簇 peer top-k, profile cosine 排序)"]
    for i, uid in enumerate(peer_uids, 1):
        sim_str = f", 相似度 {similarities[i-1]:.2f}" if similarities else ""
        lines.append(f"\n样例 {i} (designer {uid[:8]}{sim_str}):")
        # Polarity summary
        pol = profile_polarities.get(uid)
        if pol:
            for key in POLARITY_ORDER:
                if pol.get(key):
                    lines.append(
                        f"  {POLARITY_NL_LABEL[key]}: {', '.join(pol[key][:5])}"
                    )
        # Direction-specific exemplar
        resp = designer_responses.get(uid, {})
        if direction == "q7":
            for tid in ("B1", "B2", "B3"):
                seg_text = resp.get("q7", {}).get(tid, "")
                if seg_text:
                    lines.append(
                        f"  [{SEG_NAME[tid]}] {seg_text[:250]}"
                        + ("..." if len(seg_text) > 250 else "")
                    )
        else:
            for qid in ("Q1", "Q3", "Q4", "Q5", "Q6"):
                q = resp.get("q1q6", {}).get(qid)
                if q is None:
                    continue
                lines.append(
                    f"  {qid}: 选 {q.answer_letter} | 解释: {q.explanation_nl[:80]}"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SITUATION = (
    "你扮演设计师 {designer_id}. 阅读 task background + 当前情境 + trait profile +\n"
    "同簇 peer 模式, 输出 PKVC 认知路径、自然语言决策, 以及单字母答案.\n"
    "第一人称推理 50-100 字."
)
SYSTEM_PROMPT_PROFILE = (
    "你扮演设计师 {designer_id}. 阅读 task background 和当前题目, 结合自己的 trait\n"
    "profile 和同簇 peer 模式, 输出 PKVC 认知分析、自然语言决策, 以及单字母答案.\n"
    "第一人称推理 50-100 字."
)
TASK_BACKGROUND_PLACEHOLDER = "(待学姐 ⑤, dataset 常量 50-100 字)"

Q1_QUESTION_TEXT = (
    "你拿到一个新产品设计任务后, 第一反应是什么? 描述你的 PKVC 主导认知和你会采取"
    "的具体决策方向."
)

# Q3-Q6 题干 (不含选项)
Q_TEXT_NO_OPTIONS: dict[str, str] = {
    "Q3": "工程部门拿出数据证明只有'医疗器械般笨重'设计才能达 99% 准确率, 否则只有 79%. "
          "你认为那极其丑陋, 违背了消费级产品的美学. 你会如何决策?",
    "Q4": "公司发现用户脑电波数据极具商业价值, 老板提出借此打造封闭商业生态获取隐私数据. "
          "你深知这游走在道德灰度, 且开源社区呼吁共享. 你下一步会如何推进决策?",
    "Q5": "项目经理引入 AI 自动生成 100 种符合极简标准的界面但毫无人味. 老板觉得这是降本"
          "增效工具, 想淘汰一部分初级设计岗. 面对这种工具冲击, 你的第一反应是?",
    "Q6": "系统效率提升 30% 但用户长期感到焦虑异化, 老板拍板 3 个月内必须按原样上线砍掉"
          "所有人性化舒缓设计. 作为主创, 你最终怎么做?",
}


def _build_trait_profile_section(profile: "DesignerProfile",
                                  q2_cardid: list[str],
                                  s_primary_top5: list[str] | None = None) -> str:
    """β+++ Trait Profile 4 sub-sections."""
    from sft_pipeline.extract.codebook import (
        format_factor_pattern, format_q2_cardid_top3_dual,
    )
    factor_pattern = format_factor_pattern(profile.factor_scores)
    q2_top3 = format_q2_cardid_top3_dual(q2_cardid)
    # s_primary top-5 fallback: 用 4 档 polarity 强 + 的 top 5
    if not s_primary_top5:
        s_plus = profile.profile_polarity.get("strong_plus", [])[:5]
        s_primary_top5 = s_plus
    s_prim_str = ", ".join(s_primary_top5) if s_primary_top5 else "(待)"

    lines = [
        f"[Trait Profile of {profile.user_id}]",
        f"- 因子组合 (s_secondary): {factor_pattern}",
        f"- 高分项 (s_primary top-5): {s_prim_str}",
        f"- Q2 cardId top-3 自报告: {q2_top3}",
        f"- 4 档极性 (legacy):",
        f"    {profile.profile_polarity_nl}",
    ]
    return "\n".join(lines)


def _build_situation_section(qid: str, quad: dict | None) -> str:
    """Q3-Q6 [Situation] dual-form section. quad provides C / C_conflict_axes."""
    from sft_pipeline.extract.codebook import (
        C_LOCAL_NL_ZH, C_CONFLICT_TENSION_NL, get_polarity_type_nl,
    )
    c_tag = (quad or {}).get("C", "(待)")
    c_axes = (quad or {}).get("C_conflict_axes", "")
    c_nl = C_LOCAL_NL_ZH.get(c_tag, c_tag)
    tension = C_CONFLICT_TENSION_NL.get(c_axes, c_axes)
    polarity = get_polarity_type_nl(c_axes)

    lines = [
        f"[Situation]",
        f"情境: {c_nl}",
        f"     (C_local: {c_tag})",
        f"冲突类型: {polarity}",
        f"张力两端: {tension}",
    ]
    return "\n".join(lines)


def _build_peer_quad_safe_section(
    peer_uids: list[str], qid: str,
    quad_by_uid_q: dict[tuple[str, str], dict],
    profiles: dict[str, "DesignerProfile"],
    designer_responses: dict,
) -> str:
    """β+++ Peer section: dual-form quad fields only, 砍 decision NL / answer / reasoning."""
    from sft_pipeline.extract.codebook import (
        format_o_primary_dual, format_o_subtype_dual,
        format_p_surface_dual, format_p_modulation_dual,
        format_sadr_dual, format_factor_pattern,
    )
    if not peer_uids:
        return "[Cluster Peer Examples] (无可用同簇 peer)"

    lines = ["[Cluster Peer Examples] top-3 (s_primary Jaccard 排序)"]
    for i, peer_id in enumerate(peer_uids[:3], 1):
        peer_quad = quad_by_uid_q.get((peer_id, qid))
        peer_profile = profiles.get(peer_id)
        if not peer_profile:
            continue
        # Jaccard fallback: 简化用 cluster 内默认 0.6 placeholder
        jaccard = "(待算)"
        factor_p = format_factor_pattern(peer_profile.factor_scores)

        lines.append(f"peer {i} — designer {peer_id} (Jaccard={jaccard}):")
        lines.append(f"   他的 trait factor: {factor_p}")
        if peer_quad:
            lines.append(f"   {qid} cognitive routing quad:")
            lines.append(f"      主导 (o_primary): {format_o_primary_dual(peer_quad.get('o_primary'))}")
            lines.append(f"      子状态 (o_subtype): {format_o_subtype_dual(peer_quad.get('o_subtype'))}")
            lines.append(f"      决策方式 (p_surface): {format_p_surface_dual(peer_quad.get('p_surface'))}")
            lines.append(f"      调制 (p_modulation): {format_p_modulation_dual(peer_quad.get('p_modulation'))}")
            lines.append(f"      认知激活: {format_sadr_dual(peer_quad.get('sadr_activations'))}")
        else:
            lines.append(f"   {qid} quad: (该 peer 该题 quad 缺)")
        # 暴露 peer answer letter + 决策签名 (短) — cross-designer 不算 leakage,
        # 给 model "peer 选 X → 我倾向 X" 的 inductive bias (78 sample 学不到 abstract
        # trait→letter mapping, 但能学 peer→self copy).
        peer_resp = designer_responses.get(peer_id, {}).get("q1q6", {}).get(qid)
        if peer_resp and getattr(peer_resp, "answer_letter", None):
            decision_short = (peer_resp.answer_full_text or "")[:100]
            lines.append(f"   peer 决策签名: \"{decision_short}\"")
            lines.append(f"   peer 答案: {peer_resp.answer_letter}")
        else:
            lines.append("   (peer 该题 response 缺)")
    return "\n".join(lines)


def _build_pkvc_analysis_section(quad: dict | None, designer_nl: str,
                                  is_q1: bool = False) -> str:
    """β+++ [PKVC 分析] section, dual-form."""
    from sft_pipeline.extract.codebook import (
        C_LOCAL_NL_ZH, format_o_primary_dual, format_o_subtype_dual,
        format_p_surface_dual, format_p_modulation_dual, format_sadr_dual,
    )
    if not quad:
        # Q1 fallback (no quad), placeholder
        return (
            "[PKVC 分析]\n"
            "认知路径: (待 quad)\n"
            "主导 (o_primary): (待)\n"
            "子状态 (o_subtype): (待学姐 ④)\n"
            "认知激活: (待)\n"
            f"推理: {designer_nl or '(待补长解释 NL)'}\n"
            "置信度: 低 (low)\n"
            "[/PKVC 分析]"
        )

    c_tag = quad.get("C", "")
    c_nl = C_LOCAL_NL_ZH.get(c_tag, c_tag)
    conf = quad.get("confidence", "low")
    conf_nl = {"high": "高", "medium": "中", "low": "低"}.get(conf, "低")

    lines = ["[PKVC 分析]"]
    if not is_q1:
        lines.append(f"情境识别: {c_nl}")
        lines.append(f"         (C_local: {c_tag})")
    # 认知路径: 用 o_subtype 派生真实路径 (composite "X→Y" 直接用, single 加 "→ " 前缀)
    o_sub = quad.get("o_subtype") or ""
    if "→" in o_sub:
        path_str = o_sub  # e.g. "P3→K3"
    elif o_sub:
        path_str = f"→ {o_sub}"  # e.g. "→ K2"
    else:
        path_str = "(待)"
    lines.append(f"认知路径: {path_str}")
    lines.append(f"主导 (o_primary): {format_o_primary_dual(quad.get('o_primary'))}")
    lines.append(f"子状态 (o_subtype): {format_o_subtype_dual(quad.get('o_subtype'))}")
    if not is_q1:
        lines.append(f"决策方式 (p_surface): {format_p_surface_dual(quad.get('p_surface'))}")
        lines.append(f"调制 (p_modulation): {format_p_modulation_dual(quad.get('p_modulation'))}")
    lines.append(f"认知激活: {format_sadr_dual(quad.get('sadr_activations'))}")
    lines.append(f"推理: {designer_nl or '(待补长解释 NL)'}")
    lines.append(f"置信度: {conf_nl} ({conf})")
    lines.append("[/PKVC 分析]")
    return "\n".join(lines)


def build_q1q6_sample(
    profile: DesignerProfile,
    peer_block: str,  # legacy, ignored in β+++ (we use _build_peer_quad_safe_section)
    response: Q1Q6Response,
    pkvc_meta_hint: str = "",
    modulation_block: str = "",
    quad: dict[str, Any] | None = None,
    # β+++ 新加参数:
    q2_cardid: list[str] | None = None,
    peer_uids: list[str] | None = None,
    quad_by_uid_q: dict | None = None,
    profiles: dict | None = None,
    designer_responses: dict | None = None,
    option_text: str = "",  # ground truth option text 作 decision signature
) -> SFTSample | None:
    """β+++ q1q6 SFT sample. Q1=profile single (no Situation), Q3-Q6=situation single.
    Q2 返回 None (caller skip, 信息进其他题 Trait Profile).
    """
    qid = response.question_id
    if qid == "Q2":
        return None  # 不 emit Q2 sample, info 进其他题 input

    is_q1 = (qid == "Q1")
    sys_p = (SYSTEM_PROMPT_PROFILE if is_q1 else SYSTEM_PROMPT_SITUATION).format(
        designer_id=profile.user_id
    )

    # Input sections
    trait_section = _build_trait_profile_section(profile, q2_cardid or [])
    peer_section = _build_peer_quad_safe_section(
        peer_uids or [], qid, quad_by_uid_q or {}, profiles or {}, designer_responses or {},
    )

    # Question (no options)
    q_text = Q1_QUESTION_TEXT if is_q1 else Q_TEXT_NO_OPTIONS.get(qid, response.answer_full_text)

    input_parts = [
        f"<|system|>\n{sys_p}\n",
        "<|prompt|>\n",
        f"[Task Background]\n{TASK_BACKGROUND_PLACEHOLDER}\n",
    ]
    if not is_q1:
        input_parts.append(_build_situation_section(qid, quad) + "\n")
    input_parts.append(trait_section + "\n")
    input_parts.append(peer_section + "\n")
    input_parts.append(f"[Question {qid}]\n{q_text}\n")
    input_parts.append("<|response|>")
    if modulation_block:
        input_parts.insert(0, f"{modulation_block}\n")
    input_text = "\n".join(input_parts)

    # Output: 三段 [PKVC 分析] + [决策签名] + [答案]
    # β+++ v2 (2026-05-25): 加 [Peer 决策共识] 段强制 model attend peer letters,
    # 修 "model 0% unique value over peer-majority" bug (peer-majority 39% vs SFT 20%).
    pkvc_section = _build_pkvc_analysis_section(quad, response.explanation_nl, is_q1=is_q1)
    decision_sig = option_text or "(option text 缺)"
    answer = response.answer_letter or ""

    # Build [Peer 决策共识] from peer responses for this qid
    peer_letters: list[str] = []
    for pid in (peer_uids or [])[:3]:
        pr = (designer_responses or {}).get(pid, {}).get("q1q6", {}).get(qid)
        if pr and getattr(pr, "answer_letter", None):
            peer_letters.append(pr.answer_letter)
    from collections import Counter as _Counter
    if peer_letters:
        dist = _Counter(peer_letters)
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        majority = dist.most_common(1)[0][0]
        follow = "是" if (answer and answer == majority) else "否"
        peer_consensus = (
            f"[Peer 决策共识]\n"
            f"peer 答案分布: {dist_str}\n"
            f"peer 多数: {majority}\n"
            f"是否跟随多数: {follow}\n"
            f"[/Peer 决策共识]\n\n"
        )
    else:
        peer_consensus = ""

    output_text = (
        f"{pkvc_section}\n\n"
        f"{peer_consensus}"
        f"[决策签名]\n{decision_sig}\n[/决策签名]\n\n"
        f"[答案]\n{answer}\n[/答案]"
    )

    return SFTSample(
        user_id=profile.user_id, cluster_id=profile.cluster_id,
        direction="q1q6", task_id=qid, fold=profile.fold,
        input_text=input_text, output_text=output_text,
        meta={
            "answer_truth": response.answer_letter,
            "explanation_nl": response.explanation_nl,
            "pkvc_meta": pkvc_meta_hint,
            "quad": quad,
            "option_text": option_text,
            "is_q1": is_q1,
        },
        factor_scores=list(profile.factor_scores),
        factor_quartile=profile.factor_quartile,
    )


def build_q7_segment_sample(
    profile: DesignerProfile,
    peer_block: str,
    q7_segments: dict[str, str],
    task_id: str,                       # "B1" / "B2" / "B3"
    pkvc_truth: dict[str, Any] | None = None,
    modulation_block: str = "",
) -> SFTSample:
    """Construct one ``q7`` SFT sample (per-segment B1/B2/B3, brief v3.4 §4A).

    ``pkvc_truth`` should contain ``factor`` / ``sadr`` / ``criterion`` when the
    GPT-4o extractor has already labelled Q7. When absent, placeholders are
    written so the JSONL is shape-correct; downstream training will skip these
    via the curriculum mask until extractor results are available.

    ``modulation_block`` (v3.5.1 §4D.4 Path γ): trait→PKVC modulation triples
    injected into [System]. Empty when --rag-mode does not include modulation.
    """
    seg_order = ("B1", "B2", "B3")
    if task_id not in seg_order:
        raise ValueError(f"task_id must be one of {seg_order}, got {task_id}")

    # GT prefix: all preceding segments as teacher-forcing context
    prefix_lines: list[str] = []
    for prev_id in seg_order[: seg_order.index(task_id)]:
        prev_nl = q7_segments.get(prev_id, "")
        prev_name = SEG_NAME[prev_id]
        if prev_nl:
            prefix_lines.append(
                f"[{prev_name}]\n"
                f"主导因子: (GT placeholder)\n"
                f"激活节点: (GT placeholder)\n"
                f"推理: {prev_nl}\n"
                f"[/{prev_name}]"
            )
    prefix_block = "\n\n".join(prefix_lines)

    input_text = (
        "<|system|>\n"
        "你扮演一位设计师, 输出 Q7 三段思维链 (第一反应/决策链路/核心 tradeoff).\n"
        "每段格式: 主导因子 / 激活节点 / 推理.\n"
        + (f"{modulation_block}\n" if modulation_block else "")
        + "<|prompt|>\n"
        f"{profile.profile_polarity_nl}\n\n"
        f"{peer_block}\n\n"
        "[Scenario]\n场景: 老城区超市重构\n"
        "<|response|>"
        + (f"\n{prefix_block}" if prefix_block else "")
    )

    cur_nl = q7_segments.get(task_id, "")
    seg_name = SEG_NAME[task_id]
    if pkvc_truth is None:
        pkvc_truth = {}
    factor_key = "主导路径" if task_id == "B2" else "主导因子"
    factor_val = pkvc_truth.get("factor", "(待 extractor)")
    sadr_list = pkvc_truth.get("sadr", [])
    criterion = pkvc_truth.get("criterion")

    out_lines = [
        f"[{seg_name}]",
        f"{factor_key}: {factor_val}",
        f"激活节点: {', '.join(sadr_list) if sadr_list else '(待 extractor)'}",
    ]
    if task_id == "B3" and criterion:
        out_lines.append(f"核心底线: {criterion}")
    out_lines.append(f"推理: {cur_nl or '(原始 NL 缺失)'}")
    out_lines.append(f"[/{seg_name}]")

    return SFTSample(
        user_id=profile.user_id, cluster_id=profile.cluster_id,
        direction="q7", task_id=task_id, fold=profile.fold,
        input_text=input_text, output_text="\n".join(out_lines),
        meta={
            "factor_truth": factor_val,
            "sadr_truth": sadr_list,
            "criterion_truth": criterion,
            "raw_nl": cur_nl,
        },
        factor_scores=list(profile.factor_scores),     # v3.7
        factor_quartile=profile.factor_quartile,       # v3.7
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_all_samples(
    data_dir: Path,
    direction: Literal["q7", "q1q6", "both"] = "both",
    n_folds: int = 5,
    seed: int = 42,
    rag_k: int = 3,
    rag_mode: Literal["cosine", "og_rag", "modulation", "og_rag+modulation"] = "cosine",
    extracted_records_path: Path | None = None,
    pkvc_truth_path: Path | None = None,
    pkvc_meta_path: Path | None = None,
    modulation_files: dict[int, Path] | None = None,
    factor_scores_path: Path | None = None,
    cluster_filename: str = "user_cluster_efa.csv",
    quad_path: Path | None = None,  # v3.7: T_CAPS quad GT for Q3-Q6
) -> dict[str, list[SFTSample]]:
    """End-to-end build of SFTSamples for one or both directions.

    Args:
        data_dir: where the four source CSVs live.
        direction: ``"q7"`` / ``"q1q6"`` / ``"both"``.
        n_folds: 5-fold CV (default), can be reduced for pilots.
        seed: deterministic fold assignment + RAG retrieval.
        rag_k: how many peers to retrieve per target.
        use_og_rag: switch to ontology-grounded set-cover retrieval. Requires
            ``extracted_records_path`` to be a JSONL of extractor outputs.
        extracted_records_path: optional path to extracted records for og_rag.
        pkvc_truth_path: optional path to GPT-4o Q7 yaml labels.
        pkvc_meta_path: optional path to per-question PKVC metadata
            (which factor each Q1-Q6 probes).
    """
    raw = load_raw_data(data_dir, cluster_filename=cluster_filename)
    profile_matrix = build_profile_matrix(raw["scales"])
    cluster_labels = dict(zip(raw["cluster"]["user_id"], raw["cluster"]["cluster_id"]))
    designer_responses = collect_designer_responses(raw["tasks"])

    fold_map = assign_folds(raw["complete_uids"], cluster_labels, n_folds, seed)

    # --- v3.5.1: Trait-state modulation RAG (brief §4D.4 Path γ) ---
    use_modulation = "modulation" in rag_mode
    use_og_rag = "og_rag" in rag_mode
    modulations_by_fold: dict[int, list] = {}
    factor_scores_by_user: dict[str, dict[str, float]] = {}
    factor_medians: dict[str, float] = {}
    if use_modulation:
        if not modulation_files:
            raise ValueError(
                "rag_mode=%s requires modulation_files dict mapping fold_id → JSONL path. "
                "Run mapping_miner.py per fold first." % rag_mode
            )
        try:
            from sft_pipeline.rag import (
                load_modulation_triples, compute_factor_medians, load_factor_scores_dict,
            )
        except ImportError:
            from sft_pipeline.rag import (  # type: ignore[no-redef]
                load_modulation_triples, compute_factor_medians, load_factor_scores_dict,
            )
        for fid, jpath in modulation_files.items():
            modulations_by_fold[int(fid)] = load_modulation_triples(jpath)
            logger.info("loaded fold=%d modulation triples: %d", fid, len(modulations_by_fold[int(fid)]))
        if factor_scores_path and factor_scores_path.exists():
            factor_scores_by_user = load_factor_scores_dict(factor_scores_path)
            factor_medians = compute_factor_medians(factor_scores_path)
        else:
            logger.warning(
                "modulation enabled but factor_scores_path missing; "
                "ranking will fall back to |Δ|-only, no user-direction relevance"
            )

    # Per-user polarity
    profile_polarities: dict[str, dict[str, list[str]]] = {}
    profiles: dict[str, DesignerProfile] = {}

    # v3.7: 加载 factor_scores (4-d EFA, 替代 cluster_id 作 conditioning)
    # 仅 modulation 模式触发 factor_scores 加载 (上面 lines 692-694), 这里 fallback 兼容:
    # 若 factor_scores_by_user 空 (e.g. 旧 og_rag 模式), 仍然走 cluster_id legacy 路径
    f1_values: list[tuple[str, float]] = []
    for uid in raw["complete_uids"]:
        if uid in factor_scores_by_user:
            f1 = float(factor_scores_by_user[uid].get("F1", 0.0))
            f1_values.append((uid, f1))
    if f1_values:
        # 计算 F1 4 分位边界, assign quartile 0..3
        sorted_f1 = sorted(f1_values, key=lambda x: x[1])
        n = len(sorted_f1)
        quartile_bounds = [n // 4, n // 2, 3 * n // 4]
        uid_to_quartile = {}
        for i, (uid, _) in enumerate(sorted_f1):
            q = 0
            if i >= quartile_bounds[0]:
                q = 1
            if i >= quartile_bounds[1]:
                q = 2
            if i >= quartile_bounds[2]:
                q = 3
            uid_to_quartile[uid] = q
    else:
        uid_to_quartile = {}

    for uid in raw["complete_uids"]:
        if uid not in profile_matrix.index:
            continue
        row = profile_matrix.loc[uid]
        pol = numerical_to_polarity(row)
        profile_polarities[uid] = pol
        # v3.7: factor_scores 4-d (F1-F4); 若 factor_scores_by_user 空则 [0.0]*4
        user_fs = factor_scores_by_user.get(uid, {})
        fs_4d = [float(user_fs.get(f"F{i+1}", 0.0)) for i in range(4)]
        profiles[uid] = DesignerProfile(
            user_id=uid,
            cluster_id=int(cluster_labels.get(uid, -1)),
            profile_30d_numerical=row.to_dict(),
            profile_polarity=pol,
            profile_polarity_nl=polarity_to_nl(pol),
            fold=fold_map.get(uid, -1),
            q8_self_report=designer_responses.get(uid, {}).get("q8"),
            factor_scores=fs_4d,
            factor_quartile=uid_to_quartile.get(uid, -1),
        )

    # Optional truth tables
    pkvc_truth_by_user: dict[str, dict[str, Any]] = {}
    if pkvc_truth_path and pkvc_truth_path.exists():
        with pkvc_truth_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pkvc_truth_by_user.setdefault(rec["user_id"], {})[rec.get("task_id", "")] = rec

    pkvc_meta: dict[str, str] = {}
    if pkvc_meta_path and pkvc_meta_path.exists():
        pkvc_meta = json.loads(pkvc_meta_path.read_text(encoding="utf-8"))

    # v3.7: 加载 quad index — (user_id, question_id) → quad dict
    # 来源: data/processed/quad_extracted_v3_aligned.jsonl (quad_batch + quad_align_hint 输出)
    # 仅 Q3-Q6 有 quad, Q1/Q2 无 quad 进 sample (output 无 [T_CAPS quad] 段)
    quad_by_uid_q: dict[tuple[str, str], dict[str, Any]] = {}
    if quad_path and quad_path.exists():
        with quad_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "error" in rec:
                    continue
                key = (rec["designer_id"], rec["question_id"])
                quad_by_uid_q[key] = rec
        logger.info("Loaded %d quad entries from %s", len(quad_by_uid_q), quad_path)

    # Optional og_rag hypergraph
    og_rag_graph = None
    if use_og_rag:
        if not extracted_records_path or not extracted_records_path.exists():
            raise FileNotFoundError(
                f"og_rag requires extracted records, but {extracted_records_path} does not exist."
            )
        try:
            from sft_pipeline.rag import build_hypergraph
        except ImportError:
            from sft_pipeline.rag import build_hypergraph  # local import
        records = []
        with extracted_records_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        og_rag_graph = build_hypergraph(records, cluster_labels)
        logger.info("og_rag hypergraph: %s", og_rag_graph.stats())

    # Generate samples
    out: dict[str, list[SFTSample]] = {"q7": [], "q1q6": []}
    for uid, profile in profiles.items():
        # train fold (all designers not in this user's own fold) = peer pool
        peer_pool = [
            other for other in raw["complete_uids"]
            if other != uid and fold_map.get(other) != profile.fold
        ]
        peer_uids = _retrieve_peers(
            uid, profile.cluster_id, profile_matrix,
            cluster_labels, peer_pool, rag_k,
            use_og_rag=use_og_rag, og_rag_graph=og_rag_graph,
        )
        peer_block = format_cosine_peer_block(
            peer_uids, designer_responses, profile_polarities,
            direction="q7" if direction == "q7" else "q1q6",
        )

        # v3.5.1: Modulation block (Path γ system prompt insertion)
        modulation_block = ""
        if use_modulation and profile.fold in modulations_by_fold:
            try:
                from sft_pipeline.rag import (
                    retrieve_modulations, format_modulation_system_block,
                )
            except ImportError:
                from sft_pipeline.rag import (  # type: ignore[no-redef]
                    retrieve_modulations, format_modulation_system_block,
                )
            result = retrieve_modulations(
                target_user_id=uid,
                target_cluster=profile.cluster_id,
                all_triples=modulations_by_fold[profile.fold],
                target_factor_scores=factor_scores_by_user.get(uid),
                factor_medians=factor_medians or None,
                top_k=5,
                include_global=True,
            )
            modulation_block = format_modulation_system_block(result, max_chars=1200)

        if direction in {"q1q6", "both"}:
            # β+++: 收集 Q2 cardid_order 供其他题 input Trait Profile 用
            q2_resp = designer_responses.get(uid, {}).get("q1q6", {}).get("Q2")
            q2_cardid = q2_resp.cardid_order if q2_resp else []

            for qid in (*MCQ_QUESTIONS, SORT_QUESTION):
                q = designer_responses.get(uid, {}).get("q1q6", {}).get(qid)
                if q is None:
                    continue
                # v3.7: 取 quad for Q3-Q6 (Q1/Q2 quad=None)
                quad = quad_by_uid_q.get((uid, qid))
                # β+++: option text (ground truth) 作 decision signature
                opt_text = q.answer_full_text
                if "\n" in opt_text:
                    opt_text = opt_text.split("\n", 1)[1].strip().strip('"')
                sample = build_q1q6_sample(
                    profile, peer_block, q,
                    pkvc_meta_hint=pkvc_meta.get(qid, ""),
                    modulation_block=modulation_block,
                    quad=quad,
                    q2_cardid=q2_cardid,
                    peer_uids=peer_uids,
                    quad_by_uid_q=quad_by_uid_q,
                    profiles=profiles,
                    designer_responses=designer_responses,
                    option_text=opt_text,
                )
                if sample is None:
                    continue  # Q2 不 emit (方案 c)
                out["q1q6"].append(sample)
        if direction in {"q7", "both"}:
            segs = designer_responses.get(uid, {}).get("q7", {})
            if not segs:
                continue
            for tid in ("B1", "B2", "B3"):
                truth = pkvc_truth_by_user.get(uid, {}).get(tid)
                out["q7"].append(
                    build_q7_segment_sample(
                        profile, peer_block, segs, tid, truth,
                        modulation_block=modulation_block,
                    )
                )
    return out


def _retrieve_peers(
    target_uid: str, target_cluster: int,
    profile_matrix: pd.DataFrame, cluster_labels: dict[str, int],
    peer_pool: list[str], k: int,
    use_og_rag: bool, og_rag_graph: Any,
) -> list[str]:
    if not use_og_rag or og_rag_graph is None:
        return cosine_topk_same_cluster_peers(
            target_uid, target_cluster, profile_matrix,
            cluster_labels, peer_pool, k=k,
        )
    try:
        from sft_pipeline.rag import retrieve_via_og_rag
    except ImportError:
        from sft_pipeline.rag import retrieve_via_og_rag
    # Need to know scenario; data_prep iterates Q1-Q6 and Q7 segments, but
    # peer block is shared per-designer here (simplification). For tighter
    # coupling, move retrieval inside the per-question loop.
    result = retrieve_via_og_rag(
        target_user_id=target_uid,
        target_cluster=target_cluster,
        target_scenario_id="Q3",  # default; per-Q customisation TODO
        target_profile_polarity=None,
        hypergraph=og_rag_graph,
        k_max=k,
    )
    return [n.designer_id for n in result.selected_nodes][:k]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(samples: list[SFTSample], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_jsonl() + "\n")
    return len(samples)


def split_by_fold(samples: list[SFTSample]) -> dict[int, list[SFTSample]]:
    by_fold: dict[int, list[SFTSample]] = {}
    for s in samples:
        by_fold.setdefault(s.fold, []).append(s)
    return by_fold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def smoke_test(data_dir: Path) -> int:
    """Lightweight check (no I/O write): can we load, parse, fold, build a few
    samples, and emit valid JSONL lines? Returns 0 on success."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    samples = build_all_samples(data_dir, direction="both", n_folds=5)
    n_q7 = len(samples["q7"])
    n_q1q6 = len(samples["q1q6"])
    logger.info("Smoke: produced %d q7 + %d q1q6 SFT samples", n_q7, n_q1q6)
    if not samples["q7"] and not samples["q1q6"]:
        logger.error("Empty output, smoke FAIL")
        return 1
    # Validate JSONL line roundtrip on first sample
    for direction in ("q7", "q1q6"):
        if samples[direction]:
            line = samples[direction][0].to_jsonl()
            obj = json.loads(line)
            assert obj["direction"] == direction
            assert "input_text" in obj and "output_text" in obj
    logger.info("Smoke OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/data/wy/Designer_RolePlay/data"))
    parser.add_argument("--direction", choices=("q7", "q1q6", "both"), default="both")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rag-k", type=int, default=3)
    parser.add_argument(
        "--rag-mode",
        choices=("cosine", "og_rag", "modulation", "og_rag+modulation"),
        default="cosine",
        help="Retrieval mode (brief v3.5.1 §5.5 RAG-abl 4-way). "
             "cosine = profile-similarity peer; og_rag = ontology hypergraph; "
             "modulation = trait→PKVC modulation triples (§4D); "
             "og_rag+modulation = both (main contribution candidate).",
    )
    parser.add_argument("--use-og-rag", action="store_true",
                        help="DEPRECATED in v3.5.1 — equivalent to --rag-mode og_rag. Kept for back-compat.")
    parser.add_argument("--extracted-records", type=Path, default=None,
                        help="extractor_batch JSONL; required when rag-mode includes og_rag.")
    parser.add_argument("--modulation-dir", type=Path, default=None,
                        help="dir containing modulation_fold_{0..4}.jsonl (mapping_miner output). "
                             "Required when rag-mode includes modulation.")
    parser.add_argument("--factor-scores", type=Path, default=None,
                        help="user_factor_scores.csv (per designer × factor). "
                             "Optional with --rag-mode modulation; enables user-direction relevance.")
    parser.add_argument("--cluster-filename", default="user_cluster_efa.csv",
                        help="Filename inside --data-dir for cluster CSV. "
                             "Switch to user_cluster_efa_v2_200.csv post-200 EFA (v3.5.1 §4C).")
    parser.add_argument("--pkvc-truth", type=Path, default=None,
                        help="JSONL with per-(user, task) factor/sadr/criterion labels.")
    parser.add_argument("--pkvc-meta", type=Path, default=None,
                        help="JSON with per-question PKVC metadata.")
    parser.add_argument("--quad-path", type=Path,
                        default=Path("data/processed/quad_extracted_v3_aligned.jsonl"),
                        help="T_CAPS quad GT JSONL for Q3-Q6 (v3.7 加).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/data/wy/Designer_RolePlay/data/processed"))
    parser.add_argument("--dump-fold-split", type=Path, default=None,
                        help="Optional: also dump {fold_id: {train_users, test_users}} JSON for "
                             "mapping_miner --fold-json consumption (else use --derive-folds).")
    parser.add_argument("--smoke", action="store_true",
                        help="Lightweight integrity check, no JSONL output.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Resolve rag_mode + deprecated --use-og-rag
    rag_mode = args.rag_mode
    if args.use_og_rag and rag_mode == "cosine":
        rag_mode = "og_rag"
    elif args.use_og_rag and "og_rag" not in rag_mode:
        rag_mode = f"og_rag+modulation"

    # Resolve modulation files
    modulation_files: dict[int, Path] | None = None
    if "modulation" in rag_mode:
        if not args.modulation_dir or not args.modulation_dir.exists():
            parser.error(
                f"--rag-mode {rag_mode} requires --modulation-dir pointing to "
                f"modulation_fold_*.jsonl from mapping_miner.py"
            )
        modulation_files = {}
        for fid in range(args.n_folds):
            p = args.modulation_dir / f"modulation_fold_{fid}.jsonl"
            if not p.exists():
                parser.error(f"missing modulation file for fold {fid}: {p}")
            modulation_files[fid] = p

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.smoke:
        return smoke_test(args.data_dir)

    samples = build_all_samples(
        args.data_dir,
        direction=args.direction,
        n_folds=args.n_folds,
        seed=args.seed,
        rag_k=args.rag_k,
        rag_mode=rag_mode,
        extracted_records_path=args.extracted_records,
        pkvc_truth_path=args.pkvc_truth,
        pkvc_meta_path=args.pkvc_meta,
        modulation_files=modulation_files,
        factor_scores_path=args.factor_scores,
        cluster_filename=args.cluster_filename,
        quad_path=args.quad_path,
    )

    # Optional: dump fold split JSON for mapping_miner --fold-json downstream consumers
    if args.dump_fold_split is not None:
        raw = load_raw_data(args.data_dir, cluster_filename=args.cluster_filename)
        cluster_labels = dict(zip(raw["cluster"]["user_id"], raw["cluster"]["cluster_id"]))
        fold_map = assign_folds(raw["complete_uids"], cluster_labels, args.n_folds, args.seed)
        by_fold: dict[int, dict[str, list[str]]] = {
            fid: {"train_users": [], "test_users": []} for fid in range(args.n_folds)
        }
        for uid, fid in fold_map.items():
            by_fold[fid]["test_users"].append(uid)
            for other_fid in range(args.n_folds):
                if other_fid != fid:
                    by_fold[other_fid]["train_users"].append(uid)
        args.dump_fold_split.parent.mkdir(parents=True, exist_ok=True)
        args.dump_fold_split.write_text(
            json.dumps({str(k): v for k, v in by_fold.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Dumped fold split → %s", args.dump_fold_split)

    for direction, samps in samples.items():
        if not samps:
            continue
        by_fold_samples = split_by_fold(samps)
        for fold, fold_samps in sorted(by_fold_samples.items()):
            out_path = args.output_dir / f"{direction}_fold{fold}.jsonl"
            n = write_jsonl(fold_samps, out_path)
            logger.info("Wrote %d %s samples → %s", n, direction, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
