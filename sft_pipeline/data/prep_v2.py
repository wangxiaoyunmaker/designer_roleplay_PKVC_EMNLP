"""prep_v2.py — Framework v1.1 SFT 数据装配 (Path D, L_lm only).

按 docs/briefs/roleplay_prompt_framework_v1.md v1.1 spec 装配训练样本.

Pipeline:
  1. Load: users.csv / scale_responses.csv / task_responses.csv (data_v3, N=163)
           + q1q6_option_labels.jsonl (Q1-Q6 选项 PKVC label, 20 行)
           + q2_training_labels.jsonl (163 user Q2 var-len ranking + refined_nl)
           + q1q6_expansion_outputs.jsonl (Q1/Q3-Q6 expansion NL, ~32% cov)
           + efa_loadings.csv (30 x 4)
  2. EFA transform: 30-d Likert × loadings → 4-d factor scores per user
  3. trait-stratified 5-fold CV split (按 F1 quartile + random within-quartile)
  4. Per (user, qid) 装配 SFT sample:
        prompt = SYSTEM + TARGET_PROFILE + OWN_RESPONSES (mask self) + OWN_QUADS (fallback off)
               + PEER_RAW_NL (Q7 top-3) + SCENARIO + question + routing_format
        completion = "[认知路由标签]\n{pkvc_verbose}\n\n[NL 回答]\n{nl}"
  5. Augment: profile_noise × 3 (Likert ±1, polarity 重算)
  6. Output: data_v3/processed/sft_v2/fold_{k}/{train,test}.jsonl

CLI:
    python -m sft_pipeline.data.prep_v2 \\
        --config sft_pipeline/config/config_v2.yaml \\
        --baseline B5 \\
        [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("prep_v2")


# ─────────────────────────────────────────────────────────────────────
# Polarity helpers (inline, 跟 sft_pipeline.data.prep 兼容; 避免 sklearn import)
# ─────────────────────────────────────────────────────────────────────

POLARITY_ORDER = ["strong_pos", "weak_pos", "weak_neg", "strong_neg"]
POLARITY_NL_LABEL = {
    "strong_pos": "强 +",
    "weak_pos":   "弱 +",
    "weak_neg":   "弱 -",
    "strong_neg": "强 -",
}
LIKERT_TO_POLARITY = {
    1: "strong_neg", 2: "strong_neg",
    3: "weak_neg",   4: "weak_pos",
    5: "strong_pos", 6: "strong_pos",
}


def build_profile_matrix(scales: pd.DataFrame) -> pd.DataFrame:
    m = scales.pivot_table(index="user_id", columns="question_node",
                            values="score", aggfunc="mean")
    if m.isna().any().any():
        m = m.fillna(m.mean(numeric_only=True))
    return m


def numerical_to_polarity(row: pd.Series) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {k: [] for k in POLARITY_ORDER}
    for node, score in row.items():
        if pd.isna(score):
            continue
        try:
            bucket = LIKERT_TO_POLARITY[int(round(float(score)))]
        except (KeyError, ValueError):
            continue
        out[bucket].append(str(node))
    return out


def polarity_to_nl(polarity: dict[str, list[str]]) -> str:
    lines = ["[Profile] (4 档极性)"]
    for k in POLARITY_ORDER:
        items = polarity.get(k, [])
        if items:
            lines.append(f"{POLARITY_NL_LABEL[k]}: {', '.join(items)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Framework v1.1 §0.1 PKVC verbose (短 label)
# ─────────────────────────────────────────────────────────────────────

PKVC_FACTOR_VERBOSE = {
    "P": "P (Problem 现实与问题重构)",
    "K": "K (Knowledge 经验/数据/技术调用)",
    "V": "V (Value 价值判断)",
    "C": "C (Concept 构念/构建)",
}

PKVC_SUBTYPE_VERBOSE = {
    "P1": "P1 界定核心", "P2": "P2 拆解子问题", "P3": "P3 推翻假定",
    "K1": "K1 调用经验", "K2": "K2 客观数据测试", "K3": "K3 评估技术栈",
    "V1": "V1 价值立场", "V2": "V2 利益排序", "V3": "V3 设立原则",
    "C1": "C1 涌现画面", "C2": "C2 整合构建", "C3": "C3 提炼原则",
}

# Framework v1.1 §6 cognitive function (每题映射)
Q_TO_FUNCTION = {
    "Q1": "first_reaction",
    "Q2": "weighing",
    "Q3": "bottom_line",
    "Q4": "bottom_line",
    "Q5": "bottom_line",
    "Q6": "bottom_line",
}

COGNITIVE_FUNCTION_QUESTIONS = {
    "first_reaction": "面对这个项目, 您的直觉第一反应是什么? 请直接说出您不经决策权衡的本能反应。",
    "weighing":       "请按您的优先级排出 PKVC 12 个 cardId 的序列, 并说明您的权衡逻辑 (先排序, 再 NL)。",
    "bottom_line":    "面对这个具体冲突, 您会怎么决策? 您向上层提出的最核心底线是什么? 哪些是不可让步的?",
}

ROUTING_FORMAT_INSTRUCTIONS = {
    "first_reaction": "在 [认知路由标签] 段输出**单字母 + verbose 标签**, 形如 `C (Concept 构念/构建)`. 表示本次第一反应主导的认知模块. 然后空行 + [NL 回答].",
    "weighing":       "在 [认知路由标签] 段输出 **PKVC 12 cardId 的优先级序列** (可变长度, 通常 4-12 项), 用 ` > ` 分隔, 每 cardId 后跟 verbose 含义, 形如 `P1 界定核心 > V3 设立原则 > K2 客观数据测试 > C2 整合构建 > P2 拆解子问题 > K1 调用经验`. **允许重复表示强偏好** (e.g. `K1 调用经验 > K3 评估技术栈 > K1 调用经验`), 不要求 12 项全覆盖. 然后空行 + [NL 回答] (说明权衡逻辑).",
    "bottom_line":    "在 [认知路由标签] 段输出 **PKVC 单字母** (e.g. `V (Value 价值判断)`) 或 **2 段路径** (e.g. `P3 推翻假定 → K3 评估技术栈`, 头 ≠ 尾, 上限 2 段). 表示底线决策的认知路径. 然后空行 + [NL 回答] (说明不可让步的部分).",
}

SYSTEM_MESSAGE = """你扮演下面注入的特定设计师, 用第一人称回答提出的问题. 严格遵守输出 schema.

输出 schema (双段结构化, 跟问题类型对应):

  [认知路由标签]
  <一行结构化 PKVC 标注; 具体格式跟问题类型对应, scenario 段会明确要求>

  [NL 回答]
  <100-300 字一段连续中文 NL>

格式约束:
- [认知路由标签] 跟 [NL 回答] 各占独立段, 两段之间空一行
- [认知路由标签] 内容必须是合法 PKVC 表达:
    单字母:  "C (Concept 构念/构建)"
    路径:    "P3 推翻假定 → K3 评估技术栈"
    序列:    "P1 界定核心 > V3 设立原则 > K2 客观数据测试 > ..." (weighing 函数按用户优先级排序, 可变长度 4-12+ 项, 允许重复)
- [NL 回答] 100-300 字, 一段连续中文, 不要列表 / markdown / 子标题, 不要前缀
- 必须体现下面注入的设计师 trait 倾向 (强 + trait 应明显出现, 强 - 应反对或弱化)
- 若上方给了该 designer 在过去问题的认知路由, 本次应跟过去路由模式一致
- 禁止 default 到 V (Value 价值判断) 的 "balanced ethical reasoning" 套话"""


@dataclass
class PromptConfig:
    profile: bool = False
    own_responses: bool = False
    own_quads: bool = False
    peer_raw_nl: bool = False


# ─────────────────────────────────────────────────────────────────────
# Load + transform
# ─────────────────────────────────────────────────────────────────────

def compute_factor_scores(scales: pd.DataFrame, loadings_path: Path) -> pd.DataFrame:
    """30-d scale × (30 x 4) loadings → 4-d factor scores per user.

    简化 regression-style score (loadings · standardized scale). 跟 EFA 软件输出的
    Bartlett/Anderson-Rubin scores 不严格等价, 但在 v3.7 'continuous trait conditioning'
    用途下 (peer retrieval + sampler stratify) 足够.
    """
    ld = pd.read_csv(loadings_path).set_index("question_node")
    matrix = build_profile_matrix(scales)
    # align columns
    common = [q for q in ld.index if q in matrix.columns]
    if len(common) < len(ld):
        logger.warning("EFA loadings cover %d/%d items; missing: %s",
                       len(common), len(ld), set(ld.index) - set(matrix.columns))
    M = matrix[common].values                 # N × d
    L = ld.loc[common].values                 # d × 4
    M_std = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-8)
    scores = M_std @ L                        # N × 4
    return pd.DataFrame(scores, index=matrix.index, columns=["F1", "F2", "F3", "F4"])


def trait_stratified_fold_split(factor_df: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    """按 F1 quartile stratify, 同 quartile 内 hash-based fold 分配 (deterministic)."""
    f1 = factor_df["F1"].values
    q = np.quantile(f1, [0.25, 0.5, 0.75])
    quartile = np.digitize(f1, q)            # 0-3
    fold_of: dict[str, int] = {}
    bucket_assignments = {b: [] for b in range(4)}
    for uid, qt in zip(factor_df.index, quartile):
        bucket_assignments[int(qt)].append(uid)
    rng = random.Random(seed)
    for b, uids in bucket_assignments.items():
        rng.shuffle(uids)
        for i, uid in enumerate(uids):
            fold_of[str(uid)] = i % n_folds
    return fold_of


# ─────────────────────────────────────────────────────────────────────
# Source jsonl loaders
# ─────────────────────────────────────────────────────────────────────

def load_q1q6_option_labels(p: Path) -> dict[tuple[str, str], dict]:
    """key = (qid, option) → label dict."""
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["qid"], r["option"])] = r
    return out


def load_q2_training_labels(p: Path) -> dict[str, dict]:
    """key = user_id → entry."""
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[str(r["user_id"])] = r
    return out


def load_q1q6_expansion(p: Path) -> dict[tuple[str, str], dict]:
    """key = (user_id, qid) → {nl, factor, subtype}.

    Schema (q1q6_expansion_outputs.jsonl, gpt-4o output):
      {user_id, question_id, seed_nl, result: {expanded_nl, inferred_factor,
       inferred_subtype, ...}, validation: {ok, fails}, ...}

    只取 validation.ok=True 的 entry. T1 ablation 用 factor/subtype 替代 option_labels.
    """
    out: dict[tuple[str, str], dict] = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r.get("user_id"))
            qid = r.get("question_id") or r.get("qid")
            if not (uid and qid):
                continue
            valid = r.get("validation") or {}
            if not valid.get("ok", False):
                continue
            result = r.get("result") or {}
            nl = result.get("expanded_nl") or r.get("refined_nl") or r.get("output")
            if nl:
                out[(uid, qid)] = {
                    "nl": nl,
                    "factor": result.get("inferred_factor"),    # T1: v8.7 抽出来的 factor
                    "subtype": result.get("inferred_subtype"),  # T1: subtype
                }
    return out


def expansion_nl(expansion: dict[tuple[str, str], dict], uid: str, qid: str) -> str | None:
    """Helper: backward-compatible NL accessor (旧 code 期望返回 str)."""
    e = expansion.get((uid, qid))
    return e["nl"] if e else None


def load_q7_segments(p: Path) -> dict[str, dict[str, str]]:
    """key = user_id → {seg1, seg2, seg3} NL text. From v8.7 extractor outputs."""
    out: dict[str, dict[str, str]] = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r["user_id"])
            seg = r["segment_role"]
            text = r["text"]
            out.setdefault(uid, {})[seg] = text
    return out


def parse_task_responses(tasks: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """raw csv → {user_id: {qid: {answer_letter, option_text, text_supplement}}}."""
    out: dict[str, dict[str, dict]] = {}
    import re
    pat = re.compile(r'^"?([A-D])\.\s*(.+?)"?$', re.DOTALL)
    for _, row in tasks.iterrows():
        uid = str(row["user_id"])
        qid = row["question_id"]
        if qid in ("Q2", "Q7", "Q8"):  # Q2 / Q7 用 jsonl, Q8 metadata
            continue
        ans = str(row["answer_value"]).strip()
        m = pat.match(ans)
        if not m:
            continue
        out.setdefault(uid, {})[qid] = {
            "letter": m.group(1),
            "text": m.group(2).rstrip('"').strip(),
            "text_supplement": row.get("text_supplement") or "",
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Prompt 各 Section 装配
# ─────────────────────────────────────────────────────────────────────

def fmt_profile_section(polarity_nl: str, demographics: str) -> str:
    return (
        "╔═══ TARGET DESIGNER PROFILE ═══╗\n\n"
        f"[30 维信念量表 4 档极性 NL]\n{polarity_nl}\n\n"
        f"[Demographic]\n{demographics}\n\n"
        "╚═══════════════════════════════╝"
    )


def fmt_own_responses_section(
    user_resp: dict[str, dict],
    q2_entry: dict | None,
    expansion: dict[tuple[str, str], dict],
    option_labels: dict[tuple[str, str], dict],
    uid: str,
    mask_qid: str,
) -> str:
    """框架 v1.1 §4. mask 掉 mask_qid 自身防 leakage."""
    lines = ["╔═══ TARGET DESIGNER'S OWN Q1-Q6 HISTORY (这位设计师在过去 6 题的真实回答 + 自己写的解释) ═══╗"]
    question_titles = {
        "Q1": "Q1 面对委托第一启动认知模块",
        "Q2": "Q2 PKVC 优先级排序题",
        "Q3": "Q3 数据 vs 美感冲突",
        "Q4": "Q4 商业 vs 隐私 vs 开源",
        "Q5": "Q5 AI 冲击与人本/有机平衡",
        "Q6": "Q6 商业利润 vs 用户责任",
    }
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        if q == mask_qid:
            continue
        lines.append(f"\n[{question_titles[q]}]")
        if q == "Q2":
            if q2_entry is None:
                lines.append("  (未答)")
                continue
            lines.append(f"  设计师 cardId 排序: {q2_entry.get('ranking_verbose', '未排序')}")
            lines.append(f"  设计师自己写的 NL 解释: {q2_entry.get('refined_nl', '(无补充)')}")
        else:
            resp = user_resp.get(q)
            if not resp:
                lines.append("  (未答)")
                continue
            label = option_labels.get((q, resp["letter"]))
            pkvc_str = label["pkvc_verbose"] if label else "未知"
            lines.append(
                f"  选项 (内容 + 该选项激活的 PKVC factor): \"{resp['text']}\" "
                f"[该选项激活: {pkvc_str}]"
            )
            nl = expansion_nl(expansion, uid, q) or resp.get("text_supplement", "") or "(无补充解释)"
            lines.append(f"  设计师自己写的 NL 解释: {nl}")
    lines.append("\n╚═══════════════════════════════╝")
    return "\n".join(lines)


def fmt_own_quads_section_fallback() -> str:
    """v3.7 §5: quad pipeline (blocker A) 没跑 → 整段 fallback off, 返回 ''.
    Framework v1 §5 锁的 fallback 行为."""
    return ""


def fmt_peer_raw_nl_section(peers: list[dict]) -> str:
    """框架 v1.1 §5. peers: [{user_id, sim, q7_segs}, ...]."""
    if not peers:
        return ""
    lines = ["╔═══ TOP-3 TRAIT-SIMILAR PEERS' Q7 RAW NL (作风格/语气 reference, 不要照抄) ═══╗"]
    for i, p in enumerate(peers, 1):
        lines.append(f"\n[peer {i} — trait-similarity rank {i}]")
        for seg in ("seg1", "seg2", "seg3"):
            text = p["q7_segs"].get(seg, "")[:500] if p.get("q7_segs") else ""
            lines.append(f"{seg}: {text}")
    lines.append("\n╚═══════════════════════════════╝")
    return "\n".join(lines)


def fmt_scenario_section(scenario_text: str, qid: str) -> str:
    func = Q_TO_FUNCTION[qid]
    return (
        "╔═══ 场景 ═══╗\n\n"
        f"{scenario_text}\n\n"
        "╚═══════════╝\n\n"
        f"{COGNITIVE_FUNCTION_QUESTIONS[func]}\n\n"
        "[认知路由标签格式 (本次必须遵守)]\n"
        f"{ROUTING_FORMAT_INSTRUCTIONS[func]}"
    )


# ─────────────────────────────────────────────────────────────────────
# Peer retrieval (trait cosine top-3, fold-aware)
# ─────────────────────────────────────────────────────────────────────

def retrieve_peers(
    target_uid: str,
    factor_df: pd.DataFrame,
    train_uids: set[str],
    q7_segs: dict[str, dict[str, str]],
    k: int = 3,
) -> list[dict]:
    if target_uid not in factor_df.index:
        return []
    target_vec = factor_df.loc[target_uid].values
    candidates = []
    for uid in train_uids:
        if uid == target_uid or uid not in factor_df.index or uid not in q7_segs:
            continue
        v = factor_df.loc[uid].values
        sim = float(np.dot(target_vec, v) / (np.linalg.norm(target_vec) * np.linalg.norm(v) + 1e-8))
        candidates.append({"user_id": uid, "sim": sim, "q7_segs": q7_segs[uid]})
    candidates.sort(key=lambda r: -r["sim"])
    return candidates[:k]


# ─────────────────────────────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────────────────────────────

def build_prompt(
    cfg: PromptConfig,
    polarity_nl: str,
    demographics: str,
    user_resp: dict[str, dict],
    q2_entry: dict | None,
    expansion: dict[tuple[str, str], str],
    option_labels: dict[tuple[str, str], dict],
    peers: list[dict],
    scenario_text: str,
    qid: str,
    uid: str,
    is_training: bool = True,
) -> tuple[str, str]:
    """Return (system_msg, user_msg) for SFT prompt.

    framework v1.1 lock #21 (user 2026-05-25 拍):
      训练 prompt 不注入 peer_raw_nl section, 仅 eval 阶段注入.
      原因: 训练时塞 peer Q7 NL (≈ 2653 token) 引入 Q7-scenario distribution prior,
      model 注意力被 peer 占走 ~54% prompt, training target Q1-Q6 NL 信号被淹.
      train-eval schema asymmetric 是有意设计.

    is_training=True (default): 强制 peer_raw_nl 段 off
    is_training=False:           按 cfg.peer_raw_nl toggle
    """
    sections = []
    if cfg.profile:
        sections.append(fmt_profile_section(polarity_nl, demographics))
    if cfg.own_responses:
        sections.append(fmt_own_responses_section(
            user_resp, q2_entry, expansion, option_labels, uid, mask_qid=qid,
        ))
    if cfg.own_quads:
        s = fmt_own_quads_section_fallback()
        if s:
            sections.append(s)
    # lock #21: peer_raw_nl 只在 eval (推理) 阶段注入, 训练不注入
    if cfg.peer_raw_nl and not is_training:
        s = fmt_peer_raw_nl_section(peers)
        if s:
            sections.append(s)
    sections.append(fmt_scenario_section(scenario_text, qid))
    user_msg = "\n\n".join(sections)
    return SYSTEM_MESSAGE, user_msg


# ─────────────────────────────────────────────────────────────────────
# Completion (ground truth target)
# ─────────────────────────────────────────────────────────────────────

def build_completion(
    qid: str,
    uid: str,
    user_resp: dict[str, dict],
    q2_entry: dict | None,
    expansion: dict[tuple[str, str], dict],
    option_labels: dict[tuple[str, str], dict],
    label_source: str = "option_metadata",
) -> tuple[str, dict[str, Any]] | None:
    """Return (completion_str, metadata) or None if data missing.

    label_source:
      "option_metadata" (default): Q1/Q3-Q6 routing 从 q1q6_option_labels.jsonl 推
                                    (选项 metadata, framework v1.1 lock #5)
      "v87":                       Q1/Q3-Q6 routing 从 q1q6_expansion_outputs.jsonl
                                    的 inferred_factor / inferred_subtype 拿 (T1 ablation,
                                    跟 eval v8.7 schema 一致, 解 dual-prior 问题)
    """
    func = Q_TO_FUNCTION[qid]
    if qid == "Q2":
        if q2_entry is None:
            return None
        routing = q2_entry["ranking_verbose"]
        nl = q2_entry["refined_nl"]
        meta = {
            "quality_tier": q2_entry.get("quality_tier", "primary"),
            "validation_ok": q2_entry.get("validation_ok", True),
            "has_dup": q2_entry.get("has_dup", False),
            "n_cards": q2_entry.get("n_cards", 0),
            "label_source": "q2_refined",  # Q2 不受 label_source 影响
        }
    else:
        resp = user_resp.get(qid)
        if not resp:
            return None
        v87 = expansion.get((uid, qid)) or {}

        if label_source == "v87":
            factor = v87.get("factor")
            subtype = v87.get("subtype")
            if not factor:
                # v8.7 label 缺失, 不进训练 (避免 fallback 到 option_metadata 污染 ablation)
                return None
            if qid == "Q1":
                # Q1 first_reaction = 单字母 verbose
                routing = PKVC_FACTOR_VERBOSE.get(factor)
                if not routing:
                    return None
            else:
                # Q3-Q6 bottom_line = 子状态 verbose (e.g. "V1 价值立场")
                # subtype 缺失则回 factor verbose
                routing = PKVC_SUBTYPE_VERBOSE.get(subtype) or PKVC_FACTOR_VERBOSE.get(factor)
                if not routing:
                    return None
            meta = {
                "label_source": "v87",
                "inferred_factor": factor,
                "inferred_subtype": subtype,
                "option_letter": resp["letter"],
            }
        else:  # option_metadata (default, framework v1.1 lock #5)
            label = option_labels.get((qid, resp["letter"]))
            if not label:
                return None
            routing = label["pkvc_verbose"]
            meta = {
                "label_source": "option_metadata",
                "is_ambiguous": label.get("is_ambiguous", False),
                "alt_labels": label.get("alt_labels", []),
                "option_letter": resp["letter"],
            }
        # NL 段一律 expansion 优先, fallback text_supplement (跟 label_source 解耦)
        nl = v87.get("nl") or resp.get("text_supplement", "") or "(无补充解释)"
        meta["quality_tier"] = "primary" if v87.get("nl") else "fallback_supplement"

    meta["function"] = func
    completion = f"[认知路由标签]\n{routing}\n\n[NL 回答]\n{nl}"
    return completion, meta


# ─────────────────────────────────────────────────────────────────────
# Augment: profile noise (Likert ±1, polarity 重算)
# ─────────────────────────────────────────────────────────────────────

def augment_polarity_noise(profile_row: pd.Series, seed: int, delta: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-delta, delta + 1, size=len(profile_row))
    new = (profile_row.values.astype(int) + noise).clip(1, 6)
    return pd.Series(new, index=profile_row.index)


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def demographic_str(user_row: pd.Series) -> str:
    parts = [
        str(user_row.get("identity", "?")),
        f"{user_row.get('years_exp', '?')} 经验",
        f"{user_row.get('design_field', '?')} 方向",
        str(user_row.get('education_stage_current', '?')),
    ]
    return " / ".join(parts)


def run(config_path: Path, baseline: str, smoke: bool = False,
         fold_filter: int | None = None,
         label_source: str = "option_metadata",
         out_subdir: str | None = None) -> int:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    bcfg = cfg["baselines"][baseline]
    pcfg = PromptConfig(
        profile=bcfg["profile"],
        own_responses=bcfg["own_responses"],
        own_quads=bcfg["own_quads"],
        peer_raw_nl=bcfg["peer_raw_nl"],
    )

    root = Path.cwd()
    logger.info("Loading raw csv...")
    users = pd.read_csv(root / paths["users"])
    users["user_id"] = users["wechat_id"].astype(str)
    users = users.set_index("user_id")
    scales = pd.read_csv(root / paths["scale_responses"])
    tasks = pd.read_csv(root / paths["task_responses"])

    logger.info("Computing factor scores (30-d Likert × 4 loadings)...")
    factor_df = compute_factor_scores(scales, root / paths["efa_loadings"])
    factor_df.index = factor_df.index.astype(str)
    factor_df.to_csv(root / paths["factor_scores_out"])
    logger.info("Factor scores: N=%d → %s", len(factor_df), paths["factor_scores_out"])

    logger.info("Loading jsonl resources...")
    option_labels = load_q1q6_option_labels(root / paths["q1q6_option_labels"])
    q2_labels = load_q2_training_labels(root / paths["q2_training_labels"])
    expansion = load_q1q6_expansion(root / paths["q1q6_expansion_outputs"])
    q7_segs = load_q7_segments(root / paths["q7_extracted_v8_7"])
    user_resp = parse_task_responses(tasks)

    logger.info("Building profile matrix + polarity NL...")
    profile_matrix = build_profile_matrix(scales)
    profile_matrix.index = profile_matrix.index.astype(str)

    logger.info("Trait-stratified %d-fold split (stratify by F1 quartile)...",
                data_cfg["n_folds"])
    fold_of = trait_stratified_fold_split(factor_df, data_cfg["n_folds"], train_cfg["seed"])

    # User set with all required data
    uids_all = [u for u in factor_df.index
                if u in user_resp or u in q2_labels]
    if smoke:
        uids_all = uids_all[:5]
        logger.info("[SMOKE] limit to 5 users")

    out_dir = root / paths["sft_v2_out_dir"]
    if out_subdir:
        out_dir = out_dir.parent / out_subdir   # e.g. data_v3/processed/sft_v2_t1
        logger.info("label_source=%s, output overridden to: %s", label_source, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_per_fold = {k: {"train": 0, "test": 0} for k in range(data_cfg["n_folds"])}

    fold_writers: dict[tuple[int, str], list[dict]] = {}

    qid_list = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    profile_noise_mult = data_cfg["profile_noise_multiplier"]
    quality_filter = data_cfg["quality_tier_filter"]

    for uid in uids_all:
        fold = fold_of.get(uid)
        if fold is None:
            continue
        if fold_filter is not None and fold != fold_filter:
            continue

        # Q2 filter
        q2_entry = q2_labels.get(uid)
        if q2_entry and quality_filter == "primary" and q2_entry.get("quality_tier") != "primary":
            q2_entry_for_completion = None  # 训练 target 砍掉
        else:
            q2_entry_for_completion = q2_entry

        # 训练 fold = 该 user 在 test fold 时 skip prep (我们只 build train samples);
        # 但 eval prep 要单独 build test 集. 这里输出 train/test 都 build.
        # train fold: 该 user 不在 test, 即作为 train 样本 + 别人 test 时可作 peer
        # test fold: build 一份 "评测 prompt" 给 Q7 推理用 (但 Q7 不在 Q1-Q6 循环里, 留 infer_v2 处理)
        # 故 prep_v2 只 build Q1-Q6 训练样本.

        # Get train peer pool (exclude current fold uids)
        train_uids = {u for u in uids_all if fold_of.get(u) != fold}

        # Profile sources
        if uid not in profile_matrix.index:
            continue
        profile_row = profile_matrix.loc[uid]
        polarity = numerical_to_polarity(profile_row)
        polarity_nl_str = polarity_to_nl(polarity)
        demographics = demographic_str(users.loc[uid]) if uid in users.index else ""

        peers = retrieve_peers(uid, factor_df, train_uids, q7_segs, k=3)

        for qid in qid_list:
            # build completion (ground truth) first
            comp = build_completion(qid, uid, user_resp.get(uid, {}),
                                     q2_entry_for_completion, expansion, option_labels,
                                     label_source=label_source)
            if comp is None:
                continue
            completion_str, comp_meta = comp

            # build base sample (no aug)
            for noise_idx in range(profile_noise_mult + 1):  # +1 for the "no noise" original
                if noise_idx == 0:
                    pol_nl = polarity_nl_str
                else:
                    seed_int = int(hashlib.md5(f"{uid}_{qid}_{noise_idx}".encode()).hexdigest()[:8], 16)
                    perturbed = augment_polarity_noise(profile_row, seed_int,
                                                       delta=data_cfg["profile_noise_likert_delta"])
                    pol = numerical_to_polarity(perturbed)
                    pol_nl = polarity_to_nl(pol)

                sys_msg, user_msg = build_prompt(
                    pcfg, pol_nl, demographics,
                    user_resp.get(uid, {}), q2_entry, expansion, option_labels,
                    peers, cfg["scenario_text"]["training"], qid, uid,
                )

                sample = {
                    "user_id": uid,
                    "qid": qid,
                    "function": Q_TO_FUNCTION[qid],
                    "fold": fold,
                    "baseline": baseline,
                    "aug_idx": noise_idx,
                    "system": sys_msg,
                    "prompt": user_msg,
                    "completion": completion_str,
                    "meta": comp_meta,
                }
                # split: own fold = test, else train
                split = "test" if fold_of.get(uid) == fold else "train"
                # Actually since prep_v2 iterates each user with their own fold,
                # the user is always in `fold` (their own). For 5-fold CV, fold-k file's
                # test set contains users with fold_of==k, train set contains the other 4 folds.
                # So when iterating user uid with fold_of[uid]==k:
                #   in fold_k jsonl:           → split=test
                #   in fold_0..4 except k:     → split=train
                # We emit the sample to all 5 fold files (1 test + 4 train).
                for fk in range(data_cfg["n_folds"]):
                    if fold_filter is not None and fk != fold_filter:
                        continue
                    if fk == fold:
                        s = {**sample, "split": "test"}
                    else:
                        s = {**sample, "split": "train"}
                    fold_writers.setdefault((fk, s["split"]), []).append(s)
                    n_per_fold[fk][s["split"]] += 1
                    n_total += 1

    # Write
    for (fk, split), rows in fold_writers.items():
        sub = out_dir / f"fold_{fk}"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("[fold %d/%s] wrote %d samples → %s", fk, split, len(rows), path)

    print(f"[OK] total samples emitted (×5 fold replicate): {n_total}")
    for k in range(data_cfg["n_folds"]):
        print(f"  fold {k}: train={n_per_fold[k]['train']}  test={n_per_fold[k]['test']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_pipeline/config/config_v2.yaml")
    p.add_argument("--baseline", default="B5")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--fold", type=int, default=None, help="只 build 指定 fold (debug)")
    p.add_argument("--label-source", choices=["option_metadata", "v87"], default="option_metadata",
                   help="T1 ablation: Q1/Q3-Q6 routing 用选项 metadata 还是 v8.7 抽 NL 出来的 inferred_factor")
    p.add_argument("--out-subdir", default=None,
                   help="覆盖默认 sft_v2_out_dir, e.g. sft_v2_t1. label_source=v87 时建议显式指定")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # 自动给 v87 路径加 _t1 后缀, 避免覆盖默认 sft_v2/
    out_subdir = args.out_subdir
    if args.label_source == "v87" and out_subdir is None:
        out_subdir = "sft_v2_t1"
    return run(Path(args.config), args.baseline, smoke=args.smoke, fold_filter=args.fold,
               label_source=args.label_source, out_subdir=out_subdir)


if __name__ == "__main__":
    sys.exit(main())
