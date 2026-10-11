"""Baseline role-play inference for 162 designer × 3 Q7 seg.

Prompt 模块 (5 个, 累加):
  ① pi          人口统计学 (users.csv)
  ② 30维        SADR polarity persona (scale_responses.csv)
  ③ self Q1-Q6  designer 自身 Q1-Q6 选择+回答, seg-dependent:
                   seg1→Q1, seg2→Q2, seg3→Q3-Q6
  ④ peer Q7     相似 peer 在 Q7 同 seg 的回答
  ⑤ CAPS 路由   激活的 CAPS 四元组 (caps_quadruples.json)

Baselines (累加设计, 每层包含前面所有模块):
  B0  (Zero-shot):               仅场景
  B1  (+ pi):                    ①pi
  B2  (+ 30维):                  ①pi + ②30维
  B3  (+ self):                  ①pi + ②30维 + ③self
  B4  (+ peer cosine):           ①pi + ②30维 + ③self + ④peer-Q7(cosine)
  B5  (+ CAPS + peer JS):        ①pi + ②30维 + ③self + ⑤CAPS + ④peer-Q7(JS)

输出: structured PKVC JSON (dominant_factor, state_sequence, criterion, nl_answer)
不使用 CoT.

Usage:
  .venv/bin/python scripts/run_3_baselines_q7.py [--n 162] [--baselines B0,B1,B2,B3,B4,B5]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

from sft_pipeline.rag.caps_anchored import (
    load_caps_pool, CAPSPool, Designer, signature_similarity,
)
from sft_pipeline.rag.hypergraph import build_hypergraph, HyperGraph, SegmentNode
from sft_pipeline.rag.peer import retrieve_via_og_rag


# ============================================================
# Q7 scenario brief (hardcoded, source: docs / data exploration)
# ============================================================

Q7_BACKGROUND = """【超市重构困境】你接到一家零售巨头委托, 在老城区核心地段打造"全国首个全数字化、无人化的高效零售标杆超市". 上层提出 3 个核心指令:
  1. 完全拆除人工收银台, 全部换成无人自助结账;
  2. 启用 AI 动态货架/界面推荐算法, 把高利润非必需品摆到最显眼位置, 把必需品藏到深处;
  3. 三个月内必须按数字化方案完整铺开.
注意: 老城区 60+ 岁的银发族占比超过 40%, 他们不熟悉数字设备."""

SEG_TASKS = {
    "seg1": "请描述你第一时间看到这个项目时的真实直觉反应/感受 (200-400 字).",
    "seg2": "请按你的实际决策逻辑, 描述你怎么在【数据】【商业要求】【用户伦理】3 者之间权衡, 思考链路是什么 (200-400 字).",
    "seg3": "请说出你向管理层提出的核心设计底线 + 可妥协方案, 你绝不让步的是什么, 愿意妥协的是什么 (200-400 字).",
}


# ============================================================
# Demographics profile (for B1d)
# ============================================================

DEMO_FIELD_LABELS = {
    "identity": "身份",
    "education_stage_current": "教育阶段",
    "years_exp": "设计经验",
    "design_field": "设计领域",
    "design_field_other": "设计领域(其他)",
    "education_background": "教育背景",
    "work_context": "工作场景",
    "service_target": "服务对象",
    "project_role": "项目角色",
    "project_type": "项目类型",
    "interdisciplinary": "跨学科经历",
    "teaching_experience": "教学经验",
}


def load_designer_demographics(users_csv_path: Path) -> dict[str, str]:
    """Return {designer_id → formatted demo profile string}."""
    import pandas as pd
    df = pd.read_csv(users_csv_path)
    df["wechat_id"] = df["wechat_id"].astype(str)
    out = {}
    for _, row in df.iterrows():
        did = row["wechat_id"]
        lines = []
        for fld, label in DEMO_FIELD_LABELS.items():
            val = row.get(fld)
            if val is None or (isinstance(val, float) and (val != val)):  # NaN
                continue
            val = str(val).strip()
            if not val or val.lower() == "nan":
                continue
            lines.append(f"  - {label}: {val}")
        if lines:
            out[did] = "\n".join(lines)
    return out


# ============================================================
# Polarity helpers — 复用 prep_v2 同一套逻辑 + TRAIT_NL 权威表
# ============================================================

from sft_pipeline.data.trait_nl import TRAIT_NL

POLARITY_ORDER = ["strong_pos", "weak_pos", "weak_neg", "strong_neg"]
POLARITY_NL_LABEL = {
    "strong_pos": "强 +", "weak_pos": "弱 +",
    "weak_neg": "弱 -", "strong_neg": "强 -",
}
LIKERT_TO_POLARITY = {
    1: "strong_neg", 2: "strong_neg",
    3: "weak_neg",   4: "weak_pos",
    5: "strong_pos", 6: "strong_pos",
}


def load_designer_personas(scale_csv: Path) -> dict[str, dict[str, list[str]]]:
    """Load scale_responses.csv → {user_id → {polarity_bucket → [trait_name, ...]}}.

    全 30 维, 不 cap. 跟 prep_v2.numerical_to_polarity 同逻辑.
    """
    import csv
    with scale_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        uid = r["user_id"]
        score = int(round(float(r["score"])))
        bucket = LIKERT_TO_POLARITY[score]
        out.setdefault(uid, {k: [] for k in POLARITY_ORDER})
        out[uid][bucket].append(r["question_node"])
    return out


def polarity_to_nl(polarity: dict[str, list[str]]) -> str:
    """30 维 polarity-aware NL block. 与 prep_v2.polarity_to_nl 同格式."""
    lines: list[str] = []
    for k in POLARITY_ORDER:
        items = polarity.get(k, [])
        if not items:
            continue
        lines.append(f"{POLARITY_NL_LABEL[k]}:")
        for trait in items:
            info = TRAIT_NL.get(trait)
            if info is None:
                lines.append(f"  • {trait}")
                continue
            if k == "strong_pos":
                desc = f"强烈{info['high']}"
                ctx = info.get("context_high", "")
            elif k == "weak_pos":
                desc = f"较倾向{info['high']}"
                ctx = ""
            elif k == "weak_neg":
                desc = f"较反对{info['high']}, 偏{info['low']}"
                ctx = ""
            else:
                desc = f"强烈{info['low']}"
                ctx = info.get("context_low", "")
            if ctx:
                lines.append(f"  • {trait} → {desc} ({ctx})")
            else:
                lines.append(f"  • {trait} → {desc}")
    return "\n".join(lines)


def load_profile_matrix(scale_csv: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Load scale_responses.csv → (sorted_trait_names, {uid → 30-dim score vector}).

    Used by persona_cosine for peer retrieval.
    """
    import csv
    with scale_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        raw[r["user_id"]][r["question_node"]] = float(r["score"])
    traits = sorted(next(iter(raw.values())).keys())
    vectors = {uid: [scores.get(t, 3.5) for t in traits] for uid, scores in raw.items()}
    return traits, vectors


def persona_cosine(v1: list[float], v2: list[float]) -> float:
    """30-dim raw Likert cosine similarity."""
    import math
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2 + 1e-8)


# ============================================================
# ③ Self Q1-Q6: designer 自身选择 + 回答 (seg-dependent)
# ============================================================

SEG_TO_SELF_QIDS = {
    "seg1": ["Q1"],
    "seg2": ["Q2"],
    "seg3": ["Q3", "Q4", "Q5", "Q6"],
}

Q_BRIEF = {
    "Q1": "面对本场景时的第一直觉反应",
    "Q2": "PKVC 认知卡片优先级排序",
    "Q3": "「数据驱动 vs 审美直觉」冲突场景",
    "Q4": "「商业利润 vs 用户隐私」冲突场景",
    "Q5": "「技术工具 vs 人文关怀」冲突场景",
    "Q6": "「商业目标 vs 社会责任」冲突场景",
}


def load_self_q1q6(task_csv: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Load task_responses.csv → {uid → {qid → {option, supplement}}} for Q1-Q6."""
    import csv
    with task_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        qid = r["question_id"]
        if qid not in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
            continue
        out[r["user_id"]][qid] = {
            "option": r["answer_value"].strip().strip('"')[:200],
            "supplement": r["text_supplement"].strip()[:200],
        }
    return dict(out)


def format_self_q1q6(self_data: dict[str, dict[str, str]], seg: str) -> str:
    """Format designer's own Q1-Q6 responses for the relevant seg.

    Mapping: seg1→Q1, seg2→Q2, seg3→Q3-Q6.
    """
    qids = SEG_TO_SELF_QIDS.get(seg, [])
    if not qids:
        return ""
    lines = []
    for qid in qids:
        entry = self_data.get(qid)
        if not entry:
            continue
        brief = Q_BRIEF.get(qid, qid)
        lines.append(f"{qid} ({brief}):")
        lines.append(f"  选择: {entry['option']}")
        if entry["supplement"]:
            lines.append(f"  你的解释: {entry['supplement']}")
    return "\n".join(lines) if lines else ""


# ============================================================
# ④ Peer Q7 同 seg retrieval (B3: cosine, B4: CAPS JS)
# ============================================================

def retrieve_cosine_peer_q7_nl(
    target_id: str,
    vectors: dict[str, list[float]],
    pool: CAPSPool,
    scenario_key: str,
    k_peers: int = 3,
) -> list[dict]:
    """B3: top-k by 30-dim cosine, return peer's Q7 NL for same segment.

    Pre-filters to peers that have the target scenario.
    """
    target_vec = vectors.get(target_id)
    if not target_vec:
        return []
    sims = []
    for did, vec in vectors.items():
        if did == target_id:
            continue
        peer = pool.designers.get(did)
        if not peer or not peer.has_scenario(scenario_key):
            continue
        s = persona_cosine(target_vec, vec)
        sims.append((did, s))
    sims.sort(key=lambda x: -x[1])
    out = []
    for peer_id, sim in sims[:k_peers]:
        seg = pool.designers[peer_id].get_scenario(scenario_key)
        out.append({
            "peer_id": peer_id,
            "similarity": sim,
            "text": seg.text[:400],
        })
    return out


# ============================================================
# B4: CAPS quad activation + typology-JS peer retrieval
# ============================================================

def load_caps_quadruples(path: Path = Path("data/processed/caps_quadruples.json")) -> list[dict]:
    """Load 36 CAPS quads from caps_quadruples.json."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_designer_scores(scale_csv: Path) -> dict[str, dict[str, float]]:
    """Load scale_responses.csv → {uid → {trait_name → raw_score}}."""
    import csv
    with scale_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        out[r["user_id"]][r["question_node"]] = float(r["score"])
    return dict(out)


# Predicate 优先级: 越小越强
PRED_PRIORITY = {
    "strongly_forces_transition_to": 0,
    "guides_transition_to": 1,
    "strictly_inhibits": 2,
    "weakly_shifts_to": 3,
}


def get_activated_quads(
    designer_scores: dict[str, float],
    quads: list[dict],
    top_k: int = 0,
) -> list[dict]:
    """Filter + rank quads aligned with designer's scale scores.

    Activation rule (same polarity boundary as LIKERT_TO_POLARITY):
      - S_direction="+" → activated when designer score >= 4 (positive polarity)
      - S_direction="-" → activated when designer score <= 3 (negative polarity)

    Ranking: predicate priority → |β| descending.
    top_k: 0 = all activated; >0 = truncate to top_k strongest.
    """
    activated = []
    for q in quads:
        score = designer_scores.get(q["S_item"])
        if score is None:
            continue
        if q["S_direction"] == "+" and score >= 4:
            activated.append(q)
        elif q["S_direction"] == "-" and score <= 3:
            activated.append(q)

    # rank by (predicate_priority asc, |β| desc)
    activated.sort(key=lambda q: (PRED_PRIORITY.get(q["P"], 9), -abs(q["S_coef"])))

    if top_k > 0:
        activated = activated[:top_k]
    return activated


def format_activated_quads_nl(activated: list[dict]) -> str:
    """Format activated quads into a NL block for the prompt."""
    if not activated:
        return "(你的量表得分未激活任何典型认知路由模式)"
    lines = []
    for q in activated:
        lines.append(f"  • [{q['qid']}-{q['option']}] {q['nl']}")
    return "\n".join(lines)


def retrieve_caps_peer_q7_nl(
    target_id: str,
    pool: CAPSPool,
    scenario_key: str,
    k_peers: int = 3,
) -> list[dict]:
    """B4: top-k by CAPS typology JS, return peer's Q7 NL for same segment.

    Pre-filters to peers with target scenario + typology dist.
    """
    target = pool.designers.get(target_id)
    if not target or not target.typology_dist:
        return []
    sims = []
    for did, peer in pool.designers.items():
        if did == target_id:
            continue
        if not peer.typology_dist or not peer.has_scenario(scenario_key):
            continue
        sim = signature_similarity(target, peer)
        if sim > 0:
            sims.append((did, sim))
    sims.sort(key=lambda x: -x[1])
    out = []
    for peer_id, sim in sims[:k_peers]:
        seg = pool.designers[peer_id].get_scenario(scenario_key)
        out.append({
            "peer_id": peer_id,
            "similarity": sim,
            "text": seg.text[:400],
        })
    return out


# ============================================================
# OG-RAG peer retrieval (ablation, greedy set cover on hypergraph)
# ============================================================

def load_user_cluster(path: Path = Path("data_v3/user_cluster_efa.csv")) -> dict[str, int]:
    """Load user_cluster_efa.csv → {user_id → cluster_id}."""
    import csv
    with path.open(encoding="utf-8-sig") as f:
        return {r["user_id"]: int(r["cluster_id"]) for r in csv.DictReader(f)}


def build_hypergraph_from_pool(pool: CAPSPool, user_cluster: dict[str, int]) -> HyperGraph:
    """Convert CAPSPool segments → extracted_records format → build_hypergraph."""
    records = []
    for did, designer in pool.designers.items():
        for scenario_key, seg in designer.segments.items():
            records.append({
                "user_id": did,
                "scenario_id": scenario_key,
                "dominant_factor": seg.dominant_factor,
                "sadr_activations": [],   # segments don't carry SADR; set cover uses factor only
                "raw_nl": seg.text,
            })
    return build_hypergraph(records, user_cluster)


def retrieve_ograg_peer_q7_nl(
    target_id: str,
    target_cluster: int,
    scenario_key: str,
    persona_pol: dict[str, list[str]] | None,
    hypergraph: HyperGraph,
    k_peers: int = 3,
) -> list[dict]:
    """OG-RAG ablation: greedy set cover retrieval → same output format as cosine/JS."""
    result = retrieve_via_og_rag(
        target_user_id=target_id,
        target_cluster=target_cluster,
        target_scenario_id=scenario_key,
        target_profile_polarity=persona_pol,
        hypergraph=hypergraph,
        k_max=k_peers,
        require_same_cluster=False,   # v3.7 default: global
    )
    out = []
    for node in result.selected_nodes[:k_peers]:
        out.append({
            "peer_id": node.designer_id,
            "similarity": 0.0,   # set cover 没有 similarity score
            "text": node.raw_nl[:400],
            "retrieval": "ograg",
            "covered": str(result.covered_elements),
        })
    return out


# ============================================================
# Prompts
# ============================================================

# ============================================================
# Per-seg structured output schemas (v1, see docs/briefs/structured_inference_3seg_schema_v1.md)
# ============================================================

_SEG_FIELDS = {
    "seg1": {  # 第一反应: anchor 单字母
        "dominant_factor": {"type": "string", "pattern": r"^[PKVC]$"},
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 3,
        },
        "criterion": {"type": ["string", "null"], "maxLength": 25},
    },
    "seg2": {  # 决策链路: chain, 强制 multi-state
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 2, "maxItems": 7,
        },
        "criterion": {"type": ["string", "null"], "maxLength": 25},
    },
    "seg3": {  # 核心 tradeoff: anchor + 底线必填
        "dominant_factor": {"type": "string", "pattern": r"^[PKVC]$"},
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 3,
        },
        "criterion": {"type": "string", "maxLength": 25},
    },
}

_COMMON_FIELDS = {
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "reasoning_notes": {"type": "string", "maxLength": 100},
    "nl_answer": {"type": "string"},
}


def build_seg_schema(seg: str) -> dict:
    """Per-seg structured output schema. Used with json_schema strict."""
    if seg not in _SEG_FIELDS:
        raise ValueError(f"unknown seg: {seg!r}")
    properties = {**_SEG_FIELDS[seg], **_COMMON_FIELDS}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


_SEG_OUTPUT_INSTR = {
    "seg1": """## 输出约束 (seg1 — 第一反应, anchor 段)
- `dominant_factor`: **单字母** P/K/V/C (第一反应不应该是推理链路, 禁 X→Y)
- `state_sequence`: 长度 1-3, anchor 段一般 [dom] 单元素
- `criterion`: 可 null (第一反应未必有明确底线宣言)
- `nl_answer`: 200-400 字, **直觉 + 短**, 不要 step-by-step 推理, 第一人称""",
    "seg2": """## 输出约束 (seg2 — 决策链路, chain 段)
- `dominant_factor`: 单字母或 X→Y path (描述链路头尾的过渡)
- `state_sequence`: **至少 2 个 state, 强制 multi-state** (chain 段). 示例 ["V","K","P","V"]
- `criterion`: 可 null
- `nl_answer`: 200-400 字, **显式展开权衡链路**, 描述你怎么在【数据】【商业要求】【用户伦理】之间走""",
    "seg3": """## 输出约束 (seg3 — 核心 tradeoff, anchor + 底线段)
- `dominant_factor`: **单字母** P/K/V/C (禁 X→Y, 底线段不是推理链)
- `state_sequence`: 长度 1-3, anchor 段
- `criterion`: **必填非 null** (≤ 25 字 verbatim 出现在 nl_answer 中, 是你底线短句的原文)
- `nl_answer`: 200-400 字, 必须含明确的"不可妥协 X / 可妥协 Y" 表达""",
}


SYSTEM_PROMPT = """你是设计师认知模拟器. 任务: 根据档案 + 场景, 模拟一个真实设计师对 Q7 某 seg 的回答.

输出格式 (严格 JSON, 先填 PKVC 结构化字段, 再写 nl_answer):
1. dominant_factor: 段主导 PKVC 状态 (P/K/V/C 或 X→Y, 按 seg 约束)
2. state_sequence: 段内 P/K/V/C 状态时序 array (长度按 seg 约束)
3. criterion: 核心底线短句 ≤ 25 字 verbatim, 必须能在 nl_answer 中找到 (seg3 必填, seg1/2 可 null)
4. confidence: 你对结构标的把握 (high/medium/low)
5. reasoning_notes: ≤ 100 字, 简述 PKVC 标的依据
6. nl_answer: 真实设计师风格 NL (第一人称, 中文, 口语化, 200-400 字)

要求:
- nl_answer 必须跟 dominant_factor + state_sequence + criterion 内部一致
- 不要在 nl_answer 里 echo dominant/criterion 字面值
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格"""


def _format_peer_q7_block(peers: list[dict], sim_label: str = "画像相似度") -> str:
    """Format peer Q7 same-seg NL block."""
    if not peers:
        return "(无相似 peer 可供参考)"
    blocks = []
    for i, p in enumerate(peers, 1):
        blocks.append(f"### Peer {i} ({sim_label}={p['similarity']:.2f})\n{p['text']}")
    return "\n\n".join(blocks)


# ---- 累加式 prompt builder (B0 → B5, 每层包含前面所有模块) ----

def build_b0_prompt(scenario_key: str) -> str:
    """B0: 仅场景 (zero-shot)."""
    seg = scenario_key.split("_")[1]
    return f"""# 场景
{Q7_BACKGROUND}

# 任务
{SEG_TASKS[seg]}

{_SEG_OUTPUT_INSTR[seg]}"""


def build_b1_prompt(scenario_key: str, demo_profile: str) -> str:
    """B1 = ①pi."""
    seg = scenario_key.split("_")[1]
    return f"""# 你的人口统计学信息
{demo_profile}

# 场景
{Q7_BACKGROUND}

# 任务
{SEG_TASKS[seg]}

{_SEG_OUTPUT_INSTR[seg]}"""


def build_b2_prompt(
    scenario_key: str, demo_profile: str, persona_pol: dict[str, list[str]],
) -> str:
    """B2 = ①pi + ②30维."""
    seg = scenario_key.split("_")[1]
    persona_block = polarity_to_nl(persona_pol)
    return f"""# 你的人口统计学信息
{demo_profile}

# 你的设计信念档案 (30 维 polarity-aware)
{persona_block}

# 场景
{Q7_BACKGROUND}

# 任务
{SEG_TASKS[seg]}

{_SEG_OUTPUT_INSTR[seg]}"""


def build_b3_prompt(
    scenario_key: str, demo_profile: str,
    persona_pol: dict[str, list[str]], self_block: str,
) -> str:
    """B3 = ①pi + ②30维 + ③self Q1-Q6."""
    seg = scenario_key.split("_")[1]
    persona_block = polarity_to_nl(persona_pol)
    parts = [
        f"# 你的人口统计学信息\n{demo_profile}",
        f"# 你的设计信念档案 (30 维 polarity-aware)\n{persona_block}",
    ]
    if self_block:
        parts.append(f"# 你在其他设计场景的选择与回答\n\n{self_block}")
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASKS[seg]}")
    parts.append(_SEG_OUTPUT_INSTR[seg])
    return "\n\n".join(parts)


def build_b4_prompt(
    scenario_key: str, demo_profile: str,
    persona_pol: dict[str, list[str]], self_block: str,
    peers_q7: list[dict],
) -> str:
    """B4 = ①pi + ②30维 + ③self + ④peer-Q7(cosine)."""
    seg = scenario_key.split("_")[1]
    persona_block = polarity_to_nl(persona_pol)
    peer_block = _format_peer_q7_block(peers_q7, "画像相似度")
    parts = [
        f"# 你的人口统计学信息\n{demo_profile}",
        f"# 你的设计信念档案 (30 维 polarity-aware)\n{persona_block}",
    ]
    if self_block:
        parts.append(f"# 你在其他设计场景的选择与回答\n\n{self_block}")
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASKS[seg]}")
    parts.append(f"# 跟你画像相似的设计师在本场景的回答 (供参考)\n\n{peer_block}")
    parts.append(_SEG_OUTPUT_INSTR[seg])
    return "\n\n".join(parts)


def build_b5_prompt(
    scenario_key: str, demo_profile: str,
    persona_pol: dict[str, list[str]], self_block: str,
    activated_quads_nl: str, peers_q7: list[dict],
) -> str:
    """B5 = ①pi + ②30维 + ③self + ⑤CAPS路由 + ④peer-Q7(JS)."""
    seg = scenario_key.split("_")[1]
    persona_block = polarity_to_nl(persona_pol)
    peer_block = _format_peer_q7_block(peers_q7, "认知模式相似度")
    parts = [
        f"# 你的人口统计学信息\n{demo_profile}",
        f"# 你的设计信念档案 (30 维 polarity-aware)\n{persona_block}",
        f"# 你的认知路由模式 (CAPS 四元组)\n\n"
        f"以下是基于你的量表得分激活的认知路由模式 — 描述你在不同设计冲突情境下的思维转向倾向:\n\n"
        f"{activated_quads_nl}",
    ]
    if self_block:
        parts.append(f"# 你在其他设计场景的选择与回答\n\n{self_block}")
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASKS[seg]}")
    parts.append(f"# 跟你认知模式相似的设计师在本场景的回答 (供参考)\n\n{peer_block}")
    parts.append(_SEG_OUTPUT_INSTR[seg])
    return "\n\n".join(parts)


# ============================================================
# Inference
# ============================================================

async def call_one(client, baseline: str, designer_id: str, scenario_key: str, user_prompt: str, model: str):
    seg = scenario_key.split("_")[1]  # seg1 / seg2 / seg3
    schema = build_seg_schema(seg)
    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=900,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"q7_{seg}_inference",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            content = r.choices[0].message.content
            if not content:
                raise ValueError("empty content")
            parsed = json.loads(content)
            return {
                "ok": True,
                "baseline": baseline,
                "designer_id": designer_id,
                "scenario": scenario_key,
                "claimed_pkvc": {
                    "dominant_factor": parsed["dominant_factor"],
                    "state_sequence": parsed["state_sequence"],
                    "criterion": parsed["criterion"],
                    "confidence": parsed["confidence"],
                    "reasoning_notes": parsed["reasoning_notes"],
                },
                "simulated_nl": parsed["nl_answer"].strip(),    # 兼容旧 schema 字段名
                "prompt_chars": len(user_prompt),
            }
        except Exception as e:
            if attempt == 2:
                return {"ok": False, "baseline": baseline, "designer_id": designer_id,
                        "scenario": scenario_key, "error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(1 + attempt)


async def run_all(jobs: list[dict], model: str, concurrency: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(idx, job):
        async with sem:
            if idx % 50 == 0:
                print(f"[{idx}/{len(jobs)}] {job['baseline']} {job['designer_id'][:10]} {job['scenario']}", flush=True)
            return await call_one(client, job["baseline"], job["designer_id"], job["scenario"],
                                  job["prompt"], model)

    tasks = [asyncio.create_task(_wrap(i + 1, j)) for i, j in enumerate(jobs)]
    return await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="number of designers to sample (0=all)")
    ap.add_argument("--baselines", default="B0,B1,B2,B3,B4,B5")
    ap.add_argument("--caps-topk", default="5",
                    help="comma-sep top-k for CAPS quads in B5, e.g. '3,5,8,0'. 0=all.")
    ap.add_argument("--peer-abl", default="",
                    help="extra peer ablation: 'ograg' adds B5_ograg variant. comma-sep.")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default="reports/datav3/baselines_q7_inference.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retry-from-fails", default="",
                    help="jsonl path with previous run; only retry rows where ok=False "
                         "(filtered by baseline+designer_id+scenario triple)")
    args = ap.parse_args()

    baselines = args.baselines.split(",")
    valid_bl = {"B0", "B1", "B2", "B3", "B4", "B5"}
    for b in baselines:
        if b not in valid_bl:
            print(f"ERROR: unknown baseline {b!r}, valid: {sorted(valid_bl)}")
            return 1

    # B5 CAPS top-k variants: "3,5,8,0" → [3,5,8,0] (0=all)
    caps_topk_list = [int(x) for x in args.caps_topk.split(",")]
    # Expand B5 → B5_k3, B5_k5, ... (single value stays as "B5")
    b5_variants = []
    if "B5" in baselines:
        for k in caps_topk_list:
            label = f"B5_k{k}" if len(caps_topk_list) > 1 else "B5"
            b5_variants.append((label, k))

    # Peer ablation variants
    peer_abls = [x.strip() for x in args.peer_abl.split(",") if x.strip()]
    need_ograg = "ograg" in peer_abls and "B5" in baselines

    print(f"[config] baselines={baselines} model={args.model} concurrency={args.concurrency}")
    if b5_variants:
        print(f"  B5 CAPS top-k variants: {[(l,k) for l,k in b5_variants]}")
    if need_ograg:
        print(f"  peer ablation: +B5_ograg")

    need_pi   = any(b in baselines for b in ("B1","B2","B3","B4","B5"))
    need_30d  = any(b in baselines for b in ("B2","B3","B4","B5"))
    need_self = any(b in baselines for b in ("B3","B4","B5"))
    need_cos  = "B4" in baselines
    need_caps = "B5" in baselines

    print("[load] CAPSPool + persona + demographics + profile matrix...")
    pool = load_caps_pool()
    scale_csv = Path("data_v3/scale_responses.csv")
    persona = load_designer_personas(scale_csv) if need_30d else {}
    demo = load_designer_demographics(Path("data_v3/users.csv")) if need_pi else {}
    trait_names, vectors = load_profile_matrix(scale_csv) if (need_cos or need_30d) else ([], {})
    print(f"  {len(pool.designers)} designers in pool, {len(persona)} have persona, "
          f"{len(demo)} have demo, {len(vectors)} have 30-dim vectors")

    # ③ Self Q1-Q6 data
    self_q1q6_data: dict[str, dict[str, dict[str, str]]] = {}
    if need_self:
        self_q1q6_data = load_self_q1q6(Path("data_v3/task_responses.csv"))
        print(f"  self Q1-Q6: {len(self_q1q6_data)} designers")

    # ⑤ B5-specific: CAPS quads + raw scores
    caps_quads: list[dict] = []
    designer_scores: dict[str, dict[str, float]] = {}
    if need_caps:
        caps_quads = load_caps_quadruples()
        designer_scores = load_designer_scores(scale_csv)
        print(f"  B5: {len(caps_quads)} CAPS quads, {len(designer_scores)} designers with scores")

    # OG-RAG ablation: hypergraph + cluster
    user_cluster: dict[str, int] = {}
    hg: HyperGraph | None = None
    if need_ograg:
        user_cluster = load_user_cluster()
        hg = build_hypergraph_from_pool(pool, user_cluster)
        stats = hg.stats()
        print(f"  OG-RAG hypergraph: {stats['n_nodes']} nodes, {stats['n_hyperedges']} edges, "
              f"{stats['n_designers']} designers, avg edge size {stats['avg_edge_size']:.1f}")

    # eligible designers (have Q7 + persona)
    eligible = [did for did in pool.designers
                if persona.get(did) and any(pool.designers[did].has_scenario(f"Q7_{s}") for s in ("seg1","seg2","seg3"))]
    print(f"  {len(eligible)} eligible designers (have Q7 + persona)")
    if args.n > 0:
        rng = random.Random(args.seed)
        rng.shuffle(eligible)
        eligible = eligible[:args.n]
        print(f"  sampling {len(eligible)} designers")

    # build all jobs
    jobs = []
    for did in eligible:
        pol = persona.get(did, {})
        demo_profile = demo.get(did, "")
        self_data = self_q1q6_data.get(did, {})

        # B5: CAPS quads per top-k variant (per-designer, 跨 seg 共用)
        b5_quads_nl_map: dict[str, str] = {}  # label → NL block
        if need_caps:
            scores_d = designer_scores.get(did, {})
            for label, topk in b5_variants:
                activated = get_activated_quads(scores_d, caps_quads, top_k=topk)
                b5_quads_nl_map[label] = format_activated_quads_nl(activated)

        for seg in ("seg1", "seg2", "seg3"):
            scenario = f"Q7_{seg}"
            if not pool.designers[did].has_scenario(scenario):
                continue

            # ③ self Q1-Q6 (per-seg)
            self_block = format_self_q1q6(self_data, seg) if self_data else ""

            # ④ peer Q7 同 seg (per-seg, B4 cosine / B5 JS / B5_ograg set cover)
            b4_peers = retrieve_cosine_peer_q7_nl(did, vectors, pool, scenario) if need_cos else []
            b5_peers = retrieve_caps_peer_q7_nl(did, pool, scenario) if need_caps else []
            ograg_peers: list[dict] = []
            if need_ograg and hg is not None:
                cid = user_cluster.get(did, 0)
                ograg_peers = retrieve_ograg_peer_q7_nl(did, cid, scenario, pol, hg)

            for b in baselines:
                if b == "B0":
                    prompt = build_b0_prompt(scenario)
                elif b == "B1":
                    if not demo_profile:
                        continue
                    prompt = build_b1_prompt(scenario, demo_profile)
                elif b == "B2":
                    if not demo_profile:
                        continue
                    prompt = build_b2_prompt(scenario, demo_profile, pol)
                elif b == "B3":
                    if not demo_profile:
                        continue
                    prompt = build_b3_prompt(scenario, demo_profile, pol, self_block)
                elif b == "B4":
                    if not demo_profile:
                        continue
                    prompt = build_b4_prompt(scenario, demo_profile, pol, self_block, b4_peers)
                elif b == "B5":
                    if not demo_profile:
                        continue
                    # B5 top-k CAPS variants (JS peer)
                    for label, _ in b5_variants:
                        prompt = build_b5_prompt(scenario, demo_profile, pol, self_block,
                                                b5_quads_nl_map[label], b5_peers)
                        jobs.append({"baseline": label, "designer_id": did,
                                     "scenario": scenario, "prompt": prompt})
                    # B5_ograg ablation (OG-RAG peer, default CAPS top-k)
                    if need_ograg and ograg_peers:
                        default_topk = caps_topk_list[0]
                        caps_nl = b5_quads_nl_map.get(b5_variants[0][0], "") if b5_variants else ""
                        prompt = build_b5_prompt(scenario, demo_profile, pol, self_block,
                                                caps_nl, ograg_peers)
                        jobs.append({"baseline": "B5_ograg", "designer_id": did,
                                     "scenario": scenario, "prompt": prompt})
                    continue
                else:
                    continue
                jobs.append({"baseline": b, "designer_id": did, "scenario": scenario, "prompt": prompt})

    print(f"[build] {len(jobs)} inference jobs total")

    # --retry-from-fails: filter jobs to only failed (baseline, designer_id, scenario) triples
    if args.retry_from_fails:
        prev = [json.loads(l) for l in Path(args.retry_from_fails).open(encoding="utf-8") if l.strip()]
        fails = {(r["baseline"], r["designer_id"], r["scenario"]) for r in prev if not r.get("ok")}
        print(f"[retry] loaded {len(prev)} prev rows, {len(fails)} failed; filtering jobs...")
        jobs = [j for j in jobs if (j["baseline"], j["designer_id"], j["scenario"]) in fails]
        print(f"[retry] {len(jobs)} retry jobs after filter")
        if not jobs:
            print("nothing to retry"); return 0

    t0 = time.time()
    results = asyncio.run(run_all(jobs, args.model, args.concurrency))
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    # attach ground truth + persona metadata
    for r in results:
        if not r["ok"]:
            continue
        did = r["designer_id"]
        target = pool.designers[did]
        seg = target.get_scenario(r["scenario"])
        if seg:
            r["ground_truth_nl"] = seg.text
            r["ground_truth_pkvc"] = {
                "dominant": seg.dominant_factor,
                "sequence": seg.state_sequence,
                "criterion": seg.criterion,
                "confidence": seg.confidence,
            }
        pol = persona.get(did, {})
        r["persona_trait_count"] = sum(len(v) for v in pol.values())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    # summary per baseline
    from collections import Counter
    all_labels = sorted(set(r.get("baseline", "?") for r in results))
    print("\n=== per-baseline summary ===")
    for b in all_labels:
        b_r = [r for r in results if r.get("baseline") == b]
        ok_r = [r for r in b_r if r["ok"]]
        print(f"  {b}: {len(ok_r)}/{len(b_r)} ok, avg prompt chars: "
              f"{sum(r.get('prompt_chars', 0) for r in ok_r) / max(len(ok_r), 1):.0f}")
        if ok_r:
            avg_out = sum(len(r["simulated_nl"]) for r in ok_r) / len(ok_r)
            print(f"    avg simulated_nl chars: {avg_out:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
