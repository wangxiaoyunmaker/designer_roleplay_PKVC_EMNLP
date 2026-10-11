"""B6 修订版 sample inference: 20 designer × 3 seg = 60 calls.

B6 prompt 模块 (5 个, 砍 ④ peer + ⑥ criterion vs B5):
  - PKVC 4 字母含义 (新)
  - ① pi (5 高区分度字段)
  - ② 30维 stance labels (strong only, 来自 STANCE_LABELS)
  - ③ self Q3-Q6 (面对冲突 + 选择 + 解释, **无 PKVC 映射**)
  - ⑤ CAPS quad 4 类平衡 top-1 per class + 1 overall

Sample: 12 seg3 GT 非 V (7P+4K+1C, 关键测试) + 8 random GT=V (sanity).

Goal: 看 macro-F1 vs B5 是否改善 seg3 P/K/C recall (B5 现 0%).
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

# Reuse existing helpers
from scripts.run_3_baselines_q7 import (
    Q7_BACKGROUND, SEG_TASKS, _SEG_OUTPUT_INSTR, build_seg_schema,
    load_designer_demographics, load_designer_personas, load_designer_scores,
    load_self_q1q6, load_caps_quadruples, load_caps_pool,
    get_activated_quads, format_activated_quads_nl,    # v2_clean: B5-style CAPS top-k
    retrieve_caps_peer_q7_nl, _format_peer_q7_block,    # peer retrieval (B6_h)
)


# ============================================================
# 20 designer sample (12 non-V seg3 GT + 8 random V)
# ============================================================
SAMPLE_DESIGNERS = [
    # non-V seg3 GT (P=7, K=4, C=1)
    "1078765532", "695475971", "BuzhichunD", "F29293918",
    "Vivian11320", "acho20050223", "faymeng666",     # P (7)
    "AyiWe4Real", "Hreiy888", "cccclare6354", "kuronim",  # K (4)
    "Yyd3144117584",                                  # C (1)
    # random V seg3 GT (8)
    "WuShanYuXi", "wxid_negbygwl2lw522", "mz18757140975", "wxid_wqyptcofey4a22",
    "a123164981", "WUXY0953", "ww2001811920", "Jimmy_Fang18",
]


# ============================================================
# PKVC 4 letter definitions (system prompt prefix)
# ============================================================

PKVC_DEF = """# PKVC 4 类认知状态 (设计师思考过程的 4 种基本状态)
- P (Problem) 问题界定 / 重定义: 质疑题目本身, 重新框定问题, 拆解子问题
  - P1 界定核心 / P2 拆解子问题 / P3 推翻假定 (或重定义角色边界)
- K (Knowledge) 知识调用 / 数据驱动: 调研, 找案例, 用数据决策
  - K1 调用经验 / K2 客观数据 / K3 评估技术栈
- V (Value) 价值立场 / 底线宣言: 设计师道德判断, 用户优先, 底线声明
  - V1 价值立场 / V2 利益排序 / V3 设立原则
- C (Concept) 概念整合 / 抽象框架: 涌现画面, 整合方案, 提炼原则
  - C1 涌现画面 / C2 整合构建 / C3 提炼原则"""


_BASE_SYSTEM_NO_PKVC_DEF = """你是设计师认知模拟器. 任务: 根据档案 + 场景, 模拟一个真实设计师对 Q7 某 seg 的回答.

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
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格
"""

_BASE_SYSTEM = f"""你是设计师认知模拟器. 任务: 根据档案 + 场景, 模拟一个真实设计师对 Q7 某 seg 的回答.

{PKVC_DEF}

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
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格
"""

# v1: 强 anti-V (P/C 100% recall but V 牺牲)
SEG3_RULE_V1 = """
- **关键**: seg3 的 dominant_factor **不应默认是 V**. 看你的核心立场 + CAPS routing 强度,
  P 类信号最强就标 P, K 强就标 K, 不要因为 V 类例子常见而默认 V."""

# v2: 软化 anti-V + β 速览 (V 回升 but P 崩)
SEG3_RULE_V2 = """
## seg3 dominant_factor 判断规则 (重要):
- 看 CAPS routing 4 类 (P/K/V/C) 的**正向激活强度** (β > 0, "推动" 类; β < 0 是 inhibit 类排除)
- 哪类的 β 正向最大 → 你的底线属于那类:
  * P 强 → 底线是 "重新定义问题边界 / 推翻题目假定"
  * K 强 → 底线是 "数据先行 / 调研真实需求 / 技术可行性评估" (不是道德底线!)
  * V 强 → 底线是 "用户优先 / 设立专业立场原则" (常见的道德声明类底线)
  * C 强 → 底线是 "整合方案 / 提炼抽象框架"
- 若 4 类 β 强度近似, 看 ③ 你在 Q3-Q6 选择倾向哪类
- 不要无脑选 V (V 例子常见), 也不要 over-correct 到 P (K/C 强时该选 K/C)"""

# v3: 分级 + 兼顾 P 救回 + V 回稳 (拿 v1 的 P 信号 + v2 的 V 平衡)
SEG3_RULE_V3 = """
## seg3 dominant_factor 判断规则 (分级决策):
4 类 GT 分布 prior (162 designer 上的统计): V ≈ 75%, P ≈ 10%, K ≈ 10%, C ≈ 5%.
关键: V 是 majority 但 P/K/C 占 25%, 你可能就是其中之一.

判断顺序 (按 routing β 强度 + ③ 实际行为):
1. **C 类**: 若 C 类 β > +0.5 且其他类 β < +0.3 → 选 C (强且独占信号)
2. **P 类**: 若 ③ 中 ≥ 2 个 Q 选了 "重新质疑 / 拒绝 framing / 推翻假定" (例如 Q3-D, Q4-D 等),
   就选 P, 即便 P 类 β 不是最大 (P 信号在 trait 数据上常被低估)
3. **K 类**: 若 K 类 β 正向最强 (> +0.3) 且 ③ 中有 "调研数据 / 技术评估" 选择 → 选 K
4. **V 类 (default)**: 上面都不触发时, 选 V (V 是统计 majority + 道德声明常见)

不要 over-correct: P/K/C 信号要清晰才选, 不清晰就 V."""

SYSTEM_PROMPT_BY_VARIANT = {
    "v1": _BASE_SYSTEM + SEG3_RULE_V1,
    "v2": _BASE_SYSTEM + SEG3_RULE_V2,
    "v3": _BASE_SYSTEM + SEG3_RULE_V3,
    # v2_clean: v2 minus 3 bias 注入点 (无 seg3 rule, CAPS 改 top-k strength, 无 β 速览 + 无 anti-V footer)
    # 保留: PKVC 4 字母定义 + 5 字段 pi + stance label format + self 面对【冲突】格式
    "v2_clean": _BASE_SYSTEM,
    # === component ablation variants (anchor everywhere, 单点改动 vs v2) ===
    # Ablation A: v2 minus PKVC 4 字母定义 (SYSTEM 顶部块删)
    "v2_no_pkvc_def": _BASE_SYSTEM_NO_PKVC_DEF + SEG3_RULE_V2,
    # Ablation B: v2 minus seg3 decision rule (SYSTEM 内 SEG3_RULE_V2 删)
    "v2_no_seg3_rule": _BASE_SYSTEM,
    # Ablation C: v2 minus β 速览 header (CAPS quad block 内 header 删, 通过 format function 处理)
    "v2_no_beta_summary": _BASE_SYSTEM + SEG3_RULE_V2,
}

# v2_hybrid: per-seg prompt strategy
#   - seg1 / seg2: 用 v2 (β 速览 + seg3 rule + 4 类平衡 CAPS) — push chain reasoning
#   - seg3: 用 v2_clean (无 rule + top-k strength CAPS) — V-default majority match
def get_system_prompt_for_seg(variant: str, seg: str) -> str:
    if variant == "v2_hybrid":
        return SYSTEM_PROMPT_BY_VARIANT["v2"] if seg in ("seg1", "seg2") else SYSTEM_PROMPT_BY_VARIANT["v2_clean"]
    return SYSTEM_PROMPT_BY_VARIANT[variant]


def get_caps_strategy_for_seg(variant: str, seg: str) -> str:
    """Returns 'balanced' (4 class top-1 + β 速览) or 'topk' (top-k strength, no header)."""
    if variant == "v2_hybrid":
        return "balanced" if seg in ("seg1", "seg2") else "topk"
    return "topk" if variant == "v2_clean" else "balanced"

# Backward-compat default for any external imports
SYSTEM_PROMPT_B6 = SYSTEM_PROMPT_BY_VARIANT["v2"]


# ============================================================
# 30 维立场标签映射表 (single source of truth, B6 doc §9.1.1)
# ============================================================

STANCE_LABELS = {
    "数据 vs 直觉": {"high": "重直觉立场", "low": "重数据立场"},
    "天赋信仰": {"high": "天赋决定论", "low": "后天努力论"},
    "AI 替代论": {"high": "AI 替代主义", "low": "岗位保护主义"},
    "技术支持": {"high": "技术激进主义", "low": "技术审慎主义"},
    "复杂工具信仰": {"high": "工具门槛论", "low": "工具无关论"},
    "混沌与灵感": {"high": "包容失控派", "low": "验证至上派"},
    "黑盒透明度": {"high": "结果至上派", "low": "过程透明派"},
    "系统微模块": {"high": "解耦优先", "low": "整体感优先"},
    "完美主义/MVP": {"high": "完美主义", "low": "速度优先"},
    "创新代价": {"high": "创新激进派", "low": "惯例守护派"},
    "审美反民粹": {"high": "审美精英主义", "low": "数据民粹妥协"},
    "情感附加值": {"high": "仪式感优先", "low": "效率优先"},
    "装饰合法性": {"high": "装饰合法派", "low": "形式追随功能"},
    "时代 vs 永恒": {"high": "时代潮流派", "low": "永恒经典派"},
    "本土文化妥协": {"high": "本土深耕派", "low": "全球通用派"},
    "仿生与有机": {"high": "有机自然派", "low": "几何工业派"},
    "独裁 vs 委员会": {"high": "独裁愿景派", "low": "民主共创派"},
    "秩序与防错": {"high": "防错约束派", "low": "探索开放派"},
    "创作者署名": {"high": "作者性优先", "low": "商业去人格化"},
    "资产开源": {"high": "开源道德派", "low": "竞争壁垒派"},
    "封闭生态竞争": {"high": "封闭生态派", "low": "开放兼容派"},
    "风险规避": {"high": "风险规避", "low": "风险偏好"},
    "团队依恋": {"high": "独立打磨派", "low": "团队共创派"},
    "商业操纵伦理": {"high": "商业操纵容忍", "low": "用户伦理底线"},
    "数据隐私交换": {"high": "体验换隐私", "low": "隐私保护派"},
    "弱势群体优先": {"high": "边缘群体优先", "low": "多数效率优先"},
    "股东服务优先": {"high": "股东忠诚派", "low": "用户忠诚派"},
    "政治中立性": {"high": "政治中立派", "low": "政治表达派"},
    "系统责任归属": {"high": "系统归责派", "low": "用户归责派"},
    "救世主情结": {"high": "救世主情结", "low": "工具中立派"},
}


def polarity_to_stance_block(scores: dict[str, float]) -> str:
    """Score >= 5 → high stance, score <= 2 → low stance, drop weak (3-4)."""
    lines = []
    for node, score in scores.items():
        if node not in STANCE_LABELS:
            continue
        if score >= 5:
            lines.append(f"  • {STANCE_LABELS[node]['high']} ({node}维度)")
        elif score <= 2:
            lines.append(f"  • {STANCE_LABELS[node]['low']} ({node}维度, 反向立场)")
    return "\n".join(lines) if lines else "  (无强 stance, 量表大部分为中间值)"


# ============================================================
# Option-trait mapping (for ③ self Q1-Q6 conflict labels)
# ============================================================

def load_option_trait_mapping(path: Path = Path("reports/datav3/per_designer/option_trait_mapping.json")) -> dict:
    """Load {qid_option → {C_conflict_axes, ...}}."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# Hand-crafted human-readable conflict labels (from C_raw + C_conflict_axes mining)
# Maps qid → list of 2-3 conflict axis labels (human readable form for 面对【XX 冲突】)
Q_CONFLICT_LABELS = {
    "Q3": ["数据效用与美感形态冲突", "数据效用与设计师立场冲突"],
    "Q4": ["商业利润与用户隐私冲突", "隐私交换与概念建构冲突"],
    "Q5": ["AI 工具与设计师自主性冲突", "人机协作分工冲突"],
    "Q6": ["商业目标与社会责任冲突", "利益与责任感冲突"],
}


def format_self_q3q6_b6(self_data: dict[str, dict[str, str]]) -> str:
    """Format Q3-Q6 self block in B6 style: 面对【冲突】+ 选择 + 解释 (无 PKVC 映射)."""
    lines = []
    for qid in ("Q3", "Q4", "Q5", "Q6"):
        entry = self_data.get(qid)
        if not entry:
            continue
        conflicts = Q_CONFLICT_LABELS.get(qid, [])
        conflict_str = "".join(f"【{c}】" for c in conflicts)
        lines.append(f"{qid} 面对{conflict_str}")
        choice = entry.get("option", "").strip()
        if choice:
            lines.append(f'  选择: "{choice[:120]}"')
        supplement = entry.get("supplement", "").strip()
        if supplement:
            lines.append(f'  你的解释: "{supplement[:80]}"')
    return "\n".join(lines) if lines else ""


# ============================================================
# CAPS quad 4 类平衡 (top-1 per class + 1 overall top, B6 doc §3)
# ============================================================

PRED_PRIORITY = {
    "strongly_forces_transition_to": 0,
    "guides_transition_to": 1,
    "strictly_inhibits": 2,
    "weakly_shifts_to": 3,
}


def get_activated_quads_balanced(
    designer_scores: dict[str, float],
    quads: list[dict],
) -> list[dict]:
    """4 类平衡 selection: top-1 per (P/K/V/C) + 1 overall top.

    Returns 4-5 quads with PKVC class diversity. Falls back gracefully if a class has no
    activated quads.
    """
    by_class = defaultdict(list)
    for q in quads:
        score = designer_scores.get(q["S_item"])
        if score is None:
            continue
        # Activation rule: direction + and score >= 4, or - and score <= 3
        if q["S_direction"] == "+" and score >= 4:
            pass
        elif q["S_direction"] == "-" and score <= 3:
            pass
        else:
            continue
        # Use |coef| as strength (predicate priority breaks tie)
        strength = abs(q.get("S_coef", 0.0))
        target_letter = q["O"][0] if q.get("O") else "?"
        if target_letter in ("P", "K", "V", "C"):
            by_class[target_letter].append((strength, q))

    chosen = []
    for cls in ("P", "K", "V", "C"):
        if by_class[cls]:
            top = max(by_class[cls], key=lambda x: (x[0], -PRED_PRIORITY.get(x[1].get("P",""), 9)))
            chosen.append({**top[1], "_strength": top[0], "_class_anchor": cls})

    # Add 1 absolute top (any class) if not already in chosen
    all_quads = [(s, q) for cls in ("P","K","V","C") for s, q in by_class[cls]]
    if all_quads:
        top_overall = max(all_quads, key=lambda x: x[0])
        if not any(c.get("id") == top_overall[1].get("id") for c in chosen):
            chosen.append({**top_overall[1], "_strength": top_overall[0], "_class_anchor": "overall_top"})
    return chosen


def format_caps_quad_balanced_nl(quads: list[dict], variant: str = "v2") -> str:
    """Format 4-5 balanced quads with class headers.

    variant:
      - v1: 仅 4 类 routing 描述 + footer (无 β 速览)
      - v2_no_beta_summary: 4 类 routing 描述, 无 β 速览 header (ablation C)
      - v2/v3/其他: 加 β 速览 header + 详细 routing
    """
    if not quads:
        return "(你的量表得分未激活任何典型认知路由模式)"
    by_anchor = {}
    for q in quads:
        by_anchor[q["_class_anchor"]] = q

    if variant == "v2_no_beta_summary":
        # Ablation C: 4 类 routing 描述, 跳过 β 速览 header
        lines = []
        for cls in ("P", "K", "V", "C"):
            if cls in by_anchor:
                q = by_anchor[cls]
                lines.append(f"### {cls} 类 top routing (target = {cls})")
                lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")
                lines.append("")
            else:
                lines.append(f"### {cls} 类 top routing")
                lines.append(f"(该 designer 在 {cls} 类没有激活的 CAPS routing — 信号弱)")
                lines.append("")
        if "overall_top" in by_anchor:
            q = by_anchor["overall_top"]
            lines.append(f"### Overall strongest routing (跨 PKVC, |β|={q['_strength']:.3f})")
            lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")
        return "\n".join(lines)

    if variant == "v1":
        # v1 原始格式: 4 类 routing 描述 + footer (无 β 速览)
        lines = []
        for cls in ("P", "K", "V", "C"):
            if cls in by_anchor:
                q = by_anchor[cls]
                lines.append(f"## {cls} 类 top routing (target = {cls})")
                lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")
                lines.append("")
            else:
                lines.append(f"## {cls} 类 top routing")
                lines.append(f"(该 designer 在 {cls} 类没有激活的 CAPS routing — 信号弱)")
                lines.append("")
        if "overall_top" in by_anchor:
            q = by_anchor["overall_top"]
            lines.append(f"## Overall strongest routing (跨 PKVC, |β|={q['_strength']:.3f})")
            lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")
        lines.append("")
        lines.append("注意: 这 4-5 条 routing 展示你在不同 PKVC 类上的最强信号. seg3 的底线类型应反映")
        lines.append("你最强信号的类别, 不是默认 V.")
        return "\n".join(lines)

    # v2/v3: β 强度速览 (header) + 详细 routing
    lines = ["## 4 类 routing 强度速览 (β 值, 正向 → 推动该状态; 负向 → 抑制该状态)"]
    summary_rows = []
    for cls in ("P", "K", "V", "C"):
        if cls in by_anchor:
            q = by_anchor[cls]
            coef = q.get("S_coef", 0.0)
            sign = "正向激活" if coef > 0 else "负向抑制"
            summary_rows.append((cls, coef, sign, q))
        else:
            summary_rows.append((cls, 0.0, "无激活", None))

    for cls, coef, sign, _ in summary_rows:
        lines.append(f"  • {cls} 类: β = {coef:+.3f} ({sign})")
    # Identify strongest positive class
    positive = [(cls, coef) for cls, coef, sign, _ in summary_rows if coef > 0]
    if positive:
        top_pos = max(positive, key=lambda x: x[1])
        lines.append(f"\n→ **正向激活最强**: {top_pos[0]} 类 (β = {top_pos[1]:+.3f}). 优先考虑此类作为你的底线类型.")
    else:
        lines.append(f"\n→ 4 类均为负向抑制 (无正向 routing 激活). 看 ③ 你在 Q3-Q6 上的实际选择倾向哪类.")
    lines.append("")
    lines.append("---")
    lines.append("详细 routing 描述:")
    lines.append("")

    for cls in ("P", "K", "V", "C"):
        if cls in by_anchor:
            q = by_anchor[cls]
            lines.append(f"### {cls} 类 top routing (target = {cls})")
            lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")
            lines.append("")
        else:
            lines.append(f"### {cls} 类 top routing")
            lines.append(f"(该 designer 在 {cls} 类没有激活的 CAPS routing — 信号弱)")
            lines.append("")

    if "overall_top" in by_anchor:
        q = by_anchor["overall_top"]
        lines.append(f"### Overall strongest routing (跨 PKVC, |β|={q['_strength']:.3f})")
        lines.append(f"[{q['qid']}-{q['option']}] {q['nl']}")

    return "\n".join(lines)


# ============================================================
# Pi 5-field demographics (subset, B6 §9.0)
# ============================================================

DEMO_HIGH_DISCRIM_FIELDS = ["identity", "years_exp", "design_field", "work_context", "project_role"]
DEMO_FIELD_LABELS = {
    "identity": "身份", "years_exp": "经验", "design_field": "领域",
    "work_context": "工作场景", "project_role": "项目角色",
}


def load_pi_5(users_csv: Path) -> dict[str, str]:
    """5 高区分度字段 pi (vs 全 11 字段)."""
    import pandas as pd
    df = pd.read_csv(users_csv)
    df["wechat_id"] = df["wechat_id"].astype(str)
    out = {}
    for _, row in df.iterrows():
        parts = []
        for fld in DEMO_HIGH_DISCRIM_FIELDS:
            val = row.get(fld)
            if val is None or (isinstance(val, float) and val != val):
                continue
            val = str(val).strip()
            if not val or val.lower() == "nan":
                continue
            parts.append(f"{DEMO_FIELD_LABELS[fld]}: {val}")
        if parts:
            out[row["wechat_id"]] = " | ".join(parts)
    return out


# ============================================================
# B6 prompt builder
# ============================================================

def build_b6_prompt(
    scenario_key: str,
    pi_block: str,
    stance_block: str,
    self_block: str,
    caps_quad_block: str,
    variant: str = "v2",
    peer_block: str = "",   # 非空时, 在 task 后加 peer block
    modules: set | None = None,   # subset of {pi, stance, self, caps}; None = all
) -> str:
    seg = scenario_key.split("_")[1]
    if modules is None:
        modules = {"pi", "stance", "self", "caps"}
    parts = []
    if "pi" in modules and pi_block:
        parts.append(f"# 你的人口统计学信息 (5 高区分度字段)\n{pi_block}")
    if "stance" in modules and stance_block:
        parts.append(f"# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)\n{stance_block}")
    if "self" in modules and self_block:
        parts.append(f"# 你在 4 个设计冲突场景的实际行为 (原始数据)\n{self_block}")
    if "caps" in modules and caps_quad_block:
        if variant == "v2_clean":
            caps_header = "# 你的认知路由模式 (CAPS 四元组, 激活的 top-5 by strength)"
        else:
            caps_header = "# 你的认知路由模式 (CAPS 四元组, PKVC 4 类平衡 top-1 per class + 1 overall top)"
        parts.append(f"{caps_header}\n\n{caps_quad_block}")
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASKS[seg]}")
    if peer_block:
        parts.append(f"# 跟你认知模式相似的设计师在本场景的回答 (供参考)\n\n{peer_block}")
    parts.append(_SEG_OUTPUT_INSTR[seg])
    return "\n\n".join(parts)


# ============================================================
# Inference (gpt-4o via yunwu.ai)
# ============================================================

def supports_strict_schema(model: str) -> bool:
    """OpenAI-family models support json_schema strict; others use json_object."""
    m = model.lower()
    return m.startswith(("gpt-4", "gpt-3.5", "o1", "o3")) and "claude" not in m and "gemini" not in m


def get_response_format_and_aug_system(model: str, system_prompt: str, schema: dict, schema_name: str):
    """Returns (response_format dict, possibly_augmented_system_prompt)."""
    if supports_strict_schema(model):
        return (
            {"type": "json_schema",
             "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
            system_prompt,
        )
    # Non-OpenAI models via yunwu aggregator — fallback to json_object + embed schema in prompt
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    augmented = (
        system_prompt
        + f"\n\n## 严格按以下 JSON Schema 输出 (必须返回有效 JSON, 所有 required 字段都要有):\n"
          f"```json\n{schema_str}\n```\n"
          f"只返回 JSON 对象, 不要任何额外文字 / markdown 围栏 / 注释."
    )
    return ({"type": "json_object"}, augmented)


async def call_one(client, designer_id: str, scenario_key: str, user_prompt: str, model: str,
                   system_prompt: str, baseline_label: str = "B6"):
    seg = scenario_key.split("_")[1]
    schema = build_seg_schema(seg)
    schema_name = f"q7_{seg}_inference"
    response_format, augmented_system = get_response_format_and_aug_system(
        model, system_prompt, schema, schema_name)

    required_fields = ["dominant_factor", "state_sequence", "criterion",
                       "confidence", "reasoning_notes", "nl_answer"]
    # Non-OpenAI models需 schema embed in system, output 也长, max_tokens 加高防截断
    max_tokens = 900 if supports_strict_schema(model) else 1500

    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            content = r.choices[0].message.content
            if not content:
                raise ValueError("empty content")
            # Robust JSON parse — strip possible markdown fences
            txt = content.strip()
            if txt.startswith("```"):
                # Remove "```json\n...\n```"
                txt = txt.split("\n", 1)[1] if "\n" in txt else txt
                if txt.endswith("```"):
                    txt = txt.rsplit("```", 1)[0]
            parsed = json.loads(txt.strip())
            # Validate required fields
            missing = [k for k in required_fields if k not in parsed]
            if missing:
                raise ValueError(f"missing fields: {missing}")
            return {
                "ok": True, "baseline": baseline_label, "designer_id": designer_id, "scenario": scenario_key,
                "claimed_pkvc": {
                    "dominant_factor": parsed["dominant_factor"],
                    "state_sequence": parsed["state_sequence"],
                    "criterion": parsed["criterion"],
                    "confidence": parsed["confidence"],
                    "reasoning_notes": parsed["reasoning_notes"],
                },
                "simulated_nl": parsed["nl_answer"].strip(),
                "prompt_chars": len(user_prompt),
                "model": model,
            }
        except Exception as e:
            if attempt == 2:
                return {"ok": False, "baseline": baseline_label, "designer_id": designer_id,
                        "scenario": scenario_key, "error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(1 + attempt)


async def run_all(jobs, model, concurrency, baseline_label, api_key_env="OPENAI_API_KEY",
                  api_base_url=None):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"ERROR: set {api_key_env}"); sys.exit(1)
    base_url = api_base_url or os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(idx, job):
        async with sem:
            if idx % 20 == 0:
                print(f"[{idx}/{len(jobs)}] {job['designer_id'][:12]} {job['scenario']}", flush=True)
            return await call_one(client, job["designer_id"], job["scenario"], job["prompt"],
                                  model, job["system_prompt"], baseline_label)

    tasks = [asyncio.create_task(_wrap(i+1, j)) for i, j in enumerate(jobs)]
    return await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["v1", "v2", "v3", "v2_clean", "v2_hybrid",
                             "v2_no_pkvc_def", "v2_no_seg3_rule", "v2_no_beta_summary"],
                    default="v2",
                    help="B6 prompt variant. v2 = anchor everywhere (final main prompt). "
                         "v2_no_pkvc_def/v2_no_seg3_rule/v2_no_beta_summary = component ablations.")
    ap.add_argument("--n", type=int, default=20,
                    help="0 = all 162 designers; 20 = SAMPLE_DESIGNERS list")
    ap.add_argument("--segs", default="seg1,seg2,seg3",
                    help="comma-sep seg names to run, default all 3")
    ap.add_argument("--with-peer", action="store_true",
                    help="加 ④ peer (CAPS-JS top-3) block. B6_h = v2_hybrid + peer")
    ap.add_argument("--modules", default="pi,stance,self,caps",
                    help="comma-sep modules: pi/stance/self/caps. "
                         "ladder: B0=(empty) B1=pi B2=pi,stance B3=pi,stance,self B4=full")
    ap.add_argument("--model", default="gpt-4o",
                    help="Model ID. Examples: gpt-4o, claude-sonnet-4-..., gemini-3-flash, "
                         "deepseek-v4-pro, qwen-3.6-27b-instruct. "
                         "Non-OpenAI uses json_object fallback (schema embedded in prompt).")
    ap.add_argument("--api-base-url", default=None,
                    help="Override OPENAI_BASE_URL env var. Default: yunwu.ai/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="Env var name for API key (default OPENAI_API_KEY for yunwu aggregator)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default="reports/datav3/b6_sample_inference.jsonl")
    args = ap.parse_args()

    modules = {m.strip() for m in args.modules.split(",") if m.strip()}
    valid_modules = {"pi", "stance", "self", "caps"}
    bad = modules - valid_modules
    if bad:
        print(f"ERROR: unknown modules {bad}, valid={valid_modules}"); sys.exit(1)

    baseline_label = f"B6_{args.variant}"
    if modules != valid_modules:
        baseline_label += f"_mods[{','.join(sorted(modules)) if modules else 'none'}]"
    if args.with_peer:
        baseline_label += "_peer"
    print(f"[config] variant={args.variant}, n={args.n}, modules={modules}, "
          f"with_peer={args.with_peer}, label={baseline_label}")

    print("[load] data sources...")
    pool = load_caps_pool()
    pi = load_pi_5(Path("data_v3/users.csv"))
    scores = load_designer_scores(Path("data_v3/scale_responses.csv"))
    self_q1q6 = load_self_q1q6(Path("data_v3/task_responses.csv"))
    quads = load_caps_quadruples()
    print(f"  {len(pi)} pi, {len(scores)} scores, {len(self_q1q6)} self, {len(quads)} CAPS quads")

    if args.n == 0:
        # full: all 162 designers with both persona scores + Q7 seg available
        target_designers = sorted([
            did for did in pi
            if did in scores and did in pool.designers
            and any(pool.designers[did].has_scenario(f"Q7_{s}") for s in ("seg1","seg2","seg3"))
        ])
    else:
        target_designers = SAMPLE_DESIGNERS
    print(f"  {len(target_designers)} target designers")

    # Build jobs
    jobs = []
    for did in target_designers:
        if did not in pi:
            print(f"  WARN: missing pi for {did}, skip")
            continue
        pi_block = pi[did]
        stance_block = polarity_to_stance_block(scores.get(did, {}))
        self_block = format_self_q3q6_b6(self_q1q6.get(did, {}))
        # Pre-compute both CAPS formats once per designer (cheap, reused per seg)
        activated_balanced = get_activated_quads_balanced(scores.get(did, {}), quads)
        if args.variant == "v2_no_beta_summary":
            caps_balanced = format_caps_quad_balanced_nl(activated_balanced, variant="v2_no_beta_summary")
        else:
            caps_balanced = format_caps_quad_balanced_nl(activated_balanced, variant="v2")
        activated_topk = get_activated_quads(scores.get(did, {}), quads, top_k=5)
        caps_topk = format_activated_quads_nl(activated_topk)

        target_segs = [s.strip() for s in args.segs.split(",") if s.strip()]
        for seg in target_segs:
            scenario = f"Q7_{seg}"
            if did not in pool.designers or not pool.designers[did].has_scenario(scenario):
                continue
            caps_strategy = get_caps_strategy_for_seg(args.variant, seg)
            caps_block = caps_balanced if caps_strategy == "balanced" else caps_topk
            effective_variant = "v2_clean" if caps_strategy == "topk" else "v2"
            peer_block = ""
            if args.with_peer:
                peers = retrieve_caps_peer_q7_nl(did, pool, scenario, k_peers=3)
                peer_block = _format_peer_q7_block(peers, "认知模式相似度")
            prompt = build_b6_prompt(scenario, pi_block, stance_block, self_block, caps_block,
                                     variant=effective_variant, peer_block=peer_block,
                                     modules=modules)
            system_prompt = get_system_prompt_for_seg(args.variant, seg)
            jobs.append({"designer_id": did, "scenario": scenario, "prompt": prompt,
                         "system_prompt": system_prompt})

    print(f"[build] {len(jobs)} inference jobs")
    avg_chars = sum(len(j["prompt"]) for j in jobs) / len(jobs)
    print(f"  avg prompt chars: {avg_chars:.0f}")

    t0 = time.time()
    results = asyncio.run(run_all(jobs, args.model, args.concurrency, baseline_label,
                                   api_key_env=args.api_key_env, api_base_url=args.api_base_url))
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[done] {n_ok}/{len(results)} ok in {elapsed:.1f}s")

    # Attach GT
    for r in results:
        if not r["ok"]:
            continue
        did = r["designer_id"]
        if did in pool.designers:
            seg_obj = pool.designers[did].get_scenario(r["scenario"])
            if seg_obj:
                r["ground_truth_nl"] = seg_obj.text
                r["ground_truth_pkvc"] = {
                    "dominant": seg_obj.dominant_factor,
                    "sequence": seg_obj.state_sequence,
                    "criterion": seg_obj.criterion,
                    "confidence": seg_obj.confidence,
                }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
