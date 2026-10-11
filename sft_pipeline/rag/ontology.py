"""PKVC + SADR ontology schema.

Ontology 三层:
  1. Factors (PKVC 4 因子): 顶层, P/K/V/C
  2. SADR nodes (30 个): 因子的细粒度子节点, 每个带 factor 归属 + 极性默认
  3. Scenario types: 每个 question (Q1-Q7) 对应 probe 的因子 / 冲突维度

加载方式:
  - 默认值 hardcoded 在 DEFAULT_ONTOLOGY (基于 brief v3.4 + Q2 cardId)
  - 学姐导出 metadata 后, 替换 SADR_NODES 和 SCENARIO_TYPES
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Factor = Literal["P", "K", "V", "C"]
Polarity = Literal["+", "-", "0"]


@dataclass(frozen=True)
class SADRNode:
    id: str                         # 例如 "#P1", "#V2", "#NEW_服务保留"
    factor: Factor                  # 归属因子
    description: str
    polarity_default: Polarity = "+"


@dataclass(frozen=True)
class FactorDef:
    code: Factor
    name: str
    description: str


@dataclass
class ScenarioType:
    """每个 question 的语义 metadata."""
    question_id: str                # "Q1" / "Q7_seg1" 等
    probe_factors: list[Factor]     # 该题主要 probe 的因子
    probe_conflict: tuple[Factor, Factor] | None = None  # Q3-Q6 因子冲突题
    description: str = ""


# ==================
# Factors (PKVC)
# ==================

FACTORS: dict[Factor, FactorDef] = {
    "P": FactorDef("P", "问题定义", "如何界定问题、识别真实需求、拆解复杂委托"),
    "K": FactorDef("K", "知识应用", "如何调用经验、案例、客观数据、工具/材料知识"),
    "V": FactorDef("V", "价值判断", "多方利益冲突中排序, 设立不可牺牲的伦理/价值底线"),
    "C": FactorDef("C", "概念生成", "从模糊愿景到具体形态, 提炼可复用设计逻辑"),
}


# ==================
# SADR Nodes
# ==================
# 11 个来自 Q2 cardId (已知), 其余 19 个等学姐补全后 update

SADR_NODES: dict[str, SADRNode] = {
    # P 因子
    "#P1": SADRNode("#P1", "P", "界定项目核心底线和真实要解决的问题"),
    "#P2": SADRNode("#P2", "P", "将复杂的总委托拆解为几个具体子问题"),
    # K 因子
    "#K1": SADRNode("#K1", "K", "调用过去的经验、竞品案例或行业规范作参考"),
    "#K2": SADRNode("#K2", "K", "使用客观数据、测试报告来证明某个逻辑"),
    "#K3": SADRNode("#K3", "K", "评估现有技术栈、材料以及工具边界能实现什么"),
    # V 因子
    "#V1": SADRNode("#V1", "V", "(待学姐补全)"),
    "#V2": SADRNode("#V2", "V", "在多方利益冲突时进行优先级排序"),
    "#V3": SADRNode("#V3", "V", "设立一条绝对不可牺牲的原则底线"),
    # C 因子
    "#C1": SADRNode("#C1", "C", "脑海中快速涌现出一个模糊但强烈的形态或愿景画面"),
    "#C2": SADRNode("#C2", "C", "把零散观察整合为一个统摄全局的核心设计逻辑"),
    "#C3": SADRNode("#C3", "C", "提炼一些可复用的设计原则"),
    # TODO #10-#30 (#NEW_*) 等学姐补全
}


# ==================
# Scenario Types
# ==================

SCENARIO_TYPES: dict[str, ScenarioType] = {
    "Q1": ScenarioType("Q1", ["P", "K", "V", "C"], description="PKVC 优先级排序"),
    "Q2": ScenarioType("Q2", ["P", "K", "V", "C"], description="PKVC 宏观流转路径"),
    "Q3": ScenarioType("Q3", ["V", "K"], probe_conflict=("V", "K"),
                       description="V vs K 冲突权重"),
    "Q4": ScenarioType("Q4", ["V", "C"], probe_conflict=("V", "C"),
                       description="V vs C 冲突权重 (placeholder, 待 confirm)"),
    "Q5": ScenarioType("Q5", ["K", "P"], probe_conflict=("K", "P"),
                       description="K vs P 冲突权重 (placeholder)"),
    "Q6": ScenarioType("Q6", ["V", "P"], probe_conflict=("V", "P"),
                       description="V vs P 冲突权重 (placeholder)"),
    "Q7_seg1": ScenarioType("Q7_seg1", ["V", "P"], description="第一反应 (Q7 开放场景)"),
    "Q7_seg2": ScenarioType("Q7_seg2", ["P", "K", "V", "C"], description="决策链路 (Q7)"),
    "Q7_seg3": ScenarioType("Q7_seg3", ["V"], description="核心 tradeoff (Q7)"),
}


# ==================
# 便捷查询
# ==================

def get_sadr_by_factor(factor: Factor) -> list[SADRNode]:
    return [n for n in SADR_NODES.values() if n.factor == factor]


def get_scenario_probe_factors(question_id: str) -> set[Factor]:
    s = SCENARIO_TYPES.get(question_id)
    return set(s.probe_factors) if s else set()


def parse_sadr_id_with_polarity(token: str) -> tuple[str, Polarity]:
    """e.g. '#V3+' → ('#V3', '+'), '#服务保留-' → ('#服务保留', '-')."""
    if token.endswith("+"):
        return token[:-1], "+"
    if token.endswith("-"):
        return token[:-1], "-"
    return token, "0"


def parse_factor_path(s: str) -> list[Factor]:
    """e.g. 'V → K → V' → ['V', 'K', 'V']."""
    out: list[Factor] = []
    for sep in ["→", "->", "|", ","]:
        if sep in s:
            for ch in s.split(sep):
                ch = ch.strip()
                if ch in FACTORS:
                    out.append(ch)  # type: ignore
            return out
    if s.strip() in FACTORS:
        return [s.strip()]  # type: ignore
    return []
