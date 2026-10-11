"""T_CAPS quad codebook v1.1 (锁定 2026-05-23).

来源: docs/briefs/T_CAPS_quad_schema_v1.md §2 + §3
应用: quad_extractor 调用方查表填 p_modulation / 注 stance / 验 C tag 闭集.
LLM 只输出 p_surface (8 词) + o_primary (4 状态) + o_subtype (12 cardId).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# C 闭集 (4 个题级情境 tag, Q3-Q6 各 1)
# 来源: reports/datav3/per_designer/option_trait_mapping.json (学姐 mapping 为 source of truth)
# 注意: C 可以是 2-way / 3-way / 4-way conflict, 不限 X_vs_Y 形式
# ---------------------------------------------------------------------------
C_LOCAL_TAGS: dict[str, str] = {
    "Q3": "K_ObjectiveData_vs_CV_Aesthetic",
    "Q4": "KP_BusinessPrivacy_vs_V_Ethics_vs_C_Construct",
    "Q5": "K_TechScale_vs_C_Organic_vs_V_HumanCentric_vs_P_Core",
    "Q6": "V_UserResponsibility_vs_CPK_Commercial",
}

# Q1/Q2/Q8 不在 quad 范围 (Q1=profile, Q2=cardid ranking, Q8=self-report)
# 但 option_trait_mapping.json 含 Q1/Q8 mapping (辅助分析用, 不进四元组)
QUAD_QUESTIONS: tuple[str, ...] = ("Q3", "Q4", "Q5", "Q6")


def load_mapping(json_path: str = "reports/datav3/per_designer/option_trait_mapping.json") -> dict:
    """加载 option_trait_mapping.json. 返回 {Q3_A: {...}, Q3_B: {...}, ...}.

    每 entry 含: qid / option / n_samples / C_formal_proposed / s_primary /
                 s_primary_with_stance / o_primary_hint / o_subtype_hint
    """
    import json
    from pathlib import Path
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def get_quad_metadata(mapping: dict, qid: str, option: str) -> dict | None:
    """从 mapping 取该 (qid, option) 的 quad metadata.

    返回 {C, s_primary, s_primary_with_stance, o_primary_hint, o_subtype_hint, n_samples}.
    若 option 不在 mapping 或 qid 不在 quad 范围, 返回 None.
    """
    key = f"{qid}_{option}"
    entry = mapping.get(key)
    if not entry or qid not in QUAD_QUESTIONS:
        return None
    return {
        "C": entry.get("C_formal_proposed", ""),
        "C_conflict_axes": entry.get("C_conflict_axes", ""),
        "s_primary": entry.get("s_primary", []),
        "s_primary_with_stance": entry.get("s_primary_with_stance", []),
        "o_primary_hint": entry.get("o_primary_hint"),
        "o_subtype_hint": entry.get("o_subtype_hint"),
        "n_samples": entry.get("n_samples", 0),
    }


# ---------------------------------------------------------------------------
# p_surface 8 词闭集 (open vocab, LLM 抽取)
# ---------------------------------------------------------------------------
P_SURFACE_VOCAB: tuple[str, ...] = (
    "forces_transition_to",
    "severely_overrides",
    "defiant_overrides",
    "gradually_shifts_to",
    "compliantly_follows",
    "negotiates_around",
    "weakly_inhibits",
    "bypasses",
)


# ---------------------------------------------------------------------------
# p_surface ↔ p_modulation mapping (codebook 多对一)
# ---------------------------------------------------------------------------
# Modulation 5 档 (字符串值):
#   "Strong+"        强正向迁移
#   "Strong+ Bypass" 强正向 + 跳过中间阶段
#   "Weak+"          弱正向迁移
#   "Weak±"          弱正向或弱负向 (negotiates_around 上下文敏感)
#   "Weak-"          弱抑制
#   "Bypass"         直接跳过
P_SURFACE_TO_MODULATION: dict[str, str] = {
    "forces_transition_to":  "Strong+",
    "severely_overrides":    "Strong+ Bypass",
    "defiant_overrides":     "Strong+ Bypass",
    "gradually_shifts_to":   "Weak+",
    "compliantly_follows":   "Weak+",
    "negotiates_around":     "Weak±",
    "weakly_inhibits":       "Weak-",
    "bypasses":              "Bypass",
}

# stance 色彩 (3 个非 neutral 的 surface 词带 stance)
P_SURFACE_TO_STANCE: dict[str, str] = {
    "forces_transition_to":  "neutral",
    "severely_overrides":    "neutral",
    "defiant_overrides":     "Defiance",
    "gradually_shifts_to":   "neutral",
    "compliantly_follows":   "Compliance",
    "negotiates_around":     "Negotiation",
    "weakly_inhibits":       "neutral",
    "bypasses":              "neutral",
}


# ---------------------------------------------------------------------------
# o 闭集
# ---------------------------------------------------------------------------
O_PRIMARY_VOCAB: tuple[str, ...] = ("P", "K", "V", "C")
O_SUBTYPE_VOCAB: tuple[str, ...] = (
    "P1", "P2", "P3",
    "K1", "K2", "K3",
    "V1", "V2", "V3",
    "C1", "C2", "C3",
)

# 复合 o_subtype (X→Y 动态关系, 学姐 metadata 标的 "p➡️k 引起" 关系)
# 来源: docs/notes/Q1-Q6每个选项的元数据.md 学姐用过的 8 个复合
# 含义: 设计师从 X 状态被引起转移到 Y 状态, 终点 Y == o_primary 大类
O_SUBTYPE_COMPOSITE_USED: tuple[str, ...] = (
    "P3→K3",   # Q3_B
    "V1→C1",   # Q3_C
    "P1→K2",   # Q3_D 分支
    "K1→P1",   # Q4_A
    "K2→P2",   # Q4_D
    "V1→C2",   # Q6_B
    "V2→P1",   # Q6_C
    "K2→V2",   # Q6_D
)

# 完整闭集 (单 + 复合)
O_SUBTYPE_VOCAB_FULL: tuple[str, ...] = O_SUBTYPE_VOCAB + O_SUBTYPE_COMPOSITE_USED

# 12 cardId 子类定义 (来源: docs/notes/Q2选项的解释.md 学姐 metadata)
# ---------------------------------------------------------------------------
# Dual-form NL 翻译表 (β+++ output 用, dual-form: "中文 (token)")
# ---------------------------------------------------------------------------

# C_local tag → 中文 NL (Q3-Q6 各 1)
C_LOCAL_NL_ZH: dict[str, str] = {
    "K_ObjectiveData_vs_CV_Aesthetic": "高 K 数据效用 vs 低 C/V 美感",
    "KP_BusinessPrivacy_vs_V_Ethics_vs_C_Construct": "高 K/P 商业隐私 vs 高 V 伦理 vs C 架构",
    "K_TechScale_vs_C_Organic_vs_V_HumanCentric_vs_P_Core": "高 K 技术规模 vs C 有机 vs V 人本 vs P 核心",
    "V_UserResponsibility_vs_CPK_Commercial": "高 V 用户责任 vs CPK 商业指令",
}

# C_conflict_axes → 张力两端 NL ("A (左) vs B (右)")
C_CONFLICT_TENSION_NL: dict[str, str] = {
    "K vs. C/V": "数据效用 (K) vs 美感/价值 (C/V)",
    "K/P vs. V vs. C": "商业隐私 (K/P) vs 伦理 (V) vs 架构 (C)",
    "K vs. C vs. V vs. P": "技术 (K) vs 有机 (C) vs 人本 (V) vs 核心 (P)",
    "V vs. C vs. P/K": "用户责任 (V) vs 概念 (C) vs 商业指令 (P/K)",
}

# 冲突类型 NL (高低组合判断 — 简化版)
def get_polarity_type_nl(conflict: str) -> str:
    """根据 C_conflict_axes 推断冲突类型 NL.
    e.g. 'K vs. C/V' → '高低冲突 (high_vs_low)' (简化版, 当前都按 high_vs_high 处理)
    """
    return "多维冲突 (multi_axis)"  # placeholder, 学姐 ⑤ confirm 后细化


# o_primary 中文释义 (4 个)
O_PRIMARY_NL: dict[str, str] = {
    "P": "问题界定",
    "K": "知识应用",
    "V": "价值判断",
    "C": "概念生成",
}

# o_subtype 中文释义 (12 个), 给 dual-form output ("界定核心 (P1)" 等)
O_SUBTYPE_NL: dict[str, str] = {
    "P1": "界定核心", "P2": "拆解子问题", "P3": "推翻假定",
    "K1": "调用经验", "K2": "客观数据", "K3": "评估技术栈",
    "V1": "价值立场", "V2": "利益排序", "V3": "立绝对底线",
    "C1": "涌现画面", "C2": "整合构建", "C3": "提炼原则",
}

# p_surface 中文释义 (8 词)
P_SURFACE_NL: dict[str, str] = {
    "forces_transition_to": "强制迁移",
    "severely_overrides": "强压倒",
    "defiant_overrides": "抗命越过",
    "gradually_shifts_to": "渐进迁移",
    "compliantly_follows": "服从跟随",
    "negotiates_around": "协商绕开",
    "weakly_inhibits": "弱抑制",
    "bypasses": "直接跳过",
}

# p_modulation 中文释义 (5 档)
P_MODULATION_NL: dict[str, str] = {
    "Strong+": "强促进",
    "Strong+ Bypass": "强促进 + 跳过",
    "Weak+": "弱促进",
    "Weak±": "弱促进 + 弱抑制",
    "Weak-": "弱抑制",
    "Bypass": "直接跳过",
}

# EFA 4 因子 中文释义 (placeholder, 等学姐定; 暂用 token-only fallback)
FACTOR_NAME_NL: dict[str, str] = {
    "F1": "F1",   # 学姐 ① 解锁后改 "F1_问题驱动" 等
    "F2": "F2",
    "F3": "F3",
    "F4": "F4",
}


# ---------------------------------------------------------------------------
# Helper: dual-form formatter
# ---------------------------------------------------------------------------

def dual_form(nl: str | None, token: str | None) -> str:
    """生成 dual-form 字符串 "中文 (token)". 若 nl 缺则 fallback "(token)"."""
    if not token:
        return "(待)"
    if not nl or nl == token:
        return f"({token})"
    return f"{nl} ({token})"


def format_o_primary_dual(o_primary: str | None) -> str:
    """e.g. 'V' → '价值判断 (V)'."""
    return dual_form(O_PRIMARY_NL.get(o_primary or ""), o_primary)


def format_o_subtype_dual(o_subtype: str | None) -> str:
    """单值 'V3' → '立绝对底线 (V3)'; 复合 'V1→C2' → '价值立场转整合 (V1→C2)'."""
    if not o_subtype:
        return "(待学姐 ④)"
    if "→" in o_subtype:
        start, end = o_subtype.split("→")
        s_nl = O_SUBTYPE_NL.get(start, start)
        e_nl = O_SUBTYPE_NL.get(end, end)
        nl = f"{s_nl}转{e_nl}"
        return f"{nl} ({o_subtype})"
    return dual_form(O_SUBTYPE_NL.get(o_subtype), o_subtype)


def format_p_surface_dual(p_surface: str | None) -> str:
    return dual_form(P_SURFACE_NL.get(p_surface or ""), p_surface)


def format_p_modulation_dual(p_mod: str | None) -> str:
    return dual_form(P_MODULATION_NL.get(p_mod or ""), p_mod)


def format_sadr_dual(sadr_list: list[dict] | None) -> str:
    """e.g. [{'id':'#V3+','label':'用户福祉优先'}, ...] → '用户福祉优先 (#V3+), 务实让步 (#P3+)'"""
    if not sadr_list:
        return "(待)"
    parts = []
    for s in sadr_list:
        if isinstance(s, dict):
            label = s.get("label", "?")
            sid = s.get("id", "?")
            parts.append(f"{label} ({sid})")
    return ", ".join(parts) if parts else "(待)"


def format_factor_pattern(factor_scores: list[float] | None) -> str:
    """从 F1-F4 scores 算 (factor, direction) 组合, 输出 top-2 by abs.
    e.g. [0.8, -0.3, 0.1, 0.5] → '(F1_问题驱动, +); (F4, +)'
    """
    if not factor_scores or len(factor_scores) < 4:
        return "(待 EFA)"
    pairs = [(f"F{i+1}", v) for i, v in enumerate(factor_scores)]
    pairs.sort(key=lambda x: -abs(x[1]))
    top2 = pairs[:2]
    parts = []
    for fname, score in top2:
        nl = FACTOR_NAME_NL.get(fname, fname)
        dir_ = "+" if score >= 0 else "-"
        parts.append(f"({nl}, {dir_})")
    return "; ".join(parts)


def format_q2_cardid_top3_dual(cardid_order: list[str] | None) -> str:
    """Q2 cardId top-3 → '[界定核心 (P1), 客观数据 (K2), 立绝对底线 (V3)]'"""
    if not cardid_order or len(cardid_order) < 3:
        return "[(待 Q2)]"
    top3 = cardid_order[:3]
    parts = []
    for c in top3:
        nl = O_SUBTYPE_NL.get(c, c)
        parts.append(f"{nl} ({c})")
    return "[" + ", ".join(parts) + "]"


O_SUBTYPE_DEFINITIONS: dict[str, str] = {
    "P1": "界定核心 — 界定项目的核心底线和真实要解决的问题",
    "P2": "拆解子问题 — 降维拆解, 分而治之",
    "P3": "推翻假定 — 推翻之前假定, 认为问题根本不在这里",
    "K1": "调用经验 — 过去经验/竞品案例/行业规范",
    "K2": "客观数据 — A/B 测试/数据/转化率/经营数据",
    "K3": "评估技术栈 — 现有技术栈/材料/工具边界",
    "V1": "价值立场 — 基于专业价值立场给出判断标准",
    "V2": "利益排序 — 多方利益冲突的优先级排序",
    "V3": "设立原则 — 不可牺牲的原则底线",
    "C1": "涌现画面 — 脑海里快速涌现的强烈形态/愿景画面",
    "C2": "整合构建 — 零散观察整合为统摄全局的核心设计逻辑",
    "C3": "提炼原则 — 提炼可复用的设计原则",
}


# ---------------------------------------------------------------------------
# Quad result schema (dataclass)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TCapsQuad:
    """T_CAPS quad (C, s, p, o), v1.1 schema.

    s_primary / s_secondary 跟 C 是调用方 metadata, 不在 LLM 输出里.
    LLM 只填 p_surface / o_primary / o_subtype.
    p_modulation / p_stance 由 codebook 查 p_surface 得到.
    """
    # context (调用方传)
    C: str                                  # 题级情境 tag, ∈ C_LOCAL_TAGS.values()
    s_primary: list[str]                    # 该选项关联的高分特质 (option_trait_mapping)
    s_secondary: dict[str, str]             # {factor: ..., direction: +/-}

    # LLM 抽取 (3 字段)
    p_surface: str                          # ∈ P_SURFACE_VOCAB
    o_primary: str                          # ∈ O_PRIMARY_VOCAB
    o_subtype: str | None                   # ∈ O_SUBTYPE_VOCAB | null

    # codebook 查表 (derived, 不由 LLM 输出)
    p_modulation: str = ""                  # 由 p_surface 查 P_SURFACE_TO_MODULATION
    p_stance: str = ""                      # 由 p_surface 查 P_SURFACE_TO_STANCE

    # 元数据
    quad_id: str = ""
    designer_id: str = ""
    question_id: str = ""                   # ∈ QUAD_QUESTIONS
    confidence: Literal["high", "medium", "low"] = "high"


def fill_codebook(quad: TCapsQuad) -> TCapsQuad:
    """根据 quad.p_surface 查 codebook 填 p_modulation / p_stance.

    若 p_surface 不在闭集, 抛 ValueError.
    """
    if quad.p_surface not in P_SURFACE_TO_MODULATION:
        raise ValueError(
            f"p_surface {quad.p_surface!r} not in 8-vocab "
            f"{P_SURFACE_VOCAB}"
        )
    quad.p_modulation = P_SURFACE_TO_MODULATION[quad.p_surface]
    quad.p_stance = P_SURFACE_TO_STANCE[quad.p_surface]
    return quad


def parse_o_subtype(o_subtype: str | None) -> tuple[str | None, str | None]:
    """Parse o_subtype 成 (起点, 终点). 单值 → (None, single); 复合 X→Y → (X, Y).

    >>> parse_o_subtype("V3")        # 单值
    (None, 'V3')
    >>> parse_o_subtype("P3→K3")     # 复合
    ('P3', 'K3')
    >>> parse_o_subtype(None)
    (None, None)
    """
    if o_subtype is None:
        return None, None
    if "→" in o_subtype:
        parts = o_subtype.split("→")
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, o_subtype


def validate_quad(quad: TCapsQuad) -> list[str]:
    """返回 quad 违反 schema 的 fail reason list (空 = ok).

    o_subtype 允许 2 种:
      - 单值: P1/P2/.../C3 (12 闭集), 终点 == o_subtype 自己
      - 复合 X→Y: e.g. P3→K3, 终点 Y 必须跟 o_primary 一致
    """
    fails: list[str] = []
    if quad.C not in C_LOCAL_TAGS.values():
        fails.append(f"C tag {quad.C!r} 不在 4 闭集 (Q3-Q6 each 1)")
    if quad.p_surface not in P_SURFACE_VOCAB:
        fails.append(f"p_surface {quad.p_surface!r} 不在 8 词闭集")
    if quad.o_primary not in O_PRIMARY_VOCAB:
        fails.append(f"o_primary {quad.o_primary!r} 不在 4 状态 (P/K/V/C)")

    if quad.o_subtype is not None:
        start, end = parse_o_subtype(quad.o_subtype)
        # 终点必须在 12 闭集
        if end not in O_SUBTYPE_VOCAB:
            fails.append(f"o_subtype 终点 {end!r} 不在 12 cardId 闭集")
        # 复合时, 起点也必须在 12 闭集
        if start is not None and start not in O_SUBTYPE_VOCAB:
            fails.append(f"o_subtype 起点 {start!r} 不在 12 cardId 闭集")
        # 终点 prefix 必须跟 o_primary 一致 (单值 or 复合都管)
        if end and quad.o_primary and not end.startswith(quad.o_primary):
            fails.append(
                f"o_subtype 终点 {end} prefix 跟 o_primary {quad.o_primary} 不一致 "
                f"(复合时 X→Y 的 Y 必须 跟 o_primary; 单值时 o_subtype 自己 跟 o_primary)"
            )

    if quad.question_id not in QUAD_QUESTIONS:
        fails.append(f"question_id {quad.question_id!r} 不在 Q3-Q6")
    return fails


if __name__ == "__main__":
    # Quick smoke
    q = TCapsQuad(
        C=C_LOCAL_TAGS["Q6"],
        s_primary=["救世主情结", "弱势群体优先"],
        s_secondary={"factor": "F_共情关怀", "direction": "+"},
        p_surface="defiant_overrides",
        o_primary="V",
        o_subtype="V3",
        question_id="Q6",
    )
    q = fill_codebook(q)
    fails = validate_quad(q)
    print(f"quad: C={q.C[:30]}...")
    print(f"     p_surface={q.p_surface} → modulation={q.p_modulation}, stance={q.p_stance}")
    print(f"     o={q.o_primary}/{q.o_subtype} ({O_SUBTYPE_DEFINITIONS[q.o_subtype]})")
    print(f"validation: {'OK' if not fails else fails}")
