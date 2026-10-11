"""T_CAPS quad schema v1.1 — context-aware quad for trait-conditioned routing.

Reference: docs/briefs/T_CAPS_quad_schema_v1.md (v1.1 锁定 2026-05-23)

四元组形式化 (升级 v3.5.1 §4D 的三元组):

    T_CAPS = (C, s, p, o)

字段拆分 (双层 framing):

    C  情境特征      C_local 进 quad (此 schema); C_global 进 prompt narrative (不在此)
    s  潜在特质      s_primary  (30 维 high-score tag 集合, retrieval 用)
                    s_secondary (factor + direction, mining 用)
    p  认知路由      p_modulation (5 档闭集, mining 用)
                    p_surface    (8 词开放表, paper 写作 + stance 编码)
    o  目标 PKVC     o_primary (4 大类 P/K/V/C, mining grid)
                    o_subtype (12 cardId P1-C3, attribute, 200 入袋后再升 primary)

leakage 防御: 方向二 only, Q7 designer NL 永远不进 quad DB (eval 时 ground truth 现抽).

依赖: 无第三方库, 纯 stdlib. 测试见 sft_pipeline/tests/test_quad_schema.py.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# =====================================================================
# 闭集词表 (与 data/processed/situation_codebook_v0.json 同步)
# =====================================================================

PKVC_FIELDS: tuple[str, ...] = ("P", "K", "V", "C")

PKVC_SUBTYPES: tuple[str, ...] = (
    "P1", "P2", "P3",
    "K1", "K2", "K3",
    "V1", "V2", "V3",
    "C1", "C2", "C3",
)

MODULATION_LEVELS: tuple[str, ...] = (
    "Strong+", "Weak+", "Weak-", "Strong-", "Bypass",
)

# v0 codebook, 8 个 surface 词; 跟 data/processed/situation_codebook_v0.json
# 末尾 predicate_codebook 段同步. 学姐 confirm 后可扩到 10.
SURFACE_PREDICATES: tuple[str, ...] = (
    "forces_transition_to",
    "severely_overrides",
    "defiant_overrides",
    "gradually_shifts_to",
    "compliantly_follows",
    "negotiates_around",
    "weakly_inhibits",
    "bypasses",
)

POLARITY_PREFIXES: tuple[str, ...] = ("High", "Low")

# C_local tag 命名约束 regex (L-8 锁定 2026-05-23):
#   [High|Low]_[P|K|V|C][1-3]?_<Name>_vs_[High|Low]_[P|K|V|C][1-3]?_<Name>
# Name: PascalCase 或 snake_case (允许字母 + 数字 + _), 不允许中文 / 空格.
C_LOCAL_TAG_RE = re.compile(
    r"^(High|Low)_([PKVC][1-3]?)_([A-Za-z][A-Za-z0-9_]*)"
    r"_vs_(High|Low)_([PKVC][1-3]?)_([A-Za-z][A-Za-z0-9_]*)$"
)


# =====================================================================
# 子类型 dataclass
# =====================================================================

@dataclass(slots=True)
class TraitSource:
    """s 字段: 双层 (primary 30 维 + secondary factor pattern).

    s_primary 是高分项 tag 集合, 用于 retrieval (Jaccard / cosine top-k).
    s_secondary 是 (factor, direction), 用于 mining (permutation test).
    """
    s_primary: list[str] = field(default_factory=list)
    s_secondary_factor: str | None = None          # e.g. "F2_共情关怀"
    s_secondary_direction: Literal["+", "-"] | None = None


@dataclass(slots=True)
class CognitiveRoutingPredicate:
    """p 字段: 双层 (modulation 5 档 + surface 8 词).

    p_modulation 是 mining 用的闭集 (Strong+ / Weak+ / Weak- / Strong- / Bypass).
    p_surface 是 paper 写作 + stance 编码用 (Defiance / Compliance / Negotiation
    通过 surface 词选择隐含表达).
    bypass flag: True 表示跳过 PKVC 默认 P→K→V→C 顺序.
    """
    p_modulation: str                              # MODULATION_LEVELS 之一 ("Strong+" 等)
    p_surface: str                                 # SURFACE_PREDICATES 之一
    bypass: bool = False


# =====================================================================
# 主 quad dataclass
# =====================================================================

@dataclass(slots=True)
class TCAPSQuad:
    """T_CAPS quad: 一条 (designer, question) → quad 记录.

    必填: quad_id, designer_id, question_id, C, s, p, o_primary.
    可选: o_subtype (v1.1 attribute, 等学姐 Q3-Q6 选项-因子 mapping), fold,
          extraction_confidence, extraction_notes.

    序列化用 .to_dict() / from_dict() (兼容 JSONL).
    """
    quad_id: str
    designer_id: str
    question_id: str                                # "Q1".."Q8"
    C: str                                          # C_local tag, 命名约束 C_LOCAL_TAG_RE
    s: TraitSource
    p: CognitiveRoutingPredicate
    o_primary: str                                  # PKVC_FIELDS 之一
    o_subtype: str | None = None                    # PKVC_SUBTYPES 之一 (v1.1)
    fold: int | None = None                         # 5-fold CV split index
    extraction_confidence: Literal["high", "medium", "low"] | None = None
    extraction_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TCAPSQuad":
        s_obj = d["s"]
        p_obj = d["p"]
        s = TraitSource(
            s_primary=list(s_obj.get("s_primary", [])),
            s_secondary_factor=s_obj.get("s_secondary_factor"),
            s_secondary_direction=s_obj.get("s_secondary_direction"),
        )
        p = CognitiveRoutingPredicate(
            p_modulation=p_obj["p_modulation"],
            p_surface=p_obj["p_surface"],
            bypass=bool(p_obj.get("bypass", False)),
        )
        return cls(
            quad_id=d["quad_id"],
            designer_id=d["designer_id"],
            question_id=d["question_id"],
            C=d["C"],
            s=s,
            p=p,
            o_primary=d["o_primary"],
            o_subtype=d.get("o_subtype"),
            fold=d.get("fold"),
            extraction_confidence=d.get("extraction_confidence"),
            extraction_notes=d.get("extraction_notes", ""),
        )


# =====================================================================
# Validators
# =====================================================================

class QuadValidationError(ValueError):
    """命名约束 / 闭集词表违反."""


def validate_c_local_tag(tag: str) -> None:
    """验证 C_local tag 符合 v1.1 命名约束.

    格式: [High|Low]_[P|K|V|C][1-3]?_<Name>_vs_[High|Low]_[P|K|V|C][1-3]?_<Name>
    两端字段必须 ∈ PKVC 12 cardId (或大类 P/K/V/C).
    """
    if not C_LOCAL_TAG_RE.match(tag):
        raise QuadValidationError(
            f"C_local tag 不符合命名约束: {tag!r}\n"
            f"  期望格式: [High|Low]_[PKVC][1-3]?_Name_vs_[High|Low]_[PKVC][1-3]?_Name\n"
            f"  例: High_K_Utility_vs_Low_C_Aesthetics\n"
            f"  常见错: 大小写 _VS_ (要小写 _vs_) / 字段非 PKVC / 缺极性前缀"
        )


def validate_modulation(level: str) -> None:
    if level not in MODULATION_LEVELS:
        raise QuadValidationError(
            f"p_modulation 非法: {level!r}, 必须 ∈ {MODULATION_LEVELS}"
        )


def validate_surface(surface: str) -> None:
    if surface not in SURFACE_PREDICATES:
        raise QuadValidationError(
            f"p_surface 非法: {surface!r}, 必须 ∈ {SURFACE_PREDICATES}\n"
            f"  (codebook v0 锁 8 词, 学姐 confirm 后可扩到 10)"
        )


def validate_o_primary(state: str) -> None:
    if state not in PKVC_FIELDS:
        raise QuadValidationError(
            f"o_primary 非法: {state!r}, 必须 ∈ {PKVC_FIELDS}"
        )


def validate_o_subtype(subtype: str | None) -> None:
    if subtype is None:
        return  # v1.1 允许空 (等学姐 Q3-Q6 mapping)
    if subtype not in PKVC_SUBTYPES:
        raise QuadValidationError(
            f"o_subtype 非法: {subtype!r}, 必须 ∈ {PKVC_SUBTYPES} 或 None"
        )


def validate_quad(quad: TCAPSQuad) -> None:
    """验证一条 quad 满足 v1.1 schema 所有约束.

    Raises QuadValidationError on first violation.
    """
    if not quad.quad_id:
        raise QuadValidationError("quad_id 不能为空")
    if not quad.designer_id:
        raise QuadValidationError("designer_id 不能为空")
    if not quad.question_id:
        raise QuadValidationError("question_id 不能为空")
    validate_c_local_tag(quad.C)
    validate_modulation(quad.p.p_modulation)
    validate_surface(quad.p.p_surface)
    validate_o_primary(quad.o_primary)
    validate_o_subtype(quad.o_subtype)
    # o_primary 跟 o_subtype 字段一致性: 若 o_subtype=V1, o_primary 必须 = V
    if quad.o_subtype is not None and quad.o_subtype[0] != quad.o_primary:
        raise QuadValidationError(
            f"o_subtype ({quad.o_subtype}) 跟 o_primary ({quad.o_primary}) 字段不一致, "
            f"前缀 {quad.o_subtype[0]!r} ≠ {quad.o_primary!r}"
        )
