"""Unit tests for T_CAPS quad schema v1.1 (rag/quad_schema.py + rag/codebook.py).

跑法:
   .venv/bin/python -m pytest sft_pipeline/tests/test_quad_schema.py -v
不依赖 GPU 不依赖 API key, 纯 stdlib + sft_pipeline 自身.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from sft_pipeline.rag.quad_schema import (
    C_LOCAL_TAG_RE,
    MODULATION_LEVELS,
    PKVC_FIELDS,
    PKVC_SUBTYPES,
    SURFACE_PREDICATES,
    CognitiveRoutingPredicate,
    QuadValidationError,
    TCAPSQuad,
    TraitSource,
    validate_c_local_tag,
    validate_modulation,
    validate_o_primary,
    validate_o_subtype,
    validate_quad,
    validate_surface,
)
from sft_pipeline.rag.codebook import (
    PredicateCodebook,
    load_predicate_codebook,
    load_situation_codebook,
    map_surface_to_modulation,
    reset_codebook_cache,
)


# =====================================================================
# C_local tag 命名约束
# =====================================================================

class TestCLocalTagNaming:
    """学姐 2026-05-23 给的 4 个真实 tag 必须全 pass."""

    @pytest.mark.parametrize("tag", [
        "High_K_Utility_vs_Low_C_Aesthetics",                            # Q3
        "High_P_Commercial_Profit_vs_High_V_Ethical_Risk",                # Q4
        "High_K_Automation_Efficiency_vs_Low_V_Authorship_Identity",      # Q5
        "High_P_Authority_Mandate_vs_High_V_User_Harm",                   # Q6
        # 子类编号也合法 (学姐后续可能升级到 P1/V2 这种)
        "High_P1_Authority_vs_High_V2_UserHarm",
        "Low_K3_Tooling_vs_High_C1_Concept",
    ])
    def test_accept_valid(self, tag):
        validate_c_local_tag(tag)  # 不抛异常 = pass

    @pytest.mark.parametrize("tag,reason", [
        ("high_K_Utility_vs_Low_C_Aesthetics", "前缀小写 (要 PascalCase High/Low)"),
        ("High_K_Utility_VS_Low_C_Aesthetics", "分隔符大写 (锁小写 _vs_)"),
        ("High_X_Utility_vs_Low_C_Aesthetics", "X 不在 PKVC 字段集"),
        ("K_Utility_vs_C_Aesthetics", "缺极性前缀"),
        ("High_K_Utility_vs_Low_C", "右端缺名字"),
        ("High_K1234_Utility_vs_Low_C_Aesthetics", "K1234 子类编号超 1-3"),
        ("", "空字符串"),
        ("不带英文字段_vs_中文不行", "中文不允许"),
    ])
    def test_reject_invalid(self, tag, reason):
        with pytest.raises(QuadValidationError):
            validate_c_local_tag(tag)


# =====================================================================
# 闭集字段验证
# =====================================================================

class TestEnumValidators:

    def test_modulation_valid(self):
        for level in MODULATION_LEVELS:
            validate_modulation(level)

    def test_modulation_invalid(self):
        for bad in ["VeryStrong+", "strong+", "Boost", "Strong"]:
            with pytest.raises(QuadValidationError):
                validate_modulation(bad)

    def test_surface_valid(self):
        for surf in SURFACE_PREDICATES:
            validate_surface(surf)

    def test_surface_invalid(self):
        for bad in ["forces_to", "OVERRIDES", "新动词", ""]:
            with pytest.raises(QuadValidationError):
                validate_surface(bad)

    def test_o_primary_valid(self):
        for s in PKVC_FIELDS:
            validate_o_primary(s)

    def test_o_primary_invalid(self):
        for bad in ["p", "X", "PK", "Problem"]:
            with pytest.raises(QuadValidationError):
                validate_o_primary(bad)

    def test_o_subtype_valid(self):
        validate_o_subtype(None)  # None 合法 (学姐数据未到位)
        for s in PKVC_SUBTYPES:
            validate_o_subtype(s)

    def test_o_subtype_invalid(self):
        for bad in ["P4", "K0", "V11", "p1", "PA"]:
            with pytest.raises(QuadValidationError):
                validate_o_subtype(bad)


# =====================================================================
# TCAPSQuad dataclass + 整体 validator
# =====================================================================

def _make_valid_quad(**overrides) -> TCAPSQuad:
    """factory: 给一个合法的 quad, 子字段可以 override."""
    defaults = dict(
        quad_id="uD001_Q6",
        designer_id="D001",
        question_id="Q6",
        C="High_P_Authority_Mandate_vs_High_V_User_Harm",
        s=TraitSource(
            s_primary=["救世主情结", "弱势群体优先"],
            s_secondary_factor="F_共情关怀",
            s_secondary_direction="+",
        ),
        p=CognitiveRoutingPredicate(
            p_modulation="Strong+",
            p_surface="defiant_overrides",
            bypass=True,
        ),
        o_primary="V",
        o_subtype="V1",
        fold=0,
    )
    defaults.update(overrides)
    return TCAPSQuad(**defaults)


class TestQuadConstruction:

    def test_minimal_quad_constructs_and_validates(self):
        quad = _make_valid_quad()
        validate_quad(quad)

    def test_o_subtype_optional(self):
        """v1.1 允许 o_subtype=None (学姐数据未齐时)."""
        quad = _make_valid_quad(o_subtype=None)
        validate_quad(quad)

    def test_round_trip_dict(self):
        quad = _make_valid_quad()
        d = quad.to_dict()
        as_json = json.dumps(d, ensure_ascii=False)
        d2 = json.loads(as_json)
        quad2 = TCAPSQuad.from_dict(d2)
        validate_quad(quad2)
        assert quad2.C == quad.C
        assert quad2.s.s_primary == quad.s.s_primary
        assert quad2.p.p_surface == quad.p.p_surface
        assert quad2.o_subtype == quad.o_subtype


class TestQuadConsistency:

    def test_subtype_field_must_match_primary(self):
        """o_subtype=V1 但 o_primary=P → 拒绝."""
        with pytest.raises(QuadValidationError, match="字段不一致"):
            validate_quad(_make_valid_quad(o_primary="P", o_subtype="V1"))

    def test_empty_required_ids(self):
        with pytest.raises(QuadValidationError, match="quad_id 不能为空"):
            validate_quad(_make_valid_quad(quad_id=""))
        with pytest.raises(QuadValidationError, match="designer_id 不能为空"):
            validate_quad(_make_valid_quad(designer_id=""))


# =====================================================================
# Predicate codebook (surface ↔ modulation 多对一)
# =====================================================================

class TestPredicateCodebook:

    def setup_method(self):
        reset_codebook_cache()

    def test_default_codebook_has_8_words(self):
        cb = load_predicate_codebook()
        assert len(cb.surface_words()) == 8
        assert set(cb.surface_words()) == set(SURFACE_PREDICATES)

    def test_surface_to_modulation_strong_plus(self):
        modu, byp, stance = map_surface_to_modulation("forces_transition_to")
        assert modu == "Strong+"
        assert byp is False
        assert stance == "neutral"

    def test_surface_to_modulation_defiant_bypass(self):
        modu, byp, stance = map_surface_to_modulation("defiant_overrides")
        assert modu == "Strong+"
        assert byp is True
        assert stance == "Defiance"

    def test_surface_to_modulation_bypass_only(self):
        modu, byp, stance = map_surface_to_modulation("bypasses")
        assert modu == "Bypass"
        assert byp is True
        assert stance == "neutral"

    def test_surface_to_modulation_unknown(self):
        with pytest.raises(KeyError):
            map_surface_to_modulation("invented_surface_word")

    def test_all_stances_covered(self):
        """4 个 stance 类别都应该至少出现一次 (neutral / Defiance / Compliance / Negotiation)."""
        cb = load_predicate_codebook()
        stances = {cb.surface_to_stance(w) for w in cb.surface_words()}
        assert "neutral" in stances
        assert "Defiance" in stances
        assert "Compliance" in stances
        assert "Negotiation" in stances


# =====================================================================
# Situation codebook (要求 v0 json 在 data/processed/ 下)
# =====================================================================

class TestSituationCodebookLoad:
    """要 data/processed/situation_codebook_v0.json 存在; 不存在则 skip."""

    def setup_method(self):
        self.codebook_path = (
            Path(__file__).parent.parent.parent
            / "data" / "processed" / "situation_codebook_v0.json"
        )

    def test_codebook_file_exists(self):
        if not self.codebook_path.exists():
            pytest.skip(f"codebook v0 不存在: {self.codebook_path}")

    def test_load_codebook(self):
        if not self.codebook_path.exists():
            pytest.skip("codebook v0 不存在")
        sc = load_situation_codebook(self.codebook_path)
        assert sc.version  # 有版本号
        assert "Q3" in sc.c_local_by_question
        assert "Q6" in sc.c_local_by_question

    def test_q3_q6_tags_are_locked(self):
        """学姐 2026-05-23 给的 Q3-Q6 4 个 tag 都应 locked, 不是 TBD."""
        if not self.codebook_path.exists():
            pytest.skip("codebook v0 不存在")
        sc = load_situation_codebook(self.codebook_path)
        for qid in ["Q3", "Q4", "Q5", "Q6"]:
            tag = sc.tag_for_question(qid)
            assert tag is not None, f"{qid} tag is TBD/None"
            # 跑命名 validator, 确保学姐 tag normalize 后符合约束
            validate_c_local_tag(tag)

    def test_q1_q2_q7_q8_pending(self):
        """学姐还没给 Q1/Q2/Q7/Q8, tag 应该 TBD."""
        if not self.codebook_path.exists():
            pytest.skip("codebook v0 不存在")
        sc = load_situation_codebook(self.codebook_path)
        for qid in ["Q1", "Q2", "Q7", "Q8"]:
            tag = sc.tag_for_question(qid)
            assert tag is None, f"{qid} 应该 None (TBD), 但是 {tag!r}"


if __name__ == "__main__":
    # python -m sft_pipeline.tests.test_quad_schema → 跑全套
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
