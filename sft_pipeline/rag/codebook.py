"""Codebook loader: situation_codebook (C_local) + predicate_codebook (p_surface ↔ p_modulation).

Reference: docs/briefs/T_CAPS_quad_schema_v1.md §2-§3
           data/processed/situation_codebook_v0.json

提供 3 个 loader + 1 个映射函数:
   load_situation_codebook(path=None)   → SituationCodebook
   load_predicate_codebook(path=None)   → PredicateCodebook
   load_question_to_c_local(path=None)  → dict[question_id, C_local tag]
   map_surface_to_modulation(surface)   → (modulation_level, bypass, stance)

学姐 metadata 来了不动代码, 只改 JSON 文件即可生效.

依赖: 仅 stdlib (json, pathlib).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 默认 codebook 路径 (相对 repo 根)
_DEFAULT_CODEBOOK_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "situation_codebook_v0.json"


# =====================================================================
# Predicate codebook (p_surface ↔ p_modulation, v0 锁 8 词)
# =====================================================================

# 多对一映射, 跟 docs/briefs/T_CAPS_quad_schema_v1.md §3 同步.
# 改这里之前先改 brief + situation_codebook json 里的 predicate_codebook 段.
_PREDICATE_CODEBOOK_DEFAULT: dict[str, dict[str, Any]] = {
    "forces_transition_to": {"modulation": "Strong+", "bypass": False, "stance": "neutral"},
    "severely_overrides":   {"modulation": "Strong+", "bypass": True,  "stance": "neutral"},
    "defiant_overrides":    {"modulation": "Strong+", "bypass": True,  "stance": "Defiance"},
    "gradually_shifts_to":  {"modulation": "Weak+",   "bypass": False, "stance": "neutral"},
    "compliantly_follows":  {"modulation": "Weak+",   "bypass": False, "stance": "Compliance"},
    "negotiates_around":    {"modulation": "Weak+",   "bypass": False, "stance": "Negotiation"},
    "weakly_inhibits":      {"modulation": "Weak-",   "bypass": False, "stance": "neutral"},
    "bypasses":             {"modulation": "Bypass",  "bypass": True,  "stance": "neutral"},
}


@dataclass(slots=True)
class PredicateCodebook:
    """p_surface ↔ p_modulation 多对一映射 (含 stance 色彩)."""
    mapping: dict[str, dict[str, Any]] = field(default_factory=dict)

    def surface_words(self) -> list[str]:
        return list(self.mapping.keys())

    def lookup(self, surface: str) -> dict[str, Any]:
        if surface not in self.mapping:
            raise KeyError(
                f"p_surface {surface!r} 不在 codebook 中 (合法: {list(self.mapping)})"
            )
        return self.mapping[surface]

    def surface_to_modulation(self, surface: str) -> str:
        return self.lookup(surface)["modulation"]

    def surface_to_bypass(self, surface: str) -> bool:
        return bool(self.lookup(surface)["bypass"])

    def surface_to_stance(self, surface: str) -> str:
        return self.lookup(surface)["stance"]


def load_predicate_codebook(path: Path | None = None) -> PredicateCodebook:
    """从 situation_codebook json 加载 predicate_codebook 段; 若无则用默认 8 词.

    优先级:
        1. JSON 文件里的 predicate_codebook 段 (学姐 confirm 后可改这里扩词)
        2. 模块内默认 (v0 锁的 8 词)
    """
    p = path or _DEFAULT_CODEBOOK_PATH
    if not p.exists():
        return PredicateCodebook(mapping=dict(_PREDICATE_CODEBOOK_DEFAULT))
    obj = json.loads(p.read_text(encoding="utf-8"))
    pred = obj.get("predicate_codebook")
    if pred is None:
        return PredicateCodebook(mapping=dict(_PREDICATE_CODEBOOK_DEFAULT))
    return PredicateCodebook(mapping=pred)


# =====================================================================
# Situation codebook (C_local tag 词表 + question → C_local 映射)
# =====================================================================

@dataclass(slots=True)
class CLocalEntry:
    """C_local codebook 里的一条 (一道题对应一个 tag)."""
    question_id: str
    tag: str
    polarity_type: str | None = None               # "high_vs_high" / "high_vs_low" / ...
    PKVC_left: str | None = None
    PKVC_right: str | None = None
    PKVC_left_subtype: str | None = None
    PKVC_right_subtype: str | None = None
    status: str | None = None                      # "locked_by_xj_..." / "pending_xj" / ...
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SituationCodebook:
    """整份 situation_codebook (包含 C_local 词表 + C_global 元信息 + naming constraint)."""
    version: str
    locked_at: str
    c_local_by_question: dict[str, CLocalEntry] = field(default_factory=dict)
    c_global: dict[str, Any] = field(default_factory=dict)
    naming_constraint: dict[str, Any] = field(default_factory=dict)
    o_subtype_codebook: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def tag_for_question(self, question_id: str) -> str | None:
        """返回该题的 C_local tag; 若是 TBD 或不存在, 返回 None."""
        entry = self.c_local_by_question.get(question_id)
        if entry is None or entry.tag in (None, "TBD"):
            return None
        return entry.tag

    def all_locked_tags(self) -> list[str]:
        """返回所有 status 为 locked 的 C_local tag (不含 TBD / pending)."""
        return [
            e.tag for e in self.c_local_by_question.values()
            if e.tag != "TBD" and e.status and e.status.startswith("locked")
        ]


def load_situation_codebook(path: Path | None = None) -> SituationCodebook:
    """加载 data/processed/situation_codebook_v0.json → SituationCodebook 对象.

    Args:
        path: 不指定则用 _DEFAULT_CODEBOOK_PATH.

    Raises:
        FileNotFoundError: 文件不存在 (学姐 metadata 没回 + 没起 v0 占位).
        json.JSONDecodeError: 文件格式坏.
    """
    p = path or _DEFAULT_CODEBOOK_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"situation codebook 不存在: {p}\n"
            f"   学姐 metadata 没回 → 先用 data/processed/situation_codebook_v0.json 占位"
        )
    obj = json.loads(p.read_text(encoding="utf-8"))

    c_local_raw = obj.get("C_local", {})
    c_local_by_q: dict[str, CLocalEntry] = {}
    for qid, entry_raw in c_local_raw.items():
        c_local_by_q[qid] = CLocalEntry(
            question_id=qid,
            tag=entry_raw.get("tag", "TBD"),
            polarity_type=entry_raw.get("polarity_type"),
            PKVC_left=entry_raw.get("PKVC_left"),
            PKVC_right=entry_raw.get("PKVC_right"),
            PKVC_left_subtype=entry_raw.get("PKVC_left_subtype"),
            PKVC_right_subtype=entry_raw.get("PKVC_right_subtype"),
            status=entry_raw.get("status"),
            raw=entry_raw,
        )

    return SituationCodebook(
        version=obj.get("version", "v0"),
        locked_at=obj.get("locked_at", "unknown"),
        c_local_by_question=c_local_by_q,
        c_global=obj.get("C_global", {}),
        naming_constraint=obj.get("naming_constraint", {}),
        o_subtype_codebook=obj.get("o_subtype_codebook", {}),
        notes=list(obj.get("notes", [])),
    )


def load_question_to_c_local(path: Path | None = None) -> dict[str, str | None]:
    """Convenience: 直接拿 {question_id: C_local tag or None} dict.

    TBD 的题返回 None, 不要 silent 用 "TBD" 字符串混淆下游.
    """
    sc = load_situation_codebook(path)
    return {
        qid: sc.tag_for_question(qid)
        for qid in sc.c_local_by_question
    }


# =====================================================================
# Surface → modulation 顶层函数 (使用最频繁, 单独导出)
# =====================================================================

# 模块级 cache, 避免每次 LLM extractor 输出 quad 都重新 load json
_PREDICATE_CODEBOOK_CACHE: PredicateCodebook | None = None


def map_surface_to_modulation(
    surface: str,
    path: Path | None = None,
) -> tuple[str, bool, str]:
    """LLM extractor 输出 p_surface 后, 查 codebook 填 (modulation, bypass, stance).

    Returns:
        (modulation_level, bypass_flag, stance) 三元.
        modulation_level ∈ MODULATION_LEVELS, bypass ∈ {True, False},
        stance ∈ {"neutral", "Defiance", "Compliance", "Negotiation"}.

    Raises:
        KeyError: surface 不在 codebook (LLM 输出脏数据).
    """
    global _PREDICATE_CODEBOOK_CACHE
    if _PREDICATE_CODEBOOK_CACHE is None:
        _PREDICATE_CODEBOOK_CACHE = load_predicate_codebook(path)
    cb = _PREDICATE_CODEBOOK_CACHE
    entry = cb.lookup(surface)
    return entry["modulation"], bool(entry["bypass"]), entry["stance"]


def reset_codebook_cache() -> None:
    """Test fixture: 清 cache, 让下次 load 重新读 json (单元测试用)."""
    global _PREDICATE_CODEBOOK_CACHE
    _PREDICATE_CODEBOOK_CACHE = None
