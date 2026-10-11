"""PKVC label comparison utilities.

Label format (per extractor v8 `dominant_factor` field):
  - 单字母: "P" / "K" / "V" / "C"
  - 路径:   "K→V" / "P→C" / ... (v8 铁律 6: 上限 2 段, 头≠尾, "→" 是 U+2192)

5 比较模式 (供 L8 RAG eval / quad validation / judge rubric 共用):
  1. exact_match           — 严格相等 (含路径方向)
  2. same_dominant_factor  — head (起点) 相同 = 同初始激活因子
  3. same_tail_factor      — tail (终点) 相同 = 同收束因子 (path 时序最终 state)
  4. pkvc_jaccard          — 字母集合交并比 (忽略路径方向)
  5. pkvc_sequence_sim     — 综合加权 (exact 0.5 + head 0.2 + tail 0.2 + jaccard 0.1)
                              连续 [0,1] 分数, 默认用于 retrieval concordance

设计原则:
  - 路径 "X→Y" 按 v8 是**时序** (X 在段首 / Y 段尾收束), 不是强弱; 因此 head/tail 各
    有独立意义 (起点 = 第一反应; 终点 = 最终落点)
  - "→" 跟 ASCII "->" 都接受 (兼容手写)
"""
from __future__ import annotations

_ARROW_CHARS = ("→", "->")


def _parse_label(label: str) -> list[str]:
    """Parse "K→V" → ["K", "V"]; "P" → ["P"]. 容错 ASCII '->'."""
    if not isinstance(label, str) or not label.strip():
        return []
    s = label.strip()
    for arrow in _ARROW_CHARS:
        if arrow in s:
            return [p.strip() for p in s.split(arrow) if p.strip()]
    return [s]


def exact_match(a: str, b: str) -> bool:
    """完全相等 (含路径方向). "K→V" != "V→K"."""
    pa, pb = _parse_label(a), _parse_label(b)
    if not pa or not pb:
        return False
    return pa == pb


def same_dominant_factor(a: str, b: str) -> bool:
    """head (路径起点 / 单字母本身) 相同 = 同初始激活因子."""
    pa, pb = _parse_label(a), _parse_label(b)
    if not pa or not pb:
        return False
    return pa[0] == pb[0]


def same_tail_factor(a: str, b: str) -> bool:
    """tail (路径终点 / 单字母本身) 相同 = 同收束因子."""
    pa, pb = _parse_label(a), _parse_label(b)
    if not pa or not pb:
        return False
    return pa[-1] == pb[-1]


def pkvc_jaccard(a: str, b: str) -> float:
    """字母集合交并比 (忽略路径方向). [0, 1].

    Examples:
      "P" vs "P"           → 1.0
      "P" vs "K"           → 0.0
      "K→V" vs "V→K"       → 1.0  (same set)
      "K→V" vs "K→P"       → 1/3  ({K,V} ∩ {K,P} = {K}, union {K,V,P})
      "K" vs "K→V"         → 0.5  ({K} ∩ {K,V} = {K}, union {K,V})
    """
    sa, sb = set(_parse_label(a)), set(_parse_label(b))
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def pkvc_sequence_sim(a: str, b: str) -> float:
    """综合序列相似度 [0, 1], 加权:
      0.5 * exact + 0.2 * head + 0.2 * tail + 0.1 * jaccard

    捕捉:
      - 路径方向 (exact)
      - 起点匹配 (head, 第一激活因子)
      - 终点匹配 (tail, 最终收束)
      - 涉及因子重叠 (jaccard)

    Examples:
      "P" vs "P"        → 1.0
      "K→V" vs "K→V"    → 1.0
      "K→V" vs "K→P"    → 0.5*0 + 0.2*1 + 0.2*0 + 0.1*(1/3) ≈ 0.233
      "K→V" vs "P→V"    → 0.5*0 + 0.2*0 + 0.2*1 + 0.1*(1/3) ≈ 0.233
      "K→V" vs "V→K"    → 0.5*0 + 0.2*0 + 0.2*0 + 0.1*1     = 0.1
      "K" vs "K→V"      → 0.5*0 + 0.2*1 + 0.2*0 + 0.1*0.5   = 0.25
      "P" vs "K"        → 0.0
    """
    if not _parse_label(a) or not _parse_label(b):
        return 0.0
    score = 0.0
    score += 0.5 * (1.0 if exact_match(a, b) else 0.0)
    score += 0.2 * (1.0 if same_dominant_factor(a, b) else 0.0)
    score += 0.2 * (1.0 if same_tail_factor(a, b) else 0.0)
    score += 0.1 * pkvc_jaccard(a, b)
    return score


# Convenience lookup for batch comparison
COMPARATORS = {
    "exact": exact_match,
    "head": same_dominant_factor,
    "tail": same_tail_factor,
    "jaccard": pkvc_jaccard,
    "sequence_sim": pkvc_sequence_sim,
}
