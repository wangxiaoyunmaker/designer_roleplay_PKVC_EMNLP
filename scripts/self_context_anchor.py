"""B4-PlusSelf: 同情境自锚定 block（替代 B4-PlusPeer 的他人锚定）。

动机与实证依据（2026-07-26/27，Q7 human adjudicated gold，n=148）：

  锚定源                     KC-NES Δ(本人−他人)   CI95                   判定
  seg1 单独                  +0.0178              [-0.0067,+0.0443]      不显著
  seg3 单独                  +0.0195              [-0.0066,+0.0466]      不显著
  seg1+seg3（本模块默认）     +0.0278              [+0.0042,+0.0536]      显著 ✓
  状态分布JS（seg1+seg3）     +0.0394              [+0.0039,+0.0730]      显著 ✓

对照：用「他人」的同情境片段全部无效——V30/画像近邻检索预测本人 seg2 得
0.2611~0.3195，均低于群体 medoid 0.3592，且近邻不优于最远邻；
Mantel(V30距离, 行为距离) r=-0.004, p=0.947。

因此 B4-PlusPeer 的 trait-pred ρ=+0.077（唯一 CI>0）很可能来自检索循环
（trait 相似者获得重叠 peer 集合 → prompt 相似 → 输出相似 → ρ 机械抬升），
这也解释了它同时付出 +41% macro-TED 的代价。把锚定源从「他人」换成
「本人其他段」即可在不牺牲保真度的前提下获得个体化。

协议：leave-one-segment-out。目标段本身**绝不**进入输入；只使用该设计者
其他已观测段。至少需要 2 段才能达到可检测的信度（单段信度仅 0.08~0.15）。
"""
from __future__ import annotations

from typing import Any

# Q7 三段的语义角色，用于在 prompt 中标注锚定片段的来源
SEG_ROLE = {
    "seg1": "即时直觉反应",
    "seg2": "决策权衡推理",
    "seg3": "底线与妥协",
}

MIN_ANCHORS = 2  # 单段不显著，必须 >=2 段


class SelfAnchorError(RuntimeError):
    """目标段泄漏或锚定不足时抛出。"""


def retrieve_self_context_q7_nl(
    target_id: str,
    pool: Any,
    scenario_key: str,
    max_chars: int = 400,
) -> list[dict]:
    """返回目标设计者在**同一 Q7 情境下其他段**的自然语言回答。

    与 retrieve_caps_peer_q7_nl 的唯一区别：不检索他人，只取本人其他段。
    scenario_key 形如 "Q7_seg2"；本函数取 "Q7_seg1"、"Q7_seg3"。

    Raises:
        SelfAnchorError: 若目标段被包含进锚定（泄漏）。
    """
    designer = pool.designers.get(target_id)
    if designer is None:
        return []

    prefix, _, target_seg = scenario_key.rpartition("_")
    out: list[dict] = []
    for key, seg in sorted(designer.segments.items()):
        other_prefix, _, other_seg = key.rpartition("_")
        if other_prefix != prefix:
            continue
        if other_seg == target_seg:
            continue  # 目标段绝不进入输入
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        out.append({
            "segment": other_seg,
            "role": SEG_ROLE.get(other_seg, other_seg),
            "text": text[:max_chars],
        })

    if any(a["segment"] == target_seg for a in out):
        raise SelfAnchorError(
            f"target segment {target_seg} leaked into self anchor for {target_id}"
        )
    return out


def format_self_context_block(anchors: list[dict]) -> str:
    """格式化自锚定 block，镜像 _format_peer_q7_block 的结构。"""
    if not anchors:
        return "(本场景下无其他已观测片段)"
    blocks = []
    for a in anchors:
        blocks.append(f"### 你在本场景的{a['role']}\n{a['text']}")
    return "\n\n".join(blocks)


SELF_BLOCK_HEADER = "# 你本人在本场景其他环节的真实回答 (保持一致)"


def build_self_anchor_section(anchors: list[dict]) -> str:
    """返回可直接 append 进 prompt parts 的完整段落。"""
    return f"{SELF_BLOCK_HEADER}\n\n{format_self_context_block(anchors)}"


# ---------------------------------------------------------------- 对照条件
def shuffled_anchor_assignment(
    target_ids: list[str], seed: int,
) -> dict[str, str]:
    """无固定点的错排：为每个 target 指派另一个设计者作为锚定来源。

    用于强制对照 `shuffled anchor`。若 correct 不显著优于 shuffled，
    则个体化主张不成立。
    """
    import random

    if len(target_ids) < 2:
        raise SelfAnchorError("shuffled anchor needs at least two participants")
    rng = random.Random(seed)
    order = list(range(len(target_ids)))
    while True:
        rng.shuffle(order)
        if all(i != order[i] for i in range(len(order))):
            break
    return {target_ids[i]: target_ids[order[i]] for i in range(len(target_ids))}


def retrieve_anchor(
    target_id: str,
    pool: Any,
    scenario_key: str,
    *,
    condition: str = "correct",
    donor_map: dict[str, str] | None = None,
    max_chars: int = 400,
) -> list[dict]:
    """统一入口，支持三种预注册条件。

    condition:
      - "correct" : 本人其他段（主方法）
      - "shuffled": 错排他人的同位置段（反事实对照）
      - "none"    : 空锚定（消融对照）
    """
    if condition == "none":
        return []
    if condition == "correct":
        return retrieve_self_context_q7_nl(
            target_id, pool, scenario_key, max_chars=max_chars,
        )
    if condition == "shuffled":
        if not donor_map or target_id not in donor_map:
            raise SelfAnchorError("shuffled condition requires a donor_map")
        return retrieve_self_context_q7_nl(
            donor_map[target_id], pool, scenario_key, max_chars=max_chars,
        )
    raise SelfAnchorError(f"unknown anchor condition: {condition}")


def anchor_audit(anchors: list[dict], target_seg: str) -> dict[str, Any]:
    """随每条结果落盘的审计记录。"""
    return {
        "anchor_count": len(anchors),
        "anchor_segments": [a["segment"] for a in anchors],
        "target_segment_excluded": all(a["segment"] != target_seg for a in anchors),
        "meets_min_anchors": len(anchors) >= MIN_ANCHORS,
        "total_anchor_chars": sum(len(a["text"]) for a in anchors),
    }
