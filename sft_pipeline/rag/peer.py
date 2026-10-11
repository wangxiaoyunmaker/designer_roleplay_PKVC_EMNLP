"""Peer retrieval v3.7: 全局 trait-similarity (default) + og_rag (兼容旧).

v3.7 default: retrieve_via_trait_similarity
  - 全局 (不分簇) 4-d factor cosine top-k
  - 替代旧 retrieve_via_og_rag 的 "簇内" 约束 (datav3 L3 cluster 弱区分)

Legacy 保留: retrieve_via_og_rag (greedy set cover over hypergraph)
  - require_same_cluster=True 是 v3.5.1 default; v3.7 起改 False (全局 fallback)
  - v3.6 §9 og_rag 仍 ablation 之一 (RAG-abl 4 档: cosine / og_rag / modulation / og_rag+modulation)
  - schema 不变, 只是 scope 改

详见: docs/briefs/v3.7_patch_post_datav3_20260524.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hypergraph import HyperGraph, HyperEdgeKey, SegmentNode
from .ontology import Factor, get_scenario_probe_factors


@dataclass
class RetrievalResult:
    selected_edges: list[HyperEdgeKey]
    selected_nodes: list[SegmentNode]
    covered_elements: set[str]
    uncovered_elements: set[str]
    n_candidates: int


def infer_target_factors_from_profile_polarity(
    polarity: dict[str, list[str]],
) -> set[Factor]:
    """从 NL 极性 profile 推断 target 显著激活的因子.

    简化版: 强 + / 弱 + 的题目数量 → 哪个因子最多 (但当前 profile 用 question_node 名,
    跟 PKVC factor 直接映射需要题目-factor metadata; placeholder 用启发).
    """
    # TODO: 等题目-factor metadata 来了, 改成精确映射
    # 当前用 placeholder: 默认所有 factor 都"需要 cover"
    return {"P", "K", "V", "C"}


def retrieve_via_trait_similarity(
    target_user_id: str,
    target_factor_scores: np.ndarray,           # shape (4,), v3.7 4-d EFA factor
    target_scenario_id: str,
    candidate_segments: list[SegmentNode],       # 全局, 不分簇
    user_to_factor_scores: dict[str, np.ndarray],
    k_max: int = 3,
) -> RetrievalResult:
    """v3.7 default: 全局 trait-similarity top-k peer retrieval.

    替代 retrieve_via_og_rag 的 "簇内" 约束 (datav3 L3 cluster 弱区分).
    similarity 直接在 4-d EFA factor 空间 cosine.

    Args:
      target_user_id: 排除自检索
      target_factor_scores: 当前 designer 的 4-d factor scores
      target_scenario_id: 仅检索同 scenario 的 segment (情境匹配)
      candidate_segments: 全部 SegmentNode (全局, 不依赖簇)
      user_to_factor_scores: lookup table {user_id: factor_scores}
      k_max: top-k

    Returns:
      RetrievalResult (跟 og_rag 兼容, covered_elements 用 scenario+factor 集)
    """
    target_norm = np.linalg.norm(target_factor_scores)
    if target_norm == 0:
        target_norm = 1.0
    target_vec = target_factor_scores / target_norm

    candidates_with_sim = []
    for seg in candidate_segments:
        if seg.designer_id == target_user_id:
            continue
        if seg.scenario_id != target_scenario_id:
            continue
        if seg.designer_id not in user_to_factor_scores:
            continue
        peer_scores = user_to_factor_scores[seg.designer_id]
        peer_norm = np.linalg.norm(peer_scores)
        if peer_norm == 0:
            continue
        sim = float(np.dot(target_vec, peer_scores / peer_norm))
        candidates_with_sim.append((seg, sim))

    candidates_with_sim.sort(key=lambda x: -x[1])
    selected_nodes = [seg for seg, _ in candidates_with_sim[:k_max]]

    # 兼容 og_rag 的 RetrievalResult schema
    return RetrievalResult(
        selected_edges=[],          # trait-similarity 不用 hyperedge 概念
        selected_nodes=selected_nodes,
        covered_elements={f"scenario:{target_scenario_id}"},
        uncovered_elements=set(),
        n_candidates=len(candidates_with_sim),
    )


def retrieve_via_og_rag(
    target_user_id: str,
    target_cluster: int,
    target_scenario_id: str,
    target_profile_polarity: dict[str, list[str]] | None,
    hypergraph: HyperGraph,
    k_max: int = 3,
    require_same_cluster: bool = False,         # v3.7: default False (cluster 弱区分)
) -> RetrievalResult:
    """Greedy set cover retrieval.

    Args:
      target_user_id: 用于 exclude self
      target_cluster: 必须同簇 (硬约束, 跟 v3.4 brief 锁定一致)
      target_scenario_id: 当前 query 的 question (用于推 probe factor)
      target_profile_polarity: NL 极性 profile (推 designer 主导因子)
      hypergraph: 已 built 的 hypergraph
      k_max: 最多选几条超边

    Returns:
      RetrievalResult 含 selected nodes
    """
    # 1. 算 cover target
    scenario_factors = get_scenario_probe_factors(target_scenario_id)
    profile_factors = (infer_target_factors_from_profile_polarity(target_profile_polarity)
                        if target_profile_polarity else set())
    target_elements: set[str] = {f"factor:{f}" for f in (scenario_factors | profile_factors)}
    # 元素也可以加 scenario 自身, 鼓励选同 scenario 的 hyperedge
    target_elements.add(f"scenario:{target_scenario_id}")

    # 2. 候选超边: 同 cluster, 排除 self
    candidates = []
    candidate_filter = (lambda e: e.cluster_id == target_cluster) if require_same_cluster else (lambda e: True)
    for edge_key, node_idxs in hypergraph.edges.items():
        if not candidate_filter(edge_key):
            continue
        edge_elements = {f"factor:{edge_key.factor}", f"scenario:{edge_key.scenario_id}"}
        non_self_nodes = [i for i in node_idxs if hypergraph.nodes[i].designer_id != target_user_id]
        if not non_self_nodes:
            continue
        candidates.append((edge_key, edge_elements, non_self_nodes))

    # 3. Greedy set cover
    covered: set[str] = set()
    selected_edges: list[HyperEdgeKey] = []
    selected_node_indices: set[int] = set()
    remaining = list(candidates)

    while remaining and len(selected_edges) < k_max and (target_elements - covered):
        # score: 新 cover / cost
        best, best_gain = None, -1.0
        for idx, (key, elems, nodes) in enumerate(remaining):
            gain = len((elems & target_elements) - covered)
            cost = max(len(nodes), 1)
            score = gain / cost if gain > 0 else 0
            if score > best_gain:
                best, best_gain = idx, score
        if best is None or best_gain <= 0:
            break
        key, elems, nodes = remaining.pop(best)
        selected_edges.append(key)
        covered |= elems
        selected_node_indices.update(nodes)

    selected_nodes = [hypergraph.nodes[i] for i in selected_node_indices]

    return RetrievalResult(
        selected_edges=selected_edges,
        selected_nodes=selected_nodes[:k_max * 3],  # 节点上限防爆
        covered_elements=covered,
        uncovered_elements=target_elements - covered,
        n_candidates=len(candidates),
    )


def format_peer_rag_block(result: RetrievalResult, max_chars: int = 1500) -> str:
    """把 retrieved nodes 渲染成 prompt block (跟 v3.4 brief peer RAG 接口一致)."""
    if not result.selected_nodes:
        return "[Cluster_peer_examples] (无可用同簇 peer)"
    lines = [f"[Cluster_peer_examples] (ontology-grounded, "
              f"覆盖 {result.covered_elements} / 缺 {result.uncovered_elements})"]
    char_count = len(lines[0])
    for i, node in enumerate(result.selected_nodes, 1):
        sadr_str = ", ".join(node.sadr_activations[:5])
        block = (f"\n样例 {i} (designer {node.designer_id[:8]}, "
                  f"scenario {node.scenario_id}, 主因子 {node.dominant_factor}):\n"
                  f"  SADR: {sadr_str}\n"
                  f"  NL: {node.raw_nl[:300]}")
        if char_count + len(block) > max_chars:
            lines.append(f"\n... ({len(result.selected_nodes) - i + 1} more samples truncated)")
            break
        lines.append(block)
        char_count += len(block)
    return "\n".join(lines)
