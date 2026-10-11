"""从 train fold designer 数据构建 ontology-grounded hypergraph.

节点 (node):  (designer_id, scenario_id) — 一个 designer 在一个 question 上的 response
超边 (hyperedge): 由 (cluster_id, dominant_factor, scenario_type) 锚定的节点集合

每条超边 = "簇 c 内 dominant 因子 f 的 designer 在 scenario s 上的 response 集合"
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .ontology import Factor, parse_factor_path, parse_sadr_id_with_polarity


@dataclass(frozen=True)
class HyperEdgeKey:
    cluster_id: int
    factor: Factor                  # dominant factor of designer at this scenario
    scenario_id: str                # "Q1" / "Q7_seg1" 等

    def __str__(self) -> str:
        return f"<c={self.cluster_id}, f={self.factor}, s={self.scenario_id}>"


@dataclass
class SegmentNode:
    """超图节点: 一个 designer 在一个 scenario 上的 response."""
    designer_id: str
    scenario_id: str
    cluster_id: int
    dominant_factor: Factor | None
    sadr_activations: list[str] = field(default_factory=list)  # ['#V3+', '#服务保留-', ...]
    raw_nl: str = ""                # 原文 (可空)
    extracted_meta: dict = field(default_factory=dict)  # extractor 返回的 dict


@dataclass
class HyperGraph:
    nodes: list[SegmentNode] = field(default_factory=list)
    edges: dict[HyperEdgeKey, list[int]] = field(default_factory=dict)  # edge → node indices
    by_designer: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    by_cluster: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: SegmentNode) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        self.by_designer[node.designer_id].append(idx)
        self.by_cluster[node.cluster_id].append(idx)

        if node.dominant_factor is None:
            return idx
        # 节点可能进多条 hyperedge (e.g., factor path V→K→V 进 V 和 K 两条)
        factors = [node.dominant_factor]  # 简化: 只用 dominant; 路径段可后续扩展
        for f in factors:
            key = HyperEdgeKey(node.cluster_id, f, node.scenario_id)
            self.edges.setdefault(key, []).append(idx)
        return idx

    def edges_in_cluster(self, cluster_id: int) -> dict[HyperEdgeKey, list[int]]:
        return {k: v for k, v in self.edges.items() if k.cluster_id == cluster_id}

    def stats(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_designers": len(self.by_designer),
            "n_clusters": len(self.by_cluster),
            "n_hyperedges": len(self.edges),
            "avg_edge_size": (sum(len(v) for v in self.edges.values()) / max(len(self.edges), 1)),
        }


def build_hypergraph(
    extracted_records: list[dict],
    user_cluster: dict[str, int],
) -> HyperGraph:
    """从 extractor 输出 build hypergraph.

    Args:
      extracted_records: list of dict, 每条包含:
        - user_id
        - scenario_id  (Q1 / Q7_seg1 etc)
        - dominant_factor (字符串如 "V" 或 "V → K → V")
        - sadr_activations (list of dict or strings)
        - raw_nl (optional)
      user_cluster: user_id → cluster_id mapping
    """
    g = HyperGraph()
    for r in extracted_records:
        uid = r.get("user_id")
        cid = user_cluster.get(uid)
        if cid is None or uid is None:
            continue
        factors = parse_factor_path(r.get("dominant_factor", "") or "")
        # 简化: 取 path 第一个因子作 dominant; 也可以为每个因子各加一个 node
        dominant = factors[0] if factors else None
        sadr = []
        for s in r.get("sadr_activations", []):
            if isinstance(s, str):
                sadr.append(s)
            elif isinstance(s, dict):
                sid = s.get("id", "")
                pol = s.get("polarity", "")
                sadr.append(f"{sid}{pol}")
        node = SegmentNode(
            designer_id=uid,
            scenario_id=r.get("scenario_id", "?"),
            cluster_id=cid,
            dominant_factor=dominant,
            sadr_activations=sadr,
            raw_nl=r.get("raw_nl", ""),
            extracted_meta=r,
        )
        g.add_node(node)
    return g
