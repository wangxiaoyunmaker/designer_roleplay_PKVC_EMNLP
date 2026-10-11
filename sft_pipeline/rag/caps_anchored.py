"""CAPS-anchored retrieval + role-play infrastructure.

Joins 3 data sources to build unified CAPS pool:
  1. data_v3/processed/typology_soft_assignment_per_qid_v3.json
     → {designer_id → {qid → {typology_id → prob}}}, per-qid distribution
  2. data_v3/processed/q7_extracted_v8.7_full.jsonl
     → 486 Q7 segments (162 designer × 3 seg) with PKVC v8.7 labels
  3. data_v3/processed/q1q6_extracted_v8.7_full.jsonl
     → 309 Q1-Q6 expansion segments with PKVC labels

设计 (v3.8):
  - 砍 cluster, 全局 pool
  - CAPS signature = per-qid typology distribution (Q3-Q6 4 dists)
  - retrieve top-k peers by typology dist similarity (Jensen-Shannon distance, lower=more similar)
  - 必须 peer 有 target_scenario 对应的 NL 才入选
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PKVCSegment:
    """1 个 PKVC-labeled segment (Q7 seg1/2/3 或 Q1-Q6 expansion)."""
    designer_id: str
    qid: str                                # "Q7" / "Q1" / ...
    segment_role: str                       # "seg1" / "seg2" / "seg3" / "single"
    text: str                               # raw NL
    dominant_factor: str                    # "V" / "K" / "V→K" 等
    state_sequence: list[str]
    criterion: Optional[str]
    confidence: str
    notes: str = ""

    @property
    def scenario_key(self) -> str:
        """e.g. 'Q7_seg1', 'Q5'."""
        if self.qid == "Q7":
            return f"Q7_{self.segment_role}"
        return self.qid


@dataclass
class Designer:
    """Unified designer representation with all 3 data joined."""
    designer_id: str
    typology_dist: dict[str, dict[str, float]] = field(default_factory=dict)
    # {"Q3": {"Q3B-1": 0.85, ...}, "Q4": {...}, ...}
    segments: dict[str, PKVCSegment] = field(default_factory=dict)
    # scenario_key → PKVCSegment, e.g. "Q7_seg1" → seg, "Q5" → seg

    def has_scenario(self, scenario_key: str) -> bool:
        return scenario_key in self.segments

    def get_scenario(self, scenario_key: str) -> Optional[PKVCSegment]:
        return self.segments.get(scenario_key)


@dataclass
class CAPSPool:
    designers: dict[str, Designer]
    typology_ids: list[str]                 # 28 typology candidate ids
    typology_by_qid: dict[str, list[str]]   # {"Q3": ["Q3A-1", ...]}

    def filter_pool_with_scenario(self, scenario_key: str) -> list[Designer]:
        """Return designers who have this scenario response."""
        return [d for d in self.designers.values() if d.has_scenario(scenario_key)]


# ============================================================
# Loaders
# ============================================================

def load_q7_segments(path: Path) -> list[PKVCSegment]:
    out = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        pred = r["pred"]
        out.append(PKVCSegment(
            designer_id=r["user_id"],
            qid="Q7",
            segment_role=r["segment_role"],
            text=r["text"],
            dominant_factor=pred["dominant_factor"],
            state_sequence=pred["state_sequence"],
            criterion=pred.get("criterion"),
            confidence=pred["confidence"],
            notes=pred.get("extraction_notes", ""),
        ))
    return out


def load_q16_segments(path: Path) -> list[PKVCSegment]:
    out = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        pred = r["pred"]
        out.append(PKVCSegment(
            designer_id=r["user_id"],
            qid=r["qid"],
            segment_role="single",
            text=r["text"],
            dominant_factor=pred["dominant_factor"],
            state_sequence=pred["state_sequence"],
            criterion=pred.get("criterion"),
            confidence=pred["confidence"],
            notes=pred.get("extraction_notes", ""),
        ))
    return out


def load_typology_dist(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_caps_pool(
    typology_path: Path = Path("data_v3/processed/typology_soft_assignment_per_qid_v3.json"),
    q7_path: Path = Path("data_v3/processed/q7_extracted_v8.7_full.jsonl"),
    q16_path: Path = Path("data_v3/processed/q1q6_extracted_v8.7_full.jsonl"),
) -> CAPSPool:
    typology_dist = load_typology_dist(typology_path)
    q7_segs = load_q7_segments(q7_path)
    q16_segs = load_q16_segments(q16_path)

    # build Designer index
    designer_index: dict[str, Designer] = {}

    def get_or_create(did: str) -> Designer:
        if did not in designer_index:
            designer_index[did] = Designer(designer_id=did)
        return designer_index[did]

    # typology dist
    for did, dists in typology_dist.items():
        d = get_or_create(did)
        d.typology_dist = dists

    # Q7 segments
    for seg in q7_segs:
        d = get_or_create(seg.designer_id)
        d.segments[seg.scenario_key] = seg

    # Q1-Q6 segments
    for seg in q16_segs:
        d = get_or_create(seg.designer_id)
        d.segments[seg.scenario_key] = seg

    # collect all typology ids
    typo_ids_set = set()
    typo_by_qid: dict[str, set[str]] = defaultdict(set)
    for did, dists in typology_dist.items():
        for qid, td in dists.items():
            for tid in td:
                typo_ids_set.add(tid)
                typo_by_qid[qid].add(tid)

    return CAPSPool(
        designers=designer_index,
        typology_ids=sorted(typo_ids_set),
        typology_by_qid={qid: sorted(ts) for qid, ts in typo_by_qid.items()},
    )


# ============================================================
# Signature similarity
# ============================================================

def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two distributions on same support.
    Returns ∈ [0, log(2)], 0 = identical.
    """
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p_arr = [p.get(k, 0.0) for k in keys]
    q_arr = [q.get(k, 0.0) for k in keys]
    p_sum = sum(p_arr) or 1.0
    q_sum = sum(q_arr) or 1.0
    p_norm = [x / p_sum for x in p_arr]
    q_norm = [x / q_sum for x in q_arr]

    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                s += ai * math.log(ai / bi)
            elif ai > 0:
                s += ai * 30  # huge penalty for zero in target
        return s

    m = [(a + b) / 2 for a, b in zip(p_norm, q_norm)]
    return 0.5 * kl(p_norm, m) + 0.5 * kl(q_norm, m)


def signature_distance(d1: Designer, d2: Designer) -> float:
    """Distance ∈ [0, ∞), lower=more similar.
    Aggregate JS divergence across 4 qids (Q3-Q6). Skip qid if either missing.
    """
    total = 0.0
    n = 0
    for qid in ("Q3", "Q4", "Q5", "Q6"):
        if qid in d1.typology_dist and qid in d2.typology_dist:
            total += js_divergence(d1.typology_dist[qid], d2.typology_dist[qid])
            n += 1
    if n == 0:
        return float("inf")
    return total / n


def signature_similarity(d1: Designer, d2: Designer) -> float:
    """Similarity ∈ (0, 1], 1=identical signature. sim = 1 / (1 + dist)."""
    dist = signature_distance(d1, d2)
    if dist == float("inf"):
        return 0.0
    return 1.0 / (1.0 + dist)


# ============================================================
# Retrieval
# ============================================================

@dataclass
class Peer:
    designer_id: str
    similarity: float
    segment: PKVCSegment


def retrieve_caps_anchored_top3(
    target_designer_id: str,
    target_scenario: str,            # "Q7_seg1" or "Q5" etc
    pool: CAPSPool,
    k: int = 3,
    exclude_self: bool = True,
) -> list[Peer]:
    """CAPS-anchored top-k peer retrieval.

    步骤:
      1. 排除 self (按 designer_id)
      2. 只保留有 target_scenario 对应 NL 的 candidate peer
      3. 按 typology distribution 相似 (Jensen-Shannon) 排序
      4. 返回 top-k peer (含 NL + PKVC label + similarity)
    """
    if target_designer_id not in pool.designers:
        raise KeyError(f"designer {target_designer_id} not in pool")
    target = pool.designers[target_designer_id]

    candidates: list[Peer] = []
    for did, peer in pool.designers.items():
        if exclude_self and did == target_designer_id:
            continue
        seg = peer.get_scenario(target_scenario)
        if seg is None:
            continue
        sim = signature_similarity(target, peer)
        candidates.append(Peer(designer_id=did, similarity=sim, segment=seg))

    candidates.sort(key=lambda p: -p.similarity)
    return candidates[:k]


# ============================================================
# CLI / sanity
# ============================================================

def main():
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Q7_seg2", help="target scenario, e.g. Q7_seg2, Q5")
    ap.add_argument("--target", default=None, help="target designer id; default=random")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    print("[load] CAPSPool...")
    pool = load_caps_pool()
    print(f"  {len(pool.designers)} designers loaded")
    print(f"  {len(pool.typology_ids)} unique typology ids (across {sorted(pool.typology_by_qid)})")

    # scenario coverage
    n_with_scenario = sum(1 for d in pool.designers.values() if d.has_scenario(args.scenario))
    print(f"  {n_with_scenario} designers have {args.scenario}")
    if n_with_scenario == 0:
        print("  ERROR: 0 coverage")
        return 1

    # pick target
    candidates_target = [d for d in pool.designers.values()
                         if d.has_scenario(args.scenario) and d.typology_dist]
    if args.target:
        target = pool.designers[args.target]
    else:
        target = candidates_target[0]
    print(f"\n[target] designer={target.designer_id} scenario={args.scenario}")
    print(f"  typology dist (Q3-Q6):")
    for qid in ("Q3","Q4","Q5","Q6"):
        if qid in target.typology_dist:
            top3 = sorted(target.typology_dist[qid].items(), key=lambda x: -x[1])[:3]
            print(f"    {qid}: {', '.join(f'{tid}={p:.2f}' for tid, p in top3)}")
    target_seg = target.get_scenario(args.scenario)
    print(f"\n  target's own {args.scenario} (ground truth):")
    print(f"    dom={target_seg.dominant_factor}  seq={target_seg.state_sequence}  conf={target_seg.confidence}")
    print(f"    text ({len(target_seg.text)}字): {target_seg.text[:250]}...")

    # retrieve
    peers = retrieve_caps_anchored_top3(target.designer_id, args.scenario, pool, k=args.k)
    print(f"\n[retrieved] top-{args.k} CAPS-anchored peers:")
    for i, peer in enumerate(peers, 1):
        print(f"\n  Peer {i}: {peer.designer_id} (sim={peer.similarity:.3f})")
        seg = peer.segment
        print(f"    dom={seg.dominant_factor}  seq={seg.state_sequence}  conf={seg.confidence}")
        print(f"    text ({len(seg.text)}字): {seg.text[:200]}...")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
