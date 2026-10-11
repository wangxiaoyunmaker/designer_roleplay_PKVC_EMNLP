"""Demo: 用 79 designer 数据跑一遍 og_rag pipeline (CPU, 不需要 GPU / API).

由于真实 extractor 还没跑 (等 GPT-4o + Q7 yaml 真值), 用 mock 数据演示 hypergraph
+ retrieval 的工程通畅性. 当数据齐了把 mock 段替换成真实 extracted records 即可.

跑法:
  cd /data/wy/Designer_RolePlay
  .venv/bin/python sft_pipeline/rag/demo.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# 允许从 sft_pipeline.rag 导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sft_pipeline.rag.hypergraph import build_hypergraph  # noqa: E402
from sft_pipeline.rag.peer import (                  # noqa: E402
    retrieve_via_og_rag, format_peer_rag_block,
)
from sft_pipeline.rag.ontology import FACTORS, SADR_NODES, SCENARIO_TYPES  # noqa: E402


DATA_DIR = Path("/data/wy/Designer_RolePlay/data")
CLUSTER_CSV = DATA_DIR / "user_cluster_efa.csv"


def make_mock_extracted_records(cluster_df: pd.DataFrame, seed: int = 42) -> list[dict]:
    """每个 designer × 7 个 scenario (Q1-Q6 + Q7_seg2) 生成 mock extracted record."""
    rng = random.Random(seed)
    factors = list(FACTORS.keys())
    scenarios = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7_seg2"]
    sadr_ids = list(SADR_NODES.keys())

    records = []
    for _, row in cluster_df.iterrows():
        uid = str(row["user_id"])
        for s in scenarios:
            f = rng.choice(factors)
            sample_sadr = rng.sample(sadr_ids, k=rng.randint(1, 3))
            records.append({
                "user_id": uid,
                "scenario_id": s,
                "dominant_factor": f,
                "sadr_activations": [f"{sid}+" for sid in sample_sadr],
                "raw_nl": f"[mock NL for {uid} on {s}, factor {f}]",
            })
    return records


def main() -> int:
    print("=== 加载 cluster ===")
    cluster_df = pd.read_csv(CLUSTER_CSV)
    cluster_df.columns = [c.strip().lstrip("﻿") for c in cluster_df.columns]
    user_cluster = dict(zip(cluster_df["user_id"].astype(str), cluster_df["cluster_id"]))
    print(f"  {len(user_cluster)} designers, "
          f"{cluster_df['cluster_id'].nunique()} clusters")

    print("\n=== Mock extracted records ===")
    records = make_mock_extracted_records(cluster_df)
    print(f"  {len(records)} records (= {len(user_cluster)} × 7 scenarios)")

    print("\n=== Build hypergraph ===")
    g = build_hypergraph(records, user_cluster)
    print(f"  Stats: {g.stats()}")

    print("\n=== Demo retrieval ===")
    sample_user = cluster_df.iloc[0]
    target_uid = str(sample_user["user_id"])
    target_cluster = int(sample_user["cluster_id"])
    print(f"  Target: {target_uid} (cluster {target_cluster})")
    print(f"  Scenario: Q3 (V vs K conflict)")

    result = retrieve_via_og_rag(
        target_user_id=target_uid,
        target_cluster=target_cluster,
        target_scenario_id="Q3",
        target_profile_polarity=None,  # 暂用 placeholder
        hypergraph=g,
        k_max=3,
    )
    print(f"\n  Selected edges ({len(result.selected_edges)}):")
    for e in result.selected_edges:
        print(f"    {e}")
    print(f"  Covered elements: {result.covered_elements}")
    print(f"  Uncovered:        {result.uncovered_elements}")
    print(f"  Candidate pool:   {result.n_candidates}")
    print(f"  Returned nodes:   {len(result.selected_nodes)}")

    print("\n=== Rendered prompt block ===\n")
    print(format_peer_rag_block(result, max_chars=800))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
