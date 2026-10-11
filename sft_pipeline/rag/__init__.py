"""RAG subpackage (v3.5.1).

Three flavors of retrieval, all returning prompt-ready blocks consumed by data.prep:

    cosine     profile-similarity peer top-k (default, no extractor needed)
               Lives in data.prep (cosine_topk_same_cluster_peers), not here —
               doesn't depend on the ontology/hypergraph machinery.
    og_rag     ontology-grounded hypergraph + set cover peer retrieval.
               Inspired by Sharma et al. EMNLP 2025. See peer.retrieve_via_og_rag.
    modulation trait→PKVC state-transition modulation triple injection (Path γ
               system prompt). Brief v3.5.1 §4D. See mapping_miner (CLI miner)
               + mapping_retrieval (load/retrieve/format).

og_rag and modulation can compose (rag_mode="og_rag+modulation"); cosine is exclusive.

mapping_miner.py is a CLI-only script (not re-exported at package level).
Run as: ``python -m sft_pipeline.rag.mapping_miner --help``.
"""
from .ontology import (
    FACTORS, SADR_NODES, SCENARIO_TYPES,
    SADRNode, FactorDef, ScenarioType,
    get_sadr_by_factor, get_scenario_probe_factors,
    parse_factor_path, parse_sadr_id_with_polarity,
)
from .hypergraph import HyperGraph, HyperEdgeKey, SegmentNode, build_hypergraph
from .peer import retrieve_via_og_rag, format_peer_rag_block, RetrievalResult
from .mapping_retrieval import (
    LoadedTriple, ModulationRetrievalResult,
    load_modulation_triples, retrieve_modulations,
    format_modulation_system_block,
    compute_factor_medians, load_factor_scores_dict,
)

__all__ = [
    "FACTORS", "SADR_NODES", "SCENARIO_TYPES",
    "SADRNode", "FactorDef", "ScenarioType",
    "get_sadr_by_factor", "get_scenario_probe_factors",
    "parse_factor_path", "parse_sadr_id_with_polarity",
    "HyperGraph", "HyperEdgeKey", "SegmentNode", "build_hypergraph",
    "retrieve_via_og_rag", "format_peer_rag_block", "RetrievalResult",
    # v3.5.1 modulation
    "LoadedTriple", "ModulationRetrievalResult",
    "load_modulation_triples", "retrieve_modulations",
    "format_modulation_system_block",
    "compute_factor_medians", "load_factor_scores_dict",
]
