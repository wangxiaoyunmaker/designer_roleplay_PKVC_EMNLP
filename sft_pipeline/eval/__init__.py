"""Eval subpackage (v3.5.1 §5 双向评测协议).

Modules
-------
runner       Top-level CLI (fold-aware, leakage-safe). Orchestrates Tier 1/2/3.
metrics      A1 trajectory match / A2 distribution / A3 individual + within-cluster floor +
             Q8 subgroup analysis.
validation   B1 EFA + Cronbach α + KMO + Bartlett (PKVC construct validation).
ablation     D2 factor-injection / D4 memorization-probe / Phase 0 Test A/B
             (Cramér's V + permutation tests).
baselines    B0/B0'/B1/B2/B3/B4 baselines (some placeholder, see TODOs).
prototype    Cluster prototype builder — fold-aware aggregation of PKVC factor /
             SADR distribution / criterion keywords (was cluster_prototype.py).
evaluate     Legacy evaluation CLI (older A1/A2/A3 implementation, kept for
             reproducibility). New work should use ``runner``.
"""
# No top-level re-exports — eval modules pull in heavy deps (transformers, sklearn,
# factor_analyzer etc) lazily. Import explicit submodules:
#   from sft_pipeline.eval.metrics import compute_a1
#   from sft_pipeline.eval.runner import main
