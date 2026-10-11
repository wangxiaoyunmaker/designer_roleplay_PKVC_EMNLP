"""Data subpackage: raw CSV → SFT JSONL.

Modules
-------
prep                Raw CSV → fold-aware SFT JSONL (main CLI entry).
dataset             PyTorch SFTDataset + ClusterBalancedSampler + collate_fn.
augment             Profile noise + lexical augmentation (protected-token-aware).
extractor_inputs    CLI: build JSONL inputs for sft_pipeline.extract.batch.
option_inventory    Q1-Q6 option frequency table (for 学姐 PKVC mapping).
"""
from .prep import (
    DesignerProfile, Q1Q6Response, Q7Response, SFTSample,
    LIKERT_TO_POLARITY, POLARITY_ORDER, POLARITY_NL_LABEL,
    MCQ_QUESTIONS, SORT_QUESTION, OPEN_QUESTION,
    load_raw_data, build_profile_matrix,
    numerical_to_polarity, polarity_to_nl,
    assign_folds, cosine_topk_same_cluster_peers, format_cosine_peer_block,
    build_q1q6_sample, build_q7_segment_sample,
    build_all_samples, write_jsonl, split_by_fold,
)
# `dataset` and `augment` pull in torch (heavy). Do NOT re-export here — keeps
# `data.prep` importable on CPU-only laptops without torch installed (needed for
# the local data smoke pipeline). Use:
#   from sft_pipeline.data.dataset import SFTDataset, ClusterBalancedSampler, collate_fn
#   from sft_pipeline.data.augment import perturb_polarity, ...

__all__ = [
    # prep
    "DesignerProfile", "Q1Q6Response", "Q7Response", "SFTSample",
    "LIKERT_TO_POLARITY", "POLARITY_ORDER", "POLARITY_NL_LABEL",
    "MCQ_QUESTIONS", "SORT_QUESTION", "OPEN_QUESTION",
    "load_raw_data", "build_profile_matrix",
    "numerical_to_polarity", "polarity_to_nl",
    "assign_folds", "cosine_topk_same_cluster_peers", "format_cosine_peer_block",
    "build_q1q6_sample", "build_q7_segment_sample",
    "build_all_samples", "write_jsonl", "split_by_fold",
]
