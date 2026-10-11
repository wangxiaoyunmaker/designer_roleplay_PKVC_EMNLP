"""sft_pipeline: Designer PKVC role-play SFT pipeline (v3.5.2).

v3.5.2 (2026-05-23): T_CAPS quad schema v1.1 落地 (schema infra only, 算法等学姐 metadata).
   - 新增 rag/quad_schema.py     T_CAPS = (C, s, p, o) dataclass + validator
   - 新增 rag/codebook.py        situation_codebook + predicate_codebook loader
   - extractor schema 待升级到 v3 (含 p_surface + o_subtype 输出, 等学姐 cardId metadata)
   - mapping_miner / mapping_retrieval 算法暂留三元组 (v3.5.1 §4D), 等 quad DB 数据齐再升

Subpackages
-----------
data/      Raw CSV → SFT JSONL (prep / dataset / augment / extractor_inputs / option_inventory).
extract/   LLM-based PKVC extraction (llm / batch / parser).
rag/       Retrieval (ontology + hypergraph + peer + mapping_miner + mapping_retrieval
           + quad_schema [v1.1] + codebook [v1.1]).
eval/      Evaluation pipeline (runner / metrics / validation / ablation / baselines / prototype).
tests/     Smoke + integration tests (含 test_quad_schema.py [v1.1]).
config/    YAML + extractor prompts.

Top-level modules
-----------------
model.py   Qwen + LoRA + L_cc/L_cls heads (QwenSFTModel).
train.py   Training loop (multi-task loss + curriculum + cluster-balanced sampler).
infer.py   Inference + best-of-N (was inference.py pre-v3.5.1).

See sft_pipeline/STRUCTURE.md for per-file 1-line descriptions.
See docs/briefs/T_CAPS_quad_schema_v1.md for the locked v1.1 quad schema.
"""
__version__ = "3.5.2"
