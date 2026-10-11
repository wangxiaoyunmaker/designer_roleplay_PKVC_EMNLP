# sft_pipeline 代码结构 (v3.5.1)

每个文件 1 行职责. 同一职责的代码必须放在同一个子包.

```
sft_pipeline/
├── __init__.py                 包标记 + __version__
├── README.md                   入门 + CLI 示例
├── STRUCTURE.md                本文件
│
├── config/                     配置 + prompt 模板
│   ├── config.yaml             所有超参 + 路径 + Loss weight (train.py 入口读)
│   ├── eval_plan.md            评测计划文档 (Tier 1/2/3 + Phase 0)
│   └── prompts/
│       ├── extractor_v1.md     旧版 PKVC extractor system + user prompt
│       └── extractor_v2.md     v2 改进版 (待学姐填 placeholder B/D/G)
│
├── data/                       Raw CSV → SFT JSONL
│   ├── __init__.py             re-exports prep symbols (不含 dataset 因为 torch 重)
│   ├── prep.py                 主管道: CSV → fold-aware SFT JSONL + 4-档 RAG mode 注入
│   ├── dataset.py              PyTorch SFTDataset + ClusterBalancedSampler + collate
│   ├── augment.py              Profile ±1 Likert 扰动 + lexical aug (P0-4 待决策)
│   ├── extractor_inputs.py     CLI: csv → extract.batch 输入 jsonl + budget 预估
│   └── option_inventory.py     Q1-Q6 选项频率表 (给学姐填 PKVC mapping)
│
├── extract/                    LLM-based PKVC 抽取
│   ├── __init__.py             re-exports llm 主接口
│   ├── llm.py                  单段 extract_pkvc + 多家族 LLM 客户端 + ExtractionResult schema
│   ├── batch.py                异步 batch driver (caching + κ pilot)
│   └── parser.py               解析 SFT-time model 输出回 structured record (推理 + eval 用)
│
├── expand/                     Q1-Q6 短解释 NL 模型辅助扩写 (grounded rewrite)
│   ├── __init__.py             re-exports screen 主接口
│   ├── screen.py               筛选 char<threshold 候选 + 装配 ground 材料 (Q7+量表+cardId+Q8)
│   ├── prompt.py               render system + user prompt (expand_q1q6_v1.md → 填模板)
│   ├── llm.py                  单条 chat completion + cache (gpt-4o 默认, 复用 yunwu.ai 中转)
│   ├── batch.py                异步 batch driver + per-candidate skip-on-failure + mock 模式
│   ├── parser.py               JSON 解析 + 5 个 validator (长度/factor/evidence_quote grounded/...)
│   ├── merge.py                回填 task_responses_expanded.csv + provenance.jsonl
│   └── cli.py                  子命令 screen/run/merge/report/full
│
├── rag/                        Retrieval (3 种 mode)
│   ├── __init__.py             re-exports peer + modulation 接口 (不含 mapping_miner CLI)
│   ├── ontology.py             PKVC 4 因子 + 30 SADR + 题目-因子 metadata
│   ├── hypergraph.py           build_hypergraph: (designer, scenario) 节点 + (cluster, factor, scenario) 超边
│   ├── peer.py                 og_rag set cover retrieval (was retrieval.py)
│   ├── mapping_miner.py        v3.5.1 §4D mining: distribution shift + permutation test
│   ├── mapping_retrieval.py    v3.5.1 §4D injection: load + retrieve + Path γ format
│   ├── demo.py                 CPU demo: og_rag pipeline + 79 designer 数据 (无 GPU/API)
│   └── README.md               og_rag adapt 来源 + 跟原 paper 对照
│
├── eval/                       双向评测协议 (brief v3.5.1 §5)
│   ├── __init__.py             (无 top-level re-exports, 按需显式 import)
│   ├── runner.py               主入口 CLI: fold-aware, leakage-safe, Tier 1/2/3 orchestrator
│   ├── metrics.py              A1 trajectory match + A2 distribution + A3 individual + Q8 subgroup
│   ├── validation.py           B1 EFA + Cronbach α + KMO + Bartlett (PKVC construct)
│   ├── ablation.py             D2/D4 + Phase 0 Test A/B (Cramér's V + permutation)
│   ├── baselines.py            B0/B0'/B1/B2/B3/B4 baseline (B0/B0'/B1/B2 是 placeholder)
│   ├── judge.py                Pairwise LLM Judge: GPT-4o model vs baseline win rate (position-debiased)
│   ├── prototype.py            Cluster prototype builder (fold-aware, was cluster_prototype.py)
│   └── evaluate.py             Legacy A1/A2/A3 CLI (保留 reproducibility, 新工作走 runner.py)
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py           Dataset + sampler + cluster_id 一致性 smoke
│
├── model.py                    QwenSFTModel: Qwen + LoRA + L_cc/L_cls heads
├── train.py                    多任务 loss + curriculum + ClusterBalancedSampler 训练 loop
└── infer.py                    3 步 (q7) / 1 步 (q1q6) 生成 + best-of-N rerank fallback (was inference.py)
```

---

## CLI 入口

```bash
# 准备 extractor 输入
python -m sft_pipeline.data.extractor_inputs

# 跑 extractor (需 API key)
python -m sft_pipeline.extract.batch \
    --input data/processed/q7_segments.jsonl \
    --output data/processed/q7_extracted.jsonl \
    --model gpt-4o --concurrency 8

# Q1-Q6 短解释扩写 pipeline (需 API key; 落 data/task_responses_expanded.csv)
python -m sft_pipeline.expand.cli full --threshold 30 --concurrency 8
# offline smoke (no API):
python -m sft_pipeline.expand.cli full --mock --limit 5

# κ pilot
python -m sft_pipeline.extract.batch \
    --input data/processed/q7_segments.jsonl --kappa-pilot 30 \
    --kappa-model-a gpt-4o --kappa-model-b claude-3-5-sonnet-20241022

# 挖 trait-state modulation (per fold)
python -m sft_pipeline.rag.mapping_miner \
    --factor-scores data/user_factor_scores.csv \
    --extracted data/processed/extracted_all.jsonl \
    --cluster-csv data/user_cluster_efa_v2_200.csv \
    --derive-folds --fold-id 0 \
    --output data/processed/modulation_fold_0.jsonl

# build SFT JSONL (4 种 RAG mode 之一)
python -m sft_pipeline.data.prep \
    --data-dir data --cluster-filename user_cluster_efa_v2_200.csv \
    --rag-mode og_rag+modulation \
    --extracted-records data/processed/extracted_all.jsonl \
    --modulation-dir data/processed \
    --factor-scores data/user_factor_scores.csv \
    --output-dir data/processed/og_rag_modulation

# 训 SFT
python -m sft_pipeline.train --direction q1q6 --fold 0

# 评测
python -m sft_pipeline.eval.runner --phase0 --validation --baselines B3,B4 --output reports/sanity.md

# Pairwise LLM Judge (GPT-4o, position-debiased win rate)
python -m sft_pipeline.eval.judge \
    --model-preds preds/q1q6__og_rag+modulation__fold0.jsonl \
    --baseline-preds preds/q1q6__no_rag__fold0.jsonl \
    --data-dir data --output reports/judge_og_rag_vs_no_rag.json
# dry run (no API):
python -m sft_pipeline.eval.judge \
    --model-preds preds/model.jsonl --baseline-preds preds/baseline.jsonl \
    --data-dir data --dry-run
```

---

## Import 约定

- **包内 import 用绝对路径**: `from sft_pipeline.data.prep import assign_folds`. 不要 `from data.prep import ...`, 因为 `data` 在根目录已经被占了.
- **重模块 lazy**: dataset/augment 用 torch, eval/* 用 sklearn/factor_analyzer 等. data/__init__.py 不 re-export torch 依赖的符号, 保 CPU-only laptop 可 import.
- **mapping_miner 是 CLI**, 不在 `from sft_pipeline.rag import` 列表里. 用 `python -m sft_pipeline.rag.mapping_miner`.

---

## v3.5.1 重组 vs v3.4 对应表

| 旧 (v3.4) | 新 (v3.5.1) |
|---|---|
| sft_pipeline/data_prep.py | sft_pipeline/data/prep.py |
| sft_pipeline/dataset.py | sft_pipeline/data/dataset.py |
| sft_pipeline/augmentation.py | sft_pipeline/data/augment.py |
| sft_pipeline/prepare_extractor_inputs.py | sft_pipeline/data/extractor_inputs.py |
| sft_pipeline/q1q6_option_inventory.py | sft_pipeline/data/option_inventory.py |
| sft_pipeline/eval_extractor.py | sft_pipeline/extract/llm.py |
| sft_pipeline/extractor_batch.py | sft_pipeline/extract/batch.py |
| sft_pipeline/inference_parser.py | sft_pipeline/extract/parser.py |
| sft_pipeline/og_rag/retrieval.py | sft_pipeline/rag/peer.py |
| sft_pipeline/og_rag/{ontology,hypergraph,mapping_retrieval,demo}.py | sft_pipeline/rag/{...同名}.py |
| sft_pipeline/mapping_miner.py | sft_pipeline/rag/mapping_miner.py |
| sft_pipeline/eval_runner.py | sft_pipeline/eval/runner.py |
| sft_pipeline/eval_metrics.py | sft_pipeline/eval/metrics.py |
| sft_pipeline/eval_validation.py | sft_pipeline/eval/validation.py |
| sft_pipeline/eval_ablation.py | sft_pipeline/eval/ablation.py |
| sft_pipeline/eval_baselines.py | sft_pipeline/eval/baselines.py |
| sft_pipeline/cluster_prototype.py | sft_pipeline/eval/prototype.py |
| sft_pipeline/evaluate.py | sft_pipeline/eval/evaluate.py |
| sft_pipeline/inference.py | sft_pipeline/infer.py |
| sft_pipeline/config.yaml | sft_pipeline/config/config.yaml |
| sft_pipeline/eval_plan.md | sft_pipeline/config/eval_plan.md |
| sft_pipeline/extractor_prompt{,_v2}.md | sft_pipeline/config/prompts/extractor_v{1,2}.md |
| sft_pipeline/test_smoke.py | sft_pipeline/tests/test_smoke.py |
