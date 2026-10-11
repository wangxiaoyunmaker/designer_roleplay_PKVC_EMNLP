# SFT Pipeline (v3.5.1)

设计师 PKVC 簇级角色扮演的监督微调 pipeline. 对应 brief
[v3.5.1](../docs/briefs/v3.5.1_20260522.md) §4-§5.

详细结构 → [STRUCTURE.md](STRUCTURE.md).

---

## 双方向

| 方向 | 训练 | 评测 | 任务切分 | 主/对照 |
|---|---|---|---|---|
| 二 | Q1-Q6 选项+解释 NL | Q7 跨场景 | per-question single-step (×6) | **主** (老师 2026-05-20) |
| 一 | Q7 三段思维链 | Q1-Q6 跨场景 | per-segment 3-task (B1/B2/B3) | 对照 |

## 模型

- **Base**: Qwen2.5-7B-Instruct (远程 `/data/dhf/chain_critic/model/Qwen2.5-7B-Instruct/`)
- **LoRA**: r=32, α=64, target=q/k/v/o_proj
- **辅助头**: L_cls (cluster K-way classifier, K 在 train.py:_resolve_n_clusters 自动 resolve) + L_cc (cluster-contrastive)
- **K 当前**: 6 (79 EFA pilot); 200 入袋后 EFA + K sweep 重定

## RAG 4 档 (brief §5.5 RAG-abl)

| Mode | 用什么 | 何时用 |
|---|---|---|
| `cosine` | profile cosine top-3 peer | 默认, no extractor 即可 |
| `og_rag` | ontology hypergraph set cover peer | 需 extracted_records.jsonl |
| `modulation` | trait→PKVC state-transition triple (§4D) | 需 mapping_fold_*.jsonl |
| `og_rag+modulation` | 两者并存, system prompt + peer 都注入 | main contribution candidate |

## 快速 CLI

```bash
# 准备 extractor 输入 jsonl
.venv/bin/python -m sft_pipeline.data.extractor_inputs

# 跑 extractor (需 API key)
.venv/bin/python -m sft_pipeline.extract.batch \
    --input data/processed/q7_segments.jsonl \
    --output data/processed/q7_extracted.jsonl \
    --model gpt-4o --concurrency 8

# 挖 modulation triple (per fold)
.venv/bin/python -m sft_pipeline.rag.mapping_miner \
    --factor-scores data/user_factor_scores.csv \
    --extracted data/processed/extracted_all.jsonl \
    --cluster-csv data/user_cluster_efa_v2_200.csv \
    --derive-folds --fold-id 0 \
    --output data/processed/modulation_fold_0.jsonl

# build SFT JSONL
.venv/bin/python -m sft_pipeline.data.prep \
    --data-dir data --cluster-filename user_cluster_efa_v2_200.csv \
    --rag-mode og_rag+modulation \
    --extracted-records data/processed/extracted_all.jsonl \
    --modulation-dir data/processed \
    --factor-scores data/user_factor_scores.csv

# 训练 (需 GPU)
.venv/bin/python -m sft_pipeline.train --direction q1q6 --fold 0

# 评测
.venv/bin/python -m sft_pipeline.eval.runner \
    --phase0 --validation --baselines B3,B4 \
    --output reports/sanity.md
```

## 数据流约束 (锁不动)

- Q1-Q6 真值任何 token 不进方向一训练
- Q7 真值任何 token 不进方向二训练
- RAG 池 = train fold 同簇 peer 的当前方向训练数据 (per fold 独立 build)
- Mapping triple per fold mining (brief §4D.6, 防 leakage)
- 5-fold stratified CV 按 cluster_id 分层 (cluster 用 **30 维 only** 算, brief §4C lock)

## TODO (按 blocker 优先级)

- [ ] 121 incremental designer 入袋 → EFA + K sweep 重跑 → user_cluster_efa_v2_200.csv
- [ ] API key 解锁后跑 extractor on 200×9 = 1800 段 + κ pilot
- [ ] mapping_miner: 跑 5 fold × 200 designer → modulation_fold_*.jsonl
- [ ] og_rag+modulation 4 档 ablation
- [ ] GPU 解锁后跑 SFT 主实验
- [ ] eval_baselines B0/B0'/B1/B2 placeholder 填充
