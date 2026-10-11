# 评测方案规划 (基于 v3.4 双方向并行)

## 整体目标

1. **主诉求**: 比 baseline 强 (老师锁定的最终标准)
2. **次诉求**: PKVC framework + cluster role-play 在认知科学维度上有可解释贡献 (EMNLP 主会加分)
3. **双方向 fair compare**: 方向一 (Q7 训) vs 方向二 (Q1-Q6 训) 用同 backbone / batch / loss 接口跑出来, 数字直接 head-to-head

## 评测 6 层架构

```
┌─────────────────────────────────────────────────────────┐
│ Layer 6: 报告 (markdown + tables, 投 paper 用)            │
├─────────────────────────────────────────────────────────┤
│ Layer 5: 指标 (A1/A2/A3 主辅 + Tier2 ablation + Phase 0) │
├─────────────────────────────────────────────────────────┤
│ Layer 4: 真值对齐 (ground truth + cluster prototype)      │
├─────────────────────────────────────────────────────────┤
│ Layer 3: PKVC 反推 (LLM-as-extractor, 共用一个 prompt)    │
├─────────────────────────────────────────────────────────┤
│ Layer 2: 预测产出 (SFT model / baseline 各跑双向)         │
├─────────────────────────────────────────────────────────┤
│ Layer 1: 数据 (79 designer × 5-fold × hold-out)          │
└─────────────────────────────────────────────────────────┘
```

## Layer-by-layer 细则

### L1 数据
- 79 designer 完整 (有 scale + task)
- 5-fold stratified by cluster_id (per fold ~16 designer)
- Contamination L1-L4 都 hold-out (见 brief §2)

### L2 预测产出 (共 7 个 model × 2 个方向 = 14 套结果)
- **Ours v3.4**: 方向一 SFT + 方向二 SFT (主战)
- **B0**: 基础大模型 prompt-only, 无 RAG 无 PKVC schema
- **B0'**: 强模型 (GPT-4o / Claude) prompt + PKVC schema + peer RAG (Self-Discover 反证)
- **B1**: SFT 但不输出 PKVC 段 (验证 PKVC 显化的贡献)
- **B2**: SFT 但 PKVC 标签随机洗牌 (验证标注信号真实性)
- **B3**: K近邻同簇投票, 无 SFT 无大模型 (trivial baseline, 反咬最危险)
- **B4**: GBM on profile_30d, 无大模型 (经典监督学习上限)

每个 model 跑两套测试:
- 方向一: Q1-Q6 跨场景 (主指标) + Q7 同分布 (诊断)
- 方向二: Q7 跨场景 (主指标) + Q1-Q6 同分布 (诊断)

### L3 PKVC 反推 (统一用 extractor_prompt.md)
- 输入: NL 段落
- 输出: JSON (dominant_factor + sadr_activations + criterion + confidence)
- 调用: GPT-4o 默认, Claude 备选 (跨家族避 self-judge bias)
- 缓存: 每段 NL 算一次 hash, 反推结果存盘, 重复评测复用

### L4 真值对齐
- Designer 真值:
  - Q1-Q6 选项 + 排序 (直接真值)
  - Q1-Q6 解释 NL → extractor 反推 (待 NL 补全)
  - Q7 三段 NL → extractor 反推
  - 30 量表 → 4 因子 score
  - cluster_id (聚类)
- Cluster prototype:
  - factor_centroid = 同簇 train designer 的 dominant_factor 众数 / 路径 mean
  - sadr_centroid = 同簇 train designer 的 sadr 节点频次向量 (TF-IDF)
  - answer_centroid = 同簇 train designer 的 Q1-Q6 答案分布 (softmax 频率)

### L5 指标

#### Tier 1 必跑

| 指标 | 含义 | 公式 |
|---|---|---|
| **A1 trajectory match** (主) | 模型预测 trajectory vs cluster prototype | factor Hamming + sadr Jaccard (加权) |
| **A2 distribution match** (辅) | per-cluster 答案分布跟真值分布 | JS-divergence (per cluster, mean) |
| **A3 individual L1 + floor** (sanity) | 个体精确匹配 + within-cluster floor baseline | accuracy + same-cluster majority baseline |
| **B1 PKVC validation** | EFA on 30 量表 → 4 因子 | factor loading + α + KMO |
| **D2 factor injection** | 因子置空看输出坍塌 | 反转率 ≥ 阈值 |
| **E1 baseline 矩阵** | 6 baseline + Ours 横扫 | 跑同样 A1/A2/A3 |
| **E2 human eval** | 设计师 N=20 盲选 | majority vote vs ours / vs baseline |

#### Tier 2 撑硬 claim

| 指标 | 含义 |
|---|---|
| SM-abl | schema mediation ablation (factor token 是否真贡献) |
| **B2 multi-group CFA** | 跨场景 measurement invariance (直接关联 R2) |
| C2 cluster discriminability | trajectory 反推 cluster 准确率 |
| C3 self-consistency | 模型自一致 (重复推理一致性) |
| D3 反向注入 | 注入相反 cluster prototype 看坍塌 |
| D4 memorization probe | 输入扰动 → 输出变化 |
| **RAG-abl** | 簇内 peer RAG vs no-RAG |

#### Tier 3 弹药 (rebuttal 用)

| 指标 | 含义 |
|---|---|
| C4 counterfactual | 改变某个 SADR 看输出 |
| D5 adversarial | 对抗输入扰动 |
| E3 per-cluster SFT | 每 cluster 单独训对照 |

### L6 Q8 subgroup 维度 (独立分析层)

Q8 答案 = 跨场景认知一致性 self-report:
- A (有差异, 27%): 自己承认变了
- B (没差异, 33%): 自己说不变
- C (没什么感受, 40%): 中立

**每个主指标都分 Q8 subgroup 报告** — 论文核心叙事之一:
> 我们的方法**不是 universally apply**, 而是在自报稳定子群 (Q8=B) 上效果显著优于 baseline, 跟设计师 metacognition 一致。

### Phase 0 sanity (训前必跑)

| Test | 触发时机 | 期望 | 失败 fallback |
|---|---|---|---|
| **Test A** 数据侧 | 数据回收后, 不训 | within-cluster < between-cluster | framing 重写 |
| **Test B** 模型侧 | 单 fold SFT 完 | trajectory → cluster 分类 acc > 50% | 砍 cross-scenario claim |

## 代码文件分工 (`sft_pipeline/eval_*`)

```
eval_plan.md              本文件
eval_extractor.py         LLM-as-extractor (调 GPT-4o, parse JSON, 缓存)
eval_metrics.py           Tier 1 指标 (A1/A2/A3 + Q8 subgroup)
eval_validation.py        B1 EFA + B2 CFA (量表统计学)
eval_ablation.py          D2/D4/RAG-abl + Phase 0 Test A/B
eval_baselines.py         B0/B0'/B1/B2/B3/B4 实现
eval_runner.py            主入口: load → predict → extract → metric → report
```

## 调用流程

```
# 完整跑一遍 (etc Tier 1):
python eval_runner.py --tier 1 --direction both --output report.md

# 只跑 Phase 0 sanity:
python eval_runner.py --phase0 --output sanity.md

# 单跑某个 baseline vs Ours:
python eval_runner.py --baselines B3,B4 --ours --output baselines.md

# Tier 2 ablation:
python eval_runner.py --tier 2 --ablate schema_mediation,rag --output ablation.md
```

## 缓存策略

- LLM extractor 结果: SHA-256(input_text) → JSON, 存 `cache/extractor/`
- 评测中间产物: cluster prototype / fold split, 存 `cache/intermediate/`
- 报告产物: 每次跑保存 timestamp 版本, 存 `reports/`

## 阻塞项 (Phase 0 跑之前必须完成)

1. ✗ Q7 yaml 标注 (GPT-4o extractor 跑 600 段) — 真值
2. ✗ Q1-Q6 解释 NL 补全 (学姐让设计师补长) — 方向二训练数据
3. ✗ Q1-Q6 选项-PKVC factor metadata (学姐 confirm 导出) — 评测对齐
4. ⚠ SADR 30 节点完整定义 (学姐补) — extractor 校准
5. ⚠ Cluster_id 最终锁定 (K=5 vs K=6 决议)

## 时间预算 (Phase 3 评测阶段, 估算)

| 任务 | 耗时 |
|---|---|
| Q7 真值 extractor 跑 (600 段 × GPT-4o) | 2-4 小时 |
| 模型推理 (7 model × 2 方向 × 79 designer × 题数) | 6-12 小时 |
| 反推 extractor (model 输出 × 7 model × 2 方向) | 4-8 小时 |
| 指标计算 + 报告 | 1-2 小时 |
| Human eval | 1 天 (人工)|
| **总计** | 2-3 天 (含人工) |
