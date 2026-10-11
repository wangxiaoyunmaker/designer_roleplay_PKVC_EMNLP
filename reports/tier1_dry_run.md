# Tier 1 评测报告 [draft] · direction=q1q6 · 2026-05-24T21:16

> 此报告 = Phase 0 sanity + B1 EFA construct + 主指标 (4 档 RAG) + baseline 矩阵 + Q8 subgroup + §5.5 D-14 判读.
> v3.5.1/v3.7 brief 锁定; missing 段表示对应输入未就位 (SFT ckpt / 推理 jsonl / extractor API key).

## §1 Phase 0 sanity (framing viability)

**Test A — 数据侧 cluster 行为异质性**

| metric | value | PASS? |
|---|---|---|
| ratio (between / within) | 1.529 | ✓ > 1 |
| Cramér's V (mean) | 0.224 | ✓ > 0.1 |
| permutation p | 0.3576 | ✗ ≥ 0.05 |
| n_perm | 1000 | — |
| **overall** | — | **✗ FAIL** |

> ratio=1.529, Cramér's V=0.224, permutation p=0.3576 (n=1000); FAIL or borderline, 跨场景 framing 需 reconsider

**Test B — 模型侧 trajectory → cluster invariance**

| metric | value | chance | PASS? |
|---|---|---|---|
| classifier acc | 0.253 | 0.167 | ✗ ≤ 50% |
| n_samples | 79 | — | — |
| n_clusters | 6 | — | — |

> trajectory → cluster classifier acc = 0.253 (chance 0.167); FAIL, 砍 cross-scenario claim

## §2 B1 PKVC construct validation (EFA + Cronbach α + KMO + Bartlett)

## B1 PKVC Validation (EFA + Cronbach's α)

### Methodology

- Factor method: **sklearn-FA (ML, varimax rotation)**
  - 注: sklearn 的 FactorAnalysis 是 ML-based, 非心理统计常用的 PAF/minres. 若审稿质疑可换 factor_analyzer.FactorAnalyzer(method='minres', rotation='varimax').
- Orthogonality residual after varimax: 2.972e-01 (⚠ 弱正交)

### 总览

| 指标 | 值 | 评级 |
|---|---:|:---:|
| Sample size (N) | 79 | ⚠ 偏小 (建议 ≥ 150) |
| Features | 30 | — |
| **KMO** | 0.478 | **不适合** |
| **Bartlett** χ² | 664.71 | p = 7.4e-12 |
| Kaiser (eigenvalue > 1) | 12 因子 | — |
| 4 因子总共同方差 (varimax) | 25.4% | ⚠ 低 |

### Cronbach's α (内部一致性, 缺失值用列均值填充)

| Factor | α | n_used | 评级 |
|:---:|---:|---:|:---:|
| F1 | 0.658 | 79 | 勉强 |
| F2 | 0.546 | 79 | 勉强 |
| F3 | 0.489 | 79 | 差 |
| F4 | — | 0 | — |

### Top loadings per factor (按 |loading| 排序, 阈值 0.4)

**F1**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 资产开源 | +0.728 | ★ |
| 复杂工具信仰 | +0.660 | ★ |
| 风险规避 | +0.412 | ✓ |
| 创作者署名 | +0.410 | ✓ |
| 仿生与有机 | +0.403 | ✓ |
| 完美主义/MVP | +0.402 | ✓ |
| 封闭生态竞争 | +0.391 | · |
| 时代 vs 永恒 | +0.384 | · |

**F2**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 审美反民粹 | +0.786 | ★ |
| 创新代价 | +0.495 | ✓ |
| 混沌与灵感 | +0.405 | ✓ |
| 系统微模块 | +0.353 | · |
| 装饰合法性 | +0.314 | · |
| 政治中立性 | +0.313 | · |
| 数据隐私交换 | +0.277 | · |
| 黑盒透明度 | -0.260 | · |

**F3**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| AI 替代论 | -0.612 | ★ |
| 商业操纵伦理 | -0.600 | ★ |
| 技术支持 | -0.393 | · |
| 黑盒透明度 | -0.364 | · |
| 政治中立性 | -0.354 | · |
| 股东服务优先 | -0.347 | · |
| 秩序与防错 | -0.293 | · |
| 数据隐私交换 | -0.283 | · |

**F4**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 独裁 vs 委员会 | -0.680 | ★ |
| 数据 vs 直觉 | -0.384 | · |
| 风险规避 | +0.352 | · |
| 装饰合法性 | -0.313 | · |
| 秩序与防错 | +0.310 | · |
| 情感附加值 | -0.277 | · |
| 创作者署名 | -0.259 | · |
| 政治中立性 | +0.235 | · |

### 前 10 特征值 (Scree)

| Factor # | Eigenvalue | > 1 ? |
|:---:|---:|:---:|
| 1 | 3.838 | ✓ |
| 2 | 2.375 | ✓ |
| 3 | 2.287 | ✓ |
| 4 | 1.887 | ✓ |
| 5 | 1.715 | ✓ |
| 6 | 1.650 | ✓ |
| 7 | 1.436 | ✓ |
| 8 | 1.350 | ✓ |
| 9 | 1.327 | ✓ |
| 10 | 1.225 | ✓ |

### 诊断与建议

- ⚠ **KMO 0.478 < 0.6**, 数据规模或量表相关性偏弱; 建议数据加倍后重跑.
- ⚠ Cronbach's α < 0.7 的因子: F1, F2, F3; 内部一致性偏弱, 因子题目可能需要重组.
- ⚠ 4 因子累计方差 25.4% < 50%, 模型可能 under-fit 真实结构 (Kaiser 给 12 因子).

## §3 主指标 × RAG mode (主方向 q1q6 → Q7 跨场景)

**指标解读**: A1↑ 越高越好 (trajectory match) · A2↓ 越低越好 (per-cluster JS divergence) · A3↑ 越高越好 (individual accuracy)

| RAG mode | n | A1_weighted ↑ | A1_factor ↓ | A1_sadr ↑ | A2_js_mean ↓ | A3_model ↑ | A3_floor ↑ |
|---|---|---|---|---|---|---|---|
| `no_rag` | 395 | 0.119 | 0.762 | 0.000 | 0.017 | 0.430 | 0.425 |
| `cosine` | 395 | 0.129 | 0.742 | 0.000 | 0.018 | 0.435 | 0.425 |
| `og_rag` | 395 | 0.128 | 0.744 | 0.000 | 0.014 | 0.461 | 0.425 |
| `og_rag+modulation` | 395 | 0.127 | 0.747 | 0.000 | 0.022 | 0.392 | 0.425 |

## §4 Baseline 矩阵

| baseline | n_predictions | status | method |
|---|---|---|---|
| B0_prime | 395 | fallback=cluster_majority (no API key) | B0_prime_cluster_majority_fallback |
| B0_random | 395 | ok | B0_random_uniform |
| B3 | 395 | ok | B3_knn_majority |
| B4 | 395 | ok | B4_gbm |

> B0/B0' offline 用 random / cluster-majority fallback; paper 用真 LLM API 重跑替换.

## §5 Q8 metacognition subgroup

**指标解读**: ↑ 越高越好 · ↓ 越低越好

| Subgroup | n | A1_weighted | A1_factor_hamming | A1_sadr_jaccard | A2_js_mean | A3_model_acc | A3_floor_acc |
| — | — | ↑ | ↓ | ↑ | ↓ | ↑ | ↑ |
|---|---|---|---|---|---|---|---|
| **Overall** | 395 | 0.129 | 0.742 | 0.000 | 0.018 | 0.435 | 0.425 |
| A — 自报有差异 (跨场景思考变化) | 105 | 0.133 | — | — | 0.046 | 0.390 | 0.438 |
| B — 自报无差异 (跨场景思考稳定) | 130 | 0.131 | — | — | 0.047 | 0.492 | 0.432 |
| C — 自报中立 / 无明显感受 | 160 | 0.125 | — | — | 0.041 | 0.419 | 0.381 |

> **核心叙事**: Q8=B (自报稳定, n=130) 的 A1 trajectory match 低于 Q8=A (自报变化, n=105) 0.003, 跟设计师 metacognition 一致.

## §6 §5.5 mapping main-claim 升级判读 (D-14)

- metric: `A1_weighted`
- reference mode: `no_rag`
- ranking (best → worst):
  - `cosine`: 0.129
  - `og_rag`: 0.128
  - `og_rag+modulation`: 0.127
  - `no_rag`: 0.119
- og_rag+modulation − og_rag = -0.001
- **rag_helps_at_all** (max(rag) − no_rag > 1%): ✓
- **mapping_promotes_main_claim** (Δ > 1%): ✗

> ↻ peer RAG 有效但 mapping 增益 < 1%, 保持 og_rag 为 main; mapping 留 ablation 表.

## §7 Caveats

- draft 状态. 主要 Tier 1; Tier 2/3 (SM-abl / B2 CFA / D3 反向 / D4 memorization) 另外跑.
- 200 入袋前所有数字 underpowered, 投稿前 200 重跑.
- B0/B0' 在 fallback 模式下不代表真 LLM 性能; 投稿前 API 重跑.
- Phase 0 Test B 需要 SFT ckpt + 推理输出 jsonl 才会生效, 否则 skip.
- §5.5 D-14 判读默认 1% threshold, 跟 GPT review 商榷可改.
- ⚠ 主指标走 mock predictions (random P/K/V/C + flip noise), 数字不代表真模型性能 — 跑通 pipeline 用.
- ⚠ cluster prototypes 用 deterministic cluster→PKVC 映射 mock 构造, A1 数字仅验工程, 投稿前必须用 extract.batch 输出替换 (--extracted-records data/processed/extracted_all.jsonl).
- ⚠ PKVC 反推走 regex fallback, 没调 extractor; A1 上限被截.
