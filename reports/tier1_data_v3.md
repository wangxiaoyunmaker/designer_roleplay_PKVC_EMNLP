# Tier 1 评测报告 [draft] · direction=q1q6 · 2026-05-25T02:08

> 此报告 = Phase 0 sanity + B1 EFA construct + 主指标 (4 档 RAG) + baseline 矩阵 + Q8 subgroup + §5.5 D-14 判读.
> v3.5.1/v3.7 brief 锁定; missing 段表示对应输入未就位 (SFT ckpt / 推理 jsonl / extractor API key).

## §1 Phase 0 sanity (framing viability)

**Test A — 数据侧 cluster 行为异质性**

| metric | value | PASS? |
|---|---|---|
| ratio (between / within) | 0.704 | ✗ ≤ 1 |
| Cramér's V (mean) | 0.139 | ✓ > 0.1 |
| permutation p | 0.3417 | ✗ ≥ 0.05 |
| n_perm | 1000 | — |
| **overall** | — | **✗ FAIL** |

> ratio=0.704, Cramér's V=0.139, permutation p=0.3417 (n=1000); FAIL or borderline, 跨场景 framing 需 reconsider

**Test B — 模型侧 trajectory → cluster invariance**

| metric | value | chance | PASS? |
|---|---|---|---|
| classifier acc | 0.306 | 0.250 | ✗ ≤ 50% |
| n_samples | 163 | — | — |
| n_clusters | 4 | — | — |

> trajectory → cluster classifier acc = 0.306 (chance 0.250); FAIL, 砍 cross-scenario claim

## §2 B1 PKVC construct validation (EFA + Cronbach α + KMO + Bartlett)

## B1 PKVC Validation (EFA + Cronbach's α)

### Methodology

- Factor method: **sklearn-FA (ML, varimax rotation)**
  - 注: sklearn 的 FactorAnalysis 是 ML-based, 非心理统计常用的 PAF/minres. 若审稿质疑可换 factor_analyzer.FactorAnalyzer(method='minres', rotation='varimax').
- Orthogonality residual after varimax: 4.369e-01 (⚠ 弱正交)

### 总览

| 指标 | 值 | 评级 |
|---|---:|:---:|
| Sample size (N) | 163 | 充足 |
| Features | 30 | — |
| **KMO** | 0.603 | **可接受** |
| **Bartlett** χ² | 962.91 | p = 5.89e-42 |
| Kaiser (eigenvalue > 1) | 11 因子 | — |
| 4 因子总共同方差 (varimax) | 23.0% | ⚠ 低 |

### Cronbach's α (内部一致性, 缺失值用列均值填充)

| Factor | α | n_used | 评级 |
|:---:|---:|---:|:---:|
| F1 | 0.652 | 163 | 勉强 |
| F2 | 0.454 | 163 | 差 |
| F3 | 0.495 | 163 | 差 |
| F4 | 0.531 | 163 | 勉强 |

### Top loadings per factor (按 |loading| 排序, 阈值 0.4)

**F1**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 复杂工具信仰 | +0.728 | ★ |
| 资产开源 | +0.524 | ★ |
| 封闭生态竞争 | +0.515 | ★ |
| 风险规避 | +0.502 | ★ |
| 仿生与有机 | +0.383 | · |
| 创作者署名 | +0.313 | · |
| 时代 vs 永恒 | +0.285 | · |
| 秩序与防错 | +0.254 | · |

**F2**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 数据 vs 直觉 | +0.558 | ★ |
| 独裁 vs 委员会 | +0.548 | ★ |
| 团队依恋 | +0.363 | · |
| 系统微模块 | +0.327 | · |
| 装饰合法性 | +0.301 | · |
| 数据隐私交换 | +0.289 | · |
| 情感附加值 | +0.277 | · |
| 创作者署名 | +0.247 | · |

**F3**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 时代 vs 永恒 | +0.492 | ✓ |
| 商业操纵伦理 | +0.490 | ✓ |
| 技术支持 | +0.432 | ✓ |
| AI 替代论 | +0.340 | · |
| 秩序与防错 | +0.315 | · |
| 黑盒透明度 | +0.313 | · |
| 政治中立性 | +0.298 | · |
| 数据隐私交换 | +0.283 | · |

**F4**:

| 题目 | Loading | 强度 |
|:---|---:|:---:|
| 完美主义/MVP | -0.642 | ★ |
| 创新代价 | -0.533 | ★ |
| 本土文化妥协 | -0.415 | ✓ |
| 混沌与灵感 | -0.366 | · |
| 仿生与有机 | -0.365 | · |
| 救世主情结 | -0.333 | · |
| 审美反民粹 | -0.332 | · |
| 政治中立性 | -0.262 | · |

### 前 10 特征值 (Scree)

| Factor # | Eigenvalue | > 1 ? |
|:---:|---:|:---:|
| 1 | 3.917 | ✓ |
| 2 | 2.144 | ✓ |
| 3 | 2.072 | ✓ |
| 4 | 1.686 | ✓ |
| 5 | 1.510 | ✓ |
| 6 | 1.431 | ✓ |
| 7 | 1.363 | ✓ |
| 8 | 1.273 | ✓ |
| 9 | 1.187 | ✓ |
| 10 | 1.114 | ✓ |

### 诊断与建议

- ⚠ Cronbach's α < 0.7 的因子: F1, F2, F3, F4; 内部一致性偏弱, 因子题目可能需要重组.
- ⚠ 4 因子累计方差 23.0% < 50%, 模型可能 under-fit 真实结构 (Kaiser 给 11 因子).

## §3 主指标 × RAG mode (主方向 q1q6 → Q7 跨场景)

**指标解读**: A1↑ 越高越好 (trajectory match) · A2↓ 越低越好 (per-cluster JS divergence) · A3↑ 越高越好 (individual accuracy)

| RAG mode | n | A1_weighted ↑ | A1_factor ↓ | A1_sadr ↑ | A2_js_mean ↓ | A3_model ↑ | A3_floor ↑ |
|---|---|---|---|---|---|---|---|
| `no_rag` | 652 | 0.128 | 0.744 | 0.000 | 0.016 | 0.434 | 0.436 |
| `cosine` | 652 | 0.138 | 0.724 | 0.000 | 0.018 | 0.418 | 0.436 |
| `og_rag` | 652 | 0.119 | 0.762 | 0.000 | 0.020 | 0.389 | 0.436 |
| `og_rag+modulation` | 652 | 0.116 | 0.768 | 0.000 | 0.016 | 0.402 | 0.436 |

## §4 Baseline 矩阵

| baseline | n_predictions | status | method |
|---|---|---|---|
| B0_prime | 815 | fallback=cluster_majority (no API key) | B0_prime_cluster_majority_fallback |
| B0_random | 815 | ok | B0_random_uniform |
| B3 | 815 | ok | B3_knn_majority |
| B4 | 815 | ok | B4_gbm |

> B0/B0' offline 用 random / cluster-majority fallback; paper 用真 LLM API 重跑替换.

## §5 Q8 metacognition subgroup

**指标解读**: ↑ 越高越好 · ↓ 越低越好

| Subgroup | n | A1_weighted | A1_factor_hamming | A1_sadr_jaccard | A2_js_mean | A3_model_acc | A3_floor_acc |
| — | — | ↑ | ↓ | ↑ | ↓ | ↑ | ↑ |
|---|---|---|---|---|---|---|---|
| **Overall** | 815 | 0.113 | 0.774 | 0.000 | 0.018 | 0.418 | 0.436 |
| A — 自报有差异 (跨场景思考变化) | 235 | 0.102 | — | — | 0.025 | 0.447 | 0.319 |
| B — 自报无差异 (跨场景思考稳定) | 305 | 0.115 | — | — | 0.026 | 0.377 | 0.426 |
| C — 自报中立 / 无明显感受 | 275 | 0.120 | — | — | 0.020 | 0.440 | 0.425 |

> **核心叙事**: Q8=B (自报稳定, n=305) 的 A1 trajectory match 高于 Q8=A (自报变化, n=235) 0.013, 跟设计师 metacognition 一致.

## §6 §5.5 mapping main-claim 升级判读 (D-14)

- metric: `A1_weighted`
- reference mode: `no_rag`
- ranking (best → worst):
  - `cosine`: 0.138
  - `no_rag`: 0.128
  - `og_rag`: 0.119
  - `og_rag+modulation`: 0.116
- og_rag+modulation − og_rag = -0.003
- **rag_helps_at_all** (max(rag) − no_rag > 1%): ✗
- **mapping_promotes_main_claim** (Δ > 1%): ✗

> ✗ RAG 整体没明显增益, 重新审视 framing (peer NL 质量 / similarity metric / k=3 是否合理).

## §7 Caveats

- draft 状态. 主要 Tier 1; Tier 2/3 (SM-abl / B2 CFA / D3 反向 / D4 memorization) 另外跑.
- 200 入袋前所有数字 underpowered, 投稿前 200 重跑.
- B0/B0' 在 fallback 模式下不代表真 LLM 性能; 投稿前 API 重跑.
- Phase 0 Test B 需要 SFT ckpt + 推理输出 jsonl 才会生效, 否则 skip.
- §5.5 D-14 判读默认 1% threshold, 跟 GPT review 商榷可改.
- ⚠ 主指标走 mock predictions (random P/K/V/C + flip noise), 数字不代表真模型性能 — 跑通 pipeline 用.
- ⚠ PKVC 反推走 regex fallback, 没调 extractor; A1 上限被截.
