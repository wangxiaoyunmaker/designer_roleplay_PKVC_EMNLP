# L8 RAG retrieval sanity — 4 modes vs random (data_v3, 2026-05-24)

测 v3.7 RAG 候选 mode 的检索质量. CPU 跑, 不依赖 GPU/extractor.

- factor scores: K=4, N=163
- k = 3 (peer top-k)
- random baseline trials = 30 (每 (uid,Q) 平均)
- modulation weights: 4 Q 有 sig association (其余 fallback trait_similarity)
- 指标: same-option concordance rate (retrieved peer 跟 target 选同一 option 的比例)

## per-Q concordance (4 modes vs random)

| qid   |   n_users |   random |   trait_sim |   modulation |   trait+mod |   trait_lift% |   mod_lift% |   combined_lift% |
|:------|----------:|---------:|------------:|-------------:|------------:|--------------:|------------:|-----------------:|
| Q1    |       163 |    0.32  |       0.35  |        0.372 |       0.329 |           9.1 |        16.2 |              2.8 |
| Q3    |       163 |    0.398 |       0.389 |        0.389 |       0.389 |          -2.4 |        -2.4 |             -2.4 |
| Q4    |       163 |    0.343 |       0.362 |        0.305 |       0.323 |           5.6 |       -11.1 |             -5.8 |
| Q5    |       163 |    0.329 |       0.358 |        0.303 |       0.348 |           8.9 |        -7.9 |              5.7 |
| Q6    |       163 |    0.383 |       0.403 |        0.403 |       0.403 |           5.2 |         5.2 |              5.2 |
| Q8    |       163 |    0.337 |       0.331 |        0.368 |       0.339 |          -1.8 |         9.1 |              0.6 |

## 综合 (avg across Qs)

| mode | avg concordance | lift vs random |
|---|---|---|
| random (baseline) | 0.352 | — |
| **trait_similarity** | 0.365 | +3.9% |
| **modulation** | 0.357 | +1.4% |
| **trait+mod (RRF)** | 0.355 | +1.0% |

**Best mode**: `trait_similarity` (lift +3.9%)

**Verdict**: ⚠ trait_similarity 微弱优于 random (3.9%) — 边缘信号, 跟其他 ablation 互补

## Caveats

- 这是 retrieval 质量 sanity, **不是** SFT downstream lift (那需 GPU 训完才知道)
- 'same-option concordance' 是 proxy for behavioral similarity, 不是 PKVC state 一致 (后者需 extractor)
- random baseline = within-Q-pool, 不是 cross-Q random (更公平)
- N=163 sample; fold-aware retrieval 没做 (sanity 不严), 真训练时排除 test fold
- modulation 权重来自 L4 quad_s_secondary_associations.jsonl (13 sig), 包含 in-sample 信息 — **小 leakage** (concordance 数字偏乐观), but lift direction 仍 informative