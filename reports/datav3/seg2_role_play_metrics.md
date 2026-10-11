# seg2 chain segment: 重新设计 path fidelity metrics

4 个新 metric (替代 V-majority + mode collapse biased 旧 metric):

1. **Macro-averaged path metrics**: 按 GT first letter 分 P/K/V/C 4 组, 各组均值, 再 macro 平均
2. **Chain diversity**: 1 - modal_freq + entropy. 量化是否真在 differentiate.
3. **Per-designer specificity**: sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)). > 1 = role-play.
4. **Trait-chain Spearman ρ**: 相似 trait → 相似 chain pred. 直接 role-play 实证.

## 1. Macro-averaged path metrics (按 GT first letter 分 P/K/V/C)

| Baseline | macro-TED ↓ | macro-STA ↑ | macro-PathJacc ↑ | macro-LCS-F1 ↑ |
|---|---|---|---|---|
| B0_h | 2.489 | 0.259 | 0.579 | 0.562 |
| B1_h | 2.434 | 0.210 | 0.581 | 0.559 |
| B2_h | 2.458 | 0.246 | 0.558 | 0.554 |
| B3_h | 2.740 | 0.225 | 0.486 | 0.454 |
| v2_hybrid | 2.121 | 0.206 | 0.564 | 0.582 |
| v2_anchor | 2.121 | 0.206 | 0.564 | 0.582 |
| B6_h | 2.812 | 0.189 | 0.546 | 0.513 |
| B5 | 2.850 | 0.194 | 0.473 | 0.438 |

### Per-class breakdown (TED ↓)

| Baseline | P (n) | K (n) | V (n) | C (n) |
|---|---|---|---|---|
| B0_h | 2.71 (38) | 2.40 (42) | 1.84 (81) | 3.00 (1) |
| B1_h | 2.58 (38) | 2.14 (42) | 2.01 (81) | 3.00 (1) |
| B2_h | 2.50 (38) | 2.33 (42) | 2.00 (81) | 3.00 (1) |
| B3_h | 2.71 (38) | 2.21 (42) | 2.04 (81) | 4.00 (1) |
| v2_hybrid | 2.63 (38) | 1.98 (42) | 1.88 (81) | 2.00 (1) |
| v2_anchor | 2.63 (38) | 1.98 (42) | 1.88 (81) | 2.00 (1) |
| B6_h | 2.76 (38) | 2.24 (42) | 2.25 (81) | 4.00 (1) |
| B5 | 2.87 (38) | 2.60 (42) | 1.94 (81) | 4.00 (1) |

### Per-class breakdown (STA ↑)

| Baseline | P | K | V | C |
|---|---|---|---|---|
| B0_h | 0.294 | 0.345 | 0.397 | 0.000 |
| B1_h | 0.236 | 0.292 | 0.311 | 0.000 |
| B2_h | 0.283 | 0.357 | 0.343 | 0.000 |
| B3_h | 0.227 | 0.325 | 0.349 | 0.000 |
| v2_hybrid | 0.239 | 0.298 | 0.286 | 0.000 |
| v2_anchor | 0.239 | 0.298 | 0.286 | 0.000 |
| B6_h | 0.212 | 0.280 | 0.262 | 0.000 |
| B5 | 0.154 | 0.282 | 0.339 | 0.000 |

## 2. Chain diversity (mode collapse 指标)

| Baseline | Modal freq ↓ | Diversity (1-modal) ↑ | Entropy bits ↑ | #unique chains |
|---|---|---|---|---|
| B0_h | 33.95% | 0.660 | 3.81 | 34 / 162 |
| B1_h | 33.95% | 0.660 | 3.78 | 30 / 162 |
| B2_h | 22.22% | 0.778 | 4.38 | 38 / 162 |
| B3_h | 20.37% | 0.796 | 4.63 | 47 / 162 |
| v2_hybrid | 17.90% | 0.821 | 4.69 | 47 / 162 |
| v2_anchor | 17.90% | 0.821 | 4.69 | 47 / 162 |
| B6_h | 9.26% | 0.907 | 5.28 | 64 / 162 |
| B5 | 20.99% | 0.790 | 4.59 | 51 / 162 |

## 3. Per-designer specificity (sim(pred_i, GT_i) / mean_j(sim(pred_i, GT_j)))

Mean / Median > 1 表示 model 输出 specifically 给 designer i (role-play). ≈ 1 表示 generic.

| Baseline | Mean spec | Median spec | #(spec > 1) / total |
|---|---|---|---|
| B0_h | 1.009 | 0.982 | 70 / 162 (43%) |
| B1_h | 1.002 | 0.982 | 66 / 162 (41%) |
| B2_h | 1.021 | 0.982 | 68 / 162 (42%) |
| B3_h | 0.998 | 0.982 | 65 / 162 (40%) |
| v2_hybrid | 1.013 | 0.982 | 72 / 162 (44%) |
| v2_anchor | 1.013 | 0.982 | 72 / 162 (44%) |
| B6_h | 0.998 | 0.982 | 75 / 162 (46%) |
| B5 | 1.021 | 0.982 | 75 / 162 (46%) |

## 4. Trait-chain Spearman correlation

相似 30维 trait → 相似 chain pred. 直接 role-play 实证.

| Baseline | Spearman ρ | n_pairs |
|---|---|---|
| B0_h | +0.0362 | 13041 |
| B1_h | -0.0389 | 13041 |
| B2_h | +0.0379 | 13041 |
| B3_h | -0.0730 | 13041 |
| v2_hybrid | +0.0076 | 13041 |
| v2_anchor | +0.0076 | 13041 |
| B6_h | +0.0494 | 13041 |
| B5 | +0.0240 | 13041 |
