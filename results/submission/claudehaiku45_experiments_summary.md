# Claude Haiku 4.5 阶段实验结果 (paper-ready)

> 整理日期: 2026-05-26 (cross-model validation)
> 数据集: 162 designer × 3 Q7 seg
> 模型: Claude Haiku 4.5 (id: `claude-haiku-4-5-20251001`)
> 输入构造: v2_anchor 系列 prompt (跟 GPT-4o 同 prompt 套件)
> Evaluation: 11-metric set + 1000-bootstrap 95% CI (designer-level resampling)

> PKVC source: v8.7 extracted (real)

---

## 0. 实验总览

### 0.1 总规模

- **10 个 condition × 162 designer × 3 seg = ~4860 inference calls**
- + PKVC extraction calls (v8.7 extractor via GPT-4o)

### 0.2 7 个 condition 分类

| Group | Condition | 用途 |
|---|---|---|
| **Main** | B0 (zero-shot) | floor baseline |
|  | **v2_anchor** ★ | paper main prompt (PKVC def + seg3 rule + β 速览 + 4 类 CAPS) |
| **组件 ablation** | v2_no_caps (删 ⑤ CAPS 整块) | **CAPS module 整体必要性** |
|  | v2_no_seg3_rule (删 SYSTEM seg3 rule) | seg3 决策规则必要性 |
|  | v2_no_beta_summary (删 β 速览 header) | β 速览价值 |
| **架构 ablation** | v2_plus_peer (+ ④ peer) | peer 加上去 net 价值 |
|  | v2_hybrid (seg3 用 v2_clean prompt) | per-seg dispatch 价值 |

---

## 1. Final 11-Metric Set (reference, locked across all models)

3 大维度 × 11 metric:

- **Fidelity (路径准度, seg2)**: macro-TED ↓, macro-LCS-F1 ↑, non-V recall ↑
- **Personalization (个性化)**: NL specificity ratio (3 segs), Persona-NL alignment (seg1), Trait-pred Spearman ρ (seg1), Criterion-persona alignment (seg3)
- **Anti-collapse (差异化)**: Chain diversity (seg2), Criterion uniqueness (seg3)

详见 `reports/datav3/gpt4o_experiments_summary.md §1`.

---

## 2. 主结果表 (1000-bootstrap 95% CI)

### 2.1 seg1 (第一反应 anchor 段)

| Baseline | NL specificity ↑ | Persona-NL ↑ | Trait-pred ρ ↑ |
|---|---|---|---|
| B0_h | 1.001 [0.996, 1.005] | 0.367 [0.324, 0.412] | 0.048 [-0.015, 0.117] |
| B1_h | 1.001 [0.991, 1.010] | 0.343 [0.291, 0.396] | 0.000 [0.000, 0.000] |
| B2_h | 1.001 [0.992, 1.010] | 0.354 [0.308, 0.402] | -0.016 [-0.122, 0.098] |
| B3_h | 0.997 [0.987, 1.007] | 0.342 [0.297, 0.391] | -0.007 [-0.059, 0.068] |
| v2_anchor | 1.003 [0.993, 1.013] | 0.352 [0.310, 0.396] | 0.078 [0.013, 0.177] |
| v2_no_caps | 1.004 [0.996, 1.012] | 0.354 [0.310, 0.401] | 0.027 [-0.016, 0.101] |
| v2_no_seg3_rule | 1.006 [0.998, 1.016] | 0.335 [0.291, 0.376] | 0.037 [-0.058, 0.156] |
| v2_no_beta_summary | 1.001 [0.992, 1.010] | 0.330 [0.285, 0.375] | 0.058 [-0.036, 0.171] |
| v2_plus_peer | 1.009 [0.998, 1.019] | 0.329 [0.287, 0.370] | 0.066 [-0.009, 0.172] |
| v2_hybrid | 1.000 [0.991, 1.009] | 0.348 [0.306, 0.389] | 0.070 [0.008, 0.163] |

### 2.2 seg2 (chain reasoning, multi-state)

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| B0_h | 2.660 [2.449, 2.746] | 0.508 [0.492, 0.560] | 0.086 [0.031, 0.149] | 0.290 [0.222, 0.358] | 1.006 [1.003, 1.010] |
| B1_h | 2.673 [2.457, 2.778] | 0.507 [0.487, 0.568] | 0.117 [0.036, 0.200] | 0.571 [0.479, 0.664] | 1.004 [0.997, 1.011] |
| B2_h | 2.451 [2.286, 2.618] | 0.579 [0.550, 0.609] | 0.212 [0.118, 0.312] | 0.706 [0.625, 0.772] | 0.996 [0.989, 1.004] |
| B3_h | 2.758 [2.578, 2.862] | 0.515 [0.496, 0.574] | 0.375 [0.262, 0.486] | 0.920 [0.861, 0.934] | 1.004 [0.998, 1.012] |
| v2_anchor | 2.535 [2.395, 2.664] | 0.550 [0.523, 0.578] | 0.554 [0.441, 0.667] | 0.944 [0.895, 0.944] | 1.005 [0.998, 1.012] |
| v2_no_caps | 2.706 [2.499, 2.813] | 0.524 [0.504, 0.587] | 0.449 [0.323, 0.561] | 0.900 [0.850, 0.921] | 0.998 [0.992, 1.005] |
| v2_no_seg3_rule | 2.281 [2.186, 2.490] | 0.554 [0.533, 0.593] | 0.342 [0.234, 0.447] | 0.899 [0.845, 0.919] | 1.003 [0.996, 1.011] |
| v2_no_beta_summary | 2.493 [2.348, 2.634] | 0.574 [0.543, 0.602] | 0.493 [0.369, 0.621] | 0.924 [0.875, 0.938] | 1.003 [0.995, 1.011] |
| v2_plus_peer | 2.651 [2.406, 2.765] | 0.533 [0.512, 0.600] | 0.358 [0.246, 0.479] | 0.933 [0.881, 0.940] | 1.002 [0.995, 1.010] |
| v2_hybrid | 2.748 [2.222, 2.833] | 0.433 [0.416, 0.596] | 0.577 [0.464, 0.680] | 0.923 [0.877, 0.942] | 1.004 [0.996, 1.010] |

### 2.3 seg3 (anchor + criterion 段)

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| B0_h | 0.998 [0.993, 1.002] | 0.049 [0.018, 0.082] | 0.220 [0.194, 0.247] |
| B1_h | 1.003 [0.995, 1.012] | 0.415 [0.390, 0.434] | 0.173 [0.147, 0.201] |
| B2_h | 1.005 [0.996, 1.014] | 0.402 [0.373, 0.425] | 0.183 [0.158, 0.208] |
| B3_h | 1.006 [0.997, 1.015] | 0.389 [0.357, 0.415] | 0.188 [0.160, 0.215] |
| v2_anchor | 0.999 [0.991, 1.007] | 0.504 [0.469, 0.530] | 0.237 [0.207, 0.267] |
| v2_no_caps | 1.006 [0.997, 1.017] | 0.463 [0.430, 0.488] | 0.215 [0.186, 0.246] |
| v2_no_seg3_rule | 1.001 [0.992, 1.009] | 0.368 [0.337, 0.396] | 0.187 [0.161, 0.210] |
| v2_no_beta_summary | 1.002 [0.993, 1.012] | 0.530 [0.504, 0.546] | 0.227 [0.194, 0.258] |
| v2_plus_peer | 0.997 [0.987, 1.008] | 0.579 [0.556, 0.592] | 0.227 [0.195, 0.257] |
| v2_hybrid | 1.003 [0.996, 1.012] | 0.352 [0.313, 0.382] | 0.186 [0.163, 0.209] |

## 3. v2_anchor vs B0 (zero-shot) 显著性测试

★ = 95% CI 不重叠 (likely significant).

| Metric | B0 CI | v2_anchor CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.996, 1.005] | [0.993, 1.013] | +0.002 |  |
| seg1_persona_align | [0.324, 0.412] | [0.310, 0.396] | -0.015 |  |
| seg1_trait_pred_rho | [-0.015, 0.117] | [0.013, 0.177] | +0.030 |  |
| seg2_macro_TED | [2.449, 2.746] | [2.395, 2.664] | -0.125 |  |
| seg2_macro_LCS_F1 | [0.492, 0.560] | [0.523, 0.578] | +0.042 |  |
| seg2_non_V_recall | [0.031, 0.149] | [0.441, 0.667] | +0.468 | ★ |
| seg2_chain_diversity | [0.222, 0.358] | [0.895, 0.944] | +0.654 | ★ |
| seg2_nl_specificity | [1.003, 1.010] | [0.998, 1.012] | -0.002 |  |
| seg3_nl_specificity | [0.993, 1.002] | [0.991, 1.007] | +0.001 |  |
| seg3_criterion_uniqueness | [0.018, 0.082] | [0.469, 0.530] | +0.455 | ★ |
| seg3_criterion_persona | [0.194, 0.247] | [0.207, 0.267] | +0.017 |  |

## 4. CAPS module 必要性 (paper main claim 关键 ablation)

| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.993, 1.013] | [0.996, 1.012] | +0.001 (Δ towards no_caps) |  |
| seg1_persona_align | [0.310, 0.396] | [0.310, 0.401] | +0.002 (Δ towards no_caps) |  |
| seg1_trait_pred_rho | [0.013, 0.177] | [-0.016, 0.101] | -0.051 (Δ towards no_caps) |  |
| seg2_macro_TED | [2.395, 2.664] | [2.499, 2.813] | +0.171 (Δ towards no_caps) |  |
| seg2_macro_LCS_F1 | [0.523, 0.578] | [0.504, 0.587] | -0.026 (Δ towards no_caps) |  |
| seg2_non_V_recall | [0.441, 0.667] | [0.323, 0.561] | -0.105 (Δ towards no_caps) |  |
| seg2_chain_diversity | [0.895, 0.944] | [0.850, 0.921] | -0.044 (Δ towards no_caps) |  |
| seg2_nl_specificity | [0.998, 1.012] | [0.992, 1.005] | -0.006 (Δ towards no_caps) |  |
| seg3_nl_specificity | [0.991, 1.007] | [0.997, 1.017] | +0.007 (Δ towards no_caps) |  |
| seg3_criterion_uniqueness | [0.469, 0.530] | [0.430, 0.488] | -0.041 (Δ towards no_caps) |  |
| seg3_criterion_persona | [0.207, 0.267] | [0.186, 0.246] | -0.021 (Δ towards no_caps) |  |

## 5. Winner per metric (within this model's 7 conditions)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_nl_specificity | **v2_plus_peer** | 1.009 | [0.998, 1.019] |
| seg1_persona_align | **B0_h** | 0.367 | [0.324, 0.412] |
| seg1_trait_pred_rho | **v2_anchor** | 0.078 | [0.013, 0.177] |
| seg2_macro_TED | **v2_no_seg3_rule** | 2.281 | [2.186, 2.490] |
| seg2_macro_LCS_F1 | **B2_h** | 0.579 | [0.550, 0.609] |
| seg2_non_V_recall | **v2_hybrid** | 0.577 | [0.464, 0.680] |
| seg2_chain_diversity | **v2_anchor** | 0.944 | [0.895, 0.944] |
| seg2_nl_specificity | **B0_h** | 1.006 | [1.003, 1.010] |
| seg3_nl_specificity | **B3_h** | 1.006 | [0.997, 1.015] |
| seg3_criterion_uniqueness | **v2_plus_peer** | 0.579 | [0.556, 0.592] |
| seg3_criterion_persona | **v2_anchor** | 0.237 | [0.207, 0.267] |

## 6. Paper Main Claims for this model

Place numerical claims here based on §3-§5 above. Template:

```
On Claude Haiku 4.5, our v2_anchor prompt achieves vs zero-shot:
  - seg2 non-V recall: 55.4% [44.1%, 66.7%] vs 8.6% [3.1%, 14.9%]
  - seg2 chain diversity: 0.944 [0.895, 0.944] vs 0.290
  - seg3 criterion uniqueness: 0.504 [0.469, 0.530] vs 0.049
```

## 7. Reproducibility

- Model ID: `claude-haiku-4-5-20251001`
- Run command:
```
# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):
python scripts/run_b6_sample.py --variant <V> --n 0 \
    --model claude-haiku-4-5-20251001 --output reports/datav3/crossmodel/claudehaiku45_<COND>_inference.jsonl
# Then PKVC extract via GPT-4o:
python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
# Then bootstrap eval:
python scripts/generate_model_summary.py --model-slug claudehaiku45 \
    --display-name 'Claude Haiku 4.5' --model-id claude-haiku-4-5-20251001 \
    --output reports/datav3/claudehaiku45_experiments_summary.md
```
