# Gemini 3.1 Flash Lite 阶段实验结果 (paper-ready)

> 整理日期: 2026-05-26 (cross-model validation)
> 数据集: 162 designer × 3 Q7 seg
> 模型: Gemini 3.1 Flash Lite (id: `gemini-3.1-flash-lite`)
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
| B0_h | 0.995 [0.988, 1.003] | 0.309 [0.273, 0.347] | 0.000 [0.000, 0.000] |
| B1_h | 0.994 [0.987, 1.002] | 0.323 [0.286, 0.363] | -0.094 [-0.154, 0.000] |
| B2_h | 0.998 [0.991, 1.006] | 0.335 [0.296, 0.377] | -0.002 [-0.071, 0.088] |
| B3_h | 1.006 [0.998, 1.014] | 0.333 [0.294, 0.376] | 0.071 [-0.056, 0.216] |
| v2_anchor | 0.997 [0.987, 1.007] | 0.346 [0.305, 0.389] | 0.088 [0.052, 0.157] |
| v2_no_caps | 1.010 [1.002, 1.018] | 0.334 [0.296, 0.374] | -0.015 [-0.129, 0.125] |
| v2_no_seg3_rule | 1.007 [0.997, 1.019] | 0.350 [0.309, 0.395] | 0.102 [0.035, 0.193] |
| v2_no_beta_summary | 1.007 [0.997, 1.018] | 0.336 [0.297, 0.378] | 0.037 [0.006, 0.104] |
| v2_plus_peer | 1.004 [0.992, 1.015] | 0.346 [0.306, 0.389] | 0.100 [0.058, 0.176] |
| v2_hybrid | 1.012 [1.001, 1.022] | 0.345 [0.305, 0.388] | 0.090 [0.050, 0.165] |

### 2.2 seg2 (chain reasoning, multi-state)

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| B0_h | 2.380 [2.082, 2.472] | 0.459 [0.444, 0.627] | 0.481 [0.368, 0.588] | 0.599 [0.525, 0.667] | 0.999 [0.992, 1.006] |
| B1_h | 2.396 [2.089, 2.492] | 0.457 [0.442, 0.625] | 0.407 [0.300, 0.513] | 0.593 [0.525, 0.667] | 0.995 [0.989, 1.002] |
| B2_h | 2.374 [2.076, 2.470] | 0.469 [0.451, 0.642] | 0.333 [0.234, 0.439] | 0.660 [0.586, 0.735] | 1.001 [0.994, 1.009] |
| B3_h | 2.324 [1.984, 2.420] | 0.573 [0.554, 0.650] | 0.395 [0.291, 0.500] | 0.784 [0.722, 0.840] | 1.005 [0.997, 1.013] |
| v2_anchor | 2.227 [2.128, 2.406] | 0.550 [0.530, 0.593] | 0.272 [0.175, 0.371] | 0.883 [0.827, 0.920] | 1.006 [0.996, 1.015] |
| v2_no_caps | 2.571 [2.001, 2.662] | 0.551 [0.535, 0.640] | 0.432 [0.333, 0.538] | 0.747 [0.679, 0.809] | 1.006 [0.998, 1.013] |
| v2_no_seg3_rule | 2.148 [2.059, 2.279] | 0.443 [0.424, 0.611] | 0.222 [0.134, 0.317] | 0.833 [0.778, 0.883] | 1.005 [0.997, 1.013] |
| v2_no_beta_summary | 2.403 [2.107, 2.487] | 0.545 [0.525, 0.616] | 0.309 [0.214, 0.418] | 0.858 [0.802, 0.889] | 1.011 [1.003, 1.020] |
| v2_plus_peer | 2.042 [1.946, 2.481] | 0.409 [0.388, 0.565] | 0.259 [0.164, 0.352] | 0.877 [0.821, 0.914] | 1.004 [0.996, 1.011] |
| v2_hybrid | 2.487 [2.191, 2.593] | 0.410 [0.387, 0.574] | 0.210 [0.128, 0.301] | 0.901 [0.858, 0.926] | 1.005 [0.997, 1.014] |

### 2.3 seg3 (anchor + criterion 段)

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| B0_h | 0.997 [0.989, 1.004] | 0.511 [0.488, 0.526] | 0.193 [0.169, 0.219] |
| B1_h | 0.998 [0.990, 1.005] | 0.490 [0.457, 0.514] | 0.183 [0.161, 0.206] |
| B2_h | 1.001 [0.991, 1.010] | 0.544 [0.528, 0.554] | 0.201 [0.176, 0.228] |
| B3_h | 1.005 [0.998, 1.012] | 0.496 [0.463, 0.522] | 0.191 [0.166, 0.217] |
| v2_anchor | 1.010 [1.002, 1.019] | 0.525 [0.494, 0.546] | 0.209 [0.183, 0.237] |
| v2_no_caps | 1.003 [0.995, 1.012] | 0.470 [0.436, 0.497] | 0.207 [0.181, 0.236] |
| v2_no_seg3_rule | 1.007 [0.998, 1.016] | 0.454 [0.414, 0.487] | 0.206 [0.181, 0.234] |
| v2_no_beta_summary | 1.012 [1.003, 1.021] | 0.521 [0.491, 0.541] | 0.223 [0.194, 0.253] |
| v2_plus_peer | 0.995 [0.987, 1.004] | 0.555 [0.530, 0.571] | 0.211 [0.185, 0.240] |
| v2_hybrid | 1.014 [1.005, 1.022] | 0.441 [0.402, 0.469] | 0.195 [0.171, 0.220] |

## 3. v2_anchor vs B0 (zero-shot) 显著性测试

★ = 95% CI 不重叠 (likely significant).

| Metric | B0 CI | v2_anchor CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.988, 1.003] | [0.987, 1.007] | +0.003 |  |
| seg1_persona_align | [0.273, 0.347] | [0.305, 0.389] | +0.037 |  |
| seg1_trait_pred_rho | [0.000, 0.000] | [0.052, 0.157] | +0.088 | ★ |
| seg2_macro_TED | [2.082, 2.472] | [2.128, 2.406] | -0.153 |  |
| seg2_macro_LCS_F1 | [0.444, 0.627] | [0.530, 0.593] | +0.092 |  |
| seg2_non_V_recall | [0.368, 0.588] | [0.175, 0.371] | -0.210 |  |
| seg2_chain_diversity | [0.525, 0.667] | [0.827, 0.920] | +0.284 | ★ |
| seg2_nl_specificity | [0.992, 1.006] | [0.996, 1.015] | +0.007 |  |
| seg3_nl_specificity | [0.989, 1.004] | [1.002, 1.019] | +0.014 |  |
| seg3_criterion_uniqueness | [0.488, 0.526] | [0.494, 0.546] | +0.014 |  |
| seg3_criterion_persona | [0.169, 0.219] | [0.183, 0.237] | +0.016 |  |

## 4. CAPS module 必要性 (paper main claim 关键 ablation)

| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.987, 1.007] | [1.002, 1.018] | +0.013 (Δ towards no_caps) |  |
| seg1_persona_align | [0.305, 0.389] | [0.296, 0.374] | -0.012 (Δ towards no_caps) |  |
| seg1_trait_pred_rho | [0.052, 0.157] | [-0.129, 0.125] | -0.104 (Δ towards no_caps) |  |
| seg2_macro_TED | [2.128, 2.406] | [2.001, 2.662] | +0.344 (Δ towards no_caps) |  |
| seg2_macro_LCS_F1 | [0.530, 0.593] | [0.535, 0.640] | +0.000 (Δ towards no_caps) |  |
| seg2_non_V_recall | [0.175, 0.371] | [0.333, 0.538] | +0.160 (Δ towards no_caps) |  |
| seg2_chain_diversity | [0.827, 0.920] | [0.679, 0.809] | -0.136 (Δ towards no_caps) | ★ |
| seg2_nl_specificity | [0.996, 1.015] | [0.998, 1.013] | -0.000 (Δ towards no_caps) |  |
| seg3_nl_specificity | [1.002, 1.019] | [0.995, 1.012] | -0.007 (Δ towards no_caps) |  |
| seg3_criterion_uniqueness | [0.494, 0.546] | [0.436, 0.497] | -0.055 (Δ towards no_caps) |  |
| seg3_criterion_persona | [0.183, 0.237] | [0.181, 0.236] | -0.002 (Δ towards no_caps) |  |

## 5. Winner per metric (within this model's 7 conditions)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_nl_specificity | **v2_hybrid** | 1.012 | [1.001, 1.022] |
| seg1_persona_align | **v2_no_seg3_rule** | 0.350 | [0.309, 0.395] |
| seg1_trait_pred_rho | **v2_no_seg3_rule** | 0.102 | [0.035, 0.193] |
| seg2_macro_TED | **v2_plus_peer** | 2.042 | [1.946, 2.481] |
| seg2_macro_LCS_F1 | **B3_h** | 0.573 | [0.554, 0.650] |
| seg2_non_V_recall | **B0_h** | 0.481 | [0.368, 0.588] |
| seg2_chain_diversity | **v2_hybrid** | 0.901 | [0.858, 0.926] |
| seg2_nl_specificity | **v2_no_beta_summary** | 1.011 | [1.003, 1.020] |
| seg3_nl_specificity | **v2_hybrid** | 1.014 | [1.005, 1.022] |
| seg3_criterion_uniqueness | **v2_plus_peer** | 0.555 | [0.530, 0.571] |
| seg3_criterion_persona | **v2_no_beta_summary** | 0.223 | [0.194, 0.253] |

## 6. Paper Main Claims for this model

Place numerical claims here based on §3-§5 above. Template:

```
On Gemini 3.1 Flash Lite, our v2_anchor prompt achieves vs zero-shot:
  - seg2 non-V recall: 27.2% [17.5%, 37.1%] vs 48.1% [36.8%, 58.8%]
  - seg2 chain diversity: 0.883 [0.827, 0.920] vs 0.599
  - seg3 criterion uniqueness: 0.525 [0.494, 0.546] vs 0.511
```

## 7. Reproducibility

- Model ID: `gemini-3.1-flash-lite`
- Run command:
```
# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):
python scripts/run_b6_sample.py --variant <V> --n 0 \
    --model gemini-3.1-flash-lite --output reports/datav3/crossmodel/gemini31flashlite_<COND>_inference.jsonl
# Then PKVC extract via GPT-4o:
python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
# Then bootstrap eval:
python scripts/generate_model_summary.py --model-slug gemini31flashlite \
    --display-name 'Gemini 3.1 Flash Lite' --model-id gemini-3.1-flash-lite \
    --output reports/datav3/gemini31flashlite_experiments_summary.md
```
