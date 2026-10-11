# Qwen 3.6 35B A3B 阶段实验结果 (paper-ready)

> 整理日期: 2026-05-26 (cross-model validation)
> 数据集: 162 designer × 3 Q7 seg
> 模型: Qwen 3.6 35B A3B (id: `qwen3.6-35b-a3b`)
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
| B0_h | 1.000 [0.991, 1.007] | 0.340 [0.300, 0.382] | 0.008 [-0.095, 0.130] |
| B1_h | 1.006 [0.998, 1.015] | 0.346 [0.305, 0.389] | 0.058 [-0.063, 0.194] |
| B2_h | 1.004 [0.996, 1.013] | 0.350 [0.309, 0.394] | -0.027 [-0.133, 0.114] |
| B3_h | 1.000 [0.991, 1.008] | 0.343 [0.303, 0.385] | -0.011 [-0.058, 0.054] |
| v2_anchor | 1.003 [0.992, 1.014] | 0.325 [0.282, 0.371] | 0.091 [0.055, 0.165] |
| v2_no_caps | 0.997 [0.988, 1.007] | 0.322 [0.277, 0.367] | 0.048 [-0.001, 0.131] |
| v2_no_seg3_rule | 1.002 [0.991, 1.014] | 0.328 [0.282, 0.373] | 0.080 [0.023, 0.169] |
| v2_no_beta_summary | 0.999 [0.990, 1.008] | 0.330 [0.282, 0.375] | 0.045 [-0.022, 0.150] |
| v2_plus_peer | 1.004 [0.993, 1.015] | 0.323 [0.278, 0.365] | 0.083 [0.052, 0.159] |
| v2_hybrid | 1.005 [0.993, 1.017] | 0.324 [0.279, 0.370] | 0.073 [0.046, 0.143] |

### 2.2 seg2 (chain reasoning, multi-state)

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| B0_h | 2.782 [2.271, 2.868] | 0.445 [0.429, 0.611] | 0.086 [0.026, 0.153] | 0.660 [0.586, 0.722] | 0.998 [0.991, 1.007] |
| B1_h | 2.723 [2.199, 2.811] | 0.533 [0.514, 0.619] | 0.210 [0.122, 0.300] | 0.790 [0.728, 0.827] | 0.998 [0.991, 1.006] |
| B2_h | 2.253 [2.158, 2.423] | 0.571 [0.551, 0.615] | 0.185 [0.108, 0.275] | 0.790 [0.728, 0.840] | 1.001 [0.993, 1.007] |
| B3_h | 2.499 [2.219, 2.593] | 0.547 [0.528, 0.615] | 0.333 [0.233, 0.440] | 0.889 [0.840, 0.920] | 1.006 [0.999, 1.014] |
| v2_anchor | 2.493 [2.202, 2.585] | 0.521 [0.498, 0.582] | 0.315 [0.208, 0.427] | 0.917 [0.868, 0.944] | 1.000 [0.991, 1.009] |
| v2_no_caps | 2.539 [2.279, 2.635] | 0.531 [0.512, 0.600] | 0.247 [0.154, 0.348] | 0.882 [0.819, 0.903] | 1.000 [0.992, 1.009] |
| v2_no_seg3_rule | 2.471 [2.189, 2.567] | 0.420 [0.398, 0.582] | 0.333 [0.229, 0.438] | 0.902 [0.853, 0.930] | 1.003 [0.995, 1.011] |
| v2_no_beta_summary | 2.527 [2.254, 2.624] | 0.424 [0.402, 0.588] | 0.279 [0.174, 0.397] | 0.919 [0.860, 0.934] | 1.006 [0.998, 1.015] |
| v2_plus_peer | 2.474 [2.191, 2.598] | 0.428 [0.403, 0.596] | 0.347 [0.243, 0.448] | 0.909 [0.860, 0.930] | 1.003 [0.995, 1.012] |
| v2_hybrid | 2.566 [2.318, 2.658] | 0.389 [0.363, 0.548] | 0.274 [0.177, 0.378] | 0.917 [0.868, 0.938] | 1.010 [1.000, 1.019] |

### 2.3 seg3 (anchor + criterion 段)

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| B0_h | 1.001 [0.992, 1.009] | 0.559 [0.542, 0.568] | 0.169 [0.149, 0.192] |
| B1_h | 0.999 [0.990, 1.008] | 0.546 [0.526, 0.558] | 0.175 [0.154, 0.198] |
| B2_h | 1.001 [0.992, 1.011] | 0.588 [0.573, 0.596] | 0.174 [0.153, 0.197] |
| B3_h | 1.007 [0.998, 1.018] | 0.599 [0.577, 0.610] | 0.181 [0.159, 0.204] |
| v2_anchor | 0.997 [0.987, 1.007] | 0.647 [0.631, 0.653] | 0.209 [0.179, 0.241] |
| v2_no_caps | 1.007 [0.997, 1.016] | 0.632 [0.615, 0.638] | 0.197 [0.168, 0.227] |
| v2_no_seg3_rule | 1.008 [0.996, 1.019] | 0.620 [0.601, 0.630] | 0.181 [0.155, 0.205] |
| v2_no_beta_summary | 1.003 [0.993, 1.013] | 0.639 [0.621, 0.646] | 0.211 [0.179, 0.241] |
| v2_plus_peer | 0.997 [0.986, 1.009] | 0.646 [0.631, 0.652] | 0.222 [0.190, 0.255] |
| v2_hybrid | 1.007 [0.997, 1.017] | 0.612 [0.594, 0.620] | 0.167 [0.144, 0.193] |

## 3. v2_anchor vs B0 (zero-shot) 显著性测试

★ = 95% CI 不重叠 (likely significant).

| Metric | B0 CI | v2_anchor CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.991, 1.007] | [0.992, 1.014] | +0.003 |  |
| seg1_persona_align | [0.300, 0.382] | [0.282, 0.371] | -0.015 |  |
| seg1_trait_pred_rho | [-0.095, 0.130] | [0.055, 0.165] | +0.083 |  |
| seg2_macro_TED | [2.271, 2.868] | [2.202, 2.585] | -0.289 |  |
| seg2_macro_LCS_F1 | [0.429, 0.611] | [0.498, 0.582] | +0.076 |  |
| seg2_non_V_recall | [0.026, 0.153] | [0.208, 0.427] | +0.229 | ★ |
| seg2_chain_diversity | [0.586, 0.722] | [0.868, 0.944] | +0.256 | ★ |
| seg2_nl_specificity | [0.991, 1.007] | [0.991, 1.009] | +0.001 |  |
| seg3_nl_specificity | [0.992, 1.009] | [0.987, 1.007] | -0.004 |  |
| seg3_criterion_uniqueness | [0.542, 0.568] | [0.631, 0.653] | +0.087 | ★ |
| seg3_criterion_persona | [0.149, 0.192] | [0.179, 0.241] | +0.040 |  |

## 4. CAPS module 必要性 (paper main claim 关键 ablation)

| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.992, 1.014] | [0.988, 1.007] | -0.006 (Δ towards no_caps) |  |
| seg1_persona_align | [0.282, 0.371] | [0.277, 0.367] | -0.003 (Δ towards no_caps) |  |
| seg1_trait_pred_rho | [0.055, 0.165] | [-0.001, 0.131] | -0.043 (Δ towards no_caps) |  |
| seg2_macro_TED | [2.202, 2.585] | [2.279, 2.635] | +0.046 (Δ towards no_caps) |  |
| seg2_macro_LCS_F1 | [0.498, 0.582] | [0.512, 0.600] | +0.011 (Δ towards no_caps) |  |
| seg2_non_V_recall | [0.208, 0.427] | [0.154, 0.348] | -0.068 (Δ towards no_caps) |  |
| seg2_chain_diversity | [0.868, 0.944] | [0.819, 0.903] | -0.035 (Δ towards no_caps) |  |
| seg2_nl_specificity | [0.991, 1.009] | [0.992, 1.009] | +0.001 (Δ towards no_caps) |  |
| seg3_nl_specificity | [0.987, 1.007] | [0.997, 1.016] | +0.010 (Δ towards no_caps) |  |
| seg3_criterion_uniqueness | [0.631, 0.653] | [0.615, 0.638] | -0.015 (Δ towards no_caps) |  |
| seg3_criterion_persona | [0.179, 0.241] | [0.168, 0.227] | -0.011 (Δ towards no_caps) |  |

## 5. Winner per metric (within this model's 7 conditions)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_nl_specificity | **B1_h** | 1.006 | [0.998, 1.015] |
| seg1_persona_align | **B2_h** | 0.350 | [0.309, 0.394] |
| seg1_trait_pred_rho | **v2_anchor** | 0.091 | [0.055, 0.165] |
| seg2_macro_TED | **B2_h** | 2.253 | [2.158, 2.423] |
| seg2_macro_LCS_F1 | **B2_h** | 0.571 | [0.551, 0.615] |
| seg2_non_V_recall | **v2_plus_peer** | 0.347 | [0.243, 0.448] |
| seg2_chain_diversity | **v2_no_beta_summary** | 0.919 | [0.860, 0.934] |
| seg2_nl_specificity | **v2_hybrid** | 1.010 | [1.000, 1.019] |
| seg3_nl_specificity | **v2_no_seg3_rule** | 1.008 | [0.996, 1.019] |
| seg3_criterion_uniqueness | **v2_anchor** | 0.647 | [0.631, 0.653] |
| seg3_criterion_persona | **v2_plus_peer** | 0.222 | [0.190, 0.255] |

## 6. Paper Main Claims for this model

Place numerical claims here based on §3-§5 above. Template:

```
On Qwen 3.6 35B A3B, our v2_anchor prompt achieves vs zero-shot:
  - seg2 non-V recall: 31.5% [20.8%, 42.7%] vs 8.6% [2.6%, 15.3%]
  - seg2 chain diversity: 0.917 [0.868, 0.944] vs 0.660
  - seg3 criterion uniqueness: 0.647 [0.631, 0.653] vs 0.559
```

## 7. Reproducibility

- Model ID: `qwen3.6-35b-a3b`
- Run command:
```
# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):
python scripts/run_b6_sample.py --variant <V> --n 0 \
    --model qwen3.6-35b-a3b --output reports/datav3/crossmodel/qwen35b_<COND>_inference.jsonl
# Then PKVC extract via GPT-4o:
python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
# Then bootstrap eval:
python scripts/generate_model_summary.py --model-slug qwen35b \
    --display-name 'Qwen 3.6 35B A3B' --model-id qwen3.6-35b-a3b \
    --output reports/datav3/qwen35b_experiments_summary.md
```
