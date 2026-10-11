# DeepSeek V4 Flash 阶段实验结果 (paper-ready)

> 整理日期: 2026-05-26 (cross-model validation)
> 数据集: 162 designer × 3 Q7 seg
> 模型: DeepSeek V4 Flash (id: `deepseek-v4-flash`)
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
| B0_h | 1.002 [0.994, 1.010] | 0.313 [0.277, 0.351] | -0.058 [-0.124, 0.016] |
| B1_h | 1.003 [0.996, 1.011] | 0.327 [0.287, 0.367] | -0.023 [-0.117, 0.088] |
| B2_h | 0.991 [0.982, 0.999] | 0.331 [0.289, 0.372] | -0.036 [-0.115, 0.054] |
| B3_h | 0.999 [0.990, 1.007] | 0.325 [0.286, 0.367] | 0.061 [-0.034, 0.180] |
| v2_anchor | 1.000 [0.991, 1.010] | 0.328 [0.285, 0.369] | 0.114 [0.024, 0.235] |
| v2_no_caps | 0.999 [0.988, 1.009] | 0.313 [0.271, 0.353] | 0.062 [-0.032, 0.177] |
| v2_no_seg3_rule | 0.998 [0.989, 1.007] | 0.316 [0.273, 0.362] | 0.059 [-0.056, 0.193] |
| v2_no_beta_summary | 1.004 [0.995, 1.013] | 0.322 [0.283, 0.368] | 0.125 [0.036, 0.246] |
| v2_plus_peer | 1.012 [1.002, 1.022] | 0.316 [0.276, 0.358] | 0.123 [0.052, 0.228] |
| v2_hybrid | 0.998 [0.987, 1.008] | 0.320 [0.277, 0.361] | 0.082 [-0.002, 0.198] |

### 2.2 seg2 (chain reasoning, multi-state)

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| B0_h | 2.838 [2.363, 2.926] | 0.437 [0.420, 0.599] | 0.074 [0.025, 0.137] | 0.741 [0.673, 0.790] | 0.995 [0.988, 1.004] |
| B1_h | 2.746 [2.234, 2.836] | 0.458 [0.441, 0.628] | 0.086 [0.029, 0.150] | 0.611 [0.537, 0.691] | 1.005 [0.997, 1.013] |
| B2_h | 2.343 [2.203, 2.484] | 0.602 [0.574, 0.628] | 0.158 [0.080, 0.244] | 0.720 [0.647, 0.793] | 1.005 [0.998, 1.012] |
| B3_h | 2.862 [2.383, 2.953] | 0.436 [0.419, 0.597] | 0.200 [0.108, 0.295] | 0.829 [0.770, 0.882] | 0.998 [0.989, 1.007] |
| v2_anchor | 2.654 [2.107, 2.749] | 0.532 [0.512, 0.620] | 0.227 [0.129, 0.333] | 0.904 [0.846, 0.919] | 1.006 [0.996, 1.015] |
| v2_no_caps | 2.521 [2.249, 2.620] | 0.545 [0.527, 0.615] | 0.263 [0.165, 0.361] | 0.831 [0.770, 0.885] | 1.006 [0.998, 1.013] |
| v2_no_seg3_rule | 2.403 [2.255, 2.551] | 0.586 [0.556, 0.614] | 0.182 [0.096, 0.279] | 0.867 [0.815, 0.896] | 1.010 [1.003, 1.018] |
| v2_no_beta_summary | 2.419 [2.272, 2.566] | 0.586 [0.559, 0.612] | 0.098 [0.031, 0.177] | 0.817 [0.756, 0.870] | 1.010 [1.001, 1.019] |
| v2_plus_peer | 2.860 [2.368, 2.953] | 0.522 [0.504, 0.606] | 0.286 [0.185, 0.397] | 0.917 [0.868, 0.931] | 1.007 [0.998, 1.016] |
| v2_hybrid | 2.331 [2.176, 2.475] | 0.575 [0.544, 0.607] | 0.266 [0.157, 0.377] | 0.868 [0.809, 0.904] | 1.008 [0.998, 1.018] |

### 2.3 seg3 (anchor + criterion 段)

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| B0_h | 1.000 [0.992, 1.008] | 0.445 [0.412, 0.469] | 0.165 [0.145, 0.187] |
| B1_h | 1.000 [0.992, 1.007] | 0.460 [0.426, 0.485] | 0.173 [0.152, 0.196] |
| B2_h | 1.006 [0.998, 1.014] | 0.498 [0.468, 0.517] | 0.166 [0.144, 0.188] |
| B3_h | 1.002 [0.994, 1.010] | 0.496 [0.463, 0.522] | 0.165 [0.145, 0.189] |
| v2_anchor | 1.001 [0.990, 1.011] | 0.628 [0.601, 0.644] | 0.186 [0.160, 0.214] |
| v2_no_caps | 1.006 [0.997, 1.015] | 0.521 [0.494, 0.541] | 0.159 [0.137, 0.183] |
| v2_no_seg3_rule | 0.998 [0.989, 1.006] | 0.531 [0.502, 0.550] | 0.166 [0.141, 0.190] |
| v2_no_beta_summary | 1.006 [0.997, 1.016] | 0.568 [0.533, 0.591] | 0.173 [0.145, 0.202] |
| v2_plus_peer | 1.004 [0.995, 1.012] | 0.584 [0.549, 0.608] | 0.197 [0.171, 0.225] |
| v2_hybrid | 1.006 [0.997, 1.015] | 0.506 [0.474, 0.528] | 0.149 [0.127, 0.172] |

## 3. v2_anchor vs B0 (zero-shot) 显著性测试

★ = 95% CI 不重叠 (likely significant).

| Metric | B0 CI | v2_anchor CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.994, 1.010] | [0.991, 1.010] | -0.002 |  |
| seg1_persona_align | [0.277, 0.351] | [0.285, 0.369] | +0.015 |  |
| seg1_trait_pred_rho | [-0.124, 0.016] | [0.024, 0.235] | +0.172 | ★ |
| seg2_macro_TED | [2.363, 2.926] | [2.107, 2.749] | -0.184 |  |
| seg2_macro_LCS_F1 | [0.420, 0.599] | [0.512, 0.620] | +0.095 |  |
| seg2_non_V_recall | [0.025, 0.137] | [0.129, 0.333] | +0.153 |  |
| seg2_chain_diversity | [0.673, 0.790] | [0.846, 0.919] | +0.164 | ★ |
| seg2_nl_specificity | [0.988, 1.004] | [0.996, 1.015] | +0.010 |  |
| seg3_nl_specificity | [0.992, 1.008] | [0.990, 1.011] | +0.001 |  |
| seg3_criterion_uniqueness | [0.412, 0.469] | [0.601, 0.644] | +0.183 | ★ |
| seg3_criterion_persona | [0.145, 0.187] | [0.160, 0.214] | +0.021 |  |

## 4. CAPS module 必要性 (paper main claim 关键 ablation)

| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.991, 1.010] | [0.988, 1.009] | -0.002 (Δ towards no_caps) |  |
| seg1_persona_align | [0.285, 0.369] | [0.271, 0.353] | -0.014 (Δ towards no_caps) |  |
| seg1_trait_pred_rho | [0.024, 0.235] | [-0.032, 0.177] | -0.052 (Δ towards no_caps) |  |
| seg2_macro_TED | [2.107, 2.749] | [2.249, 2.620] | -0.133 (Δ towards no_caps) |  |
| seg2_macro_LCS_F1 | [0.512, 0.620] | [0.527, 0.615] | +0.013 (Δ towards no_caps) |  |
| seg2_non_V_recall | [0.129, 0.333] | [0.165, 0.361] | +0.036 (Δ towards no_caps) |  |
| seg2_chain_diversity | [0.846, 0.919] | [0.770, 0.885] | -0.073 (Δ towards no_caps) |  |
| seg2_nl_specificity | [0.996, 1.015] | [0.998, 1.013] | +0.001 (Δ towards no_caps) |  |
| seg3_nl_specificity | [0.990, 1.011] | [0.997, 1.015] | +0.006 (Δ towards no_caps) |  |
| seg3_criterion_uniqueness | [0.601, 0.644] | [0.494, 0.541] | -0.107 (Δ towards no_caps) | ★ |
| seg3_criterion_persona | [0.160, 0.214] | [0.137, 0.183] | -0.027 (Δ towards no_caps) |  |

## 5. Winner per metric (within this model's 7 conditions)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_nl_specificity | **v2_plus_peer** | 1.012 | [1.002, 1.022] |
| seg1_persona_align | **B2_h** | 0.331 | [0.289, 0.372] |
| seg1_trait_pred_rho | **v2_no_beta_summary** | 0.125 | [0.036, 0.246] |
| seg2_macro_TED | **v2_hybrid** | 2.331 | [2.176, 2.475] |
| seg2_macro_LCS_F1 | **B2_h** | 0.602 | [0.574, 0.628] |
| seg2_non_V_recall | **v2_plus_peer** | 0.286 | [0.185, 0.397] |
| seg2_chain_diversity | **v2_plus_peer** | 0.917 | [0.868, 0.931] |
| seg2_nl_specificity | **v2_no_seg3_rule** | 1.010 | [1.003, 1.018] |
| seg3_nl_specificity | **v2_no_caps** | 1.006 | [0.997, 1.015] |
| seg3_criterion_uniqueness | **v2_anchor** | 0.628 | [0.601, 0.644] |
| seg3_criterion_persona | **v2_plus_peer** | 0.197 | [0.171, 0.225] |

## 6. Paper Main Claims for this model

Place numerical claims here based on §3-§5 above. Template:

```
On DeepSeek V4 Flash, our v2_anchor prompt achieves vs zero-shot:
  - seg2 non-V recall: 22.7% [12.9%, 33.3%] vs 7.4% [2.5%, 13.7%]
  - seg2 chain diversity: 0.904 [0.846, 0.919] vs 0.741
  - seg3 criterion uniqueness: 0.628 [0.601, 0.644] vs 0.445
```

## 7. Reproducibility

- Model ID: `deepseek-v4-flash`
- Run command:
```
# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):
python scripts/run_b6_sample.py --variant <V> --n 0 \
    --model deepseek-v4-flash --output reports/datav3/crossmodel/deepseekv4flash_<COND>_inference.jsonl
# Then PKVC extract via GPT-4o:
python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
# Then bootstrap eval:
python scripts/generate_model_summary.py --model-slug deepseekv4flash \
    --display-name 'DeepSeek V4 Flash' --model-id deepseek-v4-flash \
    --output reports/datav3/deepseekv4flash_experiments_summary.md
```
