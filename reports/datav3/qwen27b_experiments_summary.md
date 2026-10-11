# Qwen 3.6 27B 阶段实验结果 (paper-ready)

> 整理日期: 2026-05-26 (cross-model validation)
> 数据集: 162 designer × 3 Q7 seg
> 模型: Qwen 3.6 27B (id: `qwen3.6-27b`)
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
| B0_h | 0.998 [0.990, 1.005] | 0.324 [0.285, 0.364] | 0.126 [-0.008, 0.241] |
| B1_h | 1.006 [0.995, 1.016] | 0.326 [0.281, 0.372] | 0.046 [-0.106, 0.231] |
| B2_h | 1.004 [0.993, 1.014] | 0.299 [0.251, 0.350] | 0.093 [-0.031, 0.227] |
| B3_h | 1.007 [0.995, 1.018] | 0.320 [0.264, 0.375] | 0.013 [-0.032, 0.104] |
| v2_anchor | 1.006 [0.996, 1.016] | 0.330 [0.289, 0.375] | 0.083 [0.037, 0.164] |
| v2_no_caps | 1.000 [0.991, 1.010] | 0.332 [0.289, 0.378] | 0.025 [-0.015, 0.096] |
| v2_no_seg3_rule | 0.996 [0.986, 1.007] | 0.340 [0.298, 0.382] | 0.079 [0.016, 0.177] |
| v2_no_beta_summary | 1.005 [0.994, 1.016] | 0.337 [0.295, 0.380] | 0.090 [0.013, 0.207] |
| v2_plus_peer | 1.008 [0.998, 1.018] | 0.331 [0.287, 0.375] | 0.088 [0.050, 0.168] |
| v2_hybrid | 1.000 [0.990, 1.009] | 0.340 [0.298, 0.385] | 0.093 [0.047, 0.178] |

### 2.2 seg2 (chain reasoning, multi-state)

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| B0_h | 2.676 [2.560, 3.365] | 0.078 [0.051, 0.133] | 0.079 [0.026, 0.143] | 0.176 [0.118, 0.242] | 1.006 [0.998, 1.014] |
| B1_h | 2.809 [2.309, 2.912] | 0.443 [0.422, 0.611] | 0.328 [0.214, 0.436] | 0.898 [0.828, 0.906] | 1.004 [0.995, 1.013] |
| B2_h | 2.409 [2.220, 2.590] | 0.613 [0.578, 0.647] | 0.431 [0.302, 0.569] | 0.923 [0.856, 0.923] | 1.003 [0.992, 1.014] |
| B3_h | 2.302 [2.126, 2.477] | 0.608 [0.578, 0.638] | 0.542 [0.396, 0.682] | 0.909 [0.848, 0.919] | 1.005 [0.994, 1.014] |
| v2_anchor | 2.687 [2.575, 3.362] | 0.068 [0.044, 0.119] | 0.027 [0.000, 0.068] | 0.167 [0.113, 0.227] | 1.001 [0.992, 1.010] |
| v2_no_caps | 2.662 [2.540, 3.334] | 0.074 [0.048, 0.130] | 0.068 [0.015, 0.137] | 0.180 [0.120, 0.247] | 1.007 [0.998, 1.015] |
| v2_no_seg3_rule | 2.670 [2.561, 3.356] | 0.070 [0.043, 0.122] | 0.051 [0.012, 0.108] | 0.160 [0.103, 0.225] | 1.014 [1.007, 1.022] |
| v2_no_beta_summary | 2.680 [2.575, 3.381] | 0.075 [0.048, 0.137] | 0.055 [0.012, 0.113] | 0.182 [0.122, 0.243] | 1.005 [0.997, 1.014] |
| v2_plus_peer | 2.689 [2.573, 3.378] | 0.074 [0.049, 0.128] | 0.079 [0.025, 0.141] | 0.183 [0.124, 0.248] | 1.005 [0.996, 1.014] |
| v2_hybrid | 2.638 [2.530, 3.319] | 0.078 [0.051, 0.137] | 0.065 [0.014, 0.123] | 0.181 [0.116, 0.245] | 1.000 [0.993, 1.008] |

### 2.3 seg3 (anchor + criterion 段)

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| B0_h | 1.007 [0.999, 1.015] | 0.507 [0.484, 0.522] | 0.215 [0.186, 0.244] |
| B1_h | 1.000 [0.991, 1.010] | 0.501 [0.469, 0.522] | 0.215 [0.184, 0.247] |
| B2_h | 1.016 [1.002, 1.030] | 0.597 [0.576, 0.605] | 0.187 [0.154, 0.220] |
| B3_h | 1.002 [0.990, 1.015] | 0.594 [0.569, 0.604] | 0.193 [0.159, 0.228] |
| v2_anchor | 1.000 [0.990, 1.010] | 0.627 [0.608, 0.636] | 0.216 [0.187, 0.248] |
| v2_no_caps | 1.002 [0.993, 1.013] | 0.618 [0.600, 0.628] | 0.216 [0.187, 0.245] |
| v2_no_seg3_rule | 1.010 [1.001, 1.019] | 0.591 [0.574, 0.601] | 0.211 [0.182, 0.238] |
| v2_no_beta_summary | 1.005 [0.995, 1.014] | 0.631 [0.614, 0.640] | 0.216 [0.189, 0.245] |
| v2_plus_peer | 1.006 [0.997, 1.016] | 0.642 [0.625, 0.650] | 0.224 [0.194, 0.255] |
| v2_hybrid | 1.003 [0.994, 1.013] | 0.584 [0.566, 0.594] | 0.203 [0.176, 0.232] |

## 3. v2_anchor vs B0 (zero-shot) 显著性测试

★ = 95% CI 不重叠 (likely significant).

| Metric | B0 CI | v2_anchor CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.990, 1.005] | [0.996, 1.016] | +0.008 |  |
| seg1_persona_align | [0.285, 0.364] | [0.289, 0.375] | +0.006 |  |
| seg1_trait_pred_rho | [-0.008, 0.241] | [0.037, 0.164] | -0.044 |  |
| seg2_macro_TED | [2.560, 3.365] | [2.575, 3.362] | +0.012 |  |
| seg2_macro_LCS_F1 | [0.051, 0.133] | [0.044, 0.119] | -0.010 |  |
| seg2_non_V_recall | [0.026, 0.143] | [0.000, 0.068] | -0.052 |  |
| seg2_chain_diversity | [0.118, 0.242] | [0.113, 0.227] | -0.010 |  |
| seg2_nl_specificity | [0.998, 1.014] | [0.992, 1.010] | -0.005 |  |
| seg3_nl_specificity | [0.999, 1.015] | [0.990, 1.010] | -0.007 |  |
| seg3_criterion_uniqueness | [0.484, 0.522] | [0.608, 0.636] | +0.120 | ★ |
| seg3_criterion_persona | [0.186, 0.244] | [0.187, 0.248] | +0.001 |  |

## 4. CAPS module 必要性 (paper main claim 关键 ablation)

| Metric | v2_anchor CI | v2_no_caps CI | Δ | Sig? |
|---|---|---|---|---|
| seg1_nl_specificity | [0.996, 1.016] | [0.991, 1.010] | -0.006 (Δ towards no_caps) |  |
| seg1_persona_align | [0.289, 0.375] | [0.289, 0.378] | +0.002 (Δ towards no_caps) |  |
| seg1_trait_pred_rho | [0.037, 0.164] | [-0.015, 0.096] | -0.058 (Δ towards no_caps) |  |
| seg2_macro_TED | [2.575, 3.362] | [2.540, 3.334] | -0.025 (Δ towards no_caps) |  |
| seg2_macro_LCS_F1 | [0.044, 0.119] | [0.048, 0.130] | +0.006 (Δ towards no_caps) |  |
| seg2_non_V_recall | [0.000, 0.068] | [0.015, 0.137] | +0.041 (Δ towards no_caps) |  |
| seg2_chain_diversity | [0.113, 0.227] | [0.120, 0.247] | +0.013 (Δ towards no_caps) |  |
| seg2_nl_specificity | [0.992, 1.010] | [0.998, 1.015] | +0.005 (Δ towards no_caps) |  |
| seg3_nl_specificity | [0.990, 1.010] | [0.993, 1.013] | +0.002 (Δ towards no_caps) |  |
| seg3_criterion_uniqueness | [0.608, 0.636] | [0.600, 0.628] | -0.009 (Δ towards no_caps) |  |
| seg3_criterion_persona | [0.187, 0.248] | [0.187, 0.245] | -0.000 (Δ towards no_caps) |  |

## 5. Winner per metric (within this model's 7 conditions)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_nl_specificity | **v2_plus_peer** | 1.008 | [0.998, 1.018] |
| seg1_persona_align | **v2_hybrid** | 0.340 | [0.298, 0.385] |
| seg1_trait_pred_rho | **B0_h** | 0.126 | [-0.008, 0.241] |
| seg2_macro_TED | **B3_h** | 2.302 | [2.126, 2.477] |
| seg2_macro_LCS_F1 | **B2_h** | 0.613 | [0.578, 0.647] |
| seg2_non_V_recall | **B3_h** | 0.542 | [0.396, 0.682] |
| seg2_chain_diversity | **B2_h** | 0.923 | [0.856, 0.923] |
| seg2_nl_specificity | **v2_no_seg3_rule** | 1.014 | [1.007, 1.022] |
| seg3_nl_specificity | **B2_h** | 1.016 | [1.002, 1.030] |
| seg3_criterion_uniqueness | **v2_plus_peer** | 0.642 | [0.625, 0.650] |
| seg3_criterion_persona | **v2_plus_peer** | 0.224 | [0.194, 0.255] |

## 6. Paper Main Claims for this model

Place numerical claims here based on §3-§5 above. Template:

```
On Qwen 3.6 27B, our v2_anchor prompt achieves vs zero-shot:
  - seg2 non-V recall: 2.7% [0.0%, 6.8%] vs 7.9% [2.6%, 14.3%]
  - seg2 chain diversity: 0.167 [0.113, 0.227] vs 0.176
  - seg3 criterion uniqueness: 0.627 [0.608, 0.636] vs 0.507
```

## 7. Reproducibility

- Model ID: `qwen3.6-27b`
- Run command:
```
# For each condition (b0, v2, v2_no_caps, v2_no_seg3_rule, v2_no_beta_summary, v2_plus_peer, v2_hybrid):
python scripts/run_b6_sample.py --variant <V> --n 0 \
    --model qwen3.6-27b --output reports/datav3/crossmodel/qwen27b_<COND>_inference.jsonl
# Then PKVC extract via GPT-4o:
python scripts/eval_baselines_pkvc.py --input <inf.jsonl> --output <pkvc.jsonl>
# Then bootstrap eval:
python scripts/generate_model_summary.py --model-slug qwen27b \
    --display-name 'Qwen 3.6 27B' --model-id qwen3.6-27b \
    --output reports/datav3/qwen27b_experiments_summary.md
```
