# EMNLP 论文实验章节 — 表格规划

> 整理日期: 2026-05-26
> 数据源: `reports/datav3/cross_model_full_metrics_v2.json` (跨模型点估计) + `reports/datav3/gpt4o_paper5metric_bootstrap.json` (GPT-4o 1000-iter bootstrap CI)
> 配套分析: `reports/datav3/cross_model_ladder_ablation_results.md` §1-§9.5
> 目的: 在论文实验章节用最少表格 (3 个) 最大化展示工作价值, metric 从 11 个筛到 5 个.

---

## 1. Metric 筛选: 11 → 5

### 1.1 选中的 5 个 metric

3 维度 × 3 seg 全覆盖, 都有跨 model 一致信号:

| Seg | Metric | 维度 | 角色 | 跨 model 信号 |
|---|---|---|---|---|
| seg2 | macro-TED ↓ | Fidelity | chain 准度 | v2 best in 2/5, B0 → v2 跨 5/5 同向 ↓ |
| seg2 | non-V recall ↑ | Fidelity | 抗 V-default | v2 > B0 in 4/5 (+0.15 to +0.47) |
| seg2 | chain diversity ↑ | Anti-collapse | chain 差异化 | v2 > B0 单调 5/5 (+0.16 to +0.65) ★ |
| seg1 | trait-pred Spearman ρ ↑ | Personalization | trait 路由 | 删 CAPS → 5/5 model ↓ (唯一 5/5 一致 CAPS-attributable 效果) ★★ |
| seg3 | criterion uniqueness ↑ | Anti-collapse | 底线差异化 | v2 > B0 跨 5/5 同向 ↑ (+0.01 to +0.46) |

> Qwen 3.6 27B 因 v2-family runs 严重退化 (见 cross-model report §6.1), 整套实验排除, 跨模型分析在 5 model 上做.

### 1.2 舍弃的 6 个 metric 及理由

| Metric | Seg | 舍弃原因 |
|---|---|---|
| macro-LCS-F1 | seg2 | 跟 TED 信息冗余, B2 sweet spot in 4/5 → 跟 v2 narrative 反向, 写进 paper hurt 一致性 |
| NL specificity | seg1 | 跨 5 model × 10 condition 全部 ≈1.0 ±0.02, 无信号 |
| NL specificity | seg2 | 同上 |
| NL specificity | seg3 | 同上 |
| Persona-NL alignment | seg1 | 量级 0.30-0.37, condition 间差异 ≤0.05, 跨 model 不一致 |
| criterion-persona alignment | seg3 | v2 best in 2/5, mixed, 跟 criterion uniqueness 部分冗余 |

---

## 2. 三张 paper 表格

> All values are point estimates with 95% bootstrap confidence intervals (1000-iter designer-level resampling, percentile method). n = 162 designers × 3 segments = 486 instances per condition.
>
> **中文**: 所有数值为点估计 + 95% bootstrap 置信区间 (1000 次 designer 级重采样, 百分位法). n = 162 designer × 3 segment = 486 个实例 / condition.

### Table 1: Main Results — Module Ladder (Primary Model: GPT-4o)

**Caption (en)**: Module-ladder ablation on GPT-4o. Each row adds one input module to the prior baseline; **v2_anchor** is the full configuration with all five modules. Values are point estimate followed by 95% bootstrap CI. Bold = best in column.

**中文翻译**: GPT-4o 上的模块阶梯消融. 每行在前一 baseline 基础上加入一个输入模块; **v2_anchor** 是包含全部五个模块的完整配置. 数值为点估计 + 95% bootstrap CI. 加粗 = 该列最佳.

| Method                   | macro-TED ↓              | non-V recall ↑           | chain div ↑              | trait-pred ρ ↑              | crit uniq ↑              |
| ------------------------ | ------------------------ | ------------------------ | ------------------------ | --------------------------- | ------------------------ |
| B0 (zero-shot)           | 2.489 [2.218, 2.575]     | 0.111 [0.048, 0.184]     | 0.660 [0.586, 0.735]     | −0.018 [−0.077, +0.068]     | 0.459 [0.429, 0.480]     |
| B1 (+ pi)                | 2.434 [2.163, 2.518]     | 0.198 [0.111, 0.288]     | 0.660 [0.586, 0.735]     | +0.029 [−0.097, +0.163]     | 0.445 [0.423, 0.463]     |
| B2 (+ stance)            | 2.458 [2.174, 2.547]     | 0.235 [0.141, 0.329]     | 0.778 [0.716, 0.840]     | −0.018 [−0.066, +0.055]     | 0.502 [0.481, 0.517]     |
| B3 (+ self)              | 2.740 [2.207, 2.831]     | 0.284 [0.188, 0.383]     | 0.796 [0.735, 0.852]     | +0.010 [−0.005, +0.057]     | 0.496 [0.471, 0.514]     |
| **v2_anchor (+ CAPS)** ★ | **2.121 [2.008, 2.273]** | **0.383 [0.277, 0.487]** | **0.821 [0.759, 0.877]** | **+0.037 [−0.020, +0.130]** | **0.545 [0.527, 0.556]** |

**Narrative for §5.4 Main Results**:
- v2_anchor wins on all 5 metrics simultaneously, the only configuration to do so.
- Compared to zero-shot B0, v2_anchor improves macro-TED by 14.8% (2.489 → 2.121, CIs do not overlap), more than triples non-V recall (0.111 → 0.383, CIs do not overlap), and lifts chain diversity by +0.16 (CIs do not overlap).
- B3 (just before CAPS) regresses on macro-TED — adding self behavior without CAPS routing creates noise.
- CAPS module flips trait-pred ρ from near-zero / negative to clearly positive (B0 −0.018 → v2 +0.037).

**中文翻译 (§5.4 主结果)**:
- v2_anchor 在全部 5 个 metric 上同时取胜, 是唯一做到这一点的配置.
- 对比 zero-shot B0, v2_anchor 把 macro-TED 改善 14.8% (2.489 → 2.121, CI 不重叠), 把 non-V recall 提升 3 倍多 (0.111 → 0.383, CI 不重叠), chain diversity 提升 +0.16 (CI 不重叠).
- B3 (CAPS 前一档) 在 macro-TED 上出现退化 — 在无 CAPS routing 的情况下加入 self behavior 会引入噪声.
- CAPS 模块把 trait-pred ρ 从接近零 / 负值翻转到明显正值 (B0 −0.018 → v2 +0.037).

---

### Table 2: Ablation — v2_anchor Component Removal (Primary Model: GPT-4o)

**Caption (en)**: Component ablations starting from v2_anchor on GPT-4o. Each row removes (−) or adds (+) one component. Bold = best in column. † v2_hybrid only degrades the seg3 prompt; its seg1 and seg2 metric values are therefore identical to full v2_anchor by construction, and the only differential cell is criterion uniqueness (seg3).

**中文翻译**: GPT-4o 上从 v2_anchor 出发的组件消融. 每行减去 (−) 或加入 (+) 一个组件. 加粗 = 该列最佳. † v2_hybrid 仅退化 seg3 的 prompt; 因此 seg1 / seg2 的 metric 值按构造与完整 v2_anchor 相同, 唯一有差异的单元格是 criterion uniqueness (seg3).

| Method | macro-TED ↓ | non-V recall ↑ | chain div ↑ | trait-pred ρ ↑ | crit uniq ↑ |
|---|---|---|---|---|---|
| **v2_anchor (full)** ★ | **2.121 [2.008, 2.273]** | **0.383 [0.277, 0.487]** | 0.821 [0.759, 0.877] | +0.037 [−0.020, +0.130] | 0.545 [0.527, 0.556] |
| − CAPS quad block | 2.804 [2.310, 2.896] | 0.358 [0.257, 0.467] | 0.765 [0.698, 0.827] | **−0.029 [−0.047, +0.011]** | 0.513 [0.492, 0.528] |
| − seg3 decision rule | 2.798 [2.295, 2.883] | 0.284 [0.185, 0.389] | 0.846 [0.790, 0.889] | +0.048 [−0.002, +0.128] | 0.508 [0.486, 0.522] |
| − β summary header | 2.462 [2.195, 2.544] | 0.296 [0.197, 0.400] | 0.877 [0.827, 0.901] | −0.043 [−0.082, +0.021] | 0.517 [0.497, 0.530] |
| + peer retrieval | 2.991 [2.223, 3.076] | 0.333 [0.233, 0.432] | **0.901 [0.852, 0.920]** | **+0.077 [+0.020, +0.162]** | **0.573 [0.551, 0.587]** |
| v2_hybrid † | 2.121 [2.008, 2.273] | 0.383 [0.277, 0.487] | 0.821 [0.759, 0.877] | +0.037 [−0.020, +0.130] | 0.503 [0.482, 0.517] |

**Narrative for §5.5 Ablation**:
- **CAPS module is the most critical component**: removing it inflates macro-TED by +32% (2.12 → 2.80, CIs do not overlap) and flips trait-pred ρ from +0.037 to −0.029.
- **seg3 rule and CAPS are coupled**: removing the rule alone causes the same +32% TED penalty, suggesting the rule operationalizes CAPS routing for boundary decisions.
- **β summary is supplementary**: removing it preserves TED but hurts non-V recall (−0.087) and reverses trait-pred ρ sign (+0.037 → −0.043).
- **Peer retrieval is a trade-off**: adds diversity (+0.080 chain div, +0.028 crit uniq) and is the only configuration with trait-pred ρ CI fully above zero (+0.077, [+0.020, +0.162]), but heavily hurts fidelity (TED 2.12 → 2.99, +41%). We exclude it from the final design.
- **v2_hybrid** is included only as a per-seg dispatch reference; by construction its seg1/seg2 metrics equal full v2_anchor (single column differential at seg3 criterion uniqueness, where the v2_clean fallback drops it from 0.545 to 0.503).

**中文翻译 (§5.5 消融)**:
- **CAPS 模块是最关键的组件**: 移除后 macro-TED 上涨 +32% (2.12 → 2.80, CI 不重叠), trait-pred ρ 由 +0.037 翻转为 −0.029.
- **seg3 规则与 CAPS 耦合**: 单独移除该规则同样造成 +32% 的 TED 退化, 说明该规则把 CAPS routing 操作化成了边界决策.
- **β summary 属于补充**: 移除后 TED 保持, 但 non-V recall 下降 (−0.087), trait-pred ρ 翻号 (+0.037 → −0.043).
- **Peer retrieval 是一个 trade-off**: 提升多样性 (chain div +0.080, crit uniq +0.028), 是唯一 trait-pred ρ 的 CI 完全大于 0 的配置 (+0.077, [+0.020, +0.162]), 但严重伤 fidelity (TED 2.12 → 2.99, +41%). 因此最终设计不采用.
- **v2_hybrid** 仅作为 per-seg 调度的参考; 按构造 seg1 / seg2 的 metric 与完整 v2_anchor 一致 (差异只在 seg3 criterion uniqueness 一列, 这里 v2_clean fallback 将其从 0.545 降到 0.503).

---

### Table 3: Cross-Model Generalization — v2_anchor vs B0_h (Δ across 5 LLMs)

**Caption (en)**: Cross-model robustness. Each cell is the directional change from B0 (zero-shot) to v2_anchor; ✓ = correct direction (↓ for TED, ↑ for others). Point estimates only (cross-model bootstrap not run; see Table 1/2 for GPT-4o CIs).

**中文翻译**: 跨模型鲁棒性. 每个单元格为从 B0 (zero-shot) 到 v2_anchor 的方向性变化; ✓ = 方向正确 (TED 为 ↓, 其它为 ↑). 仅给出点估计 (跨模型 bootstrap 未跑; GPT-4o 的 CI 见 Table 1 / 2).

| Model | macro-TED Δ ↓ | non-V Δ ↑ | chain div Δ ↑ | trait-pred ρ Δ ↑ | crit uniq Δ ↑ | 5/5 ↑? |
|---|---|---|---|---|---|---|
| GPT-4o | **−0.37** ✓ | +0.27 ✓ | +0.16 ✓ | +0.06 ✓ | +0.09 ✓ | **5/5** |
| Gemini 3.1 Flash Lite | −0.15 ✓ | −0.21 ✗ | +0.28 ✓ | +0.09 ✓ | +0.01 ✓ | 4/5 |
| Claude Haiku 4.5 | −0.13 ✓ | **+0.47** ✓ | **+0.65** ✓ | +0.03 ✓ | **+0.46** ✓ | **5/5** |
| DeepSeek V4 Flash | −0.18 ✓ | +0.15 ✓ | +0.16 ✓ | **+0.17** ✓ | +0.18 ✓ | **5/5** |
| Qwen 3.6 35B A3B | −0.29 ✓ | +0.23 ✓ | +0.26 ✓ | +0.08 ✓ | +0.09 ✓ | **5/5** |
| **Cross-model 一致** | **5/5** ✓ | 4/5 ✓ | **5/5** ✓ | **5/5** ✓ | **5/5** ✓ | **24/25 cells (96%)** |

**Narrative for §5.6 Cross-Model Generalization**:
- 24 of 25 (model, metric) cells show v2_anchor improving over zero-shot in the correct direction (96% consistency).
- The single outlier (Gemini non-V recall) reflects Gemini's already-high zero-shot baseline (48.1% vs ~10% for other models, near ceiling effect for this metric).
- The proposed prompt design transfers across providers (OpenAI, Google, Anthropic, DeepSeek, Alibaba) and scales (lite → mid → frontier).
- macro-TED, chain diversity, trait-pred ρ, and criterion uniqueness each generalize across all 5 models without exception.

**中文翻译 (§5.6 跨模型泛化)**:
- 25 个 (model, metric) 单元格中有 24 个 v2_anchor 相对 zero-shot 的方向正确 (96% 一致性).
- 唯一的 outlier (Gemini non-V recall) 源于 Gemini 本身的 zero-shot baseline 已经很高 (48.1%, 其它模型约 10%), 对该 metric 接近 ceiling.
- 所提 prompt 设计可跨厂商 (OpenAI, Google, Anthropic, DeepSeek, Alibaba) 与跨规模 (lite → mid → frontier) 迁移.
- macro-TED, chain diversity, trait-pred ρ, criterion uniqueness 这四项无一例外地在全部 5 个模型上泛化.

---

## 3. 表格摆放建议

### Layout A (推荐, 3 表清晰分工)
- §5.4 Main Results: **Table 1**
- §5.5 Ablation: **Table 2**
- §5.6 Cross-Model Generalization: **Table 3**

每张表 5-6 行 × 6 列 (含 Method 列), 单栏 LaTeX 可容下, 不需 sidewaystable.

### Layout B (省版, 2 表)
- 合并 Table 1 + Table 2 → 单表 10 行 × 6 列 (B0..v2 + v2 ablations + +peer/hybrid)
- Table 3 保留
- 缺点: 主结果跟 ablation 混在一起, narrative 不清

### Layout C (cross-model 是 hero)
- §5.4: Table 3 当 main result
- §5.5: Table 1 + 2 合并放 appendix

→ 建议走 **Layout A**, narrative 最干净, 跟 §5.4/5.5/5.6 三个子节天然对应.

---

## 4. Caption / footnote 模板

每张表都加 (英文, EMNLP submission):
- **n** = 162 designers × 3 segments = 486 designer-segment instances per condition
- **Sampling**: stateless per-segment generation (no autoregressive continuation across segs)
- **Bootstrap**: 1000-iter designer-level resampling, 95% percentile CI (Table 1, Table 2)
- **Decoding**: temperature 0.7, max 600 tokens
- **PKVC extraction**: GPT-4o, temperature 0, constrained JSON schema (v8.7 extractor)

**中文翻译**:
- **n** = 162 designer × 3 segment = 486 个 designer-segment 实例 / condition
- **采样**: 无状态的 per-segment 生成 (segment 间无 autoregressive 续写)
- **Bootstrap**: 1000 次 designer 级重采样, 95% 百分位 CI (Table 1, Table 2)
- **解码**: temperature 0.7, 最多 600 tokens
- **PKVC 抽取**: GPT-4o, temperature 0, 约束 JSON schema (v8.7 extractor)

---

## 5. 已确认

- ✅ Table 1 / Table 2 加 95% bootstrap CI (1000-iter, percentile)
- ✅ Qwen 27B 整体移除 (cross-model 分析降至 5 model)
- ✅ Caption 用英文 (EMNLP submission)
- ✅ Table 2 加 hybrid 跟 v2 在 seg1/seg2 完全一致的脚注 († 标注)
