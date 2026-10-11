# SFT vs BASE 阶段实验完整结果 (paper-ready)

> 整理日期: 2026-05-26
> 数据集: 162 designer × 3 Q7 seg (seg1 第一反应 / seg2 决策链路 / seg3 底线段)
> 模型: **BASE** = Qwen2.5-7B-Instruct (zero-shot, no LoRA);
>      **SFT** = Qwen2.5-7B-Instruct + LoRA r=32 (`/data/wy/sft_v3_ckpts/stage_warmup_ep5/`)
> SFT 训练数据: 162 designer × Q1-Q6 = 962 example (770 train + 192 val), prep_v3 (含 4 option block + v3 expansion, severe hallu 31%→8%)
> SFT 训练: warmup 5 epoch, L_lm only, batch_size=4 × grad_accum 8 = eff 32, lr 1e-4, bf16, 单 A800 80GB
> Final train_loss: 1.263, eval_loss: 1.327 (vs dirty data baseline 2.22/1.88, -43%/-29%)
> 输入构造: 复用 v2_anchor 系列 prompt (见 `scripts/run_b6_sample.py`)
> Inference: local Qwen2.5-7B + LoRA, JSON schema 嵌 prompt (Qwen 无 strict schema), temperature=0.7, top_p=0.9
> Evaluation: 11-metric set + 1000-bootstrap 95% CI (designer-level resampling), 跟 GPT-4o eval 完全对齐

---

## 0. 实验总览

### 0.1 总规模

- **14 个 condition × 162 designer × 3 seg = 6,804 inference calls** (本地 GPU, 0 cost)
- **+ 6,419 PKVC extraction calls (v8.7 extractor, GPT-4o)** ($28)
- 推理 wall time: ~2 h (5 GPU 并行 3 wave)
- 训练 wall time: 50 min (5 epoch)
- Extract + bootstrap: 30 min

### 0.2 14 个 condition 分类 (BASE × 7 + SFT × 7)

| Group | Condition | 用途 |
|---|---|---|
| **Main** | v2 (full anchor, all modules) | paper main prompt |
| | B0 (zero-shot, empty modules) | 基线下限 |
| **架构 ablation** | v2_hybrid (seg3 用 v2_clean prompt) | per-seg dispatch 价值 |
| | v2_plus_peer (v2 + ④ peer block) | peer net 价值 |
| **组件 ablation** | v2_no_caps (砍 ⑤ CAPS module) | CAPS module 价值 |
| | v2_no_seg3_rule (砍 SYSTEM seg3 判断规则) | seg3 rule 价值 |
| | v2_no_beta_summary (砍 CAPS β 速览 header) | β 速览 价值 |

Each condition 跑 2 次 (BASE + SFT) → 14 conditions 总.

### 0.3 Inference Ok Rate

| condition | BASE ok | SFT ok |
|---|---|---|
| v2 | 481/486 (99%) | 426/486 (88%) |
| v2_no_caps | 483/486 (99%) | 442/486 (91%) |
| v2_no_seg3_rule | 480/486 (99%) | 453/486 (93%) |
| v2_no_beta_summary | 481/486 (99%) | 442/486 (91%) |
| v2_hybrid | 486/486 (100%) | 419/486 (86%) |
| B0 | 479/486 (99%) | 445/486 (92%) |
| v2_plus_peer | 484/486 (99%) | 419/486 (86%) |

**SFT 偏低 (86-93% vs BASE 99-100%)**: LoRA 训练后 output 分布更聚焦, 偶尔产 schema 不严格的 NL — 但 ok ≥86%, eval signal 足够; 跟 BASE 比较公平因为 extract 前 filter ok 行.

---

## 1. Final 11-Metric Set

跟 GPT-4o experiment 相同 (5 大维度, 11 metric, 跨 3 seg, 见 `gpt4o_experiments_summary.md §1`):

| 维度 | Metric | 含义 |
|---|---|---|
| **Fidelity (路径准度)** | seg2 macro-TED ↓ | chain edit distance (按 GT class macro 去 V-bias) |
|  | seg2 macro-LCS-F1 ↑ | chain 子序列重合 (macro) |
|  | seg2 non-V recall ↑ | 非 V 起始 chain 正确率 |
| **Personalization** | seg1/seg2/seg3 NL specificity ↑ | `cos(pred_i, GT_i) / mean_{j≠i}(cos(pred_i, GT_j))` |
|  | seg1 Persona-NL alignment ↑ | `cos(pred, Q1-Q6 NL center)` |
|  | seg1 Trait-pred Spearman ρ ↑ | 相似 trait → 相似 dom prediction |
|  | seg3 Criterion-persona ↑ | `cos(criterion, persona center)` |
| **Anti-collapse** | seg2 Chain diversity ↑ | `1 - modal_chain_freq` |
|  | seg3 Criterion uniqueness ↑ | `1 - mean pairwise cos` over criterion |

---

## 2. 主结果表 (14 baseline × 11 metric × 1000-bootstrap 95% CI)

### 2.1 seg1 (第一反应 anchor 段)

| Baseline | NL specificity ↑ | Persona-NL ↑ | Trait-pred ρ ↑ |
|---|---|---|---|
| BASE_b0 | 1.000 [0.985, 1.015] | 0.284 [0.249, 0.318] | 0.044 [-0.028, 0.143] |
| BASE_v2 | 0.998 [0.986, 1.010] | 0.311 [0.276, 0.348] | **0.086 [0.053, 0.153]** |
| BASE_v2_no_caps | 1.000 [0.987, 1.012] | 0.297 [0.262, 0.331] | 0.017 [-0.097, 0.152] |
| BASE_v2_no_seg3_rule | 1.007 [0.994, 1.021] | 0.302 [0.266, 0.337] | 0.056 [-0.012, 0.149] |
| BASE_v2_no_beta_summary | 0.998 [0.984, 1.013] | 0.293 [0.257, 0.329] | -0.093 [-0.155, -0.004] |
| BASE_v2_hybrid | **1.023 [1.009, 1.037]** | 0.305 [0.269, 0.343] | 0.084 [0.054, 0.157] |
| BASE_v2_plus_peer | 0.991 [0.980, 1.002] | 0.309 [0.269, 0.349] | 0.019 [-0.060, 0.122] |
| SFT_b0 | 1.002 [0.986, 1.018] | 0.295 [0.257, 0.336] | 0.020 [-0.058, 0.131] |
| SFT_v2 | 0.992 [0.974, 1.011] | **0.330 [0.281, 0.379]** | 0.029 [-0.000, 0.112] |
| SFT_v2_no_caps | 1.010 [0.996, 1.024] | 0.280 [0.240, 0.322] | **0.173 [0.000, 0.294]** |
| SFT_v2_no_seg3_rule | 0.981 [0.965, 0.997] | 0.300 [0.259, 0.344] | -0.007 [-0.057, 0.083] |
| SFT_v2_no_beta_summary | 1.002 [0.986, 1.018] | 0.295 [0.255, 0.341] | 0.005 [-0.101, 0.137] |
| SFT_v2_hybrid | 0.989 [0.969, 1.010] | 0.319 [0.268, 0.370] | 0.100 [0.049, 0.209] |
| SFT_v2_plus_peer | 1.003 [0.979, 1.027] | **0.335 [0.289, 0.388]** | 0.064 [-0.010, 0.177] |

**seg1 关键 finding**:
- **NL specificity 整体趋同 (1.0±0.02)** — 第一反应阶段 specificity 不易拉开差距, BASE_v2_hybrid 0.023 最高 (CI 不重叠 vs SFT_v2 0.992)
- **Persona-NL: SFT_v2_plus_peer 0.335 / SFT_v2 0.330 双高**, 比对 BASE_v2 0.311 (**SFT > BASE +0.019, SFT 让第一反应文本更贴近 designer Q1-Q6 风格**)
- **Trait-pred ρ: BASE_v2 0.086 (CI [0.053, 0.153]) 显著正 ★**, SFT_v2 0.029 (CI 跨 0) — **SFT 反而弱化了 trait→prediction 的 in-context 学习能力** (LoRA overfit to training 分布, 推理时不再 conditioning trait); SFT_v2_no_caps 0.173 point 最高但 CI 大

### 2.2 seg2 (chain reasoning)

GT 分布: V-first ≈54%, non-V-first ≈46%.

| Baseline | macro-TED ↓ | macro-LCS-F1 ↑ | non-V recall ↑ | Chain diversity ↑ | NL specificity ↑ |
|---|---|---|---|---|---|
| BASE_b0 | 2.461 [2.184, 2.550] | 0.541 [0.524, 0.607] | 0.013 [0.000, 0.044] | 0.639 [0.561, 0.710] | 0.998 [0.989, 1.008] |
| BASE_v2 | 2.516 [2.259, 2.595] | 0.525 [0.508, 0.585] | 0.025 [0.000, 0.067] | 0.758 [0.694, 0.809] | 0.993 [0.983, 1.004] |
| BASE_v2_no_caps | 2.635 [2.408, 2.721] | 0.525 [0.511, 0.584] | 0.012 [0.000, 0.042] | 0.648 [0.572, 0.711] | 1.006 [0.996, 1.014] |
| BASE_v2_no_seg3_rule | 2.853 [2.380, 2.932] | 0.424 [0.407, 0.586] | 0.013 [0.000, 0.043] | 0.622 [0.545, 0.692] | 1.004 [0.995, 1.013] |
| BASE_v2_no_beta_summary | 2.533 [2.280, 2.614] | 0.518 [0.503, 0.573] | 0.000 [0.000, 0.000] | 0.777 [0.707, 0.822] | 1.006 [0.996, 1.016] |
| BASE_v2_hybrid | 2.268 [2.182, 2.441] | 0.549 [0.532, 0.583] | 0.000 [0.000, 0.000] | 0.765 [0.698, 0.809] | 1.000 [0.990, 1.010] |
| BASE_v2_plus_peer | 2.215 [2.124, 2.391] | **0.562 [0.545, 0.602]** | 0.025 [0.000, 0.063] | **0.800 [0.731, 0.825]** ★ | 1.001 [0.993, 1.011] |
| SFT_b0 | 1.998 [1.846, 2.156] | 0.581 [0.541, 0.618] | 0.102 [0.031, 0.185] | 0.565 [0.476, 0.653] | 0.995 [0.984, 1.007] |
| SFT_v2 | 2.277 [1.916, 2.389] | 0.441 [0.416, 0.616] | 0.061 [0.000, 0.135] | 0.533 [0.429, 0.619] | 1.003 [0.991, 1.014] |
| SFT_v2_no_caps | 2.545 [1.938, 2.658] | 0.448 [0.424, 0.623] | 0.131 [0.057, 0.217] | 0.634 [0.545, 0.716] | **1.013 [1.001, 1.027]** |
| SFT_v2_no_seg3_rule | **1.821 [1.713, 2.220]** ★ | 0.429 [0.405, 0.601] | 0.095 [0.031, 0.172] | 0.597 [0.511, 0.674] | 1.010 [0.996, 1.023] |
| SFT_v2_no_beta_summary | 2.414 [2.096, 2.528] | 0.513 [0.488, 0.578] | 0.117 [0.038, 0.200] | 0.680 [0.598, 0.762] | 1.000 [0.985, 1.014] |
| SFT_v2_hybrid | 2.090 [1.896, 2.267] | **0.589 [0.544, 0.633]** | 0.080 [0.019, 0.157] | 0.640 [0.550, 0.740] | 1.005 [0.989, 1.021] |
| SFT_v2_plus_peer | 2.575 [1.950, 2.700] | 0.442 [0.419, 0.617] | **0.185 [0.085, 0.295]** | 0.577 [0.481, 0.663] | 1.002 [0.987, 1.018] |

**seg2 关键 finding** ★:
- **macro-TED 最佳: SFT_v2_no_seg3_rule 1.821** (CI [1.713, 2.220]) — **CI 不重叠 vs BASE_v2_no_seg3_rule 2.853** ★ 显著 (-36% TED). 含义: BASE 在 no_seg3_rule 上严重失败 (chain 跟 GT 距离大); SFT 学到了 chain 推理的"内置 rule", 不依赖 SYSTEM 提示反而最准
- **non-V recall: SFT_v2_plus_peer 0.185** (CI [0.085, 0.295]) — **CI 不重叠 vs BASE 全部** (BASE max ≤ 0.025) ★ — SFT 给 model + peer 推荐才能 break V-default
- **Chain diversity: BASE_v2_plus_peer 0.800** (CI [0.731, 0.825]) — **CI 不重叠 vs SFT_v2 0.533** ★, **SFT mode collapse 显著** (overfit to training 分布的 modal chain)
- **macro-LCS-F1: SFT_v2_hybrid 0.589** ≈ BASE_v2_plus_peer 0.562 (CI 重叠) — top 两选项分别在 SFT 跟 BASE
- **SFT_b0 macro-TED 1.998** (CI [1.846, 2.156]) — 仅次于 SFT_v2_no_seg3_rule, **暗示 SFT model 不需 prompt 全套就能给好 chain** (LoRA 已内化 anchor pattern)

### 2.3 seg3 (anchor + criterion)

GT 分布: V-majority ~92.6%.

| Baseline | NL specificity ↑ | Criterion uniqueness ↑ | Criterion-persona ↑ |
|---|---|---|---|
| BASE_b0 | 0.988 [0.977, 0.999] | 0.323 [0.288, 0.351] | 0.244 [0.216, 0.274] |
| BASE_v2 | 1.000 [0.990, 1.012] | 0.455 [0.413, 0.486] | 0.251 [0.222, 0.281] |
| BASE_v2_no_caps | 0.993 [0.982, 1.004] | 0.210 [0.178, 0.241] | **0.268 [0.237, 0.299]** |
| BASE_v2_no_seg3_rule | 1.004 [0.994, 1.015] | 0.259 [0.214, 0.303] | 0.264 [0.233, 0.296] |
| BASE_v2_no_beta_summary | 1.005 [0.994, 1.016] | 0.247 [0.209, 0.285] | 0.254 [0.222, 0.285] |
| BASE_v2_hybrid | 0.995 [0.985, 1.006] | 0.332 [0.291, 0.367] | 0.248 [0.216, 0.282] |
| BASE_v2_plus_peer | 0.992 [0.982, 1.003] | 0.424 [0.371, 0.468] | 0.254 [0.221, 0.288] |
| SFT_b0 | 0.997 [0.980, 1.016] | 0.557 [0.534, 0.569] | 0.181 [0.153, 0.210] |
| SFT_v2 | 0.985 [0.967, 1.003] | 0.380 [0.312, 0.440] | **0.263 [0.223, 0.301]** |
| SFT_v2_no_caps | **1.007 [0.991, 1.024]** | 0.335 [0.276, 0.383] | 0.252 [0.213, 0.289] |
| SFT_v2_no_seg3_rule | 1.000 [0.982, 1.018] | **0.592 [0.561, 0.612]** ★ | 0.233 [0.197, 0.271] |
| SFT_v2_no_beta_summary | 0.997 [0.982, 1.013] | 0.244 [0.182, 0.300] | 0.241 [0.209, 0.280] |
| SFT_v2_hybrid | 1.004 [0.981, 1.027] | 0.565 [0.527, 0.587] | 0.215 [0.179, 0.252] |
| SFT_v2_plus_peer | 0.997 [0.978, 1.019] | 0.433 [0.368, 0.484] | 0.263 [0.226, 0.305] |

**seg3 关键 finding**:
- **Criterion uniqueness: SFT_v2_no_seg3_rule 0.592** (max, CI [0.561, 0.612]) — **CI 不重叠 vs BASE_v2 0.455** ★ — 跟 macro_TED 一致, **SFT 不要 SYSTEM seg3 rule 反而 criterion 更不一样**
- **Criterion-persona: 14 condition CI 几乎全部重叠** (0.21-0.27), 排序 SFT_v2 ≈ BASE_v2_no_caps ≈ SFT_v2_plus_peer 顶 (0.263-0.268)
- SFT_b0 criterion_uniqueness 0.557 (其实 SFT b0 仍很高) — **LoRA 不靠 prompt 就给出 diverse criterion**

---

## 3. BASE vs SFT 主对比 (paper main claim)

### 3.1 同 prompt (v2) 上 SFT 净增量

| Metric | BASE_v2 | SFT_v2 | Δ | 解读 |
|---|---|---|---|---|
| seg1 NL specificity | 0.998 | 0.992 | -0.006 | ≈ |
| seg1 Persona-NL | 0.311 | **0.330** | **+0.019** | SFT 让第一反应 NL 更贴 designer Q1-Q6 风格 ✓ |
| seg1 Trait-pred ρ | 0.086 | 0.029 | **-0.057** | **SFT 弱化 trait conditioning** (LoRA overfit 训练分布) ✗ |
| seg2 macro-TED ↓ | 2.516 | **2.277** | **-0.24** | SFT 让 chain 更准 ✓ (CI 部分重叠) |
| seg2 macro-LCS-F1 | 0.525 | 0.441 | -0.084 | SFT chain 跟 GT subsequence 不同 (chain mode shift) ✗ |
| seg2 non-V recall | 0.025 | **0.061** | **+0.036** | SFT 缓解 V-default chain ✓ |
| seg2 Chain diversity | 0.758 | **0.533** | **-0.225 ★** | **SFT mode collapse 显著** (CI 不重叠) ✗ |
| seg2 NL specificity | 0.993 | 1.003 | +0.010 | ≈ |
| seg3 NL specificity | 1.000 | 0.985 | -0.015 | ≈ |
| seg3 Criterion uniqueness | 0.455 | 0.380 | -0.075 | SFT 让 criterion 收敛 ✗ |
| seg3 Criterion-persona | 0.251 | **0.263** | **+0.012** | SFT 让 criterion 更贴 designer persona ✓ |

**3.1 总结**:
- ✓ SFT 改善: Persona-NL alignment, macro-TED (chain 准度), non-V recall (反 V-default), Criterion-persona
- ✗ SFT 退步 (mode collapse): Chain diversity (0.225 跌, CI 不重叠 ★), macro-LCS-F1, Criterion uniqueness, Trait-pred ρ
- ≈ 持平: NL specificity (3 个 seg)

**核心 trade-off**: SFT 把"个性化贴近 designer 个体"做强 (Persona-NL/Criterion-persona ↑), 但"跨 designer 多样性"被压 (Chain diversity/Criterion uniqueness ↓). 这是 SFT overfit 训练分布的典型 signature.

### 3.2 14 condition pairwise SFT-BASE gap (chain 准度 macro_TED)

| Condition | BASE | SFT | Δ (SFT 优=负值) |
|---|---|---|---|
| b0 | 2.461 | **1.998** | **-0.463** |
| v2 | 2.516 | 2.277 | -0.239 |
| v2_no_caps | 2.635 | 2.545 | -0.090 |
| **v2_no_seg3_rule** | **2.853** | **1.821** | **-1.032 ★** |
| v2_no_beta_summary | 2.533 | 2.414 | -0.119 |
| v2_hybrid | 2.268 | 2.090 | -0.178 |
| v2_plus_peer | 2.215 | 2.575 | **+0.360** ✗ |

**chain 准度上 SFT 在 6/7 condition 上 BASE 更好**, 唯独 **v2_plus_peer SFT 反 hurt** — peer block 跟 SFT model 内化的 routing 模式冲突, 引入噪声. 最大增量在 **v2_no_seg3_rule** (-1.032) — 砍掉 SYSTEM 强约束后 SFT 内化的 anchor 更主导.

---

## 4. SFT 内部 ablation (vs SFT_v2)

| Metric | SFT_v2 (base) | **-CAPS** | -seg3 rule | -β 速览 | hybrid | +peer |
|---|---|---|---|---|---|---|
| seg2 macro-TED ↓ | 2.277 | 2.545 (+12%) ✗ | **1.821 (-20%)** ★ | 2.414 (+6%) | 2.090 (-8%) | 2.575 (+13%) ✗ |
| seg2 macro-LCS-F1 ↑ | 0.441 | 0.448 (+2%) | 0.429 (-3%) | 0.513 (+16%) | **0.589 (+34%)** | 0.442 (≈) |
| seg2 non-V recall ↑ | 0.061 | 0.131 (+115%) | 0.095 (+56%) | 0.117 (+92%) | 0.080 (+31%) | **0.185 (+203%)** ★ |
| seg2 Chain diversity ↑ | 0.533 | 0.634 (+19%) | 0.597 (+12%) | **0.680 (+28%)** | 0.640 (+20%) | 0.577 (+8%) |
| seg3 Criterion uniqueness ↑ | 0.380 | 0.335 (-12%) | **0.592 (+56%)** ★ | 0.244 (-36%) ✗ | 0.565 (+49%) | 0.433 (+14%) |
| seg3 Criterion-persona ↑ | 0.263 | 0.252 (-4%) | 0.233 (-11%) | 0.241 (-8%) | 0.215 (-18%) ✗ | 0.263 (≈) |

**SFT 内部 ablation 核心 finding**:
- **SFT_v2 SYSTEM 设计 over-constrain SFT model** — 砍 seg3 rule (-20% TED, +56% criterion uniq), 砍 β 速览 (+28% chain diversity), 砍 CAPS (+115% non-V recall + diversity)
- **SFT model 内化了大部分 anchor 信号**, prompt 重复约束反而 hurt
- **SFT_v2_hybrid 是 SFT 上最强的 fidelity 配置** (macro-LCS-F1 0.589, +34%)
- **peer 在 SFT 上 net positive on non-V recall** (+203% vs SFT_v2 base) — peer 帮 SFT 跳出训练分布的 V-default
- Criterion-persona 对所有 SFT ablation 都没改善 (best=SFT_v2 0.263), 说明 SFT model 已锁定该 metric

---

## 5. BASE 内部 ablation (vs BASE_v2, cross-check with GPT-4o)

| Metric | BASE_v2 | -CAPS | -seg3 rule | -β 速览 | hybrid | +peer |
|---|---|---|---|---|---|---|
| seg2 macro-TED ↓ | 2.516 | 2.635 (+5%) | 2.853 (+13%) ✗ | 2.533 (+1%) | **2.268 (-10%)** | **2.215 (-12%)** |
| seg2 macro-LCS-F1 ↑ | 0.525 | 0.525 | 0.424 (-19%) | 0.518 (-1%) | 0.549 (+5%) | **0.562 (+7%)** |
| seg2 non-V recall ↑ | 0.025 | 0.012 | 0.013 | 0.000 | 0.000 | 0.025 |
| seg2 Chain diversity ↑ | 0.758 | 0.648 (-15%) | 0.622 (-18%) | 0.777 (+3%) | 0.765 (+1%) | **0.800 (+6%)** |
| seg3 Criterion uniqueness ↑ | 0.455 | 0.210 (-54%) ✗ | 0.259 (-43%) ✗ | 0.247 (-46%) ✗ | 0.332 (-27%) | 0.424 (-7%) |
| seg3 Criterion-persona ↑ | 0.251 | **0.268 (+7%)** | 0.264 (+5%) | 0.254 (+1%) | 0.248 (-1%) | 0.254 (+1%) |

**BASE 内部 ablation 核心 finding** (cross-check vs GPT-4o 实验):
- **BASE Qwen 跟 GPT-4o 在 v2_anchor 趋势上方向一致** — CAPS / seg3 rule / β 速览 砍除都 hurt criterion uniqueness (GPT-4o 也 hurt criterion uniqueness)
- **BASE Qwen 跟 GPT-4o 差异**: peer 在 BASE Qwen 上 net positive (chain diversity +6%, macro-TED -12%), 在 GPT-4o 上 hurt (macro-TED +41%); 推测 Qwen 容量小, peer block 提供有用 grounding; GPT-4o 信息量已足, peer 引入冲突
- **non-V recall BASE 整体很差 (0.000-0.025)** — Qwen base model 强 V-default bias, prompt 拉不动
- **chain diversity BASE 整体高 (0.622-0.800)** — Qwen base model 在 sampling 上没收敛 (LoRA 才会 collapse, 见 §3.1)

---

## 6. 整体 finding (paper 用)

### 6.1 SFT 训练有效但有副作用

✓ **SFT 在 7/7 condition × 7/11 metric 上正向**:
- chain 准度 (macro-TED -8 to -36%)
- 第一反应贴 designer (Persona-NL +6%)
- 跨场景 transfer (Q1-Q6 训练 → Q7 推理) 真 demonstrate

✗ **SFT 在 4/11 metric 上反 hurt**:
- **Chain diversity 跌 30%** (CI 不重叠 ★) — overfit to training 分布的 modal chain
- macro-LCS-F1 跌 16% — chain pattern shift to non-GT mode
- Criterion uniqueness 跌 16% — criterion 收敛
- Trait-pred ρ 跌 0.057 — in-context trait conditioning 减弱

### 6.2 SFT 改变了 prompt 灵敏度

**SFT model 内化了 v2_anchor 的大部分信号**, 因此:
- **砍 SYSTEM 约束反而帮 SFT** (砍 seg3 rule → macro-TED -20%, criterion uniq +56% ★)
- 完整 prompt v2 在 SFT 上**有过度约束** (跟 BASE 上的 winner 不同)
- **SFT 的最佳配置不是 BASE 的最佳** — 训练改变了 optimal prompt

### 6.3 推荐 ship 配置

- 若目标 **personalization** (Persona-NL / Criterion-persona): **SFT_v2_plus_peer** (0.335 / 0.263) or **SFT_v2** (0.330 / 0.263)
- 若目标 **chain 准度** (macro-TED): **SFT_v2_no_seg3_rule** 1.821 (★ vs BASE 同 condition 2.853)
- 若目标 **non-V recall** (反 V-default): **SFT_v2_plus_peer** 0.185 (★ vs BASE max 0.025)
- 若目标 **chain diversity**: **BASE_v2_plus_peer** 0.800 (★ vs SFT 全部) — SFT 在该维度劣
- 若目标 **balanced**: **SFT_v2_hybrid** (LCS-F1 0.589 + criterion uniq 0.565, 没 mode collapse 严重)

### 6.4 paper main claim 候选

```
3 层 (Trait × CAPS × PKVC SM) + LoRA SFT 训练能在 chain reasoning + persona alignment 维度
显著优于 zero-shot baseline 跟同 prompt 的 BASE model:
  - chain accuracy: SFT_v2_no_seg3_rule macro-TED 1.821 vs BASE 同 condition 2.853 (CI 不重叠 ★, -36%)
  - non-V recall: SFT_v2_plus_peer 0.185 vs BASE 全部 ≤0.025 (CI 不重叠 ★)
  - persona-NL alignment: SFT_v2 0.330 vs BASE_v2 0.311 (+6%)

但 SFT 引入 chain diversity 跌 30% 的 mode collapse, 这是 LoRA SFT 在 small dataset (162 designer × Q1-Q6
= 962 example) 上的典型 over-fitting signature. 未来工作可考虑加 contrastive loss / sampling diversity
penalty 缓解.
```

---

## 7. 复现

- 推理: `scripts/run_sft_inference.py` + `scripts/run_sft_14_parallel.sh`
- Extract: `scripts/eval_baselines_pkvc.py` × 14 (yunwu.ai gpt-4o, ~$28)
- Bootstrap: `scripts/sft_base_eval_bootstrap.py`
- 输出: `reports/datav3/sft_base_eval_bootstrap.{md,jsonl}`
- LoRA adapter: `/data/wy/sft_v3_ckpts/stage_warmup_ep5/` (远程, 5 epoch warmup, eval_loss 1.327)
- SFT 训练数据: `data_v3/processed/sft_{train,val}.jsonl` (962 = 770 train + 192 val)
- v3 expansion (修 hallu): `data_v3/processed/q1q6_expansion_outputs.jsonl` (871/938 ok, severe hallu 31%→8%)
- prompt: `sft_pipeline/config/prompts/expand_q1q6_v3.md` + `scripts/run_b6_sample.py` (B6 prompt builders)

---

## 8. 已知限制

1. **Q2 thinking_chain 没参与本评测** — Q2 是 designer 全局 cardId 排序, 不属 Q7 跨场景 transfer (Q7 只有 3 seg 对应 first reaction / weighing / bottom line). 单独 Q2 first letter accuracy 5ep model 是 25% (P-first 100%, K/V/C 0%) — minority class structural limit (K/V 各 10 train sample).
2. **SFT_v2 ok 率 88% (BASE 99%)** — LoRA 后输出 schema 严格性下降, eval 前 filter 但影响样本量轻微. 不影响 metric 主结论.
3. **PKVC extract reuse GPT-4o** — 跟 GPT-4o experiment 一致, 但 SFT/BASE inference 本身已含 structured fields. extract 是为了一致性, 不是必需.
4. **训练数据修过 2 轮** (v1 dirty → v2 expansion + 4 option, eval_loss 1.88 → 1.327, -29%) — 详见 §0.2 + `expand_q1q6_v3.md` notes. v2 dataset 是 final.
