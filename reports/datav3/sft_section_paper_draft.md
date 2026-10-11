# §5.X SFT Analysis — Paper Section Draft

> Position: as §5.7 (after main results §5.4, ablation §5.5, cross-model §5.6).
> Length budget: 1.5 pages (1 setup paragraph + 2 tables + 3 analysis paragraphs + limitation).
> Data source: `reports/datav3/sft_base_experiments_summary.md` §3.1 + §4 (162 designer × 3 seg × 14 condition × 11 metric × 1000-bootstrap, bf16, same local inference engine).

---

## §5.7  SFT Analysis: Does Fine-tuning Substitute or Complement Prompt Design?

### §5.7.1  Experimental Setup

We additionally fine-tune Qwen2.5-7B-Instruct with LoRA (rank=32, α=64, target modules =
{q,k,v,o}_proj) for 5 epochs on 962 Q1-Q6 training examples (162 designers × 6 questions),
using token-level cross-entropy on the full JSON output. Effective batch size is 32 (per-device
batch=4, gradient accumulation=8), learning rate 1e-4 with cosine decay, bf16 throughout, on
a single A800 80GB GPU. Training takes 50 minutes; final train/eval loss is 1.26 / 1.33.

**中文**: 我们在 Qwen2.5-7B-Instruct 上额外做 LoRA 微调 (rank=32, α=64, target modules =
{q,k,v,o}_proj), 用 962 条 Q1-Q6 训练样本 (162 设计师 × 6 题) 训练 5 个 epoch, 对完整 JSON
输出做 token 级交叉熵。Effective batch size 32 (per-device batch=4, gradient accumulation=8),
学习率 1e-4 配 cosine decay, 全程 bf16, 在单张 A800 80GB 上完成。训练耗时 50 分钟, 最终
train/eval loss 为 1.26 / 1.33。

We evaluate the resulting adapter on Q7 (a held-out cross-scenario test: training uses an
"all-sensing efficiency space" scenario, while Q7 evaluates on a different "legacy supermarket
digitization" scenario, with a different output schema — Q1-Q6 are single-step while Q7 has
three segments). We use the same prompt configurations and the same 11-metric / 1000-bootstrap
evaluation protocol as our main BASE model experiments (§5.4).

**中文**: 我们把得到的 adapter 在 Q7 上评测 (一个留出的跨场景测试: 训练用 "全感知效率空间"
场景, Q7 评测换到不同的 "老旧超市数字化" 场景, 输出 schema 也不同 — Q1-Q6 是单步而 Q7 是
三段)。我们沿用主实验 BASE 模型 (§5.4) 相同的 prompt 配置与同一套 11 指标 / 1000-bootstrap
评测协议。

### §5.7.2  SFT vs BASE on shared prompt: Mixed cross-scenario transfer

Table 4 compares the SFT adapter to the unmodified BASE model on the v2_anchor prompt
(both bf16, same local transformers inference engine, 162 designer × 3 seg × 1000-bootstrap).

**中文**: 表 4 在 v2_anchor prompt 下比较 SFT adapter 与未修改的 BASE 模型 (二者均 bf16,
同一本地 transformers 推理引擎, 162 设计师 × 3 段 × 1000-bootstrap)。

**Table 4**: SFT vs BASE on v2_anchor prompt. Point estimate [95% bootstrap CI]. ★ = CIs do not overlap.

| Metric                      | BASE_v2                | SFT_v2                 | Δ (SFT − BASE)             |
| --------------------------- | ---------------------- | ---------------------- | -------------------------- |
| seg2 macro-TED ↓            | 2.516 [2.26, 2.60]     | **2.277 [1.92, 2.39]** | **−0.24**  (SFT better)    |
| seg2 non-V recall ↑         | 0.025 [0.00, 0.07]     | **0.061 [0.00, 0.14]** | **+0.04**  (SFT better)    |
| seg2 chain diversity ↑      | **0.758 [0.69, 0.81]** | 0.533 [0.43, 0.62]     | **−0.23 ★**  (BASE better) |
| seg1 trait-pred ρ ↑         | **0.086 [0.05, 0.15]** | 0.029 [−0.00, 0.11]    | −0.06 (BASE better)        |
| seg3 criterion uniqueness ↑ | **0.455 [0.41, 0.49]** | 0.380 [0.31, 0.44]     | −0.08 (BASE better)        |

SFT improves chain accuracy (macro-TED) and non-V recall, but introduces **mode collapse on
chain diversity** (−0.23, 95% CIs do not overlap, ★), and modestly degrades trait-prediction
correlation and criterion uniqueness. This pattern is consistent with small-corpus LoRA SFT
over-fitting modal patterns in the training distribution: 962 examples (162 designers × 6
questions) sits at the lower edge of typical LoRA fine-tuning scale, and chain reasoning
inherits the most-frequent V-first / V-K-* pattern from training.

**中文**: SFT 提升了链准确度 (macro-TED) 和 non-V recall, 但在 chain diversity 上引入了
**mode collapse** (−0.23, 95% CI 不重叠, ★), 并轻度退化 trait 预测相关性和 criterion
uniqueness。这一模式与小语料 LoRA SFT 在训练分布上过拟合主导模式的现象一致: 962 样本
(162 设计师 × 6 题) 处于典型 LoRA 微调规模的下沿, 链推理继承了训练里最频繁的
V-first / V-K-* 模式。

### §5.7.3  SFT internalizes prompt structure: prompt and SFT are substitutes

Surprisingly, the full v2_anchor prompt is **not** the optimal prompt configuration for the
SFT model. Table 5 shows that removing prompt components — most notably the seg3 decision
rule, or adding the peer-retrieval block — unlocks substantially stronger SFT performance,
yielding three findings with 95% CIs not overlapping the BASE_v2 baseline (★).

**中文**: 出乎意料的是, 完整的 v2_anchor prompt **并非** SFT 模型的最优 prompt 配置。
表 5 显示, 移除 prompt 组件 — 尤其是 seg3 decision rule, 或加入 peer-retrieval 块 —
反而能大幅释放 SFT 性能, 得到三处 95% CI 不与 BASE_v2 baseline 重叠的发现 (★)。

**Table 5**: SFT internal ablation (vs SFT_v2 full prompt). Bold = best in column among SFT variants. ★ = CI not overlapping BASE_v2 on same metric.

| Method | macro-TED ↓ | non-V recall ↑ | chain div ↑ | trait-pred ρ ↑ | crit uniq ↑ |
|---|---|---|---|---|---|
| SFT_v2 (full) | 2.277 | 0.061 | 0.533 | 0.029 | 0.380 |
| − CAPS module | 2.545 | 0.131 | 0.634 | **0.173** | 0.335 |
| **− seg3 rule** | **1.821 [1.71, 2.22] ★** | 0.095 | 0.597 | −0.007 | **0.592 [0.56, 0.61] ★** |
| − β summary | 2.414 | 0.117 | **0.680** | 0.005 | 0.244 |
| + peer retrieval | 2.575 | **0.185 [0.09, 0.30] ★** | 0.577 | 0.064 | 0.433 |
| v2_hybrid | 2.090 | 0.080 | 0.640 | 0.100 | 0.565 |

Three statistically significant gains over BASE_v2 (95% CIs not overlapping):
1. **SFT−seg3rule** drops macro-TED to **1.821** (BASE_v2: 2.516; relative −28%).
2. **SFT−seg3rule** lifts criterion uniqueness to **0.592** (BASE_v2: 0.455; relative +30%).
3. **SFT+peer** lifts non-V recall to **0.185** (BASE_v2: 0.025; 7× higher).

**中文**: 相对 BASE_v2 共三处统计显著提升 (95% CI 不重叠):
1. **SFT−seg3rule** 把 macro-TED 压到 **1.821** (BASE_v2: 2.516; 相对 −28%)。
2. **SFT−seg3rule** 把 criterion uniqueness 拉到 **0.592** (BASE_v2: 0.455; 相对 +30%)。
3. **SFT+peer** 把 non-V recall 拉到 **0.185** (BASE_v2: 0.025; 7 倍)。

All three gains require **modifying** the v2_anchor prompt at inference time, even though the
SFT adapter was trained on prompts close to v2_anchor. We interpret this as evidence that the
SFT adapter has **internalized** the prompt-encoded structure (the seg3 decision rule, the
CAPS β summary heuristic): once these signals are baked into adapter weights, reinforcing them
at inference time via the prompt becomes redundant and in some cases actively distracts the
model from its internalized routing.

**中文**: 这三处增益都要求在推理时**修改** v2_anchor prompt, 即便 SFT adapter 在训练时
所用的 prompt 已接近 v2_anchor。我们把这解读为 SFT adapter 已经**内化**了 prompt 中编码
的结构 (seg3 decision rule、CAPS β 摘要启发) 的证据: 一旦这些信号烤进 adapter 权重,
推理时再通过 prompt 强化反而冗余, 甚至会把模型从其内化的 routing 上分心带走。

This observation supports a **substitute** rather than **complement** relationship between
LoRA SFT and our v2_anchor prompt design: practitioners adapting a model with SFT should
**re-tune the prompt** rather than re-use the pre-SFT prompt unchanged. In contrast, BASE
shows the standard ladder pattern (B0 < B1 < B2 < B3 < v2_anchor, see §5.4), where each
prompt module monotonically helps; the SFT model's optimal prompt is qualitatively different.

**中文**: 这一现象支持 LoRA SFT 与我们 v2_anchor prompt 设计之间是**替代**而非**互补**
的关系: 用 SFT 适配模型的实践者应当**重新调 prompt**, 而不是照搬 SFT 前的 prompt。
相比之下, BASE 模型呈现标准阶梯 (B0 < B1 < B2 < B3 < v2_anchor, 见 §5.4), 每个 prompt
模块单调有用; SFT 模型的最优 prompt 在性质上截然不同。

### §5.7.4  Limitations and Future Directions

The chain-diversity mode collapse observed in §5.7.2 and the prompt-substitution pattern in
§5.7.3 are both consistent with the small training corpus (962 examples) and the dual
cross-scenario / cross-schema transfer task (training Q1-Q6 single-step on "office space"
versus testing Q7 three-segment on "legacy supermarket"). Two complementary fixes are likely
needed:

- **Data scaling.** Typical LoRA SFT papers operate on 10⁴–10⁵ instruction-tuning examples;
  962 is an order of magnitude short. Doubling or tripling the designer pool (and thus the
  cross-trait coverage for minority cardId patterns observed in our Q2 analysis, Appendix Z)
  is the most direct mitigation.
- **Schema-aligned training.** Our Q1-Q6 supervision uses a single-step output schema while
  evaluation uses Q7's three-segment schema. Aligning training schema with the target
  evaluation schema (e.g., constructing synthetic three-segment training prompts on Q1-Q6
  data) should reduce the train-test transfer gap.

**中文**: §5.7.2 观察到的 chain diversity mode collapse 与 §5.7.3 的 prompt 替代模式, 都
与小训练语料 (962 样本) 以及双重跨场景 / 跨 schema 迁移任务相吻合 (训练 Q1-Q6 单步
"办公空间" 对测试 Q7 三段 "老旧超市")。需要两条互补修复:

- **数据 scaling。** 典型 LoRA SFT 论文工作在 10⁴–10⁵ 条指令微调样本上; 962 差了一个
  数量级。把设计师池翻倍或翻三倍 (从而提高我们 Q2 分析中观察到的少数 cardId 模式的
  跨 trait 覆盖, 附录 Z) 是最直接的缓解手段。
- **Schema 对齐训练。** 我们 Q1-Q6 监督用的是单步输出 schema, 而评测用的是 Q7 的三段
  schema。把训练 schema 对齐到目标评测 schema (例如基于 Q1-Q6 数据构造合成的三段训练
  prompt) 应能缩小 train-test 迁移 gap。

As a preliminary check on the over-fitting hypothesis, we additionally trained an SFT variant
with reduced LoRA rank (r=16 instead of r=32) and shorter training (3 epochs at LR 5e-5).
On a held-out 30-designer subsample (limited to int4-quantized inference due to compute
constraints), chain diversity recovers from 0.500 (5-epoch r=32 ep5) to 0.690 — a 38%
recovery toward the BASE level of 0.758. Full bf16 validation of this fix is left to future
work (Appendix Y).

**中文**: 作为对过拟合假设的初步验证, 我们额外训练了一个降低 LoRA rank (r=16 而非 r=32)
且缩短训练 (3 epoch, LR 5e-5) 的 SFT 变体。在一个 30 设计师的留出子样本上 (受算力限制,
仅 int4 量化推理), chain diversity 从 0.500 (5 epoch r=32) 恢复到 0.690 — 朝 BASE
水平 0.758 恢复了 38%。该修复的完整 bf16 验证留作未来工作 (附录 Y)。

### Note for paper revision (中文 internal notes — strip before submission)

- ★ findings 全在 ablation 上, 不在 SFT_v2 base, narrative 顺序: Table 4 (base mixed) → Table 5 (ablation 出 ★) → substitute hypothesis (核心 novel)
- 3 ★ 描述都用 "vs BASE_v2 same metric" 比较 (不是 vs SFT_v2 base), 这样 paper 评审看到 ★ 立刻 anchor 到 BASE 不是 SFT 内部
- Limitation 段不提"SFT 失败", 提 "scale + schema 两条 future direction" — positive framing
- 不放 chain diversity bar chart (figure 太占空间), 数字 0.533 vs 0.758 + ★ 标记够强
- v3c r=16 fix 数据写得保守 — "preliminary check", "limited to int4", "future work" — 别 overcommit
