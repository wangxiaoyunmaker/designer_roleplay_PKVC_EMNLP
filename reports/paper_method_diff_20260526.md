# 论文 method 章节 ↔ 真实代码/数据 diff 报告

**对照**: `论文/全文.md` (2026-05-26 06:18) ↔ 推理代码 `sft_pipeline/infer_v2.py` + `scripts/run_b6_sample.py` + `data/processed/caps_quadruples.json` ↔ `reports/datav3/` 全套 (5-26).

**不直接改文件, 只列差异 + 改写建议.**

---

## 0. TL;DR — 必须改的 critical 项 (7 条)

| # | 位置 | 严重度 | 一句话 |
|---|---|---|---|
| C1 | §3.2 inline | **blocker** | 还有 placeholder `(JIDAN GAIed)` 没删 |
| C2 | §3.3 Eq.2 | **blocker** | LaTeX 下标全部缺 `_`: `\mathcal{O}{trip}` 应为 `\mathcal{O}_{trip}`, `\mathcal{C}{type}` 应为 `\mathcal{C}_{type}`, `e{target}` 应为 `e_{target}` — 4 处, 编译会乱 |
| C3 | §3.2/§3.3 vs §4.3 | **high** | §3 自称 "Dynamic Ontological **Triplets**", §4 又叫 "ontological **quad**-relations". 公式确实是 4 元组, 应统一为 quad |
| C4 | §3.3 末 | **high** | "triggers the following ontological rule:" — **该例子整段缺失**, 应补一个具体 quad (e.g. `Q1_D__数据vs直觉` β=+0.537 → strongly_forces V) |
| C5 | §5.1 vs §3/abstract | **high** | N **162 vs 163** 内部不一致: abstract/§3 用 163, §5.1 用 162. 实际 evaluation set = 162 (1 designer Q7 gold 缺) |
| C6 | §5.3 整个 baseline 描述 | **critical / 论文最核心错误** | 论文写的 B1-B4 ladder 跟实际跑的 B0_h-B4_h ladder 完全错配, 详 §3.4 |
| C7 | §5.4 Experimental Results | **blocker** | 整个 section 空白, 数字一行没填 |

---

## 1. 内部一致性 (paper 自相矛盾)

### 1.1 N (designer 数量)

| 出现位置 | 论文写的 | 真实 |
|---|---|---|
| Abstract | 163 | 163 ✓ |
| §1 contributions | 163 | 163 ✓ |
| §3.2 (latent trait derivation) | 没显式数字 | — |
| §4.1 dual-component tracking | 招募 200 → 筛 163 | ✓ (matches CLAUDE.md §7) |
| **§5.1 evaluation benchmark** | **162 designers / 486 instances** | cross_model 实跑 162 × 3 = 486 ✓ |

**真实情况** (`SUMMARY.md` + `cross_model_ladder_ablation_results.md` §0.1): N=163 (data_v3), 评测时 1 个 designer 因 Q7 gold NL 缺失/extract 失败被剔, 实跑 162.

**改写建议**:
- §5.1 句末加一句: "Of the 163 designers in our cohort, 162 had complete Q7 gold-standard extractions and were included in the evaluation set (1 dropped due to missing PKVC labels)."
- 或者全文统一 162 (放弃 abstract/§3 的 163)

### 1.2 "Triplets" vs "Quads"

- §3.2 标题段没用词, §3.3 第二段 "**Dynamic Ontological Triplets**" + 公式 `(𝒞_type, 𝐭_sub, p, e_target)` ← 数一下是 4 元
- §4 副标题 "Ontological Runtime Engine", §4.3 标题 "Behavioral Ontology **Quadruples**" + "36 ontology **quad-relations**"
- 真实代码 (`rag/quad_schema.py`, `caps_quadruples.json`): **quad** = (C, S, P, O), 4 字段, **每个 entry 36 条** ✓

**改写**: §3.3 把 "Dynamic Ontological Triplets" 改成 "Dynamic Ontological **Quadruples**", 同段 "(C_type, t_sub, p, e_target)" 改注释为 "a 4-element quadruple where..." 跟 §4 命名统一。

---

## 2. §3 Cognitive Agent Framework — framing 一致性

| 论文 claim | 真实 | 状态 |
|---|---|---|
| CAPS 是理论, 30 维量表 measure CAPS network weights | ✓ matches `docs/notes/framing_correction_20260522.md` | ✓ |
| PKVC = P/K/V/C 4 state runtime, 不是 trait scale 不是 behavior | ✓ matches v3.7 §2 | ✓ |
| Trait Layer = 30 维 → conditional parameter | 实际是 30 维 → EFA → **4 个 continuous factor scores** (F1-F4), 直接 conditioning | △ paper 没说降维到 4 |
| `\mathbf{t} \in \mathbb{R}^{30}` (Eq.1 上下文) | 实际 routing 用的是 **F1-F4 factor scores + Likert polarity**, 不是 raw 30-d | △ Eq.1 在 paper 是 high-level, 不冲突, 但 §3.2 末尾 "yields a 30-dimensional trait vector t ∈ ℝ^30" 跟 §4.4 "filter the ontology against the agent's 30-dimensional Likert profile" 一致, 但跟 prompt 实际还多了 4-d factor scores via peer retrieval (only used in +peer ablation) |

**改写**: §3.2 末尾可以加一句 "These 30 dimensions are further compressed via exploratory factor analysis into a small number of continuous factor scores (§4.2) used for cluster-free peer retrieval and prompt-level activation; the raw 30-d Likert profile is retained for direct ontological matching." (可选, 不补也成立)

### 2.1 §3.2 严重问题 (C1)

```
For some designers, traits such as Local Cultural Adaptation may instead become
activated, inhibiting abstract universal frameworks and reinforcing situated
cultural specificity, contextual interpretation, and locally grounded design
logic. (JIDAN GAIed).
```

**`(JIDAN GAIed)` 是编辑 marker, 必须删**。建议整个粗体段落收紧为 1-2 句, 因为它跟前后逻辑重复. 当前该段过长 (185 词), 信息密度低.

### 2.2 §3 cognitive 4-stage 列表断尾

```
each state $e_i$ dynamically transitions within a cognitive 4-stage:

Grounded in these definitions, ...
```

冒号后**没有 4-stage 的枚举**. 紧接着跳 "Grounded in these definitions" 开始新段。

**改写**: 冒号后补 4 行/4 项:
- **P** (Problem framing): redefining or reframing the dilemma;
- **K** (Knowledge retrieval): invoking past experience, data, or technical references;
- **V** (Value judgment): asserting moral/aesthetic principles or bottom lines;
- **C** (Concept generation): integrating partial insights into a synthesizing image or framework.

### 2.3 §3.3 公式问题 (C2)

LaTeX 原文 (line ~182):
```
\mathcal{O}{trip} = (\mathcal{C}{type}, \mathbf{t}{sub}, p, e{target})
```

应为:
```
\mathcal{O}_{trip} = (\mathcal{C}_{type}, \mathbf{t}_{sub}, p, e_{target})
```

(4 个 `_` 缺失) 当前会编译出 `O trip` (后跟空白上下标), 字面渲染乱。

### 2.4 §3.3 example 断尾 (C4)

```
For instance, when a situational context 𝒞 introduces a profound ethical risk,
an agent parameterized with a dominant axiological trait ($\mathbf{t}_{ethics} \gg 0$)
triggers the following ontological rule:

This rule mathematically parameterizes the context execution layer, ...
```

**rule 整段缺**. 建议补一个真实数据上的具体 quad, e.g.:
```
\[ \mathcal{O}_{trip} = \big( \texttt{Q1\_D}, \langle \text{数据 vs 直觉},\, + \rangle,\, \texttt{strongly\_forces\_transition\_to},\, V \big) \]
with empirical activation $\beta=+0.537$ ($p=0.040$, $n=14/163$).
```
(这条 quad 来自 `caps_quadruples.json` 第 4 条, 是数据上最强的 strongly_forces 之一)

---

## 3. §4 Empirical Trait-to-State Alignment — 数据 sanity check

### 3.1 §4.1 Dual-Component Empirical Tracking

| 论文 claim | 真实 | 状态 |
|---|---|---|
| 招募 200, 30-q × 6-point Likert | data_v3: N=163 final, 30 × 6-point ✓ | ✓ |
| Task 1 含单选+排序 (=Q1+Q2+Q3-Q6+Q8) | ✓ | ✓ |
| Task 2 = Q7 开放式 | ✓ | ✓ |
| 时间间隔 ≥ 半小时 | (无法从数据 verify, 信用接受) | — |
| Task 1 trait-stimulus space 跟 behavioral-state space 解耦 | ✓ matches `docs/notes/Q1-Q6每个选项的元数据.md` | ✓ |
| "163 valid responses" | ✓ N=163 final | ✓ |

→ §4.1 基本一致, 不用改 substance。

### 3.2 §4.2 Relationship Mining and Robustness Analysis

| 论文 claim | 真实 (L4/L4b/L4c/L5) | 状态 |
|---|---|---|
| EFA (没说 K) | K_EFA = 4 (PA 推荐, KMO=0.60) | △ 可加 K=4 数字 |
| Permutation Tests | L4 = 1000 perm on (factor, qid, option) | ✓ |
| Multinomial LASSO (L1) | L4c: 30 items → per option top-3, C=0.5 | ✓ |
| 1000 Bootstrap + demographic partial-out | L5 = boot stability + partial regression | ✓ |
| t-SNE + GMM/K-means → trait-continuous manifold | L3 + L3-multi + L3-GMM + L3-visual: silhouette ≤0.25, χ² 2/30 sig | ✓ paper 跟 v3.7 framing 升级一致 |

**改写建议** (可选, 增强可信度):
- 加一句 "EFA with Horn's parallel analysis selected K = 4 latent factors (KMO = 0.60, Bartlett's $\chi^2$ rejected sphericity at $p < 10^{-41}$)."

### 3.3 §4.3 Statistical Data-Driven Construction (✓ 数字全对)

我从 `caps_quadruples.json` 直接重新统计了一遍:

```python
Total quads: 36                              ✓ paper said "36"
Predicates:
  strictly_inhibits: 22                      ✓ paper "Strictly inhibits"
  guides_transition_to: 6                    ✓ paper "Guides transition to"
  weakly_shifts_to: 5                        ✓ paper "Weakly shifts to"
  strongly_forces_transition_to: 3           ✓ paper "Strongly forces transition To"
Directions: 14 +, 22 -                       ✓ paper "14 positive and 22 negative"
|coef| min/max: 0.209 / 0.861                ✓ all in [0.20, ...] (negative threshold |coef|≥0.20)
p<0.05 count: 25                             (consistent with paper threshold rules)
p<0.10 count: 36 (= all 36)                  ✓ "p < 0.10" filter
Per O state: V=17, K=10, P=5, C=4            (paper doesn't claim per-O distribution)
```

**全部数字 sanity-check 通过**. §4.3 thresholds 描述跟 `caps_quadruples.json` 完全对得上:

| paper rule | 实际 count | 状态 |
|---|---|---|
| Strongly forces (p<0.05, |coef|≥0.50, +) | 3 | ✓ |
| Guides (p<0.05, 0.20≤|coef|<0.50, +) | 6 | ✓ |
| Weakly shifts (0.05≤p<0.10, +) | 5 | ✓ |
| Strictly inhibits (|coef|≥0.20, -) | 22 | ✓ |
| **Total** | **36** | ✓ |

**§4.3 数字唯一小 bug** (line 218 原文):
```
\item \textbf{Guides transition to):} Triggered by ...
```
`to):` 多了个右括号. 改成 `to:`。

### 3.4 §4.4 Prompt-Guided State Routing — 跟实际 prompt 对照

`docs/briefs/prompt_v2_contrastive_diverse_v1.md` §1 + `paper_appendix_prompts_zh.md` A.4 给出 v2_anchor 真实 prompt 结构. 对照如下:

| 论文 §4.4 claim | 真实 prompt | 状态 |
|---|---|---|
| filter quads by agent's 30-d Likert direction agreement | `get_activated_quads_balanced()` | ✓ |
| PKVC-stratified: top-1 per state + 1 overall = 4-5 quads | ✓ exact | ✓ |
| stratification 避免 V crowd-out | data: V=17 quads (47%), 不分层确实会被 V 主宰 | ✓ |
| 4-line β summary header | ✓ 跟 `format_caps_quad_balanced_nl` 输出一致 | ✓ |
| **(i) per-segment JSON schema constraining `state_sequence` and `nl_answer`** | 实际 schema 字段是 `dominant_factor` (seg1/3) / `chain` (seg2) + `criterion` (seg3 必填) + `nl_answer`. **没有 `state_sequence` 字段** | ✗ **改写**: 把 `state_sequence` 改为 `dominant_factor`/`chain` + `criterion` |
| (ii) PI + Likert-strong stance labels as static priors | ✓ B1_h (PI 5 字段) + B2_h (30 维 stance) | ✓ |
| (iii) agent's Q3-Q6 conflict-resolution behaviors as trajectory exemplars | ✓ B3_h, 但**实际 hardcode Q3-Q6 across all seg (seg1/2/3 都用 Q3-Q6, intent 原本是 seg1↔Q1, seg2↔Q2, seg3↔Q3-Q6 seg-aware, 实现误)** | △ paper 描述跟 actual run 一致, 不算错; 但 limitation 节可以坦白这个 implementation 偏差 |
| (iv) seg3 deterministic rule mapping strongest positive-β state → bottom-line anchor | ✓ SEG3_RULE_V2 (~300 chars in SYSTEM) | ✓ |

**改写**: §4.4 line 234 把 `\texttt{state\_sequence}` 改为 `\texttt{dominant\_factor}` (或 `\texttt{chain}` for seg2) + `\texttt{criterion}` (seg3-only).

---

## 4. §5 Experiments — 大改区, 论文这里最弱

### 4.1 §5.1 Evaluation Benchmark

| 论文 claim | 真实 | 状态 |
|---|---|---|
| 162 designers / 486 instances | ✓ matches cross_model 实跑 | ✓ (但跟 §3 abstract 的 163 冲突, 见 C5) |
| 3 segs: Seg1 immediate intuitive / Seg2 weighing / Seg3 bottom-line | ✓ matches `infer_v2.py::SEG_TO_QID_HINT` + `COGNITIVE_FUNCTION_QUESTIONS` | ✓ |
| stateless prompting per seg, not autoregressive continuation | ✓ `infer_v2.py` 每个 seg 独立 generate_once | ✓ |
| Q1-Q6 → Q7 cross-scenario transfer ("conditioned on SADR and PKVC from Q1-Q6") | ✓ training scenario = 全感知效率空间, eval scenario = 老城区超市 | ✓ |

### 4.2 §5.2 Model Configuration — 4 处不一致

| 论文 claim | 真实 (cross_model_ladder_ablation_results.md §0.2) | 状态 |
|---|---|---|
| **5 LLMs**: GPT-4o, Gemini 3.1 Flash Lite, Claude Haiku 4.5, DeepSeek V4 Flash, Qwen 3.6 35B A3B | 实际跑 **6 LLMs** (上面 5 个 + Qwen 3.6 27B) | △ Qwen 27B 在 B0/v2 condition broken (yunwu 限流, §6.1), main table 不报合理; 应在 §5.2 加注或限定到 5 |
| temperature = 0.7, max length **600 tokens** | `config_v2.yaml::infer.max_new_tokens = 512` (这是 SFT 端) | △ 需 verify cross-model run 的实际 max_tokens (paper 写 600 跟 SFT 不冲突, 但需跟 `scripts/run_b6_sample.py` 对) |
| PKVC sequence extraction: GPT-4o, temperature=0, JSON schema constrained | "v8.7 extractor (GPT-4o backend)" ✓ | ✓ |
| text-embedding-3-small, 1536-d | ✓ | ✓ |

**改写**: 
- 选 (a) 把 "five" 改 "six" 并加注 Qwen 27B 的 caveat; 或
- (b) 维持 "five", 但 §5.2 末尾或 limitation 加一句 "We additionally evaluated Qwen 3.6 27B but excluded it from the main table due to API throttling artifacts on two conditions (see Appendix X)."

### 4.3 §5.3 Baseline Configurations — **paper 最大错误**

**论文写的 ladder** (§5.3 line 260-270):
```
B1: Zero-shot                          (= scenario only)
B2: Zero-shot + PI                     (= +5 字段 identity)
B3: PI + SADR Persona + CoT            (= +30 维 SADR + Chain-of-Thought)
B4: PI + SADR Persona + RAG            (= +peer top-3 RAG; 检索 Q1-Q6 的 NL)
v2_anchor (SOTA): PI + SADR + Q3-Q6 + CAPS quads + β + seg3 rule
```

**实际跑的 ladder** (cross_model `B0_h ... B4_h`):
```
B0_h: scenario + PKVC primer (SYSTEM 里已含 PKVC 4 字母 def + JSON schema + seg3 rule)
B1_h: + ① PI (5 字段)
B2_h: + ② 30 维 stance labels   ← 跟 paper 的 "SADR" 不同概念 (这是 Likert 强立场, 不是 SADR 12 子类)
B3_h: + ③ Q3-Q6 self conflict 行为
B4_h = v2_anchor: + ⑤ CAPS quad 4 类平衡 + β 速览  ★

(peer RAG 不在主 ladder, 是 ablation +peer = v2_plus_peer)
```

**错配点**:
1. **paper B3 = "+CoT" → 实际 B2_h = "+30 维 stance labels"**, 不是 CoT
2. **paper B4 = "+peer top-3 RAG" → 实际 B3_h = "+Q3-Q6 行为"**, peer RAG 没在主 ladder
3. **paper v2_anchor 描述提了 "Q3-Q6 behaviors" + "CAPS quadruples", 但顺序应该说清: Q3-Q6 是 B3_h 那步加的, CAPS quad 是 B4_h (=v2_anchor) 这步才加, 因此 v2_anchor = B3_h + CAPS quad**
4. **CoT 字样在实际 prompt 里没有专门标记**, 别用 CoT 这词描述

**改写建议** — §5.3 ladder 重写:

```
We construct a 5-level cumulative module ladder (B0_h–B4_h), each adding
one component to the previous:

B0_h (Zero-shot + framework primer): SYSTEM prompt only — PKVC 4-state
  definitions, per-segment output JSON schema, and the seg3 dominant-factor
  decision rule. No designer-specific information.

B1_h (+Personal Identity): adds five high-discriminative demographic fields
  (identity, years of experience, design field, working context, project
  role).

B2_h (+30-dim Stance Labels): adds the agent's "strong-stance" labels —
  Likert items with score ≥ 5 (strong agree) or ≤ 2 (strong disagree) on
  each of the 30 dimensions, injected as natural-language polarity tags.

B3_h (+Self Conflict Trajectories): adds the agent's own answers to Q3–Q6
  (four single-choice + free-text conflict resolutions) as trajectory
  exemplars.

B4_h = v2_anchor (+CAPS Quadruple Routing) ★: adds the activated CAPS
  quadruple block. For each agent, quads are filtered by direction
  agreement with the 30-d Likert profile, PKVC-stratified into 4–5
  routing rules, and prefixed by a 4-line β-activation summary header.
  This is our SOTA configuration.
```

然后 §5.5 ablation 里, peer RAG 列为 +peer ablation (= v2_plus_peer), 不放在 main ladder。

### 4.4 §5.4 Experimental Results — 当前空白, 应填如下

从 `reports/datav3/cross_model_ladder_ablation_results.md` §1-§9 + `cross_model_full_metrics_v2.json` 填:

**主表 (跨 5 model 平均, v2_anchor vs B0_h)**:

| Metric | B0_h | v2_anchor | Δ |
|---|---|---|---|
| macro-TED ↓ | ~2.63 (avg) | ~2.41 | **−0.22** (5/5 model 一致下降) |
| chain diversity ↑ | ~0.59 | ~0.88 | **+0.29** (5/5 model 单调上升) |
| non-V recall ↑ | ~0.16 | ~0.35 | **+0.19** (4/5, Gemini 反向因 B0 已 ceiling) |
| macro-LCS-F1 ↑ | ~0.48 | ~0.55 | +0.07 (混合, 跨 model 1/6 单调) |
| criterion uniqueness (seg3) ↑ | ~0.40 | ~0.57 | +0.17 (5/5 一致上升) |
| **Trait-pred Spearman ρ (seg1) ↑** | ~−0.01 | ~+0.09 | **+0.10** (5/6 一致, **删 CAPS 后 6/6 model 全降** — 唯一强 CAPS-attributable NL 信号) |

**Aggregate rank** (12 condition × 4 main metric × 5 model = 240 cells):

| Prompt | Mean rank | Win count |
|---|---|---|
| **v2_anchor** ★ | **3.45/10** | 8/20 |
| v2_plus_peer | 3.55 | 5 |
| v2_hybrid | 4.10 | 3 |
| ... | | |
| B0_h | 7.85 | 1 |

**主 claim (强, 跨 5 model 一致, 可写)**:
1. Anchor family (v2-anchor + variants) > zero-shot on chain diversity (5/5) + macro-TED (5/5) + criterion uniqueness (5/5)
2. CAPS quad block 是 trait-conditioned NL personalization 的唯一一致驱动器 (Trait-pred ρ 删 CAPS 后 6/6 model 下降, magnitude −0.04 to −0.10)
3. v2_anchor 是 most robust single configuration (mean rank 3.45/10)

**不能 claim (cross-model 不一致, 不要写)**:
1. ✗ "CAPS module universally critical" — DSv4F 上删 CAPS 反而 better
2. ✗ "seg3 rule universally critical" — 仅 2/5 model 删后 hurt
3. ✗ "v2_anchor universally optimal on chain accuracy" — DSv4F/Qwen35B 上 B2 (+stance only) 是 sweet spot
4. ✗ "non-V recall improvement generalizes" — Gemini 反向

**推荐 paper framing (来自 cross_model_ladder_ablation_results.md §9.3)**:

> Anchor prompts (v2-family) consistently outperform zero-shot baselines
> across 5 LLMs on chain reasoning diversity (5/5 models) and chain
> accuracy (5/5 models). v2_anchor is the most robust single configuration
> (aggregate rank 3.45/10, 8/20 metric × model cells win). Component
> contributions vary by model: explicit decision rules and structured
> anchors are most beneficial on large instruction-tuned models (GPT-4o,
> Claude), while smaller models often prefer simpler configurations.

### 4.5 §5.5 Ablation Analysis — 描述基本对, 缺数字

| ablation 名 | paper 描述 | actual | 状态 |
|---|---|---|---|
| v2_no_caps | ✓ | ✓ | 6 个 ablation 命名/语义跟 `scripts/run_b6_sample.py --variant` 一致 |
| v2_no_seg3_rule | ✓ | ✓ | ✓ |
| v2_no_beta_summary | ✓ | ✓ | ✓ |
| v2_no_pkvc_def | ✓ | ✓ | ✓ |
| v2_plus_peer | ✓ | ✓ | ✓ |
| v2_hybrid | ✓ | ✓ | ✓ |

→ §5.5 描述 OK, 但**全部数字需要从 §1-§6 跨 model 表填**. 填法见 4.4 节主表。

---

## 5. Abstract / §1 一致性

| 论文 claim | 真实 | 状态 |
|---|---|---|
| "163 designers" | ✓ | ✓ |
| "30-dimensional trait layer" | ✓ | ✓ |
| "4-state generator termed PKVC" | ✓ | ✓ |
| "operationalizes CAPS theory" | ✓ matches framing | ✓ |
| "ontological procedures dynamically parameterize state-transition probabilities" | ✓ (虽然实际是 in-context prompt routing, 不是 token-level transition prob, framing 上 OK) | △ 弱包装, 严格说是 in-context guidance 不是 modulating P(e_t \| ...). 可保留, 但 §3 公式跟实际实现的 gap 应该意识到 |

Abstract 没 specific 错, 但 (paper)→(code) gap: "dynamically modulating state transitions" 实际是把 quad 当 NL routing 描述拼到 prompt 里, 让 LLM 自己按 in-context instruction 走 — 不是真的 token-level state machine. 论文用 "controlled transition process" + 公式 P(e_t|...) 是合法 abstraction, 但 reviewer 可能追问. 准备 rebuttal 时要心里有数。

---

## 6. 跟 v3.7 brief 的 framing alignment 检查

paper 写的 framing 跟 `v3.7_patch_post_datav3_20260524.md` 一致性:

| v3.7 framing | paper | 状态 |
|---|---|---|
| 砍 cluster, trait-continuous | paper 不提 cluster, 提 "trait-continuous manifold" | ✓ |
| K_EFA=4 (PA), K_state=4 (theory lock), 数值相等是巧合 | paper 没明说 K_EFA, 但 "30-d trait + 4-state PKVC" 跟 v3.7 一致 | ✓ |
| global mapping main (D-15 翻案) | paper 没说 cluster, 默认 global ✓ | ✓ |
| bypass<0.06 / Weak<0.28 / Strong≥0.28 (data-driven) | paper §4.3 thresholds: bypass 没提, 直接给 Strong/Weakly/Inhibit |coef| 阈值 ≥0.50/0.20 | △ paper 用的是 effect-size 阈值 (0.50, 0.20), 跟 brief 的 data-driven 0.28 不同; 实际 `caps_quadruples.json` 阈值是按 0.50/0.20 切的 (我 verify 过), 跟 paper 一致 |
| 三独立 K | paper OK (没硬绑 K_EFA=K_state) | ✓ |

→ framing 上 paper 跟 v3.7 一致 (v3.6 cluster 描述已经清掉)。**对立面**: brief v3.7 §5.4 给的"data-driven 阈值 0.06/0.28"实际**没用在 caps_quadruples.json 里**, paper 用的是 0.20/0.50 hard threshold. 需要 decide:
- 论文锁 0.20/0.50 (current paper choice + current quad DB)
- 或重跑 quad DB 用 0.06/0.28, 重写 §4.3

(建议保持 paper 现状: 0.20/0.50 是更 conservative + 出来的 36 个 quad 看起来 sane. brief 的 data-driven 阈值 没 finalize 到 quad DB.)

---

## 7. Open questions (需要你拍板再改)

| Q | 选项 | 推荐 |
|---|---|---|
| Q1: 162 vs 163 怎么统一? | (a) 全文 162 / (b) 全文 163 + §5.1 加 "1 dropped" 注 / (c) abstract 用 163, §5 用 162 + caveat | (b) — 数据 spent on 163, eval set 162 + 一句 caveat 最干净 |
| Q2: 5 model vs 6 model? | (a) 全文 "five" + Qwen 27B 限到 appendix / (b) "six" + 主表标注 Qwen 27B partial | (a) — main table cleaner, Qwen 27B 数据 broken 不值 main |
| Q3: §5.3 baseline 是否完全重写跟 actual ladder 对齐? | (a) 重写 B1→B4 跟 actual B0_h-B4_h 一致 / (b) 维持 paper 当前 narrative, 加 footnote 说 actual ladder | **(a) — 当前 paper §5.3 跟实际跑的 baseline 严重错配, 不重写会被 reviewer 当 fabrication** |
| Q4: §3.3 trip 字样改成 quad? | (a) 全文统一 quad / (b) 维持 triplet/quad 混用 | (a) — 跟 §4 一致 |
| Q5: §3.3 ontological rule 例子用真数据 quad 还是抽象描述? | (a) 真 quad: Q1_D__数据vs直觉 / (b) 抽象 ethical example | (a) — example 接地, 也呼应 §4.3 数字 |
| Q6: SFT inference 结果 (sft_v2_* 那一坨) 跟 paper 什么关系? | (a) paper 完全 prompt-based, SFT 数据另起一篇 / (b) paper appendix 加 SFT comparison | 看你跟一作要不要把 SFT 放进 main; 当前 paper §3-§5 没提 SFT, framing 是 prompt-only |

---

## 8. 一句话改写优先级

1. 立刻删 `(JIDAN GAIed)` (C1)
2. 立刻修 4 个 `_` 下标 (C2)
3. 立刻补 §3.3 ontological rule 例子 (C4) 用 `Q1_D__数据vs直觉` β=+0.537 那条
4. §5.3 baseline ladder 整段重写跟 actual B0_h-B4_h 对齐 (C6)
5. §5.4 填 6 个 metric 数字 + framing 用 "anchor family > zero-shot generalizes" (C7)
6. 全文 trip→quad 统一 (C3)
7. §5.1 N 描述消歧 162/163 (C5)
