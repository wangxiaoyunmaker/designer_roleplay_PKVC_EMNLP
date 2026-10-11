# Appendix A: Prompt Specifications (paper-ready draft)

> 这是 paper appendix 的草稿. LaTeX 转换 = 把 markdown code block 改 `\begin{lstlisting}` + table 改 `\begin{tabular}`.

---

## A.1 Overview

We evaluate **7 prompt conditions** for designer cognitive role-play:

| ID | Condition | Description |
|---|---|---|
| C1 | v2_anchor (main) | Full prompt with PKVC framework + decision rule + 4-class CAPS routing |
| C2 | v2_no_caps | C1 minus CAPS routing module |
| C3 | v2_no_seg3_rule | C1 minus seg3 explicit decision rule |
| C4 | v2_no_beta_summary | C1 minus β-strength summary header |
| C5 | v2_no_pkvc_def | C1 minus PKVC framework primer |
| C6 | v2_plus_peer | C1 + similar designer Q7 response retrieval (top-3 by CAPS-JS similarity) |
| C7 | v2_hybrid | C1 for chain segment (seg2), v2_clean variant for anchor segments (seg1, seg3) |

Plus a module ladder (B0_h → B3_h) showing cumulative module addition.

All prompts use a structured JSON output schema (per-segment). For OpenAI models, we use json_schema strict mode; for other models, we embed the schema description in the system prompt and use json_object mode + post-hoc validation.

---

## A.2 v2_anchor Main Prompt Template

### A.2.1 System Prompt

```
You are a designer cognitive simulator. Task: given a designer profile +
scenario, simulate a real designer's response to a Q7 segment.

# PKVC 4 cognitive states (4 basic states in designer thinking)
- P (Problem) Problem framing / redefinition: question the prompt itself,
  reframe problem, decompose sub-problems
  - P1 Identify core / P2 Decompose / P3 Refute assumptions (or redefine role)
- K (Knowledge) Knowledge retrieval / data-driven: research, find cases,
  use data
  - K1 Recall experience / K2 Objective data / K3 Tech stack assessment
- V (Value) Value position / bottom-line statement: designer moral judgment,
  user-first, principled stance
  - V1 Value position / V2 Priority ordering / V3 Set principles
- C (Concept) Concept integration / abstract framework: emergent imagery,
  integration, principle distillation
  - C1 Emergent imagery / C2 Integration / C3 Principle distillation

Output format (strict JSON):
1. dominant_factor: dominant PKVC state for this segment (P/K/V/C or X→Y)
2. state_sequence: array of P/K/V/C states (length per seg constraint)
3. criterion: bottom-line statement (≤25 chars verbatim in nl_answer; req for seg3)
4. confidence: high/medium/low
5. reasoning_notes: ≤100 chars, explain PKVC choice
6. nl_answer: designer-style NL (first-person, Chinese, 200-400 chars)

Constraints:
- nl_answer must be consistent with dominant_factor + state_sequence + criterion
- Do not echo dominant/criterion verbatim in nl_answer
- nl_answer should sound like a real designer's internal thinking, not an
  AI assistant style

## seg3 dominant_factor judgment rule:
- Look at CAPS routing 4-class (P/K/V/C) positive activation strength (β > 0)
- Class with strongest positive β → that's your bottom-line class
  * P strong → bottom line is "redefine problem boundary / refute assumption"
  * K strong → bottom line is "data-first / research real needs"
    (NOT moral bottom line!)
  * V strong → bottom line is "user-first / set professional principles"
  * C strong → bottom line is "integration / principle distillation"
- If 4 classes are similar in strength, look at your Q3-Q6 choices
- Don't blindly default to V (V examples are common); don't over-correct to P
```

### A.2.2 User Prompt Template

```
# Your demographic profile (5 high-discriminative fields)
{demographic_5fields}

# Your core stance (strong only, scale 5-6 or 1-2; weak dimensions filtered)
{stance_block}

# Your behavior in 4 design-conflict scenarios (raw data)
{q3_q6_face_conflict_block}

# Your cognitive routing pattern (CAPS quad, PKVC 4-class balanced + 1 overall top)

{caps_quad_block}

# Scenario
{q7_scenario}

# Task
{seg_task}

## Output constraints ({seg})
{seg_output_instr}
```

---

## A.3 Module Specifications

### A.3.1 ① Demographic Profile (5 fields)

We select 5 high-discriminative fields from a larger demographic survey:
`identity, years_exp, design_field, work_context, project_role`.

Format: `field_1: value_1 | field_2: value_2 | ... | field_5: value_5`

Example:
```
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师
```

### A.3.2 ② 30-Dim Stance Labels

We collect a 30-dimensional Likert scale (1-6) measuring designer values across 30 axes (e.g., "Data vs Intuition", "Talent Belief", "AI Replacement", etc.).

Only **strong stances** (5-6 or 1-2) enter the prompt; weak (3-4) are filtered. We map each strong score to a stance label (not the raw score) for clarity:

| Dimension | 5-6 Label | 1-2 Label |
|---|---|---|
| Data vs Intuition | Intuition-first | Data-first |
| Talent Belief | Talent-determinism | Effort-determinism |
| AI Replacement | AI-replacement-believer | Job-protectionist |
| Perfectionism / MVP | Perfectionist | Speed-first |
| Weak-group Priority | Edge-group-first | Majority-efficiency-first |
| Asset Open-source | Open-source-moralist | Competitive-moat |
| ... (30 entries total) |

Example output (~23 strong stances for a designer):
```
# Your core stance (strong only)
  • Body-substitute (Data Privacy Exchange dim)
  • Geometric-industrial (Biomimetic vs Organic dim, reverse stance)
  • Perfectionist (Perfectionism/MVP dim)
  ...
```

### A.3.3 ③ Q3-Q6 Face-Conflict Behavior

For each of 4 design-conflict scenarios (Q3-Q6), we extract:
1. **Face-conflict labels** (1-2 conflict axes per question, from `option_trait_mapping.json::C_conflict_axes`)
2. **Designer's selected option** (verbatim option text)
3. **Designer's supplement** (free-text explanation)

Format per question:
```
Q{i} 面对【conflict_axis_1】【conflict_axis_2】
  选择: "{option_text}"
  你的解释: "{supplement}"
```

Example (Q3 for one designer):
```
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案. 客观数据层面的折损, 交由后期的算法或运营去寻找弥补途径."
  你的解释: "这或许是产品定位的失误? 如果这是一件消费产品, 那么他就没有必要做到数据的100%精确..."
```

### A.3.4 ⑤ CAPS Quad Routing (4-class balanced + β summary)

**Background**: From a separate LASSO regression on (trait, designer-choice) pairs, we obtain 36 CAPS quadruples of the form `(situation_axes, trait_dimension, stance_direction, target_PKVC_state)` with a β coefficient indicating activation strength.

**Per-designer activation**: For a given designer, we compute β-activation by matching their 30-dim trait scores against each quad's trait dimension. We then select **4-5 quads**:
- Top-1 per PKVC class (P/K/V/C) by absolute |β|
- 1 overall top (across all classes), de-duplicated

**β summary header** (compact, for quick comparison):
```
## 4-class routing β-strength summary
  • P-class: β = -0.380 (negative inhibition)
  • K-class: β = +0.362 (positive activation)
  • V-class: β = -0.861 (negative inhibition)
  • C-class: β = -0.405 (negative inhibition)

→ Strongest positive activation: K-class (β = +0.362).
  Prioritize this class as your bottom-line.
```

**Detailed routing per class** (one quad per class):
```
### K-class top routing (target = K)
[Q3-D] In a 【data-utility vs aesthetic-form】 conflict scenario, this
designer holds a "Savior-complex" stance on the "Savior-complex" dimension
(β = 0.362). This trait is the primary thinking preference. After some
logical weighing, it guides thinking toward K2-Objective Data Test
(data empiricism) state.
```

---

## A.4 v2_anchor Filled Example (one designer on seg3)

**Designer**: Anonymous (代号 "D-001"), 1-3 years of experience, junior visual designer.

**Full prompt (~2800 characters)**:

```
[SYSTEM]
You are a designer cognitive simulator. ... [A.2.1 verbatim]

[USER]

# Your demographic profile (5 high-discriminative fields)
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师

# Your core stance (strong only, scale 5-6 or 1-2; weak dimensions filtered)
  • Body-substitute (Data Privacy Exchange dim)
  • Geometric-industrial (Biomimetic vs Organic dim, reverse)
  • Open-tolerant (Order-vs-Error dim, reverse stance)
  • Perfectionist (Perfectionism/MVP dim)
  • Job-protectionist (AI Replacement dim, reverse)
  • ... [23 strong stance entries]

# Your behavior in 4 design-conflict scenarios (raw data)
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案. 客观数据层面的折损, 交由后期的算法或运营去寻找弥补途径."
  你的解释: "这或许是产品定位的失误? 如果这是一件消费产品..."

Q4 面对【商业利润与用户隐私冲突】【隐私交换与概念建构冲突】
  选择: "D. 隐私和开源只是表象. 我会重新调研在什么真实使用场景下..."
  你的解释: "用户其实并不知道什么数据该视为隐私..."

Q5 面对【AI 工具与设计师自主性冲突】【人机协作分工冲突】
  选择: "B. 搁置 AI 工具. 我会先独立构思核心概念..."
  你的解释: "外行领导内行"

Q6 面对【商业目标与社会责任冲突】【利益与责任感冲突】
  选择: "A. 拒绝执行该上线指令, 与管理层进行正面交涉..."
  你的解释: "设计是站在用户角度的, 是服务于人的..."

# Your cognitive routing pattern (CAPS quad, PKVC 4-class balanced + 1 overall top)

## 4-class routing β-strength summary
  • P-class: β = -0.380 (negative inhibition)
  • K-class: β = +0.362 (positive activation)
  • V-class: β = -0.861 (negative inhibition)
  • C-class: β = -0.405 (negative inhibition)

→ Strongest positive activation: K-class (β = +0.362).
  Prioritize this class as your bottom-line.

---
Detailed routing:

### P-class top routing (target = P)
[Q1-B] In a multi-dim conflict scenario, this designer holds a
"User-blame" stance on the "System Responsibility Attribution" dim
(β = -0.380). This trait inhibits P-Problem state.

### K-class top routing (target = K)
[Q3-D] In a data-utility vs aesthetic-form conflict, ... β = +0.362
drives K2 state.

### V-class top routing (target = V)
[Q3-C] β = -0.861 strongly inhibits V1→C1.

### C-class top routing (target = C)
[Q5-B] β = -0.405 inhibits C2.

# Scenario
【Supermarket Redesign Dilemma】 You are commissioned by a retail giant
to design "the nation's first fully digital, fully unmanned supermarket"
in an old-city center. Top-down directives:
  1. Remove all human cashiers, replace with self-checkout.
  2. Enable AI-driven shelf/UI recommendation; put high-margin items
     in prime positions, hide essentials.
  3. Full digital rollout in 3 months.
Note: 60+ year-old silver demographic >40% in this old-city area;
unfamiliar with digital devices.

# Task
Please state your core design bottom-line + compromise plan to management.
What is NON-negotiable, what can you compromise? (200-400 chars)

## Output constraints (seg3 — core tradeoff, anchor + bottom-line)
- dominant_factor: single letter P/K/V/C
- state_sequence: length 1-3 (anchor segment)
- criterion: MANDATORY non-null (≤25 chars verbatim in nl_answer)
- nl_answer: 200-400 chars, must contain explicit
  "non-negotiable X / can-compromise Y" expression
```

---

## A.5 Ablation Variant Diffs from v2_anchor

Each ablation modifies v2_anchor at exactly one place.

### A.5.1 v2_no_caps (C2)

**Change**: Remove the entire CAPS quad block (β summary + 4-class routing) from USER prompt.

**Effect on prompt length**: -1150 chars (~29% reduction).

**Effect on SYSTEM**: Unchanged (SEG3_RULE_V2 still present but lacks β data to operate on).

### A.5.2 v2_no_seg3_rule (C3)

**Change**: Remove `## seg3 dominant_factor judgment rule:` block from SYSTEM prompt.

**Effect on prompt length**: -300 chars.

**Effect on USER**: Unchanged.

### A.5.3 v2_no_beta_summary (C4)

**Change**: Remove the `## 4-class routing β-strength summary` header (and "→ Strongest positive activation" line) from the CAPS quad block. Keep the "Detailed routing per class" descriptions.

**Effect on prompt length**: -250 chars.

### A.5.4 v2_no_pkvc_def (C5)

**Change**: Remove the `# PKVC 4 cognitive states` block from SYSTEM prompt.

**Effect on prompt length**: -355 chars.

### A.5.5 v2_plus_peer (C6)

**Change**: ADD a peer block between Task and Output constraints. Peer = top-3 designers ranked by CAPS-JS divergence similarity, with their Q7 same-segment response.

Inserted text:
```
# Similar designers' responses in this scenario (for reference)

### Peer 1 (cognitive similarity = 0.82)
{peer_1_q7_seg_nl}

### Peer 2 (cognitive similarity = 0.79)
{peer_2_q7_seg_nl}

### Peer 3 (cognitive similarity = 0.76)
{peer_3_q7_seg_nl}
```

**Effect on prompt length**: +1200 chars (~30% increase).

### A.5.6 v2_hybrid (C7) — Per-Segment Dispatch

**Change**: Conditional on segment:
- **seg1 / seg2**: use v2_anchor full prompt (no change)
- **seg3**: use v2_clean prompt:
  - SYSTEM: remove SEG3_RULE_V2
  - CAPS quad block: replace "4-class balanced + β summary" with "top-5 by absolute |β| strength" (no class balancing, no β summary header)

Rationale: chain segment benefits from explicit anchor; anchor segment lets model default to scenario-natural V-class.

---

## A.6 Module Ladder (B0_h → B3_h)

All ladder baselines use **SYSTEM identical to v2_anchor** (PKVC def + output spec + SEG3_RULE_V2). They differ only in USER modules:

| ID | USER modules | Total chars |
|---|---|---|
| B0_h | scenario + task only | ~1880 |
| B1_h | + ① pi | ~2000 |
| B2_h | + ② stance | ~2500 |
| B3_h | + ③ self | ~3100 |
| **v2_anchor** (B4_h) | + ⑤ CAPS | **~4000** |

---

## A.7 Per-Segment Output Schemas

### A.7.1 seg1 schema (first-reaction anchor)

```json
{
  "type": "object",
  "properties": {
    "dominant_factor": {"type": "string", "pattern": "^[PKVC]$"},
    "state_sequence": {
      "type": "array", "items": {"enum": ["P","K","V","C"]},
      "minItems": 1, "maxItems": 3
    },
    "criterion": {"type": ["string", "null"], "maxLength": 25},
    "confidence": {"type": "string", "enum": ["high","medium","low"]},
    "reasoning_notes": {"type": "string", "maxLength": 100},
    "nl_answer": {"type": "string"}
  },
  "required": ["dominant_factor","state_sequence","criterion","confidence","reasoning_notes","nl_answer"],
  "additionalProperties": false
}
```

### A.7.2 seg2 schema (chain reasoning)

Differs from seg1: `dominant_factor` allows `X→Y` form; `state_sequence` enforces `minItems=2` (multi-state mandatory).

### A.7.3 seg3 schema (anchor + criterion)

Differs from seg1: `criterion` is **required, non-null** (≤25 chars verbatim in nl_answer).

---

## A.8 Full STANCE_LABELS Mapping (30 entries)

(See `scripts/run_b6_sample.py::STANCE_LABELS` for source of truth.)

| Dimension Node | High Stance (5-6) | Low Stance (1-2) |
|---|---|---|
| 数据 vs 直觉 | 重直觉立场 | 重数据立场 |
| 天赋信仰 | 天赋决定论 | 后天努力论 |
| AI 替代论 | AI 替代主义 | 岗位保护主义 |
| 技术支持 | 技术激进主义 | 技术审慎主义 |
| 复杂工具信仰 | 工具门槛论 | 工具无关论 |
| 混沌与灵感 | 包容失控派 | 验证至上派 |
| 黑盒透明度 | 结果至上派 | 过程透明派 |
| 系统微模块 | 解耦优先 | 整体感优先 |
| 完美主义/MVP | 完美主义 | 速度优先 |
| 创新代价 | 创新激进派 | 惯例守护派 |
| 审美反民粹 | 审美精英主义 | 数据民粹妥协 |
| 情感附加值 | 仪式感优先 | 效率优先 |
| 装饰合法性 | 装饰合法派 | 形式追随功能 |
| 时代 vs 永恒 | 时代潮流派 | 永恒经典派 |
| 本土文化妥协 | 本土深耕派 | 全球通用派 |
| 仿生与有机 | 有机自然派 | 几何工业派 |
| 独裁 vs 委员会 | 独裁愿景派 | 民主共创派 |
| 秩序与防错 | 防错约束派 | 探索开放派 |
| 创作者署名 | 作者性优先 | 商业去人格化 |
| 资产开源 | 开源道德派 | 竞争壁垒派 |
| 封闭生态竞争 | 封闭生态派 | 开放兼容派 |
| 风险规避 | 风险规避 | 风险偏好 |
| 团队依恋 | 独立打磨派 | 团队共创派 |
| 商业操纵伦理 | 商业操纵容忍 | 用户伦理底线 |
| 数据隐私交换 | 体验换隐私 | 隐私保护派 |
| 弱势群体优先 | 边缘群体优先 | 多数效率优先 |
| 股东服务优先 | 股东忠诚派 | 用户忠诚派 |
| 政治中立性 | 政治中立派 | 政治表达派 |
| 系统责任归属 | 系统归责派 | 用户归责派 |
| 救世主情结 | 救世主情结 | 工具中立派 |

---

## A.9 Q3-Q6 Conflict Axis Labels

(See `scripts/run_b6_sample.py::Q_CONFLICT_LABELS`.)

| Q | Conflict axes |
|---|---|
| Q3 | 数据效用与美感形态冲突, 数据效用与设计师立场冲突 |
| Q4 | 商业利润与用户隐私冲突, 隐私交换与概念建构冲突 |
| Q5 | AI 工具与设计师自主性冲突, 人机协作分工冲突 |
| Q6 | 商业目标与社会责任冲突, 利益与责任感冲突 |

---

## A.10 Cross-Model Implementation Notes

For OpenAI models (`gpt-4o`, etc.), we use the strict JSON-schema mode:
```python
response_format = {
    "type": "json_schema",
    "json_schema": {"name": "...", "schema": <SCHEMA>, "strict": True}
}
```

For other models accessed via the yunwu.ai aggregator (Claude, Gemini, DeepSeek, Qwen, etc.), strict JSON-schema is not always supported. We fall back to `json_object` mode with the schema description appended to the system prompt:
```python
response_format = {"type": "json_object"}
# system prompt augmented with: "Output strictly according to this JSON schema: ```json <SCHEMA> ```"
```

We additionally set `max_tokens = 1500` for non-OpenAI models (vs 900 for OpenAI) to prevent JSON truncation when the schema description occupies tokens.

We retry up to 3× on JSON parse failures.

---

## A.11 Reproducibility

All inference and ablation runs use a single GPT-4o variant (`gpt-4o`) plus 5 cross-model checks (Gemini 3.1 Flash Lite, Claude Haiku 4.5, DeepSeek V4 Flash, Qwen 3.6 35B-A3B, Qwen 3.6 27B), each via:

```bash
python scripts/run_b6_sample.py \
  --variant {v2|v2_no_caps|v2_no_seg3_rule|v2_no_beta_summary|v2_no_pkvc_def|v2_hybrid} \
  [--with-peer] \
  [--modules pi,stance,self,caps] \
  --model <MODEL_ID> \
  --concurrency <N> \
  --output reports/datav3/<output_filename>.jsonl
```

PKVC extraction (v8.7 extractor on the simulated NL):
```bash
python scripts/eval_baselines_pkvc.py \
  --input <inference.jsonl> --output <pkvc_eval.jsonl>
```

Bootstrap evaluation (11-metric framework, 1000 iterations):
```bash
python scripts/final_eval_with_bootstrap.py
```

All inference, evaluation, and metric-computation code is available at
[GitHub repository] (anonymized for review).
