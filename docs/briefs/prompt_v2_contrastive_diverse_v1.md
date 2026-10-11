# Prompt v2_anchor: Final Spec + 7-Condition Ablation Suite

> 2026-05-26 final. v2_anchor 是 paper main prompt, 跨 5 model 验证 (GPT-4o + Gemini 3.1 Flash Lite + Claude Haiku 4.5 + DeepSeek V4 Flash + Qwen 3.6 35B A3B).
> 7 个 condition = 1 main + 6 ablations, 全在 `scripts/run_b6_sample.py` 实现.
> Aggregate rank: v2_anchor 跨 20 (model, metric) cells mean rank **2.40 (best of 6 v2-family variants)**.

> ⚠ 之前 `prompt_v2_contrastive_diverse_v1.md` 老版本是 B6 contrastive peer + criterion library 设计 (M21~M24 探索期). 实证后**砍掉** ④ contrastive peer 和 ⑥ criterion library (paper main 不要), 现版本 v2_anchor 是迭代后的最终设计.

---

## 0. Summary 一句话

```
v2_anchor = SYSTEM (PKVC 4 字母 def + 输出格式 + seg3 决策规则)
          + USER (① pi 5 字段 + ② 30维 stance labels + ③ self Q3-Q6 面对【冲突】
                   + ⑤ CAPS quad 4 类平衡 + β 速览 + Q7 scenario + per-seg task)
          ≈ 4000 chars / ~4400 tokens
```

7 个 condition:
1. **v2_anchor** (main) ★
2. v2_no_caps (-CAPS module)
3. v2_no_seg3_rule (-seg3 决策规则)
4. v2_no_beta_summary (-β 速览 header)
5. v2_no_pkvc_def (-PKVC 4 字母定义)
6. v2_plus_peer (+ ④ peer block)
7. v2_hybrid (seg3 用 v2_clean prompt)

加: B0_h~B3_h 模块 ladder (cumulative module add).

---

## 1. v2_anchor 主 prompt 完整 spec

### 1.1 SYSTEM prompt (~1330 chars)

**结构**:
1. PKVC 4 字母定义 (~355 chars)
2. 输出格式说明 (~600 chars)
3. seg3 dominant_factor 判断规则 SEG3_RULE_V2 (~300 chars)

#### 1.1.1 PKVC 4 字母定义

```
# PKVC 4 类认知状态 (设计师思考过程的 4 种基本状态)
- P (Problem) 问题界定 / 重定义: 质疑题目本身, 重新框定问题, 拆解子问题
  - P1 界定核心 / P2 拆解子问题 / P3 推翻假定 (或重定义角色边界)
- K (Knowledge) 知识调用 / 数据驱动: 调研, 找案例, 用数据决策
  - K1 调用经验 / K2 客观数据 / K3 评估技术栈
- V (Value) 价值立场 / 底线宣言: 设计师道德判断, 用户优先, 底线声明
  - V1 价值立场 / V2 利益排序 / V3 设立原则
- C (Concept) 概念整合 / 抽象框架: 涌现画面, 整合方案, 提炼原则
  - C1 涌现画面 / C2 整合构建 / C3 提炼原则
```

#### 1.1.2 输出格式 (per-seg JSON schema)

```
输出格式 (严格 JSON, 先填 PKVC 结构化字段, 再写 nl_answer):
1. dominant_factor: 段主导 PKVC 状态 (P/K/V/C 或 X→Y, 按 seg 约束)
2. state_sequence: 段内 P/K/V/C 状态时序 array (长度按 seg 约束)
3. criterion: 核心底线短句 ≤ 25 字 verbatim, 必须能在 nl_answer 中找到 (seg3 必填, seg1/2 可 null)
4. confidence: 你对结构标的把握 (high/medium/low)
5. reasoning_notes: ≤ 100 字, 简述 PKVC 标的依据
6. nl_answer: 真实设计师风格 NL (第一人称, 中文, 口语化, 200-400 字)

要求:
- nl_answer 必须跟 dominant_factor + state_sequence + criterion 内部一致
- 不要在 nl_answer 里 echo dominant/criterion 字面值
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格
```

#### 1.1.3 SEG3_RULE_V2 (seg3 决策规则)

```
## seg3 dominant_factor 判断规则 (重要):
- 看 CAPS routing 4 类 (P/K/V/C) 的**正向激活强度** (β > 0, "推动" 类; β < 0 是 inhibit 类排除)
- 哪类的 β 正向最大 → 你的底线属于那类:
  * P 强 → 底线是 "重新定义问题边界 / 推翻题目假定"
  * K 强 → 底线是 "数据先行 / 调研真实需求 / 技术可行性评估" (不是道德底线!)
  * V 强 → 底线是 "用户优先 / 设立专业立场原则" (常见的道德声明类底线)
  * C 强 → 底线是 "整合方案 / 提炼抽象框架"
- 若 4 类 β 强度近似, 看 ③ 你在 Q3-Q6 选择倾向哪类
- 不要无脑选 V (V 例子常见), 也不要 over-correct 到 P (K/C 强时该选 K/C)
```

### 1.2 USER prompt 结构 (~2700 chars, 5 module + scenario + task)

```
[# 你的人口统计学信息]      ← ① pi
[# 你的核心立场]            ← ② 30维 stance
[# 你在 4 个设计冲突场景的实际行为]  ← ③ self Q3-Q6
[# 你的认知路由模式]        ← ⑤ CAPS quad 4 类平衡 + β 速览
# 场景                      ← Q7 background
# 任务                      ← per-seg task
## 输出约束                 ← per-seg output instr
```

### 1.3 各模块详细 spec

#### ① pi — 5 高区分度字段 (~120 chars)

字段顺序: `identity / years_exp / design_field / work_context / project_role`

实现: `scripts/run_b6_sample.py::load_pi_5()` + `DEMO_HIGH_DISCRIM_FIELDS`.

Example (BuzhichunD):
```
# 你的人口统计学信息 (5 高区分度字段)
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师
```

注: 5 字段 vs 11 字段 ablation 已 deprecated (cost 不显著, 默认 5).

#### ② 30维 stance labels — strong only, 不写 +/- (~500 chars)

只把量表分 5-6 分或 1-2 分的强 stance 进 prompt, 弱 (3-4 分) drop.

不写 `+/-` 符号, 用直接立场名 (STANCE_LABELS 30 entries 表):

| 维度 (节点名) | 5-6 分 stance | 1-2 分 stance |
|---|---|---|
| 数据 vs 直觉 | 重直觉立场 | 重数据立场 |
| 天赋信仰 | 天赋决定论 | 后天努力论 |
| AI 替代论 | AI 替代主义 | 岗位保护主义 |
| 完美主义/MVP | 完美主义 | 速度优先 |
| 弱势群体优先 | 边缘群体优先 | 多数效率优先 |
| 资产开源 | 开源道德派 | 竞争壁垒派 |
| ... (30 维全表见 `scripts/run_b6_sample.py::STANCE_LABELS`) |

实现: `polarity_to_stance_block()`.

Example (BuzhichunD, 23 个强 stance):
```
# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)
  • 体验换隐私 (数据隐私交换维度)
  • 几何工业派 (仿生与有机维度, 反向立场)
  • 完美主义 (完美主义/MVP 维度)
  ... (省略, 23 行)
```

#### ③ self Q3-Q6 — 面对【冲突】+ 选择 + 解释 (~600 chars)

每 Q 包含 3 件:
1. `面对【conflict_axis_1】【conflict_axis_2】` (1-2 个 conflict axes, 从 `option_trait_mapping.json::C_conflict_axes` 拿; 在 `Q_CONFLICT_LABELS` 表 hardcode)
2. `选择: "<designer 选的 option NL>"`
3. `你的解释: "<designer 自己 supplement>"`

实现: `format_self_q3q6_b6()` + `Q_CONFLICT_LABELS` (Q3-Q6 各冲突轴 hardcoded).

Example:
```
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案. 客观数据层面的折损, 交由后期的算法或运营去寻找弥补途径."
  你的解释: "这或许是产品定位的失误?如果这是一件消费产品, 那么他就没有必要做到数据的100%精确..."
Q4 面对【商业利润与用户隐私冲突】【隐私交换与概念建构冲突】
  ...
Q5 面对【AI 工具与设计师自主性冲突】【人机协作分工冲突】
  ...
Q6 面对【商业目标与社会责任冲突】【利益与责任感冲突】
  ...
```

#### ⑤ CAPS quad block — 4 类平衡 + β 速览 (~1100 chars)

**核心 anchor module**. 由 2 部分组成:

##### ⑤.1 β 速览 header (短)
```
## 4 类 routing 强度速览 (β 值, 正向 → 推动该状态; 负向 → 抑制该状态)
  • P 类: β = -0.380 (负向抑制)
  • K 类: β = +0.362 (正向激活)
  • V 类: β = -0.861 (负向抑制)
  • C 类: β = -0.405 (负向抑制)

→ **正向激活最强**: K 类 (β = +0.362). 优先考虑此类作为你的底线类型.
```

##### ⑤.2 详细 routing 描述 (4 类 + 1 overall top)

**Selection**: 跨 36 CAPS quads 池, 按 designer trait 激活, 每 PKVC 类 top-1 by |β| + 1 overall top across all classes = 4-5 quads.

实现: `get_activated_quads_balanced()` + `format_caps_quad_balanced_nl(variant='v2')`.

每 quad 长这样:
```
### K 类 top routing (target = K)
[Q3-D] 在【数据效用与美感形态冲突】【数据效用与设计师立场冲突】情境下, 该设计师在「救世主情结」维度上持「救世主情结」立场(β=0.362), 这一特质是其主要思考偏好, 经过一定逻辑权衡后, 顺理成章地将思维引导向 K2-客观数据测试(数据实证主义) 状态.
```

### 1.4 Per-seg JSON schema (3 个)

实现: `build_seg_schema(seg)` in `run_3_baselines_q7.py`.

| Seg | dominant_factor | state_sequence | criterion |
|---|---|---|---|
| seg1 (第一反应) | 单字母 P/K/V/C | 长度 1-3, 通常 [dom] | 可 null |
| seg2 (chain) | 单字母 或 X→Y | 长度 2-7, 强制 multi-state | 可 null |
| seg3 (anchor + criterion) | 单字母 P/K/V/C | 长度 1-3, anchor 段 | **必填** ≤25 字 verbatim |

---

## 2. 7 个 condition 详细 spec (main + 6 ablations)

### Condition 1: **v2_anchor (main)** ★

完整 v2_anchor prompt (§1 全部). `--variant v2`.

### Condition 2: **v2_no_caps** (-CAPS module)

**改动**: 删整个 ⑤ CAPS quad block (β 速览 + detailed routing 都没).

**保留**: SYSTEM (含 seg3 rule + PKVC def) + ① pi + ② stance + ③ self + scenario + task.

注: SEG3_RULE_V2 仍在 SYSTEM, 但没 CAPS β 给 model 看 → rule 失去依据.

**Test**: CAPS routing module 整体必要性. Paper main claim 关键 ablation.

实现: `--variant v2 --modules pi,stance,self`.

### Condition 3: **v2_no_seg3_rule** (-SYSTEM seg3 决策规则)

**改动**: 删 SYSTEM 中 `SEG3_RULE_V2` 整块 (§1.1.3).

**保留**: SYSTEM 其他 (PKVC def + 输出格式) + 完整 USER 5 模块.

**Test**: 显式 seg3 决策规则 vs implicit signal 必要性.

实现: `--variant v2_no_seg3_rule`. SYSTEM 是 `_BASE_SYSTEM` (不带 SEG3_RULE).

### Condition 4: **v2_no_beta_summary** (-β 速览 header)

**改动**: 删 ⑤ CAPS quad block 内的 β 速览 header (§1.3.5.1).

**保留**: 4 类 detailed routing description (§1.3.5.2) 全在.

**Test**: β cheat sheet 是否帮 model 快速把 trait → routing 转成可比较的数值.

实现: `--variant v2_no_beta_summary`. `format_caps_quad_balanced_nl(variant='v2_no_beta_summary')` 跳过 β header.

### Condition 5: **v2_no_pkvc_def** (-SYSTEM PKVC 4 字母定义)

**改动**: 删 SYSTEM 顶部 PKVC 4 字母定义块 (§1.1.1).

**保留**: SYSTEM 其他 (输出格式 + seg3 rule) + 完整 USER 5 模块.

**Test**: PKVC framework primer 必要性. 删了 model 不知道 PKVC 是什么概念.

实现: `--variant v2_no_pkvc_def`. SYSTEM 是 `_BASE_SYSTEM_NO_PKVC_DEF + SEG3_RULE_V2`.

### Condition 6: **v2_plus_peer** (+ ④ peer block)

**改动**: 在 task 后, output instr 前, 加 ④ peer block.

**Peer block 内容**: CAPS-JS top-3 peer designer's Q7 同 seg NL response.

```
# 跟你认知模式相似的设计师在本场景的回答 (供参考)

### Peer 1 (认知模式相似度=0.82)
<peer 1 的 Q7 seg NL, 截至 400 字>

### Peer 2 (认知模式相似度=0.79)
<peer 2 NL>

### Peer 3 (认知模式相似度=0.76)
<peer 3 NL>
```

**Test**: Peer retrieval 是否帮 model 在 chain 上更准 / 增加 differentiation.

实现: `--variant v2 --with-peer`. Peer 检索用 `retrieve_caps_peer_q7_nl()` (跟 B5 ④ peer 同实现).

### Condition 7: **v2_hybrid** (seg3 用 v2_clean prompt)

**改动**: seg1 / seg2 用 v2_anchor (§1 完整). seg3 用 v2_clean:
- SYSTEM: 删 SEG3_RULE_V2 (no seg3 rule)
- ⑤ CAPS quad: 改 **top-k by strength** (top-5 by |β|, 不分类) — 用 `format_activated_quads_nl()` 而非 4 类平衡

**Test**: per-seg prompt dispatch — chain seg 用 anchor push multi-state, anchor seg 让 model 自然 V-default.

实现: `--variant v2_hybrid`. `get_system_prompt_for_seg()` 根据 seg 返回 v2 或 v2_clean SYSTEM.

---

## 3. B0_h ~ B3_h Module Ladder (cumulative add)

跟 v2_anchor SYSTEM 一致 (PKVC def + seg3 rule + 输出格式), USER 逐步加 module:

| Baseline | --modules flag | USER modules |
|---|---|---|
| **B0_h** | `""` (empty) | scenario + task 仅 |
| **B1_h** | `"pi"` | + ① pi |
| **B2_h** | `"pi,stance"` | + ② stance |
| **B3_h** | `"pi,stance,self"` | + ③ self |
| **v2_anchor** | `"pi,stance,self,caps"` (default) | + ⑤ CAPS = full |

实现: `--variant v2_hybrid --modules <list>`. 注: ladder baseline 用 `v2_hybrid` (seg3 退化), 跟 main result table 严格匹配上可能略有差 — 一致性细节见 limitations.

---

## 4. Output 模块对比表

| Module | v2_anchor | -caps | -seg3rule | -βsum | -pkvcdef | +peer | hybrid |
|---|---|---|---|---|---|---|---|
| SYSTEM PKVC def | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ (seg1/2), ✓ (seg3) |
| SYSTEM 输出 format | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SYSTEM seg3 rule | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ (seg1/2), **✗** (seg3) |
| USER ① pi | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| USER ② stance | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| USER ③ self | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| USER ⑤ CAPS β 速览 | ✓ | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ (seg1/2), **✗** (seg3) |
| USER ⑤ CAPS routing | 4 类平衡 | **✗** | 4 类平衡 | 4 类平衡 | 4 类平衡 | 4 类平衡 | 4 类平衡 (seg1/2), top-k strength (seg3) |
| USER ④ peer | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | ✗ |

---

## 5. 完整 v2_anchor prompt 例 (BuzhichunD on seg3, ~2800 chars)

### 5.1 SYSTEM
```
你是设计师认知模拟器. 任务: 根据档案 + 场景, 模拟一个真实设计师对 Q7 某 seg 的回答.

# PKVC 4 类认知状态 (设计师思考过程的 4 种基本状态)
- P (Problem) 问题界定 / 重定义: P1 / P2 / P3
- K (Knowledge) 知识调用 / 数据驱动: K1 / K2 / K3
- V (Value) 价值立场 / 底线宣言: V1 / V2 / V3
- C (Concept) 概念整合 / 抽象框架: C1 / C2 / C3

输出格式 (严格 JSON, ...):
1. dominant_factor: ...
2. state_sequence: ...
3. criterion: ...
4. confidence: ...
5. reasoning_notes: ...
6. nl_answer: ...

要求:
- nl_answer 必须跟 dominant_factor + state_sequence + criterion 内部一致
- 不要在 nl_answer 里 echo dominant/criterion 字面值
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格

## seg3 dominant_factor 判断规则 (重要):
- 看 CAPS routing 4 类 (P/K/V/C) 的正向激活强度 (β > 0)
- 哪类的 β 正向最大 → 你的底线属于那类:
  * P 强 → 底线是 "重新定义问题边界 / 推翻题目假定"
  * K 强 → 底线是 "数据先行 / 调研真实需求"
  * V 强 → 底线是 "用户优先 / 设立专业立场原则"
  * C 强 → 底线是 "整合方案 / 提炼抽象框架"
- 若 4 类 β 强度近似, 看 ③ Q3-Q6 选择倾向哪类
- 不要无脑选 V (V 例子常见), 也不要 over-correct 到 P
```

### 5.2 USER

```
# 你的人口统计学信息 (5 高区分度字段)
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师

# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)
  • 体验换隐私 (数据隐私交换维度)
  • 几何工业派 (仿生与有机维度, 反向立场)
  • 完美主义 (完美主义/MVP 维度)
  • 探索开放派 (秩序与防错维度, 反向立场)
  ... (23 行)

# 你在 4 个设计冲突场景的实际行为 (原始数据)
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案..."
  你的解释: "这或许是产品定位的失误?..."
Q4 面对【商业利润与用户隐私冲突】【隐私交换与概念建构冲突】
  选择: "D. 隐私和开源只是表象..."
  你的解释: "用户其实并不知道什么数据该视为隐私..."
Q5 面对【AI 工具与设计师自主性冲突】【人机协作分工冲突】
  ...
Q6 面对【商业目标与社会责任冲突】【利益与责任感冲突】
  ...

# 你的认知路由模式 (CAPS 四元组, PKVC 4 类平衡 top-1 per class + 1 overall top)

## 4 类 routing 强度速览 (β 值, 正向 → 推动该状态; 负向 → 抑制该状态)
  • P 类: β = -0.380 (负向抑制)
  • K 类: β = +0.362 (正向激活)
  • V 类: β = -0.861 (负向抑制)
  • C 类: β = -0.405 (负向抑制)

→ **正向激活最强**: K 类 (β = +0.362). 优先考虑此类作为你的底线类型.

---
详细 routing 描述:

### P 类 top routing (target = P)
[Q1-B] 在【多维度冲突】... β=-0.38 排斥 P3 状态.
### K 类 top routing (target = K)
[Q3-D] 在【数据效用与美感形态冲突】... β=+0.362 驱动 K2 状态.
### V 类 top routing (target = V)
[Q3-C] 在【数据效用与美感形态冲突】... β=-0.861 排斥 V1→C1 状态.
### C 类 top routing (target = C)
[Q5-B] 在【技术与人本冲突】... β=-0.405 排斥 C2 状态.

# 场景
【超市重构困境】你接到一家零售巨头委托, ...

# 任务
请说出你向管理层提出的核心设计底线 + 可妥协方案... (200-400 字)

## 输出约束 (seg3 — 核心 tradeoff, anchor + 底线段)
- dominant_factor: 单字母 P/K/V/C
- state_sequence: 长度 1-3
- criterion: 必填 ≤ 25 字 verbatim
- nl_answer: 200-400 字, 含 "不可妥协 X / 可妥协 Y" 表达
```

---

## 6. Cross-model 实现 notes

### 6.1 Structured output mode 自动选择

`scripts/run_b6_sample.py::supports_strict_schema()`:
- **OpenAI family** (gpt-4o, gpt-4o-mini, gpt-4.x, o1, o3): json_schema strict
- **Non-OpenAI** (Claude, Gemini, DeepSeek, Qwen, etc.): json_object + 在 system prompt 中 embed schema description, parse + validate retry

实现 `get_response_format_and_aug_system()`:
- Strict 模式: 直接 `{"type": "json_schema", "json_schema": {..., "strict": True}}`
- Fallback 模式: `{"type": "json_object"}` + system prompt augmented with `"严格按以下 JSON Schema 输出"` + embed schema as JSON code block

### 6.2 max_tokens 自适应

- Strict 模式 (OpenAI): max_tokens=900 (够)
- Fallback 模式 (non-OpenAI, schema embed 长): max_tokens=1500 (防 JSON 截断)

### 6.3 Cross-model 跑过的 model IDs

`yunwu.ai` 中转 API + same prompt:
- `gpt-4o` (paper main)
- `gemini-3.1-flash-lite`
- `claude-haiku-4-5-20251001`
- `deepseek-v4-flash`
- `qwen3.6-35b-a3b`
- `qwen3.6-27b`

### 6.4 Concurrency 选择

- OpenAI: concurrency=8 (yunwu 限制宽)
- Non-OpenAI: concurrency=2-4 (避 rate limit)
- Qwen 27B 严重限流, concurrency=1-2

---

## 7. 长度估算 (字符)

| Module | 长度 |
|---|---|
| SYSTEM PKVC def | ~355 |
| SYSTEM 输出 format | ~600 |
| SYSTEM seg3 rule | ~300 |
| USER ① pi | ~120 |
| USER ② stance (23 行 strong) | ~500 |
| USER ③ self (Q3-Q6 each ~150) | ~600 |
| USER ⑤ CAPS β 速览 | ~250 |
| USER ⑤ CAPS 4 类 routing | ~900 |
| USER Q7 scenario | ~200 |
| USER per-seg task + output instr | ~180 |
| **v2_anchor full SYSTEM+USER** | **~4000 chars** |

各 ablation 长度差异:
- -PKVC def: -355 chars → ~3645
- -seg3 rule: -300 → ~3700
- -β 速览: -250 → ~3750
- -CAPS module: -1150 (β 速览 + routing) → ~2850
- +peer: +1200 (3 peer × 400 chars) → ~5200
- hybrid (seg3): seg3 用 v2_clean format → ~3500 (top-k strength 替 4 类平衡)
- B0_h (modules empty): -2120 (无 ①②③⑤) → ~1880

---

## 8. Cross-Model main result (v2_anchor vs B0) 验证 paper claim

跨 5 model 跑完 7 condition × 162 designer × 3 seg = 17,010 inference:

| Metric | GPT-4o | Gemini 3.1F | Claude H4.5 | DSv4F | Qwen35B | 一致? |
|---|---|---|---|---|---|---|
| macro-TED ↓ | -0.368 ✓ | -0.153 ✓ | -0.125 ✓ | -0.184 ✓ | -0.289 ✓ | **5/5** |
| macro-LCS-F1 ↑ | +0.020 ✓ | +0.092 ✓ | +0.042 ✓ | +0.095 ✓ | +0.076 ✓ | **5/5** |
| chain diversity ↑ | +0.160 ✓ | +0.284 ✓ | +0.654 ✓ | +0.164 ✓ | +0.256 ✓ | **5/5** |
| non-V recall ↑ | +0.272 ✓ | -0.210 ✗ | +0.468 ✓ | +0.153 ✓ | +0.229 ✓ | 4/5 |

**18/20 cells 一致** → v2_anchor 跨 model generalization 强证据.

Aggregate rank (5 model × 4 metric = 20 cells):
- **v2_anchor**: mean rank 2.40 ⭐ (best of 6 v2-family variants)
- +peer: 2.60 (variable, peaks but troughs)
- 其他 ablation: 3.30~4.55 (consistently worse)

→ v2_anchor 是 **most robust** prompt choice across model × metric.

---

## 9. Limitation: GPT-4o-tuned components

Component ablations (单删 PKVC def / seg3 rule / β 速览 / CAPS module) 仅在 **GPT-4o** 上显著 hurt; 其他 model 上影响 mixed:

- **GPT-4o**: 删 CAPS module → macro-TED +32% (★ sig CI 不重叠)
- **Gemini / Claude / DeepSeek / Qwen**: 删某些 component 反而 better

诊断:
- v2_anchor 各 component 是 **GPT-4o-optimized** (explicit rule + structured anchor 适合 instruction-tuned strong model)
- 小 / 中 model 上 simpler prompt 也 work
- v2_anchor 仍是 robust default (mean rank 最低)

→ Paper 不应 claim "component X universally necessary", 应 claim "**v2_anchor robust prompt family > zero-shot, cross-model generalized**".

---

## 10. 一句话总结

```
v2_anchor 是 paper main prompt (PKVC def + 输出 spec + seg3 rule + 5 字段 pi
+ 30 维 stance + Q3-Q6 face-conflict + CAPS 4 类平衡 + β 速览, ~4000 chars).
6 个 ablation (–PKVC def / –seg3 rule / –β 速览 / –CAPS / +peer / hybrid)
+ B0_h~B3_h ladder. 跨 5 model 验证: macro-TED / chain diversity 5/5 一致改善,
非 V recall 4/5 一致改善, aggregate rank 2.40 (best of 6 variants).
SFT (Qwen2.5-7B + LoRA on Q1-Q6 v2_anchor format) 待跑.
```
