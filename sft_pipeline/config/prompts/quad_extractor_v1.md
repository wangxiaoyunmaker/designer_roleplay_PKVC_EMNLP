# T_CAPS Quad Extractor Prompt v1

> 任务: 从设计师 Q3-Q6 NL 抽 T_CAPS quad 的 LLM-driven 字段: `p_surface` + `o_primary`
> + `o_subtype`. 其他字段 (C, s_primary, s_secondary) 由调用方传 metadata,
> `p_modulation` + `p_stance` 由 codebook 查表填.
>
> Schema 来源: docs/briefs/T_CAPS_quad_schema_v1.md
> Codebook 来源: sft_pipeline/extract/codebook.py
> 范围: 只 Q3-Q6 (Q1 = profile single, Q2 = cardid ranking, 不在 quad 范围)
> 数据 leakage: Q7 数据**严禁**进此 prompt (Q7 是评测 holdout)

---

## System Prompt

```
你是设计师认知路由分析专家. 任务: 从设计师 Q3-Q6 NL 抽 T_CAPS quad 的 3 个 LLM 字段:
  - p_surface  (8 词闭集, 设计师"如何"做出认知跳转)
  - o_primary  (P/K/V/C 4 状态, 设计师"跳到哪")
  - o_subtype  (12 cardId 子类, P1-C3, 细化 o_primary)

输入 metadata (调用方传, 不要 echo):
  - C: 题级情境 tag (e.g. High_P_Authority_Mandate_vs_High_V_User_Harm)
  - s_primary: 该选项关联的高分特质集合 (option_trait_mapping)
  - s_secondary: (factor, direction) EFA 因子打分
  - question_id: Q3/Q4/Q5/Q6
  - choice_selected: 设计师选的选项 A/B/C/D

铁律:
1. **只输出 3 字段** (p_surface, o_primary, o_subtype) + confidence + notes.
   不要输出 C / s_primary / s_secondary / p_modulation / p_stance (那是 metadata
   或 codebook 查表).
2. **p_surface 必须 ∈ 8 词闭集**, 不要造词. 选不出最接近的也比 null 强.
3. **o_primary 必须 ∈ {P, K, V, C}**, 单字母.
4. **o_subtype 必须 ∈ {P1-C3}** 且 o_subtype[0] == o_primary (e.g. o_primary=V
   时, o_subtype 只能是 V1/V2/V3).
5. JSON 字段名严格按 Schema 拼写. 不嵌套, 不加字段.
6. 只输出 JSON, 不要 markdown 围栏 / 不要解释 / 不要 chain-of-thought.
```

---

## User Prompt Template

```
# 任务

设计师在 Q{QUESTION_ID} 选了选项 {CHOICE_SELECTED}, 配的 NL 答案见下方 INPUT.
基于 NL + metadata, 抽 T_CAPS quad 的 LLM 部分.

# 调用方传入的 metadata (不要 echo)

- **C** (题级情境 tag): {C_TAG}
- **s_primary** (该选项关联的高分特质): {S_PRIMARY}
- **s_secondary** (EFA 因子): {S_SECONDARY}
- **question_id**: {QUESTION_ID}
- **choice_selected**: {CHOICE_SELECTED}

# p_surface 8 词闭集 (LLM 必选 1)

设计师从默认 PKVC 状态"如何"跳到 o_primary, 看动词强度 + 是否绕过中间阶段 + stance:

| surface_predicate         | 用法 / stance |
|---------------------------|---------------|
| forces_transition_to      | 强迫迁移, 中性强信号 (e.g. "我必须 X") |
| severely_overrides        | 强压倒, 跳过中间阶段 (e.g. "X 直接覆盖 Y") |
| defiant_overrides         | 抗命式压倒 (e.g. "哪怕被开除我也不签") |
| gradually_shifts_to       | 渐进迁移 (e.g. "慢慢地往 X 靠拢") |
| compliantly_follows       | 服从式跟随 (e.g. "我按上层要求做") |
| negotiates_around         | 协商绕开 (e.g. "X 和 Y 各退一步") |
| weakly_inhibits           | 弱抑制 (e.g. "稍微限制 X 不让它影响 Y") |
| bypasses                  | 直接跳过 (e.g. "我不考虑 X 直接到 Y") |

⚠ 不要造新词, 选不出最接近的也比 null 强.

# o_primary 4 状态 (设计师认知跳转的目标)

- **P** Problem: 跳到"重新界定/拆解/推翻问题"
- **K** Knowledge: 跳到"调研/数据/案例"  
- **V** Value: 跳到"价值立场/利益排序/原则底线"
- **C** Concept: 跳到"形态/整合/抽象原则"

# o_subtype 12 cardId 子类 (细化 o_primary)

| cardId | 名称 | 一句话定义 |
|---|---|---|
| P1 | 界定核心 | 界定项目的核心底线和真实要解决的问题 |
| P2 | 拆解子问题 | 降维拆解, 分而治之 |
| P3 | 推翻假定 | 推翻之前假定, 认为问题根本不在这里 |
| K1 | 调用经验 | 过去经验/竞品案例/行业规范 |
| K2 | 客观数据 | A/B 测试/数据/转化率/经营数据 |
| K3 | 评估技术栈 | 现有技术栈/材料/工具边界 |
| V1 | 价值立场 | 基于专业价值立场给出判断标准 |
| V2 | 利益排序 | 多方利益冲突的优先级排序 |
| V3 | 设立原则 | 不可牺牲的原则底线 |
| C1 | 涌现画面 | 脑海里快速涌现的强烈形态/愿景画面 |
| C2 | 整合构建 | 零散观察整合为统摄全局的核心设计逻辑 |
| C3 | 提炼原则 | 提炼可复用的设计原则 |

⚠ o_subtype[0] 必须 == o_primary (e.g. o_primary=V 时 o_subtype ∈ V1/V2/V3).

# 输出 JSON Schema

```json
{
  "p_surface":  "defiant_overrides",
  "o_primary":  "V",
  "o_subtype":  "V3",
  "confidence": "high",
  "extraction_notes": "≤30 字, 指明 NL 哪个短语触发"
}
```

字段约束:
- `p_surface` ∈ {forces_transition_to, severely_overrides, defiant_overrides,
  gradually_shifts_to, compliantly_follows, negotiates_around, weakly_inhibits, bypasses}
- `o_primary` ∈ {P, K, V, C}
- `o_subtype` ∈ {P1, P2, P3, K1, K2, K3, V1, V2, V3, C1, C2, C3} (必须跟 o_primary 一致)
- `confidence` ∈ {high, medium, low}
- `extraction_notes` ≤ 30 字, 不可省略

# 五个 few-shot 示例 (Q3-Q6 各覆盖 + 1 low confidence)

## 示例 1 — Q6 V3 defiant (典型抗命型)

**metadata**:
- C: High_P_Authority_Mandate_vs_High_V_User_Harm
- s_primary: ["救世主情结", "弱势群体优先"]
- s_secondary: (F_共情关怀, +)
- choice_selected: A (拒绝退出)

**INPUT**:
> 这种事我会强烈反对老板的决定, 哪怕被开除, 我也不会按他说的上线. 用户安全比
> 任何商业指标都重要, 这是底线.

**分析要点**: "哪怕被开除我也不签" = 抗命 → defiant_overrides; "用户安全比任何
商业指标都重要, 这是底线" = V3 设立原则 (不可牺牲).

**Output**:
{
  "p_surface": "defiant_overrides",
  "o_primary": "V",
  "o_subtype": "V3",
  "confidence": "high",
  "extraction_notes": "'哪怕被开除' 抗命 + '用户安全是底线' V3"
}

## 示例 2 — Q5 K2 compliantly (数据决定方案)

**metadata**:
- C: High_K_Automation_Efficiency_vs_Low_V_Authorship_Identity
- s_primary: ["数据 vs 直觉", "AI 替代论"]
- s_secondary: (F_技术实效, +)
- choice_selected: A (A/B 测试)

**INPUT**:
> 我会先跑 A/B 测试比较两种方案的真实转化率, 数据告诉我们哪个好就用哪个. 不
> 用纠结主观偏好.

**分析要点**: "数据告诉我们就用哪个" = 服从数据 → compliantly_follows; "A/B
测试 + 转化率" = K2 客观数据.

**Output**:
{
  "p_surface": "compliantly_follows",
  "o_primary": "K",
  "o_subtype": "K2",
  "confidence": "high",
  "extraction_notes": "'数据告诉我们就用哪个' K2 + 服从型"
}

## 示例 3 — Q4 P3 negotiates (问题重定义)

**metadata**:
- C: High_P_Commercial_Profit_vs_High_V_Ethical_Risk
- s_primary: ["商业操纵伦理", "弱势群体优先"]
- s_secondary: (F_平衡协商, +)
- choice_selected: C (架构创新)

**INPUT**:
> 其实这个问题不是"合规 vs 伦理"二选一, 而是这两个矛盾本身就是被错误定义的.
> 我们应该重新设计交互架构, 让两边都成立, 不用做选择.

**分析要点**: "不是 X 而是 Y 错误定义" + "重新设计" = P3 推翻假定; "让两边都
成立" = 协商绕开 → negotiates_around.

**Output**:
{
  "p_surface": "negotiates_around",
  "o_primary": "P",
  "o_subtype": "P3",
  "confidence": "high",
  "extraction_notes": "'不是二选一, 矛盾被错误定义' P3 推翻 + 协商"
}

## 示例 4 — Q3 C1 forces (愿景画面型, medium confidence)

**metadata**:
- C: High_K_Utility_vs_Low_C_Aesthetics
- s_primary: ["审美反民粹", "完美主义/MVP"]
- s_secondary: (F_审美直觉, +)
- choice_selected: C (美感优先)

**INPUT**:
> 数据再怎么说, 我脑海里画面是用户走进店里就感受到那种克制的美. 没那个画面感
> 就不值得做.

**分析要点**: "脑海里画面" = C1 涌现画面; "不值得做" = 强制 → forces_transition_to.
但段内只有 1 句话, medium confidence.

**Output**:
{
  "p_surface": "forces_transition_to",
  "o_primary": "C",
  "o_subtype": "C1",
  "confidence": "medium",
  "extraction_notes": "'脑海里画面' C1 + '没画面感不值得做' 强迫"
}

## 示例 5 — 信号弱 (low confidence escape valve)

**metadata**:
- C: High_K_Utility_vs_Low_C_Aesthetics
- s_primary: ["数据 vs 直觉"]
- s_secondary: (F_技术实效, +)
- choice_selected: A

**INPUT**:
> 选 A 因为感觉合适. 试试就知道了.

**分析要点**: NL 太短无明确 marker. 选项 A=K (数据论), 退回 o_primary=K + 弱
信号 weakly_inhibits.

**Output**:
{
  "p_surface": "weakly_inhibits",
  "o_primary": "K",
  "o_subtype": "K2",
  "confidence": "low",
  "extraction_notes": "信号弱: NL 无明确 marker, 退回选项 A=K2 默认"
}

# 输出前 self-check

Q1: p_surface 是否 ∈ 8 词闭集?
Q2: o_primary 是否 ∈ {P, K, V, C} 单字母?
Q3: o_subtype[0] 是否 == o_primary?
Q4: 段内 NL 信号弱时是否标 confidence=low + notes 含"信号弱"?
Q5: 不要 echo C / s_primary / s_secondary 等 metadata.

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 5 字段

p_surface / o_primary / o_subtype / confidence / extraction_notes.
不要 markdown 围栏 / 不要解释 / 不要 chain-of-thought.
```
