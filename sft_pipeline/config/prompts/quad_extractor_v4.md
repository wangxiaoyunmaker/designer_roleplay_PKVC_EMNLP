# T_CAPS Quad Extractor Prompt v4 (加 sadr_activations)

> v4 vs v3: 加 sadr_activations 字段, 抽 designer 在该段表现的具体 SADR 激活点
> (dual-form, e.g. "用户福祉优先 (#V3+)"). 给 β+++ output 的 [认知激活] 段供数据.

> v3 vs v2 改动 (基于 v2 full 316 case 发现):
> 1. **o_subtype 允许复合 X→Y** — 前置标注 metadata 里 `p➡️k` 是"存在引起/动态关系", v2
>    当 ambiguity 处理选起点 X 是错的. v3 schema 加复合形式 (e.g. P3→K3, V1→C2),
>    o_subtype 可单可复合, 终点 Y prefix 必须 == o_primary.
> 2. **Hint usage 重写** — v2 "复合 hint 跟 NL 自由定" 改为 "复合 hint 直接保留 X→Y
>    形式作 o_subtype". NL 矛盾时 override.
> 3. **加复合 few-shot example** — 1 例展示 X→Y 输出格式.
>
> Schema 来源: docs/briefs/T_CAPS_quad_schema_v1.md
> Codebook 来源: sft_pipeline/extract/codebook.py
> 范围: 只 Q3-Q6 (Q1=profile, Q2=cardid ranking, 不在 quad 范围)
> 数据 leakage: Q7 数据**严禁**进此 prompt

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
2. **p_surface 必须 ∈ 8 词闭集**, 不要造词.
3. **o_primary 必须 ∈ {P, K, V, C}**, 单字母.
4. **o_subtype 允许两种形式**:
   - 单值: P1/P2/P3/K1/K2/K3/V1/V2/V3/C1/C2/C3 (12 闭集), 终点 == 单值自己
   - 复合: X→Y (e.g. "P3→K3", "V1→C2"), 表示设计师从 X 状态被引起转移到 Y 状态
   两种形式终点 (单值自己 / 复合的 Y) 都必须跟 o_primary 一致前缀.
5. JSON 字段名严格. 只输出 JSON, 不要 markdown 围栏 / 解释 / chain-of-thought.

★ Anti-bias 铁律 (v2 新加, 修 v1 LLM 默认中位档问题):
6. **12 cardId 都是 valid 选项, 不要默认选 V2/C2/K2 中位档**. V1/V3/C1/C3/K1/K3
   全部应该按 NL 信号公平选. v1 smoke 67 case 里 LLM 100% 选 *2 (V2/C2/K2) 是 bias,
   v2 必须打破.

★ Hint usage 铁律 (v2 新加):
7. **NL 信号优先, hint 作 fallback default**. 调用方会传 o_primary_hint + o_subtype_hint:
   - NL 信号明确 + 跟 hint 一致 → 跟 hint, confidence=high
   - NL 信号明确 + 跟 hint 矛盾 → 跟 NL (信号强为准), notes 注明 "NL override hint"
   - NL 信号弱 + 有 hint → 跟 hint, confidence=medium
   - NL 信号弱 + 无 hint → 自己推, confidence=low + notes 标"信号弱"
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
- **o_primary_hint** (前置标注的 PKVC 主导): {O_PRIMARY_HINT}
- **o_subtype_hint** (前置标注的子类): {O_SUBTYPE_HINT}

⭐ Hint format 说明 (v3 重写):
- 单字母 `V`/`K`/`P`/`C` → o_primary 已定, o_subtype 你按 NL 推 (3 选 1 在该大类内)
- 小写 `k2`/`v3`/`p1` → o_primary+o_subtype 都已定 (单值形式), 跟 hint 除非 NL 信号矛盾
- 复合 `p3➡️k3`/`v1➡️c1` → 前置标注的**动态关系** (X 引起 Y 转移). 你必须**保留复合形式**
  作 o_subtype 输出 "X→Y" (用 → 不是 ➡️). 终点 Y 决定 o_primary (e.g. p3➡️k3 →
  o_primary=K, o_subtype="P3→K3"). NL 矛盾时 override.
- `None` → 无 hint, 完全靠 NL 推

⭐ 看 hint 时记住: hint 是来自统计学习的"该选项共同特质", 不是该 designer 的特质.
NL 是该 designer 真实写的, 信号更强. 矛盾时优先 NL.

⭐ 复合 (X→Y) 表达原则:
- X→Y 表示设计师认知状态有"起点 X, 终点 Y"的转移. 不是 X 或 Y 二选一.
- 终点 Y 是 designer 最终的认知 commit state, 用作 o_primary.
- 起点 X 是 trigger (源 trait 或 cognitive entry).
- 输出 o_subtype 时用 ASCII 箭头 → 不用 emoji ➡️.

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

# sadr_activations 字段 (v4 新加)

抽 designer 在该段表现的具体 SADR (Schema-Anchored Design Reasoning) 激活点.
输出 1-4 个激活点, 每个含: cardId + 极性 (+/-) + 中文 label.

格式: list of {"id": "#V3+", "label": "用户福祉优先"}

- id 形式: `#cardId极性`, e.g. "#V3+", "#P1-"
- label 是该激活点的中文 designer-voice 表达 (≤10 字)
- 1-4 个 (1 = 单激活, 4 = 复合激活), 不要 0 个 (信号弱时给 1 个最强)
- 极性 + 表示赞同/激活, - 表示反对/抑制

# 输出 JSON Schema

```json
{
  "p_surface":  "defiant_overrides",
  "o_primary":  "V",
  "o_subtype":  "V3",
  "sadr_activations": [
    {"id": "#V3+", "label": "用户安全是底线"},
    {"id": "#P1-", "label": "权威指令受抑"}
  ],
  "confidence": "high",
  "extraction_notes": "≤30 字, 指明 NL 哪个短语触发"
}
```

字段约束:
- `p_surface` ∈ 8 词闭集
- `o_primary` ∈ {P, K, V, C}
- `o_subtype` ∈ 12 单 + 7 复合 (X→Y), 终点 prefix 必须跟 o_primary 一致
- `sadr_activations`: 1-4 个 dict {id, label}; id 形式 #cardId极性 (e.g. #V3+, #P1-)
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
  "sadr_activations": [
    {"id": "#V3+", "label": "用户安全是底线"},
    {"id": "#P1-", "label": "权威指令受抑"}
  ],
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
  "sadr_activations": [
    {"id": "#K2+", "label": "数据决定方案"},
    {"id": "#V1-", "label": "主观偏好受抑"}
  ],
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
  "sadr_activations": [
    {"id": "#P3+", "label": "重定义二元对立"},
    {"id": "#C2+", "label": "架构整合"}
  ],
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
  "sadr_activations": [
    {"id": "#C1+", "label": "脑海里画面感"},
    {"id": "#K2-", "label": "数据让步"}
  ],
  "confidence": "medium",
  "extraction_notes": "'脑海里画面' C1 + '没画面感不值得做' 强迫"
}

## 示例 5 — Q4 V1 weakly_inhibits (个人价值立场, NL override hint)

**metadata**:
- C: KP_BusinessPrivacy_vs_V_Ethics_vs_C_Construct
- s_primary: ["商业操纵伦理", "弱势群体优先"]
- choice_selected: B
- **o_primary_hint**: K  (统计预标 K, 但实际 NL 偏 V)
- **o_subtype_hint**: k1

**INPUT**:
> 我个人觉得这事不太对劲. 我相信用户隐私是设计师的责任, 不能为了商业便利出卖.
> 数据再多也支持不了不道德的决策.

**分析要点**: NL 反复"我个人觉得 / 我相信 / 是设计师的责任" = **V1 价值立场** (主观判断);
"不能为了 X 出卖" = weakly_inhibits; "数据再多也支持不了" → 不是 K 主导. NL 信号
跟 hint=K 矛盾, **跟 NL** (V1 主导).

**Output**:
{
  "p_surface": "weakly_inhibits",
  "o_primary": "V",
  "o_subtype": "V1",
  "sadr_activations": [
    {"id": "#V1+", "label": "主观伦理立场"},
    {"id": "#K2-", "label": "数据无法支持不道德"}
  ],
  "confidence": "high",
  "extraction_notes": "V1 价值立场 (我个人/我相信), NL override hint=K"
}

## 示例 6 — Q3 K1 compliantly (调用历史经验)

**metadata**:
- C: K_ObjectiveData_vs_CV_Aesthetic
- s_primary: ["天赋信仰", "团队依恋"]
- choice_selected: A
- o_primary_hint: K, o_subtype_hint: k1

**INPUT**:
> 这种事我看以前 X 项目类似情况怎么做的. 行业里一般是先做小范围试点, 再 scale.
> 我会参考竞品 Y 的做法.

**分析要点**: "以前 X 项目 / 行业一般 / 参考竞品 Y" = **K1 调用经验** (不是 K2 数据);
"我会参考" = compliantly_follows. 跟 hint=K/k1 一致.

**Output**:
{
  "p_surface": "compliantly_follows",
  "o_primary": "K",
  "o_subtype": "K1",
  "sadr_activations": [
    {"id": "#K1+", "label": "调用历史经验"},
    {"id": "#K2-", "label": "数据是工具非主"}
  ],
  "confidence": "high",
  "extraction_notes": "K1 调用经验 (历史项目/竞品), 跟 hint 一致"
}

## 示例 7 — Q5 C3 forces (提炼可复用原则)

**metadata**:
- C: K_TechScale_vs_C_Organic_vs_V_HumanCentric_vs_P_Core
- s_primary: ["复杂工具信仰", "完美主义/MVP"]
- choice_selected: C

**INPUT**:
> 这件事让我总结一条原则: 凡是新技术介入, 必先建立 fallback 机制. 这是可以推广
> 到所有 AI 设计的通用 design principle.

**分析要点**: "总结一条原则 / 可以推广到所有 X 的通用原则" = **C3 提炼可复用原则**
(不是 C1 画面 也不是 C2 整合); "必先" = forces_transition_to.

**Output**:
{
  "p_surface": "forces_transition_to",
  "o_primary": "C",
  "o_subtype": "C3",
  "sadr_activations": [
    {"id": "#C3+", "label": "提炼可复用原则"},
    {"id": "#K3+", "label": "fallback 机制必先"}
  ],
  "confidence": "high",
  "extraction_notes": "C3 提炼原则 (总结 / 可推广 / 通用 principle)"
}

## 示例 8 — Q6 V2 negotiates (多方利益排序)

**metadata**:
- C: V_UserResponsibility_vs_CPK_Commercial
- s_primary: ["本土文化妥协", "AI 替代论", "商业操纵伦理"]
- choice_selected: B
- o_primary_hint: V, o_subtype_hint: v1➡️c2

**INPUT**:
> 我同意商业目标重要, 但坚持用户权益不能完全牺牲. 我会在两者之间找平衡点, 给
> 商业方让 70%, 给用户保 30% 退路.

**分析要点**: "我同意 X 但坚持 Y" + "在两者之间找平衡点" + 显式比例 (70/30) =
**V2 利益排序** (不是 V1 主观立场 也不是 V3 不可妥协); "找平衡点" = negotiates_around.

**Output**:
{
  "p_surface": "negotiates_around",
  "o_primary": "V",
  "o_subtype": "V2",
  "sadr_activations": [
    {"id": "#V2+", "label": "多方利益排序"},
    {"id": "#V3-", "label": "妥协 30% 退路"}
  ],
  "confidence": "high",
  "extraction_notes": "V2 利益排序 (同意 X 但坚持 Y, 70/30 比例)"
}

## 示例 9 — Q4 K1→P1 复合 (从经验调用引起重新界定核心)

**metadata**:
- C: KP_BusinessPrivacy_vs_V_Ethics_vs_C_Construct
- s_primary: ["商业操纵伦理", "AI 替代论", "数据隐私交换"]
- choice_selected: A
- o_primary_hint: K
- **o_subtype_hint**: 'k1➡️p1'   ← 复合, 前置标注的动态关系

**INPUT**:
> 我会先调阅合规法案和竞品隐私协议这类历史经验, 然后用这些信息重新界定"我们到底
> 能做什么"这个核心边界. 不是争论, 是用法律和先例画出可行域.

**分析要点**: "调阅历史经验" = K1 起点; "重新界定核心边界 / 用先例画出可行域" =
P1 终点. 跟 hint k1➡️p1 一致, 直接保留复合形式. o_primary=P (终点 P1 的 prefix).

**Output**:
{
  "p_surface": "compliantly_follows",
  "o_primary": "P",
  "o_subtype": "K1→P1",
  "sadr_activations": [
    {"id": "#K1+", "label": "调用合规法案"},
    {"id": "#P1+", "label": "重新界定可行域"}
  ],
  "confidence": "high",
  "extraction_notes": "K1 起点 (调阅经验) → P1 终点 (重新界定), 跟 hint 复合"
}

## 示例 10 — 信号弱 (low confidence escape valve)

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
  "sadr_activations": [
    {"id": "#K2+", "label": "选项默认 K2"}
  ],
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
