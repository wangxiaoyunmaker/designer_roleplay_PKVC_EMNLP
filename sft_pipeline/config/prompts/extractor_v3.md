# PKVC + SADR + T_CAPS Quad 提取器 Prompt v3 (WIP, 等学姐 metadata)

> v3 改动 (相对 v2):
> - **A**: 输出 schema 加 T_CAPS quad 字段 (`p_surface` / `p_modulation` / `o_primary` / `o_subtype`)
>   对接 docs/briefs/T_CAPS_quad_schema_v1.md (v1.1 锁定 2026-05-23)
> - **B**: 8 词 `p_surface` 封闭词表 + 多对一映射到 5 档 `p_modulation`
> - **C**: 12 cardId `o_subtype` 封闭枚举 (P1-P3, K1-K3, V1-V3, C1-C3, 学姐 metadata 来了填语义)
> - **D**: `C_local` 由调用方在 metadata 里给 (不让 LLM 推, 节省 budget + 保证闭集)
> - **E**: `s_primary` / `s_secondary` 也由调用方传入 (从 30 维量表 + EFA 算, 不依赖 NL)
> - **F**: LLM 只负责输出 p_surface + o_primary + o_subtype + 原 v2 字段
>
> ⚠ 当前状态: WIP, schema 锁了, 但 12 cardId 语义定义 + p_surface 边界 case 等学姐 reply.
> ⚠ 学妹 D 包 (κ pilot) 当前还跑 v2, 等学姐 ① 回了再切 v3.

---

## System Prompt

```
你是一位设计师认知分析专家. 任务: 从设计师的思维链段落中提取
(1) PKVC + SADR 信号 (v2 兼容),
(2) T_CAPS quad 字段 (p_surface 路由谓语 + o_primary/o_subtype 目标状态).
严格按规则提取, 只输出 JSON, 不要任何解释 / 不要 markdown 围栏 / 不要 CoT 痕迹.
```

---

## User Prompt Template

```
# 任务

从下方设计师思维链段落中提取 PKVC + SADR + T_CAPS quad 字段. 严格输出 JSON.

# 输入 metadata (由 extractor_batch.py 自动填充, 不要重新推断)

- segment_role: {SEGMENT_ROLE}              ← 第一反应 | 决策链路 | 核心 tradeoff
- scenario:     {SCENARIO}                  ← e.g. "Q7_老城区超市重构"
- C_local:      {C_LOCAL_TAG}               ← 题级情境 tag (由 situation_codebook 查表填入)
- task_background: {TASK_BACKGROUND}         ← 50-100 字全局背景 (等学姐补)
- s_primary:    {S_PRIMARY_TAGS}            ← 该设计师 30 维 high-score tag 集合 (调用方算好)
- s_secondary:  {S_SECONDARY_FACTOR}        ← (factor, direction), e.g. "(F_共情关怀, +)"

# T_CAPS 输出字段 (v1.1 锁定, 见 docs/briefs/T_CAPS_quad_schema_v1.md)

## p_surface (路由谓语, 闭集 8 词)

| surface_predicate         | 对应 modulation | 隐含 stance      |
|---------------------------|-----------------|------------------|
| forces_transition_to      | Strong+         | neutral          |
| severely_overrides        | Strong+ Bypass  | neutral          |
| defiant_overrides         | Strong+ Bypass  | Defiance (抗命)  |
| gradually_shifts_to       | Weak+           | neutral          |
| compliantly_follows       | Weak+           | Compliance (服从) |
| negotiates_around         | Weak+ / Weak-   | Negotiation (协商)|
| weakly_inhibits           | Weak-           | neutral          |
| bypasses                  | Bypass          | neutral          |

**只能从上面 8 个词选 1 个**, 不要造新词. 看不出来就选最接近的; 不一致写 confidence=low.

## o_primary (目标 PKVC 状态, 闭集 4)

```
P  Problem Definition       界定 / 拆解 / 真实需求
K  Knowledge Activation     调用经验 / 数据 / 案例
V  Value Judgment           底线 / 价值排序 / 伦理
C  Commitment Action        概念形态 / 设计逻辑 / 整合
```

## o_subtype (PKVC 子类, 闭集 12, v1.1 attribute)

⚠ 12 cardId 语义定义等学姐 metadata. 当前 placeholder, 学姐 confirm 后填:

| cardId | 名称 | 一句话定义 |
|---|---|---|
| P1 | (TBD) | 学姐补 |
| P2 | (TBD) | 学姐补 |
| P3 | (TBD) | 学姐补 |
| K1 | (TBD) | 学姐补 |
| K2 | (TBD) | 学姐补 |
| K3 | (TBD) | 学姐补 |
| V1 | (TBD) | 学姐补 |
| V2 | (TBD) | 学姐补 |
| V3 | (TBD) | 学姐补 |
| C1 | (TBD) | 学姐补 |
| C2 | (TBD) | 学姐补 |
| C3 | (TBD) | 学姐补 |

学姐数据来了之前, o_subtype 可以输出 null (合法), 不要瞎猜.

# SADR + PKVC 4 因子框架 (v2 兼容, 完整内容见 extractor_v2.md)

(略, v2 内容不变, 在 v3 跑通后从 v2 ctrl+C 过来)

# 输出 JSON Schema (v3, 在 v2 基础上加 quad 字段)

```json
{
  "segment_role":      "第一反应" | "决策链路" | "核心 tradeoff",
  "dominant_factor":   "<P|K|V|C>" 或路径 "<P→K→V>",
  "sadr_activations":  [ ... v2 schema 不变 ... ],
  "criterion":          "<短语>" | null,
  "criterion_strength": "不可妥协" | "可妥协" | null,
  "confidence":         "high" | "medium" | "low",
  "extraction_notes":   "<一句话>",

  // ─── v1.1 quad 字段 ───
  "p_surface":          "<上面 8 词之一>" | null,
  "o_primary":          "P" | "K" | "V" | "C" | null,
  "o_subtype":          "P1" | ... | "C3" | null
}
```

# 提取规则 (v3 新增, 在 v2 之外加)

1. **p_surface 选词**:
   - 看动词强度 + 是否绕过默认 PKVC 阶段 + stance 色彩
   - "强烈反对 / 必须 / 绝对" 类 + 跳过中间 → defiant_overrides
   - "顺从 / 执行 / 接受" → compliantly_follows
   - "协商 / 折中 / 平衡" → negotiates_around
   - "渐进 / 逐步 / 慢慢" → gradually_shifts_to
   - 不强烈但有方向 → forces_transition_to OR weakly_inhibits
   - 直接跳过中间阶段 → bypasses

2. **o_primary 跟 dominant_factor 关系**:
   - 如果 dominant_factor 是单字母 (V), o_primary = 该字母
   - 如果 dominant_factor 是路径 (V → K → V), o_primary = 路径终点 (V)
   - 异常 case: dominant_factor 是 null → o_primary 也 null

3. **o_subtype 推断** (等学姐 cardId metadata 后启用):
   - 从 sadr_activations 里找最显著的 cardId (V1 / V2 / V3 之一)
   - 跟 o_primary 字段一致 (o_primary=V → o_subtype 必须 ∈ {V1, V2, V3})
   - 不确定 → 输出 null, 不要瞎猜

# Self-check (输出前内部 verify, 不写进 JSON)

```
Q1 (p_surface ∈ 8 词闭集): 是不是从 forces_transition_to / severely_overrides /
   defiant_overrides / gradually_shifts_to / compliantly_follows /
   negotiates_around / weakly_inhibits / bypasses 选的?
Q2 (o_primary ↔ dominant_factor 一致): o_primary 是否跟 dominant_factor 终点对得上?
Q3 (o_subtype ↔ o_primary 字段一致): o_subtype[0] == o_primary?
   (o_subtype=V1, o_primary 必须 V; 否则降 confidence)
```

# Few-shot 示例 (v3 quad 字段新增, v2 SADR 字段从 v2 ctrl+C)

## 示例 1 — Q6 决策链路, defiant_overrides

**Input metadata**:
- segment_role: 决策链路
- scenario: Q6_老板上线指令
- C_local: High_P_Authority_Mandate_vs_High_V_User_Harm
- s_primary: ["救世主情结", "弱势群体优先"]
- s_secondary: (F_共情关怀, +)

**Input text**:
> 这种事我会强烈反对老板的决定, 哪怕被开除, 我也不会按他说的上线.
> 用户安全比任何商业指标都重要, 这是底线.

**Output**:
```json
{
  "segment_role": "决策链路",
  "dominant_factor": "V",
  "sadr_activations": [
    {"id": "#V3", "polarity": "+", "evidence": "用户安全比任何商业指标都重要, 这是底线",
     "evidence_offset_hint": "段尾"}
  ],
  "criterion": null,
  "criterion_strength": null,
  "confidence": "high",
  "extraction_notes": "强烈反对 + 跳过权衡阶段直接到价值, 经典 defiant_overrides",

  "p_surface": "defiant_overrides",
  "o_primary": "V",
  "o_subtype": null
}
```

(o_subtype null 因为学姐 V1/V2/V3 cardId 语义还没回; 学姐数据齐时填 V3)

## 示例 2 — Q5 第一反应, gradually_shifts_to

(等学姐金标 NL 后补, 当前 placeholder)

---

# 输入文本

```
{INPUT_TEXT}
```

只输出 JSON, 严格遵循 schema. 不要 markdown 围栏, 不要解释, 不要 chain-of-thought.

---

# 阻塞项 (学姐回复后填充, 然后 v3 锁 v3.1)

| 项 | 状态 |
|---|---|
| 12 cardId 完整语义定义 (P1-C3) | ⚠ 等学姐 ③ |
| Q1, Q2, Q7, Q8 的 C_local tag | ⚠ 等学姐 ① + Q4 字段归属 ② |
| Q3-Q6 选项-因子 mapping (P1/P3 精确 cardId) | ⚠ 等学姐 ④ |
| 50-100 字 task background | ⚠ 等学姐 ⑤ |
| p_surface 8 词是否够 / 扩到 10 | ⚠ 等学姐 confirm |
| 金标 NL 3-5 段 (替换示例 2 placeholder) | ⚠ 等学姐 |

# 调用方接口约定

`extractor_batch.py` 在调用时, 把以下占位符替换为实际值:
- {SEGMENT_ROLE} / {SCENARIO} / {INPUT_TEXT}     (v2 已有)
- {C_LOCAL_TAG}                                   (v3 新增, 从 situation_codebook 查表)
- {TASK_BACKGROUND}                                (v3 新增, 全 dataset 常量, 等学姐 ⑤)
- {S_PRIMARY_TAGS} / {S_SECONDARY_FACTOR}         (v3 新增, 从 30 维量表 + EFA 算)
