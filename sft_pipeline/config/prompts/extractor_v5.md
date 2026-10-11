# PKVC + T_CAPS Quad 提取器 Prompt v5.0

> v5.0 任务范围收紧 (相对 v4.x): **砍 sadr_activations 30 节点细抽取**, 只保留:
> - PKVC dominant_factor (单字母 / 路径)
> - criterion + criterion_strength (核心底线)
> - T_CAPS quad (p_surface / o_primary / o_subtype)
> - confidence + extraction_notes
>
> Rationale: v4.x holdout 显示 30 SADR 节点抽取在 GPT-4o 上不稳定 (LLM 倾向于 hallucinate
> evidence 或选错节点). 简化到只标 dominant_factor + criterion 后 LLM 任务清晰, prompt
> 体积砍 1/3+, 注意力更聚焦.
>
> SADR 30 节点信息保留在 docs/notes/PKVC_30节点本体.md, 后续如需可 post-process 加上.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师思维链段落中提取 PKVC 主导因子 + 核心底线
+ T_CAPS 路由字段, 严格输出 JSON.

铁律 (违反任意一条 → confidence=low):
1. criterion 字段 (若非 null) 必须 verbatim 节选自 INPUT BEGIN/END 区段, ≤ 20 字.
   不能复制 prompt 中的 4 因子诊断特征或 T_CAPS 词表内容.
2. JSON 字段名严格按下面 Schema 拼写 (dominant_factor, criterion, p_surface,
   o_primary). 不换名, 不嵌套.
3. dominant_factor / o_primary 必须 ∈ {"P","K","V","C"} 闭集 (或路径字符串如 "K→V").
4. p_surface 必须 ∈ 下方 8 词闭集.
5. 只输出 JSON, 不要 markdown 围栏 / 不要解释 / 不要 chain-of-thought.
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段的设计师思维链段落中提取 PKVC 主导因子 + 核心底线 + T_CAPS 字段.

# 输入 metadata

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}
- C_local:      {C_LOCAL_TAG}
- task_background: {TASK_BACKGROUND}
- s_primary:    {S_PRIMARY_TAGS}
- s_secondary:  {S_SECONDARY_FACTOR}

# PKVC 4 因子诊断特征 (判 dominant_factor)

判断基于**认知入口 + 推理结构 + 收束姿态**, 不基于关键词频率.

| 因子 | 诊断特征 |
|---|---|
| **P** Problem | 质疑问题框架本身, 把冲突"去稳定化", 问"这个问题对吗" |
| **K** Knowledge | 先理解再判断, 把价值操作化为可证伪经验命题 (需要调研/数据/案例) |
| **V** Value | 先判断后推导, 价值立场段首确立, 后续是该立场的展开 |
| **C** Concept | 从形态/愿景入口, 而非问题/数据/价值入口 (5 范例中仅 1 份) |

多因子路径用 "K→V" / "P→K→V" 字符串表达. 段尾总结性立场优先于段中工具性提及.

⚠ 常见误判:
- "需要调研" / "需要数据" → K 入口 (不是 P, P 是质疑问题框架本身)
- "成本至上" / "提高收益" 作底线 → 看 V 是先行还是 K 搜索后浮现: 先行 = V 主导,
  搜索后 = K 主导 V 工具性激活
- 第一句"直觉" 不一定是 V 入口 — 看后续展开

# T_CAPS p_surface (路由谓语, 闭集 8 词)

| surface_predicate         | 对应 modulation | stance      |
|---------------------------|-----------------|-------------|
| forces_transition_to      | Strong+         | neutral     |
| severely_overrides        | Strong+ Bypass  | neutral     |
| defiant_overrides         | Strong+ Bypass  | Defiance    |
| gradually_shifts_to       | Weak+           | neutral     |
| compliantly_follows       | Weak+           | Compliance  |
| negotiates_around         | Weak±           | Negotiation |
| weakly_inhibits           | Weak-           | neutral     |
| bypasses                  | Bypass          | neutral     |

p_surface 看动词强度 + 是否绕过默认阶段 + stance:
- "强烈反对/必须/绝对" + 跳过中间 → defiant_overrides
- "顺从/执行/接受" → compliantly_follows
- "协商/折中/平衡" → negotiates_around
- "渐进/逐步/慢慢" → gradually_shifts_to
- 不强烈但有方向 → forces_transition_to OR weakly_inhibits
- 直接跳过中间 → bypasses

# 输出 JSON Schema (字段名严格按下方拼写)

```json
{
  "segment_role":       "决策链路",
  "dominant_factor":    "K→V",
  "criterion":          "成本至上",
  "criterion_strength": "可妥协",
  "confidence":         "high",
  "extraction_notes":   "K 发动机型: K 全程主导, V 在 K 搜索饱和后以成本至上锚定",
  "p_surface":          "compliantly_follows",
  "o_primary":          "V",
  "o_subtype":          null
}
```

字段取值约束:
- `segment_role` ∈ {"第一反应", "决策链路", "核心 tradeoff"}
- `dominant_factor`: "P"/"K"/"V"/"C" 单字母 或 路径如 "K→V" / "P→V"
- `criterion`: ≤ 20 字 verbatim 节选自 INPUT, 或 null (段内无明确底线时)
- `criterion_strength` ∈ {"不可妥协", "可妥协"} | null
- `confidence` ∈ {"high", "medium", "low"}
- `extraction_notes`: ≤ 50 字, 一句话总结主导模式
- `p_surface` ∈ 上面 8 词闭集 | null
- `o_primary` ∈ {"P","K","V","C"} (= dominant_factor 路径终点) | null
- `o_subtype` 暂时 null (学姐 cardId metadata 未到)

# 输出前 self-check (不写进 JSON)

Q1: dominant_factor 是否符合 4 因子诊断特征? 不是关键词数, 是认知流程.
Q2: criterion (若非 null) 是否能在下方 INPUT BEGIN/END 之间 verbatim 找到?
    找不到 → criterion = null, confidence 降一档.
Q3: o_primary 是否 == dominant_factor 路径终点?
Q4: p_surface 是否 ∈ 8 词闭集?
Q5: 输出是不是干净 JSON, 没 markdown 围栏?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON

必须包含全部 9 个字段: segment_role / dominant_factor / criterion / criterion_strength
/ confidence / extraction_notes / p_surface / o_primary / o_subtype.

字段名按 Schema (英文下划线), 字段值按约束 (PKVC 单字母 / 8 词闭集 / verbatim INPUT).
不要 markdown 围栏 / 不要解释 / 不要 chain-of-thought.
```

---

# 阻塞项

| 项 | 状态 |
|---|---|
| 12 cardId 完整语义定义 (P1-C3) | ⚠ 等学姐 ③ → 触发 o_subtype 启用 |
| C_local tag (Q1, Q2, Q7, Q8) | ⚠ 等学姐 ① + Q4 字段归属 ② |
| 50-100 字 task background | ⚠ 等学姐 ⑤ |
| 跨家族 κ pilot (gpt-4o vs claude-3-5-sonnet) | ⚠ 待跑 |
| SADR 30 节点抽取 (v4.x 范围, 现 deferred) | ⚠ 任务现简化到只标 PKVC + criterion, SADR 待后续 |

# Holdout

附录 F (C 入口) + 附录 G (P 先启 V 双形态) 评测脚本 scripts/test_extractor_v5_holdout.py.
评测指标: dominant_factor 跟专家标注一致 / criterion 文字接近.

# 调用方接口约定

extractor_batch.py 替换占位符: {SEGMENT_ROLE} / {SCENARIO} / {INPUT_TEXT} +
{C_LOCAL_TAG} / {TASK_BACKGROUND} / {S_PRIMARY_TAGS} / {S_SECONDARY_FACTOR}.
