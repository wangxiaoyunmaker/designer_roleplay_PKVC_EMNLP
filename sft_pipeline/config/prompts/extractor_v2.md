# PKVC + SADR 提取器 Prompt v2

> v2 改动 (相对 `extractor_prompt.md` v1):
> - **A**: `segment_role` 改为输入参数, LLM 不再推断段角色, 节省 budget + 提高 κ
> - **B**: 每个 PKVC 因子加 lexical marker 清单(⚠ 待学姐确认词汇覆盖)
> - **C**: `evidence` 字段强制 verbatim 原文片段, 不允许 paraphrase
> - **D**: few-shot 加 long / mixed / polarity-reversal / low-confidence 4 个 edge case
>   (⚠ 等学姐 3-5 段金标 NL → 当前用 v1 例子的简化版作 placeholder)
> - **E**: 新增 negative examples 章节("不要做什么")
> - **F**: 新增 self-check chain (输出前 LLM 内部 verify)
> - **G**: 12 cardId 完整列出(2026-05-22 数据 verify 后 P1/P2/P3 + V1/V2/V3 都存在, 不是 11)

---

## System Prompt

```
你是一位设计师认知分析专家. 任务: 从设计师的思维链段落中提取 PKVC 框架下的
关键信号. 严格按规则提取, 只输出 JSON, 不要任何解释 / 不要 markdown 围栏 /
不要 chain-of-thought 痕迹.
```

---

## User Prompt Template

```
# 任务

从下方设计师思维链段落中提取主导因子、SADR 激活节点(含极性)、以及核心底线
(仅 B3 段). 严格输出 JSON.

# 输入 metadata (由调用方填充, 你不要重新推断)

- segment_role: {SEGMENT_ROLE}   ← "第一反应" | "决策链路" | "核心 tradeoff"
- scenario:     {SCENARIO}        ← e.g. "Q7_老城区超市重构"

# PKVC 4 因子框架

| 字母 | 名称        | 含义                                                                       | 典型 lexical marker (⚠ 学姐确认覆盖) |
|:---:|:---|:---|:---|
| P  | 问题定义     | 界定问题、识别真实需求、拆解复杂委托                                          | 界定 / 拆解 / 核心问题 / 真实需求 / 边界 / 范围  |
| K  | 知识应用     | 调用经验、案例、客观数据、工具/材料知识                                       | 数据 / 案例 / 经验 / 测试 / 工具 / 行业规范 / 测算 |
| V  | 价值判断     | 多方利益冲突时排序, 设立不可牺牲的伦理/价值底线                               | 底线 / 不可妥协 / 优先 / 反对 / 伦理 / 必须 / 绝对 |
| C  | 概念生成     | 模糊愿景到具体形态, 提炼可复用设计逻辑                                        | 形态 / 概念 / 愿景 / 整合 / 提炼 / 设计逻辑 / 原则 |

# SADR 节点参考清单 (12 cardId, 4 因子 × 3 子节点)

| ID | 因子 | 一句话定义 | 默认极性 |
|:---:|:---:|:---|:---:|
| #P1 | P | 界定项目核心底线和真实要解决的问题                              | + |
| #P2 | P | 将复杂的总委托拆解为几个具体子问题                              | + |
| #P3 | P | (⚠ 学姐补全定义; 当前 verbatim 来自问卷 Q2 cardId)              | + |
| #K1 | K | 调用过去的经验、竞品案例或行业规范作参考                        | + |
| #K2 | K | 使用客观数据、测试报告来证明某个逻辑                            | + |
| #K3 | K | 评估现有技术栈、材料以及工具边界能实现什么                      | + |
| #V1 | V | (⚠ 学姐补全定义; 来自问卷 Q2 cardId, 数据中 34 designer 选过)   | + |
| #V2 | V | 在多方利益(商业/用户/社会)发生冲突时进行优先级排序             | + |
| #V3 | V | 设立一条绝对不可牺牲的原则底线                                  | + |
| #C1 | C | 脑海中快速涌现出一个模糊但强烈的形态或愿景画面                  | + |
| #C2 | C | 把零散观察整合为一个统摄全局的核心设计逻辑                      | + |
| #C3 | C | 提炼一些可复用的设计原则                                        | + |

**扩展节点 (#NEW_<描述>)**: 当文本激活的语义不在上述 12 个 cardId 内,
用 `#NEW_<2-6 中文字描述>` 占位 (例: `#NEW_服务保留`, `#NEW_文化纹理`).
扩展节点会在 600 段全跑完后按 frequency 跟学姐 align 到正式 SADR 30 节点编号.

# 极性规则

| 极性 | 含义 | 例子 |
|:---:|:---|:---|
| `+` | 设计师**采纳 / 推动 / 优先** 该方向 | "数据显示 X 应该采用" → `#K2+` |
| `-` | 设计师**反对 / 保留 / 排斥** 该方向 (反向锚定) | "人工服务必须保留" → `#NEW_服务保留-` (设计师反向锚定"消除服务"这个方向) |
| `0` | 中立 / 未明确表态 | (一般不输出, 除非极性确实模糊) |

**反向锚定经验**: "保留 / 不消除 / 必须维持 / 不能动" 这类语句, 设计师是
**反对**"消除该事物"的方向, 写作 `#<事物>-`. 不要错写成 `+`.

**反转模式**: "以前我会 X, 现在反而不 X" / "我曾经认同 X, 但现在反对" — 这种段
里同一个 SADR 节点应该取**当前** 立场 (现在的极性), 不取过去的极性.

# 输出 JSON Schema

```json
{
  "segment_role":      "第一反应" | "决策链路" | "核心 tradeoff",
  "dominant_factor":   "<P|K|V|C>" 或路径 "<P→K→V>",
  "sadr_activations":  [
    {
      "id":       "#P1" | "#NEW_服务保留" | ...,
      "polarity": "+" | "-" | "0",
      "evidence": "<原文 verbatim 短句, 5-25 字>",
      "evidence_offset_hint": "<原文中大致位置: 段首 | 段中 | 段尾>"
    },
    ...
  ],
  "criterion":          "<短语>" | null,    (仅 B3 核心 tradeoff 段)
  "criterion_strength": "不可妥协" | "可妥协" | null,
  "confidence":         "high" | "medium" | "low",
  "extraction_notes":   "<一句话: 你为什么这么定 dominant_factor>"
}
```

# 提取规则

1. **dominant_factor**:
   - 单段单一主导 → 单字母 (如 `V`)
   - 段内清晰流转 → 用 `→` 连接 (如 `V→K→V`: 价值起手 → 调用知识 → 回归价值)
   - 不确定 → `confidence: low` 且 `extraction_notes` 写明歧义

2. **sadr_activations**:
   - 优先使用上面 12 个已知 cardId; 不在清单的用 `#NEW_<描述>` 占位
   - **evidence 必须是原文逐字 sub-string** (5-25 字; 不要 paraphrase, 不要总结, 不要意译)
   - 极性参考"反向锚定"规则
   - 同一节点不要重复出现; 同一节点不同极性出现 → 取**最后表态的极性**

3. **criterion**: 仅 B3 段提取; 找原文中 "底线 / 不能 / 必须 / 绝对 / 不可妥协"
   周边短语 (5-20 字). criterion 应该是**设计师本人的措辞** (原文短语), 不要重写.

4. **confidence**:
   - `high`: factor 主导清晰 (单一字母强信号 或 路径明确), SADR 节点匹配 cardId 清单
   - `medium`: factor 明确但 SADR 节点边界模糊 / 大量用 `#NEW_*`
   - `low`: 多 factor 打架 / 信号弱 / 段落太短 / 设计师答非所问

# Self-check (输出 JSON 前内部验证 3 条)

⚠ 不要把这段 reasoning 写进 JSON. 内部 verify 通过后只输出 JSON.

```
Q1 (verbatim): 每个 sadr_activations[i].evidence 是不是原文里能 find() 到的 sub-string?
              如果不是, 修正成原文实际短句.
Q2 (factor↔evidence 一致): dominant_factor = V 时, 至少有一个 evidence
              触发 V 因子 lexical marker (底线/优先/反对/...); 否则降 confidence.
Q3 (sadr 因子归属一致): sadr_activations[i].id 是 #V_ 开头时,
              dominant_factor 路径里至少出现 V; 否则降 confidence.
```

# Negative examples (常见错误, 不要犯)

- **不要从语气词 / 连接词提 SADR**: "啊", "呢", "吧", "然后", "所以"
  这些不是 SADR 节点触发词.
- **不要 invent criterion**: B1 / B2 段没有 criterion 字段 → 严格写 `null`.
  B3 段如果找不到明确底线表述 → 写 `null` 而不是编一个.
- **不要把段角色搞错**: `segment_role` 已经在 metadata 给定, 直接 echo, 不要
  重新推断. 如果你内部判断跟 metadata 不符, 写在 `extraction_notes` 提示
  调用方, 但不要修改 segment_role 字段.
- **不要 over-extract 多 factor**: 文本明显单一主导 (一通到底都是 V) 时不要
  生编 `V→K→V` 路径. 路径只在原文真有跳转时用.
- **不要 paraphrase evidence**: evidence 必须是原文 sub-string. 如果你写的
  evidence 在原文搜不到, 就是不合格.

---

# Few-shot 示例

## 示例 1 — B3 核心 tradeoff, high confidence

**Input metadata**:
- segment_role: 核心 tradeoff
- scenario: Q7_老城区超市重构

**Input text**:
> 我的核心底线只有一句话: 人工服务不消除. 哪怕经营数据说自助收银能省 30%
> 人力, 我也不会建议. 因为这家店的核心资产不在货架上, 在收银台后面那个
> 认识每个老顾客的人.

**Output**:
```json
{
  "segment_role": "核心 tradeoff",
  "dominant_factor": "V",
  "sadr_activations": [
    {"id": "#V3", "polarity": "+", "evidence": "我的核心底线只有一句话",
     "evidence_offset_hint": "段首"},
    {"id": "#NEW_服务保留", "polarity": "-", "evidence": "人工服务不消除",
     "evidence_offset_hint": "段首"},
    {"id": "#K2", "polarity": "-", "evidence": "哪怕经营数据说自助收银能省 30% 人力",
     "evidence_offset_hint": "段中"}
  ],
  "criterion": "人工服务不消除",
  "criterion_strength": "不可妥协",
  "confidence": "high",
  "extraction_notes": "明确划出 V 主导底线, K2 数据反例被主动反向锚定, 即数据让步于价值"
}
```

## 示例 2 — B2 决策链路, factor 流转 V→K→V

**Input metadata**:
- segment_role: 决策链路
- scenario: Q7_老城区超市重构

**Input text**:
> 但完全不变也不行. 经营压力是真的. 所以问题不是"现代化 vs 保留", 是
> "现代化到什么程度". 空间和动线可以现代化, 是 K 层面的优化; 但 service
> touchpoint 必须保留, 是 V 内核. 增加的现代化收益要能撑起保留下来的
> 低效率部分.

**Output**:
```json
{
  "segment_role": "决策链路",
  "dominant_factor": "V→K→V",
  "sadr_activations": [
    {"id": "#P1", "polarity": "+",
     "evidence": "问题不是\"现代化 vs 保留\", 是\"现代化到什么程度\"",
     "evidence_offset_hint": "段中"},
    {"id": "#K3", "polarity": "+", "evidence": "空间和动线可以现代化",
     "evidence_offset_hint": "段中"},
    {"id": "#V2", "polarity": "+",
     "evidence": "service touchpoint 必须保留, 是 V 内核",
     "evidence_offset_hint": "段中"},
    {"id": "#NEW_经济可持续", "polarity": "+",
     "evidence": "现代化收益要能撑起保留下来的低效率部分",
     "evidence_offset_hint": "段尾"}
  ],
  "criterion": null,
  "criterion_strength": null,
  "confidence": "high",
  "extraction_notes": "价值底线先稳 → 调局部知识做优化 → 回到价值锚定. 经典 V→K→V 权衡链"
}
```

## 示例 3 — B1 第一反应, low confidence (短段 / 信号弱)

**Input metadata**:
- segment_role: 第一反应
- scenario: Q7_老城区超市重构

**Input text**:
> 第一反应是不能动核心. 30 年的社区超市不只是个"门店", 是社区的一部分.

**Output**:
```json
{
  "segment_role": "第一反应",
  "dominant_factor": "V",
  "sadr_activations": [
    {"id": "#V3", "polarity": "+", "evidence": "第一反应是不能动核心",
     "evidence_offset_hint": "段首"},
    {"id": "#NEW_社区性", "polarity": "+", "evidence": "是社区的一部分",
     "evidence_offset_hint": "段尾"}
  ],
  "criterion": null,
  "criterion_strength": null,
  "confidence": "medium",
  "extraction_notes": "短文本, V 主导但缺细分 SADR 节点信号"
}
```

## 示例 4 — Edge case: 极性反转 (placeholder, 待学姐补金标)

> ⚠ 等学姐 3-5 段金标 NL 后, 这里加一个"以前我会 K, 现在反而 V" 类型的
> 反转例子. 当前 placeholder 跳过. 学姐的金标段建议覆盖:
> - 长段(>300 字)且多 factor 混合
> - 极性反转 ("以前认同 X, 现在反对")
> - B3 criterion 强度 "可妥协" (跟示例 1 的"不可妥协"对照)

---

# 输入文本

```
{INPUT_TEXT}
```

只输出 JSON, 严格遵循 schema. 不要 markdown 围栏, 不要解释, 不要 chain-of-thought.

---

# 调用方接口约定 (extractor_batch.py 自动填充)

`extractor_batch.py` 在调用时把 ``SEGMENT_ROLE`` / ``SCENARIO`` / ``INPUT_TEXT`` 三个占位符
替换为实际值, 通过 OpenAI / Anthropic / DeepSeek 的 JSON mode (response_format) 强制
结构化输出. 失败重试 3 次, 全失败时 batch 中该条写 ``{"error": ...}`` 但 batch 继续.

## 校准建议

1. **κ pilot 必跑**: 抽 30 段 (10 × 3 段 designer), GPT-4o + Claude-3.5-sonnet 各跑一遍,
   factor agreement rate ≥ 0.7 & sadr Jaccard mean ≥ 0.5 才放行全量.
2. **人工 audit 必跑**: 学姐独立标注 10-15 段, 跟 LLM κ ≥ 0.6.
3. **temperature = 0** + **response_format JSON mode**: 强制 deterministic + JSON.
4. **#NEW_* 频次后置 align**: 全 600 段跑完后, 收集所有 `#NEW_<...>` 节点 frequency,
   跟学姐 align 到正式 SADR 30 节点编号.

## 阻塞项 (学姐回复后填充)

| 项 | 状态 |
|---|---|
| PKVC 4 因子 lexical marker 完整词表 (B 章节) | ⚠ 当前清单是我的初稿, 等学姐 confirm 覆盖 |
| #P3 / #V1 语义定义 | ⚠ 学姐补 |
| Q3-Q6 题目 PKVC 因子 mapping | ⚠ 学姐补 (参考 `q1q6_option_inventory.md`) |
| 3-5 段金标 NL (含 long / mixed / reversal) | ⚠ 学姐补 (替换示例 4 placeholder) |
| SADR 30 节点完整定义表 | ⚠ 学姐补 (替换 #NEW_* 启发) |
