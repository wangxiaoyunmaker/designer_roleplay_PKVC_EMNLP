# PKVC + SADR 提取器 Prompt

用于:
1. **训练数据加工** (方向一): Q7 三段 NL CoT → factor + sadr + criterion yaml
2. **评测端反推** (方向二): 模型 Q7 输出 NL → 反推 PKVC trait, 跟 Q1-Q6 真值比

调用方: GPT-4o / Claude / DeepSeek-V4-Flash (任选不同家族, 避开 self-judge bias)。

---

## System Prompt

```
你是一位设计师认知分析专家, 任务是从设计师的思维链文本中提取出 PKVC 认知
框架下的关键信号。请严格按规则提取, 输出 JSON, 不要有任何解释或额外文字。
```

---

## User Prompt Template

```
# 任务

从下方给定的设计师思维链段落中, 提取 PKVC 主导因子、SADR 激活节点 (含极性)、
以及核心底线 (若有)。输出严格 JSON。

# PKVC 4 因子框架

| 字母 | 名称 | 含义 |
|---|---|---|
| P | 问题定义 (Problem framing) | 设计师如何界定问题、识别真实需求、拆解复杂委托 |
| K | 知识应用 (Knowledge)       | 设计师如何调用经验、案例、客观数据、工具/材料知识 |
| V | 价值判断 (Value)           | 设计师如何在多方利益冲突中排序、设立不可牺牲的伦理 / 价值底线 |
| C | 概念生成 (Concept)         | 设计师如何从模糊愿景到具体形态、提炼设计逻辑 |

# SADR 节点参考清单

(SADR = Schema-Anchored Design Reasoning, 30 个节点; 编号 #1-#30)

## 已知节点 (来自 Q2 排序题 cardId)

- #P1: 界定项目核心底线和真实要解决的问题
- #P2: 将复杂的总委托拆解为几个具体子问题
- #K1: 调用过去的经验、竞品案例或行业规范作参考
- #K2: 使用客观数据、测试报告来证明某个逻辑
- #K3: 评估现有技术栈、材料以及工具边界能实现什么
- #V1: (待补)
- #V2: 在多方利益(商业、用户、社会)发生冲突时进行优先级排序
- #V3: 设立一条绝对不可牺牲的原则底线
- #C1: 脑海中快速涌现出一个模糊但强烈的形态或愿景画面
- #C2: 把零散观察整合为一个统摄全局的核心设计逻辑
- #C3: 提炼一些可复用的设计原则

## 扩展节点 (#10-#30, 待补全完整定义)

⚠ 占位: 上面 11 个是 Q2 cardId 给到的。30 节点完整列表待补充。
当前若文本激活了清单外的 SADR, 用编号 #NEW_<简短描述> 标记 (例如 #NEW_服务保留),
后续人工 align 到正式编号。

# 极性规则

- `+` (正激活): 设计师**采纳 / 推动 / 优先** 该方向
- `-` (负激活): 设计师**反对 / 保留 / 排斥** 该方向 (重要: "保留人工服务" 是
  对"消除人工服务"的负激活, 写作 `#服务保留-` 或 `#人工服务消除-`)

# 输出 JSON Schema

```json
{
  "segment_role": "第一反应" | "决策链路" | "核心 tradeoff" | "未识别",
  "dominant_factor": "<P|K|V|C>" 或 "<P→K→V>" 路径 (多步流转用 → 连接),
  "sadr_activations": [
    {"id": "#P1", "polarity": "+", "evidence": "原文中触发的短句"},
    {"id": "#NEW_服务保留", "polarity": "-", "evidence": "..."}
  ],
  "criterion": "<短语>" 或 null   (仅 B3 核心 tradeoff 段),
  "criterion_strength": "不可妥协" | "可妥协" | null,
  "confidence": "high" | "medium" | "low",
  "extraction_notes": "<一句话: 你为什么选这些激活节点>"
}
```

# 提取规则

1. **segment_role**: 看文本语气判断
   - 第一反应: 直觉性的初判、强烈情绪 (惊讶 / 警惕 / 兴奋)
   - 决策链路: 多方权衡、步骤化推理、"如果…那么…"
   - 核心 tradeoff: 明确划出底线、不可妥协陈述

2. **dominant_factor**:
   - 单段单一主导 → 单字母 (如 `V`)
   - 段内有清晰流转 → 用箭头 (如 `V→K→V`: 价值起手 → 调用知识 → 回归价值)
   - 不能确定 → confidence 标 low

3. **sadr_activations**:
   - 严格选自 SADR 列表; 不在列表的用 `#NEW_<描述>` 占位
   - 必须给 evidence 短句 (原文 5-20 字)
   - 极性参考上文"极性规则"

4. **criterion**: 仅 B3 段填; 找原文中"底线 / 不能 / 必须 / 绝对" 周边短语

5. **confidence**:
   - high: 因子主导清晰, SADR 节点匹配明确
   - medium: 因子明确但 SADR 节点边界模糊
   - low: 多个因子打架 / 信号弱

# Few-shot 示例

## 示例 1 (B3 核心 tradeoff)

输入文本:
> 我的核心底线只有一句话: 人工服务不消除. 哪怕经营数据说自助收银能省 30%
> 人力, 我也不会建议. 因为这家店的核心资产不在货架上, 在收银台后面那个
> 认识每个老顾客的人.

输出:
```json
{
  "segment_role": "核心 tradeoff",
  "dominant_factor": "V",
  "sadr_activations": [
    {"id": "#V3", "polarity": "+", "evidence": "我的核心底线只有一句话"},
    {"id": "#NEW_服务保留", "polarity": "-", "evidence": "人工服务不消除"},
    {"id": "#K2", "polarity": "-", "evidence": "哪怕经营数据说...我也不会建议"}
  ],
  "criterion": "人工服务保留",
  "criterion_strength": "不可妥协",
  "confidence": "high",
  "extraction_notes": "明确划出 V 主导底线, 主动用 K2 数据反例论证, 即数据让步于价值"
}
```

## 示例 2 (B2 决策链路)

输入文本:
> 但完全不变也不行. 经营压力是真的. 所以问题不是"现代化 vs 保留", 是
> "现代化到什么程度". 空间和动线可以现代化, 是 K 层面的优化; 但 service
> touchpoint 必须保留, 是 V 内核. 增加的现代化收益要能撑起保留下来的低
> 效率部分.

输出:
```json
{
  "segment_role": "决策链路",
  "dominant_factor": "V→K→V",
  "sadr_activations": [
    {"id": "#P1", "polarity": "+", "evidence": "问题不是X vs Y, 是Y到什么程度"},
    {"id": "#K3", "polarity": "+", "evidence": "空间和动线可以现代化, K 层面优化"},
    {"id": "#V2", "polarity": "+", "evidence": "service touchpoint 必须保留, V 内核"},
    {"id": "#NEW_经济可持续", "polarity": "+", "evidence": "现代化收益撑起低效率部分"}
  ],
  "criterion": null,
  "criterion_strength": null,
  "confidence": "high",
  "extraction_notes": "经典权衡链: 价值底线先稳 → 拉知识做局部优化 → 回到价值锚定"
}
```

## 示例 3 (B1 第一反应, low confidence)

输入文本:
> 第一反应是不能动核心. 30 年的社区超市不只是个"门店", 是社区的一部分.

输出:
```json
{
  "segment_role": "第一反应",
  "dominant_factor": "V",
  "sadr_activations": [
    {"id": "#V3", "polarity": "+", "evidence": "第一反应是不能动核心"},
    {"id": "#NEW_社区价值", "polarity": "+", "evidence": "是社区的一部分"}
  ],
  "criterion": null,
  "criterion_strength": null,
  "confidence": "medium",
  "extraction_notes": "短文本, V 主导但 SADR 节点细分难, #社区价值 需后续 align"
}
```

# 待提取文本

```
{INPUT_TEXT}
```

请只输出 JSON, 不要任何额外解释。
```

---

## 校准建议 (调用前)

1. **温度 = 0**, JSON mode (OpenAI / Claude 都支持) 强制结构化输出
2. **κ pilot**: 30 段先抽出来, 同时调 GPT-4o + Claude 跑一遍, 算两家 dominant_factor / sadr_id 的一致率, κ ≥ 0.7 才能放心用
3. **人工 audit**: 第二标注员抽 10-15 段独立标注, 跟 LLM 比对, κ ≥ 0.6
4. **SADR 列表补全前先 free-form**: 让 LLM 用 `#NEW_<描述>` 自由 propose, 跑完 200 designer × 3 段 = 600 段后, 收集所有 `#NEW_*` 做 frequency 统计 → 跟 30 SADR 节点定义 align (这是个 bottom-up 验证, 顺便检查 30 节点设计是否覆盖真实数据)

## 阻塞项 (需协作方补)

1. **SADR 30 节点完整定义** (id / 名称 / 触发关键词 / 极性约定)
2. **#V1 cardId 的具体定义** (从 Q2 cardId 看, V1 缺失, V2/V3 有)
3. **B1/B2/B3 段切分规则** (现在是 GPT-4o 看文本判断, 后续可能需协作方给出更严格的语义切分准则)
