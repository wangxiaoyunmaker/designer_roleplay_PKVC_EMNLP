# PKVC Extractor Prompt v7b (conditional, segment-aware)

> v7b vs v7a: 同 4 字段 schema, 加 segment-role 条件指令.
> 每个 segment (B1/B2/B3) 有不同认知特征, prompt 内部告诉 LLM 按 segment_role 调整重点.
> 单 prompt 维护, 但内部有 if/else 分支引导.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 思维链段落中提取 PKVC 主导因子 + 核心底线.
严格输出 JSON 4 个字段.

铁律:
1. criterion (非 null) 必须 verbatim 节选自 INPUT BEGIN/END 区段, ≤ 20 字.
2. dominant_factor 必填非 null. 单字母 P/K/V/C 或路径 K→V.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
4. 根据 segment_role 调整重点 — 见 user prompt "segment 先验".
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段提取 PKVC 主导因子 + 核心底线.

# 输入 metadata

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}

# ★ Segment 先验 (根据 segment_role 调整重点)

| segment_role | 该段认知特征 | 抽取重点 |
|---|---|---|
| **第一反应** | 设计师听到委托后的第一直觉, 短促, 信号往往单纯 | 抓**认知入口** = dominant_factor (单字母多, 路径少). criterion 一般 null. |
| **决策链路** | 中段权衡 + 推理, 因子可能切换 | 抓 dominant_factor + **路径** (e.g. K→V / P→K). criterion 一般 null 或弱. |
| **核心 tradeoff** | 收束段, 设计师明确"我的底线" | 重点抓 **criterion** (段尾"我的底线/必须/不能"). dominant_factor 多半 V. |

⚠ 这是先验不是硬规则: 如果该段实际信号跟先验不符, 跟随实际信号 (不要硬套).

# PKVC 4 因子诊断特征

| 因子 | 诊断特征 |
|---|---|
| **P** Problem | 质疑问题框架本身, 把冲突"去稳定化", 问"这个问题对吗" |
| **K** Knowledge | 先理解再判断, 把价值操作化为可证伪经验命题 (调研/数据) |
| **V** Value | 先判断后推导, 价值立场段首确立, 后续是该立场的展开 |
| **C** Concept | 从形态/愿景入口, 而非问题/数据/价值入口 |

⚠ 常见误判:
- "需要调研" → K (不是 P, P 是质疑问题框架)
- "成本至上"作底线: V 先行 = V 主导, K 搜索后浮现 = K 主导
- 第一句"直觉" 不一定 V — 看后续展开

# 输出 Schema (4 字段)

- `dominant_factor`: 必填非 null. "P"/"K"/"V"/"C" 或路径如 "K→V"
- `criterion`: ≤20 字 verbatim INPUT, 或 null
- `confidence`: "high"|"medium"|"low"
- `extraction_notes`: ≤30 字一句话

# 三个 few-shot

## 示例 1 (segment_role=第一反应, K 主导)

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 是否会对这个超市的营收带来影响? 这些都是需要调研的.

(段类型=第一反应, 抓认知入口. criterion null 符合先验.)

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K 入口: 第一反应需要调研, 信息收集驱动"
}

## 示例 2 (segment_role=决策链路, P 主导)

INPUT (附录 D seg2):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? 老年人
> 除本超市外是否有可替代的选择? 这些问题综合考虑起来才有价值讨论是否应该放弃
> 部分老年客户. 零售行业面向不同的客群可以做精细化的、定制化的转型. 每一类服
> 务一类客群、服务好即可. 至于诱导和操纵? 商业算法本来就是引导和欺诈来赚取更
> 多利润, 除非重新定义商业行为.

(段类型=决策链路, 抓路径. 表面 K 实际 P 重构.)

Output:
{
  "dominant_factor": "P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "P 主导: 重构客群划分 + 重新定义商业行为"
}

## 示例 3 (segment_role=核心 tradeoff, V 主导)

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 如果你要牺牲用户去换这些短期收益, 那
> 迟早会崩盘. 我会觉得这种东西最核心的是让人幸福、而不是得利.

(段类型=核心 tradeoff, 抓 criterion. 段尾"让人幸福而不是得利"明确底线.)

Output:
{
  "dominant_factor": "V",
  "criterion": "让人幸福而不是得利",
  "confidence": "high",
  "extraction_notes": "V 主导: 价值前置 + 段尾宣言型底线"
}

# 输出前 self-check

Q1: dominant_factor 是否非 null?
Q2: 根据 segment_role 先验, criterion 该不该 null? 不该 null 则确认 verbatim 在 INPUT.
Q3: JSON 只 4 字段?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON

只 4 字段, 字段名严格. 不要 markdown 围栏 / 不要解释.
```
