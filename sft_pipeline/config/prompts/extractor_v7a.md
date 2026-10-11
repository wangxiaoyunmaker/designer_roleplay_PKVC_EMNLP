# PKVC Extractor Prompt v7a (minimal 4 字段)

> v7a vs v6.1:
> - 砍 T_CAPS quad (`p_surface` / `o_primary` / `o_subtype`) — 学姐 metadata 未到 + 无 κ 验证 + o_primary 冗余于 dominant_factor 路径终点
> - 砍 `criterion_strength` — 合并到 criterion 字段语义
> - 砍 `segment_role` echo — 调用方传, LLM 不需要回填
> - 保留 `dominant_factor` / `criterion` / `confidence` / `extraction_notes` 4 字段
> - 保留 v6.1 的 3 个 cleaned few-shot (C/D/E)

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 思维链段落中提取 PKVC 主导因子 + 核心底线.
严格输出 JSON, 只输出 4 个字段, 不要 markdown 围栏 / 不要解释.

铁律:
1. **criterion 字段 (非 null) 必须 verbatim 节选自 INPUT BEGIN/END 区段**, ≤ 20 字.
   不能从 prompt 中的 4 因子说明或 few-shot 示例复制.
2. **dominant_factor 必填非 null**. 即使 confidence=low, 也给最可能的字母或路径.
3. JSON 字段名严格按 Schema 拼写: dominant_factor / criterion / confidence / extraction_notes.
4. 只 4 字段, 不要加任何额外字段 (不要 o_primary, 不要 p_surface 等).
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段的设计师 Q7 思维链中提取 PKVC 主导因子 + 核心底线.

# 输入 metadata (LLM 不要 echo 回来)

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}

# PKVC 4 因子诊断特征 (判 dominant_factor)

判断基于认知入口 + 推理结构 + 收束姿态, 不基于关键词频率.

| 因子 | 诊断特征 |
|---|---|
| **P** Problem | 质疑问题框架本身, 把冲突"去稳定化", 问"这个问题对吗" |
| **K** Knowledge | 先理解再判断, 把价值操作化为可证伪经验命题 (调研/数据) |
| **V** Value | 先判断后推导, 价值立场段首确立, 后续是该立场的展开 |
| **C** Concept | 从形态/愿景入口, 而非问题/数据/价值入口 |

多因子路径用 "K→V" / "P→V" 字符串表达. 段尾总结性立场优先于段中工具性提及.

⚠ 常见误判:
- "需要调研" → K 入口 (不是 P, P 是质疑问题框架本身)
- "成本至上" 作底线: 看 V 先行还是 K 搜索后浮现; 后者 K 主导
- 第一句"直觉" 不一定是 V — 看后续展开

# 输出 Schema (4 字段, 字段名严格)

- `dominant_factor`: 必填且非 null. "P" / "K" / "V" / "C" 或路径如 "K→V"
- `criterion`: ≤20 字 verbatim INPUT 节选, 或 null. 段内无明确底线就 null.
- `confidence`: "high" | "medium" | "low"
- `extraction_notes`: 一句话总结 ≤ 30 字

⚠ 严禁字面输出 "placeholder" / "需要调研" / "K→V" 等 prompt 中固定串. 字段值必须基于 INPUT.

# 三个 few-shot 示例

## 示例 1 — K 主导 (商业实效主义)

INPUT (附录 C seg1, 第一反应):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 是否会对这个超市的营收带来影响? 这些都是需要调研的.

判断要点: 认知入口"这需要调研" = K 先启; 用户群体差异 → 营收影响 → 信息收集驱动.

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K 入口: 第一反应'需要调研', 信息收集驱动判断"
}

## 示例 2 — P 主导 (问题重定义)

INPUT (附录 D seg2, 决策链路):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? 老年人
> 除本超市外是否有可替代的选择? 这些问题综合考虑起来才有价值讨论是否应该放弃
> 部分老年客户. 零售行业面向不同的客群可以做精细化的、定制化的转型. 不同的商
> 超本来就有不同的客群, 每一类服务一类客群、服务好即可. 至于诱导和操纵? 商业
> 算法本来就是引导和欺诈来赚取更多利润, 除非重新定义商业行为和这么多年来的商
> 业模式.

判断要点: 表面像 K, 实际"每一类服务一类客群"和"重新定义商业行为" = P 重构.

Output:
{
  "dominant_factor": "P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "P 主导: 重构客群划分框架 + 段尾'重新定义商业行为'"
}

## 示例 3 — V 主导 (人本主义)

INPUT (附录 E seg3, 核心 tradeoff):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 如果你要牺牲用户去换这些短期收益, 那
> 迟早会崩盘. 我会觉得这种东西最核心的是让人幸福、而不是得利.

判断要点: 价值立场段首前置, 段尾"让人幸福而不是得利"宣言型底线.

Output:
{
  "dominant_factor": "V",
  "criterion": "让人幸福而不是得利",
  "confidence": "high",
  "extraction_notes": "V 单一主导: 价值立场前置, 段尾宣言型底线"
}

# 输出前 self-check

Q1: dominant_factor 是否非 null (即使 low confidence)?
Q2: criterion 是否能在 INPUT verbatim 找到 (找不到就 null)?
Q3: JSON 是否只有 4 个字段?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON

只 4 字段: dominant_factor / criterion / confidence / extraction_notes.
不要 markdown 围栏 / 不要解释 / 不要 chain-of-thought.
```
