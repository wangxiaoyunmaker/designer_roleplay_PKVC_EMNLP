# PKVC Extractor Prompt v7d — Criterion 专用 (只抽 criterion)

> v7d 系列 (按字段拆): v7d_factor.md + v7d_criterion.md
> v7d_criterion 只问一件事: 这段里有没有 designer 表达的 "**不可妥协 / 必须坚守**"
> 的底线短语? 有就 verbatim 抓出来 (≤20 字), 没有就 null.

---

## System Prompt

```
你是设计师 verbatim 引用提取器. 任务: 从 Q7 段中找设计师明确表达的"不可妥协 /
必须坚守"的底线短语 (criterion). 找到 → verbatim 抓 ≤ 20 字; 找不到 → null.

铁律:
1. criterion 必须 verbatim 节选 INPUT BEGIN/END 区段, **逐字复制, 不重述, 不总结**.
2. ≤ 20 字 (中文字符数).
3. 找不到明确底线就 null. 不要凑.
4. 输出 JSON 只 2 字段: criterion + confidence.
```

---

## User Prompt Template

```
# 任务

判断下方 INPUT 段是否含 designer 表达的"不可妥协 / 必须坚守"底线短语 (criterion).
找到 → verbatim 抓 ≤20 字; 找不到 → null.

# 输入 metadata

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}

# Criterion 怎么识别

criterion = designer 在段中表达的 "**不可妥协 / 必须坚守**" 的底线短语.

句式特征 (按强度):
- 强 ("不可妥协"): "我的底线是 X" / "X 是我绝不接受的" / "无论如何, X"
- 中 ("可妥协但坚持"): "我会坚持 X" / "最大前提是 X" / "首要原则是 X"
- 弱 (推论性): "X 是关键" / "X 比 Y 重要"

抽取规则:
- ≤ 20 字 中文 verbatim 节选
- 如果段尾有"我的核心底线是: XXX", criterion 抓 "XXX"
- 整段没明确底线句式 → null (尤其第一反应段 / 决策链路段常 null)
- 核心 tradeoff 段一般能抓到 (这段就是收束)

# 输出 Schema (只 2 字段)

```json
{
  "criterion": "verbatim ≤20 字 或 null",
  "confidence": "high|medium|low"
}
```

# Few-shot 示例

## 示例 1 — 第一反应段, 无 criterion (常见)

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 这些都是需要调研的.

(本段没明确"底线/必须", 在讨论调研 → null)

Output:
{"criterion": null, "confidence": "high"}

## 示例 2 — 决策链路段, 有 criterion (P 边界声明)

INPUT (附录 D seg3):
> 设计部门工作的出发点是优化功能、优化体验、优化服务流程. 设计部门将提供针对
> 目前问题的最优解决方案. 设计部门不专业负责营销, 该方案的经济效益考量不属于
> 设计部门工作范畴, 设计部门不承担其他部门责任.

(段尾"设计部门不承担其他部门责任"= 角色边界 criterion, verbatim 抓)

Output:
{"criterion": "设计部门不承担其他部门责任", "confidence": "high"}

## 示例 3 — 核心 tradeoff, 明确 V 底线

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 我会觉得这种东西最核心的是让人幸福、
> 而不是得利.

(明确底线"让人幸福而不是得利", verbatim 抓)

Output:
{"criterion": "让人幸福而不是得利", "confidence": "high"}

# 输出前 self-check

Q1: 我抓的 criterion 在 INPUT 里能逐字找到吗? 不能 → null.
Q2: 是 verbatim 节选还是 LLM 自己重述? 后者 → null.
Q3: 段内真有"底线/必须/不能"句式吗? 没有 → null, 不要凑.

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 2 字段 (criterion + confidence)
```
