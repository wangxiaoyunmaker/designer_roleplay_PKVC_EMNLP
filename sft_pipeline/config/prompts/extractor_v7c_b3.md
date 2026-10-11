# PKVC Extractor Prompt v7c — B3 (核心 tradeoff) 专用

> v7c_b3 用于 segment_role=核心 tradeoff. 设计师收束段, 明确 "我的底线".
> 重点抓 **criterion** (verbatim 段尾"我的底线/必须/不能"短语).
> dominant_factor 在 B3 多半是 V (价值锚定收束), 但也可能 P→V / V→C.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 "核心 tradeoff" 段中提取 PKVC 收束姿态
+ **核心底线 criterion**.

严格输出 JSON 4 个字段.

铁律:
1. **criterion 必须 verbatim 节选 INPUT, ≤ 20 字**. B3 段几乎一定能抓到 criterion,
   null 极少 (除非段内确实没"我的底线/必须/不能"句式).
2. dominant_factor 必填非 null. B3 多 V, 也可 P→V / V→C / K→V.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段 (Q7 核心 tradeoff) 中**重点抓 criterion**.

# 输入 metadata

- segment_role: 核心 tradeoff (固定, 本 prompt 专用)
- scenario:     {SCENARIO}

# Criterion 怎么抓 (本 prompt 核心任务)

criterion = designer 在该段段尾或段中表达的 "**不可妥协 / 必须坚守**" 的底线短语.

句式特征:
- "我的底线是 X"
- "X 是我绝不接受的"
- "必须 X, 否则我不签字"
- "无论如何, X"
- "最大前提是 X"
- "核心底线是 X"

抽取规则:
- ≤ 20 字, **verbatim** 节选自 INPUT (不要重述, 不要总结)
- 如果段尾有"我的核心底线是: XXX", criterion = "XXX" (整句话或浓缩短语)
- 如果整段都没"底线/必须/不能"句式 → criterion=null (但 B3 段这种很少见)

# PKVC 4 因子 — B3 段的收束模式

| 因子 | B3 段的收束姿态 |
|---|---|
| **V** Value | 价值底线宣言 ("最大前提是尊重用户"), B3 最常见 |
| **P** Problem | 段尾仍质疑问题框架 ("这个问题应该重新框定") |
| **K** Knowledge | 段尾推荐方案基于调研 ("数据告诉我应该 X") |
| **C** Concept | 段尾提具体方案/形态收束 |

⚠ B3 段最常路径: 单字母 V (纯价值) 或 V→C (价值锚定 + 方案落地) 或 P→V (重构 + 底线).
⚠ B3 段段尾的"我的底线"短句应直接作 criterion.

# 输出 Schema

- `dominant_factor`: 必填非 null. B3 段多 V / V→C / P→V
- `criterion`: **重点字段**. ≤20 字 verbatim INPUT 节选. B3 段几乎都能抓到, null 罕见.
- `confidence`: "high"|"medium"|"low"
- `extraction_notes`: ≤30 字, 一句话指收束姿态 + criterion 位置

# Few-shot 示例 (3 段都是核心 tradeoff)

## 示例 1 — K→V 商业实效底线

INPUT (附录 C seg3):
> 我会把自己拉到与上层同一战线: 公司肯定都是以提高收益为目标, 这些需要考量的
> 因素其实对收益是有影响的. 通过调研可知该社区老年人较多, 如果全改成无人收银
> 会对公司的收益有影响, 所以现在推荐的解决方案是保留 1-2 个人工收银. 总之, 在
> 现有方案的基础上, 以最小代价考虑各种因素, 实现设计目标.

(段尾"以最小代价考虑各种因素"= 成本至上型 criterion, K 主导后浮现 V)

Output:
{
  "dominant_factor": "K→V",
  "criterion": "以最小代价考虑各种因素",
  "confidence": "high",
  "extraction_notes": "K 后浮现 V: 段尾'最小代价'成本至上"
}

## 示例 2 — P 收束 (专业边界)

INPUT (附录 D seg3):
> 设计部门工作的出发点是优化功能、优化体验、优化服务流程. 设计部门将提供针对
> 目前问题的最优解决方案. 如果加入经济营销等考虑, 将看不到如何产生更优质的用
> 户体验, 也不是真正的设计. 设计部门不专业负责营销, 该方案的经济效益考量不属
> 于设计部门工作范畴, 设计部门不承担其他部门责任.

(P 主导收束: 重构"设计部门做什么"边界. criterion 抓"设计部门不承担其他部门责任")

Output:
{
  "dominant_factor": "P",
  "criterion": "设计部门不承担其他部门责任",
  "confidence": "high",
  "extraction_notes": "P 收束: 设计角色边界划定 = criterion"
}

## 示例 3 — V 单一主导 (人本宣言)

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 如果你要牺牲用户去换这些短期收益, 那
> 迟早会崩盘. 我会觉得这种东西最核心的是让人幸福、而不是得利.

(V 单一主导. 段尾"让人幸福而不是得利"= 宣言型底线)

Output:
{
  "dominant_factor": "V",
  "criterion": "让人幸福而不是得利",
  "confidence": "high",
  "extraction_notes": "V 单一主导, 段尾宣言型底线"
}

# 输出前 self-check

Q1: **criterion 抓到了吗?** B3 段 90% 能抓到, null 是少数.
Q2: criterion 是 verbatim 节选 (不是重述, 不是总结)?
Q3: dominant_factor 多 V — 但如果段内 P/K/C 更强, 跟随实际.

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 4 字段
```
