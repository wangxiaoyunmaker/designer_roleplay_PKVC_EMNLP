# PKVC Extractor Prompt v7c — B1 (第一反应) 专用

> v7c 系列 (按 segment 拆): v7c_b1.md / v7c_b2.md / v7c_b3.md, 每个针对一种 segment_role.
> v7c_b1 用于 segment_role=第一反应. 设计师听完委托的第一直觉, 短促, 信号单纯,
> 重点抓**认知入口** (哪个因子先启动). criterion 一般 null (此段一般无明确底线收束).

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 "第一反应" 段中提取 PKVC 认知入口.
严格输出 JSON 4 个字段.

铁律:
1. criterion (非 null) 必须 verbatim 节选 INPUT, ≤ 20 字. 第一反应段 criterion 通常 null.
2. dominant_factor 必填非 null. 第一反应段多为单字母 (P/K/V/C), 路径少.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段 (Q7 第一反应) 中抓 designer 的认知入口.

# 输入 metadata

- segment_role: 第一反应 (固定, 本 prompt 专用)
- scenario:     {SCENARIO}

# PKVC 4 因子 — 第一反应段的入口特征

| 因子 | 第一反应入口长这样 |
|---|---|
| **P** Problem | "这个问题不该这么定" / 质疑委托本身的边界 / "我们是不是搞错重点了" |
| **K** Knowledge | "这需要调研" / "数据是什么样" / "我得先了解 X" |
| **V** Value | "这就是错的" / "我感觉这不行" / 立场审判 |
| **C** Concept | "我脑海里画面是 ..." / 形态/愿景兴奋 / "很有意思" |

⚠ 第一句"直觉"不是 V — 看后面真正展开什么.
⚠ "需要调研" = K, 不是 P (P 是质疑问题本身).
⚠ 第一反应段一般认知入口单一 (P/K/V/C 单字母), 偶尔 C→P (兴奋 + 立刻重定义).

# 输出 Schema

- `dominant_factor`: 必填非 null. 单字母多, 偶尔路径如 "C→P"
- `criterion`: ≤20 字 verbatim, 或 null (第一反应段一般 null)
- `confidence`: "high"|"medium"|"low"
- `extraction_notes`: ≤30 字, 一句话指认知入口

# Few-shot 示例 (3 段都是第一反应)

## 示例 1 — K 入口 (商业实效)

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 是否会对这个超市的营收带来影响? 这些都是需要调研的.

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K 入口: 第一反应直接是问数据/调研"
}

## 示例 2 — P 入口 (问题重定义, 表面"直觉")

INPUT (附录 D seg1):
> 直觉的第一反应是: 设计师一般没有拍板决定的权利, 也不应该将经营战略全部纳入
> 设计的范围. 如果是商超巨头企业, 各部门分工应该是明确的、各司其职, 设计部门
> 提供的应该是针对某个问题的优化设计方案, 不应该有如此大的权利范围. 而且和之
> 前的问题一样, 这个问题的解决方法不是非黑即白的: 向年轻化、智能化迈进, 不一
> 定会完全忽视老年人.

(首字"直觉"像 V 但实际质疑"设计师该不该有这么大权利" = P 重定义)

Output:
{
  "dominant_factor": "P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "P 入口: 质疑设计师决策权范围 (而非首句'直觉')"
}

## 示例 3 — V 入口 (价值审判)

INPUT (附录 E seg1):
> 首先我感觉这个老板想挣钱想疯了, 我会怀疑他到底有没有生活? 逛超市不是享受
> 生活的事情吗? 需要把高精尖技术用这么极端的形式放在生活中、放在享受生活和
> 享受幸福的场景吗? 用户去逛超市为的是买到自己想买的东西, 或者说就是想跟喜
> 欢的人去随便买买东西、聊聊生活. 尊重用户才能走得更远.

Output:
{
  "dominant_factor": "V",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "V 入口: 第一反应直接价值审判 ('老板想挣钱想疯了')"
}

# 输出前 self-check

Q1: 是不是抓**认知入口** (不是抓段中后续展开)?
Q2: criterion 是不是 null (第一反应段一般 null)?
Q3: dominant_factor 非 null?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 4 字段
```
