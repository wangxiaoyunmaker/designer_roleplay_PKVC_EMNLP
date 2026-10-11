# PKVC Extractor Prompt v7c — B2 (决策链路) 专用

> v7c_b2 用于 segment_role=决策链路. 设计师的中段权衡 + 推理展开,
> 因子常切换, 重点抓 **dominant_factor 路径** (e.g. K→V / P→K / V→C).
> criterion 一般 null 或弱 (此段以推理为主, 不是宣言段).

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 "决策链路" 段中提取 PKVC 主导因子 + 路径.
严格输出 JSON 4 个字段.

铁律:
1. criterion (非 null) 必须 verbatim 节选 INPUT, ≤ 20 字. 决策链路段常 null.
2. dominant_factor 必填非 null. 决策链路常出现**路径** (e.g. "K→V"), 但单字母也合法.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段 (Q7 决策链路) 中抓 designer 的认知因子路径.

# 输入 metadata

- segment_role: 决策链路 (固定, 本 prompt 专用)
- scenario:     {SCENARIO}

# PKVC 4 因子 — 决策链路段的路径模式

| 因子 | 决策链路段的体现 |
|---|---|
| **P** Problem | 反复重构问题框架, 提出"或者其实问题是 X" |
| **K** Knowledge | 调用数据/案例/调研结果做权衡, "如果数据显示 X 那 Y" |
| **V** Value | 价值排序, "X 比 Y 更重要", 但本段 V 多为推理性而非宣言性 |
| **C** Concept | 提出具体方案/形态, "我会做一个 X 的设计" |

⚠ 决策链路段常**路径**: K→V (调研后浮现价值底线), P→K (重构问题后再查数据),
   V→K→C (价值先行 + 数据验证 + 具体方案)
⚠ 看转折点 + 段尾收束姿态判终点 = o_primary.
⚠ 不要过度抽路径 — 如果整段单一因子主导, 给单字母 (例如全段都在调数据 = K).

# 输出 Schema

- `dominant_factor`: 必填非 null. 决策链路段多为路径如 "K→V", 也可单字母
- `criterion`: ≤20 字 verbatim, 或 null (决策链路段常 null)
- `confidence`: "high"|"medium"|"low"
- `extraction_notes`: ≤30 字, 指明路径起点 + 终点

# Few-shot 示例 (3 段都是决策链路)

## 示例 1 — K 主导无切换 (商业实效)

INPUT (附录 C seg2):
> 首先我会明确自己的设计目标. 比如这个超市有一个很实际的目标就是提高收益. 其
> 次我认为这三个需要考量的因素并不矛盾, 通过设计是能够进行有效权衡的. 因为目
> 前已经有初步的方案了, 我会通过调研和测试去验证这个方案是否真的能提高收益:
> 比如这个社区的用户群体是怎么样的, 他们的需求到底是什么. 如果调研显示对收益
> 没有影响那是最好; 如果有影响, 我会基于这些访谈与数据对现有的设计方案进行优
> 化和迭代.

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K 单一主导: 全段调研驱动, 无明显切换"
}

## 示例 2 — P 主导 + K 工具 (问题重构)

INPUT (附录 D seg2):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? 老年
> 人除本超市外是否有可替代的选择? 这些问题综合考虑起来才有价值讨论是否应该
> 放弃部分老年客户. 零售行业面向不同的客群可以做精细化的、定制化的转型. 不
> 同的商超本来就有不同的客群, 每一类服务一类客群、服务好即可. 至于诱导和操
> 纵? 商业算法本来就是引导和欺诈来赚取更多利润, 除非重新定义商业行为.

(K 数据需求 → P 客群划分重构 → P 段尾"重新定义商业行为", 终点 P)

Output:
{
  "dominant_factor": "K→P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K (数据需求) → P (重构客群划分) → 段尾 P 终点"
}

## 示例 3 — V→K 价值锚定后调用工具 (人本)

INPUT (附录 E seg2):
> 首先, 数据和商业要求是完成的硬指标, 要先保证在这个基础上进行优化. 用户伦理
> 我认为是较为弹性的指标, 有点像连接数据和商业要求各个板块的一条河流: 先确定
> 好数据和商业要求的基本线之后, 基于用户伦理针对服务细节进行润滑优化. 但是否
> 尊重用户、是否符合所谓的用户伦理, 是更贴近人、贴近真实生活和情感流动的部
> 分. 如果这个东西的本质是欺骗, 那它也无法实现细水长流.

(V 价值排序贯穿全段, 调 K 作工具; 终点 V)

Output:
{
  "dominant_factor": "V",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "V 主导: 价值排序贯穿, K 仅为工具"
}

# 输出前 self-check

Q1: 路径起点 + 转折点 + 终点是否都看了? (终点 = o_primary)
Q2: 是不是过度抽路径 (单一主导就不要硬拆)?
Q3: criterion null 还是 verbatim?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 4 字段
```
