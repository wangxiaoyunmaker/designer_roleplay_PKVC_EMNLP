# PKVC Extractor Prompt v7d — Factor 专用 (只抽 dominant_factor)

> v7d 系列 (按字段拆 2 个): v7d_factor.md + v7d_criterion.md
> v7d_factor 只问一件事: 这段的 dominant_factor 是 P/K/V/C 哪个 (或路径).
> 不抽 criterion, 不抽 T_CAPS — 任务最简单, LLM 注意力集中.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 判断一段 Q7 思维链的 PKVC 主导认知模式.
只输出 2 字段 JSON: dominant_factor + confidence.

铁律:
1. dominant_factor 必填非 null. "P" / "K" / "V" / "C" 单字母 或 路径如 "K→V" / "P→V".
2. confidence: "high" / "medium" / "low".
3. JSON 只 2 字段, 不要任何其他字段 (不要 criterion, 不要 extraction_notes).
4. 只输出 JSON, 不要 markdown 围栏 / 解释 / chain-of-thought.
```

---

## User Prompt Template

```
# 任务

判断下方 INPUT 段的 PKVC 主导认知模式. 只输出 dominant_factor + confidence.

# 输入 metadata (供参考, 不需 echo)

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}

# PKVC 4 因子诊断特征

判断基于**认知入口 + 推理结构 + 收束姿态**, 不基于关键词频率.

| 因子 | 诊断特征 |
|---|---|
| **P** Problem | 质疑问题框架本身, 把冲突"去稳定化", 问"这个问题对吗" |
| **K** Knowledge | 先理解再判断, 把价值操作化为可证伪经验命题 (调研/数据) |
| **V** Value | 先判断后推导, 价值立场段首确立, 后续是该立场的展开 |
| **C** Concept | 从形态/愿景入口, 而非问题/数据/价值入口 |

⚠ 常见误判:
- "需要调研" → K 入口 (不是 P, P 是质疑问题框架本身)
- "成本至上"作底线: 看 V 先行还是 K 搜索后浮现; 后者 K 主导
- 第一句"直觉" 不一定 V — 看后续展开

# 路径表达

如果段内主导因子切换 (e.g. K 入口 → V 锚定), dominant_factor 写路径字符串 "K→V".
段尾收束姿态决定路径终点. 单一主导写单字母.

# 输出 Schema (只 2 字段)

```json
{
  "dominant_factor": "K→V",
  "confidence": "high"
}
```

# 三个 few-shot (不同 segment_role 各一)

## 示例 1 — 第一反应段, K 入口

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? ... 这些都是需要调研的.

Output:
{"dominant_factor": "K", "confidence": "high"}

## 示例 2 — 决策链路段, P 主导

INPUT (附录 D seg2):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? ...
> 不同的商超本来就有不同的客群, 每一类服务一类客群、服务好即可. ... 除非重新
> 定义商业行为和这么多年来的商业模式.

Output:
{"dominant_factor": "P", "confidence": "high"}

## 示例 3 — 核心 tradeoff 段, V 主导

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. ... 我会觉得这种
> 东西最核心的是让人幸福、而不是得利.

Output:
{"dominant_factor": "V", "confidence": "high"}

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 2 字段 (dominant_factor + confidence)

不要 criterion 不要 extraction_notes 不要 markdown 围栏.
```
