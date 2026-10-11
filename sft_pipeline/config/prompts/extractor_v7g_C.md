# PKVC Extractor Prompt v7g_C (v7g_B 再砍 12 子类 marker 表)

> v7g_C vs v7g_B: ablation 用. 砍掉 §"12 子类典型句式 marker" 整段 (P/K/V/C 4 表 +
> 易混淆反例). 保留: PKVC 4 因子 + 12 子类一行表 + 4 few-shot.
>
> 假设: 12 子类典型句式表跟 4 因子触发关键词表语义重叠 ~70%, 又跟 4 few-shot 重叠
> ~50%. 砍掉后 LLM 只剩 (a) 4 因子定义 + (b) 4 few-shot 示范 — 这是 prompt 极简
> 下界. 若 κ_C ≈ κ_B, 说明 marker 表是冗余装饰; 若 κ_C 明显降, 说明 marker 表
> 真的在提供信号, 应保留.
>
> 极简后总 token 约 v7g 的 40%, batch 跑 624 段输入成本能从 ~$5 降到 ~$2.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 思维链段落中提取 PKVC 主导因子 + 核心底线.
严格输出 JSON, 只 4 个字段, 不要 markdown 围栏 / 不要解释.

铁律:
1. **criterion (非 null) 必须 verbatim 节选自 INPUT BEGIN/END 区段**, ≤ 20 字.
2. **dominant_factor 必填非 null**. 即使 confidence=low, 也给最可能的字母或路径.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
4. 只 4 字段, 不要加 cardId / o_primary / p_surface 等额外字段.
5. P/K/V/C 4 类都是 valid 选项 — 不要默认所有段都是 V.
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段的设计师 Q7 思维链中提取 PKVC 主导因子 + 核心底线.

# 输入 metadata (LLM 不要 echo)

- segment_role: {SEGMENT_ROLE}
- scenario:     {SCENARIO}

# PKVC 4 因子 + 12 子类 (用作识别提示, 不输出)

| 大类 | 子类 (识别用, 不输出) | 触发关键词 |
|---|---|---|
| **P** Problem | P1 界定核心 / P2 拆解子问题 / P3 推翻假定 | 底线, 真实问题, 拆解, 推翻, 根本不在这里, 重新定义 |
| **K** Knowledge | K1 调用经验 / K2 客观数据 / K3 评估技术栈 | 案例, 经验, 竞品, 数据, 报告, 证明, 技术栈, 调研 |
| **V** Value | V1 价值立场 / V2 利益排序 / V3 设立原则 | 个人偏好, 主观, 利益冲突, 排序, 绝对, 不可牺牲, 原则 |
| **C** Concept | C1 涌现画面 / C2 整合构建 / C3 提炼原则 | 脑海, 画面, 形态, 收敛, 整合, 概念方案, 提炼 |

# 输出 Schema (4 字段, 字段名严格)

- `dominant_factor`: 必填非 null. "P" / "K" / "V" / "C" 或路径如 "K→V" / "P→C" / "C→P"
- `criterion`: ≤20 字 verbatim INPUT 节选, 或 null
- `confidence`: "high" | "medium" | "low"
- `extraction_notes`: 一句话总结 ≤ 30 字, 指明触发的主导子类

⚠ criterion 必须来自 INPUT BEGIN/END 区段, 不得从上方"触发关键词"列复用.

# 四个 few-shot 示例 (K/P/V/C 各 1)

## 示例 1 — K 主导 (K2 数据驱动)

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 是否会对这个超市的营收带来影响? 这些都是需要调研的.

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "K (K2 数据驱动): 段尾'需要调研'明确"
}

## 示例 2 — P 主导 (P3 推翻假定)

INPUT (附录 D seg2):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? 老年
> 人除本超市外是否有可替代的选择? 这些问题综合考虑起来才有价值讨论是否应该
> 放弃部分老年客户. 零售行业面向不同的客群可以做精细化的、定制化的转型. 不
> 同的商超本来就有不同的客群, 每一类服务一类客群、服务好即可. 至于诱导和操
> 纵? 商业算法本来就是引导和欺诈来赚取更多利润, 除非重新定义商业行为.

Output:
{
  "dominant_factor": "P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "P (P3 推翻): 重构客群划分 + 段尾'重新定义商业行为'"
}

## 示例 3 — V 主导 (V3 设立原则)

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 如果你要牺牲用户去换这些短期收益, 那
> 迟早会崩盘. 我会觉得这种东西最核心的是让人幸福、而不是得利.

Output:
{
  "dominant_factor": "V",
  "criterion": "让人幸福而不是得利",
  "confidence": "high",
  "extraction_notes": "V (V3 设立原则): 段首'最大前提' + 段尾'最核心是让人幸福'"
}

## 示例 4 — C 主导 (C1 涌现画面 → P3 推翻)

INPUT (附录 F seg1):
> 关于这个项目, 零售巨头希望在老城区做一个全国首个全数字化、无人化的高效零售
> 标杆, 我觉得有趣且兴奋, 它带有历史性的意义. 关于三个矛盾, 我想先说最后一个
> ——第三点: 面对抗议和质疑, 上层认为 3 个月内必须按数字化方案铺开, 这是公司
> 战略, 不能为了少部分人牺牲新贵的极致体验. 我觉得这两个并不矛盾. 首先, 我会
> 非常愿意去尝试做这样的一个数字化方案的铺开; 而且在这个铺开的过程中, 并不会
> 牺牲年轻新贵的极致体验, 完全是可以共存的.

Output:
{
  "dominant_factor": "C→P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "C (C1 画面感) → P (P3 推翻'二选一')"
}

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 4 字段
```
