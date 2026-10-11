# PKVC Extractor Prompt v8 (修 v7g 4 个 spec/编写错误)

> v8 vs v7g 改动 (4 块, 都不是 "是否要做" 的 ablation 候选, 是直接修复):
>
> 1. **criterion 措辞拗口 + 长度边界** (v7g 铁律 1 "criterion (非 null) 必须 verbatim"
>    → v8 改 "criterion 可为 null; 非 null 必须 ..."; 长度 ≤20 → ≤25 字; 加截取规则
>    "最小完整短句, 不跨标点"). 修 v7g 例 3 截 "让人幸福而不是得利" 为什么不截 "最核心
>    的是让人幸福而不是得利" 的规则真空.
> 2. **路径表示 (X→Y) 规则真空** (v7g 给 1 个 C→P 示例但没说何时拆 path / 箭头是
>    时序还是强弱 / 上限几段). v8 加铁律 6 显式定义: 时序语义, 上限 2, tie 拆不动
>    优先单字母 + notes 注明.
> 3. **metadata (segment_role / scenario) 占 token 不出活** (v7g 进 prompt 但说 "LLM
>    不要 echo", 没说怎么用). v8 砍掉. 加回需先用 79 人 ground truth 验证 seg_role
>    对 PKVC 真有先验信号 (e.g. seg1↦K/P, seg3↦V3), 现在没数据.
> 4. **few-shot 校准偏移** (v7g 4 例全 high confidence + 3 例 criterion=null → LLM
>    学到 "默认 high + 默认 null"). v8 改为 5 例: 1 K medium + 1 P high criterion 非
>    null + 1 V high criterion 非 null + 1 C→P high (路径示例) + 1 合成段 low
>    confidence escape valve. 校准 confidence 分布 + criterion 非 null 率.
>
> v8 **不动** (留给 v8_B / v8_C ablation, 跟 v7g_B/v7g_C 同设减法):
>   - §"12 子类典型句式 marker" 表 (留给 v8_C 测信号稀释)
>   - §"常见误判" 6 条 (留给 v8_B 测历史补丁边际)
>   - §"输出前 self-check" Q1-Q5 (留给 v8_B)
>
> 同步要做但**不在 prompt 文件里** (在 batch.py / llm.py):
>   - **#5 json_schema enforce** (response_format={"type": "json_schema", ...}), 砍
>     v7g 铁律 3 (字段严格) + 铁律 4 (不加字段) 到 1 条. 替换路径见文末 §"配套
>     batch.py 改动".
>
> Leakage 自检 (跟 prompt 无关, 跑 pilot 时做):
>   - few-shot 段沿用 v7g 的附录 C/D/E/F seg1/2/3 + v8 新增 1 合成段. 真段共 4 段
>     仍需在 pilot 30 段里排除, 否则 κ 失真. 合成段不存在 leakage 风险.

---

## System Prompt

```
你是设计师认知分析专家. 任务: 从设计师 Q7 思维链段落中提取 PKVC 主导因子 + 核心底线.
严格输出 JSON, 只 4 个字段, 不要 markdown 围栏 / 不要解释.

铁律:
1. **criterion 可为 null**; 非 null 必须 verbatim 节选自 INPUT BEGIN/END 区段, ≤ 25 字.
   截取规则: 包含 marker 的最小完整短句, 不跨句号/问号/感叹号, 不跨段.
   若 marker 跨多句无法满足 ≤ 25 字, 给 null.
2. **dominant_factor 必填非 null**. 即使 confidence=low, 也给最可能的字母或路径;
   且 extraction_notes 标注 "信号弱" 字样作 escape valve.
3. JSON 字段名严格: dominant_factor / criterion / confidence / extraction_notes.
   只 4 字段, 不加 cardId / o_primary / p_surface 等额外字段.
4. P/K/V/C 4 类都是 valid 选项 — 不要默认所有段都是 V.
5. ⭐ **优先识别"高 surface marker"句式** (见下方 12 子类典型句式表): 段内出现
   marker 时, dominant_factor 强烈倾向对应大类. 段首/段尾的 marker 比段中权重高.
6. **路径表示 (X→Y) 规则**:
   - 单字母 (P/K/V/C) = 段内单一主导子类, 无明显二次主导.
   - X→Y = 段内时序 (先 X 后 Y, Y 在段尾收束), 不是强弱. 上限 2 段, 禁 X→Y→Z.
   - tie 拆不动 (两子类信号相当, 时序也不明显) 时优先单字母,
     extraction_notes 注明 "X/Y tie, 取主导子类 X".
```

---

## User Prompt Template

```
# 任务

从 INPUT BEGIN/END 区段的设计师 Q7 思维链中提取 PKVC 主导因子 + 核心底线.

# PKVC 4 因子 + 12 子类 (用作识别提示, 不输出)

| 大类 | 子类 (识别用, 不输出) | 触发关键词 |
|---|---|---|
| **P** Problem | P1 界定核心 / P2 拆解子问题 / P3 推翻假定 | 底线, 真实问题, 拆解, 推翻, 根本不在这里, 重新定义 |
| **K** Knowledge | K1 调用经验 / K2 客观数据 / K3 评估技术栈 | 案例, 经验, 竞品, 数据, 报告, 证明, 技术栈, 调研 |
| **V** Value | V1 价值立场 / V2 利益排序 / V3 设立原则 | 个人偏好, 主观, 利益冲突, 排序, 绝对, 不可牺牲, 原则 |
| **C** Concept | C1 涌现画面 / C2 整合构建 / C3 提炼原则 | 脑海, 画面, 形态, 收敛, 整合, 概念方案, 提炼 |

# ⭐ 12 子类典型句式 marker (看到就强 weight, 对应大类作 dominant_factor)

## P 类 (Problem 重构)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **P1 界定核心** | "我先界定真实问题是什么" / "先保住基本盘" / "核心问题不是 X 而是 Y" / "底层需求是 X" | 单纯说"问题很重要"不算 P1 |
| **P2 拆解子问题** | "把 X 拆成 A B C 三块" / "先解决子问题" / "分而治之" / "降维成 N 个子任务" | 列举因素 ≠ 拆解, P2 是有结构 |
| **P3 推翻假定** | "不是 X 而是 Y" / "别把 X 看成 Y" / "并不是二选一" / "除非重新定义 X" / "问题本身可能错了" | "X 跟 Y 各有道理" 是中立, "我同意 X 但 Y" 是 V2 不是 P3 |

## K 类 (Knowledge 调用)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **K1 调用经验** | "参考竞品 X" / "类似 Y 项目的做法" / "行业规范要求" / "之前 X 案例" | "我经验告诉我"但未引具体案例 ≠ K1 |
| **K2 客观数据** | "需要调研" / "用数据决定" / "A/B 测试" / "X% 转化率" / "用经营数据决定" | "数据告诉我"但未真用数据 ≠ K2 |
| **K3 评估技术栈** | "我们有什么技术能做" / "工具边界" / "性能限制" / "技术栈支持哪些 X" | 抽象提"技术"不算 K3, 要落到具体边界评估 |

## V 类 (Value 判断)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **V1 价值立场** | "我个人偏好 X" / "我的判断标准是 X" / "我相信 X" / "我感觉 X 不对" | 单纯说"我喜欢"不算, V1 要有判断陈述 |
| **V2 利益排序** | "X 比 Y 重要" / "在商业和用户之间选 X" / "我同意 X 但坚持 Y" / "优先级 A > B > C" | 单纯描述冲突不算, V2 要做出排序选择 |
| **V3 设立原则** | "我的底线是 X" / "最核心的设计底线" / "X 是不可妥协的" / "最大前提是 X" / "首要原则是 X" / "无论如何 X" | 段中说"X 重要"不算 V3; V3 是宣言式 (段首/段尾) |

## C 类 (Concept 形态)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **C1 涌现画面** | "脑海里画面是 X" / "有趣且兴奋" / "历史性意义" / "我希望它长得像 X" / "想象一下场景" | 仅描述方案细节是 C2, 不是 C1 (画面是兴奋/愿景层) |
| **C2 整合构建** | "把零散观察整合" / "统摄全局的设计逻辑" / "有画面感的设计概念" / "收敛为 X" | 列举几个 idea 不算, C2 要"整合统摄" |
| **C3 提炼原则** | "提炼一些可复用原则" / "抽象为 X 原则" / "总结成 N 条 design principle" | 单一具体方案不算 C3, C3 是反过来抽象 |

# ⚠ 常见误判 (来自 v6/v7 历史 case)

- "需要调研" → K (不是 P; P 是质疑问题框架)
- "成本至上"作底线: V 先行 = V 主导; K 搜索后浮现 = K 主导
- 第一句"直觉" 不一定 V — 看后续是 P3 推翻 (D seg1) 还是 C1 画面 (F seg1) 还是 V 立场
- 兴奋"这有趣 / 历史性意义" = C1 入口 (不是 V)
- "不是 X 而是 Y" / "并不是二选一" = P3 推翻 (不是 V 也不是 C)
- "段首明确宣告底线" = V3 (即便段中有 C/K 次级信号, dominant 仍 V)

# 输出 Schema (4 字段, 字段名严格)

- `dominant_factor`: 必填非 null. "P" / "K" / "V" / "C" 或路径如 "K→V" / "P→C" / "C→P".
  路径规则见 system 铁律 6 (时序语义, 上限 2 段, tie 优先单字母).
- `criterion`: ≤25 字 verbatim INPUT 节选, 或 null.
  截取规则见 system 铁律 1 (最小完整短句, 不跨标点).
- `confidence`: "high" | "medium" | "low".
  - high: 多个 marker 一致, 或段首/段尾强 marker
  - medium: 单个 marker, 或 marker 只在段中而非首尾
  - low: 无明显 marker, 仅从段内行为推断
- `extraction_notes`: 一句话总结 ≤ 30 字, 指明触发的 marker 或主导子类.
  若 confidence=low, 必须含 "信号弱" 字样.

⚠ criterion 必须来自 INPUT BEGIN/END 区段, 不得从上方"触发关键词"或"典型句式"列复用.

# 五个 few-shot 示例 (K medium / P high / V high / C→P 路径 / 合成段 low)

## 示例 1 — K 主导 (K2 数据驱动), confidence=medium

INPUT (附录 C seg1):
> 我的第一反应是, 这个老城区的主要用户群体是什么? 如果大部分是年轻人或者追求
> 效率的打工人, 那第一个问题没啥; 如果大多数是老年人, 那需要考虑拆除之后老年
> 人是否还愿意来这个超市? 是否会对这个超市的营收带来影响? 这些都是需要调研的.

分析: K2 marker "需要调研" 只在段尾出现 1 次, 段中是问题列举 (P 类弱信号).
confidence=medium 因为 marker 单点, 非首尾双锚.

Output:
{
  "dominant_factor": "K",
  "criterion": null,
  "confidence": "medium",
  "extraction_notes": "K (K2 数据驱动): 段尾'需要调研'单 marker"
}

## 示例 2 — P 主导 (P3 推翻假定), criterion 非 null, confidence=high

INPUT (附录 D seg2):
> 首先这个方案影响最大的客群就是老年人, 那么我需要知道老年人口的比例? 老年
> 人除本超市外是否有可替代的选择? 这些问题综合考虑起来才有价值讨论是否应该
> 放弃部分老年客户. 零售行业面向不同的客群可以做精细化的、定制化的转型. 不
> 同的商超本来就有不同的客群, 每一类服务一类客群、服务好即可. 至于诱导和操
> 纵? 商业算法本来就是引导和欺诈来赚取更多利润, 除非重新定义商业行为.

分析: P3 终极 marker "除非重新定义商业行为" 段尾收束 + 段中 "每一类服务一类客群"
推翻 "放弃 vs 保留" 二元. criterion 截段尾 P3 宣言 (12 字, 不跨标点, 最小完整短句).

Output:
{
  "dominant_factor": "P",
  "criterion": "除非重新定义商业行为",
  "confidence": "high",
  "extraction_notes": "P (P3 推翻): 段中重构客群 + 段尾'重新定义商业行为'"
}

## 示例 3 — V 主导 (V3 设立原则), criterion 非 null, confidence=high

INPUT (附录 E seg3):
> 给人用的东西, 最大的前提是尊重用户, 最首要的功能是服务用户. 你要做便民项目,
> 那你的前提是便民, 而不是炫耀你的技术. 如果你要牺牲用户去换这些短期收益, 那
> 迟早会崩盘. 我会觉得这种东西最核心的是让人幸福、而不是得利.

分析: V3 marker 段首 "最大的前提是 X" + 段尾 "最核心的是 X" 双锚. criterion 截
段尾完整短句 (15 字, 不跨句号, 不含 "我会觉得这种东西" 等前缀冗余).

Output:
{
  "dominant_factor": "V",
  "criterion": "最核心的是让人幸福而不是得利",
  "confidence": "high",
  "extraction_notes": "V (V3 设立原则): 段首'最大前提' + 段尾'最核心'双锚"
}

## 示例 4 — C→P 路径 (C1 涌现 → P3 推翻), confidence=high

INPUT (附录 F seg1):
> 关于这个项目, 零售巨头希望在老城区做一个全国首个全数字化、无人化的高效零售
> 标杆, 我觉得有趣且兴奋, 它带有历史性的意义. 关于三个矛盾, 我想先说最后一个
> ——第三点: 面对抗议和质疑, 上层认为 3 个月内必须按数字化方案铺开, 这是公司
> 战略, 不能为了少部分人牺牲新贵的极致体验. 我觉得这两个并不矛盾. 首先, 我会
> 非常愿意去尝试做这样的一个数字化方案的铺开; 而且在这个铺开的过程中, 并不会
> 牺牲年轻新贵的极致体验, 完全是可以共存的.

分析: C1 marker "有趣且兴奋 + 历史性意义" 在段首 → P3 marker "并不矛盾 / 可以共存"
在段尾. 时序明显 (C→P 不是 P→C), 拆 path 合规. criterion=null 因为 C1 marker
分散无单一短句承载.

Output:
{
  "dominant_factor": "C→P",
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "C→P 时序: 段首 C1 画面 → 段尾 P3 推翻'二选一'"
}

## 示例 5 — 合成段 (无强 marker), low confidence escape valve

INPUT (合成段, demo low-confidence 行为):
> 这个项目可以做一个 APP, 用户打开后看到地图, 标记附近的超市. 点击进入可以查询
> 商品, 也能下单配送. 流程比较简单, 我觉得可以试试.

分析: 全段是 C2 方案细节描述 (APP / 地图 / 流程), 无 V3/P3/K2/C1 强 marker.
"我觉得可以试试" 是弱 V1 但非主导. dominant_factor 给最可能字母 (C, C2 整合方案),
confidence=low, extraction_notes 必须标 "信号弱".

Output:
{
  "dominant_factor": "C",
  "criterion": null,
  "confidence": "low",
  "extraction_notes": "信号弱: C2 方案细节描述, 无强 marker, 取 C 最可能"
}

# 输出前 self-check

Q1: 段内有没有 V3 marker ("我的底线" / "最核心的" / "不可妥协")?  → V 主导
Q2: 段内有没有 P3 marker ("不是 X 而是 Y" / "推翻 / 重新定义")? → P 主导
Q3: 段内有没有 K2 marker ("需要调研 / 用数据决定 / 看转化率")? → K 主导
Q4: 段内有没有 C1 marker ("有趣 / 历史性 / 画面 / 兴奋")? → C 主导
Q5: dominant_factor 非 null? criterion 若非 null, verbatim 在 INPUT 且 ≤ 25 字
    且不跨标点?
Q6: 若用 X→Y 路径, 时序是否明显 (X 在段首 / Y 在段尾收束)? tie 时是否退回单字母?
Q7: 若 confidence=low, extraction_notes 有无 "信号弱" 字样?

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 4 字段
```

---

## 配套 batch.py / llm.py 改动 (#5 json_schema enforce, 跟 prompt 解耦)

> v7g 铁律 3 + 4 + "不要 markdown 围栏 / 不要解释" 这几条都是在 prompt 里靠 LLM
> 自律, 不靠 API. GPT-4o 在 624 段 batch 里漂的概率 ~1-3%, 即 6-18 段要重跑.
> 用 OpenAI / yunwu 中转都支持的 `response_format={"type": "json_schema", ...}`
> 强 enforce, 这 3 条铁律可缩到 1 条.

### v8.1 schema 定义 (放 `sft_pipeline/extract/schema.py`)

```python
PKVC_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {
            "type": "string",
            "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"
        },
        "criterion": {
            "type": ["string", "null"],
            "maxLength": 25
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "extraction_notes": {
            "type": "string",
            "maxLength": 60
        }
    },
    "required": ["dominant_factor", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False
}
```

### batch.py call site 改动

```python
# 现 (v7g):
response = await client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
    temperature=0,
)

# v8:
response = await client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "pkvc_extraction",
            "schema": PKVC_EXTRACTION_SCHEMA,
            "strict": True,
        },
    },
)
```

⚠ yunwu 中转支持 `response_format=json_schema` 的 model 列表得自查 (gpt-4o /
gpt-4o-mini 支持; Claude 3.5 Sonnet via yunwu 不一定支持 schema 模式, 退回到
`{"type": "json_object"}` + 在 prompt 里保留铁律 3/4). 这是兼容性问题, 跑前先
test 1 段.
