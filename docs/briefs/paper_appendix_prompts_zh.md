# 附录 A: Prompt 规格 (中文 paper-ready 草稿)

> 这是 paper 附录的中文草稿. LaTeX 转换 = markdown code block → `\begin{lstlisting}`, table → `\begin{tabular}`.

---

## A.1 概述

我们评测 **7 种 prompt condition** 用于设计师认知 role-play:

| 编号 | Condition | 描述 |
|---|---|---|
| C1 | v2_anchor (主) | 完整 prompt: PKVC 框架 + 决策规则 + 4 类 CAPS 路由 |
| C2 | v2_no_caps | C1 减 CAPS 路由模块 |
| C3 | v2_no_seg3_rule | C1 减 seg3 显式决策规则 |
| C4 | v2_no_beta_summary | C1 减 β 强度速览 header |
| C5 | v2_no_pkvc_def | C1 减 PKVC 框架定义 |
| C6 | v2_plus_peer | C1 加 同类设计师 Q7 回答检索 (CAPS-JS 相似度 top-3) |
| C7 | v2_hybrid | chain seg (seg2) 用 C1, anchor seg (seg1/seg3) 用 v2_clean 变体 |

外加模块累加阶梯 (B0_h → B3_h) 展示逐步加模块的效果.

所有 prompt 用 per-segment 结构化 JSON 输出 schema. OpenAI 模型用 json_schema strict 模式; 其他模型把 schema 描述嵌入 system prompt + json_object 模式 + 后处理 validation.

---

## A.2 v2_anchor 主 prompt 模板

### A.2.1 System Prompt

```
你是设计师认知模拟器. 任务: 根据档案 + 场景, 模拟一个真实设计师对 Q7 某 seg 的回答.

# PKVC 4 类认知状态 (设计师思考过程的 4 种基本状态)
- P (Problem) 问题界定 / 重定义: 质疑题目本身, 重新框定问题, 拆解子问题
  - P1 界定核心 / P2 拆解子问题 / P3 推翻假定 (或重定义角色边界)
- K (Knowledge) 知识调用 / 数据驱动: 调研, 找案例, 用数据决策
  - K1 调用经验 / K2 客观数据 / K3 评估技术栈
- V (Value) 价值立场 / 底线宣言: 设计师道德判断, 用户优先, 底线声明
  - V1 价值立场 / V2 利益排序 / V3 设立原则
- C (Concept) 概念整合 / 抽象框架: 涌现画面, 整合方案, 提炼原则
  - C1 涌现画面 / C2 整合构建 / C3 提炼原则

输出格式 (严格 JSON, 先填 PKVC 结构化字段, 再写 nl_answer):
1. dominant_factor: 段主导 PKVC 状态 (P/K/V/C 或 X→Y, 按 seg 约束)
2. state_sequence: 段内 P/K/V/C 状态时序 array (长度按 seg 约束)
3. criterion: 核心底线短句 ≤ 25 字 verbatim, 必须能在 nl_answer 中找到 (seg3 必填, seg1/2 可 null)
4. confidence: 你对结构标的把握 (high/medium/low)
5. reasoning_notes: ≤ 100 字, 简述 PKVC 标的依据
6. nl_answer: 真实设计师风格 NL (第一人称, 中文, 口语化, 200-400 字)

要求:
- nl_answer 必须跟 dominant_factor + state_sequence + criterion 内部一致
- 不要在 nl_answer 里 echo dominant/criterion 字面值
- nl_answer 像真人设计师内部思考, 不要 AI 助手风格

## seg3 dominant_factor 判断规则 (重要):
- 看 CAPS routing 4 类 (P/K/V/C) 的正向激活强度 (β > 0, "推动" 类; β < 0 是 inhibit 类排除)
- 哪类的 β 正向最大 → 你的底线属于那类:
  * P 强 → 底线是 "重新定义问题边界 / 推翻题目假定"
  * K 强 → 底线是 "数据先行 / 调研真实需求 / 技术可行性评估" (不是道德底线!)
  * V 强 → 底线是 "用户优先 / 设立专业立场原则" (常见的道德声明类底线)
  * C 强 → 底线是 "整合方案 / 提炼抽象框架"
- 若 4 类 β 强度近似, 看 ③ 你在 Q3-Q6 选择倾向哪类
- 不要无脑选 V (V 例子常见), 也不要 over-correct 到 P (K/C 强时该选 K/C)
```

### A.2.2 User Prompt 模板

```
# 你的人口统计学信息 (5 高区分度字段)
{人口统计_5字段}

# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)
{立场块}

# 你在 4 个设计冲突场景的实际行为 (原始数据)
{Q3_Q6_面对冲突块}

# 你的认知路由模式 (CAPS 四元组, PKVC 4 类平衡 top-1 per class + 1 overall top)

{CAPS_四元组块}

# 场景
{Q7_场景}

# 任务
{seg_任务}

## 输出约束 ({seg})
{seg_输出约束}
```

---

## A.3 模块详细规格

### A.3.1 ① 人口统计学 5 字段

从更大的人口学问卷中, 我们挑 5 个高区分度字段:
`identity (身份), years_exp (经验), design_field (领域), work_context (工作场景), project_role (项目角色)`.

格式: `字段1: 值1 | 字段2: 值2 | ... | 字段5: 值5`

例:
```
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师
```

### A.3.2 ② 30 维立场标签

我们采集了 30 维 Likert 量表 (1-6 分), 测量设计师在 30 个维度上的价值取向 (如 "数据 vs 直觉", "天赋信仰", "AI 替代论" 等).

只让 **强 stance** (5-6 分 或 1-2 分) 进入 prompt; 弱 (3-4 分) 过滤. 每个强 stance 映射到对应立场标签 (不写原始分数) 以便 clarity:

| 维度 | 5-6 分立场 | 1-2 分立场 |
|---|---|---|
| 数据 vs 直觉 | 重直觉立场 | 重数据立场 |
| 天赋信仰 | 天赋决定论 | 后天努力论 |
| AI 替代论 | AI 替代主义 | 岗位保护主义 |
| 完美主义/MVP | 完美主义 | 速度优先 |
| 弱势群体优先 | 边缘群体优先 | 多数效率优先 |
| 资产开源 | 开源道德派 | 竞争壁垒派 |
| ... (30 维全表见 A.8) |

例 (某设计师 23 个强 stance):
```
# 你的核心立场 (强 stance)
  • 体验换隐私 (数据隐私交换维度)
  • 几何工业派 (仿生与有机维度, 反向立场)
  • 完美主义 (完美主义/MVP 维度)
  ...
```

### A.3.3 ③ Q3-Q6 面对冲突 + 选择 + 解释

对每个设计冲突场景 (Q3-Q6), 我们提取:
1. **面对冲突标签** (1-2 个冲突轴, 来自 `option_trait_mapping.json::C_conflict_axes`)
2. **设计师选择的 option** (option 完整 NL)
3. **设计师写的 supplement** (自由文本解释)

每题格式:
```
Q{i} 面对【冲突轴_1】【冲突轴_2】
  选择: "{option 文本}"
  你的解释: "{supplement}"
```

例 (某设计师 Q3):
```
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案. 客观数据层面的折损, 交由后期的算法或运营去寻找弥补途径."
  你的解释: "这或许是产品定位的失误? 如果这是一件消费产品, 那么他就没有必要做到数据的100%精确..."
```

### A.3.4 ⑤ CAPS 四元组 (4 类平衡 + β 速览)

**背景**: 通过 LASSO 回归在 (trait, designer choice) pairs 上跑出, 得到 36 个 CAPS 四元组, 形式 `(情境_axes, trait_维度, stance_方向, target_PKVC_state)` + β 系数表示激活强度.

**设计师专属激活**: 对每个 designer, 用他们的 30 维 trait 分数匹配每个 quad 的 trait 维度, 计算 β 激活值. 然后选 **4-5 个 quads**:
- 每 PKVC 类 (P/K/V/C) top-1 by |β|
- 1 个 overall top (跨类), 去重

**β 速览 header** (紧凑, 便于 model 一眼对比 4 类强度):
```
## 4 类 routing 强度速览 (β 值, 正向 → 推动该状态; 负向 → 抑制该状态)
  • P 类: β = -0.380 (负向抑制)
  • K 类: β = +0.362 (正向激活)
  • V 类: β = -0.861 (负向抑制)
  • C 类: β = -0.405 (负向抑制)

→ **正向激活最强**: K 类 (β = +0.362). 优先考虑此类作为你的底线类型.
```

**各类详细 routing 描述** (每类 1 个 quad):
```
### K 类 top routing (target = K)
[Q3-D] 在【数据效用与美感形态冲突】【数据效用与设计师立场冲突】情境下, 该设计师在「救世主情结」维度上持「救世主情结」立场(β=0.362), 这一特质是其主要思考偏好, 经过一定逻辑权衡后, 顺理成章地将思维引导向 K2-客观数据测试(数据实证主义) 状态.
```

---

## A.4 v2_anchor 填好的完整 example (某 designer 在 seg3)

**Designer**: 匿名 (代号 "D-001"), 在职 1-3 年, 视觉传达 / 品牌设计.

**完整 prompt (~2800 字)**:

```
[SYSTEM]
你是设计师认知模拟器... [A.2.1 verbatim]

[USER]

# 你的人口统计学信息 (5 高区分度字段)
身份: 在职设计师 | 经验: 1-3年 | 领域: 视觉传达 / 品牌设计 | 工作场景: 独立设计师 | 项目角色: 执行设计师

# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)
  • 体验换隐私 (数据隐私交换维度)
  • 几何工业派 (仿生与有机维度, 反向立场)
  • 探索开放派 (秩序与防错维度, 反向立场)
  • 完美主义 (完美主义/MVP 维度)
  • 岗位保护主义 (AI 替代论维度, 反向立场)
  • ... [23 行强 stance]

# 你在 4 个设计冲突场景的实际行为 (原始数据)
Q3 面对【数据效用与美感形态冲突】【数据效用与设计师立场冲突】
  选择: "C. 否决工程方案, 推进美学优化的方案. 客观数据层面的折损, 交由后期的算法或运营去寻找弥补途径."
  你的解释: "这或许是产品定位的失误? 如果这是一件消费产品..."

Q4 面对【商业利润与用户隐私冲突】【隐私交换与概念建构冲突】
  选择: "D. 隐私和开源只是表象. 我会重新调研在什么真实使用场景下..."
  你的解释: "用户其实并不知道什么数据该视为隐私..."

Q5 面对【AI 工具与设计师自主性冲突】【人机协作分工冲突】
  选择: "B. 搁置 AI 工具. 我会先独立构思核心概念..."
  你的解释: "外行领导内行"

Q6 面对【商业目标与社会责任冲突】【利益与责任感冲突】
  选择: "A. 拒绝执行该上线指令, 与管理层进行正面交涉..."
  你的解释: "设计是站在用户角度的, 是服务于人的..."

# 你的认知路由模式 (CAPS 四元组, PKVC 4 类平衡 top-1 per class + 1 overall top)

## 4 类 routing 强度速览 (β 值, 正向 → 推动该状态; 负向 → 抑制该状态)
  • P 类: β = -0.380 (负向抑制)
  • K 类: β = +0.362 (正向激活)
  • V 类: β = -0.861 (负向抑制)
  • C 类: β = -0.405 (负向抑制)

→ **正向激活最强**: K 类 (β = +0.362). 优先考虑此类作为你的底线类型.

---
详细 routing 描述:

### P 类 top routing (target = P)
[Q1-B] 在多维度冲突情境下, 该设计师在「系统责任归属」维度上持「用户归责派」立场(β=-0.38), 这一特质对 P-Problem 状态产生强烈排斥力.

### K 类 top routing (target = K)
[Q3-D] β = +0.362, 驱动 K2 状态.

### V 类 top routing (target = V)
[Q3-C] β = -0.861, 强烈排斥 V1→C1.

### C 类 top routing (target = C)
[Q5-B] β = -0.405, 抑制 C2.

# 场景
【超市重构困境】你接到一家零售巨头委托, 在老城区核心地段打造"全国首个全数字化、无人化的高效零售标杆超市". 上层提出 3 个核心指令:
  1. 完全拆除人工收银台, 全部换成无人自助结账;
  2. 启用 AI 动态货架/界面推荐算法, 把高利润非必需品摆到最显眼位置, 把必需品藏到深处;
  3. 三个月内必须按数字化方案完整铺开.
注意: 老城区 60+ 岁的银发族占比超过 40%, 他们不熟悉数字设备.

# 任务
请说出你向管理层提出的核心设计底线 + 可妥协方案, 你绝不让步的是什么, 愿意妥协的是什么 (200-400 字).

## 输出约束 (seg3 — 核心 tradeoff, anchor + 底线段)
- dominant_factor: 单字母 P/K/V/C (禁 X→Y, 底线段不是推理链)
- state_sequence: 长度 1-3, anchor 段
- criterion: 必填非 null (≤ 25 字 verbatim 出现在 nl_answer 中, 是你底线短句的原文)
- nl_answer: 200-400 字, 必须含明确的 "不可妥协 X / 可妥协 Y" 表达
```

---

## A.5 各 Ablation 跟 v2_anchor 的差异

每个 ablation 只在一处改 v2_anchor.

### A.5.1 v2_no_caps (C2)

**改动**: 从 USER prompt 删整个 CAPS 四元组块 (β 速览 + 4 类 routing).

**长度影响**: -1150 字 (~29% 减).

**SYSTEM 影响**: 不变 (SEG3_RULE_V2 仍在, 但失去 β 数据无法执行).

### A.5.2 v2_no_seg3_rule (C3)

**改动**: 从 SYSTEM 删 `## seg3 dominant_factor 判断规则:` 块.

**长度影响**: -300 字.

**USER 影响**: 不变.

### A.5.3 v2_no_beta_summary (C4)

**改动**: 从 CAPS 四元组块删 `## 4 类 routing 强度速览` header (以及 "→ 正向激活最强" 一行). 保留 "各类详细 routing 描述".

**长度影响**: -250 字.

### A.5.4 v2_no_pkvc_def (C5)

**改动**: 从 SYSTEM 删 `# PKVC 4 类认知状态` 定义块.

**长度影响**: -355 字.

### A.5.5 v2_plus_peer (C6)

**改动**: 在 任务 + 输出约束 之间加入 peer block. Peer = top-3 设计师按 CAPS-JS divergence 相似度排序, 给出他们的 Q7 同 seg 回答.

插入文本:
```
# 跟你认知模式相似的设计师在本场景的回答 (供参考)

### Peer 1 (认知模式相似度 = 0.82)
{peer_1_q7_seg_nl}

### Peer 2 (认知模式相似度 = 0.79)
{peer_2_q7_seg_nl}

### Peer 3 (认知模式相似度 = 0.76)
{peer_3_q7_seg_nl}
```

**长度影响**: +1200 字 (~30% 增加).

### A.5.6 v2_hybrid (C7) — Per-Segment Dispatch (分段调度)

**改动**: 根据 segment 不同:
- **seg1 / seg2**: 用 v2_anchor 完整 prompt (不变)
- **seg3**: 用 v2_clean prompt:
  - SYSTEM: 删 SEG3_RULE_V2
  - CAPS 四元组块: 把 "4 类平衡 + β 速览" 换成 "top-5 by 绝对 |β| 强度" (不分类, 无 β 速览 header)

依据: chain segment 需要 explicit anchor push multi-state; anchor segment 让 model 按场景自然走 V default.

---

## A.6 模块累加阶梯 (B0_h → B3_h)

所有阶梯 baseline 用跟 v2_anchor 一致的 **SYSTEM** (PKVC def + 输出 spec + SEG3_RULE_V2). 它们只在 USER modules 上有差异:

| 编号 | USER modules | 总字符 |
|---|---|---|
| B0_h | 仅 scenario + task | ~1880 |
| B1_h | + ① pi | ~2000 |
| B2_h | + ② stance | ~2500 |
| B3_h | + ③ self | ~3100 |
| **v2_anchor** (B4_h) | + ⑤ CAPS | **~4000** |

---

## A.7 Per-Segment 输出 schemas

### A.7.1 seg1 schema (第一反应 anchor)

```json
{
  "type": "object",
  "properties": {
    "dominant_factor": {"type": "string", "pattern": "^[PKVC]$"},
    "state_sequence": {
      "type": "array", "items": {"enum": ["P","K","V","C"]},
      "minItems": 1, "maxItems": 3
    },
    "criterion": {"type": ["string", "null"], "maxLength": 25},
    "confidence": {"type": "string", "enum": ["high","medium","low"]},
    "reasoning_notes": {"type": "string", "maxLength": 100},
    "nl_answer": {"type": "string"}
  },
  "required": ["dominant_factor","state_sequence","criterion","confidence","reasoning_notes","nl_answer"],
  "additionalProperties": false
}
```

### A.7.2 seg2 schema (chain 推理)

跟 seg1 差异: `dominant_factor` 允许 `X→Y` 形式; `state_sequence` 强制 `minItems=2` (多 state 必须).

### A.7.3 seg3 schema (anchor + criterion)

跟 seg1 差异: `criterion` 是**必填非 null** (≤ 25 字 verbatim 出现在 nl_answer 中).

---

## A.8 STANCE_LABELS 全表 (30 entries)

(权威源代码: `scripts/run_b6_sample.py::STANCE_LABELS`.)

| 维度节点 | 5-6 分立场 | 1-2 分立场 |
|---|---|---|
| 数据 vs 直觉 | 重直觉立场 | 重数据立场 |
| 天赋信仰 | 天赋决定论 | 后天努力论 |
| AI 替代论 | AI 替代主义 | 岗位保护主义 |
| 技术支持 | 技术激进主义 | 技术审慎主义 |
| 复杂工具信仰 | 工具门槛论 | 工具无关论 |
| 混沌与灵感 | 包容失控派 | 验证至上派 |
| 黑盒透明度 | 结果至上派 | 过程透明派 |
| 系统微模块 | 解耦优先 | 整体感优先 |
| 完美主义/MVP | 完美主义 | 速度优先 |
| 创新代价 | 创新激进派 | 惯例守护派 |
| 审美反民粹 | 审美精英主义 | 数据民粹妥协 |
| 情感附加值 | 仪式感优先 | 效率优先 |
| 装饰合法性 | 装饰合法派 | 形式追随功能 |
| 时代 vs 永恒 | 时代潮流派 | 永恒经典派 |
| 本土文化妥协 | 本土深耕派 | 全球通用派 |
| 仿生与有机 | 有机自然派 | 几何工业派 |
| 独裁 vs 委员会 | 独裁愿景派 | 民主共创派 |
| 秩序与防错 | 防错约束派 | 探索开放派 |
| 创作者署名 | 作者性优先 | 商业去人格化 |
| 资产开源 | 开源道德派 | 竞争壁垒派 |
| 封闭生态竞争 | 封闭生态派 | 开放兼容派 |
| 风险规避 | 风险规避 | 风险偏好 |
| 团队依恋 | 独立打磨派 | 团队共创派 |
| 商业操纵伦理 | 商业操纵容忍 | 用户伦理底线 |
| 数据隐私交换 | 体验换隐私 | 隐私保护派 |
| 弱势群体优先 | 边缘群体优先 | 多数效率优先 |
| 股东服务优先 | 股东忠诚派 | 用户忠诚派 |
| 政治中立性 | 政治中立派 | 政治表达派 |
| 系统责任归属 | 系统归责派 | 用户归责派 |
| 救世主情结 | 救世主情结 | 工具中立派 |

---

## A.9 Q3-Q6 冲突轴标签

(权威源代码: `scripts/run_b6_sample.py::Q_CONFLICT_LABELS`.)

| Q | 冲突轴 |
|---|---|
| Q3 | 数据效用与美感形态冲突, 数据效用与设计师立场冲突 |
| Q4 | 商业利润与用户隐私冲突, 隐私交换与概念建构冲突 |
| Q5 | AI 工具与设计师自主性冲突, 人机协作分工冲突 |
| Q6 | 商业目标与社会责任冲突, 利益与责任感冲突 |

---

## A.10 跨模型实现说明

对 OpenAI 系模型 (`gpt-4o` 等), 我们用严格 JSON schema 模式:
```python
response_format = {
    "type": "json_schema",
    "json_schema": {"name": "...", "schema": <SCHEMA>, "strict": True}
}
```

对其他通过 yunwu.ai 聚合 API 访问的模型 (Claude, Gemini, DeepSeek, Qwen 等), strict JSON schema 不一定被支持. 我们 fallback 到 `json_object` 模式 + system prompt 中嵌入 schema 描述:
```python
response_format = {"type": "json_object"}
# system prompt 加上: "严格按以下 JSON Schema 输出: ```json <SCHEMA> ```"
```

非 OpenAI 模型设 `max_tokens = 1500` (vs OpenAI 900), 防止 JSON 截断 (schema 描述占 token).

JSON parse 失败重试 3 次.

---

## A.11 复现性 (Reproducibility)

所有 inference + ablation runs 用 GPT-4o (`gpt-4o`) 作主, 加 5 个跨模型 (Gemini 3.1 Flash Lite, Claude Haiku 4.5, DeepSeek V4 Flash, Qwen 3.6 35B-A3B, Qwen 3.6 27B), 命令一致:

```bash
python scripts/run_b6_sample.py \
  --variant {v2|v2_no_caps|v2_no_seg3_rule|v2_no_beta_summary|v2_no_pkvc_def|v2_hybrid} \
  [--with-peer] \
  [--modules pi,stance,self,caps] \
  --model <MODEL_ID> \
  --concurrency <N> \
  --output reports/datav3/<output_filename>.jsonl
```

PKVC 抽取 (v8.7 extractor on simulated NL):
```bash
python scripts/eval_baselines_pkvc.py \
  --input <inference.jsonl> --output <pkvc_eval.jsonl>
```

Bootstrap 评测 (11-metric framework, 1000 iterations):
```bash
python scripts/final_eval_with_bootstrap.py
```

所有 inference / evaluation / metric 计算代码见
[GitHub 仓库] (review 时匿名化).

---

## A.12 完整 metric 列表 (与 paper 主表对应)

| 维度                        | Metric                             | 计算说明                                                                                  |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| **路径准度 (Fidelity)**       | macro-TED ↓                        | 按 GT first-letter 分 P/K/V/C 4 组, 各组 mean edit distance, 再 macro 平均                    |
|                           | macro-LCS-F1 ↑                     | 同上, 用 LCS-F1                                                                          |
|                           | non-V recall ↑                     | 非 V GT first-letter designer 中, pred first-letter 正确的比例                               |
| **个性化 (Personalization)** | NL specificity ratio (3 segs)      | cos(pred_NL_i, GT_NL_i) / mean_{j≠i}(cos(pred_NL_i, GT_NL_j)) > 1 即 designer-specific |
|                           | Persona-NL alignment (seg1)        | cos(pred_NL, designer Q1-Q6 NL 中心)                                                    |
|                           | Trait-pred Spearman ρ (seg1)       | 相似 30 维 trait → 相似 dominant prediction                                                |
|                           | Criterion-persona alignment (seg3) | cos(criterion 嵌入, designer Q1-Q6 NL 中心)                                               |
| **抗坍缩 (Anti-collapse)**   | Chain diversity (seg2)             | 1 − modal_chain_freq, 反映 162 个 designer 输出多样性                                         |
|                           | Criterion uniqueness (seg3)        | 1 − mean pairwise cos of criterion embeddings, 跨 162 designer                         |

3 大维度 × 11 metric 一起捕捉 "路径是否准 / NL 是否 designer-specific / 输出是否 differentiated" 3 个独立面.

Statistical significance: 1000-bootstrap (designer-level resample), 95% CI percentile method, 不做 Bonferroni correction (3 个正交维度).
