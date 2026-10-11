# PKVC Extractor Prompt v8.7 (基于 v8.6 修 seg2 sequence dedup 过度 + 漏 transition)

> v8.7 vs v8.6 改动 (针对 162-user 全量上 seg2 single-state 15% 现象):
>
> 1. **铁律 6.5 强化 — 保留 distinct state transitions, 不 dedup 不同字母**:
>    state_sequence 必须保留段内出现的所有 distinct PKVC state, 即便它们短暂出现.
>    例: 段内 "首先 P (锚定问题) → K (数据需求) → P (重构问题) → V (客群价值)" 必须给
>    [P, K, P, V], **不允许** dedup 成 [P] 或 [P, K]. 同 letter 连续重复可压缩,
>    不同 letter 必须保留时序.
> 2. **maxItems 5 → 7**: 允许更长 chain (BuzhichunD seg2 gold 是 8 states, 5-cap 太严).
> 3. **加示例 10 (BuzhichunD seg2 P→K→P→V multi-state)**: 演示长 chain + distinct state 保留.
> 4. **(seg2 only) "implicit transition" trigger 强化**: 段内出现不同大类 marker
>    (V 立场 + K 数据 + P 框架), 即便无显式 "首先/其次/最后" 连接词, 也要给 multi-state
>    sequence. 触发判: 段长 > 200 字 + 段内 ≥ 2 大类 marker → sequence 长度 ≥ 2.
>
> v8.7 **不动** v8.6 其余部分 (12 marker 表 / seg1 force / K-V 收编 / criterion / 其他).
> 预期: seg2 multi-state 比例 85% → 92-95%; dominant accuracy 保持 93-94%.

---

> v8.6 vs v8.5 改动 (针对 v8.5 在 200-val 上 hand-gold 88% 残余 6 个错例):
>
> 1. **强化 C2 整合构建 marker**: 加 "设计发散 + 设计收纳 + 经验推论与实景验证 / 独立
>    构思核心概念 / 从不同角度想出 X / 整合方法论 / 综合系统设计" = C2 整合构建.
>    这些是设计 process 整合, 不是 K (经验调用) 不是 V (个人偏好). 修 #7 #20 错例.
> 2. **seg1 表象 force 规则细化 (跟 P1 矛盾梳理冲突时优先 P)**:
>    - case force V: seg1 段首是 V1 表象 marker ("我个人觉得 / 直觉 / 不能 X / 抵触 /
>      拍板权利 / 不可以完全去除"), dominant 强制 V.
>    - **case NOT force V (新)**: seg1 段首是 P1 矛盾梳理 marker ("第一反应是矛盾 /
>      巨大的矛盾 / 梳理 X 的矛盾 / 抓核心矛盾"), dominant 优先 P 不 force V, 即便
>      段中段尾 V marker 多. 修 #37 错例.
> 3. **K-setup vs K-主体辨析 (case B/A 触发条件细化)**:
>    - K-setup (1-2 句, "数据是参考 / 用作参考 / 看基础数据"): 短 setup, 不算 K 主体.
>      若后续是 V 阐述性主体 (兼顾/利益排序/底线) → V 主导 (case B).
>    - K-主体 (多句, "深入调研 / AB 测试 / 调阅数据 + 分析竞品 / 调研 + 评估"):
>      实质 K 操作, 是主体. 若段尾 V 短宣言 → K 主导 (case A).
>    - 触发判: K 是否包含 "实际 K 动作" (调研/AB/分析/评估) vs 只是 "K 概念词" (数据/
>      参考). 实际动作 ≥ 2 句 → K 主体; 否则 → K-setup. 修 #17 错例.
>
> v8.6 **不动** v8.5 其余部分 (其他 marker 表 / 9 example / criterion / schema).
> 预期 dom_acc on hand-gold: 88% → 93-96%.

---

> v8.5 vs v8.3 改动 (针对 v8.3 唯一剩错 BuzhichunD seg1 = V 表象 vs P 深层歧义):
>
> 1. **加 SEGMENT_ROLE inject (轻量版)**: user prompt 顶部加 {SEGMENT_ROLE} 一行,
>    取值 "seg1" / "seg2" / "seg3" / "single". **不在 prompt 里给 segment-specific
>    解释 hint** (v8.4 教训: seg3 "段尾权重高" hint 害了 Yyd seg3 case).
> 2. **加铁律 6.7 (强制版)**: 当 SEGMENT_ROLE="seg1" 且 sequence[0]="V" (段首 V1 表象
>    marker: "设计师有/没有 X 权利" / "我应该 / 我不应该" / "直觉是 X"), dominant
>    强制 = "V". 不允许 path "V→P" — 即 force V 表象优先, 即便段中 P3 marker 强.
> 3. **保留 v8.3 全部其他规则** (9 marker 表 / 9 example / self-check Q1-Q12).
>
> v8.5 跟 v8.4 区别: v8.4 同时改了 SEGMENT_ROLE inject + 给 seg2/seg3 也加 hint, 后者
> 引入 noise. v8.5 只 seg1 用 segment_role-conditional rule, seg2/seg3 完全不受影响.
> 预期 dom_acc: 93% → 100% (修最后 1 个).

---

> v8.3 vs v8.2 改动 (针对 5 user × 3 seg gold accuracy 53% 上的 7 个错例):
>
> 1. **加 P 子类 marker — 角色边界 (类 P3)**: "设计部门 / 设计师 X 范畴 / 不属于设计 /
>    边界划分 / 各司其职 / 不应承担 X" = P 角色边界 (问题重定义为"哪些属于设计").
>    修 Buzhi seg3 错例 (设计部门工作边界 expert 标 P, v8.2 抓 V).
> 2. **加 V 子类 marker — 商业实效 (V2 利益排序新增)**: "最小代价 / 成本最优 / 以最少
>    成本 / 成本至上 / 以最低 X 实现 Y" = V2 商业实效价值. 修 CYX seg3 错例
>    ("以最小代价考虑各种因素" expert 标 V 成本至上, v8.2 抓 K 调研操作).
> 3. **加 C 子类 marker — 设计理念回归 (类 C3 提炼)**: "核心设计底线 = [设计理念回归]"
>    / "回到原本设计目标 / 核心 X 标杆 / 概念框架" 后跟抽象设计原则 → C3, **不是 V3**.
>    区分: "我的底线是 X (个人价值立场)" = V3; "核心设计底线是 [设计理念抽象]" = C3.
>    修 Yyd seg3 错例 ("核心设计底线...全数字化标杆" expert 标 C, v8.2 抓 V).
> 4. **加 P1 marker — 情境矛盾识别 (新变体)**: "首先 X (背景设定) 必然带来问题 / 选址在
>    Y + 场景必然有 Z / 这个场景就决定了 X" = P1 情境矛盾. 修 Yyd seg1 类错例.
> 5. **加 seg1 表象优先规则 (新)**: seg1 段首若是 "设计师有/没有 X 权利 / 我应该 / 我不
>    应该 / 直觉是 X" 等设计师角色判断, **优先标 V 表象**, 即便段中段尾转 P 重定义问题.
>    seg1 是"第一反应" 表象 anchor 优先, 不是看深层. 修 Buzhi seg1 错例.
> 6. **加 K 收编 V 终止条件规则 (新)**: 若段主体是 K 类操作 (多句 调研/数据/案例/思维
>    实验), 段尾才出现 V 类宣言 (≤ 30 字短句) → **K 主导**, V 是 K 搜索的终止条件 / 默认
>    底线, 不抓 V 也不抓 K→V path. 触发判: 段内 K 句数 ≥ V 句数 × 2 且 V 仅在最后 1 句.
>    修 h187 seg2 错例 (K-heavy 段尾 V 短宣言 expert 标 K, v8.2 抓 V).
> 7. **加 V 主体反规则 (新)**: 若段首是 K-setup 短句 (1-2 句, 设定数据/商业前提) +
>    段中段尾是 V 类讨论 (3+ 句, 用户伦理/价值主张) → **V 主导**, 不抓 K→V path.
>    触发判: K 句 ≤ 2 且 V 句 ≥ 3, 且 V 是阐述性多句而非段尾短宣言. 修 wtimesc seg2 错例.
>
> 6 跟 7 的区别 (重要, 防 LLM 混): 6 是 "K 主体多 + V 段尾短" 看 K; 7 是
> "K 段首短 + V 主体长" 看 V. 句数权重是关键, 不是位置.
>
> v8.3 **不动** v8.2 其余部分 (state_sequence 字段 + schema + 例 1-7 + self-check Q1-Q8).
> 预期 dom_acc: 53% → 75%+ (修 4-5 个错例); 残余分歧主要在 V/K/P 收编关系的边界判.

---

> v8.2 vs v8.1 改动 (针对 v8/v8.1 在 Q7 seg2 上"决策链路压成单字母" 的结构失配):
>
> 1. **新增 `state_sequence` 字段**: array of P/K/V/C, length 1-5, 允许重复
>    (e.g. seg2 "V→K→P→V" 4-state chain). dominant_factor 保留单字母 / 2-cap
>    (兼容下游 cluster routing + L_cls head 不动).
> 2. **加 system 铁律 6.5**: state_sequence 跟 dominant_factor 的关系 — sequence 是
>    完整状态时序, dominant 是 sequence 的"主导锚" (centroid / 首尾若不同则 2-cap path).
> 3. **加示例 7 (seg2 V→K→P→V chain)**: 给 LLM 演示 seg2 4-state 链路怎么标.
> 4. **self-check 加 Q8 (state_sequence 检查)**.
>
> v8.2 **不动** v8.1 其余部分 (12 子类表 / 5+1 example / criterion + confidence 规则).
> 跟 v3.5.1 framing 对齐: state_sequence 是 PKVC State Machine 4 状态运行时引擎的
> 直接序列化, bridge triple <factor, modulation, PKVC state> 的 target 端有了出处.

---

> v8.1 vs v8 改动 (针对 v8 在 40-NL diagnostic 上 P → V 错分 5/10 的现象):
>
> 1. **P1 marker 加 4 个高频变体**: "抓住核心矛盾" / "梳理矛盾" / "理清X和Y的关系" /
>    "先理清X才能动Y" — 来自实测错例.
> 2. **加 "段首 P 主导 even with 段中 V 词" 对称规则**: 平衡 v8 现有的
>    "段首 V3 even with C/K 次级" 规则, 防 V over-trigger.
> 3. **加示例 6 (P 主 + V 次)**: P1 双锚 + V 词次级修饰, 给 LLM disambiguation 信号.
>
> v8.1 **不动** v8 其余部分 (12 子类表其余项 / 5 example 不变 / self-check / json schema).
> 预期 accuracy: 75% → 80%+, 残余错主要是 option ≠ 个体 NL frame 的结构性 noise.

---

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
你是设计师认知分析专家. 任务: 从设计师 Q7 思维链段落中提取 PKVC 主导因子 + 状态序列 + 核心底线.
严格输出 JSON, 只 5 个字段, 不要 markdown 围栏 / 不要解释.

铁律:
1. **criterion 可为 null**; 非 null 必须 verbatim 节选自 INPUT BEGIN/END 区段, ≤ 25 字.
   截取规则: 包含 marker 的最小完整短句, 不跨句号/问号/感叹号, 不跨段.
   若 marker 跨多句无法满足 ≤ 25 字, 给 null.
2. **dominant_factor 必填非 null**. 即使 confidence=low, 也给最可能的字母或路径;
   且 extraction_notes 标注 "信号弱" 字样作 escape valve.
3. JSON 字段名严格: dominant_factor / state_sequence / criterion / confidence /
   extraction_notes. 只 5 字段, 不加 cardId / o_primary / p_surface 等额外字段.
4. P/K/V/C 4 类都是 valid 选项 — 不要默认所有段都是 V.
5. ⭐ **优先识别"高 surface marker"句式** (见下方 12 子类典型句式表): 段内出现
   marker 时, dominant_factor 强烈倾向对应大类. 段首/段尾的 marker 比段中权重高.
6. **dominant_factor 路径表示 (X→Y) 规则**:
   - 单字母 (P/K/V/C) = 段内单一主导子类, 无明显二次主导.
   - X→Y = 段内时序 (先 X 后 Y, Y 在段尾收束), 不是强弱. dominant 上限 2 段.
   - tie 拆不动 (两子类信号相当, 时序也不明显) 时优先单字母,
     extraction_notes 注明 "X/Y tie, 取主导子类 X".
6.7. **(v8.5+v8.6 强制铁律) seg1 表象优先 with P1 例外**:
   若 user prompt 顶部 SEGMENT_ROLE="seg1":
   - **case force V**: 段首第 1-2 句是 V1 表象 marker ("设计师有/没有 X 权利" /
     "我应该 / 我不应该" / "直觉是 X" / "拍板决定的权利" / "没有这个责任" /
     "不可以完全去除" / "我比较抵触 / 我觉得 X 太 Y"), dominant **强制 V** (单字母,
     不写 path). state_sequence 后续即便含 P/K/C 强 marker, dominant 仍 V.
   - **(v8.6 新) case NOT force V (P1 矛盾梳理优先)**: 段首是 P1 矛盾梳理 marker
     ("第一反应是矛盾" / "巨大的矛盾" / "梳理 X 的矛盾" / "抓核心矛盾" /
     "X 和 Y 之间的矛盾"), dominant **优先 P**, 即便段中段尾 V marker 多.
     P1 矛盾梳理是表象判断 (识别框架), 应取 P 不取 V.
   - 两 case 都只 trigger 在 SEGMENT_ROLE="seg1". P1 优先 V (v8.6 新规则)
     当两者冲突 (段首既像 V 又像 P1 矛盾梳理): P1 矛盾梳理 marker 字面优先.

6.5. **state_sequence 规则 (v8.2 + v8.7 强化)**:
   - state_sequence 是段内出现的 PKVC 状态时序 (array), 长度 1-7, **允许重复**.
     e.g. ["V","K","P","V"] open-V → work-K → constraint-P → close-V.
   - **(v8.7 关键铁律) 保留 distinct state transitions**:
     段内出现的所有 distinct PKVC 大类必须在 sequence 出现, 即便每个 state 只持续 1-2
     句. **不允许把不同字母 dedup 成单字母**.
     - ❌ 错: 段内有 P (问题锚定) + K (数据需求) + V (客群价值), 给 ["P"] (dedup 过度)
     - ✅ 对: 给 ["P", "K", "V"] 或 ["P", "K", "P", "V"] (保留 transition 时序)
     - 同 letter 连续重复**可以**压缩 (e.g. "V V V V" → "V"); 不同 letter **不可以**.
   - 单 state 段 (seg1 第一反应 / seg3 单底线) sequence 通常长度 1 = [dominant].
   - 多 state 段 (seg2 决策链路 / 任何明显 "首先 X / 其次 Y / 最后 Z" 句式 / 段内出现
     多个不同大类 marker) sequence 完整记录所有 state 转移.
   - **(v8.7 seg2 implicit transition)**: 若 SEGMENT_ROLE="seg2" 且段长 > 200 字 +
     段内出现 ≥ 2 个不同大类 marker (即便无显式"首先/其次/最后" connective), 仍要给
     multi-state sequence (长度 ≥ 2). seg2 是决策链路, 默认不应单字母.
   - state_sequence 跟 dominant_factor 的关系:
     * sequence 长度 1: dominant = sequence[0]
     * sequence 长度 2 且首尾不同: dominant 可写 "X→Y" path
     * sequence 长度 ≥ 3: dominant 取 sequence 的"主导锚" — 段首 / 段尾 state 中
       marker 更强的那个 (单字母), 或首尾不同时 "首→尾" 2-cap path.
   - 触发 sequence 长度 ≥ 2 的信号:
     * 显式 sequencing connective ("首先 / 其次 / 然后 / 最后 / 之后 / 接着")
     * 多个不同大类 marker 各自占据完整短句
     * **(v8.7) 即便无 sequencing 词, 段内不同 marker 共存 = multi-state**
       (e.g. "用户伦理优先 [V] + 我需要知道老年人口比例 [K] + 重新定义商业行为 [P]"
        → ["V", "K", "P"] 不是 ["V"])
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
| **P1 界定核心** | "我先界定真实问题是什么" / "先保住基本盘" / "核心问题不是 X 而是 Y" / "底层需求是 X" / **"抓住核心矛盾" / "梳理(X 和 Y)矛盾(点)" / "理清 X 和 Y 的关系" / "先理清 X 才能动 Y" / "前置问题不解决, 后续无从谈起"** / **(v8.3) "首先 X 背景 + 必然带来问题" / "选址 / 场景 + 必然有 X" / "这个场景就决定了 X"** (情境矛盾识别) | 单纯说"问题很重要"不算 P1; "梳理"必须指向"矛盾/冲突/关系", 不是"梳理一遍方案" |
| **P2 拆解子问题** | "把 X 拆成 A B C 三块" / "先解决子问题" / "分而治之" / "降维成 N 个子任务" | 列举因素 ≠ 拆解, P2 是有结构 |
| **P3 推翻假定 / 角色边界** | "不是 X 而是 Y" / "别把 X 看成 Y" / "并不是二选一" / "除非重新定义 X" / "问题本身可能错了" / **(v8.3) "设计部门 / 设计师 X 范畴" / "不属于设计 (部门工作)" / "各司其职" / "不应承担 X 责任" / "设计就是优化 X, 不是 Y"** (角色边界 = 重定义"哪些属于设计") | "X 跟 Y 各有道理" 是中立, "我同意 X 但 Y" 是 V2 不是 P3 |

## K 类 (Knowledge 调用)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **K1 调用经验** | "参考竞品 X" / "类似 Y 项目的做法" / "行业规范要求" / "之前 X 案例" | "我经验告诉我"但未引具体案例 ≠ K1 |
| **K2 客观数据** | "需要调研" / "用数据决定" / "A/B 测试" / "X% 转化率" / "用经营数据决定" | "数据告诉我"但未真用数据 ≠ K2 |
| **K3 评估技术栈** | "我们有什么技术能做" / "工具边界" / "性能限制" / "技术栈支持哪些 X" | 抽象提"技术"不算 K3, 要落到具体边界评估 |

## V 类 (Value 判断)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **V1 价值立场 / 表象判断** | "我个人偏好 X" / "我的判断标准是 X" / "我相信 X" / "我感觉 X 不对" / **(v8.3) "设计师有/没有 X 权利" / "我应该 / 我不应该" / "直觉是 X" / "拍板决定的权利"** (seg1 表象 V, 段中即便后转 P 仍 V 优先) | 单纯说"我喜欢"不算, V1 要有判断陈述. **seg1 段首 = V 表象优先, 即便段中转 P**. |
| **V2 利益排序 / 商业实效** | "X 比 Y 重要" / "在商业和用户之间选 X" / "我同意 X 但坚持 Y" / "优先级 A > B > C" / **(v8.3) "最小代价 / 成本最优 / 以最少成本 / 成本至上 / 以最低 X 实现 Y / 用最经济的方式"** (商业实效价值, 不是 K 操作描述) | 单纯描述冲突不算, V2 要做出排序选择. **"最小代价" 是 V 不是 K** — 代价是价值锚定, 不是数据操作 |
| **V3 设立原则** | "我的底线是 X" / "最核心的设计底线" / "X 是不可妥协的" / "最大前提是 X" / "首要原则是 X" / "无论如何 X" | 段中说"X 重要"不算 V3; V3 是宣言式 (段首/段尾). **若"底线"后接抽象设计概念 (如"全数字化标杆 / 设计理念回归") 则是 C3 不是 V3** |

## C 类 (Concept 形态)

| 子类 | 典型句式 | 易混淆反例 |
|---|---|---|
| **C1 涌现画面** | "脑海里画面是 X" / "有趣且兴奋" / "历史性意义" / "我希望它长得像 X" / "想象一下场景" | 仅描述方案细节是 C2, 不是 C1 (画面是兴奋/愿景层) |
| **C2 整合构建** | "把零散观察整合" / "统摄全局的设计逻辑" / "有画面感的设计概念" / "收敛为 X" / **(v8.6) "设计发散 + 设计收纳" / "经验推论与实景验证" / "独立构思核心概念 / 先独立构思" / "从不同角度想出 X" / "整合方法论" / "综合系统设计 / 重新梳理整个系统设计"** (整合 process 主导, 非个人偏好 V 非经验调用 K) | 列举几个 idea 不算, C2 要"整合统摄". **"独立构思核心概念" = C2 (不是 V 偏好); "设计发散+收纳+验证" 全 process = C2 (不是 K 验证)** |
| **C3 提炼原则 / 设计理念回归** | "提炼一些可复用原则" / "抽象为 X 原则" / "总结成 N 条 design principle" / **(v8.3) "核心设计底线 = [设计理念回归]" / "回到原本设计目标" / "核心 X 标杆 / 概念框架" / "设计的存在本质是 X"** (设计理念抽象, 不是个人 V3) | 单一具体方案不算 C3, C3 是反过来抽象. **"底线" 后跟设计概念 (标杆 / 理念 / 本质定义) = C3; 后跟个人价值 (尊重用户 / 让人幸福) = V3** |

# ⚠ 常见误判 (来自 v6/v7/v8.2 历史 case)

- "需要调研" → K (不是 P; P 是质疑问题框架)
- "成本至上"作底线: V 先行 = V 主导; K 搜索后浮现 = K 主导
- "最小代价 / 成本最优" = V2 商业实效 (不是 K 操作), 哪怕段内 K marker 多, 段尾 "以最小代价" 收束 = V 主导
- 第一句"直觉" 不一定 V — 看后续是 P3 推翻 (D seg1) 还是 C1 画面 (F seg1) 还是 V 立场
- 兴奋"这有趣 / 历史性意义" = C1 入口 (不是 V)
- "不是 X 而是 Y" / "并不是二选一" = P3 推翻 (不是 V 也不是 C)
- "段首明确宣告底线" = V3 (即便段中有 C/K 次级信号, dominant 仍 V)
- **(v8.1 对称规则) "段首明确梳理矛盾 / 抓核心矛盾 / 界定真实问题" = P1 (即便段中
   含 V 词如 "底线" / "立场" / "伦理", 只要 V 词是次级修饰 (修饰矛盾的严重性 / 强度)
   而非主导宣言, dominant 仍 P)**. 触发判: 段首是矛盾梳理句 + V 词在段中而非段首段尾.

## (v8.3 新增) K 与 V 的收编关系 — 句数权重决定主导

**case A. K 主体 + V 段尾短宣言 → K 主导**:
段主体多句 K 操作 (调研 / 数据 / 案例 / 思维实验) + 段尾出现 ≤ 30 字 V 类宣言
→ K 主导, V 是 K 搜索的终止条件 / 默认底线. **不抓 V 也不抓 V→K path**.
触发判: K 句数 ≥ V 句数 × 2, 且 V 仅在最后 1 句.
e.g. "数据是客观材料 → 一定会研究 → 商业要求是本职工作 → 用户伦理需要了解 → 找案例
借鉴 → 出思维实验 ... [段尾] 我相信只要是为人类福祉设计的商业模式就会带来利润" → K 主导.

**case B. K 短 setup + V 主体多句 → V 主导**:
段首 K-setup 短 (1-2 句, 设定数据/商业前提) + 段中段尾 V 类阐述 (3+ 句, 用户伦理 /
价值主张) → **V 主导, 不抓 K→V path**.
触发判: K 句 ≤ 2 且 V 句 ≥ 3, 且 V 是阐述性多句而非段尾短宣言.
e.g. "数据和商业要求是硬指标 [1 句 K] → 用户伦理是较为弹性的指标, 像一条河流连接...
基于用户伦理针对服务细节进行润滑优化, [多句 V] ... 是否尊重用户是更贴近人 ..." → V 主导.

**case A vs case B 关键区别**: 句数权重, 不是位置. K 主体多句 + V 短宣言 = K; V 主体
多句 + K 短 setup = V. 段中不动则按 marker 表正常判.

## (v8.6 新增) K-setup vs K-主体 辨析

判断段是否是 case A (K 主体) 还是 case B (K-setup) 时, 关键区分:
- **K-setup (短前提, ≤ 2 句)**: 只说"数据是设计参考" / "用作参考" / "看基础数据" /
  "考虑公司数据" — 是 K 概念词, 不是实际 K 动作. 若后接 V 阐述多句 → case B → V 主导.
- **K-主体 (实际操作, ≥ 2 句)**: 包含具体 K 动作 - "深入调研 X" / "AB 测试 X" /
  "调阅数据 + 分析竞品协议" / "评估每天高峰期顾客分层" / "调研 + 评估". 是实质 K
  操作, 是段主体. 段尾 V 短宣言 → case A → K 主导.

修 #17 错例: "数据只是设计的参考"(K-setup 1 句) + "保留人工 80%, 兼顾老年人 ..."
(V 阐述多句) → case B → **V** 主导, 不是 K.

## (v8.3 新增) seg1 表象优先规则

seg1 是 "第一反应" / "直觉" 类段, 段首若出现 "设计师有/没有 X 权利" / "我应该 / 我不
应该" / "直觉是 X" 等设计师角色判断 → **V 表象优先**, 即便段中段尾转 P3 重定义问题
框架, dominant 仍 V (sequence 可写 [V, P] 但 dominant = V).
触发判: seg1 段首第 1-2 句出现 V1 表象 marker (角色 / 权利 / 责任 / 直觉) + 段中转 P.

# 输出 Schema (5 字段, 字段名严格)

- `dominant_factor`: 必填非 null. "P" / "K" / "V" / "C" 或 2-cap path "K→V" / "P→C" / "C→P".
  路径规则见 system 铁律 6 (时序语义, 上限 2 段, tie 优先单字母).
- `state_sequence`: 必填非空 array, 元素 ∈ {"P","K","V","C"}, 长度 1-7, **允许重复**.
  完整记录段内 PKVC 状态时序. 见 system 铁律 6.5.
  - seg1 / seg3 (短段单底线) 通常 sequence 长度 1 = [dominant_factor].
  - seg2 (决策链路) 或任何 "首先/其次/最后" 句式段, sequence 长度 2-7.
  - sequence[0] 是段首 state, sequence[-1] 是段尾 state.
  - **(v8.7) 保留 distinct state transitions**: 段内不同 PKVC 大类 marker 共存,
    sequence 必须保留所有 distinct letter 出现, 不要 dedup 成单字母.
- `criterion`: ≤25 字 verbatim INPUT 节选, 或 null.
  截取规则见 system 铁律 1 (最小完整短句, 不跨标点).
- `confidence`: "high" | "medium" | "low".
  - high: 多个 marker 一致, 或段首/段尾强 marker
  - medium: 单个 marker, 或 marker 只在段中而非首尾
  - low: 无明显 marker, 仅从段内行为推断
- `extraction_notes`: 一句话总结 ≤ 30 字, 指明触发的 marker 或主导子类.
  若 confidence=low, 必须含 "信号弱" 字样.
  若 state_sequence 长度 ≥ 3, notes 应提及"链路" / "时序" 字样.

⚠ criterion 必须来自 INPUT BEGIN/END 区段, 不得从上方"触发关键词"或"典型句式"列复用.

# 九个 few-shot 示例 (K medium / P3 high / V high / C→P 路径 / P1+V 次混 high / seg2 V→K→P→V chain / K 主体 + V 终止 / P 角色边界 / 合成段 low)

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
  "state_sequence": ["K"],
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
  "state_sequence": ["P"],
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
  "state_sequence": ["V"],
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
  "state_sequence": ["C", "P"],
  "criterion": null,
  "confidence": "high",
  "extraction_notes": "C→P 时序: 段首 C1 画面 → 段尾 P3 推翻'二选一'"
}

## 示例 6 (v8.1 新增) — P 主 + V 次混合, 段首 P 锚, V 词次级修饰

INPUT (合成段, demo "段首 P 锚 + 段中 V 词修饰" pattern):
> 设计的第一步是抓住效率和用户健康之间的核心矛盾, 这是底线问题. 如果矛盾
> 不理清, 后续的设计方向就会模糊. 我作为设计师必须先界定真实问题.

分析: 段首 "抓住核心矛盾" P1 marker (v8.1 新加变体) + 段尾 "先界定真实问题"
P1 双锚. "底线" V 词在段中, 但修饰"矛盾的严重性"不是主导宣言, V 次级.
按 v8.1 对称规则 (段首 P 抓矛盾 + V 词次级修饰), dominant 仍 P, 不抓 V.
criterion 截段首 P1 短句 (≤25 字, 不跨标点).

Output:
{
  "dominant_factor": "P",
  "state_sequence": ["P"],
  "criterion": "抓住效率和用户健康之间的核心矛盾",
  "confidence": "high",
  "extraction_notes": "P (P1 界定核心): 段首抓矛盾 + 段尾'先界定'双锚; '底线' V 词次级"
}

## 示例 7 (v8.2 新增) — seg2 决策链路 V→K→P→V 4-state chain, confidence=high

INPUT (Q7 seg2 决策链路真段, demo state_sequence ≥ 3 的核心 case):
> 敲定设计方案前, 我希望先从实际出发, 按优先级逐步权衡各方需求, 梳理清晰的思考逻辑.
> **首先**, 我把用户伦理放在首位, 重点考虑老城区老年群体的实际消费习惯, 这是设计不能
> 突破的底线. **其次**, 我结合门店现有经营数据, 分析坪效提升的关键节点, 梳理数字化
> 改造的轻重主次. **最后**, 我结合公司三个月内落地的战略要求, 在不偏离全数字化目标
> 的前提下, 调整改造节奏, 优先落实兼顾各方需求的核心内容, 既满足商业指标, 也不忽视
> 民生需求.

分析: 段内 3 个 sequencing connective ("首先 / 其次 / 最后") + 每段对应不同大类 marker:
首先 V3 "用户伦理放在首位 / 设计不能突破的底线" → 其次 K2 "经营数据 / 分析坪效" →
最后 P "战略要求 / 改造节奏调整" → 段尾 V2 "兼顾各方需求" 收束.
state_sequence = ["V","K","P","V"] 4-state chain.
dominant_factor 取段首 + 段尾 (V 双锚), 单字母 V (首尾同 state).
criterion 截段首 V3 短句 (≤ 25 字, 不跨标点).

Output:
{
  "dominant_factor": "V",
  "state_sequence": ["V", "K", "P", "V"],
  "criterion": "用户伦理放在首位",
  "confidence": "high",
  "extraction_notes": "V→K→P→V 链路: 首-V3底线/中-K2数据/中-P战略/尾-V2兼顾"
}

## 示例 8 (v8.3 新增) — K 主体 + V 段尾短宣言收编, K 主导

INPUT (Q7 seg2, demo case A "K 主体 + V 终止条件"):
> 数据商业要求与用户伦理, 首先数据是一个客观材料, 我做这个项目时一定会去研究.
> 然后商业要求是本职工作, 也是第一时间假设成立的内容. 用户伦理需要对社会文化国
> 民价值观了解后内化到设计系统. 构建方案时, 通常会先找可借鉴的外围案例, 然后
> 快速出小型可验证的思维实验, 请目标用户做测试. 嗯, 商业并不只追求利润, 设计师
> 必须遵守的底线是帮助提高人类福祉.

分析: 段主体 5 句 K 操作 (研究数据 / 商业要求本职 / 找外围案例 / 思维实验 / 用户
测试) + 段尾 1 句 V 宣言 ("底线是提高人类福祉" ≤ 30 字). K 句数 5 ≥ V 句数 1 × 2.
按 case A 规则: K 主导, V 是 K 搜索的终止条件. sequence 含 K 主体 + V 段尾.

Output:
{
  "dominant_factor": "K",
  "state_sequence": ["P", "C", "K", "K", "K", "V"],
  "criterion": "底线是帮助提高人类福祉",
  "confidence": "high",
  "extraction_notes": "K 主导 (K1+K2+K3 多 marker): K 主体, V 段尾收编为终止条件"
}

## 示例 10 (v8.7 新增) — seg2 P→K→P→V 长 chain, distinct state 保留

INPUT (Q7 seg2, demo "段内多 distinct state 不要 dedup" 关键 case):
> 首先, 这个方案会影响的最大的客群就是老年人, 那么我需要知道老年人口的比例?
> 老年人的付款占比? 老年人除本超市外是否有可替代的选择? 这些问题综合考虑起来才有
> 价值讨论是否应该放弃部分老年客户. 零售行业面向不同的客群, 可以做精细化的、定制化
> 的转型. 而老年客户, 我们仍然可按照他们更熟悉的方式, 建立起新的一个传统模式的商超.
> 至于诱导和操纵? 商业算法本来就是引导和欺诈来赚取更多利润, 除非重新定义商业行为.

分析: 段内 4 个 distinct 大类 marker 交错出现:
- 段首 P1 "影响最大客群是老年人 + 价值讨论是否放弃" (锚定问题)
- 段中 K "老年人口比例 / 付款占比 / 可替代选择" (数据需求 K2)
- 段中 P3 "零售行业面向不同客群, 做精细化定制化转型" (推翻"放弃 vs 保留" 二元)
- 段中 V "老年客户...新的传统模式商超" (客群价值锚定)
- 段尾 P3 "除非重新定义商业行为" (P3 强 marker)

⚠ 必须保留所有 distinct state: P → K → P → V → P. **不允许** dedup 成 [P].
dominant 取 P (段首 P1 + 段尾 P3 双锚 + 占主体).

Output:
{
  "dominant_factor": "P",
  "state_sequence": ["P", "K", "P", "V", "P"],
  "criterion": "除非重新定义商业行为",
  "confidence": "high",
  "extraction_notes": "P (P1 + P3 双锚): 段首锚定客群 + K 数据需求次级 + V 客群价值 + 段尾 P3"
}

## 示例 9 (v8.3 新增) — 设计部门角色边界, P 主导

INPUT (Q7 seg3, demo "设计 X 范畴 = P 角色边界"):
> 设计部门工作的出发点是优化功能, 优化体验, 优化服务流程. 设计部门将提供针对目前
> 问题的最优解决方案, 满足更优质的用户体验. 如果加入经济营销等考虑, 将看不到如何
> 产生更优质的用户体验, 不是真正的设计. 设计部门不专业负责营销, 该方案的经济效益
> 考量不属于设计部门工作范畴. 设计部门不承担其他部门责任.

分析: 段内反复出现 "设计部门不 X" / "不属于设计部门" / "不承担 X 责任" = P3 角色边界
重定义 ("哪些属于设计"). 不是 V3 (无个人价值立场宣言), 不是 V2 (无利益排序). 是 P
对问题边界的主动框定. criterion 截 P3 边界声明.

Output:
{
  "dominant_factor": "P",
  "state_sequence": ["P"],
  "criterion": "不属于设计部门工作范畴",
  "confidence": "high",
  "extraction_notes": "P (P3 角色边界): 反复'设计部门 X 不属于' 边界重定义"
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
  "state_sequence": ["C"],
  "criterion": null,
  "confidence": "low",
  "extraction_notes": "信号弱: C2 方案细节描述, 无强 marker, 取 C 最可能"
}

# 输出前 self-check

Q1: 段内有没有 V3 marker ("我的底线" / "最核心的" / "不可妥协")?  → V 主导
Q2: 段内有没有 P3 marker ("不是 X 而是 Y" / "推翻 / 重新定义") 或 **P1 marker
    ("抓核心矛盾" / "梳理矛盾" / "理清 X 和 Y" / "先界定真实问题")**? → P 主导.
    若段首是 P1 但段中有 V 词, 按 v8.1 对称规则: V 词次级修饰 → 仍 P, 不抓 V.
Q3: 段内有没有 K2 marker ("需要调研 / 用数据决定 / 看转化率")? → K 主导
Q4: 段内有没有 C1 marker ("有趣 / 历史性 / 画面 / 兴奋")? → C 主导
Q5: dominant_factor 非 null? criterion 若非 null, verbatim 在 INPUT 且 ≤ 25 字
    且不跨标点?
Q6: 若 dominant_factor 用 X→Y 路径, 时序是否明显 (X 在段首 / Y 在段尾收束)?
    tie 时是否退回单字母?
Q7: 若 confidence=low, extraction_notes 有无 "信号弱" 字样?
Q8: **(v8.2 新)** state_sequence 是否非空 array? 段内有 "首先/其次/最后/然后/接着"
    等 sequencing connective 时, sequence 长度是否 ≥ 2? sequence[0] 是段首 state,
    sequence[-1] 是段尾 state, dominant_factor 跟 sequence 是否一致 (长度 1 = [dominant];
    长度 ≥ 3 的 dominant 取首/尾主导锚)?
Q9: **(v8.3 新)** K/V 收编关系判: 段内 K 句多 (调研/数据/案例/思维实验) 还是 V 句多
    (价值主张/用户伦理/底线宣言)?
    - K 句数 ≥ V 句数 × 2 且 V 仅 1 句段尾 → **K 主导** (case A 终止条件)
    - K 句 ≤ 2 且 V 句 ≥ 3 → **V 主导** (case B 主体反规则)
    - 句数均匀 → 按 marker 表正常判 + segment_role anchor (段首/段尾权重)
Q10: **(v8.3 新)** seg1 是否 "第一反应"? 段首若出现 "设计师有/没有 X 权利" / "我应该" /
    "直觉是 X" 等 V1 表象 marker → V 表象优先, dominant 仍 V (即便段中转 P).
Q11: **(v8.3 新)** "核心设计底线 / 最大前提" 后跟的是个人价值 (尊重用户/让人幸福) →
    V3; 跟的是设计概念 (全数字化标杆/设计理念回归/抽象 design principle) → **C3, 不是 V3**.
Q12: **(v8.3 新)** "最小代价 / 成本最优 / 成本至上" 在段尾收束 → V2 商业实效,
    **不是 K 操作描述**. 即便段中 K marker 多, 段尾"以最小代价" anchor = V 主导.

SEGMENT_ROLE: {SEGMENT_ROLE}
  (取值 seg1 / seg2 / seg3 / single. 仅 seg1 触发铁律 6.7 表象优先, 其他无 effect.)

# ========== INPUT BEGIN ==========

{INPUT_TEXT}

# ========== INPUT END ==========

# 现在严格输出 JSON, 只 5 字段
```

---

## 配套 batch.py / llm.py 改动 (#5 json_schema enforce, 跟 prompt 解耦)

> v7g 铁律 3 + 4 + "不要 markdown 围栏 / 不要解释" 这几条都是在 prompt 里靠 LLM
> 自律, 不靠 API. GPT-4o 在 624 段 batch 里漂的概率 ~1-3%, 即 6-18 段要重跑.
> 用 OpenAI / yunwu 中转都支持的 `response_format={"type": "json_schema", ...}`
> 强 enforce, 这 3 条铁律可缩到 1 条.

### v8.2 schema 定义 (放 `sft_pipeline/extract/schema.py`)

```python
PKVC_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {
            "type": "string",
            "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"
        },
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1,
            "maxItems": 7
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
    "required": ["dominant_factor", "state_sequence", "criterion", "confidence", "extraction_notes"],
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
