---
date: 2026-05-22
status: framing-correction-note
purpose: 防止下次 session 又把 CAPS 跟 30 维量表混淆 (我 2026-05-22 这次 session 内犯了该错, 当日修正)
references:
  - /Users/wy/Desktop/RolePlay_Designer/30维-CAPS-pkvc 撰写框架.md  (权威 framing 文档)
  - docs/briefs/v3.5.1_20260522.md  (已 in-place 修正到 v3.5.1)
  - introduction_zh.md                  (论文 intro, framing 跟撰写框架一致)
---

# Framing 修正笔记 — CAPS / PKVC / Trait-State Modulation

## TL;DR

我之前 (2026-05-22 上午写 v3.5 brief / mapping_miner.py 时) 把 **CAPS 等同于 30 维量表**,
把 **PKVC 当成"输出行为层"**, 写了一个 (trait_dim, polarity, SADR_node) 的静态 mapping schema.
**这三件事都错了**. 当日修正为 v3.5.1, 这份笔记记录错误溯源, 防下次再漂.

---

## 正确的 3 层架构 (摘自撰写框架.md 行 60-67 + introduction_zh.md 行 200-202)

```
┌────────────────────────────────────────────────────────────────┐
│ Trait Layer (静态)                                              │
│   30 维设计师领域信念量表                                       │
│   = designer 个人的 CAPS network weights (借鉴 PsychoBench)     │
│   ⚠ 30 维 是被 CAPS 理论指导下"测量出来的网络权重",             │
│      不是 CAPS 本身                                              │
└────────────────────────────────────────────────────────────────┘
                       │
                       │  CAPS Theory (Mischel & Shoda 1995):
                       │  连接机制, 不是 layer
                       │
                       │  "长期固有信念 (trait) 通过系统性改变瞬时
                       │   认知过程 (state) 来影响行为" — If...Then...
                       │
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ PKVC State Machine (动态运行时)                                 │
│   P (Problem-framing) / K (Knowledge-retrieval)                 │
│   V (Value-judgment) / C (Concept-generation)                   │
│   4 状态处理引擎, Agent 推理时穿越的 4 个状态                   │
│   30 SADR 节点是 4 状态的子细分                                 │
│   ⚠ PKVC 是处理状态机, 不是"输出行为层"                          │
└────────────────────────────────────────────────────────────────┘

Bridge 机制 (本工作 NEW):
  30 维 trait 作为 ontological routing parameters,
  动态调节 PKVC state 间的 state-transition probabilities.
  Triple 形式: <trait pattern, modulation, PKVC state>
  例: <"Data-Driven Belief", "Strong-Inhibit", "Concept-Phase">
```

---

## 3 个错误 + 当日修正

| # | 错误 | 哪儿写错 | 正确表述 |
|---|---|---|---|
| 1 | "CAPS Network = 30 维领域专业信念网络" | brief v3.5 §1 / mapping_miner.py docstring / introduction_zh.md §16 一处 | CAPS 是 Mischel 1995 理论 (trait modulates state-transition 机制); 30 维是 CAPS 理论指导下的 *测量结果*, 是 designer 个人的 network weights, 不是 CAPS 本身 |
| 2 | "PKVC = 输出侧行为层" | brief v3.5 §1 框架表 | PKVC 是处理状态机 (P/K/V/C 4 状态运行时引擎). 不是"输出", 是"推理过程经过的状态". 30 SADR 是 4 状态内部的子节点细分 |
| 3 | Triple = (trait_dim, polarity, SADR_node, polarity), 静态 co-occurrence | brief v3.5 §4D.2 / mapping_miner.py v1 schema + mining 逻辑 | Triple = <factor pattern, modulation (4 档 + bypass), PKVC state>, 动态 distribution shift; mining 用 permutation test 测 P(state | trait_pattern) − P(state | population) |

---

## 为什么会犯这个错

1. **没读权威文档**. 我看到 introduction_zh.md 行 16 写 "**特质表征层(CAPS Network)**: 基于 Mischel 的认知—情感人格系统(CAPS)理论, 我们映射出一个 30 维的领域特定专业信念网络", 直接推断 "CAPS Network = 30 维". 没读 30维-CAPS-pkvc 撰写框架.md 行 40-67 的完整 framing.

2. **CLAUDE.md §13 第 1 条**说"读完本文件, 摸清状态". 我读了 CLAUDE.md 没读项目目录其他 framing 文档. CLAUDE.md §6 文件地图也没列 30维-CAPS-pkvc 撰写框架.md (因为该文件不在 0512技术方案/ 目录, 在 RolePlay_Designer/ 上一级). 跨目录文档默认看不见.

3. **CLAUDE.md §3 用户偏好**说"涉及版本号 / framing 变化时, 立刻拎隐含假设, 不顺着改". 我把 framing 当成"已知", 直接动笔, 没拎"我对 CAPS 的理解是不是对的"这条假设.

---

## 下次 session 启动 checklist (防漂)

```
1. 读 CLAUDE.md (§6 文件地图)
2. **也读** /Users/wy/Desktop/RolePlay_Designer/30维-CAPS-pkvc 撰写框架.md
   — 这是 framing 权威文档, 不在 0512技术方案/ 目录但优先级最高
3. 读 docs/briefs/v3.5.1_20260522.md (或更新版本)
4. 读 docs/notes/framing_correction_20260522.md (本文件, 防再犯同错;
   **特别看下面"2026-05-23 复发"那段 + 加强版三层映射表**)
5. 任何提到 CAPS / PKVC / mapping / triple 的任务, 先 grep 一次:
   grep -rn "CAPS\|PKVC\|mapping\|triple" --include="*.md" 上级目录
   确认 framing 跟撰写框架.md 一致, 再动笔
6. **EFA / 聚类结果解释前, 先对照"加强版三层映射表"** —
   factor / cluster 名词跟 PKVC 状态名词出现在同一段话时立刻警觉
```

---

## 已修正的文件 (2026-05-22)

| 文件 | 修正幅度 |
|---|---|
| `docs/briefs/v3.5.1_20260522.md` | header 升 v3.5.1; §1 框架表行 → "3 层架构"; §4D 整段重写 (motivation / schema / mining / injection / impl / fold); §6 anchor 表 Mischel 行; §14 一句话 takeaway |
| `sft_pipeline/mapping_miner.py` | 完整重写: schema 从 (trait_dim, polarity, SADR, polarity) → (factor, direction, modulation, PKVC state); mining 从 co-occurrence → distribution shift + permutation test; CLI 加 --state-dist / --extracted / --n-perm 等 |
| `CLAUDE.md` §2 | working assumption 重写 3 层架构语言 + 记录 ㉘-㉜ 修正条 |

---

## 2026-05-23 复发 — Trait factor ↔ PKVC state 混淆 (第 2 次防漂失败)

跑完 N=133 EFA pilot 后, 我向 user 解释结果时再次漂了:

| 错误表述 (我说的) | 正确表述 |
|---|---|
| "30 道量表题背后是 4 个心智维度 **(P/K/V/C)** 驱动的" | 30 题背后是 **trait** 的 4-5 潜在结构 (无名, F1-F4/F5), 跟 PKVC 4 状态无等价关系 |
| "**4 因子 PKVC 模型**解释力弱 (累计方差 22.3%)" | 4-5 **trait factor** 累计 22.3%, 是 trait 量表降维丢信息多, 跟 PKVC framing 验证无关 |
| "Kaiser 给 12 因子 → **PKVC 框架可能不成立**" | Kaiser 12 是 trait 量表结构特性, PKVC 4 状态根本不靠 EFA 验证, 靠 extractor κ pilot + 学姐 metadata + Q7 三段覆盖度 |
| "**因子起名: F1 在 PKVC 里到底属于 P/K/V/C 哪个?**" | F1-F4 是 trait factor (无名), 跟 PKVC 4 state **不映射**, 中间隔 bridge mapping triple 做 dynamic modulation |

### 为什么又错了

1. **2026-05-22 修正只动了 brief / mapping_miner.py / CLAUDE.md, 没动我对 EFA 结果的解释模板**.
   修正记录里写 "30 维 ≠ CAPS" 是抽象论, 没写 "EFA 抽出的 factor ≠ PKVC state" 这条具体推论.
2. **聚类 + EFA 出现在 pipeline 里时, 默认按"4 因子 = PKVC"反射性解释**, 因为 PKVC 是 4 个, EFA 假设也是 4-5 个, 数字撞车诱导漂移.
3. user 听完直接拎"PKVC 不是 4 因子是行为表现, 为什么要根据 PKVC 分类" — 反映 user 已经识别了这种漂移, 我 framing 不一致让 user 怀疑整个聚类设计被错配.

### 补丁: 加强版 framing 三层映射表 (任何 EFA / 聚类 / mapping 讨论必查)

```
Layer            | 数据来源              | 是否聚类 | 是否 EFA | 是否 PKVC label |
─────────────────┼───────────────────────┼──────────┼──────────┼─────────────────┤
Trait            | 30 量表               | ✓ K=6    | ✓ 4-5 F  | ✗               |
Bridge           | trait factor × PKVC   | ✗        | ✗        | ✗ (是 triple)   |
PKVC State       | Q1-Q7 → extractor 抽  | ✗        | ✗        | ✓ (P/K/V/C 标签) |
```

**3 个层次的数据来源、统计工具、产物 完全独立**. 看到"EFA 因子" 不准联想"PKVC 状态", 看到"K=6 cluster" 不准联想"PKVC label".

### 给下次 session 的硬规则

```
聚类 / K-means / silhouette / cluster_id 出现时:
  → 这是 Trait Layer 操作, 跟 PKVC 无关
EFA / 因子 / loading / KMO / Bartlett 出现时:
  → 这是 Trait Layer 降维, 抽出的 factor 跟 PKVC state 无映射
PKVC / P/K/V/C / 状态机 / extractor / κ pilot 出现时:
  → 这是 PKVC State Machine Layer, 不靠 EFA 或聚类验证
mapping triple / bridge / modulation 出现时:
  → 这是 Bridge Layer, 是 trait factor pattern → PKVC state 的动态调节
```

---

## 一句话

> 30 维 ≠ CAPS. PKVC ≠ 输出. Triple ≠ 静态映射.
> CAPS = 理论 (连接). PKVC = 状态机 (动态). Triple = state-transition modulation (动态调节).
> **EFA factor ≠ PKVC state. K-means cluster ≠ PKVC label** (2026-05-23 加).
