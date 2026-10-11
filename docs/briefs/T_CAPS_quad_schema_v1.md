# T_CAPS 四元组 schema v1.1 (方向二 only)

**日期**: 2026-05-23
**状态**: proposal, 待学姐 + 老师 review
**前置**: docs/briefs/v3.5.1_20260522.md §4D, docs/notes/framing_correction_20260522.md
**升级路径**: v3.5.1 §4D 三元组 → v1 四元组 (C, s, p, o) → v1.1 加 o_subtype attribute 字段
**v1.1 改动**: 加 `o_subtype` (12 PKVC cardId 子类) 作 attribute 字段, 不参与 main mining; 应对学姐 2026-05-23 反馈"Q3-Q6 选项已标到 P1/P3 子类粒度"

---

## 0. 一句话定位

把 v3.5.1 §4D 锁的 mapping triple 升级为 **context-aware quad** T_CAPS = (C, s, p, o), 显式引入情境字段 C, 真正承接 CAPS 理论 "situation × person → behavior" 的 if-then 机制。同时锁 leakage 防御: 方向二 only, Q7 任何 designer 数据不进 quad DB。

---

## 1. 起源 (学姐 2026-05-23 示例)

学姐发的扩展三元组 JSON:

```json
{
  "If_Situation": "High_P_Authority_Mandate_VS_High_V_User_Harm",
  "Given_Trait": "High_Score_in_Design_Savior_Complex",
  "Predicate_Action": "severely_overrides",
  "Target_Transition": "V_State_Defiance"
}
```

LaTeX 形式: `T_CAPS = (C, s, p, o)`, 在 RDF 标准三元组 (s, p, o) 上加一个情境维度 C。

经过迭代讨论 (2026-05-23 session), 锁定本文档 v1 schema。

---

## 2. 主 schema

| 字段 | 子字段 | 取值 / 词表 | 来源 | 用途 | 例 |
|---|---|---|---|---|---|
| **C** 情境特征 | — | 闭集 ≤ 8 个 tag, 命名格式 `X_field_vs_Y_field` (两端 ∈ PKVC 12 cardId) | Q1-Q8 **题干** (⚠ 不是 designer 答案); 三人各写 → 合并去重 | RAG 过滤 + paper if-then 表达 | `P1_AuthorityMandate_vs_V2_UserWelfare` |
| **s** 潜在特质 | `s_primary` | 30 维信念量表的 top-k high-score tag 集合 | designer 30 维量表分数, 取 high-score 项 | RAG 检索 key (Jaccard / cosine) | `{"救世主情结", "弱势群体优先"}` |
|  | `s_secondary` | (factor, direction), factor ∈ EFA 4-5 因子, direction ∈ {+, -} | 由 `s_primary` 经 EFA factor scores 脚本自动算, 不二次标注 | mapping mining (permutation test) | `(F_共情关怀, +)` |
| **p** 认知路由谓语 | `p_modulation` | 闭集 5 档 {Strong+, Weak+, Weak-, Strong-, Bypass} | 由 `p_surface` 经 codebook 多对一映射 | mining (统计可枚举) | `Strong+ Bypass` |
|  | `p_surface` | 开放词表 8 个 (codebook 锁) | 设计师 **Q1-Q6** NL, LLM 抽取 | paper 写作 + stance 编码 (替代砍掉的 o_attribute) | `defiant_overrides` |
| **o** 目标 PKVC 状态 | `o_primary` | 闭集 4 个 {P, K, V, C} | 设计师 **Q1-Q6** NL, LLM 抽取 | quad target (main mining), 评测打分 | `V` |
|  | `o_subtype` | 闭集 12 个 {P1-P3, K1-K3, V1-V3, C1-C3} | 学姐 Q3-Q6 题目-选项-因子 mapping + LLM 抽取 | attribute (不参与 main mining), paper qualitative + 200 入袋后升 primary | `V1` |

---

## 3. `p_surface` ↔ `p_modulation` codebook v0

| surface_predicate | modulation | 隐含 stance |
|---|---|---|
| `forces_transition_to` | Strong+ | neutral |
| `severely_overrides` | Strong+ Bypass | neutral |
| `defiant_overrides` | Strong+ Bypass | Defiance |
| `gradually_shifts_to` | Weak+ | neutral |
| `compliantly_follows` | Weak+ | Compliance |
| `negotiates_around` | Weak+ / Weak- | Negotiation |
| `weakly_inhibits` | Weak- | neutral |
| `bypasses` | Bypass | neutral |

设计原则:
- surface 词表对外, modulation 词表对内, 多对一映射锁死
- stance 色彩 (Defiance / Compliance / Negotiation) 编码到 surface 词里, 不开独立字段
- 此举替代了砍掉的 `o_attribute` 字段
- LLM 抽取时只输出 surface 词, modulation 由 codebook 查表填

---

## 4. 方向二 数据使用规则 (leakage-clean)

| 阶段 | C | s | p | o | 数据源限制 |
|---|---|---|---|---|---|
| codebook 构建 | Q1-Q8 题干 | EFA on 30 维量表 | 8 词表锁定 | 4 状态锁定 | ⚠ 不用任何 designer 答案 |
| quad DB 构建 (train fold) | 题级标签查表 | train fold designer 量表 | train fold designer **Q1-Q6** NL | train fold designer **Q1-Q6** NL | ⚠ **Q7 任何数据一行不进** |
| SFT 训练 | 输入 prompt | conditioning | supervision target | supervision target | 只看 Q1-Q6 |
| eval (Q7) | Q7 题级标签 (codebook 内) | eval designer 量表 (跨折共享) | 现抽 Q7 NL **仅打分用** | 现抽 Q7 NL **仅打分用** | Q7 NL 抽完一次性, 不入 DB |

**核心规则**: Q7 NL **不进任何训练阶段 retrieval index, 不参与任何 SFT supervision**. 它只在 eval 阶段一次性出现, 用于:
1. Q7 题干 + 题级 C → 作为模型 input
2. Q7 designer NL → 现抽 ground truth quad, 跟模型预测比对打分

**fold-aware 设置**: 5-fold CV split 按 `designer_id`, 不是按 `question_id`. 每折独立建 quad DB, peer retrieval scope 限制在 train fold + same cluster_id.

---

## 5. C 字段: 双层 framing (学姐 2026-05-23 升级)

学姐提出"组合情境"概念: CAPS situation 不应是单层 tag, 而应是 **background 全局压力源 + local trigger 局部触发器** 的组合。本文档接住该 framing, 但**做工程拆分**:

```
C_global  Task-level Background Pressure
          闭集 ≤ 5 个 background tag (后续多 task 时定义, 当前 Q1-Q6 共用 1 个)
          来源: 题目设计者 (学姐) 写的 task background 文本
          颗粒度: task-level (1 个 task scenario 1 个 background)
          落点: ⚠ 进 prompt narrative, ❌ 不入 quad schema
          原因: 当前训练 dataset 全用 1 个 task background (Q1-Q6 同 task), C_global 是 dataset 常量,
                进 quad 字段无 variance, mining 不出来; 进 prompt 让 LLM in-context 处理即可.
                未来加第二个 task 时再升级 schema 把 C_global 入 quad.

C_local   Question-level Local Trigger
          闭集 ≤ 8 个 trigger tag (学姐 5/23 给了 Q3-Q6 共 4 个)
          来源: Q1-Q8 题干 extract
          颗粒度: question-level (1 题 1 个 trigger, 跟 designer 答案脱钩)
          落点: ✅ 进 quad schema (`C` 字段就是这个)
```

**`C_local` codebook v0 (学姐 2026-05-23 真实版本)**:

| 题号 | 题干主轴 | C_local tag |
|---|---|---|
| Q3 | 医疗器械般笨重但准确率 99% vs 美学 79% | `High_K_Utility_vs_Low_C_Aesthetics` |
| Q4 | 封闭生态获取隐私 vs 开源协议 | `High_P_Commercial_Profit_vs_High_V_Ethical_Risk` |
| Q5 | AI 自动生成效率高但无机没人味 | `High_K_Automation_Efficiency_vs_Low_V_Authorship_Identity` |
| Q6 | 用户焦虑异化 vs 老板强令上线 | `High_P_Authority_Mandate_vs_High_V_User_Harm` |
| Q1 | (待学姐补) | TBD |
| Q2 | (待学姐补) | TBD |
| Q7 | (待学姐补, eval 用) | TBD |
| Q8 | (defer, P0-1 待拍) | TBD |

**命名约束** (v1 锁定):
- 格式: `[High/Low]_X_field_[Name]_vs_[High/Low]_Y_field_[Name]`
- High/Low 极性前缀必须有 (学姐版表达力升级, 区分"双高冲突" vs "高低对立")
- 中间分隔符锁小写 `_vs_`
- 两端字段名 ∈ PKVC 12 cardId 子项 (Q4 的 `Commercial_Profit` 暂归 P 字段, 等学姐 cardId metadata 最终校准)

**`C_global` prompt narrative 段** (示例, 当前 Q1-Q6 task):
```
[Task Background]:
设计师正在某科技公司主导一个 AI 产品设计项目, 全局承受以下压力:
  - 商业压力: 公司战略要求短期内完成数字化转型
  - 时间压力: 产品 deadline 紧迫
  - 道德压力: 用户福祉与商业指标存在张力
  (具体文本由学姐补 50-100 字, 写进 prompt template)
```

**paper 表达**:
> We formalize situation as a 2-layer construct: a task-level Background pressure
> (C_global), instantiated once per task scenario and entering as prompt context;
> and question-level Local triggers (C_local), extracted from question stems and
> stored as a structured field in the quad. Our T_CAPS = (C_local, s, p, o)
> operationalizes the local layer; the global layer enters as scenario narrative.

---

## 6. 一条完整 quad 示例 (训练数据)

```json
{
  "quad_id": "uD001_Q6",
  "designer_id": "D001",
  "question_id": "Q6",
  "fold": 0,

  "C": "P1_AuthorityMandate_vs_V2_UserWelfare",

  "s_primary": ["救世主情结", "弱势群体优先", "完美主义/MVP"],
  "s_secondary": {"factor": "F_共情关怀", "direction": "+"},

  "p_surface": "defiant_overrides",
  "p_modulation": {"level": "Strong+", "bypass": true},

  "o_primary": "V",
  "o_subtype": "V1"
}
```

**注**: `o_primary` 进 mining grid 跑 permutation test; `o_subtype` 仅作 attribute, paper qualitative + 200 入袋后再评估是否升 primary。

### paper 表达 (unfold 后)

```
情境 P1_AuthorityMandate_vs_V2_UserWelfare 下,
该设计师高度认同"救世主情结 / 弱势群体优先 / 完美主义"三项
(共属共情关怀因子, 正向),
其认知路由表现为 defiant_overrides — 强烈压制默认 Problem 阶段
并跳过 (bypass), 强制跳转到 V (Value Judgment).
```

---

## 7. 跟 v3.5.1 §4D 三元组的差异

| 维度 | v3.5.1 §4D | 本文档 v1 |
|---|---|---|
| 元组阶 | 3 元 (source, modulation, target) | 4 元 (C, s, p, o) |
| 情境 C | 隐式塞进 PKVC state machine | 显式独立字段 |
| s 表达 | factor pattern 单一形式 | 双字段 (s_primary 30 维检索 / s_secondary factor mining) |
| p 表达 | 4 档 modulation + Bypass flag | 双层 (modulation 5 档 / surface 8 词) |
| o 表达 | PKVC 4 state | PKVC 4 state primary + 12 子类 attribute (v1.1) |
| stance 色彩 | 无 | 编码到 surface 词里 (Defiance/Compliance/Negotiation) |
| mining 算法 | distribution shift + permutation test | 同 (用 s_secondary, p_modulation, o 三元跑) |
| paper anchor | trait-state modulation | situation-conditioned cognitive routing (升级) |

**main claim 条件路径不变**:
- §5.5 跑 og_rag+modulation vs og_rag-only
- 若涨点 > 1%, 升级 mapping 为 main claim
- 升级时 framing hook 从 "trait → state modulation" 改成 "situation-conditioned cognitive routing"

---

## 8. 已锁的决策 (2026-05-23 session)

| 编号 | 决策 | 锁定方 |
|---|---|---|
| L-1 | C 字段加入, 显式承接 CAPS situation × person 机制 | user + 学姐 |
| L-2 | s 双字段 (s_primary 30 维 + s_secondary factor pattern) | user |
| L-3 | p 双层 (modulation 5 档 + surface 8 词, codebook 多对一) | user |
| L-4 | o_attribute 砍, stance 色彩归并到 p_surface 词里 | user |
| L-5 | C 词表来源: Q1-Q8 题干 (⚠ 不是 designer 答案), bottom-up extract | user + 学姐示例 |
| L-6 | 方向二 only, Q7 NL 不进 quad DB, 只 eval 时一次性抽 | user (待跟老师同步, 见 §10 ③) |
| L-7 | C 双层 framing: C_global 进 prompt narrative, C_local 进 quad schema | user (学姐 2026-05-23 "组合情境" + "按技术可行性来") |
| L-8 | C_local 命名: `[High/Low]_X_field_[Name]_vs_[High/Low]_Y_field_[Name]` 锁定 | user (学姐 2026-05-23 4 个真实 tag normalize 后) |
| L-9 | o 拆分 primary (4 大类, main mining) + subtype (12 cardId, attribute); 200 入袋后重评估是否升 subtype 到 primary | user (学姐 2026-05-23 反馈 Q3-Q6 选项已标到子类 + "按技术可行性来") |

---

## 9. 数据稀疏度算账

```
mining 用 o_primary (4 大类) 而不是 o_subtype (12), 这是 v1.1 的核心决策:

mining (s_secondary, p_modulation, o_primary, 不含 C):
   |s_sec| × |p_mod| × |o_pri| = 8 × 5 × 4 = 160 cell
   200 designer × ~3 quad/人 = 600 quad
   600 / 160 = 3.75 quad/cell
   → permutation test 可跑, fold-aware 后仍 ~2 quad/cell

加 C 后 (4 元联合):
   |C| × |s| × |p| × |o_pri| = 8 × 8 × 5 × 4 = 1280 cell
   600 / 1280 = 0.47 quad/cell (稀)
   → 不直接 4 元 mining, 走 hierarchical:
     step 1: (s_secondary, p_modulation, o_primary) 三元先挖 main pattern
     step 2: 检验 C 是否对挖到的 pattern 有 distribution shift (moderator analysis)

⚠ 不用 o_subtype 跑 mining 的原因:
   |s_sec| × |p_mod| × |o_sub| = 8 × 5 × 12 = 480 cell
   600 / 480 = 1.25 quad/cell, permutation power 边缘
   加 C 后 = 3840 cell, 600/3840 = 0.16 quad/cell, 跑不动
   → 200 入袋后重算 (1200 quad / 480 = 2.5 quad/cell, 边缘 working), 再评估升 primary

retrieval (s_primary):
   key = 30 维 high-score tag 集合, 集合 Jaccard / cosine top-k
   稀疏不影响 retrieval 成立
```

---

## 10. 待拍 / 待补 4 项

| 编号 | 事项 | 阻塞方 | 优先级 |
|---|---|---|---|
| ① | PKVC 12 cardId 完整定义 → C 词表 v0 落地 | 学姐 (阻塞 D) | P0 |
| ② | `p_surface` 8 个词够不够 / 扩到 10 | 学姐 | P1 |
| ③ | 方向一砍掉是否已跟老师同步 (CLAUDE.md / v3.5.1 brief 还锁着"对照") | 自己 | P0 (framing 决策不能默漂) |
| ④ | `mapping_miner.py` / `mapping_retrieval.py` schema 从三元组升级到四元组 | 自己 (等 ①) | P1 |

---

## 11. 工程改动清单 (等学姐 ① 之后启动)

```
新增:
  data/processed/situation_codebook.json        C 词表
  data/processed/predicate_codebook.json        p_surface ↔ p_modulation 映射
  sft_pipeline/rag/quad_schema.py               dataclass T_CAPS quad

改造:
  sft_pipeline/rag/mapping_miner.py             升级 schema 从三元组到四元组
    + hierarchical mining (step 1 三元 + step 2 C moderator)
  sft_pipeline/rag/mapping_retrieval.py         升级 retrieval key 从 s 单字段到 (C, s_primary)
    + filter_by_situation(C_tag) 一层过滤
  sft_pipeline/extract/llm.py                   extractor output schema 加 p_surface 字段

不动:
  sft_pipeline/model.py                         L_cc / L_cls heads 不变
  sft_pipeline/train.py                         curriculum + sampler 不变
  fold split 逻辑                                 仍按 designer_id 5-fold
```

---

## 12. 下次 session 接手要先看的

1. 本文档 §2 主 schema 表 + §3 codebook
2. §4 数据使用规则 (leakage 防御)
3. §10 待拍 4 项 (尤其 ③ 方向一确认)
4. v3.5.1 brief §4D (历史 trace, 看升级前是什么样)

---

## 一句话给学姐 / 老师 review 时强调

T_CAPS = (C, s, p, o) 这版相比 v3.5.1 §4D, 主要改动是:
**显式加 C (情境字段, 承接 CAPS if-then)**,
**s 和 p 都拆成"检索用 / mining 用"双字段**,
**stance 色彩从独立 attribute 编码进 p_surface 词**,
**严格锁 Q7 NL 只在 eval 一次性出现, 不进任何 retrieval index**.

如果 §5.5 跑出 og_rag+mapping > og_rag-only > 1%, paper main framing 从
"trait → state modulation" 升级为 "situation-conditioned cognitive routing".
