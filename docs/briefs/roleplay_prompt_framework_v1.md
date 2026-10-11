---
date: 2026-05-25 (v1.1 patch, base v1 2026-05-24)
version: v1.1
status: spec lock (q1q6_option_labels.jsonl + q2_training_labels.jsonl 已就绪, 可进 implement)
authors: 二作
target: gpt-4o zero-shot baseline 集 + SFT prompt 模板 (训练 prompt 同结构)
---

# v1.1 patch (2026-05-25, Path D 拍板)

跟 v1 (2026-05-24) 的 diff:

1. **routing_format["weighing"]**: 从 "12 cardId 必须完整" 改成 "user real partial+dup ordering"
   - 触发依据: 实测 N=163 里 **0 user 真排满 12 unique cardId**.
     长度分布 median=8, range 4-20, 29% user 有 dup. Framework v1 原 spec 跟数据现实完全 mismatch.
   - 决策: 接受 partial+dup 当 fidelity-preserving ground truth (Path D).
     用 q2_nl_refined.jsonl 已 LLM-refine 的 100-300 字 NL 段补齐解释信号.
   - 影响: §7 routing_format_instruction["weighing"] + §8 Python `ROUTING_FORMAT_INSTRUCTIONS` + §12 lock #5

2. **Q7.seg2 评测端 alignment**: Q7.seg2 v8.7 抽出来的 state_sequence median=3 (range 1-6).
   训练-eval schema 异质 (训 4-20 项 cardId, eval 1-6 factor letter), eval 时需要 factor-collapse + Kendall-tau on common factors.
   q2_v8_7_ranking_eval.py 已经实现这个 collapse, 现成可用 (top-1 96%, Jaccard 0.90).

3. **lock #5 Q2 部分修订**: Q2 训练 ground truth = q2_nl_refined.jsonl entries
   (115/163 primary + 48 fallback), 不是 12 cardId 完整序列.
   Q3-D / Q6-A ambiguous primary 已拍板入 q1q6_option_labels.jsonl (alt_labels 留备 eval).

4. **v1 砍掉 prior_segs / teacher-forcing** (原 lock #14) — §10 ablation 矩阵里
   "teacher-forcing seg2" 这一行作废, paper §10 不再考虑.

5. **场景 grounding verify**: §7 lock #15 "训练 Q1-Q6 共用全感知效率空间" 已在 N=163 上 verify
   — Q4 选项 100% 含 "隐私", Q5 100% 含 "AI", Q3 题面 EEG 传感器精度 ✓.
   原 v1 担忧的 "ground truth 跟 prompt scenario mismatch" 不成立.

6. **peer_raw_nl train-eval asymmetric** (lock #21 新, 2026-05-25 拍):
   训练 prompt 不注入 peer_raw_nl, 仅 eval 阶段注入.
   原因: train-time prompt 塞 peer Q7 NL 占 54% token, training target 信号被淹 +
   model over-anchored 在 train fold Q7 distribution.
   token 长度从 max 6145 (超 4096) 降到 3546 (全 fit max_seq=4096), batch=32 不动.
   工程: prep_v2 `build_prompt(is_training=True)` 默认强制 peer off; infer_v2 按 baseline config.

---

# Role-play Q7 Prompt Framework v1 (可拆卸 ablation)

## 0. 设计原则

1. **模块化**: 6 个独立 section, 每个 section 有 boolean toggle, 关掉就**整段不出现**在 prompt 里 (不留占位符空骨架, 避免给 LLM 暗示某段 "应该有但缺失")
2. **placeholder 命名空间**: 所有数据注入用 `{{section.field}}` 模板, 工程上易扫
3. **ablation 矩阵驱动**: 标准 baseline 是 toggle 的 fixed 组合, 也允许任意自定义组合
4. **section 顺序固定** (即使部分关掉): 跟 LLM 注意力 prior 一致 (近 prompt 末权重高), 不要让 ablation 改变信息位置
5. **section 间用 hard delimiter** (`╔═...═╗ / ╚═...═╝`) 隔开, 不用 Markdown `#` (防 LLM 模仿格式输出 markdown 标题)
6. **PKVC 不写单字母**: 所有出现 P/K/V/C 的地方一律展开为自解释中英文标签, LLM 不依赖外部约定
7. **per-segment 生成**: 不要求 LLM 一次输出 seg1+seg2+seg3, 改 3 次独立调用, 每次生成 1 段; seg2/seg3 可选注入前段 real-context 做 teacher-forcing eval

---

## 0.1 PKVC 标签命名约定 (所有 section 共用)

LLM 不懂 "P" "K2" 这种单字母 / 编号是什么意思. 所有 section 必须用 verbose 标签:

**因子级 (o_primary, dominant_factor 等)**:

| 单字母 (内部) | LLM-facing verbose 标签 |
|---|---|
| `P` | **P (Problem 现实与问题重构)** |
| `K` | **K (Knowledge 经验/数据/技术调用)** |
| `V` | **V (Value 价值判断)** |
| `C` | **C (Concept 构念/构建)** |

**子类级 (o_subtype, p_modulation 路径终点等, 12 个 SADR)**:

| 单字母 (内部) | LLM-facing verbose |
|---|---|
| `P1` | P1 界定核心 |
| `P2` | P2 拆解子问题 |
| `P3` | P3 推翻假定 |
| `K1` | K1 调用经验 |
| `K2` | K2 客观数据测试 |
| `K3` | K3 评估技术栈 |
| `V1` | V1 价值立场 |
| `V2` | V2 利益排序 |
| `V3` | V3 设立原则 |
| `C1` | C1 涌现画面 |
| `C2` | C2 整合构建 |
| `C3` | C3 提炼原则 |

**路径表示** (e.g. `K→V`): `K (Knowledge) → V (Value)`, 自解释.

**实现**: helper `pkvc_verbose(letter_or_path: str) -> str` lookup, 所有 template injection 前过这个函数. 单字母原值仅保留在内部 dataclass / JSON output 字段, **prompt 永远 verbose**.

---

## 1. Section 一览

| #   | Section                                                | Toggle          | 出现位置            | 必含 baseline |
| --- | ------------------------------------------------------ | --------------- | --------------- | ----------- |
| 1   | SYSTEM (role + per-segment format spec)                | `system`        | system message  | always      |
| 2   | TARGET_PROFILE (30 维 trait 画像)                         | `profile`       | user message §1 | B1+         |
| 3   | TARGET_OWN_RESPONSES (Q1-Q6 NL 历史 + option PKVC)       | `own_responses` | user message §2 | B3+         |
| 4   | TARGET_OWN_QUADS (Q3-Q6 CAPS 路由)                       | `own_quads`     | user message §3 | B3+         |
| 5   | PEER_RAW_NL (3 peer Q7 raw NL)                         | `peer_raw_nl`   | user message §4 | B2          |
| 6   | SCENARIO + OUTPUT (Q7 单段 + 前段 teacher-forcing context) | `scenario`      | user message 底  | always      |

**v1 修订** (跟初版 draft 差异):
- 砍掉 PEER_QUADS section (跟 OWN_QUADS 重复, 79 user 都有 own quads 时优先用 own)
- per-segment 生成: SYSTEM + SCENARIO 都改成 "只输出 1 段", 3 次独立调用 (seg1/seg2/seg3)
- OWN_RESPONSES 加 Q2 NL + option_text 合并 PKVC 标签 (不再裸字母)
- OWN_QUADS 内部 PKVC 单字母改 verbose

**Ablation 矩阵** (固定 baseline):

| Baseline | system | profile | own_resp | own_quads | peer_raw_nl | scenario |
|---|---|---|---|---|---|---|
| B0 cold | ✓ |   |   |   |   | ✓ |
| B1 profile only | ✓ | ✓ |   |   |   | ✓ |
| B2 + peer raw-NL | ✓ | ✓ |   |   | ✓ | ✓ |
| B3 + own quads | ✓ | ✓ |   | ✓ |   | ✓ |
| B4 + own resp + own quads | ✓ | ✓ | ✓ | ✓ |   | ✓ |
| **B5 known-user full** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| custom | 任意组合 |

**额外 segment 维度**: 每 baseline × 3 segments (seg1/seg2/seg3) × {无 prior / real prior teacher-forcing} = 6 个细 variant per baseline.

**Coverage 提醒**:
- own_quads 限 79 in-hand user (84 incremental 没跑 extractor); target 不在 79 池时该 toggle 自动 off
- own_resp.expansion NL 32% (cell 级); 缺时 fallback 短 text_supplement

---

## 2. Section 1 — SYSTEM (always on)

**Toggle**: `system` (强制 on)

**Position**: API `messages[0].role = "system"`

**Template** (干净版, 不暴露 segment / 思考链 meta 给 LLM):

```
你扮演下面注入的特定设计师, 用第一人称回答提出的问题. 严格遵守输出 schema.

输出 schema (双段结构化, 跟问题类型对应):

  [认知路由标签]
  <一行结构化 PKVC 标注; 具体格式跟问题类型对应, scenario 段会明确要求>

  [NL 回答]
  <100-300 字一段连续中文 NL>

格式约束:
- [认知路由标签] 跟 [NL 回答] 各占独立段, 两段之间空一行
- [认知路由标签] 内容必须是合法 PKVC 表达 (见 §0.1 verbose 约定):
    单字母:  "C (Concept 构念/构建)"
    路径:    "P3 推翻假定 → K3 评估技术栈"
    序列:    "P1 界定核心 > V3 设立原则 > K2 客观数据测试 > ..." (weighing 函数按用户优先级排序, 可变长度 4-12+ 项, 允许重复)
- [NL 回答] 100-300 字, 一段连续中文, 不要列表 / markdown / 子标题, 不要前缀
  "我会..." 之外的元话语 ("我的回答:" / "首先" / "总结一下" 都不要)
- 必须体现下面注入的设计师 trait 倾向 (强 + 的 trait 应明显出现, 强 - 的应反对或弱化)
- 若上方给了该 designer 在过去问题的认知路由 (own_quads), 本次 [认知路由标签] 跟 [NL]
  应跟过去路由模式一致 (e.g. 过去走 K2 客观数据测试的 designer, 本次也应倾向 K-driven)
- 禁止 default 到 V (Value 价值判断) 的 "balanced ethical reasoning" 套话
```

**注入数据**: 无 — SYSTEM 完全静态, 跟问哪一段无关。LLM 完全不知道 "seg1/seg2/seg3" 或 "思考链 3 段" 的存在, 只看到具体问题 (在 §7 SCENARIO 段提) + 该问题的 [认知路由标签] 格式说明。

---

## 3. Section 2 — TARGET_PROFILE (30 维 trait 画像)

**Toggle**: `profile`

**Position**: user message 第一段

**Template**:

```
╔═══ TARGET DESIGNER PROFILE ═══╗

[30 维信念量表 4 档极性 NL]
{{profile.polarity_nl}}

[Demographic]
{{profile.demographics}}

╚═══════════════════════════════╝
```

**注入数据**:

| 字段 | 来源 | 格式 |
|---|---|---|
| `profile.polarity_nl` | `numerical_to_polarity()` + `polarity_to_nl()` (sft_pipeline.data.prep) | "强 +: 天赋信仰, 黑盒透明度\n弱 +: ..." 4 行 |
| `profile.demographics` | `users.csv` 选 `identity`, `years_exp`, `design_field`, `education_stage_current` 拼成 1 行 | "学生 / 5 年经验 / UX 方向 / 硕士" |

**v3.7 修正**: 之前 draft 写了 `EFA 4 因子 score F1-F4` 这段, **已删** —
- EFA cluster (K=6) 在 v3.7 撤掉 (datav3 L3 cluster 弱区分)
- EFA factor scores (4-d 连续) 仍保留**作内部 use**:
  - peer retrieval (§5/§6 trait cosine 用)
  - SFT 训练 trait-factor prefix / L_trait_reg head
- 但 **factor scores 不该进 LLM-facing prompt** — LLM 不懂 F1=+0.62 是什么意思, 给只是噪声
- LLM 只需要看 **30 维 polarity NL** (中文 trait 名字 + 4 档极性), 这个 LLM 能理解

---

## 4. Section 3 — TARGET_OWN_RESPONSES (Q1-Q6 NL 历史)

**Toggle**: `own_responses`

**Position**: user message §3

**Template**:

```
╔═══ TARGET DESIGNER'S OWN Q1-Q6 HISTORY (这位设计师在过去 6 题的真实回答 + 自己写的解释) ═══╗

[Q1 面对委托第一启动认知模块]
  选项 (内容 + 该选项激活的 PKVC factor): {{own_responses.q1.option_text_with_pkvc}}
  设计师自己写的 NL 解释: {{own_responses.q1.nl}}

[Q2 PKVC 优先级排序题]
  设计师 cardId 排序: {{own_responses.q2.cardids_ordered_verbose}}
  设计师自己写的 NL 解释: {{own_responses.q2.nl}}

[Q3 数据 vs 美感冲突]
  选项 (内容 + 该选项激活的 PKVC factor): {{own_responses.q3.option_text_with_pkvc}}
  设计师自己写的 NL 解释: {{own_responses.q3.nl}}

[Q4 商业 vs 隐私 vs 开源]
  选项 (内容 + 该选项激活的 PKVC factor): {{own_responses.q4.option_text_with_pkvc}}
  设计师自己写的 NL 解释: {{own_responses.q4.nl}}

[Q5 AI 冲击与人本/有机平衡]
  选项 (内容 + 该选项激活的 PKVC factor): {{own_responses.q5.option_text_with_pkvc}}
  设计师自己写的 NL 解释: {{own_responses.q5.nl}}

[Q6 商业利润 vs 用户责任]
  选项 (内容 + 该选项激活的 PKVC factor): {{own_responses.q6.option_text_with_pkvc}}
  设计师自己写的 NL 解释: {{own_responses.q6.nl}}

╚═══════════════════════════════╝
```

**option_text_with_pkvc 格式示例**:

```
"我会先在脑中形成产品形态和使用场景，想象它如何与手机 App 协同..." 
  [该选项激活: C (Concept 构念/构建), 子状态 C1 涌现画面]
```

**注入数据**:

| 字段 | 来源 | 格式 | fallback |
| --- | --- | --- | --- |
| `own_responses.qN.option_text_with_pkvc` | task_responses answer_value (去引号 + 去 "A. ") + `option_trait_mapping.json` o_primary_hint + o_subtype_hint, verbose 化 | "选项文本... [该选项激活: V (Value 价值判断), 子状态 V3 设立原则]" | "未答" |
| `own_responses.qN.nl` (N=1..6, 含 Q2) | **expansion outputs (102 字, 优先)**, fallback text_supplement (raw, 100% 覆盖但短 25 字) | NL 100-130 字 | "(无补充解释)" |
| `own_responses.q2.cardids_ordered_verbose` | task_responses Q2 JSON → cardId list, 每 cardId 展开 + label (e.g. "P1 界定核心 > P2 拆解子问题 > K1 调用经验 > ...") | verbose 序列 | "未排序" |

**覆盖率**:
- option (Q1, Q3-Q6, Q8): 100% (163/163)
- Q2 cardId order: 100%
- expansion NL: 32% cell-level (309/978); 但 Q2 expansion 覆盖率单独统计 (需 verify)
- text_supplement: 100% 但短 (median 25 字, 信号弱), 作 fallback

**v1 修订**: 之前 draft 把 option 字母 (A/B/C/D) 单独列了, **已删** — 字母无信息量, LLM 不知道 A 是什么. 改成 option 文本 + PKVC 激活标签合并展示 (中文 verbose).

---

## 5. Section 4 — TARGET_OWN_QUADS (Q3-Q6 CAPS 路由)

**Toggle**: `own_quads`

**Position**: user message §4

**Template** (所有 PKVC 字段 verbose 化):

```
╔═══ TARGET DESIGNER'S CAPS COGNITIVE ROUTING (从 Q3-Q6 NL 抽出的结构化路由) ═══╗

[Q3 数据 vs 美感冲突 (情境冲突轴: K vs C/V)]
  trait 画像 (该 designer 在此题的群体共同高分 trait): {{own_quads.q3.s_primary}}
  路由动词: {{own_quads.q3.p_surface_verbose}}
  跳转目标状态: {{own_quads.q3.o_primary_verbose}}, 子状态 {{own_quads.q3.o_subtype_verbose}}
  底线 (criterion verbatim 节选): {{own_quads.q3.criterion}}

[Q4 商业 vs 隐私 vs 开源 (情境冲突轴: K/P vs V vs C)]
  trait 画像: {{own_quads.q4.s_primary}}
  路由动词: {{own_quads.q4.p_surface_verbose}}
  跳转目标状态: {{own_quads.q4.o_primary_verbose}}, 子状态 {{own_quads.q4.o_subtype_verbose}}
  底线: {{own_quads.q4.criterion}}

[Q5 AI 冲击与人本/有机 (情境冲突轴: K vs C vs V vs P)]
  trait 画像: {{own_quads.q5.s_primary}}
  路由动词: {{own_quads.q5.p_surface_verbose}}
  跳转目标状态: {{own_quads.q5.o_primary_verbose}}, 子状态 {{own_quads.q5.o_subtype_verbose}}
  底线: {{own_quads.q5.criterion}}

[Q6 商业 vs 用户责任 (情境冲突轴: V vs C vs P/K)]
  trait 画像: {{own_quads.q6.s_primary}}
  路由动词: {{own_quads.q6.p_surface_verbose}}
  跳转目标状态: {{own_quads.q6.o_primary_verbose}}, 子状态 {{own_quads.q6.o_subtype_verbose}}
  底线: {{own_quads.q6.criterion}}

p_surface 8 词路由动词含义 (供 LLM 参考, 不用输出, 已是中英自解释):
  forces_transition_to     = Strong+ 强推转移 (无情绪)
  severely_overrides       = Strong+ 强覆盖 (bypass 跳过现状)
  defiant_overrides        = Strong+ 对抗性覆盖 (Defiance 情绪)
  gradually_shifts_to      = Weak+ 渐进转移
  compliantly_follows      = Weak+ 顺从跟随 (Compliance)
  negotiates_around        = Weak+ 协商绕过 (Negotiation)
  weakly_inhibits          = Weak- 轻微抑制
  bypasses                 = Bypass 直接跳过/无效

╚═══════════════════════════════╝
```

**verbose 化 helpers**:

| 字段 | 注入前转换 |
|---|---|
| `own_quads.qN.o_primary_verbose` | `pkvc_verbose(quad['o_primary'])` → "K (Knowledge 经验/数据/技术调用)" 等 |
| `own_quads.qN.o_subtype_verbose` | `pkvc_verbose(quad['o_subtype'])` → "K2 客观数据测试" 等 |
| `own_quads.qN.p_surface_verbose` | quad['p_surface'] 已是 8 词 verbose, 但加 `(p_modulation)` 注释, e.g. "forces_transition_to (Strong+ 强推转移)" |

**注入数据**:

| 字段 | 来源 |
|---|---|
| `own_quads.qN.*` | `data/processed/quad_extracted_v3_aligned.jsonl` filter by `designer_id == target_uid && question_id == "QN"` |

**覆盖率**: 79/163 user 有 own quads (in-hand 那批已跑 extractor); 84 incremental 没。
**Fallback**: target 不在 79 池 → 整段 toggle off (跟 B3 一致)

---

## 6. Section 5 — PEER_RAW_NL (3 peer Q7 NL)

**Toggle**: `peer_raw_nl`  

**⚠ lock #21 (v1.1)**: 该 section 仅 eval (推理) 时注入, **训练 prompt 强制不注入**.
train-eval schema asymmetric, 见 v1.1 patch §6 + lock 表 #21.

**Position**: user message §5

**Template**:

```
╔═══ TOP-3 TRAIT-SIMILAR PEERS' Q7 RAW NL (作风格/语气 reference, 不要照抄) ═══╗

[peer 1 — trait-similarity rank 1]
seg1: {{peer_raw_nl.p1.seg1}}
seg2: {{peer_raw_nl.p1.seg2}}
seg3: {{peer_raw_nl.p1.seg3}}

[peer 2 — trait-similarity rank 2]
seg1: {{peer_raw_nl.p2.seg1}}
seg2: {{peer_raw_nl.p2.seg2}}
seg3: {{peer_raw_nl.p2.seg3}}

[peer 3 — trait-similarity rank 3]
seg1: {{peer_raw_nl.p3.seg1}}
seg2: {{peer_raw_nl.p3.seg2}}
seg3: {{peer_raw_nl.p3.seg3}}

╚═══════════════════════════════╝
```

**注入数据**:

| 字段 | 来源 |
|---|---|
| `peer_raw_nl.pN.sim` | 4-d factor cosine score (`np.dot / np.linalg.norm`) |
| `peer_raw_nl.pN.seg{1,2,3}` | task_responses Q7 `answer_value` 解析 JSON → `seg1/seg2/seg3.text` |

**Peer 选取**: 4-d EFA factor space cosine top-3 from `eligible ∩ has_Q7` pool (163 用户里 ~160 有 Q7)
**长度上限**: 每 seg 取前 500 字 (防爆 context, 3 peer × 3 seg × 500 = 4500 字 ≈ 6K tokens 可控)

---

## 7. Section 6 — SCENARIO + OUTPUT_INSTRUCTION (always on, Path A 无 prior_segs)

**Toggle**: `scenario` (强制 on)

**Position**: user message 最后一段

**Template** (Path A 已锁: 无 prior_segs, 每次调用独立; 注入认知路由格式说明):

```
╔═══ 场景 ═══╗

{{scenario.text}}

╚═══════════╝

{{scenario.question}}

[认知路由标签格式 (本次必须遵守)]
{{scenario.routing_format_instruction}}
```

**注入数据**:

| 字段 | 内容 |
|---|---|
| `scenario.text` | 跟问题对应的场景描述. **训练 Q1-Q6** = "全感知效率空间" 项目场景 (穿戴 + EEG + APP 协同的设计师项目); **评测 Q7** = "老城区超市数字化改造" 场景. 两套场景文本固定 hard-coded |
| `scenario.question` | 跟 `cognitive_function` 对应的自然问题 (3 个 function, 见下表) |
| `scenario.routing_format_instruction` | 跟 `cognitive_function` 对应的 `[认知路由标签]` 输出格式说明 (3 个, 见下表) |

---

**3 个自然问题** (COGNITIVE_FUNCTION_QUESTIONS, 训练 + 推理共享):

| function | question (LLM 看到的问句) | 训练用 | 推理用 |
|---|---|---|---|
| `first_reaction` | "面对这个项目, 您的直觉第一反应是什么? 请直接说出您不经决策权衡的本能反应。" | Q1 训练 | Q7.seg1 |
| `weighing` | "请按您的优先级排出 PKVC 12 个 cardId 的序列, 并说明您的权衡逻辑 (先排序, 再 NL)。" | Q2 训练 | Q7.seg2 |
| `bottom_line` | "面对这个具体冲突, 您会怎么决策? 您向上层提出的最核心底线是什么? 哪些是不可让步的?" | Q3-Q6 训练 | Q7.seg3 |

---

**3 个 routing_format_instruction** (跟 function 一一对应):

| function         | scenario.routing_format_instruction                                                                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `first_reaction` | "在 [认知路由标签] 段输出**单字母 + verbose 标签**, 形如 `C (Concept 构念/构建)` 或 `P (Problem 现实与问题重构)`. 表示本次第一反应主导的认知模块. 然后空行 + [NL 回答]."                                                            |
| `weighing`       | "在 [认知路由标签] 段输出 **PKVC 12 cardId 的优先级序列** (可变长度, 通常 4-12 项), 用 ` > ` 分隔, 每 cardId 后跟 verbose 含义, 形如 `P1 界定核心 > V3 设立原则 > K2 客观数据测试 > C2 整合构建 > P2 拆解子问题 > K1 调用经验`. **允许重复表示强偏好** (e.g. `K1 调用经验 > K3 评估技术栈 > K1 调用经验`), 不要求 12 项全覆盖. 然后空行 + [NL 回答] (说明权衡逻辑)." |
| `bottom_line`    | "在 [认知路由标签] 段输出 **PKVC 单字母** (e.g. `V (Value 价值判断)`) 或 **2 段路径** (e.g. `P3 推翻假定 → K3 评估技术栈`, 头 ≠ 尾, 上限 2 段). 表示底线决策的认知路径. 然后空行 + [NL 回答] (说明不可让步的部分)."                            |

---

**输出 schema (LLM 应该返回的, 跟 §2 SYSTEM 同步)**:

```
[认知路由标签]
<按 routing_format_instruction 要求的格式: 单字母 / 路径 / 可变长 ranking>

[NL 回答]
<100-300 字一段连续中文>
```

---

**Path A 锁定后的工程约定 (v1 修订)**:

- 砍掉 `prior_segs` / `prior_context_block` / PRIOR_REFERENCE_INTRO dict — 训练-推理严格 parity
- 砍掉 "real-context vs autoregressive teacher-forcing" 选项
- 每个 cognitive function 1 次独立 API 调用, 跟 brief v3.5.1 方向二 per-question SFT 训练对齐
- 同 cognitive function 在训练 (e.g. Q3 训) 跟 eval (Q7.seg3 评) 的 prompt schema 严格一致 (only scenario.text 不同, question 跟 routing_format_instruction 完全相同)

---

**对齐 brief v3.5.1 方向二 (direction-q1q6)**:

- 训练源: Q1 (→ first_reaction) + Q2 (→ weighing) + Q3-Q6 (→ bottom_line) = **6 个训练样本 per user**
- 评测源: Q7 (→ seg1 first_reaction + seg2 weighing + seg3 bottom_line) = **3 个 eval segment per user**
- bottom_line function 训练 sample 数最多 (Q3+Q4+Q5+Q6 = 4×), first_reaction / weighing 各 1× → seg3 评测信号最强
- cognitive function 是真正训练目标; Q 题号只是数据载体; Q7 跨场景 transfer 验证 function-level generalization

---


## 8. Python interface (实施约定, v1 修订)

```python
from typing import Literal

# ---------------- cognitive function 是真正训练目标 ----------------

CognitiveFunction = Literal["first_reaction", "weighing", "bottom_line"]

# Q (数据 qid) → cognitive function 映射. Q1-Q6 训练用; Q7.segN 推理用.
# 同 function 的训练样本跟推理 prompt 用同一组 question + routing_format.
Q_TO_FUNCTION: dict[str, CognitiveFunction] = {
    "Q1":      "first_reaction",   # 训练 → 对应 Q7.seg1 推理
    "Q2":      "weighing",         # 训练 → 对应 Q7.seg2 推理
    "Q3":      "bottom_line",      # 训练 → 对应 Q7.seg3 推理 (4×)
    "Q4":      "bottom_line",
    "Q5":      "bottom_line",
    "Q6":      "bottom_line",
    "Q7.seg1": "first_reaction",
    "Q7.seg2": "weighing",
    "Q7.seg3": "bottom_line",
}

# LLM-facing 自然问题. 训练 + 推理共享, 同 function 同 wording.
COGNITIVE_FUNCTION_QUESTIONS: dict[CognitiveFunction, str] = {
    "first_reaction": "面对这个项目, 您的直觉第一反应是什么? 请直接说出您不经决策权衡的本能反应。",
    "weighing":       "请按您的优先级排出 PKVC 12 个 cardId 的序列, 并说明您的权衡逻辑 (先排序, 再 NL)。",
    "bottom_line":    "面对这个具体冲突, 您会怎么决策? 您向上层提出的最核心底线是什么? 哪些是不可让步的?",
}

# [认知路由标签] 输出格式说明, 跟 function 一一对应.
ROUTING_FORMAT_INSTRUCTIONS: dict[CognitiveFunction, str] = {
    "first_reaction": "在 [认知路由标签] 段输出**单字母 + verbose 标签**, 形如 `C (Concept 构念/构建)`. 表示本次第一反应主导的认知模块. 然后空行 + [NL 回答].",
    "weighing":       "在 [认知路由标签] 段输出 **PKVC 12 cardId 的优先级序列** (可变长度, 通常 4-12 项), 用 ` > ` 分隔, 每 cardId 后跟 verbose 含义, 形如 `P1 界定核心 > V3 设立原则 > K2 客观数据测试 > C2 整合构建 > P2 拆解子问题 > K1 调用经验`. **允许重复表示强偏好** (e.g. `K1 调用经验 > K3 评估技术栈 > K1 调用经验`), 不要求 12 项全覆盖. 然后空行 + [NL 回答] (说明权衡逻辑).",
    "bottom_line":    "在 [认知路由标签] 段输出 **PKVC 单字母** (e.g. `V (Value 价值判断)`) 或 **2 段路径** (e.g. `P3 推翻假定 → K3 评估技术栈`, 头 ≠ 尾, 上限 2 段). 表示底线决策的认知路径. 然后空行 + [NL 回答].",
}

# scenario_text 池: 训练 Q1-Q6 共用一个 (全感知效率空间项目), 评测 Q7 单独一个 (老城区超市)
SCENARIO_TEXT_POOL: dict[str, str] = {
    "training":   "...[全感知效率空间项目场景描述, ~400 字, 训练 Q1-Q6 共用]...",
    "evaluation": "...[老城区超市数字化改造场景描述, ~400 字, 评测 Q7 用]...",
}

# Q2 训练特殊设计: scenario_text 不含 user 答案 (D1 锁), 只 Q2 题面 + 12 cardId 池
# Q2_TRAINING_SCENARIO 由 build_q2_training_prompt 单独拼装
# (Q1 / Q3-Q6 用 SCENARIO_TEXT_POOL["training"], Q2 用 Q2 特殊 scenario)

# ---------------- baseline / 配置 ----------------

@dataclass
class RolePlayPromptConfig:
    profile: bool = False
    own_responses: bool = False        # 仅当训练 / 评测 user 不是本次的 generation target 时打开
    own_quads: bool = False             # 同上 (79 in-hand user 才有)
    peer_raw_nl: bool = False
    # system + scenario always on
    # NOTE: 训练 Q_n 时, own_responses / own_quads 必须 mask 掉 Q_n 自身 (防 leakage)


BASELINE_CONFIGS = {
    "B0": RolePlayPromptConfig(),
    "B1": RolePlayPromptConfig(profile=True),
    "B2": RolePlayPromptConfig(profile=True, peer_raw_nl=True),
    "B3": RolePlayPromptConfig(profile=True, own_quads=True),
    "B4": RolePlayPromptConfig(profile=True, own_responses=True, own_quads=True),
    "B5": RolePlayPromptConfig(profile=True, own_responses=True, own_quads=True, peer_raw_nl=True),
}

# ---------------- data classes ----------------

@dataclass
class OwnResponse:
    qid: str
    option_text_with_pkvc: str | None   # Q1/Q3-Q6/Q8: 选项文本 + PKVC 激活标签; Q2: None
    nl: str                              # Q1/Q3-Q6: expansion 优先, fallback text_supplement
                                          # Q2: q2_nl_refined.jsonl 的 refined_nl (100-300 字)
    cardids_ordered_verbose: str | None = None   # 仅 Q2: user 真实 partial+dup ranking 的简洁
                                                   # verbose, 可变长 4-20 项, 允许 dup. 
                                                   # 来源: data_v3/processed/q2_training_labels.jsonl
                                                   # 字段 ranking_verbose


@dataclass
class QuadRecord:
    qid: str
    s_primary: list[str]
    p_surface_verbose: str       # "forces_transition_to (Strong+ 强推转移)"
    o_primary_verbose: str       # "K (Knowledge 经验/数据/技术调用)"
    o_subtype_verbose: str       # "K2 客观数据测试"
    criterion: str | None


@dataclass
class TargetData:
    user_id: str
    profile_polarity_nl: str
    factor_scores: list[float]     # 内部 retrieval 用, 不进 prompt
    demographics: str
    own_responses: dict[str, OwnResponse] | None    # {qid: OwnResponse}, 含 Q2
    own_quads: dict[str, QuadRecord] | None         # {qid: quad}, None 若不在 79 池


@dataclass
class PeerData:
    user_id: str
    sim: float                      # 4-d factor cosine, 内部排序用
    q7_segs: dict[str, str] | None  # {seg1: ..., seg2: ..., seg3: ...}


# ---------------- helpers ----------------

def pkvc_verbose(letter_or_path: str) -> str:
    """单字母 / 编号 / 路径 → verbose 标签. 详 §0.1 标签表."""
    ...


def get_question(qid_or_seg: str) -> str:
    """Q1 / Q2 / Q3 / ... / Q7.seg1 / Q7.seg2 / Q7.seg3 → 自然问题."""
    return COGNITIVE_FUNCTION_QUESTIONS[Q_TO_FUNCTION[qid_or_seg]]


def get_routing_format(qid_or_seg: str) -> str:
    """同上, 拿 [认知路由标签] 输出格式说明."""
    return ROUTING_FORMAT_INSTRUCTIONS[Q_TO_FUNCTION[qid_or_seg]]


def build_prompt(
    config: RolePlayPromptConfig,
    target: TargetData,
    peers: list[PeerData],          # len 3, 已按 sim 降序
    scenario_text: str,
    qid_or_seg: str,                 # Q1 / Q2 / Q3 / Q4 / Q5 / Q6 / Q7.seg1 / Q7.seg2 / Q7.seg3
) -> tuple[str, str]:
    """Return (system_message, user_message).

    路径 A 已锁: 无 prior_segs, 每次调用独立; 3 段独立 stateless 调用.

    Args:
      scenario_text: 跟问题对应的场景描述 (训练用 SCENARIO_TEXT_POOL['training'],
                     评测用 SCENARIO_TEXT_POOL['evaluation']; Q2 训练特殊单独 build)
      qid_or_seg: 决定 scenario.question + scenario.routing_format_instruction
                  (通过 Q_TO_FUNCTION lookup)

    Returns:
      (system_msg, user_msg) for OpenAI chat completion API
    """
    ...


# ---------------- 训练数据构造 ----------------

def build_training_samples(target_users: list[str]) -> list[dict]:
    """为每个 user 构造 6 个训练样本 (Q1, Q2, Q3, Q4, Q5, Q6).

    每个样本 = {
      "prompt": build_prompt(...) 的 user_msg,
      "completion": "[认知路由标签]\n<...>\n\n[NL 回答]\n<user real NL>",
      "qid": qid, "cognitive_function": function, "user_id": uid,
    }

    LEAKAGE RULES:
      - 训练 Qn 样本时, own_responses 必须 mask 掉 Qn 本身 (只给其他 5 个 Q)
      - 训练 Qn 样本时, own_quads 必须 mask 掉 Qn 本身 (Q3-Q6 quad 各自 mask)
      - 训练 Q2 样本时, scenario_text 用 Q2 特殊 prompt (不含 user 答案), 不含 expansion
        explanation (D1, 防 leakage)

    PKVC 标签 (训练 target 的 [认知路由标签] 段) 来源:
      - Q1: 该 user 选项的 main_factor 单字母 verbose
            (e.g. "C (Concept 构念/构建)")
            数据源: data_v3/processed/q1q6_option_labels.jsonl (qid=Q1, 4 行)
      - Q2: 该 user 真实 partial+dup cardId ranking 的简洁 verbose 序列
            (可变长度 4-20 项, 允许重复)
            数据源: data_v3/processed/q2_training_labels.jsonl
                    字段 ranking_verbose, quality_tier in ('primary','fallback')
            sampler 默认只取 primary (115/163, validation ok=True)
      - Q3-Q6: 该 user 选项的 chain (e.g. "P3 推翻假定 → K3 评估技术栈") 或单字母
            数据源: data_v3/processed/q1q6_option_labels.jsonl (qid=Q3..Q6, 16 行)
            ambiguous primary (Q3-D / Q6-A) 拍板用 primary, eval 时 hit alt_labels 也算

    NL 段 (训练 target 的 [NL 回答] 段) 来源:
      - Q1, Q3-Q6: q1q6_expansion_outputs.jsonl 优先 (cell cover 28-39%),
                  fallback task_responses.text_supplement (median 25 字)
      - Q2:       q2_training_labels.jsonl 字段 refined_nl (100-400 字, 100% cover)
    """
    ...
```

**Fallback 行为**:
- toggle on 但数据缺 (e.g. `own_quads=True` 但 target 不在 79 池) → log warning, 该 section 跳过, **不报错**
- 输出 prompt 文件后 dump 到 `reports/<runid>/prompts/<user_id>__<qid_or_seg>__<baseline>.txt` 供审计
- 训练时 own_responses / own_quads 自动 mask 当前 Qn (防 leakage), 由 `build_training_samples` 负责

---

## 8.5 Q2 NL 优化 pipeline (训练数据预处理, SFT-critical)

**问题**: Q2 真实数据里 ranking (var-length cardId 序列) 跟 user 写的 text_supplement NL **几乎不对应**:
- ranking 是 cardId 序列 (可变长 4-20 项, N=163 median=8, 29% 含 dup)
- NL 短 (median 25 字 raw), 经常是 "题目好难" / "不知道怎么排" / 跟 cardId 无关的吐槽
- 直接拿这俩做 SFT target → 模型学到 (trait → ranking + 不解释 ranking 的乱 NL), 推理时 Q7.seg2 也输出**结构化 ranking + 跟 ranking 脱节的 NL**, eval 信号失真

**Solution**: **LLM 离线优化 NL**, 让 NL 真正解释 user 的 ranking 决策, **同时保留 user 原 NL 的语气 / 价值倾向**.

### Pipeline 结构

```
input:  (user_id, raw_ranking, raw_text_supplement, user_profile_polarity_NL)
   │
   ↓  [LLM optimizer prompt, 模型 = gpt-4o]
   ↓
output: refined_NL (100-300 字, 解释 ranking 决策 + 保留 user 语气)
   │
   ↓
存到: data/processed/q2_nl_refined.jsonl
   │
   ↓
SFT 训练 Q2 sample 的 target completion 用 refined_NL (不用 raw NL)
```

### LLM optimizer prompt

```
SYSTEM:
你是设计师认知风格分析专家. 任务: 把一个设计师写的"短随手 NL"重写成"解释她/他
PKVC 排序决策"的完整 NL, 严格保留她/他原 NL 的语气 + 用词偏好 + 价值倾向, 不要
加 LLM 通用套话.

铁律:
1. 输出必须**逐项呼应**给定的 PKVC ranking (尤其 top 3 + bottom 3), 说明为什么这
   样排
2. 保留原 NL 出现的所有用词 / 情绪 / 立场倾向 (如果 user 表达不耐烦, refined NL
   也应有不耐烦的痕迹; 不要洗成"我经过深思熟虑")
3. 保留原 NL 中明确的关键概念 (e.g. user 说 "底线", refined NL 必须含 "底线")
4. 100-300 字, 一段连续中文, 不要列表 / 标题
5. 第一人称, 设计师视角
6. 禁止编造 user 没有的事实 (e.g. user 没说"我有 5 年经验"就别加)

USER TEMPLATE:
请基于以下材料, 重写一段 NL 解释该设计师的 PKVC 排序决策.

[设计师信念画像 (帮你定调子, 不要直接 quote)]
{{profile_polarity_nl}}

[设计师真实的 PKVC 排序 (从最重要到最不重要)]
{{ranking_verbose}}
e.g. "P1 界定核心 > V3 设立原则 > K2 客观数据测试 > C2 整合构建 > ... (共 12)"

[设计师自己写的原始短 NL]
{{raw_text_supplement}}

输出: 一段 100-300 字 refined NL, 解释这个排序, 保留原 NL 语气.
```

### output validation (写到 jsonl 前)

| check | 规则 | fail 处理 |
|---|---|---|
| 长度 | 80-400 字 | 重抽 (temperature 0.5 → 0.7) |
| 包含 top-3 cardId 短词 | refined NL 必须提及 ranking top-3 的至少 2 个 verbose 名 | 重抽 |
| 包含 raw NL 关键词 | 如 raw NL 有 "底线" / "用户" / 关键 trait, refined 必须保留 | 重抽 |
| 没有 LLM 套话 | 不含 "需要综合考虑" / "balance" / "权衡多方面" | 重抽 (改 prompt 加 negative example) |

### 跑法 + 输出格式

```bash
.venv/bin/python -m sft_pipeline.data.q2_nl_refine \
    --model gpt-4o \
    --output data/processed/q2_nl_refined.jsonl
```

输出 schema:

```json
{
  "user_id": "abc123",
  "raw_text_supplement": "题目好难...感觉随便排了",
  "ranking_verbose": "P1 界定核心 > V3 设立原则 > K2 客观数据测试 > ...",
  "refined_nl": "我把 P1 界定核心放最前因为...; V3 设立原则次之是为了...; K2 客观数据测试主要用来验证...; 至于把 C3 提炼原则放最后, 是觉得设计师要先做事再总结; 至于'随便排'这个感觉, 其实是因为 12 个选项里有 4-5 个我都觉得差不多重要, 这是我惯常的纠结...",
  "validation": {"length_ok": true, "top3_covered": true, "voice_preserved": true},
  "optimizer_model": "gpt-4o",
  "refine_date": "2026-05-25"
}
```

### caveat (paper §4 必标)

1. **训练 target 是 LLM-refined NL, 不是 user 真 NL** — 严格说 model 学的是 "LLM 写
   作风格 conditioned on user profile + ranking", 不是纯人类 voice. **distribution
   drift 风险**. fallback: 跑 ablation B5_raw_NL (用 raw text_supplement 不优化), 比
   B5_refined_NL 看 lift, 看 refine 加多少
2. **优化前后 PKVC 标签不变** (ranking 不动, 只优化 NL), 标签信号无 drift
3. **冷启动覆盖**: 哪些 user 没有 raw NL? 应该多少? coverage 验证
4. **manual spot-check**: 取 10 个 refined NL 人工读, 看是否真保留 user 语气 (而不
   是变成 GPT 的"理性分析师"声音)

### 跟 framework v1 §4 OWN_RESPONSES 的对接

OWN_RESPONSES section 现在的 `own_responses.q2.nl` 字段, **应该读 refined_nl 而不
是 raw text_supplement** (在 Q2 优化跑完后). expansion outputs (q1q6_expansion) 是
Q1-Q6 的等价 pipeline, 已经跑过 309 段; Q2 是补充这个 pipeline.

---

## 9. 跟 SFT 训练 prompt 对齐

**关键**: 这个 framework 也是 SFT 训练 prompt 模板, 训练 / 推理同 schema 避免 distribution mismatch.

训练时:
- target = train fold user, ground truth = real Q7 seg1-3 NL
- target's own_responses / own_quads 必须 fold-aware filter (Q3-Q6 跟 Q7 不同 fold? 不需要, Q3-Q6 是训练源, Q7 是评测源, 不冲突)
- peer 从 train fold 选, 排除 self

推理 (eval) 时:
- target = test fold user
- 同样 prompt 结构, 但 peer 从 train fold 选 (fold-aware leakage 防御)

---

## 10. ablation 实验菜单 (paper §9 候选, v1 修订)

| 实验 | 用 baseline | 看什么 |
|---|---|---|
| **profile alone 信号** | B1 vs B0 | profile NL 自己有用吗 (现 -0.005 实测) |
| **peer raw-NL RAG 信号** | B2 vs B1 | peer Q7 NL 加多少 |
| **own quads value** | B3 vs B1 | 自己 Q3-Q6 路由 vs 无 quad |
| **own response value** | B4 vs B3 | 自己 Q1-Q6 NL 历史多加多少 |
| **full known-user max** | B5 vs B1 | 完全信息加多少 (天花板) |
| **per-segment vs joint** | B5 per-seg vs B5 joint | 一次输出 3 段 vs 分 3 次的输出质量差异 |
| **own_quads alone (no own_resp)** | B3 vs B1 | quad 结构化够不够, 不要 raw NL |
| **own_resp alone (no own_quads)** | custom B1+own_resp vs B1 | NL 历史够不够, 不要 quad |

## 11. 评测指标 (per-segment)

跟 §1 7 baseline × 3 segment × {real/auto teacher-forcing} 矩阵对应:

- **per-seg PKVC label** (v8 extractor 抽 generated seg, 跟 real seg 抽出来的比 pkvc_compare)
- **per-seg embed cosine** (text-embedding-3-small)
- **per-seg LLM-as-judge** (可选, 用 gpt-4o 比 generated vs real "哪个更像该 designer")
- **joint trajectory** (3 seg 拼起来跟 real 3 seg 拼一起 embed 对比)

main 看 per-seg 各项 lift 是否一致 (如果只 seg1 涨 seg2/3 不涨, 说明 RAG 只帮第一段, 不帮思考链).

---

## 12. 已 lock 的设计选择 (v1 修订, user 2026-05-25 confirm)

| # | 选择 | 决定 |
|---|---|---|
| 1 | 训练只跑 direction-q1q6, 砍 direction-q7 对照 | ✅ lock (user 明确不做 q7 训练) |
| 2 | mapping: Q1→seg1, Q2→seg2, Q3-Q6→seg3 (function-level) | ✅ lock |
| 3 | per-segment 3 次独立调用, 无 prior_segs (Path A) | ✅ lock |
| 4 | 3 段全结构化输出: `[认知路由标签] + [NL 回答]` | ✅ lock |
| 5 | 训练 PKVC 标签来源: 选项 metadata (Q1 单字母 / Q3-Q6 路径) + user real ranking (Q2 partial+dup, var-length 4-20); 详 v1.1 patch §1 | ✅ lock (v1.1 改) |
| 6 | Q7 评测 PKVC 标签来源: LLM 标注 (v8 extractor 抽 real Q7.segN) | ✅ lock |
| 7 | Q2 训练 scenario_text 不含 user 真答案 (D1, 防 leakage) | ✅ lock |
| 8 | own_responses / own_quads 训练 Qn 时 mask Qn 自身 (防 leakage) | ✅ lock |
| 9 | EFA factor scores 不进 LLM prompt (LLM 不懂数字), 仅内部 retrieval / SFT prefix | ✅ lock |
| 10 | PKVC 单字母 / 编号一律 verbose 化 (pkvc_verbose helper) | ✅ lock |
| 11 | own_responses NL 字段: expansion 优先 fallback text_supplement | ✅ default |
| 12 | peer_raw_nl 给 peer Q7 全 3 段 (~1500 字 per peer) | ✅ default |
| 13 | 撤掉 peer_quads section (跟 own_quads 重复) | ✅ lock |
| 14 | 撤掉 prior_segs / teacher-forcing autoregressive 选项 | ✅ lock |
| 15 | scenario_text: 训练 Q1-Q6 共用"全感知效率空间", 评测 Q7 用"老城区超市" | ✅ lock |
| 16 | Q2 训练 target NL = LLM-refined (gpt-4o 用 ranking + raw NL 生成解释 NL), 不用 raw text_supplement; 详 §8.5 pipeline + v1.1 patch §1 | ✅ lock (v1, v1.1 校准) |
| 17 | Q2 训练 [认知路由标签] = user real partial+dup ranking, 不强制 12 项 (N=163 实测 0 user 真排满 12 unique); fallback 池 (validation ok=False, 48/163) sampler 默认不取; 详 v1.1 patch §1 + data_v3/processed/q2_training_labels.jsonl | ✅ lock (v1.1 新) |
| 18 | Q7.seg2 评测 alignment: model 输出 var-length cardId → factor-collapse → 跟 v8.7 extractor 抽出来的 state_sequence (median=3, range 1-6) 用 Kendall-tau on common factors 比, q2_v8_7_ranking_eval.py 已实现 (top-1 96%, Jaccard 0.90) | ✅ lock (v1.1 新) |
| 19 | Q3-D / Q6-A ambiguous option PKVC primary 拍板 (Q3-D = P1→K2, Q6-A = V1), V2 / V3 入 alt_labels; eval 时 hit primary 或 alt 都算命中 | ✅ lock (v1.1 新) |
| 20 | data 路径全 propagate 到 data_v3/ (含 task_responses.csv, processed/*.jsonl); data/ 那份只留 q2_nl_refined.jsonl + q2_v8_7_ranking_eval.json 作 source-of-truth, 其余 deprecated | ✅ lock (v1.1 新) |
| 21 | **peer_raw_nl section 仅在 eval (推理) prompt 注入, 训练 prompt 强制 off** (train-eval schema asymmetric, 不是 bug). 原因: 训练 Q1-Q6 时塞 peer Q7 NL (≈ 2653 token) 引入 Q7-scenario distribution prior, model 注意力被 peer 占走 ~54% prompt, training target (Q1-Q6 NL ~80 token) 信号被淹. 同时 train-time 大量暴露 train fold Q7 NL → over-anchored. 移除 peer 后 max prompt token 从 6145 降到 3546, max_seq_len=4096 batch=32 spec 不动. 副作用: §10 ablation B2 vs B1 / B5 vs B1 改成 eval-time peer 加多少 lift, paper §10 重 frame | ✅ lock (v1.1 新, 2026-05-25 user 拍) |

**v1.1 状态**: 全部 design 选择已 lock, training data 就绪:
  - q1q6_option_labels.jsonl: 20 行 (Q1×4 + Q3-Q6×16, 含 2 ambiguous primary)
  - q2_training_labels.jsonl:  163 行 (115 primary + 48 fallback)
  - q1q6_expansion_outputs.jsonl: 309 段 (cell-level 28-39% cover, fallback text_supplement)

下一步: 实施 `scripts/roleplay_pilot_v2.py` + `sft_pipeline/data/prep_v2.py` (按 framework v1.1) → run 6 个 baseline × 10 user × 3 segment = 180 generation, 看新 schema 下 RAG mode 排名。
