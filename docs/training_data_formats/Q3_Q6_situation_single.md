# Q3-Q6 训练格式 — 情境单选 (β+++ 隐含选项 + 完整 quad + dual-form)

**版本**: β+++ (2026-05-23 锁), 取代 β++ (隐含选项 + dual-form + 三轨监督)
**适用题号**: Q3, Q4, Q5, Q6
**题型**: 情境驱动单选, 4-5 选项, 每选项暗示一个 PKVC 路由
**关键设计** (β+++):
- ⚠ **Input 不暴露 MCQ 选项文本** (跟 Q7 评测形式对齐, 跨场景迁移最直接)
- ⚠ **Output 三轨监督**: [PKVC 分析] + [决策签名] + [字母]
- ⚠ **Dual-form 字段**: 结构化字段自然语言主 + token 附
- ⚠ **完整 quad 字段**: o_primary / o_subtype / p_surface / p_modulation / sadr 全输出
- ⚠ **Peer 砍 decision NL**: 暴露 quad fields 不暴露完整 NL

**含 C_local**: ✅ (学姐 5/23 给 Q3-Q6 4 个 tag)
**进 quad mining**: ✅ (主要训练 + mining 目标)

---

## 1. 题目背景

Q3-Q6 各自一个**情境张力** (跟 PKVC 字段对应), 设计师在 4-5 个选项里选反映他**在该情境下的认知路由**。每选项隐含 (o_primary, o_subtype, p_surface) 三元 — 学姐 ④ 校准后明确, 当前推测:

| 题号 | C_local tag | 张力 | 选项→PKVC 推测 |
|---|---|---|---|
| Q3 | `High_K_Utility_vs_Low_C_Aesthetics` | 数据效用 vs 美感 | A=K (数据论), B=K (质疑数据), C=C (美感优先?), D=K (市场测试) |
| Q4 | `High_P_Commercial_Profit_vs_High_V_Ethical_Risk` | 商业 vs 伦理 | A=K (合规), B=V (伦理底线), C=C (架构创新), D=K (拆解需求) |
| Q5 | `High_K_Automation_Efficiency_vs_Low_V_Authorship_Identity` | AI 效率 vs 作者性 | A=K (A/B 测试), B=P (搁置 AI), C=P (限制规范), D=V (人本深挖) |
| Q6 | `High_P_Authority_Mandate_vs_High_V_User_Harm` | 权威 vs 用户 | A=V (拒绝退出), B=V (暗中对抗), C=P (按时上线), D=P (完全服从) |

---

## 2. Input 结构 (β+++ 砍选项)

```
<|system|>
你扮演设计师 {designer_id}. 阅读 task background + 当前情境 + trait profile +
同簇 peer 模式, 输出 PKVC 认知路径、自然语言决策, 以及单字母答案.
第一人称推理 50-100 字.

<|prompt|>

[Task Background]                                            ← v1.1 C_global, 学姐 ⑤
{task_background_text}                                       ← 50-100 字

[Situation]                                                  ← v1.1 C_local 段, β+++ dual-form
情境: {c_local_nl_zh}                                        ← 中文释义, e.g. "高权威指令 vs 高用户伤害"
     (C_local: {c_local_tag})                                ← e.g. High_P_Authority_Mandate_vs_High_V_User_Harm
冲突类型: {polarity_nl} ({polarity_type})                     ← e.g. "双高冲突 (high_vs_high)"
张力两端: {left_field_nl} ({left_field}) vs {right_field_nl} ({right_field})
                                                              ← e.g. "权威指令 (P) vs 用户福祉 (V)"

[Trait Profile of {designer_id}]                             ← v1.1 s 双字段
- 因子组合 (s_secondary): {factor_pattern_with_polarity}
- 高分项 (s_primary top-5): {tag_list}
- Q2 cardId top-3 自报告: {q2_top3_dual}                     ← 来自 Q2 方案 c, dual-form
- 4 档极性 (legacy):
    强 +: ...
    弱 +: ...
    弱 -: ...
    强 -: ...

[Cluster Peer Examples] top-3 (s_primary Jaccard 排序)        ← β+++ 砍 decision NL
peer 1 — designer {peer_id} (Jaccard={score}, 共享: {shared_tags}):
   他的 trait factor: {peer_factor_pattern}
   {Qn} cognitive routing quad:
      认知路径: {peer_path_nl} ({peer_path_token})              ← e.g. "权威到价值跳转 (P → V)"
      主导 (o_primary): {peer_primary_nl} ({peer_primary_token})
      子状态 (o_subtype): {peer_subtype_nl} ({peer_subtype_token})  ← 等学姐 ④
      决策方式 (p_surface): {peer_surface_nl} ({peer_surface_token})
      调制 (p_modulation): {peer_modulation_nl} ({peer_modulation_token})
      认知激活: {peer_sadr_nl_list} ({peer_sadr_token_list})
   (decision NL / answer letter 不展示, 防 copy + leakage)

peer 2 — ...
peer 3 — ...

[Question {Qn}]
{题干, 不含 ABCDE 选项}                                       ← ⚠ 选项不进 input

<|response|>
```

**关键: 题干描述情境 + 决策点, 但不列 ABCDE 选项**。模型从 trait + situation 自主生成 strategy, 不靠 5 选 1。

---

## 3. Output 结构 (β+++ 三轨监督)

```
[PKVC 分析]                                                  ← weight α = 0.5
                                                              supervise from designer text_supplement
                                                              (designer voice fidelity)

情境识别: {c_local_nl_zh}                                     ← echo, e.g. "高权威指令 vs 高用户伤害"
         (C_local: {c_local_tag})

认知路径: {自然语言路径描述}                                    ← e.g. "我先被权威锚定, 价值反弹, 最后跳到 V"
         (路径: {token_path})                                  ← e.g. "(路径: V → P → V, bypass=True)"

主导 (o_primary): {主导自然语言} ({字母})                       ← e.g. "价值判断 (V)"

子状态 (o_subtype): {子状态自然语言} ({cardId})                 ← 等学姐 ④
                                                              e.g. "立绝对底线 (V3)"
                                                              fallback: 字段省略

决策方式 (p_surface): {决策方式自然语言} ({surface_token})       ← e.g. "协商式微调 (negotiates_around)"

调制 (p_modulation): {调制自然语言} ({modulation_token})         ← e.g. "弱促进 + 弱抑制 (Weak+/Weak-)"

认知激活: {激活点1_nl} ({sadr_token1}),
         {激活点2_nl} ({sadr_token2}),
         ...                                                  ← e.g. "用户福祉优先 (#V2+), 务实让步 (#P3+)"

推理: {designer 第一人称 NL, supervise from text_supplement}    ← designer voice, ≥50 字

置信度: {高|中|低} ({high|medium|low})                          ← NL 长度 derived

[/PKVC 分析]

[决策签名]                                                    ← weight β = 0.3
                                                              supervise from ground truth option text
                                                              (paraphrase × 3 + token dropout 10%)

{ground truth option text, action signature, 不带 token}

[/决策签名]

[答案]                                                        ← weight γ = 0.2
                                                              single letter CE

{letter}

[/答案]
```

**字段约束**:
- `c_local_tag` echo input (节省 budget + 保证闭集)
- `o_primary` ∈ {P, K, V, C}, dual-form: "价值判断 (V)"
- `o_subtype` ∈ 12 cardId 或 null (学姐 ④ 前), dual-form: "立绝对底线 (V3)"
- `o_subtype[0] == o_primary` (validator 强约束, 见 `rag/quad_schema.py`)
- `p_surface` ∈ 8 词闭集 (见 `data/processed/situation_codebook_v0.json` predicate_codebook)
- `p_modulation` 不让 LLM 直出, 由代码 `map_surface_to_modulation()` 查表
- `推理` 50-100 字 第一人称, designer voice (text_supplement)
- `决策签名` ≥ 50 字, action signature, 学姐 option text 风格 OK
- `letter` 单字母 ground truth, MCQ aux

---

## 4. Loss weights + augmentation (β+++)

```yaml
# sft_pipeline/config/config.yaml
loss_weights:
  pkvc_analysis: 0.5      # alpha, designer voice
  decision_signature: 0.3 # beta,  action signature
  answer_letter: 0.2      # gamma, MCQ aux

augmentation:
  decision_token_dropout: 0.10
  decision_paraphrase_n: 3
```

**ablation sweep** (paper §6 必跑):
- (α, β, γ) over [(0.5,0.3,0.2), (0.7,0.2,0.1), (0.3,0.4,0.3), (0.6,0.2,0.2)]
- metric: Q7 cross-scenario A1 / A2 / A3
- 报告 best 组合 + sensitivity

---

## 5. Peer 设计 (β+++ 防 leakage)

```
β++ 原始 peer: 显示完整 decision NL → 模型 copy NL 风险

β+++ peer 砍掉:
   ❌ peer.decision_signature_nl (完整选项文本)
   ❌ peer.answer_letter (字母绕过 input 选项隐藏)
   ❌ peer.reasoning_nl (designer 自述 NL, 可能含敏感个人风格)

β+++ peer 保留 (全 dual-form):
   ✅ peer.designer_id + Jaccard
   ✅ peer.shared_high_score_tags                              # 相似性出处
   ✅ peer.factor_pattern                                       # trait 信号
   ✅ peer.cognitive_path (dual-form)                           # 路径 quad
   ✅ peer.o_primary / o_subtype (dual-form)                    # 目标 PKVC 状态
   ✅ peer.p_surface / p_modulation (dual-form)                 # 路由谓语
   ✅ peer.sadr_activations (dual-form)                         # 激活点

paper §3 写: "Peer examples expose abstract cognitive routing as structured
            quad fields (with natural-language gloss) but not verbatim decision
            text. This forces the model to internalize the trait-to-strategy
            mapping rather than copy peer NL."
```

---

## 6. 跟 v1.1 quad schema 的对应

```
input/output 的字段 ↔ TCAPSQuad dataclass:

input:
  [Situation].c_local_tag           → quad.C
  [Trait Profile].factor + dir       → quad.s.s_secondary
  [Trait Profile].top-5 tags         → quad.s.s_primary

output:
  [PKVC 分析].o_primary (dual-form) → quad.o_primary
  [PKVC 分析].o_subtype (dual-form) → quad.o_subtype
  [PKVC 分析].p_surface (dual-form) → quad.p.p_surface
  [PKVC 分析].p_modulation (dual-form) → quad.p.p_modulation (含 bypass)
  [决策签名]                          → quad 无对应 (supervision-only artifact)
  [答案] letter                       → quad 无对应 (MCQ aux)

→ Q3-Q6 训练完成后, 模型在 hold-out fold 上的 [PKVC 分析] 段 output
  可直接 parse 成 TCAPSQuad → 灌 mapping_miner.py 跑 distribution shift permutation test.
```

---

## 7. 学姐 metadata 解锁前的占位 (现在能跑 fallback)

| 字段 | 解锁前 | 解锁后 |
|---|---|---|
| `task_background` | "N/A (待学姐 ⑤)" | 50-100 字 |
| `c_local_tag` | Q3-Q6 已有 (学姐 5/23) | 不变 |
| `c_local_nl_zh` | 学姐 5/23 给的中文释义 | 不变 |
| `o_subtype` (dual-form) | 字段省略 | "立绝对底线 (V3)" 等 |
| `peer.o_subtype` (dual-form) | 字段省略 | 同 |
| `推理` (NL) | text_supplement 原文 (大多短) | 50-100 字 (等 E 阻塞) |
| `决策签名` paraphrase | option text 原版 (无 paraphrase) | option text + 3 paraphrase |
| `置信度` | 多 "低" | 多 "中 / 高" |
| `选项 → PKVC` 映射 | 推测 | 学姐校准 (④) |
| `p_modulation_nl` | "强促进 (Strong+)" 这种字面翻译 | 学姐 confirm 是否扩到 10 词 |

⚠ 解锁前能跑, 但 [推理] 段 supervision 弱 (NL 短) + [决策签名] 无 paraphrase → 训出来易过拟合到学姐措辞。**强烈建议**至少等 E 阻塞 (NL 补长) 解了再训。

---

## 8. 数据样本统计

```
Q3 (4 distinct): D 41 (52%), B 29 (37%), A 6, C 3
Q4 (4 distinct): A 37 (47%), D 28 (35%), C 12, B 2
Q5 (5 distinct): D 41 (52%), B 14, A 14, C 9, A_var 1
Q6 (4 distinct): C 33 (42%), B 30 (38%), A 13, D 3

NL 字数分布 (text_supplement):
  ≤ 10 字 ("市场检验为王道"):  ~70 designer
  50-100 字:                  ~5 designer (D002 等)
  > 100 字:                   ~3 designer
```

→ NL 长 designer 数据稀缺是 E 阻塞核心痛点。β+++ 训练 NL 不够长时, [推理] 段权重 0.5 但实际信号被压缩。

---

## 9. 跟 prep.py 的对接

```python
SITUATION_QIDS = {"Q3", "Q4", "Q5", "Q6"}

def build_situation_input(designer_id, question_id, fold, codebook, peer_finder) -> str:
    c_local_entry = codebook.c_local_by_question[question_id]
    return assemble(
        system=SYSTEM_SITUATION_BETA3,
        task_background=codebook.c_global.get("narrative_text", "N/A (待学姐 ⑤)"),
        situation=build_situation_section_dual_form(c_local_entry),
        trait_profile=build_trait_profile(designer_id),
        peers=peer_finder.top3_quad_safe(                       # β+++ 防 leakage
            designer_id, fold, question_id,
            include_decision_nl=False,
            include_answer_letter=False,
            include_reasoning_nl=False,
            dual_form=True,
        ),
        question=load_question_text_no_options(question_id),     # ⚠ 不带 ABCDE
    )

def build_situation_output(designer_id, question_id) -> dict:
    """三轨监督, 各段独立返回, dataset.py 应用 loss weights."""
    gt = load_ground_truth(designer_id, question_id)
    nl = load_text_supplement(designer_id, question_id)
    quad = synthesize_quad(designer_id, question_id, gt, nl)     # extractor 或人工
    paraphrase_pool = load_paraphrase_pool(question_id, gt.answer_letter)

    return {
        "pkvc_analysis": assemble_pkvc_block_dual_form(quad, reasoning=nl),
        "decision_signature": sample_paraphrase(paraphrase_pool),
        "answer": gt.answer_letter,
    }
```

---

## 10. 待拍 (Q3-Q6 题级)

```
1. p_surface 选项 → ground truth p_surface 怎么定:
   (a) LLM extractor 跑 (extractor_v3.md)
   (b) 学姐 ④ 手动标注 17 个 option × p_surface
   (c) (a) + (b) 并行算 κ
   我推 c.

2. p_modulation_nl 翻译:
   "Strong+" → "强促进" / "强烈推动" / "强 trigger"?
   建议跟学姐确认中文表达, 锁 codebook short_nl 字段.

3. o_subtype 训练时不输出还是输出 null:
   - 不输出 (字段省略): 模型不学这字段, 学姐 ④ 来了再启用
   - 输出 "null": 模型学到 "null" 字符串占位
   我推第 1 个 (字段省略), 避免学到 null 字面.

4. 决策签名 paraphrase 是否含 negative samples:
   只用 ground truth option 的 paraphrase (positive only)
   还是 mix 其他选项的 paraphrase 作 negative contrastive?
   v1.1 不引入 contrastive, 走 positive only.

5. peer 数量 top-3 vs top-2:
   token 预算紧时降到 top-2 (节省 ~25% input length)
   信号损失: 第 3 个 peer Jaccard 通常 < 0.6, 影响小
   默认 top-3, 紧张时 fallback top-2
```
