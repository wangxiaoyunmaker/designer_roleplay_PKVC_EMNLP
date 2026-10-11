# Q1 训练格式 — profile 单选 (β+++ 隐含选项 + 三轨监督)

**版本**: β+++ (2026-05-23 锁), 取代 β++ (隐含选项 + dual-form)
**适用题号**: Q1
**题型**: 设计师 PKVC 主导自报告, 5 个候选选项, 单选
**关键设计** (β+++):
- ⚠ **Input 不暴露 MCQ 选项文本** (跟 Q7 评测形式对齐, 防止模型 anchor 学姐措辞)
- ⚠ **Output 三轨监督**: [推理 NL] + [决策签名] + [字母], 各自不同的 supervision target
- ⚠ **Dual-form 字段**: 结构化字段同时给自然语言 (主) + token (附), 利用 Qwen 预训练语义
- ⚠ **Peer 砍 decision NL**: 防 model 直接 copy peer 措辞

**含 C_local**: ❌ (profile 题, 无情境张力, prompt 不加 [Situation] 段)
**进 quad mining**: 部分 (作 designer profile metadata)

---

## 1. 题目背景

Q1 测设计师拿到新产品设计任务时的"第一反应主导因子"。**5 个选项不进 input**, 模型必须从 trait profile 自主推断 designer 在此场景下的主导 PKVC 状态 + 决策风格。

学姐 metadata 解锁后会给 "选项 → PKVC 字段" 完整映射, 当前从选项文本推测:

| 选项 (训练时不进 input, 仅作 supervision target) | 暗示主导 |
|---|---|
| A "形态先行 / 使用场景脑中浮现" | C (Concept) |
| B "梳理核心矛盾 / 厘清效率 vs 健康关系" | P (Problem) |
| C "要求数据 / 测试报告 / 拿事实说话" | K (Knowledge) |
| D (待 enumerate) | V 等 |
| E (待 enumerate) | ... |

---

## 2. Input 结构 (β+++ 砍选项)

```
<|system|>
你扮演设计师 {designer_id}. 阅读 task background 和当前题目, 结合自己的 trait
profile 和同簇 peer 模式, 输出 PKVC 认知分析、自然语言决策, 以及单字母答案.
第一人称推理 50-100 字.

<|prompt|>

[Task Background]                                            ← v1.1 C_global, 学姐 ⑤
{task_background_text}                                       ← 50-100 字, dataset 常量

[Trait Profile of {designer_id}]                             ← v1.1 s 双字段
- 因子组合 (s_secondary): {factor_pattern}                   ← e.g. (F2_共情关怀, +); (F1_问题驱动, +)
- 高分项 (s_primary top-5): {tag_list}
- Q2 cardId top-3 自报告: {q2_top3}                          ← 来自 Q2 方案 c
- 4 档极性 (legacy 兼容):
    强 +: {tags}
    弱 +: {tags}
    弱 -: {tags}
    强 -: {tags}

[Cluster Peer Examples] top-3 (s_primary Jaccard 排序)        ← β+++ 砍 decision NL
peer 1 — designer {peer_id} (Jaccard={score}, 共享: {shared_tags}):
   他的 trait factor: {peer_factor_pattern}
   Q1 cognitive routing:
      主导 (o_primary): {peer_primary_nl} ({peer_primary_token})    ← 双形式
      认知激活: {peer_sadr_nl} ({peer_sadr_token})
   (decision NL 不展示, 防 copy)

peer 2 — ...
peer 3 — ...

[Question Q1]
你拿到一个新产品设计任务后, 第一反应是什么? 描述你的 PKVC 主导认知和你会采取的
具体决策方向.                                                 ← ⚠ 不列 ABCDE 选项

<|response|>
```

**对比 β++ (砍掉的)**: 选项 A/B/C/D/E 文本不进 prompt。**对比 v3.5.1 (砍掉的)**: 4 档极性平铺被升级到 factor 组合 + top-5 + 4 档兼容。

---

## 3. Output 结构 (β+++ 三轨监督)

```
[PKVC 分析]                                                  ← weight α = 0.5
                                                              supervise from designer text_supplement (NL)
                                                              (designer voice fidelity)

认知路径: {自然语言路径描述}
         (路径: {token_path})                                  ← dual-form: e.g. "(路径: P → V)"

主导 (o_primary): {主导自然语言} ({字母})                       ← e.g. "问题界定 (P)"

子状态 (o_subtype): {子状态自然语言} ({cardId})                 ← 等学姐 ④, 解锁前可空
                                                              e.g. "界定核心底线 (P1)"

认知激活: {激活点自然语言1} ({sadr_token1}),
         {激活点自然语言2} ({sadr_token2})                     ← e.g. "用户福祉优先 (#V3+)"

推理: {designer 第一人称 NL, supervise from text_supplement}    ← 推理段, designer voice

置信度: {高|中|低} ({high|medium|low})

[/PKVC 分析]

[决策签名]                                                    ← weight β = 0.3
                                                              supervise from ground truth option text
                                                              (paraphrase × 3 + token dropout 10%)

{ground truth option text, 学姐措辞, action signature}

[/决策签名]

[答案]                                                        ← weight γ = 0.2
                                                              supervise from ground truth letter (CE loss)

{letter}

[/答案]
```

**Q1 注意点**:
- **不加 `[Situation]` 段** (profile 题无情境张力)
- 路径写法: Q1 单步, `o_primary` 多数是单字母, 极少 path (路径段可省)
- `o_subtype` 学姐 metadata 来前为空; 来后填 `cardId`
- `推理` 段是 designer voice, 哪怕只有 ≤10 字 (像 "直觉") 也保留, 不强制 padding
- `决策签名` token-level dropout 10% + 3 paraphrase 选 1 (`data/prep.py` augmentation)

---

## 4. Loss weights + augmentation 配置 (β+++)

```yaml
# sft_pipeline/config/config.yaml 新增
loss_weights:
  pkvc_analysis: 0.5      # alpha, designer voice (推理段)
  decision_signature: 0.3 # beta,  action signature (选项 NL)
  answer_letter: 0.2      # gamma, MCQ aux (字母 CE)

augmentation:
  decision_token_dropout: 0.10
  decision_paraphrase_n: 3    # 每个 option text 用 GPT-4o 生成 3 个语义等价 paraphrase
                              # 训练时 sample 一个作 target, 防 verbatim memorize
```

**ablation 必跑** (paper §6):
- (α, β, γ) sweep over [(0.5,0.3,0.2), (0.7,0.2,0.1), (0.3,0.4,0.3), (0.6,0.2,0.2)]
- 评价: Q7 cross-scenario A1 / A2 / A3 metric
- 选 Q7 metric 最高的组合作 main report

---

## 5. Peer 设计 (β+++ 防 leakage)

```
β++ 原始 peer 显示完整 decision NL → 风险: 模型直接 copy peer 措辞, 失去 trait→strategy 能力

β+++ peer 砍掉:
   ❌ peer.decision_nl (完整 100 字)        — 不展示
   ❌ peer.answer_letter                    — 不展示 (避免 ABCD 信号绕过 input 选项隐藏)
   
β+++ peer 保留:
   ✅ peer.designer_id + Jaccard score      — 出处可追溯
   ✅ peer.shared_high_score_tags           — 模型看到"为什么这个 peer 相似"
   ✅ peer.factor_pattern                   — trait 信号
   ✅ peer.o_primary (dual-form)            — 主导状态 quad 字段
   ✅ peer.sadr_activations (dual-form)     — 激活点 quad 字段
   (no p_surface for Q1 because Q1 单步无路由动作)

paper §3 写: "Peer examples expose abstract cognitive routing structure (quad fields)
            without verbatim decision text, forcing the model to internalize the
            trait-to-strategy mapping rather than copy surface NL."
```

---

## 6. 跟 v1.1 quad schema 的对应

```
input/output 的字段 ↔ TCAPSQuad dataclass:

input:
  [Trait Profile].factor + dir      → quad.s.s_secondary
  [Trait Profile].top-5 tags         → quad.s.s_primary

output:
  [PKVC 分析].o_primary              → quad.o_primary
  [PKVC 分析].o_subtype              → quad.o_subtype
  [决策签名]                          → quad 没对应字段 (是 supervision-only artifact)
  [答案] letter                       → quad 没对应字段 (MCQ aux)

→ Q1 训练目标主要在 PKVC 分析段 (生成 quad fields) + 决策签名段 (生成 strategy NL),
  字母段作 auxiliary letter CE.
→ Q1 不构成完整 quad (没 p / 没 C_local), 不进 quad mining DB.
→ Q1 只作 "trait → primary state mapping" 的 supervised signal.
```

---

## 7. 学姐 metadata 解锁前的占位 (现在能跑 fallback)

| 字段 | 解锁前 | 解锁后 |
|---|---|---|
| `task_background` | "N/A (待学姐 ⑤)" | 50-100 字正式背景 |
| `s_secondary.factor` 命名 | "F1+/F1-/F2+/F2-" 通用编号 | "F_共情关怀" 语义命名 |
| `o_subtype` | 字段省略 (不输出) | "界定核心底线 (P1)" 等 |
| `peer.sadr` | "#P1+" 等 cardId 占位 | "界定底线 (#P1+)" dual-form |
| `推理` (NL) | designer 原 NL (大多 ≤20 字) | 50-100 字 (等学姐 E 阻塞解) |
| `置信度` | 多 "低 (low)" (NL 短) | 多 "中 / 高" |
| `选项 → PKVC` 映射 | 推测 | 学姐校准 |

⚠ 解锁前可以训, 但 Q1 NL 字段大多 ≤20 字 ("直觉" / "市场检验为王道" 这种), 推理段 supervision 信号弱。**建议至少等 E 阻塞 (NL 补长) 或 ④ Q3-Q6 选项-因子 mapping 解一个再训**。

---

## 8. 数据样本统计 (基于当前 79 designer)

```
Q1 答案分布 (单选, distinct 5):
  A "形态先行 (C)"        35  (44.3%)
  B "矛盾梳理 (P)"        24  (30.4%)
  C "数据求证 (K)"        11  (13.9%)
  D ?                       8 (推测 V)
  E ?                       1

text_supplement (NL) 长度:
  ≤ 10 字  ("直觉" 这种):  多数 designer
  > 50 字:                极少 (D002 等 < 5 人)
```

**训练 sample 不均衡**: A 占近半, C/D/E 数据稀疏; 但这是 designer 真实分布, 不要 oversample (会破坏 trait distribution)。

---

## 9. 跟 prep.py 的对接

```python
def build_q1_input(designer_id, fold, codebook, peer_finder) -> str:
    """Q1 是 profile 题, 不加 [Situation] 段, 不暴露 5 个选项文本."""
    return assemble(
        system=SYSTEM_PROFILE_BETA3,                          # β+++ system prompt
        task_background=codebook.c_global.get("narrative_text", "N/A (待学姐 ⑤)"),
        trait_profile=build_trait_profile(designer_id),       # factor + top-5 + Q2 top-3 + 4档
        peers=peer_finder.top3_quad_safe(                     # β+++ 砍 NL + 砍 letter
            designer_id, fold, question="Q1",
            include_decision_nl=False,
            include_answer_letter=False,
        ),
        question=Q1_PROMPT_NO_OPTIONS,                        # ⚠ 不带 ABCDE 选项
    )

def build_q1_output(designer_id, question_id="Q1") -> dict:
    """返回 dict: {pkvc_analysis: str, decision_signature: str, answer: str}
    每段独立 supervise, loss weights 在 dataset.py 应用."""
    gt = load_ground_truth(designer_id, "Q1")
    nl = load_text_supplement(designer_id, "Q1")
    o_primary = infer_pkvc_from_option(gt.answer_value)      # A→C, B→P, C→K, ...
    paraphrase_pool = load_paraphrase_pool("Q1", gt.answer_letter)  # 3 paraphrase

    return {
        "pkvc_analysis": assemble_pkvc_block_dual_form(
            o_primary=o_primary,
            o_subtype=None,                                    # 等学姐 ④
            sadr_activations=[],
            reasoning=nl,
            confidence=confidence_from_nl_length(nl),
        ),
        "decision_signature": sample_paraphrase(paraphrase_pool),  # paraphrase × token dropout
        "answer": gt.answer_letter,
    }
```

---

## 10. 待拍 (Q1 单题级)

```
1. paraphrase pool 生成: 每个 option text × 3 paraphrase 用 GPT-4o 跑, $5 / 全部 Q1-Q6
2. token dropout 0.10 是否合理: 0.05 / 0.10 / 0.15 sweep
3. 推理 NL ≤10 字 designer 的样本是否 skip: 我推荐不 skip (保 trait 分布完整)
4. Q1 训作 SFT target 还是只作 input metadata: 我推荐训 (loss weight 1.0)
```
