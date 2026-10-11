# Q2 训练格式 — cardId 优先级排序 (β+++ 适配)

**版本**: β+++ (2026-05-23)
**适用题号**: Q2
**题型**: 12 cardId 优先级排序 (designer 自定义 ordering)
**output 形态**: JSON array (cardId + order), 79 distinct answers
**含 C_local**: ❌ (profile 题)
**进 quad mining**: ❌ (作 designer profile metadata)

---

## 0. 总建议 (β+++ 不变): **方案 c (不训 Q2 作 SFT target)**

```
Q2 不是 MCQ, β+++ 隐含选项 + dual-form 升级对 Q2 影响小. 整体决策仍走方案 c.

理由 (回顾):
1. Q2 信号已被 30 维量表 + EFA factor scores 间接编码
2. 训 ranking 需要 pairwise / listwise loss head, 跟 Q1/Q3-Q6 单选 head 异构
3. Q2 有 79 distinct 答案, ranking 训练目标"学一个 designer-specific permutation" 易过拟合
4. 时间 ROI: 不如投到 Q3-Q6 quad mining
```

**方案 c 实操**: Q2 不出 output template, 只把 Q2 cardId top-3 ranking **进 input** 作 trait profile 的扩展段 (见 §3)。

下面 §4-§6 是"若反悔走方案 a/b 训 Q2"的备用 schema, 已升级到 β+++ dual-form。

---

## 3. Q2 信息怎么进其他题的 input (方案 c 主线)

```python
# sft_pipeline/data/prep.py
def build_trait_profile_section(designer_id) -> str:
    q2_data = load_q2_answer(designer_id)               # JSON list of cardId rankings
    q2_top3_cardid = [item["cardId"] for item in q2_data[:3]]
    q2_top3_nl = [load_cardid_nl(cid) for cid in q2_top3_cardid]  # 自然语言描述

    # β+++ dual-form: 自然语言 (主) + cardId token (附)
    q2_top3_dual = [f"{nl} ({cid})" for nl, cid in zip(q2_top3_nl, q2_top3_cardid)]

    return (
        f"[Trait Profile of {designer_id}]\n"
        f"- 因子组合 (s_secondary): {get_factor_pattern(designer_id)}\n"
        f"- 高分项 (s_primary top-5): {get_top5_high_score_tags(designer_id)}\n"
        f"- Q2 cardId top-3 自报告: {q2_top3_dual}\n"   # ← β+++ dual-form
        f"- 4 档极性: ...\n"
    )

# Q2 cardId top-3 自报告 示例输出:
#   [界定核心底线 (P1), 使用客观数据 (K2), 立绝对底线 (V3)]
#   学姐 metadata 解锁前 fallback:
#   [(P1), (K2), (V3)]   ← 只有 token, 没自然语言, 等学姐 ③
```

⚠ Q2 cardId 的自然语言描述要等学姐 ③ 12 cardId metadata 回了才有, 当前 fallback 用 cardId token only。

---

## 4. (备用方案 a/b) 若坚持训 Q2 — β+++ Input 结构

```
<|system|>
你扮演设计师 {designer_id}. 阅读 12 cardId 候选 + 你的 trait profile,
按你的认知优先级输出 cardId 排序. 第一人称推理 50-100 字.

<|prompt|>

[Task Background]                                            ← 同 Q1
{task_background_text}

[Trait Profile]                                              ← 同 Q1, 但 *不* 包含 Q2 ranking (避免自指)
- 因子组合 (s_secondary): {factor_pattern}
- 高分项 (s_primary top-5): {tag_list}
- 4 档极性: ... (同 Q1)

[Cluster Peer Examples] top-3 (β+++ 砍 decision NL)
peer 1 — designer {peer_id} (Jaccard={score}):
   他的 trait factor: {peer_factor_pattern}
   Q2 cardId top-5 自报告: [界定核心底线 (P1), 拆解子问题 (P2), 使用客观数据 (K2),
                          模糊形态愿景 (C1), 立绝对底线 (V3)]            ← dual-form

peer 2 — ...
peer 3 — ...

[12 cardId 候选列表]  ⚠ dual-form, 自然语言 + cardId token
1. 界定项目核心底线和真实问题 (P1)
2. 把复杂总委托拆解为子问题 (P2)
3. (TBD 学姐) (P3)
4. 调用经验 / 案例 / 行业规范 (K1)
5. 使用客观数据 / 测试报告 (K2)
6. 评估技术 / 材料 / 工具边界 (K3)
7. (TBD 学姐) (V1)
8. 多方利益冲突时排优先级 (V2)
9. 立绝对不可牺牲的底线 (V3)
10. 模糊但强烈的形态愿景 (C1)
11. 把零散观察整合为核心设计逻辑 (C2)
12. 提炼可复用的设计原则 (C3)

[Question Q2]
拿到一个复杂设计任务时, 你认为以上 12 个认知活动的重要性优先级是什么?
按你的真实习惯, 从最重要到次要选出 top-3.

<|response|>
```

---

## 5. (备用方案 b) Output 结构 — β+++ 三轨监督 (top-3 set)

```
[PKVC 分析]                                                  ← weight α = 0.5
                                                              supervise from designer text_supplement

推理: {designer NL, 50-100 字, 描述选 top-3 的认知逻辑}

置信度: {高|中|低} ({high|medium|low})

[/PKVC 分析]

[决策签名]                                                    ← weight β = 0.3
                                                              supervise from ground truth Q2 top-3 cardId 文本
                                                              (token-level dropout 10%)

我会优先 {cardId1_nl}, 然后是 {cardId2_nl}, 最后是 {cardId3_nl}.
具体地, 这种顺序反映了我先 {理由1}, 再 {理由2}, 最后 {理由3} 的认知习惯.
                                                              (paraphrase × 3 if available)

[/决策签名]

[答案]                                                        ← weight γ = 0.2
                                                              top-3 cardId set (multi-label binary CE)

[界定核心底线 (P1), 使用客观数据 (K2), 立绝对底线 (V3)]

[/答案]
```

(若走方案 a 完整 ranking, [答案] 段改成 12 cardId full permutation list + ListMLE loss)

---

## 6. 评估指标 (若训, 备用)

```
方案 a (完整 ranking):
  - Spearman ρ (predicted vs gt ranking)
  - NDCG@3, NDCG@5

方案 b (top-3 set):
  - Precision@3, Recall@3
  - Jaccard(predicted_set, gt_set)
  - + decision signature BLEU vs ground truth NL (auxiliary)
```

---

## 7. Q2 真值数据格式 (现成)

`data/task_responses.csv` Q2 行:

```json
[
  {"cardId": "C2", "label": "C2", "text": "把零散观察整合为一个统摄全局的核心设计逻辑", "order": 0},
  {"cardId": "P1", "label": "P1", "text": "界定这个项目的核心底线和真实要解决的问题", "order": 1},
  {"cardId": "K2", "label": "K2", "text": "使用客观数据、测试报告来证明某个逻辑", "order": 2},
  ...
]
```

每个 designer 1 个 list, 包含 12 cardId 的 ordering (full permutation 或 subset)。

注意 `text` 字段已经是自然语言描述, 训练时直接用 (不用等学姐 ③), 但语义清晰度不及学姐 metadata 给的官方定义。

---

## 8. 待拍 (Q2 单题级)

```
1. 方案 a / b / c 选哪个: 我推荐 c (Q2 不训, 信息进 input top-3 dual-form)
2. 若走 c, Q2 进 input top-3 / top-5 / full: 我推 top-3 (token 预算 + 主要信号在 top)
3. Q2 ranking 跟 EFA factor pattern 相关性 sanity:
   - 计算 Spearman(Q2_ranking, factor_scores_projection)
   - > 0.6 → Q2 已被 factor 包含, 不训无损
   - < 0.4 → Q2 信号独立, 考虑走方案 b 补回
```

---

## 9. 跟 v1.1 quad schema 的关系

```
Q2 不形成 quad. 它是 designer profile 的 self-report 维度,
跟 30 维量表 / EFA factor / 4 档极性 平行, 都属 trait layer.

若 §8 第 3 项 sanity 显示 Q2 跟 factor 高度相关, Q2 可以彻底从 input 移除,
减少 token 预算 (节省 ~50 token / sample). 这是 paper §6 一个潜在 ablation.
```

---

## 10. β+++ 对 Q2 的影响汇总

```
Input 段 (方案 c 主线):
   ✅ Q2 信息进 input 时升级到 dual-form (自然语言 + cardId token 双形式)
   ⚠ 等学姐 ③ 12 cardId 语义解锁前, fallback 用 cardId token only

Input 段 (方案 a/b 备用, 若训):
   ✅ 12 cardId 候选列表升级到 dual-form
   ✅ peer Q2 ranking 升级到 dual-form
   ✅ peer 砍 decision NL (跟 Q1 / Q3-Q6 一致)

Output 段 (方案 b 备用, 若训):
   ✅ 三轨监督: [PKVC 分析] + [决策签名] + [top-3 set]
   ✅ Loss weights 跟 Q1 / Q3-Q6 共用 alpha/beta/gamma 配置

Output 段 (方案 c 主线):
   N/A (Q2 不训)
```
