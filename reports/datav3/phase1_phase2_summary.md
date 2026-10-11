# Phase 1 + Phase 2 完成报告

> 2026-05-25. Phase 1 = 163 designer 个人 quad extraction; Phase 2 = quad similarity + soft assignment.

## Phase 1: Individual quad extraction (163 designer × Q3-Q6)

**输入**: `data_v3/task_responses_expanded.csv` (163 designer × Q1-Q8)
**Prompt**: `sft_pipeline/config/prompts/quad_extractor_v4.md`
**Script**: `sft_pipeline.extract.quad_batch` + `quad_fix_and_retry`

**结果**:
- 652 jobs (163 × 4 = Q3-Q6 each designer), 469 直接 ok + 183 schema_fail auto-fixed = **652/652 ok (100%)**
- 跑时 212s, ~$5 GPT-4o
- 输出: **`data_v3/processed/quad_extracted_163_fixed.jsonl`** (652 quad)

**Quad 分布**:
- o_primary: K 197 / V 128 / P 79 / C 65 (4 类均衡)
- p_surface: gradually_shifts_to 208 / forces_transition_to 153 / compliantly_follows 105 / negotiates_around 3
- 4 个 C 情境 (跟 Q3-Q6 一一对应), 每个 163 个 quad

注: 183 schema_fail 都是 `o_primary` 跟 `o_subtype` 终点 prefix 不一致 (LLM 给 X→Y path 但 o_primary 选 X 不选 Y). `quad_fix_and_retry` 自动修复.

## Phase 2: Typology similarity + soft assignment

### 2a. Quad similarity 函数 (`sft_pipeline/rag/typology_match.py`)

`sim(individual_quad, typology_quad) ∈ [0, 1]`, 4 component 加权:
- α=0.40 × s_jaccard (trait 重合)
- β=0.25 × p_surface_match (路由表面 exact)
- γ=0.15 × p_modulation_compat (Strong+/Weak+ 等的 compatibility matrix)
- δ=0.20 × o_match (o_primary 主类 + o_subtype 子类层级匹配)

13 unit tests (per-component + integration + real data sanity), 全 pass.

### 2b. Soft assignment

**两种模式**:

| 模式 | 形式 | 用途 |
|---|---|---|
| Global | `designer → {typology_id → prob}`, 28-dim distribution | 单一 designer-level signature |
| **Per-qid** (推荐) | `designer → {qid → {typology_id → prob}}`, 4 个 distribution per designer | **CAPS If-Then framing** — 每个 situation 下的 typology activation |

**结果** (163 designer × 28 typology 候选, temperature=0.1):

| 模式 | top-1 median prob | sharp >0.5 占比 |
|---|---|---|
| Global | 0.22 | low |
| **Per-qid Q3** | **0.56** | 136/163 (83%) |
| **Per-qid Q4** | **0.69** | 159/163 (98%) |
| **Per-qid Q5** | **0.77** | 144/163 (88%) |
| Per-qid Q6 | 0.43 | 71/163 (44%) |

Per-qid 显著优于 global (CAPS 理论上也更对 — 每个 situation 一个 If-Then routing).

**最 popular typology (top-1 count per qid)**:
- Q3: Q3B-1 (60), Q3D-2 (43), Q3D-1 (41) — Q3 option B 最 popular (28/29 strong support)
- Q4: Q4A-1 (60), Q4D-1 (51), Q4C-1 (30)
- Q5: Q5D-1 (60), Q5B-1 (32), Q5B-2 (23)
- Q6: Q6A-1 (86) 占主导, Q6B-1 (43)

每个 (qid, option) cell 跟 expert candidate 的对应明确, 没有 mismatch 异常.

### 输出文件

| 文件 | 内容 |
|---|---|
| `data_v3/processed/quad_extracted_163_fixed.jsonl` | **652 individual quad** (Phase 1) |
| `data_v3/processed/typology_soft_assignment_per_qid.json` | **163 designer × 4 qid distribution** (推荐用) |
| `data_v3/processed/typology_soft_assignment_global.json` | 163 designer × global 28-dim distribution (对比用) |
| `sft_pipeline/rag/typology_match.py` | 核心模块 (~280 行) |
| `sft_pipeline/tests/test_typology_match.py` | 13 unit + integration tests |

## 下一步 (Phase 3 + 4 等专家 / 训练)

### Phase 3 — 等专家 (`expert_review_quad_candidates_v1.md` review 完成)

专家把 28 candidate 的 `[ ] 收 / [ ] 不收 / [ ] 待定` 勾完, 收的 ~20 个变成 final typology.

完成后, 跑:
```
.venv/bin/python -m sft_pipeline.rag.typology_match --only-accepted --per-qid \
    --output data_v3/processed/typology_assignment_final.json
```
就拿 final ~20 typology 的 distribution.

### Phase 4 — 接 SFT + RAG (final assignment 后)

1. **SFT condition**: 每个 designer 的 per-qid typology distribution 作 input embedding, model 学 "this designer 在 Q3 是 Q3B-1 型 (V→K 转向), Q5 是 Q5D-1 型 (...)"
2. **RAG retrieval**: target designer query → 找 same typology pool 内 top-3 peer NL 作 in-context
3. **Bridge mining**: 群体 level — typology × scenario → state machine 分布 (mapping_miner)

## 已知局限

1. **专家未 review 完**: 当前 28 candidates 全用作 mock typology. 专家收的部分可能比 28 少 (~20), 跑 `--only-accepted` 后 distribution sharpness 会上升 (typology pool 小 = softmax 更 peaked).

2. **Q6 sharpness 低 (43%)**: Q6 8 typology 区分细 (内部 p_surface/o_subtype 差异小), softmax 分散. 专家 prune 后应改善.

3. **s_primary 跨 designer 同 option 是常数** (因为来自 LASSO option-level): 相同 option 下 designer 的 s component 没区分力. 真正个体差异来自 p_surface, o_primary, o_subtype 三个 LLM 标的字段. 现在权重 α=0.40 (s) + δ=0.20 (o) + β=0.25 (p surface) + γ=0.15 (p mod) — 可能要 rebalance 后续 sweep.

4. **没用 confidence weighting**: 现 individual quad 的 confidence (high/medium/low) 没传入 similarity. 可后续加 weighting (low confidence 的 quad 贡献小).
