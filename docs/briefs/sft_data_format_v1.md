# SFT 训练数据格式设计 v1.1

> 2026-05-25 落盘 + 同日 user 反馈修订: 砍 ④ peer + 加 Q1/Q2 进训 + 砍 cluster 概念 + reasoning_notes 已 pre-label 不重跑.
> 基于 `docs/briefs/prompt_v2_contrastive_diverse_v1.md` (B6 inference prompt).
> 训练目标: LoRA fine-tune Qwen2.5-7B 模拟设计师 PKVC trajectory + NL 生成.
> 评测目标: Q7 3 seg (held-out, paper main eval).

## 0. 一句话

SFT 在 **Q1-Q6 全部** (Q7 严格留 eval) 训练; Q1 ~ Q7_seg1 同构 (单字母), Q2 ~ Q7_seg2 同构 (sequence), Q3-Q6 ~ Q7_seg3 同构 (dom + nl). input prompt 复用 B6 模块**但砍 ④ peer + ⑥ criterion**, 严格做 **leave-one-out** 防 leakage.

## 1. 训练 scope

### 1.1 数据可用性 (修正)

| 问题 | 数据形态 | Q7 对应 seg | SFT 训练性 |
|---|---|---|---|
| **Q1** | 单字母 PKVC 第一反应 + supplement NL | seg1 (single dominant) | ✅ **训** (同构) |
| **Q2** | designer 在 Q1 设计问题上的**思考链路** (cardId sequence, ~3-7 cards) + supplement | seg2 (sequence) | ✅ **训** (**1:1 mirror seg2**) |
| **Q3-Q6** | 设计冲突场景 + 4 选项 ABCD + supplement NL | seg3 (dom + nl) | ✅ **训** (option_trait_mapping.json PKVC 标) |
| Q7 | 3 seg 长 NL (open-ended) | — | ❌ **留 eval, 严禁进训练 input** |

**关键洞察**: Q1/Q2/Q3-Q6 跟 Q7 三个 seg **一一同构**. 训 Q1-Q6 = 隐性训 Q7 三种段类型的 routing pattern.

### 1.2 训练集规模

- 162 designer × **6 question** (Q1 + Q2 + Q3-Q6) = **972 examples**
- partial 数据 designer 单独处理, 可能 ~960 有效
- Train / val split: 80/20 random by (designer, question) → **~770 train + ~200 val**
- 真 test: Q7 162 × 3 seg = 486 (paper main eval)

## 2. Output target schema (3 个 task-type variant)

按题型分 3 个 schema, 共享 `supplement_nl` + `reasoning_notes` + `task_type` 字段. 训练时 model 看 task_type 决定生成哪几个字段.

### 2.1 Q1 schema (~ Q7 seg1, single dominant)

```python
Q1_SCHEMA = {
    "task_type": "Q1",                                                    # 显式标记
    "chosen_letter": {"type": "string", "enum": ["P","K","V","C"]},       # 第一反应单字母
    "supplement_nl": {"type": "string", "minLength": 10, "maxLength": 200},
    "dominant_factor": {"type": "string", "pattern": r"^[PKVC]$"},        # = chosen_letter
    "state_sequence": {                                                    # 长度=1, 跟 dominant 一致
        "type": "array", "items": {"enum": ["P","K","V","C"]}, "minItems": 1, "maxItems": 1,
    },
    "reasoning_notes": {"type": "string", "maxLength": 100},
}
```

### 2.2 Q2 schema (~ Q7 seg2, **1:1 mirror sequence**)

**澄清**: Q2 不是 12 cardId 优先级**全排序**, 而是 designer 在 Q1 设计问题上的**实际思考链路** — 即 designer 自报的 cardId sequence (长度 ~3-7, 表示思考从哪个 state 走到哪个 state). 这跟 Q7 seg2 的 state_sequence 几乎是 1:1 mirror.

```python
Q2_SCHEMA = {
    "task_type": "Q2",
    "thinking_chain": {                                                    # designer 思考链路 cardId 序列
        "type": "array",
        "items": {"enum": ["P1","P2","P3","K1","K2","K3","V1","V2","V3","C1","C2","C3"]},
        "minItems": 2, "maxItems": 7,                                       # 实际链路长度, 非全 12
    },
    "supplement_nl": {"type": "string", "minLength": 10, "maxLength": 200},
    "state_sequence": {                                                    # cardId → PKVC letter sequence
        "type": "array", "items": {"enum": ["P","K","V","C"]},
        "minItems": 2, "maxItems": 7,                                       # 直接 letter-strip thinking_chain
    },
    "reasoning_notes": {"type": "string", "maxLength": 100},
}
```

Note: `state_sequence` 由 `thinking_chain` 直接派生 (e.g., `["P3","K2","V1"]` → `["P","K","V"]`), 用于 SFT 阶段对齐 Q7 seg2 的 letter sequence output.

### 2.3 Q3-Q6 schema (~ Q7 seg3, dom + nl)

```python
Q3Q6_SCHEMA = {
    "task_type": "Q3"|"Q4"|"Q5"|"Q6",
    "chosen_option": {"type": "string", "enum": ["A","B","C","D"]},
    "supplement_nl": {"type": "string", "minLength": 10, "maxLength": 400},
    "dominant_factor": {"type": "string", "pattern": r"^[PKVC]$"},
    "state_sequence": {                                                    # subtype 解析: "P3→K3" → ["P","K"]
        "type": "array", "items": {"enum": ["P","K","V","C"]},
        "minItems": 1, "maxItems": 3,
    },
    "o_subtype": {"type": "string"},                                       # e.g. "K2" 或 "P3→K3"
    "reasoning_notes": {"type": "string", "maxLength": 100},
}
```

### 2.4 GT 字段来源

| 字段 | Q1 | Q2 | Q3-Q6 |
|---|---|---|---|
| chosen_letter / thinking_chain / chosen_option | task_responses.csv Q1 answer | task_responses.csv Q2 **思考链路 cardId 序列** | task_responses.csv Q{i} option |
| supplement_nl | **q1q6_extracted_v8.7_full.jsonl cleaned NL** (Q1) | **q1q6_extracted cleaned NL** (Q2) | **q1q6_extracted cleaned NL** (Q{i}) |
| dominant_factor | = Q1 letter | — | option_trait_mapping.json `o_primary_hint` |
| state_sequence | [Q1 letter] | letter-strip(thinking_chain) | parse_subtype(o_subtype_hint) |
| o_subtype | — | — | option_trait_mapping.json `o_subtype_hint` |
| reasoning_notes | **已 pre-label (cached)** | **已 pre-label** | **已 pre-label** |

⚠ **supplement_nl 用 extractor 输出的 cleaned 版**: v1 用户原始 text_supplement 跑出来质量差 (typo / 半截句 / 跑题). v8.7 extractor 调用时已让 model rewrite / normalize 一次, 输出在 `q1q6_extracted_v8.7_full.jsonl` 的 `nl_cleaned` 字段. SFT target 用 cleaned 版作 supervision signal, 比 raw 干净.

### 2.5 真实样本

**Q1 (BuzhichunD)**:
```json
{
  "task_type": "Q1",
  "chosen_letter": "P",
  "supplement_nl": "第一反应是要质疑这个题目本身, 设计师不应该被框住.",
  "dominant_factor": "P",
  "state_sequence": ["P"],
  "reasoning_notes": "完美主义 + 边缘群体优先 trait 驱动质疑思维, 第一反应进 P 类问题界定."
}
```

**Q2 (某 designer 在 Q1 设计问题上的思考链路)**:
```json
{
  "task_type": "Q2",
  "thinking_chain": ["P3","K2","V1","C2"],
  "supplement_nl": "先质疑设计部门要管的范围, 再调研数据看用户真实需求, 然后亮明用户优先底线, 最后整合方案.",
  "state_sequence": ["P","K","V","C"],
  "reasoning_notes": "完美主义 + 边缘群体优先 trait → P3 推翻假定开头, 再走 K2 调研客观数据, V1 立用户优先, C2 整合."
}
```

**Q3-Q6 (BuzhichunD 选 Q3-B)**:
```json
{
  "task_type": "Q3",
  "chosen_option": "B",
  "supplement_nl": "原则上, 要解决问题的功能是最重要的, 但是有更美观符合产品调性的外观更好...",
  "dominant_factor": "K",
  "state_sequence": ["P", "K"],
  "o_subtype": "P3→K3",
  "reasoning_notes": "完美主义 (β=+0.16) + 边缘群体优先 → 先质疑数据 (P3) 再要求工程给真实差异报告 (K3)."
}
```

## 3. Input prompt structure (B6 adapted, **砍 ④ peer + ⑥ criterion**)

### 3.1 模块清单 (5 个, 砍 2 个)

| Module | B6 inference (target=Q7) | SFT (target=Q1/Q2/Q3-Q6) |
|---|---|---|
| PKVC 4 字母含义 | ✓ | ✓ 不动 |
| ① pi (5 字段) | ✓ | ✓ 不动 |
| ② 30维 stance labels (strong only) | ✓ | ✓ 不动 (trait 静态, 无 leak) |
| ③ self Q1-Q6 (raw behavior) | 4 个 (Q3-Q6 全展示) | **mask target Q** (训 Qi 时排除 Qi) |
| ⑤' CAPS quad (4 类平衡) | 从全部 quad 池选 | **mask target Q** (filter qid==target) |
| ~~④' contrastive peer~~ | (B6 有) | **❌ 砍 — 没有簇 / peer 不必要** |
| ~~⑥ criterion library~~ | seg3 only | **❌ 砍 — user confirm** |
| 任务 + output schema | Q7 per-seg structured | **task-type variant (Q1/Q2/Q3-Q6)** |

### 3.2 ③ self Q1-Q6 (mask target) 示例

**重要**: 块内 NL 一律用 v8.7 extractor cleaned 版 (`q1q6_extracted_v8.7_full.jsonl[nl_cleaned]`), 不用 raw text_supplement (v1 跑下来发现 raw 太 noisy).

#### 训 Q1 时, ③ 展示 Q2-Q6:
```
你在 Q2-Q6 (其他 5 道题) 的实际行为:

Q2 思考链路: P3 → K2 → V1 → C2
  你的解释 (cleaned): "先质疑设计部门要管的范围, 再调研数据看用户真实需求, 然后亮明用户优先底线, 最后整合方案."

Q3 面对【数据效用与美感形态冲突】...
  选择: "..."
  你的解释 (cleaned): "..."

Q4 面对... / Q5 面对... / Q6 面对...
```

#### 训 Q3 时, ③ 展示 Q1, Q2, Q4-Q6:
```
你在 Q1, Q2, Q4-Q6 的实际行为:

Q1 第一反应单字母: P
  你的解释 (cleaned): "第一反应质疑题目本身, 设计师不应被框住."

Q2 思考链路: P3 → K2 → V1 → C2
  你的解释 (cleaned): "..."

Q4 面对... / Q5 面对... / Q6 面对...
```

### 3.3 ⑤' CAPS quad (mask target Q) 示例

CAPS quad 池基于 Q3-Q6 的 option_trait_mapping 算 (Q1/Q2 没有 CAPS quad). 训 Q3 时 filter qid=="Q3-*":

```
你的认知路由模式 (CAPS 四元组, 4 类平衡, 排除 target Q routing)

## P 类 top routing
[Q4-D] 在【商业利润与用户隐私冲突】... β=+0.759 → P3→K3

## K 类 top routing
[Q5-A] 在【AI 工具与设计师自主性冲突】... β=+0.32 → K2

## V 类 top routing
[Q6-A] ...

## C 类 top routing
[Q4-C] ...
```

注: 训 Q1/Q2 时不需 mask (CAPS quad pool 本来就不含 Q1/Q2 quad), 全部 4 类 top 直接展示.

### 3.4 任务 + output schema (按 task_type 分支)

```
# 场景 (训练 target = Q3 — 数据效用 vs 美感冲突)
[Q3 场景题面, 长度 ~200 字]

# 任务
请基于你的人口统计学 + 立场 + 行为数据 + CAPS 路由模式, 输出:
1. 你会选 ABCD 哪个 option
2. 你的 supplement (简评) NL, 50-200 字, 第一人称
3. 该 option 对应的 PKVC trajectory (dominant + sequence + o_subtype)
4. 推理简述 (reasoning_notes ≤ 100 字)

# 输出严格 JSON, schema 见上方 SFT_TARGET_SCHEMA
```

## 4. Leakage prevention 规则 (砍 peer 后简化)

| 类型 | 规则 | 实现 |
|---|---|---|
| **Q7 leak (最严)** | input 任何 module 都不能含 Q7 数据 | filter `qid == "Q7"` 全部 |
| **target Q leak** | ③ + ⑤ 不能含 target Q{i} 的 data | filter `qid == target_q` |

(④ peer 砍了, 不再需要 peer-related 防 leak 规则.)

## 5. Data prep pipeline

### 5.1 输入数据源

- `data_v3/task_responses.csv` — 162 designer × Q1-Q6 raw answer (Q1 letter / Q2 **思考链路 cardId 序列** / Q3-Q6 option + text_supplement)
- `data_v3/processed/q1q6_extracted_v8.7_full.jsonl` — **162 × 6 = 972 条 extractor 输出**: cleaned NL + PKVC label + state_sequence (Q2 sequence 直接从这里取)
- `data_v3/users.csv` — pi 字段
- `data_v3/scale_responses.csv` — 30 维 polarity
- `reports/datav3/per_designer/option_trait_mapping.json` — Q3-Q6 (qid, option) → o_primary_hint / o_subtype_hint
- `data_v3/processed/quad_extracted_163_fixed.jsonl` — Q3-Q6 LASSO + LLM-labeled CAPS quad
- `data_v3/processed/reasoning_notes_pre_label.jsonl` — **已 pre-label, 162 × 6 = 972 条 cached** (one-time GPT-4o 调用结果)

### 5.2 单条训练样本 build pipeline (3 个 task_type 分支)

```python
def build_sft_example(designer_id: str, target_q: str) -> dict:
    """生成 1 条 SFT 训练 example: (input prompt, target JSON). target_q ∈ {Q1, Q2, Q3, Q4, Q5, Q6}."""
    # === Input prompt (5 module, mask target Q) ===
    pi_block = format_pi(users_df, designer_id)
    stance_block = format_stance_labels(scale_responses_df, designer_id)
    self_q16_block = format_self_q1q6(
        designer_id, exclude_qids=[target_q],
        source=q1q6_extracted_v8_7,   # 用 extractor cleaned NL, 不用 raw text_supplement
    )
    caps_quad_block = format_caps_quad_balanced(
        caps_quads_df, designer_id,
        exclude_qid=target_q if target_q in ("Q3","Q4","Q5","Q6") else None,
    )
    task_block = format_task_by_type(target_q)   # Q1 / Q2 / Q3-Q6 分支

    input_prompt = "\n\n".join([
        PKVC_4_LETTER_DEF,
        f"# 你的人口统计学信息\n{pi_block}",
        f"# 你的核心立场\n{stance_block}",
        f"# 你在其他题上的实际行为 (排除 {target_q})\n{self_q16_block}",
        f"# 你的认知路由模式 (CAPS 四元组)\n{caps_quad_block}",
        task_block,
    ])

    # === Target JSON (按 task_type 分支, NL 用 extractor cleaned 版) ===
    extracted = q1q6_extracted_v8_7[(designer_id, target_q)]
    supplement_nl_cleaned = extracted["nl_cleaned"]                      # ⚠ extractor 输出, 非 raw
    reasoning = reasoning_notes_cache[(designer_id, target_q)]

    if target_q == "Q1":
        letter = extracted["dominant_factor"]                            # extractor 已 parse
        target = {
            "task_type": "Q1",
            "chosen_letter": letter,
            "supplement_nl": supplement_nl_cleaned,
            "dominant_factor": letter,
            "state_sequence": [letter],
            "reasoning_notes": reasoning,
        }
    elif target_q == "Q2":
        thinking_chain = extracted["state_sequence_cardid"]              # designer 思考链路 (~3-7 cards)
        letter_seq = [c[0] for c in thinking_chain]                       # ["P3","K2",...] → ["P","K",...]
        target = {
            "task_type": "Q2",
            "thinking_chain": thinking_chain,
            "supplement_nl": supplement_nl_cleaned,
            "state_sequence": letter_seq,
            "reasoning_notes": reasoning,
        }
    else:  # Q3-Q6
        chosen_option = extracted["chosen_option"]
        mapping = option_trait_mapping[f"{target_q}_{chosen_option}"]
        dom = mapping["o_primary_hint"]
        subtype = mapping["o_subtype_hint"]
        state_seq = parse_subtype_to_seq(subtype)
        target = {
            "task_type": target_q,
            "chosen_option": chosen_option,
            "supplement_nl": supplement_nl_cleaned,
            "dominant_factor": dom,
            "state_sequence": state_seq,
            "o_subtype": subtype,
            "reasoning_notes": reasoning,
        }

    return {"input": input_prompt, "target": target}
```

### 5.3 `reasoning_notes` — 已 pre-label, 直接 cache 取

972 条 reasoning_notes 之前已用 GPT-4o 跑过 (one-time $15), 现存 `data_v3/processed/reasoning_notes_pre_label.jsonl`. SFT 直接 dict-lookup, **无新增 LLM 调用**.

## 6. Train / val / test split

| Split | Source | Size | Purpose |
|---|---|---|---|
| **Train** | Q1-Q6 (80%) | ~770 | LoRA gradient |
| **Val** | Q1-Q6 (20% holdout) | ~200 | Early stop / LoRA hyper-tune |
| **Test (Q1-Q6)** | val 同集 | ~200 | In-domain accuracy check |
| **Test (Q7)** | 162 designer × 3 seg = 486 | 486 | **paper main eval** (跨场景 generalization) |

注意:
- Train/val split by **(designer, Q)** pair, 162 designer 全部见过, 仅 hold out 个别 (designer, Q) 组合
- Designer 不分 train/test, 防止 cold start 问题污染 Q7 eval
- 真 test 是 Q7 (跨场景), 验证 train-on-Q1-Q6 学到的 routing 能否泛化到 Q7 (Q1→seg1, Q2→seg2, Q3-Q6→seg3 同构 transfer)

## 7. Loss design

### 7.1 多任务 loss (跟 v3.5.1 framing 一致)

```
L_total = L_lm + λ_cc · L_cc + λ_cls · L_cls

L_lm   = cross-entropy on entire JSON output (含 chosen_option + supplement_nl + PKVC fields)
L_cc   = consistency loss — chosen_option 跟 PKVC trajectory 一致性
         (e.g., 若 dominant=K, supplement 不能明显是 V 类语言)
L_cls  = K-way classification head on chosen_option (auxiliary)
```

初始 λ_cc = 0.3, λ_cls = 0.1. tune by val accuracy.

### 7.2 Sequential 训练顺序 (curriculum)

1. **Stage 1 (warmup, 1 epoch)**: 仅 L_lm, 让 model 学 JSON 输出 format
2. **Stage 2 (main, 3 epoch)**: L_lm + L_cls (auxiliary classification)
3. **Stage 3 (refine, 1 epoch)**: L_lm + L_cls + L_cc (加 consistency)

## 8. 完整训练样本示例 (designer=BuzhichunD, target Q3, 砍 ④ peer 后)

### 8.1 Input prompt (~2900 chars, 比 B6 short ~1000 因砍 peer + criterion)

```
你是设计师认知模拟器. 任务: 根据档案, 模拟设计师对 Q3 的回答.

# PKVC 4 类认知状态
- P (Problem) 问题界定 / 重定义: P1 / P2 / P3
- K (Knowledge) 知识调用 / 数据驱动: K1 / K2 / K3
- V (Value) 价值立场 / 底线宣言: V1 / V2 / V3
- C (Concept) 概念整合 / 抽象框架: C1 / C2 / C3

# 你的人口统计学信息
身份: 在职设计师 | 经验: 3-5年 | 领域: 交互/UX | 工作场景: 甲方内部团队 | 项目角色: 主创设计师

# 你的核心立场 (强 stance, 5-6 分或 1-2 分; 弱维度已过滤)
  • 完美主义 (完美主义/MVP 维度)
  • 边缘群体优先 (弱势群体优先维度)
  • 救世主情结 (救世主情结维度)
  ... (7-9 行)

# 你在其他题上的实际行为 (排除 Q3, NL 用 extractor cleaned 版)
Q1 第一反应单字母: P
  你的解释 (cleaned): "第一反应质疑题目本身, 设计师不应被框住."

Q2 思考链路: P3 → K2 → V1 → C2
  你的解释 (cleaned): "先质疑设计部门要管的范围, 再调研数据看用户真实需求, 然后亮明用户优先底线, 最后整合方案."

Q4 面对【商业利润与用户隐私冲突】【隐私交换与概念建构冲突】
  选择: "隐私和开源只是表象. 我会重新调研在什么真实使用场景下..."
  你的解释 (cleaned): "隐私这些大议题, 容易让人忽略具体用户需求..."

Q5 面对【AI 工具与设计师自主性冲突】【人机协作分工冲突】
  选择: "先独立构思核心概念..."
  你的解释 (cleaned): "保持设计原创性..."

Q6 面对【商业目标与社会责任冲突】【利益与责任感冲突】
  选择: "拒绝执行该上线指令..."
  你的解释 (cleaned): "好的产品与发布时间无关..."

# 你的认知路由模式 (CAPS 四元组, 4 类平衡, 排除 Q3 相关 routing)
## P 类 top routing
[Q4-D] 在【商业利润与用户隐私冲突】情境下, 该设计师在「资产开源」维度上持「开源道德派」立场(β=+0.759), 强力驱动思维进入 P3→K3 状态.
## K 类 top routing
[Q5-A] ...
## V 类 top routing
[Q6-A] ...
## C 类 top routing
[Q4-C] ...

# 场景 (target = Q3 — 数据效用 vs 美感冲突)
设计你的下一代图标系统时, UX 工程团队给出客观数据: 方案 A 设计美感测试得分 79%, 但用户实际识别准确率 99% (基于 10000 人 A/B 测试). 方案 B 反过来. 你选哪个方案推进? 请说明思考路径.

A. 数据是不可违背的铁律. 直接采用工程部门的 99% 准确率方案.
B. 质疑数据. 要求工程团队给出 79% 和 99% 在实际用户体验中的真实差异报告...
C. 走中间路线...
D. 重新质疑这个二选一...

# 任务
输出 task_type=Q3 schema JSON: chosen_option + supplement_nl + dominant_factor + state_sequence + o_subtype + reasoning_notes.
```

### 8.2 Target JSON

```json
{
  "task_type": "Q3",
  "chosen_option": "B",
  "supplement_nl": "原则上, 要解决问题的功能是最重要的, 但是有更美观符合产品调性的外观更好, 需要尽量保留. 因此要尽量保留美观的设计.",
  "dominant_factor": "K",
  "state_sequence": ["P", "K"],
  "o_subtype": "P3→K3",
  "reasoning_notes": "完美主义 trait (β=+0.16) + 边缘群体优先 → 不愿轻易让步, 先质疑数据 (P3 推翻假定) 再要求工程给 K3 评估真实差异."
}
```

### 8.3 长度估算 (砍 ④ + ⑥ 后)

| Module | chars |
|---|---|
| PKVC 4 字母含义 | 350 |
| ① pi (5 字段) | 120 |
| ② 30维 stance labels | 500 |
| ③ self Q1-Q6 (mask 1 target, 含 Q1+Q2+3 Qs) | 600 |
| ⑤ CAPS quad 4 类平衡 (mask target) | 1100 |
| Q3 场景 + 任务 + output schema | 700 |
| **Total (训 Q3-Q6)** | **~3370** |
| **Total (训 Q1/Q2)** | **~3170** (场景题面更短) |

## 9. SFT 跟 B6 inference 的字段对齐

| 字段 | SFT target (Q3-Q6) | B6 inference (Q7) | 对齐性 |
|---|---|---|---|
| dominant_factor | ✓ 单字母 (P/K/V/C) | ✓ 单字母 (seg1/3) 或 X→Y (seg2) | seg2 推理需 X→Y, SFT 训练只 anchor 单字母 → 可能 cold start |
| state_sequence | length 1-3 | length 1-7 (seg2 ≥ 2) | SFT seq 短, Q7 seg2 推理可能长 → 模型需 stretch |
| o_subtype | ✓ (有 mapping) | ❌ (Q7 v8.7 没抽 subtype) | inference 时 model 自己写 reasoning_notes, 不强制 subtype |
| criterion | ❌ (Q3-Q6 无 criterion) | ✓ (seg3 必填) | SFT 训不到 criterion 字段 → Q7 seg3 推理时 model 凭 cross-context 学 (类似 zero-shot transfer) |
| chosen_option | ✓ A/B/C/D | ❌ Q7 无 option | SFT 学的 categorical, Q7 推理时 schema 不要这字段 |
| supplement_nl / nl_answer | supplement_nl (短) | nl_answer (200-400 字) | 字段名不同, 但都是 free NL — 训练时 model 学风格 transfer 到 Q7 长 NL |

### 9.1 跨任务字段映射

- SFT 时 schema (task_type-dispatched):
  - Q1: `chosen_letter + supplement_nl + dominant_factor + state_sequence + reasoning_notes`
  - Q2: `thinking_chain + supplement_nl + state_sequence + reasoning_notes`
  - Q3-Q6: `chosen_option + supplement_nl + dominant_factor + state_sequence + o_subtype + reasoning_notes`
- Q7 inference 时 schema (per-seg):
  - seg1: `dominant_factor + confidence + reasoning_notes + nl_answer`
  - seg2: `state_sequence + dominant_factor (X→Y) + reasoning_notes + nl_answer`
  - seg3: `dominant_factor + state_sequence + criterion + confidence + reasoning_notes + nl_answer`

**共享字段** (训-推共用): dominant_factor / state_sequence / reasoning_notes
**SFT 独有**: chosen_letter / chosen_option / thinking_chain / supplement_nl / o_subtype
**Q7 推理独有**: criterion / confidence / nl_answer

### 9.2 同构 transfer 路径

```
SFT Q1 (chosen_letter + state_sequence=[letter])    → Q7 seg1 (dominant_factor)
SFT Q2 (thinking_chain + state_sequence letters)    → Q7 seg2 (state_sequence X→Y)  [1:1 mirror]
SFT Q3-Q6 (chosen_option + dominant + subtype + supplement)
                                                     → Q7 seg3 (dominant + nl_answer + criterion)
```

Q1/Q2 直接 mirror Q7 seg1/seg2 的 structure, Q3-Q6 训 dom + nl 风格. **criterion 字段是 zero-shot transfer** (训不到), 靠 prompt 的 PKVC 4 字母含义 + reasoning_notes 教 model 概念.

→ 训练目标: model 学到 trait → routing 内在规律, 不严格学每个字段名. inference 时按 Q7 per-seg schema 输出.

## 10. 已锁决策 (user 反馈)

| 项 | 锁定 |
|---|---|
| ④' contrastive peer | **❌ 砍** (没有簇 / peer 不必要) |
| ⑥ criterion library 进 SFT input | **❌ 不加** |
| Q1 + Q2 进训练 | **✅ 进** (跟 Q7 seg1/seg2 同构) |
| reasoning_notes pre-label | **✅ 已跑过, cache 取** |
| cluster 概念 | **❌ 砍** (本设计全程无 cluster) |
| Train/val split | **by (designer, Q) pair** |
| Loss curriculum | Stage 1 (L_lm) → Stage 2 (+ L_cls) → Stage 3 (+ L_cc) |

## 11. 期望

训完 LoRA, 在 Q7 上跑同 B6 inference prompt (砍 peer / criterion 一致):
- seg1 macro-F1 (mirror Q1 训练): 显著 boost, designer-specific dom 学到
- seg2 macro-F1 (mirror Q2 训练): chain 段 SFT 强, **+10pp 期望** (单纯 prompting 0.24 → 0.34+)
- seg3 macro-F1 (mirror Q3-Q6 训练): **B6 prompting 0.23 → 0.40+** (砍 peer / criterion 后 prompting 基线降, 但 SFT 补足)
- self-consistency: **B5 ~78% → 92%+** (训练专门强化结构一致性)
- 跨场景 transfer 真 demonstrate (训 Q1-Q6 → eval Q7 同构 seg)

paper main claim 升级: **"3 层 (Trait × CAPS × PKVC SM) + LoRA SFT 同构 transfer" 才是 main contribution**, 单 prompting 不够.

## 12. 文件清单 (待 implement)

| 文件 | 用途 |
|---|---|
| `scripts/build_sft_data_q1q6.py` | 主管道, 162 × 6 = 972 example build (Q1+Q2+Q3-Q6) |
| `data_v3/processed/sft_train.jsonl` | ~770 train rows |
| `data_v3/processed/sft_val.jsonl` | ~200 val rows |
| `data_v3/processed/reasoning_notes_pre_label.jsonl` | 972 条 pre-label cache (已有) |
| `sft_pipeline/data/sft_format.py` | 3 个 task_type schema + format helper |
| `sft_pipeline/data/prep.py` | 复用现有, 加 Q1/Q2 handler |
| `sft_pipeline/train.py` | LoRA fine-tune entrypoint (复用现有, schema dispatch) |
