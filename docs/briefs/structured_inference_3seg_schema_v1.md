# Structured Q7 Inference — Per-Seg 3 Schema 方案 v1

> 2026-05-25. 实现方案落盘, 待 RAG 改动结束后 implement.
> Author: claude (跟 user 对齐过).

## 0. 背景

现状: 4 baseline (B1/B1d/B2/B3) inference 输出 **free-form NL**, 后跑 v8.7 PKVC extractor 抽 dom/sequence/criterion. 2-step pipeline 有 extract noise, model 不被强制 commit PKVC.

升级: inference 输出 **per-seg structured JSON**, model 先输出 PKVC 字段再输出 nl_answer. 跟未来 SFT target format 一致.

## 1. 核心 framing — Q7 三 seg 性质不同

数据验证 (162 designer × 3 seg ground truth via v8.7):

| Seg | 性质 | 主要 sequence 长度 | dominant 含 X→Y path | dominant V 占比 |
|---|---|---|---|---|
| **seg1** 第一反应 | anchor (单状态) | 1 (80%) | **1% (≈0)** | 78% |
| **seg2** 决策链路 | chain (多状态) | 3-4 (72%) | 2% | 54% 均衡 |
| **seg3** 核心 tradeoff | anchor (单状态 + 底线) | 1 (85%) | **0% (从不)** | 93% |

→ seg1 + seg3 是 **anchor 段** (单 PKVC state); seg2 是 **chain 段** (multi-state transitions). 跟 v3.5.1 "PKVC State Machine" 理论一致.

## 2. 3 套 JSON schema

### 2.1 共有字段 (5 个)

| 字段 | 类型 | 描述 |
|---|---|---|
| `dominant_factor` | string | 段主导 PKVC 状态 |
| `state_sequence` | array of "P"/"K"/"V"/"C" | 段内状态时序 |
| `criterion` | string or null (seg3: string only) | 核心底线短句 verbatim |
| `confidence` | "high"/"medium"/"low" | model 对结构标的把握 |
| `reasoning_notes` | string ≤ 100 字 | model 推理简述 |
| `nl_answer` | string | 设计师 NL 回答 (200-400 字 expected, 不硬约束) |

### 2.2 Per-seg 差异

```python
SEG_SCHEMAS = {
    "seg1": {                                  # 第一反应: anchor
        "dominant_factor": {
            "type": "string",
            "pattern": "^[PKVC]$"              # 单字母, 禁 X→Y
        },
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 3       # 短 sequence
        },
        "criterion": {
            "type": ["string", "null"],         # nullable
            "maxLength": 25
        },
    },
    "seg2": {                                  # 决策链路: chain
        "dominant_factor": {
            "type": "string",
            "pattern": "^(P|K|V|C)(→(P|K|V|C))?$"  # 允许 X→Y
        },
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 2, "maxItems": 7        # 强制 multi-state (>=2)
        },
        "criterion": {
            "type": ["string", "null"],
            "maxLength": 25
        },
    },
    "seg3": {                                  # 核心 tradeoff: anchor + 底线
        "dominant_factor": {
            "type": "string",
            "pattern": "^[PKVC]$"               # 单字母, 禁 X→Y
        },
        "state_sequence": {
            "type": "array",
            "items": {"type": "string", "enum": ["P", "K", "V", "C"]},
            "minItems": 1, "maxItems": 3
        },
        "criterion": {
            "type": "string",                   # required non-null
            "maxLength": 25
        },
    },
}
```

### 2.3 完整 schema builder

```python
def build_seg_schema(seg: str) -> dict:
    seg_fields = SEG_SCHEMAS[seg]
    common_fields = {
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning_notes": {"type": "string", "maxLength": 100},
        "nl_answer": {"type": "string", "minLength": 100},
    }
    return {
        "type": "object",
        "properties": {**seg_fields, **common_fields},
        "required": ["dominant_factor", "state_sequence", "criterion",
                     "confidence", "reasoning_notes", "nl_answer"],
        "additionalProperties": False,
    }
```

`json_schema strict: True` (OpenAI API) 强 enforce.

## 3. 系统 prompt + per-seg user prompt 设计

### 3.1 系统 prompt (所有 seg 共用)

```
你是设计师认知模拟器. 任务: 根据给定档案 + 场景, 模拟一个真实设计师对 Q7 场景某 seg 的回答.

输出格式 (严格 JSON, 必须先填 PKVC 结构化字段, 再写 nl_answer):
1. dominant_factor: 段主导 PKVC 状态 (P/K/V/C 或 X→Y path, 按 seg 约束)
2. state_sequence: 段内 P/K/V/C 状态时序 array (长度按 seg 约束)
3. criterion: 核心底线短句 ≤ 25 字 verbatim, 必须在 nl_answer 里能找到 (seg3 必填, seg1/2 可 null)
4. confidence: 你对结构标的把握 (high/medium/low)
5. reasoning_notes: ≤ 100 字, 简述 PKVC 标的依据
6. nl_answer: 真实设计师风格的 NL 回答 (200-400 字, 第一人称, 中文, 口语化)

要求:
- nl_answer 必须跟 dominant_factor + state_sequence + criterion 内部一致
- 不要在 nl_answer 里 echo dominant/criterion 字面值 (e.g., 不要写 "我的 dominant 是 V")
- nl_answer 必须自然, 像真人设计师在内部讨论, 不要"AI助手"风格
```

### 3.2 Per-seg user prompt 末段 (区别)

**Seg1** (anchor, 直觉):
```
# 当前 seg: seg1 — 第一反应

任务: 描述你**第一时间看到这个项目时的真实直觉感受**.
约束:
- dominant_factor 必须**单字母** (P/K/V/C), 不要写 X→Y path (第一反应不应该是推理链)
- state_sequence 长度 1-3 (anchor 段)
- criterion 可 null (第一反应不必有明确底线宣言)
- nl_answer 200-400 字, **直觉 + 短**, 不要 step-by-step 推理
```

**Seg2** (chain, 决策权衡):
```
# 当前 seg: seg2 — 决策链路

任务: 按你的实际决策逻辑, 描述你怎么在【数据】【商业要求】【用户伦理】3 者之间权衡, 思考链路是什么.
约束:
- dominant_factor 可单字母或 X→Y path (链路头尾)
- state_sequence **必须至少 2 个不同 state** (chain 段, 强制 multi-state)
  示例: ["V", "K", "P", "V"] (用户伦理 → 数据评估 → 问题重定义 → 价值收束)
- criterion 可 null
- nl_answer 200-400 字, 显式展开你的权衡推理
```

**Seg3** (anchor + 底线):
```
# 当前 seg: seg3 — 核心 tradeoff

任务: 说出你向管理层提出的**核心设计底线** + 可妥协方案, 绝不让步的是什么, 愿意妥协的是什么.
约束:
- dominant_factor **单字母** (P/K/V/C), 不要 X→Y path
- state_sequence 长度 1-3 (anchor 段, 通常单字母重复或单一状态)
- criterion **必填** (≤ 25 字 verbatim 出现在 nl_answer 中, 是你底线短句的原文)
- nl_answer 200-400 字, 必须包含明确的"不可妥协 X / 可妥协 Y" 表达
```

## 4. 代码改动 file-by-file

### 4.1 `scripts/run_3_baselines_q7.py`

**改动**:
- 新加 `SEG_SCHEMAS` + `build_seg_schema(seg)` 函数 (放文件头)
- `build_b1_prompt` / `build_b1d_prompt` / `build_b2_prompt` / `build_b3_prompt` / `build_b4_prompt`:
  - 末段加 per-seg structured instruction (§3.2)
  - 系统 prompt 用 §3.1 (替代当前 SYSTEM_PROMPT)
- `call_one(client, baseline, designer_id, scenario, user_prompt, model)`:
  - 加 `seg = scenario.split("_")[1]`
  - `schema = build_seg_schema(seg)`
  - `response_format={"type": "json_schema", "json_schema": {"name": f"q7_{seg}_inference", "schema": schema, "strict": True}}`
  - parse `r.choices[0].message.content` 拿 dict (不是 str)
- 输出 jsonl row 字段:
  ```json
  {
    "ok": bool,
    "baseline": "B1d",
    "designer_id": "...",
    "scenario": "Q7_seg2",
    "claimed_pkvc": {                          // model 自己声明 (新)
      "dominant_factor": "V→K",
      "state_sequence": ["V", "K", "P", "V"],
      "criterion": "...",
      "confidence": "high",
      "reasoning_notes": "..."
    },
    "simulated_nl": "<nl_answer 字段>",        // 兼容旧 schema
    "ground_truth_nl": "...",
    "ground_truth_pkvc": {...},
    "retrieved_peers": [...] // B3/B4
  }
  ```
  注: `simulated_nl` 字段保留旧名, 值是 `claimed_pkvc.nl_answer`, 让 eval_5metric 不用改字段名.

### 4.2 `scripts/eval_baselines_pkvc.py`

**改动**:
- 加 CLI flag `--use-claimed`:
  - `--use-claimed`: 跳过 v8.7 extract, 直接用 inference jsonl 里的 `claimed_pkvc.dominant_factor` + `state_sequence` 作 pred
  - default (无 flag): 仍跑 v8.7 extract on nl_answer (用于 self-consistency 对比)
- 输出 jsonl 加 `pred_source` 字段 ("claimed" 或 "v8.7_extract")
- 若两个都跑, 加 `claimed_vs_extracted_agree` 字段 (新指标)

### 4.3 `scripts/eval_baselines_5metric.py`

**改动**:
- 加 CLI flag `--use-claimed-pkvc`:
  - True: 用 inference jsonl 的 `claimed_pkvc.state_sequence` 算 path metric
  - False: 用 PKVC eval jsonl 的 extracted state_sequence (现行为)
- 新指标 (optional, 加 flag `--also-self-consistency`):
  - PKVC self-consistency = (claimed dom == extracted dom) 比例 — model 自我一致性
- 11 metric 公式不变, 只换 input source

### 4.4 SFT 训练 target format

未来 SFT 训练时, target 就是 inference 的 structured output (跟 baseline 同 schema, per-seg):

```jsonl
{"input": {... scenario + designer profile ...},
 "target": {
   "dominant_factor": "V",
   "state_sequence": ["V"],
   "criterion": "用户体验不可妥协",
   "confidence": "high",
   "reasoning_notes": "段首明确底线",
   "nl_answer": "我觉得这项目的核心..."
 }}
```

直接 supervised fine-tune Qwen2.5-7B + LoRA r=32 用 cross-entropy on JSON token sequence.

## 5. Eval pipeline 变化

### 5.1 现有 11 metric 全部保留

PKVC dom acc / TED / STA / syn / sem / prag / 4 path / persona — 全保留.

**source 切换**:
- 旧 (free-form + post-hoc extract): inference NL → v8.7 extract → pred → compare with gt
- 新 (structured): inference claimed → pred → compare with gt

### 5.2 新指标 (per structured output)

| 指标 | 计算 | 意义 |
|---|---|---|
| **schema_pass_rate** | model 输出 JSON 通过 strict schema 验证比例 | 衡量 model 遵守约束能力 (per-seg) |
| **PKVC self-consistency** | claimed dom == v8.7 extracted dom on nl_answer | model 内部一致性 (claim vs 实际 NL) |
| **criterion_in_nl** (seg3) | criterion verbatim 出现在 nl_answer 中 | criterion 真不真 (不是凭空造) |

### 5.3 对照价值 (paper ablation)

- B1/B1d/B2/B3 same baselines but **structured** 后是否提升?
- Hypothesis: structured 让 model 在 path metric 上提升 (因强制 commit sequence, 不能模糊处理)
- 跟 free-form baseline 1944 row 对比 → 可写 paper 一个 ablation "structured output > free-form"

## 6. 迁移步骤 (待 RAG 改完后做)

### Step 1: schema 落盘 + 单测 (1h)

- 写 `sft_pipeline/extract/schema.py:get_q7_seg_schema(seg)` (复用现有 schema 模块)
- 加 unit test: 3 seg schema valid, 正负样本 strict enforce 通过

### Step 2: prompt builder 改动 (1h)

- `scripts/run_3_baselines_q7.py`:
  - 加 SYSTEM_PROMPT_STRUCTURED (§3.1)
  - 改 5 个 build_*_prompt 函数末段 (§3.2)
  - 改 call_one + 输出 schema

### Step 3: smoke 5 designer × 3 seg × 4 baseline = 60 calls (5 min, $1)

- 验证 JSON schema 强 enforce 没崩 (schema_pass_rate ≥ 95%)
- spot check 3 个 sample 看 dominant/sequence/criterion 是否合理

### Step 4: 全量重跑 162 × 3 × 4 = 1944 (30 min, $25)

- 输出 `reports/datav3/baselines_q7_structured_inference.jsonl`
- 保留旧 free-form 作对比 (rename 加 `_v1_freeform` 后缀)

### Step 5: eval 重算 11 metric + 新 3 个指标 (10 min, 0 API if use claimed)

- claimed PKVC 直接拿 metric, 0 API
- self-consistency: optional v8.7 extract on nl_answer (+$25)

### Step 6: 报告对比 (free-form vs structured) → `reports/datav3/baselines_q7_structured_vs_freeform.md`

## 7. Open questions (待 user 决定)

| Q | Default | 备选 |
|---|---|---|
| 1. Self-consistency 跑不跑 (+$25)? | 不跑 | 跑, 加 paper 一个指标 |
| 2. nl_answer 长度硬约束 (minLength=100)? | 软 (just expected) | 硬 (强 schema 验证) |
| 3. reasoning_notes 字段保留? | 保留 (≤100 字, 给 SFT 用) | 砍 (省 token) |
| 4. 重跑全 4 baseline 还是只 B1/B3? | 全 4 (公平对比) | 只 B1/B3 (省一半 $$$, 主要 finding 看 endpoints) |
| 5. v8.7 ground truth 重抽 with strict schema? | 不动 (旧 gt 保留) | 重抽 162 × 3 = 486 calls ($7) |

## 8. 跟当前 RAG 改动的关系

User 在改 RAG, 这个 structured 方案 **跟 RAG 解耦**:
- RAG (caps_anchored / B4 等) 修改的是 retrieval logic + prompt 上半段 (peer context)
- Structured 修改的是 response_format + prompt 下半段 (output instruction)
- 两者可独立改, 但 implementation 顺序: 等 RAG 稳定后再上 structured (避免 2 个变量同时变)

## 9. 验收标准

- [ ] 3 schema + unit test 通过 (positive + negative samples)
- [ ] Smoke 60 calls: schema_pass_rate ≥ 95% (允许 ≤ 5% 边缘 case)
- [ ] 全量 1944 calls 0 schema fail (json_schema strict 应该 0)
- [ ] B1d STA / Path metric 不退步 (跟 free-form 比), 期望 +1-2 pp
- [ ] 4 baseline ordering 不翻 (B3 > B1d > B1 > B2 on STA, B3 > B2 > B1d > B1 on dom)

## 10. 一句话

Q7 3 seg 性质不同 (anchor / chain / anchor + 底线), per-seg structured schema 让 inference 强 commit PKVC 再输出 NL, 跟未来 SFT target 完全一致, 不破坏现有 11 metric eval, 待 RAG 改完后即可 implement.
