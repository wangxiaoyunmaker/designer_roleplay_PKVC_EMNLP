# Extractor Prompt v7g Ablation Plan

> 测的不是"哪版 prompt 最好", 测的是 "prompt 哪些段在出力, 哪些是冗余装饰".
> 出来的结论指导 v8 该加什么 / 该砍什么. **A/B/C 必须同模型 + 同 30 段 + 同 seed**,
> 否则不是 ablation.

---

## 1. 三个变体

| 变体 | 文件 | 跟 A 的差 | 假设 |
|---|---|---|---|
| **A** (control) | `extractor_v7g.md` | — (full) | 当前 production prompt, baseline κ |
| **B** | `extractor_v7g_B.md` | 砍 §"常见误判" 6 条 + §"输出前 self-check" Q1-Q5 | 这两块是 v6/v7 历史补丁, 现已挤掉 LLM 自有判断带宽 |
| **C** | `extractor_v7g_C.md` | B 再砍 §"12 子类典型句式 marker" 表 (P/K/V/C 4 张子表) | 12 子类 marker 跟 4 因子触发关键词 + 4 few-shot 重叠 ~70%, 是冗余装饰 |

⚠ 变量隔离: B/C **不引入**任何 v8 改进 (criterion 长度上限 / 路径语义 / metadata
先验 / json_schema enforce / few-shot 加 medium-low). 否则 κ 差就不能归因到"砍了
什么", 而是"砍了 + 加了". 把 v8 改进留到下一轮.

---

## 2. 跑法 (假设 batch.py 支持 `--prompt-template` 参数; 不支持就先改 batch.py)

### 2.1 选 30 段 pilot set (与之前 κ pilot 相同 seed)

```bash
# 假设 q7_segments.jsonl 已 ready (234 段)
.venv/bin/python -c "
import json, random
random.seed(20260524)
with open('data/processed/q7_segments.jsonl') as f:
    segs = [json.loads(l) for l in f]
# 排除 leakage: few-shot 用到的 4 段
LEAK = {('C','seg1'), ('D','seg2'), ('E','seg3'), ('F','seg1')}
segs = [s for s in segs if (s.get('appendix'), s.get('seg_id')) not in LEAK]
pilot = random.sample(segs, 30)
with open('data/processed/q7_pilot30.jsonl', 'w') as f:
    for s in pilot: f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'wrote {len(pilot)} segs')
"
```

⚠ **leakage 自检**: few-shot 段来自附录 C-seg1 / D-seg2 / E-seg3 / F-seg1, 不能
进 pilot. 如果 `q7_segments.jsonl` 里没有 `appendix` / `seg_id` 字段, 改成按 text
前 30 字 hash 匹配排除. 这一步不做, κ 失真.

### 2.2 三版各跑一次 (同模型, 默认 GPT-4o)

```bash
for V in v7g v7g_B v7g_C; do
  .venv/bin/python -m sft_pipeline.extract.batch \
      --input data/processed/q7_pilot30.jsonl \
      --output data/processed/q7_pilot30_${V}.jsonl \
      --prompt-template sft_pipeline/config/prompts/extractor_${V}.md \
      --model gpt-4o \
      --temperature 0 \
      --concurrency 4 \
      --seed 20260524
done
```

### 2.3 算指标

```bash
.venv/bin/python -m sft_pipeline.extract.batch \
    --kappa-eval \
    --input-a data/processed/q7_pilot30_v7g.jsonl \
    --input-b data/processed/q7_pilot30_v7g_B.jsonl \
    --report data/processed/extractor_v7g_vs_v7g_B.md

.venv/bin/python -m sft_pipeline.extract.batch \
    --kappa-eval \
    --input-a data/processed/q7_pilot30_v7g.jsonl \
    --input-b data/processed/q7_pilot30_v7g_C.jsonl \
    --report data/processed/extractor_v7g_vs_v7g_C.md
```

---

## 3. 评测指标 (3 个一起看, 别只看 κ)

| 指标 | 算法 | 解读 |
|---|---|---|
| **κ (dominant_factor)** | Cohen's κ between A vs B (or A vs C) 4-class (P/K/V/C, 路径合并到首字母) | 主指标. κ ≥ 0.7 视作"砍掉无损" |
| **criterion 非 null 率** | non-null 比例差 | v7g→B/C 砍掉 marker 表后, criterion 抓不抓得到 V3 宣言? 这个会暴露 marker 表对 criterion 的辅助价值 |
| **input token / 段** | tiktoken 统计 prompt + INPUT | B vs A 应降 ~300 token, C vs A 应降 ~800 token. 配合成本估算 |
| **factor 分布** | P:K:V:C 4 类计数 | 若 C 版 V 类占比突涨 (V default bias 抬头), 说明 marker 表在压制 V bias, 应保留 |
| **路径 (X→Y) 触发率** | 含 "→" 的 dominant_factor 比例 | 三版差大说明路径表示对 prompt context 高度敏感, v8 必须显式定路径规则 |

---

## 4. 决策树

```
if κ(A,B) ≥ 0.85:
    "常见误判 + self-check" 是装饰, v8 砍掉.
    → 进 if κ(B,C) 分支
elif 0.7 ≤ κ(A,B) < 0.85:
    "常见误判 + self-check" 救场 ~10-15% 段, 保留但精简到 3 条最有 lift 的.
else (κ(A,B) < 0.7):
    它俩在救场, v8 保留 + 加强. 不再考虑砍 marker 表.

if κ(B,C) ≥ 0.85:
    12 子类 marker 表也是装饰, v8 直接极简 (4 因子 + few-shot 即可).
    总 token 砍 60%, 成本 $5 → $2.
elif 0.7 ≤ κ(B,C) < 0.85:
    marker 表有边际价值, v8 保留 12 表但合并触发关键词列 (消重叠).
else:
    marker 表关键, v8 保留 + 加 medium/low confidence few-shot.
```

---

## 5. 同步要做的检查 (跟 ablation 独立, 但都跑了顺手做)

1. **V default bias 量化**: 三版各自 V 类占比 vs 79 人 baseline pilot. baseline 是
   什么不清楚 (现 v3.5.1 没记录), 跑前先算 ground truth (学姐手标 30 段) 的 V 类
   比例作锚.
2. **criterion 截取规则模糊**: 例 3 截 "让人幸福而不是得利" (9 字) 是 sub-string 还是
   完整短句? 三版 LLM 截取行为差异 → v8 加规则.
3. **路径 (X→Y) 语义**: 三版输出里所有 path case dump 出来, 人工标"时序" vs "强弱",
   看 LLM 默认是哪种 → v8 显式声明.

---

## 6. 下一步 (v8 候选, 不进本轮 ablation)

按 v7g_B/C ablation 结果排序优先级:

- **v8-1** json_schema enforce (response_format), 砍 JSON 铁律到 1 条
- **v8-2** few-shot 加 1 例 medium confidence + 1 例 criterion 非 null V3 真实场景
- **v8-3** 路径规则显式: "X→Y = 段内时序; 单字母 = 单一主导; 上限 2 段"
- **v8-4** criterion 长度 20→25 字, 加截取规则 "包含 marker 的最小完整短句, 不跨标点"
- **v8-5** metadata (segment_role / scenario) 先验: seg1↦K/P, seg2↦V/C, seg3↦V3
  (先用 79 人 ground truth 验证此先验真存在, 不然 bias 起反作用)
- **v8-6** confidence=low 加 escape valve: "若信号过弱, dominant 给最可能字母 +
  extraction_notes 标注'信号弱'", 当前是硬猜

---

## 7. Open question (跑 ablation 前需要 user 确认)

- Q1: pilot N=30 还是 60? 30 段 κ 置信区间宽 (CI95% ≈ ±0.15), 60 段更稳但成本 2×.
- Q2: 用 GPT-4o 还是 Claude 3.5 Sonnet? 之前 κ pilot 是 GPT-4o vs Claude (跨模型一致性),
  这次 ablation 用单模型 (固定 GPT-4o 测 prompt 影响). 跨模型 ablation 留下一轮.
- Q3: temperature 设 0 还是 0.3? 0 最 deterministic, 但实际 production 跑全量是 0.3
  (待 batch.py 确认). ablation 用 production 同设比较有意义.
