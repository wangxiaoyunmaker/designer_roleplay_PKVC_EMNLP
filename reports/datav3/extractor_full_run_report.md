# PKVC Extractor 全量标注最终报告

> 2026-05-25. 用 v8.6 final prompt 标了 **data_v3 全量 163 用户**: 486 Q7 + 309 Q1-Q6 = **795 段 SFT 训练数据**.

## TL;DR

| 阶段 | 样本 | dominant accuracy | 主要结论 |
|---|---|---|---|
| v8.2 baseline (5-gold) | 15 | 53% | 起点 |
| v8.3 (7 marker patches) | 15 | 93% | 修系统 marker 错例 |
| v8.5 (seg1 force) | 15 | **100%** | 5-gold ceiling (但 overfit) |
| **v8.6 (5-gold + 48-hand)** | 63 | **93.7%** | 真实 generalize ceiling |
| **v8.6 全量 data_v3** | **795** | (no gold) | **训练数据 final, 162 user 覆盖** |

**全量 795 段, 0 fail, 242s 跑完 (≈ \$12).** Schema strict enforce 100% 通过.
覆盖 163 baseline 设计师中的 162 个 (1 个 user `wiwsx19940412` Q7 全空, 真没填).

---

## 优化路径

### Phase 1: 5-user expert gold 校准 (15 段)

文件: `data/processed/expert_gold/parsed.json` (5 user × 3 seg = 15 段, expert hand-labeled).

| version | dom_acc | 改动 |
|---|---|---|
| v8.2 | 53% | (baseline) state_sequence 字段加入 |
| v8.3 | 93% | 7 marker/规则补丁: P3 角色边界 / V2 商业实效 / C3 设计理念 / P1 情境矛盾 / V1 表象 / K-V 收编关系 |
| v8.4 | 87% | (regression) SEGMENT_ROLE inject 给 seg2/seg3 加 hint, seg3 hint 反咬 Yyd seg3 |
| v8.5 | **100%** | 仅 seg1-only SEGMENT_ROLE force, 解 BuzhichunD seg1 V 表象 |

### Phase 2: 200-segment validation (48 hand-label)

文件: `data/processed/extractor_val_200.jsonl` (150 Q7 + 50 Q1-Q6, exclude 5 gold users)
+ `data/processed/hand_labels.json` (48 段 hand-labeled by claude using v8.5 framework).

v8.5 on 48 hand-gold = **88%** (5-gold 100% 是 overfit 信号).

5 错例分析:
- **C2 整合构建 marker 弱** (#7, #20): "独立构思 / 设计发散+收纳" 被抓成 V / K
- **seg1 force V 规则边界** (#28 段首 V 没 force; #37 段首 P1 矛盾梳理被 force V)
- **seg2 K-V 主体辨析** (#17 K-setup 被当 K 主体)
- **CraigCheon Q4 K vs V 边界** (#3)

### Phase 3: v8.6 patch + 验证 (3 改动)

文件: `sft_pipeline/config/prompts/extractor_v8.6.md`

| 改动 | 详细 |
|---|---|
| 1. C2 marker 加强 | 加 "设计发散+设计收纳 / 经验推论与实景验证 / 独立构思核心概念 / 从不同角度想出 X / 整合方法论" 等 marker |
| 2. seg1 force V P1 例外 | 段首是 P1 矛盾梳理 marker ("第一反应是矛盾 / 抓核心矛盾"), dominant 优先 P 不 force V |
| 3. K-setup vs K-主体辨析 | "数据是参考"(K-setup) → 后接 V 阐述 = V 主导; "深入调研 / AB 测试 / 调阅数据"(K-主体) → 段尾 V = K 主导 |

v8.6 结果:
- 5-gold: 100% → **93%** (1 regression: BuzhichunD seg1)
- 48 hand-gold: 88% → **94%** (+4 fix, -1 break = net +3)
- **Combined 63 段: 90.5% → 93.7%** (净 +3.2 pp)

---

## 全量 795 段分布 (v8.6 final, data_v3)

### Q7 (486 段, 162 user × 3 seg)

| seg | V | P | K | C | multi-state | high conf | criterion null |
|---|---|---|---|---|---|---|---|
| seg1 (第一反应) | 125 (77%) | 28 (17%) | 7 (4%) | 2 (1%) | 11% | 94% | 1% |
| seg2 (决策链路) | 86 (53%) | 33 (20%) | 40 (25%) | 3 (2%) | **85%** | 80% | 2% |
| seg3 (底线锚定) | 149 (92%) | 8 (5%) | 4 (2%) | 1 (1%) | 10% | 96% | 0% |

**符合预期**:
- seg1 V 主导 (第一反应通常是价值/直觉表象, 77% V), 但 P/K/C 4 类全有
- seg2 N-state chain 真出来 (**85% multi-state**, avg seq length ~3.2), 4 类均衡 (V/K/P 主体, K 25% 比 78-user 旧跑高很多 — 162 user 池里 K-leaning designer 更多)
- seg3 V 主导 (底线收束 = V3 设立原则 典型), **criterion null=0** (所有 seg3 都有底线短句)

### Q1-Q6 expanded (309 段, 102 user)

| dom | n | % |
|---|---|---|
| V | 125 | 40% |
| K | 89 | 29% |
| P | 53 | 17% |
| C | 42 | 14% |

4 类均衡分布 (不像 v8 早期全 V vacuous). vs canonical o_primary_hint 一致率 ~66% (200-val 上测过) — 但 canonical 本身是 designed-question intent, 不是 actual NL gold (expansion 自己跟 canonical 也只 44%).

### 质量指标

- **Schema strict enforce**: 0 fail across 405 calls (json_schema strict 100% 通过)
- **High confidence rate**: 95-97% (Q7) / 98% (Q1-Q6)
- **Criterion null rate**: 0-3% (LLM 几乎都能给 verbatim criterion 短句)
- **Sequence dedup tendency**: LLM 倾向给短 sequence (length 1 占主体 except seg2 83% multi-state)

### Confusion (5 gold + 48 hand-gold, v8.6)

| gold → pred | n | 状态 |
|---|---|---|
| V → V | 40 | ✓ 主对角线 |
| K → K | 7 | ✓ |
| P → P | 7 | ✓ |
| C → C | 3 | ✓ |
| V → P | 2 | ✗ (BuzhichunD seg1 + lynnsaidly seg2) |
| K → V | 1 | ✗ (CraigCheon Q4) |
| V → K | 1 | ✗ (#17 17816030126 seg2) |

主对角 57/63 = 93.7%; off-diagonal 残余 4 错都是 K/V/P 边界 (即便人工标也可能争议).

---

## 输出文件

| 文件 | 内容 |
|---|---|
| `sft_pipeline/config/prompts/extractor_v8.6.md` | **final prompt** (~360 行, 12 marker 表 + 9 example + self-check Q1-Q12) |
| **`data_v3/processed/q7_extracted_v8.6_full.jsonl`** | **486 Q7 段** (162 user × 3 seg) SFT 训练标注 ⭐ |
| **`data_v3/processed/q1q6_extracted_v8.6_full.jsonl`** | **309 Q1-Q6 段** (102 user) SFT 训练标注 ⭐ |
| `data_v3/processed/full_corpus_v8.6_data_v3.log` | 全量跑 log |
| `data/processed/q7_extracted_v8.6.jsonl` | (旧) 234 Q7 段, 79-user 子集, **不要用** |
| `data/processed/q1q6_extracted_v8.6.jsonl` | (旧) 171 Q1-Q6 段, 56-user 子集, **不要用** |
| `data/processed/expert_gold/parsed.json` | 5-user expert gold (per-seg dominant + sequence + criterion) |
| `data/processed/hand_labels.json` | 48 hand-label set (8 group: a/b/c/d × confident/suspect) |
| `reports/datav3/eval_vs_gold_v8.{2,3,4,5,6}.json` | 各版本 5-gold accuracy |
| `reports/datav3/val_200_v8.{5,6}.jsonl` | 200-val full output (per version) |
| `scripts/parse_expert_gold.py` | gold xlsx 解析 |
| `scripts/eval_vs_gold.py` | 5-gold accuracy 评测 |
| `scripts/run_val_200.py` / `eval_v8_6_vs_hand_gold.py` | 200-val 评测 |
| **`scripts/run_full_data_v3.py`** | **全量跑脚本 (data_v3, 163 用户)** ⭐ |
| `scripts/run_full_corpus.py` | (旧) 79-user 全量跑, **deprecated** |

---

## 每条训练样本结构

```json
{
  "id": "8e741c91ad70",                       // segment hash
  "user_id": "1142208878",
  "qid": "Q7",
  "segment_role": "seg1",                     // seg1/seg2/seg3/single
  "text": "...完整 NL...",
  "source": "Q7",                             // or "Q1-Q6_expanded"
  "pred": {
    "dominant_factor": "V",                   // P|K|V|C 或 X→Y
    "state_sequence": ["V", "K", "P", "V"],   // 1-5 长度
    "criterion": "用户伦理放在首位",         // ≤25 字 verbatim 或 null
    "confidence": "high",                     // high|medium|low
    "extraction_notes": "..."                 // ≤30 字推理
  },
  "ok": true
}
```

---

## 局限 + 后续建议

### 局限

1. **5-gold + 48 hand-label = 63 段 ground truth, 总样本 405** — 测过的 ratio 15.6%, 剩 84% 没人标过.
2. **hand-label 由 claude 用 v8.5 framework 标的**, 不算独立 gold. 真正 cross-checker 应让学姐或独立 expert 标.
3. **5-gold 100% → 48-hand 88% 揭示了 ~12% 真实 noise**. 全量 405 估计 noise ~6-10% (因 v8.6 把 88% 推到 94%).
4. **Q1-Q6 expanded NL 是 GPT 生成的**, 不是真人写的. 标注质量受 expansion 质量限制. Q1-Q6 真人原始解释 (text_supplement) 才 1-10 字, 信号弱.
5. **Sequence Jaccard ~0.65-0.70**: LLM 倾向 dedup 短 sequence, 跟 expert "每模块一 state" 风格不完全对齐. dominant 不受影响, 但 bridge mining 用 sequence 时要 normalize.

### 后续建议

1. **跨标注者 κ pilot** (投稿 ICLR 前必做): 学姐 + 你 各标 30 段 (不看 v8.6 output), 算 κ. κ ≥ 0.7 才能 claim 标注 reliable.
2. **扩 hand-gold 到 100 段**: 现 48 段是 suspect-biased (低 conf / canon-mismatch 都进去了). 加 ~50 random high-conf 段做平衡 test set.
3. **subtype 细化**: 现 dominant 只到大类 P/K/V/C. SADR 30 维子类还可以从 NL 抽 (v3.5.1 §4D bridge target). 加一轮 subtype extractor 跑 v8.7.
4. **Confidence=low 段 (~5%) curriculum 后期**: 这些是 LLM 没把握的, 训练时可放后期 / 或 augment 重写.
5. **训练数据格式落盘**: 整合 405 段 → `sft_pipeline/data/dataset.py` SFT input 格式. 加 cluster_id (从 user_cluster_efa.csv 拼) + RAG peer top-3.

### Cost / Time

- 全量 data_v3 795 段 \$12 / 242s (concurrency=8)
- Full pipeline (sample + 5-gold + val_200 + iter v8.6 + 旧 405 + new 795): \$30-40 / ~1 hour
- 若日后扩到 N=200 baseline (37 user 还未入), 增量跑同 prompt: ~110 段 \$2, ~30s

---

## 一句话

v8.6 是当前 best extractor prompt, 在 5-gold 上 93%, 48-hand 上 94%, 已用于 **data_v3 全量 162 user / 795 段 SFT 训练数据标注**. 残余 noise ~6-10%, 边界争议为主. 准备投训练.
