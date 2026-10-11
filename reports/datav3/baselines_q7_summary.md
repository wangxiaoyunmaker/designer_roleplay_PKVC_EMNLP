# Q7 6-Baseline Role-Play 最终结果 (macro-F1 corrected, per-seg + ablation)

> 2026-05-25. 162 designer × 3 Q7 seg × 6 baseline = 2916 inference (100% coverage 含 retry). PKVC v8.7 extract on nl_answer + 11 base metric + self-consistency. Structured output (claimed PKVC + nl_answer JSON).

## ⚠️ 数据偏 + 指标修正 (2026-05-25 加)

**GT seg3 V=93% 严重不平衡** — dom acc / self-consistency 在 seg3 上有 **majority-class artifact**. B5 在 seg3 上 dom acc 看似 85.2% (其他 baseline 5-12%), 实际是把全部 12 个 non-V 个体也 claim 成 V (Macro-F1 0.230 反 **低** 于"全猜 V" baseline 0.240).

**修正策略**:
1. 主指标改 **macro-F1** (4 类均权), 而非 dom acc (majority-class biased)
2. seg3 标"saturated by majority class" — 不作 paper 主信号
3. **真正 trajectory understanding 信号在 seg2 (chain, GT 均衡)**: B5 macro-F1 0.240 vs all-V 0.176, +6.4pp 真实 gain

下面表格 dom acc / STA / self-consistency 等仍保留, 但**解读要看 macro-F1**.

## Baseline 定义 (累加 structured, 无 CoT)

| | 输入 prompt 内容 |
|---|---|
| **B0** Zero-shot | Q7 scenario + task |
| **B1** +pi | + 设计师人口统计学 |
| **B2** +30维 | + 30 维 SADR polarity persona |
| **B3** +self | + designer 自身 Q1-Q6 选择+回答 |
| **B4** +peer cosine | + 3 个 trait-cosine 相似 peer 的 Q7 同 seg NL |
| **B5** +CAPS quad + peer JS | + 激活的 CAPS 四元组 + 3 个 typology JS-similar peer 同 seg NL |

输出: structured JSON (`dominant_factor` + `state_sequence` + `criterion` + `confidence` + `reasoning_notes` + `nl_answer`)
Per-seg schema: seg1/seg3 = single letter anchor, seg2 = multi-state chain (≥ 2).

## 主指标 — Macro-F1 (修正版, primary)

| Baseline | **Macro-F1** ↑ | Macro-Recall ↑ | dom acc (artifact 警告) | all-V baseline |
|---|---|---|---|---|
| B0 | 0.167 | 0.309 | 19.8% | 0.214 |
| B1 | 0.181 | 0.308 | 21.4% | 0.214 |
| B2 | 0.165 | 0.290 | 19.8% | 0.214 |
| B3 | 0.160 | 0.240 | 21.2% | 0.214 |
| B4 | 0.151 | 0.228 | 19.1% | 0.214 |
| **B5** | **0.281** | 0.273 | 60.9% | 0.214 |

**真正 gap**: B5 macro-F1 0.281 vs B1 0.181 = **+0.10**. 跟 dom acc 看似的 3× ratio 不一致 — 后者是 majority-class artifact.

### Per-seg Macro-F1 (拆解, 真信号定位)

| Seg | GT V 占比 | B0 | B1 | B2 | B3 | B4 | **B5** | **all-V baseline** | **B5 vs all-V** |
|---|---|---|---|---|---|---|---|---|---|
| seg1 (第一反应) | 78% | 0.194 | **0.211** | 0.166 | 0.162 | 0.133 | 0.213 | 0.219 | **−0.006** |
| **seg2 (决策链路)** | 54% | 0.142 | 0.130 | 0.162 | 0.155 | 0.209 | **0.240** | 0.176 | **+0.064** ⭐ |
| seg3 (核心底线) | 93% | 0.055 | 0.108 | 0.042 | 0.094 | 0.042 | 0.230 | 0.240 | **−0.010** |

🎯 **B5 唯一 beat all-V baseline 的 seg 是 seg2** (+6.4pp). seg1 和 seg3 都不如 all-V — GT 太偏 V, "全猜 V" 已是强 baseline.

→ **seg2 (chain segment, 决策权衡) 是 paper main contribution 的 真实证据**.

## Main 完整 metric (overall mean, 11+1 metric, claimed source 主)

| Metric | B0 | B1 | B2 | B3 | B4 | **B5** | Winner |
|---|---|---|---|---|---|---|---|
| **TED** ↓ | 1.564 | 1.570 | 1.558 | 1.481 | 1.611 | **1.284** | **B5** |
| **STA** ↑ | 0.169 | 0.165 | 0.168 | 0.193 | 0.136 | **0.345** | **B5** (1.8× vs B3) |
| PKVC dom acc | 19.8% | 21.4% | 19.8% | 21.2% | 19.1% | **60.9%** | **B5** (3.0×) |
| Path Jaccard | 0.336 | 0.335 | 0.325 | 0.359 | 0.309 | **0.567** | **B5** (1.6×) |
| Path Edit norm | 0.262 | 0.255 | 0.250 | 0.276 | 0.229 | **0.498** | **B5** (1.8×) |
| Path Trans Jacc | 0.147 | 0.139 | 0.142 | 0.165 | 0.110 | **0.323** | **B5** (1.9×) |
| Path LCS F1 | 0.337 | 0.335 | 0.320 | 0.353 | 0.304 | **0.575** | **B5** (1.6×) |
| **Self-Consistency** | 21.6% | 24.4% | 27.1% | 31.5% | 29.3% | **78.0%** ⭐ | **B5** (2.5×) |
| Syntactic | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 | 0.961 | tie |
| Semantic cos | 0.612 | 0.609 | 0.618 | 0.619 | 0.633 | **0.637** | **B5** |
| Pragmatic cos | **0.684** | 0.661 | 0.665 | 0.669 | 0.656 | **0.684** | B0 ≈ B5 |
| Persona cos | 0.534 | 0.543 | 0.552 | **0.556** | 0.546 | 0.549 | B3 |

(PKVC dom acc 用 claimed PKVC 比 GT, B5 60.9% 是因为 seg3 dom acc 跃迁拉高总均值. 见 per-seg.)

## Self-consistency: B5 是唯一"知行合一"

| Baseline | self-consistency = (claimed dom == v8.7 extracted dom) |
|---|---|
| B0 zero-shot | 21.6% |
| B1 +pi | 24.4% |
| B2 +30维 | 27.1% |
| B3 +self | 31.5% |
| B4 +peer cosine | 29.3% |
| **B5 +CAPS+peer JS** | **78.0%** ⭐ |

**B0-B4 model 嘴上说一套 NL 写另一套** (21-32%). B5 让 model "知行合一" — claimed PKVC 跟 NL 里真表达的 PKVC 一致 78%.

**Paper-worthy claim**: CAPS routing 不仅改善 trajectory fidelity, 还让 model 对自己的 cognitive state 有自我认知 (self-aware reasoning).

## Per-seg dom accuracy

| Seg | B0 | B1 | B2 | B3 | B4 | **B5** |
|---|---|---|---|---|---|---|
| seg1 第一反应 | 29.0% | 32.7% | 28.4% | 28.4% | 22.8% | **46.3%** |
| seg2 决策链路 | 22.8% | 21.0% | 25.3% | 23.5% | 29.0% | **51.2%** |
| seg3 核心底线 | 7.4% | 10.5% | 5.6% | 11.7% | 5.6% | **85.2%** |

🔥 **seg3 dom acc**: B0-B4 全卡在 5-12%, B5 跃到 85.2% (8 × better). **CAPS routing 解锁底线段识别**.

## Per-seg STA

| Seg | B0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| seg1 | 0.241 | 0.247 | 0.228 | 0.235 | 0.154 | **0.383** |
| seg2 | 0.220 | 0.223 | 0.247 | **0.267** | 0.236 | 0.203 |
| seg3 | 0.046 | 0.025 | 0.029 | 0.077 | 0.019 | **0.451** |

**B5 在 seg1+seg3 (anchor 段) STA 远超**, 但 **seg2 (chain 段) 反不如 B3**. 解读: B5 让 model 在单状态段精确 commit, 但在多状态 chain 段过度压缩 sequence.

## Per-seg Self-Consistency

| Seg | B0 | B1 | B2 | B3 | B4 | **B5** |
|---|---|---|---|---|---|---|
| seg1 | 13.0% | 19.3% | 21.0% | 26.5% | 29.6% | **51.9%** |
| seg2 | 45.1% | 46.6% | 56.2% | 59.3% | 54.3% | **89.5%** |
| seg3 | 6.8% | 7.5% | 3.7% | 8.6% | 3.7% | **92.6%** |

**seg3 self-cons 跃迁最猛** (B0-B4 ~5%, B5 92.6%). seg3 是 paper main impact 维度.

## Per-seg Path LCS F1

| Seg | B0 | B1 | B2 | B3 | B4 | **B5** |
|---|---|---|---|---|---|---|
| seg1 | 0.311 | 0.325 | 0.287 | 0.309 | 0.246 | **0.461** |
| seg2 | 0.559 | 0.552 | 0.573 | 0.579 | 0.569 | 0.545 |
| seg3 | 0.139 | 0.129 | 0.102 | 0.171 | 0.095 | **0.718** |

类似模式: seg1/3 (anchor) B5 巨大领先, seg2 (chain) 其他 baseline 跟上.

## Ablation: B5 CAPS top-k + OG-RAG peer

| | **B5 main (k=5)** | B5_k0 (all) | **B5_k3** | B5_k8 | B5_ograg |
|---|---|---|---|---|---|
| TED ↓ | 1.284 | 1.297 | 1.284 | **1.267** | 1.292 |
| STA ↑ | 0.345 | 0.317 | **0.359** | 0.355 | 0.347 |
| Path Jaccard | 0.567 | 0.540 | **0.583** | 0.579 | 0.566 |
| Path LCS F1 | 0.575 | 0.557 | **0.596** | 0.595 | 0.578 |
| Self-Consistency | 78.0% | 79.0% | **80.0%** | 79.0% | 76.0% |
| Semantic | 0.637 | 0.634 | 0.634 | **0.637** | 0.629 |
| Pragmatic | 0.682 | 0.658 | 0.673 | 0.670 | **0.684** |
| Persona cos | 0.549 | 0.545 | 0.544 | 0.546 | **0.563** |

### Ablation findings

1. **CAPS top-k=3 是 sweet spot**: STA / Path Jaccard / Path LCS / Self-Consistency 都 best 或 second-best.
2. **k=0 (all CAPS) 反退步**: 信息过载 dilute trajectory 信号 (STA 0.317).
3. **B5_ograg 在 persona/pragmatic 占优**: persona 0.563 vs main 0.549 (+1.4pp), pragmatic 0.684 vs 0.682. Set-cover 检索给 voice 多样性.

## 跟前 free-form vs structured 对比 (旧 4-baseline 数据存档)

| | B1 zero-shot | B1d +demo | B2 +CoT | B3 +RAG cos (旧) |
|---|---|---|---|---|
| 旧 STA (free-form + extract) | 0.383 | 0.403 | 0.345 | 0.389 |
| 新 B0 (new) STA (structured claimed) | 0.169 | 0.165 | 0.169 | 0.193 |

结构化让 STA 看似下降, 实际是 model commit 更保守. 但 **B5 (structured + CAPS) STA = 0.345**, 接近 free-form 旧水平, 同时 self-consistency 高 (78% vs 旧 ~30%) — 这是真正的 trajectory understanding 而非 NL 偶然冒出.

## 关键 finding 总结 (修正版, paper-worthy)

### Finding 1 — B5 真实 gain 在 seg2 (chain segment) macro-F1 +6.4pp
- seg2 GT 4 类均衡 (V 54% / K 21% / P 22% / C 3%), 是干净的 multi-class signal
- B5 macro-F1 0.240 vs all-V baseline 0.176 = **+6.4pp 真实 gain**
- 其他 baseline (B0-B4) seg2 macro-F1 都 0.13-0.21, 在 all-V 附近徘徊
- → **CAPS routing 唯一 demonstrated work 的场景是 seg2 决策链路**

### Finding 2 — seg3 上 B5 高 dom acc 是 majority-class artifact
- GT V=93%, "全猜 V" 就 93% dom acc
- B5 seg3 dom acc 85.2% 看似突出, 实际**低于** 93% baseline
- B5 把 12 个 non-V (P/K/C) 全 claim V, **P/K/C 类 recall = 0%**
- macro-F1 B5=0.230 < all-V=0.240
- → seg3 不能作 paper main signal

### Finding 3 — seg1 (第一反应) 三 baseline 几乎打平 (无显著区分)
- B5 macro-F1 0.213, B1 0.211, B0 0.194 — gap < 0.02
- all-V baseline 0.219 — B5 不如全猜 V
- seg1 GT V=78%, 模型很容易 default V
- → seg1 信号弱, 不作主区分

### Finding 4 — Self-consistency 是 process metric, 看 seg2 才有信号
- 整体 self-cons B5=78%, others 21-32% (gap 大)
- 但 seg3 上 self-cons B5=92.6% 是 "model claim V + NL 写 V" 一致, 不证明真理解
- seg2 self-cons B5=89.5%, others 45-59% — seg2 上的 +30pp 才是真正的 cognitive consistency
- → paper 报 self-cons 要按 seg 拆开, 强调 seg2 数字

### Finding 5 — B4 (+cosine peer 无 CAPS) 比 B3 trajectory 退步
- B3 STA 0.193 → B4 STA 0.136 (-0.057)
- cosine peer (trait-similar) 的 sequence 跟 target 不一定 align
- B5 (CAPS quad + JS-similar peer) 才解锁 trajectory
- → **peer retrieval 必须 CAPS-anchored 才有用**, 仅 cosine 反 hurt

### Finding 6 — Demographic / persona 边际改进有限
- B0 → B4 macro-F1 0.151-0.181, 跟 all-V 0.214 比都不行
- 加 pi (B1) / 30维 (B2) / self (B3) / cosine peer (B4) 都不破 all-V
- 唯一打破 all-V 的是 B5 macro-F1 0.281 → all-V 0.214 (+6.7pp)
- → **CAPS routing framework 是 paper unique value, 不是 prompt-engineering 堆 module**

## 输入 prompt 长度

| Baseline | prompt chars |
|---|---|
| B0 | ~530 |
| B1 | ~750 |
| B2 | ~1870 |
| B3 | ~2240 |
| B4 | ~3160 |
| B5 | ~5520 (含 CAPS quad NL + peer JS NL) |

## 局限 + 下一步

1. **seg2 (chain 段) 是 ceiling**: 所有 baseline STA 都 0.20-0.27, struct schema 限制 multi-state 展开. SFT 训才能突破.
2. **PKVC dom acc 60.9% 是 prompting 上限**: B5 60.9%, 后续 LoRA SFT 应突破 70-80%.
3. **没接 bridge mining**: mapping_miner ready 但没 join 进 retrieval, design feature pending.
4. **Q1-Q6 没跑 baseline**: 仅 Q7 数据.
5. **B5_ograg 跟主 B5 几乎平**: ograg peer 主要 persona/pragmatic 增益 1pp, 不是主战场.

## 产物清单

| 文件 | 内容 |
|---|---|
| **inference data** | |
| `reports/datav3/baselines_q7_main_final_inference.jsonl` | 2916 main (B0-B5) 100% coverage |
| `reports/datav3/baselines_q7_ablation_final_inference.jsonl` | 1944 ablation (B5 variants) 100% |
| **extract (v8.7 PKVC on nl_answer)** | |
| `reports/datav3/baselines_q7_main_final_pkvc_eval.jsonl` | 2913/2916 ok (99.9%) |
| `reports/datav3/baselines_q7_ablation_final_pkvc_eval.jsonl` | 1944/1944 ok |
| **eval (per-pair 13 metric)** | |
| `reports/datav3/baselines_q7_main_final_perpair.jsonl` | 2916 rows × 13 metric fields |
| `reports/datav3/baselines_q7_ablation_final_perpair.jsonl` | 1944 rows × 13 |
| **report** | |
| `reports/datav3/baselines_q7_main_final_metric.md` | main 11+1 主表 + per-seg |
| `reports/datav3/baselines_q7_ablation_final_metric.md` | ablation 主表 |
| **scripts** | |
| `scripts/run_3_baselines_q7.py` | 推理 (6 baseline + ablation 支持) |
| `scripts/eval_baselines_pkvc.py` | v8.7 extract |
| `scripts/eval_baselines_5metric.py` | 11+1 metric (claimed/extracted/both) |
| `scripts/finalize_baselines.py` | retry merge + relabel |

## 一句话 (修正版)

CAPS routing (B5) **在 seg2 决策链路上 macro-F1 0.240 vs all-V baseline 0.176 (+6.4pp 真实 gain)** — 这是 paper 的真实 main contribution. seg1 / seg3 因 GT 严重不平衡 (V 78% / 93%), dom acc 看似的 big jump 是 majority-class artifact; B5 在 seg3 把 12 个 non-V 个体全 claim V, P/K/C 类 recall = 0%, **不能作 paper 主信号**.
