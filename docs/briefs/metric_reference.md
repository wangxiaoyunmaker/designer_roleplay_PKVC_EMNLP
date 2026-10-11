# Metric Reference: 11 Final Metrics + Deprecated Alternatives

> 2026-05-26 final.
> Coverage: 11 paper-locked metrics + 12 deprecated/supplementary metrics with rationale.
> 配套 bootstrap 95% CI (1000-iter, designer-level resample, percentile method).

---

## 0. Quick reference (11 final metrics)

| # | Metric | Dim | Seg | Direction | Captures |
|---|---|---|---|---|---|
| 1 | macro-TED | Fidelity | seg2 | ↓ | Chain edit distance to GT, class-balanced |
| 2 | macro-LCS-F1 | Fidelity | seg2 | ↑ | Chain longest subsequence overlap, class-balanced |
| 3 | non-V recall | Fidelity | seg2 | ↑ | Correctly predict non-V first-letter |
| 4 | Chain diversity | Anti-collapse | seg2 | ↑ | 1 − modal chain frequency |
| 5 | NL specificity (seg1) | Personalization | seg1 | ↑ | NL is designer-specific vs generic |
| 6 | Persona-NL alignment | Personalization | seg1 | ↑ | Designer style match (Q1-Q6 NL center) |
| 7 | Trait-pred Spearman ρ | Personalization | seg1 | ↑ | 30维 trait → dom prediction correlation |
| 8 | NL specificity (seg2) | Personalization | seg2 | ↑ | Same as #5 but on seg2 NL |
| 9 | NL specificity (seg3) | Personalization | seg3 | ↑ | Same as #5 but on seg3 NL |
| 10 | Criterion uniqueness | Anti-collapse | seg3 | ↑ | 1 − mean pairwise cos of criterion embeddings |
| 11 | Criterion-persona alignment | Personalization | seg3 | ↑ | criterion 跟 designer 个人风格 cosine |

3 大维度:
- **Fidelity (路径准度)**: 3 metric (seg2 only)
- **Personalization (个性化)**: 6 metric (3 NL specificity × 3 seg + 3 alignment metrics)
- **Anti-collapse (差异化)**: 2 metric (chain diversity + criterion uniqueness)

---

## 1. Fidelity Metrics (seg2 chain segment only)

seg2 是 chain reasoning segment, state sequence 长度 2-7. Fidelity 测 pred chain vs GT chain 距离.

### 1.1 macro-TED ↓ (Tree Edit Distance, class-balanced)

**Definition**: 跨 P/K/V/C 4 类 GT first-letter 分组, 各组内取 edit distance 均值, 再 macro 平均.

**Purpose**: 测 chain accuracy. Class-balanced 避免 V-majority dominance (seg2 V-first 设计师占 54%).

**Formula**:
```
TED(a, b) = Levenshtein edit distance (insertion / deletion / substitution = 1)

macro_TED = (1/|S|) × Σ_{c ∈ S} mean_{i ∈ class c} TED(gt_seq_i, pred_seq_i)

where S = {P, K, V, C} ∩ {classes with non-zero support}
```

**Implementation** (`scripts/final_eval_with_bootstrap.py::m_macro_ted`):
```python
def edit_distance(a, b):  # standard DP
    n, m = len(a), len(b); dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def m_macro_ted(d, sample):
    groups = defaultdict(list)
    for idx in sample:
        groups[d["Q7_seg2_gt_first"][idx]].append(d["seg2_ted"][idx])
    classes = [c for c in "PKVC" if c in groups and groups[c]]
    return float(np.mean([np.mean(groups[c]) for c in classes]))
```

**Caveat**: TED 跟 sequence 长度有关 (长 seq 自然 TED 大); GT 大部分 length 2-4, pred 也类似, 故公平.

### 1.2 macro-LCS-F1 ↑ (Longest Common Subsequence F1)

**Definition**: 跨 P/K/V/C 4 类 macro 平均 LCS-F1.

**Purpose**: 测 chain 子序列重合度 (不要求 contiguous). 跟 TED 互补 — LCS 看保留多少, TED 看改了多少.

**Formula**:
```
LCS_len(a, b) = longest common subsequence length (standard DP)
LCS_precision = LCS_len(a, b) / |b|       # pred 中 GT 序列保留比例
LCS_recall    = LCS_len(a, b) / |a|       # GT 中被 pred 覆盖比例
LCS_F1 = 2 × prec × rec / (prec + rec)
macro_LCS_F1 = mean over P/K/V/C classes
```

**Implementation**:
```python
def lcs_length(a, b):
    n, m = len(a), len(b); dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def lcs_f1(a, b):
    L = lcs_length(a, b)
    if not a or not b: return 0.0
    prec, rec = L/len(b), L/len(a)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
```

### 1.3 non-V recall ↑ (non-V first-letter recall)

**Definition**: GT first-letter ≠ V 的 designer 中, pred first-letter == GT first-letter 的比例.

**Purpose**: 直接测 model 是否能识别非 V 类 designer (不被 V-default 收集). 在 V-majority 数据上是关键 role-play 信号.

**Formula**:
```
non_V_recall = #{i: GT_first_i ≠ V & pred_first_i == GT_first_i} / #{i: GT_first_i ≠ V}
```

**Implementation**:
```python
def m_non_v_recall_seg2(d, sample):
    gt_first = d["Q7_seg2_gt_first"]; pred_first = d["seg2_pred_first"]
    num, den = 0, 0
    for idx in sample:
        if gt_first[idx] != "V":
            den += 1
            if pred_first[idx] == gt_first[idx]: num += 1
    return num / den if den > 0 else 0.0
```

**Why** (specifically chosen): GT seg2 first-letter 分布: V 88, P 38, K 42, C 1 (P+K+C = 81 非 V). V-majority 让 dom acc 给 V-default 假高分; non-V recall 直接看 model 区分能力.

---

## 2. Personalization Metrics (个性化)

### 2.1 NL specificity ratio ↑ (3 segments)

**Definition**: 设计师 i 的 pred_NL_i 跟自己 GT_NL_i 的 cosine 相似度, 跟自己 vs 其他 designer GT_NL_j 平均 cosine 的比值.

**Purpose**: 测 pred 是否真"为这个 designer 量身定做" (> 1 = designer-specific, ≈ 1 = generic).

**Formula**:
```
sim(pred_i, GT_j) = cosine(emb(pred_NL_i), emb(GT_NL_j))   # text-embedding-3-small, 1536-dim

specificity_i = sim(pred_i, GT_i) / mean_{j ≠ i}(sim(pred_i, GT_j))

mean_NL_specificity = (1/n) × Σ_i specificity_i
```

**Implementation** (`m_nl_specificity`):
```python
def m_nl_specificity(d, sample, seg):
    pred_gt_cos = d[f"{seg}_pred_gt_cos"]  # precomputed n×n matrix
    n = len(sample)
    specs = []
    for k, idx_k in enumerate(sample):
        own = pred_gt_cos[idx_k, idx_k]
        others_sum = sum(pred_gt_cos[idx_k, sample[l]] for l in range(n) if l != k)
        others_mean = others_sum / (n - 1)
        if others_mean > 0: specs.append(own / others_mean)
    return float(np.mean(specs))
```

**Caveat**: 自动归一化但仍取决于 NL embedding 模型. 用 `text-embedding-3-small` (1536-dim).

### 2.2 Persona-NL alignment ↑ (seg1)

**Definition**: pred_NL_i 跟 designer i 的 "persona center" (Q1-Q6 NL 平均 embedding) 的 cosine.

**Purpose**: 测 NL 风格是否 match designer 个人写作 (词汇 / 句法 / 价值表达).

**Formula**:
```
persona_center_i = mean over Q1-Q6 NL of (emb(NL))   # 1536-dim
persona_align_i = cosine(emb(pred_NL_i), persona_center_i)
mean_persona_align = (1/n) × Σ_i persona_align_i
```

**Implementation**:
```python
# Pre-compute persona center per designer
persona_center = {did: mean(embeddings of Q1-Q6 NL)}

def m_persona_align(d, sample, seg):
    arr = d[f"{seg}_pred_persona_cos"]
    return float(np.mean([arr[idx] for idx in sample]))
```

### 2.3 Trait-prediction Spearman ρ ↑ (seg1)

**Definition**: 跨 designer pairs (i, j), Spearman correlation 测 "30 维 trait 相似度" 跟 "dominant_factor 匹配" 的相关性.

**Purpose**: 测 model 是否真在用 trait 信号决定 dom prediction. 高 ρ = 相似 trait → 相似 prediction (role-play 核心).

**Formula**:
```
trait_sim(i, j) = cosine(trait_30d_i, trait_30d_j)         # 30-dim raw Likert
pred_match(i, j) = 1 if pred_dom_i == pred_dom_j else 0    # binary

trait_pred_rho = Spearman(trait_sim_pairs, pred_match_pairs)
                 over n(n-1)/2 ≈ 13041 pairs (n=162)
```

**Implementation**:
```python
from scipy.stats import spearmanr

def m_trait_pred_rho(d, sample, seg):
    trait_cos = d["trait_cos"]; pred_dom = d[f"{seg}_pred_dom"]
    n = len(sample); t_sims = []; matches = []
    for k in range(n):
        for l in range(k+1, n):
            i, j = sample[k], sample[l]
            t_sims.append(trait_cos[i, j])
            matches.append(1.0 if pred_dom[i] == pred_dom[j] else 0.0)
    if not t_sims or np.std(t_sims) == 0 or np.std(matches) == 0:
        return 0.0
    rho, _ = spearmanr(t_sims, matches)
    return float(rho) if not np.isnan(rho) else 0.0
```

**Caveat (seg3)**: V-majority 让大部分 pair pred_match=1 (both V) → matches 方差近零 → Spearman 不可靠. 故 **只 seg1 用** (seg1 V-first 78%, 仍 borderline; seg3 V-first 93% 完全废).

### 2.4 Criterion-persona alignment ↑ (seg3)

**Definition**: pred 的 criterion 短句嵌入跟 designer persona center 的 cosine.

**Purpose**: 测底线短句是否反映 designer 个人风格 (vs 通用 "保护银发族" 等).

**Formula**:
```
criterion_persona_align_i = cosine(emb(criterion_i), persona_center_i)
mean_crit_persona = (1/n) × Σ_i criterion_persona_align_i
```

**Implementation**:
```python
# Pre-compute criterion embedding + persona center
data["crit_persona_cos"][i] = cosine(crit_emb_i_normalized, persona_emb_i_normalized)

def m_criterion_persona(d, sample):
    return float(np.mean([d["crit_persona_cos"][idx] for idx in sample]))
```

---

## 3. Anti-collapse Metrics (差异化, 抗坍缩)

### 3.1 Chain diversity ↑ (seg2)

**Definition**: 1 − (frequency of the most common predicted chain).

**Purpose**: 量化 model 是否给所有 designer 输出同一个 chain (mode collapse), 还是真在 differentiate.

**Formula**:
```
chain_counter = Counter(tuple(pred_seq_i) for i in sample)
modal_freq = chain_counter.most_common(1)[0][1] / n
chain_diversity = 1.0 - modal_freq
```

**Implementation**:
```python
def m_chain_diversity(d, sample):
    pred_seq = d["Q7_seg2_pred_seq"]
    chain_counter = Counter(tuple(pred_seq[idx]) for idx in sample)
    n = sum(chain_counter.values())
    modal = chain_counter.most_common(1)[0][1] / n
    return 1.0 - modal
```

**Range**: [0, 1−1/n]. 高 = 差异化, 低 = mode collapse.
- v2_anchor on GPT-4o: 0.821 (top chain 频率 17.9%)
- B0_h on GPT-4o: 0.660 (top chain 频率 34%, 1/3 designer 同 chain)

### 3.2 Criterion uniqueness ↑ (seg3)

**Definition**: 1 − mean pairwise cosine of criterion embeddings 跨 162 designers.

**Purpose**: 测 162 个 designer 的底线短句是否真"每人不同". 高 = differentiated, 低 = 大家说一样的话.

**Formula**:
```
crit_cos_matrix[i,j] = cosine(emb(criterion_i), emb(criterion_j))

uniqueness = 1 − mean over upper triangle (i<j) of crit_cos_matrix[i,j]
```

**Implementation**:
```python
def m_criterion_uniqueness(d, sample):
    crit_cos = d["crit_cos"]    # precomputed n×n
    n = len(sample)
    sims = []
    for k in range(n):
        for l in range(k+1, n):
            sims.append(crit_cos[sample[k], sample[l]])
    return 1.0 - float(np.mean(sims))
```

**Why** "跨设计师 pairwise" 而不是单 designer: criterion 是 designer-level output, 必须看 designer 之间是否 differentiated.

---

## 4. Bootstrap CI Methodology

### 4.1 Setup

```
N = 162 (designer pool)
B = 1000 (bootstrap iterations)
seed = 42
```

### 4.2 Resampling

```python
rng = np.random.default_rng(seed)
for b in range(B):
    # Sample n designer indices with replacement, position-based
    sample = rng.integers(0, n, size=n).tolist()
    for metric_fn in METRICS:
        results[metric].append(metric_fn(d, sample))
```

**Position-based**: 同一 designer 可能多次出现, 当作不同 position 处理. Pair-wise metric (NL specificity, Trait-pred ρ, Criterion uniqueness) 在 bootstrap sample 内**重算 pair structure**, 不复用原 pair.

### 4.3 CI computation

**Percentile method**:
```python
CI_lo = np.percentile(values, 2.5)
CI_hi = np.percentile(values, 97.5)
```

### 4.4 Significance test (CI overlap)

Two baselines A and B 在某 metric 上 sig 不同 iff 95% CI 不重叠.

For ↑ better metric:
```python
sig = "★" if (B["ci_lo"] > A["ci_hi"]) else ""  # B sig better
sig = "★" if (B["ci_hi"] < A["ci_lo"]) else ""  # B sig worse
```

For ↓ better metric (e.g. macro-TED):
```python
sig = "★" if (B["ci_hi"] < A["ci_lo"]) else ""  # B sig better (lower)
sig = "★" if (B["ci_lo"] > A["ci_hi"]) else ""  # B sig worse (higher)
```

**No Bonferroni correction**: 11 metrics 跨 3 个**正交维度** (fidelity / personalization / anti-collapse), per-metric CI 独立 informative.

---

## 5. Embedding-based Metrics 技术 notes

### 5.1 Embedding model

`text-embedding-3-small` (OpenAI), 1536-dim.

### 5.2 Cosine similarity

```python
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na * nb) else 0.0
```

L2-normalize 后 matrix product 更快:
```python
embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-9)
cos_matrix = embs_n @ embs_n.T   # n×n
```

### 5.3 Cache

File: `data_v3/processed/embedding_cache_text_embed_3_small.jsonl`
每行: `{"key": md5_hex_of_text, "vec": [...]}`

加载/检查/embed missing helper: `scripts/final_eval_with_bootstrap.py::load_cache + embed_missing`.

### 5.4 Pre-compute matrices (per baseline)

```python
data["trait_cos"]               = 30维 trait n×n cosine matrix
data[f"{seg}_pred_gt_cos"]      = pred_NL × GT_NL n×n matrix
data[f"{seg}_pred_persona_cos"] = pred_NL × persona_center 1D array (n,)
data["crit_cos"]                = criterion × criterion n×n matrix
data["crit_persona_cos"]        = criterion × persona_center 1D array
```

Bootstrap 1000 iterations 只需 indexing, 不重算 cosine.

---

## 6. Deprecated / Supplementary Metrics

下面 metric 在迭代中考虑过, 最后**未进入 final 11 个**. 列出 + 弃用理由.

### 6.1 macro-F1 (DEPRECATED for seg3)

**Definition**: Per-class P/R/F1 取平均.

**Why deprecated on seg3**: GT V=92.6%, P/K/C 各 1-7 samples, per-class F1 噪声大 (e.g., 1 wrong P pred 让 P F1 跳大). 数据集不平衡使 macro-F1 unreliable.

**Where kept**: seg1 (V=78%, P=17%) 上 macro-F1 仍 borderline 可用 (但 P/K/C class support 仍小).

### 6.2 dom acc (DEPRECATED, V-majority artifact)

**Definition**: pred_dom == gt_dom 的比例.

**Why deprecated**: seg3 V=93%, V-default model trivially 拿 93% acc. **不反映 role-play**.

**Where kept**: 作为 auxiliary number, 加 caveat.

### 6.3 Distribution KL (DEPRECATED)

**Definition**: KL(pred_distribution || GT_distribution) 跨 162 designer 的 dom prediction 分布.

**Why deprecated**: V-default model 输出 100% V → KL(全 V || 93% V) 极低 (trivial). 无区分度.

### 6.4 Prediction entropy ↑ (DEPRECATED on seg3)

**Definition**: Shannon entropy of pred PKVC distribution.

**Why deprecated on seg3**: GT entropy = 0.477 (V 集中). 正确 model 应 entropy 低. 高 entropy 反而是错的 (over-differentiation). **方向矛盾**.

**Where kept**: seg2 chain segment 上 chain diversity 替代它 (chain-level entropy 比 dom-level entropy 更细).

### 6.5 STA — State Transition Accuracy (DEPRECATED, V-bias)

**Definition**: # GT transitions that appear in pred / # GT transitions.

**Why deprecated**: B0_h V-default 输出 [V] 或 [V, V] 容易 cover V→V 等常见 GT transitions, trivially 高分. Macro-TED 跟 LCS-F1 已替代.

### 6.6 Path Jaccard / Path Trans Jaccard (DEPRECATED in main, kept in supp)

**Definition**: Set-based Jaccard on state set / transition set.

**Why deprecated**: 跟 macro-LCS-F1 / macro-TED 高度相关 (3 个 metric 都测 chain overlap). 选 2 个互补即可.

### 6.7 Syntactic / Pragmatic cosine (DEPRECATED)

**Definition**: Cosine on 5-dim NL stats / 4-dim pragmatic marker density.

**Why deprecated**:
- Syntactic: 不区分 baseline (统一约束 200-400 chars → syntactic stats 都 high cosine)
- Pragmatic: 跟 NL specificity 高度重叠

### 6.8 Semantic cosine (DEPRECATED in main, included in macro-semantic but cut)

**Definition**: cosine of `simulated_nl` vs `ground_truth_nl` via embedding.

**Why dropped**: 用 NL specificity ratio 更强 (相对 vs 绝对). Semantic 只看 own pred 跟 own GT, specificity 加了"vs 其他 designer" denominator.

### 6.9 Supplement reference cosine (DEPRECATED)

**Definition**: cosine of `simulated_nl` vs designer 的 Q3-Q6 supplement 拼接.

**Why deprecated**: 跟 Persona-NL alignment 高度重叠 (都是 cosine with designer 个人文本). 选 Persona-NL (更广泛: Q1-Q6 NL, 不仅 Q3-Q6).

### 6.10 macro-semantic cosine (DEPRECATED in main)

**Definition**: 按 GT class 分组取 mean semantic cosine 再 macro.

**Why dropped**: 跟 NL specificity 重叠. NL specificity 已是 designer-level personalization signal.

### 6.11 Per-designer specificity (Jaccard-based) (DEPRECATED on seg2)

**Definition**: cosine ratio 用 state-set Jaccard 替代 NL cosine, 测 pred chain 是否 specifically match own GT chain.

**Why deprecated on seg2**: Jaccard is coarse (set-based, not order-aware) — 大部分 designer mean spec ≈ 1, 区分度低.

### 6.12 Macro-STA (考虑过但未实现)

**Definition**: 按 GT class macro 平均 STA.

**Why deprecated**: macro-TED + macro-LCS-F1 已覆盖 chain accuracy 双角度. Macro-STA 跟它们正相关, 信息冗余.

---

## 7. Why the 11 Final Metrics (设计 rationale)

### 7.1 Per-segment 配置 design

| Seg | Metric | 理由 |
|---|---|---|
| seg1 | NL specificity, Persona-NL, Trait-pred ρ | seg1 是 single dominant + NL, path metric 退化. Personalization 是 paper main signal |
| seg2 | macro-TED, macro-LCS-F1, non-V recall, Chain diversity, NL specificity | seg2 chain segment, fidelity + anti-collapse + personalization 3 维度都关键 |
| seg3 | NL specificity, Criterion uniqueness, Criterion-persona | seg3 是 criterion 段, criterion-specific metrics 主导; dom-level metric 因 V-majority 废弃 |

### 7.2 3 维度独立性

- **Fidelity** (path 准): macro-TED, macro-LCS-F1, non-V recall
- **Personalization** (个性化): NL specificity (×3 segs), Persona-NL, Trait-pred ρ, Criterion-persona
- **Anti-collapse** (差异化): Chain diversity, Criterion uniqueness

3 维度的 correlation 设计上独立 (一个 model 可在某维度强某维度弱). 故 no Bonferroni correction.

### 7.3 V-majority bias resistance

- Macro-prefix (按 GT class 平均): macro-TED, macro-LCS-F1 — 平均 P/K/V/C 4 类得分, V majority 不主导
- Non-V recall: 直接 focus 非 V designers
- Chain diversity, Criterion uniqueness: pred-side metric, 不依赖 GT distribution
- NL specificity: ratio-based, 自动归一化
- Persona-NL, Criterion-persona: cos with designer 个人 NL, 不依赖 V-bias

---

## 8. 一句话总结

```
11 个 final metric (3 维度 × per-segment 配置), 1000-bootstrap 95% CI,
全 V-majority bias-resistant. Embedding 用 text-embedding-3-small (1536-dim) +
local cache. 实现见 scripts/final_eval_with_bootstrap.py:: m_macro_ted /
m_macro_lcs / m_non_v_recall_seg2 / m_chain_diversity / m_nl_specificity /
m_persona_align / m_trait_pred_rho / m_criterion_uniqueness / m_criterion_persona.
```
