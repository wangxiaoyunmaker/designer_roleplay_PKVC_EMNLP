# data_v3 Statistical Observation — SUMMARY (2026-05-24)

N = 163 designers, 30 量表 + Q1-Q8 (Q2 排序 / Q7 NL / Q8 单选)

## Layer verdicts

| Layer | Headline | Verdict |
|---|---|---|
| L1 health | N=163, 30 量表完整, PKVC imbalance ratio=3.1× (C:K vs V:P) | PASS |
| L1b expansion | coverage=309/978 (31.6%), PASS (扩写选了短的, 符合预期) | INFO |
| L2 EFA | KMO=0.60, Bartlett p<1e-41, **PA 推荐 K=4** | PASS |
| L3 cluster (4-d EFA) | K-sweep max silhouette=0.199, picked K=3, 累计 2 sig | **WARN — cluster 弱区分** |
| L3-multi (3 source spaces) | best_sil/χ²sig: 30d_trait=0.07/2, Q_option_behavior=0.11/6, joint_trait_plus_behavior=0.06/3 | **WARN — 30-d 比 4-d 还差, behavior 略好但 gap=负** |
| L3-GMM (behavior) | BIC monotonic=YES (over-fit), BIC-K=10, silhouette-K=4 | **WARN — Gaussian-on-binary 不对路, behavior = sparse fingerprint 不是 clean mode** |
| L4 quad s_secondary | 13/92 (factor,q,opt) associations sig (14%) | PASS |
| L4b LASSO (4-factor) | **13/13 L4 sig agreed** with LASSO 同号 | PASS |
| L4c LASSO (30-item) | **22 段 Statistical Observation** (CPO Compiler Prompt 格式), 28 个 item-level marginal p<.05 | INFO |
| L5 bridge | **13/13 robust** (boot>0.9 + covariate drop<30%), 0 被 demographic 杀 | PASS |
| L7 audit | wechat_dup=0, 完整 Q1-Q8 user = 163/163 | PASS |

## 3 个独立 K — 解耦, 数值相等是巧合不是对齐

| 维度 | 值 | 数据来源 | 用途 |
|---|---|---|---|
| **EFA factor count** K_EFA | 4 | 30 题量表 + Horn parallel analysis | L_cls head 维度 + L4 mapping 的 source factor 空间 |
| **K-means K** K_clust | 3 | 163 designer 在 trait-factor 空间聚类 (L3 任务导向规则: fallback: 无 K 同时满足 (min_n>=20, sig_Q>=2, sil knee within 0.02); 取 silhouette 最优 K=3) | cluster conditioning + 簇内 peer RAG scope (注: L3 显示 cluster 弱区分, 下游价值待评估) |
| **PKVC state count** K_state | 4 | **理论锁** (Mischel CAPS → P/K/V/C state machine), **不是 data-driven** | runtime state machine target + L4 v2 target (extractor 解锁后用) |

关键 framing (brief v3.5.1 §3):
- CAPS 是 *理论* (Mischel 1995 connectionist network), **不是** 30 题量表
- 30 题量表测的是 CAPS network weights (static individual difference), EFA factor count = 这些 weights 的潜变量结构
- PKVC 是 *runtime 状态机*, **不是** trait scale, **不是** 输出 behavior
- K_EFA=4 跟 K_state=4 数值相等是巧合; 即使 PA 出 K_EFA=3 或 5, PKVC 仍然 4 state, 不冲突
- K_clust=3 跟 K_EFA / K_state 也无关, 是第 3 个独立量

## Decisions unlocked (brief v3.5.1 §8.4 / §8.5)

- **N final**: 200 → **163**, brief §1 + sft_pipeline/__init__.py + 写稿 grep 全部统一
- **D-18 EFA factor count final**: PA 推荐 K_EFA = 4, 不再等 200 入袋 (跟 PKVC 4 state 无关, 两者独立)
- **K cluster**: L3 picked K = 3, 但 cluster 在 Q-option 上 χ² 极弱 (累计 2 sig)
- **D-15 (per-cluster vs global mapping)**: cluster 弱区分 → **global mapping main**, per-cluster sensitivity (推荐翻 D-15 default)
- **bypass / strength 阈值**: data-driven calibration (L5 §5.4): bypass<0.06, Weak<0.28, Strong>=0.28
- **blocker E**: L1b 显示 expansion 已 309 行 (selection bias PASS, median 102 字), blocker 实质已解 → 改 CLAUDE.md §5
- **4 层 cluster conditioning**: 经验基础不牢 (L3 + L3-multi + L3-GMM + L3-visual), framing 改 trait-continuous + behavior-fingerprint (quad ontology 不变, 见 docs/notes/CAPS四元组与簇内RAG_20260523.md)
- **T_CAPS quad pipeline 进度**: s_primary (L4c option_trait_mapping.json) + s_secondary (L4 quad_s_secondary_associations.jsonl) + p_modulation 阈值 (L5) 已就绪; ⏳ p_surface + o_primary LLM 抽待 blocker A

## 输出文件 (reports/datav3/, gitignored — 留本地)

- L1_health.md, L1b_expansion_audit.md
- L2_efa.md, L3_cluster.md
- L4_mapping.md, L4b_lasso_check.md
- L5_bridge.md, L7_audit.md
- SUMMARY.md (this file)
- per_designer/ (gitignored): factor_scores_k{3,4,5,6}.csv, loadings_k{3,4,5,6}.csv, user_clusters_k{3,4,5,6,7}.csv, **quad_s_secondary_associations.jsonl** (L4), **option_trait_mapping.json** (L4c, quad Step 2a 标准 schema)

## 下一步建议

1. 跟一作对齐 L1b expansion subset bias + L2/L3 推荐 K + L3 cluster 弱区分发现
2. 把 L4 + L5 通过的 13 个 robust quad s_secondary association 当 paper §4 main set; blocker A 解后跑 quad pipeline Step 2b/c 生成 800 完整 quad
3. 跟老师同步:
   - D-15 翻案 (per-cluster → global mapping)
   - 4 层 cluster conditioning 必要性重审 (cluster χ² 弱)
   - K cluster 选 K=3 还是去掉 cluster conditioning
4. update CLAUDE.md §5 blocker E (已解) + brief v3.5.1 §1 数字 (200→163) + 加 v3.5.2 升级