# L4 — T_CAPS quad s_secondary 统计关联 (data_v3, 2026-05-24)

跟 quad pipeline 对应: 本表给 quad schema 的 s_secondary_factor + direction
字段提供 individual-level 统计证据 (factor 跟 option 的显著关联). 互补于 L4c
option_trait_mapping (s_primary 群体级特质). 见 docs/notes/CAPS四元组与簇内RAG_20260523.md.

- factor scores: K = 4, N designer = 163
- (factor, qid, option) 关联探查总数: 92 (sample 量过低 cell 已 filter)
- permutation 次数: 1000
- **significant (p<.05)**: 13 / 92

## strength 分布 (4 档 modulation + bypass calibration 候选)

| strength   |   count |
|:-----------|--------:|
| bypass     |      35 |
| Weak+      |      30 |
| Weak-      |      23 |
| Strong-    |       3 |
| Strong+    |       1 |

## top 20 显著关联 (按 |Cohen's d|)

| factor   | qid   | option   |   n_chose |   n_total |   delta_mean |   cohens_d |   p_perm | strength   | significant   |
|:---------|:------|:---------|----------:|----------:|-------------:|-----------:|---------:|:-----------|:--------------|
| F3       | Q4    | B        |         8 |       163 |       -0.633 |     -0.786 |    0.033 | Strong-    | True          |
| F3       | Q1    | D        |        14 |       163 |       -0.568 |     -0.709 |    0.016 | Strong-    | True          |
| F3       | Q5    | C        |        20 |       163 |       -0.452 |     -0.563 |    0.02  | Strong-    | True          |
| F1       | Q4    | C        |        29 |       163 |        0.395 |      0.465 |    0.019 | Weak+      | True          |
| F3       | Q5    | A        |        28 |       163 |        0.355 |      0.441 |    0.047 | Weak+      | True          |
| F2       | Q8    | C        |        55 |       163 |        0.359 |      0.435 |    0.005 | Weak+      | True          |
| F2       | Q8    | A        |        47 |       163 |       -0.354 |     -0.428 |    0.015 | Weak-      | True          |
| F3       | Q5    | B        |        36 |       163 |       -0.342 |     -0.424 |    0.026 | Weak-      | True          |
| F3       | Q8    | C        |        55 |       163 |        0.338 |      0.422 |    0.014 | Weak+      | True          |
| F3       | Q8    | B        |        61 |       163 |       -0.334 |     -0.417 |    0.013 | Weak-      | True          |
| F1       | Q4    | D        |        48 |       163 |       -0.32  |     -0.377 |    0.026 | Weak-      | True          |
| F4       | Q5    | D        |        79 |       163 |        0.296 |      0.369 |    0.022 | Weak+      | True          |
| F4       | Q8    | C        |        55 |       163 |        0.28  |      0.349 |    0.036 | Weak+      | True          |

## per-factor 显著关联数

| factor   |   n_sig |
|:---------|--------:|
| F1       |       2 |
| F2       |       2 |
| F3       |       7 |
| F4       |       2 |

## per-question 显著关联数

| qid   |   n_sig |
|:------|--------:|
| Q1    |       1 |
| Q3    |       0 |
| Q4    |       3 |
| Q5    |       4 |
| Q6    |       0 |
| Q8    |       5 |

## Verdict

- 显著率 = 14.1%  (moderate)
- 推荐: 跑 L4b LASSO 跟本表做 intersect, 取共同显著 = robust quad s_secondary 候选
- 下游: 这些关联当 quad schema s_secondary 的 prior, blocker A 解后再加 p/o 抽取