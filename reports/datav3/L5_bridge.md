# L5 — Bridge stats for quad s_secondary associations (data_v3, 2026-05-24)

- 输入: 13 个 L4 显著 quad s_secondary association

跟 quad pipeline 对应: 本表验证 quad s_secondary 候选的 robustness (bootstrap
sign stability + demographic partial-out + inhibit/excite asymmetry + bypass
threshold data-driven calibration). 通过的 association → quad s_secondary 主集.

## 5.1 / 5.2 / 5.3 — per-triple bridge stats

| factor   | qid   | option   |   raw_d |   partial_d |   d_dropped_pct |   boot_stability |   top_q_rate |   bot_q_rate |   asym |
|:---------|:------|:---------|--------:|------------:|----------------:|-----------------:|-------------:|-------------:|-------:|
| F1       | Q4    | C        |   0.465 |       0.42  |             9.8 |            0.97  |        0.268 |        0.098 |  0.171 |
| F1       | Q4    | D        |  -0.377 |      -0.315 |            16.4 |            0.985 |        0.22  |        0.439 | -0.22  |
| F2       | Q8    | A        |  -0.428 |      -0.464 |            -8.5 |            0.992 |        0.195 |        0.341 | -0.146 |
| F2       | Q8    | C        |   0.435 |       0.439 |            -1   |            0.994 |        0.512 |        0.244 |  0.268 |
| F3       | Q1    | D        |  -0.709 |      -0.718 |            -1.3 |            0.979 |        0.049 |        0.244 | -0.195 |
| F3       | Q4    | B        |  -0.786 |      -0.799 |            -1.7 |            0.977 |        0     |        0.122 | -0.122 |
| F3       | Q5    | A        |   0.441 |       0.421 |             4.6 |            0.995 |        0.244 |        0.073 |  0.171 |
| F3       | Q5    | B        |  -0.424 |      -0.41  |             3.2 |            0.978 |        0.146 |        0.366 | -0.22  |
| F3       | Q5    | C        |  -0.563 |      -0.561 |             0.4 |            0.986 |        0.098 |        0.171 | -0.073 |
| F3       | Q8    | B        |  -0.417 |      -0.393 |             5.8 |            0.995 |        0.293 |        0.585 | -0.293 |
| F3       | Q8    | C        |   0.422 |       0.397 |             5.9 |            1     |        0.463 |        0.171 |  0.293 |
| F4       | Q5    | D        |   0.369 |       0.335 |             9.2 |            0.988 |        0.585 |        0.39  |  0.195 |
| F4       | Q8    | C        |   0.349 |       0.344 |             1.5 |            0.985 |        0.415 |        0.195 |  0.22  |

## Robustness summary

- **robust 通过** (boot>0.9 且 covariate drop<30%): **13** / 13
- **被 covariate 几乎杀掉** (d_dropped >50%): 0

## 5.4 Bypass / strength threshold calibration

- |d| 分布 (across all 92 探查 triples): 25%=0.055, 50%=0.140, 75%=0.283, 90%=0.424
- 当前 mapping.py 阈值: bypass<0.10, Weak<0.50, Strong>=0.50
- **建议 (data-driven)**: bypass<0.06, Weak<0.28, Strong>=0.28

## Verdict

- L4 sig 且 L5 robust 通过的 association = T_CAPS quad s_secondary 主集候选 (paper main)
- 被 covariate 杀掉的 association 写 paper §4 caveat