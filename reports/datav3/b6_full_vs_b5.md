# B5 vs B6_v1 vs B6_v2 — Full 162 designer comparison

Common designer pool: 162 (all 3 segs each).

## Overall macro-F1 (all 3 segs combined)

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.281 | 0.273 | 60.9% |
| B6_v1 | 0.169 | 0.275 | 21.8% |
| B6_v2 | 0.270 | 0.336 | 47.3% |

## Per-seg macro-F1

### Q7_seg1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.213 | 0.239 | 46.3% |
| B6_v1 | 0.087 | 0.256 | 19.1% |
| B6_v2 | 0.247 | 0.330 | 49.4% |

### Q7_seg2

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.240 | 0.267 | 51.2% |
| B6_v1 | 0.197 | 0.233 | 25.9% |
| B6_v2 | 0.248 | 0.252 | 36.4% |

### Q7_seg3

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.230 | 0.230 | 85.2% |
| B6_v1 | 0.096 | 0.225 | 20.4% |
| B6_v2 | 0.198 | 0.400 | 56.2% |

## seg3 per-class P/R/F1

### B5
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.000 | 0.000 | 0.000 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 0.920 | 0.920 | 0.920 | 150 |
| C | 0.000 | 0.000 | 0.000 | 1 |

Macro-F1: **0.230**, Macro-Recall: **0.230**

### B6_v1
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.042 | 0.714 | 0.079 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 0.848 | 0.187 | 0.306 | 150 |
| C | 0.000 | 0.000 | 0.000 | 1 |

Macro-F1: **0.096**, Macro-Recall: **0.225**

### B6_v2
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.000 | 0.000 | 0.000 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 0.938 | 0.600 | 0.732 | 150 |
| C | 0.031 | 1.000 | 0.061 | 1 |

Macro-F1: **0.198**, Macro-Recall: **0.400**

## seg3 prediction distribution

| Baseline | Pred=P | Pred=K | Pred=V | Pred=C |
|---|---|---|---|---|
| B5 | 2 (1%) | 0 (0%) | 150 (93%) | 10 (6%) |
| B6_v1 | 120 (74%) | 6 (4%) | 33 (20%) | 3 (2%) |
| B6_v2 | 10 (6%) | 24 (15%) | 96 (59%) | 32 (20%) |
