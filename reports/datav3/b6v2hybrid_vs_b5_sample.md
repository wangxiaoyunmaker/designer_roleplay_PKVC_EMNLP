# B6_v2hybrid sample vs B5 (same 20 designer) macro-F1

Sample size: 20 designer × 3 seg = 60 calls per baseline. B5 source: main_final.jsonl.

## Overall macro-F1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.178 | 0.209 | 36.7% |
| B6_v2hybrid | 0.196 | 0.227 | 36.7% |

## Per-seg macro-F1

### Q7_seg1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.415 | 0.394 | 40.0% |
| B6_v2hybrid | 0.455 | 0.419 | 45.0% |

### Q7_seg2

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.140 | 0.219 | 35.0% |
| B6_v2hybrid | 0.161 | 0.188 | 25.0% |

### Q7_seg3

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.130 | 0.219 | 35.0% |
| B6_v2hybrid | 0.143 | 0.250 | 40.0% |

## seg3 per-class P/R/F1 (关键: 看 B6 能否拉起 P/K/C recall)

GT in sample: 7 P + 4 K + 1 C + 8 V = 20

### B5
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.000 | 0.000 | 0.000 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 0.368 | 0.875 | 0.519 | 8 |
| C | 0.000 | 0.000 | 0.000 | 1 |

Macro-F1: **0.130**, Macro-Recall: **0.219**

### B6_v2hybrid
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.000 | 0.000 | 0.000 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 0.400 | 1.000 | 0.571 | 8 |
| C | 0.000 | 0.000 | 0.000 | 1 |

Macro-F1: **0.143**, Macro-Recall: **0.250**

## Per-designer seg3 prediction (sorted by GT class)

| Designer | GT | B5 pred | B6_v2hybrid pred | B5 ✓? | B6_v2hybrid ✓? |
|---|---|---|---|---|---|
| 1078765532 | P | V | V | ✗ | ✗ |
| 695475971 | P | V | V | ✗ | ✗ |
| BuzhichunD | P | V | V | ✗ | ✗ |
| F29293918 | P | V | V | ✗ | ✗ |
| Vivian11320 | P | V | V | ✗ | ✗ |
| acho20050223 | P | V | V | ✗ | ✗ |
| faymeng666 | P | V | V | ✗ | ✗ |
| AyiWe4Real | K | V | V | ✗ | ✗ |
| Hreiy888 | K | V | V | ✗ | ✗ |
| cccclare6354 | K | V | V | ✗ | ✗ |
| kuronim | K | V | V | ✗ | ✗ |
| Yyd3144117584 | C | V | V | ✗ | ✗ |
| Jimmy_Fang18 | V | V | V | ✓ | ✓ |
| WUXY0953 | V | V | V | ✓ | ✓ |
| WuShanYuXi | V | C | V | ✗ | ✓ |
| a123164981 | V | V | V | ✓ | ✓ |
| mz18757140975 | V | V | V | ✓ | ✓ |
| ww2001811920 | V | V | V | ✓ | ✓ |
| wxid_negbygwl2 | V | V | V | ✓ | ✓ |
| wxid_wqyptcofe | V | V | V | ✓ | ✓ |
