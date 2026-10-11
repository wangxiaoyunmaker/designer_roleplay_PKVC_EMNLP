# B6v2 sample vs B5 (same 20 designer) macro-F1

Sample size: 20 designer × 3 seg = 60 calls per baseline. B5 source: main_final.jsonl.

## Overall macro-F1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.178 | 0.209 | 36.7% |
| B6v2 | 0.273 | 0.384 | 40.0% |

## Per-seg macro-F1

### Q7_seg1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.415 | 0.394 | 40.0% |
| B6v2 | 0.439 | 0.419 | 45.0% |

### Q7_seg2

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.140 | 0.219 | 35.0% |
| B6v2 | 0.181 | 0.219 | 30.0% |

### Q7_seg3

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.130 | 0.219 | 35.0% |
| B6v2 | 0.339 | 0.531 | 45.0% |

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

### B6v2
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.000 | 0.000 | 0.000 | 7 |
| K | 0.333 | 0.250 | 0.286 | 4 |
| V | 0.636 | 0.875 | 0.737 | 8 |
| C | 0.200 | 1.000 | 0.333 | 1 |

Macro-F1: **0.339**, Macro-Recall: **0.531**

## Per-designer seg3 prediction (sorted by GT class)

| Designer | GT | B5 pred | B6v2 pred | B5 ✓? | B6v2 ✓? |
|---|---|---|---|---|---|
| 1078765532 | P | V | C | ✗ | ✗ |
| 695475971 | P | V | V | ✗ | ✗ |
| BuzhichunD | P | V | K | ✗ | ✗ |
| F29293918 | P | V | V | ✗ | ✗ |
| Vivian11320 | P | V | C | ✗ | ✗ |
| acho20050223 | P | V | V | ✗ | ✗ |
| faymeng666 | P | V | K | ✗ | ✗ |
| AyiWe4Real | K | V | C | ✗ | ✗ |
| Hreiy888 | K | V | V | ✗ | ✗ |
| cccclare6354 | K | V | P | ✗ | ✗ |
| kuronim | K | V | K | ✗ | ✓ |
| Yyd3144117584 | C | V | C | ✗ | ✓ |
| Jimmy_Fang18 | V | V | V | ✓ | ✓ |
| WUXY0953 | V | V | C | ✓ | ✗ |
| WuShanYuXi | V | C | V | ✗ | ✓ |
| a123164981 | V | V | V | ✓ | ✓ |
| mz18757140975 | V | V | V | ✓ | ✓ |
| ww2001811920 | V | V | V | ✓ | ✓ |
| wxid_negbygwl2 | V | V | V | ✓ | ✓ |
| wxid_wqyptcofe | V | V | V | ✓ | ✓ |
