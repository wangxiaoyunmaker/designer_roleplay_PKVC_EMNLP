# B6 sample vs B5 (same 20 designer) macro-F1

Sample size: 20 designer × 3 seg = 60 calls per baseline. B5 source: main_final.jsonl.

## Overall macro-F1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.178 | 0.209 | 36.7% |
| B6 | 0.425 | 0.455 | 43.3% |

## Per-seg macro-F1

### Q7_seg1

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.415 | 0.394 | 40.0% |
| B6 | 0.310 | 0.500 | 45.0% |

### Q7_seg2

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.140 | 0.219 | 35.0% |
| B6 | 0.288 | 0.362 | 40.0% |

### Q7_seg3

| Baseline | Macro-F1 | Macro-Recall | dom acc |
|---|---|---|---|
| B5 | 0.130 | 0.219 | 35.0% |
| B6 | 0.451 | 0.531 | 45.0% |

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

### B6
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P | 0.412 | 1.000 | 0.583 | 7 |
| K | 0.000 | 0.000 | 0.000 | 4 |
| V | 1.000 | 0.125 | 0.222 | 8 |
| C | 1.000 | 1.000 | 1.000 | 1 |

Macro-F1: **0.451**, Macro-Recall: **0.531**

## Per-designer seg3 prediction (sorted by GT class)

| Designer | GT | B5 pred | B6 pred | B5 ✓? | B6 ✓? |
|---|---|---|---|---|---|
| 1078765532 | P | V | P | ✗ | ✓ |
| 695475971 | P | V | P | ✗ | ✓ |
| BuzhichunD | P | V | P | ✗ | ✓ |
| F29293918 | P | V | P | ✗ | ✓ |
| Vivian11320 | P | V | P | ✗ | ✓ |
| acho20050223 | P | V | P | ✗ | ✓ |
| faymeng666 | P | V | P | ✗ | ✓ |
| AyiWe4Real | K | V | P | ✗ | ✗ |
| Hreiy888 | K | V | P | ✗ | ✗ |
| cccclare6354 | K | V | P | ✗ | ✗ |
| kuronim | K | V | P | ✗ | ✗ |
| Yyd3144117584 | C | V | C | ✗ | ✓ |
| Jimmy_Fang18 | V | V | P | ✓ | ✗ |
| WUXY0953 | V | V | P | ✓ | ✗ |
| WuShanYuXi | V | C | P | ✗ | ✗ |
| a123164981 | V | V | V | ✓ | ✓ |
| mz18757140975 | V | V | P | ✓ | ✗ |
| ww2001811920 | V | V | K | ✓ | ✗ |
| wxid_negbygwl2 | V | V | P | ✓ | ✗ |
| wxid_wqyptcofe | V | V | P | ✓ | ✗ |
