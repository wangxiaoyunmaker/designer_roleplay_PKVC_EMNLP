# Q7 3-Baseline Inference 分析报告

输入: reports/datav3/baselines_q7_inference.jsonl

总样本: 1458 inference calls, 486 unique (designer, scenario) pairs

## 长度分布

| 来源 | n | median 字数 | mean 字数 |
|---|---|---|---|
| Ground truth | 486 | 252 | 282 |
| B1 | 486 | 492 | 503 |
| B2 | 486 | 532 | 538 |
| B3 | 485 | 571 | 584 |

## Lexical 重合 (Jaccard char 1+2-gram, vs ground truth)

| baseline | n | median | mean |
|---|---|---|---|
| B1 | 486 | 0.107 | 0.108 |
| B2 | 486 | 0.110 | 0.111 |
| B3 | 485 | 0.109 | 0.112 |

## Ground truth PKVC dominant 分布 (per seg)

| scenario | n | distribution |
|---|---|---|
| Q7_seg1 | 162 | {'V': 126, 'P': 28, 'K': 6, 'C': 2} |
| Q7_seg2 | 162 | {'K': 34, 'V': 88, 'P': 36, 'C': 4} |
| Q7_seg3 | 162 | {'V': 150, 'P': 7, 'K': 4, 'C': 1} |