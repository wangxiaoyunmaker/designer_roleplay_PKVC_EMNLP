# L1b Expansion Subset Audit (data_v3, 2026-05-24)

- expansion ok rows: **309**
- 理论上限 163 × 6 = 978 cells, **coverage = 31.6%**

## 被扩 vs 没被扩 的 raw supplement 长度 (selection bias)

(预期: 被扩组的 raw 长度应明显短于未被扩组, 因为扩写挑短的)

| was_expanded   |   mean |   median |   max |   min |
|:---------------|-------:|---------:|------:|------:|
| False          |   84.6 |       64 |  1811 |     0 |
| True           |   13.5 |       14 |    25 |     2 |

**Verdict**: PASS (扩写选了短的, 符合预期) (被扩 median=14.0, 未扩 median=64.0)

## per-Q 扩写覆盖率

| question_id   |   sum |   count |   coverage_pct |
|:--------------|------:|--------:|---------------:|
| Q1            |    46 |     163 |           28.2 |
| Q2            |    43 |     163 |           26.4 |
| Q3            |    53 |     163 |           32.5 |
| Q4            |    63 |     163 |           38.7 |
| Q5            |    47 |     163 |           28.8 |
| Q6            |    57 |     163 |           35   |

## inferred_factor 分布 (P/K/V/C)

- P: 43
- K: 110
- V: 38
- C: 118

## inferred_subtype top 15

- K2: 70
- C1: 54
- C2: 50
- P1: 29
- K1: 27
- V2: 24
- P2: 14
- C3: 14
- K3: 13
- V3: 8
- V1: 6

## confidence 分布

- high: 309

## validation_ok 分布

- True: 301
- False: 8

## expanded_nl 长度分布 (字符)

- min=56 median=102 max=131 mean=100.1