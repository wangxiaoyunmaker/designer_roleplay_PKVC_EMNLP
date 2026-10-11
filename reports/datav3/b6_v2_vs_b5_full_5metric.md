# Q7 3-Baseline 5-metric 综合评测

输入: ['reports/datav3/baselines_q7_main_final_inference.jsonl', 'reports/datav3/b6_v2_full_inference.jsonl']

per-pair 结果: reports/datav3/b6_v2_vs_b5_full_5metric_perpair.jsonl (3402 rows)

## 主表 (mean ± std)

### Core (用户严格定义 trajectory fidelity)

| Baseline | **TED** ↓ | **STA** ↑ | PKVC dom acc ↑ |
|---|---|---|---|
| B5 | 1.284±1.103 | 0.345±0.443 | 60.9% |
| B6_v2 | 1.313±1.015 | 0.296±0.416 | 47.3% |

### Auxiliary path metrics + lexical/embedding

| Baseline | Syntactic | Semantic | Pragmatic | Path Jaccard | Path Edit norm | Path Trans Jacc | Path LCS F1 | Persona Cos |
|---|---|---|---|---|---|---|---|---|
| B5 | 0.961±0.145 | 0.637±0.098 | 0.684±0.241 | 0.567±0.363 | 0.498±0.368 | 0.323±0.437 | 0.575±0.352 | 0.549±0.085 |
| B6_v2 | 0.959±0.144 | 0.624±0.094 | 0.660±0.255 | 0.511±0.380 | 0.445±0.369 | 0.276±0.411 | 0.527±0.369 | 0.556±0.083 |

## Per-seg 拆解 (mean)

### Q7_seg1

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.858 | 0.383 | 0.952 | 0.625 | 0.633 | 0.448 | 0.436 | 0.383 | 0.461 | 0.512 |
| B6_v2 | 0.858 | 0.395 | 0.950 | 0.602 | 0.581 | 0.455 | 0.444 | 0.395 | 0.467 | 0.514 |

### Q7_seg2

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 2.241 | 0.203 | 0.964 | 0.657 | 0.684 | 0.594 | 0.406 | 0.135 | 0.545 | 0.589 |
| B6_v2 | 2.043 | 0.231 | 0.961 | 0.645 | 0.642 | 0.616 | 0.440 | 0.171 | 0.598 | 0.583 |

### Q7_seg3

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.753 | 0.451 | 0.967 | 0.628 | 0.736 | 0.660 | 0.651 | 0.451 | 0.718 | 0.547 |
| B6_v2 | 1.037 | 0.262 | 0.967 | 0.623 | 0.758 | 0.463 | 0.450 | 0.262 | 0.518 | 0.570 |
