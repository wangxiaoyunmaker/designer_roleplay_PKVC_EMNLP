# Q7 3-Baseline 5-metric 综合评测

输入: ['reports/datav3/baselines_q7_main_final_inference.jsonl', 'reports/datav3/b6_v2hybrid_full_inference.jsonl', 'reports/datav3/b6_h_full_inference.jsonl']

per-pair 结果: reports/datav3/b6h_vs_v2hybrid_vs_b5_full_5metric_perpair.jsonl (3888 rows)

## 主表 (mean ± std)

### Core (用户严格定义 trajectory fidelity)

| Baseline | **TED** ↓ | **STA** ↑ | PKVC dom acc ↑ |
|---|---|---|---|
| B5 | 1.284±1.103 | 0.345±0.443 | 60.9% |
| B6_v2_hybrid | 1.222±1.052 | 0.358±0.442 | 59.5% |
| B6_v2_hybrid_peer | 1.344±1.063 | 0.302±0.423 | 56.2% |

### Auxiliary path metrics + lexical/embedding

| Baseline | Syntactic | Semantic | Pragmatic | Path Jaccard | Path Edit norm | Path Trans Jacc | Path LCS F1 | Persona Cos |
|---|---|---|---|---|---|---|---|---|
| B5 | 0.961±0.145 | 0.637±0.098 | 0.684±0.241 | 0.567±0.363 | 0.498±0.368 | 0.323±0.437 | 0.575±0.352 | 0.549±0.085 |
| B6_v2_hybrid | 0.959±0.144 | 0.621±0.096 | 0.654±0.253 | 0.588±0.361 | 0.517±0.366 | 0.338±0.440 | 0.606±0.345 | 0.549±0.084 |
| B6_v2_hybrid_peer | 0.959±0.145 | 0.636±0.099 | 0.680±0.245 | 0.538±0.353 | 0.471±0.354 | 0.280±0.415 | 0.558±0.345 | 0.546±0.091 |

## Per-seg 拆解 (mean)

### Q7_seg1

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.858 | 0.383 | 0.952 | 0.625 | 0.633 | 0.448 | 0.436 | 0.383 | 0.461 | 0.512 |
| B6_v2_hybrid | 0.858 | 0.395 | 0.950 | 0.602 | 0.581 | 0.455 | 0.444 | 0.395 | 0.467 | 0.514 |
| B6_v2_hybrid_peer | 0.932 | 0.333 | 0.946 | 0.623 | 0.632 | 0.388 | 0.377 | 0.333 | 0.398 | 0.503 |

### Q7_seg2

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 2.241 | 0.203 | 0.964 | 0.657 | 0.684 | 0.594 | 0.406 | 0.135 | 0.545 | 0.589 |
| B6_v2_hybrid | 2.043 | 0.231 | 0.961 | 0.645 | 0.642 | 0.616 | 0.440 | 0.171 | 0.598 | 0.583 |
| B6_v2_hybrid_peer | 2.247 | 0.210 | 0.963 | 0.659 | 0.649 | 0.577 | 0.406 | 0.143 | 0.547 | 0.589 |

### Q7_seg3

| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.753 | 0.451 | 0.967 | 0.628 | 0.736 | 0.660 | 0.651 | 0.451 | 0.718 | 0.547 |
| B6_v2_hybrid | 0.765 | 0.448 | 0.967 | 0.617 | 0.738 | 0.692 | 0.665 | 0.448 | 0.753 | 0.550 |
| B6_v2_hybrid_peer | 0.852 | 0.363 | 0.967 | 0.626 | 0.759 | 0.647 | 0.629 | 0.363 | 0.728 | 0.545 |
