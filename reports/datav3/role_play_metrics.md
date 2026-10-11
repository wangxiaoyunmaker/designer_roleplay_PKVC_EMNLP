# Role-play metric eval (seg1/seg3 differentiation)

Compares baselines on designer-DIFFERENTIATION metrics, beyond GT-match.

## A+B. PKVC dom 分布 KL ↓ + 预测 entropy ↑ (seg1, seg3)

- KL: model 预测分布 vs GT 分布. Low = match population GT distribution.
- Entropy: model 预测的多样性. High = differentiation, Low = V-default collapse.

### Q7_seg1

GT distribution (n=162): P=17.3% K=3.7% V=77.8% C=1.2%, entropy=0.975

| Baseline | Pred dist (P/K/V/C) | KL(pred‖GT) ↓ | Entropy ↑ |
|---|---|---|---|
| B5 (orig) | P=47.5% K=1.2% V=48.8% C=2.5% | 0.2566 | 1.226 |
| B0_h | P=30.2% K=0.0% V=69.7% C=0.0% | 0.0926 | 0.886 |
| B1_h | P=14.8% K=0.0% V=85.2% C=0.0% | 0.0539 | 0.607 |
| B2_h | P=32.7% K=0.0% V=67.3% C=0.0% | 0.1105 | 0.914 |
| B3_h | P=52.5% K=0.6% V=46.9% C=0.0% | 0.3339 | 1.047 |
| v2_hybrid | P=21.0% K=12.3% V=54.3% C=12.3% | 0.2781 | 1.696 |
| v2 (anchor) | P=21.0% K=12.3% V=54.3% C=12.3% | 0.2781 | 1.696 |
| B6_h (+peer) | P=23.5% K=12.3% V=46.3% C=17.9% | 0.4579 | 1.822 |

### Q7_seg3

GT distribution (n=162): P=4.3% K=2.5% V=92.6% C=0.6%, entropy=0.477

| Baseline | Pred dist (P/K/V/C) | KL(pred‖GT) ↓ | Entropy ↑ |
|---|---|---|---|
| B5 (orig) | P=1.2% K=0.0% V=92.6% C=6.2% | 0.1258 | 0.431 |
| B0_h | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |
| B1_h | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |
| B2_h | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |
| B3_h | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |
| v2_hybrid | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |
| v2 (anchor) | P=6.2% K=14.8% V=59.3% C=19.8% | 0.7055 | 1.566 |
| B6_h (+peer) | P=0.0% K=0.0% V=100.0% C=0.0% | 0.0759 | 0.003 |

## C. Trait-prediction correlation (seg1, seg3)

Spearman(trait cosine sim, pred match) across designer pairs. Positive = similar 30维 traits → similar predictions (role-play 信号).

### Q7_seg1

| Baseline | Spearman ρ | Δ mean sim (matched − unmatched) | N pairs |
|---|---|---|---|
| B5 (orig) | +0.0409 | +0.0047 | 13041 |
| B0_h | -0.0181 | -0.0015 | 13041 |
| B1_h | +0.0285 | +0.0060 | 13041 |
| B2_h | -0.0177 | -0.0016 | 13041 |
| B3_h | +0.0104 | +0.0010 | 13041 |
| v2_hybrid | +0.0372 | +0.0038 | 13041 |
| v2 (anchor) | +0.0372 | +0.0038 | 13041 |
| B6_h (+peer) | +0.1015 | +0.0103 | 13041 |

### Q7_seg3

| Baseline | Spearman ρ | Δ mean sim (matched − unmatched) | N pairs |
|---|---|---|---|
| B5 (orig) | +0.0646 | +0.0117 | 13041 |
| B0_h | +0.0000 | +0.8873 | 13041 |
| B1_h | +0.0000 | +0.8873 | 13041 |
| B2_h | +0.0000 | +0.8873 | 13041 |
| B3_h | +0.0000 | +0.8873 | 13041 |
| v2_hybrid | +0.0000 | +0.8873 | 13041 |
| v2 (anchor) | +0.0705 | +0.0079 | 13041 |
| B6_h (+peer) | +0.0000 | +0.8873 | 13041 |

## D + E. Criterion uniqueness + persona alignment (seg3)

- Uniqueness: 162 designer criteria 的 mean pairwise cosine distance. High = each designer has unique criterion (role-play). Low = everyone says same.

- Persona align: cosine(criterion_emb, designer_Q1-Q6_emb_center). High = criterion matches designer style.

| Baseline | Criterion uniqueness ↑ | Persona alignment ↑ | n_criteria |
|---|---|---|---|
| B5 (orig) | 0.5511 | 0.3161 | 162/162 cached |
| B0_h | 0.4588 | 0.2977 | 162/162 cached |
| B1_h | 0.4447 | 0.3029 | 162/162 cached |
| B2_h | 0.5020 | 0.3221 | 162/162 cached |
| B3_h | 0.4957 | 0.3239 | 162/162 cached |
| v2_hybrid | 0.5026 | 0.3184 | 162/162 cached |
| v2 (anchor) | 0.5452 | 0.3764 | 162/162 cached |
| B6_h (+peer) | 0.5576 | 0.3211 | 162/162 cached |
