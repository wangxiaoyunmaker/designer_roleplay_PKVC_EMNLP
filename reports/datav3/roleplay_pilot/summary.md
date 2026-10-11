# Role-play Q7 pilot — B0/B1/B2 × 10 users (data_v3, 2026-05-24)

- model: gpt-4o
- baselines: B0 (cold) / B1 (+profile NL) / B2 (B1 + trait-sim top-3 peer RAG)
- target: real user Q7 seg1 PKVC label (v8 extractor)
- metric: pkvc_compare 5 modes (exact / head / tail / jaccard / sequence_sim)

## Per-baseline avg score (higher = closer to real user)

| baseline | exact | head | tail | jaccard | sequence_sim |
|---|---|---|---|---|---|
| b0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Lift vs B0 (cold baseline)

| baseline | sequence_sim | lift vs B0 (abs) |
|---|---|---|
| B0 | 1.000 | — |
| B1 | 1.000 | +0.000 |
| B2 | 1.000 | +0.000 |

## PKVC label distribution

- **real_pkvc**: {'V': 10}
- **b0_pkvc**: {'V': 10}
- **b1_pkvc**: {'V': 10}
- **b2_pkvc**: {'V': 10}

## v8 extractor confidence on real / generated

- **real_pkvc**: {'low': 8, 'medium': 2}
- **b0_pkvc**: {'low': 6, 'medium': 4}
- **b1_pkvc**: {'medium': 5, 'low': 5}
- **b2_pkvc**: {'low': 7, 'high': 1, 'medium': 2}