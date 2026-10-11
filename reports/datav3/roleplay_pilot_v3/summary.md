# Role-play Q7 pilot — B0/B1/B2 × 10 users (data_v3, 2026-05-24)

- model: gpt-4o
- baselines: B0 (cold) / B1 (+profile) / B2 (+raw NL peer RAG) / B3 (+quad-context peer RAG)
- target: real user Q7 seg1 PKVC label (v8 extractor)
- metric: pkvc_compare 5 modes (exact / head / tail / jaccard / sequence_sim)

## Per-baseline avg score (higher = closer to real user)

| baseline | exact | head | tail | jaccard | sequence_sim |
|---|---|---|---|---|---|
| b0 | 0.600 | 0.600 | 0.800 | 0.700 | 0.650 |
| b1 | 0.600 | 0.600 | 0.800 | 0.700 | 0.650 |
| b2 | 0.600 | 0.600 | 0.800 | 0.700 | 0.650 |
| b3 | 0.600 | 0.600 | 0.800 | 0.700 | 0.650 |

## Lift vs B0 (cold baseline)

| baseline | sequence_sim | lift vs B0 (abs) |
|---|---|---|
| B0 | 0.650 | — |
| B1 | 0.650 | +0.000 |
| B2 | 0.650 | +0.000 |
| B3 | 0.650 | +0.000 |

## PKVC label distribution

- **real_pkvc**: {'V': 6, 'K→V': 1, 'C→P': 1, 'P': 1, 'C→V': 1}
- **b0_pkvc**: {'V': 10}
- **b1_pkvc**: {'V': 10}
- **b2_pkvc**: {'V': 10}
- **b3_pkvc**: {'V': 10}

## v8 extractor confidence on real / generated

- **real_pkvc**: {'high': 7, 'medium': 3}
- **b0_pkvc**: {'high': 10}
- **b1_pkvc**: {'high': 10}
- **b2_pkvc**: {'high': 10}
- **b3_pkvc**: {'high': 10}