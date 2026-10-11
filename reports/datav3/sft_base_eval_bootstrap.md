# Final 11-metric eval with 1000-bootstrap 95% CI

Baselines: ['BASE_api_b0', 'BASE_api_b1', 'BASE_api_b2', 'BASE_api_b3', 'BASE_api_v2', 'BASE_api_v2_no_caps', 'BASE_api_v2_no_seg3_rule', 'BASE_api_v2_no_beta_summary', 'BASE_api_v2_hybrid', 'BASE_api_v2_plus_peer', 'BASE_local_b0', 'BASE_local_v2', 'SFT_ep5_v2', 'SFT_v3c_v2', 'SFT_v3c_v2_no_caps', 'SFT_v3c_v2_no_seg3_rule', 'SFT_v3c_v2_no_beta_summary', 'SFT_v3c_v2_hybrid']
Bootstrap: 1000 resamples × 162 designers (with replacement)
CI: percentile method (2.5%–97.5%). No Bonferroni correction (orthogonal dimensions).


## seg1 (第一反应 anchor)

| Baseline | seg1_trait_pred_rho |
|---|---|
| BASE_api_b0 | 0.000 [0.000, 0.000] |
| BASE_api_b1 | -0.018 [-0.102, 0.099] |
| BASE_api_b2 | -0.080 [-0.178, 0.050] |
| BASE_api_b3 | -0.042 [-0.134, 0.091] |
| BASE_api_v2 | 0.065 [0.021, 0.140] |
| BASE_api_v2_no_caps | -0.044 [-0.140, 0.103] |
| BASE_api_v2_no_seg3_rule | 0.063 [0.025, 0.131] |
| BASE_api_v2_no_beta_summary | -0.081 [-0.148, 0.005] |
| BASE_api_v2_hybrid | 0.065 [0.021, 0.140] |
| BASE_api_v2_plus_peer | 0.042 [-0.027, 0.133] |
| BASE_local_b0 | 0.044 [-0.028, 0.143] |
| BASE_local_v2 | 0.086 [0.053, 0.153] |
| SFT_ep5_v2 | 0.029 [-0.000, 0.112] |
| SFT_v3c_v2 | -0.001 [-0.050, 0.083] |
| SFT_v3c_v2_no_caps | 0.030 [-0.077, 0.158] |
| SFT_v3c_v2_no_seg3_rule | -0.022 [-0.122, 0.103] |
| SFT_v3c_v2_no_beta_summary | -0.070 [-0.134, 0.022] |
| SFT_v3c_v2_hybrid | -0.004 [-0.058, 0.081] |

- seg1_trait_pred_rho: ↑

## seg2 (chain reasoning)

| Baseline | seg2_macro_TED | seg2_non_V_recall | seg2_chain_diversity |
|---|---|---|---|
| BASE_api_b0 | 2.662 [2.123, 2.751] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| BASE_api_b1 | 2.679 [2.478, 2.762] | 0.000 [0.000, 0.000] | 0.211 [0.149, 0.273] |
| BASE_api_b2 | 2.659 [2.445, 2.739] | 0.000 [0.000, 0.000] | 0.363 [0.287, 0.438] |
| BASE_api_b3 | 2.658 [2.449, 2.739] | 0.025 [0.000, 0.066] | 0.578 [0.503, 0.646] |
| BASE_api_v2 | 2.194 [2.097, 2.348] | 0.150 [0.070, 0.230] | 0.752 [0.683, 0.807] |
| BASE_api_v2_no_caps | 2.706 [2.509, 2.795] | 0.025 [0.000, 0.065] | 0.588 [0.506, 0.662] |
| BASE_api_v2_no_seg3_rule | 2.683 [2.474, 2.774] | 0.086 [0.028, 0.152] | 0.728 [0.660, 0.784] |
| BASE_api_v2_no_beta_summary | 2.539 [2.290, 2.613] | 0.074 [0.024, 0.138] | 0.753 [0.685, 0.815] |
| BASE_api_v2_hybrid | 2.249 [2.152, 2.431] | 0.138 [0.062, 0.212] | 0.758 [0.696, 0.820] |
| BASE_api_v2_plus_peer | 2.467 [2.190, 2.565] | 0.198 [0.117, 0.289] | 0.858 [0.802, 0.895] |
| BASE_local_b0 | 2.461 [2.184, 2.550] | 0.013 [0.000, 0.044] | 0.639 [0.561, 0.710] |
| BASE_local_v2 | 2.516 [2.259, 2.595] | 0.025 [0.000, 0.067] | 0.758 [0.694, 0.809] |
| SFT_ep5_v2 | 2.277 [1.916, 2.389] | 0.061 [0.000, 0.135] | 0.533 [0.429, 0.619] |
| SFT_v3c_v2 | 2.594 [2.005, 2.692] | 0.054 [0.013, 0.110] | 0.582 [0.507, 0.664] |
| SFT_v3c_v2_no_caps | 2.350 [2.033, 2.435] | 0.169 [0.084, 0.261] | 0.706 [0.634, 0.771] |
| SFT_v3c_v2_no_seg3_rule | 2.595 [2.022, 2.695] | 0.051 [0.000, 0.105] | 0.613 [0.535, 0.690] |
| SFT_v3c_v2_no_beta_summary | 2.126 [2.021, 2.265] | 0.039 [0.000, 0.089] | 0.558 [0.474, 0.628] |
| SFT_v3c_v2_hybrid | 2.333 [2.010, 2.439] | 0.095 [0.031, 0.162] | 0.593 [0.520, 0.667] |

- seg2_macro_TED: ↓
- seg2_non_V_recall: ↑
- seg2_chain_diversity: ↑

## seg3 (anchor + criterion)

| Baseline | seg3_criterion_uniqueness |
|---|---|
| BASE_api_b0 | 0.000 [0.000, 0.000] |
| BASE_api_b1 | 0.295 [0.277, 0.309] |
| BASE_api_b2 | 0.150 [0.115, 0.182] |
| BASE_api_b3 | 0.276 [0.224, 0.321] |
| BASE_api_v2 | 0.454 [0.411, 0.484] |
| BASE_api_v2_no_caps | 0.099 [0.060, 0.143] |
| BASE_api_v2_no_seg3_rule | 0.328 [0.276, 0.375] |
| BASE_api_v2_no_beta_summary | 0.274 [0.222, 0.320] |
| BASE_api_v2_hybrid | 0.386 [0.345, 0.421] |
| BASE_api_v2_plus_peer | 0.409 [0.356, 0.450] |
| BASE_local_b0 | 0.323 [0.288, 0.351] |
| BASE_local_v2 | 0.455 [0.413, 0.486] |
| SFT_ep5_v2 | 0.380 [0.312, 0.440] |
| SFT_v3c_v2 | 0.377 [0.318, 0.426] |
| SFT_v3c_v2_no_caps | 0.251 [0.199, 0.302] |
| SFT_v3c_v2_no_seg3_rule | 0.387 [0.329, 0.440] |
| SFT_v3c_v2_no_beta_summary | 0.257 [0.199, 0.308] |
| SFT_v3c_v2_hybrid | 0.439 [0.389, 0.479] |

- seg3_criterion_uniqueness: ↑

## Winner per metric (point estimate, CI annotation)

| Metric | Winner | Point | CI |
|---|---|---|---|
| seg1_trait_pred_rho | **BASE_local_v2** | 0.0863 | [0.0527, 0.1530] |
| seg2_macro_TED | **SFT_v3c_v2_no_beta_summary** | 2.1261 | [2.0206, 2.2651] |
| seg2_non_V_recall | **BASE_api_v2_plus_peer** | 0.1975 | [0.1169, 0.2895] |
| seg2_chain_diversity | **BASE_api_v2_plus_peer** | 0.8580 | [0.8025, 0.8951] |
| seg3_criterion_uniqueness | **BASE_local_v2** | 0.4547 | [0.4134, 0.4858] |


## Δ vs B0_h (zero-shot baseline), 95% CI overlap test

Marked ★ if CI of v2_anchor and B0_h do not overlap (likely significant).

| Metric | B0_h point [CI] | v2_anchor point [CI] | Δ | Sig? |
|---|---|---|---|---|
