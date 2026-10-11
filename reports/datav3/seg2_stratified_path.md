# seg2 stratified path fidelity (by GT first letter)

Goal: prove v2_hybrid does chain reasoning on non-V GT designers, where B0_h's V-default fails.

## B0_h

  **V-first GT (易于 V-default)** (n=81):  TED↓=1.840  STA↑=0.397  Path-Jacc↑=0.642  Trans-Jacc↑=0.239  LCS-F1↑=0.683
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.556  STA↑=0.317  Path-Jacc↑=0.665  Trans-Jacc↑=0.195  LCS-F1↑=0.582

## B1_h

  **V-first GT (易于 V-default)** (n=81):  TED↓=2.012  STA↑=0.311  Path-Jacc↑=0.606  Trans-Jacc↑=0.188  LCS-F1↑=0.638
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.358  STA↑=0.262  Path-Jacc↑=0.687  Trans-Jacc↑=0.182  LCS-F1↑=0.599

## v2_hybrid

  **V-first GT (易于 V-default)** (n=81):  TED↓=1.877  STA↑=0.286  Path-Jacc↑=0.627  Trans-Jacc↑=0.189  LCS-F1↑=0.639
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.284  STA↑=0.266  Path-Jacc↑=0.645  Trans-Jacc↑=0.189  LCS-F1↑=0.596

## v2_anchor

  **V-first GT (易于 V-default)** (n=81):  TED↓=1.877  STA↑=0.286  Path-Jacc↑=0.627  Trans-Jacc↑=0.189  LCS-F1↑=0.639
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.284  STA↑=0.266  Path-Jacc↑=0.645  Trans-Jacc↑=0.189  LCS-F1↑=0.596

## B6_h

  **V-first GT (易于 V-default)** (n=81):  TED↓=2.247  STA↑=0.262  Path-Jacc↑=0.593  Trans-Jacc↑=0.159  LCS-F1↑=0.592
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.506  STA↑=0.245  Path-Jacc↑=0.624  Trans-Jacc↑=0.150  LCS-F1↑=0.562

## B5

  **V-first GT (易于 V-default)** (n=81):  TED↓=1.938  STA↑=0.339  Path-Jacc↑=0.637  Trans-Jacc↑=0.193  LCS-F1↑=0.668
  **non-V-first GT (真 chain reasoning 测试)** (n=81):  TED↓=2.741  STA↑=0.218  Path-Jacc↑=0.622  Trans-Jacc↑=0.138  LCS-F1↑=0.537


## 对比总结 (non-V-first GT, 真 chain reasoning)

| Baseline | TED↓ | STA↑ | Path-Jacc↑ | Trans-Jacc↑ | LCS-F1↑ |
|---|---|---|---|---|---|
| B0_h | 2.556 | 0.317 | 0.665 | 0.195 | 0.582 |
| B1_h | 2.358 | 0.262 | 0.687 | 0.182 | 0.599 |
| v2_hybrid | 2.284 | 0.266 | 0.645 | 0.189 | 0.596 |
| v2_anchor | 2.284 | 0.266 | 0.645 | 0.189 | 0.596 |
| B6_h | 2.506 | 0.245 | 0.624 | 0.150 | 0.562 |
| B5 | 2.741 | 0.218 | 0.622 | 0.138 | 0.537 |

## V-first GT (V-default 容易胜出 — 提供对照)

| Baseline | TED↓ | STA↑ | Path-Jacc↑ | LCS-F1↑ |
|---|---|---|---|---|
| B0_h | 1.840 | 0.397 | 0.642 | 0.683 |
| B1_h | 2.012 | 0.311 | 0.606 | 0.638 |
| v2_hybrid | 1.877 | 0.286 | 0.627 | 0.639 |
| v2_anchor | 1.877 | 0.286 | 0.627 | 0.639 |
| B6_h | 2.247 | 0.262 | 0.593 | 0.592 |
| B5 | 1.938 | 0.339 | 0.637 | 0.668 |