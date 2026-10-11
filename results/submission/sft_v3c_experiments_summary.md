# SFT (Qwen2.5-7B + LoRA r=16) on Q7 cross-scenario transfer — paper experiments

> 整理日期: 2026-05-26
> 数据集: 162 designer × 3 Q7 seg = 486 designer-segment instances per condition
> 模型对比:
>   - **BASE**: Qwen2.5-7B-Instruct via yunwu.ai API (no fine-tuning, bf16)
>   - **SFT_v3c**: Qwen2.5-7B-Instruct + LoRA **r=16** (3 epoch, LR 5e-5), 当前推理 int4 量化 (因 GPU zombie 限制)
>   - **SFT_ep5** (history): LoRA r=32, 5 epoch, LR 1e-4 — over-fit reference (chain diversity 塌)
> Training data: 162 designer × Q1-Q6 = 962 example (770 train / 192 val), v3 expansion 含 trait-aware NL rewrite
> Final train_loss=1.793, eval_loss=1.497 (vs ep5 train 1.263 eval 1.327 — SFT_v3c 训得更轻保多样性)
> Evaluation: 5 metric × 1000-bootstrap 95% CI (designer-level resampling); PKVC extraction via GPT-4o v8.7 extractor
> Inference: BASE via API (concurrency 8); SFT_v3c via local LoRA int4 (10 condition, 5 GPU parallel)

---

## 0. 实验设计

### 0.1 5 metric subset (EMNLP table 锁)

| Seg | Metric | 维度 | 角色 |
|---|---|---|---|
| seg2 | macro-TED ↓ | Fidelity | chain 准度 |
| seg2 | non-V recall ↑ | Fidelity | 抗 V-default |
| seg2 | chain diversity ↑ | Anti-collapse | chain 差异化 |
| seg1 | trait-pred Spearman ρ ↑ | Personalization | trait 路由 |
| seg3 | criterion uniqueness ↑ | Anti-collapse | 底线差异化 |

砍掉 6 个 metric (NL specificity × 3, persona-NL, criterion-persona, macro-LCS-F1) 因为信息冗余 / condition 间差异 ≤0.05 / 跨 model 不一致.

### 0.2 10 condition (per model: BASE_api + SFT_v3c)

| Group | Condition | CLI |
|---|---|---|
| **Ladder** | B0 (zero-shot) | `--variant v2_hybrid --modules ""` |
| | B1 (+ pi) | `--variant v2_hybrid --modules pi` |
| | B2 (+ stance) | `--variant v2_hybrid --modules pi,stance` |
| | B3 (+ self) | `--variant v2_hybrid --modules pi,stance,self` |
| | **v2 (+ CAPS)** ★ | `--variant v2` (full) |
| **Component ablation** | − CAPS | `--variant v2 --modules pi,stance,self` |
| | − seg3 rule | `--variant v2_no_seg3_rule` |
| | − β summary | `--variant v2_no_beta_summary` |
| | + peer | `--variant v2 --with-peer` |
| | hybrid | `--variant v2_hybrid` |

Each condition 跑 162 designer × 3 seg = 486 inference.

---

## 1. 关键 finding (paper claim 候选)

> _[Pending: 等 v3c wave 2 + bootstrap]_

**Mode collapse fix**: 早期 SFT_ep5 (r=32, 5 epoch) 在 chain diversity 上比 BASE 跌 30% (CI 不重叠 ★, over-fit signature). SFT_v3c (**r=16, 3 epoch, LR 5e-5**) 恢复 diversity, in probe 上 v3c 0.690 > BASE 0.636 > ep5 0.500.

---

## 2. Table 1 — Module Ladder on SFT_v3c (paper main result, primary model)

> _[Pending bootstrap data, 用 SFT_v3c]_

| Method | macro-TED ↓ | non-V recall ↑ | chain div ↑ | trait-pred ρ ↑ | crit uniq ↑ |
|---|---|---|---|---|---|
| B0 (zero-shot) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| B1 (+ pi) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| B2 (+ stance) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| B3 (+ self) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **v2 (+ CAPS)** ★ | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## 3. Table 2 — Component Ablation on SFT_v3c (vs full v2)

> _[Pending bootstrap data]_

| Method | macro-TED ↓ | non-V recall ↑ | chain div ↑ | trait-pred ρ ↑ | crit uniq ↑ |
|---|---|---|---|---|---|
| **v2 (full)** ★ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| − CAPS | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| − seg3 rule | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| − β summary | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + peer | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| v2_hybrid | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 4. Table 3 — Cross-Model: BASE vs SFT_v3c (Δ per metric)

> _[Pending bootstrap data]_

| Model × Condition | macro-TED ↓ | non-V recall ↑ | chain div ↑ | trait-pred ρ ↑ | crit uniq ↑ |
|---|---|---|---|---|---|
| BASE_v2 (api) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SFT_v3c_v2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Δ (SFT − BASE)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 5. Mode collapse fix — Ablation across SFT 变体 (paper section)

5 个 LoRA 变体 chain diversity 对比 (30-designer subsample, int4 probe):

| variant | LoRA r | epoch | LR | KL λ | chain_div |
|---|---|---|---|---|---|
| ep5 (over-fit) | 32 | 5 | 1e-4 | 0 | 0.500 |
| v3a | 32 | 2 | 5e-5 | 0 | 0.633 |
| v3b | 32 | 3 | 5e-5 | 0 | 0.483 |
| **v3c** ★ | **16** | 3 | 5e-5 | 0 | **0.690** |
| v3d | 32 | 3 | 5e-5 | 0.05 | 0.483 |
| v3e | 32 | 3 | 5e-5 | 0.2 | 0.567 |

**finding**: **LoRA r 减半 (32→16) 是 mode collapse 最有效 fix**. KL-to-base loss 无效 (λ=0.05) 到中等 (λ=0.2); epoch/LR 调整次要.

---

## 6. 复现

- Inference (BASE API): `scripts/run_base_qwen_api_10cond.sh` (yunwu.ai, qwen2.5-7b-instruct)
- Inference (SFT_v3c): `scripts/run_sft_v3c_7cond_int4.sh` + `scripts/run_sft_inference.py --quantize int4`
- Training: `sft_pipeline/train_v3.py --epochs 3 --lr 5e-5 --lora-r 16 --out-suffix _v3c`
- Extract PKVC: `scripts/eval_baselines_pkvc.py × 20` (~$15 total)
- Bootstrap: `scripts/sft_base_eval_bootstrap.py` (5-metric subset)
- Adapter: `/data/wy/sft_v3_ckpts/stage_warmup_ep3_v3c` (remote)
- Outputs: `reports/datav3/sft_base_eval_bootstrap.{md,jsonl}` + `reports/datav3/sft_v3c_experiments_summary.md`

---

## 7. 已知限制

1. **v3c 用 int4 量化推理** — bf16 GPU 被 CUDA zombie 占, 只能 int4. quantization noise 存在, 但 SFT 内部 ranking 可信
2. **BASE API vs SFT_v3c int4 跨精度对比** — quantization noise 在两 model 间不对等. 可作 directional signal, 严格 magnitude 有 caveat
3. **Q2 thinking_chain 未评估** — Q2 是 designer 全局 cardId 排序, 跟 Q7 transfer 分离
4. **第 5 节 chain diversity probe 用 30-designer subsample int4**, 跟 paper main result 162-designer 不完全 apples-to-apples — 仅作 ablation reference
