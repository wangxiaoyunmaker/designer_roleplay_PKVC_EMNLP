# 最终论文与文件对应

唯一论文依据：用户提供的 `17317_Bridging_Latent_Traits_a.pdf`，保存为 `paper/final.pdf`。

| 论文内容 | 必要代码 / 产物 | 核对情况 |
|---|---|---|
| §3–4：traits、EFA、LASSO、clustering / t-SNE | `sft_pipeline/eval/datav3_stats/{efa,lasso_30_items,cluster,visualize}.py`，`L2_efa` / `L3_cluster` / `L3_visual` / `L4c_lasso_30items`、factor/loadings CSV | 仅保留正文方法的统计阶段和元数据依赖 |
| §4：36 条 CAPS 程序 | `data/processed/caps_quadruples.json`、`per_designer/option_trait_mapping.json`、`extract_caps_quadruples.py` | 36 条；保留对应抽取源码与映射 |
| §5 / 附录 C：最终 B4 | `run_b6_sample.py`，`--variant v2 --n 0 --modules pi,stance,self,caps` | v2_anchor；未采用 pilot 默认 20 人 |
| 表 3：B0–B4 | `ladder_b0_h` 至 `ladder_b3_h`、`b6_v2_full`，每项 inference + pkvc_eval | 表 3 五项指标全部与保存数值一致 |
| 表 4：跨模型 B0→B4 | `crossmodel/{gemini31flashlite,claudehaiku45,deepseekv4flash,qwen35b}_{b0,v2}_*` | 只保留论文列出的模型/条件；移除 qwen27b 和额外跨模型消融 |
| 表 5–6：五项 GPT-4o 消融 | `ablation_v2_no_caps`、`ablation_v2_no_seg3_rule`、`ablation_v2_no_beta_summary`、`ablation_v2_plus_peer`、`b6_v2hybrid_full` | 表 6 五项指标全部与保存数值一致；PlusPeer 是论文中报告的消融 |
| §5.4 / 图 4：SFT | `train_v3.py`、`config_v3.yaml`、`build_sft_data_q1q6.py`、`sft_train` / `sft_val`、`run_sft_inference.py`、7 个 BASE + 7 个 epoch-5 SFT 条件 | 保留与表 6 配置对应的本地训练验证；图 4 正文 1.821 / 0.592 / 0.185 来自 `sft_base_experiments_summary.md` |
| PKVC 提取及评估 | `run_full_data_v3.py`、`eval_baselines_pkvc.py`、`extractor_v8.7.md`、`final_eval_with_bootstrap.py` | 保留最终 v8.7 金标/提取结果与度量函数 |

## 可运行入口与来源

`evaluate_paper.py` 限定表 3/6 的 GPT-4o 十个条件，或 §5.4 的十四个 BASE/SFT 条件；使用原始度量和 1000 次 bootstrap。
`aggregate_cross_model_11metrics.py` 对本仓库保留的五个模型和已存在条件聚合；缺 embedding cache 时立即报错，避免把缺失向量生成的空值写入最终结果。
`verify_paper.py` 无网络调用，核对表 3 / 6 的 50 个指标单元、表 4 的 25 个差值，并报告已有差异。
原始 `cross_model_full_metrics_v2.json` 已只保留论文对应条件，未更改保留条件的数值。
若 API 缓存不存在，`evaluate_paper.py` 会通过环境变量指定的 embedding API 补齐。已保存的论文结果不代表在本轮重新进行了训练或 API 推理。

## 论文与已有材料的差异

1. 训练数据：论文 §5.4 写 162×6=972；实际保存的训练/验证 JSONL 为 770+192=962，历史 SFT 报告也写 962。保留原始记录，没有补造 10 条样本。
2. 表 4 Claude Haiku 的 macro-TED delta 在保存汇总中为 -0.1247346485，论文写 -0.13；criterion uniqueness delta 为 +0.4547425116，论文写 +0.46。其余差值与论文精度对应。
3. Trait-prediction rho：正文称 Seg2；现有聚合使用 `seg1_trait_pred_rho`。表 3/6 的数值能与现有汇总对应，但该 segment 定义差异未被静默改写。
4. 图 4 正文写 SFT NoRule TED=1.821、uniqueness=0.592；PDF 图内柱形与这些正文数字存在视觉不一致。仓库保存的是能追溯到正文数字的 epoch-5 报告和原始输出，不重绘论文图。
5. 论文对生成上限写 600；保留的源码中不同 API backend 的上限设置有差异。未为表面一致而改写既有实验协议。
6. 模型 LoRA 权重在服务器路径中，本地无完整权重文件；embedding cache 不随仓库上传，可由保存文本再生成。因此不是自包含的完全离线复现包。

共享源码文件内仍有历史 variant 分支；仅按上表明确列出的最终入口使用。独立旧版本实验、rank-16 v3c/v3a-v3e、extractor v2–v8.6 比较、缓存单条 JSON、失败重试、旧设计文档和重复产物均已移出当前目录。
