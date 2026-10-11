# EMNLP：最终论文代码与关键研究产物

整理依据：[最终论文](paper/final.pdf)，标题 *Bridging Latent Traits and Epistemic States: Aligning Agent Reasoning under Dilemmas with Expert Cognition*。
已移除早期 extractor 对照、pilot、rank-16 后续训练、未在最终论文出现的模型/实验、缓存碎片和重复结果。

| 内容 | 位置 |
|---|---|
| 最终 B4 / v2_anchor、模块阶梯与消融 | `scripts/run_b6_sample.py`（`--variant v2 --n 0`） |
| §4 统计挖掘与最终 36 条 CAPS 程序 | `sft_pipeline/eval/datav3_stats/`、`data/processed/caps_quadruples.json` |
| 表 3–6 的最终输出、提取与指标 | `reports/datav3/`（crossmodel 只保留表 4 的 B0 / B4） |
| §5.4 / 图 4 的 rank-32、5 epoch SFT | `sft_pipeline/train_v3.py`、`scripts/run_sft_inference.py`、`base_*` / `sft_v2*` 输出 |
| 原始问卷、关键中间产物、训练/验证集 | `data_v3/` |
| 无 API 调用的论文数值核对 | `scripts/verify_paper.py` |

从仓库根目录运行：

```bash
python3 -m scripts.verify_paper
# 需要 OPENAI_API_KEY；实际生成会调用 API：
python3 -m scripts.run_b6_sample --variant v2 --n 0 --model gpt-4o --output /tmp/emnlp_b4.jsonl
# 需要 embedding API 或已有本地缓存：
python3 -m scripts.evaluate_paper --group gpt4o
python3 -m scripts.evaluate_paper --group sft
```

依赖见 `requirements.txt`。服务器模型路径、LoRA 权重和 embedding 缓存需另行配置；权重不在本次本地材料中。
数据为私有研究产物，密钥仅通过环境变量提供。
详细文件对应和论文与保存记录的差异见 [PAPER_MAP.md](PAPER_MAP.md)。
