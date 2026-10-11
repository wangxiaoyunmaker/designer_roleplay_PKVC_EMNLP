# EMNLP — RolePlay Designer / PKVC–CAPS（私有研究归档）

2026-10-11 整理。本地最终方法线索来自原项目 git `499af2a`（M29: v2_anchor final spec）。保留当前本地工作文件，不等同于纯净 commit 检出。尚未核实 Overleaf 最终投稿源。

## 最终代码与关键产物

| 内容 | 入口 | 说明 |
|---|---|---|
| 最终 prompt | `docs/briefs/prompt_v2_contrastive_diverse_v1.md` | v2_anchor，peer 是额外消融 |
| 最终 prompt runner | `scripts/run_b6_sample.py` | `--variant v2`，保留历史脚本名 |
| baselines | `scripts/run_3_baselines_q7.py` | B0–B5 |
| 指标与跨模型聚合 | `scripts/role_play_metrics.py`, `scripts/final_eval_with_bootstrap.py`, `scripts/aggregate_cross_model_11metrics.py` | 保存原实现 |
| SFT、extractor、RAG | `sft_pipeline/` | 训练、推理、v8.7 抽取配置 |
| 原始数据与较早版本 | `data/`, `data_0523/`, `data_v3/` | 保留身份字段；仅用于私有仓库 |
| 关键处理产物 | `data_v3/processed/` | EFA 分数、Q1–Q6 扩写和 provenance、v8.7 标签、四元组、SFT train/val、fold 划分 |
| 规则四元组与本体 | `artifacts/ontology/` | 提炼出的规则、选项本体 |
| EFA 与相关中间产物 | `artifacts/statistics/` | loadings、相关和偏相关矩阵 |
| 最终跨模型点估计 | `results/submission/cross_model_full_metrics_v2.json` | 与 paper_table_plan.md 指向一致 |
| GPT-4o bootstrap | `results/submission/gpt4o_paper5metric_bootstrap.json` | 论文五指标 CI |
| 全部逐人生成与评测 | `reports/datav3/` | inference、PKVC eval、perpair、跨模型与 SFT 结果 |
| 原始专家标注及 mapping | `private_data_pack/` | 现有私有数据包及身份映射 |
| 方法和表格版本 | `docs/briefs/`, `paper_drafts/` | 最终 prompt 有明确版本；paper_drafts 是历史稿，未认证为提交源 |

## 使用与版本边界

从仓库根目录执行脚本。原始 ID、匿名 D 编号和较早数据版本各按源文件保留，不在归档时重排 ID。与 `private_data_pack/uid_mapping.csv` 对照后再跨数据版本连接。原研究纳入 163 人，部分 Q7 评测为 162 人，见原实验报告；不要把它与 KDD 的 162 人处理版混用。

依赖声明见 requirements.txt；此次没有重新训练或调用 LLM。远端权重、GPU 环境与未同步的远端文件没有完整本地副本，因此不声称完整训练复现。历史报告按原文保留，不代表重新确认其研究结论。模型 checkpoint 和 embedding cache 未包含。

远端为私有仓库 [EMNLP](https://github.com/wangxiaoyunmaker/EMNLP)，默认分支为 `main`。见 `PUSH.md`，必须使用 Private 仓库。来源与 hash 见 `SOURCE_MANIFEST.json`、`CHECKSUMS.sha256`。原目录均保留。

## 本次归档检查

文件复制 hash、Python 语法、密钥模式扫描和 GitHub 单文件大小检查通过；没有 50 MiB 以上文件。已使用原项目现存 Python 3.11 环境验证最终 prompt runner 的 --help；环境目录本身未复制。系统 Python 3.14 尚缺 openai，需要按 requirements.txt 在新环境安装依赖。批量运行脚本里的硬编码密钥已改为从环境变量读取。

GitHub 私有仓库：https://github.com/wangxiaoyunmaker/EMNLP。后续在本目录提交更新，并运行 `git push origin main`。
