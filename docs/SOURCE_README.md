# Designer PKVC Role-Play (EMNLP 2026 main)

设计师 PKVC 簇级 role-play 的 LoRA SFT + 簇内 peer RAG + trait-state modulation
(operationalize Mischel 1995 CAPS theory). 投 **EMNLP 2026 main**, backup Findings.

> **本 README 给人看**. 给 Claude Code 看的是 [CLAUDE.md](CLAUDE.md).

---

## 当前 working assumption

**v3.5.1 (2026-05-22)** — 3 层架构 + 单 LoRA + 30 维 trait-only 聚类 + trait-state
modulation triple bridge.

- **方向二 (主)**:Q1-Q6 + 解释 NL SFT → Q7 跨场景评测
- **方向一 (对照)**:Q7 三段 CoT SFT → Q1-Q6 跨场景评测
- **Trait Layer**: 30 维设计师领域信念量表 (designer 个人 CAPS network weights)
- **PKVC State Machine**: P/K/V/C 4 状态运行时引擎 (30 SADR 是子细分)
- **Bridge**: trait-state modulation triple, 4 档 + Bypass flag

详 [docs/briefs/v3.5.1_20260522.md](docs/briefs/v3.5.1_20260522.md).

---

## 仓库结构 (v3.5.1 lock)

```
0512技术方案/
├── README.md                        本文件 (给人看)
├── CLAUDE.md                        给 Claude Code 看的 onboarding
├── docs/
│   ├── briefs/                      技术方案 brief (按版本)
│   │   ├── v3.4_20260519.md         双方向并行 framing 锁
│   │   └── v3.5.1_20260522.md       3 层架构 + modulation, 当前 source of truth
│   ├── notes/
│   │   ├── decisions_20260519.md    决策清单
│   │   └── framing_correction_20260522.md   CAPS framing 错误溯源
│   └── archive/                     62 篇前期项目文档 (6 tier)
├── sft_pipeline/                    主代码 (见 sft_pipeline/STRUCTURE.md)
├── scripts/                         EFA / cluster pilot 等独立脚本
├── data/                            真人 PII, **不进 git** (.gitignore)
│   └── processed/                   extractor 输出 / mapping triple JSONL
└── reports/                         评测报告输出 (gitignored)
```

---

## 快速入口

| 想做什么 | 文件 / 命令 |
|---|---|
| 看技术 framing | [docs/briefs/v3.5.1_20260522.md](docs/briefs/v3.5.1_20260522.md) |
| 看代码结构 | [sft_pipeline/STRUCTURE.md](sft_pipeline/STRUCTURE.md) |
| build SFT JSONL | `python -m sft_pipeline.data.prep --rag-mode cosine ...` |
| 跑 extractor (需 API key) | `python -m sft_pipeline.extract.batch --input ...` |
| 挖 mapping triple | `python -m sft_pipeline.rag.mapping_miner --derive-folds ...` |
| 跑 SFT 训练 | `python -m sft_pipeline.train --config sft_pipeline/config/config.yaml --direction q1q6 --fold 0` |
| 跑评测 | `python -m sft_pipeline.eval.runner ...` |

---

## 开发约定 (跟 CLAUDE.md 一致)

- 终端不渲染 LaTeX → 回复禁用 `$...$`, 用 Unicode + 代码块. 写文件可以用 LaTeX.
- 介入方案时主动拎隐含假设 + 给对立面.
- Framing 决策先于实现; 启动 SFT 实验前先对齐 framing 不漂移.
- 论文缝合 ≤ 3 个 core anchor (Self-Discover / USP / P-React + CAPS + OG-RAG).
- 数据 `data/*.csv` 是真人 PII, 不进 git.
- 远程服务器 `wy@101.69.162.1 -p 11024`, `/data` 经常满会静默写零字节, 改 `/home/wy/`.

---

## 当前 blocker (CLAUDE.md §5)

```
🔴 API key (yunwu.ai)         旧 key 已泄露需 rotate, 阻塞 extractor + mapping mining
🔴 GPU (远程服务器)            被多 PID 占, 阻塞 SFT 训练
🔴 磁盘 (/data 100%)           部署到远程暂停
🟡 学姐 reply (PKVC metadata)  12 cardId + 题目-因子 mapping + 金标 Q7 NL
🟡 121 incremental designer    入袋时间未承诺 (200 入袋触发 EFA 重跑)
```

---

## 投稿 timeline

详见 [CLAUDE.md §12](CLAUDE.md) 跟 [docs/briefs/v3.5.1_20260522.md §11](docs/briefs/v3.5.1_20260522.md).
