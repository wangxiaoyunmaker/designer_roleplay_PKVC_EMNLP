# Ontology-Grounded RAG (PKVC adaptation)

设计灵感来自 OG-RAG (Sharma et al., EMNLP 2025, [arxiv:2412.15235](https://arxiv.org/abs/2412.15235), [microsoft/ograg2](https://github.com/microsoft/ograg2)), 但**重新实现**以适配 v3.4 PKVC + 簇内 peer 框架。

## 跟 OG-RAG 原架构的对照

| OG-RAG | 我们的实现 |
|---|---|
| JSON-LD ontology (entity + relation) | `ontology.py` PKVC 4 因子 + 30 SADR 节点 + 题目-因子 metadata |
| 文档 chunk 作为超图节点 | designer-segment (Q7 三段 / Q1-Q6 单题) |
| Hyperedge = grounded fact cluster | (cluster_id, dominant_factor, scenario_type) 三元组下的 designer 集合 |
| Min-hyperedge retrieval (优化算法) | Greedy set cover, 目标覆盖 query 的 PKVC factor + scenario probe factor |
| LLM context 构建 | 跟 v3.4 brief peer RAG 同接口, 替换 cosine top-k |

## 跟 v3.4 brief §4 簇内 peer RAG 的升级关系

```
v3.4 默认 RAG:    profile cosine top-k=3, 同簇内 nearest neighbor
og_rag (本模块):  ontology-grounded set cover, 同簇内保 PKVC factor 覆盖均衡

trade-off:
  + 检索更智能 (避免 top-3 都是同 factor 的 peer, 信号冗余)
  + 论文叙事强 (OG-RAG anchor + PKVC framework 结合, EMNLP main contribution)
  - 实现复杂 (要维护 ontology + hypergraph + 优化)
  - 速度慢 (greedy set cover O(k × |hyperedges|))
```

## 文件分工

```
og_rag/
├── README.md         本文档
├── ontology.py       PKVC + SADR ontology schema 定义 + 加载
├── hypergraph.py     从 train fold 数据构建 hypergraph
├── retrieval.py      Greedy set cover retrieval (query → peer examples)
└── demo.py           用 79 designer 数据跑 mini demo (CPU OK)
```

## 接入 brief v3.4 的方式

在 `sft_pipeline/data_prep.py` 中, 把 `retrieve_same_cluster_peers()` 函数从
profile-cosine 实现切换到 `og_rag.retrieval.retrieve_via_og_rag()`, 其他不变。

## 评测 (跟 §5 RAG-abl 对接)

```
RAG-abl 对照组:
  - no_rag:        无 peer context
  - cosine_topk:   profile cosine top-3 (v3.4 默认)
  - og_rag:        本模块 ontology-grounded
对比指标: A1 trajectory match / A2 distribution match
```

## Citation (paper 写作时)

> 本工作的 RAG 模块设计 inspired by OG-RAG (Sharma et al., 2025), 我们将 OG-RAG 的
> hypergraph + 优化检索范式 adapt 到设计师认知簇模拟场景: 用 PKVC 4 因子作为
> ontology 主结构, 30 SADR 节点作为 sub-entity, 把同簇 peer 的 segment 作为
> hypergraph 节点, 按 (cluster, factor, scenario) 三元组聚合为超边。
