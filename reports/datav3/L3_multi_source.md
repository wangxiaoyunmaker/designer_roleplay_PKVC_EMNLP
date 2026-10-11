# L3-multi — cluster sweep on 3 source spaces (data_v3, 2026-05-24)

L3 主报告只在 **4-d EFA factor** 空间跑 K-means, picked K=3 fallback, cluster 弱区分.
这里换 3 个不同 source space, 比较是否 method artifact (4-d EFA 太压缩) 还是真信号.
方法: 同 K-means + silhouette/DB/gap + per-cluster Q-option χ² (跟 L3 主同).


## Source: `30d_trait`  (shape = (163, 30))

**Sweep metrics:**

|   K |   silhouette |   davies_bouldin |   gap |   min_cluster_n | per_cluster_n                |
|----:|-------------:|-----------------:|------:|----------------:|:-----------------------------|
|   3 |        0.065 |            3.132 | 0.206 |              30 | [30, 48, 85]                 |
|   4 |        0.05  |            3.289 | 0.206 |              28 | [40, 57, 38, 28]             |
|   5 |        0.04  |            3.103 | 0.208 |              12 | [44, 34, 12, 36, 37]         |
|   6 |        0.046 |            3.149 | 0.207 |              19 | [19, 42, 19, 25, 22, 36]     |
|   7 |        0.037 |            3.096 | 0.212 |              16 | [18, 21, 22, 16, 29, 32, 25] |

**per-K χ² sig count (over Q1/Q3-Q6/Q8 单选题):**

- K=3: **0 sig** ([])
- K=4: **0 sig** ([])
- K=5: **2 sig** (['Q5', 'Q8'])
- K=6: **2 sig** (['Q5', 'Q8'])
- K=7: **0 sig** ([])

**picked K (任务导向规则)**: 3
- fallback: 无 K 同时满足 (min_n>=20, sig_Q>=2, sil knee within 0.02); 取 silhouette 最优 K=3

## Source: `Q_option_behavior`  (shape = (163, 23))

**Sweep metrics:**

|   K |   silhouette |   davies_bouldin |    gap |   min_cluster_n | per_cluster_n                |
|----:|-------------:|-----------------:|-------:|----------------:|:-----------------------------|
|   3 |        0.099 |            2.558 | -0.616 |              42 | [64, 57, 42]                 |
|   4 |        0.096 |            2.52  | -0.592 |              31 | [31, 50, 39, 43]             |
|   5 |        0.109 |            2.363 | -0.562 |              26 | [26, 29, 37, 31, 40]         |
|   6 |        0.099 |            2.2   | -0.559 |              15 | [32, 34, 20, 30, 15, 32]     |
|   7 |        0.099 |            2.171 | -0.541 |              11 | [11, 26, 29, 32, 17, 27, 21] |

**per-K χ² sig count (over Q1/Q3-Q6/Q8 单选题):**

- K=3: **5 sig** (['Q1', 'Q3', 'Q4', 'Q6', 'Q8'])
- K=4: **6 sig** (['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8'])
- K=5: **6 sig** (['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8'])
- K=6: **6 sig** (['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8'])
- K=7: **6 sig** (['Q1', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8'])

**picked K (任务导向规则)**: 3
- min K with per-cluster N>=20, sig on 5 Q, silhouette within 0.02 of max (0.109)

## Source: `joint_trait_plus_behavior`  (shape = (163, 53))

**Sweep metrics:**

|   K |   silhouette |   davies_bouldin |   gap |   min_cluster_n | per_cluster_n               |
|----:|-------------:|-----------------:|------:|----------------:|:----------------------------|
|   3 |        0.06  |            3.249 | 0.128 |              25 | [25, 55, 83]                |
|   4 |        0.04  |            3.447 | 0.129 |              31 | [44, 31, 45, 43]            |
|   5 |        0.04  |            3.261 | 0.138 |               9 | [9, 21, 38, 52, 43]         |
|   6 |        0.033 |            3.43  | 0.128 |              16 | [28, 32, 16, 32, 34, 21]    |
|   7 |        0.042 |            3.087 | 0.137 |               5 | [20, 28, 34, 20, 12, 44, 5] |

**per-K χ² sig count (over Q1/Q3-Q6/Q8 单选题):**

- K=3: **0 sig** ([])
- K=4: **3 sig** (['Q4', 'Q5', 'Q8'])
- K=5: **2 sig** (['Q5', 'Q8'])
- K=6: **2 sig** (['Q5', 'Q8'])
- K=7: **1 sig** (['Q8'])

**picked K (任务导向规则)**: 4
- min K with per-cluster N>=20, sig on 3 Q, silhouette within 0.02 of max (0.060)

---

## 跨 source 对照总结

| Source | shape | best silhouette | picked K | max χ² sig Q (across K) |
|---|---|---|---|---|
| 30d_trait | (163, 30) | 0.065 | 3 | 2 |
| Q_option_behavior | (163, 23) | 0.109 | 3 | 6 |
| joint_trait_plus_behavior | (163, 53) | 0.060 | 4 | 3 |

参考: L3 主报告 4-d EFA factor 空间 best silhouette ≈ 0.199, max χ² sig = 1.


## Verdict

- silhouette 都弱, 但 source `Q_option_behavior` 在 χ² 上 sig 较多
- → cluster 形状可能非球状 (K-means 假设错), 建议下一步加 GMM / DBSCAN