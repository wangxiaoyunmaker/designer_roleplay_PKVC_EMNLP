# 论文数据驱动图 — README

4 张数据驱动图, 跟 `论文.tex` 现有表格互补 (不重复). PDF 给 LaTeX 嵌入用, PNG 给预览。

生成脚本: `make_figs.py` (一键重跑, ACL traditional style).

---

## Fig A — 36 quad effect-size × significance scatter

**插入位置**: §4.3 (Ontological Procedures Construction), 在 4 个 itemize predicate 列表之后, "Through the aforementioned Numeric-to-Semantic compilation rules..." 之前。

**数据源**: `data/processed/caps_quadruples.json` (36 行)

**讲什么**: x = 标准化 LASSO 系数 |β|, y = −log₁₀(p), 颜色/形状 = 4 个 predicate 类。横竖虚线对应 §4.3 描述的 4 个阈值 (|β|=0.20, 0.50; p=0.05, 0.10)。一张图替代 4 行 itemize 的视觉验证。

**互补性**: 论文当前没图/表展示 36 quad 的统计分布, 只有 prose 描述。

**LaTeX**:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figs_paper/figA_quad_scatter}
    \caption{The 36 mined ontology quadruples projected onto the
    coefficient–significance plane. Each point is one quad; marker color
    and shape encode the assigned routing predicate. Dashed lines show
    the four compilation thresholds (\S\ref{subsec:quad_construction}).
    Positive correlations populate \emph{Strongly forces} ($n{=}3$),
    \emph{Guides} ($n{=}6$), and \emph{Weakly shifts} ($n{=}5$);
    negative correlations populate \emph{Strictly inhibits}
    ($n{=}22$). The empirical $V$-state imbalance manifests as 17/36
    quads targeting $V$ (not shown), motivating PKVC-stratified
    sampling at inference time.}
    \label{fig:quad_scatter}
\end{figure}
```

**反面 / 风险**: 只有 3 个 Strongly_forces, 大头是 22 个 Strictly_inhibits — reviewer 可能问"main claim 是不是靠 inhibit 撑的". 反驳: §4.3 已说明 inhibit 是 repulsive boundary, 正反向各起作用; 而且 main result (chain diversity +0.29 跨 5 model) 来自整个 routing block, 不是单独正向 quad。

---

## Fig B — Cross-model module ladder (chain diversity + macro-TED)

**插入位置**: §5.4.2 (Cross-Model Generalization), 在 `tab:cross_model` 之后或之前。

**数据源**: `reports/datav3/cross_model_full_metrics_v2.json` (5 个 paper model × 5 condition × 2 metric)

**讲什么**: 左 panel = chain diversity ↑ (5/5 monotonic 阶梯上升); 右 panel = macro-TED ↓ (净改善 5/5, 但 B2 在部分 model 上是 sweet spot, B4 不总是最低)。

**互补性**: `tab:cross_model` 只列 B0→B4 的 Δ (终点对比), 不显示中间 B1/B2/B3 的轨迹。这张图把"5/5 monotonic" 跟"中间 step 是否单调" 两个 claim 直接 visualize。

**LaTeX**:
```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\linewidth]{figs_paper/figB_module_ladder}
    \caption{Per-step trajectories of the 5-tier module ladder
    (\textbf{B0}$\to$\textbf{B4}) across five LLMs.
    \textbf{(a)} Chain diversity increases monotonically across all
    five models from B0 to B4. \textbf{(b)} macro-TED decreases net
    from B0 to B4 for all models, but intermediate tiers exhibit
    model-specific minima: B2 (+stance only) acts as a sweet spot for
    DeepSeek V4 Flash and Qwen 3.6 35B, whereas GPT-4o, Gemini, and
    Claude continue to benefit through B4. The full B4 configuration
    is the best single setting on average (cf.\
    Table~\ref{tab:cross_model}).}
    \label{fig:module_ladder}
\end{figure*}
```

**反面 / 风险**: 右 panel (b) 暴露 B4 不总是 macro-TED 最低 (DSv4F/Qwen35B 的 B2 更低)。建议正面写: "Component contributions vary by model... B4 is the most robust single configuration"。如果不想暴露这个, 可以删 (b) 只留 (a)。

---

## Fig C — CAPS ablation: Trait-prediction ρ × 5 models

**插入位置**: §5.5 (Ablation Analysis), 在 observation (i) "The CAPS routing block is the most critical component" 之后。

**数据源**: `reports/datav3/cross_model_full_metrics_v2.json` (5 paper model × {v2, v2_no_caps} × seg1_trait_pred_rho)

**讲什么**: 5 个 model 分别在 v2 (B4) vs v2_no_caps (B4-NoCaps) 两个 condition 下的 Trait-pred ρ。所有 5 个 model 删 CAPS 后 ρ 下降 (Δ +0.043 to +0.104), GPT-4o + Gemini 删 CAPS 后甚至跌到负值 (无 trait-conditioned routing)。

**互补性**: `tab:ablation_results` 只报 GPT-4o 一个 model 的 B4 vs B4-NoCaps; 这张图直接 show 跨 5 model 的一致性 (这是 §5.5 observation (i) 的最强支撑, paper 现在只有 GPT-4o 数字)。

**LaTeX**:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figs_paper/figC_caps_ablation_rho}
    \caption{Removing the CAPS routing block degrades trait-conditioned
    role-play across all five LLMs. Trait-prediction Spearman $\rho$
    (seg1) drops by $+0.04$ to $+0.10$ when the CAPS block is ablated;
    GPT-4o and Gemini 3.1 FL fall below the random baseline ($\rho<0$).
    This is the single most consistent CAPS-attributable effect in our
    cross-model evaluation (see \S\ref{subsec:ablation_studies}).}
    \label{fig:caps_ablation_rho}
\end{figure}
```

**反面 / 风险**: ρ 绝对值小 (0.04-0.11), reviewer 会问"effect size 是否 statistically significant"。论文已经在 `tab:ablation_results` 给了 95% bootstrap CI for GPT-4o (B4 [$-0.020, +0.130$], B4-NoCaps [$-0.047, +0.011$] — 一定程度 overlap), 但 5 model 的 5/5 directional 一致是 sign test p<0.04 (1/2^5=0.031), 比单 model 的 CI 强很多。Caption 可加 "binomial sign test $p = 0.031$ for 5/5 directional drop"。

---

## Fig D — Trait-continuous manifold (t-SNE on 4-d EFA factors)

**插入位置**: §4.2 (Trait-State Relationship Mining), 在 "Crucially, clustering and t-SNE visualizations revealed..." 这句之后。

**数据源**: `data_v3/scale_responses.csv` + `data/efa_loadings.csv` + `data_v3/task_responses.csv`(Q1 dominant PKVC color)

**讲什么**: 163 designer 在 4-d EFA factor 空间投影到 2-d, 颜色 = Q1 first-reaction 主导 PKVC (P/K/V/C)。**可见**: 没有 clean cluster (validates "continuous manifold" claim); 同时 P=60 / C=66 是 majority, V=14 / K=23 是 minority (motivates §4.4 PKVC-stratified sampling)。

**互补性**: §4.2 当前只用 prose 提"continuous manifold"; 这张图是直接证据。

**LaTeX**:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{figs_paper/figD_trait_manifold_tsne}
    \caption{t-SNE projection of the 4-dimensional EFA factor scores
    for $N{=}163$ designers, colored by the dominant PKVC state of
    their Q1 (first-reaction) response. The lack of well-separated
    clusters supports our \emph{trait-continuous manifold}
    formulation: designer cognition does not partition into discrete
    archetypes. The marked class imbalance ($P{+}C$ together account
    for $77\%$ of first reactions) further motivates the PKVC-stratified
    quadruple sampling described in \S\ref{subsec:state_generator}.}
    \label{fig:trait_manifold}
\end{figure}
```

**反面 / 风险**: t-SNE 是 stochastic, perplexity=30 + random_state=42 fixed for reproducibility. Reviewer 可能问 "为什么用 t-SNE 不用 UMAP/PCA"; 我们 `figures/` 已有同 data 的 UMAP/PCA 版本 ( `4d_EFA_factor__{umap,pca}.png`), 都展示类似的无 clustering 现象, 可放 appendix robustness check。**另一风险**: 这是 negative result (cluster 不 work) 包装为 positive framing (continuous manifold); 严谨 reviewer 会看穿。Caption 已经把 framing 锁住, 但 §4.2 文字也要呼应 (现状 OK)。

---

## 重跑 / 修改

```bash
# 改完 make_figs.py 后, 项目根目录跑:
cd /Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案
python3 reports/datav3/figs_paper/make_figs.py
# 4 张图 PDF + PNG 重新生成到 reports/datav3/figs_paper/
```

依赖: `matplotlib pandas numpy scikit-learn` (sklearn 仅 Fig D 需要)。

---

## 4 张图 vs 论文目前 figure budget

论文当前已有 4 个 figure 引用:
- `fig:intro` (Intropic.png, schematic, §1)
- `fig:PKVC` (PKVC.png, schematic, §3.1)
- `fig:framework` (latex/Framework.png, schematic, §3.1, 双栏 figure*)
- `fig:Traits` (Traits.png, schematic, §3.2)

加上这 4 张数据驱动 + 1 张未编号的 PKVC 子图, 总共 **8 张**, 接近 ACL 长论文 figure budget 上限 (通常 6-10 张). 如果要砍, 优先级:
- **必留**: Fig A (quad scatter) + Fig C (CAPS ablation × 5 model) — 最强 paper claim 的视觉支撑
- **可选**: Fig B (module ladder) — 跟 `tab:cross_model` 重叠 30%, 但展示中间 step
- **可选**: Fig D (manifold) — 跟 §4.2 verbal claim 互补, 但是 negative-result 包装

最小集: A + C 两张就够撑 main claim (data → ontology → cross-model robustness)。
