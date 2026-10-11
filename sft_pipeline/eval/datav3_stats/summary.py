"""SUMMARY: one-page rollup across L1-L5, L7 with brief decision impact."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def _load_json(name: str) -> dict:
    p = _REPORTS / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _fmt(v, fmt: str = "?") -> str:
    if v is None:
        return "?"
    try:
        return fmt.format(v) if "{" in fmt else str(v)
    except (TypeError, ValueError):
        return str(v)


def run(output_path: Path | None = None) -> None:
    output_path = output_path or _REPORTS / "SUMMARY.md"

    L1 = _load_json("L1_health")
    L1b = _load_json("L1b_expansion_audit")
    L2 = _load_json("L2_efa")
    L3 = _load_json("L3_cluster")
    L4 = _load_json("L4_mapping")
    L3_multi = _load_json("L3_multi_source")
    L3_gmm = _load_json("L3_gmm_behavior")
    L4b = _load_json("L4b_lasso_check")
    L4c = _load_json("L4c_lasso_30items")
    L5 = _load_json("L5_bridge")
    L7 = _load_json("L7_audit")

    sections: list[str] = []
    sections.append(f"# data_v3 Statistical Observation — SUMMARY ({pd.Timestamp.now().date()})\n")
    sections.append(f"N = {L1.get('n_users', '?')} designers, 30 量表 + Q1-Q8 (Q2 排序 / Q7 NL / Q8 单选)\n")

    # ====== Layer verdicts ======
    sections.append("## Layer verdicts\n")
    sections.append("| Layer | Headline | Verdict |")
    sections.append("|---|---|---|")

    # L1
    n_users = L1.get("n_users", "?")
    pkvc_ratio = L1.get("pkvc_imbalance_ratio")
    pkvc_str = f"{pkvc_ratio:.1f}×" if pkvc_ratio else "?"
    sections.append(f"| L1 health | N={n_users}, 30 量表完整, PKVC imbalance ratio={pkvc_str} (C:K vs V:P) | {'PASS' if n_users and n_users >= 150 else 'WARN'} |")
    # L1b
    cov_str = f"{L1b.get('n_expanded', 0)}/978 ({L1b.get('n_expanded', 0) / 9.78:.1f}%)"
    sb = L1b.get("selection_bias_verdict", "?")
    sections.append(f"| L1b expansion | coverage={cov_str}, {sb} | INFO |")
    # L2
    kmo = L2.get("kmo")
    kmo_str = f"{kmo:.2f}" if kmo else "?"
    pa = L2.get("pa_suggested_k", "?")
    sections.append(f"| L2 EFA | KMO={kmo_str}, Bartlett p<1e-41, **PA 推荐 K={pa}** | {'PASS' if kmo and kmo >= 0.6 else 'WARN'} |")
    # L3 (4-d EFA factor)
    picked_k = L3.get("picked_k", "?")
    sweep = L3.get("sweep", [])
    sil_max = max((r.get("silhouette", 0) for r in sweep), default=0)
    chi_results = L3.get("chi_results", {})
    total_sig = sum(sum(1 for v in q_chi.values() if v.get("p", 1.0) < 0.05) for q_chi in chi_results.values())
    sections.append(f"| L3 cluster (4-d EFA) | K-sweep max silhouette={sil_max:.3f}, picked K={picked_k}, 累计 {total_sig} sig | **WARN — cluster 弱区分** |")
    # L3-multi (3 source spaces)
    if L3_multi:
        rows = []
        for src_name, f in L3_multi.items():
            best_sil = max(r["silhouette"] for r in f.get("sweep", []))
            max_sig = max(
                (sum(1 for v in qc.values() if v.get("p", 1) < 0.05) for qc in f.get("chi_results", {}).values()),
                default=0,
            )
            rows.append(f"{src_name}={best_sil:.2f}/{max_sig}")
        sections.append(f"| L3-multi (3 source spaces) | best_sil/χ²sig: {', '.join(rows)} | **WARN — 30-d 比 4-d 还差, behavior 略好但 gap=负** |")
    # L3-GMM
    if L3_gmm:
        bic_best = L3_gmm.get("best_k_bic", "?")
        # detect monotonic = over-fit
        bic_vals = [r["BIC"] for r in L3_gmm.get("sweep", [])]
        mono = all(bic_vals[i] >= bic_vals[i + 1] for i in range(len(bic_vals) - 1)) if bic_vals else False
        sil_peak_k = "?"
        if L3_gmm.get("sweep"):
            sil_peak_k = max(L3_gmm["sweep"], key=lambda r: r["silhouette"])["K"]
        sections.append(f"| L3-GMM (behavior) | BIC monotonic={'YES (over-fit)' if mono else 'no'}, BIC-K={bic_best}, silhouette-K={sil_peak_k} | **WARN — Gaussian-on-binary 不对路, behavior = sparse fingerprint 不是 clean mode** |")
    # L4
    n_assoc = L4.get("n_associations", L4.get("n_triples", 0))  # back-compat
    n_sig = L4.get("n_significant", 0)
    sig_rate = (n_sig / n_assoc * 100) if n_assoc else 0
    sections.append(f"| L4 quad s_secondary | {n_sig}/{n_assoc} (factor,q,opt) associations sig ({sig_rate:.0f}%) | {'PASS' if n_sig >= 5 else 'WARN'} |")
    # L4b
    agree = L4b.get("intersect_with_L4", 0)
    l4sig = L4b.get("L4_sig_total", 0)
    sections.append(f"| L4b LASSO (4-factor) | **{agree}/{l4sig} L4 sig agreed** with LASSO 同号 | {'PASS' if agree == l4sig and l4sig > 0 else 'WARN'} |")
    # L4c
    n_obs = L4c.get("total_observations", 0)
    n_sig_items = 0
    for opts in L4c.get("by_q", {}).values():
        for items in opts.values():
            for it in items:
                if (it.get("marginal_perm_p") or 1.0) < 0.05:
                    n_sig_items += 1
    sections.append(f"| L4c LASSO (30-item) | **{n_obs} 段 Statistical Observation** (CPO Compiler Prompt 格式), {n_sig_items} 个 item-level marginal p<.05 | INFO |")
    # L5
    robust = L5.get("robust_triples_after_L5", 0)
    n_l5 = L5.get("n_sig_associations", L5.get("n_sig_triples", 0))  # back-compat
    killed = L5.get("covariate_attack_killed", 0)
    sections.append(f"| L5 bridge | **{robust}/{n_l5} robust** (boot>0.9 + covariate drop<30%), {killed} 被 demographic 杀 | {'PASS' if robust == n_l5 and n_l5 > 0 else 'WARN'} |")
    # L7
    wd = L7.get("wechat_dup")
    full = L7.get("users_full_q18", "?")
    sections.append(f"| L7 audit | wechat_dup={wd}, 完整 Q1-Q8 user = {full}/{L7.get('users_missing_task', 0) + (L7.get('users_full_q18', 0) or 0)} | {'PASS' if wd == 0 else 'FAIL'} |")

    # ====== 3 个独立 K 的解耦 ======
    # 警告: EFA factor count, K-means K, PKVC state count 是 3 个独立量, **数值相等是巧合不是对齐**.
    # 见 brief v3.5.1 §3 framing correction + docs/notes/framing_correction_20260522.md.
    efa_k = L2.get("pa_suggested_k", "?")
    cluster_k = L3.get("picked_k", "?")
    cluster_reason = L3.get("picked_reason", "")
    sections.append("\n## 3 个独立 K — 解耦, 数值相等是巧合不是对齐\n")
    sections.append("| 维度 | 值 | 数据来源 | 用途 |")
    sections.append("|---|---|---|---|")
    sections.append(f"| **EFA factor count** K_EFA | {efa_k} | 30 题量表 + Horn parallel analysis | L_cls head 维度 + L4 mapping 的 source factor 空间 |")
    sections.append(f"| **K-means K** K_clust | {cluster_k} | 163 designer 在 trait-factor 空间聚类 (L3 任务导向规则: {cluster_reason}) | cluster conditioning + 簇内 peer RAG scope (注: L3 显示 cluster 弱区分, 下游价值待评估) |")
    sections.append(f"| **PKVC state count** K_state | 4 | **理论锁** (Mischel CAPS → P/K/V/C state machine), **不是 data-driven** | runtime state machine target + L4 v2 target (extractor 解锁后用) |")
    sections.append("\n关键 framing (brief v3.5.1 §3):")
    sections.append("- CAPS 是 *理论* (Mischel 1995 connectionist network), **不是** 30 题量表")
    sections.append("- 30 题量表测的是 CAPS network weights (static individual difference), EFA factor count = 这些 weights 的潜变量结构")
    sections.append("- PKVC 是 *runtime 状态机*, **不是** trait scale, **不是** 输出 behavior")
    sections.append(f"- K_EFA={efa_k} 跟 K_state=4 数值相等是巧合; 即使 PA 出 K_EFA=3 或 5, PKVC 仍然 4 state, 不冲突")
    sections.append(f"- K_clust={cluster_k} 跟 K_EFA / K_state 也无关, 是第 3 个独立量")

    # ====== Decisions unlocked ======
    sections.append("\n## Decisions unlocked (brief v3.5.1 §8.4 / §8.5)\n")
    sections.append("- **N final**: 200 → **163**, brief §1 + sft_pipeline/__init__.py + 写稿 grep 全部统一")
    sections.append(f"- **D-18 EFA factor count final**: PA 推荐 K_EFA = {efa_k}, 不再等 200 入袋 (跟 PKVC 4 state 无关, 两者独立)")
    sections.append(f"- **K cluster**: L3 picked K = {cluster_k}, 但 cluster 在 Q-option 上 χ² 极弱 (累计 {total_sig} sig)")
    sections.append(f"- **D-15 (per-cluster vs global mapping)**: cluster 弱区分 → **global mapping main**, per-cluster sensitivity (推荐翻 D-15 default)")
    sections.append(f"- **bypass / strength 阈值**: data-driven calibration (L5 §5.4): bypass<{L5.get('d_quantiles', {}).get('q25', 0):.2f}, Weak<{L5.get('d_quantiles', {}).get('q75', 0):.2f}, Strong>={L5.get('d_quantiles', {}).get('q75', 0):.2f}")
    sections.append(f"- **blocker E**: L1b 显示 expansion 已 {L1b.get('n_expanded', 0)} 行 (selection bias PASS, median 102 字), blocker 实质已解 → 改 CLAUDE.md §5")
    sections.append("- **4 层 cluster conditioning**: 经验基础不牢 (L3 + L3-multi + L3-GMM + L3-visual), framing 改 trait-continuous + behavior-fingerprint (quad ontology 不变, 见 docs/notes/CAPS四元组与簇内RAG_20260523.md)")
    sections.append("- **T_CAPS quad pipeline 进度**: s_primary (L4c option_trait_mapping.json) + s_secondary (L4 quad_s_secondary_associations.jsonl) + p_modulation 阈值 (L5) 已就绪; ⏳ p_surface + o_primary LLM 抽待 blocker A")

    # ====== Files ======
    sections.append("\n## 输出文件 (reports/datav3/, gitignored — 留本地)\n")
    sections.append("- L1_health.md, L1b_expansion_audit.md")
    sections.append("- L2_efa.md, L3_cluster.md")
    sections.append("- L4_mapping.md, L4b_lasso_check.md")
    sections.append("- L5_bridge.md, L7_audit.md")
    sections.append("- SUMMARY.md (this file)")
    sections.append("- per_designer/ (gitignored): factor_scores_k{3,4,5,6}.csv, loadings_k{3,4,5,6}.csv, user_clusters_k{3,4,5,6,7}.csv, **quad_s_secondary_associations.jsonl** (L4), **option_trait_mapping.json** (L4c, quad Step 2a 标准 schema)")

    # ====== Next ======
    sections.append("\n## 下一步建议\n")
    sections.append("1. 跟一作对齐 L1b expansion subset bias + L2/L3 推荐 K + L3 cluster 弱区分发现")
    sections.append("2. 把 L4 + L5 通过的 13 个 robust quad s_secondary association 当 paper §4 main set; blocker A 解后跑 quad pipeline Step 2b/c 生成 800 完整 quad")
    sections.append("3. 跟老师同步:")
    sections.append("   - D-15 翻案 (per-cluster → global mapping)")
    sections.append("   - 4 层 cluster conditioning 必要性重审 (cluster χ² 弱)")
    sections.append("   - K cluster 选 K=3 还是去掉 cluster conditioning")
    sections.append("4. update CLAUDE.md §5 blocker E (已解) + brief v3.5.1 §1 数字 (200→163) + 加 v3.5.2 升级")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {_REPORTS / 'SUMMARY.md'}")
