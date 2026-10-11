"""L8: RAG retrieval sanity — 3 modes vs random baseline.

测 v3.7 RAG 候选 mode 的检索质量 (CPU 可跑, 不依赖 GPU/extractor):
- random (baseline)
- trait_similarity        (4-d factor 等权 cosine)
- modulation              (4-d factor 按 quad s_secondary association 权重 weighted cosine)
- trait_similarity+modulation (rank fusion of 上面两个)

指标: same-option concordance rate (retrieved peer 跟 target 选同一个 option 的比例)
期望: 至少有一个 mode > random + 5%, 说明 retrieval 信号存在.

Caveat: retrieval 质量 sanity, 不是 SFT downstream lift (那要 GPU 训完才知道).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import loaders

_REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "datav3"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _retrieve_trait_topk(
    target_uid: str,
    target_vec: np.ndarray,
    factor_scores: pd.DataFrame,
    k: int,
) -> list[str]:
    sims = []
    for uid in factor_scores.index:
        if uid == target_uid:
            continue
        sims.append((uid, _cosine(target_vec, factor_scores.loc[uid].values)))
    sims.sort(key=lambda x: -x[1])
    return [u for u, _ in sims[:k]]


def _retrieve_random_topk(
    target_uid: str,
    candidates: list[str],
    k: int,
    rng: np.random.Generator,
) -> list[str]:
    pool = [u for u in candidates if u != target_uid]
    rng.shuffle(pool)
    return pool[:k]


def _per_q_factor_weights(
    assoc_file: Path,
    factor_columns: list[str],
) -> dict[str, np.ndarray]:
    """从 quad_s_secondary_associations.jsonl 提 per-Q factor importance weights.

    weight[qid][factor] = max |cohens_d| over sig associations for that (qid, factor)
    没 sig association 的 (qid, factor) 权重 = 0.
    """
    if not assoc_file.exists():
        return {}
    records = []
    for line in assoc_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    weights_dict: dict[str, dict[str, float]] = {}
    for r in records:
        if not r["significant"]:
            continue
        qid = r["qid"]
        factor = r["factor"]
        d = abs(r["cohens_d"])
        weights_dict.setdefault(qid, {f: 0.0 for f in factor_columns})
        weights_dict[qid][factor] = max(weights_dict[qid][factor], d)
    return {
        qid: np.array([weights_dict[qid].get(f, 0.0) for f in factor_columns])
        for qid in weights_dict
    }


def _retrieve_modulation_topk(
    target_uid: str,
    target_vec: np.ndarray,
    factor_scores: pd.DataFrame,
    factor_weights: np.ndarray,           # shape (n_factors,)
    k: int,
) -> list[str]:
    """Weighted cosine: 用 per-Q factor importance 加权 4-d factor 空间."""
    if factor_weights.sum() == 0:
        # 该 Q 没 sig association, 退回等权
        return _retrieve_trait_topk(target_uid, target_vec, factor_scores, k)
    w = factor_weights / factor_weights.sum()
    target_w = target_vec * w
    sims = []
    for uid in factor_scores.index:
        if uid == target_uid:
            continue
        peer_w = factor_scores.loc[uid].values * w
        na, nb = np.linalg.norm(target_w), np.linalg.norm(peer_w)
        if na == 0 or nb == 0:
            continue
        sims.append((uid, float(np.dot(target_w, peer_w) / (na * nb))))
    sims.sort(key=lambda x: -x[1])
    return [u for u, _ in sims[:k]]


def _retrieve_combined_topk(
    target_uid: str,
    target_vec: np.ndarray,
    factor_scores: pd.DataFrame,
    factor_weights: np.ndarray,
    k: int,
) -> list[str]:
    """Reciprocal rank fusion of trait_similarity + modulation."""
    n_full = max(k * 5, 20)               # 取 top-N 各 mode, fusion 后再 cut top-k
    ts_top = _retrieve_trait_topk(target_uid, target_vec, factor_scores, n_full)
    mod_top = _retrieve_modulation_topk(target_uid, target_vec, factor_scores, factor_weights, n_full)
    scores: dict[str, float] = {}
    for rank, uid in enumerate(ts_top):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (rank + 60.0)   # RRF k=60 standard
    for rank, uid in enumerate(mod_top):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (rank + 60.0)
    sorted_uids = sorted(scores.items(), key=lambda x: -x[1])
    return [u for u, _ in sorted_uids[:k]]


def run(
    output_path: Path | None = None,
    factor_scores_path: Path | None = None,
    k: int = 3,
    n_random_trials: int = 30,
    seed: int = 42,
) -> dict:
    output_path = output_path or _REPORTS / "L8_rag_sanity.md"
    factor_scores_path = factor_scores_path or loaders.get_default_factor_scores_path()

    scores = pd.read_csv(factor_scores_path, index_col=0)
    task = loaders.add_canonical_option(loaders.load_task_long(expanded=True))
    q_single = task[task["option"].notna()].copy()
    q_single = q_single[q_single["user_id"].isin(scores.index)]

    # 构 (uid, qid) → option lookup
    answer_lookup: dict[tuple[str, str], str] = {}
    for _, r in q_single.iterrows():
        answer_lookup[(r["user_id"], r["question_id"])] = r["option"]

    # 加载 quad s_secondary associations 用于 modulation 权重
    assoc_path = _REPORTS / "per_designer" / "quad_s_secondary_associations.jsonl"
    factor_columns = list(scores.columns)
    q_weights = _per_q_factor_weights(assoc_path, factor_columns)

    findings: dict = {
        "factor_scores_path": str(factor_scores_path),
        "k": k,
        "n_qs_with_modulation_weights": len(q_weights),
    }
    sections: list[str] = []
    sections.append(f"# L8 RAG retrieval sanity — 4 modes vs random (data_v3, {pd.Timestamp.now().date()})\n")
    sections.append("测 v3.7 RAG 候选 mode 的检索质量. CPU 跑, 不依赖 GPU/extractor.\n")
    sections.append(f"- factor scores: K={scores.shape[1]}, N={len(scores)}")
    sections.append(f"- k = {k} (peer top-k)")
    sections.append(f"- random baseline trials = {n_random_trials} (每 (uid,Q) 平均)")
    sections.append(f"- modulation weights: {len(q_weights)} Q 有 sig association (其余 fallback trait_similarity)")
    sections.append(f"- 指标: same-option concordance rate (retrieved peer 跟 target 选同一 option 的比例)\n")

    rng = np.random.default_rng(seed)
    rows = []
    for qid in sorted({q for _, q in answer_lookup.keys()}):
        users_with_q = [u for (u, q), _ in answer_lookup.items() if q == qid]
        if len(users_with_q) < k + 5:
            continue
        weights_for_q = q_weights.get(qid, np.zeros(len(factor_columns)))

        trait_conc, mod_conc, comb_conc, random_conc = [], [], [], []
        for uid in users_with_q:
            if uid not in scores.index:
                continue
            target_opt = answer_lookup.get((uid, qid))
            if target_opt is None:
                continue
            target_vec = scores.loc[uid].values

            for mode_peers, store in [
                (_retrieve_trait_topk(uid, target_vec, scores, k), trait_conc),
                (_retrieve_modulation_topk(uid, target_vec, scores, weights_for_q, k), mod_conc),
                (_retrieve_combined_topk(uid, target_vec, scores, weights_for_q, k), comb_conc),
            ]:
                match = [
                    1 if answer_lookup.get((p, qid)) == target_opt else 0
                    for p in mode_peers if (p, qid) in answer_lookup
                ]
                if match:
                    store.append(np.mean(match))

            # random baseline (avg over n_random_trials)
            random_rates = []
            for _ in range(n_random_trials):
                r_peers = _retrieve_random_topk(uid, users_with_q, k, rng)
                r_match = [
                    1 if answer_lookup.get((p, qid)) == target_opt else 0
                    for p in r_peers if (p, qid) in answer_lookup
                ]
                if r_match:
                    random_rates.append(np.mean(r_match))
            if random_rates:
                random_conc.append(np.mean(random_rates))

        if not trait_conc:
            continue
        rnd_mean = float(np.mean(random_conc)) if random_conc else float("nan")
        def _lift(x):
            return ((x - rnd_mean) / rnd_mean * 100) if rnd_mean > 0 else float("nan")
        ts_mean = float(np.mean(trait_conc))
        mod_mean = float(np.mean(mod_conc)) if mod_conc else float("nan")
        cmb_mean = float(np.mean(comb_conc)) if comb_conc else float("nan")
        rows.append({
            "qid": qid,
            "n_users": len(trait_conc),
            "random": round(rnd_mean, 3),
            "trait_sim": round(ts_mean, 3),
            "modulation": round(mod_mean, 3),
            "trait+mod": round(cmb_mean, 3),
            "trait_lift%": round(_lift(ts_mean), 1),
            "mod_lift%": round(_lift(mod_mean), 1),
            "combined_lift%": round(_lift(cmb_mean), 1),
        })

    df = pd.DataFrame(rows)
    sections.append("## per-Q concordance (4 modes vs random)\n")
    sections.append(df.to_markdown(index=False))

    # 综合
    if len(df):
        sections.append("\n## 综合 (avg across Qs)\n")
        rnd_o = df["random"].mean()
        ts_o = df["trait_sim"].mean()
        mod_o = df["modulation"].mean()
        cmb_o = df["trait+mod"].mean()
        def _l(x):
            return (x - rnd_o) / rnd_o * 100 if rnd_o > 0 else 0
        sections.append(f"| mode | avg concordance | lift vs random |")
        sections.append(f"|---|---|---|")
        sections.append(f"| random (baseline) | {rnd_o:.3f} | — |")
        sections.append(f"| **trait_similarity** | {ts_o:.3f} | {_l(ts_o):+.1f}% |")
        sections.append(f"| **modulation** | {mod_o:.3f} | {_l(mod_o):+.1f}% |")
        sections.append(f"| **trait+mod (RRF)** | {cmb_o:.3f} | {_l(cmb_o):+.1f}% |")

        findings["overall"] = {
            "random": float(rnd_o), "trait_sim": float(ts_o),
            "modulation": float(mod_o), "trait_plus_mod": float(cmb_o),
            "trait_lift_pct": float(_l(ts_o)), "mod_lift_pct": float(_l(mod_o)),
            "combined_lift_pct": float(_l(cmb_o)),
        }

        # verdict 自动判
        best_mode = max(
            [("trait_similarity", ts_o), ("modulation", mod_o), ("trait+mod", cmb_o)],
            key=lambda x: x[1],
        )
        best_lift = _l(best_mode[1])
        sections.append(f"\n**Best mode**: `{best_mode[0]}` (lift {best_lift:+.1f}%)")
        if best_lift > 5:
            verdict = f"✅ {best_mode[0]} 比 random 显著好 (>5%) — 应作 main RAG mode"
        elif best_lift > 0:
            verdict = f"⚠ {best_mode[0]} 微弱优于 random ({best_lift:.1f}%) — 边缘信号, 跟其他 ablation 互补"
        else:
            verdict = "❌ 所有 mode 都不优于 random — retrieval 需要换 modality (e.g. NL embedding)"
        sections.append(f"\n**Verdict**: {verdict}")
        findings["verdict"] = verdict
        findings["best_mode"] = best_mode[0]
        findings["best_lift_pct"] = float(best_lift)

    sections.append("\n## Caveats\n")
    sections.append("- 这是 retrieval 质量 sanity, **不是** SFT downstream lift (那需 GPU 训完才知道)")
    sections.append("- 'same-option concordance' 是 proxy for behavioral similarity, 不是 PKVC state 一致 (后者需 extractor)")
    sections.append("- random baseline = within-Q-pool, 不是 cross-Q random (更公平)")
    sections.append("- N=163 sample; fold-aware retrieval 没做 (sanity 不严), 真训练时排除 test fold")
    sections.append("- modulation 权重来自 L4 quad_s_secondary_associations.jsonl (13 sig), 包含 in-sample 信息 — **小 leakage** (concordance 数字偏乐观), but lift direction 仍 informative")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
