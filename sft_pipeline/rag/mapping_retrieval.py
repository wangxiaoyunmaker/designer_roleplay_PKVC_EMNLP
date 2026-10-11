"""Modulation-triple retrieval + prompt block formatting.

⚠ v3.5.2 (2026-05-23) schema 升级状态:
    schema 已升级到 T_CAPS quad v1.1 (见 docs/briefs/T_CAPS_quad_schema_v1.md
    + sft_pipeline/rag/quad_schema.py). **本文件实现暂留 v3.5.1 §4D 三元组**,
    等学姐数据齐 → mapping_miner.py 升级 → 本文件接 quad DB 实现:
    - retrieval key = (C_local, s_primary), Jaccard / cosine top-k
    - 加 filter_by_situation(C_local_tag) 作题级过滤层
    - 学姐数据 ready 之前, 走原 (factor, direction) → (state, modulation) 检索
    TODO (v3.5.3): rewrite to consume TCAPSQuad records, add C_local filter.

─── 以下是 v3.5.1 §4D 原 docstring (保留供 reference) ───

Pairs with mapping_miner.py: that module emits per-fold modulation_fold_{k}.jsonl,
this module loads them and formats top-K triples into the [System] prompt block
for SFT inputs.

Path γ injection (brief §4D.4 default): triples appear in the *system* prompt as
trait-conditioned routing hints, separate from the [Peers] / [Profile] blocks.

Retrieval strategy (v1, intentionally simple):
    1. Filter triples by target_user's cluster_id (per-cluster). Optionally also
       include global triples (cluster_id == null) as supplementary signal.
    2. Score each triple by relevance to the target user's actual factor profile:
         - If target user's factor[F] direction matches triple.source.direction: keep,
           boost by |Δ| × (1 + |z_score|)
         - Otherwise: down-weight (×0.1) so it can still appear but ranks low
    3. Take top-K (default 5).

If you don't have the target user's factor scores at retrieval time (e.g. eval-time
deployment with only 30-d Likert), pass `target_factor_scores=None` and the function
falls back to ranking by |Δ| alone, which is still cluster-conditioned because of
step 1.

API mirrors rag.peer.retrieve_via_og_rag for drop-in use in data_prep.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("sft_pipeline.rag.mapping_retrieval")


# =====================================================================
# Schema (mirrors mapping_miner.ModulationTriple.to_dict())
# =====================================================================

@dataclass
class LoadedTriple:
    factor_id: str
    factor_name: str
    direction: str          # "high" / "low"
    modulation_type: str    # Strong-Inhibit / Weak-Inhibit / Weak-Excite / Strong-Excite
    bypass: bool
    delta: float
    p_value: float
    pkvc_state: str         # P / K / V / C
    cluster_id: int | None  # None == global
    support: int
    fold_id: int


@dataclass
class ModulationRetrievalResult:
    selected: list[LoadedTriple]
    n_candidates: int
    target_cluster: int | None
    fold_id: int


# =====================================================================
# Loading
# =====================================================================

def load_modulation_triples(jsonl_path: Path) -> list[LoadedTriple]:
    """Load mapping_miner output JSONL into LoadedTriple objects."""
    triples: list[LoadedTriple] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj.get("source", {})
            mod = obj.get("modulation", {})
            tgt = obj.get("target", {})
            triples.append(LoadedTriple(
                factor_id=str(src.get("factor_id", "")),
                factor_name=str(src.get("factor_name", "")),
                direction=str(src.get("direction", "")),
                modulation_type=str(mod.get("type", "")),
                bypass=bool(mod.get("bypass", False)),
                delta=float(mod.get("delta", 0.0)),
                p_value=float(mod.get("p_value", 1.0)),
                pkvc_state=str(tgt.get("pkvc_state", "")),
                cluster_id=obj.get("cluster_id"),
                support=int(obj.get("support", 0)),
                fold_id=int(obj.get("fold_id", -1)),
            ))
    logger.info("loaded %d modulation triples from %s", len(triples), jsonl_path)
    return triples


# =====================================================================
# Retrieval
# =====================================================================

def retrieve_modulations(
    target_user_id: str,
    target_cluster: int,
    all_triples: list[LoadedTriple],
    target_factor_scores: dict[str, float] | None = None,
    factor_medians: dict[str, float] | None = None,
    top_k: int = 5,
    include_global: bool = True,
) -> ModulationRetrievalResult:
    """Retrieve top-K cluster-scoped triples relevant to target user.

    Args:
        target_user_id: for logging only (kept for API symmetry with retrieve_via_og_rag)
        target_cluster: target user's cluster_id; only same-cluster triples scored higher
        all_triples: full pool loaded via load_modulation_triples
        target_factor_scores: optional {factor_id: score} for relevance ranking;
            if absent, ranks by |Δ| only
        factor_medians: optional {factor_id: median} train-fold split point;
            needed to decide if target_user is "high" or "low" on each factor.
            If absent and target_factor_scores given, falls back to 0.0 as median.
        top_k: how many triples to include in prompt
        include_global: also include cluster_id=None triples (population-level rules)

    Returns:
        ModulationRetrievalResult with selected list (length ≤ top_k)
    """
    candidates = [
        t for t in all_triples
        if t.cluster_id == target_cluster
        or (include_global and t.cluster_id is None)
    ]
    if not candidates:
        return ModulationRetrievalResult(
            selected=[], n_candidates=0,
            target_cluster=target_cluster, fold_id=-1,
        )

    def score(t: LoadedTriple) -> float:
        s = abs(t.delta) * (1.5 if t.cluster_id == target_cluster else 1.0)
        if t.bypass:
            s += 0.2  # bypass is high-information-density flag
        if target_factor_scores is None:
            return s
        user_score = target_factor_scores.get(t.factor_id)
        if user_score is None:
            return s * 0.5
        median = (factor_medians or {}).get(t.factor_id, 0.0)
        user_dir = "high" if user_score > median else "low"
        if user_dir == t.direction:
            return s * (1.0 + min(abs(user_score - median), 3.0))
        return s * 0.1   # opposite direction — keep but down-rank

    ranked = sorted(candidates, key=score, reverse=True)
    return ModulationRetrievalResult(
        selected=ranked[:top_k],
        n_candidates=len(candidates),
        target_cluster=target_cluster,
        fold_id=ranked[0].fold_id if ranked else -1,
    )


# =====================================================================
# Prompt block formatting (Path γ: system prompt)
# =====================================================================

def format_modulation_system_block(
    result: ModulationRetrievalResult,
    max_chars: int = 1200,
) -> str:
    """Render top-K triples as a [System] sub-block per brief v3.5.1 §4D.4 example."""
    if not result.selected:
        return ""
    lines = [
        f"[Modulation_hints] (你所在 cluster {result.target_cluster} 的 trait→PKVC 调节规律, "
        f"fold {result.fold_id}, top {len(result.selected)} of {result.n_candidates})",
    ]
    char_count = len(lines[0])
    for i, t in enumerate(result.selected, 1):
        scope = f"cluster {t.cluster_id}" if t.cluster_id is not None else "global"
        bypass = " [Bypass]" if t.bypass else ""
        block = (
            f"\n  规则 {i}: 在 '{t.factor_name}' ({t.factor_id}) 上 {t.direction} → "
            f"PKVC {t.pkvc_state} 状态 {t.modulation_type}{bypass} "
            f"(Δ={t.delta:+.2f}, p={t.p_value:.3f}, support={t.support}, {scope})"
        )
        if char_count + len(block) > max_chars:
            lines.append(f"\n  ... ({len(result.selected) - i + 1} more truncated)")
            break
        lines.append(block)
        char_count += len(block)
    return "".join(lines)


# =====================================================================
# Helpers for data_prep integration
# =====================================================================

def compute_factor_medians(factor_scores_csv: Path) -> dict[str, float]:
    """Population-level factor median, for "high vs low" inference at inject time.

    Read once at startup; used for every target user during data_prep prompt build.
    """
    df = pd.read_csv(factor_scores_csv)
    out: dict[str, float] = {}
    for col in df.columns:
        if df[col].dtype.kind in "fc":
            out[col] = float(np.median(df[col].dropna()))
    return out


def load_factor_scores_dict(
    factor_scores_csv: Path,
) -> dict[str, dict[str, float]]:
    """{user_id: {factor_id: score}}."""
    df = pd.read_csv(factor_scores_csv)
    user_col = next((c for c in ("user_id", "designer_id", "wechat_id") if c in df.columns), None)
    if not user_col:
        raise ValueError(f"factor_scores_csv {factor_scores_csv} missing user_id-like col")
    out: dict[str, dict[str, float]] = {}
    factor_cols = [c for c in df.columns if c != user_col and df[c].dtype.kind in "fc"]
    for _, row in df.iterrows():
        uid = str(row[user_col])
        out[uid] = {fc: float(row[fc]) for fc in factor_cols}
    return out
