"""mapping_miner.py — Mine trait→PKVC modulation triples.

⚠ v3.5.2 (2026-05-23) schema 升级状态:
    schema 定义已经升级到 T_CAPS quad v1.1 (见 docs/briefs/T_CAPS_quad_schema_v1.md
    + sft_pipeline/rag/quad_schema.py). **本文件算法实现暂留 v3.5.1 §4D 三元组**,
    等学姐 PKVC 12 cardId metadata + Q3-Q6 选项-因子 mapping 数据齐了, 再升级到:
    - hierarchical mining: step 1 (s_secondary, p_modulation, o_primary) 三元
                           step 2 C_local 作 moderator (permutation test)
    - o_primary 4 大类作 main grid (不用 o_subtype, 数据量不够支撑 12 子类)
    TODO (v3.5.3): rewrite to consume TCAPSQuad records instead of separate
                   factor_scores + state_distributions, 简化输入接口.

─── 以下是 v3.5.1 §4D 原 docstring (保留供 reference) ───

Operationalize Mischel 1995 CAPS theory: 30-dim trait factor scores modulate
PKVC state-transition probabilities. For each (factor, direction, PKVC state)
candidate, test whether the observed state activation rate in designers exhibiting
that trait pattern differs significantly from the population baseline (permutation
test, n=1000), and classify the modulation strength into 4 buckets symmetric to the
trait 4-bucket polarity.

Background — brief v3.5.1 §4D framing (3-layer architecture):
    Trait Layer    : 30-dim belief survey → EFA → 4-5 factors (factor scores per
                     designer = that designer's CAPS network weights, static).
    CAPS Theory    : Mischel 1995 — connecting mechanism, not a layer. Asserts that
                     trait modulates state-transition probabilities in an
                     "If [trait + context] Then [state activation]" form.
    PKVC State Mach: 4-state runtime engine (P/K/V/C). 30 SADR nodes are sub-divisions.
    This miner    : Empirically estimates the "If-Then" function as triples of the form
                     <factor pattern, modulation, PKVC state>, e.g.
                     <"高 F2 (数据驱动)", "Strong-Inhibit", "Concept-Phase">.

Inputs:
    1. user_factor_scores.csv     designer × {F1, F2, ...} EFA factor scores
    2. state_distributions.jsonl  per-designer PKVC state activation distribution
                                   {user_id, distribution: {P, K, V, C}} (each row
                                   sums to 1; defensively renormalized on load)
       --or--
       extracted.jsonl            extractor_batch.py output; state distribution
                                   computed inline by counting `dominant_factor`
                                   mentions (parsed as factor path) per segment
                                   (--extracted flag)
    3. user_cluster_efa_v2_200.csv designer_id → cluster_id (30-dim K-means)
    4. fold_split  one of:
                   --fold-json fold_split.json   external dump
                                                  ({"0": {"train_users": [...], ...}})
                   --derive-folds                inline compute via data_prep.assign_folds
                                                  using cluster_csv + --fold-seed + --n-folds
    5. factor_name_map.json       optional {factor_id: factor_name} for readable triples

Output (one line per triple, JSONL):

    {
        "source": {"factor_id": "F2", "factor_name": "数据驱动倾向",
                   "direction": "high", "composition": []},
        "modulation": {"type": "Strong-Inhibit", "bypass": false,
                       "delta": -0.42, "p_value": 0.0119},
        "target":  {"pkvc_state": "C"},
        "context": {"scenario_type": null},
        "cluster_id": 3,
        "support":  12,
        "fold_id":  0
    }

Algorithm (brief §4D.3 方案 A, distribution shift):
    For each train fold k:
        baseline = mean PKVC state distribution across train_users
        For each factor F, direction d ∈ {high, low}:
            group = median split on F in direction d
            For each PKVC state s ∈ {P, K, V, C}:
                delta = mean(group state_dist[s]) - baseline[s]
                if |delta| < delta_threshold:     skip
                p = permutation_test(scores, state_rates, n=n_perm)
                if p ≥ p_threshold:               skip
                bypass = (group_mean_rate < bypass_threshold)
                emit triple with classified modulation bucket

Per-cluster mode scopes the loop to designers inside each designer cluster.
fold-awareness is hard (brief §4D.6): per call, only train_users contribute.

History — v3.5.1 supersedes the v3.5 co-occurrence design. The co-occurrence schema
treated (trait_polarity_atom, SADR_node) pairs as the unit of mining, which conflated
CAPS (a connecting mechanism) with the 30-dim survey (a measurement). See
framing_correction_20260522.md for the rationale.

CLI
---
::

    python -m sft_pipeline.mapping_miner \
        --factor-scores  data/user_factor_scores.csv \
        --state-dist     data/processed/state_distributions.jsonl \
        --cluster-csv    data/user_cluster_efa_v2_200.csv \
        --fold-json      data/processed/fold_split.json \
        --fold-id        0 \
        --output         data/processed/modulation_fold_0.jsonl \
        --delta-threshold 0.1 --p-threshold 0.05 --n-perm 1000 \
        --support        10 \
        --per-cluster --also-global
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("mapping_miner")


# =====================================================================
# Schema
# =====================================================================

PKVCState = Literal["P", "K", "V", "C"]
ModulationType = Literal["Strong-Inhibit", "Weak-Inhibit", "Weak-Excite", "Strong-Excite"]
Direction = Literal["high", "low"]
PKVC_STATES: tuple[PKVCState, ...] = ("P", "K", "V", "C")


@dataclass(frozen=True)
class SourcePattern:
    """A trait pattern: (factor, direction). v1 single-factor; multi-factor reserved
    for Tier 3 ablation via the composition[] field on output."""

    factor_id: str           # e.g. "F2" (EFA factor index / column name)
    factor_name: str         # e.g. "数据驱动倾向"
    direction: Direction


@dataclass
class Modulation:
    type: ModulationType
    bypass: bool             # group state activation < bypass_threshold (≈ "跳过" state)
    delta: float             # observed Δ = group_rate − baseline_rate
    p_value: float           # two-sided permutation test


@dataclass
class ModulationTriple:
    source: SourcePattern
    modulation: Modulation
    target_state: PKVCState
    scenario_type: str | None
    cluster_id: int | None    # None == global (across all train designers in fold)
    support: int              # group size after median split, ≥ min_support
    fold_id: int

    def to_dict(self) -> dict:
        return {
            "source": {
                "factor_id": self.source.factor_id,
                "factor_name": self.source.factor_name,
                "direction": self.source.direction,
                "composition": [],   # v1 single-factor; reserved for fine ablation
            },
            "modulation": {
                "type": self.modulation.type,
                "bypass": self.modulation.bypass,
                "delta": round(self.modulation.delta, 4),
                "p_value": round(self.modulation.p_value, 4),
            },
            "target": {"pkvc_state": self.target_state},
            "context": {"scenario_type": self.scenario_type},
            "cluster_id": self.cluster_id,
            "support": self.support,
            "fold_id": self.fold_id,
        }


@dataclass
class MiningConfig:
    fold_id: int
    delta_threshold: float = 0.1      # min |Δ| to even consider permutation test
    p_threshold: float = 0.05         # max p-value to emit
    n_perm: int = 1000
    min_support: int = 10             # min group size post median split
    bypass_threshold: float = 0.05    # group state mean below this → bypass flag
    per_cluster: bool = True
    also_global: bool = True
    random_seed: int = 42


# =====================================================================
# Loaders
# =====================================================================

def load_factor_scores(
    factor_csv: Path,
    factor_name_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """designer × {F1, F2, ...} factor scores.

    Returns (df indexed by user_id, list of factor column names, factor_id→name map).
    """
    df = pd.read_csv(factor_csv)
    user_col = _first_present(df, ["user_id", "designer_id", "wechat_id"])
    if not user_col:
        raise ValueError(f"factor_csv {factor_csv} missing user_id-like col; got {list(df.columns)}")
    factor_cols = [c for c in df.columns if c != user_col and df[c].dtype.kind in "fc"]
    if not factor_cols:
        raise ValueError(f"factor_csv {factor_csv} has no numeric factor columns")
    indexed = df.set_index(df[user_col].astype(str))[factor_cols]
    name_map = {
        fc: (factor_name_map.get(fc, fc) if factor_name_map else fc)
        for fc in factor_cols
    }
    logger.info("loaded factor scores: %d designers × %d factors", len(indexed), len(factor_cols))
    return indexed, factor_cols, name_map


def load_state_distributions(state_jsonl: Path) -> pd.DataFrame:
    """Per-designer PKVC state distribution.

    Each line: {"user_id": ..., "distribution": {"P": 0.3, "K": 0.2, "V": 0.4, "C": 0.1}}
    Returns a DataFrame indexed by user_id with columns P, K, V, C (row-normalized).
    """
    rows = []
    with open(state_jsonl, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj.get("user_id") or obj.get("designer_id") or "")
            if not uid:
                continue
            dist = obj.get("distribution") or {}
            rows.append({"user_id": uid, **{s: float(dist.get(s, 0.0)) for s in PKVC_STATES}})
    df = pd.DataFrame(rows).set_index("user_id")
    if df.empty:
        raise ValueError(f"state_jsonl {state_jsonl} produced no rows")
    sums = df[list(PKVC_STATES)].sum(axis=1)
    nonzero = sums > 0
    df.loc[nonzero, list(PKVC_STATES)] = df.loc[nonzero, list(PKVC_STATES)].div(sums[nonzero], axis=0)
    logger.info("loaded state distributions for %d designers", len(df))
    return df


def build_state_distributions_from_extracted(extracted_jsonl: Path) -> pd.DataFrame:
    """Build per-designer PKVC distribution from extractor_batch.py output.

    Counts PKVC factor mentions across all segments of a designer:
      * `extracted.dominant_factor`: may be a single letter ("V") or a factor path
        ("V → K → V"). Parsed via `_parse_factor_path`; every parsed factor counts +1.
      * `extracted.sadr_activations`: list of dicts {"id": "#V3", "polarity": "+", ...};
        each activation contributes +1 to its parent factor (inferred from sadr id prefix).

    Defensive about extracted-shape variations. Schema source of truth:
    `eval_extractor.ExtractionResult` + `extractor_batch._extract_one` (returns
    "extracted" dict with keys segment_role / dominant_factor / sadr_activations /
    criterion / criterion_strength / confidence / extraction_notes).
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in PKVC_STATES})
    with open(extracted_jsonl, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get("user_id") or obj.get("designer_id") or "")
            if not uid:
                continue
            extracted = obj.get("extracted") or {}
            if not isinstance(extracted, dict):
                continue
            # dominant_factor: "V" or "V → K → V" (B2 segments often a path)
            for f in _parse_factor_path(extracted.get("dominant_factor")):
                if f in PKVC_STATES:
                    counts[uid][f] += 1
            # sadr_activations: list[dict] with `id` like "#V3"; parent factor is 2nd char
            for sadr in (extracted.get("sadr_activations") or []):
                if not isinstance(sadr, dict):
                    continue
                sid = str(sadr.get("id", "")).strip()
                # Normalize: "#V3" / "V3" / "#V13" → first letter that's PKVC
                for ch in sid.lstrip("#"):
                    if ch in PKVC_STATES:
                        counts[uid][ch] += 1
                        break
    rows = []
    for uid, c in counts.items():
        total = sum(c.values()) or 1
        rows.append({"user_id": uid, **{s: c[s] / total for s in PKVC_STATES}})
    df = pd.DataFrame(rows).set_index("user_id")
    if df.empty:
        raise ValueError(f"extracted_jsonl {extracted_jsonl} produced no PKVC events")
    logger.info("built state distributions for %d designers from %s", len(df), extracted_jsonl)
    return df


def load_clusters(cluster_csv: Path) -> dict[str, int]:
    df = pd.read_csv(cluster_csv)
    user_col = _first_present(df, ["user_id", "designer_id", "wechat_id"])
    cluster_col = _first_present(df, ["cluster_id", "cluster", "label"])
    if not (user_col and cluster_col):
        raise ValueError(f"cluster_csv {cluster_csv} missing required columns; got {list(df.columns)}")
    return {str(row[user_col]): int(row[cluster_col]) for _, row in df.iterrows()}


def load_fold_split(fold_json: Path, fold_id: int) -> tuple[list[str], list[str]]:
    """Load external fold_split.json dump.

    Schema: {"0": {"train_users": [...], "test_users": [...]}, "1": {...}, ...}
    """
    with open(fold_json, encoding="utf-8") as f:
        splits = json.load(f)
    key = str(fold_id)
    if key not in splits:
        raise KeyError(f"fold_id={fold_id} not in {fold_json}; available: {list(splits.keys())}")
    s = splits[key]
    return list(map(str, s["train_users"])), list(map(str, s["test_users"]))


def derive_fold_split(
    cluster_labels: dict[str, int],
    fold_id: int,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Inline fold split using the same logic as data_prep.assign_folds.

    Avoids needing an external fold_split.json dump. Identical output to data_prep
    given the same (cluster_labels, n_folds, seed). Imports data_prep.assign_folds
    when available; falls back to a local reimplementation otherwise (e.g. when
    running mapping_miner standalone without sklearn).
    """
    try:
        from sft_pipeline.data.prep import assign_folds
    except ImportError:
        try:
            from sft_pipeline.data.prep import assign_folds  # type: ignore[no-redef]
        except ImportError:
            return _fallback_fold_split(cluster_labels, fold_id, n_folds, seed)
    user_ids = list(cluster_labels.keys())
    fold_map = assign_folds(user_ids, cluster_labels, n_folds=n_folds, seed=seed)
    train = [u for u, fid in fold_map.items() if fid != fold_id]
    test = [u for u, fid in fold_map.items() if fid == fold_id]
    return train, test


def _fallback_fold_split(
    cluster_labels: dict[str, int],
    fold_id: int,
    n_folds: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministic hash fallback (mirrors data_prep.assign_folds small-cluster branch)."""
    fold_map = {
        u: (hash((u, seed)) & 0x7FFFFFFF) % n_folds
        for u in cluster_labels
    }
    train = [u for u, fid in fold_map.items() if fid != fold_id]
    test = [u for u, fid in fold_map.items() if fid == fold_id]
    return train, test


# =====================================================================
# Mining core
# =====================================================================

def classify_modulation(delta: float) -> ModulationType:
    """4-bucket modulation, symmetric to trait 4-bucket polarity (brief §4D.2)."""
    if delta > 0.3:
        return "Strong-Excite"
    if delta > 0.1:
        return "Weak-Excite"
    if delta < -0.3:
        return "Strong-Inhibit"
    return "Weak-Inhibit"     # -0.3 ≤ delta < -0.1


def permutation_p_value(
    factor_scores: np.ndarray,
    state_rates: np.ndarray,
    direction: Direction,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Two-sided permutation test on |Δ|.

    Returns (p_value, observed_delta, group_size). Adds +1 to numerator/denominator
    so the p-value is unbiased and never exactly zero.
    """
    n = len(factor_scores)
    median = float(np.median(factor_scores))
    if direction == "high":
        mask = factor_scores > median
    else:
        mask = factor_scores <= median
    group_size = int(mask.sum())
    if group_size < 5 or group_size >= n - 5:
        return 1.0, 0.0, group_size
    baseline = state_rates.mean()
    observed_delta = float(state_rates[mask].mean() - baseline)
    observed_abs = abs(observed_delta)
    n_more_extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(factor_scores)
        if direction == "high":
            sh_mask = shuffled > median
        else:
            sh_mask = shuffled <= median
        sz = int(sh_mask.sum())
        if sz < 5 or sz >= n - 5:
            continue
        sh_delta = state_rates[sh_mask].mean() - baseline
        if abs(sh_delta) >= observed_abs:
            n_more_extreme += 1
    p_value = (n_more_extreme + 1) / (n_perm + 1)
    return p_value, observed_delta, group_size


def mine_modulation_triples(
    factor_df: pd.DataFrame,
    state_df: pd.DataFrame,
    factor_name_map: dict[str, str],
    train_users: list[str],
    clusters: dict[str, int],
    cfg: MiningConfig,
) -> list[ModulationTriple]:
    """Top-level miner. Per-cluster + global per fold, both gated by cfg flags."""
    present = [u for u in train_users if u in factor_df.index and u in state_df.index]
    n_missing = len(train_users) - len(present)
    if n_missing:
        logger.warning("dropping %d train_users lacking factor/state data", n_missing)
    if not present:
        logger.error("no designers with both factor and state data — aborting")
        return []

    rng = np.random.default_rng(cfg.random_seed)
    triples: list[ModulationTriple] = []

    if cfg.per_cluster:
        by_cluster: dict[int, list[str]] = defaultdict(list)
        for u in present:
            c = clusters.get(u)
            if c is None:
                logger.warning("designer %s missing cluster_id, skipping per-cluster", u)
                continue
            by_cluster[c].append(u)
        for c_id, members in sorted(by_cluster.items()):
            ts = _mine_one_scope(factor_df, state_df, factor_name_map,
                                  members, cluster_id=c_id, cfg=cfg, rng=rng)
            triples.extend(ts)
            logger.info("cluster=%s n=%d → %d triples", c_id, len(members), len(ts))

    if cfg.also_global:
        ts = _mine_one_scope(factor_df, state_df, factor_name_map,
                              present, cluster_id=None, cfg=cfg, rng=rng)
        triples.extend(ts)
        logger.info("global n=%d → %d triples", len(present), len(ts))

    logger.info("fold=%d total %d triples (|Δ|≥%.2f, p<%.2f, support≥%d)",
                cfg.fold_id, len(triples), cfg.delta_threshold,
                cfg.p_threshold, cfg.min_support)
    return triples


def _mine_one_scope(
    factor_df: pd.DataFrame,
    state_df: pd.DataFrame,
    factor_name_map: dict[str, str],
    designer_ids: list[str],
    cluster_id: int | None,
    cfg: MiningConfig,
    rng: np.random.Generator,
) -> list[ModulationTriple]:
    if len(designer_ids) < max(cfg.min_support * 2, 10):
        logger.warning("scope=cluster=%s too small (%d), skip", cluster_id, len(designer_ids))
        return []
    sub_f = factor_df.loc[designer_ids]
    sub_s = state_df.loc[designer_ids]
    out: list[ModulationTriple] = []
    for factor_id in sub_f.columns:
        scores = sub_f[factor_id].to_numpy(dtype=float)
        if scores.std() < 1e-9:
            continue
        for direction in ("high", "low"):
            median = float(np.median(scores))
            mask = scores > median if direction == "high" else scores <= median
            group_size = int(mask.sum())
            if group_size < cfg.min_support:
                continue
            for state in PKVC_STATES:
                rates = sub_s[state].to_numpy(dtype=float)
                p, observed_delta, _ = permutation_p_value(
                    scores, rates, direction, cfg.n_perm, rng,
                )
                if abs(observed_delta) < cfg.delta_threshold:
                    continue
                if p >= cfg.p_threshold:
                    continue
                group_mean = float(rates[mask].mean())
                bypass = bool(group_mean < cfg.bypass_threshold)
                mod = Modulation(
                    type=classify_modulation(observed_delta),
                    bypass=bypass,
                    delta=observed_delta,
                    p_value=p,
                )
                src = SourcePattern(
                    factor_id=str(factor_id),
                    factor_name=factor_name_map.get(str(factor_id), str(factor_id)),
                    direction=direction,
                )
                out.append(ModulationTriple(
                    source=src, modulation=mod, target_state=state,
                    scenario_type=None,
                    cluster_id=cluster_id, support=group_size, fold_id=cfg.fold_id,
                ))
    # Sort: largest |Δ| first, then most significant p-value (asc)
    out.sort(key=lambda t: (-abs(t.modulation.delta), t.modulation.p_value))
    return out


def write_triples(triples: list[ModulationTriple], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


# =====================================================================
# Helpers
# =====================================================================

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_factor_path(value) -> list[str]:
    """'V → K → V' → ['V', 'K', 'V']. Tolerates list input or None."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    if not isinstance(value, str):
        return []
    for sep in ["→", "->", "|", ",", ";"]:
        if sep in value:
            return [x.strip() for x in value.split(sep) if x.strip()]
    return [value.strip()] if value.strip() else []


# =====================================================================
# CLI
# =====================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mine trait→PKVC modulation triples (brief v3.5.1 §4D, distribution shift, fold-aware).",
    )
    p.add_argument("--factor-scores", type=Path, required=True,
                   help="user_factor_scores.csv: designer × EFA factor scores")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-dist", type=Path,
                     help="prebuilt PKVC state distribution JSONL")
    src.add_argument("--extracted", type=Path,
                     help="extractor_batch.py JSONL (state dist computed inline)")
    p.add_argument("--cluster-csv", type=Path, required=True)
    fold = p.add_mutually_exclusive_group(required=True)
    fold.add_argument("--fold-json", type=Path,
                      help='external fold dump: {"0": {"train_users": [...], "test_users": [...]}, ...}')
    fold.add_argument("--derive-folds", action="store_true",
                      help="derive fold split inline via data_prep.assign_folds")
    p.add_argument("--n-folds", type=int, default=5,
                   help="only used with --derive-folds (default 5)")
    p.add_argument("--fold-seed", type=int, default=42,
                   help="only used with --derive-folds (must match data_prep seed)")
    p.add_argument("--fold-id", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factor-name-map", type=Path, default=None,
                   help="optional JSON dict {factor_id: 中文名} for readable triples")
    p.add_argument("--delta-threshold", type=float, default=0.1)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--support", type=int, default=10,
                   help="min post-median-split group size to keep a triple")
    p.add_argument("--bypass-threshold", type=float, default=0.05,
                   help="if group_mean[state] < this, flag bypass=true")
    p.add_argument("--per-cluster", action="store_true", default=True)
    p.add_argument("--no-per-cluster", dest="per_cluster", action="store_false")
    p.add_argument("--also-global", action="store_true", default=True)
    p.add_argument("--no-also-global", dest="also_global", action="store_false")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    name_map: dict[str, str] = {}
    if args.factor_name_map and args.factor_name_map.exists():
        name_map = json.loads(args.factor_name_map.read_text(encoding="utf-8"))

    factor_df, _, full_name_map = load_factor_scores(args.factor_scores, name_map)
    if args.state_dist:
        state_df = load_state_distributions(args.state_dist)
    else:
        state_df = build_state_distributions_from_extracted(args.extracted)
    clusters = load_clusters(args.cluster_csv)
    if args.derive_folds:
        train_users, _ = derive_fold_split(
            clusters, args.fold_id, n_folds=args.n_folds, seed=args.fold_seed,
        )
        logger.info("derived fold split inline (n_folds=%d, seed=%d): train=%d",
                    args.n_folds, args.fold_seed, len(train_users))
    else:
        train_users, _ = load_fold_split(args.fold_json, args.fold_id)

    cfg = MiningConfig(
        fold_id=args.fold_id,
        delta_threshold=args.delta_threshold,
        p_threshold=args.p_threshold,
        n_perm=args.n_perm,
        min_support=args.support,
        bypass_threshold=args.bypass_threshold,
        per_cluster=args.per_cluster,
        also_global=args.also_global,
        random_seed=args.random_seed,
    )
    triples = mine_modulation_triples(
        factor_df, state_df, full_name_map, train_users, clusters, cfg,
    )
    write_triples(triples, args.output)
    logger.info("wrote %d modulation triples → %s", len(triples), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
