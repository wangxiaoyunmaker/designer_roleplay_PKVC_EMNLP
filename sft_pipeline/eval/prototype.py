"""Build per-cluster PKVC prototype from extracted records.

A *prototype* for cluster ``c`` is the aggregated PKVC trait signature of all
**training-fold** designers in that cluster on a particular scenario. It is
the comparison target for the main metric A1 (trajectory match).

Schema of a prototype
---------------------
::

    {
        "cluster_id": int,
        "scenario_id": "Q7_seg1" | "Q1" | ...,
        "n_designers": int,
        "dominant_factor": "V"   # mode across designers (single-factor segs)
                       or "V → K → V" # for multi-step segs like Q7_seg2
                       or None,
        "factor_distribution": { "V": 0.6, "K": 0.3, ... },
        "sadr_frequency": { "#V3": 0.55, "#服务保留-": 0.40, ... },
        "criterion_keywords": [ "人工服务", ... ]  # only B3 / BOTTOM_LINE
    }

Cluster prototypes can be loaded by ``eval_metrics.compute_a1`` to score
predictions: for each prediction we compare its (dominant_factor, sadr_set)
against the prototype using Hamming + Jaccard.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cluster_prototype")

PKVC_FACTORS = ("P", "K", "V", "C")
_FACTOR_PATH_SEP_RE = re.compile(r"\s*(?:→|->|>|\|)\s*")


@dataclass
class Prototype:
    """Per (cluster, scenario) PKVC prototype."""
    cluster_id: int
    scenario_id: str
    n_designers: int
    dominant_factor: str | None
    factor_distribution: dict[str, float]
    sadr_frequency: dict[str, float]
    criterion_keywords: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_factor_field(value: Any) -> list[str]:
    """Normalize a ``dominant_factor`` field into a factor sequence.

    Accepts ``"V"``, ``"V → K → V"``, ``"P->K"`` and returns ``["V"]`` /
    ``["V", "K", "V"]`` / ``["P", "K"]`` respectively. Garbage in → ``[]``.
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = _FACTOR_PATH_SEP_RE.split(s) if any(sep in s for sep in ("→", "->", ">", "|")) else [s]
    return [p.strip() for p in parts if p.strip() in PKVC_FACTORS]


def parse_sadr_field(value: Any) -> list[str]:
    """Normalize ``sadr_activations`` into a flat list of ``#ID±`` tokens."""
    if value is None:
        return []
    out: list[str] = []
    if isinstance(value, list):
        for s in value:
            if isinstance(s, str):
                out.append(s)
            elif isinstance(s, dict):
                node_id = s.get("id", "")
                polarity = s.get("polarity", "")
                if node_id:
                    out.append(f"{node_id}{polarity}")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_factor_distribution(
    factor_sequences: list[list[str]],
) -> tuple[str | None, dict[str, float]]:
    """Aggregate factor sequences across same-cluster designers.

    - If all sequences are length 1 (single-factor segments) we report the
      mode factor as ``dominant_factor`` and the relative-frequency dict.
    - If sequences are length > 1 (multi-step paths), the most common full
      path string is reported as ``dominant_factor`` (e.g. ``"V → K → V"``).

    Returns ``(dominant_factor, distribution)``.
    """
    if not factor_sequences:
        return None, {}

    flat_lengths = {len(seq) for seq in factor_sequences if seq}
    # Multi-step: mode of joined path
    if flat_lengths - {1}:
        path_strs = [" → ".join(seq) for seq in factor_sequences if seq]
        if not path_strs:
            return None, {}
        c = Counter(path_strs)
        total = sum(c.values())
        return (
            c.most_common(1)[0][0],
            {k: v / total for k, v in c.items()},
        )

    # Single-factor: mode of first element
    tokens = [seq[0] for seq in factor_sequences if seq]
    if not tokens:
        return None, {}
    c = Counter(tokens)
    total = sum(c.values())
    return (
        c.most_common(1)[0][0],
        {k: v / total for k, v in c.items()},
    )


def aggregate_sadr_frequency(
    sadr_lists: list[list[str]],
    min_frequency: float = 0.3,
) -> dict[str, float]:
    """Aggregate SADR tokens; keep nodes present in ≥ ``min_frequency`` of the
    designer-segments to filter out idiosyncratic activations."""
    n = len(sadr_lists)
    if n == 0:
        return {}
    counter: Counter[str] = Counter()
    for lst in sadr_lists:
        # deduplicate within one designer-segment so we count "designer-presence"
        counter.update(set(lst))
    return {
        token: count / n
        for token, count in counter.items()
        if count / n >= min_frequency
    }


_CRITERION_TOKEN_RE = re.compile(r"[一-鿿]{2,}")


def aggregate_criterion_keywords(criteria: list[str], top_k: int = 5) -> list[str]:
    """Extract the most common 2+ character CJK tokens from criterion phrases.

    A very rough proxy for "what the cluster considers their non-negotiable
    bottom line"; replace with a proper tokenizer later if needed.
    """
    counter: Counter[str] = Counter()
    for c in criteria:
        if not c:
            continue
        for m in _CRITERION_TOKEN_RE.findall(c):
            counter[m] += 1
    return [tok for tok, _ in counter.most_common(top_k)]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_prototypes(
    extracted_records: list[dict[str, Any]],
    user_cluster: dict[str, int],
    train_users: set[str] | None = None,
    min_sadr_freq: float = 0.3,
) -> list[Prototype]:
    """Compute (cluster_id, scenario_id) → Prototype.

    Args:
        extracted_records: each is the output JSON of
            :func:`eval_extractor.extract_pkvc` plus user_id + scenario_id.
        user_cluster: user_id → cluster_id.
        train_users: if given, restrict aggregation to this subset (use for
            fold-aware prototyping; pass the train-fold users for a 5-fold CV).
    """
    # bucket by (cluster_id, scenario_id)
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for rec in extracted_records:
        uid = rec.get("user_id")
        if uid is None or uid not in user_cluster:
            continue
        if train_users is not None and uid not in train_users:
            continue
        cid = int(user_cluster[uid])
        sid = rec.get("scenario_id") or rec.get("extracted", {}).get("segment_role") or "?"
        buckets.setdefault((cid, sid), []).append(rec)

    prototypes: list[Prototype] = []
    for (cid, sid), recs in buckets.items():
        # Each rec contributes one factor sequence + sadr list
        factor_seqs = [parse_factor_field((r.get("extracted") or r).get("dominant_factor"))
                       for r in recs]
        sadr_lists = [parse_sadr_field((r.get("extracted") or r).get("sadr_activations", []))
                       for r in recs]
        criteria = [(r.get("extracted") or r).get("criterion") or "" for r in recs]

        dominant, distribution = aggregate_factor_distribution(factor_seqs)
        sadr_freq = aggregate_sadr_frequency(sadr_lists, min_frequency=min_sadr_freq)
        criterion_kw = aggregate_criterion_keywords(criteria) if sid.endswith("seg3") or sid == "B3" else []

        prototypes.append(Prototype(
            cluster_id=cid,
            scenario_id=sid,
            n_designers=len({r.get("user_id") for r in recs}),
            dominant_factor=dominant,
            factor_distribution=distribution,
            sadr_frequency=sadr_freq,
            criterion_keywords=criterion_kw,
        ))
    return prototypes


def index_prototypes(prototypes: list[Prototype]) -> dict[tuple[int, str], Prototype]:
    return {(p.cluster_id, p.scenario_id): p for p in prototypes}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_prototypes(prototypes: list[Prototype], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prototypes:
            f.write(json.dumps(p.asdict(), ensure_ascii=False) + "\n")


def read_prototypes(path: Path) -> list[Prototype]:
    out: list[Prototype] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Prototype(**obj))
    return out


def read_extracted_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--extracted", type=Path, required=True,
                   help="JSONL produced by extractor_batch.py.")
    p.add_argument("--cluster-csv", type=Path,
                   default=Path("/data/wy/Designer_RolePlay/data/user_cluster_efa.csv"))
    p.add_argument("--train-users", type=Path, default=None,
                   help="Optional file with one user_id per line restricting aggregation.")
    p.add_argument("--min-sadr-freq", type=float, default=0.3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import pandas as pd
    cdf = pd.read_csv(args.cluster_csv)
    cdf.columns = [c.strip().lstrip("﻿") for c in cdf.columns]
    user_cluster = dict(zip(cdf["user_id"].astype(str), cdf["cluster_id"].astype(int)))

    train_users: set[str] | None = None
    if args.train_users and args.train_users.exists():
        train_users = {l.strip() for l in args.train_users.read_text().splitlines() if l.strip()}

    records = read_extracted_records(args.extracted)
    logger.info("Loaded %d extracted records", len(records))

    prototypes = build_prototypes(records, user_cluster, train_users, args.min_sadr_freq)
    write_prototypes(prototypes, args.output)
    logger.info("Wrote %d prototypes → %s", len(prototypes), args.output)

    by_cluster = Counter(p.cluster_id for p in prototypes)
    for c in sorted(by_cluster):
        logger.info("  cluster %d: %d scenarios", c, by_cluster[c])
    return 0


if __name__ == "__main__":
    sys.exit(main())
