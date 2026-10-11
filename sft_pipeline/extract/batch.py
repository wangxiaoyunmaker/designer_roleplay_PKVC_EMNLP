"""Batch driver for the PKVC + SADR extractor.

Runs ``eval_extractor.extract_pkvc`` asynchronously on a large input JSONL
(typical scale: 600 Q7 segments + 79*6 Q1-Q6 explanation NLs), with:

* Concurrency control (semaphore, default 8 in-flight)
* Disk cache (re-use of previously extracted results)
* Per-line skip-on-failure (writes ``{"error": ...}`` so downstream can filter)
* Optional cross-family κ pilot (GPT-4o vs Claude) on a random sub-sample
* Progress logging every N lines

Input JSONL schema (one object per line)
----------------------------------------
::

    {
        "id":          "<unique id, used for dedup>",
        "user_id":     "<designer id>",
        "scenario_id": "Q7_seg1" | "Q1" | ...,
        "text":        "<NL to extract from>",
        "meta":        { ... optional passthrough ... }
    }

Output JSONL schema
-------------------
::

    {
        "id":          ...,
        "user_id":     ...,
        "scenario_id": ...,
        "text":        ...,
        "extracted":   { ...JSON from extractor... },
        "cache_hit":   true|false,
        "model":       "gpt-4o",
        "error":       null|"<exception str>"
    }

CLI
---
::

    python extractor_batch.py --input data/q7_nl.jsonl --output data/q7_extracted.jsonl
    python extractor_batch.py --input data/q7_nl.jsonl --kappa-pilot 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("extractor_batch")


@dataclass
class BatchStats:
    total: int = 0
    cache_hits: int = 0
    new_extractions: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        rate = self.total / max(self.elapsed_sec, 1e-9)
        return (
            f"total={self.total} cache_hit={self.cache_hits} "
            f"new={self.new_extractions} err={self.errors} "
            f"elapsed={self.elapsed_sec:.1f}s ({rate:.2f}/s)"
        )


def _import_extractor():
    """Late import so this module is testable without the API SDKs installed."""
    from sft_pipeline.extract.llm import extract_pkvc, ExtractionResult
    return extract_pkvc, ExtractionResult


async def _extract_one(
    line_obj: dict[str, Any],
    sem: asyncio.Semaphore,
    model: str,
    use_cache: bool,
    api_key: str | None,
    extract_fn,
) -> dict[str, Any]:
    async with sem:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: extract_fn(line_obj["text"], model=model,
                                   use_cache=use_cache, api_key=api_key),
            )
            return {
                **{k: v for k, v in line_obj.items() if k != "text"},
                "text": line_obj["text"],
                "extracted": {
                    "segment_role": result.segment_role,
                    "dominant_factor": result.dominant_factor,
                    "sadr_activations": result.sadr_activations,
                    "criterion": result.criterion,
                    "criterion_strength": result.criterion_strength,
                    "confidence": result.confidence,
                    "extraction_notes": result.extraction_notes,
                },
                "cache_hit": result.cache_hit,
                "model": model,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - we want to keep batch alive
            logger.warning("Extract failed on id=%s: %s", line_obj.get("id"), exc)
            return {
                **{k: v for k, v in line_obj.items() if k != "text"},
                "text": line_obj["text"],
                "extracted": None,
                "cache_hit": False,
                "model": model,
                "error": str(exc),
            }


async def run_batch_async(
    input_records: list[dict[str, Any]],
    model: str,
    concurrency: int,
    use_cache: bool,
    api_key: str | None,
    log_every: int = 25,
) -> tuple[list[dict[str, Any]], BatchStats]:
    extract_fn, _ = _import_extractor()
    sem = asyncio.Semaphore(concurrency)
    stats = BatchStats(total=len(input_records))
    t0 = time.time()

    async def _wrap(idx: int, obj: dict[str, Any]) -> dict[str, Any]:
        out = await _extract_one(obj, sem, model, use_cache, api_key, extract_fn)
        if out["error"]:
            stats.errors += 1
        elif out["cache_hit"]:
            stats.cache_hits += 1
        else:
            stats.new_extractions += 1
        if (idx + 1) % log_every == 0 or idx + 1 == stats.total:
            logger.info("Progress %d/%d", idx + 1, stats.total)
        return out

    tasks = [asyncio.create_task(_wrap(i, obj)) for i, obj in enumerate(input_records)]
    results = await asyncio.gather(*tasks)
    stats.elapsed_sec = time.time() - t0
    return results, stats


# ---------------------------------------------------------------------------
# κ pilot
# ---------------------------------------------------------------------------

async def kappa_pilot_async(
    sample_records: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    concurrency: int,
    use_cache: bool,
    api_key_a: str | None,
    api_key_b: str | None,
) -> dict[str, Any]:
    """Run both extractors over the same sample, return inter-rater agreement."""
    out_a, stats_a = await run_batch_async(
        sample_records, model_a, concurrency, use_cache, api_key_a,
    )
    out_b, stats_b = await run_batch_async(
        sample_records, model_b, concurrency, use_cache, api_key_b,
    )

    factor_match = 0
    sadr_jacc_sum = 0.0
    counted = 0
    for ra, rb in zip(out_a, out_b):
        ea, eb = ra.get("extracted"), rb.get("extracted")
        if not ea or not eb:
            continue
        counted += 1
        if ea.get("dominant_factor") == eb.get("dominant_factor"):
            factor_match += 1
        a_ids = {s.get("id") for s in (ea.get("sadr_activations") or []) if isinstance(s, dict)}
        b_ids = {s.get("id") for s in (eb.get("sadr_activations") or []) if isinstance(s, dict)}
        union = a_ids | b_ids
        sadr_jacc_sum += (len(a_ids & b_ids) / len(union)) if union else 1.0

    return {
        "n_sampled": len(sample_records),
        "n_compared": counted,
        "model_a": model_a, "model_b": model_b,
        "factor_agreement_rate": factor_match / max(counted, 1),
        "sadr_jaccard_mean": sadr_jacc_sum / max(counted, 1),
        "passes_kappa_07": (
            factor_match / max(counted, 1) >= 0.7
            and sadr_jacc_sum / max(counted, 1) >= 0.5
        ),
        "model_a_stats": stats_a.summary(),
        "model_b_stats": stats_b.summary(),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip malformed line %d: %s", ln, e)
                continue
            if "text" not in obj:
                logger.warning("Skip line %d: missing 'text' field", ln)
                continue
            records.append(obj)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL with {id, text, ...}.")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL with extracted results. Default: input.with_suffix('.extracted.jsonl').")
    parser.add_argument("--model", default="gpt-4o",
                        help="Extractor model name (gpt-4o, claude-3-5-sonnet-..., deepseek-chat).")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-key", default=None,
                        help="Override env var (OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY).")
    parser.add_argument("--kappa-pilot", type=int, default=0,
                        help="If >0, sample this many records, run model_a vs model_b agreement.")
    parser.add_argument("--kappa-model-a", default="gpt-4o")
    parser.add_argument("--kappa-model-b", default="claude-3-5-sonnet-20241022")
    parser.add_argument("--kappa-api-key-a", default=None,
                        help="API key for κ pilot model A (overrides env var).")
    parser.add_argument("--kappa-api-key-b", default=None,
                        help="API key for κ pilot model B (overrides env var).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = read_jsonl(args.input)
    if not records:
        logger.error("Empty input: %s", args.input)
        return 1
    logger.info("Loaded %d records from %s", len(records), args.input)

    if args.kappa_pilot:
        rng = random.Random(args.seed)
        sample = rng.sample(records, min(args.kappa_pilot, len(records)))
        result = asyncio.run(kappa_pilot_async(
            sample, args.kappa_model_a, args.kappa_model_b,
            args.concurrency, not args.no_cache,
            args.kappa_api_key_a, args.kappa_api_key_b,
        ))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passes_kappa_07"] else 1

    out_path = args.output or args.input.with_suffix(".extracted.jsonl")
    results, stats = asyncio.run(run_batch_async(
        records, args.model, args.concurrency, not args.no_cache, args.api_key,
    ))
    write_jsonl(results, out_path)
    logger.info("Done: %s", stats.summary())
    logger.info("Wrote → %s", out_path)
    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
