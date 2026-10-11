"""异步批跑驱动 + 进度日志 + per-candidate skip-on-failure.

输入: q1q6_expansion_candidates.jsonl (screen 输出)
输出: q1q6_expansion_outputs.jsonl, 每行含 {candidate, result, validation, model}

调用: ``python -m sft_pipeline.expand.batch --input ... --output ... --model gpt-4o``
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sft_pipeline.expand.llm import call_chat, mock_call_chat_factory
from sft_pipeline.expand.parser import parse_and_validate, result_to_jsonl_dict
from sft_pipeline.expand.prompt import render
from sft_pipeline.expand.screen import ExpansionCandidate, load_candidates

logger = logging.getLogger("expand.batch")


@dataclass
class BatchStats:
    total: int = 0
    cache_hits: int = 0
    new_calls: int = 0
    validation_ok: int = 0
    validation_fail: int = 0
    api_errors: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        rate = self.total / max(self.elapsed_sec, 1e-9)
        return (
            f"total={self.total} cache={self.cache_hits} new={self.new_calls} "
            f"ok={self.validation_ok} fail={self.validation_fail} "
            f"api_err={self.api_errors} elapsed={self.elapsed_sec:.1f}s "
            f"({rate:.2f}/s)"
        )


async def _expand_one(
    candidate: ExpansionCandidate,
    sem: asyncio.Semaphore,
    model: str,
    api_key: str | None,
    base_url: str | None,
    use_cache: bool,
    temperature: float,
    call_fn: Callable[..., tuple[str, bool]],
) -> dict[str, Any]:
    async with sem:
        system, user = render(candidate)
        loop = asyncio.get_event_loop()
        try:
            raw, cache_hit = await loop.run_in_executor(
                None,
                lambda: call_fn(system, user, model=model, api_key=api_key,
                                base_url=base_url, use_cache=use_cache,
                                temperature=temperature),
            )
        except Exception as exc:  # noqa: BLE001 keep batch alive
            logger.warning("API call failed on %s: %s", candidate.candidate_id, exc)
            return {
                "candidate_id": candidate.candidate_id,
                "user_id": candidate.user_id,
                "question_id": candidate.question_id,
                "seed_nl": candidate.seed_nl,
                "seed_char_count": candidate.seed_char_count,
                "result": None,
                "validation": {"ok": False, "fails": [f"api_error: {exc}"]},
                "model": model,
                "api_error": True,
            }
        result, report = parse_and_validate(raw, candidate, cache_hit=cache_hit)
        rec = result_to_jsonl_dict(candidate, result, report, model)
        rec["api_error"] = False
        return rec


async def run_batch_async(
    candidates: list[ExpansionCandidate],
    model: str = "gpt-4o",
    concurrency: int = 8,
    api_key: str | None = None,
    base_url: str | None = None,
    use_cache: bool = True,
    temperature: float = 0.3,
    log_every: int = 10,
    mock: bool = False,
) -> tuple[list[dict[str, Any]], BatchStats]:
    sem = asyncio.Semaphore(concurrency)
    stats = BatchStats(total=len(candidates))
    t0 = time.time()
    call_fn = mock_call_chat_factory() if mock else call_chat

    async def _wrap(idx: int, c: ExpansionCandidate) -> dict[str, Any]:
        out = await _expand_one(
            c, sem, model, api_key, base_url, use_cache, temperature, call_fn,
        )
        if out.get("api_error"):
            stats.api_errors += 1
        else:
            if out["result"] and out["result"].get("cache_hit"):
                stats.cache_hits += 1
            else:
                stats.new_calls += 1
            if out["validation"]["ok"]:
                stats.validation_ok += 1
            else:
                stats.validation_fail += 1
        if (idx + 1) % log_every == 0 or idx + 1 == stats.total:
            logger.info(
                "Progress %d/%d (ok=%d fail=%d err=%d cache=%d)",
                idx + 1, stats.total, stats.validation_ok,
                stats.validation_fail, stats.api_errors, stats.cache_hits,
            )
        return out

    tasks = [asyncio.create_task(_wrap(i, c)) for i, c in enumerate(candidates)]
    results = await asyncio.gather(*tasks)
    stats.elapsed_sec = time.time() - t0
    return results, stats


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/q1q6_expansion_candidates.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/q1q6_expansion_outputs.jsonl"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-key", default=None,
                        help="Override OPENAI_API_KEY env var.")
    parser.add_argument("--base-url", default=None,
                        help="Override OPENAI_BASE_URL (e.g. yunwu.ai 中转).")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no network). For pipeline smoke testing.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N candidates (0 = all).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cands = load_candidates(args.input)
    if args.limit:
        cands = cands[: args.limit]
    if not cands:
        logger.error("No candidates loaded from %s", args.input)
        return 1
    logger.info("Loaded %d candidates", len(cands))

    results, stats = asyncio.run(run_batch_async(
        cands, args.model, args.concurrency, args.api_key, args.base_url,
        not args.no_cache, args.temperature, mock=args.mock,
    ))
    write_jsonl(results, args.output)
    logger.info("Done: %s", stats.summary())
    logger.info("Wrote → %s", args.output)
    return 0 if stats.api_errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
