"""Q1-Q6 扩写 pipeline orchestration entry point.

子命令:
  screen   - 筛选候选 + 装配 ground 材料, 输出 candidates.jsonl
  run      - 批跑 LLM, 输出 outputs.jsonl + cache
  merge    - 回填到 task_responses_expanded.csv + provenance.jsonl
  report   - 统计扩写质量 (validation 通过率 / factor 分布 / κ 数据准备)
  full     - 依次跑 screen → run → merge → report

典型用法:
  # offline smoke (no API)
  python -m sft_pipeline.expand.cli full --mock --limit 5

  # production (yunwu.ai 中转 OPENAI 兼容)
  export OPENAI_API_KEY=sk-...
  export OPENAI_BASE_URL=https://yunwu.ai/v1
  python -m sft_pipeline.expand.cli full --threshold 25 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sft_pipeline.expand.batch import run_batch_async, write_jsonl
from sft_pipeline.expand.merge import merge_into_task_responses
from sft_pipeline.expand.screen import (
    load_candidates, screen_candidates, write_candidates,
)

logger = logging.getLogger("expand.cli")

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROCESSED = Path("data/processed")
CANDIDATES_PATH = DEFAULT_PROCESSED / "q1q6_expansion_candidates.jsonl"
OUTPUTS_PATH = DEFAULT_PROCESSED / "q1q6_expansion_outputs.jsonl"
PROVENANCE_PATH = DEFAULT_PROCESSED / "q1q6_expansion_provenance.jsonl"
EXPANDED_CSV = Path("data/task_responses_expanded.csv")
REPORT_PATH = DEFAULT_PROCESSED / "q1q6_expansion_report.md"


def cmd_screen(args: argparse.Namespace) -> int:
    cands = screen_candidates(args.data_dir, args.threshold, args.cluster_filename)
    n = write_candidates(cands, args.candidates)
    logger.info("Screened %d → %s", n, args.candidates)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cands = load_candidates(args.candidates)
    if args.limit:
        cands = cands[: args.limit]
    if not cands:
        logger.error("No candidates loaded from %s", args.candidates)
        return 1
    logger.info("Loaded %d candidates", len(cands))
    results, stats = asyncio.run(run_batch_async(
        cands, model=args.model, concurrency=args.concurrency,
        api_key=args.api_key, base_url=args.base_url,
        use_cache=not args.no_cache, temperature=args.temperature,
        mock=args.mock,
    ))
    write_jsonl(results, args.outputs)
    logger.info("Batch done: %s", stats.summary())
    logger.info("Wrote → %s", args.outputs)
    return 0 if stats.api_errors == 0 else 2


def cmd_merge(args: argparse.Namespace) -> int:
    merge_into_task_responses(
        args.task_csv, args.outputs, args.expanded_csv, args.provenance,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if not args.outputs.exists():
        logger.error("Outputs file missing: %s. Run `run` first.", args.outputs)
        return 1
    rows: list[dict[str, Any]] = []
    with args.outputs.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    ok = sum(1 for r in rows if r["validation"]["ok"])
    api_err = sum(1 for r in rows if r.get("api_error"))
    by_q = Counter(r["question_id"] for r in rows)
    by_q_ok = Counter(r["question_id"] for r in rows if r["validation"]["ok"])

    fail_reasons: Counter[str] = Counter()
    for r in rows:
        for f in r["validation"]["fails"]:
            fail_reasons[f.split(":")[0]] += 1

    by_factor = Counter(
        (r["result"] or {}).get("inferred_factor") or "?"
        for r in rows if r["validation"]["ok"]
    )
    by_confidence = Counter(
        (r["result"] or {}).get("confidence") or "?"
        for r in rows if r.get("result")
    )

    # Length distribution of expanded NLs
    expanded_lens = [
        len((r["result"] or {}).get("expanded_nl", ""))
        for r in rows if r["validation"]["ok"]
    ]
    if expanded_lens:
        med = sorted(expanded_lens)[len(expanded_lens) // 2]
        mn, mx = min(expanded_lens), max(expanded_lens)
    else:
        med = mn = mx = 0

    lines = [
        "# Q1-Q6 扩写 pipeline report",
        "",
        f"- total candidates: **{total}**",
        f"- validation ok: **{ok}** ({100 * ok / max(total, 1):.1f}%)",
        f"- api errors: {api_err}",
        f"- expanded length (chars): min={mn} median={med} max={mx}",
        "",
        "## By question",
        "",
        "| qid | total | ok | ok% |",
        "|---|---|---|---|",
    ]
    for q in sorted(by_q):
        t = by_q[q]
        o = by_q_ok[q]
        lines.append(f"| {q} | {t} | {o} | {100 * o / max(t, 1):.0f}% |")

    lines.extend([
        "",
        "## By inferred PKVC factor (ok rows only)",
        "",
        "| factor | count |",
        "|---|---|",
    ])
    for fac in sorted(by_factor):
        lines.append(f"| {fac} | {by_factor[fac]} |")

    lines.extend([
        "",
        "## By confidence",
        "",
        "| level | count |",
        "|---|---|",
    ])
    for c in sorted(by_confidence):
        lines.append(f"| {c} | {by_confidence[c]} |")

    lines.extend([
        "",
        "## Validation failure reasons",
        "",
        "| reason | count |",
        "|---|---|",
    ])
    for rsn, n in fail_reasons.most_common():
        lines.append(f"| {rsn} | {n} |")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report → %s", args.report)
    print("\n".join(lines))
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    rc = cmd_screen(args)
    if rc != 0:
        return rc
    rc = cmd_run(args)
    if rc != 0:
        return rc
    rc = cmd_merge(args)
    if rc != 0:
        return rc
    rc = cmd_report(args)
    return rc


def _common_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    p.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    p.add_argument("--task-csv", type=Path, default=Path("data/task_responses.csv"))
    p.add_argument("--expanded-csv", type=Path, default=EXPANDED_CSV)
    p.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    p.add_argument("--cluster-filename", default="user_cluster_efa.csv")
    p.add_argument("--threshold", type=int, default=25)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subs = root.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("screen", cmd_screen), ("run", cmd_run), ("merge", cmd_merge),
        ("report", cmd_report), ("full", cmd_full),
    ):
        sp = subs.add_parser(name, help=fn.__doc__ or "")
        _common_paths(sp)
        sp.set_defaults(func=fn)

    args = root.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
