"""Pairwise LLM Judge: GPT-4o 评测 model vs baseline win rate.

Position-debiased: 每对样本 swap A/B 跑两轮, 消除 GPT-4o A-position bias.
4 维评判: persona_fidelity, cognitive_authenticity, scenario_grounding, linguistic_naturalness.
支持 4 档 RAG 模式 predictions 直接喂入.

CLI
---
::

    # 跑 judge (需要 OPENAI_API_KEY, yunwu.ai 中转)
    python -m sft_pipeline.eval.judge \\
        --model-preds preds/q1q6__og_rag+modulation__fold0.jsonl \\
        --baseline-preds preds/q1q6__no_rag__fold0.jsonl \\
        --data-dir data \\
        --output reports/judge_og_rag_vs_no_rag_fold0.json

    # 指定 judge model / 并发
    python -m sft_pipeline.eval.judge \\
        --model-preds preds/q1q6__og_rag__fold0.jsonl \\
        --baseline-preds preds/q1q6__no_rag__fold0.jsonl \\
        --data-dir data \\
        --judge-model gpt-4o \\
        --concurrency 4 \\
        --output reports/judge.json

    # dry run (不调 API, 随机 verdict, 验 pipeline)
    python -m sft_pipeline.eval.judge \\
        --model-preds preds/q1q6__og_rag__fold0.jsonl \\
        --baseline-preds preds/q1q6__no_rag__fold0.jsonl \\
        --data-dir data \\
        --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval.judge")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluator for designer cognitive role-play. You will compare \
two responses generated for the same designer under the same scenario, and \
judge which response better captures that designer's authentic cognitive style.

You must evaluate along FOUR dimensions, then give a final verdict.

### Dimensions

1. **Persona Fidelity** — Does the response reflect this specific designer's \
trait profile (belief tendencies, reasoning habits)? Penalize generic \
designer-speak that could belong to anyone.

2. **Cognitive State Authenticity** — Does the response exhibit plausible \
cognitive states (perception, knowledge activation, value weighing, choice \
commitment) rather than surface-level opinion statements? Look for evidence \
of internal reasoning process, not just conclusions.

3. **Scenario Grounding** — Is the response grounded in the specific design \
scenario presented? Penalize responses that ignore scenario constraints or \
give advice applicable to any scenario.

4. **Linguistic Naturalness** — Does the response read like a real designer \
thinking aloud, with appropriate hedging, domain jargon, and personal style? \
Penalize robotic, overly structured, or excessively verbose outputs.

### Scoring Rules

- Judge ONLY based on the four dimensions above.
- Do NOT favor longer responses. Concise and precise beats verbose and vague.
- Do NOT let position bias affect your judgment (Response A appearing first \
does not make it better).
- If both responses are equally good or equally bad, you MUST say Tie.
- Base your judgment on the designer profile provided. A response that is \
eloquent but inconsistent with the profile is WORSE than a rough but \
authentic one.

### Output Format (strict JSON)

{
  "persona_fidelity":        {"winner": "A" or "B" or "Tie", "reason": "<1 sentence>"},
  "cognitive_authenticity":   {"winner": "A" or "B" or "Tie", "reason": "<1 sentence>"},
  "scenario_grounding":      {"winner": "A" or "B" or "Tie", "reason": "<1 sentence>"},
  "linguistic_naturalness":  {"winner": "A" or "B" or "Tie", "reason": "<1 sentence>"},
  "final_verdict":           {"winner": "A" or "B" or "Tie", "reason": "<1 sentence>"}
}

The final_verdict should reflect the overall winner considering all four \
dimensions. It is NOT a simple majority vote — persona_fidelity and \
cognitive_authenticity should weigh more heavily than the other two.\
"""

USER_TEMPLATE = """\
### Designer Profile
- **Designer ID**: {designer_id}
- **Trait Summary**: {trait_summary}
{factor_scores_line}

### Scenario
- **Question**: {question_id}
{scenario_context_line}

### Response A
{response_a}

### Response B
{response_b}

Compare Response A and Response B for this designer under this scenario. \
Output your evaluation as JSON.\
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    winner: str          # "A" | "B" | "Tie"
    reason: str = ""


@dataclass
class RoundResult:
    swap: bool
    raw_verdict: dict[str, Any]
    actual_winner: str   # "model" | "baseline" | "tie"


@dataclass
class JudgePairResult:
    user_id: str
    task_id: str
    final: str           # "win" | "lose" | "tie"
    rounds: list[RoundResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class WinRateReport:
    win: int
    lose: int
    tie: int
    total: int
    win_rate: float            # (wins + 0.5*ties) / total
    win_rate_strict: float     # wins / total
    per_question: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Profile builder (from raw data, same as prep.py)
# ---------------------------------------------------------------------------

LIKERT_TO_POLARITY: dict[int, str] = {
    1: "强-", 2: "弱-", 3: "弱-", 4: "弱+", 5: "弱+", 6: "强+",
}


def _build_profile_summaries(
    data_dir: Path,
    factor_scores_file: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build {user_id: {trait_summary, factor_scores}} from raw CSVs."""
    import pandas as pd

    scales = pd.read_csv(data_dir / "scale_responses.csv")
    scales.columns = [c.strip().lstrip("﻿") for c in scales.columns]
    scales["user_id"] = scales["user_id"].astype(str)

    pivot = scales.pivot_table(
        index="user_id", columns="question_node", values="score", aggfunc="mean",
    )
    pivot = pivot.fillna(pivot.mean(numeric_only=True))

    profiles: dict[str, dict[str, Any]] = {}
    for uid in pivot.index:
        row = pivot.loc[uid]
        parts = []
        for col in sorted(row.index):
            score = row[col]
            polarity = LIKERT_TO_POLARITY.get(round(score), "弱+")
            parts.append(f"{col}={polarity}")
        profiles[uid] = {
            "trait_summary": "; ".join(parts),
            "factor_scores": "",
        }

    if factor_scores_file and factor_scores_file.exists():
        fs = pd.read_csv(factor_scores_file)
        fs.columns = [c.strip().lstrip("﻿") for c in fs.columns]
        fs["user_id"] = fs["user_id"].astype(str)
        factor_cols = [c for c in fs.columns if c.startswith("F") or c.startswith("factor")]
        for _, r in fs.iterrows():
            uid = str(r["user_id"])
            if uid in profiles:
                scores = [f"{c}={r[c]:.2f}" for c in factor_cols if c in r.index]
                profiles[uid]["factor_scores"] = "; ".join(scores)

    return profiles


# ---------------------------------------------------------------------------
# Prediction loader (reuse predictions.py schema)
# ---------------------------------------------------------------------------

def _load_preds(path: Path) -> dict[tuple[str, str], str]:
    """Load jsonl -> {(user_id, task_id): full_response}."""
    out: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            tid = str(rec.get("task_id", rec.get("question_id", "")))
            resp = rec.get("full_response", "")
            if not resp:
                pred = rec.get("prediction", {})
                if isinstance(pred, dict):
                    resp = pred.get("full_response", pred.get("text", ""))
                elif isinstance(pred, str):
                    resp = pred
            out[(uid, tid)] = resp
    return out


# ---------------------------------------------------------------------------
# Judge core (sync, single pair)
# ---------------------------------------------------------------------------

def _call_judge_api(
    system: str,
    user_msg: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Call OpenAI-compatible chat completion, return parsed JSON."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def _build_user_msg(
    profile: dict[str, Any],
    user_id: str,
    task_id: str,
    response_a: str,
    response_b: str,
) -> str:
    factor_line = ""
    if profile.get("factor_scores"):
        factor_line = f"- **Factor Scores**: {profile['factor_scores']}"
    return USER_TEMPLATE.format(
        designer_id=user_id,
        trait_summary=profile.get("trait_summary", "N/A"),
        factor_scores_line=factor_line,
        question_id=task_id,
        scenario_context_line="",
        response_a=response_a,
        response_b=response_b,
    )


def _parse_winner(verdict: dict[str, Any]) -> str:
    """Extract final_verdict.winner, tolerant to format variation."""
    fv = verdict.get("final_verdict", {})
    if isinstance(fv, dict):
        w = fv.get("winner", "Tie")
    elif isinstance(fv, str):
        w = fv
    else:
        w = "Tie"
    w = w.strip().upper()
    if w in ("A", "RESPONSE A"):
        return "A"
    if w in ("B", "RESPONSE B"):
        return "B"
    return "Tie"


def _mock_verdict() -> dict[str, Any]:
    """Dry-run: random verdict without API call."""
    dims = ["persona_fidelity", "cognitive_authenticity",
            "scenario_grounding", "linguistic_naturalness"]
    choices = ["A", "B", "Tie"]
    v: dict[str, Any] = {}
    for d in dims:
        w = random.choice(choices)
        v[d] = {"winner": w, "reason": f"[mock] random pick {w}"}
    final = random.choice(choices)
    v["final_verdict"] = {"winner": final, "reason": f"[mock] random pick {final}"}
    return v


def judge_single_pair(
    profile: dict[str, Any],
    user_id: str,
    task_id: str,
    model_response: str,
    baseline_response: str,
    *,
    judge_model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "https://yunwu.ai/v1",
    dry_run: bool = False,
) -> JudgePairResult:
    """Judge one (user, task) pair with position debiasing (2 rounds)."""
    rounds: list[RoundResult] = []

    for swap in (False, True):
        if swap:
            resp_a, resp_b = baseline_response, model_response
            label_a, label_b = "baseline", "model"
        else:
            resp_a, resp_b = model_response, baseline_response
            label_a, label_b = "model", "baseline"

        user_msg = _build_user_msg(profile, user_id, task_id, resp_a, resp_b)

        if dry_run:
            verdict = _mock_verdict()
        else:
            verdict = _call_judge_api(
                SYSTEM_PROMPT, user_msg, judge_model, api_key, base_url,
            )

        raw_winner = _parse_winner(verdict)
        if raw_winner == "A":
            actual = label_a
        elif raw_winner == "B":
            actual = label_b
        else:
            actual = "tie"

        rounds.append(RoundResult(swap=swap, raw_verdict=verdict, actual_winner=actual))

    wins = [r.actual_winner for r in rounds]
    if wins[0] == "model" and wins[1] == "model":
        final = "win"
    elif wins[0] == "baseline" and wins[1] == "baseline":
        final = "lose"
    else:
        final = "tie"

    return JudgePairResult(user_id=user_id, task_id=task_id, final=final, rounds=rounds)


# ---------------------------------------------------------------------------
# Async batch runner (concurrency-limited)
# ---------------------------------------------------------------------------

async def _judge_one_async(
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    **kwargs: Any,
) -> JudgePairResult:
    async with sem:
        return await loop.run_in_executor(None, lambda: judge_single_pair(**kwargs))


async def judge_batch_async(
    pairs: list[dict[str, Any]],
    *,
    judge_model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "https://yunwu.ai/v1",
    concurrency: int = 4,
    dry_run: bool = False,
) -> list[JudgePairResult]:
    """Run all pairs with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    tasks = [
        _judge_one_async(
            sem, loop,
            profile=p["profile"],
            user_id=p["user_id"],
            task_id=p["task_id"],
            model_response=p["model_response"],
            baseline_response=p["baseline_response"],
            judge_model=judge_model,
            api_key=api_key,
            base_url=base_url,
            dry_run=dry_run,
        )
        for p in pairs
    ]
    results = []
    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        r = await coro
        results.append(r)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("  judged %d / %d pairs", i + 1, total)
    return results


# ---------------------------------------------------------------------------
# Win rate computation
# ---------------------------------------------------------------------------

def compute_win_rate(judgments: list[JudgePairResult]) -> WinRateReport:
    w = sum(1 for j in judgments if j.final == "win")
    l = sum(1 for j in judgments if j.final == "lose")
    t = sum(1 for j in judgments if j.final == "tie")
    n = len(judgments)
    win_rate = (w + 0.5 * t) / n if n > 0 else 0.0
    strict = w / n if n > 0 else 0.0

    per_q: dict[str, dict[str, Any]] = {}
    from collections import defaultdict
    by_q: dict[str, list[JudgePairResult]] = defaultdict(list)
    for j in judgments:
        by_q[j.task_id].append(j)
    for qid, jlist in sorted(by_q.items()):
        qw = sum(1 for j in jlist if j.final == "win")
        ql = sum(1 for j in jlist if j.final == "lose")
        qt = sum(1 for j in jlist if j.final == "tie")
        qn = len(jlist)
        per_q[qid] = {
            "win": qw, "lose": ql, "tie": qt, "total": qn,
            "win_rate": round((qw + 0.5 * qt) / qn, 4) if qn else 0.0,
        }

    return WinRateReport(
        win=w, lose=l, tie=t, total=n,
        win_rate=round(win_rate, 4),
        win_rate_strict=round(strict, 4),
        per_question=per_q,
    )


# ---------------------------------------------------------------------------
# Report renderer (markdown)
# ---------------------------------------------------------------------------

def render_report(report: WinRateReport, label: str = "") -> str:
    lines = [f"## Pairwise Win Rate{f' — {label}' if label else ''}", ""]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Win    | {report.win} |")
    lines.append(f"| Lose   | {report.lose} |")
    lines.append(f"| Tie    | {report.tie} |")
    lines.append(f"| Total  | {report.total} |")
    lines.append(f"| **Win Rate** | **{report.win_rate:.2%}** |")
    lines.append(f"| Win Rate (strict) | {report.win_rate_strict:.2%} |")
    lines.append("")

    if report.per_question:
        lines.append("### Per-Question Breakdown")
        lines.append("")
        lines.append("| Question | Win | Lose | Tie | Win Rate |")
        lines.append("|----------|-----|------|-----|----------|")
        for qid, stats in sorted(report.per_question.items()):
            wr = stats["win_rate"]
            lines.append(
                f"| {qid} | {stats['win']} | {stats['lose']} | {stats['tie']} | {wr:.2%} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise LLM Judge: model vs baseline win rate",
    )
    parser.add_argument("--model-preds", type=Path, required=True,
                        help="model predictions jsonl")
    parser.add_argument("--baseline-preds", type=Path, required=True,
                        help="baseline predictions jsonl")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="raw data directory (for profile building)")
    parser.add_argument("--factor-scores-file", type=Path, default=None,
                        help="EFA factor scores CSV (optional)")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--base-url", default="https://yunwu.ai/v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path (default: stdout)")
    parser.add_argument("--label", default="",
                        help="label for the comparison (e.g., 'og_rag vs no_rag')")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip API calls, random verdicts")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error("No API key. Set OPENAI_API_KEY or pass --api-key.")
        sys.exit(1)

    logger.info("Loading predictions ...")
    model_preds = _load_preds(args.model_preds)
    baseline_preds = _load_preds(args.baseline_preds)

    common_keys = sorted(set(model_preds) & set(baseline_preds))
    if not common_keys:
        logger.error("No overlapping (user_id, task_id) between model and baseline.")
        sys.exit(1)
    logger.info("  %d common (user_id, task_id) pairs", len(common_keys))

    logger.info("Building designer profiles ...")
    profiles = _build_profile_summaries(args.data_dir, args.factor_scores_file)

    pairs: list[dict[str, Any]] = []
    skipped = 0
    for uid, tid in common_keys:
        if uid not in profiles:
            skipped += 1
            continue
        mr = model_preds[(uid, tid)]
        br = baseline_preds[(uid, tid)]
        if not mr.strip() or not br.strip():
            skipped += 1
            continue
        pairs.append({
            "profile": profiles[uid],
            "user_id": uid,
            "task_id": tid,
            "model_response": mr,
            "baseline_response": br,
        })
    if skipped:
        logger.warning("  skipped %d pairs (no profile or empty response)", skipped)
    logger.info("  %d pairs to judge", len(pairs))

    if not pairs:
        logger.error("Nothing to judge.")
        sys.exit(1)

    logger.info("Running judge (model=%s, concurrency=%d, dry_run=%s) ...",
                args.judge_model, args.concurrency, args.dry_run)
    t0 = time.time()
    results = asyncio.run(judge_batch_async(
        pairs,
        judge_model=args.judge_model,
        api_key=api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))
    elapsed = time.time() - t0
    logger.info("  done in %.1fs (%.2f pairs/s)", elapsed,
                len(results) / elapsed if elapsed > 0 else 0)

    report = compute_win_rate(results)
    md = render_report(report, label=args.label)

    output_data = {
        "label": args.label,
        "judge_model": args.judge_model,
        "model_preds": str(args.model_preds),
        "baseline_preds": str(args.baseline_preds),
        "n_pairs": len(pairs),
        "elapsed_seconds": round(elapsed, 1),
        "dry_run": args.dry_run,
        "report": report.to_dict(),
        "judgments": [j.to_dict() for j in results],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        md_path = args.output.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved JSON: %s", args.output)
        logger.info("Saved report: %s", md_path)
    else:
        print(md)
        print("---")
        print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
