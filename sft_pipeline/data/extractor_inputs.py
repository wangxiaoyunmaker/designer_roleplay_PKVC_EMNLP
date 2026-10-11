"""CSV → extractor JSONL conversion + token budget estimation.

Reads the raw participant CSVs and emits JSONL files in the schema consumed by
``extractor_batch.py`` (one record per (designer, scenario) pair):

::

    {
        "id":          "<unique, used for dedup + caching>",
        "user_id":     "<wechat_id>",
        "scenario_id": "Q7_seg1" | "Q7_seg2" | "Q7_seg3" | "Q1" | "Q3" ... ,
        "text":        "<NL to extract>",
        "meta":        { "method": "text"|"voice", "letter": "A"|..., ... }
    }

Outputs (under ``data/processed/``):

* ``q7_segments.jsonl``         — 79 × 3 = up to 237 Q7 segment NLs (P0, long
  reasoning chains, the high-leverage extractor input).
* ``q1q6_explanations.jsonl``   — 79 × ~5 Q1-Q6 short explanation NLs (currently
  ``text_supplement`` is 1-10 chars; will rerun once 学姐 补长 to 50-100 字).
* ``extraction_budget.md``       — token counts + GPT-4o cost estimate.

CLI
---
::

    # Default: write everything under data/processed/
    python prepare_extractor_inputs.py

    # Custom paths / token model
    python prepare_extractor_inputs.py \\
        --csv data/task_responses.csv \\
        --output-dir data/processed/ \\
        --token-model gpt-4o
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("prepare_extractor")

# 中文 ≈ 1.5 tokens / char, 英文 ≈ 0.25 tokens / char.
# Q7 内容主要是中文, 用 1.5 估算上界, 实际 GPT-4o 中文 tokenization ~ 1-2 tokens/char.
CHARS_PER_TOKEN_CN = 1.5

# Pricing as of 2026-05 (US$/1M tokens). Source: openai.com/pricing
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "deepseek-chat":   {"input": 0.27, "output": 1.10},
}

MCQ_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6")
SORT_QUESTION = "Q2"
OPEN_QUESTION = "Q7"
META_QUESTION = "Q8"
Q7_SEG_KEYS = ("seg1", "seg2", "seg3")
Q7_SEG_TO_ID = {"seg1": "Q7_seg1", "seg2": "Q7_seg2", "seg3": "Q7_seg3"}


@dataclass(slots=True)
class ExtractorRecord:
    """Schema mirroring ``extractor_batch.py`` input."""
    id: str
    user_id: str
    scenario_id: str
    text: str
    meta: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


_LETTER_RE = re.compile(r"[A-E]\.")


def parse_answer_letter(answer: Any) -> str | None:
    if answer is None:
        return None
    s = str(answer)
    m = _LETTER_RE.search(s)
    return m.group(0)[0] if m else None


def parse_q7_segments(answer: Any) -> dict[str, dict[str, str]]:
    """Returns ``{"seg1": {"text": "...", "method": "text"|"voice"}, ...}``."""
    if answer is None or (isinstance(answer, float)):
        return {}
    try:
        obj = json.loads(str(answer))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for seg_key in Q7_SEG_KEYS:
        seg = obj.get(seg_key, {})
        if isinstance(seg, dict):
            text = str(seg.get("text", "")).strip()
            if text:
                out[seg_key] = {"text": text, "method": str(seg.get("method", "text"))}
    return out


def _stable_id(*parts: str) -> str:
    """Short deterministic ID for dedup."""
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def emit_q7_segments(rows: Iterable[dict[str, Any]]) -> list[ExtractorRecord]:
    out: list[ExtractorRecord] = []
    for row in rows:
        if row.get("question_id") != OPEN_QUESTION:
            continue
        uid = str(row.get("user_id", ""))
        segments = parse_q7_segments(row.get("answer_value"))
        for seg_key, payload in segments.items():
            scen_id = Q7_SEG_TO_ID[seg_key]
            out.append(ExtractorRecord(
                id=_stable_id(uid, scen_id),
                user_id=uid,
                scenario_id=scen_id,
                text=payload["text"],
                meta={"method": payload["method"], "source": "Q7"},
            ))
    return out


def emit_q1q6_explanations(rows: Iterable[dict[str, Any]]) -> list[ExtractorRecord]:
    out: list[ExtractorRecord] = []
    for row in rows:
        qid = row.get("question_id")
        if qid not in MCQ_QUESTIONS:
            continue
        uid = str(row.get("user_id", ""))
        explanation = str(row.get("text_supplement", "") or "").strip()
        if not explanation:
            continue
        letter = parse_answer_letter(row.get("answer_value"))
        out.append(ExtractorRecord(
            id=_stable_id(uid, qid, "exp"),
            user_id=uid,
            scenario_id=qid,
            text=explanation,
            meta={"letter": letter, "source": "Q1-Q6 explanation",
                  "explanation_length_chars": len(explanation)},
        ))
    return out


def estimate_tokens(text: str) -> int:
    """Rough upper-bound token count via char-density heuristic (Chinese-heavy)."""
    return int(len(text) * CHARS_PER_TOKEN_CN) + 1


def render_budget(
    q7_records: list[ExtractorRecord],
    q1q6_records: list[ExtractorRecord],
    model: str,
    prompt_overhead_tokens: int = 1500,
    output_tokens_per_record: int = 300,
) -> str:
    """Markdown budget report.

    Assumes one extractor call per record, with a fixed prompt overhead
    (system + few-shot template) and ~300 output tokens per JSON response.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        avail = ", ".join(PRICING.keys())
        raise ValueError(f"Unknown model {model!r}. Known: {avail}")
    in_rate = pricing["input"] / 1_000_000
    out_rate = pricing["output"] / 1_000_000

    lines = [f"# Extractor token budget (model: `{model}`)", ""]

    for label, recs in (("Q7 segments", q7_records), ("Q1-Q6 explanations", q1q6_records)):
        if not recs:
            continue
        n = len(recs)
        char_lengths = [len(r.text) for r in recs]
        per_record_input_tokens = [
            estimate_tokens(r.text) + prompt_overhead_tokens for r in recs
        ]
        total_in = sum(per_record_input_tokens)
        total_out = output_tokens_per_record * n
        cost = total_in * in_rate + total_out * out_rate

        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Records: **{n}**")
        lines.append(f"- Char length: min={min(char_lengths)}, "
                     f"median={sorted(char_lengths)[n//2]}, "
                     f"max={max(char_lengths)}, mean={sum(char_lengths)/n:.0f}")
        lines.append(f"- Estimated input tokens (incl. {prompt_overhead_tokens} prompt overhead/call): "
                     f"**{total_in:,}**")
        lines.append(f"- Estimated output tokens ({output_tokens_per_record}/record): "
                     f"**{total_out:,}**")
        lines.append(f"- **Cost: ${cost:.2f}** (in ${total_in*in_rate:.2f} + out ${total_out*out_rate:.2f})")
        lines.append("")

    # κ pilot estimate: 30 segments × 2 LLM calls
    if q7_records:
        lines.append("## κ pilot (30 Q7 segments × 2 models)")
        lines.append("")
        sample_tokens = sum(estimate_tokens(r.text) + prompt_overhead_tokens
                            for r in q7_records[:30])
        sample_out = output_tokens_per_record * 30
        sample_cost = sample_tokens * in_rate + sample_out * out_rate
        lines.append(f"- Per model: ~${sample_cost:.2f}; both: **~${sample_cost*2:.2f}**")
        lines.append("")

    lines.append("> 假设: prompt overhead per call = {0} tokens (system + few-shot); "
                 "output per call ≈ {1} tokens (JSON). 中文 token 估算用 1.5 chars/token "
                 "上界, 实际可能略低.".format(prompt_overhead_tokens, output_tokens_per_record))
    return "\n".join(lines)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV stripping UTF-8 BOM from header keys."""
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append({k.strip(): v for k, v in row.items()})
    return out


def write_jsonl(records: list[ExtractorRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_jsonl() + "\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("0512技术方案/data/task_responses.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("0512技术方案/data/processed"))
    parser.add_argument("--token-model", default="gpt-4o",
                        help="Model used to estimate budget. One of: "
                             + ", ".join(PRICING.keys()))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from sft_pipeline.data.prep import EXCLUDED_USERS
    rows = read_csv_rows(args.csv)
    n_raw = len(rows)
    rows = [r for r in rows if r.get("user_id") not in EXCLUDED_USERS]
    n_excluded = n_raw - len(rows)
    if n_excluded:
        logger.warning("Dropped %d rows from excluded users: %s",
                       n_excluded, sorted(EXCLUDED_USERS))
    logger.info("Loaded %d task_responses rows (post-exclusion)", len(rows))

    q7_records = emit_q7_segments(rows)
    q1q6_records = emit_q1q6_explanations(rows)
    logger.info("Q7 segments  : %d", len(q7_records))
    logger.info("Q1-Q6 expl.  : %d (text_supplement non-empty)", len(q1q6_records))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    q7_path = args.output_dir / "q7_segments.jsonl"
    q1q6_path = args.output_dir / "q1q6_explanations.jsonl"
    budget_path = args.output_dir / "extraction_budget.md"

    write_jsonl(q7_records, q7_path)
    write_jsonl(q1q6_records, q1q6_path)
    budget_md = render_budget(q7_records, q1q6_records, args.token_model)
    budget_path.write_text(budget_md, encoding="utf-8")

    logger.info("Wrote %s", q7_path)
    logger.info("Wrote %s", q1q6_path)
    logger.info("Wrote %s", budget_path)

    # Quick sanity: scenario_id distribution
    scen_count = Counter(r.scenario_id for r in q7_records + q1q6_records)
    for sid in sorted(scen_count):
        logger.info("  %s: %d records", sid, scen_count[sid])
    return 0


if __name__ == "__main__":
    sys.exit(main())
