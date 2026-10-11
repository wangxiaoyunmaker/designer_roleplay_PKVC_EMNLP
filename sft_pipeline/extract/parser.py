"""Inference output parsing: model generations → normalized JSONL.

Models trained with this pipeline emit text matching brief v3.4 §4 schema:

* **q7 direction** (3 steps): each step yields one segment block bracketed by
  ``[INTUITION] ... [/INTUITION]`` / ``[WEIGH]`` / ``[BOTTOM_LINE]``.
* **q1q6 direction** (single step): a ``[PKVC 分析] ...`` block followed by
  ``[答案] ... [/答案]``.

This module:

1. Reads raw inference output JSONL (one row = one inference call).
2. Parses the embedded block format into structured fields
   (``dominant_factor`` / ``sadr_activations`` / ``criterion`` / ``answer``).
3. Optionally calls :func:`eval_extractor.extract_pkvc` on the free-text
   ``推理`` (reasoning) sub-field so we get an extractor-grade trajectory
   for downstream metric computation.
4. Emits normalized JSONL consumed by ``eval_runner.py``.

Input JSONL row schema
----------------------
::

    {
        "user_id":     "...",
        "task_id":     "B1"|"B2"|"B3"|"Q1"|...|"Q6",
        "direction":   "q7"|"q1q6",
        "scenario_id": "Q7_seg1" | "Q3" | ...,
        "model_output": "<raw decoded text>"
    }

Output JSONL row schema
-----------------------
::

    {
        "user_id":          ...,
        "task_id":          ...,
        "direction":        ...,
        "scenario_id":      ...,
        "parsed": {
            "dominant_factor":  "V" | "V → K → V" | None,
            "sadr_activations": ["#V3+", "#服务保留-", ...],
            "criterion":        "..." | None,
            "answer":           "B" | "V > K > P > C" | None,
            "reasoning_nl":     "<free text>"
        },
        "extracted": {...optional, from LLM extractor...},
        "parse_warnings": [...]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("inference_parser")

# Regexes for segment headers
_SEG_BLOCK_RE = re.compile(
    r"\[(INTUITION|WEIGH|BOTTOM_LINE)\]"
    r"(.*?)"
    r"\[/\1\]",
    re.DOTALL,
)
_PKVC_BLOCK_RE = re.compile(
    r"\[PKVC\s*分析\](.*?)\[/PKVC\s*分析\]",
    re.DOTALL,
)
_ANSWER_BLOCK_RE = re.compile(
    r"\[答案\](.*?)\[/答案\]",
    re.DOTALL,
)
_FACTOR_RE = re.compile(
    r"主导(?:因子|路径)[:\s：]+([PKVC](?:\s*(?:→|->|>)\s*[PKVC])*|[PKVC])"
)
_SADR_RE = re.compile(r"(#[A-Za-z0-9_一-鿿]+)([+\-]?)")
_CRITERION_RE = re.compile(r"核心底线[:\s：]+([^\n\[]+)")
_REASONING_RE = re.compile(r"推理[:\s：]+(.+?)(?=\[/|\Z)", re.DOTALL)
_LETTER_RE = re.compile(r"[A-E]")
_PATH_ANSWER_RE = re.compile(r"([PKVC](?:\s*(?:>|→|->)\s*[PKVC]){1,})")


@dataclass(slots=True)
class ParsedSegment:
    seg_id: str                       # B1 / B2 / B3
    dominant_factor: str | None       # "V" or "V → K → V"
    sadr_activations: list[str] = field(default_factory=list)
    criterion: str | None = None
    reasoning_nl: str = ""


@dataclass(slots=True)
class ParsedInference:
    user_id: str
    task_id: str
    direction: Literal["q7", "q1q6"]
    scenario_id: str
    dominant_factor: str | None = None
    sadr_activations: list[str] = field(default_factory=list)
    criterion: str | None = None
    answer: str | None = None
    reasoning_nl: str = ""
    parse_warnings: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_factor(text: str) -> str | None:
    m = _FACTOR_RE.search(text)
    if not m:
        return None
    return re.sub(r"\s*->\s*", " → ", m.group(1)).replace(">", " → ").strip()


def parse_sadr_tokens(text: str) -> list[str]:
    """Extract ``#NAME±`` tokens; preserve order, deduplicate."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _SADR_RE.finditer(text):
        tok = f"{m.group(1)}{m.group(2)}"
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    return found


def parse_criterion(text: str) -> str | None:
    m = _CRITERION_RE.search(text)
    return m.group(1).strip() if m else None


def parse_reasoning(text: str) -> str:
    m = _REASONING_RE.search(text)
    return m.group(1).strip() if m else ""


def parse_answer_q1q6(text: str, question_id: str) -> str | None:
    """Parse the ``[答案] ... [/答案]`` block.

    * Q1 (priority) expects a path like ``V > K > P > C``.
    * Q2 (trajectory) expects an arrow path like ``P → K → V``.
    * Q3-Q6 (single MCQ) expect a single letter A-E.
    """
    m = _ANSWER_BLOCK_RE.search(text)
    body = m.group(1).strip() if m else text.strip()
    if question_id == "Q1":
        p = _PATH_ANSWER_RE.search(body)
        return p.group(1).replace(" ", "") if p else None
    if question_id == "Q2":
        p = _PATH_ANSWER_RE.search(body)
        if p:
            return re.sub(r"\s*->\s*", " → ", p.group(1)).replace(">", " → ").strip()
        return None
    # Q3-Q6: single letter
    lm = _LETTER_RE.search(body)
    return lm.group(0) if lm else None


# ---------------------------------------------------------------------------
# Direction-specific
# ---------------------------------------------------------------------------

def parse_q7_segment(text: str, task_id: str) -> ParsedSegment:
    """Parse one q7 segment block. ``task_id`` is the expected seg id."""
    expected_name = {"B1": "INTUITION", "B2": "WEIGH", "B3": "BOTTOM_LINE"}[task_id]
    blocks = _SEG_BLOCK_RE.findall(text)
    matched_body: str | None = None
    for name, body in blocks:
        if name == expected_name:
            matched_body = body
            break
    if matched_body is None:
        # tolerate model emitting just the segment without the [/.../] closer
        matched_body = text

    return ParsedSegment(
        seg_id=task_id,
        dominant_factor=parse_factor(matched_body),
        sadr_activations=parse_sadr_tokens(matched_body),
        criterion=parse_criterion(matched_body) if task_id == "B3" else None,
        reasoning_nl=parse_reasoning(matched_body) or matched_body.strip(),
    )


def parse_q1q6_response(text: str, question_id: str) -> dict[str, Any]:
    """Parse a single q1q6 inference response."""
    pkvc_match = _PKVC_BLOCK_RE.search(text)
    pkvc_body = pkvc_match.group(1) if pkvc_match else text

    return {
        "dominant_factor": parse_factor(pkvc_body),
        "sadr_activations": parse_sadr_tokens(pkvc_body),
        "reasoning_nl": parse_reasoning(pkvc_body) or pkvc_body.strip(),
        "answer": parse_answer_q1q6(text, question_id),
        "criterion": None,
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def parse_inference_row(row: dict[str, Any]) -> ParsedInference:
    """Parse one raw inference JSONL row."""
    user_id = str(row.get("user_id", ""))
    task_id = str(row.get("task_id", ""))
    direction = row.get("direction") or ("q7" if task_id.startswith("B") else "q1q6")
    scenario_id = row.get("scenario_id") or task_id
    text = row.get("model_output", "")
    warnings: list[str] = []

    parsed = ParsedInference(
        user_id=user_id, task_id=task_id, direction=direction,
        scenario_id=scenario_id,
    )

    if not text:
        warnings.append("empty model_output")
        parsed.parse_warnings = warnings
        return parsed

    if direction == "q7":
        seg = parse_q7_segment(text, task_id)
        parsed.dominant_factor = seg.dominant_factor
        parsed.sadr_activations = seg.sadr_activations
        parsed.criterion = seg.criterion
        parsed.reasoning_nl = seg.reasoning_nl
        if seg.dominant_factor is None:
            warnings.append("missing dominant_factor")
        if not seg.sadr_activations:
            warnings.append("missing sadr_activations")
    else:
        q = parse_q1q6_response(text, task_id)
        parsed.dominant_factor = q["dominant_factor"]
        parsed.sadr_activations = q["sadr_activations"]
        parsed.criterion = q["criterion"]
        parsed.reasoning_nl = q["reasoning_nl"]
        parsed.answer = q["answer"]
        if parsed.answer is None:
            warnings.append("could not extract answer letter / path")

    parsed.parse_warnings = warnings
    return parsed


def parse_all(rows: list[dict[str, Any]]) -> list[ParsedInference]:
    return [parse_inference_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Optional LLM-extractor pass on reasoning NL
# ---------------------------------------------------------------------------

def enrich_with_extractor(
    parsed: list[ParsedInference],
    model: str = "gpt-4o",
    use_cache: bool = True,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """For each parsed row, call the extractor on ``reasoning_nl`` and attach
    the result under ``extracted`` field.

    Important: returns *dicts* not ``ParsedInference`` because we add the
    ``extracted`` field. Skips rows whose reasoning_nl is empty.
    """
    try:
        from sft_pipeline.eval_extractor import extract_pkvc
    except ImportError:
        from eval_extractor import extract_pkvc  # type: ignore

    out: list[dict[str, Any]] = []
    for p in parsed:
        d = p.asdict()
        if p.reasoning_nl:
            try:
                res = extract_pkvc(p.reasoning_nl, model=model,
                                   use_cache=use_cache, api_key=api_key)
                d["extracted"] = {
                    "segment_role": res.segment_role,
                    "dominant_factor": res.dominant_factor,
                    "sadr_activations": res.sadr_activations,
                    "criterion": res.criterion,
                    "criterion_strength": res.criterion_strength,
                    "confidence": res.confidence,
                    "extraction_notes": res.extraction_notes,
                }
            except Exception as exc:  # noqa: BLE001
                d["extracted"] = None
                d["parse_warnings"] = d.get("parse_warnings", []) + [f"extractor failed: {exc}"]
        else:
            d["extracted"] = None
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Raw model inference JSONL.")
    p.add_argument("--output", type=Path, required=True,
                   help="Normalized parsed JSONL.")
    p.add_argument("--enrich-extractor", action="store_true",
                   help="Also call LLM extractor on reasoning_nl.")
    p.add_argument("--extractor-model", default="gpt-4o")
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rows = read_jsonl(args.input)
    logger.info("Loaded %d inference rows", len(rows))
    parsed = parse_all(rows)
    n_warn = sum(1 for x in parsed if x.parse_warnings)
    if n_warn:
        logger.warning("%d rows had parse warnings", n_warn)

    if args.enrich_extractor:
        logger.info("Calling extractor on reasoning_nl ...")
        enriched = enrich_with_extractor(
            parsed, model=args.extractor_model,
            use_cache=not args.no_cache, api_key=args.api_key,
        )
        write_jsonl(enriched, args.output)
    else:
        write_jsonl([p.asdict() for p in parsed], args.output)
    logger.info("Wrote → %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
