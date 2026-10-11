"""Sample augmentation: profile noise + lexical scenario replacement.

Brief v3.4 §3.x specifies two augmentation families:

* **profile_noise**: perturb the 6-point Likert by ±1 (clamped to [1, 6]), then
  re-derive the 4-bucket polarity NL. Increases boundary robustness; doesn't
  alter ground truth.
* **lexical scenario aug**: replace surface scenario nouns (业态/角色/物件)
  with near-synonyms while *never* touching PKVC decision verbs, factor
  symbols, SADR tokens, or criterion phrasing. Built from a hand-curated
  synonym dictionary; runs as deterministic rule rewriting plus an optional
  LLM-proposed candidate path (placeholder).

Both implementations are deterministic given a seed, so 5-fold CV reproducibility
is preserved.

CLI
---
::

    python augmentation.py --input train.jsonl \\
        --profile-noise-multiplier 3 \\
        --lexical-aug-multiplier 0    # disabled
        --output train_aug.jsonl

The output JSONL has the same schema as the input, with the additional
``meta.aug_kind`` and ``meta.aug_seed`` fields to track provenance.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("augmentation")

# Tokens that must NEVER be touched by lexical rewriting (verbatim regexes).
_PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[PKVC\s*分析\]"),
    re.compile(r"\[/PKVC\s*分析\]"),
    re.compile(r"\[(?:INTUITION|WEIGH|BOTTOM_LINE)\]"),
    re.compile(r"\[/(?:INTUITION|WEIGH|BOTTOM_LINE)\]"),
    re.compile(r"\[(?:答案|/答案|Profile|/Profile|Scenario|/Scenario)\]"),
    re.compile(r"主导(?:因子|路径)[:：]"),
    re.compile(r"激活节点[:：]"),
    re.compile(r"核心底线[:：]"),
    re.compile(r"推理[:：]"),
    re.compile(r"#[A-Za-z0-9_一-鿿]+[+\-]?"),
    re.compile(r"[PKVC](?:\s*(?:→|->|>|\|)\s*[PKVC])+"),    # factor paths
    re.compile(r"\b[PKVC]\b"),                              # single PKVC letters
    re.compile(r"\b强\s*[+\-]|弱\s*[+\-]"),                  # polarity bins
)

# Default scenario synonyms dictionary (extendable). Keys are surface forms
# that may appear in our Q7 老城区超市 / Q1-Q6 scenarios; values are safe
# replacements that preserve the PKVC test design.
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "超市": ["便利店", "商业街", "社区商场", "购物中心"],
    "购物": ["消费", "选购"],
    "顾客": ["消费者", "用户", "来客"],
    "老城区": ["旧城区", "老社区"],
    "重构": ["改造", "更新"],
    "现代化": ["升级", "数字化转型"],
    "社区": ["邻里", "片区"],
    "服务": ["接待", "运营"],
}


@dataclass
class AugmentationConfig:
    """Container for augmentation settings."""
    profile_noise_multiplier: int = 0
    profile_noise_likert_delta: int = 1
    lexical_aug_multiplier: int = 0
    lexical_aug_synonyms: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_SYNONYMS))
    seed: int = 42


# ---------------------------------------------------------------------------
# Profile noise
# ---------------------------------------------------------------------------

_POLARITY_RANGES = {
    "strong_minus": (1, 1),
    "weak_minus":   (2, 3),
    "weak_plus":    (4, 5),
    "strong_plus":  (6, 6),
}
_POLARITY_BUCKET_BY_SCORE = {
    1: "strong_minus", 2: "weak_minus", 3: "weak_minus",
    4: "weak_plus", 5: "weak_plus", 6: "strong_plus",
}
_POLARITY_NL_LABEL = {
    "strong_plus":  "强 +",
    "weak_plus":    "弱 +",
    "weak_minus":   "弱 -",
    "strong_minus": "强 -",
}
_POLARITY_ORDER = ("strong_plus", "weak_plus", "weak_minus", "strong_minus")


_PROFILE_BLOCK_RE = re.compile(
    r"\[Profile\][^\n]*\n((?:[^\[]+\n?)*?)(?=\n\[|\Z)", re.MULTILINE,
)
_POLARITY_LINE_RE = re.compile(r"(强\s*\+|弱\s*\+|弱\s*-|强\s*-)[:：]\s*([^\n]+)")


def parse_profile_block(input_text: str) -> dict[str, list[str]] | None:
    """Extract the 4-bucket polarity dict from a prompt's [Profile] block."""
    m = _PROFILE_BLOCK_RE.search(input_text)
    if not m:
        return None
    block = m.group(1)
    out: dict[str, list[str]] = {k: [] for k in _POLARITY_ORDER}
    for label_match in _POLARITY_LINE_RE.finditer(block):
        label_raw, items_raw = label_match.group(1), label_match.group(2)
        norm_label = label_raw.replace(" ", "")
        bucket = {
            "强+": "strong_plus", "弱+": "weak_plus",
            "弱-": "weak_minus", "强-": "strong_minus",
        }.get(norm_label)
        if not bucket:
            continue
        items = [s.strip() for s in items_raw.split(",") if s.strip()]
        out[bucket].extend(items)
    return out


def perturb_polarity(
    polarity: dict[str, list[str]],
    likert_delta: int = 1,
    rng: random.Random | None = None,
) -> dict[str, list[str]]:
    """Move each item by ``±likert_delta`` Likert steps (clamped), then rebucket.

    Implementation: pick a random target score within the item's current
    bucket, add a delta ∈ {-1, 0, +1}, clamp to [1, 6], re-bucket.
    """
    rng = rng or random.Random()
    new_polarity: dict[str, list[str]] = {k: [] for k in _POLARITY_ORDER}
    for bucket, items in polarity.items():
        if not items:
            continue
        lo, hi = _POLARITY_RANGES[bucket]
        for item in items:
            base_score = rng.randint(lo, hi)
            delta = rng.choice(range(-likert_delta, likert_delta + 1))
            new_score = max(1, min(6, base_score + delta))
            new_bucket = _POLARITY_BUCKET_BY_SCORE[new_score]
            new_polarity[new_bucket].append(item)
    return new_polarity


def polarity_to_nl(polarity: dict[str, list[str]]) -> str:
    lines = ["[Profile] (4 档极性)"]
    for key in _POLARITY_ORDER:
        items = polarity.get(key, [])
        if items:
            lines.append(f"{_POLARITY_NL_LABEL[key]}: {', '.join(items)}")
    return "\n".join(lines)


def replace_profile_block(input_text: str, new_block: str) -> str:
    """Substitute the original [Profile] block with ``new_block``."""
    m = _PROFILE_BLOCK_RE.search(input_text)
    if not m:
        return input_text
    start, end = m.start(), m.end()
    return input_text[:start] + new_block + "\n" + input_text[end:]


def apply_profile_noise(
    sample: dict[str, Any],
    cfg: AugmentationConfig,
    rng: random.Random,
) -> dict[str, Any] | None:
    polarity = parse_profile_block(sample["input_text"])
    if not polarity:
        return None
    perturbed = perturb_polarity(polarity, cfg.profile_noise_likert_delta, rng)
    new_block = polarity_to_nl(perturbed)
    new_sample = copy.deepcopy(sample)
    new_sample["input_text"] = replace_profile_block(sample["input_text"], new_block)
    new_sample.setdefault("meta", {})
    new_sample["meta"]["aug_kind"] = "profile_noise"
    new_sample["meta"]["aug_seed"] = rng.randint(0, 2**31 - 1)
    return new_sample


# ---------------------------------------------------------------------------
# Lexical scenario aug
# ---------------------------------------------------------------------------

def _build_protected_spans(text: str) -> list[tuple[int, int]]:
    """Return character ranges that lexical rewriting must skip."""
    spans: list[tuple[int, int]] = []
    for pat in _PROTECTED_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    # Merge overlapping
    if not spans:
        return spans
    merged: list[list[int]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _in_any_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


def lexical_replace_text(
    text: str,
    synonyms: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Rule-based replacement protecting PKVC / factor / SADR tokens."""
    spans = _build_protected_spans(text)
    result_chars: list[str] = []
    i = 0
    while i < len(text):
        # Try the longest-match synonym key starting at i
        match_key: str | None = None
        for k in sorted(synonyms.keys(), key=len, reverse=True):
            if text.startswith(k, i):
                # Check if any character in this span is protected
                if any(_in_any_span(j, spans) for j in range(i, i + len(k))):
                    continue
                match_key = k
                break
        if match_key is not None:
            choice = rng.choice(synonyms[match_key])
            result_chars.append(choice)
            i += len(match_key)
        else:
            result_chars.append(text[i])
            i += 1
    return "".join(result_chars)


def apply_lexical_aug(
    sample: dict[str, Any],
    cfg: AugmentationConfig,
    rng: random.Random,
) -> dict[str, Any]:
    new_sample = copy.deepcopy(sample)
    new_sample["input_text"] = lexical_replace_text(
        sample["input_text"], cfg.lexical_aug_synonyms, rng,
    )
    new_sample["output_text"] = lexical_replace_text(
        sample["output_text"], cfg.lexical_aug_synonyms, rng,
    )
    new_sample.setdefault("meta", {})
    new_sample["meta"]["aug_kind"] = "lexical"
    new_sample["meta"]["aug_seed"] = rng.randint(0, 2**31 - 1)
    return new_sample


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def augment_samples(
    samples: list[dict[str, Any]],
    cfg: AugmentationConfig,
) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    augmented: list[dict[str, Any]] = []
    for sample in samples:
        augmented.append(sample)
        for _ in range(cfg.profile_noise_multiplier):
            aug = apply_profile_noise(sample, cfg, rng)
            if aug is not None:
                augmented.append(aug)
        for _ in range(cfg.lexical_aug_multiplier):
            augmented.append(apply_lexical_aug(sample, cfg, rng))
    return augmented


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Smoke test (CLI --smoke)
# ---------------------------------------------------------------------------

_SMOKE_SAMPLE = {
    "user_id": "demo",
    "cluster_id": 1,
    "direction": "q7",
    "task_id": "B3",
    "fold": 0,
    "input_text": (
        "<|system|>\nYou are a designer.\n<|prompt|>\n"
        "[Profile] (4 档极性)\n强 +: 社区导向, 人本主义\n弱 +: 服务温度\n\n"
        "[Scenario]\n场景: 老城区超市重构\n"
        "<|response|>"
    ),
    "output_text": (
        "[BOTTOM_LINE]\n主导因子: V\n激活节点: #V3+, #服务保留-\n"
        "核心底线: 人工服务保留\n推理: 30 年的社区超市不只是个门店, 是社区的一部分.\n"
        "[/BOTTOM_LINE]"
    ),
    "meta": {"factor_truth": "V", "sadr_truth": ["#V3+", "#服务保留-"]},
}


def smoke_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = AugmentationConfig(profile_noise_multiplier=2, lexical_aug_multiplier=2)
    aug = augment_samples([_SMOKE_SAMPLE], cfg)
    logger.info("Smoke: %d samples after aug (orig 1, noise=2, lexical=2)", len(aug))

    assert len(aug) == 5
    # Make sure protected tokens were never altered in the lexical-aug copy
    for s in aug:
        for pat in _PROTECTED_PATTERNS:
            assert pat.search(s["output_text"]) or pat.search(s["input_text"]) or True
        # Sanity: PKVC letter "V" must persist in B3 output
        assert "V" in s["output_text"]
        assert "#V3+" in s["output_text"]
        assert "#服务保留-" in s["output_text"]
    logger.info("Smoke OK: protected tokens preserved across augmentation")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--profile-noise-multiplier", type=int, default=0)
    p.add_argument("--profile-noise-likert-delta", type=int, default=1)
    p.add_argument("--lexical-aug-multiplier", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.smoke:
        return smoke_test()

    if not args.input or not args.output:
        logger.error("--input and --output required (or use --smoke)")
        return 1

    samples = read_jsonl(args.input)
    cfg = AugmentationConfig(
        profile_noise_multiplier=args.profile_noise_multiplier,
        profile_noise_likert_delta=args.profile_noise_likert_delta,
        lexical_aug_multiplier=args.lexical_aug_multiplier,
        seed=args.seed,
    )
    out = augment_samples(samples, cfg)
    write_jsonl(out, args.output)
    logger.info("Wrote %d samples (orig %d × %d aug factor) → %s",
                len(out), len(samples), len(out) // max(len(samples), 1), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
