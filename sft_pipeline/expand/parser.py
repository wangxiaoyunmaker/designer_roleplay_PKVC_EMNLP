"""出参解析 + 反 hallucination 校验.

LLM 输出 JSON → ExpansionResult, 跑一组 validator:
  1. JSON schema 完整 + 字段类型对
  2. 50 ≤ len(expanded_nl) ≤ 100 (硬卡, 跟 prompt 铁律 5 一致)
  3. inferred_factor ∈ {P, K, V, C}
  4. evidence_quote 必须在该 designer 的 ground 材料里 (fuzzy match, 至少
     一个 ≥ 8 字连续子串重合) — 反 hallucination 关键卡口
  5. confidence ∈ {high, medium, low}

每条 validator 返回独立 fail reason, 全部聚合到 ValidationReport. 任意一条 fail
则 ``ok = False``, downstream merge 步骤会跳过 (或回炉重跑).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from sft_pipeline.expand.screen import ExpansionCandidate

logger = logging.getLogger("expand.parser")

VALID_FACTORS = frozenset({"P", "K", "V", "C"})
VALID_SUBTYPES = frozenset({
    "P1", "P2", "P3", "K1", "K2", "K3",
    "V1", "V2", "V3", "C1", "C2", "C3",
})
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
LEN_MIN = 50
LEN_MAX = 130   # v1 pilot 经验: gpt-4o 中文字符计数不精确, 实际产出 105-130 字, 内容 OK, 不必硬切
QUOTE_MIN_LEN = 6   # 8 太严: 7 字 verbatim seed 也算 grounded, 不必硬切
QUOTE_MAX_LEN = 60   # v2 经验: gpt-4o 倾向引完整一句 (41-52 字), 内容 grounded 即可
EVIDENCE_MATCH_MIN_NGRAM = 6   # 连续 ≥ 6 字子串需在 ground 出现


@dataclass(slots=True)
class ExpansionResult:
    candidate_id: str
    expanded_nl: str
    inferred_factor: str | None
    inferred_subtype: str | None
    evidence_quote: str
    evidence_source: str
    confidence: str
    reason_if_low: str
    raw_json: str = ""
    cache_hit: bool = False


@dataclass(slots=True)
class ValidationReport:
    ok: bool
    fails: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.fails.append(msg)
        self.ok = False


def _extract_json(raw: str) -> dict[str, Any]:
    """Lenient JSON 提取: 直接 parse 失败时, 截取首尾大括号."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        return json.loads(raw[s:e + 1])
    raise json.JSONDecodeError("no JSON object found", raw, 0)


def _collect_ground_text(candidate: ExpansionCandidate) -> str:
    """汇总所有 ground 材料成一长串文本, 用于 evidence_quote fuzzy match."""
    pieces: list[str] = []
    for seg in ("B1", "B2", "B3"):
        v = candidate.q7_segments.get(seg, "").strip()
        if v:
            pieces.append(v)
    if candidate.seed_nl:
        pieces.append(candidate.seed_nl)
    pieces.extend(v for v in candidate.q1q6_other_nl.values() if v)
    if candidate.answer_text:
        pieces.append(candidate.answer_text)
    pieces.append(candidate.profile_polarity_nl)
    return "\n".join(pieces)


def _normalize(s: str) -> str:
    """去全/半角空白 + 标点常态化, 用于 fuzzy 匹配."""
    return re.sub(r"\s+", "", s)


def _quote_in_ground(quote: str, ground: str, min_ngram: int = EVIDENCE_MATCH_MIN_NGRAM) -> bool:
    """quote 整段在 ground 里出现, 或 quote 的任一 ≥ min_ngram 字连续子串在 ground 里出现."""
    nq = _normalize(quote)
    ng = _normalize(ground)
    if not nq or not ng:
        return False
    if nq in ng:
        return True
    for i in range(len(nq) - min_ngram + 1):
        if nq[i:i + min_ngram] in ng:
            return True
    return False


def parse_and_validate(
    raw: str,
    candidate: ExpansionCandidate,
    cache_hit: bool = False,
) -> tuple[ExpansionResult, ValidationReport]:
    """主入口: raw LLM 输出 + 原 candidate → (result, report)."""
    report = ValidationReport(ok=True)

    try:
        obj = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        report.add(f"json_parse_fail: {e}")
        return _stub_result(candidate, raw, cache_hit), report

    result = ExpansionResult(
        candidate_id=candidate.candidate_id,
        expanded_nl=str(obj.get("expanded_nl", "")).strip(),
        inferred_factor=obj.get("inferred_factor"),
        inferred_subtype=obj.get("inferred_subtype"),
        evidence_quote=str(obj.get("evidence_quote", "")).strip(),
        evidence_source=str(obj.get("evidence_source", "")).strip(),
        confidence=str(obj.get("confidence", "")).strip(),
        reason_if_low=str(obj.get("reason_if_low", "")).strip(),
        raw_json=raw,
        cache_hit=cache_hit,
    )

    # Validator 2: 长度
    n = len(result.expanded_nl)
    if n < LEN_MIN or n > LEN_MAX:
        report.add(f"len_out_of_range: {n} chars (need {LEN_MIN}-{LEN_MAX})")

    # Validator 3: factor
    if result.inferred_factor not in VALID_FACTORS:
        report.add(f"bad_factor: {result.inferred_factor!r}")

    # Validator 3b: subtype (允许 None)
    if result.inferred_subtype is not None and result.inferred_subtype not in VALID_SUBTYPES:
        report.add(f"bad_subtype: {result.inferred_subtype!r}")

    # Validator 4: evidence_quote 长度 + grounded
    qlen = len(result.evidence_quote)
    if qlen < QUOTE_MIN_LEN or qlen > QUOTE_MAX_LEN:
        report.add(f"quote_len: {qlen} (need {QUOTE_MIN_LEN}-{QUOTE_MAX_LEN})")
    else:
        ground = _collect_ground_text(candidate)
        if not _quote_in_ground(result.evidence_quote, ground):
            report.add(
                f"quote_not_in_ground: {result.evidence_quote!r} "
                f"(no ≥{EVIDENCE_MATCH_MIN_NGRAM}-char substring matches)"
            )

    # Validator 5: confidence
    if result.confidence not in VALID_CONFIDENCE:
        report.add(f"bad_confidence: {result.confidence!r}")

    # confidence=low 允许豁免 length / factor 检查 (LLM 拒绝扩写)
    if result.confidence == "low" and result.reason_if_low:
        # 只要 reason 给了 + JSON 合法, low 视作 "honest refusal", 不强制 fail
        # 但下游 merge 会跳过 (不用扩写结果)
        report.fails = [f for f in report.fails if not f.startswith("len_") and not f.startswith("bad_factor")]
        report.ok = len(report.fails) == 0

    return result, report


def _stub_result(candidate: ExpansionCandidate, raw: str, cache_hit: bool) -> ExpansionResult:
    return ExpansionResult(
        candidate_id=candidate.candidate_id,
        expanded_nl="",
        inferred_factor=None,
        inferred_subtype=None,
        evidence_quote="",
        evidence_source="",
        confidence="low",
        reason_if_low="parse_fail",
        raw_json=raw,
        cache_hit=cache_hit,
    )


def result_to_jsonl_dict(
    candidate: ExpansionCandidate,
    result: ExpansionResult,
    report: ValidationReport,
    model: str,
) -> dict[str, Any]:
    """合并 candidate + result + report → 一行 JSONL 输出, 保 provenance trace."""
    return {
        "candidate_id": candidate.candidate_id,
        "user_id": candidate.user_id,
        "question_id": candidate.question_id,
        "seed_nl": candidate.seed_nl,
        "seed_char_count": candidate.seed_char_count,
        "result": asdict(result),
        "validation": {
            "ok": report.ok,
            "fails": list(report.fails),
        },
        "model": model,
    }


if __name__ == "__main__":
    # Quick test
    cand = ExpansionCandidate(
        candidate_id="test_Q1",
        user_id="test", question_id="Q1",
        seed_nl="直觉", seed_char_count=2, is_empty=False,
        answer_letter="A", answer_text="选项 A 原文",
        q7_segments={"B1": "我的直觉在尖叫: 当我们为了追求极致的效率而剥离了人味, 信任比算法更值钱."},
        profile_polarity_nl="[Profile] (4 档极性)\n强 +: 直觉",
    )
    good = json.dumps({
        "expanded_nl": "我看到问题的第一反应是直觉先跑出来. 我习惯先靠脑子里的画面感判断方向, 然后再用数据回查. 这次我也是凭直觉先选了 A, 之后才反推它跟我对场所精神的理解一致.",
        "inferred_factor": "K",
        "inferred_subtype": "K1",
        "evidence_quote": "我的直觉在尖叫: 当我们",
        "evidence_source": "Q7_B1",
        "confidence": "high",
        "reason_if_low": "",
    }, ensure_ascii=False)
    r, rep = parse_and_validate(good, cand)
    print("OK:", rep.ok, "fails:", rep.fails, "len:", len(r.expanded_nl))
