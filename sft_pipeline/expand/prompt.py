"""扩写 prompt 渲染.

从 expand_q1q6_v1.md 解析出 system + user template, 用 ExpansionCandidate
填充 placeholder. 返回 (system_msg, user_msg) tuple.
"""
from __future__ import annotations

import re
from pathlib import Path

from sft_pipeline.expand.screen import ExpansionCandidate

_FENCE_RE = re.compile(r"```\s*\n(.*?)\n```", re.DOTALL)

PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "expand_q1q6_v3.md"

_CARDID_NAME: dict[str, str] = {
    "P1": "P1-需求", "P2": "P2-冲突", "P3": "P3-边界",
    "K1": "K1-经验", "K2": "K2-数据", "K3": "K3-工具",
    "V1": "V1-用户", "V2": "V2-社会", "V3": "V3-底线",
    "C1": "C1-形态", "C2": "C2-语义", "C3": "C3-叙事",
}


def load_prompt_template() -> tuple[str, str]:
    """从 expand_q1q6_v2.md 提取 system + user template.

    跟 extract/llm.py 的 _load_prompt_template 同策略: 第一个 ``` 代码块是
    system, 第二个是 user template.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH}")
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in _FENCE_RE.finditer(content)]
    if len(fences) < 2:
        raise ValueError(
            f"Prompt template missing fenced blocks (got {len(fences)}, need ≥ 2)."
        )
    return fences[0], fences[1]


def _format_cardid_order(cardid_order: list[str]) -> str:
    if not cardid_order:
        return "(未填)"
    named = [_CARDID_NAME.get(c, c) for c in cardid_order]
    return " > ".join(named)


def _format_q1q6_other_nl(other_nl: dict[str, str]) -> str:
    if not other_nl:
        return "(无其他题解释)"
    lines = []
    for qid in sorted(other_nl):
        v = other_nl[qid].strip()
        if v:
            lines.append(f"- [{qid}] {v}")
    return "\n".join(lines) if lines else "(无其他题解释)"


def render(candidate: ExpansionCandidate) -> tuple[str, str]:
    """返回 (system, user) prompt 字符串, 直接喂给 chat completion."""
    system, user_tmpl = load_prompt_template()

    placeholders = {
        "{QUESTION_ID}": candidate.question_id,
        "{ANSWER_LETTER_OR_NA}": candidate.answer_letter or "N/A (Q2 是排序题)",
        "{ANSWER_TEXT}": (candidate.answer_text or "(无)").strip()[:400],
        "{SEED_NL}": candidate.seed_nl or "(原始解释为空)",
        "{SEED_CHAR_COUNT}": str(candidate.seed_char_count),
        "{Q7_B1}": (candidate.q7_segments.get("B1") or "(未填)").strip(),
        "{Q7_B2}": (candidate.q7_segments.get("B2") or "(未填)").strip(),
        "{Q7_B3}": (candidate.q7_segments.get("B3") or "(未填)").strip(),
        "{PROFILE_POLARITY_NL}": candidate.profile_polarity_nl,
        "{Q2_CARDID_ORDER}": _format_cardid_order(candidate.cardid_order),
        "{Q8_SELF_REPORT}": candidate.q8_self_report or "(未填)",
        "{Q1Q6_OTHER_NL}": _format_q1q6_other_nl(candidate.q1q6_other_nl),
    }
    user = user_tmpl
    for k, v in placeholders.items():
        user = user.replace(k, str(v))
    return system, user


def estimate_token_count(text: str) -> int:
    """中文 token 粗估: 1 char ≈ 1 token. 用于预算检查."""
    return len(text)


if __name__ == "__main__":
    import argparse
    import json
    from sft_pipeline.expand.screen import load_candidates

    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path,
                   default=Path("data/processed/q1q6_expansion_candidates.jsonl"))
    p.add_argument("--show", type=int, default=1, help="show first N rendered prompts")
    args = p.parse_args()

    cands = load_candidates(args.candidates)
    for i, c in enumerate(cands[: args.show]):
        sys_msg, user_msg = render(c)
        print(f"=== candidate {i} ({c.candidate_id}) ===")
        print("--- system ---")
        print(sys_msg)
        print("--- user ---")
        print(user_msg)
        print(f"--- est tokens: sys={len(sys_msg)} user={len(user_msg)} ---")
