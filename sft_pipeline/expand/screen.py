"""筛选需扩充的 Q1-Q6 解释 + 装配 ground 材料.

输入: data/ 目录 (task_responses.csv + scale_responses.csv + users.csv)
输出: JSONL, 每条一个 ExpansionCandidate, 含该 designer 的全部 ground 材料 inline,
      供 prompt.py 直接渲染.

筛选规则:
  - char_count(text_supplement) < threshold (默认 30) → 进 pipeline
  - char_count == 0 (空) 也进, 标记 ``is_empty=True``
  - alpha-test 用户 wiwsx19940412 排除 (跟 prep.py 一致)
  - Q1-Q6 + Q2 都纳入 (Q8 是 metadata 不在 prep.py 训练里, 跳过)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sft_pipeline.data.prep import (
    EXCLUDED_USERS,
    POLARITY_NL_LABEL,
    POLARITY_ORDER,
    _read_csv,
    build_canonical_option_text,
    build_profile_matrix,
    parse_answer_letter,
    parse_q2_cardid_order,
    parse_q7_segments,
    polarity_to_nl,
)

# 2026-05-26 fix: prep.py 默认 5→弱+ (6 才算强+), 但 sft_format / audit 用 prep_v2 的
# strict mapping (5-6→强+). expand pipeline 之前用 prep mapping 导致 trait 信号被弱化
# (score=5 designer 在 prompt 里只看到 "弱+ AI 替代论"), v2 anti-hallu 铁律 7 无法触发.
# Override 到 prep_v2 strict mapping, 保持跟下游 SFT polarity 一致.
LIKERT_TO_POLARITY = {
    1: "strong_minus", 2: "strong_minus",
    3: "weak_minus",   4: "weak_plus",
    5: "strong_plus",  6: "strong_plus",
}
import pandas as _pd


def numerical_to_polarity(row: "_pd.Series") -> dict[str, list[str]]:
    """Strict mapping override (5-6→强+). 跟 sft_format / prep_v2 一致."""
    out: dict[str, list[str]] = {k: [] for k in POLARITY_ORDER}
    for node, score in row.items():
        if _pd.isna(score):
            continue
        try:
            bucket = LIKERT_TO_POLARITY[int(round(float(score)))]
        except (KeyError, ValueError):
            continue
        out[bucket].append(str(node))
    return out

logger = logging.getLogger("expand.screen")

#: Questions we expand. Q8 是 metadata, prep.py 不训, 不处理.
EXPAND_QUESTIONS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")


@dataclass(slots=True)
class ExpansionCandidate:
    """单条扩写候选, 带全部 ground 材料 inline.

    候选 ID: ``f"{user_id}_{question_id}"``, 用于 cache key + dedup.
    """
    candidate_id: str
    user_id: str
    question_id: str
    seed_nl: str                        # 原 1-10 字短解释 (可能为空)
    seed_char_count: int
    is_empty: bool

    # Question-specific
    answer_letter: str | None           # Q1/Q3-Q6: A-E; Q2: None
    answer_text: str                    # 选项原文 (Q1/Q3-Q6) 或 cardId 排序 JSON (Q2)
    cardid_order: list[str] = field(default_factory=list)  # Q2 only

    # Cross-question ground (same designer)
    q7_segments: dict[str, str] = field(default_factory=dict)  # B1/B2/B3 raw NL
    q1q6_other_nl: dict[str, str] = field(default_factory=dict)  # 其他题 NL
    q8_self_report: str | None = None   # A/B/C
    profile_polarity_nl: str = ""       # 4 档 polarity NL block

    # Routing / book-keeping
    cluster_id: int = -1

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def screen_candidates(
    data_dir: Path,
    threshold: int = 25,
    cluster_filename: str = "user_cluster_efa.csv",
) -> list[ExpansionCandidate]:
    """读 data/, 返回所有需扩写候选 (含 ground 材料)."""
    users = _read_csv(data_dir / "users.csv")
    tasks = _read_csv(data_dir / "task_responses.csv")
    scales = _read_csv(data_dir / "scale_responses.csv")

    cluster_path = data_dir / cluster_filename
    cluster = _read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()

    # 排除 alpha-test + 限制到 (users ∩ scales)
    scale_uids = set(scales["user_id"].astype(str))
    user_uids = set(users["wechat_id"].astype(str))
    complete_uids = sorted((scale_uids & user_uids) - EXCLUDED_USERS)
    if not complete_uids:
        raise RuntimeError("No complete designers after exclusion.")
    tasks = tasks.copy()
    tasks["user_id"] = tasks["user_id"].astype(str)
    tasks = tasks[tasks["user_id"].isin(complete_uids)]
    scales = scales.copy()
    scales["user_id"] = scales["user_id"].astype(str)
    scales = scales[scales["user_id"].isin(complete_uids)]

    # Per-designer ground material 预处理
    profile_matrix = build_profile_matrix(scales)
    canonical_option = build_canonical_option_text(tasks)
    cluster_labels: dict[str, int] = {}
    if not cluster.empty:
        cluster = cluster[~cluster["user_id"].astype(str).isin(EXCLUDED_USERS)]
        cluster_labels = dict(zip(cluster["user_id"].astype(str), cluster["cluster_id"].astype(int)))

    by_user_q: dict[str, dict[str, dict[str, Any]]] = {uid: {} for uid in complete_uids}
    q7_by_user: dict[str, dict[str, str]] = {uid: {} for uid in complete_uids}
    q8_by_user: dict[str, str | None] = {uid: None for uid in complete_uids}

    for row in tasks.itertuples(index=False):
        uid = str(row.user_id)
        qid = row.question_id
        nl = str(getattr(row, "text_supplement", "") or "").strip()
        ans = str(row.answer_value or "")

        if qid in {"Q1", "Q3", "Q4", "Q5", "Q6"}:
            letter = parse_answer_letter(row.answer_value)
            text = canonical_option.get((qid, letter), ans)
            by_user_q[uid][qid] = {
                "answer_letter": letter,
                "answer_text": text,
                "seed_nl": nl,
                "cardid_order": [],
            }
        elif qid == "Q2":
            by_user_q[uid][qid] = {
                "answer_letter": None,
                "answer_text": ans,
                "seed_nl": nl,
                "cardid_order": parse_q2_cardid_order(row.answer_value),
            }
        elif qid == "Q7":
            q7_by_user[uid] = parse_q7_segments(row.answer_value)
        elif qid == "Q8":
            q8_by_user[uid] = parse_answer_letter(row.answer_value)

    # 装配候选
    candidates: list[ExpansionCandidate] = []
    for uid in complete_uids:
        if uid not in profile_matrix.index:
            continue
        polarity = numerical_to_polarity(profile_matrix.loc[uid])
        polarity_nl = polarity_to_nl(polarity)
        q1q6_other_nl = {
            q: by_user_q[uid][q]["seed_nl"]
            for q in by_user_q[uid]
            if by_user_q[uid][q].get("seed_nl")
        }
        for qid in EXPAND_QUESTIONS:
            entry = by_user_q[uid].get(qid)
            if entry is None:
                continue
            seed = entry["seed_nl"]
            n = len(seed)
            if n >= threshold:
                continue  # 已达标, 不扩
            other_nl = {q: v for q, v in q1q6_other_nl.items() if q != qid}
            candidates.append(ExpansionCandidate(
                candidate_id=f"{uid}_{qid}",
                user_id=uid,
                question_id=qid,
                seed_nl=seed,
                seed_char_count=n,
                is_empty=(n == 0),
                answer_letter=entry["answer_letter"],
                answer_text=entry["answer_text"],
                cardid_order=entry["cardid_order"],
                q7_segments=q7_by_user.get(uid, {}),
                q1q6_other_nl=other_nl,
                q8_self_report=q8_by_user.get(uid),
                profile_polarity_nl=polarity_nl,
                cluster_id=int(cluster_labels.get(uid, -1)),
            ))
    logger.info(
        "Screened %d candidates from %d designers (threshold < %d chars)",
        len(candidates), len(complete_uids), threshold,
    )
    return candidates


def write_candidates(candidates: list[ExpansionCandidate], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.to_jsonl() + "\n")
    return len(candidates)


def load_candidates(path: Path) -> list[ExpansionCandidate]:
    out: list[ExpansionCandidate] = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(ExpansionCandidate(**obj))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--threshold", type=int, default=25,
                        help="char count < threshold → expand. Default 25 "
                             "(25-29 字 seed 一般已成完整短句, 留原样不洗 designer 风格).")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/q1q6_expansion_candidates.jsonl"))
    parser.add_argument("--cluster-filename", default="user_cluster_efa.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cands = screen_candidates(args.data_dir, args.threshold, args.cluster_filename)
    n = write_candidates(cands, args.output)

    # 分布报告
    from collections import Counter
    by_q = Counter(c.question_id for c in cands)
    empty = sum(1 for c in cands if c.is_empty)
    logger.info("Wrote %d candidates → %s", n, args.output)
    logger.info("  by question: %s", dict(by_q))
    logger.info("  empty seeds: %d", empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
