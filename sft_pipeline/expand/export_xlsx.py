"""导出 designer 级 xlsx 总览.

1 row = 1 designer (post 排除 alpha-test, ~78 行). 列:
  - user_id, cluster_id
  - strong_plus_30d:  量表 score=6 的 item (强 +)
  - strong_minus_30d: 量表 score=1 的 item (强 -)
  - Q1_answer / Q1_explanation
  - Q3_answer / Q3_explanation
  - Q4_answer / Q4_explanation
  - Q5_answer / Q5_explanation
  - Q6_answer / Q6_explanation
  - Q2_cardid_order / Q2_explanation
  - Q8_self_report

explanation 列优先取扩写后的, 没扩的 (短 seed 也可能 validation fail 回退) 显示原 seed
+ "(seed only)" 后缀.

依赖: openpyxl (pandas.to_excel).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from sft_pipeline.data.prep import (
    EXCLUDED_USERS,
    LIKERT_TO_POLARITY,
    _read_csv,
    build_canonical_option_text,
    parse_answer_letter,
    parse_q2_cardid_order,
)

logger = logging.getLogger("expand.export_xlsx")


# 4-档极性映射 (xlsx 专用; 跟 prep.py 的 SFT prompt 映射不同):
#   强 + = score 5 或 6
#   弱 + = score 4
#   弱 - = score 3
#   强 - = score 1 或 2
LIKERT_TO_POLARITY_XLSX: dict[int, str] = {
    1: "strong_minus", 2: "strong_minus",
    3: "weak_minus",
    4: "weak_plus",
    5: "strong_plus", 6: "strong_plus",
}
POLARITY_BUCKETS = ("strong_plus", "weak_plus", "weak_minus", "strong_minus")


def collect_30d_polarity_4bucket(
    scales: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """每 user → {bucket: [item, ...]} 4-档. item 在 bucket 内按字典序."""
    out: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {b: [] for b in POLARITY_BUCKETS}
    )
    for row in scales.itertuples(index=False):
        uid = str(row.user_id)
        try:
            s = int(round(float(row.score)))
        except (TypeError, ValueError):
            continue
        bucket = LIKERT_TO_POLARITY_XLSX.get(s)
        if bucket is None:
            continue
        out[uid][bucket].append(str(row.question_node))
    # 排序每个 bucket
    for uid in out:
        for b in POLARITY_BUCKETS:
            out[uid][b].sort()
    return out


def build_xlsx(
    data_dir: Path,
    task_csv: Path,
    provenance_path: Path,
    out_xlsx: Path,
) -> None:
    """主入口. task_csv 应该是扩写后的 task_responses_expanded.csv."""
    users = _read_csv(data_dir / "users.csv")
    scales = _read_csv(data_dir / "scale_responses.csv")
    tasks = _read_csv(task_csv)
    # 同时读未扩写的原始 csv, 拿 seed 文本作对照列
    orig_csv = data_dir / "task_responses.csv"
    tasks_orig = _read_csv(orig_csv) if orig_csv.exists() else None
    cluster_path = data_dir / "user_cluster_efa.csv"
    cluster = _read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()

    # 一致性: cohort = users ∩ scales − alpha-test
    scale_uids = set(scales["user_id"].astype(str))
    user_uids = set(users["wechat_id"].astype(str))
    complete_uids = sorted((scale_uids & user_uids) - EXCLUDED_USERS)
    logger.info("Cohort: %d designers", len(complete_uids))

    cluster_labels: dict[str, int] = {}
    if not cluster.empty:
        cluster = cluster[~cluster["user_id"].astype(str).isin(EXCLUDED_USERS)]
        cluster_labels = dict(
            zip(cluster["user_id"].astype(str), cluster["cluster_id"].astype(int))
        )

    # 30 维 4 档极性 (强是 5-6 / 1-2, 弱是 4 / 3)
    polarity_4 = collect_30d_polarity_4bucket(scales)

    # 选项 canonical text (跨问卷版本去 drift)
    tasks = tasks.copy()
    tasks["user_id"] = tasks["user_id"].astype(str)
    tasks = tasks[tasks["user_id"].isin(complete_uids)]
    canonical = build_canonical_option_text(tasks)

    # 原始 seed text 索引: (user_id, qid) → 原 text_supplement
    seed_by_uid_q: dict[tuple[str, str], str] = {}
    if tasks_orig is not None:
        tasks_orig = tasks_orig.copy()
        tasks_orig["user_id"] = tasks_orig["user_id"].astype(str)
        for row in tasks_orig.itertuples(index=False):
            uid = str(row.user_id)
            if uid in complete_uids:
                nl = str(getattr(row, "text_supplement", "") or "").strip()
                seed_by_uid_q[(uid, row.question_id)] = nl

    # 装配 per-user per-Q answer + explanation
    per_user_q: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    q8_self: dict[str, str | None] = {}
    for row in tasks.itertuples(index=False):
        uid = str(row.user_id)
        qid = row.question_id
        nl = str(getattr(row, "text_supplement", "") or "").strip()

        if qid in {"Q1", "Q3", "Q4", "Q5", "Q6"}:
            letter = parse_answer_letter(row.answer_value)
            text = canonical.get((qid, letter), str(row.answer_value or ""))
            # 去掉前缀 "A." 等, 保留选项内容
            if "\n" in text:
                opt_text = text.split("\n", 1)[1].strip().strip('"')
            else:
                opt_text = text.strip().strip('"')
            per_user_q[uid][qid] = {
                "answer": f"{letter}: {opt_text}" if letter else opt_text,
                "explanation": nl,
            }
        elif qid == "Q2":
            order = parse_q2_cardid_order(row.answer_value)
            per_user_q[uid][qid] = {
                "answer": " > ".join(order) if order else "",
                "explanation": nl,
            }
        elif qid == "Q8":
            q8_self[uid] = parse_answer_letter(row.answer_value)

    # 加载 provenance 标记哪些是扩写过的
    expanded_keys: set[tuple[str, str]] = set()
    if provenance_path.exists():
        with provenance_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                if p.get("was_expanded"):
                    expanded_keys.add((p["user_id"], p["question_id"]))
    logger.info("Provenance: %d (user, qid) marked as expanded", len(expanded_keys))

    # 构造 DataFrame 行
    rows: list[dict] = []
    for uid in complete_uids:
        pol = polarity_4.get(uid, {b: [] for b in POLARITY_BUCKETS})
        row: dict = {
            "user_id": uid,
            "cluster_id": cluster_labels.get(uid, -1),
            "strong_plus_30d": ", ".join(pol["strong_plus"]),    # score 5-6
            "weak_plus_30d":   ", ".join(pol["weak_plus"]),      # score 4
            "weak_minus_30d":  ", ".join(pol["weak_minus"]),     # score 3
            "strong_minus_30d": ", ".join(pol["strong_minus"]),  # score 1-2
            "q8_self_report": q8_self.get(uid) or "",
        }
        qe = per_user_q.get(uid, {})
        # 顺序: Q1, Q2 (排序题独立), Q3, Q4, Q5, Q6
        order = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        for qid in order:
            entry = qe.get(qid, {})
            row[f"{qid}_answer"] = entry.get("answer", "")
            row[f"{qid}_seed"] = seed_by_uid_q.get((uid, qid), "")
            nl = entry.get("explanation", "")
            tag = "" if (uid, qid) in expanded_keys else "  ⟂seed"
            row[f"{qid}_expanded"] = f"{nl}{tag}"
        rows.append(row)

    df = pd.DataFrame(rows)
    # 期望列顺序
    cols = [
        "user_id", "cluster_id",
        "strong_plus_30d", "weak_plus_30d", "weak_minus_30d", "strong_minus_30d",
        "q8_self_report",
        "Q1_answer", "Q1_seed", "Q1_expanded",
        "Q2_answer", "Q2_seed", "Q2_expanded",
        "Q3_answer", "Q3_seed", "Q3_expanded",
        "Q4_answer", "Q4_seed", "Q4_expanded",
        "Q5_answer", "Q5_seed", "Q5_expanded",
        "Q6_answer", "Q6_seed", "Q6_expanded",
    ]
    df = df[cols]

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="designer_summary", index=False)
        # 调列宽
        ws = writer.sheets["designer_summary"]
        widths = {
            "user_id": 16, "cluster_id": 8,
            "strong_plus_30d": 50, "weak_plus_30d": 40,
            "weak_minus_30d": 40, "strong_minus_30d": 40,
            "q8_self_report": 6,
            "Q1_answer": 50, "Q1_seed": 20, "Q1_expanded": 80,
            "Q2_answer": 30, "Q2_seed": 20, "Q2_expanded": 80,
            "Q3_answer": 50, "Q3_seed": 20, "Q3_expanded": 80,
            "Q4_answer": 50, "Q4_seed": 20, "Q4_expanded": 80,
            "Q5_answer": 50, "Q5_seed": 20, "Q5_expanded": 80,
            "Q6_answer": 50, "Q6_seed": 20, "Q6_expanded": 80,
        }
        for i, col in enumerate(df.columns, 1):
            letter = ws.cell(row=1, column=i).column_letter
            ws.column_dimensions[letter].width = widths.get(col, 20)
        # 首行加粗 + freeze
        from openpyxl.styles import Font, Alignment
        bold = Font(bold=True)
        for cell in ws[1]:
            cell.font = bold
        ws.freeze_panes = "B2"
        # 内容自动换行
        wrap = Alignment(wrap_text=True, vertical="top")
        for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row_cells:
                cell.alignment = wrap

    logger.info("Wrote xlsx → %s (%d designers × %d cols)", out_xlsx, len(rows), len(cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--task-csv", type=Path, default=Path("data/task_responses_expanded.csv"))
    parser.add_argument("--provenance", type=Path,
                        default=Path("data/processed/q1q6_expansion_provenance.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/designer_summary.xlsx"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    build_xlsx(args.data_dir, args.task_csv, args.provenance, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
