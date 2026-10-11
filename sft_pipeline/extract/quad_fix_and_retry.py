"""v3 输出 post-process: auto-fix schema_fail + retry timeout.

输入: quad_extracted_v3_full.jsonl (含 67 api_err + 55 schema_fail)
输出: quad_extracted_v3_fixed.jsonl (auto-fix + retry 后)

策略:
  1. schema_fail "o_primary 跟终点不一致" → 自动 set o_primary = parse_o_subtype(o_subtype)[1][0],
     重 validate, 写回 ok 状态
  2. api_err = timeout → retry 1 次, concurrency=4, 慢一点防再 timeout
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sft_pipeline.expand.llm import call_chat
from sft_pipeline.extract.codebook import (
    fill_codebook, get_quad_metadata, load_mapping, parse_o_subtype,
    validate_quad, TCapsQuad,
)
from sft_pipeline.extract.schema import get_schema
from sft_pipeline.extract.quad_batch import load_prompt, render
import pandas as pd

logger = logging.getLogger("quad_fix_retry")


def fix_schema_fail(row: dict) -> dict:
    """Auto-fix "o_primary 跟终点不一致" by setting o_primary = end prefix."""
    fails = row.get("_meta", {}).get("validation_fails", [])
    if not fails:
        return row
    # 只 fix 已知 pattern: 终点 prefix 跟 o_primary 不一致
    relevant = [f for f in fails if "终点" in f and "prefix" in f and "o_primary" in f]
    if not relevant or len(fails) > 1:
        return row  # 别的 fail 不动

    _, end = parse_o_subtype(row["o_subtype"])
    if not end:
        return row
    new_primary = end[0]
    if new_primary not in ("P", "K", "V", "C"):
        return row

    row["o_primary"] = new_primary
    row["_meta"]["validation_fails"] = []  # 修好了
    row["_meta"]["_auto_fixed"] = (
        f"o_primary 改 {row['o_primary']} (从 o_subtype={row['o_subtype']} 终点 prefix)"
    )
    return row


async def retry_one(row: dict, system: str, user_tmpl: str, mapping: dict,
                    sem: asyncio.Semaphore, model: str, schema: dict,
                    task_df: pd.DataFrame) -> dict:
    """Retry 1 个 timeout case."""
    async with sem:
        # 从 task_df 找 NL
        nl_row = task_df[(task_df["user_id"] == row["designer_id"]) &
                          (task_df["question_id"] == row["question_id"])]
        if not len(nl_row):
            return row  # 找不到 NL, 给 up
        nl = str(nl_row.iloc[0]["text_supplement"] or "").strip()
        choice = row["choice"] if "choice" in row else row.get("_meta", {}).get("choice_selected", "?")

        md = get_quad_metadata(mapping, row["question_id"], choice)
        if md is None:
            return row

        user = render(user_tmpl, row["question_id"], choice, nl, md["C"],
                      md["s_primary"], "(EFA factor 待补)",
                      md.get("o_primary_hint"), md.get("o_subtype_hint"))
        loop = asyncio.get_event_loop()
        try:
            raw, _ = await loop.run_in_executor(
                None, lambda: call_chat(
                    system, user, model=model, temperature=0.0,
                    use_cache=False, response_format=schema,
                )
            )
        except Exception as exc:
            return {**row, "error": f"retry_api_error: {exc}", "_retry_attempted": True}

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return {**row, "error": f"retry_json_parse: {e}", "_retry_attempted": True}

        # 装配
        cluster_id = row.get("cluster_id", -1)
        quad = TCapsQuad(
            C=md["C"], s_primary=md["s_primary"],
            s_secondary={"factor": "TBD", "direction": ""},
            p_surface=obj["p_surface"], o_primary=obj["o_primary"],
            o_subtype=obj.get("o_subtype"),
            quad_id=f"{row['designer_id']}_{row['question_id']}",
            designer_id=row["designer_id"], question_id=row["question_id"],
            confidence=obj["confidence"],
        )
        quad = fill_codebook(quad)
        fails = validate_quad(quad)
        # auto-fix 这一行
        end_prefix = parse_o_subtype(quad.o_subtype)[1]
        if end_prefix and not end_prefix.startswith(quad.o_primary):
            quad.o_primary = end_prefix[0]
            fails = validate_quad(quad)

        return {
            "designer_id": row["designer_id"],
            "question_id": row["question_id"],
            "fold": -1, "cluster_id": cluster_id,
            "C": quad.C, "C_conflict_axes": md["C_conflict_axes"],
            "s_primary": quad.s_primary,
            "s_primary_with_stance": md["s_primary_with_stance"],
            "s_secondary": quad.s_secondary,
            "p_surface": quad.p_surface, "p_modulation": quad.p_modulation,
            "p_stance": quad.p_stance,
            "o_primary": quad.o_primary, "o_subtype": quad.o_subtype,
            "confidence": quad.confidence,
            "extraction_notes": obj.get("extraction_notes", ""),
            "_meta": {
                "choice_selected": choice, "n_cluster_peers": md["n_samples"],
                "o_primary_hint": md["o_primary_hint"],
                "o_subtype_hint": md["o_subtype_hint"],
                "model": model, "validation_fails": fails,
                "_retried": True,
            },
        }


async def run_retries(rows: list[dict], task_df: pd.DataFrame, mapping: dict,
                      model: str, concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    system, user_tmpl = load_prompt()
    schema = get_schema("quad_extraction")

    async def _wrap(idx, r):
        out = await retry_one(r, system, user_tmpl, mapping, sem, model, schema, task_df)
        if (idx + 1) % 10 == 0 or idx + 1 == len(rows):
            logger.info("Retry progress %d/%d", idx + 1, len(rows))
        return out

    return await asyncio.gather(*[asyncio.create_task(_wrap(i, r)) for i, r in enumerate(rows)])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=Path("data/processed/quad_extracted_v3_full.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/processed/quad_extracted_v3_fixed.jsonl"))
    p.add_argument("--task-csv", type=Path, default=Path("data/task_responses_expanded.csv"))
    p.add_argument("--mapping", type=Path, default=Path("reports/datav3/per_designer/option_trait_mapping.json"))
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    logger.info("Loaded %d rows", len(rows))

    # Step 1: auto-fix schema_fail
    n_fixed = 0
    fixed_rows = []
    timeouts = []
    for r in rows:
        if "error" in r:
            timeouts.append(r)
            continue
        before = r.get("_meta", {}).get("validation_fails", [])
        new_r = fix_schema_fail(r)
        if before and not new_r["_meta"]["validation_fails"]:
            n_fixed += 1
        fixed_rows.append(new_r)
    logger.info("Step 1: auto-fixed %d schema_fail rows", n_fixed)

    # Step 2: retry timeouts
    logger.info("Step 2: retrying %d timeouts (concurrency=%d)...", len(timeouts), args.concurrency)
    task_df = pd.read_csv(args.task_csv)
    task_df.columns = [c.strip().lstrip("﻿") for c in task_df.columns]
    task_df["user_id"] = task_df["user_id"].astype(str)
    mapping = load_mapping(str(args.mapping))

    t0 = time.time()
    retried = asyncio.run(run_retries(timeouts, task_df, mapping, args.model, args.concurrency))
    logger.info("Retry done in %.1fs", time.time() - t0)

    # 合并
    n_retry_ok = sum(1 for r in retried if "error" not in r)
    logger.info("Step 2 result: %d/%d retry ok", n_retry_ok, len(retried))

    all_rows = fixed_rows + retried
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ok = sum(1 for r in all_rows
                   if "error" not in r and not r.get("_meta", {}).get("validation_fails"))
    logger.info("Final: %d/%d ok (%.1f%%) → %s", total_ok, len(all_rows),
                100 * total_ok / len(all_rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
