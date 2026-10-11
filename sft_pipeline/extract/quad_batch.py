"""T_CAPS Quad async batch extractor.

输入:
  - data/task_responses_expanded.csv (designer × Q3-Q6 NL, 扩写后)
  - reports/datav3/per_designer/option_trait_mapping.json (学姐 mapping)
  - data/user_cluster_efa.csv (cluster_id per designer)
  - sft_pipeline/config/prompts/quad_extractor_v1.md (prompt)
  - sft_pipeline/extract/schema.py (json_schema strict)

输出:
  - data/processed/quad_extracted.jsonl (1 quad per line)

跑法:
  # smoke 1 cluster
  python -m sft_pipeline.extract.quad_batch --cluster 1 --limit 0
  # 全量
  python -m sft_pipeline.extract.quad_batch

CLI flags:
  --cluster N    只跑该 cluster designers
  --limit N      只跑前 N 个 (debug, 0 = 全量)
  --concurrency  并发数 (默认 8)
  --output PATH  输出 jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sft_pipeline.data.prep import parse_answer_letter
from sft_pipeline.expand.llm import call_chat
from sft_pipeline.extract.codebook import (
    QUAD_QUESTIONS, fill_codebook, get_quad_metadata, load_mapping, validate_quad,
    TCapsQuad,
)
from sft_pipeline.extract.schema import get_schema

logger = logging.getLogger("quad_batch")

PROMPT_PATH = Path("sft_pipeline/config/prompts/quad_extractor_v4.md")
DEFAULT_TASK_CSV = Path("data/task_responses_expanded.csv")
DEFAULT_MAPPING = Path("reports/datav3/per_designer/option_trait_mapping.json")
DEFAULT_CLUSTER = Path("data/user_cluster_efa.csv")
DEFAULT_OUTPUT = Path("data/processed/quad_extracted.jsonl")


@dataclass
class BatchStats:
    total: int = 0
    ok: int = 0
    schema_fail: int = 0
    api_error: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        rate = self.total / max(self.elapsed_sec, 1e-9)
        return (f"total={self.total} ok={self.ok} schema_fail={self.schema_fail} "
                f"api_err={self.api_error} elapsed={self.elapsed_sec:.1f}s ({rate:.2f}/s)")


def load_prompt() -> tuple[str, str]:
    content = PROMPT_PATH.read_text(encoding="utf-8")
    fences = [m.group(1).strip() for m in re.finditer(r"```\s*\n(.*?)\n```", content, re.DOTALL)]
    return fences[0], fences[1]


def render(template: str, q: str, choice: str, nl: str, c_tag: str,
           s_primary: list, s_secondary: str,
           o_primary_hint: str | None, o_subtype_hint: str | None) -> str:
    out = template
    placeholders = {
        "{C_TAG}": c_tag,
        "{S_PRIMARY}": str(s_primary),
        "{S_SECONDARY}": s_secondary,
        "{QUESTION_ID}": q.lstrip("Q"),
        "{CHOICE_SELECTED}": choice or "?",
        "{INPUT_TEXT}": nl or "(NL 为空)",
        "{O_PRIMARY_HINT}": str(o_primary_hint or "None"),
        "{O_SUBTYPE_HINT}": repr(o_subtype_hint or "None"),
    }
    for k, v in placeholders.items():
        out = out.replace(k, str(v))
    return out


async def extract_one(
    designer_id: str, question_id: str, choice: str, nl: str,
    cluster_id: int, mapping: dict,
    system: str, user_tmpl: str,
    sem: asyncio.Semaphore, model: str, schema: dict,
) -> dict:
    """单 quad 抽取. 返回 1 行 jsonl dict, 含 quad fields + provenance."""
    async with sem:
        md = get_quad_metadata(mapping, question_id, choice)
        if md is None:
            return {
                "designer_id": designer_id, "question_id": question_id,
                "choice": choice, "error": f"no mapping for {question_id}_{choice}",
            }

        user = render(user_tmpl, question_id, choice, nl, md["C"],
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
            return {
                "designer_id": designer_id, "question_id": question_id,
                "choice": choice, "error": f"api_error: {exc}",
            }

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return {
                "designer_id": designer_id, "question_id": question_id,
                "choice": choice, "error": f"json_parse: {e}", "raw": raw[:200],
            }

        # 装配 TCapsQuad, 查 codebook, validate
        quad = TCapsQuad(
            C=md["C"],
            s_primary=md["s_primary"],
            s_secondary={"factor": "TBD", "direction": ""},  # placeholder
            p_surface=obj["p_surface"],
            o_primary=obj["o_primary"],
            o_subtype=obj.get("o_subtype"),
            quad_id=f"{designer_id}_{question_id}",
            designer_id=designer_id,
            question_id=question_id,
            confidence=obj["confidence"],
        )
        quad = fill_codebook(quad)
        fails = validate_quad(quad)

        return {
            "designer_id": designer_id,
            "question_id": question_id,
            "fold": -1,  # set later if needed
            "cluster_id": cluster_id,
            "C": quad.C,
            "C_conflict_axes": md["C_conflict_axes"],
            "s_primary": quad.s_primary,
            "s_primary_with_stance": md["s_primary_with_stance"],
            "s_secondary": quad.s_secondary,
            "p_surface": quad.p_surface,
            "p_modulation": quad.p_modulation,
            "p_stance": quad.p_stance,
            "o_primary": quad.o_primary,
            "o_subtype": quad.o_subtype,
            "sadr_activations": obj.get("sadr_activations", []),  # v4 new
            "confidence": quad.confidence,
            "extraction_notes": obj.get("extraction_notes", ""),
            "_meta": {
                "choice_selected": choice,
                "n_cluster_peers": md["n_samples"],
                "o_primary_hint": md["o_primary_hint"],
                "o_subtype_hint": md["o_subtype_hint"],
                "model": model,
                "validation_fails": fails,
            },
        }


async def run_batch(jobs: list[tuple], mapping: dict, system: str, user_tmpl: str,
                    model: str, schema: dict, concurrency: int) -> tuple[list[dict], BatchStats]:
    sem = asyncio.Semaphore(concurrency)
    stats = BatchStats(total=len(jobs))
    t0 = time.time()

    async def _wrap(idx, args):
        out = await extract_one(*args, mapping=mapping, system=system,
                                user_tmpl=user_tmpl, sem=sem, model=model, schema=schema)
        if "error" in out:
            stats.api_error += 1
        elif out.get("_meta", {}).get("validation_fails"):
            stats.schema_fail += 1
        else:
            stats.ok += 1
        if (idx + 1) % 10 == 0 or idx + 1 == stats.total:
            logger.info("Progress %d/%d ok=%d fail=%d err=%d",
                        idx + 1, stats.total, stats.ok, stats.schema_fail, stats.api_error)
        return out

    tasks = [asyncio.create_task(_wrap(i, j)) for i, j in enumerate(jobs)]
    results = await asyncio.gather(*tasks)
    stats.elapsed_sec = time.time() - t0
    return results, stats


def collect_jobs(task_csv: Path, cluster_csv: Path,
                 cluster_filter: int | None, limit: int) -> list[tuple]:
    """返回 list of (designer_id, qid, choice, nl, cluster_id)."""
    df = pd.read_csv(task_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["question_id"].isin(QUAD_QUESTIONS)]
    df["letter"] = df["answer_value"].apply(parse_answer_letter)

    cluster_df = pd.read_csv(cluster_csv)
    cluster_df["user_id"] = cluster_df["user_id"].astype(str)
    cluster_map = dict(zip(cluster_df["user_id"], cluster_df["cluster_id"]))

    if cluster_filter is not None:
        df = df[df["user_id"].map(lambda u: cluster_map.get(u, -1)) == cluster_filter]

    jobs = []
    for row in df.itertuples(index=False):
        nl = str(row.text_supplement or "").strip()
        if not nl or not row.letter:
            continue
        jobs.append((str(row.user_id), row.question_id, row.letter, nl,
                     int(cluster_map.get(str(row.user_id), -1))))
    if limit:
        jobs = jobs[:limit]
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-csv", type=Path, default=DEFAULT_TASK_CSV)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--cluster-csv", type=Path, default=DEFAULT_CLUSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster", type=int, default=None,
                        help="只跑该 cluster designers (smoke), None=全量")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    mapping = load_mapping(str(args.mapping))
    jobs = collect_jobs(args.task_csv, args.cluster_csv, args.cluster, args.limit)
    logger.info("Collected %d (designer, Q) jobs", len(jobs))
    if not jobs:
        logger.error("No jobs. Check cluster/limit/csv path.")
        return 1

    system, user_tmpl = load_prompt()
    schema = get_schema("quad_extraction")

    results, stats = asyncio.run(run_batch(
        jobs, mapping, system, user_tmpl, args.model, schema, args.concurrency,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Done: %s", stats.summary())
    logger.info("Wrote → %s", args.output)
    return 0 if stats.api_error == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
