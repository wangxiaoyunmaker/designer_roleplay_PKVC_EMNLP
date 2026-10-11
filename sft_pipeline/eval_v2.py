"""eval_v2.py — Per-segment metric (framework v1.1 §11).

输入: gen/v2/fold_{k}/{baseline}/{uid}__seg{1,2,3}.json (infer_v2 输出)
跟 GT 比对:
  - GT routing: v8.7 extractor 抽 real Q7.segN NL → state_sequence (data_v3/processed/q7_extracted_v8.7_full.jsonl)
  - GT NL:      real user Q7.segN text (从 task_responses.csv Q7 解析)

Metrics (per seg):
  1. PKVC label match:
       seg1 first_reaction: dominant_factor 单字母 match
       seg2 weighing:       factor-collapse + Kendall-tau on common factors
                            (复用 q2_v8_7_ranking_eval.py 思路)
       seg3 bottom_line:    factor-level dominant_factor match + chain 末端 match
  2. NL embed cosine: text-embedding-3-small (cached, optional)
  3. LLM-as-judge: gpt-4o pairwise "更像该 designer" (optional)
  4. parsed_ok 率: 多少 generated completion schema 合法

输出: reports/v2/fold_{k}/{baseline}_eval.json + .md (人读)

CLI:
    python -m sft_pipeline.eval_v2 \\
        --config sft_pipeline/config/config_v2.yaml \\
        --fold 0 --baselines B1 B5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("eval_v2")


# ─────────────────────────────────────────────────────────────────────
# GT loaders
# ─────────────────────────────────────────────────────────────────────

def load_real_q7_segs_text(tasks_csv: Path) -> dict[str, dict[str, str]]:
    """从 task_responses.csv Q7 行解析 JSON → {uid: {seg1, seg2, seg3}}."""
    df = pd.read_csv(tasks_csv)
    out: dict[str, dict[str, str]] = {}
    for _, row in df[df.question_id == "Q7"].iterrows():
        try:
            d = json.loads(row["answer_value"])
            uid = str(row["user_id"])
            out[uid] = {
                "seg1": d.get("seg1", {}).get("text", ""),
                "seg2": d.get("seg2", {}).get("text", ""),
                "seg3": d.get("seg3", {}).get("text", ""),
            }
        except Exception:
            continue
    return out


def load_v8_7_gt_routing(jsonl: Path) -> dict[tuple[str, str], dict]:
    """key = (uid, seg) → {dominant_factor, state_sequence, criterion, confidence}."""
    out = {}
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            key = (str(r["user_id"]), r["segment_role"])
            out[key] = r.get("pred", {})
    return out


# ─────────────────────────────────────────────────────────────────────
# Routing parse: generated string → factor sequence
# ─────────────────────────────────────────────────────────────────────

_FACTOR_TOKEN_RE = re.compile(r"\b([PKVC])(?:[1-3])?\b")


def routing_str_to_factor_seq(routing: str) -> list[str]:
    """'P3 推翻假定 → K3 评估技术栈' → ['P','K']
       'P1 界定 > V3 设立 > K2 测试' → ['P','V','K']
       'C (Concept ...)' → ['C']
       first-occurrence dedup, max 4 letter.
    """
    tokens = _FACTOR_TOKEN_RE.findall(routing)
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return seen


def routing_str_to_first_factor(routing: str) -> str | None:
    m = _FACTOR_TOKEN_RE.search(routing)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def kendall_tau_common(a: list[str], b: list[str]) -> float | None:
    """Kendall tau on common elements only. Range -1 to 1."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    a_pos = {x: i for i, x in enumerate(a) if x in common}
    b_pos = {x: i for i, x in enumerate(b) if x in common}
    n = len(common)
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = common[i], common[j]
            da = a_pos[x] - a_pos[y]
            db = b_pos[x] - b_pos[y]
            if da * db > 0:
                concord += 1
            elif da * db < 0:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else None


def seg_routing_metrics(gen_seq: list[str], gt_seq: list[str]) -> dict:
    if not gen_seq or not gt_seq:
        return {"top1_match": False, "top3_overlap": 0.0, "jaccard": 0.0,
                "kendall_tau": None}
    top1 = gen_seq[0] == gt_seq[0]
    s_gen, s_gt = set(gen_seq), set(gt_seq)
    jacc = len(s_gen & s_gt) / max(1, len(s_gen | s_gt))
    top3_overlap = len(set(gen_seq[:3]) & set(gt_seq[:3])) / 3.0
    kt = kendall_tau_common(gen_seq, gt_seq)
    return {"top1_match": top1, "top3_overlap": top3_overlap, "jaccard": jacc,
            "kendall_tau": kt}


def embed_cosine(text_a: str, text_b: str, embed_cache: dict[str, list[float]],
                 client, model_name: str) -> float | None:
    if not text_a.strip() or not text_b.strip():
        return None
    def _embed(t):
        if t in embed_cache:
            return np.array(embed_cache[t])
        resp = client.embeddings.create(model=model_name, input=t)
        v = resp.data[0].embedding
        embed_cache[t] = v
        return np.array(v)
    try:
        va, vb = _embed(text_a), _embed(text_b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
    except Exception as e:
        logger.warning("embed failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_eval(cfg: dict, fold: int, baselines: list[str], do_embed: bool = False) -> None:
    paths = cfg["paths"]
    root = Path.cwd()
    gen_dir = Path(paths["gen_dir"]) / f"fold_{fold}"
    report_dir = Path(paths["report_dir"]) / f"fold_{fold}"
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GT...")
    real_q7 = load_real_q7_segs_text(root / paths["task_responses"])
    v87_gt = load_v8_7_gt_routing(root / paths["q7_extracted_v8_7"])

    embed_cache: dict[str, list[float]] = {}
    client = None
    if do_embed:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.warning("OpenAI client init failed; skipping embed: %s", e)
            do_embed = False

    for bname in baselines:
        b_dir = gen_dir / bname
        if not b_dir.exists():
            logger.warning("No generations for %s, skipping", bname)
            continue

        per_sample: list[dict] = []
        for gen_file in sorted(b_dir.glob("*__seg*.json")):
            with gen_file.open(encoding="utf-8") as f:
                g = json.load(f)
            uid = g["user_id"]
            seg = g["seg"]
            # GT routing factor seq
            gt_pred = v87_gt.get((uid, seg), {})
            gt_seq = gt_pred.get("state_sequence", [])
            # Generated routing factor seq
            gen_seq = routing_str_to_factor_seq(g.get("routing", ""))
            rm = seg_routing_metrics(gen_seq, gt_seq)

            # NL embed cosine
            real_nl = real_q7.get(uid, {}).get(seg, "")
            cos = None
            if do_embed and real_nl and g.get("nl"):
                cos = embed_cosine(g["nl"], real_nl, embed_cache, client,
                                    cfg["eval"]["embed_model"])

            per_sample.append({
                "user_id": uid, "seg": seg, "baseline": bname,
                "parsed_ok": g.get("parsed_ok", False),
                "gen_routing": g.get("routing", ""),
                "gen_factor_seq": gen_seq,
                "gt_factor_seq": gt_seq,
                "gt_dominant": gt_pred.get("dominant_factor"),
                **rm,
                "nl_embed_cos": cos,
            })

        # aggregate
        if not per_sample:
            logger.warning("No per-sample for %s", bname)
            continue
        df = pd.DataFrame(per_sample)
        agg: dict = {}
        for seg in ("seg1", "seg2", "seg3"):
            sub = df[df.seg == seg]
            if sub.empty: continue
            agg[seg] = {
                "n": len(sub),
                "parsed_ok_pct": float(sub["parsed_ok"].mean() * 100),
                "top1_match_pct": float(sub["top1_match"].mean() * 100),
                "top3_overlap_mean": float(sub["top3_overlap"].mean()),
                "jaccard_mean": float(sub["jaccard"].mean()),
                "kendall_tau_mean": (
                    float(sub["kendall_tau"].dropna().mean())
                    if sub["kendall_tau"].dropna().size > 0 else None
                ),
            }
            if do_embed:
                agg[seg]["nl_embed_cos_mean"] = (
                    float(sub["nl_embed_cos"].dropna().mean())
                    if sub["nl_embed_cos"].dropna().size > 0 else None
                )

        report = {
            "fold": fold, "baseline": bname,
            "n_samples": len(per_sample),
            "per_segment": agg,
            "samples": per_sample,
        }
        out_json = report_dir / f"{bname}_eval.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # human-readable md
        md_lines = [f"# {bname} fold {fold} eval", "", f"n_samples = {len(per_sample)}", ""]
        for seg in ("seg1", "seg2", "seg3"):
            if seg not in agg: continue
            a = agg[seg]
            md_lines.append(f"## {seg}  n={a['n']}")
            md_lines.append(f"- parsed_ok: {a['parsed_ok_pct']:.1f}%")
            md_lines.append(f"- PKVC top-1 (dominant) match: {a['top1_match_pct']:.1f}%")
            md_lines.append(f"- Top-3 overlap: {a['top3_overlap_mean']:.2f}")
            md_lines.append(f"- Jaccard: {a['jaccard_mean']:.2f}")
            kt = a["kendall_tau_mean"]
            md_lines.append(f"- Kendall tau (common): {kt:.2f}" if kt is not None else "- Kendall tau: n/a")
            if do_embed and a.get("nl_embed_cos_mean") is not None:
                md_lines.append(f"- NL embed cosine: {a['nl_embed_cos_mean']:.3f}")
            md_lines.append("")
        out_md = report_dir / f"{bname}_eval.md"
        out_md.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("[%s] eval done → %s + %s", bname, out_json.name, out_md.name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_pipeline/config/config_v2.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--baselines", nargs="+", default=["B5"])
    p.add_argument("--embed", action="store_true", help="启用 NL embed cosine (需 OPENAI_API_KEY)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg, args.fold, args.baselines, do_embed=args.embed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
