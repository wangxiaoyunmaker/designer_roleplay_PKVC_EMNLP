"""分析 3 baseline inference 结果, 跟 ground truth 比.

简单指标:
  - length 分布: B1/B2/B3 输出长度 vs ground truth 长度
  - PKVC dominant 一致性: 把 simulated_nl 喂回 v8.7 extractor (单次, 不 batch), 看 dom == ground_truth.dominant 比例
    (注: 跑全 1458 个 extract 是 $$$, 这里只算 subset)
  - lexical overlap: 简单 Jaccard 词重合 (基线指标, 真要 BLEU/ROUGE 后续加)

输出: reports/datav3/baselines_q7_analysis.md
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def tokenize_zh(s: str) -> set[str]:
    """字符级 unigram + bi-gram for Chinese lexical overlap (简单基线)."""
    s = re.sub(r"[\s\W_]+", "", s)
    if not s:
        return set()
    unigrams = set(s)
    bigrams = {s[i:i+2] for i in range(len(s) - 1)}
    return unigrams | bigrams


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/datav3/baselines_q7_inference.jsonl")
    ap.add_argument("--output", default="reports/datav3/baselines_q7_analysis.md")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).open(encoding="utf-8")]
    print(f"loaded {len(rows)} rows")

    # group: (designer, scenario) → {baseline: row}
    by_key: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if not r.get("ok"): continue
        by_key[(r["designer_id"], r["scenario"])][r["baseline"]] = r

    # ============= analysis 1: length distribution =============
    print("\n== length distribution ==")
    lens = defaultdict(list)
    gt_lens = []
    for key, three in by_key.items():
        if "B1" in three:
            gt = three["B1"].get("ground_truth_nl", "")
            if gt:
                gt_lens.append(len(gt))
        for b in ("B1", "B2", "B3"):
            if b in three:
                lens[b].append(len(three[b]["simulated_nl"]))
    print(f"  ground truth: n={len(gt_lens)}, median={np.median(gt_lens):.0f}, mean={np.mean(gt_lens):.0f}")
    for b in ("B1", "B2", "B3"):
        if lens[b]:
            print(f"  {b}: n={len(lens[b])}, median={np.median(lens[b]):.0f}, mean={np.mean(lens[b]):.0f}")

    # ============= analysis 2: lexical jaccard vs ground truth =============
    print("\n== lexical overlap (jaccard char unigram+bigram, vs GT) ==")
    jacc_by_b = defaultdict(list)
    for key, three in by_key.items():
        gt = three.get("B1", {}).get("ground_truth_nl", "")
        if not gt:
            continue
        gt_tokens = tokenize_zh(gt)
        for b in ("B1", "B2", "B3"):
            if b not in three:
                continue
            sim_tokens = tokenize_zh(three[b]["simulated_nl"])
            jacc_by_b[b].append(jaccard(gt_tokens, sim_tokens))
    for b in ("B1", "B2", "B3"):
        if jacc_by_b[b]:
            print(f"  {b}: n={len(jacc_by_b[b])}, median jaccard={np.median(jacc_by_b[b]):.3f}, "
                  f"mean={np.mean(jacc_by_b[b]):.3f}")

    # ============= analysis 3: ground-truth PKVC dominant distribution =============
    print("\n== ground-truth PKVC dominant distribution per seg ==")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        sc_keys = [(k, v) for k, v in by_key.items() if k[1] == sc]
        gt_doms = []
        for (did, _), three in sc_keys:
            gt = three.get("B1", {}).get("ground_truth_pkvc")
            if gt:
                gt_doms.append(gt["dominant"][0])
        if gt_doms:
            print(f"  {sc}: n={len(gt_doms)}, {dict(Counter(gt_doms))}")

    # ============= save markdown report =============
    md = ["# Q7 3-Baseline Inference 分析报告\n"]
    md.append(f"输入: {args.input}\n")
    md.append(f"总样本: {len(rows)} inference calls, {len(by_key)} unique (designer, scenario) pairs\n")

    md.append("## 长度分布\n")
    md.append("| 来源 | n | median 字数 | mean 字数 |")
    md.append("|---|---|---|---|")
    md.append(f"| Ground truth | {len(gt_lens)} | {np.median(gt_lens):.0f} | {np.mean(gt_lens):.0f} |")
    for b in ("B1", "B2", "B3"):
        if lens[b]:
            md.append(f"| {b} | {len(lens[b])} | {np.median(lens[b]):.0f} | {np.mean(lens[b]):.0f} |")
    md.append("")

    md.append("## Lexical 重合 (Jaccard char 1+2-gram, vs ground truth)\n")
    md.append("| baseline | n | median | mean |")
    md.append("|---|---|---|---|")
    for b in ("B1", "B2", "B3"):
        if jacc_by_b[b]:
            md.append(f"| {b} | {len(jacc_by_b[b])} | {np.median(jacc_by_b[b]):.3f} | {np.mean(jacc_by_b[b]):.3f} |")
    md.append("")

    md.append("## Ground truth PKVC dominant 分布 (per seg)\n")
    md.append("| scenario | n | distribution |")
    md.append("|---|---|---|")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        sc_keys = [v for k, v in by_key.items() if k[1] == sc]
        gt_doms = []
        for three in sc_keys:
            gt = three.get("B1", {}).get("ground_truth_pkvc")
            if gt:
                gt_doms.append(gt["dominant"][0])
        if gt_doms:
            md.append(f"| {sc} | {len(gt_doms)} | {dict(Counter(gt_doms))} |")

    out_path = Path(args.output)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
