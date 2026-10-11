"""评测: extractor 反推 PKVC + A1/A2/A3 三指标.

A1 trajectory match (主): 拼接预测段的 factor + sadr → vs cluster prototype Hamming
A2 cluster distribution match (辅): per-cluster soft 分布 JS-divergence
A3 individual L1 + within-cluster floor (sanity)

数据流:
  pred jsonl → extract_pkvc_trait() → 计算指标 vs ground truth
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PKVC_FACTORS = ["P", "K", "V", "C"]


def extract_pkvc_from_text(text: str) -> dict:
    """从生成的 NL 文本里反推 PKVC trait.

    简化版: 匹配 '主导因子: V'/'主导路径: V → K → V'/'激活节点: #13-, #14-'/'核心底线: ...'
    """
    factor_match = re.search(r"主导因子[: ：]\s*([PKVC])", text)
    path_match = re.search(r"主导路径[: ：]\s*([PKVC\s→\-]+)", text)
    sadr_match = re.findall(r"#(\d+)([+-])", text)
    criterion_match = re.search(r"核心底线[: ：]\s*([^\n\[]+)", text)
    return {
        "factor": factor_match.group(1) if factor_match else None,
        "factor_path": [c for c in (path_match.group(1) if path_match else "")
                         if c in PKVC_FACTORS],
        "sadr": [f"#{n}{s}" for n, s in sadr_match],
        "criterion": criterion_match.group(1).strip() if criterion_match else None,
    }


def hamming_factor(pred: list[str], truth: list[str]) -> float:
    """factor sequence 编辑距离 (简化为 Hamming, 长度不一致补 None)."""
    L = max(len(pred), len(truth))
    if L == 0:
        return 0.0
    p = pred + [None] * (L - len(pred))
    t = truth + [None] * (L - len(truth))
    return sum(pi != ti for pi, ti in zip(p, t)) / L


def jaccard_sadr(pred: list[str], truth: list[str]) -> float:
    """SADR 节点集合 Jaccard 相似度."""
    s_pred, s_truth = set(pred), set(truth)
    if not s_pred and not s_truth:
        return 1.0
    return len(s_pred & s_truth) / max(len(s_pred | s_truth), 1)


def compute_a1_trajectory_match(preds: list[dict], truths: list[dict]) -> dict:
    """A1 主指标: 每 designer × 每 题 trajectory 跟簇原型比 Hamming.

    简化版直接用 designer 真值做对比 (而不是簇原型 mean), 后续可换原型版本.
    """
    factor_dist = []
    sadr_jac = []
    for p, t in zip(preds, truths):
        extracted = extract_pkvc_from_text(p["prediction"].get("full_response", ""))
        pred_factors = [extracted["factor"]] if extracted["factor"] else extracted["factor_path"]
        truth_factors = t.get("factor_truth", [])
        if isinstance(truth_factors, str):
            truth_factors = [truth_factors]
        factor_dist.append(hamming_factor(pred_factors, truth_factors))
        sadr_jac.append(jaccard_sadr(extracted["sadr"], t.get("sadr_truth", [])))
    return {
        "factor_hamming_mean": float(np.mean(factor_dist)) if factor_dist else float("nan"),
        "sadr_jaccard_mean": float(np.mean(sadr_jac)) if sadr_jac else float("nan"),
        "n": len(preds),
    }


def compute_a2_cluster_distribution(preds: list[dict], truths: list[dict],
                                       cluster_labels: dict[str, int]) -> dict:
    """A2: per-cluster 答案分布 JS-divergence (方向二 MCQ; 方向一暂跳过).

    Args: preds 每条含 user_id + 预测的 letter (从 [答案] 段提取).
    """
    # 按 cluster 聚合
    cluster_pred: dict[int, list[str]] = defaultdict(list)
    cluster_truth: dict[int, list[str]] = defaultdict(list)
    for p, t in zip(preds, truths):
        c = cluster_labels.get(p["user_id"])
        if c is None:
            continue
        pred_letter = extract_letter(p["prediction"].get("full_response", ""))
        truth_letter = t.get("answer_truth")
        if pred_letter and truth_letter:
            cluster_pred[c].append(pred_letter)
            cluster_truth[c].append(truth_letter)

    js_per_cluster = {}
    for c in sorted(set(cluster_pred) | set(cluster_truth)):
        pred_dist = pd.Series(cluster_pred[c]).value_counts(normalize=True).to_dict()
        truth_dist = pd.Series(cluster_truth[c]).value_counts(normalize=True).to_dict()
        letters = sorted(set(pred_dist) | set(truth_dist))
        p_vec = np.array([pred_dist.get(l, 0) for l in letters])
        t_vec = np.array([truth_dist.get(l, 0) for l in letters])
        m = 0.5 * (p_vec + t_vec)
        # KL(p||m), KL(t||m) → JS
        def safe_kl(a, b):
            return sum(ai * np.log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)
        js = 0.5 * (safe_kl(p_vec, m) + safe_kl(t_vec, m))
        js_per_cluster[c] = float(js)
    return {"js_per_cluster": js_per_cluster,
             "js_mean": float(np.mean(list(js_per_cluster.values()))) if js_per_cluster else float("nan")}


def compute_a3_individual_l1(preds: list[dict], truths: list[dict],
                                 cluster_labels: dict[str, int]) -> dict:
    """A3 sanity: individual prediction L1 + within-cluster floor baseline."""
    correct = []
    for p, t in zip(preds, truths):
        pred_letter = extract_letter(p["prediction"].get("full_response", ""))
        truth_letter = t.get("answer_truth")
        correct.append(1.0 if pred_letter == truth_letter else 0.0)

    # within-cluster floor: 同簇其他人 majority 当作 prediction 的 accuracy
    cluster_truth_dist: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, t in zip(preds, truths):
        c = cluster_labels.get(p["user_id"])
        truth = t.get("answer_truth")
        if c is not None and truth:
            cluster_truth_dist[c][truth] += 1

    floor_correct = []
    for p, t in zip(preds, truths):
        c = cluster_labels.get(p["user_id"])
        if c is None:
            continue
        dist = cluster_truth_dist[c]
        # 排除当前 designer 自己
        own = t.get("answer_truth")
        adj = {k: (v - (1 if k == own else 0)) for k, v in dist.items()}
        majority = max(adj, key=adj.get) if any(v > 0 for v in adj.values()) else None
        floor_correct.append(1.0 if own == majority else 0.0)

    return {
        "model_accuracy": float(np.mean(correct)) if correct else float("nan"),
        "within_cluster_floor_accuracy": float(np.mean(floor_correct)) if floor_correct else float("nan"),
        "n": len(preds),
    }


def extract_letter(text: str) -> str | None:
    m = re.search(r"\[答案\]\s*([A-E])", text)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--cluster-file", required=True)
    args = parser.parse_args()

    preds = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    truths = [json.loads(l) for l in open(args.ground_truth, encoding="utf-8")]
    assert len(preds) == len(truths), f"长度不匹配: {len(preds)} vs {len(truths)}"

    cluster_df = pd.read_csv(args.cluster_file)
    cluster_labels = dict(zip(cluster_df["user_id"], cluster_df["cluster_id"]))

    print("=== A1 Trajectory Match (主) ===")
    print(compute_a1_trajectory_match(preds, truths))
    print("\n=== A2 Cluster Distribution Match (辅) ===")
    print(compute_a2_cluster_distribution(preds, truths, cluster_labels))
    print("\n=== A3 Individual L1 + Floor (sanity) ===")
    print(compute_a3_individual_l1(preds, truths, cluster_labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
