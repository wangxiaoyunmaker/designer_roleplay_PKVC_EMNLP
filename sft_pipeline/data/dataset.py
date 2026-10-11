"""PyTorch Dataset + batch sampler (v3.7: cluster → trait-stratified).

设计要点:
  - SFTDataset: 从 jsonl 加载, tokenize, mask prompt 部分 loss
  - TraitStratifiedSampler (v3.7 default): 按 factor F1-F4 分位数 stratify batch,
    满足 L_cc CAPS-connection 多样性约束 (替代旧 ClusterBalancedSampler)
  - ClusterBalancedSampler (legacy, deprecated): cluster_id × N — v3.7 起 cluster
    弱区分实证 (datav3 L3-multi/GMM/visual), 仅保留兼容旧 cluster_file 路径
  - 双方向共用此 dataset (direction 是字段)

v3.7 patch: v3.6 → v3.7 sampler 升级理由 见 docs/briefs/v3.7_patch_post_datav3_20260524.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """从 jsonl 加载 SFT samples, tokenize, loss mask only on output."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        input_text = s["input_text"]
        output_text = s["output_text"]

        # tokenize prompt
        prompt_ids = self.tokenizer(input_text + "\n", add_special_tokens=False).input_ids

        # β+++: 拆 output 三段 (PKVC 分析 / 决策签名 / 答案) 分别 tokenize
        # 给每段 token 标 segment_id (0/1/2) 供三轨 loss 加权
        pkvc_end_str = "[/PKVC 分析]\n\n"
        dec_end_str = "[/决策签名]\n\n"

        if pkvc_end_str in output_text:
            sec1_end = output_text.index(pkvc_end_str) + len(pkvc_end_str)
        else:
            sec1_end = len(output_text)  # legacy fallback (无三段, 整段 segment=0)
        if dec_end_str in output_text:
            sec2_end = output_text.index(dec_end_str) + len(dec_end_str)
        else:
            sec2_end = sec1_end

        sec1_text = output_text[:sec1_end]                # [PKVC 分析] 段
        sec2_text = output_text[sec1_end:sec2_end]        # [决策签名] 段
        sec3_text = output_text[sec2_end:] + self.tokenizer.eos_token  # [答案] + eos

        sec1_ids = self.tokenizer(sec1_text, add_special_tokens=False).input_ids if sec1_text else []
        sec2_ids = self.tokenizer(sec2_text, add_special_tokens=False).input_ids if sec2_text else []
        sec3_ids = self.tokenizer(sec3_text, add_special_tokens=False).input_ids if sec3_text else []

        full_ids = prompt_ids + sec1_ids + sec2_ids + sec3_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(sec1_ids) + list(sec2_ids) + list(sec3_ids)
        # segment_id: -1 = prompt (no loss), 0 = PKVC 分析, 1 = 决策签名, 2 = 答案
        segment_ids = (
            [-1] * len(prompt_ids)
            + [0] * len(sec1_ids)
            + [1] * len(sec2_ids)
            + [2] * len(sec3_ids)
        )

        # truncate from front (keep response 段)
        if len(full_ids) > self.max_length:
            keep = self.max_length
            full_ids = full_ids[-keep:]
            labels = labels[-keep:]
            segment_ids = segment_ids[-keep:]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),  # β+++ 三轨 loss
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "cluster_id": s.get("cluster_id", -1),
            "factor_scores": s.get("factor_scores", [0.0] * 4),
            "factor_quartile": s.get("factor_quartile", -1),
            "task_id": s["task_id"],
            "direction": s["direction"],
            "user_id": s["user_id"],
        }


class TraitStratifiedSampler(Sampler[list[int]]):
    """v3.7 default: 每 batch 按 factor_quartile (F1 4 分位) stratify, 满足 L_cc
    CAPS connection 多样性约束 (替代旧 cluster-based 多样性).

    rationale (datav3 L3): cluster 弱区分 + trait 连续, 用 factor 分位代替 cluster id
    保证 batch 内 trait 谱系覆盖.

    Args:
      dataset: SFTDataset, sample 需要含 'factor_quartile' (0..n_strata-1)
      batch_size: total batch size (= n_strata × samples_per_stratum)
      n_strata: F1 分位数, 默认 4 (quartile)
      samples_per_stratum: 每 stratum 内取多少 sample
    """

    def __init__(self, dataset: SFTDataset, batch_size: int,
                 n_strata: int = 4, samples_per_stratum: int | None = None,
                 seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_strata = n_strata
        if samples_per_stratum is None:
            assert batch_size % n_strata == 0, \
                f"batch_size {batch_size} 不能被 n_strata {n_strata} 整除"
            samples_per_stratum = batch_size // n_strata
        self.sps = samples_per_stratum
        assert n_strata * samples_per_stratum == batch_size, \
            f"batch_size {batch_size} = {n_strata} stratum × {samples_per_stratum} sample"

        self.stratum_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            q = s.get("factor_quartile", -1)
            if q < 0:
                # 兼容旧 jsonl: 没 factor_quartile 字段则用 user_id hash 假分位
                q = hash(s["user_id"]) % n_strata
            self.stratum_to_indices[int(q)].append(i)

        observed = len(self.stratum_to_indices)
        if observed != n_strata:
            import warnings
            warnings.warn(
                f"TraitStratifiedSampler: declared n_strata={n_strata}, 数据中只有 "
                f"{observed} 个 unique quartile — 检查 factor_scores upstream",
                stacklevel=2,
            )
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[list[int]]:
        shuffled = {}
        for q, idxs in self.stratum_to_indices.items():
            perm = torch.randperm(len(idxs), generator=self.generator).tolist()
            shuffled[q] = [idxs[i] for i in perm]
        n_batches = min(len(v) for v in shuffled.values()) // self.sps
        for b in range(n_batches):
            batch = []
            for q in sorted(shuffled.keys()):
                batch.extend(shuffled[q][b * self.sps:(b + 1) * self.sps])
            yield batch

    def __len__(self) -> int:
        return min(len(v) for v in self.stratum_to_indices.values()) // self.sps


class ClusterBalancedSampler(Sampler[list[int]]):
    """[DEPRECATED v3.7] 每 batch 包含每 cluster N 样本.

    v3.7 起改用 TraitStratifiedSampler (cluster 弱区分实证, 见 v3.7 patch).
    保留兼容 v3.5.1 训练流, 但新实验不应使用.
    """

    def __init__(self, dataset: SFTDataset, batch_size: int, n_clusters: int,
                 samples_per_cluster: int, seed: int = 42):
        """Caller MUST resolve n_clusters from the actual cluster_file (brief
        v3.5.1: K may be 6 on 79-pilot, ? after 200-EFA). No default — passing
        n_clusters=5 silently used to be wrong for K=6 even on the 79 cohort.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.spc = samples_per_cluster
        assert n_clusters * samples_per_cluster == batch_size, \
            f"batch_size {batch_size} = {n_clusters} cluster × {samples_per_cluster} sample"
        self.cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self.cluster_to_indices[s["cluster_id"]].append(i)
        # Sanity: warn if observed unique cluster count diverges from declared n_clusters
        observed = len(self.cluster_to_indices)
        if observed != n_clusters:
            import warnings
            warnings.warn(
                f"ClusterBalancedSampler: declared n_clusters={n_clusters} but data "
                f"contains {observed} unique cluster_ids — check upstream EFA / "
                f"cluster_file consistency",
                stacklevel=2,
            )
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[list[int]]:
        # 每 epoch 随机 shuffle 每个 cluster 内的 sample order
        shuffled = {}
        for c, idxs in self.cluster_to_indices.items():
            perm = torch.randperm(len(idxs), generator=self.generator).tolist()
            shuffled[c] = [idxs[i] for i in perm]
        # 最短 cluster 决定 epoch 内 batch 数
        n_batches = min(len(v) for v in shuffled.values()) // self.spc
        for b in range(n_batches):
            batch = []
            for c in sorted(shuffled.keys()):
                batch.extend(shuffled[c][b * self.spc:(b + 1) * self.spc])
            yield batch

    def __len__(self) -> int:
        return min(len(v) for v in self.cluster_to_indices.values()) // self.spc


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad input_ids / labels / segment_ids to longest in batch, stack metadata."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    segment_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)  # β+++ pad=-1
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        if "segment_ids" in b:
            segment_ids[i, :L] = b["segment_ids"]
        attention_mask[i, :L] = b["attention_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "segment_ids": segment_ids,  # β+++ 三轨 loss
        "attention_mask": attention_mask,
        # v3.7: 新加 factor_scores [B, 4] + factor_quartile [B] (CAPS-conditioning input)
        "factor_scores": torch.tensor(
            [b["factor_scores"] for b in batch], dtype=torch.float32,
        ),
        "factor_quartiles": torch.tensor(
            [b["factor_quartile"] for b in batch], dtype=torch.long,
        ),
        # [DEPRECATED v3.7] cluster_ids — 保留兼容旧 collator caller
        "cluster_ids": torch.tensor([b["cluster_id"] for b in batch], dtype=torch.long),
        "task_ids": [b["task_id"] for b in batch],
        "directions": [b["direction"] for b in batch],
        "user_ids": [b["user_id"] for b in batch],
    }
