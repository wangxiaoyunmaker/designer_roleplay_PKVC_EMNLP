"""Smoke test (CPU): tokenizer/config load + mock data → Dataset → Sampler → collate.

不依赖 GPU, 不需要真实 yaml 标注. 用于验证 SFT pipeline 工程是否跑通.

跑法:
  .venv/bin/python sft_pipeline/test_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader

# 同目录 import
sys.path.insert(0, str(Path(__file__).parent))
from sft_pipeline.data.dataset import SFTDataset, ClusterBalancedSampler, collate_fn


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
CLUSTER_CSV = Path("/data/wy/Designer_RolePlay/data/user_cluster_efa.csv")


def test_tokenizer_and_config_load(model_path: str) -> None:
    print(f"\n=== (a) Tokenizer + Model Config load ({model_path}) ===")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"  tokenizer class: {tok.__class__.__name__}")
    print(f"  vocab_size:      {tok.vocab_size}")
    print(f"  eos / pad:       {tok.eos_token} / {tok.pad_token}")
    print(f"  model arch:      {cfg.architectures}")
    print(f"  hidden_size:     {cfg.hidden_size}")
    print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
    print(f"  num_attention_heads: {cfg.num_attention_heads}")
    print(f"  max_position_embeddings: {cfg.max_position_embeddings}")
    return tok


def make_mock_jsonl(out_path: Path, cluster_df: pd.DataFrame, samples_per_user: int = 3) -> int:
    """每个 user 生成 samples_per_user 个 mock sample, 模拟方向一 B1/B2/B3."""
    rows = []
    for _, r in cluster_df.iterrows():
        for seg_idx, seg in enumerate(["B1", "B2", "B3"][:samples_per_user]):
            rows.append({
                "user_id": str(r["user_id"]),
                "cluster_id": int(r["cluster_id"]),
                "direction": "q7",
                "task_id": seg,
                "fold": 0,
                "input_text": f"[PLACEHOLDER PROMPT for {r['user_id']} {seg}]\n"
                              f"[Profile] 强+: 社区导向, 人本主义\n弱+: 服务温度\n"
                              f"[Scenario] 老城区超市重构 + 题干\n<|response|>",
                "output_text": f"[{seg}] 主导因子: V | sadr: #13-, #14- | "
                               f"推理: <placeholder NL segment for {r['user_id']}>",
                "meta": {"factor_truth": "V", "sadr_truth": ["#13-", "#14-"]},
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def test_dataset_pipeline(tokenizer, jsonl_path: Path, batch_size: int = 20,
                           n_clusters: int = 5) -> None:
    print(f"\n=== (b) Dataset + ClusterBalancedSampler + collate ===")
    ds = SFTDataset(jsonl_path, tokenizer, max_length=2048)
    print(f"  Dataset size: {len(ds)}")
    print(f"  Per-cluster sample count:")
    cluster_counts = {}
    for s in ds.samples:
        cluster_counts[s["cluster_id"]] = cluster_counts.get(s["cluster_id"], 0) + 1
    for c in sorted(cluster_counts):
        print(f"    cluster {c}: {cluster_counts[c]}")

    samples_per_cluster = batch_size // n_clusters
    sampler = ClusterBalancedSampler(
        ds, batch_size=batch_size, n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster, seed=42,
    )
    print(f"  Sampler: {batch_size} per batch ({n_clusters} cluster × {samples_per_cluster})")
    print(f"  Batches per epoch: {len(sampler)}")

    loader = DataLoader(ds, batch_sampler=sampler,
                         collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))

    batch = next(iter(loader))
    print(f"\n  First batch:")
    print(f"    input_ids shape:      {tuple(batch['input_ids'].shape)}")
    print(f"    labels shape:         {tuple(batch['labels'].shape)}")
    print(f"    attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"    cluster_ids:          {batch['cluster_ids'].tolist()}")
    print(f"    task_ids:             {batch['task_ids']}")
    print(f"    directions:           {set(batch['directions'])}")

    # 验证 batch 内 cluster 平衡
    import collections
    cd = collections.Counter(batch["cluster_ids"].tolist())
    print(f"\n  Cluster balance in batch: {dict(cd)}")
    assert len(cd) == n_clusters, f"批内 cluster 数 {len(cd)} ≠ {n_clusters}"
    assert all(v == samples_per_cluster for v in cd.values()), "批内 cluster 不平衡"
    print(f"  ✓ Sampler 正确每 cluster {samples_per_cluster} 个样本")

    # 验证 labels mask 正确 (prompt 部分应该是 -100, response 部分是 token id)
    sample0_labels = batch["labels"][0].tolist()
    n_ignored = sum(1 for x in sample0_labels if x == -100)
    n_response = sum(1 for x in sample0_labels if x != -100)
    print(f"\n  Loss mask in sample 0: {n_ignored} ignored + {n_response} response tokens")
    assert n_ignored > 0 and n_response > 0, "labels mask 出错"
    print(f"  ✓ Loss mask 正确")


def main() -> int:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg["base_model"]["name"]

    # (a) Tokenizer + Config
    tok = test_tokenizer_and_config_load(model_path)

    # (b) Mock data + Dataset pipeline
    print(f"\n=== Mock data 准备 ===")
    cluster_df = pd.read_csv(CLUSTER_CSV)
    cluster_df.columns = [c.strip().lstrip("﻿") for c in cluster_df.columns]
    print(f"  Cluster CSV: {len(cluster_df)} users, "
          f"clusters = {sorted(cluster_df['cluster_id'].unique().tolist())}")

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        mock_path = Path(f.name)
    n_rows = make_mock_jsonl(mock_path, cluster_df, samples_per_user=3)
    print(f"  Mock jsonl: {n_rows} samples → {mock_path}")

    # ⚠ 当前 user_cluster_efa.csv 是 K=6 (EFA best silhouette),
    # brief v3.4 锁 K=5, 后续要 align (重跑 EFA force K=5 or 更新 brief)
    actual_n_clusters = cluster_df["cluster_id"].nunique()
    spc = 4
    test_dataset_pipeline(tok, mock_path, batch_size=actual_n_clusters * spc,
                            n_clusters=actual_n_clusters)

    print("\n=== ALL SMOKE TESTS PASSED ✓ ===")
    print("  pipeline 工程跑通, 等真实数据加工 + GPU 释放即可启动训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
