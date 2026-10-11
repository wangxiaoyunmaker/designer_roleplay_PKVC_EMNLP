"""Build wechat_id <-> user_id mapping via scale completion timestamp.

users.csv 用 wechat_id 标识,task_responses / scale_responses 用数字 user_id。
通过 scale_completed_at (users.csv) 跟每个 user_id 的 scale submitted_at 时间戳精准匹配。
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"D:\Designer_RolePlay\0512技术方案\data")
OUT_PATH = DATA_DIR / "user_id_mapping.csv"


def main() -> int:
    users = pd.read_csv(DATA_DIR / "users.csv")
    scales = pd.read_csv(DATA_DIR / "scale_responses.csv")
    tasks = pd.read_csv(DATA_DIR / "task_responses.csv")

    users["scale_completed_at"] = pd.to_datetime(users["scale_completed_at"])

    # 每个 user_id 取一行(30 题 batch 提交,submitted_at 应该一致)
    scale_first = scales.groupby("user_id").first().reset_index()
    scale_first["submitted_at"] = pd.to_datetime(scale_first["submitted_at"])

    rows = []
    for _, u in users.iterrows():
        diff = (scale_first["submitted_at"] - u["scale_completed_at"]).abs()
        idx = diff.idxmin()
        matched_uid = scale_first.loc[idx, "user_id"]
        matched_time = scale_first.loc[idx, "submitted_at"]
        rows.append({
            "wechat_id": u["wechat_id"],
            "user_id": matched_uid,
            "wechat_scale_completed_at": u["scale_completed_at"],
            "numeric_scale_submitted_at": matched_time,
            "time_diff_seconds": abs((matched_time - u["scale_completed_at"]).total_seconds()),
        })

    mapping = pd.DataFrame(rows)

    print(f"Total mappings built: {len(mapping)}")
    print(f"Unique wechat_id: {mapping['wechat_id'].nunique()}")
    print(f"Unique user_id:   {mapping['user_id'].nunique()}")
    print(f"Time diff (s) — max: {mapping['time_diff_seconds'].max():.3f}, "
          f"mean: {mapping['time_diff_seconds'].mean():.3f}, "
          f"median: {mapping['time_diff_seconds'].median():.3f}")

    dup_uid = mapping["user_id"].duplicated().sum()
    dup_wechat = mapping["wechat_id"].duplicated().sum()
    print(f"Duplicate user_id mapping: {dup_uid}")
    print(f"Duplicate wechat_id:      {dup_wechat}")

    task_uids = set(tasks["user_id"].unique())
    mapped_uids = set(mapping["user_id"])
    overlap = task_uids & mapped_uids
    only_task = task_uids - mapped_uids
    print(f"\nMapped user_ids also in task_responses:   {len(overlap)} / 79 expected")
    print(f"Task user_ids without mapping (no scale): {len(only_task)}")

    mapping[["wechat_id", "user_id", "time_diff_seconds"]].to_csv(OUT_PATH, index=False)
    print(f"\nSaved mapping to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
