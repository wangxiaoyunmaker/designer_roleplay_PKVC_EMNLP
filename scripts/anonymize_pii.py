"""scripts/anonymize_pii.py — 一次性 wechat_id → D001-DNNN anonymization.

跑法:
    python scripts/anonymize_pii.py [--seed 42]

输入 (data/, gitignored 真人 PII):
    data/users.csv                        79 designer + wechat_id + identity / 教育 / 经验
    data/task_responses.csv               632 行 Q1-Q8 答案 + text_supplement
    data/scale_responses.csv              2370 行 (79 user × 30 量表)
    data/user_cluster_efa.csv             K-means cluster_id
    data/user_factor_scores.csv           EFA factor scores

输出 (data/anonymized/, ⚠ 同样 gitignored 但安全等级 lower):
    data/anonymized/users.csv             wechat_id → D001-D079 anonymized code
    data/anonymized/task_responses.csv    同上
    data/anonymized/scale_responses.csv   同上
    data/anonymized/user_cluster_efa.csv  同上
    data/anonymized/user_factor_scores.csv 同上
    data/anonymized/uid_mapping.csv       D code ↔ wechat_id mapping (⚠ 最高敏感, 只 user / 学姐手里)

设计:
    1. anonymized code 用 sequential D001-D079 (不用 hash, 防 length leak)
    2. order 按 first appearance in users.csv 锁死 (deterministic + reproducible)
    3. mapping 文件单独输出, 不跟 anonymized 数据混在一起
    4. 任意一次跑 produces same mapping (给定 users.csv 不变), 跨 session 一致
    5. text_supplement / Q7 NL 内容 *不* 二次脱敏 (entity removal), 79 数据短 + entity 少, 风险可控
       121 入袋后重审, 如有命名实体 (公司名/产品名) 再加 NER pipeline.

⚠ 跑完 sanity:
    grep -E '(wxid_|^1[0-9]{10}$|^Zz|^Yyd|^MH)' data/anonymized/*.csv
    应该返回 0 行, 否则有漏网 wechat_id
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ANON_DIR = DATA_DIR / "anonymized"

# 主 PII 字段名 (跨 csv 一致)
WECHAT_ID_COLS = ("user_id", "﻿user_id", "wechat_id", "designer_id")

# 示例文档里固定使用的 designer 编号 (md 引用的 sample / peer designer 锁这里).
# 不在此 map 的 designer 按 users.csv 顺序填补剩下 D 编号.
# 改这里之前先 sync docs/training_data_formats/*.md + docs/briefs/T_CAPS_quad_schema_v1.md.
PINNED_MAPPING: dict[str, str] = {
    # 主示例 designer (brief + quad example)
    "1142208878":         "D001",
    # NL 长 designer (示例.md 主)
    "Zz18661092892":      "D002",
    # 旧训练数据 sample peer (user 一开始 quote 的烂训练数据示例)
    "MH632120202":        "D012",
    "wtimesc":            "D034",
    "nexTVTforoseoo_ilo": "D056",
    # 示例.md 新 peer (NL 长 top-5 designer)
    "wxid_higyz7x19rs422": "D023",
    "Flower_foreverrrr":  "D045",
    "Yyd3144117584":      "D058",
}


def load_users(users_csv: Path) -> list[dict]:
    """加载 users.csv, return rows in file order."""
    rows = []
    with users_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_mapping(users: list[dict]) -> dict[str, str]:
    """生成 wechat_id → D001-DNNN mapping.

    策略:
        1. 先应用 PINNED_MAPPING (示例 designer 锁定 D 编号, 跟 md 文档同步)
        2. 剩下 designer 按 users.csv 顺序填入未占用的 D 编号
    """
    mapping: dict[str, str] = {}
    used_codes: set[str] = set()
    wechat_ids_in_users: list[str] = []

    # Pass 1: collect all wechat_ids from users.csv
    for i, row in enumerate(users, start=1):
        wid = (row.get("user_id") or row.get("﻿user_id") or
               row.get("wechat_id") or "").strip()
        if not wid:
            raise ValueError(f"row {i}: 找不到 user_id / wechat_id, row={row}")
        if wid in wechat_ids_in_users:
            raise ValueError(f"row {i}: 重复 wechat_id {wid!r}")
        wechat_ids_in_users.append(wid)

    # Pass 2: apply PINNED_MAPPING (顺序无关, 学姐 / 老师 / md 引用的 designer 锁码)
    for wid, code in PINNED_MAPPING.items():
        if wid not in wechat_ids_in_users:
            print(f"  ⚠ pinned {wid!r}→{code} 不在 users.csv 里, 跳过")
            continue
        if code in used_codes:
            raise ValueError(f"PINNED_MAPPING 内部冲突: 多个 wechat_id 映射到 {code}")
        mapping[wid] = code
        used_codes.add(code)

    # Pass 3: 给剩下 designer 按 users.csv 顺序填补
    free_codes_iter = iter(
        f"D{i:03d}" for i in range(1, len(users) + 1) if f"D{i:03d}" not in used_codes
    )
    for wid in wechat_ids_in_users:
        if wid in mapping:
            continue  # pinned
        try:
            code = next(free_codes_iter)
        except StopIteration:
            raise ValueError(f"D 编号不够分: PINNED_MAPPING 占用太多 + users.csv {len(users)} 人")
        mapping[wid] = code

    return mapping


def anonymize_csv(
    in_path: Path,
    out_path: Path,
    mapping: dict[str, str],
    pii_cols: tuple[str, ...] = WECHAT_ID_COLS,
) -> int:
    """读 in_path, 把 PII 列 (任一 ∈ pii_cols) 的 value 替换成 mapping[value],
    其他列保留. 输出 out_path. 返回替换行数."""
    if not in_path.exists():
        print(f"  ⚠ skip {in_path.name} (不存在)")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_repl = 0
    with in_path.open(encoding="utf-8-sig") as fin, out_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        # 标准化字段名 (去 BOM)
        fieldnames = [f.lstrip("﻿") for f in reader.fieldnames or []]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            new_row = {}
            for k, v in row.items():
                key = k.lstrip("﻿")
                if key in pii_cols and v in mapping:
                    new_row[key] = mapping[v]
                    n_repl += 1
                else:
                    new_row[key] = v
            writer.writerow(new_row)
    return n_repl


def write_mapping_csv(mapping: dict[str, str], out_path: Path) -> None:
    """输出 uid_mapping.csv — ⚠ 最高敏感, .gitignore 必须防."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anonymized_code", "wechat_id"])
        for wid, code in mapping.items():
            w.writerow([code, wid])
    print(f"  ✅ {out_path.name}: {len(mapping)} mapping entries (⚠ 最高敏感, 不进 git)")


def sanity_check(anon_dir: Path) -> bool:
    """跑完 grep 残留 wechat_id 模式."""
    leak_patterns = [
        r"wxid_[a-zA-Z0-9_]+",                 # wxid_xxx
        r"\b1[0-9]{10}\b",                      # 11 位数字 (常见 wechat 数字号)
        r"\bZz[a-zA-Z0-9]+\b",                  # Zz 开头
        r"\bYyd[a-zA-Z0-9]+\b",                 # Yyd 开头
        r"\bMH\d+\b",                            # MH+数字
    ]
    combined = re.compile("|".join(leak_patterns))
    leaks = []
    for csv_file in anon_dir.glob("*.csv"):
        if csv_file.name == "uid_mapping.csv":
            continue  # mapping file 本身含 wechat_id, 跳过
        with csv_file.open(encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if combined.search(line):
                    leaks.append((csv_file.name, ln, line.rstrip()[:80]))
    if leaks:
        print("\n❌ sanity FAIL — 有 wechat_id 残留:")
        for name, ln, content in leaks[:10]:
            print(f"  {name}:{ln}  {content}")
        return False
    print("\n✅ sanity PASS — anonymized/*.csv 无 wechat_id 残留")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=42,
                   help="(unused, 编号按 users.csv 顺序锁死)")
    p.add_argument("--users-csv", type=Path, default=DATA_DIR / "users.csv")
    p.add_argument("--out-dir", type=Path, default=ANON_DIR)
    p.add_argument("--force", action="store_true",
                   help="覆盖已有 anonymized 输出 (默认拒)")
    args = p.parse_args()

    if not args.users_csv.exists():
        print(f"❌ users.csv 不存在: {args.users_csv}")
        return 1

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        print(f"⚠ {args.out_dir} 已有内容, 用 --force 覆盖, 否则停止")
        return 1

    print(f"读 users.csv: {args.users_csv}")
    users = load_users(args.users_csv)
    print(f"  → {len(users)} designer")

    print(f"生成 wechat_id → D001-D{len(users):03d} mapping")
    mapping = build_mapping(users)

    # 输出 anonymized 数据文件
    csv_files = [
        "users.csv",
        "task_responses.csv",
        "scale_responses.csv",
        "user_cluster_efa.csv",
        "user_factor_scores.csv",
        "user_cluster.csv",         # legacy
        "user_cluster_v2.csv",      # legacy
        "user_id_mapping.csv",      # ⚠ 这个是 v3.4 老 mapping 不是 anonymized mapping, 检查后决定
    ]
    print(f"\n输出 anonymized 数据到 {args.out_dir}/")
    total_repl = 0
    for name in csv_files:
        in_p = DATA_DIR / name
        out_p = args.out_dir / name
        n = anonymize_csv(in_p, out_p, mapping)
        if n > 0:
            print(f"  ✅ {name}: 替换 {n} 处 wechat_id")
        total_repl += n
    print(f"\n总共替换 {total_repl} 处")

    # 输出 mapping 文件 (⚠ 最高敏感)
    write_mapping_csv(mapping, args.out_dir / "uid_mapping.csv")

    # sanity check
    ok = sanity_check(args.out_dir)
    if not ok:
        return 1

    # 提示信息
    print("\n下一步:")
    print(f"  1. 把 {args.out_dir.relative_to(ROOT)} 加进 .gitignore (如未加)")
    print(f"  2. uid_mapping.csv 保管严格, 只给学姐 / user / 老师")
    print(f"  3. 给学妹的数据从 {args.out_dir.relative_to(ROOT)} 取, 不要从 data/raw")
    print(f"  4. 训练 pipeline 应该消费 anonymized 数据, raw data/ 不参与训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
