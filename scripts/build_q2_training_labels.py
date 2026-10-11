"""build_q2_training_labels.py

Q2 weighing function 训练数据 build (Path D, user 2026-05-25 拍板).

Path D 决策:
  - [认知路由标签] 段 = user raw partial+dup cardId ranking 的简洁 verbose
                     (保 data fidelity, 不 dedup, 不 collapse 到 factor)
  - [NL 回答] 段     = q2_nl_refined.jsonl 已 LLM-refine 的 100-300 字解释
  - validation_ok=True 用户进 primary 池 (115/163)
  - validation_ok=False 用户进 fallback 池 (48/163), sampler 默认不取

输入:
  data/processed/q2_nl_refined.jsonl    (163 user, refined NL + raw ranking)

输出:
  data_v3/processed/q2_training_labels.jsonl   (163 user, 训练 ready)

每行 schema:
  {
    "user_id":              "-DevolS",
    "qid":                  "Q2",
    "function":             "weighing",
    "raw_cardids":          ["C1","P1","P2","V2","V3","K3","K2","C3","K1"],   # user real
    "ranking_verbose":      "C1 涌现画面 > P1 界定核心 > ...",                  # 简洁 verbose
    "ranking_verbose_long": "C1 脑海中快速涌现出一个模糊但...",                   # 长 verbose (alt, jsonl 原版)
    "refined_nl":           "做这个项目时, 虽然是我没碰过的高端领域...",
    "n_cards":              9,           # raw 长度
    "n_unique":             9,           # dedup 后
    "has_dup":              false,
    "quality_tier":         "primary" | "fallback",
    "validation_ok":        true | false,
    "validation_fail_reason": "top-3 cardId 覆盖不足: 仅 0/3"  # 仅 fallback
  }
"""

from __future__ import annotations

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# framework v1 §0.1 PKVC SUBTYPE verbose (简洁版, 短 label)
# ─────────────────────────────────────────────────────────────────────

PKVC_SUBTYPE_VERBOSE = {
    "P1": "P1 界定核心",
    "P2": "P2 拆解子问题",
    "P3": "P3 推翻假定",
    "K1": "K1 调用经验",
    "K2": "K2 客观数据测试",
    "K3": "K3 评估技术栈",
    "V1": "V1 价值立场",
    "V2": "V2 利益排序",
    "V3": "V3 设立原则",
    "C1": "C1 涌现画面",
    "C2": "C2 整合构建",
    "C3": "C3 提炼原则",
}


def cardids_to_concise_verbose(cards: list[str]) -> str:
    """[P1, V3, K2] → 'P1 界定核心 > V3 设立原则 > K2 客观数据测试'"""
    return " > ".join(PKVC_SUBTYPE_VERBOSE[c] for c in cards if c in PKVC_SUBTYPE_VERBOSE)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    root = Path(__file__).resolve().parent.parent
    src = root / "data" / "processed" / "q2_nl_refined.jsonl"
    out = root / "data_v3" / "processed" / "q2_training_labels.jsonl"

    if not src.exists():
        raise FileNotFoundError(f"missing {src}")

    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cards = d["ranking_cardids"]
            valid = d.get("validation", {}) or {}
            ok = bool(valid.get("ok", False))

            row = {
                "user_id": d["user_id"],
                "qid": "Q2",
                "function": "weighing",
                "raw_cardids": cards,
                "ranking_verbose": cardids_to_concise_verbose(cards),
                "ranking_verbose_long": d.get("ranking_verbose", ""),
                "refined_nl": d["refined_nl"],
                "n_cards": len(cards),
                "n_unique": len(set(cards)),
                "has_dup": len(cards) != len(set(cards)),
                "quality_tier": "primary" if ok else "fallback",
                "validation_ok": ok,
                "validation_fail_reason": valid.get("last_fail") if not ok else None,
            }
            rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    n = len(rows)
    n_primary = sum(1 for r in rows if r["quality_tier"] == "primary")
    n_fallback = n - n_primary
    n_dup = sum(1 for r in rows if r["has_dup"])
    lens = sorted([r["n_cards"] for r in rows])
    nl_lens = [len(r["refined_nl"]) for r in rows]

    print(f"[OK] wrote {n} rows → {out}")
    print(f"     primary  (validation ok=True):  {n_primary}/{n}  ({n_primary/n*100:.1f}%)")
    print(f"     fallback (validation ok=False): {n_fallback}/{n}")
    print(f"     has_dup (raw 有重复 cardId):     {n_dup}/{n}  ({n_dup/n*100:.1f}%)")
    print(f"     n_cards: min={lens[0]}  median={lens[n//2]}  max={lens[-1]}  mean={sum(lens)/n:.1f}")
    print(f"     refined_nl 长度: min={min(nl_lens)}  median={sorted(nl_lens)[n//2]}  max={max(nl_lens)}  mean={sum(nl_lens)/n:.0f}")
    # sanity: 看一条 primary
    primary = [r for r in rows if r["quality_tier"] == "primary"]
    if primary:
        s = primary[0]
        print()
        print(f"     === primary sample ({s['user_id']}, n_cards={s['n_cards']}, has_dup={s['has_dup']}) ===")
        print(f"     ranking_verbose: {s['ranking_verbose']}")
        print(f"     refined_nl[:120]: {s['refined_nl'][:120]}")


if __name__ == "__main__":
    main()
