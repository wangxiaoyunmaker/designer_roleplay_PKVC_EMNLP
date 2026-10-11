"""build_q1q6_training_labels.py

Parse 学姐 Q1-Q6 选项 PKVC metadata (markdown)
  → 结构化 jsonl, 供 framework v1 lock #5 训练 ground truth 用.

Input:
  docs/notes/Q1-Q6每个选项的元数据.md
  data_v3/task_responses.csv      (拿选项文本 ground truth, N=163)

Output:
  data_v3/processed/q1q6_option_labels.jsonl

每行 schema (跟 framework v1 §0.1 verbose 约定对齐):
  {
    "qid":           "Q1" | "Q3" | "Q4" | "Q5" | "Q6",  (Q2 是排序题, 单独 build, 不在本表)
    "option":        "A" | "B" | "C" | "D",
    "option_text":   "我会先在脑中形成产品形态和使用场景...",     (从 task_responses 反查 unique)
    "raw_label":     "C初始激活" / "P3➡️K3" / "V1/V3——v" 等学姐原写法,
    "pkvc_chain":    ["C"] | ["P3","K3"] | ["V1"] | ["V2","P1","K2"] (head -> tail),
    "pkvc_verbose":  "C (Concept 构念/构建)" |
                     "P3 推翻假定 → K3 评估技术栈",
    "is_ambiguous":  true 当学姐写了 "V2/P1➡️K2" 这种 alt 标,
    "alt_labels":    ["V2 利益优先级排序"]  (ambiguous 时 second-best),
    "function":      "first_reaction" (Q1) | "bottom_line" (Q3-Q6)
  }

Q1: single letter only (C / P / K / V), 无 subtype
Q3-Q6: single subtype (e.g. K2) 或 2 段路径 (e.g. P3→K3)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# PKVC verbose 表 (跟 framework v1 §0.1 严格对齐)
# ─────────────────────────────────────────────────────────────────────

PKVC_FACTOR_VERBOSE = {
    "P": "P (Problem 现实与问题重构)",
    "K": "K (Knowledge 经验/数据/技术调用)",
    "V": "V (Value 价值判断)",
    "C": "C (Concept 构念/构建)",
}

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


def pkvc_verbose(token: str) -> str:
    """单字母 / 编号 → verbose 标签."""
    token = token.strip().upper()
    if token in PKVC_FACTOR_VERBOSE:
        return PKVC_FACTOR_VERBOSE[token]
    if token in PKVC_SUBTYPE_VERBOSE:
        return PKVC_SUBTYPE_VERBOSE[token]
    raise ValueError(f"unknown PKVC token: {token!r}")


def chain_verbose(chain: list[str]) -> str:
    """chain ['P3','K3'] → 'P3 推翻假定 → K3 评估技术栈'.
    单元素 ['C'] → 'C (Concept 构念/构建)'."""
    if len(chain) == 1:
        return pkvc_verbose(chain[0])
    return " → ".join(pkvc_verbose(t) for t in chain)


# ─────────────────────────────────────────────────────────────────────
# Hand-coded metadata 提取 (因为学姐 markdown 格式不齐, regex 不稳, 直接转译)
# 来源: docs/notes/Q1-Q6每个选项的元数据.md (2026-05-25 读取)
# ─────────────────────────────────────────────────────────────────────

# 每条:
#   raw_label          (学姐原写)
#   primary_chain      (取 head→tail 主路径, 用作 training target)
#   alt_chains         (学姐写 "V2/P1→K2" 时, V2 单字母 是 alt)
RAW_METADATA: dict[str, dict[str, dict]] = {
    "Q1": {
        "A": {"raw": "C初始激活",      "primary": ["C"], "alts": [], "amb": False},
        "B": {"raw": "P初始激活",      "primary": ["P"], "alts": [], "amb": False},
        "C": {"raw": "K初始激活",      "primary": ["K"], "alts": [], "amb": False},
        "D": {"raw": "V初始激活",      "primary": ["V"], "alts": [], "amb": False},
    },
    "Q3": {
        "A": {"raw": "K2纯导向 — k2",                 "primary": ["K2"],       "alts": [],     "amb": False},
        "B": {"raw": "P3→K3",                          "primary": ["P3", "K3"], "alts": [],     "amb": False},
        "C": {"raw": "V1→C1",                          "primary": ["V1", "C1"], "alts": [],     "amb": False},
        # Q3 D ambiguous: V2 单标 / P1→K2 chain — 拍板取 chain 当 primary, V2 入 alt
        "D": {"raw": "V2 / P1→K2 (学姐双标 ambiguous)", "primary": ["P1", "K2"], "alts": [["V2"]], "amb": True},
    },
    "Q4": {
        "A": {"raw": "K1→P1",  "primary": ["K1", "P1"], "alts": [], "amb": False},
        "B": {"raw": "V3",     "primary": ["V3"],       "alts": [], "amb": False},
        "C": {"raw": "C2",     "primary": ["C2"],       "alts": [], "amb": False},
        "D": {"raw": "K2→P2",  "primary": ["K2", "P2"], "alts": [], "amb": False},
    },
    "Q5": {
        "A": {"raw": "K2",     "primary": ["K2"], "alts": [], "amb": False},
        "B": {"raw": "C2",     "primary": ["C2"], "alts": [], "amb": False},
        "C": {"raw": "V3",     "primary": ["V3"], "alts": [], "amb": False},
        "D": {"raw": "P1",     "primary": ["P1"], "alts": [], "amb": False},
    },
    "Q6": {
        # Q6 A ambiguous: V1 / V3 双单字母 (学姐写 V1/V3——v) — 取 V1 当 primary, V3 入 alt
        "A": {"raw": "V1 / V3 (学姐双标 ambiguous, 合 V)", "primary": ["V1"], "alts": [["V3"]], "amb": True},
        "B": {"raw": "V1→C2", "primary": ["V1", "C2"], "alts": [], "amb": False},
        "C": {"raw": "V2→P1", "primary": ["V2", "P1"], "alts": [], "amb": False},
        "D": {"raw": "K2→V2", "primary": ["K2", "V2"], "alts": [], "amb": False},
    },
}

Q_TO_FUNCTION = {
    "Q1": "first_reaction",
    "Q3": "bottom_line",
    "Q4": "bottom_line",
    "Q5": "bottom_line",
    "Q6": "bottom_line",
}


# ─────────────────────────────────────────────────────────────────────
# 从 task_responses.csv 反查每个 qid+option 的 verbose 文本
# answer_value 形如:  "A. 我会先在脑中形成产品形态..."
# ─────────────────────────────────────────────────────────────────────

def extract_option_texts(csv_path: Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(csv_path)
    bucket: dict[tuple[str, str], set[str]] = {}
    pat = re.compile(r'^"?([A-D])\.\s*(.+?)"?$', re.DOTALL)
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid not in RAW_METADATA:  # 跳 Q2 / Q7 / Q8
            continue
        ans = str(row["answer_value"]).strip()
        m = pat.match(ans)
        if not m:
            continue
        opt, txt = m.group(1), m.group(2).rstrip('"').strip()
        bucket.setdefault((qid, opt), set()).add(txt)

    # 同 qid+option 应该所有 user 一致, sanity 检查 + 取 majority
    out: dict[tuple[str, str], str] = {}
    for (qid, opt), texts in bucket.items():
        if len(texts) > 1:
            # 可能是空白/标点差异, 取最长那条 (info 更完整)
            chosen = max(texts, key=len)
            print(f"[WARN] {qid}-{opt} has {len(texts)} text variants, choosing longest")
        else:
            chosen = next(iter(texts))
        out[(qid, opt)] = chosen
    return out


# ─────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────

def build_rows(option_texts: dict[tuple[str, str], str]) -> list[dict]:
    rows = []
    for qid, opts in RAW_METADATA.items():
        for opt, meta in opts.items():
            text = option_texts.get((qid, opt), "")
            row = {
                "qid": qid,
                "option": opt,
                "option_text": text,
                "raw_label": meta["raw"],
                "pkvc_chain": meta["primary"],
                "pkvc_verbose": chain_verbose(meta["primary"]),
                "is_ambiguous": meta["amb"],
                "alt_labels": [chain_verbose(a) for a in meta["alts"]],
                "function": Q_TO_FUNCTION[qid],
            }
            rows.append(row)
    return rows


def main():
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data_v3" / "task_responses.csv"
    out_path = root / "data_v3" / "processed" / "q1q6_option_labels.jsonl"

    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")

    option_texts = extract_option_texts(csv_path)
    rows = build_rows(option_texts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    n_total = len(rows)
    n_amb = sum(1 for r in rows if r["is_ambiguous"])
    n_chain = sum(1 for r in rows if len(r["pkvc_chain"]) > 1)
    n_single = sum(1 for r in rows if len(r["pkvc_chain"]) == 1)
    missing_text = [(r["qid"], r["option"]) for r in rows if not r["option_text"]]

    print(f"[OK] wrote {n_total} rows → {out_path}")
    print(f"     - single-letter labels:  {n_single}")
    print(f"     - 2-step chain labels:   {n_chain}")
    print(f"     - ambiguous (有 alt):    {n_amb}")
    if missing_text:
        print(f"[WARN] {len(missing_text)} option(s) missing text: {missing_text}")
    else:
        print(f"     - 所有 {n_total} 选项 option_text 已 join 成功")


if __name__ == "__main__":
    main()
