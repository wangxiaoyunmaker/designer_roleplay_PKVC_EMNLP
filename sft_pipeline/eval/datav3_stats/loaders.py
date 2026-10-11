"""data_v3 standard loaders: users / scale_wide / task_long / expansion + metadata.

每个 layer 共用这些 DataFrame, 不再自己解析 CSV.
Metadata loader (item polarity + Q1-Q6 option PKVC) 从 docs/notes/ 直接解析 markdown.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_V3 = _REPO_ROOT / "data_v3"

_REPORTS_DIR = _REPO_ROOT / "reports" / "datav3"
_DOCS_NOTES = _REPO_ROOT / "docs" / "notes"
_EFA_SWEEP_KS = (3, 4, 5, 6)


def _resolve_root(root: Path | str | None) -> Path:
    if root is None:
        return DEFAULT_DATA_V3
    return Path(root)


def _strip_bom(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def load_users(root: Path | str | None = None) -> pd.DataFrame:
    """users.csv → DataFrame, wechat_id 是 user 主键."""
    root = _resolve_root(root)
    df = pd.read_csv(root / "users.csv")
    df = _strip_bom(df)
    if "wechat_id" not in df.columns:
        raise ValueError(f"users.csv missing wechat_id col, got {df.columns.tolist()}")
    return df


def load_scale_wide(root: Path | str | None = None) -> pd.DataFrame:
    """scale_responses.csv (long) → wide (user_id × 30 量表题).

    Index = user_id, columns 按 question_node 字典序.
    """
    root = _resolve_root(root)
    df = pd.read_csv(root / "scale_responses.csv")
    df = _strip_bom(df)
    wide = df.pivot_table(
        index="user_id",
        columns="question_node",
        values="score",
        aggfunc="first",
    )
    wide = wide.reindex(sorted(wide.columns), axis=1)
    return wide


def load_task_long(
    root: Path | str | None = None,
    expanded: bool = True,
) -> pd.DataFrame:
    """task_responses[_expanded].csv → long DataFrame."""
    root = _resolve_root(root)
    name = "task_responses_expanded.csv" if expanded else "task_responses.csv"
    df = pd.read_csv(root / name)
    df = _strip_bom(df)
    return df


def load_expansion(root: Path | str | None = None) -> pd.DataFrame:
    """q1q6_expansion_outputs.jsonl → DataFrame.

    展平 result.{inferred_factor, inferred_subtype, expanded_nl, confidence} 到顶层列,
    方便下游统计.
    """
    root = _resolve_root(root)
    path = root / "processed" / "q1q6_expansion_outputs.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    # flatten result dict
    if "result" in df.columns:
        for k in ["inferred_factor", "inferred_subtype", "expanded_nl", "confidence"]:
            df[k] = df["result"].apply(lambda r: r.get(k) if isinstance(r, dict) else None)
    # flatten validation
    if "validation" in df.columns:
        df["validation_ok"] = df["validation"].apply(lambda v: v.get("ok", False) if isinstance(v, dict) else False)
    return df


# ---------------------------------------------------------------------
# Answer canonicalization helpers
# ---------------------------------------------------------------------

_OPTION_LETTER_RE = None  # lazy compile

def canonicalize_option(answer_value: str, question_id: str) -> str | None:
    """把 task answer_value 规范化成单字母选项 "A"/"B"/"C"/"D" 或 None.

    适用 Q1, Q3-Q6, Q8 单选题 (前缀 "A. text" 形式; 含双引号包裹).
    Q2 是排序题, 返回 None (调用方应跳过).
    Q7 是 NL, 返回 None.
    """
    if not isinstance(answer_value, str):
        return None
    if question_id in {"Q2", "Q7"}:
        return None
    # 去掉前导引号 + 空白
    s = answer_value.lstrip('"\'').lstrip()
    if len(s) >= 2 and s[0] in "ABCDE" and s[1] in ".、 ":
        return s[0]
    return None


def parse_q2_ranking(answer_value: str) -> list[str]:
    """Q2 cardId 排序 → ordered list (按 'order' 字段, 升序).

    返回 ['P1', 'P2', 'K1', ...]; 解析失败返回 []."""
    if not isinstance(answer_value, str):
        return []
    try:
        items = json.loads(answer_value)
        if not isinstance(items, list):
            return []
        ordered = sorted(items, key=lambda x: x.get("order", 999))
        return [x["cardId"] for x in ordered if "cardId" in x]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


# ---------------------------------------------------------------------
# Metadata loaders — parse markdown in docs/notes/
# ---------------------------------------------------------------------

def load_item_polarity() -> dict[str, dict]:
    """Parse docs/notes/PKVC_30节点本体.md §1 table → {item_name: {plus, minus, node_id, scale_q}}.

    +/- 极性 stance label (5-6 分 = +, 1-2 分 = -, 3-4 = neutral 不激活).
    返回 dict 主键 = 30 题节点名 (跟 scale_responses.question_node 一致).
    """
    path = _DOCS_NOTES / "PKVC_30节点本体.md"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ID |"):
            in_table = True
            continue
        if in_table and (line.startswith("|---") or line.startswith("| ---")):
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 5:
                continue
            node_id, item_name, plus_label, minus_label, scale_q = parts[:5]
            if not item_name or item_name == "节点名":
                continue
            out[item_name] = {
                "node_id": node_id,
                "plus_label": plus_label,
                "minus_label": minus_label,
                "scale_q": scale_q,
            }
    return out


def load_q1q6_option_pkvc() -> dict[str, dict]:
    """Parse docs/notes/Q1-Q6每个选项的元数据.md → {qid: {option: {main_factor, chain}}}.

    main_factor: P / K / V / C (chain 里的第一个字母, 主导因子)
    chain: e.g. 'p3➡️k3' 或 None (纯主导单选时)
    """
    path = _DOCS_NOTES / "Q1-Q6每个选项的元数据.md"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    current_qid: str | None = None
    qid_pattern = re.compile(r"\*\*(Q\d)[:：]")
    option_pattern = re.compile(r"^([ABCD])[.、]")
    # 多种 marker 格式 (兼容 Q1 / Q3-Q6 / Q5 各自的标注风格)
    chain_pattern = re.compile(r"——\s*([pkvcPKVC][\d]?(?:\s*➡️\s*[pkvcPKVC][\d]?)?)")
    single_factor_pattern = re.compile(r"——\s*([pkvcPKVC])")
    # Q5 风格: 选项末尾 `**K2: [客观数据测试]**`
    bold_factor_pattern = re.compile(r"\*\*([PKVC])(\d)?[:：]")

    for line in path.read_text(encoding="utf-8").splitlines():
        m = qid_pattern.search(line)
        if m:
            current_qid = m.group(1)
            out.setdefault(current_qid, {})
            continue
        if current_qid is None:
            continue
        opt_m = option_pattern.match(line.lstrip())
        if not opt_m:
            continue
        opt = opt_m.group(1)
        chain_m = chain_pattern.search(line)
        if chain_m:
            chain = chain_m.group(1)
            out[current_qid][opt] = {"main_factor": chain[0].upper(), "chain": chain}
            continue
        sf_m = single_factor_pattern.search(line)
        if sf_m:
            out[current_qid][opt] = {"main_factor": sf_m.group(1).upper(), "chain": None}
            continue
        bf_m = bold_factor_pattern.search(line)
        if bf_m:
            sub = bf_m.group(1) + (bf_m.group(2) or "")
            out[current_qid][opt] = {"main_factor": bf_m.group(1).upper(), "chain": sub.lower()}
    return out


def load_q3_q6_context_c() -> dict[str, dict]:
    """Parse docs/notes/Q1-Q6每个选项的元数据.md → {qid: {C_raw, conflict_axes, C_formal_proposed}}.

    quad doc Step 9 标 "Q3-Q6 4 个 C 值待一作 confirm 正式命名 string",
    但内容 (情境特征 + 冲突 PKVC 标识) 已在项目 metadata 里给.
    本 loader 抽出来, 提供 working 4 个 C, 一作 confirm 正式命名后只需改 dict.
    """
    path = _DOCS_NOTES / "Q1-Q6每个选项的元数据.md"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    current_qid: str | None = None
    qid_pat = re.compile(r"\*\*(Q[3-6])[:：]")
    c_pat = re.compile(r"情境特征[：:]\s*(.+?)(?:\*\*|$)")
    axes_pat = re.compile(r"\(([PKVC/\s.vs]+)\)")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = qid_pat.search(line)
        if m:
            current_qid = m.group(1)
            out[current_qid] = {}
            continue
        if current_qid is None:
            continue
        cm = c_pat.search(line)
        if cm and "C_raw" not in out[current_qid]:
            out[current_qid]["C_raw"] = cm.group(1).strip().rstrip("】").rstrip("*").strip()
        am = axes_pat.search(line)
        if am and "conflict_axes" not in out[current_qid]:
            out[current_qid]["conflict_axes"] = am.group(1).strip()
    # Claude 提议的正式命名 (跟 quad doc 'K_ObjectiveData_vs_C_AestheticVision' 风格一致)
    # 一作 confirm 后改这里:
    proposed_formal = {
        "Q3": "K_ObjectiveData_vs_CV_Aesthetic",
        "Q4": "KP_BusinessPrivacy_vs_V_Ethics_vs_C_Construct",
        "Q5": "K_TechScale_vs_C_Organic_vs_V_HumanCentric_vs_P_Core",
        "Q6": "V_UserResponsibility_vs_CPK_Commercial",
    }
    for qid, fn in proposed_formal.items():
        if qid in out:
            out[qid]["C_formal_proposed"] = fn
    return out


def add_canonical_option(task_df: pd.DataFrame) -> pd.DataFrame:
    """在 task_long DataFrame 上加 'option' 列 (canonicalized A/B/C/D or None).

    in-place 安全: 返回新 DataFrame.
    """
    df = task_df.copy()
    df["option"] = df.apply(
        lambda r: canonicalize_option(r["answer_value"], r["question_id"]),
        axis=1,
    )
    return df


def get_default_factor_scores_path(fallback_k: int = 4) -> Path:
    """读 reports/datav3/L2_efa.json 拿 pa_suggested_k, 返回对应 factor_scores 路径.

    PA 推荐超出 sweep 范围 → clamp 到最近的 sweep K.
    L2 未跑 → 回退 fallback_k.
    """
    l2_json = _REPORTS_DIR / "L2_efa.json"
    pa_k: int = fallback_k
    if l2_json.exists():
        try:
            data = json.loads(l2_json.read_text(encoding="utf-8"))
            raw = data.get("pa_suggested_k", fallback_k)
            pa_k = int(raw) if raw else fallback_k
        except (json.JSONDecodeError, ValueError):
            pa_k = fallback_k
    if pa_k not in _EFA_SWEEP_KS:
        pa_k = min(_EFA_SWEEP_KS, key=lambda k: abs(k - pa_k))
    return _REPORTS_DIR / "per_designer" / f"factor_scores_k{pa_k}.csv"
