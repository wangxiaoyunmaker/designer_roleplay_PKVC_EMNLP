"""推理输出 IO + 4 档 RAG 模式 dispatch (v3.5.1 §5.5 D-14 / v3.7 §6).

主要职责
--------
1. 加载 ``infer.py`` 输出的 prediction jsonl, 标准化成
   ``[{user_id, task_id, answer, dominant_factor, sadr_activations, ...}]``
   供 ``metrics.compute_a1/a2/a3`` 直接消费.
2. 自动 dispatch 4 档 RAG 模式的预测文件 (按命名约定),
   并兼容 v3.5.1 (og_rag) ↔ v3.7 (trait_similarity) 名字漂移.
3. SFT ckpt 未就位时, 用 mock 生成器 (cluster majority + 小噪声) 产出占位预测,
   保 smoke test + Tier 1 pipeline 离线跑通.
4. 反推 PKVC: 把 ``prediction.full_response`` 喂给 ``extract.llm.extract_pkvc``
   反推 dominant_factor + sadr; 离线 (无 API key) 时用 regex fallback.
5. Phase 0 Test B 用的 trajectory vec builder (sadr 频次 + factor one-hot).

文件命名约定 (per fold per RAG mode)
-----------------------------------
::

    {pred_dir}/{direction}__{rag_mode}__fold{k}.jsonl
    例: preds/q1q6__og_rag+modulation__fold0.jsonl
        preds/q1q6__no_rag__fold0.jsonl

mode 名字 v3.5.1 ↔ v3.7 别名映射在 ``RAG_MODE_ALIASES``.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

logger = logging.getLogger("eval.predictions")


# ---------------------------------------------------------------------------
# RAG mode 规范 + 别名 (v3.5.1 ↔ v3.7)
# ---------------------------------------------------------------------------

#: 4 档 RAG 模式 (v3.5.1 §5.5 brief, 主方向 ablation 标准集)
RAG_MODES_V351: tuple[str, ...] = (
    "no_rag",
    "cosine",
    "og_rag",
    "og_rag+modulation",
)

#: v3.7 等价集 (config.yaml: cluster 弱区分实证后 cosine → trait_similarity)
RAG_MODES_V37: tuple[str, ...] = (
    "no_rag",
    "trait_similarity",
    "og_rag",
    "trait_similarity+modulation",
)

#: 双向别名 → 规范名 (用 v3.5.1 名字作内部规范, 容忍 v3.7 输入)
RAG_MODE_ALIASES: dict[str, str] = {
    "trait_similarity": "cosine",
    "trait_similarity+modulation": "og_rag+modulation",
    "modulation": "og_rag+modulation",   # 单 modulation 罕见, 归入主候选
}


def canonical_rag_mode(name: str) -> str:
    return RAG_MODE_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Prediction record schema (统一传入 metrics)
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """metrics.compute_a1/a2/a3 期望的标准化 record.

    answer / dominant_factor / sadr_activations 是反推后的, 不是原始 generation.
    """
    user_id: str
    task_id: str                         # Q1..Q6 / Q7_seg1..3
    scenario_id: str | None = None       # 同 task_id (compat)
    answer: str | None = None            # A..E 单字母 (MCQ 题)
    dominant_factor: str | None = None   # 'V' or 'V → K → V'
    sadr_activations: list[Any] = field(default_factory=list)
    criterion: str | None = None
    full_response: str = ""              # 原始 generation, 保留供 audit
    rag_mode: str = ""                   # 标记 prediction 来源
    fold: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regex fallback parser (无 API key 时反推 answer / factor / sadr)
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"\[?答案\]?[:：\s]*([A-E])", re.IGNORECASE)
_ANSWER_FALLBACK_RE = re.compile(r"\b([A-E])\.?\s*[、,，。]?")
_FACTOR_RE = re.compile(r"主导因子[:：\s]*([PKVC](?:\s*[→\->]\s*[PKVC])*)")
_SADR_RE = re.compile(r"(#[A-Za-z0-9_一-鿿\-]+[+\-]?)")


def parse_response_regex(full_response: str) -> tuple[str | None, str | None, list[str]]:
    """离线反推 (answer_letter, dominant_factor, sadr_list).

    用于 (1) 无 API key 离线 smoke, (2) extractor 失败时 fallback.
    严格 < extractor LLM, 但能保 pipeline 跑通.
    """
    if not full_response:
        return None, None, []
    ans_m = _ANSWER_RE.search(full_response)
    answer = ans_m.group(1).upper() if ans_m else None
    if answer is None:
        fb = _ANSWER_FALLBACK_RE.search(full_response)
        answer = fb.group(1).upper() if fb else None

    fac_m = _FACTOR_RE.search(full_response)
    factor = fac_m.group(1).replace(" ", "").replace("->", "→") if fac_m else None

    sadrs = list(dict.fromkeys(_SADR_RE.findall(full_response)))  # dedup, preserve order
    return answer, factor, sadrs


# ---------------------------------------------------------------------------
# Predictions IO
# ---------------------------------------------------------------------------

def load_predictions_jsonl(
    path: Path,
    rag_mode: str = "",
    use_extractor: bool = False,
    extractor_model: str = "gpt-4o",
) -> list[PredictionRecord]:
    """读 infer.py 输出的 jsonl, 反推 PKVC 字段.

    Args:
      path: jsonl 路径. 每行 ``{user_id, task_id, prediction: {full_response}, meta?}``.
      rag_mode: 标注本批次的 RAG 模式 (写入 record.rag_mode).
      use_extractor: True 用 ``extract.llm.extract_pkvc`` (要 API key); False
                     用 regex fallback. 默认 False 保 offline 可跑.
      extractor_model: extractor 模型名, 仅 use_extractor=True 时用.
    """
    if not path.exists():
        logger.warning("predictions file missing: %s", path)
        return []
    out: list[PredictionRecord] = []
    extractor = None
    if use_extractor:
        try:
            from sft_pipeline.extract.llm import extract_pkvc
            extractor = extract_pkvc
        except Exception as e:  # noqa: BLE001
            logger.warning("extractor import failed (%s); fall back to regex", e)
            extractor = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("user_id", ""))
            tid = str(row.get("task_id") or row.get("scenario_id") or "")
            pred = row.get("prediction") or {}
            full = pred.get("full_response") or pred.get("text") or ""

            answer, factor, sadrs = parse_response_regex(full)
            sadr_objs: list[Any] = sadrs

            # extractor 优先 (更准, 给 dict-form sadr_activations)
            if extractor is not None and full:
                try:
                    er = extractor(full, model=extractor_model)
                    if er.dominant_factor:
                        factor = er.dominant_factor
                    if er.sadr_activations:
                        sadr_objs = er.sadr_activations
                except Exception as e:  # noqa: BLE001
                    logger.debug("extractor failed on uid=%s task=%s: %s", uid, tid, e)

            out.append(PredictionRecord(
                user_id=uid,
                task_id=tid,
                scenario_id=tid,
                answer=answer,
                dominant_factor=factor,
                sadr_activations=sadr_objs,
                full_response=full,
                rag_mode=rag_mode,
                fold=row.get("fold"),
            ))
    logger.info("Loaded %d preds from %s (mode=%s, extractor=%s)",
                len(out), path.name, rag_mode, bool(extractor))
    return out


def discover_predictions(
    pred_dir: Path,
    direction: str,
    rag_modes: Iterable[str],
    n_folds: int = 5,
) -> dict[str, list[Path]]:
    """按命名约定 ``{dir}/{direction}__{mode}__fold{k}.jsonl`` 找文件.

    Returns ``{rag_mode: [fold0_path, fold1_path, ...]}``; 缺失文件返回空 list.
    """
    out: dict[str, list[Path]] = {}
    for mode in rag_modes:
        canon = canonical_rag_mode(mode)
        paths: list[Path] = []
        for k in range(n_folds):
            # 接受 canonical + alias 两种文件名
            candidates = [
                pred_dir / f"{direction}__{canon}__fold{k}.jsonl",
                pred_dir / f"{direction}__{mode}__fold{k}.jsonl",
            ]
            found = next((c for c in candidates if c.exists()), None)
            if found:
                paths.append(found)
        out[canon] = paths
    return out


def load_predictions_for_mode(
    pred_dir: Path,
    direction: str,
    rag_mode: str,
    n_folds: int = 5,
    use_extractor: bool = False,
    extractor_model: str = "gpt-4o",
) -> list[PredictionRecord]:
    """聚合所有 fold 的某个 RAG mode 预测."""
    discovered = discover_predictions(pred_dir, direction, [rag_mode], n_folds)
    paths = discovered[canonical_rag_mode(rag_mode)]
    all_preds: list[PredictionRecord] = []
    for p in paths:
        all_preds.extend(load_predictions_jsonl(p, rag_mode, use_extractor, extractor_model))
    return all_preds


# ---------------------------------------------------------------------------
# Mock predictions (SFT ckpt 没就位 / smoke test 用)
# ---------------------------------------------------------------------------

def build_mock_predictions(
    user_cluster: dict[str, int],
    truth_records: list[dict[str, Any]],
    rag_mode: str = "mock",
    factor_pool: tuple[str, ...] = ("P", "K", "V", "C"),
    seed: int = 42,
    flip_rate: float = 0.25,
) -> list[PredictionRecord]:
    """生成离线 mock predictions: 同簇 (cluster, task) majority answer + flip 噪声.

    用途:
      - smoke test: 无需 SFT ckpt 就能跑通 A1/A2/A3
      - dry-run report: 看 markdown 表渲染对不对
      - lower-bound sanity: mock 表现接近 B3 (按定义), 投稿不用

    Args:
      user_cluster: user_id → cluster_id
      truth_records: ``[{user_id, task_id, answer}]``; 提供 (cluster, task)
                      majority 学习信号. 没有时退化为 random.
      flip_rate: per-record flip 到非 majority 的概率, 用来跟 B3 拉开差距.
    """
    rng = np.random.default_rng(seed)
    # build (cluster, task) -> Counter(answer)
    ct_dist: dict[tuple[int, str], Counter] = defaultdict(Counter)
    task_options: dict[str, list[str]] = defaultdict(set)
    for t in truth_records:
        c = user_cluster.get(str(t["user_id"]))
        if c is None or not t.get("answer"):
            continue
        tid = str(t.get("task_id") or t.get("question_id") or "")
        ct_dist[(int(c), tid)][t["answer"]] += 1
        task_options[tid].add(t["answer"])

    preds: list[PredictionRecord] = []
    for t in truth_records:
        uid = str(t["user_id"])
        c = user_cluster.get(uid)
        if c is None:
            continue
        tid = str(t.get("task_id") or t.get("question_id") or "")
        opts = sorted(task_options.get(tid, set()))
        if not opts:
            continue
        dist = ct_dist.get((int(c), tid))
        if dist:
            majority = dist.most_common(1)[0][0]
        else:
            majority = opts[0]
        # flip 噪声 → 让 mock 不是完美 B3 复刻
        if rng.random() < flip_rate:
            alt = [o for o in opts if o != majority]
            ans = rng.choice(alt) if alt else majority
        else:
            ans = majority
        preds.append(PredictionRecord(
            user_id=uid,
            task_id=tid,
            scenario_id=tid,
            answer=str(ans),
            dominant_factor=str(rng.choice(factor_pool)),
            sadr_activations=[f"#mock{rng.integers(1, 30)}{rng.choice(['+', '-'])}"],
            full_response=f"[MOCK rag_mode={rag_mode}] 答案: {ans}",
            rag_mode=rag_mode,
        ))
    logger.info("Mock predictions built: n=%d, mode=%s, flip_rate=%.2f",
                len(preds), rag_mode, flip_rate)
    return preds


# ---------------------------------------------------------------------------
# Phase 0 Test B 用的 trajectory vec builder
# ---------------------------------------------------------------------------

def build_trajectory_vec(
    preds: list[PredictionRecord],
    sadr_vocab: list[str] | None = None,
    factor_pool: tuple[str, ...] = ("P", "K", "V", "C"),
) -> list[dict[str, Any]]:
    """聚合 per-user 的 trajectory embedding (factor one-hot 累加 + sadr freq).

    输出 ``[{user_id, trajectory_vec}]`` 直接喂 ``ablation.phase0_test_b``.

    每个 user 的 vec = concat([factor_count_normalized (4d),
                              sadr_freq_over_vocab (|vocab|d)])
    """
    # 先扫一遍 vocab
    if sadr_vocab is None:
        vocab = sorted({s if isinstance(s, str) else (s.get("id", "") + s.get("polarity", ""))
                        for p in preds for s in p.sadr_activations})
        vocab = [v for v in vocab if v]
    else:
        vocab = list(sadr_vocab)
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    by_user: dict[str, list[PredictionRecord]] = defaultdict(list)
    for p in preds:
        by_user[p.user_id].append(p)

    out: list[dict[str, Any]] = []
    for uid, plist in by_user.items():
        factor_count = np.zeros(len(factor_pool), dtype=float)
        for p in plist:
            if p.dominant_factor:
                for tok in p.dominant_factor.replace(" ", "").replace("->", "→").split("→"):
                    if tok in factor_pool:
                        factor_count[factor_pool.index(tok)] += 1
        s = factor_count.sum()
        if s > 0:
            factor_count = factor_count / s

        sadr_freq = np.zeros(max(len(vocab), 1), dtype=float)
        for p in plist:
            for tok in p.sadr_activations:
                key = tok if isinstance(tok, str) else (tok.get("id", "") + tok.get("polarity", ""))
                if key in vocab_idx:
                    sadr_freq[vocab_idx[key]] += 1
        s2 = sadr_freq.sum()
        if s2 > 0:
            sadr_freq = sadr_freq / s2

        vec = np.concatenate([factor_count, sadr_freq])
        out.append({"user_id": uid, "trajectory_vec": vec.tolist()})
    return out


# ---------------------------------------------------------------------------
# Extracted record schema normalizer (data_v3 兼容)
# ---------------------------------------------------------------------------

def normalize_extracted_record(rec: dict[str, Any]) -> dict[str, Any]:
    """把多种 extractor 输出 schema 统一成 ``prototype.build_prototypes`` 期望的:

      ``{user_id, scenario_id, dominant_factor, sadr_activations, criterion}``

    支持的输入 schema:

    A. **quad_extracted (v3.5.2+ quad schema, data_v3 主)**:
       ``{designer_id, question_id, o_primary, sadr_activations:[{id,label}], ...}``

    B. **q7_extracted_v8.7 / q1q6_extracted_v8.7 (single-pred wrapper)**:
       ``{user_id, qid, segment_role, pred:{dominant_factor, criterion, ...}}``

    C. **legacy extractor.batch (extract.llm 单段输出)**:
       ``{user_id, scenario_id, extracted:{dominant_factor, sadr_activations}}``
       (本函数对 C 是 no-op pass-through; prototype 已经能直接吃)
    """
    if not rec:
        return {}
    out: dict[str, Any] = {}

    # user_id
    uid = rec.get("user_id") or rec.get("designer_id")
    out["user_id"] = str(uid) if uid is not None else None

    # scenario_id
    sid = rec.get("scenario_id") or rec.get("question_id")
    if not sid:
        qid = rec.get("qid")
        seg = rec.get("segment_role")
        if qid and seg and seg not in ("-", "single", "single_step", "full"):
            sid = f"{qid}_{seg}"
        else:
            sid = qid
    out["scenario_id"] = str(sid) if sid else "?"

    # dominant_factor: o_primary (quad) | pred.dominant_factor (single-pred) | extracted.dominant_factor (legacy)
    factor = (rec.get("o_primary")
              or (rec.get("pred") or {}).get("dominant_factor")
              or (rec.get("extracted") or {}).get("dominant_factor")
              or rec.get("dominant_factor"))
    out["dominant_factor"] = factor

    # state_sequence override (q7 pred 有 state_sequence list, 比 dominant_factor 信息多)
    pred = rec.get("pred") or {}
    if pred.get("state_sequence"):
        seq = pred["state_sequence"]
        if isinstance(seq, list) and seq:
            out["dominant_factor"] = " → ".join(str(s) for s in seq)

    # sadr_activations: quad has [{id,label,...}], normalize to '#ID±' string list
    raw_sadr = (rec.get("sadr_activations")
                or pred.get("sadr_activations")
                or (rec.get("extracted") or {}).get("sadr_activations")
                or [])
    sadr_norm: list[str] = []
    for s in raw_sadr:
        if isinstance(s, str):
            sadr_norm.append(s)
        elif isinstance(s, dict):
            sid_s = str(s.get("id", "")).strip()
            pol = str(s.get("polarity", "")).strip()
            tok = f"{sid_s}{pol}" if pol and not sid_s.endswith(("+", "-")) else sid_s
            if tok:
                sadr_norm.append(tok)
    out["sadr_activations"] = sadr_norm

    # criterion / o_subtype 留一份方便 prototype.aggregate_criterion_keywords 用
    out["criterion"] = (pred.get("criterion")
                          or rec.get("criterion")
                          or rec.get("o_subtype"))

    return out


def load_extracted_records_normalized(path: Path) -> list[dict[str, Any]]:
    """加载并 normalize extracted jsonl. 任何已知 schema 一把过."""
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = normalize_extracted_record(rec)
            if n.get("user_id") and n.get("dominant_factor"):
                out.append(n)
    logger.info("Loaded + normalized %d extracted records from %s", len(out), path.name)
    return out


# ---------------------------------------------------------------------------
# Truth converter: tasks DataFrame → metrics-friendly truth_records
# ---------------------------------------------------------------------------

def tasks_df_to_truth_records(tasks_df, mcq_questions: Iterable[str]) -> list[dict[str, Any]]:
    """Convert task_responses-style DataFrame → ``[{user_id, task_id, answer}]``.

    Mirrors ``runner._prepare_mcq_answers`` but returns dicts for metrics consumption.
    """
    import pandas as pd
    t = tasks_df[tasks_df["question_id"].isin(list(mcq_questions))].copy()
    t["letter"] = t["answer_value"].astype(str).str.extract(r"([A-E])\.")
    t = t.dropna(subset=["letter"])
    out: list[dict[str, Any]] = []
    for r in t.itertuples():
        out.append({
            "user_id": str(r.user_id),
            "task_id": str(r.question_id),
            "answer": str(r.letter),
        })
    return out
