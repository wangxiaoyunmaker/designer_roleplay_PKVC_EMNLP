"""q1q6_nl_refine.py — Q1/Q3-Q6 text_supplement → 100-200 字 refined NL.

照搬 scripts/q2_nl_refine.py 思路, 但适配 single-option Q (不是 Q2 ranking).

目的:
  Q1/Q3-Q6 训练 NL 现状: 28-39% 有 expansion (~100 字), 62-72% fallback 到 raw text_supplement (median 25 字).
  这一不齐导致训练 NL 质量分布 bimodal, model 学到 "短水文" 跟 "长解释" 两个模式混合.

做法:
  对剩余 62-72% 跑 gpt-4o refine:
    输入: raw text_supplement + 该选项 PKVC label + 该选项 verbose 文本 + profile polarity NL
    输出: 100-200 字 第一人称 NL, 严格 anchor 选项的 PKVC routing, 保留 user 原 NL 语气/关键词

  输出 append 到 data_v3/processed/q1q6_expansion_outputs.jsonl (跟现有 schema 一致),
  prep_v2 load 时自动 union.

CLI:
    export OPENAI_API_KEY=$(cat .env | grep OPENAI_API_KEY | cut -d= -f2)
    python scripts/q1q6_nl_refine.py --model gpt-4o

    # 测试模式: 只跑 5 个 sample, 看 prompt + 输出
    python scripts/q1q6_nl_refine.py --smoke 5

  Output schema (match existing q1q6_expansion_outputs.jsonl):
    {candidate_id, user_id, question_id, seed_nl, seed_char_count,
     result: {expanded_nl, inferred_factor, inferred_subtype, evidence_quote,
              evidence_source, confidence, reason_if_low},
     validation: {ok, fails},
     model: 'gpt-4o', api_error: false}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("q1q6_nl_refine")

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────
# Constants from framework v1.1 §0.1 (跟 prep_v2 / build_q1q6_training_labels 一致)
# ─────────────────────────────────────────────────────────────────────

PKVC_FACTOR_VERBOSE = {
    "P": "P (Problem 现实与问题重构)",
    "K": "K (Knowledge 经验/数据/技术调用)",
    "V": "V (Value 价值判断)",
    "C": "C (Concept 构念/构建)",
}

PKVC_SUBTYPE_VERBOSE = {
    "P1": "P1 界定核心", "P2": "P2 拆解子问题", "P3": "P3 推翻假定",
    "K1": "K1 调用经验", "K2": "K2 客观数据测试", "K3": "K3 评估技术栈",
    "V1": "V1 价值立场", "V2": "V2 利益排序", "V3": "V3 设立原则",
    "C1": "C1 涌现画面", "C2": "C2 整合构建", "C3": "C3 提炼原则",
}


def chain_verbose(chain: list[str]) -> str:
    if len(chain) == 1:
        return PKVC_FACTOR_VERBOSE.get(chain[0]) or PKVC_SUBTYPE_VERBOSE.get(chain[0], chain[0])
    return " → ".join(PKVC_SUBTYPE_VERBOSE.get(c, c) for c in chain)


# ─────────────────────────────────────────────────────────────────────
# Polarity helpers (inline, 跟 prep_v2 一致)
# ─────────────────────────────────────────────────────────────────────

POLARITY_ORDER = ["strong_pos", "weak_pos", "weak_neg", "strong_neg"]
POLARITY_NL_LABEL = {"strong_pos": "强 +", "weak_pos": "弱 +", "weak_neg": "弱 -", "strong_neg": "强 -"}
LIKERT_TO_POLARITY = {1: "strong_neg", 2: "strong_neg", 3: "weak_neg", 4: "weak_pos", 5: "strong_pos", 6: "strong_pos"}


def build_profile_polarity_nl(uid: str, scales: pd.DataFrame) -> str:
    sub = scales[scales.user_id.astype(str) == uid]
    if sub.empty:
        return "(profile 缺)"
    polarity = {k: [] for k in POLARITY_ORDER}
    for _, row in sub.iterrows():
        try:
            bucket = LIKERT_TO_POLARITY[int(round(float(row["score"])))]
            polarity[bucket].append(str(row["question_node"]))
        except (KeyError, ValueError):
            continue
    lines = []
    for k in POLARITY_ORDER:
        if polarity[k]:
            lines.append(f"{POLARITY_NL_LABEL[k]}: {', '.join(polarity[k])}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是设计师认知风格分析专家. 任务: 把一个设计师写的"短随手 NL"重写成"解释她/他选这个选项的设计师 first-person NL", 严格保留她/他原 NL 的语气 + 用词偏好 + 价值倾向, 不加 LLM 通用套话.

铁律:
1. 输出必须**呼应给定的 PKVC 选项激活路径** (single factor / chain). 解释该 designer 为啥选这个选项, NL 中要体现 PKVC routing.
2. 保留原 NL 出现的所有用词 / 情绪 / 立场倾向 (如果 user 表达不耐烦, refined NL 也应有不耐烦的痕迹).
3. 保留原 NL 中明确的关键概念 (e.g. user 说 "底线" → refined 必须含 "底线"; user 说 "数据" → refined 也要含).
4. 100-200 字, 一段连续中文, 不要列表 / 标题 / markdown / 子标题.
5. 第一人称 (设计师视角), 不要用 "设计师" / "她" 作主语.
6. 禁止编造 user 没有的事实 (不要凭空加经验 / 角色 / 具体项目细节).
7. 禁止 LLM 套话: "需要综合考虑" / "balance" / "权衡多方面" / "深入分析" 都不要.
8. 严格输出 JSON, 不要 prefix / 注释 / markdown wrapper."""

USER_PROMPT_TEMPLATE = """请基于以下材料, 重写一段 NL 解释该设计师选这个选项的设计决策.

[设计师 30 维信念画像 (帮你定调子, 不要直接 quote)]
{profile_polarity_nl}

[当前问题]
{question_title}

[该设计师选的选项 (内容 + PKVC 激活路径)]
选项 {option_letter}: "{option_text}"
PKVC 激活: {pkvc_verbose}

[该设计师自己写的原始短 NL]
"{raw_text_supplement}"

输出严格 JSON, schema:
{{
  "expanded_nl": "<100-200 字 第一人称 NL>",
  "inferred_factor": "<P|K|V|C 单字母, 跟 PKVC 激活第一段一致>",
  "inferred_subtype": "<P1-C3 子类, 跟 PKVC 激活第一段子类一致>",
  "evidence_quote": "<原 raw NL 里你保留的关键短语 (5-15 字)>",
  "confidence": "<high|medium|low>"
}}"""


QUESTION_TITLES = {
    "Q1": "Q1 面对委托第一启动认知模块",
    "Q3": "Q3 数据 vs 美感冲突 (硬件形态)",
    "Q4": "Q4 商业 vs 隐私 vs 开源 (脑电波数据商业化)",
    "Q5": "Q5 AI 冲击与人本/有机平衡 (AI 自动 UI)",
    "Q6": "Q6 商业利润 vs 用户责任 (上线焦虑)",
}


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def validate_output(result: dict, raw_nl: str, expected_chain: list[str]) -> tuple[bool, list[str]]:
    fails = []
    nl = result.get("expanded_nl", "")
    if not (80 <= len(nl) <= 300):
        fails.append(f"length out of [80, 300] = {len(nl)}")
    # factor 必须 match chain head
    inf_factor = result.get("inferred_factor")
    if expected_chain:
        expected_head = expected_chain[0][0] if expected_chain[0] else None
        if inf_factor != expected_head:
            fails.append(f"factor mismatch: expected {expected_head}, got {inf_factor}")
    # 关键词保留: split raw NL by 标点 → 每 segment 抽 2-4 字 substring → 看 refined 命中 ≥ 1
    stopwords = {"是", "的", "了", "和", "在", "我", "你", "他", "这", "那", "也", "都", "就",
                 "因为", "所以", "我会", "我想", "可以", "应该", "需要", "可能", "但是", "如果",
                 "什么", "怎么", "时候", "他们", "我们", "自己", "一种", "一个", "这个", "那个"}
    if raw_nl:
        raw_keywords = set()
        # split by 中/英文标点
        segments = re.split(r"[，。！？、；：…,.!?;:\s]+", raw_nl)
        for seg in segments:
            chinese_only = re.sub(r"[^一-鿿]", "", seg)
            # 抽 2-4 字 substring (sliding window)
            for w_len in (2, 3, 4):
                for i in range(len(chinese_only) - w_len + 1):
                    w = chinese_only[i:i + w_len]
                    if w not in stopwords:
                        raw_keywords.add(w)
        if raw_keywords:
            hit = sum(1 for kw in raw_keywords if kw in nl)
            if hit == 0:
                # 只 fail 当真的 0 命中 (refined 跟 raw 完全无交集)
                sample = list(raw_keywords)[:8]
                fails.append(f"raw NL 关键词 0 命中 (sample): {sample}")
    # 套话黑名单
    for bad in ["需要综合考虑", "balance", "权衡多方面", "深入分析", "全面思考"]:
        if bad in nl:
            fails.append(f"LLM 套话: {bad}")
            break
    return (len(fails) == 0), fails


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def load_existing_coverage(p: Path) -> set[tuple[str, str]]:
    """Read existing q1q6_expansion_outputs to know what's already covered."""
    covered = set()
    if not p.exists():
        return covered
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = str(r.get("user_id", ""))
            qid = r.get("question_id", "")
            if r.get("validation", {}).get("ok", False) and uid and qid:
                covered.add((uid, qid))
    return covered


def build_missing_inputs(tasks_csv: Path, option_labels_jsonl: Path,
                          covered: set[tuple[str, str]]) -> list[dict]:
    """Identify (uid, qid) pairs missing from coverage. Build refine inputs."""
    tasks = pd.read_csv(tasks_csv)
    pat = re.compile(r'^"?([A-D])\.\s*(.+?)"?$', re.DOTALL)

    # Load option labels
    option_labels = {}
    with option_labels_jsonl.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            option_labels[(r["qid"], r["option"])] = r

    inputs = []
    for _, row in tasks.iterrows():
        qid = row["question_id"]
        if qid not in ("Q1", "Q3", "Q4", "Q5", "Q6"):
            continue
        uid = str(row["user_id"])
        if (uid, qid) in covered:
            continue
        ans = str(row["answer_value"]).strip()
        m = pat.match(ans)
        if not m:
            continue
        letter, option_text = m.group(1), m.group(2).rstrip('"').strip()
        label = option_labels.get((qid, letter))
        if not label:
            continue
        raw_nl = str(row.get("text_supplement", "") or "").strip()
        inputs.append({
            "uid": uid, "qid": qid, "letter": letter,
            "option_text": option_text, "raw_nl": raw_nl,
            "pkvc_chain": label["pkvc_chain"],
            "pkvc_verbose": label["pkvc_verbose"],
        })
    return inputs


def call_gpt4o(client, inp: dict, profile_nl: str, model: str, max_retry: int = 3) -> dict:
    """Single API call with retry."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        profile_polarity_nl=profile_nl,
        question_title=QUESTION_TITLES[inp["qid"]],
        option_letter=inp["letter"],
        option_text=inp["option_text"],
        pkvc_verbose=inp["pkvc_verbose"],
        raw_text_supplement=inp["raw_nl"] or "(无补充, 仅选了该选项)",
    )

    last_err = None
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.5 + 0.1 * attempt,  # bump temp on retry
                max_tokens=400,
            )
            content = resp.choices[0].message.content
            result = json.loads(content)
            ok, fails = validate_output(result, inp["raw_nl"], inp["pkvc_chain"])
            if ok:
                return {"result": result, "validation": {"ok": True, "fails": []},
                        "attempts": attempt + 1}
            elif attempt == max_retry - 1:
                # Last attempt, return with fails for downstream visibility
                return {"result": result, "validation": {"ok": False, "fails": fails},
                        "attempts": attempt + 1}
            # else continue to retry
        except Exception as e:
            last_err = str(e)
            time.sleep(1 + attempt)

    return {"result": None, "validation": {"ok": False, "fails": [f"API error: {last_err}"]},
            "attempts": max_retry}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--smoke", type=int, default=0, help="只跑 N 个 (debug)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-append", action="store_true",
                    help="不 append, 输出到 _new.jsonl 单独审计")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if not api_key and line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif not base_url and line.startswith("OPENAI_BASE_URL="):
                base_url = line.split("=", 1)[1].strip()
    if not api_key:
        logger.error("OPENAI_API_KEY not found in env or .env")
        return 1
    logger.info("Using base_url: %s", base_url or "(default openai.com)")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    tasks_csv = ROOT / "data_v3" / "task_responses.csv"
    option_labels_jsonl = ROOT / "data_v3" / "processed" / "q1q6_option_labels.jsonl"
    expansion_jsonl = ROOT / "data_v3" / "processed" / "q1q6_expansion_outputs.jsonl"
    scales_csv = ROOT / "data_v3" / "scale_responses.csv"

    logger.info("Loading existing coverage...")
    covered = load_existing_coverage(expansion_jsonl)
    logger.info("Already covered (user, qid) pairs: %d", len(covered))

    logger.info("Building missing inputs...")
    inputs = build_missing_inputs(tasks_csv, option_labels_jsonl, covered)
    logger.info("Missing (need refine): %d", len(inputs))

    if args.smoke:
        inputs = inputs[: args.smoke]
        logger.info("[SMOKE] limit to %d", args.smoke)

    if not inputs:
        logger.info("All covered. Nothing to do.")
        return 0

    # Cost estimate
    est_cost = len(inputs) * 0.05
    logger.info("Estimated cost: ~$%.2f (%d calls × $0.05 / call, gpt-4o)", est_cost, len(inputs))
    if not args.smoke:
        confirm = input(f"Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            logger.info("Aborted by user.")
            return 0

    logger.info("Loading scales for profile NL...")
    scales = pd.read_csv(scales_csv)

    # Pre-build profile NL cache (avoid re-computing per call)
    profile_cache: dict[str, str] = {}
    for uid in set(inp["uid"] for inp in inputs):
        profile_cache[uid] = build_profile_polarity_nl(uid, scales)

    # Output path
    out_path = expansion_jsonl if not args.no_append else expansion_jsonl.with_suffix(".new.jsonl")
    mode = "a" if not args.no_append else "w"
    out_f = out_path.open(mode, encoding="utf-8")

    def worker(inp: dict) -> dict:
        profile_nl = profile_cache.get(inp["uid"], "")
        result = call_gpt4o(client, inp, profile_nl, args.model)
        entry = {
            "candidate_id": f"{inp['uid']}_{inp['qid']}",
            "user_id": inp["uid"],
            "question_id": inp["qid"],
            "seed_nl": inp["raw_nl"],
            "seed_char_count": len(inp["raw_nl"]),
            "result": result["result"],
            "validation": result["validation"],
            "model": args.model,
            "api_error": result["result"] is None,
        }
        return entry

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(worker, inp): inp for inp in inputs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                entry = fut.result()
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                if entry["validation"]["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / i * (len(inputs) - i)
                    logger.info("[%d/%d] ok=%d fail=%d  elapsed=%.0fs eta=%.0fs",
                                i, len(inputs), n_ok, n_fail, elapsed, eta)
            except Exception as e:
                logger.error("worker exception: %s", e)
                n_fail += 1

    out_f.close()
    logger.info("DONE: total=%d ok=%d fail=%d (%.1f%% ok)",
                len(inputs), n_ok, n_fail, n_ok / len(inputs) * 100)
    logger.info("Output appended to: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
