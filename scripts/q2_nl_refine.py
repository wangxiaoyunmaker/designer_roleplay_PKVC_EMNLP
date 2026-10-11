"""Q2 NL refine pipeline (framework v1 §8.5).

输入: (user_id, raw_ranking, raw_text_supplement, profile_polarity_nl)
LLM: gpt-4o
输出: refined_nl (100-300 字, 解释 ranking 决策 + 保留 user 原 NL 语气)

存到: data/processed/q2_nl_refined.jsonl

run:
    export OPENAI_API_KEY=...
    .venv/bin/python scripts/q2_nl_refine.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from sft_pipeline.data.prep import numerical_to_polarity, polarity_to_nl
from sft_pipeline.eval.datav3_stats import loaders


SYSTEM_PROMPT = """你是设计师认知风格分析专家. 任务: 把一个设计师写的"短随手 NL" 重写成"解释她/他 PKVC 排序决策"的完整 NL, 严格保留她/他原 NL 的语气 + 用词偏好 + 价值倾向, 不要加 LLM 通用套话.

铁律:
1. 输出必须**逐项呼应**给定的 PKVC ranking (尤其 top 3 + bottom 3), 说明为什么这样排.
2. 保留原 NL 出现的所有用词 / 情绪 / 立场倾向 (如果 user 表达不耐烦, refined NL 也应有不耐烦的痕迹; 不要洗成"我经过深思熟虑").
3. 保留原 NL 中明确的关键概念 (e.g. user 说"底线", refined NL 必须含"底线"; user 说"用户", refined 也要含; user 提到具体角色或场景, refined 要保留).
4. 100-300 字, 一段连续中文, 不要列表 / 标题 / markdown.
5. 第一人称 (设计师视角).
6. 禁止编造 user 没有的事实 (不要凭空加经验 / 角色 / 项目细节).
7. 禁止 LLM 套话: "需要综合考虑" / "balance" / "权衡多方面" / "深入分析" 这种话不要出现.
8. 严格输出 NL 段, 不要任何 prefix 或 metadata."""


USER_PROMPT_TEMPLATE = """请基于以下材料, 重写一段 NL 解释该设计师的 PKVC 排序决策.

[设计师 30 维信念画像 — 帮你定调子, 不要直接 quote]
{profile_polarity_nl}

[该设计师真实的 PKVC 排序 — 从最重要 (rank 0) 到最不重要]
{ranking_verbose}

[该设计师自己写的原始短 NL — 必须保留语气 + 关键词]
"{raw_text_supplement}"

请输出: 一段 100-300 字 refined NL, 解释排序决策 (尤其 top-3 + bottom-3 为什么这样排), 保留原 NL 语气. 不要前缀, 直接段内容."""


def parse_q2_ranking(answer_value: str) -> list[dict] | None:
    """parse Q2 answer_value (JSON list) → ordered cardId list. 返回 None 若解析失败."""
    try:
        cards = json.loads(answer_value)
        if not isinstance(cards, list):
            return None
        return sorted(cards, key=lambda x: x.get("order", 999))
    except (json.JSONDecodeError, TypeError):
        return None


def cards_to_verbose_ranking(cards: list[dict]) -> str:
    """[{cardId, text, order}, ...] → 'P1 界定核心 > V3 设立原则 > ...' verbose 字符串."""
    parts = []
    for c in cards:
        cid = c.get("cardId", "?")
        text = c.get("text", "").strip()
        # 短截断 text 避免太长
        if len(text) > 30:
            text = text[:30] + "..."
        parts.append(f"{cid} {text}")
    return " > ".join(parts)


def validate_refined(refined: str, raw_nl: str, top3_cards: list[str]) -> tuple[bool, str]:
    """简单 validation: 长度 + raw NL 关键词 + top-3 cardId 出现 + 无套话.

    Returns (ok, reason_if_fail).
    """
    n = len(refined.strip())
    if n < 80 or n > 500:
        return False, f"length out of [80,500] = {n}"

    # raw NL 里关键名词应保留
    keywords = ["底线", "用户", "老板", "设计", "原则", "矛盾", "效率", "经验", "数据"]
    raw_keys_present = [k for k in keywords if k in raw_nl]
    raw_keys_kept = [k for k in raw_keys_present if k in refined]
    if raw_keys_present and len(raw_keys_kept) < len(raw_keys_present) * 0.5:
        return False, f"raw NL 关键词丢失太多: {set(raw_keys_present) - set(raw_keys_kept)}"

    # top-3 cardId 至少 2 个出现 (verbose 或 cardId)
    top3_hits = 0
    for cid_text in top3_cards:
        cid = cid_text.split()[0]  # "P1" from "P1 界定..."
        text = " ".join(cid_text.split()[1:])
        if cid in refined or (text and text[:6] in refined):
            top3_hits += 1
    if top3_hits < 2:
        return False, f"top-3 cardId 覆盖不足: 仅 {top3_hits}/3"

    # 无套话
    bad_phrases = ["需要综合考虑", "balance", "权衡多方面", "深入分析"]
    for bp in bad_phrases:
        if bp in refined:
            return False, f"出现套话: {bp}"

    return True, ""


def refine_one(
    client: OpenAI,
    model: str,
    profile_nl: str,
    ranking_verbose: str,
    raw_nl: str,
    top3_cards: list[str],
    max_retries: int = 2,
) -> tuple[str, dict]:
    """跑 1 个 user, 失败重试. 返回 (refined_nl, validation_dict)."""
    last_resp = ""
    last_reason = ""
    for attempt in range(max_retries + 1):
        temp = 0.5 + attempt * 0.2
        prompt = USER_PROMPT_TEMPLATE.format(
            profile_polarity_nl=profile_nl,
            ranking_verbose=ranking_verbose,
            raw_text_supplement=raw_nl,
        )
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=500,
        )
        refined = r.choices[0].message.content.strip()
        ok, reason = validate_refined(refined, raw_nl, top3_cards)
        if ok:
            return refined, {"ok": True, "attempts": attempt + 1, "length": len(refined)}
        last_resp = refined
        last_reason = reason
    # 重试都 fail, 接受最后一次
    return last_resp, {"ok": False, "attempts": max_retries + 1, "length": len(last_resp), "last_fail": last_reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--output", default="data/processed/q2_nl_refined.jsonl")
    ap.add_argument("--max-users", type=int, default=0, help="0 = all 163")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    print("[data] loading Q2 task + scale ...")
    raw_task = loaders.load_task_long(expanded=False)
    q2 = raw_task[raw_task["question_id"] == "Q2"].copy()
    scale = loaders.load_scale_wide()
    print(f"[data] Q2 users: {len(q2)}; scale users: {len(scale)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url)

    selected_uids = q2["user_id"].tolist()
    if args.max_users > 0:
        selected_uids = selected_uids[: args.max_users]
    print(f"[run] processing {len(selected_uids)} users, model={args.model}\n")

    results = []
    t0 = time.time()
    for i, uid in enumerate(selected_uids, 1):
        row = q2[q2["user_id"] == uid].iloc[0]
        raw_nl = str(row.get("text_supplement", "")).strip()
        if not raw_nl or raw_nl == "nan":
            raw_nl = "(无补充)"

        cards = parse_q2_ranking(row["answer_value"])
        if not cards:
            print(f"[{i}/{len(selected_uids)}] {uid[:8]} SKIP (ranking parse fail)")
            continue
        ranking_verbose = cards_to_verbose_ranking(cards)
        top3 = [f"{c['cardId']} {c.get('text','')[:30]}" for c in cards[:3]]

        if uid not in scale.index:
            print(f"[{i}/{len(selected_uids)}] {uid[:8]} SKIP (no scale)")
            continue
        pol = numerical_to_polarity(scale.loc[uid])
        profile_nl = polarity_to_nl(pol)

        try:
            refined, val = refine_one(client, args.model, profile_nl, ranking_verbose, raw_nl, top3)
        except Exception as e:
            print(f"[{i}/{len(selected_uids)}] {uid[:8]} FAIL: {e}")
            continue

        results.append({
            "user_id": uid,
            "raw_text_supplement": raw_nl,
            "ranking_cardids": [c["cardId"] for c in cards],
            "ranking_verbose": ranking_verbose,
            "refined_nl": refined,
            "validation": val,
            "optimizer_model": args.model,
        })

        # incremental save (防 crash 丢)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        flag = "✓" if val["ok"] else "✗"
        print(f"[{i}/{len(selected_uids)}] {uid[:8]} {flag} len={val['length']} attempts={val['attempts']} elapsed={elapsed:.0f}s")

    print(f"\n[done] {len(results)}/{len(selected_uids)} refined. saved to {out_path}")
    n_ok = sum(1 for r in results if r["validation"]["ok"])
    print(f"[done] validation pass: {n_ok}/{len(results)} = {n_ok/max(len(results),1):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
