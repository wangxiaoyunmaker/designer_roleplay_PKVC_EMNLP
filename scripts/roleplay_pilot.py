"""Role-play Q7 pilot: 10 users × 3 baselines (B0/B1/B2) + PKVC eval.

Pre-SFT sanity: zero-shot gpt-4o 在不同 context (cold / profile / RAG) 下
role-play 生成 Q7 seg1, 跟 real user 的 PKVC label 比 similarity.

期望: B1 (profile) > B0 (cold); B2 (RAG) > B1 (signal exists if RAG 有用)
若 B1/B2 ≈ B0: 信号弱, RAG 走 SFT 路径前需重新设计
若 B2 > B1 + 5%: RAG 有效, 训完 SFT 后预期 lift 更高

run: .venv/bin/python scripts/roleplay_pilot.py [--n-users 10] [--model gpt-4o]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from sft_pipeline.data.prep import numerical_to_polarity, polarity_to_nl
from sft_pipeline.eval.datav3_stats import loaders
from sft_pipeline.eval.pkvc_compare import COMPARATORS


# Q7 scenario (from data_v3, summary 版, 给 gpt-4o role-play 用)
Q7_SCENARIO = """
背景: 一个大型零售品牌想在老城区改造一家老旧的社区超市. 拟拆除几乎所有人工
收银台改全自动结算, 把高利润非生活必需品藏到深处迫使消费者动线变长, 引入 AI
视觉识别 + 推荐算法. 公司说: '我们要做全国首个全数字化、无人化、高效零售标杆'.
工程师说可行; 法务说边缘 OK; 但年轻设计师内部抗议'这是剥削老年用户'. 老板拍板:
'商业第一, 竞品马上发布, 3 个月内必须按数字化方案铺开'.

请输出 3 段思考:
- seg1 第一反应 (你最先想到什么, 100-200 字)
- seg2 决策链路 (你怎么权衡, 100-200 字)
- seg3 核心 tradeoff (你给上层提的底线, 100-200 字)
"""


V8_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {"type": "string", "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$"},
        "criterion": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": {"type": "string"},
    },
    "required": ["dominant_factor", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def load_v8_prompt() -> tuple[str, str]:
    """Parse extractor_v8.md → (system_prompt, user_template).

    用 line-anchored '\\n---\\n' delimiter, 避免误匹配 markdown table separator
    '|---|---|---|' (这个 bug 之前让 user template 被截到 127 字).
    """
    pt = Path("sft_pipeline/config/prompts/extractor_v8.md").read_text(encoding="utf-8")
    # System prompt block: 在 ## System Prompt 后第一个 ``` ... ``` 之间
    sys_section_start = pt.find("## System Prompt")
    sys_fence_open = pt.find("```", sys_section_start)
    sys_fence_close = pt.find("```", sys_fence_open + 3)
    sys_block = pt[sys_fence_open + 3 : sys_fence_close].strip()

    # User template block: 在 ## User Prompt Template 后第一个 ``` ... ``` 之间
    usr_section_start = pt.find("## User Prompt Template")
    usr_fence_open = pt.find("```", usr_section_start)
    usr_fence_close = pt.find("```", usr_fence_open + 3)
    usr_template = pt[usr_fence_open + 3 : usr_fence_close].strip()
    return sys_block, usr_template


def gen_b0(client: OpenAI, model: str) -> str:
    """B0: cold, no profile, no RAG."""
    prompt = f"你扮演一位设计师, 回答下面的设计 dilemma.\n{Q7_SCENARIO}"
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=800,
    )
    return r.choices[0].message.content


def gen_b1(client: OpenAI, model: str, profile_nl: str) -> str:
    """B1: profile NL injected."""
    prompt = f"""你扮演下面这位特定的设计师, 严格按照她/他的信念画像回答.

[设计师信念画像]
{profile_nl}
[/设计师信念画像]

{Q7_SCENARIO}"""
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=800,
    )
    return r.choices[0].message.content


def gen_b2(client: OpenAI, model: str, profile_nl: str, peer_q7s: list[str]) -> str:
    """B2: B1 + top-3 trait-similar peers' real Q7 seg1 作 RAG context (raw NL)."""
    peer_block = "\n\n".join(
        [f"[trait 相似 peer {i+1}, 真实回答 seg1]\n{q[:400]}" for i, q in enumerate(peer_q7s)]
    )
    prompt = f"""你扮演下面这位特定的设计师, 严格按照她/他的信念画像回答.

[设计师信念画像]
{profile_nl}
[/设计师信念画像]

[3 位 trait 相似设计师的真实回答, 作语气/思路参考, 不要照抄]
{peer_block}
[/参考]

{Q7_SCENARIO}"""
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=800,
    )
    return r.choices[0].message.content


def gen_b3(client: OpenAI, model: str, profile_nl: str,
            peer_quads: list[tuple[str, list[dict]]]) -> str:
    """B3: B1 + top-3 trait-similar peers' QUAD structured routing pattern.

    peer_quads: list of (peer_uid, [quad_records_for_peer]).
    每 peer 展示 Q3-Q6 的 quad (C / s_primary / p_surface / o_primary).
    """
    peer_block_parts = []
    for i, (peer_uid, quads) in enumerate(peer_quads, 1):
        lines = [f"peer {i} (trait-similarity 邻居, 跨场景认知路由):"]
        for q in quads:
            s_pri = ", ".join(q.get("s_primary", []))
            lines.append(
                f"  {q['question_id']} 情境 '{q['C']}': "
                f"trait 画像 [{s_pri}] → "
                f"采用 {q['p_surface']} ({q['p_modulation']}) "
                f"→ 跳转 {q['o_primary']} state ({q.get('o_subtype', '?')})"
            )
        peer_block_parts.append("\n".join(lines))
    peer_block = "\n\n".join(peer_block_parts)

    prompt = f"""你扮演下面这位特定的设计师, 严格按照她/他的信念画像回答.

[设计师信念画像]
{profile_nl}
[/设计师信念画像]

[3 位 trait 相似设计师在过去 Q3-Q6 场景中的认知路由 (T_CAPS quad ontology)]
{peer_block}
[/参考路由]

提示: 上面 quad 反映了 trait-similar peer 在不同情境下的 PKVC routing 模式
(p_surface 是 8 词 codebook: forces_transition_to=Strong+, gradually_shifts_to=Weak+,
compliantly_follows=Weak+ Compliance, negotiates_around=Weak+ Negotiation 等).
你的 Q7 回答应展现类似 trait → routing 的 cognitive style, 但**不要照抄 peer 内容**,
按你自己 profile 在 Q7 老城区超市情境下输出.

{Q7_SCENARIO}"""
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=800,
    )
    return r.choices[0].message.content


def load_quads(quad_path: Path) -> dict[str, list[dict]]:
    """加载 quad jsonl, 按 designer_id 聚合."""
    from collections import defaultdict
    out: dict[str, list[dict]] = defaultdict(list)
    for line in quad_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[q["designer_id"]].append(q)
    return dict(out)


def extract_pkvc(client: OpenAI, model: str, nl: str, sys_block: str, usr_template: str) -> dict:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_block},
            {"role": "user", "content": usr_template.replace("{INPUT_TEXT}", nl)},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pkvc_extraction", "schema": V8_SCHEMA, "strict": True},
        },
    )
    return json.loads(r.choices[0].message.content)


def get_q7_seg1_by_user() -> dict[str, str]:
    raw = loaders.load_task_long(expanded=False)
    q7 = raw[raw["question_id"] == "Q7"]
    out = {}
    for _, r in q7.iterrows():
        try:
            blob = json.loads(r["answer_value"])
            out[r["user_id"]] = blob["seg1"]["text"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def stratified_select(factor_scores: pd.DataFrame, eligible_uids: set[str], n: int, seed: int) -> list[str]:
    """按 F1 4 分位均匀选 n 个 user (从 eligible 池里)."""
    rng = np.random.default_rng(seed)
    f1 = factor_scores["F1"].dropna()
    f1 = f1[f1.index.isin(eligible_uids)]
    sorted_uids = f1.sort_values().index.tolist()
    N = len(sorted_uids)
    quartiles = [sorted_uids[i * N // 4: (i + 1) * N // 4] for i in range(4)]
    per_q = max(1, n // 4 + 1)
    selected = []
    for q in quartiles:
        q_copy = q.copy()
        rng.shuffle(q_copy)
        selected.extend(q_copy[:per_q])
    return selected[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=10)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--output-dir", default="reports/datav3/roleplay_pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY env (eg. `export $(cat .env | xargs)`)")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.md"

    # data
    print("[data] loading factor scores + Q7 + scale + quads...")
    factor_scores = pd.read_csv(loaders.get_default_factor_scores_path(), index_col=0)
    q7_seg1 = get_q7_seg1_by_user()
    scale = loaders.load_scale_wide()

    quad_path = Path("data/processed/quad_extracted_v3_aligned.jsonl")
    quads_by_user = load_quads(quad_path) if quad_path.exists() else {}
    quad_users = set(quads_by_user.keys())
    print(f"[data] quad pool: {len(quad_users)} designers with Q3-Q6 quads")

    eligible = set(factor_scores.index) & set(q7_seg1.keys()) & set(scale.index)
    print(f"[data] eligible (factor + Q7 + scale): {len(eligible)}")
    print(f"[data] eligible ∩ quad_pool: {len(eligible & quad_users)} (peer 池 for B3)")

    selected = stratified_select(factor_scores, eligible, args.n_users, args.seed)
    print(f"[data] selected {len(selected)} users (F1 4 分位 stratified): {[u[:8] for u in selected]}")

    # profile NL precompute
    profile_nls = {}
    for uid in eligible:
        if uid in scale.index:
            pol = numerical_to_polarity(scale.loc[uid])
            profile_nls[uid] = polarity_to_nl(pol)

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_block, usr_template = load_v8_prompt()
    print(f"[v8] system prompt {len(sys_block)} chars, user template {len(usr_template)} chars\n")

    results = []
    t_start = time.time()
    for i, uid in enumerate(selected, 1):
        target_vec = factor_scores.loc[uid].values
        real_seg1 = q7_seg1[uid]

        # trait_sim top-3 peers (excluding self, only peers w/ Q7) — for B2
        sims = []
        for p in eligible:
            if p == uid:
                continue
            pv = factor_scores.loc[p].values
            na, nb = np.linalg.norm(target_vec), np.linalg.norm(pv)
            if na == 0 or nb == 0:
                continue
            sims.append((p, float(np.dot(target_vec, pv) / (na * nb))))
        sims.sort(key=lambda x: -x[1])
        top3_b2 = [p for p, _ in sims[:3]]
        peer_q7s = [q7_seg1[p] for p in top3_b2]

        # B3: 限定 quad_users 池
        quad_sims = [(p, s) for p, s in sims if p in quad_users]
        top3_b3 = [p for p, _ in quad_sims[:3]]
        peer_quads_b3 = [(p, quads_by_user[p]) for p in top3_b3] if top3_b3 else []

        print(f"[{i}/{len(selected)}] {uid[:8]} — generating B0/B1/B2/B3 + extracting...")
        try:
            b0 = gen_b0(client, args.model)
            b1 = gen_b1(client, args.model, profile_nls[uid])
            b2 = gen_b2(client, args.model, profile_nls[uid], peer_q7s)
            b3 = gen_b3(client, args.model, profile_nls[uid], peer_quads_b3) if peer_quads_b3 else ""

            real_pkvc = extract_pkvc(client, args.model, real_seg1, sys_block, usr_template)
            b0_pkvc = extract_pkvc(client, args.model, b0, sys_block, usr_template)
            b1_pkvc = extract_pkvc(client, args.model, b1, sys_block, usr_template)
            b2_pkvc = extract_pkvc(client, args.model, b2, sys_block, usr_template)
            b3_pkvc = extract_pkvc(client, args.model, b3, sys_block, usr_template) if b3 else None
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {e}")
            continue

        scores = {}
        real_df = real_pkvc["dominant_factor"]
        for bname, bpkvc in [("b0", b0_pkvc), ("b1", b1_pkvc), ("b2", b2_pkvc)] + (
            [("b3", b3_pkvc)] if b3_pkvc else []
        ):
            scores[bname] = {
                metric: float(fn(bpkvc["dominant_factor"], real_df))
                for metric, fn in COMPARATORS.items()
            }

        result = {
            "user_id": uid,
            "peer_top3_b2": top3_b2,
            "peer_top3_b3": top3_b3,
            "real_pkvc": real_pkvc,
            "b0_pkvc": b0_pkvc,
            "b1_pkvc": b1_pkvc,
            "b2_pkvc": b2_pkvc,
            "b3_pkvc": b3_pkvc,
            "scores": scores,
            "generations": {"real_seg1": real_seg1, "b0": b0, "b1": b1, "b2": b2, "b3": b3},
        }
        results.append(result)

        # incremental save
        with results_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.time() - t_start
        b3_str = b3_pkvc["dominant_factor"] if b3_pkvc else "—"
        print(f"  done. real={real_df} b0={b0_pkvc['dominant_factor']} b1={b1_pkvc['dominant_factor']} b2={b2_pkvc['dominant_factor']} b3={b3_str} | elapsed {elapsed:.0f}s")

    # summary
    print(f"\n=== SUMMARY ({len(results)} users) ===\n")
    if not results:
        print("no results")
        return 1

    summary_lines = [
        f"# Role-play Q7 pilot — B0/B1/B2 × {len(results)} users (data_v3, {pd.Timestamp.now().date()})\n",
        f"- model: {args.model}",
        f"- baselines: B0 (cold) / B1 (+profile) / B2 (+raw NL peer RAG) / B3 (+quad-context peer RAG)",
        f"- target: real user Q7 seg1 PKVC label (v8 extractor)",
        f"- metric: pkvc_compare 5 modes (exact / head / tail / jaccard / sequence_sim)\n",
        f"## Per-baseline avg score (higher = closer to real user)\n",
        f"| baseline | exact | head | tail | jaccard | sequence_sim |",
        f"|---|---|---|---|---|---|",
    ]
    all_baselines = ["b0", "b1", "b2", "b3"]
    for bname in all_baselines:
        valid_scores = [r["scores"].get(bname) for r in results if bname in r.get("scores", {})]
        if not valid_scores:
            continue
        row = [bname]
        for metric in ["exact", "head", "tail", "jaccard", "sequence_sim"]:
            avg = np.mean([s[metric] for s in valid_scores])
            row.append(f"{avg:.3f}")
        summary_lines.append("| " + " | ".join(row) + " |")

    # lift
    summary_lines.append("\n## Lift vs B0 (cold baseline)\n")
    summary_lines.append("| baseline | sequence_sim | lift vs B0 (abs) |")
    summary_lines.append("|---|---|---|")
    b0_avg = np.mean([r["scores"]["b0"]["sequence_sim"] for r in results])
    summary_lines.append(f"| B0 | {b0_avg:.3f} | — |")
    for bname in ["b1", "b2", "b3"]:
        valid = [r["scores"][bname]["sequence_sim"] for r in results if bname in r.get("scores", {})]
        if not valid:
            continue
        avg = np.mean(valid)
        lift = avg - b0_avg
        summary_lines.append(f"| {bname.upper()} | {avg:.3f} | {lift:+.3f} |")

    # PKVC distribution
    summary_lines.append("\n## PKVC label distribution\n")
    from collections import Counter
    for bname in ["real_pkvc", "b0_pkvc", "b1_pkvc", "b2_pkvc", "b3_pkvc"]:
        c = Counter(r[bname]["dominant_factor"] for r in results if r.get(bname))
        summary_lines.append(f"- **{bname}**: {dict(c)}")

    # confidence
    summary_lines.append("\n## v8 extractor confidence on real / generated\n")
    for bname in ["real_pkvc", "b0_pkvc", "b1_pkvc", "b2_pkvc", "b3_pkvc"]:
        c = Counter(r[bname]["confidence"] for r in results if r.get(bname))
        summary_lines.append(f"- **{bname}**: {dict(c)}")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\n[saved] {results_path}")
    print(f"[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
