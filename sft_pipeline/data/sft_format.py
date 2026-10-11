"""sft_format.py — SFT v3 data format (docs/briefs/sft_data_format_v1.md v1.1).

3 个 task_type variant target schema (Q1 / Q2 / Q3-Q6).
Prompt 5 module: PKVC def + pi + stance + self Q1-Q6 (mask) + CAPS quad (mask).
砍 ④ peer + ⑥ criterion (spec §3.1 lock).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ────────────────────────────────────────────────────────────────────
# Module 0: PKVC 4 字母含义 (spec §3.1)
# ────────────────────────────────────────────────────────────────────

PKVC_4_LETTER_DEF = """# PKVC 4 类认知状态
- P (Problem) 现实与问题重构: P1 界定核心 / P2 拆解子问题 / P3 推翻假定
- K (Knowledge) 经验/数据/技术调用: K1 调用经验 / K2 客观数据测试 / K3 评估技术栈
- V (Value) 价值判断 / 底线宣言: V1 价值立场 / V2 利益排序 / V3 设立原则
- C (Concept) 构念/构建 / 抽象框架: C1 涌现画面 / C2 整合构建 / C3 提炼原则"""


# ────────────────────────────────────────────────────────────────────
# Q-level metadata
# ────────────────────────────────────────────────────────────────────

Q_CONFLICT_LABEL = {
    "Q3": "数据效用 vs 美感形态冲突 (K vs C/V)",
    "Q4": "商业利润 vs 用户隐私 vs 资产开源冲突 (K/V multi-axis)",
    "Q5": "AI 工具与设计师自主性冲突 (人机协作分工)",
    "Q6": "商业目标 vs 用户责任 / 社会责任冲突 (利益 vs 责任感)",
}

CARDID_ENUM = ["P1","P2","P3","K1","K2","K3","V1","V2","V3","C1","C2","C3"]


# ────────────────────────────────────────────────────────────────────
# PKVC chain → target field 派生
# ────────────────────────────────────────────────────────────────────

def parse_pkvc_chain(pkvc_chain: list[str]) -> dict[str, Any]:
    """e.g. ['P3','K3'] → {dom:'K', seq:['P','K'], subtype:'P3→K3'}.

    Rule (spec §2.5 example):
      - dom = chain 终点 letter (outcome). 单格情况 ['K2'] → dom='K'.
      - seq = letter strip 全部.
      - subtype = "→".join(chain).
    """
    if not pkvc_chain:
        return {"dom": "", "seq": [], "subtype": ""}
    return {
        "dom": pkvc_chain[-1][0],
        "seq": [c[0] for c in pkvc_chain],
        "subtype": "→".join(pkvc_chain),
    }


# ────────────────────────────────────────────────────────────────────
# Target JSON builders (3 schema)
# ────────────────────────────────────────────────────────────────────

def build_q1_target(
    user_id: str,
    parsed_resp: dict,
    option_labels: dict[tuple[str, str], dict],
    expansion: dict[tuple[str, str], dict],
) -> dict | None:
    """Q1 target (single dominant)."""
    if not parsed_resp:
        return None
    letter_opt = parsed_resp["letter"]
    label = option_labels.get(("Q1", letter_opt))
    if not label:
        return None
    parsed = parse_pkvc_chain(label["pkvc_chain"])
    chosen_letter = parsed["dom"]
    if chosen_letter not in ("P", "K", "V", "C"):
        return None
    nl = (expansion.get((user_id, "Q1"), {}) or {}).get("nl") or parsed_resp.get("text_supplement", "")
    nl = "" if (nl is None or (isinstance(nl, float))) else str(nl).strip()
    if not nl or len(nl) < 10:
        return None
    return {
        "task_type": "Q1",
        "chosen_letter": chosen_letter,
        "supplement_nl": nl,
        "dominant_factor": chosen_letter,
        "state_sequence": [chosen_letter],
    }


def build_q2_target(user_id: str, q2_entry: dict | None) -> dict | None:
    """Q2 target (cardId sequence ~3-9)."""
    if q2_entry is None:
        return None
    chain = q2_entry.get("raw_cardids") or []
    chain = [c for c in chain if c in CARDID_ENUM]
    if len(chain) < 2:
        return None
    letter_seq = [c[0] for c in chain]
    nl = q2_entry.get("refined_nl") or ""
    if not nl or len(nl) < 10:
        return None
    return {
        "task_type": "Q2",
        "thinking_chain": chain,
        "supplement_nl": nl,
        "state_sequence": letter_seq,
    }


def build_q3q6_target(
    user_id: str,
    qid: str,
    parsed_resp: dict,
    option_labels: dict[tuple[str, str], dict],
    expansion: dict[tuple[str, str], dict],
) -> dict | None:
    """Q3-Q6 target (dom + nl + subtype)."""
    if not parsed_resp:
        return None
    letter_opt = parsed_resp["letter"]
    label = option_labels.get((qid, letter_opt))
    if not label:
        return None
    parsed = parse_pkvc_chain(label["pkvc_chain"])
    if parsed["dom"] not in ("P", "K", "V", "C"):
        return None
    nl = (expansion.get((user_id, qid), {}) or {}).get("nl") or parsed_resp.get("text_supplement", "")
    nl = "" if (nl is None or (isinstance(nl, float))) else str(nl).strip()
    if not nl or len(nl) < 10:
        return None
    return {
        "task_type": qid,
        "chosen_option": letter_opt,
        "supplement_nl": nl,
        "dominant_factor": parsed["dom"],
        "state_sequence": parsed["seq"],
        "o_subtype": parsed["subtype"],
    }


# ────────────────────────────────────────────────────────────────────
# Module ① pi (demographic 5-field)
# ────────────────────────────────────────────────────────────────────

def fmt_pi(user_row: dict) -> str:
    def g(k, default="?"):
        v = user_row.get(k)
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return default
        return str(v)
    parts = [
        f"身份: {g('identity')}",
        f"经验: {g('years_exp')}",
        f"领域: {g('design_field')}",
        f"工作场景: {g('work_context')}",
        f"项目角色: {g('project_role')}",
    ]
    return " | ".join(parts)


# ────────────────────────────────────────────────────────────────────
# Module ② 30 维 stance labels (strong only, spec §3.1)
# ────────────────────────────────────────────────────────────────────

def fmt_stance(polarity: dict[str, list[str]]) -> str:
    """polarity: {strong_pos:[...], weak_pos:[...], weak_neg:[...], strong_neg:[...]}.
    Spec §3.1 ② 只展示强极性 (Likert 1,2 / 5,6)."""
    lines = []
    sp = polarity.get("strong_pos", [])
    sn = polarity.get("strong_neg", [])
    if sp:
        lines.append("强认同 (Likert 5-6): " + ", ".join(sp))
    if sn:
        lines.append("强反对 (Likert 1-2): " + ", ".join(sn))
    return "\n".join(lines) if lines else "(无明显强极性 trait)"


# ────────────────────────────────────────────────────────────────────
# Module ③ self Q1-Q6 (mask target, NL cleaned)
# ────────────────────────────────────────────────────────────────────

_Q_TITLES = {
    "Q1": "第一反应",
    "Q2": "思考链路 (cardId 序列)",
    "Q3": "Q3 数据效用 vs 美感冲突",
    "Q4": "Q4 商业利润 vs 用户隐私冲突",
    "Q5": "Q5 AI 工具与设计师自主性冲突",
    "Q6": "Q6 商业目标 vs 用户责任冲突",
}


def fmt_self_q1q6(
    user_id: str,
    parsed_responses: dict[str, dict],
    q2_entry: dict | None,
    expansion: dict[tuple[str, str], dict],
    option_labels: dict[tuple[str, str], dict],
    exclude_qid: str,
) -> str:
    """spec §3.2: 训 target_q 时, 展示其他 5 题. NL 一律用 expansion cleaned."""
    lines = []
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        if q == exclude_qid:
            continue
        lines.append(f"\n[{_Q_TITLES[q]}]")
        if q == "Q2":
            if q2_entry is None:
                lines.append("  (未答)")
                continue
            chain_str = " → ".join(q2_entry.get("raw_cardids") or [])
            nl = q2_entry.get("refined_nl") or "(无补充)"
            lines.append(f"  你的 cardId 思考链路: {chain_str}")
            lines.append(f"  你的解释 (cleaned): {nl}")
        else:
            resp = parsed_responses.get(q)
            if not resp:
                lines.append("  (未答)")
                continue
            label = option_labels.get((q, resp["letter"]))
            chain = label["pkvc_chain"] if label else []
            parsed = parse_pkvc_chain(chain)
            nl = (expansion.get((user_id, q), {}) or {}).get("nl") or resp.get("text_supplement", "") or "(无补充)"
            if q == "Q1":
                lines.append(f"  第一反应单字母: {parsed['dom']} (PKVC chain: {parsed['subtype']})")
            else:
                # Q3-Q6: 不暴露 dominant_factor (那是当前 target 字段同款), 只给 option_text + cleaned NL
                lines.append(f"  选择 ({resp['letter']}): \"{resp['text'][:120]}\"")
                lines.append(f"  PKVC chain: {parsed['subtype']} (dom={parsed['dom']}, seq={parsed['seq']})")
            lines.append(f"  你的解释 (cleaned): {nl}")
    return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────────────
# Module ⑤' CAPS quad (4 类平衡, mask target Q)
# ────────────────────────────────────────────────────────────────────

def fmt_caps_quad_balanced(
    quads_for_designer: list[dict],
    exclude_qid: str | None,
) -> str:
    """spec §3.3: 4 类平衡 (P/K/V/C 每类 top 1), 排除 target Q.

    quad row schema (data_v3/processed/quad_extracted_163_fixed.jsonl):
      designer_id, question_id, o_primary, o_subtype, p_modulation, p_stance,
      s_primary_with_stance: [{item, stance, direction, coef, marginal_perm_p}],
      C (conflict label), _meta.choice_selected, confidence
    """
    by_class: dict[str, list[dict]] = {"P": [], "K": [], "V": [], "C": []}
    for q in quads_for_designer:
        if exclude_qid and q.get("question_id") == exclude_qid:
            continue
        primary = q.get("o_primary")
        if primary in by_class:
            by_class[primary].append(q)

    lines = []
    for cls in "PKVC":
        bucket = by_class[cls]
        if not bucket:
            lines.append(f"## {cls} 类 top routing\n  (该 designer 在 P/K/V/C 中此类无 routing)")
            continue
        bucket.sort(key=lambda x: (0 if x.get("confidence") == "high" else 1))
        top = bucket[0]
        stance_parts = []
        for s in (top.get("s_primary_with_stance") or [])[:3]:
            sign = "+" if s.get("direction") == "+" else "-"
            coef = s.get("coef", 0.0) or 0.0
            stance_parts.append(f"{s.get('item', '?')} (β={sign}{abs(float(coef)):.2f})")
        stance_nl = ", ".join(stance_parts) if stance_parts else "(无 stance)"
        modulation = top.get("p_modulation", "")
        p_stance = top.get("p_stance", "")
        subtype = top.get("o_subtype", "")
        scenario_short = (top.get("C") or "").replace("_", " ")
        choice = (top.get("_meta") or {}).get("choice_selected", "?")
        qid = top.get("question_id", "?")
        lines.append(
            f"## {cls} 类 top routing\n"
            f"  [{qid}-{choice}] {scenario_short} | stance: {stance_nl}"
            f" | modulation: {modulation} ({p_stance}) → {subtype}"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Module: 任务 + output schema (按 task_type)
# ────────────────────────────────────────────────────────────────────

_SCENARIO_TRAINING_FALLBACK = """项目场景: 全感知效率空间

某互联网大厂提出一项设计委托: 为公司 5,000 名员工打造一个"全感知效率空间" — 该空间将通过穿戴设备 (EEG 脑电监测 + 心率 + 皮电) + 工位 IoT (光照/温湿度自动调节) + 手机 App (任务调度 + 焦点状态可视化) 三层协同, 实时感知员工的认知状态 (专注度 / 焦虑值 / 疲劳度), 自动微调环境参数 + 推送任务建议, 目标"提升 30% 工作效率, 同时改善心理健康指标"。预算 5,000 万元, 12 个月交付一期."""


def _fmt_options_block(qid: str, chosen_letter: str | None,
                        option_labels: dict[tuple[str, str], dict] | None,
                        brief_len: int = 80) -> str:
    """spec v1.1 fix C: 4 option 注入 — 被选项全文 + 3 未选项简述."""
    if not option_labels or not chosen_letter:
        return ""
    lines = [f"# 4 个候选选项 (我这次选了 {chosen_letter})"]
    for letter in ("A", "B", "C", "D"):
        label = option_labels.get((qid, letter))
        if not label:
            continue
        text = (label.get("option_text") or "").strip().strip('"').strip()
        pkvc = "/".join(label.get("pkvc_chain") or [])
        if letter == chosen_letter:
            lines.append(f"\n[{letter}] (我的选择, 全文 | PKVC chain: {pkvc}):")
            lines.append(f"  {text}")
        else:
            brief = text[:brief_len].rstrip() + ("..." if len(text) > brief_len else "")
            lines.append(f"[{letter}] (未选 | PKVC: {pkvc}): {brief}")
    return "\n".join(lines)


def fmt_task_q1(scenario_text: str, chosen_letter_opt: str | None = None,
                 option_labels: dict[tuple[str, str], dict] | None = None) -> str:
    options_block = _fmt_options_block("Q1", chosen_letter_opt, option_labels)
    return (
        f"# 场景\n{scenario_text}\n\n"
        f"{options_block + chr(10) + chr(10) if options_block else ''}"
        "# 任务 (target = Q1, 第一反应单字母)\n"
        "面对该委托, 列出你第一反应主导的认知模块 (单字母 P/K/V/C, 不经决策权衡).\n\n"
        "# 输出严格 JSON, 字段:\n"
        '  task_type: "Q1"\n'
        '  chosen_letter: ∈ {"P","K","V","C"}\n'
        "  supplement_nl: 50-200 字第一人称解释\n"
        "  dominant_factor: = chosen_letter\n"
        "  state_sequence: [chosen_letter]"
    )


def fmt_task_q2(scenario_text: str) -> str:
    return (
        f"# 场景\n{scenario_text}\n\n"
        "# 任务 (target = Q2, 思考链路 cardId 序列)\n"
        "列出你在该委托上的实际思考链路 — 用 PKVC 12 个 cardId 按你本人优先级排序.\n"
        "cardId 枚举: P1 P2 P3 K1 K2 K3 V1 V2 V3 C1 C2 C3 (但**不要**直接按这字母顺序输出).\n\n"
        "★ 重要提醒 (Q2 specific):\n"
        "  1. 起头 cardId **取决于你的 trait + CAPS routing**, 不是设计师群体平均.\n"
        "     - 群体里 P1 起头是 default (61%), 但你**不一定**是 — 看你 ② 立场跟 ⑤' CAPS routing.\n"
        "     - 强\"混沌与灵感\" / \"天赋信仰\" / C 类 CAPS top routing → 大概率 C1/C2 起头.\n"
        "     - 强\"完美主义/MVP\" + 弱\"混沌\" → 大概率 P1 起头.\n"
        "     - 强\"数据 vs 直觉\"(直觉派) + 强\"天赋信仰\" → 可能 C 或 V 起头.\n"
        "  2. 主体长度 4-12. 重复 OK 表强偏好 (e.g. K1 出现两次说明你特别依赖经验).\n"
        "  3. ⑤' 上面的 CAPS quad 哪类 modulation 最强, 起头大概率就是那类.\n\n"
        "# 输出严格 JSON, 字段:\n"
        '  task_type: "Q2"\n'
        "  thinking_chain: cardId 序列 array (典型长度 4-12)\n"
        "  supplement_nl: 100-300 字第一人称解释\n"
        "  state_sequence: letter strip (e.g. [P3,K2] → [P,K])"
    )


def fmt_task_q3q6(qid: str, scenario_text: str,
                   chosen_letter_opt: str | None = None,
                   option_labels: dict[tuple[str, str], dict] | None = None) -> str:
    conflict = Q_CONFLICT_LABEL.get(qid, "")
    options_block = _fmt_options_block(qid, chosen_letter_opt, option_labels)
    return (
        f"# 场景 (target = {qid} — {conflict})\n{scenario_text}\n\n"
        f"{options_block + chr(10) + chr(10) if options_block else ''}"
        "# 任务\n"
        f"基于你的人口统计学 + 立场 + 行为数据 + CAPS 路由模式, 输出你在 {qid} 上的回答.\n\n"
        "# 输出严格 JSON, 字段:\n"
        f'  task_type: "{qid}"\n'
        '  chosen_option: ∈ {"A","B","C","D"}\n'
        "  supplement_nl: 50-400 字第一人称解释\n"
        '  dominant_factor: ∈ {"P","K","V","C"} (chain 终点)\n'
        "  state_sequence: letter array (e.g. [\"P\",\"K\"])\n"
        '  o_subtype: PKVC chain, 例 "P3→K3" 或 "K2"'
    )


def fmt_task_block(target_q: str, scenario_text: str,
                    chosen_letter_opt: str | None = None,
                    option_labels: dict[tuple[str, str], dict] | None = None) -> str:
    if target_q == "Q1":
        return fmt_task_q1(scenario_text, chosen_letter_opt, option_labels)
    if target_q == "Q2":
        return fmt_task_q2(scenario_text)
    return fmt_task_q3q6(target_q, scenario_text, chosen_letter_opt, option_labels)


# ────────────────────────────────────────────────────────────────────
# Final assembly
# ────────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "你扮演下面注入的特定设计师, 用第一人称回答. 严格按 output schema 输出 JSON, "
    "不要前后缀 / 不要 markdown / 不要解释 schema 本身. 体现注入的 trait + CAPS routing pattern, "
    "禁止 default 到 V (Value) 套话."
)


def assemble_prompt(
    target_q: str,
    pi_block: str,
    stance_block: str,
    self_q1q6_block: str,
    caps_quad_block: str,
    task_block: str,
) -> str:
    """组合 5 module + task = full user prompt (spec §3.1).

    Layout:
      PKVC_4_LETTER_DEF
      # ① pi
      # ② stance (strong only)
      # ③ self Q1-Q6 (mask target)
      # ⑤' CAPS quad (4 类平衡, mask target)
      # task + output schema
    """
    sections = [
        PKVC_4_LETTER_DEF,
        f"# ① 你的人口统计学\n{pi_block}",
        f"# ② 你的核心立场 (强极性, 弱维度已过滤)\n{stance_block}",
        f"# ③ 你在其他题上的实际行为 (mask target={target_q}, NL 用 LLM cleaned 版)\n{self_q1q6_block}",
        f"# ⑤' 你的认知路由模式 (CAPS 四元组, 4 类平衡, 排除 target={target_q})\n{caps_quad_block}",
        task_block,
    ]
    return "\n\n".join(sections)


def build_sft_example(
    designer_id: str,
    target_q: str,
    *,
    user_row: dict,
    polarity: dict[str, list[str]],
    parsed_responses: dict[str, dict],
    q2_entry: dict | None,
    expansion: dict[tuple[str, str], dict],
    option_labels: dict[tuple[str, str], dict],
    quads_for_designer: list[dict],
    scenario_text: str,
) -> dict | None:
    """Build 1 SFT (input, target) example for (designer, target_q).

    Returns None if 数据缺失 (e.g., Q1 cleaned NL 没 + raw text_supplement 也短).
    """
    # === Target JSON ===
    if target_q == "Q1":
        target = build_q1_target(designer_id, parsed_responses.get("Q1") or {}, option_labels, expansion)
    elif target_q == "Q2":
        target = build_q2_target(designer_id, q2_entry)
    elif target_q in ("Q3", "Q4", "Q5", "Q6"):
        target = build_q3q6_target(designer_id, target_q, parsed_responses.get(target_q) or {}, option_labels, expansion)
    else:
        return None
    if target is None:
        return None

    # === Input prompt ===
    pi_block = fmt_pi(user_row)
    stance_block = fmt_stance(polarity)
    self_block = fmt_self_q1q6(designer_id, parsed_responses, q2_entry, expansion, option_labels,
                                exclude_qid=target_q)
    quad_exclude = target_q if target_q in ("Q3", "Q4", "Q5", "Q6") else None
    quad_block = fmt_caps_quad_balanced(quads_for_designer, exclude_qid=quad_exclude)

    # 4 option 注入 (Q1 / Q3-Q6 用), chosen_letter_opt = task_responses 里的 A/B/C/D
    chosen_letter_opt = None
    if target_q in ("Q1", "Q3", "Q4", "Q5", "Q6"):
        resp = parsed_responses.get(target_q) or {}
        chosen_letter_opt = resp.get("letter")
    task_block = fmt_task_block(target_q, scenario_text, chosen_letter_opt, option_labels)

    input_prompt = assemble_prompt(target_q, pi_block, stance_block, self_block, quad_block, task_block)

    target_str = json.dumps(target, ensure_ascii=False)
    return {
        "user_id": designer_id,                # train_v2 SFTDataset 兼容字段
        "qid": target_q,                        # train_v2 SFTDataset 兼容字段
        "designer_id": designer_id,
        "target_q": target_q,
        "system": SYSTEM_MESSAGE,
        "prompt": input_prompt,
        "target": target,
        "target_str": target_str,
        "completion": target_str,               # train_v2 SFTDataset 期望字段
    }
