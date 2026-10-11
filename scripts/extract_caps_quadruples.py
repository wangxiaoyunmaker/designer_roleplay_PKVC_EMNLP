#!/usr/bin/env python3
"""
从 option_trait_mapping.json 提取 CAPS 四元组 (C, S, P, O)。
输出原始单库 (不做正负镜像), 保持 +/- direction 原样。

数据源: option_trait_mapping.json (学姐标注的 Q1-Q6 选项-trait mapping)
输出:   data/processed/caps_quadruples.json (36 条, 原始 direction 不翻转)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parent.parent / "reports/datav3/per_designer/option_trait_mapping.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = OUT_DIR / "caps_quadruples.json"

# ── 30 维量表双极 stance 表 (权威来源) ────────────────────

BIPOLAR_STANCES = {
    "数据 vs 直觉":    {"+": "重直觉立场",     "-": "重数据立场"},
    "天赋信仰":        {"+": "天赋决定论",     "-": "后天努力论"},
    "AI 替代论":       {"+": "AI替代主义",     "-": "岗位保护主义"},
    "技术支持":        {"+": "技术激进主义",    "-": "技术审慎主义"},
    "复杂工具信仰":    {"+": "工具门槛论",     "-": "工具无关论"},
    "混沌与灵感":      {"+": "包容失控派",     "-": "验证至上派"},
    "黑盒透明度":      {"+": "结果至上派",     "-": "过程透明派"},
    "系统微模块":      {"+": "解耦优先",       "-": "整体感优先"},
    "完美主义/MVP":    {"+": "完美主义",       "-": "速度优先"},
    "创新代价":        {"+": "创新激进派",     "-": "惯例守护派"},
    "审美反民粹":      {"+": "审美精英主义",    "-": "数据民粹妥协"},
    "情感附加值":      {"+": "仪式感优先",     "-": "效率优先"},
    "装饰合法性":      {"+": "装饰合法派",     "-": "形式追随功能"},
    "时代 vs 永恒":    {"+": "时代潮流派",     "-": "永恒经典派"},
    "本土文化妥协":    {"+": "本土深耕派",     "-": "全球通用派"},
    "仿生与有机":      {"+": "有机自然派",     "-": "几何工业派"},
    "独裁 vs 委员会":  {"+": "独裁愿景派",     "-": "民主共创派"},
    "秩序与防错":      {"+": "防错约束派",     "-": "探索开放派"},
    "创作者署名":      {"+": "作者性优先",     "-": "商业去人格化"},
    "资产开源":        {"+": "开源道德派",     "-": "竞争壁垒派"},
    "封闭生态竞争":    {"+": "封闭生态派",     "-": "开放兼容派"},
    "风险规避":        {"+": "风险规避",       "-": "风险偏好"},
    "团队依恋":        {"+": "独立打磨派",     "-": "团队共创派"},
    "商业操纵伦理":    {"+": "商业操纵容忍",    "-": "用户伦理底线"},
    "数据隐私交换":    {"+": "体验换隐私",     "-": "隐私保护派"},
    "弱势群体优先":    {"+": "边缘群体优先",    "-": "多数效率优先"},
    "股东服务优先":    {"+": "股东忠诚派",     "-": "用户忠诚派"},
    "政治中立性":      {"+": "政治中立派",     "-": "政治表达派"},
    "系统责任归属":    {"+": "系统归责派",     "-": "用户归责派"},
    "救世主情结":      {"+": "救世主情结",     "-": "工具中立派"},
}

# ── 分类规则 ──────────────────────────────────────────────

def classify_predicate(p: float, coef: float, direction: str) -> str | None:
    abs_coef = abs(coef)
    if direction == "-" and p < 0.10 and abs_coef >= 0.20:
        return "strictly_inhibits"
    if direction == "+":
        if p < 0.05 and abs_coef >= 0.50:
            return "strongly_forces_transition_to"
        if p < 0.05 and abs_coef >= 0.20:
            return "guides_transition_to"
        if 0.05 <= p < 0.10 and abs_coef >= 0.20:
            return "weakly_shifts_to"
    return None

# ── PKVC 词表 ─────────────────────────────────────────────

PKVC_SUBTYPE_VERBOSE = {
    "P":  "P-Problem(现实与问题重构)",
    "p1": "P1-界定核心(现实约束与宏观收敛)",
    "p2": "P2-拆解子问题(降维与工程化)",
    "p3": "P3-推翻假定(批判性思维与重构)",
    "K":  "K-Knowledge(经验/数据/技术调用)",
    "k1": "K1-调用经验(经验主义)",
    "k2": "K2-客观数据测试(数据实证主义)",
    "k3": "K3-评估技术栈(技术可行性边界)",
    "V":  "V-Value(价值判断)",
    "v":  "V-Value(价值判断)",
    "v1": "V1-价值立场(专业价值立场判断)",
    "v2": "V2-利益优先级排序(实用伦理与多边协调)",
    "v3": "V3-设立原则(道德绝对主义)",
    "C":  "C-Concept(构念/构建)",
    "c1": "C1-涌现强烈画面(直觉与视觉先导)",
    "c2": "C2-整合构建(整合构建执行)",
    "c3": "C3-提炼复用原则(抽象与理论化)",
}

C_LABELS = {
    "Q1": ["多维度冲突", "多条件限制", "问题边界模糊"],
    "Q3": ["数据效用与美感形态冲突", "数据效用与设计师立场冲突"],
    "Q4": ["商业利润与隐私风险冲突", "封闭与开源冲突"],
    "Q5": ["技术与人本冲突", "国际通用与本土化冲突", "极简与有机冲突"],
    "Q6": ["商业导向与用户导向冲突", "利益与责任感冲突",
           "多数人效率至上与边缘群体优先冲突"],
}

SKIP_QIDS = {"Q8"}

# ── NL 模板 ───────────────────────────────────────────────

NL_TEMPLATES = {
    "strongly_forces_transition_to":
        "在{C}情境下，该设计师在「{item}」维度上持「{stance}」立场(β={coef})，"
        "这一特质具有绝对主导权，强力驱动其思维毫不犹豫地直接进入{O}状态。",
    "guides_transition_to":
        "在{C}情境下，该设计师在「{item}」维度上持「{stance}」立场(β={coef})，"
        "这一特质是其主要思考偏好，经过一定逻辑权衡后，顺理成章地将思维引导向{O}状态。",
    "weakly_shifts_to":
        "在{C}情境下，该设计师在「{item}」维度上持「{stance}」立场(β={coef})，"
        "当没有更强约束时，这一特质会使其轻微转向{O}状态。",
    "strictly_inhibits":
        "在{C}情境下，该设计师在「{item}」维度上持「{stance}」立场(β={coef})，"
        "这一特质对{O}状态产生强烈排斥力，思考时会明确拒绝进入该状态。",
}

# ── 工具函数 ──────────────────────────────────────────────

def c_to_nl(tags: list[str]) -> str:
    return "".join(f"【{t}】" for t in tags)


def _lookup(key: str) -> str:
    return PKVC_SUBTYPE_VERBOSE.get(key.strip().lower(), key)


def resolve_o_verbose(o_primary: str | None, o_subtype: str | None) -> str:
    if o_subtype:
        if "➡️" in o_subtype:
            parts = [p.strip() for p in o_subtype.split("➡️")]
            return " → ".join(_lookup(p) for p in parts)
        if o_subtype.lower() in PKVC_SUBTYPE_VERBOSE:
            return PKVC_SUBTYPE_VERBOSE[o_subtype.lower()]
    if o_primary and o_primary.strip() in PKVC_SUBTYPE_VERBOSE:
        return PKVC_SUBTYPE_VERBOSE[o_primary.strip()]
    return "未知状态"


def render_nl(predicate: str, c_raw, o_primary, o_subtype,
              item: str, stance: str, coef: float) -> str:
    c_text = c_to_nl(c_raw) if isinstance(c_raw, list) else (c_raw or "?")
    o_text = resolve_o_verbose(o_primary, o_subtype)
    return NL_TEMPLATES[predicate].format(
        C=c_text, item=item, stance=stance, coef=coef, O=o_text)


# ── 主提取 ────────────────────────────────────────────────

def extract(data: dict):
    quads = []
    skipped = []

    for key, entry in data.items():
        qid = entry["qid"]
        if qid in SKIP_QIDS:
            continue
        option = entry["option"]
        conflict = C_LABELS.get(qid, entry.get("C_formal_proposed") or entry.get("C_raw"))
        o_primary = entry.get("o_primary_hint")
        o_subtype = entry.get("o_subtype_hint")

        for trait in entry.get("s_primary_with_stance", []):
            p_val = trait["marginal_perm_p"]
            if p_val > 0.10:
                continue

            coef = trait["coef"]
            direction = trait["direction"]
            predicate = classify_predicate(p_val, coef, direction)

            if predicate is None:
                skipped.append({"key": key, "item": trait["item"],
                                "p": p_val, "coef": coef, "direction": direction})
                continue

            item = trait["item"]
            quad_id = f"{qid}_{option}__{item}"

            nl = render_nl(predicate, c_raw=conflict, o_primary=o_primary,
                           o_subtype=o_subtype, item=item,
                           stance=trait["stance"], coef=coef)

            quads.append({
                "id": quad_id,
                "qid": qid,
                "option": option,
                "C": conflict,
                "S_item": item,
                "S_stance": trait["stance"],
                "S_direction": direction,
                "S_coef": coef,
                "S_p": p_val,
                "P": predicate,
                "O": o_primary,
                "O_subtype": o_subtype,
                "nl": nl,
            })

    return quads, skipped


# ── 输出 ──────────────────────────────────────────────────

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    quads, skipped = extract(data)

    print(f"=== CAPS 四元组 (原始单库) ===")
    print(f"有效: {len(quads)} 条  |  跳过: {len(skipped)}")
    print()

    dirs = Counter(q["S_direction"] for q in quads)
    preds = Counter(q["P"] for q in quads)
    qids = Counter(q["qid"] for q in quads)
    print(f"  direction 分布: {dict(dirs)}")
    print(f"  predicate 分布: {dict(sorted(preds.items()))}")
    print(f"  qid 分布: {dict(sorted(qids.items()))}")
    print()

    # 抽样
    print("--- 样本 (前 3 条) ---")
    for q in quads[:3]:
        print(f"  [{q['id']}] S={q['S_stance']}({q['S_direction']}) P={q['P']} β={q['S_coef']}")
        print(f"    {q['nl']}")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(quads, f, ensure_ascii=False, indent=2)

    print(f"已写入: {OUT_PATH} ({len(quads)} 条)")


if __name__ == "__main__":
    main()
