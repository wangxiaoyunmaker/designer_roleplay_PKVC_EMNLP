"""Parse 5 expert-labeled rows from /Users/wy/Downloads/问卷结果标注.xlsx into structured
per-segment gold labels.

每个 user 的 Factor 激活列表 给出 N 个 process modules, 每 module 有 dominant + evidence
原话. 用 evidence substring match seg1/seg2/seg3 NL 来分配 module → seg, 然后聚合.

输出 data/processed/expert_gold/parsed.json, 结构:
{
  "users": [
    {
      "user_id": str,
      "seg1": {"text": ..., "dominant": "K", "sequence": ["K"], "modules": [...]},
      "seg2": {"text": ..., "dominant": "K", "sequence": ["K","V","K","K"], "modules": [...]},
      "seg3": {"text": ..., "dominant": "V", "sequence": ["V"], "modules": [...]},
      "criterion": "成本至上 ...",
      "overall_sequence": "K → K(V微) → K←V → K → V",
    },
    ...
  ]
}
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path


def extract_modules(long_a: str) -> list[dict]:
    """Parse Factor 激活列表 table: rows of (顺序, 模块, 主导, 次级, 原话)."""
    idx = long_a.find("Factor 激活列表")
    if idx < 0:
        return []
    # find end (next section)
    end = len(long_a)
    for stop in ("SADR 激活列表", "因子序列符号表示", "核心底线"):
        i = long_a.find(stop, idx + 10)
        if i > idx and i < end:
            end = i
    table = long_a[idx:end]
    lines = [l.strip() for l in table.split("\n") if l.strip()]
    modules = []
    for l in lines:
        # skip header & non-data
        if l.startswith("Factor") or l.startswith("顺序"):
            continue
        # split by tab (or multi-space)
        parts = re.split(r"\t+", l)
        if len(parts) < 5:
            continue
        try:
            order = int(parts[0].strip())
        except ValueError:
            continue
        modules.append({
            "order": order,
            "module": parts[1].strip(),
            "dominant": parts[2].strip(),
            "secondary": parts[3].strip(),
            "evidence": parts[4].strip(),
        })
    return modules


def extract_overall_sequence(long_a: str) -> str:
    """Extract the 因子序列符号表示 line."""
    idx = long_a.find("因子序列符号表示")
    if idx < 0:
        return ""
    line_end = long_a.find("\n", idx)
    if line_end < 0:
        line_end = idx + 200
    return long_a[idx:line_end].split("：", 1)[-1].split(":", 1)[-1].strip()


def extract_criterion(long_a: str) -> str:
    """Extract 核心底线 — first non-table line after 核心底线 section header."""
    # 核心底线 may appear twice (once as Factor 表 row, once as section header).
    # Find LAST occurrence — that's the section header in narrative.
    idx = long_a.rfind("核心底线")
    if idx < 0:
        return ""
    after = long_a[idx + len("核心底线"):]
    for line in after.split("\n"):
        line = line.strip()
        if not line:
            continue
        # skip table-row-shaped lines (contain tab-separated cells)
        if "\t" in line and line.split("\t")[0].strip() in ("P", "K", "V", "C", "—"):
            continue
        # skip table header
        if line.startswith("维度") or line.startswith("标注理据"):
            continue
        return line[:200]
    return ""


def normalize_dominant(d: str) -> str:
    """Map gold dominant string to canonical letter.
    e.g.  "K"  → "K"
          "V（表象）" → "V"
          "K ← V（被 K 收编）" → "K" (first letter is the dominant)
          "V₁" / "V₂" → "V"
          "C (V弱)" / "P(K)" → "P" (first letter)
    """
    s = d.strip()
    # find first P/K/V/C
    m = re.search(r"[PKVC]", s)
    return m.group(0) if m else ""


def map_module_to_seg(module: dict, seg_texts: dict[str, str], order_max: int) -> str:
    """Match module evidence to seg1/seg2/seg3 with order-based prior + substring tie-break.

    Priority:
      1. Module-name keyword (第一反应/直觉 → seg1; 总结/底线/最终/收束/锚定 → seg3).
      2. Substring match of quoted evidence phrases against seg texts.
      3. Order-based fallback (order 1 → seg1; last → seg3; else seg2).
    """
    name = module["module"]
    order = module["order"]

    # Priority 1: module name keywords
    seg1_kw = ("第一反应", "直觉", "面对委托", "角色代入")
    seg3_kw = ("最终", "底线锚定", "收束", "总结", "辩证综合", "核心底线")
    for kw in seg1_kw:
        if kw in name:
            return "seg1"
    for kw in seg3_kw:
        if kw in name:
            return "seg3"

    # Priority 2: substring match
    ev = module["evidence"]
    phrases = re.findall(r"[\"“”]([^\"“”]{4,})[\"“”]", ev)
    if not phrases:
        phrases = [ev[:30]]
    scores = {"seg1": 0, "seg2": 0, "seg3": 0}
    for ph in phrases:
        for n in range(min(len(ph), 12), 3, -1):
            matched_any = False
            for i in range(len(ph) - n + 1):
                substr = ph[i:i + n]
                for seg in ("seg1", "seg2", "seg3"):
                    if substr in seg_texts[seg]:
                        scores[seg] += n
                        matched_any = True
                        break
                if matched_any:
                    break
            if matched_any:
                break
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # Priority 3: order-based fallback
    if order == 1:
        return "seg1"
    if order == order_max:
        return "seg3"
    return "seg2"


def main():
    in_dir = Path("data/processed/expert_gold")
    rows = []
    for f in sorted(in_dir.glob("*.json")):
        if f.name == "parsed.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        uid = d["user_id"]
        # strip control chars from segments_json (some files have raw \n inside strings)
        seg_raw = d["segments_json"]
        seg_clean = re.sub(r"[\x00-\x1f]+", " ", seg_raw)
        try:
            segs_blob = json.loads(seg_clean)
        except json.JSONDecodeError:
            # try removing all control chars then escape backslash issues
            seg_clean2 = "".join(ch for ch in seg_raw if ord(ch) >= 32 or ch == " ")
            segs_blob = json.loads(seg_clean2)
        seg_texts = {k: segs_blob[k]["text"] for k in ("seg1", "seg2", "seg3")}

        modules = extract_modules(d["long_analysis"])
        overall_seq = extract_overall_sequence(d["long_analysis"])
        criterion_raw = extract_criterion(d["long_analysis"])

        # assign each module to a seg
        max_order = max((m["order"] for m in modules), default=1)
        for m in modules:
            m["seg"] = map_module_to_seg(m, seg_texts, max_order)
            m["dominant_letter"] = normalize_dominant(m["dominant"])

        # aggregate per seg
        per_seg = {}
        for seg in ("seg1", "seg2", "seg3"):
            seg_mods = [m for m in modules if m["seg"] == seg]
            sequence = [m["dominant_letter"] for m in sorted(seg_mods, key=lambda x: x["order"])]
            # dominant per seg: mode (most frequent), tie broken by first
            if sequence:
                cnt = Counter(sequence)
                top_n = cnt.most_common(1)[0][1]
                top_letters = [l for l, n in cnt.items() if n == top_n]
                if len(top_letters) == 1:
                    dom = top_letters[0]
                else:
                    # tie: prefer first state in sequence among ties
                    dom = next(l for l in sequence if l in top_letters)
            else:
                dom = ""
            per_seg[seg] = {
                "text": seg_texts[seg],
                "dominant": dom,
                "sequence": sequence,
                "modules": seg_mods,
            }

        rows.append({
            "user_id": uid,
            "qid": "Q7",
            "seg1": per_seg["seg1"],
            "seg2": per_seg["seg2"],
            "seg3": per_seg["seg3"],
            "criterion": criterion_raw,
            "overall_sequence": overall_seq,
        })

        print(f"=== {uid} ===")
        print(f"  overall: {overall_seq}")
        print(f"  seg1: dom={per_seg['seg1']['dominant']:1s}  seq={per_seg['seg1']['sequence']}")
        print(f"  seg2: dom={per_seg['seg2']['dominant']:1s}  seq={per_seg['seg2']['sequence']}")
        print(f"  seg3: dom={per_seg['seg3']['dominant']:1s}  seq={per_seg['seg3']['sequence']}")
        print(f"  criterion: {criterion_raw[:80]}")

    out_path = in_dir / "parsed.json"
    out_path.write_text(json.dumps({"users": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
