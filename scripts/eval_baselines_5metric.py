"""5-metric comprehensive evaluation for 3 baselines.

Metrics:
  1. Syntactic similarity — 5-dim stats (字数/句数/逗号/问号/省略号 density) cosine vs GT
  2. Semantic similarity — text-embedding-3-small cosine
  3. Pragmatic similarity — marker density vector (hedge/modal/stance/discourse) cosine vs GT
  4. PKVC 路径保真度 — sequence Jaccard + normalized edit distance vs GT sequence
  5. 人格一致性 — cosine(simulated_emb, designer_Q3Q6_NL_mean_emb)

Inputs:
  - reports/datav3/baselines_q7_inference.jsonl  (1458 simulated NL + GT NL)
  - reports/datav3/baselines_q7_pkvc_eval.jsonl  (1457 sim_NL → extracted PKVC seq)
  - data_v3/processed/q1q6_extracted_v8.7_full.jsonl  (designer Q1-Q6 NL → persona center)

Embedding: text-embedding-3-small (OpenAI), batch up to 2048 inputs. Cache to file.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_CACHE = Path("data_v3/processed/embedding_cache_text_embed_3_small.jsonl")
EMBED_BATCH_SIZE = 200  # text-embedding API recommended batch


# ============================================================
# 1. Syntactic features
# ============================================================

def syntactic_features(text: str) -> np.ndarray:
    """5-dim vector: [char_count, sentence_count, comma_density, qmark_density, ellipsis_density]."""
    if not text:
        return np.zeros(5)
    n_chars = len(text)
    sents = re.split(r"[。！？.!?]+", text)
    n_sents = max(len([s for s in sents if s.strip()]), 1)
    n_commas = text.count(",") + text.count("，")
    n_q = text.count("?") + text.count("？")
    n_ellip = text.count("…") + text.count("...")
    # density per 100 chars + normalize char count to range
    return np.array([
        min(n_chars / 500.0, 2.0),       # length proxy, capped
        min(n_sents / 10.0, 2.0),        # sentence count
        n_commas * 100 / n_chars,
        n_q * 100 / n_chars,
        n_ellip * 100 / n_chars,
    ])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 3. Pragmatic markers
# ============================================================

PRAGMATIC_MARKERS = {
    "hedge": [
        "我觉得", "可能", "也许", "大概", "应该", "大约", "或许", "似乎", "好像",
        "我感觉", "感觉", "我认为", "我想",
    ],
    "modal": [
        "必须", "应当", "应该", "不能", "不可以", "一定", "肯定", "绝对", "千万",
        "需要", "得", "必要", "禁止",
    ],
    "stance": [
        "底线", "原则", "立场", "信念", "核心", "本质", "宗旨", "首要", "前提",
        "出发点", "根本", "最大", "最核心", "重要", "关键",
    ],
    "discourse": [
        "首先", "其次", "再次", "最后", "然后", "接着", "另外", "但是", "不过",
        "因此", "所以", "总之", "总而言之", "综上", "之后", "前面", "后面",
    ],
}


def pragmatic_features(text: str) -> np.ndarray:
    """4-dim density vector for 4 marker categories (per 100 chars)."""
    if not text:
        return np.zeros(4)
    n_chars = max(len(text), 1)
    out = []
    for cat in ("hedge", "modal", "stance", "discourse"):
        count = sum(text.count(m) for m in PRAGMATIC_MARKERS[cat])
        out.append(count * 100 / n_chars)
    return np.array(out)


# ============================================================
# 4. PKVC path fidelity
# ============================================================

def levenshtein(a: list, b: list) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): dp[i][0] = i
    for j in range(len(b) + 1): dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len(a)][len(b)]


def transition_set(seq: list) -> set:
    """Pairs of consecutive states. e.g. [V,K,P,V] → {(V,K),(K,P),(P,V)}."""
    return {(seq[i], seq[i+1]) for i in range(len(seq) - 1)}


def lcs_length(a: list, b: list) -> int:
    """Longest Common Subsequence length (DP, O(|a|·|b|))."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[len(a)][len(b)]


def path_fidelity(pred_seq: list, gt_seq: list) -> dict:
    """PKVC Trajectory Fidelity 6 sub-metrics.

    严格按用户定义实现 TED + STA:
      - **TED** (Topological Edit Distance): raw Levenshtein integer distance,
        把 pred 序列变成 gt 序列所需的最少 增/删/改 操作. lower = better.
      - **STA** (State Transition Accuracy): gt 出现的特定 bigram 跳转 (X,Y),
        在 pred 中命中的比例. = |gt_bigrams ∩ pred_bigrams| / |gt_bigrams|.
        asymmetric recall (focus on gt's transitions). higher = better.

    其余 4 个保留作参考:
      - jaccard          state set 交并比 (loose, 不顾顺序/重复, symmetric)
      - norm_edit        1 - TED / max_len (顺序敏感, 归一到 [0,1])
      - transition_jacc  bigram set Jaccard (symmetric, intersection/union)
      - lcs_f1           LCS × 2 / (|pred|+|gt|) (顺序敏感 F1-like)
    """
    if not pred_seq and not gt_seq:
        return {"jaccard": 1.0, "norm_edit": 1.0, "transition_jacc": 1.0, "lcs_f1": 1.0,
                "ted": 0, "sta": 1.0}
    if not pred_seq or not gt_seq:
        return {"jaccard": 0.0, "norm_edit": 0.0, "transition_jacc": 0.0, "lcs_f1": 0.0,
                "ted": max(len(pred_seq), len(gt_seq)), "sta": 0.0}

    jacc = len(set(pred_seq) & set(gt_seq)) / len(set(pred_seq) | set(gt_seq))

    # === TED (raw int) ===
    ted = levenshtein(pred_seq, gt_seq)
    max_len = max(len(pred_seq), len(gt_seq))
    norm_edit = 1.0 - ted / max_len if max_len else 1.0

    # transition set + jaccard (symmetric, 参考)
    t_pred, t_gt = transition_set(pred_seq), transition_set(gt_seq)
    if not t_pred and not t_gt:
        # 两个 sequence 都 length 1 → fall back to sequence equality
        trans_jacc = 1.0 if pred_seq == gt_seq else 0.0
    elif not t_pred or not t_gt:
        trans_jacc = 0.0
    else:
        trans_jacc = len(t_pred & t_gt) / len(t_pred | t_gt)

    # === STA (asymmetric recall on gt bigrams) ===
    # focus on "did pred reproduce the specific transitions human (gt) exhibited"
    if not t_gt:
        # gt is length 1, no bigrams. fall back: pred 也 length 1 且字母同 → 1.0; else → 0.0
        sta = 1.0 if pred_seq == gt_seq else 0.0
    else:
        sta = len(t_pred & t_gt) / len(t_gt)

    lcs_len = lcs_length(pred_seq, gt_seq)
    lcs_f1 = 2 * lcs_len / (len(pred_seq) + len(gt_seq))

    return {
        "jaccard": jacc, "norm_edit": norm_edit,
        "transition_jacc": trans_jacc, "lcs_f1": lcs_f1,
        "ted": ted, "sta": sta,
    }


# ============================================================
# 2 + 5. Embeddings (with cache)
# ============================================================

def text_key(text: str) -> str:
    """Stable hash key for cache."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    cache = {}
    if EMBED_CACHE.exists():
        for line in EMBED_CACHE.open():
            r = json.loads(line)
            cache[r["key"]] = r["vec"]
    return cache


def save_cache_append(records: list[dict]) -> None:
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with EMBED_CACHE.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def embed_texts(texts: list[str], cache: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Embed (with cache). Returns dict[text → np.ndarray]."""
    # Dedup texts
    unique = list(dict.fromkeys(texts))
    new_texts = [t for t in unique if text_key(t) not in cache]
    print(f"  cache: {len(cache)} entries; new texts to embed: {len(new_texts)}")
    if new_texts:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: set OPENAI_API_KEY"); sys.exit(1)
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"))
        new_records = []
        for i in range(0, len(new_texts), EMBED_BATCH_SIZE):
            batch = new_texts[i:i+EMBED_BATCH_SIZE]
            print(f"  embedding batch {i//EMBED_BATCH_SIZE + 1}/{(len(new_texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE} ({len(batch)} texts)...", flush=True)
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for txt, item in zip(batch, resp.data):
                vec = item.embedding
                cache[text_key(txt)] = vec
                new_records.append({"key": text_key(txt), "vec": vec})
        save_cache_append(new_records)
    return {t: np.array(cache[text_key(t)]) for t in unique}


# ============================================================
# Main: 5-metric eval
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", nargs="+",
                    default=["reports/datav3/baselines_q7_inference.jsonl"],
                    help="one or more inference jsonl files (will be concat)")
    ap.add_argument("--pkvc-eval", nargs="*",
                    default=["reports/datav3/baselines_q7_pkvc_eval.jsonl"],
                    help="one or more PKVC eval jsonl files (will be concat). "
                         "Required if --pkvc-source=extracted or both.")
    ap.add_argument("--pkvc-source", default="extracted",
                    choices=["extracted", "claimed", "both"],
                    help="path metric source: "
                         "'extracted' = v8.7 extract on simulated_nl (from --pkvc-eval); "
                         "'claimed' = inference jsonl claimed_pkvc.state_sequence; "
                         "'both' = compute both + add claimed/extracted dom agreement metric.")
    ap.add_argument("--q16", default="data_v3/processed/q1q6_extracted_v8.7_full.jsonl")
    ap.add_argument("--output", default="reports/datav3/baselines_q7_5metric.md")
    ap.add_argument("--output-jsonl", default="reports/datav3/baselines_q7_5metric_perpair.jsonl")
    ap.add_argument("--baselines", default="B1,B2,B3",
                    help="comma-separated baseline names to include in report tables")
    args = ap.parse_args()

    baselines_to_show = args.baselines.split(",")
    use_claimed = args.pkvc_source in ("claimed", "both")
    use_extracted = args.pkvc_source in ("extracted", "both")

    print("[load] inference...")
    inf_rows = []
    for p in args.inference:
        rows = [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r.get("ok") and r.get("simulated_nl")]
        print(f"  {p}: +{len(rows)} ok rows")
        inf_rows.extend(rows)
    print(f"  total: {len(inf_rows)} ok inference rows")

    pkvc_map = {}
    if use_extracted:
        print("[load] PKVC eval (for path fidelity, extracted source)...")
        pkvc_rows = []
        for p in args.pkvc_eval:
            rows = [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]
            pkvc_rows.extend(rows)
        pkvc_map = {(r["baseline"], r["designer_id"], r["scenario"]): r for r in pkvc_rows if r["ok"]}
        print(f"  {len(pkvc_map)} pkvc entries")

    print("[load] Q1-Q6 NL (for persona center)...")
    q16_rows = [json.loads(l) for l in Path(args.q16).open(encoding="utf-8") if l.strip()]
    designer_q16_nls = defaultdict(list)
    for r in q16_rows:
        if r.get("ok") and r.get("text"):
            designer_q16_nls[r["user_id"]].append(r["text"])
    print(f"  {len(designer_q16_nls)} designers with Q1-Q6 NL (avg {sum(len(v) for v in designer_q16_nls.values())/len(designer_q16_nls):.1f} NLs each)")

    # ============= prepare all unique texts to embed =============
    print("\n[embed] preparing texts...")
    all_texts = set()
    for r in inf_rows:
        all_texts.add(r["simulated_nl"])
        if r.get("ground_truth_nl"):
            all_texts.add(r["ground_truth_nl"])
    for nls in designer_q16_nls.values():
        for nl in nls:
            all_texts.add(nl)
    print(f"  {len(all_texts)} unique texts to embed")
    cache = load_cache()
    emb_map = embed_texts(list(all_texts), cache)
    print(f"  done. embedding dim: {len(next(iter(emb_map.values())))}")

    # designer persona center = mean embedding of their Q1-Q6 NL
    print("\n[persona] building designer persona centers...")
    persona_center: dict[str, np.ndarray] = {}
    for did, nls in designer_q16_nls.items():
        embs = [emb_map[nl] for nl in nls if nl in emb_map]
        if embs:
            persona_center[did] = np.mean(embs, axis=0)
    print(f"  {len(persona_center)} designer centers")

    # ============= compute 5 metrics per row =============
    print("\n[compute] per-row 5 metrics...")
    per_row = []
    for i, r in enumerate(inf_rows):
        sim_text = r["simulated_nl"]
        gt_text = r.get("ground_truth_nl", "")
        did = r["designer_id"]
        scenario = r["scenario"]
        baseline = r["baseline"]
        if not gt_text:
            continue
        # 1. syntactic
        sim_syn = syntactic_features(sim_text)
        gt_syn = syntactic_features(gt_text)
        syn_cos = cosine(sim_syn, gt_syn)
        # 2. semantic
        sim_emb = emb_map.get(sim_text)
        gt_emb = emb_map.get(gt_text)
        sem_cos = cosine(sim_emb, gt_emb) if (sim_emb is not None and gt_emb is not None) else 0.0
        # 3. pragmatic
        sim_prag = pragmatic_features(sim_text)
        gt_prag = pragmatic_features(gt_text)
        prag_cos = cosine(sim_prag, gt_prag)
        # 4. PKVC path fidelity (6 sub-metrics 含 TED + STA)
        gt_seq = r.get("ground_truth_pkvc", {}).get("sequence", [])
        empty_path = {"jaccard": None, "norm_edit": None, "transition_jacc": None,
                      "lcs_f1": None, "ted": None, "sta": None}

        # claimed source — inference jsonl 的 claimed_pkvc.state_sequence
        path_claimed = empty_path.copy()
        claimed_pkvc = r.get("claimed_pkvc")
        if use_claimed and claimed_pkvc and gt_seq:
            claimed_seq = claimed_pkvc.get("state_sequence", [])
            if claimed_seq:
                path_claimed = path_fidelity(claimed_seq, gt_seq)

        # extracted source — pkvc_eval jsonl 的 v8.7 extract
        path_extracted = empty_path.copy()
        if use_extracted:
            pkvc_r = pkvc_map.get((baseline, did, scenario))
            if pkvc_r and gt_seq:
                pred_seq = pkvc_r["pred"]["state_sequence"]
                path_extracted = path_fidelity(pred_seq, gt_seq)

        # Pick primary path source for downstream metrics
        if args.pkvc_source == "claimed":
            path = path_claimed
        elif args.pkvc_source == "extracted":
            path = path_extracted
        else:  # both → use claimed as primary, keep extracted as secondary
            path = path_claimed

        # Dominant agreement (only meaningful in 'both' mode)
        claimed_dom = (claimed_pkvc or {}).get("dominant_factor")
        extracted_dom = None
        if use_extracted:
            pkvc_r = pkvc_map.get((baseline, did, scenario))
            if pkvc_r:
                extracted_dom = pkvc_r["pred"].get("dominant_factor")
        self_consistency = None
        if args.pkvc_source == "both" and claimed_dom and extracted_dom:
            self_consistency = 1.0 if claimed_dom[0] == extracted_dom[0] else 0.0
        # 5. persona consistency
        center = persona_center.get(did)
        persona_cos = cosine(sim_emb, center) if (sim_emb is not None and center is not None) else None

        per_row.append({
            "baseline": baseline, "designer_id": did, "scenario": scenario,
            "syntactic_cos": syn_cos,
            "semantic_cos": sem_cos,
            "pragmatic_cos": prag_cos,
            # path metric (primary source)
            "path_jaccard": path["jaccard"],
            "path_norm_edit": path["norm_edit"],
            "path_transition_jacc": path["transition_jacc"],
            "path_lcs_f1": path["lcs_f1"],
            "ted": path["ted"],
            "sta": path["sta"],
            # path metric (claimed/extracted explicit for both mode)
            "path_jaccard_claimed": path_claimed["jaccard"],
            "ted_claimed": path_claimed["ted"],
            "sta_claimed": path_claimed["sta"],
            "path_jaccard_extracted": path_extracted["jaccard"],
            "ted_extracted": path_extracted["ted"],
            "sta_extracted": path_extracted["sta"],
            # claimed-extracted PKVC agreement (self-consistency)
            "claimed_dom": claimed_dom[0] if claimed_dom else None,
            "extracted_dom": extracted_dom[0] if extracted_dom else None,
            "self_consistency": self_consistency,
            "persona_cos": persona_cos,
        })

    print(f"  {len(per_row)} per-row results")
    Path(args.output_jsonl).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_row), encoding="utf-8"
    )
    print(f"  [saved] {args.output_jsonl}")

    # ============= aggregate per baseline =============
    print("\n[aggregate] per baseline...")
    md = ["# Q7 3-Baseline 5-metric 综合评测\n"]
    md.append(f"输入: {args.inference}\n")
    md.append(f"per-pair 结果: {args.output_jsonl} ({len(per_row)} rows)\n")

    md.append("## 主表 (mean ± std)\n")
    md.append("### Core (用户严格定义 trajectory fidelity)\n")
    md.append("| Baseline | **TED** ↓ | **STA** ↑ | PKVC dom acc ↑ |")
    md.append("|---|---|---|---|")
    # compute dom acc per baseline from claimed or extracted (per --pkvc-source)
    pkvc_dom_acc = {}
    for b in baselines_to_show:
        pairs = []
        for r in inf_rows:
            if r.get("baseline") != b: continue
            gt_dom = r.get("ground_truth_pkvc", {}).get("dominant")
            if not gt_dom: continue
            # Pick pred source
            if args.pkvc_source == "claimed":
                pred_dom = (r.get("claimed_pkvc") or {}).get("dominant_factor")
            elif args.pkvc_source == "extracted":
                pkvc_r = pkvc_map.get((b, r["designer_id"], r["scenario"]))
                pred_dom = pkvc_r["pred"]["dominant_factor"] if pkvc_r else None
            else:  # both → use claimed as primary
                pred_dom = (r.get("claimed_pkvc") or {}).get("dominant_factor")
            if pred_dom:
                pairs.append((gt_dom[0], pred_dom[0]))
        pkvc_dom_acc[b] = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    for b in baselines_to_show:
        rows = [r for r in per_row if r["baseline"] == b]
        if not rows: continue
        def stats(key):
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals: return ("-", "-")
            return (f"{np.mean(vals):.3f}", f"{np.std(vals):.3f}")
        ted_m, ted_s = stats("ted")
        sta_m, sta_s = stats("sta")
        md.append(f"| {b} | {ted_m}±{ted_s} | {sta_m}±{sta_s} | {pkvc_dom_acc[b]:.1%} |")
    md.append("")

    md.append("### Auxiliary path metrics + lexical/embedding\n")
    md.append("| Baseline | Syntactic | Semantic | Pragmatic | Path Jaccard | Path Edit norm | Path Trans Jacc | Path LCS F1 | Persona Cos |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for b in baselines_to_show:
        rows = [r for r in per_row if r["baseline"] == b]
        if not rows: continue
        def stats(key):
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals: return ("-", "-")
            return (f"{np.mean(vals):.3f}", f"{np.std(vals):.3f}")
        syn_m, syn_s = stats("syntactic_cos")
        sem_m, sem_s = stats("semantic_cos")
        prag_m, prag_s = stats("pragmatic_cos")
        pj_m, pj_s = stats("path_jaccard")
        pe_m, pe_s = stats("path_norm_edit")
        pt_m, pt_s = stats("path_transition_jacc")
        pl_m, pl_s = stats("path_lcs_f1")
        per_m, per_s = stats("persona_cos")
        md.append(f"| {b} | {syn_m}±{syn_s} | {sem_m}±{sem_s} | {prag_m}±{prag_s} | "
                  f"{pj_m}±{pj_s} | {pe_m}±{pe_s} | {pt_m}±{pt_s} | {pl_m}±{pl_s} | "
                  f"{per_m}±{per_s} |")
    md.append("")

    # per-seg
    md.append("## Per-seg 拆解 (mean)\n")
    for sc in ("Q7_seg1", "Q7_seg2", "Q7_seg3"):
        md.append(f"### {sc}\n")
        md.append("| Baseline | TED ↓ | STA ↑ | Syn | Sem | Prag | Path Jacc | Path Edit | Path Trans | Path LCS | Persona |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for b in baselines_to_show:
            rows = [r for r in per_row if r["baseline"] == b and r["scenario"] == sc]
            if not rows: continue
            def m(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return f"{np.mean(vals):.3f}" if vals else "-"
            md.append(f"| {b} | {m('ted')} | {m('sta')} | "
                      f"{m('syntactic_cos')} | {m('semantic_cos')} | {m('pragmatic_cos')} | "
                      f"{m('path_jaccard')} | {m('path_norm_edit')} | "
                      f"{m('path_transition_jacc')} | {m('path_lcs_f1')} | "
                      f"{m('persona_cos')} |")
        md.append("")

    # print summary to stdout too
    print("\n=== summary (overall mean) ===")
    for b in baselines_to_show:
        rows = [r for r in per_row if r["baseline"] == b]
        if not rows: continue
        def m(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return np.mean(vals) if vals else 0
        print(f"  {b}: TED={m('ted'):.3f} ↓  STA={m('sta'):.3f} ↑")
        print(f"     syn={m('syntactic_cos'):.3f} sem={m('semantic_cos'):.3f} prag={m('pragmatic_cos'):.3f}")
        print(f"     path_jacc={m('path_jaccard'):.3f} path_edit_norm={m('path_norm_edit'):.3f} "
              f"path_trans_jacc={m('path_transition_jacc'):.3f} path_lcs={m('path_lcs_f1'):.3f}")
        print(f"     persona={m('persona_cos'):.3f}")

    Path(args.output).write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
