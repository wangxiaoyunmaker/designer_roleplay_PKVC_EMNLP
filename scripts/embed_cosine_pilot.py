"""A: embedding cosine pilot — re-eval existing roleplay pilot via NL semantic similarity.

复用 reports/datav3/roleplay_pilot/results.jsonl (10 user × {real, B0, B1, B2}):
  1. embed 每个 NL (text-embedding-3-small)
  2. cosine(real, B0/B1/B2) per user
  3. avg + B1/B2 lift vs B0

不依赖 v8 extractor, 不依赖 PKVC label, 直接看语义距离.
RAG 若有效: B2 cosine > B1 cosine > B0 cosine.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI


def embed_batch(client: OpenAI, texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Batch embed, return [N, D] array."""
    r = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in r.data])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="reports/datav3/roleplay_pilot")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY")
        return 1
    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")

    results_path = Path(args.input_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found, run scripts/roleplay_pilot.py first")
        return 1

    results = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[data] loaded {len(results)} pilot results from {args.input_dir}")

    # batch all NLs (含 b3 quad-context, 若存在)
    all_texts = []
    indices: list[tuple[str, str]] = []  # (user_id, baseline_name)
    for r in results:
        gens = r["generations"]
        for bname in ["real_seg1", "b0", "b1", "b2", "b3"]:
            nl = gens.get(bname, "")
            if nl:
                all_texts.append(nl)
                indices.append((r["user_id"], bname))

    print(f"[embed] embedding {len(all_texts)} NLs ({sum(len(t) for t in all_texts)} chars total)...")
    client = OpenAI(api_key=api_key, base_url=base_url)
    # batch in chunks of 50 (under token limit)
    embeddings = []
    BATCH = 20
    for i in range(0, len(all_texts), BATCH):
        chunk = all_texts[i : i + BATCH]
        emb_chunk = embed_batch(client, chunk)
        embeddings.append(emb_chunk)
        print(f"  chunk {i//BATCH + 1}/{(len(all_texts)+BATCH-1)//BATCH}: shape {emb_chunk.shape}")
    embeddings = np.vstack(embeddings)
    print(f"[embed] full shape: {embeddings.shape}")

    # build (user_id → {bname: embedding})
    by_user: dict[str, dict[str, np.ndarray]] = {}
    for (uid, bname), emb in zip(indices, embeddings):
        by_user.setdefault(uid, {})[bname] = emb

    # cosine per user
    cosines: dict[str, list[float]] = {"b0": [], "b1": [], "b2": [], "b3": []}
    per_user_rows = []
    for uid, d in by_user.items():
        if "real_seg1" not in d:
            continue
        real = d["real_seg1"]
        row = {"user_id": uid[:8]}
        for bname in ["b0", "b1", "b2", "b3"]:
            if bname in d:
                c = cosine(real, d[bname])
                cosines[bname].append(c)
                row[bname] = round(c, 4)
        per_user_rows.append(row)

    # summary
    print(f"\n=== RESULTS ({len(per_user_rows)} users) ===\n")
    print("per-user cosine (real seg1 vs generated):")
    print(f"  {'user':<10} | {'b0':<7} | {'b1':<7} | {'b2':<7} | {'b3':<7} | b3-b2")
    for r in per_user_rows:
        b0 = r.get("b0", 0)
        b1 = r.get("b1", 0)
        b2 = r.get("b2", 0)
        b3 = r.get("b3", 0)
        b3_minus_b2 = (b3 - b2) if b3 else 0
        b3_str = f"{b3:.4f}" if b3 else "  —   "
        b3_diff_str = f"{b3_minus_b2:+.4f}" if b3 else "  —  "
        print(f"  {r['user_id']:<10} | {b0:.4f} | {b1:.4f} | {b2:.4f} | {b3_str} | {b3_diff_str}")

    print("\nAvg cosine:")
    b0_avg = float(np.mean(cosines["b0"])) if cosines["b0"] else 0
    b1_avg = float(np.mean(cosines["b1"])) if cosines["b1"] else 0
    b2_avg = float(np.mean(cosines["b2"])) if cosines["b2"] else 0
    b3_avg = float(np.mean(cosines["b3"])) if cosines["b3"] else None
    print(f"  B0 (cold):                          {b0_avg:.4f}")
    print(f"  B1 (profile):                       {b1_avg:.4f}  lift vs B0: {b1_avg - b0_avg:+.4f} ({(b1_avg-b0_avg)/b0_avg*100:+.2f}%)")
    print(f"  B2 (profile + raw-NL RAG):          {b2_avg:.4f}  lift vs B0: {b2_avg - b0_avg:+.4f} ({(b2_avg-b0_avg)/b0_avg*100:+.2f}%)")
    if b3_avg is not None:
        print(f"  B3 (profile + quad-context RAG):    {b3_avg:.4f}  lift vs B0: {b3_avg - b0_avg:+.4f} ({(b3_avg-b0_avg)/b0_avg*100:+.2f}%)")
        print(f"\nKey deltas:")
        print(f"  RAG-vs-profile  B2 - B1: {b2_avg - b1_avg:+.4f}")
        print(f"  RAG-vs-profile  B3 - B1: {b3_avg - b1_avg:+.4f}")
        print(f"  quad-vs-rawNL   B3 - B2: {b3_avg - b2_avg:+.4f}  ← CAPS quad context 是否更强?")

    # verdict
    print("\n=== VERDICT ===")
    if b3_avg is not None:
        quad_vs_rawnl = b3_avg - b2_avg
        if quad_vs_rawnl > 0.005:
            verdict = f"✅ CAPS quad RAG (B3) > raw NL RAG (B2): +{quad_vs_rawnl:.4f} — quad context 有效升级"
        elif quad_vs_rawnl > 0:
            verdict = f"⚠ CAPS quad RAG ≈ raw NL RAG: +{quad_vs_rawnl:.4f} — 微弱差, 不显著"
        else:
            verdict = f"❌ CAPS quad RAG < raw NL RAG: {quad_vs_rawnl:.4f} — quad 结构化反而损失风格信号"
    else:
        verdict = "(no B3 data)"
    print(verdict)

    # save
    out = {
        "n_users": len(per_user_rows),
        "avg_cosine": {"b0": b0_avg, "b1": b1_avg, "b2": b2_avg, "b3": b3_avg},
        "lift_vs_b0": {
            "b1": b1_avg - b0_avg, "b2": b2_avg - b0_avg,
            "b3": (b3_avg - b0_avg) if b3_avg is not None else None,
        },
        "rag_lift_b2_vs_b1": b2_avg - b1_avg,
        "rag_lift_b3_vs_b1": (b3_avg - b1_avg) if b3_avg is not None else None,
        "quad_vs_rawnl_b3_vs_b2": (b3_avg - b2_avg) if b3_avg is not None else None,
        "per_user": per_user_rows,
        "verdict": verdict,
        "embedding_model": "text-embedding-3-small",
        "input_dir": args.input_dir,
    }
    out_path = Path(args.input_dir) / "embed_cosine.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
