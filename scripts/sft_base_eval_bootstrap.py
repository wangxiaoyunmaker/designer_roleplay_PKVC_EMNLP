"""SFT vs BASE 11-metric eval with 1000-bootstrap 95% CI on 162 designer.

复用 scripts/final_eval_with_bootstrap.py 的所有 metric / bootstrap logic, 仅 override
BASELINES 列表为 14 SFT/BASE conditions.

Output:
  reports/datav3/sft_base_eval_bootstrap.md
  reports/datav3/sft_base_eval_bootstrap.jsonl
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.final_eval_with_bootstrap as feb


# Override BASELINES with 14 SFT + BASE conditions.
# Tuple format: (label, inference_jsonl, pkvc_eval_jsonl, baseline_filter)
# Auto-discover: only include baselines whose pkvc_eval exists
import os.path as _osp
_ALL_CANDIDATE_BASELINES = [
    # ── BASE Qwen2.5-7B via yunwu API (bf16) ──
    ("BASE_api_b0",                "reports/datav3/base_api_b0_inference.jsonl",                "reports/datav3/base_api_b0_pkvc_eval.jsonl",                None),
    ("BASE_api_b1",                "reports/datav3/base_api_b1_inference.jsonl",                "reports/datav3/base_api_b1_pkvc_eval.jsonl",                None),
    ("BASE_api_b2",                "reports/datav3/base_api_b2_inference.jsonl",                "reports/datav3/base_api_b2_pkvc_eval.jsonl",                None),
    ("BASE_api_b3",                "reports/datav3/base_api_b3_inference.jsonl",                "reports/datav3/base_api_b3_pkvc_eval.jsonl",                None),
    ("BASE_api_v2",                "reports/datav3/base_api_v2_inference.jsonl",                "reports/datav3/base_api_v2_pkvc_eval.jsonl",                None),
    ("BASE_api_v2_no_caps",        "reports/datav3/base_api_v2_no_caps_inference.jsonl",        "reports/datav3/base_api_v2_no_caps_pkvc_eval.jsonl",        None),
    ("BASE_api_v2_no_seg3_rule",   "reports/datav3/base_api_v2_no_seg3_rule_inference.jsonl",   "reports/datav3/base_api_v2_no_seg3_rule_pkvc_eval.jsonl",   None),
    ("BASE_api_v2_no_beta_summary","reports/datav3/base_api_v2_no_beta_summary_inference.jsonl","reports/datav3/base_api_v2_no_beta_summary_pkvc_eval.jsonl",None),
    ("BASE_api_v2_hybrid",         "reports/datav3/base_api_v2_hybrid_inference.jsonl",         "reports/datav3/base_api_v2_hybrid_pkvc_eval.jsonl",         None),
    ("BASE_api_v2_plus_peer",      "reports/datav3/base_api_v2_plus_peer_inference.jsonl",      "reports/datav3/base_api_v2_plus_peer_pkvc_eval.jsonl",      None),
    # ── BASE Qwen2.5-7B local int4 (legacy fallback) ──
    ("BASE_local_b0",                "reports/datav3/base_b0_inference.jsonl",                "reports/datav3/base_b0_pkvc_eval.jsonl",                None),
    ("BASE_local_v2",                "reports/datav3/base_v2_inference.jsonl",                "reports/datav3/base_v2_pkvc_eval.jsonl",                None),
    # ── SFT_ep5 (over-fit) ──
    ("SFT_ep5_v2",             "reports/datav3/sft_v2_inference.jsonl",                "reports/datav3/sft_v2_pkvc_eval.jsonl",                None),
    # ── SFT_v3c (r=16, fixed) — int4 ──
    ("SFT_v3c_b0",                "reports/datav3/sft_v3c_b0_inference.jsonl",                "reports/datav3/sft_v3c_b0_pkvc_eval.jsonl",                None),
    ("SFT_v3c_b1",                "reports/datav3/sft_v3c_b1_inference.jsonl",                "reports/datav3/sft_v3c_b1_pkvc_eval.jsonl",                None),
    ("SFT_v3c_b2",                "reports/datav3/sft_v3c_b2_inference.jsonl",                "reports/datav3/sft_v3c_b2_pkvc_eval.jsonl",                None),
    ("SFT_v3c_b3",                "reports/datav3/sft_v3c_b3_inference.jsonl",                "reports/datav3/sft_v3c_b3_pkvc_eval.jsonl",                None),
    ("SFT_v3c_v2",                "reports/datav3/sft_v3c_v2_inference.jsonl",                "reports/datav3/sft_v3c_v2_pkvc_eval.jsonl",                None),
    ("SFT_v3c_v2_no_caps",        "reports/datav3/sft_v3c_v2_no_caps_inference.jsonl",        "reports/datav3/sft_v3c_v2_no_caps_pkvc_eval.jsonl",        None),
    ("SFT_v3c_v2_no_seg3_rule",   "reports/datav3/sft_v3c_v2_no_seg3_rule_inference.jsonl",   "reports/datav3/sft_v3c_v2_no_seg3_rule_pkvc_eval.jsonl",   None),
    ("SFT_v3c_v2_no_beta_summary","reports/datav3/sft_v3c_v2_no_beta_summary_inference.jsonl","reports/datav3/sft_v3c_v2_no_beta_summary_pkvc_eval.jsonl",None),
    ("SFT_v3c_v2_hybrid",         "reports/datav3/sft_v3c_v2_hybrid_inference.jsonl",         "reports/datav3/sft_v3c_v2_hybrid_pkvc_eval.jsonl",         None),
    ("SFT_v3c_v2_plus_peer",      "reports/datav3/sft_v3c_v2_plus_peer_inference.jsonl",      "reports/datav3/sft_v3c_v2_plus_peer_pkvc_eval.jsonl",      None),
]
SFT_BASE_BASELINES = [b for b in _ALL_CANDIDATE_BASELINES if _osp.exists(b[2])]
print(f"[auto-discover] {len(SFT_BASE_BASELINES)} baselines with pkvc_eval present (of {len(_ALL_CANDIDATE_BASELINES)} candidates)")

# 5-metric subset (EMNLP paper 锁的)
KEEP_METRICS = {
    "seg2_macro_TED", "seg2_non_V_recall", "seg2_chain_diversity",
    "seg1_trait_pred_rho", "seg3_criterion_uniqueness",
}


def patched_main():
    # Override BASELINES + output paths, then call original main()
    feb.BASELINES = SFT_BASE_BASELINES
    # 5-metric subset filter
    feb.METRICS = [(n, s, fn) for (n, s, fn) in feb.METRICS if n in KEEP_METRICS]
    print(f"[filter] kept {len(feb.METRICS)} metrics: {[n for n,_,_ in feb.METRICS]}")
    # Monkey-patch output paths
    import scripts.final_eval_with_bootstrap as m

    original_main = m.main
    # Patch path constants by editing source — easier: just rerun main() after override
    # The output paths in main() are hard-coded; we override after main() writes them
    original_main()

    # Rename outputs from final_eval_bootstrap.* to sft_base_eval_bootstrap.*
    import shutil
    for ext in ("md", "jsonl"):
        src = Path(f"reports/datav3/final_eval_bootstrap.{ext}")
        dst = Path(f"reports/datav3/sft_base_eval_bootstrap.{ext}")
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"[rename] {src} → {dst}")


if __name__ == "__main__":
    patched_main()
