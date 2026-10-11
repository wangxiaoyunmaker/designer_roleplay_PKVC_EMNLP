#!/bin/bash
# Chain 7 SFT inference conditions sequentially on Q7 seg1/2/3.
# 162 designer × 3 seg = 486 jobs per condition × 7 conditions = 3402 total jobs
# ~5s/job → ~4-5h total wall time.

set -e
cd /data/wy/Designer_RolePlay/0512技术方案

ADAPTER="${ADAPTER:-/data/wy/sft_v3_ckpts/stage_warmup_ep5}"
PY="${PY:-/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=.

mkdir -p reports/datav3
echo "[start] $(date)  adapter=$ADAPTER"

run_one() {
    local label="$1"; shift
    echo
    echo "=== [$label] $(date) ==="
    $PY scripts/run_sft_inference.py --adapter "$ADAPTER" --n 0 "$@"
}

# 1. v2 (full anchor)
run_one "1/7 v2" --variant v2 --output reports/datav3/sft_v2_inference.jsonl

# 2. v2_no_caps (variant=v2 minus caps module — user 澄清: 砍 caps module)
run_one "2/7 v2_no_caps" --variant v2 --modules pi,stance,self \
    --output reports/datav3/sft_v2_no_caps_inference.jsonl

# 3. v2_no_seg3_rule
run_one "3/7 v2_no_seg3_rule" --variant v2_no_seg3_rule \
    --output reports/datav3/sft_v2_no_seg3_rule_inference.jsonl

# 4. v2_no_beta_summary
run_one "4/7 v2_no_beta_summary" --variant v2_no_beta_summary \
    --output reports/datav3/sft_v2_no_beta_summary_inference.jsonl

# 5. v2_hybrid
run_one "5/7 v2_hybrid" --variant v2_hybrid \
    --output reports/datav3/sft_v2_hybrid_inference.jsonl

# 6. B0 (empty modules)
run_one "6/7 B0" --variant v2_hybrid --modules "" \
    --output reports/datav3/sft_b0_inference.jsonl

# 7. v2_plus_peer
run_one "7/7 v2_plus_peer" --variant v2 --with-peer \
    --output reports/datav3/sft_v2_plus_peer_inference.jsonl

echo
echo "[done] $(date)"
ls -la reports/datav3/sft_*_inference.jsonl | head -10
