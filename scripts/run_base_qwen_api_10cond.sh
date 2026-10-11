#!/bin/bash
# BASE Qwen2.5-7B-Instruct via yunwu.ai API (no local GPU needed).
# 10 conditions, 5 parallel per wave, 2 waves.
# Each condition: 162 designer × 3 seg = 486 calls, concurrency=8 internal.
# Total: ~20-30 min wall, ~$3-5 cost.

set -u
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
export OPENAI_API_KEY
export OPENAI_BASE_URL='https://yunwu.ai/v1'
MODEL=qwen2.5-7b-instruct
LOG_DIR=/tmp/base_qwen_api_logs
mkdir -p $LOG_DIR

echo "[start] $(date)  model=$MODEL"

run_one() {
    local name="$1"; shift
    local outfile="reports/datav3/base_api_${name}_inference.jsonl"
    local logfile="$LOG_DIR/${name}.log"
    echo "  $name → $outfile"
    .venv/bin/python scripts/run_b6_sample.py \
        --model $MODEL --n 0 --concurrency 8 \
        --output "$outfile" "$@" > "$logfile" 2>&1 &
}

# Wave 1: 5 conditions
echo "=== Wave 1 $(date) ==="
run_one v2                  --variant v2
run_one v2_no_caps          --variant v2 --modules pi,stance,self
run_one v2_no_seg3_rule     --variant v2_no_seg3_rule
run_one v2_no_beta_summary  --variant v2_no_beta_summary
run_one v2_hybrid           --variant v2_hybrid
wait
echo "=== Wave 1 done $(date) ==="

# Wave 2: 5 conditions (B0/B1/B2/B3 ladder + plus_peer)
echo "=== Wave 2 $(date) ==="
run_one b0                  --variant v2_hybrid --modules ""
run_one b1                  --variant v2_hybrid --modules pi
run_one b2                  --variant v2_hybrid --modules pi,stance
run_one b3                  --variant v2_hybrid --modules pi,stance,self
run_one v2_plus_peer        --variant v2 --with-peer
wait
echo "=== Wave 2 done $(date) ==="

echo "[done] $(date)"
ls -la reports/datav3/base_api_*_inference.jsonl 2>/dev/null
