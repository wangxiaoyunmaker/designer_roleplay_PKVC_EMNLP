#!/bin/bash
# v3c (LoRA r=16) full 7-condition eval, int4 quantization, 5 GPU parallel 2 waves.
# Wave 1: 5 conditions (v2 / v2_no_caps / v2_no_seg3_rule / v2_no_beta_summary / v2_hybrid)
# Wave 2: 2 conditions (b0 / v2_plus_peer)
# 每 condition int4 ~90 min, total ~3h (2 wave × ~90 min).

set -u
cd /data/wy/Designer_RolePlay/0512技术方案

ADAPTER="${ADAPTER:-/data/wy/sft_v3_ckpts/stage_warmup_ep3_v3c}"
PY="${PY:-/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python}"
LOG_DIR=/home/wy/sft_v3c_7cond_logs
mkdir -p $LOG_DIR reports/datav3
export PYTHONPATH=.

echo "[start] $(date)  adapter=$ADAPTER"

run_one() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="$LOG_DIR/${name}.log"
    local outfile="reports/datav3/sft_v3c_${name}_inference.jsonl"
    echo "  GPU $gpu: $name → $outfile"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_sft_inference.py \
        --n 0 --adapter "$ADAPTER" --quantize int4 --output "$outfile" "$@" > "$logfile" 2>&1 &
}

# ─── Wave 1 (5 conditions) ───
echo
echo "=== Wave 1 $(date) ==="
run_one 3 v2                 --variant v2
run_one 4 v2_no_caps         --variant v2 --modules pi,stance,self
run_one 5 v2_no_seg3_rule    --variant v2_no_seg3_rule
run_one 6 v2_no_beta_summary --variant v2_no_beta_summary
run_one 7 v2_hybrid          --variant v2_hybrid
wait
echo "=== Wave 1 done $(date) ==="

# ─── Wave 2 (5 conditions: b0 / peer / B1 / B2 / B3 ladder) ───
echo
echo "=== Wave 2 $(date) ==="
run_one 3 b0                 --variant v2_hybrid --modules ""
run_one 4 v2_plus_peer       --variant v2 --with-peer
run_one 5 b1                 --variant v2_hybrid --modules pi
run_one 6 b2                 --variant v2_hybrid --modules pi,stance
run_one 7 b3                 --variant v2_hybrid --modules pi,stance,self
wait
echo "=== Wave 2 done $(date) ==="

echo
echo "[done] $(date)"
ls -la reports/datav3/sft_v3c_*_inference.jsonl 2>/dev/null
