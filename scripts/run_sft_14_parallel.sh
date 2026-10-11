#!/bin/bash
# Parallel 14-condition inference: 7 SFT (LoRA) + 7 BASE (no LoRA), 跨 5 GPU.
# Wave 1: 5 (SFT v2 / v2_no_caps / v2_no_seg3_rule / v2_no_beta_summary / v2_hybrid)
# Wave 2: 5 (SFT B0 / v2_plus_peer / BASE v2 / v2_no_caps / v2_no_seg3_rule)
# Wave 3: 4 (BASE v2_no_beta_summary / v2_hybrid / B0 / v2_plus_peer)
# 每 wave ~40 min, 总 ~2h.

set -u
cd /data/wy/Designer_RolePlay/0512技术方案

ADAPTER="${ADAPTER:-/data/wy/sft_v3_ckpts/stage_warmup_ep5}"
PY="${PY:-/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python}"
LOG_DIR=/home/wy/sft_14_logs
mkdir -p $LOG_DIR
mkdir -p reports/datav3
export PYTHONPATH=.

echo "[start] $(date)  adapter=$ADAPTER"

# launch_one GPU MODEL_TAG CONDITION_NAME EXTRA_ARGS...
launch_one() {
    local gpu="$1"; local tag="$2"; local name="$3"; shift 3
    local logfile="$LOG_DIR/${tag}_${name}.log"
    local outfile="reports/datav3/${tag,,}_${name}_inference.jsonl"
    local adapter_arg=""
    if [ "$tag" = "SFT" ]; then
        adapter_arg="--adapter $ADAPTER"
    else
        adapter_arg="--adapter none"
    fi
    echo "  GPU $gpu: $tag $name → $outfile (log: $logfile)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_sft_inference.py \
        --n 0 $adapter_arg --output "$outfile" "$@" > "$logfile" 2>&1 &
}

# ─── Wave 1 ───
echo
echo "=== Wave 1 $(date) ==="
launch_one 3 SFT v2          --variant v2
launch_one 4 SFT v2_no_caps  --variant v2 --modules pi,stance,self
launch_one 5 SFT v2_no_seg3_rule    --variant v2_no_seg3_rule
launch_one 6 SFT v2_no_beta_summary --variant v2_no_beta_summary
launch_one 7 SFT v2_hybrid          --variant v2_hybrid
wait
echo "=== Wave 1 done $(date) ==="

# ─── Wave 2 ───
echo
echo "=== Wave 2 $(date) ==="
launch_one 3 SFT b0                  --variant v2_hybrid --modules ""
launch_one 4 SFT v2_plus_peer        --variant v2 --with-peer
launch_one 5 BASE v2                 --variant v2
launch_one 6 BASE v2_no_caps         --variant v2 --modules pi,stance,self
launch_one 7 BASE v2_no_seg3_rule    --variant v2_no_seg3_rule
wait
echo "=== Wave 2 done $(date) ==="

# ─── Wave 3 ───
echo
echo "=== Wave 3 $(date) ==="
launch_one 3 BASE v2_no_beta_summary --variant v2_no_beta_summary
launch_one 4 BASE v2_hybrid          --variant v2_hybrid
launch_one 5 BASE b0                 --variant v2_hybrid --modules ""
launch_one 6 BASE v2_plus_peer       --variant v2 --with-peer
wait
echo "=== Wave 3 done $(date) ==="

echo
echo "[done] $(date)"
ls -la reports/datav3/sft_*_inference.jsonl reports/datav3/base_*_inference.jsonl 2>/dev/null | head -20
