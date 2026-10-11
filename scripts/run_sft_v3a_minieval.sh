#!/bin/bash
# Mini eval for v3a SFT (3 SFT conditions, BASE_v2 reused from previous run).
# 3 conditions × 486 jobs × ~5s = ~30 min serial on 1 GPU.
# Or parallel on 3 GPUs = ~10 min.

set -u
cd /data/wy/Designer_RolePlay/0512技术方案

ADAPTER="${ADAPTER:-/data/wy/sft_v3_ckpts/stage_warmup_ep2_v3a}"
PY="${PY:-/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python}"
LOG_DIR=/home/wy/sft_v3a_logs
mkdir -p $LOG_DIR
export PYTHONPATH=.

echo "[start] $(date)  adapter=$ADAPTER"

launch_one() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="$LOG_DIR/${name}.log"
    local outfile="reports/datav3/sft_v3a_${name}_inference.jsonl"
    echo "  GPU $gpu: $name → $outfile"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_sft_inference.py \
        --n 0 --adapter "$ADAPTER" --output "$outfile" "$@" > "$logfile" 2>&1 &
}

# 3 SFT in parallel on GPU 3, 4, 5
launch_one 3 v2                 --variant v2
launch_one 4 v2_no_seg3_rule    --variant v2_no_seg3_rule
launch_one 5 v2_plus_peer       --variant v2 --with-peer
wait
echo "[done] $(date)"
ls -la reports/datav3/sft_v3a_*_inference.jsonl 2>/dev/null
