#!/bin/bash
# Probe chain diversity on 5 v3 adapters (SFT_v2 prompt only, 162 designer × 3 seg = 486 jobs each).
# 5 adapters × 1 condition, 1 per GPU = ~10 min wall.

set -u
cd /data/wy/Designer_RolePlay/0512技术方案

PY="${PY:-/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python}"
CKPT=/data/wy/sft_v3_ckpts
LOG_DIR=/home/wy/sft_v3_probe_logs
mkdir -p $LOG_DIR reports/datav3
export PYTHONPATH=.

echo "[start] $(date)"

probe_one() {
    local gpu="$1"; local name="$2"; local adapter="$3"
    local outfile="reports/datav3/sft_${name}_inference.jsonl"
    local logfile="$LOG_DIR/${name}.log"
    echo "  GPU $gpu: $name (adapter=$adapter)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_sft_inference.py \
        --variant v2 --n 0 --adapter "$adapter" --output "$outfile" > "$logfile" 2>&1 &
}

probe_one 3 v3a              $CKPT/stage_warmup_ep2_v3a
probe_one 4 v3b              $CKPT/stage_warmup_ep3_v3b
probe_one 5 v3c              $CKPT/stage_warmup_ep3_v3c
probe_one 6 v3d_kl05         $CKPT/stage_warmup_ep3_v3d_kl05bs2
probe_one 7 v3e_kl20         $CKPT/stage_warmup_ep3_v3e_kl20bs2
wait
echo "[done] $(date)"
ls -la reports/datav3/sft_v3*_inference.jsonl 2>/dev/null
