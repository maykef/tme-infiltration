#!/usr/bin/env bash
set -uo pipefail
cd /mnt/nvme8tb/tme-infiltration
source /home/microscopy-rig/miniforge3/etc/profile.d/conda.sh
conda activate tme-infiltration
echo "[$(date '+%F %T')] CORRECTION — LOPO eval (full-corpus encoder): primary" >> results/progress.log
python src/finetune_eval.py --seed 42 2>&1 | tail -6
echo "[$(date '+%F %T')] CORRECTION — LOPO eval: frozen trunk" >> results/progress.log
python src/finetune_eval.py --seed 42 --freeze_trunk --tag frozen 2>&1 | tail -6
echo "[$(date '+%F %T')] CORRECTION — LOPO eval: no pretrain" >> results/progress.log
python src/finetune_eval.py --seed 42 --no_pretrain --tag nopretrain 2>&1 | tail -6
echo "[$(date '+%F %T')] CORRECTION — all 3 LOPO runs done" >> results/progress.log
touch results/.corrected_eval_done
