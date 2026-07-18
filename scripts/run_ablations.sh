#!/usr/bin/env bash
set -uo pipefail
cd /mnt/nvme8tb/tme-infiltration
source /home/microscopy-rig/miniforge3/etc/profile.d/conda.sh
conda activate tme-infiltration
echo "[$(date '+%F %T')] STAGE 5 ablation: frozen trunk" >> results/progress.log
python src/finetune_eval.py --seed 42 --freeze_trunk --tag frozen 2>&1 | tail -6
echo "[$(date '+%F %T')] STAGE 5 ablation: no pretrain" >> results/progress.log
python src/finetune_eval.py --seed 42 --no_pretrain --tag nopretrain 2>&1 | tail -6
echo "[$(date '+%F %T')] STAGE 5 ablations done" >> results/progress.log
touch results/.ablations_done
