#!/usr/bin/env bash
# Full pipeline, Stages 1..6 (non-interactive). Assumes the `tme-infiltration` conda env
# exists (build it once with `bash scripts/build_env.sh`). Re-runnable; deterministic (--seed).
# Usage: bash run_all.sh [SEED] [K]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
SEED="${1:-42}"
K="${2:-12}"
LOG="$REPO/results/progress.log"
mkdir -p results
ts() { date '+%Y-%m-%d %H:%M:%S'; }
section() { echo ""; echo "=================================================================="; \
            echo "[$(ts)] $*"; echo "=================================================================="; \
            echo "[$(ts)] RUN_ALL: $*" >> "$LOG"; }

source /home/microscopy-rig/miniforge3/etc/profile.d/conda.sh
conda activate tme-infiltration

# Stage 0b (only if env build never completed) --------------------------------
if ! python -c "import torch, torch_geometric" 2>/dev/null; then
  section "STAGE 0b — build environment (missing deps detected)"
  bash scripts/build_env.sh
fi

# Stage 1a — Jackson/Fischer export (skip if parquet already present) ----------
if [ ! -f data/jackson_processed/jackson_cells.parquet ]; then
  section "STAGE 1a — export Jackson/Fischer (imcdatasets -> parquet)"
  Rscript src/export_jackson.R
fi

# Stage 1b — Moldoveanu download + extract (skip if extracted) -----------------
if [ ! -f data/moldoveanu_raw/Moldoveanu_2022_CyTOF_melanoma/Data/ICI_meanIntensity.tsv.gz ]; then
  section "STAGE 1b — download + extract Moldoveanu (Zenodo)"
  mkdir -p data/moldoveanu_raw
  curl -L --retry 3 -o data/moldoveanu_raw/moldoveanu_2022.tar.gz \
    "https://zenodo.org/records/5903179/files/Moldoveanu_2022_CyTOF_melanoma.tar.gz?download=1"
  echo "f03adede5b7b81c11569608393aad5df  data/moldoveanu_raw/moldoveanu_2022.tar.gz" | md5sum -c -
  tar -xzf data/moldoveanu_raw/moldoveanu_2022.tar.gz -C data/moldoveanu_raw
fi

# Stage 3 — graph construction -------------------------------------------------
section "STAGE 3 — build spatial kNN graphs (k=$K)"
python src/build_graphs.py --dataset both --k "$K" --n_jobs 32 --seed "$SEED"

# Stage 4 — self-supervised pretraining ---------------------------------------
section "STAGE 4 — masked-marker pretraining (Jackson)"
python src/pretrain.py --seed "$SEED"

# Stage 5 — LOPO fine-tuning + Giuliani baseline ------------------------------
section "STAGE 5 — LOPO fine-tune + baseline (Moldoveanu)"
python src/finetune_eval.py --seed "$SEED"

# Stage 6 — counterfactual gradient search ------------------------------------
section "STAGE 6 — counterfactual gradient search"
python src/counterfactual.py --seed "$SEED" --targets nonresponders

# Stage 7 — assemble RESULTS.md -----------------------------------------------
section "STAGE 7 — assemble RESULTS.md"
python src/make_results.py

section "PIPELINE COMPLETE — see RESULTS.md"
