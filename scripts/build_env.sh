#!/usr/bin/env bash
# Stage 0b — dedicated environment build. Runs unattended; logs to results/progress.log.
# Idempotent-ish: safe to re-run; conda create will error if env exists (that's fine, caught).
set -uo pipefail
REPO=/mnt/nvme8tb/tme-infiltration
LOG=$REPO/results/progress.log
ENVN=tme-infiltration
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

source /home/microscopy-rig/miniforge3/etc/profile.d/conda.sh
source /home/microscopy-rig/miniforge3/etc/profile.d/mamba.sh 2>/dev/null || true

say "STAGE 0b START — env build ($ENVN)"

# 1. Base env with python + R + core R build deps (bioconda has prebuilt bioconductor pkgs)
if mamba env list | grep -qE "^\s*$ENVN\s"; then
  say "env $ENVN already exists — reusing"
else
  say "creating env $ENVN (python 3.11 + r-base 4.3)"
  mamba create -n $ENVN -c conda-forge -c bioconda python=3.11 r-base=4.3 r-biocmanager r-remotes r-data.table -y >>"$LOG" 2>&1
  say "base env create exit=$?"
fi

mamba activate $ENVN
say "active python: $(which python)  R: $(which R)"

# 2. Bioconductor packages via bioconda (prebuilt — far more reliable than BiocManager in conda R)
say "installing bioconductor pkgs via bioconda"
mamba install -n $ENVN -c conda-forge -c bioconda \
  bioconductor-singlecellexperiment bioconductor-spatialexperiment \
  bioconductor-cytomapper bioconductor-imcdatasets r-arrow -y >>"$LOG" 2>&1
say "bioconda bioconductor install exit=$?"

# 3. PyTorch — Blackwell sm_120 needs cu128 wheels (verified compatible on this GPU)
say "installing torch (cu128)"
pip install --no-input torch --index-url https://download.pytorch.org/whl/cu128 >>"$LOG" 2>&1
say "torch install exit=$?"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'cap',torch.cuda.get_device_capability(),'archs',torch.cuda.get_arch_list())" 2>&1 | tee -a "$LOG"

# 4. PyTorch Geometric — matching wheels for the installed torch/cuda
TVER=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null)
say "installing torch_geometric + extensions for torch $TVER cu128"
pip install --no-input torch_geometric >>"$LOG" 2>&1
say "pyg core install exit=$?"
pip install --no-input pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-${TVER}+cu128.html >>"$LOG" 2>&1
say "pyg extensions install exit=$? (extensions optional; core PyG works without them)"

# 5. Python ML/data stack
say "installing python ML stack"
pip install --no-input pandas numpy scipy scikit-learn anndata pyarrow tqdm matplotlib joblib threadpoolctl >>"$LOG" 2>&1
say "python stack install exit=$?"

# 6. Verification summary
say "STAGE 0b VERIFY"
python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib
mods = ["torch","torch_geometric","pandas","numpy","scipy","sklearn","anndata","pyarrow","joblib","matplotlib","threadpoolctl"]
for m in mods:
    try:
        mod = importlib.import_module(m); print(f"OK  {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"ERR {m}: {e}")
import torch
print("CUDA avail:", torch.cuda.is_available(), "cap:", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
try:
    from torch_geometric.nn import GATv2Conv, GlobalAttention; print("OK  PyG layers importable")
except Exception as e:
    print("ERR PyG layers:", e)
PY
R --version | head -1 | tee -a "$LOG"
mamba run -n $ENVN Rscript -e 'suppressMessages({library(imcdatasets);library(SingleCellExperiment);library(SpatialExperiment)}); cat("R BIOC OK\n")' 2>&1 | tee -a "$LOG"

say "STAGE 0b COMPLETE — env build finished"
touch $REPO/results/.env_build_done
