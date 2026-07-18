# Task: TME spatial-graph infiltration predictor — pretrain on Jackson/Fischer, fine-tune on Moldoveanu

> This file is the verbatim historical record of the original build plan (Stage 0d.1).
> Update only if the plan itself materially changes (e.g. the igraph fallback is taken, a
> different k for kNN is settled on after tuning), not for routine status updates. For
> current status see `STATUS.md`; for the resolved data contract see `docs/data_schema.md`.

## Context and goal

Build a pipeline that (1) represents tumor microenvironment imaging mass cytometry (IMC)
slides as spatial cell graphs, (2) pretrains a graph neural network with a self-supervised
masked-marker task on the Jackson/Fischer 2020 breast cancer cohort, (3) fine-tunes and
evaluates it on the Moldoveanu 2022 melanoma immunotherapy-response cohort using
leave-one-patient-out cross-validation, and (4) implements a gradient-based counterfactual
search over the fine-tuned model. Machine: AMD Threadripper 7970X (32 cores / 64 threads),
128GB RAM, single NVIDIA RTX PRO 6000 Blackwell (96GB VRAM). No internet-restricted steps —
all data below is public.

Work in stages. Do not write the training pipeline before Stage 2 (schema discovery) is
complete and its findings are written to `docs/data_schema.md` — the exact column names and
label locations below are not fully known in advance and must be confirmed from the real
files, not assumed.

---

## Operating mode: autonomous execution

I am monitoring this run remotely and asynchronously, not watching the terminal live.
Structure the interaction accordingly:

1. Run Stage 0a (system and environment audit) first and present a short plan: what already
   exists on this machine and can be reused, what will be newly created or installed,
   estimated disk usage, estimated total runtime.
2. Ask for confirmation **exactly once**, after that plan is presented, before making any
   changes to the system (creating environments, installing packages, downloading data).
3. After that single confirmation, execute Stages 0b through 7 to completion without further
   prompts. Do not pause to ask permission again for anything inside the scope of this
   plan — including taking the igraph 1.2.5 fallback if needed, retrying failed downloads,
   or picking reasonable hyperparameters not pinned in this document.
4. If something genuinely outside the scope of this plan comes up and is destructive or
   hard to reverse (e.g. needing to remove an existing environment to free disk space, a
   dependency conflict with no safe fallback), do not silently guess and do not halt and
   wait — write it clearly to `results/decisions_needed.log` with full context, take the
   least destructive reasonable path, and keep going.
5. Log continuously, not just at the end: timestamped stage-start/stage-complete lines to
   `results/progress.log`, an updated `STATUS.md` (see Stage 0d) after every stage, and on
   any error, a clear plain-text description of what failed and what fallback (if any) was
   taken, written to the log rather than only printed to a terminal no one is watching live.
6. Commit and push to GitHub at the end of every stage, not just once at the end — since
   I'm monitoring remotely, `STATUS.md` and the GitHub web UI are how I check progress
   without needing terminal access. Use clear, conventional commit messages (`stage0:
   environment audit and setup`, `stage3: graph construction`, etc.) and keep Claude Code's
   default commit attribution as described in Stage 0d. If a stage fails partway, still
   commit what succeeded before moving to the fallback, so failed attempts are visible in
   history rather than silently overwritten.

---

## Stage 0c — GitHub repository setup

The repo `tme-infiltration` has already been created (empty) on GitHub by the user — do not
create a new remote repo. Check `gh auth status` first; if the `gh` CLI is installed and
authenticated, get the owner from `gh api user --jq .login` and clone with
`gh repo clone <owner>/tme-infiltration`. If `gh` is not available or not authenticated, ask
for the clone URL as part of the single up-front confirmation in step 1-2 above — bundle it
into that one question rather than making it a separate interruption.

Clone (or if a local `tme-infiltration/` folder already exists from a prior partial run,
`cd` into it and confirm the remote matches instead of re-cloning) into the working
directory, then before any other file is added: [.gitignore contents — see repo .gitignore]

Data files, model checkpoints, and the downloaded archives are excluded deliberately — they
don't belong in git history (multi-GB, regenerable, and this repo may end up public later).
`RESULTS.md`, `docs/data_schema.md`, `results/run_log.jsonl`, `results/progress.log`,
`STATUS.md`, and `CLAUDE.md` are small text and should be committed normally. Write a short
`README.md` up front stating what this repo does, the two data sources with their access
instructions from Stage 1, and how to reproduce (`bash run_all.sh` after Stage 0's
environment step) — then commit this as the first commit before any other work begins.

---

## Stage 0d — Repository memory: CLAUDE.md, status file, and build spec

This repo will outlive this single conversation. Set up in order:
1. `docs/BUILD_SPEC.md` — copy this entire task document into the repo verbatim (this file).
2. `CLAUDE.md` (repo root, committed) — short durable instruction file, <100 lines, points
   to BUILD_SPEC and data_schema for detail. Includes: env activation + run_all.sh; the
   Stage 3 parallelization rule (32 workers with OMP_NUM_THREADS=1 each for slide-level
   preprocessing, modest num_workers for the small training loop); the leave-one-patient-out
   requirement for the 30-patient cohort; the commit attribution convention.
3. `CLAUDE.local.md` (repo root, gitignored) — what Stage 0a's audit found on this machine.
4. `STATUS.md` (repo root, committed, overwritten not appended) — 30-second dashboard.

**Commit attribution**: leave Claude Code's default `Co-Authored-By: Claude
<noreply@anthropic.com>` trailer on. Honest human/LLM co-authored record.

---

## Stage 0 — Environment

### 0a. System and environment audit — run first, unconditionally.
Record uname, os-release, mamba env list, nvidia-smi, nvcc, lscpu, free, df, which R, and
per-env python/R/torch versions. Determine: (a) any env with working R+Bioconductor, (b) any
env with CUDA PyTorch matching sm_120 (Blackwell), (c) free disk vs downloads. Prefer reusing
a suitable existing environment. R is needed because Jackson/Fischer ships as a Bioconductor
package.

### 0b. Environment creation (only for what 0a shows is missing).
If no env covers both R/Bioconductor and Blackwell PyTorch, create a dedicated one. PyTorch:
fetch current install command for CUDA build compatible with sm_120 (CUDA 12.8+). PyG: fetch
current matching install instructions. Additional: pandas numpy scipy scikit-learn anndata
pyarrow tqdm matplotlib joblib threadpoolctl. R: BiocManager::install(c("SingleCellExperiment",
"SpatialExperiment","cytomapper","imcdatasets")).

---

## Stage 1 — Data acquisition

### 1a. Jackson/Fischer 2020 breast cancer cohort (pretraining set, no outcome labels)
Via `imcdatasets`: `sce <- JacksonFischer_2020_BreastCancer(data_type = "sce")`. Before
export, log dim(sce), colnames(rowData(sce)), colnames(colData(sce)), head(colData(sce)),
assayNames(sce). Confirm x,y coordinate location and image/slide id column. Export to one
parquet: `[image_id, cell_id, x, y, <marker_1..marker_35>]`, using transformed (not raw)
assay — prefer an arcsinh-transformed assay if present, else apply asinh(x/5). Note which in
docs/data_schema.md.

### 1b. Moldoveanu 2022 melanoma ICI-response cohort (fine-tuning + eval set)
Download:
```
curl -L -o data/moldoveanu_raw/moldoveanu_2022.tar.gz \
  "https://zenodo.org/records/5903179/files/Moldoveanu_2022_CyTOF_melanoma.tar.gz?download=1"
```
Archive contains R analysis scripts. `cell_type_clustering.R` processes
Data/ICI_meanIntensity.tsv.gz and Data/non-ICI_meanIntensity.tsv.gz -> Data/ST4_cell_data.txt.
`build_spatial_graph.R` shows coordinate + sample/patient id columns. Two things to resolve
by inspection: (1) igraph 1.2.5 pin — try remotes::install_version; if it fails within ~30
min, fall back to leidenalg + python-igraph community detection, record deviation. (2)
responder/non-responder label source — grep the extracted scripts; likely a metadata table
joined by patient/sample id; if absent, from Moldoveanu et al. Science Immunology 2022
eabi5072 supplement. Document both in docs/data_schema.md before Stage 3.

---

## Stage 2 — Schema discovery checkpoint
Write docs/data_schema.md: exact column names for cell id, image/slide/patient id, x/y,
marker channels (all, with transform), and for Moldoveanu the response label column + mapping,
plus resolution of both Stage 1b open items. This is the contract.

---

## Stage 3 — Graph construction
`src/build_graphs.py` -> one torch_geometric.data.Data per slide to
data/processed/{dataset}/{slide_id}.pt.
- Nodes = cells; features = marker vector (arcsinh), z-scored per marker within each dataset
  separately (no pooling across datasets).
- Edges = kNN in (x,y), k=12 (CLI arg). sklearn NearestNeighbors n_jobs=-1 or scipy cKDTree.
- Store slide_id, and for Moldoveanu patient_id + response_label, as graph attributes.
- Parallelization: joblib.Parallel(n_jobs=32, backend="loky") over slides; inside each worker
  set OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1 before numpy/sklearn import.

---

## Stage 4 — Self-supervised pretraining (Jackson/Fischer)
`src/pretrain.py`. Model: 3-4 layer GATv2 encoder (GATv2Conv), hidden 128, 4 heads. Task:
mask 15% of marker channels for 15% of random nodes per graph per epoch (zero them + pass a
mask flag feature), predict true values at masked positions with MSE. Train with early
stopping (patience ~15) on validation masked-MSE. DataLoader num_workers=8. Batch via
torch_geometric.loader.DataLoader. Save encoder to checkpoints/pretrained_encoder.pt.

---

## Stage 5 — Fine-tuning and leave-one-patient-out evaluation (Moldoveanu)
`src/finetune_eval.py`. Load pretrained encoder, add GlobalAttention pooling -> slide
embedding -> 2-layer MLP head -> binary responder logit. N=30 patients: leave-one-patient-out
CV (30 folds, not random split). Collect out-of-fold predicted probs, compute AUROC with
bootstrap CI (~2000 resamples of patients, 2.5/97.5 percentiles).
**Mandatory baseline (same script, same data):** Giuliani et al. pair-correlation statistic —
spatial correlation between macrophage/monocyte and activated CD8+ T cell density at
r=10.5um, per slide, AUROC of that scalar vs same 30 labels with same LOPO logic. Reference:
that feature separates responders at p=0.0005 vs raw CD8+ density p=0.15. Print both AUROCs
side by side. If GNN doesn't beat the statistic, say so directly.

---

## Stage 6 — Counterfactual gradient search
`src/counterfactual.py`. For a chosen slide (or all non-responder slides), gradient of
predicted response prob wrt node input features (torch.autograd.grad, retain_graph=True,
inputs = node feature tensor requires_grad_(True)). Aggregate per-node gradient magnitude,
rank cells/neighborhoods whose marker shift would most increase predicted response. Output
per-slide ranked CSV (cell id, dominant marker direction, magnitude) to
results/counterfactuals/{slide_id}.csv + a scatter PNG of cell positions colored by gradient
magnitude per slide.

---

## Stage 7 — Reproducibility and output
- Fix all seeds (random, numpy, torch, CUDA determinism) via single --seed CLI arg, default 42.
- Log every run (hyperparams, git commit hash, timing) to results/run_log.jsonl, one line/run.
- Final deliverable RESULTS.md stating in order: dataset sizes used after filtering; the two
  AUROC numbers (GNN vs pair-correlation) with CIs; wall-clock for pretrain + finetune; top 5
  counterfactual findings in one sentence each.

## Definition of done
After the single up-front confirmation, audit->counterfactual runs unattended and produces
RESULTS.md with the two real AUROC numbers, plus complete results/progress.log. Also write
run_all.sh chaining Stages 0b-6. Every stage's commit pushed to tme-infiltration on GitHub as
it completes.
