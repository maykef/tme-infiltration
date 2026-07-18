# CLAUDE.md — durable instructions for any Claude Code session in this repo

This is a human/LLM co-authored prototype: a spatial-graph GNN that pretrains on breast-cancer
IMC (Jackson/Fischer 2020) and fine-tunes to predict melanoma immunotherapy response
(Moldoveanu 2022). Read `docs/BUILD_SPEC.md` for the full original plan and
`docs/data_schema.md` for the resolved data contract before changing pipeline code. `STATUS.md`
is the live dashboard.

## Environment

Dedicated conda env `tme-infiltration` (python 3.11, torch 2.11+cu128 for Blackwell sm_120,
PyG 2.8, R 4.5 + Bioconductor). Activate:

```bash
source /home/microscopy-rig/miniforge3/etc/profile.d/conda.sh
conda activate tme-infiltration
```

Rebuild from scratch with `bash scripts/build_env.sh`. Machine-specific paths/versions the
audit found are in `CLAUDE.local.md` (gitignored — do not commit).

## Run

```bash
bash run_all.sh        # Stages 1..6: data -> graphs -> pretrain -> finetune/eval -> counterfactual
```

Individual stages live in `src/` (see README layout). All scripts take `--seed` (default 42).

## Rules that are easy to get wrong

- **Parallelization split.** Slide-level preprocessing (`src/build_graphs.py`) is
  CPU-embarrassingly-parallel: use `joblib.Parallel(n_jobs=32, backend="loky")` over slides,
  and inside each worker set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` **before**
  numpy/sklearn import, or BLAS oversubscription makes it slower than serial. The GPU training
  loop is the opposite: it is GPU-bound, so use a modest DataLoader `num_workers` (~8), not 32.
- **Evaluation on Moldoveanu is leave-one-patient-out only.** N=30 patients — never a random
  train/test split. 30 folds, out-of-fold predictions, AUROC with a bootstrap CI over patients.
- **Normalize per dataset, never pooled.** Jackson and Moldoveanu are different panels/
  instruments; z-score markers within each dataset separately.
- **The GNN must be reported against the Giuliani pair-correlation baseline** (macrophage vs.
  activated-CD8 density correlation at r=10.5um), computed on the same labels with the same LOPO
  logic. If the GNN doesn't beat it, say so in `RESULTS.md` — don't bury it.

## Conventions

- Keep the default `Co-Authored-By: Claude <noreply@anthropic.com>` commit trailer (honest
  co-authorship record for this prototype).
- Commit + push at the end of every stage; overwrite `STATUS.md` each time; append detail to
  `results/progress.log`; one JSON line per run to `results/run_log.jsonl`.
- Data / checkpoints / `*.pt` / archives are gitignored — regenerable, not for history.
- Anything needing a human decision goes to `results/decisions_needed.log` (take the least
  destructive path and keep going; don't halt).
