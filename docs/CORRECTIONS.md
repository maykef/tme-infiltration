# CORRECTIONS

Post-hoc correction pass on the original Stage 0–7 run. The original run was real and
correctly reported, but two issues had to be fixed before the negative result (GNN loses to
the hand-crafted baseline) could be trusted. This document records what was wrong and what
changed. The original numbers are preserved (archived in `results/basel100/`) and shown side
by side with the corrected numbers in `RESULTS.md` — nothing was silently overwritten.

## Issue 1 — Jackson pretraining set was silently truncated to the Basel subset

**What was wrong.** Stage 1a called
`JacksonFischer_2020_BreastCancer(data_type = "sce")` with the **default argument
`full_dataset = FALSE`**. That default is undocumented in our Stage 1a code and silently
returns only the **Basel** cohort — **100 images / 285,851 cells** — not the full
Jackson & Fischer dataset. The pretraining corpus was therefore ~⅕ of what was intended, and
"pretraining doesn't help" was concluded on a truncated corpus.

**Fix.** `src/export_jackson.R` now calls it with **`full_dataset = TRUE`**, returning the
**Basel + Zurich** cohorts:

| | Original (default) | Corrected (`full_dataset=TRUE`) |
|---|---|---|
| Jackson images (slide graphs) | 100 | **723** |
| Jackson cells | 285,851 | **1,240,267** |
| Jackson markers used | 33 | **36** |

Note the marker count rose 33 → 36: three channels (`EpCAM`, `CTNNB`, `SOX9`) are entirely
NaN in the Basel-subset `exprs` assay and were dropped there, but are populated in the full
(Basel+Zurich) assay, so `build_graphs.py`'s "drop all-NaN channels" rule keeps them now.
Pretraining val masked-MSE improved 0.228 → **0.175** with the larger corpus.

Re-run in order on the corrected pull, reusing the existing scripts unchanged except the one
argument: Stage 3 (graphs) → Stage 4 (pretrain) → Stage 5 (LOPO + ablations). Moldoveanu was
untouched.

## Issue 2 — marker panel overlap was never quantified

Negative cross-panel transfer is only interpretable if you know how much the two antibody
panels actually share. `src/check_marker_overlap.py` (→ `docs/marker_overlap.md`) now
quantifies it: **only 8 of 36/35 markers are shared** (synonym-aware: CD20, CD3, CD45, CD68,
B.Catenin/CTNNB, Histone.H3, Ki67, SMA), **Jaccard ≈ 12.7%**, and **zero channel-order
correspondence**. Jackson is a breast-epithelial/tumor panel; Moldoveanu is a melanoma immune
panel. This low semantic overlap is a structural, sufficient reason to expect weak transfer
regardless of pretraining-corpus size — and it is now stated up front.

## Issue 3 — ablations reported as point estimates without CIs

The original ablation panel (frozen trunk, no-pretrain) gave point AUROCs only. The corrected
run applies the **same 2000× patient-resample bootstrap** used for the headline to **all
three** GNN configurations, and `RESULTS.md` states plainly whether the
pretrained / frozen / no-pretrain ordering is statistically distinguishable or within noise
(it is within noise — all CIs overlap and cross 0.5).

## Counterfactual findings — gated

The Stage 6 counterfactual outputs are recomputed on the corrected model but remain **gated**:
`RESULTS.md` presents them only under an explicit "**NOT biological findings**" caveat unless
and until the corrected GNN AUROC's lower CI clears the baseline. They must not be described
as biology anywhere before that bar is met.

## Files changed
- `src/export_jackson.R` — `full_dataset = TRUE`.
- `src/check_marker_overlap.py`, `docs/marker_overlap.md` — new (Issue 2).
- `src/make_results.py`, `RESULTS.md` — side-by-side original vs corrected + CI + gating.
- `results/basel100/` — archived original run artifacts.
- `docs/data_schema.md` — updated Jackson counts (723 images, 36 markers).
