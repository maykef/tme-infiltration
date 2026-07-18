# STATUS — tme-infiltration

**Updated:** 2026-07-18 11:30 · **Current stage:** Stage 3 (graph construction)

## Headline numbers
| Metric | Value |
|---|---|
| GNN AUROC (LOPO, Moldoveanu) | _pending_ |
| Giuliani pair-correlation baseline AUROC | _pending_ |

## Stage status
| Stage | Status |
|---|---|
| 0a — system/env audit | ✅ done |
| 0c — repo setup | ✅ done |
| 0b — env build | ✅ done — torch 2.11+cu128 (sm_120 ✓), PyG 2.8, R 4.5+Bioc |
| 1a — Jackson/Fischer export | ✅ done — 45ch × 285,851 cells × 100 images → parquet (used `exprs` arcsinh assay) |
| 1b — Moldoveanu download/extract | ✅ done — md5 ✓; both open items resolved (labels + no igraph needed) |
| 2 — schema discovery doc | ✅ done — `docs/data_schema.md` |
| 3 — graph construction | ⏳ in progress |
| 4 — SSL pretraining | ⬜ not started |
| 5 — LOPO fine-tune + baseline | ⬜ not started |
| 6 — counterfactual search | ⬜ not started |
| 7 — reproducibility + RESULTS.md | ⬜ not started |

## Key resolved facts
- **Moldoveanu N=30** ICI tumor samples with `Response` label (`ST1_sample_info.txt`): **14 responders / 16 non-responders**. Each sample = one patient = one slide.
- **No igraph 1.2.5 needed:** published per-cell types ship as `ST4_cell_data.txt`; joined by `Cell_ID` (macro.mono + Tc.ae available for the Giuliani baseline).
- **Panels differ:** Jackson 36 markers vs Moldoveanu 35 → shared GATv2 trunk + per-dataset input stem; z-score within each dataset separately.

## Open items needing human input
_None._

## Notes
- Full detail: `results/progress.log`; original plan: `docs/BUILD_SPEC.md`; contract: `docs/data_schema.md`.
