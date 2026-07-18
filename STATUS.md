# STATUS — tme-infiltration

**Updated:** 2026-07-18 11:24 · **Current stage:** Stage 1 (data acquisition)

## Headline numbers
| Metric | Value |
|---|---|
| GNN AUROC (LOPO, Moldoveanu) | _pending_ |
| Giuliani pair-correlation baseline AUROC | _pending_ |

## Stage status
| Stage | Status |
|---|---|
| 0a — system/env audit | ✅ done |
| 0c — repo setup | ✅ done (docs, CLAUDE.md, README committed) |
| 0b — env build | ✅ done — `tme-infiltration` env: torch 2.11+cu128 (sm_120 ✓), PyG 2.8, R 4.5+Bioc |
| 1a — Jackson/Fischer export | ⏳ in progress |
| 1b — Moldoveanu download/extract | ✅ archive downloaded (md5 ✓); extraction + label/igraph resolution pending |
| 2 — schema discovery doc | ⬜ not started |
| 3 — graph construction | ⬜ not started |
| 4 — SSL pretraining | ⬜ not started |
| 5 — LOPO fine-tune + baseline | ⬜ not started |
| 6 — counterfactual search | ⬜ not started |
| 7 — reproducibility + RESULTS.md | ⬜ not started |

## Open items needing human input
_None._ (see `results/decisions_needed.log` if any appear)

## Notes
- Dedicated env created; no existing env modified. PyG C++ extensions unavailable (optional).
- Full detail: `results/progress.log`; original plan: `docs/BUILD_SPEC.md`.
