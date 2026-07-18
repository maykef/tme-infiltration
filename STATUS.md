# STATUS — tme-infiltration

**Updated:** 2026-07-18 14:25 · **Current stage:** ✅ COMPLETE + correction pass done

## Headline numbers (corrected, full-dataset run)
| Metric | Value |
|---|---|
| **GNN AUROC (LOPO, pretrained+finetune)** | **0.576** [0.360, 0.789] |
| **Giuliani pair-correlation baseline AUROC** | **0.833** [0.660, 0.964] |
| Verdict | GNN still does **not** beat the baseline (CI crosses 0.5, overlaps baseline) |

Original 100-image run: GNN 0.460 [0.253, 0.679] — see `RESULTS.md` for the side-by-side.

## Correction pass (2026-07-18)
| Item | Status |
|---|---|
| 1 — marker overlap check | ✅ `docs/marker_overlap.md`: 8 shared, **12.7% Jaccard** (LOW) |
| 2 — full-dataset Jackson (`full_dataset=TRUE`) | ✅ 723 images / 1,240,267 cells (was 100 / 285,851); re-pretrain val-MSE 0.228→0.175; re-ran Stages 3–5 |
| 3 — bootstrap CIs on all 3 ablations | ✅ all CIs overlap → pretrain/frozen/no-pretrain **not distinguishable** (within noise) |
| Report — side-by-side + `docs/CORRECTIONS.md` | ✅ RESULTS.md shows both runs; counterfactuals **gated** (NOT biological findings) |

## Corrected ablations (full run)
| Config | AUROC [95% CI] |
|---|---|
| Pretrained + finetune | 0.576 [0.360, 0.789] |
| Frozen trunk | 0.567 [0.348, 0.796] |
| No pretraining | 0.545 [0.330, 0.760] |

## Key conclusions (trustworthy after correction)
- **The negative result holds:** even with 4.3× the pretraining corpus, the GNN (0.576) does not beat the hand-crafted baseline (0.833).
- The earlier "pretraining hurts" impression was a **truncation artifact**; with full data pretraining gives a small bump but it is **within bootstrap noise** — no directional claim supported.
- **Marker panels barely overlap (12.7%)** — a structural reason weak transfer was expected.
- Counterfactual findings remain **gated**; not to be cited as biology until a model beats the baseline.

## Original 7-stage pipeline
All stages 0–7 done (see git history stage0…stage7). Reproduce: `bash scripts/build_env.sh` then `bash run_all.sh`.

## Open items needing human input
_None._
