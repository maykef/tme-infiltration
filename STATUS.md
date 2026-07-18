# STATUS — tme-infiltration

**Updated:** 2026-07-18 15:10 · **Current stage:** ✅ COMPLETE + correction pass + panel-match test done

## Headline numbers (corrected, full-dataset run)
| Metric | Value |
|---|---|
| **GNN AUROC (LOPO, pretrained+finetune)** | **0.576** [0.360, 0.789] |
| **Giuliani pair-correlation baseline AUROC** | **0.833** [0.660, 0.964] |
| **Paired test (baseline − GNN)** | **+0.257 [+0.029, +0.493] → baseline significantly wins** |
| Verdict | GNN does **not** beat the baseline; paired test confirms baseline wins significantly |

Original 100-image run: GNN 0.460 [0.253, 0.679] — see `RESULTS.md` for the side-by-side.

## Panel-matched pretraining test (2026-07-18, Part 1 go/no-go)
| Candidate | Jaccard vs Moldoveanu | Checkpoints (of 4) | Clears bar? |
|---|---|---|---|
| IMMUcan_2022_CancerExample | 37.7% | 2/4 | ❌ (< 40%) |
| HochSchulz_2022_Melanoma | 19.4% | 1/4 | ❌ |

**NO-GO — stopped at Part 1** (pre-registered bar: Jaccard ≥40% AND ≥2/4 checkpoints). No
public IMC panel-matched source clears it → panel mismatch ruled out as a *fixable* confound.
Part 2 (retrain on new corpus) not started, per the gate. `docs/panel_matched_candidates.md`.

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
