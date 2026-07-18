# STATUS — tme-infiltration

**Updated:** 2026-07-18 13:25 · **Current stage:** ✅ COMPLETE (all stages done)

## Headline numbers
| Metric | Value |
|---|---|
| **GNN AUROC (LOPO, Moldoveanu)** | **0.460** [0.253, 0.679] |
| **Giuliani pair-correlation baseline AUROC** | **0.833** [0.660, 0.964] |
| Verdict | GNN does **not** beat the hand-crafted spatial statistic (reported plainly) |

## Stage status
| Stage | Status |
|---|---|
| 0a — system/env audit | ✅ done |
| 0c — repo setup | ✅ done |
| 0b — env build | ✅ done — torch 2.11+cu128 (sm_120 ✓), PyG 2.8, R 4.5+Bioc |
| 1a — Jackson/Fischer export | ✅ done — 100 slides × 285,851 cells → parquet |
| 1b — Moldoveanu download/extract | ✅ done — md5 ✓; both open items resolved |
| 2 — schema discovery doc | ✅ done — `docs/data_schema.md` |
| 3 — graph construction | ✅ done — 100 jackson (33mk) + 30 moldoveanu (35mk) graphs, k=12 |
| 4 — SSL pretraining | ✅ done — val masked-MSE 0.58→0.23, 76 s, encoder saved |
| 5 — LOPO fine-tune + baseline | ✅ done — GNN 0.460 vs baseline 0.833 (+2 transparent ablations) |
| 6 — counterfactual search | ✅ done — 16 non-responder slides, CSV+PNG+top findings |
| 7 — reproducibility + RESULTS.md | ✅ done — RESULTS.md, run_all.sh, run_log.jsonl |

## Key findings
- **The GNN does not beat the Giuliani baseline** at N=30. All 3 GNN configs (finetune 0.460, frozen 0.406, no-pretrain 0.545) sit near chance with 0.5-crossing CIs. Pretraining did not help — breast→melanoma transfer was ineffective (a clean negative result).
- Baseline (macrophage ↔ activated-CD8 co-localization @10.5 µm) is a strong single feature (0.833).
- Counterfactuals most often point to TIM3 / CD45RA / CCR7 increases on melanoma & macrophage cells to raise predicted response (interpret cautiously given near-chance GNN).

## Open items needing human input
_None._

## Notes
- Full detail: `results/progress.log` + `results/run_log.jsonl`; contract: `docs/data_schema.md`;
  headline: `RESULTS.md`. Reproduce: `bash scripts/build_env.sh` then `bash run_all.sh`.
