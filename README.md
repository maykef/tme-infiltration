# tme-infiltration

Spatial-graph deep learning for predicting immunotherapy response from the tumor
microenvironment (TME) in imaging mass cytometry (IMC) data.

**What this repo does.** It represents IMC tumor slides as spatial cell graphs (cells =
nodes, k-nearest-neighbor spatial edges), self-supervised **pretrains** a GATv2 graph neural
network with a masked-marker task on a large breast-cancer cohort, then **fine-tunes and
evaluates** it on a small melanoma immune-checkpoint-inhibitor (ICI) cohort to predict
responder vs. non-responder, using leave-one-patient-out cross-validation. It also runs a
gradient-based **counterfactual** search over the fine-tuned model to rank which cells /
marker shifts would most increase predicted response, and benchmarks the GNN against a
published hand-crafted spatial statistic.

## Data sources

| Role | Cohort | Access |
|---|---|---|
| Pretrain (no labels) | **Jackson & Fischer 2020**, breast cancer IMC (~350 images) | Bioconductor `imcdatasets::JacksonFischer_2020_BreastCancer(data_type="sce")` — auto-downloads via ExperimentHub, no login |
| Fine-tune + eval | **Moldoveanu et al. 2022**, melanoma ICI CyTOF/IMC (30 patients) | Zenodo record [5903179](https://zenodo.org/records/5903179) — `Moldoveanu_2022_CyTOF_melanoma.tar.gz`, direct download, no login |

Both are public. Raw data, processed graphs, checkpoints and counterfactual outputs are
git-ignored (large / regenerable); the code, docs, logs and `RESULTS.md` are tracked.

## Reproduce

```bash
# 0. one-time environment build (conda env `tme-infiltration`: python + torch/PyG + R/Bioconductor)
bash scripts/build_env.sh

# 1..6 full pipeline (data -> graphs -> pretrain -> finetune/eval -> counterfactual)
bash run_all.sh
```

Outputs land in `results/` (`RESULTS.md`, `progress.log`, `run_log.jsonl`,
`counterfactuals/`) and `checkpoints/`. See `STATUS.md` for the live dashboard,
`docs/data_schema.md` for the resolved data contract, and `docs/BUILD_SPEC.md` for the full
original build specification.

## Layout

```
scripts/build_env.sh     environment build (Stage 0b)
src/export_jackson.R     Jackson/Fischer SCE -> parquet (Stage 1a)
src/build_graphs.py      spatial kNN graph construction (Stage 3)
src/pretrain.py          self-supervised masked-marker pretraining (Stage 4)
src/finetune_eval.py     LOPO fine-tuning + AUROC + Giuliani baseline (Stage 5)
src/counterfactual.py    gradient counterfactual search (Stage 6)
run_all.sh               chains Stages 1..6 non-interactively
docs/                    BUILD_SPEC.md, data_schema.md
```

Human/LLM co-authored prototype; commits keep the `Co-Authored-By: Claude` trailer as an
honest record of what was agent-built.
