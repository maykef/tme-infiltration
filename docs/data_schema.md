# data_schema.md — resolved data contract (Stage 2)

This is the contract the rest of the pipeline is built against. All fields below were
confirmed by inspecting the real files on 2026-07-18, not assumed. If anything downstream
disagrees with a real file, fix it here first.

---

## 1. Jackson & Fischer 2020 — breast cancer IMC (pretraining, no outcome labels used)

**Source.** Bioconductor `imcdatasets::JacksonFischer_2020_BreastCancer(data_type = "sce")`
(ExperimentHub, cached under `~/.cache/R/ExperimentHub`). Exported by `src/export_jackson.R`
to `data/jackson_processed/jackson_cells.parquet` (+ `jackson_meta.json`).

**Object.** `SingleCellExperiment`, dim **45 channels × 285,851 cells**, **100 images**.
`assayNames = counts, exprs, quant_norm`.

**Resolved columns (from `colData`):**
| Field | Column |
|---|---|
| image / slide id (one graph per) | `image_name` (e.g. `BaselTMA_SP41_257_X3Y1`) — 100 unique |
| cell id | `cell_id` (e.g. `1_1`) |
| x coordinate | `cell_x` |
| y coordinate | `cell_y` |
| patient id (metadata only) | `patient_id` |

**Transform.** Used the **pre-existing `exprs` assay**, which is the package's
arcsinh-transformed expression (cofactor-scaled). No additional transform applied. (Fallback
`asinh(counts/5)` was not needed since `exprs` exists.)

**Parquet columns.** `[image_id, cell_id, x, y, <45 channel columns>]`.

**Channel selection for node features.** The 45 channels include 7 ruthenium counterstain
channels (`Ru96, Ru98, Ru99, Ru100, Ru101, Ru102, Ru104`) and 2 DNA intercalators
(`DNA1, DNA2`) — pure technical/segmentation channels with no antibody signal. `build_graphs.py`
drops these 9, **plus 3 channels entirely NaN in the `exprs` assay** (`EpCAM, CTNNB, SOX9`,
excluded by the package's expression computation), leaving **33 biological markers**:
`H3, H3K27me3, KRT5, FN1, KRT19, KRT8_18, TWIST1, CD68, KRT14, SMA, VIM, c_Myc, HER2, CD3e,
p_H3, SNAI2, ERa, PGR, p53, CD44, CD45, GATA3, CD20, CA9, CDH1, Ki67, EGFR,
p_S6, vWF, p_mTOR, KRT7, PanCK, cPARP_cCASP3`.

No outcome label is used from this cohort — pretraining is self-supervised (masked marker).

---

## 2. Moldoveanu 2022 — melanoma ICI IMC (fine-tuning + evaluation)

**Source.** Zenodo record 5903179, `Moldoveanu_2022_CyTOF_melanoma.tar.gz` (md5
`f03adede5b7b81c11569608393aad5df`, verified). Extracted to
`data/moldoveanu_raw/Moldoveanu_2022_CyTOF_melanoma/`.

**Primary table.** `Data/ICI_meanIntensity.tsv.gz` — the ICI (immunotherapy-treated) cohort,
**118,364 cells**, one row per cell, self-contained (coords + markers + cell type):

| Field | Column |
|---|---|
| x coordinate | `Location_Center_X` |
| y coordinate | `Location_Center_Y` |
| sample / slide id | `Sample_ID` (e.g. `26BL`, `14RD`) |
| cell id | `Cell_ID` (e.g. `26BL:1`) |
| coarse cell type | `Cluster` / `Cluster.v2` |
| 38 channels (cols 5–42) | see below |

**38 channels:** `SMA, PDL1, OX40, CD45, LAG3, TIM3, FoxP3, CD4, CCR7, CD68, VISTA, MEK1.2,
CD20, CD8a, pMEK1.2, SOX10, B.Catenin, CD45RA, GranzymeB, CD40, CollagenI, CD3, Ki67,
pERK1.2, cleaved.Caspase3, CD45RO, HLA.DR, S100, Histone.H3, X190BCKG, X191Ir, X193Ir, CD14,
ERK1.2, CD16, CD31, ICOS, CD29`.

**Channel selection for node features.** Drop 3 technical channels (`X190BCKG` background,
`X191Ir`/`X193Ir` DNA intercalators), leaving **35 biological markers**. (Marker intensities
in this file are already CyTOF mean intensities; `build_graphs.py` applies `asinh(x/5)` for
consistency with the Jackson `exprs` transform, then per-dataset z-scoring — see below.)

**Panels differ (45→36 vs 38→35).** The two cohorts use different antibody panels and
instruments. Node-feature marker vectors therefore have **different dimensionality** (Jackson
36, Moldoveanu 35) and are **z-scored within each dataset separately** (never pooled). The
transferable part of the network is the shared GATv2 message-passing trunk that operates in
hidden dimension; each dataset has its own input projection ("stem") mapping its marker
vector → hidden. On fine-tuning, the GATv2 trunk weights load from the pretrained encoder and
a fresh Moldoveanu stem is trained. See `src/pretrain.py` / `src/finetune_eval.py`.

### Fine-tuning cohort and label — RESOLVED (open item 2)

`ICI_meanIntensity.tsv.gz` contains 34 `Sample_ID`s: 30 tumor samples, plus `19RD` and
`Spleen1/2/3`. The responder label comes from **`Data/ST1_sample_info.txt`, column `Response`
(`Yes`/`No`)**, joined by `Sample_ID`. **Exactly 30 tumor samples carry a Response label**
(`19RD` and the three spleen controls have none → excluded). This is the N=30 cohort:

- **14 responders (`Response=Yes`)** / **16 non-responders (`Response=No`)**.
- `irRC` column (`CR/PR/SD/PD`) is the underlying RECIST-like criterion; `Response=Yes` ≈
  CR/PR/SD-with-benefit, `No` ≈ PD. We use the pre-computed binary `Response` directly.
- **Each `Sample_ID` = one patient = one slide = one graph.** The 30 IDs are distinct numbers
  (no patient contributes two samples), so `Sample_ID` serves as `patient_id` for
  leave-one-patient-out. `data.y = 1` if `Response=Yes` else `0`; `data.patient_id =
  data.slide_id = Sample_ID`.

### Cell typing for the Giuliani baseline — RESOLVED (open item 1)

**igraph 1.2.5 is NOT needed and was NOT installed.** `cell_type_clustering.R` would rebuild
cell types from the mean-intensity files, but its published output —
**`Data/ST4_cell_data.txt`** (Supplementary Table 4, the paper's final per-cell type calls) —
**ships in the archive**. We use it directly, bypassing clustering entirely. No deviation in
cell typing versus the paper (we use the paper's own output), and no legacy-toolchain build.

`ST4_cell_data.txt` columns: `obj.id, sample.id, coord.x, coord.y, Cluster, Ki67.prob`.
`obj.id` == ICI `Cell_ID` (verified: 100% cell overlap per sample, identical scheme). Its
`Cluster` uses the paper's **fine** cell types. Restricted to the 30 ICI-labeled samples:
`melano, others, macro.mono (13,199), Tc.ae (4,588), B, Th.ae, CD31, Tc.naive, Th.naive, Treg`.

For the **Giuliani pair-correlation baseline** (Stage 5):
- **macrophage/monocyte** = `Cluster == "macro.mono"` (joined from ST4).
- **activated CD8+ T cell** = `Cluster == "Tc.ae"` (activated/effector cytotoxic T — the
  paper's activated CD8 population), joined from ST4.

`build_graphs.py` merges ST4's fine `Cluster` onto the ICI cells by `Cell_ID == obj.id` and
stores per-node `cell_type` so both the baseline and counterfactual interpretation can use it.

---

## Summary of resolved open items (Stage 1b)

1. **igraph 1.2.5 clustering** → not required; published `ST4_cell_data.txt` cell types used
   directly. No fallback clustering needed.
2. **Responder/non-responder label** → `ST1_sample_info.txt::Response` (Yes/No), joined by
   `Sample_ID`; 30 labeled ICI tumor samples, 14 responders / 16 non-responders.
