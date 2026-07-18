#!/usr/bin/env Rscript
# Panel-match Part 1 — dump marker panels of candidate imcdatasets (metadata only, no export).
suppressMessages({library(imcdatasets); library(SingleCellExperiment)})
repo <- "/mnt/nvme8tb/tme-infiltration"
outdir <- file.path(repo, "data", "panels"); dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

d <- as.data.frame(listDatasets())
write.csv(d, file.path(outdir, "all_datasets.csv"), row.names = FALSE)

cands <- c("HochSchulz_2022_Melanoma", "IMMUcan_2022_CancerExample")
for (fn in cands) {
  sce <- tryCatch(get(fn)(data_type = "sce"), error = function(e) NULL)
  if (is.null(sce)) { cat("FAILED", fn, "\n"); next }
  rn <- rownames(sce)
  nm <- if ("name" %in% colnames(rowData(sce))) as.character(rowData(sce)$name) else rn
  # write both the rowname id and the clean 'name' (one per line, tab-separated)
  writeLines(paste(rn, nm, sep = "\t"), file.path(outdir, paste0(fn, "_panel.tsv")))
  cat(fn, ":", nrow(sce), "channels,", ncol(sce), "cells\n")
}
cat("PANELS DUMPED\n")
