#!/usr/bin/env Rscript
# Stage 1a — Jackson & Fischer 2020 breast cancer IMC -> flat parquet for Python.
# Downloads via Bioconductor ExperimentHub (imcdatasets). Logs schema before exporting.
suppressMessages({
  library(imcdatasets)
  library(SingleCellExperiment)
  library(SpatialExperiment)
  library(S4Vectors)
  library(arrow)
})

repo <- "/mnt/nvme8tb/tme-infiltration"
outdir <- file.path(repo, "data", "jackson_processed")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
logf <- file.path(repo, "results", "jackson_schema.log")
log <- function(...) cat(..., "\n", file = logf, append = TRUE)
cat("Jackson/Fischer 2020 export —", format(Sys.time()), "\n", file = logf)

log("Loading JacksonFischer_2020_BreastCancer(data_type='sce') ...")
sce <- JacksonFischer_2020_BreastCancer(data_type = "sce")

log("class:", class(sce))
log("dim (markers x cells):", paste(dim(sce), collapse = " x "))
log("assayNames:", paste(assayNames(sce), collapse = ", "))
log("rowData cols:", paste(colnames(rowData(sce)), collapse = ", "))
log("colData cols:", paste(colnames(colData(sce)), collapse = ", "))
cd <- as.data.frame(colData(sce))
log("colData head:")
capture.output(print(utils::head(cd, 3)), file = logf, append = TRUE)
log("rowData head:")
capture.output(print(utils::head(as.data.frame(rowData(sce)), 40)), file = logf, append = TRUE)

# --- locate spatial coordinates ---
xcol <- NULL; ycol <- NULL
if (is(sce, "SpatialExperiment") && ncol(spatialCoords(sce)) >= 2) {
  sc <- spatialCoords(sce)
  coords <- data.frame(x = sc[, 1], y = sc[, 2])
  log("coords: from spatialCoords() cols", paste(colnames(sc), collapse = ","))
} else {
  cand_x <- c("Location_Center_X", "Center_X", "x", "X", "Pos_X", "cell_x")
  cand_y <- c("Location_Center_Y", "Center_Y", "y", "Y", "Pos_Y", "cell_y")
  xcol <- cand_x[cand_x %in% colnames(cd)][1]
  ycol <- cand_y[cand_y %in% colnames(cd)][1]
  if (is.na(xcol) || is.na(ycol)) {
    # fall back: any column pair matching x/y ignoring case
    xcol <- grep("_x$|center.?x|pos.?x|^x$", colnames(cd), ignore.case = TRUE, value = TRUE)[1]
    ycol <- grep("_y$|center.?y|pos.?y|^y$", colnames(cd), ignore.case = TRUE, value = TRUE)[1]
  }
  coords <- data.frame(x = cd[[xcol]], y = cd[[ycol]])
  log("coords: from colData cols x=", xcol, " y=", ycol)
}

# --- locate image / slide id ---
cand_img <- c("ImageNumber", "image_id", "ImageId", "core", "Core", "image", "ImageName", "sample_id")
imgcol <- cand_img[cand_img %in% colnames(cd)][1]
if (is.na(imgcol)) imgcol <- grep("image|core|sample", colnames(cd), ignore.case = TRUE, value = TRUE)[1]
log("image/slide id col:", imgcol)

# --- cell id ---
cand_cell <- c("CellId", "cell_id", "CellNumber", "ObjectNumber", "id")
cellcol <- cand_cell[cand_cell %in% colnames(cd)][1]
cell_id <- if (!is.na(cellcol)) cd[[cellcol]] else seq_len(ncol(sce))
log("cell id col:", ifelse(is.na(cellcol), "<generated 1..N>", cellcol))

# --- choose assay: prefer arcsinh-transformed if present, else asinh(counts/5) ---
an <- assayNames(sce)
pick <- NULL
for (nm in c("exprs", "asinh", "arcsinh", "normalized", "logcounts")) {
  if (nm %in% an) { pick <- nm; break }
}
if (!is.null(pick)) {
  M <- assay(sce, pick)
  transform_note <- paste0("used pre-existing assay '", pick, "' (assumed arcsinh/transformed)")
} else {
  raw_nm <- if ("counts" %in% an) "counts" else an[1]
  M <- asinh(assay(sce, raw_nm) / 5)
  transform_note <- paste0("applied asinh(x/5) to assay '", raw_nm, "'")
}
log("assay transform:", transform_note)

# marker names = rownames
markers <- rownames(sce)
if (is.null(markers)) markers <- paste0("marker_", seq_len(nrow(sce)))
# sanitize marker names for column safety
markers_clean <- make.names(markers, unique = TRUE)
log("n markers:", length(markers))
log("markers:", paste(markers_clean, collapse = ", "))

# Build data.frame: one row per cell
expr <- t(as.matrix(M))                     # cells x markers
colnames(expr) <- markers_clean
df <- data.frame(
  image_id = as.character(cd[[imgcol]]),
  cell_id  = as.character(cell_id),
  x = as.numeric(coords$x),
  y = as.numeric(coords$y),
  check.names = FALSE
)
df <- cbind(df, as.data.frame(expr, check.names = FALSE))
log("final table dim:", paste(dim(df), collapse = " x "))
log("n unique images:", length(unique(df$image_id)))

outfile <- file.path(outdir, "jackson_cells.parquet")
arrow::write_parquet(df, outfile)
log("WROTE", outfile, "(", round(file.info(outfile)$size / 1e6, 1), "MB )")

# also dump the marker list + resolved columns as a small json-ish sidecar for Python/schema doc
meta <- list(
  image_id_col = imgcol, cell_id_col = ifelse(is.na(cellcol), "<generated>", cellcol),
  x_col = ifelse(is.null(xcol), "spatialCoords[,1]", xcol),
  y_col = ifelse(is.null(ycol), "spatialCoords[,2]", ycol),
  transform = transform_note, n_markers = length(markers), n_images = length(unique(df$image_id)),
  n_cells = nrow(df), markers = markers_clean
)
saveRDS(meta, file.path(outdir, "jackson_meta.rds"))
tryCatch(
  writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
             file.path(outdir, "jackson_meta.json")),
  error = function(e) log("jsonlite unavailable, skipped json sidecar:", conditionMessage(e))
)
cat("JACKSON EXPORT DONE\n", file = logf, append = TRUE)
cat("JACKSON EXPORT DONE\n")
