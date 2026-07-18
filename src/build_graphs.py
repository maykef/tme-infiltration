#!/usr/bin/env python
"""Stage 3 — spatial kNN graph construction.

One torch_geometric.data.Data per slide -> data/processed/{dataset}/{slide_id}.pt

Node = cell. Node features = arcsinh-transformed marker vector, z-scored per marker
*within each dataset separately*. Edges = kNN in (x, y), k configurable (default 12),
made undirected. Graph-level attributes: slide_id (both); patient_id + y (response) for
Moldoveanu. Also stores pos (coords), cell_id, and cell_type (Moldoveanu) so the
counterfactual and Giuliani-baseline stages can map back to cells.

Parallelization: joblib.Parallel(n_jobs, backend="loky") over slides; single-threaded BLAS
per worker (set below, before numpy import) to avoid oversubscription.
"""
# --- single-threaded BLAS in every process (parent + loky workers re-import this module) ---
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from torch_geometric.data import Data

REPO = "/mnt/nvme8tb/tme-infiltration"

# ---- marker selections (from docs/data_schema.md) ----
JACKSON_DROP = {"Ru96", "Ru98", "Ru99", "Ru100", "Ru101", "Ru102", "Ru104", "DNA1", "DNA2"}
MOLD_DROP = {"X190BCKG", "X191Ir", "X193Ir"}
MOLD_MARKER_COLS = [  # cols 5..42 of ICI_meanIntensity, in file order
    "SMA", "PDL1", "OX40", "CD45", "LAG3", "TIM3", "FoxP3", "CD4", "CCR7", "CD68", "VISTA",
    "MEK1.2", "CD20", "CD8a", "pMEK1.2", "SOX10", "B.Catenin", "CD45RA", "GranzymeB", "CD40",
    "CollagenI", "CD3", "Ki67", "pERK1.2", "cleaved.Caspase3", "CD45RO", "HLA.DR", "S100",
    "Histone.H3", "X190BCKG", "X191Ir", "X193Ir", "CD14", "ERK1.2", "CD16", "CD31", "ICOS",
    "CD29",
]


def build_edges(xy, k):
    """kNN edge_index (undirected, no self loops) from (N,2) coords."""
    n = xy.shape[0]
    kk = min(k, n - 1)
    if kk < 1:
        return torch.empty((2, 0), dtype=torch.long)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=kk + 1)  # includes self at col 0
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n), kk)
    dst = idx[:, 1:].reshape(-1)
    ei = np.stack([src, dst], axis=0)
    # make undirected: add reverse, dedup
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    ei = np.unique(ei, axis=1)
    return torch.from_numpy(ei).long()


# ----------------------- Jackson -----------------------
def build_jackson_slide(slide_id, sub, markers, mean, std, k):
    feats = sub[markers].to_numpy(dtype=np.float32)          # already arcsinh (exprs assay)
    feats = np.nan_to_num(feats, nan=0.0)                     # safety net for stray NaNs
    z = (feats - mean) / std
    xy = sub[["x", "y"]].to_numpy(dtype=np.float64)
    data = Data(
        x=torch.from_numpy(z.astype(np.float32)),
        edge_index=build_edges(xy, k),
        pos=torch.from_numpy(xy.astype(np.float32)),
    )
    data.slide_id = str(slide_id)
    data.dataset = "jackson"
    data.cell_id = list(sub["cell_id"].astype(str))
    data.num_nodes = z.shape[0]
    return slide_id, data


def run_jackson(k, n_jobs):
    pq = os.path.join(REPO, "data", "jackson_processed", "jackson_cells.parquet")
    df = pd.read_parquet(pq)
    all_cols = [c for c in df.columns if c not in ("image_id", "cell_id", "x", "y")]
    markers = [c for c in all_cols if c not in JACKSON_DROP]
    # drop channels that are entirely NaN in the exprs assay (package excludes EpCAM/CTNNB/SOX9)
    allnan = [m for m in markers if df[m].isna().all()]
    if allnan:
        print(f"[jackson] dropping {len(allnan)} all-NaN exprs channels: {allnan}")
        markers = [m for m in markers if m not in allnan]
    print(f"[jackson] {len(df)} cells, {df['image_id'].nunique()} slides, "
          f"{len(markers)} markers (dropped {len(all_cols)-len(markers)} technical/empty)")
    # per-dataset z-score stats (global over all cells)
    X = df[markers].to_numpy(dtype=np.float32)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std < 1e-6] = 1.0
    outdir = os.path.join(REPO, "data", "processed", "jackson")
    os.makedirs(outdir, exist_ok=True)
    groups = list(df.groupby("image_id"))
    print(f"[jackson] building {len(groups)} graphs with n_jobs={n_jobs}, k={k}")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(build_jackson_slide)(sid, sub.reset_index(drop=True), markers, mean, std, k)
        for sid, sub in groups
    )
    for sid, data in results:
        torch.save(data, os.path.join(outdir, f"{_safe(sid)}.pt"))
    _write_marker_list("jackson", markers)
    print(f"[jackson] wrote {len(results)} graphs to {outdir}")
    return len(results)


# ----------------------- Moldoveanu -----------------------
def load_moldoveanu_cells():
    base = os.path.join(REPO, "data", "moldoveanu_raw", "Moldoveanu_2022_CyTOF_melanoma", "Data")
    ici = pd.read_csv(os.path.join(base, "ICI_meanIntensity.tsv.gz"), sep="\t")
    st1 = pd.read_csv(os.path.join(base, "ST1_sample_info.txt"), sep="\t")
    st4 = pd.read_csv(os.path.join(base, "ST4_cell_data.txt"), sep="\t")
    # response label per sample
    resp = st1[["Sample_ID", "Response"]].dropna(subset=["Response"])
    resp = resp[resp["Response"].isin(["Yes", "No"])]
    resp_map = dict(zip(resp["Sample_ID"], (resp["Response"] == "Yes").astype(int)))
    # fine cell type from ST4 by Cell_ID == obj.id
    st4_map = dict(zip(st4["obj.id"].astype(str), st4["Cluster"].astype(str)))
    ici["fine_type"] = ici["Cell_ID"].astype(str).map(st4_map).fillna("NA")
    # keep only labeled samples (the 30-patient cohort)
    ici = ici[ici["Sample_ID"].isin(resp_map)].copy()
    ici["response"] = ici["Sample_ID"].map(resp_map).astype(int)
    return ici, resp_map


def build_mold_slide(slide_id, sub, markers, mean, std, k):
    raw = sub[markers].to_numpy(dtype=np.float32)
    arc = np.arcsinh(raw / 5.0)                              # match Jackson exprs transform
    z = (arc - mean) / std
    xy = sub[["Location_Center_X", "Location_Center_Y"]].to_numpy(dtype=np.float64)
    y = int(sub["response"].iloc[0])
    data = Data(
        x=torch.from_numpy(z.astype(np.float32)),
        edge_index=build_edges(xy, k),
        pos=torch.from_numpy(xy.astype(np.float32)),
        y=torch.tensor([y], dtype=torch.long),
    )
    data.slide_id = str(slide_id)
    data.patient_id = str(slide_id)
    data.response_label = y
    data.dataset = "moldoveanu"
    data.cell_id = list(sub["Cell_ID"].astype(str))
    data.cell_type = list(sub["fine_type"].astype(str))
    data.num_nodes = z.shape[0]
    return slide_id, data


def run_moldoveanu(k, n_jobs):
    ici, resp_map = load_moldoveanu_cells()
    markers = [m for m in MOLD_MARKER_COLS if m not in MOLD_DROP]
    n_resp = sum(resp_map.values())
    print(f"[moldoveanu] {len(ici)} cells, {ici['Sample_ID'].nunique()} labeled slides "
          f"({n_resp} responders / {len(resp_map)-n_resp} non-responders), {len(markers)} markers")
    # per-dataset z-score on arcsinh(x/5) over all cells
    arc = np.arcsinh(ici[markers].to_numpy(dtype=np.float32) / 5.0)
    mean = arc.mean(axis=0)
    std = arc.std(axis=0)
    std[std < 1e-6] = 1.0
    outdir = os.path.join(REPO, "data", "processed", "moldoveanu")
    os.makedirs(outdir, exist_ok=True)
    groups = list(ici.groupby("Sample_ID"))
    print(f"[moldoveanu] building {len(groups)} graphs with n_jobs={n_jobs}, k={k}")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(build_mold_slide)(sid, sub.reset_index(drop=True), markers, mean, std, k)
        for sid, sub in groups
    )
    for sid, data in results:
        torch.save(data, os.path.join(outdir, f"{_safe(sid)}.pt"))
    _write_marker_list("moldoveanu", markers)
    # label sidecar for quick reference
    lbl = pd.DataFrame({"slide_id": list(resp_map.keys()),
                        "response": list(resp_map.values())})
    lbl.to_csv(os.path.join(outdir, "_labels.csv"), index=False)
    print(f"[moldoveanu] wrote {len(results)} graphs to {outdir}")
    return len(results)


# ----------------------- utils -----------------------
def _safe(s):
    return str(s).replace("/", "_").replace(" ", "_")


def _write_marker_list(dataset, markers):
    d = os.path.join(REPO, "data", "processed", dataset)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "_markers.txt"), "w") as f:
        f.write("\n".join(markers) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jackson", "moldoveanu", "both"], default="both")
    ap.add_argument("--k", type=int, default=12, help="kNN neighbors")
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    counts = {}
    if args.dataset in ("jackson", "both"):
        counts["jackson"] = run_jackson(args.k, args.n_jobs)
    if args.dataset in ("moldoveanu", "both"):
        counts["moldoveanu"] = run_moldoveanu(args.k, args.n_jobs)
    print(f"[done] {counts} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
