#!/usr/bin/env python
"""Stage 5 — fine-tuning + leave-one-patient-out (LOPO) evaluation on Moldoveanu.

GNN: load the pretrained GATv2 trunk, attach a fresh Moldoveanu input stem + attention pooling
+ 2-layer MLP head -> binary responder logit. 30-fold LOPO (train 29, predict 1), collect
out-of-fold probabilities, AUROC with a bootstrap CI over patients.

Baseline (same data, same labels): Giuliani et al. cross pair-correlation between
macrophage/monocyte and activated CD8+ T (Tc.ae) at r=10.5um per slide; AUROC of that single
scalar. Both AUROCs printed side by side, non-responders flagged if the GNN doesn't win.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from model import SlideClassifier
from utils import set_seed, load_graphs, log_run, progress, REPO


# ---------------- Giuliani cross pair-correlation baseline ----------------
def cross_pcf(pos_a, pos_b, window_area, r=10.5, dr=5.0):
    """Bivariate pair-correlation g_AB(r): density of B cells in annulus [r-dr/2, r+dr/2]
    around A cells, normalized by CSR expectation. >1 co-localization, <1 avoidance."""
    na, nb = len(pos_a), len(pos_b)
    if na == 0 or nb == 0 or window_area <= 0:
        return 0.0
    r_out, r_in = r + dr / 2.0, max(0.0, r - dr / 2.0)
    tree_b = cKDTree(pos_b)
    n_out = np.sum(tree_b.query_ball_point(pos_a, r_out, return_length=True))
    n_in = np.sum(tree_b.query_ball_point(pos_a, r_in, return_length=True))
    count_annulus = float(n_out - n_in)
    lambda_b = nb / window_area
    annulus_area = np.pi * (r_out ** 2 - r_in ** 2)
    expected = na * lambda_b * annulus_area
    if expected <= 0:
        return 0.0
    return count_annulus / expected


def giuliani_feature(g, r=10.5, dr=5.0):
    pos = g.pos.numpy()
    ct = np.array(g.cell_type)
    a = pos[ct == "macro.mono"]
    b = pos[ct == "Tc.ae"]
    xmin, ymin = pos.min(0)
    xmax, ymax = pos.max(0)
    window_area = max((xmax - xmin) * (ymax - ymin), 1.0)
    return cross_pcf(a, b, window_area, r, dr)


# ---------------- bootstrap CI ----------------
def bootstrap_auc_ci(labels, scores, n_boot=2000, seed=42):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    point = roc_auc_score(labels, scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi), len(aucs)


# ---------------- GNN LOPO ----------------
def train_fold(train_graphs, model, device, epochs, lr, wd):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossfn = nn.BCEWithLogitsLoss()
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            logit = model(batch.x, batch.edge_index, batch.batch)
            loss = lossfn(logit, batch.y.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


@torch.no_grad()
def predict_graph(g, model, device):
    model.eval()
    x = g.x.to(device)                       # move copies only; do not mutate stored graph
    ei = g.edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    logit = model(x, ei, batch)
    return float(torch.sigmoid(logit).item())


def run_gnn_lopo(graphs, args, device, ckpt):
    n_markers = graphs[0].x.shape[1]
    labels = np.array([int(g.y) for g in graphs])
    slide_ids = [g.slide_id for g in graphs]
    oof = np.full(len(graphs), np.nan)
    for i in range(len(graphs)):
        set_seed(args.seed + i)
        train_g = [graphs[j] for j in range(len(graphs)) if j != i]
        model = SlideClassifier(n_markers, args.hidden, args.heads, args.layers,
                                dropout=args.dropout).to(device)
        if ckpt is not None:
            model.encoder.load_trunk(ckpt["trunk_state"], freeze=args.freeze_trunk)
        train_fold(train_g, model, device, args.epochs, args.lr, args.wd)
        oof[i] = predict_graph(graphs[i], model, device)
        if (i + 1) % 5 == 0:
            progress(f"  LOPO fold {i+1}/{len(graphs)} done (held out {slide_ids[i]})")
    return labels, oof, slide_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--freeze_trunk", action="store_true",
                    help="freeze pretrained GATv2 trunk, train only stem/pool/head")
    ap.add_argument("--no_pretrain", action="store_true", help="random init (ablation)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--r", type=float, default=10.5)
    ap.add_argument("--tag", type=str, default="", help="suffix for output json (ablations)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    progress(f"STAGE 5 START — LOPO fine-tune + baseline (device={device})")

    graphs = load_graphs("moldoveanu")
    labels = np.array([int(g.y) for g in graphs])
    progress(f"  {len(graphs)} slides, responders={int(labels.sum())} "
             f"non-responders={int((1-labels).sum())}")

    # ---- Giuliani baseline ----
    feats = np.array([giuliani_feature(g, r=args.r) for g in graphs])
    b_auc, b_lo, b_hi, b_n = bootstrap_auc_ci(labels, feats, args.n_boot, args.seed)
    progress(f"  Giuliani cross-PCF(macro.mono~Tc.ae, r={args.r}um) AUROC="
             f"{b_auc:.3f} [{b_lo:.3f}, {b_hi:.3f}]")

    # ---- GNN LOPO ----
    ckpt = None
    if not args.no_pretrain:
        cp = os.path.join(REPO, "checkpoints", "pretrained_encoder.pt")
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        progress(f"  loaded pretrained trunk (val_mse={ckpt.get('val_mse'):.4f})")
    labels2, oof, slide_ids = run_gnn_lopo(graphs, args, device, ckpt)
    g_auc, g_lo, g_hi, g_n = bootstrap_auc_ci(labels2, oof, args.n_boot, args.seed)
    elapsed = time.time() - t0

    # ---- report ----
    verdict = "GNN BEATS baseline" if g_auc > b_auc else "GNN does NOT beat baseline"
    progress("  " + "=" * 62)
    progress(f"  GNN (LOPO)          AUROC = {g_auc:.3f}  95% CI [{g_lo:.3f}, {g_hi:.3f}]")
    progress(f"  Giuliani baseline   AUROC = {b_auc:.3f}  95% CI [{b_lo:.3f}, {b_hi:.3f}]")
    progress(f"  --> {verdict}")
    progress("  " + "=" * 62)

    # persist out-of-fold predictions + features for RESULTS + counterfactual
    out = {
        "slide_ids": slide_ids, "labels": labels2.tolist(),
        "gnn_oof_prob": oof.tolist(), "giuliani_feature": feats.tolist(),
        "gnn_auc": g_auc, "gnn_ci": [g_lo, g_hi],
        "baseline_auc": b_auc, "baseline_ci": [b_lo, b_hi],
        "verdict": verdict, "n_boot_effective": {"gnn": g_n, "baseline": b_n},
        "r_um": args.r, "elapsed_s": round(elapsed, 1),
        "config": {k: getattr(args, k) for k in
                   ["seed", "hidden", "heads", "layers", "dropout", "epochs", "lr", "wd",
                    "freeze_trunk", "no_pretrain"]},
    }
    fname = f"eval_results{('_' + args.tag) if args.tag else ''}.json"
    with open(os.path.join(REPO, "results", fname), "w") as f:
        json.dump(out, f, indent=2)
    progress(f"STAGE 5 COMPLETE — {elapsed:.1f}s, wrote results/{fname}")
    log_run({"stage": "finetune_eval", "gnn_auc": g_auc, "gnn_ci": [g_lo, g_hi],
             "baseline_auc": b_auc, "baseline_ci": [b_lo, b_hi], "verdict": verdict,
             "seconds": round(elapsed, 1), "seed": args.seed, "epochs": args.epochs,
             "lr": args.lr, "freeze_trunk": args.freeze_trunk, "no_pretrain": args.no_pretrain})


if __name__ == "__main__":
    main()
