#!/usr/bin/env python
"""Stage 6 — gradient-based counterfactual search over the fine-tuned model.

Trains one SlideClassifier on all 30 Moldoveanu slides (pretrained trunk + fine-tune), then
for each target slide computes d P(response) / d(node features). Per-node gradient magnitude
ranks which cells, if their marker profile shifted (in the gradient direction), would most
increase predicted response probability; the dominant marker direction per node names the
shift. Outputs a ranked CSV + a position scatter PNG (colored by gradient magnitude) per slide,
and an aggregate top-findings summary for RESULTS.md.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import SlideClassifier
from utils import set_seed, load_graphs, log_run, progress, REPO


def load_markers():
    with open(os.path.join(REPO, "data", "processed", "moldoveanu", "_markers.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def train_full(graphs, args, device, ckpt):
    from torch_geometric.loader import DataLoader
    n_markers = graphs[0].x.shape[1]
    model = SlideClassifier(n_markers, args.hidden, args.heads, args.layers,
                            dropout=args.dropout).to(device)
    if ckpt is not None:
        model.encoder.load_trunk(ckpt["trunk_state"], freeze=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossfn = nn.BCEWithLogitsLoss()
    loader = DataLoader(graphs, batch_size=8, shuffle=True)
    model.train()
    for _ in range(args.epochs):
        for batch in loader:
            batch = batch.to(device)
            logit = model(batch.x, batch.edge_index, batch.batch)
            loss = lossfn(logit, batch.y.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def counterfactual_slide(g, model, device, markers):
    """Return (per_node DataFrame, prob) with grad of P(resp) wrt node features."""
    model.eval()
    x = g.x.to(device).clone().requires_grad_(True)
    ei = g.edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    logit = model(x, ei, batch)
    prob = torch.sigmoid(logit)
    grad = torch.autograd.grad(prob.sum(), x, retain_graph=True)[0]  # (N, F) dP/dx
    grad = grad.detach().cpu().numpy()
    mag = np.linalg.norm(grad, axis=1)                              # per-node magnitude
    dom_idx = np.argmax(np.abs(grad), axis=1)
    dom_marker = [markers[j] for j in dom_idx]
    dom_signed = grad[np.arange(len(grad)), dom_idx]
    dom_dir = np.where(dom_signed >= 0, "increase", "decrease")
    df = pd.DataFrame({
        "cell_id": list(g.cell_id),
        "x": g.pos[:, 0].numpy(), "y": g.pos[:, 1].numpy(),
        "cell_type": list(g.cell_type),
        "grad_magnitude": mag,
        "dominant_marker": dom_marker,
        "dominant_direction": dom_dir,
        "dominant_grad": dom_signed,
    }).sort_values("grad_magnitude", ascending=False).reset_index(drop=True)
    return df, float(prob.item())


def plot_slide(df, slide_id, prob, outpng):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["x"], df["y"], c=df["grad_magnitude"], s=6, cmap="magma")
    ax.set_title(f"{slide_id}  P(response)={prob:.3f}\ncells colored by |d P/d features|")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_aspect("equal"); ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="gradient magnitude")
    fig.tight_layout(); fig.savefig(outpng, dpi=110); plt.close(fig)


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
    ap.add_argument("--targets", choices=["nonresponders", "all"], default="nonresponders")
    ap.add_argument("--top_k", type=int, default=25, help="top cells per slide kept in summary")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    progress(f"STAGE 6 START — counterfactual gradient search (device={device})")

    graphs = load_graphs("moldoveanu")
    markers = load_markers()
    cp = os.path.join(REPO, "checkpoints", "pretrained_encoder.pt")
    ckpt = torch.load(cp, map_location=device, weights_only=False) if os.path.exists(cp) else None
    model = train_full(graphs, args, device, ckpt)
    torch.save(model.state_dict(), os.path.join(REPO, "checkpoints", "finetuned_full.pt"))
    progress("  trained final model on all 30 slides")

    outdir = os.path.join(REPO, "results", "counterfactuals")
    os.makedirs(outdir, exist_ok=True)
    targets = [g for g in graphs if (args.targets == "all" or int(g.y) == 0)]
    progress(f"  computing counterfactuals for {len(targets)} slides ({args.targets})")

    summary_rows = []
    for g in targets:
        df, prob = counterfactual_slide(g, model, device, markers)
        df.to_csv(os.path.join(outdir, f"{g.slide_id}.csv"), index=False)
        plot_slide(df, g.slide_id, prob, os.path.join(outdir, f"{g.slide_id}.png"))
        top = df.head(args.top_k)
        # dominant marker+direction among this slide's most influential cells
        combo = (top["dominant_marker"] + " (" + top["dominant_direction"] + ")").value_counts()
        summary_rows.append({
            "slide_id": g.slide_id, "response_label": int(g.y), "pred_prob": round(prob, 3),
            "n_cells": len(df), "max_grad": round(float(df["grad_magnitude"].max()), 5),
            "top_marker_shift": combo.index[0], "top_marker_count": int(combo.iloc[0]),
            "top_cell_type": top["cell_type"].value_counts().index[0],
        })
    summ = pd.DataFrame(summary_rows).sort_values("max_grad", ascending=False)
    summ.to_csv(os.path.join(outdir, "_summary.csv"), index=False)

    # aggregate top-5 findings across slides (by max gradient)
    findings = []
    for _, row in summ.head(5).iterrows():
        findings.append(
            f"In non-responder slide {row['slide_id']} (model P(response)={row['pred_prob']}), "
            f"the most influential cells are {row['top_cell_type']} cells, and the counterfactual "
            f"gradient most often points to '{row['top_marker_shift']}' "
            f"({row['top_marker_count']}/{args.top_k} of the top cells) as the marker shift that "
            f"would most raise predicted response."
        )
    with open(os.path.join(outdir, "_top_findings.json"), "w") as f:
        json.dump({"findings": findings,
                   "global_top_marker_shifts":
                       summ["top_marker_shift"].value_counts().head(8).to_dict()}, f, indent=2)

    elapsed = time.time() - t0
    for i, fnd in enumerate(findings, 1):
        progress(f"  finding {i}: {fnd}")
    progress(f"STAGE 6 COMPLETE — {elapsed:.1f}s, {len(targets)} slides -> {outdir}")
    log_run({"stage": "counterfactual", "n_target_slides": len(targets),
             "targets": args.targets, "seconds": round(elapsed, 1), "seed": args.seed})


if __name__ == "__main__":
    main()
