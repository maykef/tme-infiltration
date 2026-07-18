#!/usr/bin/env python
"""Stage 4 — self-supervised masked-marker pretraining on Jackson/Fischer.

Task: for each graph each epoch, pick 15% of nodes; for those nodes mask (zero) 15% of marker
channels and flag them; reconstruct the true (z-scored) values at masked positions with MSE.
GATv2 encoder (hidden 128, 4 heads, 3 layers). Early stopping on held-out masked-MSE.
Saves the encoder (incl. transferable trunk) to checkpoints/pretrained_encoder.pt.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import MaskedMarkerModel
from utils import set_seed, load_graphs, log_run, progress, REPO


def make_mask(x, node_frac, chan_frac, generator):
    """Return (masked_x, mask_flag) with mask_flag=1 at masked (node,channel) positions.

    Vectorized: pick node_frac of nodes, then mask each channel of those nodes independently
    with probability chan_frac (Bernoulli) — ~chan_frac of channels masked per chosen node.
    RNG runs on CPU (reproducible via `generator`), result moved to x's device.
    """
    n, f = x.shape
    n_nodes = max(1, int(round(node_frac * n)))
    node_idx = torch.randperm(n, generator=generator)[:n_nodes]
    chan_mask = (torch.rand(n_nodes, f, generator=generator) < chan_frac).float()
    mask = torch.zeros(n, f)
    mask[node_idx] = chan_mask
    mask = mask.to(x.device)
    masked_x = x * (1.0 - mask)
    return masked_x, mask


def run_epoch(model, loader, device, node_frac, chan_frac, gen, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        masked_x, mask = make_mask(batch.x, node_frac, chan_frac, gen)
        with torch.set_grad_enabled(train):
            pred = model(masked_x, mask, batch.edge_index)
            diff = (pred - batch.x) ** 2 * mask
            denom = mask.sum().clamp(min=1.0)
            loss = diff.sum() / denom
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        m = int(mask.sum().item())
        tot_loss += float(loss.item()) * m
        tot_n += m
    return tot_loss / max(tot_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--node_frac", type=float, default=0.15)
    ap.add_argument("--chan_frac", type=float, default=0.15)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    progress(f"STAGE 4 START — pretraining on Jackson (device={device})")

    graphs = load_graphs("jackson")
    n_markers = graphs[0].x.shape[1]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(graphs))
    n_val = max(1, int(round(args.val_frac * len(graphs))))
    val_idx, train_idx = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
    train_g = [graphs[i] for i in range(len(graphs)) if i in train_idx]
    val_g = [graphs[i] for i in range(len(graphs)) if i in val_idx]
    progress(f"  {len(graphs)} graphs, {n_markers} markers; train={len(train_g)} val={len(val_g)}")

    train_loader = DataLoader(train_g, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_g, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = MaskedMarkerModel(n_markers, args.hidden, args.heads, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(args.seed)

    ckpt_dir = os.path.join(REPO, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "pretrained_encoder.pt")

    best_val, best_epoch, bad = float("inf"), -1, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        tr = run_epoch(model, train_loader, device, args.node_frac, args.chan_frac, gen, opt)
        va = run_epoch(model, val_loader, device, args.node_frac, args.chan_frac, gen, None)
        history.append((epoch, tr, va))
        if va < best_val - 1e-5:
            best_val, best_epoch, bad = va, epoch, 0
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "trunk_state": model.encoder.trunk_state_dict(),
                "n_markers": n_markers, "hidden": args.hidden, "heads": args.heads,
                "layers": args.layers, "val_mse": best_val, "epoch": epoch,
            }, ckpt_path)
        else:
            bad += 1
        if epoch % 5 == 0 or epoch == 1:
            progress(f"  epoch {epoch:3d}  train_mse={tr:.4f}  val_mse={va:.4f}  best={best_val:.4f}@{best_epoch}")
        if bad >= args.patience:
            progress(f"  early stop at epoch {epoch} (no val improvement for {args.patience})")
            break

    elapsed = time.time() - t0
    progress(f"STAGE 4 COMPLETE — best val_mse={best_val:.4f}@epoch{best_epoch}, "
             f"{elapsed:.1f}s, saved {ckpt_path}")
    log_run({"stage": "pretrain", "dataset": "jackson", "n_graphs": len(graphs),
             "n_markers": n_markers, "best_val_mse": best_val, "best_epoch": best_epoch,
             "epochs_ran": len(history), "seconds": round(elapsed, 1),
             "hidden": args.hidden, "heads": args.heads, "layers": args.layers,
             "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed})


if __name__ == "__main__":
    main()
