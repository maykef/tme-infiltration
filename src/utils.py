"""Shared helpers: seeding, run logging, graph loading."""
import os
import json
import glob
import random
import subprocess
import time
import numpy as np
import torch

REPO = "/mnt/nvme8tb/tme-infiltration"


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def log_run(record):
    """Append one JSON line to results/run_log.jsonl."""
    record = dict(record)
    record.setdefault("git", git_hash())
    record.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(REPO, "results", "run_log.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_graphs(dataset):
    paths = sorted(glob.glob(os.path.join(REPO, "data", "processed", dataset, "*.pt")))
    return [torch.load(p, weights_only=False) for p in paths]


def progress(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(os.path.join(REPO, "results", "progress.log"), "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
