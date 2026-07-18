#!/usr/bin/env python
"""Paired-difference significance test: (baseline AUROC - GNN AUROC).

Requested last round; applied here RETROACTIVELY to the already-completed full-Jackson
corrected result only. It uses the stored out-of-fold GNN probabilities and Giuliani features
in results/eval_results.json — no retraining, no new data. (The Part-2 application to a
panel-matched corpus is gated off by the Part-1 NO-GO and is not run.)

In each of N bootstrap iterations we resample the SAME patient indices for both models, compute
each model's AUROC on that resample, and record the difference. If the 95% CI on
(baseline - GNN) excludes zero, the baseline significantly outperforms the GNN in a paired
sense.
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import REPO, log_run


def paired_test(labels, gnn, base, n_boot=2000, seed=42):
    labels, gnn, base = map(np.asarray, (labels, gnn, base))
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = roc_auc_score(labels, base) - roc_auc_score(labels, gnn)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                     # SAME indices for both models
        if len(np.unique(labels[idx])) < 2:
            continue
        d = roc_auc_score(labels[idx], base[idx]) - roc_auc_score(labels[idx], gnn[idx])
        diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_gt0 = float(np.mean(np.array(diffs) > 0))          # fraction favouring baseline
    return {"point_diff": float(point), "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "frac_favouring_baseline": p_gt0,
            "n_effective": len(diffs)}


def main():
    src = os.path.join(REPO, "results", "eval_results.json")
    d = json.load(open(src))
    res = paired_test(d["labels"], d["gnn_oof_prob"], d["giuliani_feature"],
                      n_boot=2000, seed=42)
    res["source"] = "full-Jackson corrected run (retroactive)"
    res["gnn_auc"] = d["gnn_auc"]
    res["baseline_auc"] = d["baseline_auc"]
    out = os.path.join(REPO, "results", "paired_diff.json")
    json.dump(res, open(out, "w"), indent=2)
    verdict = ("baseline significantly beats GNN (paired CI excludes 0)"
               if res["excludes_zero"] else
               "NOT significant (paired CI on baseline-GNN includes 0)")
    print(f"paired (baseline - GNN) diff = {res['point_diff']:+.3f} "
          f"95% CI [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}] -> {verdict}; "
          f"{100*res['frac_favouring_baseline']:.0f}% of resamples favour baseline")
    log_run({"stage": "paired_diff_test", "source": "full_jackson_retroactive",
             **{k: res[k] for k in ["point_diff", "ci", "excludes_zero",
                                    "frac_favouring_baseline"]}})
    print("wrote", out)


if __name__ == "__main__":
    main()
