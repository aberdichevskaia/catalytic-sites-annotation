#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seed-ensemble evaluation for cross-validation models.

For a given experiment prefix (e.g. ablate09_esm2_CataloDB_3B_graphV2):
  1. Discover all fold×seed directories automatically.
  2. Average predictions across all seeds within each fold.
  3. Average the per-fold ensembles into a global test ensemble.
  4. Find the best F1 threshold on OOF ensemble validation predictions.
  5. Report test AUCPR, F1, Precision, Recall.

Usage:
  python evaluate_ensemble.py \\
      --base_dir /path/to/scannet_retrains \\
      --prefix   ablate09_esm2_CataloDB_3B_graphV2

Directory structure expected:
  {base_dir}/{prefix}_fold{F}_v{V}/
      test_results.pkl
      validation_results.pkl
"""

import argparse
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    auc, f1_score, precision_recall_curve, precision_score, recall_score,
)


# ---- pickle helpers ----

def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_score(arr: np.ndarray) -> np.ndarray:
    """Collapse (L, C) predictions to 1D score (last/positive channel)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        return arr[..., -1].reshape(-1)
    return arr.reshape(-1)


def _to_binary(arr: np.ndarray) -> np.ndarray:
    """Collapse labels to 0/1."""
    arr = np.asarray(arr)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        return (np.argmax(arr, axis=-1) > 0).astype(np.int32).reshape(-1)
    return (arr.reshape(-1) > 0).astype(np.int32)


# ---- averaging ----

def average_pickles(pkl_list: list) -> dict:
    """
    Average predictions from multiple pickles (same ids, same structure).
    Aligns by id string; skips entries with shape mismatches.
    """
    base = pkl_list[0]
    ids = [str(i) for i in base["ids"]]
    sum_preds = [np.asarray(p, dtype=np.float32).copy() for p in base["predictions"]]
    counts = [1] * len(ids)

    for pl in pkl_list[1:]:
        other = {str(i): np.asarray(p, dtype=np.float32)
                 for i, p in zip(pl["ids"], pl["predictions"])}
        for j, id_ in enumerate(ids):
            if id_ in other and other[id_].shape == sum_preds[j].shape:
                sum_preds[j] += other[id_]
                counts[j] += 1

    avg_preds = [s / c for s, c in zip(sum_preds, counts)]
    return {
        "ids": ids,
        "labels": list(base["labels"]),
        "predictions": avg_preds,
        "weights": list(base["weights"]),
        "splits": list(base["splits"]),
    }


# ---- metrics ----

def compute_aucpr(payload: dict) -> float:
    y = np.concatenate([_to_binary(l) for l in payload["labels"]])
    p = np.concatenate([_to_score(pr) for pr in payload["predictions"]])
    ok = np.isfinite(p) & np.isfinite(y.astype(float))
    prec, rec, _ = precision_recall_curve(y[ok], p[ok])
    return float(auc(rec, prec))


def find_best_threshold(payloads: list) -> tuple[float, float]:
    """Find threshold maximising F1 on concatenated OOF predictions."""
    all_y, all_p = [], []
    for pl in payloads:
        all_y.append(np.concatenate([_to_binary(l) for l in pl["labels"]]))
        all_p.append(np.concatenate([_to_score(pr) for pr in pl["predictions"]]))
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    ok = np.isfinite(p) & np.isfinite(y.astype(float))
    prec, rec, thrs = precision_recall_curve(y[ok], p[ok])
    denom = prec[1:] + rec[1:]
    f1s = np.where(denom > 0, 2 * prec[1:] * rec[1:] / denom, 0.0)
    best = int(np.argmax(f1s))
    return float(thrs[best]), float(f1s[best])


def eval_at_threshold(payload: dict, threshold: float) -> dict:
    y = np.concatenate([_to_binary(l) for l in payload["labels"]])
    p = np.concatenate([_to_score(pr) for pr in payload["predictions"]])
    y_hat = (p >= threshold).astype(np.int32)
    return {
        "f1":        float(f1_score(y, y_hat, zero_division=0)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall":    float(recall_score(y, y_hat, zero_division=0)),
    }


# ---- discovery ----

def discover_runs(base_dir: str, prefix: str) -> dict[int, list[int]]:
    """
    Find all {prefix}_fold{F}_v{V} directories.
    Returns {fold: [versions]} sorted.
    """
    pattern = os.path.join(base_dir, f"{prefix}_fold*_v*")
    dirs = glob.glob(pattern)
    runs = defaultdict(list)
    for d in dirs:
        m = re.search(r"_fold(\d+)_v(\d+)$", os.path.basename(d))
        if m:
            runs[int(m.group(1))].append(int(m.group(2)))
    return {f: sorted(vs) for f, vs in sorted(runs.items())}


# ---- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True,
                    help="Directory containing the fold×seed model directories.")
    ap.add_argument("--prefix", required=True,
                    help="Common prefix of model directories, e.g. ablate09_esm2_CataloDB_3B_graphV2.")
    args = ap.parse_args()

    runs = discover_runs(args.base_dir, args.prefix)
    if not runs:
        raise SystemExit(f"No directories found matching {args.prefix}_fold*_v* in {args.base_dir}")

    print(f"Found folds: { {f: len(vs) for f, vs in runs.items()} }")

    fold_test_ensembles = []
    fold_val_ensembles  = []

    for fold, versions in runs.items():
        test_pkls, val_pkls = [], []
        for v in versions:
            d = os.path.join(args.base_dir, f"{args.prefix}_fold{fold}_v{v}")
            test_pkls.append(load_pickle(os.path.join(d, "test_results.pkl")))
            val_pkls.append(load_pickle(os.path.join(d, "validation_results.pkl")))

        fold_test = average_pickles(test_pkls)
        fold_val  = average_pickles(val_pkls)
        fold_test_ensembles.append(fold_test)
        fold_val_ensembles.append(fold_val)
        print(f"  fold{fold} ({len(versions)} seeds)  val AUCPR={compute_aucpr(fold_val):.4f}")

    test_global = average_pickles(fold_test_ensembles)
    test_aucpr  = compute_aucpr(test_global)
    print(f"\nTest AUCPR (all folds × seeds): {test_aucpr:.4f}")

    best_thr, val_f1 = find_best_threshold(fold_val_ensembles)
    print(f"OOF val — best threshold: {best_thr:.4f}  val F1={val_f1:.4f}")

    res = eval_at_threshold(test_global, best_thr)
    print(f"\nTest F1        = {res['f1']:.4f}")
    print(f"Test Precision = {res['precision']:.4f}")
    print(f"Test Recall    = {res['recall']:.4f}")


if __name__ == "__main__":
    main()
