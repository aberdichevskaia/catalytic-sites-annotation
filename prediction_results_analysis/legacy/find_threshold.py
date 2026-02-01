#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# ---------- IO ----------

def find_existing(path_dir, candidates):
    """Return first existing path in candidates, joined with path_dir."""
    for name in candidates:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found under {path_dir}")

def load_results(path):
    """Load merged pickles: flatten labels/preds; repeat per-chain weight."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # lists of arrays/lists → flatten
    labels_per_chain = [np.asarray(x, dtype=np.int32) for x in data["labels"]]
    preds_per_chain  = [np.asarray(x, dtype=np.float32) for x in data["predictions"]]
    weights_per_chain = [float(w) for w in data["weights"]]

    # sanity
    for y, p in zip(labels_per_chain, preds_per_chain):
        if len(y) != len(p):
            raise ValueError(f"Length mismatch in {path}: y={len(y)} p={len(p)}")

    y_flat = np.concatenate(labels_per_chain).astype(np.float32)
    p_flat = np.concatenate(preds_per_chain).astype(np.float32)
    w_flat = np.concatenate([
        np.full(len(y), w, dtype=np.float32)
        for y, w in zip(labels_per_chain, weights_per_chain)
    ])

    # clean NaN/Inf to be robust
    bad = np.isnan(p_flat) | np.isnan(y_flat) | np.isinf(p_flat)
    if bad.any():
        p_med = np.nanmedian(p_flat[~bad]) if (~bad).any() else 0.0
        p_flat[bad] = p_med
        y_flat[bad] = 0.0
        w_flat[bad] = 0.0

    return y_flat.astype(np.int32), p_flat.astype(np.float32), w_flat.astype(np.float32)

# ---------- Metrics on a shared threshold grid ----------

def prec_recall_vs_threshold(y_true, y_probs, y_weights, thresholds):
    """Weighted precision/recall for each t in thresholds."""
    precisions, recalls = [], []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(np.int32)
        # sklearn metrics support weights directly
        precisions.append(
            precision_score(y_true, y_pred, sample_weight=y_weights, zero_division=0)
        )
        recalls.append(
            recall_score(y_true, y_pred, sample_weight=y_weights)
        )
    return np.asarray(precisions), np.asarray(recalls)

# ---------- Main ----------

if __name__ == "__main__":
    input_dir = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/cross_validation/merged"

    # use merged filenames, fallback to old ones if present
    train_path = find_existing(input_dir, ["train_dedup.pkl", "train_results_dedup.pkl", "train.pkl", "train_results.pkl"])
    validation_path  = find_existing(input_dir, ["validation.pkl", "validation_results.pkl"])

    print(f"[use] train: {train_path}")
    print(f"[use] validation : {validation_path}")

    y_train, p_train, w_train = load_results(train_path)
    y_validation,  p_validation,  w_validation  = load_results(validation_path)

    thresholds = np.linspace(0.0, 1.0, 200)

    prec_train, rec_train = prec_recall_vs_threshold(y_train, p_train, w_train, thresholds)
    prec_validation,  rec_validation  = prec_recall_vs_threshold(y_validation,  p_validation,  w_validation,  thresholds)

    # target constraints
    tgt_prec_validation = 0.50
    tgt_rec_train = 0.80

    mask = (prec_validation >= tgt_prec_validation) & (rec_train >= tgt_rec_train)
    valid_thresholds = thresholds[mask]

    thr_min = thr_max = None
    if valid_thresholds.size > 0:
        thr_min, thr_max = float(valid_thresholds.min()), float(valid_thresholds.max())
        thr_mid = 0.5 * (thr_min + thr_max)
        # report metrics at the mid point (just for reference)
        y_pred_validation_mid = (p_validation >= thr_mid).astype(int)
        y_pred_train_mid = (p_train >= thr_mid).astype(int)
        prec_validation_mid  = precision_score(y_validation,  y_pred_validation_mid,  sample_weight=w_validation,  zero_division=0)
        rec_validation_mid   = recall_score(  y_validation,  y_pred_validation_mid,  sample_weight=w_validation)
        prec_train_mid = precision_score(y_train, y_pred_train_mid, sample_weight=w_train, zero_division=0)
        rec_train_mid  = recall_score(  y_train, y_pred_train_mid, sample_weight=w_train)

        print(f"Valid threshold range: [{thr_min:.3f}, {thr_max:.3f}]")
        print(f"Suggested (mid): {thr_mid:.3f}")
        print(f"  train: Prec={prec_train_mid:.3f}  Rec={rec_train_mid:.3f}")
        print(f"  validation : Prec={prec_validation_mid:.3f}   Rec={rec_validation_mid:.3f}")
    else:
        print("No threshold satisfies: Precision_validation ≥ 0.5 and Recall_train ≥ 0.8")
        thr_mid = None

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, prec_train, label="Precision (train)")
    plt.plot(thresholds, rec_train, label="Recall (train)")
    plt.plot(thresholds, prec_validation,  label="Precision (validation)", linestyle="--")
    plt.plot(thresholds, rec_validation,   label="Recall (validation)",  linestyle="--")

    plt.axhline(0.5, color="gray", linestyle=":", label="Precision=0.5 (target)")
    plt.axhline(0.8, color="orange", linestyle=":", label="Recall=0.8 (target train)")

    if thr_min is not None and thr_max is not None:
        plt.axvspan(thr_min, thr_max, color="green", alpha=0.2,
                    label=f"Valid range [{thr_min:.2f}, {thr_max:.2f}]")
        plt.axvline(thr_mid, linestyle=":", label=f"Chosen ~{thr_mid:.2f}")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs Threshold (weighted)")
    plt.legend()
    plt.grid(True)
    out_pdf = os.path.join(input_dir, "thresholds_curve_range_fixed.pdf")
    plt.savefig(out_pdf, dpi=300, format="pdf")
    print(f"[OK] saved plot -> {out_pdf}")
