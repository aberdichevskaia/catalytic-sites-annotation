#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Violin plot of *log(weight)* grouped by per-sequence F@1 (0 or 1).

- X axis: F@1 category (0 or 1)
- Y axis: log_base(weight) with numeric stability (clip by epsilon)
"""

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    preds = np.array(data["predictions"], dtype=object)
    labels = np.array(data["labels"][: len(preds)], dtype=object)
    weights = np.array(data["weights"][: len(preds)], dtype=float)
    return preds, labels, weights


def per_sequence_f_at_1(pred_scores, true_labels) -> int:
    """Binary F@1: 1 if top-1 is a true positive, else 0 (sequences with no positives -> 0)."""
    true = np.asarray(true_labels, dtype=int)
    if true.sum() == 0:
        return 0
    scores = np.asarray(pred_scores, dtype=float)
    bad = ~np.isfinite(scores)
    if bad.any():
        scores = scores.copy()
        fallback = 0.0 if (~bad).sum() == 0 else np.nanmin(scores[~bad])
        scores[bad] = fallback
    top1_idx = int(np.argmax(scores))
    return int(true[top1_idx] == 1)


def safe_log_transform(w: np.ndarray, base: float, eps: float) -> np.ndarray:
    """
    Apply log_base(max(w, eps)) elementwise.
    base=e for natural log, base=10 or 2 as needed.
    """
    w_clip = np.clip(np.asarray(w, dtype=float), a_min=eps, a_max=None)
    if base == np.e:
        return np.log(w_clip)
    else:
        return np.log(w_clip) / np.log(base)


def build_groups(preds, labels, weights, base: float, eps: float):
    """Group *log(weights)* by F@1=0 and F@1=1."""
    w0, w1 = [], []
    for p, y, w in zip(preds, labels, weights):
        f1 = per_sequence_f_at_1(p, y)
        lw = float(safe_log_transform(np.array([w]), base, eps)[0])
        if f1 == 1:
            w1.append(lw)
        else:
            w0.append(lw)
    return w0, w1


def plot_violin(w0, w1, title: str, out_path: str, base: float):
    """Make a violin plot for log(weights)."""
    data = [w0, w1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.violinplot(
        dataset=data,
        positions=[0, 1],
        showmeans=True,
        showextrema=True,
        showmedians=True,
        widths=0.8,
    )

    # Jittered scatter to visualize counts
    rng = np.random.default_rng(42)
    if len(w0) > 0:
        ax.scatter(0 + (rng.random(len(w0)) - 0.5) * 0.15, w0, s=6, alpha=0.35, edgecolors="none")
    if len(w1) > 0:
        ax.scatter(1 + (rng.random(len(w1)) - 0.5) * 0.15, w1, s=6, alpha=0.35, edgecolors="none")

    ax.set_xticks([0, 1], ["F@1 = 0", "F@1 = 1"])
    base_label = "e" if abs(base - np.e) < 1e-12 else (str(int(base)) if base in (2, 10) else f"{base:g}")
    ax.set_xlabel("F@1 per sequence", fontsize=12)
    ax.set_ylabel(f"log_{base_label}(weight)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/iscb/wolfson/annab4/DB/all_proteins/structural_homology_baseline/results_adaptive_cutoff_den2_cv1/train_results.pkl",
        help="Path to *_results.pkl with predictions/labels/weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="violin_f1_vs_logweight.pdf",
        help="Output image path (.pdf/.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="log(weight) by per-sequence F@1 (k=1)",
        help="Plot title",
    )
    parser.add_argument(
        "--log_base",
        type=float,
        default=float(np.e),
        help="Logarithm base: e (2.718...), 10, 2, etc. Default: e",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Floor for weights before log to avoid -inf. Default: 1e-12",
    )
    args = parser.parse_args()

    preds, labels, weights = load_results(args.input)
    w0, w1 = build_groups(preds, labels, weights, base=args.log_base, eps=args.eps)

    # Console stats in log-space
    n = len(weights)
    print(f"[info] sequences: {n}")
    print(f"[info] F@1=0: {len(w0)} | F@1=1: {len(w1)}")
    if len(w0) > 0:
        print(f"[info] log-weight stats (F@1=0): mean={np.mean(w0):.4f} median={np.median(w0):.4f}")
    if len(w1) > 0:
        print(f"[info] log-weight stats (F@1=1): mean={np.mean(w1):.4f} median={np.median(w1):.4f}")

    plot_violin(w0, w1, args.title, args.output, base=args.log_base)
    print(f"[ok] saved: {args.output}")


if __name__ == "__main__":
    main()
