#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# --- Style (as requested) ---
matplotlib.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.titlesize": 7,

    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 0.9,

    "svg.fonttype": "path",  # "none"
})

# figures sizes (for the paper)
W_FULL = 6.27
GUTTER = 0.15
W_HALF = (W_FULL - GUTTER) / 2

H_SMALL = 1.9


def _save_both(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), dpi=1000, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_hist_isoforms_per_base(base_meta: pd.DataFrame, out_path: Path):
    """
    Histogram of number of isoforms per base_id.
    - ignore 0/1 (keep n_isoforms >= 2)
    - cap at 10 (10 means 10+)
    """
    if "n_isoforms" not in base_meta.columns:
        raise ValueError("per_base_meta.csv must contain column 'n_isoforms'.")

    x = pd.to_numeric(base_meta["n_isoforms"], errors="coerce").dropna().astype(int).values
    x = x[x >= 2]
    if x.size == 0:
        raise ValueError("No base IDs with n_isoforms >= 2 found.")

    x_cap = np.minimum(x, 10)

    # bins for integers 2..10 (where 10 represents 10+ after capping)
    bin_edges = np.arange(1.5, 10.5 + 1e-9, 1.0)  # 1.5..10.5 step 1

    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()

    base = plt.colormaps["YlOrBr"](0.65)
    tail = "#999999"
    

    n, bins, patches = ax.hist(x_cap, bins=bin_edges)
    print(len(bins))
    colors = [base] * (len(bins) - 2) + [tail]   # last bin (>=cap) in gray
    for p, c in zip(patches, colors):
        p.set_facecolor(c)
        p.set_edgecolor("black")
        p.set_linewidth(0.8)

    ax.set_xlabel("Number of isoforms per base protein")
    ax.set_ylabel("Count")

    # x ticks: 2..9, 10+
    ticks = list(range(2, 11))
    ax.set_xticks(ticks)
    ticklabels = [str(t) for t in range(2, 10)] + ["10+"]
    ax.set_xticklabels(ticklabels)

    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Isoforms per base protein (capped at 10+)")

    _save_both(fig, out_path)


def plot_bar_loss_categories(per_iso: pd.DataFrame, out_path: Path):
    """
    Bar plot with 3 columns (fractions over all compared isoforms):
      - no_loss
      - loss_missing_only
      - loss_aligned_present
    """
    if "category" not in per_iso.columns:
        raise ValueError("per_isoform_loss_categories.csv must contain column 'category'.")

    cats = ["no_loss", "loss_missing_only", "loss_aligned_present"]
    labels = [
        "No loss",
        "Lost (not aligned)",
        "Lost (aligned, p<τ)",
    ]

    total = len(per_iso)
    if total == 0:
        raise ValueError("per_isoform_loss_categories.csv is empty.")

    vc = per_iso["category"].value_counts(dropna=False).to_dict()
    vals = [vc.get(c, 0) / total for c in cats]

    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()
    ax.bar(range(len(vals)), vals, color=plt.colormaps["YlOrBr"](0.55), edgecolor='black')

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction of isoforms")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Isoform-level loss categories (τ=0.35)")

    _save_both(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_csv", required=True,
                    help="CSV produced by analyse_npz.py (one row per isoform).")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.analysis_csv)

    # Derive per-base metadata: count all isoforms (reference included) per base_id.
    base_meta = (
        df.groupby("base_id")
        .size()
        .reset_index(name="n_isoforms")
    )

    # Loss categories: exclude the reference isoform row (not a "compared" isoform).
    per_iso = df[df["category"] != "reference"].copy()

    plot_hist_isoforms_per_base(base_meta, out_dir / "hist_isoforms_per_base_cap10")
    plot_bar_loss_categories(per_iso, out_dir / "bar_isoform_loss_categories_tau035")

    print("[OK] Wrote plots to:", out_dir.resolve())


if __name__ == "__main__":
    main()
