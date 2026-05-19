#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare AUCPR per RSA bin across multiple models.

Reads metrics_by_accessibility_bin.csv from each model's stratification output dir
and produces a combined table + comparative barplot with CI error bars.

Usage:
  python compare_aucpr_by_rsa_bin.py \
      --model_dirs <dir1> <dir2> ... \
      --model_names <name1> <name2> ... \
      --out_dir <out>

Or auto-discover all subdirs containing metrics_by_accessibility_bin.csv:
  python compare_aucpr_by_rsa_bin.py \
      --search_dir <stratification_results_dir> \
      --out_dir <out>
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

RSA_BIN_ORDER = [
    "buried(<=0.05)",
    "partly_buried(0.05-0.2)",
    "intermediate(0.2-0.5)",
    "exposed(>0.5)",
]
RSA_BIN_LABELS = ["buried\n≤0.05", "partly buried\n0.05–0.2", "intermediate\n0.2–0.5", "exposed\n>0.5"]


def load_model_metrics(model_dir, model_name):
    path = os.path.join(model_dir, "metrics_by_accessibility_bin.csv")
    if not os.path.exists(path):
        print(f"[WARN] not found: {path}")
        return None
    df = pd.read_csv(path)
    df["model"] = model_name
    return df


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dirs", nargs="+", help="Explicit list of model output dirs")
    group.add_argument("--search_dir", help="Auto-discover subdirs with metrics_by_accessibility_bin.csv")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="Display names for models (only with --model_dirs)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.search_dir:
        entries = sorted([
            e for e in os.scandir(args.search_dir)
            if e.is_dir() and os.path.exists(os.path.join(e.path, "metrics_by_accessibility_bin.csv"))
        ], key=lambda e: e.name)
        model_dirs  = [e.path for e in entries]
        model_names = [e.name for e in entries]
    else:
        model_dirs  = args.model_dirs
        model_names = args.model_names if args.model_names else [os.path.basename(d) for d in model_dirs]

    if not model_dirs:
        print("[ERROR] no model dirs found")
        return

    frames = []
    for d, name in zip(model_dirs, model_names):
        df = load_model_metrics(d, name)
        if df is not None:
            frames.append(df)
            print(f"[OK] loaded {name}")

    if not frames:
        print("[ERROR] no data loaded")
        return

    combined = pd.concat(frames, ignore_index=True)

    # --- Wide table: models × bins ---
    pivot = combined.pivot(index="model", columns="rsa_bin", values="AUCPR")
    pivot = pivot.reindex(columns=RSA_BIN_ORDER)
    pivot = pivot.reindex(model_names)

    out_csv = os.path.join(args.out_dir, "aucpr_by_rsa_bin_comparison.csv")
    combined.to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv}")

    wide_csv = os.path.join(args.out_dir, "aucpr_by_rsa_bin_wide.csv")
    pivot.to_csv(wide_csv)
    print(f"[OK] saved {wide_csv}")

    print("\nAUCPR per bin:\n", pivot.to_string())

    # --- Barplot: grouped by RSA bin, one bar per model ---
    n_bins   = len(RSA_BIN_ORDER)
    n_models = len(model_names)
    width    = 0.8 / n_models
    x        = np.arange(n_bins)

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        sub = combined[combined["model"] == name].set_index("rsa_bin")
        vals  = [sub.loc[b, "AUCPR"]      if b in sub.index else np.nan for b in RSA_BIN_ORDER]
        lo    = [sub.loc[b, "AUCPR_ci_lo"] if b in sub.index else np.nan for b in RSA_BIN_ORDER]
        hi    = [sub.loc[b, "AUCPR_ci_hi"] if b in sub.index else np.nan for b in RSA_BIN_ORDER]

        offsets = x + (i - n_models / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width=width * 0.9, label=name, color=color, edgecolor="black", linewidth=0.5)

        vals_a = np.array(vals, dtype=float)
        lo_a   = np.array(lo,   dtype=float)
        hi_a   = np.array(hi,   dtype=float)
        yerr   = np.vstack([vals_a - lo_a, hi_a - vals_a])
        ax.errorbar(offsets, vals_a, yerr=yerr, fmt="none", ecolor="black",
                    capsize=3, elinewidth=0.8, capthick=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(RSA_BIN_LABELS, fontsize=10)
    ax.set_ylabel("AUCPR", fontsize=11)
    ax.set_title("AUCPR by RSA bin — model comparison", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "aucpr_by_rsa_bin_comparison.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
