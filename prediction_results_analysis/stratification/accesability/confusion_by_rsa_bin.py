#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion matrix per RSA bin, using the F1-optimal global threshold.

Reads residue_table.csv produced by stratify.py (already has rsa_bin column).

Usage:
  python confusion_by_rsa_bin.py \
      --residue_table <path>/residue_table.csv \
      --out_dir <path>/confusion_rsa
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

RSA_BIN_ORDER = [
    "buried(<=0.05)",
    "partly_buried(0.05-0.2)",
    "intermediate(0.2-0.5)",
    "exposed(>0.5)",
]


def find_f1_optimal_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # precision_recall_curve appends a final point (p=1, r=0) with no threshold
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1)
    return float(thresholds[best_idx]), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def confusion_metrics(y_true, y_pred_bin):
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0)
                 else float("nan"))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                precision=precision, recall=recall, f1=f1, specificity=specificity)


def plot_confusion_matrix(tp, fp, tn, fn, title, out_path):
    cm = np.array([[tn, fp], [fn, tp]])
    total = tp + fp + tn + fn

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"])
    ax.set_yticklabels(["True: 0", "True: 1"])
    ax.set_title(title, fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residue_table", required=True,
                        help="Path to residue_table.csv from stratify.py")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.residue_table)
    eval_df = df.dropna(subset=["y_true", "y_pred"]).copy()
    eval_df["y_true"] = eval_df["y_true"].astype(int)

    # --- Global F1-optimal threshold ---
    threshold, f1_global, prec_global, rec_global = find_f1_optimal_threshold(
        eval_df["y_true"].to_numpy(),
        eval_df["y_pred"].to_numpy(),
    )
    print(f"[INFO] F1-optimal threshold (global): {threshold:.4f}  "
          f"F1={f1_global:.4f}  precision={prec_global:.4f}  recall={rec_global:.4f}")

    eval_df["y_pred_bin"] = (eval_df["y_pred"] >= threshold).astype(int)

    # --- Per-bin confusion matrices ---
    rows = []
    for rsa_bin in RSA_BIN_ORDER:
        sub = eval_df[eval_df["rsa_bin"] == rsa_bin]
        if len(sub) == 0:
            print(f"[WARN] no data for bin '{rsa_bin}'")
            continue

        m = confusion_metrics(sub["y_true"].to_numpy(), sub["y_pred_bin"].to_numpy())
        row = {"rsa_bin": rsa_bin, "n_residues": len(sub),
               "n_positive": int(sub["y_true"].sum()),
               "threshold": threshold, **m}
        rows.append(row)
        print(f"  {rsa_bin}: TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}  "
              f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  F1={m['f1']:.3f}")

        safe_name = rsa_bin.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")
        plot_confusion_matrix(
            m["TP"], m["FP"], m["TN"], m["FN"],
            title=f"{rsa_bin}\nthresh={threshold:.3f}  F1={m['f1']:.3f}",
            out_path=os.path.join(args.out_dir, f"confusion_{safe_name}.png"),
        )

    out_csv = os.path.join(args.out_dir, "confusion_by_rsa_bin.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] saved {out_csv}")

    # --- Summary bar chart: F1 per bin ---
    metrics_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, col, label in zip(axes, ["precision", "recall", "f1"], ["Precision", "Recall", "F1"]):
        vals = [metrics_df.loc[metrics_df["rsa_bin"] == b, col].values[0]
                if b in metrics_df["rsa_bin"].values else float("nan")
                for b in RSA_BIN_ORDER]
        short_labels = ["buried\n≤0.05", "partly\n0.05–0.2", "interm.\n0.2–0.5", "exposed\n>0.5"]
        ax.bar(range(len(RSA_BIN_ORDER)), vals, edgecolor="black")
        ax.set_xticks(range(len(RSA_BIN_ORDER)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Metrics by RSA bin  (threshold={threshold:.3f}, global F1={f1_global:.3f})",
                 fontsize=10)
    fig.tight_layout()
    summary_png = os.path.join(args.out_dir, "metrics_by_rsa_bin_at_f1_threshold.png")
    fig.savefig(summary_png, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {summary_png}")


if __name__ == "__main__":
    main()
