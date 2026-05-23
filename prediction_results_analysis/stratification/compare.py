#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare.py — Compare metrics across multiple models using pre-computed CSVs.

Loads metrics_by_{tag}.csv files written by stratify.py (no bootstrap recomputation).
Only PR curve plotting still reads residue_table.csv.

Analyses produced:
  1. PDB vs AF barplot
  2. Amino acid barplot + full AA heatmap
  3. EC top-level class barplot
  4. Chemotype barplot
  5. PR curves per RSA bin (one sub-panel per bin) + RSA bin barplot

Usage:
  python compare.py \\
    --model_dirs /path/model1 /path/model2 ... \\
    --model_names "Model A" "Model B" ... \\
    --out_dir /path/to/comparison
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stratify_utils import normalize_id, W_FULL, H_WIDE
from stratify import RSA_LABELS, AA_ORDER, CHEMOTYPE_NAMES

RSA_BIN_SHORT = [
    "buried\n≤0.05", "partly buried\n0.05–0.2",
    "intermediate\n0.2–0.5", "exposed\n>0.5",
]

matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 12, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7, "axes.titlesize": 11,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "svg.fonttype": "path",
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── data loading ──────────────────────────────────────────────────────────────

def load_residue_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ("y_true", "y_pred", "chain_weight", "rsa"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_model_metrics(model_dirs: List[str], model_names: List[str], tag: str) -> Dict[str, pd.DataFrame]:
    """Load metrics_by_{tag}.csv from each model dir."""
    out: Dict[str, pd.DataFrame] = {}
    for d, name in zip(model_dirs, model_names):
        path = os.path.join(d, f"metrics_by_{tag}.csv")
        if not os.path.exists(path):
            log.warning("%s: metrics_by_%s.csv not found in %s — skipping", name, tag, d)
            continue
        out[name] = pd.read_csv(path)
    return out


# ─── multi-model grouped barplot ───────────────────────────────────────────────

def multi_barplot(
    ax,
    groups:          List,
    model_names:     List[str],
    metrics_by_model: Dict[str, pd.DataFrame],
    group_col:       str,
    colors,
    xlabels:    Optional[List[str]] = None,
    ylabel:     str = "AUCPR",
    metric_col: str = "AUCPR",
    ci_lo_col:  str = "AUCPR_ci_lo",
    ci_hi_col:  str = "AUCPR_ci_hi",
    rotation:   int = 20,
    ha:         str = "right",
    legend_loc: str = "upper right",
) -> None:
    n     = len(model_names)
    width = 0.8 / n

    for i, (name, color) in enumerate(zip(model_names, colors)):
        mdf = metrics_by_model.get(name)
        if mdf is None or len(mdf) == 0:
            continue

        idx_map  = {str(row[group_col]): row for _, row in mdf.iterrows()}
        vals = np.array([idx_map[str(g)][metric_col]  if str(g) in idx_map else np.nan for g in groups], float)
        lo   = np.array([idx_map[str(g)][ci_lo_col]   if str(g) in idx_map and ci_lo_col in idx_map[str(g)] else np.nan for g in groups], float)
        hi   = np.array([idx_map[str(g)][ci_hi_col]   if str(g) in idx_map and ci_hi_col in idx_map[str(g)] else np.nan for g in groups], float)

        x = np.arange(len(groups)) + (i - n / 2 + 0.5) * width
        ax.bar(x, vals, width * 0.9, label=name, color=color, edgecolor="black", linewidth=0.5)

        bad = np.isnan(lo) | np.isnan(hi)
        lo2, hi2 = lo.copy(), hi.copy()
        lo2[bad], hi2[bad] = vals[bad], vals[bad]
        ax.errorbar(x, vals, yerr=np.vstack([vals - lo2, hi2 - vals]),
                    fmt="none", ecolor="black", capsize=2, elinewidth=0.7)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(xlabels if xlabels else [str(g) for g in groups],
                       rotation=rotation, ha=ha)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── comparison analyses ───────────────────────────────────────────────────────

def compare_by_group(
    model_dirs:  List[str],
    model_names: List[str],
    group_col:   str,
    groups:      Optional[List],
    out_dir:     str,
    tag:         str,
    title:       str,
    colors,
    xlabels:    Optional[List[str]] = None,
    height:     float = H_WIDE,
    rotation:   int = 20,
    ha:         str = "right",
    legend_loc: str = "upper right",
) -> None:
    metrics_by_model = load_model_metrics(model_dirs, model_names, tag)

    if not metrics_by_model:
        log.warning("no data for %s", tag)
        return

    all_rows = []
    for name, m in metrics_by_model.items():
        m2 = m.copy()
        m2.insert(0, "model", name)
        all_rows.append(m2)
    pd.concat(all_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, f"{tag}_comparison.csv"), index=False
    )

    actual_groups = groups
    if actual_groups is None:
        seen: set = set()
        for mdf in metrics_by_model.values():
            seen.update(mdf[group_col].astype(str).tolist())
        actual_groups = sorted(seen, key=str)

    xlbls = xlabels if xlabels else [str(g) for g in actual_groups]
    fig_w = max(W_FULL, len(actual_groups) * 1.2 * max(1, len(model_names) * 0.5))

    bar_kw = dict(rotation=rotation, ha=ha, legend_loc=legend_loc)

    for metric, ci_lo, ci_hi, ylabel, fname_suffix, ylim_override in (
        ("AUCPR",        "AUCPR_ci_lo",  "AUCPR_ci_hi",  "AUCPR",               "aucpr",      None),
        ("AUCPR_norm",   None,           None,           "AUCPR (normalised)",  "aucpr_norm", (-0.1, 1.0)),
        ("AUCROC",       "AUCROC_ci_lo", "AUCROC_ci_hi", "AUC-ROC",             "aucroc",     None),
        ("max_F1",       "F1_ci_lo",     "F1_ci_hi",     "max F1",              "f1",         None),
        ("Recall_at_F1", None,           None,           "Recall @ max F1",     "recall",     None),
    ):
        if not any(metric in mdf.columns for mdf in metrics_by_model.values()):
            continue
        _ci_lo = ci_lo or metric
        _ci_hi = ci_hi or metric
        fig, ax = plt.subplots(figsize=(fig_w, height))
        multi_barplot(ax, actual_groups, model_names, metrics_by_model, group_col, colors,
                      xlabels=xlbls, ylabel=ylabel,
                      metric_col=metric, ci_lo_col=_ci_lo, ci_hi_col=_ci_hi,
                      **bar_kw)
        if ylim_override:
            ax.set_ylim(*ylim_override)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="random")
        ax.set_title(f"{title} — {ylabel}", fontsize=11)
        ax.legend(fontsize=7, loc=legend_loc, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_{fname_suffix}_comparison.png"), dpi=200)
        plt.close(fig)

    log.info("saved %s_{aucpr,aucpr_norm,aucroc,f1,recall}_comparison.*", tag)


def _aa_heatmap(all_metrics: Dict[str, pd.DataFrame], metric_col: str,
                model_names: List[str], out_dir: str, fname: str, label: str) -> None:
    cols = {name: m.set_index("aa")[metric_col] for name, m in all_metrics.items()}
    mat  = pd.DataFrame(cols).reindex(AA_ORDER)
    mat.to_csv(os.path.join(out_dir, f"{fname}.csv"))

    fig, ax = plt.subplots(figsize=(max(4, len(model_names) * 1.6 + 1), 8))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_yticklabels(AA_ORDER, fontsize=9)
    for i in range(len(AA_ORDER)):
        for j in range(len(model_names)):
            v = mat.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.8 else "white")
    ax.set_title(f"{label} per amino acid", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=200)
    plt.close(fig)


def compare_aa_heatmap(
    model_dirs:  List[str],
    model_names: List[str],
    out_dir:     str,
) -> None:
    all_metrics = load_model_metrics(model_dirs, model_names, "aa")
    if not all_metrics:
        log.warning("no AA data for heatmap")
        return
    _aa_heatmap(all_metrics, "AUCPR",  model_names, out_dir, "aa_aucpr_heatmap",  "AUCPR")
    _aa_heatmap(all_metrics, "AUCROC", model_names, out_dir, "aa_aucroc_heatmap", "AUC-ROC")
    log.info("saved aa_aucpr_heatmap.* and aa_aucroc_heatmap.*")


def compare_pr_curves_by_rsa(
    model_dirs:  List[str],
    model_names: List[str],
    colors,
    out_dir:     str,
) -> None:
    # ── PR curves (one sub-panel per RSA bin) — load residue_table.csv here ───
    model_dfs = []
    for d, name in zip(model_dirs, model_names):
        rt = os.path.join(d, "residue_table.csv")
        if not os.path.exists(rt):
            log.warning("%s: residue_table.csv not found — PR curves will be partial", name)
            model_dfs.append(pd.DataFrame())
        else:
            model_dfs.append(load_residue_table(rt))

    fig, axes = plt.subplots(1, len(RSA_LABELS), figsize=(4.5 * len(RSA_LABELS), 4.5))

    for ax, rsa_bin, short_label in zip(axes, RSA_LABELS, RSA_BIN_SHORT):
        baseline_vals = []
        for df, name, color in zip(model_dfs, model_names, colors):
            if df.empty or "rsa_bin" not in df.columns:
                continue
            sub = df[(df["rsa_bin"].astype(str) == rsa_bin)
                     & df["y_true"].notna() & df["y_pred"].notna()].copy()
            sub["y_true"] = sub["y_true"].astype(int)
            if len(sub) == 0 or sub["y_true"].sum() == 0:
                continue
            y = sub["y_true"].to_numpy()
            p = sub["y_pred"].to_numpy()
            mask = ~(np.isnan(p) | np.isinf(p))
            prec, rec, _ = precision_recall_curve(y[mask], p[mask])
            a = float(auc(rec, prec))
            ax.plot(rec, prec, label=f"{name} ({a:.3f})", color=color, linewidth=1.4, alpha=0.9)
            baseline_vals.append(float(sub["y_true"].mean()))

        baseline = float(np.mean(baseline_vals)) if baseline_vals else 0.0
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.8,
                   label=f"baseline ({baseline:.3f})")
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title(short_label, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Precision-Recall curves per RSA bin", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curves_by_rsa_bin.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("saved pr_curves_by_rsa_bin.png")

    # ── barplot from pre-computed metrics ──────────────────────────────────────
    compare_by_group(
        model_dirs=model_dirs, model_names=model_names,
        group_col="rsa_bin", groups=RSA_LABELS,
        out_dir=out_dir, tag="rsa_bin",
        title="by RSA bin",
        colors=colors, xlabels=RSA_BIN_SHORT,
        height=4.5, rotation=0, ha="center", legend_loc="upper left",
    )
    log.info("saved rsa_bin_{aucpr,aucroc,aucpr_norm}_comparison.*")


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model_dirs",  nargs="+", required=True,
                    help="Dirs containing metrics_by_*.csv files (produced by stratify.py)")
    ap.add_argument("--model_names", nargs="+", default=None,
                    help="Display names (same order as --model_dirs; defaults to dir basenames)")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_names and len(args.model_names) != len(args.model_dirs):
        raise ValueError("--model_names and --model_dirs must have the same length")

    model_names = args.model_names or [os.path.basename(d.rstrip("/")) for d in args.model_dirs]
    colors      = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

    kw = dict(model_dirs=args.model_dirs, model_names=model_names,
              out_dir=args.out_dir, colors=colors)

    log.info("[1] PDB vs AF")
    compare_by_group(group_col="model_source", groups=["AF", "PDB"],
                     tag="model_source", title="PDB vs AlphaFold", **kw)

    log.info("[2] Amino acid")
    compare_by_group(group_col="aa", groups=AA_ORDER,
                     tag="aa", title="By amino acid", **kw)
    compare_aa_heatmap(args.model_dirs, model_names, args.out_dir)

    log.info("[3] EC top-level class")
    compare_by_group(group_col="ec_top", groups=None,
                     tag="ec_top", title="By EC top-level class", **kw)

    log.info("[4] Chemotype")
    chem_groups = [str(c) for c in range(8)]
    chem_labels = [CHEMOTYPE_NAMES[c] for c in range(8)]
    compare_by_group(group_col="chemotype", groups=chem_groups, xlabels=chem_labels,
                     tag="chemotype", title="By chemotype", **kw)

    log.info("[5] PR curves + metrics by RSA bin")
    compare_pr_curves_by_rsa(model_dirs=args.model_dirs, model_names=model_names,
                              colors=colors, out_dir=args.out_dir)

    log.info("all outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
