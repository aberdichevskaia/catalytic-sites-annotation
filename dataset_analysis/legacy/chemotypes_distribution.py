#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute chemotype statistics (counts + weights) and plot stacked bars across CV splits.

Inputs:
  --dataset_csv   path to dataset.csv with columns: Sequence_ID, Set_Type, W_Structure (or custom weight col)
  --label_txts    one or more split*.txt label files in format:
                    >SEQID
                    A 1 M 0
                    A 2 G 0
                    ...

Outputs (in --out_dir):
  - chemotype_overall.csv              (like EC numbers: counts + % + weight + %)
  - chemotype_overall_table.tex        (LaTeX table matching your EC table style)
  - chemotype_by_split_long.csv        (split × chemotype, counts + weight)
  - stackedbar_chemotype_by_split_weight.pdf
  - stackedbar_chemotype_by_split_weight_norm.pdf
  - (optionally png versions if --also_png)

Notes:
- Chemotype is computed from TRUE catalytic residues in label files (label==1), using a fixed priority rule.
- Proteins missing from label_txts are dropped (with a warning).
"""

import argparse
import os
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


# ----------------- Chemotype definition (0..7) -----------------

CHEMOTYPE_LABELS = {
    0: "Class 0 (ILMVWF)",
    1: "Class 1 (AGP)",
    2: "Class 2 (QN)",
    3: "Class 3 (KR)",
    4: "Class 4 (S)",
    5: "Class 5 (T)",
    6: "Class 6 (DE)",
    7: "Class 7 (other/none)",
}


def chemotype_from_residue_set(residues: Set[str]) -> int:
    """Return chemotype class 0..7 from a set of AA types at catalytic positions."""
    if any(r in residues for r in "ILMVWF"):
        return 0
    if any(r in residues for r in "AGP"):
        return 1
    if any(r in residues for r in "QN"):
        return 2
    if any(r in residues for r in "KR"):
        return 3
    if "S" in residues:
        return 4
    if "T" in residues:
        return 5
    if any(r in residues for r in "DE"):
        return 6
    return 7


# ----------------- Parsing label txts -----------------

def parse_label_txt(path: str) -> Dict[str, Tuple[Set[str], int]]:
    """
    Parse a split*.txt file:
      >SEQID
      A 1 M 0
      ...
    Returns: id -> (set_of_catalytic_residue_types, n_pos)
    """
    out: Dict[str, Tuple[Set[str], int]] = {}
    cur_id = None
    cat_set: Set[str] = set()
    n_pos = 0

    def flush():
        nonlocal cur_id, cat_set, n_pos
        if cur_id is not None:
            out[cur_id] = (set(cat_set), int(n_pos))

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_id = line[1:].strip()
                cat_set = set()
                n_pos = 0
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            aa = parts[2].strip()
            try:
                y = int(parts[3])
            except ValueError:
                continue

            if y == 1:
                cat_set.add(aa)
                n_pos += 1

    flush()
    return out


def build_catalytic_maps(label_txts: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Merge multiple label files.
    Returns:
      id -> set(residue types at catalytic positions)
      id -> n_pos (# catalytic residues)
    If an id appears multiple times, we union sets and sum n_pos (shouldn't happen in clean CV).
    """
    id_to_set: Dict[str, Set[str]] = {}
    id_to_npos: Dict[str, int] = {}

    for p in label_txts:
        part = parse_label_txt(p)
        for k, (s, npos) in part.items():
            if k not in id_to_set:
                id_to_set[k] = set(s)
                id_to_npos[k] = int(npos)
            else:
                id_to_set[k].update(s)
                id_to_npos[k] += int(npos)

    return id_to_set, id_to_npos


# ----------------- LaTeX table helper -----------------

def write_latex_chemotype_table(df_overall: pd.DataFrame, out_path: str):
    """
    Write a LaTeX table in the same spirit as your EC numbers table:
      Chemotype | # chains | Chains (% of dataset) | Total weight | Weight (% of total W_Structure)
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\adjustbox{max width=\columnwidth}{%")
    lines.append(r"\begin{tabular}{|l|r|r|r|r|}")
    lines.append(r"\hline")
    lines.append(r"Chemotype & \# chains & \makecell[l]{Chains\\(\% of dataset)} & Total weight & \makecell[l]{Weight\\(\% of total $W_\mathrm{Structure}$)} \\")
    lines.append(r"\hline")

    for _, r in df_overall.iterrows():
        name = r["Chemotype"]
        n = int(r["n_chains"])
        p_ch = float(r["chains_pct"])
        w = float(r["weight_sum"])
        p_w = float(r["weight_pct"])
        lines.append(f"{name} & {n} & {p_ch:.1f} & {w:.2f} & {p_w:.1f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Distribution of catalytic chemotypes (overall).}")
    lines.append(r"\label{tab:chemotype_overall}")
    lines.append(r"\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------- Plotting -----------------

def stacked_bar_by_split(df_long: pd.DataFrame,
                         split_order: List[str],
                         class_order: List[int],
                         value_col: str,
                         normalize: bool,
                         out_path_pdf: str,
                         also_png: bool = False):
    """
    df_long columns: Set_Type, chemotype, value_col (weight_sum or n_chains)
    """
    cmap = matplotlib.colormaps.get("Paired")  # try: "Set2", "Dark2", "Paired"
    colors = [cmap(i) for i in range(8)]
    colors[7] = (0.6, 0.6, 0.6, 1.0)  # make "other/none" gray

    # Pivot to wide: rows=splits, cols=chemotypes
    piv = (
        df_long.pivot_table(index="Set_Type", columns="chemotype", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(split_order)
    )
    # Ensure all classes are present
    for c in class_order:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[class_order]

    mat = piv.to_numpy(dtype=float)

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    x = np.arange(len(split_order))
    bottom = np.zeros(len(split_order), dtype=float)

    for j, c in enumerate(class_order):
        vals = mat[:, j]
        ax.bar(x, vals, bottom=bottom, label=CHEMOTYPE_LABELS.get(c, str(c)), color=colors[j])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=0)

    if normalize:
        ax.set_ylabel("Fraction of split" if value_col == "weight_sum" else "Fraction of split (chains)")
        ax.set_ylim(0.0, 1.0)
        title = "Chemotype composition across splits (normalized)"
    else:
        ax.set_ylabel("Total structural weight" if value_col == "weight_sum" else "# chains")
        title = "Chemotype composition across splits"

    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    fig.savefig(out_path_pdf, dpi=300, format="pdf")
    if also_png:
        fig.savefig(out_path_pdf.replace(".pdf", ".png"), dpi=300, format="png")
    plt.close(fig)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, required=True)
    ap.add_argument("--label_txts", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--weight_col", type=str, default="W_Structure")
    ap.add_argument("--also_png", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset metadata
    df = pd.read_csv(args.dataset_csv)
    need_cols = {"Sequence_ID", "Set_Type", args.weight_col}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"dataset_csv missing columns: {sorted(missing)}")

    df = df.drop_duplicates(subset=["Sequence_ID"]).copy()
    df[args.weight_col] = df[args.weight_col].astype(float)

    # Parse catalytic residue sets from label files
    id_to_set, id_to_npos = build_catalytic_maps(args.label_txts)

    # Attach chemotype and npos
    chemos = []
    npos_list = []
    missing_labels = 0

    for sid in df["Sequence_ID"].astype(str).tolist():
        s = id_to_set.get(sid, None)
        if s is None:
            missing_labels += 1
            chemos.append(np.nan)
            npos_list.append(np.nan)
        else:
            chemos.append(int(chemotype_from_residue_set(s)))
            npos_list.append(int(id_to_npos.get(sid, 0)))

    if missing_labels > 0:
        print(f"[WARN] Missing label entries for {missing_labels} / {len(df)} proteins; they will be dropped.")

    df["chemotype"] = chemos
    df["n_pos"] = npos_list
    df = df.dropna(subset=["chemotype"]).copy()
    df["chemotype"] = df["chemotype"].astype(int)

    total_n = len(df)
    total_w = float(df[args.weight_col].sum())

    # Overall chemotype table (like EC numbers)
    overall = (
        df.groupby("chemotype")
        .agg(n_chains=("Sequence_ID", "count"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
    )
    overall["chains_pct"] = 100.0 * overall["n_chains"] / max(total_n, 1)
    overall["weight_pct"] = 100.0 * overall["weight_sum"] / max(total_w, 1e-12)
    overall["Chemotype"] = overall["chemotype"].map(CHEMOTYPE_LABELS)

    # Sort by chemotype index 0..7
    overall = overall.sort_values("chemotype").reset_index(drop=True)

    out_csv = os.path.join(args.out_dir, "chemotype_overall.csv")
    overall[["chemotype", "Chemotype", "n_chains", "chains_pct", "weight_sum", "weight_pct"]].to_csv(out_csv, index=False)

    out_tex = os.path.join(args.out_dir, "chemotype_overall_table.tex")
    write_latex_chemotype_table(overall, out_tex)

    # By split (long)
    by_split = (
        df.groupby(["Set_Type", "chemotype"])
        .agg(n_chains=("Sequence_ID", "count"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
        .sort_values(["Set_Type", "chemotype"])
        .reset_index(drop=True)
    )
    by_split.to_csv(os.path.join(args.out_dir, "chemotype_by_split_long.csv"), index=False)

    # Plot stacked bars by split (weight)
    split_order = sorted(df["Set_Type"].unique().tolist(), key=lambda x: (str(x)[:5], str(x)))
    class_order = list(range(0, 8))

    stacked_bar_by_split(
        df_long=by_split,
        split_order=split_order,
        class_order=class_order,
        value_col="weight_sum",
        normalize=False,
        out_path_pdf=os.path.join(args.out_dir, "stackedbar_chemotype_by_split_weight.pdf"),
        also_png=args.also_png,
    )
    stacked_bar_by_split(
        df_long=by_split,
        split_order=split_order,
        class_order=class_order,
        value_col="weight_sum",
        normalize=True,
        out_path_pdf=os.path.join(args.out_dir, "stackedbar_chemotype_by_split_weight_norm.pdf"),
        also_png=args.also_png,
    )

    print("[OK] Wrote:")
    print(" - chemotype_overall.csv")
    print(" - chemotype_overall_table.tex")
    print(" - chemotype_by_split_long.csv")
    print(" - stackedbar_chemotype_by_split_weight.pdf")
    print(" - stackedbar_chemotype_by_split_weight_norm.pdf")


if __name__ == "__main__":
    main()
