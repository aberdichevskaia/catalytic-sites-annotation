#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summaries and plots for catalytic-site dataset.

- Reads dataset.csv with metadata.
- Parses split*.txt with per-residue labels.
- Produces:
  - per_chain_stats.csv
  - table_overview.csv
  - table_ec_distribution.csv
  - several pdf plots (lengths, imbalance, weights).
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def parse_split_file(path):
    """
    Parse splitX.txt like:

        >A0A1L8G2K9_A
        A 1 M 0
        A 2 G 0
        ...

    Returns DataFrame with columns:
      Sequence_ID, Length, n_catalytic
    """
    records = []
    current_id = None
    length = 0
    n_pos = 0

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None:
                    records.append((current_id, length, n_pos))
                current_id = line[1:].strip()
                length = 0
                n_pos = 0
            else:
                parts = line.split()
                # Expect: chain_id, index, aa, label
                if len(parts) < 4:
                    continue
                try:
                    label = int(parts[3])
                except ValueError:
                    continue
                length += 1
                if label == 1:
                    n_pos += 1

        # Save last sequence
        if current_id is not None:
            records.append((current_id, length, n_pos))

    df = pd.DataFrame(records, columns=["Sequence_ID", "Length", "n_catalytic"])
    return df


def compute_overview_table(per_chain: pd.DataFrame, output_path: str) -> None:
    """Compute Table 1: overview by split + overall row."""
    grouped = (
        per_chain
        .groupby("Split", dropna=False)
        .agg(
            n_chains=("Sequence_ID", "nunique"),
            n_residues=("Length", "sum"),
            n_catalytic=("n_catalytic", "sum"),
            mean_length=("Length", "mean"),
            p25_length=("Length", lambda x: x.quantile(0.25)),
            p75_length=("Length", lambda x: x.quantile(0.75)),
            sum_w_structure=("W_Structure", "sum"),
            sum_w_sequence=("W_Sequence", "sum"),
        )
        .reset_index()
    )

    grouped["frac_catalytic"] = (
        grouped["n_catalytic"] / grouped["n_residues"].replace(0, np.nan)
    )

    total_row = {
        "Split": "all",
        "n_chains": per_chain["Sequence_ID"].nunique(),
        "n_residues": per_chain["Length"].sum(),
        "n_catalytic": per_chain["n_catalytic"].sum(),
        "mean_length": per_chain["Length"].mean(),
        "p25_length": per_chain["Length"].quantile(0.25),
        "p75_length": per_chain["Length"].quantile(0.75),
        "sum_w_structure": per_chain["W_Structure"].sum(),
        "sum_w_sequence": per_chain["W_Sequence"].sum(),
    }
    total_row["frac_catalytic"] = (
        total_row["n_catalytic"] / total_row["n_residues"]
        if total_row["n_residues"] > 0
        else np.nan
    )

    total_df = pd.DataFrame([total_row])
    summary = pd.concat([grouped, total_df], ignore_index=True)
    summary.to_csv(output_path, index=False)


def add_ec_class_column(per_chain: pd.DataFrame) -> pd.DataFrame:
    """Add high-level EC class (first digit) as EC_class column."""

    def top_ec(ec_value):
        if pd.isna(ec_value):
            return "NA"
        if not isinstance(ec_value, str):
            return "NA"

        s = ec_value.strip()
        if s == "":
            return "NA"

        # If multiple ECs separated by ';', take first
        s = s.split(";")[0].strip()
        if s == "" or s[0] in ("-", "?"):
            return "NA"

        # Take first number before dot
        first = s.split(".")[0]
        return first if first else "NA"

    per_chain = per_chain.copy()
    per_chain["EC_class"] = per_chain["EC_number"].apply(top_ec)
    return per_chain


def compute_ec_table(per_chain: pd.DataFrame, output_path: str) -> None:
    """Table 2: distribution by EC_class and split."""
    df = add_ec_class_column(per_chain)

    ec_summary = (
        df
        .groupby(["Split", "EC_class"], dropna=False)
        .agg(
            n_chains=("Sequence_ID", "nunique"),
            n_catalytic=("n_catalytic", "sum"),
            n_residues=("Length", "sum"),
        )
        .reset_index()
    )
    ec_summary["frac_catalytic"] = (
        ec_summary["n_catalytic"] /
        ec_summary["n_residues"].replace(0, np.nan)
    )

    ec_summary.to_csv(output_path, index=False)


def plot_length_histograms(per_chain: pd.DataFrame, out_dir: str) -> None:
    """Overall and per-split histograms of sequence lengths."""
    # Overall
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_chain["Length"], bins=50)
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Number of chains")
    ax.set_title("Sequence length distribution (all splits)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_length_overall.pdf"), dpi=300)
    plt.close(fig)

    # By split (overlaid)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split_name, df_split in per_chain.groupby("Split"):
        ax.hist(
            df_split["Length"],
            bins=50,
            alpha=0.4,
            label=split_name,
        )
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Number of chains")
    ax.set_title("Sequence length distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_length_by_split.pdf"), dpi=300)
    plt.close(fig)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_catalytic_per_chain_binned(per_chain: pd.DataFrame, out_dir: str, cap: int = 5) -> None:
    """
    Bar plot: number of catalytic residues per chain binned as
    [1, 2, ..., cap-1, >=cap]. Zeros are dropped.

    cap=5 -> bins [1,2,3,4,>=5]
    cap=8 -> bins [1,2,3,4,5,6,7,>=8]
    """
    # English comments as requested
    if cap < 2:
        raise ValueError("cap must be >= 2 (so that we have at least one exact bin and one >=cap bin).")

    x = per_chain["n_catalytic"].to_numpy()
    x = x[np.isfinite(x)].astype(int)
    x = x[x >= 1]  # drop zeros and negatives

    if x.size == 0:
        raise ValueError("No chains with n_catalytic >= 1 found.")

    # Exact bins: 1..cap-1
    exact_max = cap - 1
    exact_counts = np.array([(x == i).sum() for i in range(1, exact_max + 1)], dtype=int)

    # Tail: >= cap
    tail_count = int((x >= cap).sum())

    labels = [str(i) for i in range(1, exact_max + 1)] + [f"≥{cap}"]
    values = exact_counts.tolist() + [tail_count]

    fig_w = max(6.0, 0.6 * len(values))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(np.arange(len(values)), values)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of catalytic residues per chain")
    ax.set_ylabel("Number of chains")
    ax.set_title(f"Catalytic residues per chain (binned, cap={cap})")

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hist_catalytic_per_chain_binned_1_to_{cap-1}_ge{cap}.pdf"), dpi=300)
    plt.close(fig)


def plot_catalytic_per_chain_log(per_chain: pd.DataFrame, out_dir: str, cap: int | None = None) -> None:
    """Histogram with log-scale Y: bins are integers from 1..max (or 1..cap)."""
    # English comments as requested
    x = per_chain["n_catalytic"].to_numpy()
    x = x[np.isfinite(x)].astype(int)
    x = x[x >= 1]  # drop zeros and negatives

    if x.size == 0:
        raise ValueError("No chains with n_catalytic >= 1 found.")

    max_cats = int(x.max())
    max_plot = min(max_cats, int(cap)) if cap is not None else max_cats

    # Optionally fold tail into a final bin >max_plot
    if max_cats > max_plot:
        x_plot = np.minimum(x, max_plot + 1)
        bins = np.arange(0.5, max_plot + 1.5 + 1.0, 1.0)  # includes last bin for max_plot+1
        xticks = list(range(1, max_plot + 1)) + [max_plot + 1]
        xticklabels = [str(i) for i in range(1, max_plot + 1)] + [f">{max_plot}"]
    else:
        x_plot = x
        bins = np.arange(0.5, max_plot + 1.5, 1.0)
        xticks = list(range(1, max_plot + 1))
        xticklabels = [str(i) for i in xticks]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(x_plot, bins=bins)
    ax.set_yscale("log")
    ax.set_xlabel("Number of catalytic residues per chain")
    ax.set_ylabel("Number of chains (log scale)")
    ax.set_title("Catalytic residues per chain (log scale)")

    # Make x-axis readable if range is large
    if len(xticks) <= 25:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    else:
        # show every 2nd or 5th tick depending on width
        step = 2 if len(xticks) <= 60 else 5
        show = xticks[::step]
        ax.set_xticks(show)
        ax.set_xticklabels([xticklabels[i - 1] if i - 1 < len(xticklabels) else str(i) for i in show])

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_catalytic_per_chain_log.pdf"), dpi=300)
    plt.close(fig)


def plot_catalytic_per_chain(per_chain: pd.DataFrame, out_dir: str) -> None:
    """Histogram: number of catalytic residues per chain."""
    max_cats = int(per_chain["n_catalytic"].max())
    bins = np.arange(-0.5, max_cats + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_chain["n_catalytic"], bins=bins)
    ax.set_xlabel("Number of catalytic residues per chain")
    ax.set_ylabel("Number of chains")
    ax.set_title("Catalytic residues per chain")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_catalytic_per_chain.pdf"), dpi=300)
    plt.close(fig)


def plot_class_imbalance(per_chain: pd.DataFrame, out_dir: str) -> None:
    """Barplot: positives vs negatives per split (log Y)."""
    posneg = (
        per_chain
        .groupby("Split", dropna=False)
        .agg(
            n_pos=("n_catalytic", "sum"),
            n_total=("Length", "sum"),
        )
        .reset_index()
    )
    posneg["n_neg"] = posneg["n_total"] - posneg["n_pos"]

    x = np.arange(len(posneg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, posneg["n_neg"], width, label="non-catalytic")
    ax.bar(x + width / 2, posneg["n_pos"], width, label="catalytic")

    ax.set_xticks(x)
    ax.set_xticklabels(posneg["Split"], rotation=45)
    ax.set_yscale("log")
    ax.set_ylabel("Number of residues (log scale)")
    ax.set_title("Class imbalance by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_imbalance_by_split.pdf"), dpi=300)
    plt.close(fig)


def plot_weights_boxplot(per_chain: pd.DataFrame, out_dir: str) -> None:
    """Boxplot of W_Structure by split (chain level)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    data = []
    labels = []
    for split_name, df_split in per_chain.groupby("Split"):
        vals = df_split["W_Structure"].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(split_name)

    if not data:
        plt.close(fig)
        return

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("W_Structure")
    ax.set_title("Chain weights (W_Structure) by split")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boxplot_w_structure_by_split.pdf"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset summary and plots for catalytic-site dataset."
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=(
            "/home/iscb/wolfson/annab4/DB/all_proteins/"
            "cross_validation_chem/weight_based_v9/dataset.csv"
        ),
        help="Path to dataset.csv with metadata.",
    )
    parser.add_argument(
        "--splits_pattern",
        type=str,
        default=(
            "/home/iscb/wolfson/annab4/DB/all_proteins/"
            "cross_validation_chem/weight_based_v9/split*.txt"
        ),
        help="Glob pattern for split*.txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/home/iscb/wolfson/annab4/catalytic-sites-annotation/data_plots/plots/"
        ),
        help="Directory to save tables and figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    dataset = pd.read_csv(args.dataset_csv)
    dataset = dataset.drop_duplicates(subset=["Sequence_ID"])

    # Parse all splits
    split_paths = sorted(glob(args.splits_pattern))
    if not split_paths:
        raise FileNotFoundError(f"No split files matched pattern: {args.splits_pattern}")

    per_chain_list = []
    for path in split_paths:
        base = os.path.basename(path)
        split_name = os.path.splitext(base)[0]  # e.g. "split1"
        df_split = parse_split_file(path)
        df_split["Split"] = split_name
        per_chain_list.append(df_split)

    per_chain = pd.concat(per_chain_list, ignore_index=True)

    # Merge metadata (weights, EC, etc.)
    per_chain = per_chain.merge(
        dataset,
        on="Sequence_ID",
        how="left",
        validate="many_to_one",
    )

    # Save raw per-chain stats
    per_chain.to_csv(
        os.path.join(args.output_dir, "per_chain_stats.csv"),
        index=False,
    )

    # Tables
    compute_overview_table(
        per_chain,
        os.path.join(args.output_dir, "table_overview.csv"),
    )
    compute_ec_table(
        per_chain,
        os.path.join(args.output_dir, "table_ec_distribution.csv"),
    )

    # Plots
    # plot_length_histograms(per_chain, args.output_dir)
    # plot_catalytic_per_chain(per_chain, args.output_dir)
    # plot_class_imbalance(per_chain, args.output_dir)
    # plot_weights_boxplot(per_chain, args.output_dir)

    plot_catalytic_per_chain_binned(per_chain, args.output_dir, 5)
    plot_catalytic_per_chain_binned(per_chain, args.output_dir, 4)
    plot_catalytic_per_chain_log(per_chain, args.output_dir, cap=None)



if __name__ == "__main__":
    main()
