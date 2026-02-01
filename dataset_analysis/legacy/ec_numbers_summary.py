#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute EC class distribution (counts and W_Structure weights)
for the catalytic sites dataset.

Input:
    /home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv

Output:
    CSV summary + LaTeX table snippet.
"""

import os
import pandas as pd
import numpy as np

# ---- paths ----

DATASET_CSV = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv"
OUT_DIR = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/data_plots/plots"
OUT_CSV = os.path.join(OUT_DIR, "ec_distribution.csv")


# ---- helpers ----

def get_top_ec(ec_value):
    """Extract top-level EC class (1..7) or 'Unknown'."""
    if pd.isna(ec_value):
        return "Unknown"
    ec_str = str(ec_value).strip()
    if not ec_str or ec_str in {"-", "nan", "None"}:
        return "Unknown"
    top = ec_str.split(".", 1)[0]
    if top in {"1", "2", "3", "4", "5", "6", "7"}:
        return top
    return "Unknown"


EC_LABELS = {
    "1": "EC 1: Oxidoreductases",
    "2": "EC 2: Transferases",
    "3": "EC 3: Hydrolases",
    "4": "EC 4: Lyases",
    "5": "EC 5: Isomerases",
    "6": "EC 6: Ligases",
    "7": "EC 7: Translocases",
    "Unknown": "Unknown / no EC",
}

EC_ORDER = ["1", "2", "3", "4", "5", "6", "7", "Unknown"]


# ---- main ----

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_CSV)
    print(f"[INFO] Raw rows in dataset.csv: {len(df)}")

    # Use unique Sequence_IDs (как в пайплайне)
    df = df.drop_duplicates(subset=["Sequence_ID"])
    print(f"[INFO] Unique Sequence_ID:       {len(df)}")

    # Make sure weight column exists
    if "W_Structure" not in df.columns:
        raise ValueError("Column 'W_Structure' not found in dataset.csv")

    df["W_Structure"] = df["W_Structure"].fillna(0.0)

    # Top-level EC class
    df["EC_top"] = df["EC_number"].apply(get_top_ec)
    df["EC_label"] = df["EC_top"].map(EC_LABELS)

    total_chains = len(df)
    total_weight = df["W_Structure"].sum()

    print(f"[INFO] Total chains (dedup):     {total_chains}")
    print(f"[INFO] Total W_Structure:        {total_weight:.3f}")

    # Aggregate by EC_top
    grouped = (
        df.groupby("EC_top")
        .agg(
            n_chains=("Sequence_ID", "nunique"),
            total_weight=("W_Structure", "sum"),
        )
        .reset_index()
    )

    # Add labels and percentages
    grouped["EC_label"] = grouped["EC_top"].map(EC_LABELS)
    grouped["pct_chains"] = 100.0 * grouped["n_chains"] / float(total_chains)
    grouped["pct_weight"] = 100.0 * grouped["total_weight"] / float(total_weight)

    # Reorder rows
    grouped["EC_top"] = pd.Categorical(grouped["EC_top"], categories=EC_ORDER, ordered=True)
    grouped = grouped.sort_values("EC_top").reset_index(drop=True)

    # Pretty rounding для вывода/CSV
    grouped_rounded = grouped.copy()
    grouped_rounded["total_weight"] = grouped_rounded["total_weight"].round(3)
    grouped_rounded["pct_chains"] = grouped_rounded["pct_chains"].round(2)
    grouped_rounded["pct_weight"] = grouped_rounded["pct_weight"].round(2)

    # Save CSV
    grouped_rounded.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved EC distribution CSV to {OUT_CSV}")
    print()
    print("[INFO] EC distribution (rounded):")
    print(grouped_rounded)

    # Print LaTeX snippet
    print("\n[LaTeX] Table rows (paste into tabular):\n")
    for _, row in grouped_rounded.iterrows():
        label = row["EC_label"]
        n = int(row["n_chains"])
        pc = row["pct_chains"]
        wt = row["total_weight"]
        pw = row["pct_weight"]
        # EC X: Name  &  #chains & %chains & Total weight & %weight \\
        print(f"{label} & {n} & {pc:.1f} & {wt:.1f} & {pw:.1f} \\\\")


if __name__ == "__main__":
    main()
