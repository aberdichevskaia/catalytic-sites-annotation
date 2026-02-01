#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset report utilities for catalytic-site dataset.

Inputs:
  1) dataset.csv with columns (at least):
     Sequence_ID, Set_Type, Component_ID, EC_number, W_Structure, W_Sequence, ...
  2) A directory with label .txt files (any names), each in format:
       >SEQID
       A 1 M 0
       A 2 G 0
       ...
  3) (optional) all_protein_table_modified.json with per-accession pdb_ids:
       { "A0A0...": { "pdb_ids": ["4Q6X", ...], ... }, ... }

Outputs (in out_dir), separate svgs (no multi-panel figures):
  (1) Catalytic residues per protein (bins 1..>=4) + log-y:
      - hist_catalytic_per_protein_binned_ge4.svg
      - hist_catalytic_per_protein_binned_ge4_logy.svg

  (2) Chemical classes (chemotypes) across CV splits (raw + normalized):
      - chemotype_by_split_weight_raw.svg
      - chemotype_by_split_weight_norm.svg
      - chemotype_overall.csv
      - chemotype_by_split_long.csv

  (3) Experimental structural coverage per UniProt accession (3 separate plots):
      - structcov_fraction_ge1_pdb.svg
      - structcov_n_pdb_ge1_capped10.svg
      - structcov_top10_accessions_by_pdb.svg

  (4) Connected component sizes (2 separate plots):
      - cc_sizes_hist_ge100.svg
      - cc_sizes_top10.svg
      - connected_components_summary.csv

Usage examples:
  python dataset_report.py all \
    --dataset_csv /path/to/dataset.csv \
    --labels_dir /path/to/labels_dir \
    --protein_json /path/to/all_protein_table_modified.json \
    --out_dir /path/to/out \
    --weight_col W_Structure

Notes:
- "Classes" here means CHEMICAL CLASSES (chemotypes) defined by catalytic AA types.
- EC numbers are treated as EC numbers, not "classes".
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

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

    "svg.fonttype": "path" #"none",  # текст в SVG остаётся текстом
})


# figures sizes (for the paper)
W_FULL = 6.27
GUTTER = 0.15
W_HALF = (W_FULL - GUTTER) / 2
W_THIRD = (W_FULL - GUTTER) / 3
W_TWO_THIRDS = 2 * (W_FULL - GUTTER) / 3

H_SMALL = 1.9
H_WIDE = 2.7


# ------------------------- Constants -------------------------

CHEMOTYPE_LABELS: Dict[int, str] = {
    0: "Class 0 (ILMVWF)",
    1: "Class 1 (AGP)",
    2: "Class 2 (QN)",
    3: "Class 3 (KR)",
    4: "Class 4 (S)",
    5: "Class 5 (T)",
    6: "Class 6 (DE)",
    7: "Class 7 (other/none)",
}


# ------------------------- IO helpers -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_label_txts(labels_dir: str) -> List[str]:
    """Return all .txt files in a directory (non-recursive), sorted."""
    if not os.path.isdir(labels_dir):
        raise NotADirectoryError(f"labels_dir is not a directory: {labels_dir}")
    paths = sorted(glob(os.path.join(labels_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in labels_dir: {labels_dir}")
    return paths


def load_dataset_csv(dataset_csv: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "Sequence_ID" not in df.columns:
        raise KeyError("dataset.csv must contain column 'Sequence_ID'")
    df = df.drop_duplicates(subset=["Sequence_ID"]).copy()
    df["Sequence_ID"] = df["Sequence_ID"].astype(str)
    return df


# ------------------------- Parsing labels -------------------------

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


def parse_label_files(
    label_paths: List[str],
    strict_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Parse label .txt files and return:
      - per_chain DataFrame with columns: Sequence_ID, Length, n_catalytic
      - id_to_resset: Sequence_ID -> set of AA types at catalytic positions (label==1)

    Duplicate Sequence_ID across files is considered an error by default.
    """
    records: List[Tuple[str, int, int]] = []
    id_to_resset: Dict[str, Set[str]] = {}
    seen: Set[str] = set()

    for path in label_paths:
        cur_id: Optional[str] = None
        length = 0
        n_pos = 0
        resset: Set[str] = set()

        def flush() -> None:
            nonlocal cur_id, length, n_pos, resset
            if cur_id is None:
                return
            if cur_id in seen:
                msg = f"Duplicate Sequence_ID in label files: {cur_id} (found again in {path})"
                if strict_duplicates:
                    raise ValueError(msg)
                else:
                    print(f"[WARN] {msg} -> keeping first occurrence, skipping duplicate.")
                    return
            seen.add(cur_id)
            records.append((cur_id, int(length), int(n_pos)))
            id_to_resset[cur_id] = set(resset)

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush()
                    cur_id = line[1:].split()[0].strip()
                    length = 0
                    n_pos = 0
                    resset = set()
                    continue

                # Expected: chain_id, index, aa, label
                parts = line.split()
                if len(parts) < 4:
                    continue
                aa = parts[2].strip()
                try:
                    y = int(parts[3])
                except ValueError:
                    continue

                length += 1
                if y == 1:
                    n_pos += 1
                    resset.add(aa)

        flush()

    per_chain = pd.DataFrame(records, columns=["Sequence_ID", "Length", "n_catalytic"])
    return per_chain, id_to_resset


def build_per_chain_with_metadata(
    dataset_df: pd.DataFrame,
    labels_dir: str,
    strict_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Merge label-derived per-chain stats with dataset.csv metadata.
    Keeps only Sequence_ID present in BOTH dataset.csv and labels.
    """
    label_paths = list_label_txts(labels_dir)
    per_chain, id_to_resset = parse_label_files(label_paths, strict_duplicates=strict_duplicates)

    common = sorted(set(dataset_df["Sequence_ID"]) & set(per_chain["Sequence_ID"]))
    if len(common) == 0:
        raise ValueError("No overlap between dataset.csv Sequence_IDs and label files Sequence_IDs")

    if len(common) < len(dataset_df):
        print(f"[WARN] Missing labels for {len(dataset_df) - len(common)} / {len(dataset_df)} dataset chains. Dropping them.")
    if len(common) < len(per_chain):
        print(f"[WARN] Labels contain {len(per_chain) - len(common)} chains not in dataset.csv. Dropping them.")

    per_chain = per_chain[per_chain["Sequence_ID"].isin(common)].copy()
    dataset_sub = dataset_df[dataset_df["Sequence_ID"].isin(common)].copy()

    merged = per_chain.merge(dataset_sub, on="Sequence_ID", how="left", validate="one_to_one")
    return merged, id_to_resset


# ------------------------- Tables -------------------------

def compute_table_overview(per_chain: pd.DataFrame, weight_col: str, out_csv: str) -> None:
    """
    Overview by split + overall row:
      n_chains, n_residues, n_catalytic, mean_length, p25, p75,
      sum weights (W_Structure, W_Sequence if present), and frac_catalytic.
    """
    need = {"Set_Type", "Sequence_ID", "Length", "n_catalytic"}
    missing = need - set(per_chain.columns)
    if missing:
        raise KeyError(f"per_chain missing required columns: {sorted(missing)}")

    # Allow weight columns if present
    has_w_struct = "W_Structure" in per_chain.columns
    has_w_seq = "W_Sequence" in per_chain.columns

    agg_dict = {
        "n_chains": ("Sequence_ID", "nunique"),
        "n_residues": ("Length", "sum"),
        "n_catalytic": ("n_catalytic", "sum"),
        "mean_length": ("Length", "mean"),
        "p25_length": ("Length", lambda x: x.quantile(0.25)),
        "p75_length": ("Length", lambda x: x.quantile(0.75)),
    }
    if has_w_struct:
        agg_dict["sum_w_structure"] = ("W_Structure", "sum")
    if has_w_seq:
        agg_dict["sum_w_sequence"] = ("W_Sequence", "sum")
    if weight_col in per_chain.columns and weight_col not in {"W_Structure", "W_Sequence"}:
        agg_dict[f"sum_{weight_col}"] = (weight_col, "sum")

    grouped = (
        per_chain
        .groupby("Set_Type", dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"Set_Type": "Split"})
    )
    grouped["frac_catalytic"] = grouped["n_catalytic"] / grouped["n_residues"].replace(0, np.nan)

    # Overall row
    total = {
        "Split": "all",
        "n_chains": int(per_chain["Sequence_ID"].nunique()),
        "n_residues": int(per_chain["Length"].sum()),
        "n_catalytic": int(per_chain["n_catalytic"].sum()),
        "mean_length": float(per_chain["Length"].mean()),
        "p25_length": float(per_chain["Length"].quantile(0.25)),
        "p75_length": float(per_chain["Length"].quantile(0.75)),
    }
    if has_w_struct:
        total["sum_w_structure"] = float(per_chain["W_Structure"].sum())
    if has_w_seq:
        total["sum_w_sequence"] = float(per_chain["W_Sequence"].sum())
    if weight_col in per_chain.columns and weight_col not in {"W_Structure", "W_Sequence"}:
        total[f"sum_{weight_col}"] = float(per_chain[weight_col].sum())

    total["frac_catalytic"] = total["n_catalytic"] / total["n_residues"] if total["n_residues"] > 0 else np.nan

    out = pd.concat([grouped, pd.DataFrame([total])], ignore_index=True)
    out.to_csv(out_csv, index=False)


def build_chemotype_table(
    per_chain: pd.DataFrame,
    id_to_resset: Dict[str, Set[str]],
    weight_col: str,
) -> pd.DataFrame:
    """
    Attach chemotype to per_chain rows. Uses true catalytic residue types from labels.
    """
    if weight_col not in per_chain.columns:
        raise KeyError(f"weight_col '{weight_col}' not found in dataset.csv merged columns")

    df = per_chain.copy()
    df["chemotype"] = df["Sequence_ID"].map(lambda sid: chemotype_from_residue_set(id_to_resset.get(sid, set())))
    df["chemotype_name"] = df["chemotype"].map(CHEMOTYPE_LABELS)
    df[weight_col] = df[weight_col].astype(float)
    return df


# ------------------------- Plotting: catalytic residues per protein -------------------------

def plot_catalytic_per_protein_binned_ge4(per_chain: pd.DataFrame, out_svg: str, cap: int = 4) -> None:
    """
    Histogram: bins are 1, 2, ..., cap-1, >=cap (cap=4 -> 1,2,3,>=4).
    Zeros are dropped. Tail is folded into the last bin by clipping.
    """
    base = plt.colormaps["Greens"](0.65)
    tail = "#999999"
    colors = [base] * (cap - 1) + [tail]   # last bin (>=cap) in gray

    x = per_chain["n_catalytic"].to_numpy()
    x = x[np.isfinite(x)].astype(int)
    x = x[x >= 1]
    if x.size == 0:
        print("[WARN] No proteins with n_catalytic>=1. Skip plot.")
        return

    # Fold tail into >=cap by clipping everything >=cap to exactly cap
    x_clip = np.minimum(x, cap)

    # Bin edges so that each integer k sits in its own bin:
    # 0.5-1.5 => 1, ..., (cap-0.5)-(cap+0.5) => cap (which represents >=cap)
    bin_edges = np.arange(0.5, cap + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(W_THIRD, H_SMALL))
    n, bins, patches = ax.hist(x_clip, bins=bin_edges)

    # Color each bin patch
    for p, c in zip(patches, colors):
        p.set_facecolor(c)
        p.set_edgecolor("black")
        p.set_linewidth(0.8)

    xs = np.arange(1, cap + 1)
    labels = [str(i) for i in range(1, cap)] + [f">={cap}"]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    ax.set_xlabel("# catalytic residues per protein")
    ax.set_ylabel("Number of proteins")
    #ax.set_title("Catalytic residues per protein (binned)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)

def plot_catalytic_per_chain_log(per_chain: pd.DataFrame, out_svg: str) -> None:
    """Histogram with log-scale Y: integer bins from 1..max (NO capping)."""
    # English comments as requested
    base = plt.colormaps["Greens"](0.65)
    x = per_chain["n_catalytic"].to_numpy()
    x = x[np.isfinite(x)].astype(int)
    x = x[x >= 1]  # drop zeros and negatives

    if x.size == 0:
        raise ValueError("No chains with n_catalytic >= 1 found.")

    max_cats = int(x.max())

    bins = np.arange(0.5, max_cats + 1.5, 1.0)
    xticks = list(range(1, max_cats + 1))

    fig, ax = plt.subplots(figsize=(W_TWO_THIRDS, H_SMALL))
    ax.hist(x, bins=bins, color=base)
    ax.set_yscale("log")
    ax.set_xlabel("# catalytic residues per protein")
    ax.set_ylabel("Number of proteins (log scale)")
    #ax.set_title("Catalytic residues per chain (log scale)")

    # Make x-axis readable if range is large
    if len(xticks) <= 25:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks])
    else:
        step = 2 if len(xticks) <= 60 else 5
        show = xticks[::step]
        ax.set_xticks(show)
        ax.set_xticklabels([str(i) for i in show])

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)



# ------------------------- Plotting: chemotypes across splits -------------------------

import matplotlib
from matplotlib.patches import Patch

def get_chemotype_colors():
    cmap = matplotlib.colormaps.get("Paired")
    colors = [cmap(i) for i in range(8)]
    colors[7] = (0.6, 0.6, 0.6, 1.0)  # other/none gray
    return colors


def plot_chemotype_by_split_stacked(
    df_long: pd.DataFrame,
    split_order: List[str],
    class_order: List[int],
    value_col: str,
    normalize: bool,
    out_svg: str,
    title: str,
    ylabel: str,
) -> None:
    """
    Stacked bar plot for chemotypes across splits.
    df_long columns: Split(Set_Type), chemotype, value_col.
    """
    cmap = matplotlib.colormaps.get("Paired")  # try: "Set2", "Dark2", "Paired"
    colors = [cmap(i) for i in range(8)]
    colors[7] = (0.6, 0.6, 0.6, 1.0)  # make "other/none" gray

    piv = (
        df_long.pivot_table(index="Split", columns="chemotype", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(split_order)
    )
    for c in class_order:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[class_order]

    mat = piv.to_numpy(dtype=float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(W_THIRD, H_SMALL))
    x = np.arange(len(split_order))
    bottom = np.zeros(len(split_order), dtype=float)

    for j, c in enumerate(class_order):
        vals = mat[:, j]
        ax.bar(x, vals, bottom=bottom, label=CHEMOTYPE_LABELS.get(c, str(c)), color=colors[j])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=0)
    ax.set_ylabel(ylabel)
    #ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    #fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)

def save_chemotype_legend(out_svg: str, fontsize: int = 10) -> None:
    colors = get_chemotype_colors()
    handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=CHEMOTYPE_LABELS[i])
        for i in range(8)
    ]

    fig = plt.figure(figsize=(W_THIRD, H_SMALL))  # размер не важен: мы tight-crop'нем
    fig.legend(handles=handles, loc="center left", frameon=False)

    fig.savefig(out_svg, dpi=300, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



# ------------------------- Plotting: structural coverage per accession -------------------------

def plot_fraction_accessions_ge1_pdb(n_struct: np.ndarray, out_svg: str) -> None:
    """Bar plot: fraction of accessions with 0 PDB vs >=1 PDB."""
    base = plt.colormaps["Purples"](0.65)
    n_struct = np.asarray(n_struct, dtype=int)
    if n_struct.size == 0:
        print("[WARN] Empty n_struct. Skip plot.")
        return

    frac_ge1 = float((n_struct >= 1).mean())
    frac_0 = 1.0 - frac_ge1

    fig, ax = plt.subplots(figsize=(W_THIRD, H_SMALL))
    ax.bar([0, 1], [frac_0, frac_ge1], color=base)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 PDB", "≥1 PDB"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of accessions")
    #ax.set_title("PDB coverage per UniProt accession")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0, frac_0 + 0.03, f"{100*frac_0:.1f}%", ha="center")
    ax.text(1, frac_ge1 + 0.03, f"{100*frac_ge1:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)


def plot_distribution_pdb_counts_ge1_capped(n_struct: np.ndarray, cap: int, out_svg: str) -> None:
    """Bar plot for accessions with >=1 PDB: bins 1..cap plus final bin >cap."""
    base = plt.colormaps["Purples"](0.65)
    tail = "#999999"
    colors = [base]*(cap) + [tail]

    n_struct = np.asarray(n_struct, dtype=int)
    n_pos = n_struct[n_struct >= 1]

    xs = np.arange(1, cap + 2)  # 1..cap+1
    labels = [str(i) for i in range(1, cap + 1)] + [f">{cap}"]

    if n_pos.size == 0:
        y = np.zeros_like(xs)
    else:
        clipped = np.minimum(n_pos, cap + 1)  # cap+1 means >cap
        counts = np.bincount(clipped, minlength=cap + 2)
        y = counts[1:cap + 2]

    fig, ax = plt.subplots(figsize=(W_HALF, H_SMALL))
    ax.bar(xs, y, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# PDB structures per accession")
    ax.set_ylabel("# accessions")
    #ax.set_title(f"Multiplicity among accessions with PDB structures (cap={cap})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)


def plot_top_accessions_by_pdb_count(
    acc_to_n: Dict[str, int],
    acc_to_name: Dict[str, str],
    top_k: int,
    out_svg: str,
) -> None:
    """Horizontal bar plot: top-K proteins by #PDB structures (label = protein name)."""
    base = plt.colormaps["Purples"](0.65)
    items = [(str(acc), int(n)) for acc, n in acc_to_n.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    top = items[:top_k]
    if not top:
        print("[WARN] Empty acc_to_n. Skip plot.")
        return

    # Use protein names; fallback to accession if name missing/empty
    def pretty_label(acc: str) -> str:
        name = acc_to_name.get(acc, "")
        name = str(name).strip()
        return name if name else acc

    labels = [pretty_label(acc) for acc, _ in top][::-1]
    values = [v for _, v in top][::-1]

    fig_h = max(3.2, 0.35 * len(values) + 1.2)
    fig, ax = plt.subplots(figsize=(W_TWO_THIRDS, H_SMALL))#(figsize=(10.5, fig_h))  # чуть шире, имена длинные
    ax.barh(np.arange(len(values)), values, color=base)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("# PDB structures")
    #ax.set_title(f"Top-{top_k} proteins by #PDB structures")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)



# ------------------------- Plotting: connected components -------------------------

def plot_component_sizes_hist_gecap(comp_sizes: np.ndarray, cap: int, out_svg: str) -> None:
    """
    Histogram: bins are 1..(cap-1) and a final bin >=cap.
    No bin for 0 (component size cannot be 0).
    """
    base = plt.colormaps["Oranges"](0.65)
    tail = "#999999"
    colors = [base] * (cap - 1) + [tail]

    sizes = np.asarray(comp_sizes, dtype=int)
    sizes = sizes[sizes >= 1]
    if sizes.size == 0:
        print("[WARN] Empty component sizes. Skip plot.")
        return

    # Fold tail into >=cap by clipping
    x = np.minimum(sizes, cap)

    # Bin edges: 0.5-1.5 => 1, ..., (cap-0.5)-(cap+0.5) => cap (represents >=cap)
    bin_edges = np.arange(0.5, cap + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(W_FULL, H_WIDE))
    n, bins, patches = ax.hist(x, bins=bin_edges)

    # Color each bin patch
    for p, c in zip(patches, colors):
        p.set_facecolor(c)
        p.set_edgecolor("black")
        p.set_linewidth(0.8)

    # Ticks: show a subset + last as ">=cap"
    xs = np.arange(1, cap + 1)
    step = 10
    ticks = list(range(1, cap, step)) + [cap]
    ticklabels = [str(t) for t in range(1, cap, step)] + [f">={cap}"]
    ax.set_xticks(ticks)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticklabels(ticklabels)

    ax.set_xlabel("Component size (# chains)")
    ax.set_ylabel("Number of components")
    #ax.set_title(f"Connected component sizes (last bin ≥{cap})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)



def plot_top_components(comp_sizes: np.ndarray, top_k: int, out_svg: str) -> None:
    """Bar plot: sizes of the top-K largest connected components."""
    base = plt.colormaps["Oranges"](0.65)
    sizes = np.asarray(comp_sizes, dtype=int)
    sizes = sizes[sizes >= 1]
    if sizes.size == 0:
        print("[WARN] Empty component sizes. Skip plot.")
        return

    top = np.sort(sizes)[::-1][:top_k]
    x = np.arange(1, len(top) + 1)

    fig, ax = plt.subplots(figsize=(W_HALF, H_SMALL))
    ax.bar(x, top, color=base)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in x])
    ax.set_xlabel("Largest components (rank)")
    ax.set_ylabel("Component size (# chains)")
    #ax.set_title(f"Top-{top_k} largest connected components")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values
    y_pad = max(1, int(0.02 * top[0])) if len(top) > 0 else 1
    for i, v in enumerate(top, start=1):
        ax.text(i, v + y_pad, str(int(v)), ha="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300, format="svg")
    plt.close(fig)


# ------------------------- Commands -------------------------

def cmd_overview(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    dataset_df = load_dataset_csv(args.dataset_csv)
    per_chain, _ = build_per_chain_with_metadata(dataset_df, args.labels_dir, strict_duplicates=args.strict_duplicates)

    per_chain.to_csv(os.path.join(args.out_dir, "per_chain_stats.csv"), index=False)

    out_table = os.path.join(args.out_dir, "table_overview.csv")
    compute_table_overview(per_chain, args.weight_col, out_table)
    print(f"[OK] Wrote {out_table}")

    plot_catalytic_per_protein_binned_ge4(
        per_chain,
        out_svg=os.path.join(args.out_dir, "hist_catalytic_per_protein_binned_ge4.svg"),
        cap=4,
    )
    plot_catalytic_per_chain_log(
        per_chain,
        out_svg=os.path.join(args.out_dir, "hist_catalytic_per_protein_binned_ge4_logy.svg"),
    )
    print("[OK] Wrote catalytic-per-protein plots")


def cmd_chem(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    dataset_df = load_dataset_csv(args.dataset_csv)
    per_chain, id_to_resset = build_per_chain_with_metadata(dataset_df, args.labels_dir, strict_duplicates=args.strict_duplicates)

    df = build_chemotype_table(per_chain, id_to_resset, args.weight_col)

    # Overall chemotype distribution
    total_n = int(df["Sequence_ID"].nunique())
    total_w = float(df[args.weight_col].sum())

    overall = (
        df.groupby("chemotype")
        .agg(n_chains=("Sequence_ID", "nunique"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
        .sort_values("chemotype")
        .reset_index(drop=True)
    )
    overall["chains_pct"] = 100.0 * overall["n_chains"] / max(total_n, 1)
    overall["weight_pct"] = 100.0 * overall["weight_sum"] / max(total_w, 1e-12)
    overall["chemotype_name"] = overall["chemotype"].map(CHEMOTYPE_LABELS)

    overall_out = os.path.join(args.out_dir, "chemotype_overall.csv")
    overall.to_csv(overall_out, index=False)
    print(f"[OK] Wrote {overall_out}")

    # By split (long)
    by_split = (
        df.groupby(["Set_Type", "chemotype"])
        .agg(n_chains=("Sequence_ID", "nunique"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
        .rename(columns={"Set_Type": "Split"})
        .sort_values(["Split", "chemotype"])
        .reset_index(drop=True)
    )
    by_split_out = os.path.join(args.out_dir, "chemotype_by_split_long.csv")
    by_split.to_csv(by_split_out, index=False)
    print(f"[OK] Wrote {by_split_out}")

    split_order = sorted(df["Set_Type"].astype(str).unique().tolist(), key=lambda s: (int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else 10**9, s))
    class_order = list(range(8))

    # Raw stacked by weight
    plot_chemotype_by_split_stacked(
        df_long=by_split,
        split_order=split_order,
        class_order=class_order,
        value_col="weight_sum",
        normalize=False,
        out_svg=os.path.join(args.out_dir, "chemotype_by_split_weight_raw.svg"),
        title=f"Chemical class distribution across CV splits (raw {args.weight_col})",
        ylabel=args.weight_col,
    )

    # Normalized stacked by weight
    plot_chemotype_by_split_stacked(
        df_long=by_split,
        split_order=split_order,
        class_order=class_order,
        value_col="weight_sum",
        normalize=True,
        out_svg=os.path.join(args.out_dir, "chemotype_by_split_weight_norm.svg"),
        title=f"Chemical class distribution across CV splits (normalized by {args.weight_col})",
        ylabel="Fraction of split",
    )

    save_chemotype_legend(os.path.join(args.out_dir, "chemotype_legend.svg"), fontsize=10)

    print("[OK] Wrote chemotype plots (raw + normalized)")


def cmd_redundancy(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    dataset_df = load_dataset_csv(args.dataset_csv)

    need = {"Set_Type", "Component_ID", "Sequence_ID"}
    missing = need - set(dataset_df.columns)
    if missing:
        raise KeyError(f"dataset.csv missing columns required for redundancy: {sorted(missing)}")

    dataset_df["Component_ID"] = dataset_df["Component_ID"].astype(int)

    # Connected components
    comp = (
        dataset_df.groupby("Component_ID")
        .agg(n_chains=("Sequence_ID", "count"))
        .reset_index()
        .sort_values("n_chains", ascending=False)
        .reset_index(drop=True)
    )
    comp_out = os.path.join(args.out_dir, "connected_components_summary.csv")
    comp.to_csv(comp_out, index=False)
    print(f"[OK] Wrote {comp_out}")

    sizes = comp["n_chains"].to_numpy(dtype=int)
    plot_component_sizes_hist_gecap(
        sizes, cap=100, out_svg=os.path.join(args.out_dir, "cc_sizes_hist_ge100.svg")
    )
    plot_top_components(
        sizes, top_k=10, out_svg=os.path.join(args.out_dir, "cc_sizes_top10.svg")
    )
    print("[OK] Wrote connected-components plots")

    # Structural coverage (optional)
    if args.protein_json:
        with open(args.protein_json, "r", encoding="utf-8") as f:
            prot = json.load(f)
            acc_to_name = {acc: rec.get("full_name", "") for acc, rec in prot.items()}

        dataset_acc = sorted(set(dataset_df["Sequence_ID"].astype(str).str.split("_").str[0].tolist()))
        acc_to_n: Dict[str, int] = {}
        for acc in dataset_acc:
            rec = prot.get(acc, None)
            pdb_ids = [] if rec is None else (rec.get("pdb_ids", []) or [])
            acc_to_n[acc] = int(len(pdb_ids))

        n_struct = np.array([acc_to_n[a] for a in dataset_acc], dtype=int)
        pd.DataFrame({"accession": dataset_acc, "n_pdb_structures": n_struct}).to_csv(
            os.path.join(args.out_dir, "structures_per_accession_values.csv"), index=False
        )

        plot_fraction_accessions_ge1_pdb(
            n_struct, os.path.join(args.out_dir, "structcov_fraction_ge1_pdb.svg")
        )
        plot_distribution_pdb_counts_ge1_capped(
            n_struct, cap=10, out_svg=os.path.join(args.out_dir, "structcov_n_pdb_ge1_capped10.svg")
        )
        plot_top_accessions_by_pdb_count(
            acc_to_n, acc_to_name, top_k=10, out_svg=os.path.join(args.out_dir, "structcov_top10_accessions_by_pdb.svg")
        )
        print("[OK] Wrote structural-coverage plots")


def cmd_all(args: argparse.Namespace) -> None:
    # Reuse the same args; just call the subcommands in a fixed order.
    cmd_overview(args)
    cmd_chem(args)
    cmd_redundancy(args)


# ------------------------- CLI -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset analysis + plots (separate svgs).")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset_csv", type=str, required=True, help="Path to dataset.csv")
        sp.add_argument("--labels_dir", type=str, required=True, help="Directory with label .txt files")
        sp.add_argument("--out_dir", type=str, required=True, help="Output directory")
        sp.add_argument("--weight_col", type=str, default="W_Structure", help="Weight column for chemotype plots/tables")
        sp.add_argument(
            "--strict_duplicates",
            action="store_true",
            help="Fail if the same Sequence_ID appears in multiple label files (recommended).",
        )

    sp_over = sub.add_parser("overview", help="Overview tables + catalytic-per-protein plots")
    add_common(sp_over)

    sp_chem = sub.add_parser("chem", help="Chemotype tables + chemotype-by-split plots")
    add_common(sp_chem)

    sp_red = sub.add_parser("redundancy", help="Redundancy / dataset skew plots (CC + structural coverage)")
    add_common(sp_red)
    sp_red.add_argument(
        "--protein_json",
        type=str,
        default=None,
        help="Path to all_protein_table_modified.json (for PDB coverage per accession).",
    )

    sp_all = sub.add_parser("all", help="Run overview + chem + redundancy")
    add_common(sp_all)
    sp_all.add_argument(
        "--protein_json",
        type=str,
        default=None,
        help="Path to all_protein_table_modified.json (for PDB coverage per accession).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default: strict duplicates ON unless user explicitly disables (we only have --strict_duplicates flag)
    # For safety, if user didn't provide the flag, we still want strict behavior:
    if not getattr(args, "strict_duplicates", False):
        # Keep backward compatibility: default strict True.
        args.strict_duplicates = True

    if args.command == "overview":
        cmd_overview(args)
    elif args.command == "chem":
        cmd_chem(args)
    elif args.command == "redundancy":
        cmd_redundancy(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
