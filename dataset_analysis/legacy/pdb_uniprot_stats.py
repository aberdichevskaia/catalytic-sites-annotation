#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset characterization plots & tables.

Reads:
  1) dataset.csv with columns:
     Sequence_ID, Set_Type, Component_ID, W_Structure (and others)
  2) all_protein_table_modified.json (for #structures per accession)
  3) split*.txt label files (optional; for chemotypes from TRUE catalytic residues)

Produces (in out_dir):
  - connected_components_size_hist.pdf
  - pdb_afdb_fraction_by_split_counts_norm.pdf
  - pdb_afdb_fraction_by_split_weight_norm.pdf
  - structures_per_accession_hist.pdf
  - (optional) chemotype_overall.csv + chemotype_overall_table.tex
  - (optional) chemotype_by_split_long.csv + chemotype_by_split_weight_norm.pdf

Notes:
  - PDB vs AFDB heuristic:
      base = Sequence_ID.split('_')[0]
      if base matches 4-char PDB id (e.g., 3FII), -> "PDB", else -> "AFDB/UniProt"
  - Chemotypes are computed from TRUE catalytic residues in label txts (label==1),
    using your priority rule (0..7).
"""

import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


# ------------------------- Helpers -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_sort_key(s: str) -> Tuple[int, str]:
    """Sort split names like split1, split2, ..., then fallback to string."""
    m = re.match(r"^split(\d+)$", str(s).strip())
    if m:
        return (int(m.group(1)), str(s))
    return (10**9, str(s))


def infer_source_from_sequence_id(seq_id: str) -> str:
    """
    Infer PDB vs AFDB/UniProt from Sequence_ID.
    PDB ids are 4 chars; common pattern: digit + 3 alnum (e.g., 3FII, 2J44).
    """
    base = str(seq_id).split("_")[0]
    if re.match(r"^[0-9][A-Za-z0-9]{3}$", base):
        return "PDB"
    return "AFDB/UniProt"


def get_palette_colors(n: int, palette: str) -> List:
    """Return a list of colors for n categories."""
    palette = (palette or "").lower()

    if palette in {"okabe_ito", "okabe-ito", "okabe"}:
        # Colorblind-friendly 8-color set; last one gray works well for "other"
        colors = [
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
            "#999999",  # gray
        ]
        if n <= len(colors):
            return colors[:n]
        # If more than 8, repeat (rare in your use-case)
        return [colors[i % len(colors)] for i in range(n)]

    # Matplotlib categorical colormaps
    import matplotlib as mpl  # local import

    cmap_name = palette if palette else "tab10"
    try:
        cmap = mpl.colormaps.get(cmap_name)
    except Exception:
        cmap = mpl.colormaps.get("tab10")

    return [cmap(i) for i in range(n)]


# ------------------------- Plotting -------------------------

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_has_pdb_bar(n_struct: np.ndarray, out_pdf: str):
    # English comments as requested
    n_struct = np.asarray(n_struct, dtype=int)
    has = (n_struct >= 1).sum()
    
    no = (n_struct == 0).sum()

    print(has, no)
    print(100*has/(has+no), 100*no/(has+no))

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar([0, 1], [no, has])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no PDB", "≥1 PDB"])
    ax.set_ylabel("# accessions")
    ax.set_title("PDB coverage per accession")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_structures_per_accession_positive_only(n_struct: np.ndarray, cap: int, out_pdf: str):
    # English comments as requested
    n_struct = np.asarray(n_struct, dtype=int)
    n_struct = n_struct[n_struct >= 1]  # drop zeros

    clipped = np.minimum(n_struct, cap + 1)
    counts = np.bincount(clipped, minlength=cap + 2)[:cap + 2]

    # We want bins 1..cap and >cap
    x = np.arange(1, cap + 2)  # 1..cap+1
    y = counts[1:cap + 2]
    print(y[-1], 100*y[-1]/(sum(y)))
    labels = [str(i) for i in range(1, cap + 1)] + [f">{cap}"]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# PDB structures per accession (given ≥1)")
    ax.set_ylabel("# accessions")
    ax.set_title("Multiplicity among accessions with PDB structures")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_structures_per_accession_capped(n_struct: np.ndarray,
                                         cap: int,
                                         out_pdf: str,
                                         title: str = "#structures per accession (from JSON pdb_ids)"):
    # English comments as requested
    n_struct = np.asarray(n_struct, dtype=int)
    n_struct = n_struct[n_struct >= 0]

    # Clip everything > cap into a single bucket (cap+1)
    clipped = np.minimum(n_struct, cap + 1)

    # Count frequencies for 0..cap and >cap (stored as cap+1)
    counts = np.bincount(clipped, minlength=cap + 2)[:cap + 2]

    x = np.arange(cap + 2)
    labels = [str(i) for i in range(cap + 1)] + [f">{cap}"]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x, counts)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# PDB structures per accession")
    ax.set_ylabel("# accessions")
    ax.set_title(title)

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_hist(values: np.ndarray, bins: int, xlabel: str, ylabel: str, title: str, out_pdf: str) -> None:
    """Plot a simple histogram."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_component_sizes_zoom(sizes: np.ndarray, xmax: int, bins: int, out_pdf: str):
    sizes = np.asarray(sizes, dtype=int)
    sizes_zoom = sizes[sizes <= xmax]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(sizes_zoom, bins=bins)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Component size (# chains)")
    ax.set_ylabel("Number of components")
    ax.set_title(f"Connected component sizes (zoom: ≤{xmax})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_top_components(comp_sizes: np.ndarray, top_k: int, out_pdf: str):
    comp_sizes = np.asarray(comp_sizes, dtype=int)
    top = np.sort(comp_sizes)[::-1][:top_k]

    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x, top)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in range(len(top))])
    ax.set_xlabel("Largest components (rank)")
    ax.set_ylabel("Component size (# chains)")
    ax.set_title(f"Top-{top_k} largest connected components")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)



def plot_stacked_bar_2groups_norm(df_long: pd.DataFrame,
                                  split_order: List[str],
                                  value_col: str,
                                  group_col: str,
                                  groups: List[str],
                                  title: str,
                                  ylabel: str,
                                  out_pdf: str,
                                  colors: Optional[List] = None) -> None:
    """
    Normalized (100%) stacked bar plot for two groups (e.g., PDB vs AFDB/UniProt).
    df_long columns: Set_Type, group_col, value_col
    """
    piv = (
        df_long.pivot_table(index="Set_Type", columns=group_col, values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(split_order)
    )
    for g in groups:
        if g not in piv.columns:
            piv[g] = 0.0
    piv = piv[groups]

    mat = piv.to_numpy(dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    mat = mat / row_sum

    if colors is None:
        colors = get_palette_colors(2, "tab10")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(split_order))
    bottom = np.zeros(len(split_order), dtype=float)

    for j, g in enumerate(groups):
        vals = mat[:, j]
        ax.bar(x, vals, bottom=bottom, label=g, color=colors[j])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


def plot_stacked_bar_multi_norm(df_long: pd.DataFrame,
                                split_order: List[str],
                                class_order: List[int],
                                value_col: str,
                                title: str,
                                ylabel: str,
                                out_pdf: str,
                                labels: Dict[int, str],
                                palette: str) -> None:
    """
    Normalized (100%) stacked bar plot for multiple classes (e.g., 8 chemotypes).
    df_long columns: Set_Type, chemotype, value_col
    """
    piv = (
        df_long.pivot_table(index="Set_Type", columns="chemotype", values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(split_order)
    )
    for c in class_order:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[class_order]

    mat = piv.to_numpy(dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    mat = mat / row_sum

    colors = get_palette_colors(len(class_order), palette)

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(split_order))
    bottom = np.zeros(len(split_order), dtype=float)

    for j, c in enumerate(class_order):
        vals = mat[:, j]
        ax.bar(x, vals, bottom=bottom, label=labels.get(c, str(c)), color=colors[j])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(split_order, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


# ------------------------- Chemotypes (optional) -------------------------

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


def parse_label_txt(path: str) -> Dict[str, Set[str]]:
    """
    Parse split*.txt:
      >SEQID
      A 1 M 0
      ...
    Returns: id -> set(residue types at catalytic positions)  (label==1)
    """
    out: Dict[str, Set[str]] = {}
    cur_id: Optional[str] = None
    cat_set: Set[str] = set()

    def flush() -> None:
        nonlocal cur_id, cat_set
        if cur_id is not None:
            out[cur_id] = set(cat_set)

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_id = line[1:].strip()
                cat_set = set()
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

    flush()
    return out


def build_chemotype_map(label_txts: List[str]) -> Dict[str, int]:
    """Merge label files -> id -> chemotype."""
    merged: Dict[str, Set[str]] = {}
    for p in label_txts:
        part = parse_label_txt(p)
        for sid, s in part.items():
            if sid not in merged:
                merged[sid] = set(s)
            else:
                merged[sid].update(s)

    return {sid: chemotype_from_residue_set(res_set) for sid, res_set in merged.items()}


def write_latex_table_like_ec(df_overall: pd.DataFrame, out_tex: str, caption: str, label: str) -> None:
    """Write LaTeX table in the same style as EC numbers (counts+percents+weights)."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\adjustbox{max width=\columnwidth}{%")
    lines.append(r"\begin{tabular}{|l|r|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        r"Chemotype & \# chains & \makecell[l]{Chains\\(\% of dataset)} & Total weight & "
        r"\makecell[l]{Weight\\(\% of total $W_\mathrm{Structure}$)} \\"
    )
    lines.append(r"\hline")

    for _, r in df_overall.iterrows():
        name = r["chemotype_name"]
        n = int(r["n_chains"])
        p_ch = float(r["chains_pct"])
        w = float(r["weight_sum"])
        p_w = float(r["weight_pct"])
        lines.append(f"{name} & {n} & {p_ch:.1f} & {w:.2f} & {p_w:.1f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


import numpy as np
import matplotlib.pyplot as plt

def plot_top_accessions_by_pdb_count(acc_to_n: dict,
                                    top_k: int,
                                    out_pdf: str,
                                    title: str = "Top accessions by #PDB structures"):
    """
    acc_to_n: dict {accession (str): n_pdb (int)}
    """
    # English comments as requested
    items = [(str(k), int(v)) for k, v in acc_to_n.items() if v is not None]
    items = [(k, v) for k, v in items if v >= 0]
    items.sort(key=lambda x: x[1], reverse=True)

    top = items[:top_k]
    if len(top) == 0:
        raise ValueError("No accessions with non-negative PDB counts found.")

    labels = [k for k, _ in top][::-1]   # reverse for horizontal bar
    values = [v for _, v in top][::-1]

    fig_h = max(3.2, 0.35 * len(top) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.barh(np.arange(len(values)), values)

    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("# PDB structures")
    ax.set_title(title)

    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate counts at the end of bars
    for i, v in enumerate(values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, format="pdf")
    plt.close(fig)


# ------------------------- Main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--weight_col", type=str, default="W_Structure")

    # Optional inputs
    ap.add_argument("--protein_json", type=str, default=None,
                    help="Path to all_protein_table_modified.json (for #structures per accession).")
    ap.add_argument("--chemotype_label_txts", type=str, nargs="*", default=None,
                    help="Paths to split*.txt label files (for chemotypes from TRUE labels).")

    # Plot options
    ap.add_argument("--bins_components", type=int, default=50)
    ap.add_argument("--bins_structures", type=int, default=30)
    ap.add_argument("--chemotype_palette", type=str, default="okabe_ito",
                    help="Palette for 8 chemotypes: okabe_ito, tab10, Set2, Dark2, Paired, ...")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # ---------- Load dataset.csv ----------
    df = pd.read_csv(args.dataset_csv)
    need = {"Sequence_ID", "Set_Type", "Component_ID", args.weight_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"dataset_csv missing columns: {sorted(missing)}")

    df = df.drop_duplicates(subset=["Sequence_ID"]).copy()
    df["Sequence_ID"] = df["Sequence_ID"].astype(str)
    df["Set_Type"] = df["Set_Type"].astype(str)
    df["Component_ID"] = df["Component_ID"].astype(int)
    df[args.weight_col] = df[args.weight_col].astype(float)

    split_order = sorted(df["Set_Type"].unique().tolist(), key=split_sort_key)

    # ---------- (1) Connected components size histogram ----------
    comp = (
        df.groupby("Component_ID")
        .agg(n_chains=("Sequence_ID", "count"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
        .sort_values("n_chains", ascending=False)
        .reset_index(drop=True)
    )
    comp.to_csv(os.path.join(args.out_dir, "connected_components_summary.csv"), index=False)

    plot_hist(
        values=comp["n_chains"].to_numpy(dtype=int),
        bins=args.bins_components,
        xlabel="Component size (# chains)",
        ylabel="Number of components",
        title="Connected component sizes",
        out_pdf=os.path.join(args.out_dir, "connected_components_size_hist.pdf"),
    )

    sizes = comp["n_chains"].to_numpy(dtype=int)

    plot_component_sizes_zoom(sizes, xmax=200, bins=40,
                            out_pdf=os.path.join(args.out_dir, "cc_sizes_zoom_le200.pdf"))
    plot_top_components(sizes, top_k=10,
                        out_pdf=os.path.join(args.out_dir, "cc_sizes_top10.pdf"))


    # ---------- (2) PDB / AFDB per split (100% stacked) ----------
    df["Source"] = df["Sequence_ID"].apply(infer_source_from_sequence_id)
    src_long = (
        df.groupby(["Set_Type", "Source"])
        .agg(n_chains=("Sequence_ID", "count"), weight_sum=(args.weight_col, "sum"))
        .reset_index()
        .sort_values(["Set_Type", "Source"])
        .reset_index(drop=True)
    )
    src_long.to_csv(os.path.join(args.out_dir, "pdb_afdb_by_split_long.csv"), index=False)

    # Two-group colors: PDB + AFDB/UniProt
    colors_2 = ["#0072B2", "#999999"]  # blue + gray (reads well)
    plot_stacked_bar_2groups_norm(
        df_long=src_long,
        split_order=split_order,
        value_col="n_chains",
        group_col="Source",
        groups=["PDB", "AFDB/UniProt"],
        title="PDB vs AFDB/UniProt fraction across splits (by chain count)",
        ylabel="Fraction of split",
        out_pdf=os.path.join(args.out_dir, "pdb_afdb_fraction_by_split_counts_norm.pdf"),
        colors=colors_2,
    )
    plot_stacked_bar_2groups_norm(
        df_long=src_long,
        split_order=split_order,
        value_col="weight_sum",
        group_col="Source",
        groups=["PDB", "AFDB/UniProt"],
        title=f"PDB vs AFDB/UniProt fraction across splits (by {args.weight_col})",
        ylabel="Fraction of split",
        out_pdf=os.path.join(args.out_dir, "pdb_afdb_fraction_by_split_weight_norm.pdf"),
        colors=colors_2,
    )

    # ---------- (3) #structures per accession histogram (from JSON) ----------
    if args.protein_json:
        with open(args.protein_json, "r") as f:
            data = json.load(f)

        n_struct_list = []
        for acc, rec in data.items():
            pdb_ids = data.get(acc, {}).get("pdb_ids", [])
            #print(acc, len(pdb_ids))
            if pdb_ids is None:
                pdb_ids = []
            n_struct_list.append(int(len(pdb_ids)))

        n_struct = np.array(n_struct_list, dtype=int)
        pd.DataFrame({"n_structures": n_struct}).to_csv(
            os.path.join(args.out_dir, "structures_per_accession_values.csv"), index=False
        )

        # plot_structures_per_accession_capped(
        #     n_struct=n_struct,
        #     cap=10,
        #     out_pdf=os.path.join(args.out_dir, "structures_per_accession_hist_capped.pdf"),
        # )
        #print(n_struct)
        plot_has_pdb_bar(
            n_struct, 
            os.path.join(args.out_dir, "has_pdb_bar.pdf"
            ))
        plot_structures_per_accession_positive_only(
            n_struct, cap=20, 
            out_pdf=os.path.join(args.out_dir, "structures_per_accession_ge1.pdf"))

        acc_to_n = {}
        for acc, rec in data.items():           # here acc is Uniprot accession key in JSON
            pdb_ids = data.get(acc, {}).get("pdb_ids", [])
            acc_to_n[acc] = len(pdb_ids)

        dataset_acc = set(df["Sequence_ID"].str.split("_").str[0].tolist())
        acc_to_n = {acc: n for acc, n in acc_to_n.items() if acc in dataset_acc}


        plot_top_accessions_by_pdb_count(acc_to_n, top_k=10,
                                        out_pdf=os.path.join(args.out_dir, "top10_accessions_by_pdb.pdf"))

        # Quick summary (useful for text)
        summary = {
            "n_accessions": int(n_struct.size),
            "fraction_with_0": float((n_struct == 0).mean()),
            "fraction_with_1": float((n_struct == 1).mean()),
            "fraction_with_ge_2": float((n_struct >= 2).mean()),
            "max_structures": int(n_struct.max()) if n_struct.size > 0 else 0,
        }
        with open(os.path.join(args.out_dir, "structures_per_accession_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ---------- (4) Chemotype overall table + stacked by split (optional) ----------
    if args.chemotype_label_txts:
        chem_map = build_chemotype_map(args.chemotype_label_txts)

        df_ch = df.copy()
        df_ch["chemotype"] = df_ch["Sequence_ID"].map(chem_map)
        missing = int(df_ch["chemotype"].isna().sum())
        if missing > 0:
            print(f"[WARN] chemotype: missing label entries for {missing} proteins; they will be dropped.")
        df_ch = df_ch.dropna(subset=["chemotype"]).copy()
        df_ch["chemotype"] = df_ch["chemotype"].astype(int)

        total_n = int(df_ch.shape[0])
        total_w = float(df_ch[args.weight_col].sum())

        # Overall (like EC numbers)
        overall = (
            df_ch.groupby("chemotype")
            .agg(n_chains=("Sequence_ID", "count"), weight_sum=(args.weight_col, "sum"))
            .reset_index()
            .sort_values("chemotype")
            .reset_index(drop=True)
        )
        overall["chains_pct"] = 100.0 * overall["n_chains"] / max(total_n, 1)
        overall["weight_pct"] = 100.0 * overall["weight_sum"] / max(total_w, 1e-12)
        overall["chemotype_name"] = overall["chemotype"].map(CHEMOTYPE_LABELS)

        overall.to_csv(os.path.join(args.out_dir, "chemotype_overall.csv"), index=False)

        write_latex_table_like_ec(
            df_overall=overall,
            out_tex=os.path.join(args.out_dir, "chemotype_overall_table.tex"),
            caption="Distribution of catalytic chemotypes.",
            label="tab:chemotype_overall",
        )

        # By split (long) + normalized stacked by weight
        by_split = (
            df_ch.groupby(["Set_Type", "chemotype"])
            .agg(weight_sum=(args.weight_col, "sum"), n_chains=("Sequence_ID", "count"))
            .reset_index()
            .sort_values(["Set_Type", "chemotype"])
            .reset_index(drop=True)
        )
        by_split.to_csv(os.path.join(args.out_dir, "chemotype_by_split_long.csv"), index=False)

        plot_stacked_bar_multi_norm(
            df_long=by_split,
            split_order=split_order,
            class_order=list(range(0, 8)),
            value_col="weight_sum",
            title=f"Chemotype composition across splits (normalized by {args.weight_col})",
            ylabel="Fraction of split",
            out_pdf=os.path.join(args.out_dir, "chemotype_by_split_weight_norm.pdf"),
            labels=CHEMOTYPE_LABELS,
            palette=args.chemotype_palette,
        )

    print(f"[OK] Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
