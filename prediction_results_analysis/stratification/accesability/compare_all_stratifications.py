#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare 4 models across 4 stratifications:
  1. PDB vs AF (from metrics_by_model_source.csv)
  2. Amino acid type (from metrics_by_aa.csv) — heatmap + barplot
  3. EC top-level class + chemotype (from residue_table.csv + dataset_csv)
  4. PR curves per RSA bin (from residue_table.csv)

Usage:
  python compare_all_stratifications.py \
      --search_dir <stratification_results_dir> \
      --dataset_csv <path>/dataset.csv \
      --out_dir <out>
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import precision_recall_curve, auc

# ── constants ─────────────────────────────────────────────────────────────────

RSA_BIN_ORDER = [
    "buried(<=0.05)", "partly_buried(0.05-0.2)",
    "intermediate(0.2-0.5)", "exposed(>0.5)",
]
RSA_BIN_SHORT = ["buried\n≤0.05", "partly buried\n0.05–0.2",
                 "intermediate\n0.2–0.5", "exposed\n>0.5"]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

CHEMOTYPE_RULE = [
    (0, "0 (ILMVWF)", set("ILMVWF")),
    (1, "1 (AGP)",   set("AGP")),
    (2, "2 (QN)",    set("QN")),
    (3, "3 (KR)",    set("KR")),
    (4, "4 (S)",     {"S"}),
    (5, "5 (T)",     {"T"}),
    (6, "6 (DE)",    set("DE")),
    (7, "7 (other)", set()),
]

def get_chemotype(cat_aas):
    s = set(cat_aas)
    for idx, label, residues in CHEMOTYPE_RULE:
        if residues and s & residues:
            return label
    return "7 (other)"

# ── helpers ───────────────────────────────────────────────────────────────────

def aucpr(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    mask = ~(np.isnan(p) | np.isinf(p))
    y, p = y[mask], p[mask]
    if y.sum() == 0 or len(y) == 0:
        return np.nan
    prec, rec, _ = precision_recall_curve(y, p)
    return float(auc(rec, prec))


def bootstrap_ci(y, p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(aucpr(y[s], p[s]))
    vals = np.array(vals)
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def load_residue_table(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    df["y_true"] = df["y_true"].astype(int)
    return df


def grouped_aucpr(df, group_col, group_order=None):
    rows = []
    groups = group_order if group_order else sorted(df[group_col].dropna().unique())
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) == 0:
            continue
        y = sub["y_true"].to_numpy()
        p = sub["y_pred"].to_numpy()
        a = aucpr(y, p)
        lo, hi = bootstrap_ci(y, p)
        rows.append({group_col: g, "AUCPR": a, "ci_lo": lo, "ci_hi": hi,
                     "n_residues": len(sub), "n_positive": int(y.sum())})
    return pd.DataFrame(rows)

# ── plot helpers ──────────────────────────────────────────────────────────────

def grouped_barplot(ax, x, model_names, vals_dict, ci_lo_dict, ci_hi_dict,
                    xlabels, colors, ylabel="AUCPR"):
    n_models = len(model_names)
    width = 0.8 / n_models
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = np.array([vals_dict[name].get(g, np.nan) for g in x], float)
        lo   = np.array([ci_lo_dict[name].get(g, np.nan) for g in x], float)
        hi   = np.array([ci_hi_dict[name].get(g, np.nan) for g in x], float)
        xpos = np.arange(len(x)) + (i - n_models/2 + 0.5) * width
        ax.bar(xpos, vals, width*0.9, label=name, color=color,
               edgecolor="black", linewidth=0.5)
        yerr = np.vstack([vals - lo, hi - vals])
        ax.errorbar(xpos, vals, yerr=yerr, fmt="none",
                    ecolor="black", capsize=2, elinewidth=0.7)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── analysis 1: PDB vs AF ─────────────────────────────────────────────────────

def analysis_pdb_af(model_dirs, model_names, colors, out_dir):
    print("\n[1] PDB vs AF")
    groups = ["AF", "PDB"]
    vals_dict  = {n: {} for n in model_names}
    lo_dict    = {n: {} for n in model_names}
    hi_dict    = {n: {} for n in model_names}

    rows_all = []
    for d, name in zip(model_dirs, model_names):
        path = os.path.join(d, "metrics_by_model_source.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            g = row["model_source"]
            vals_dict[name][g] = row["AUCPR"]
            lo_dict[name][g]   = row["AUCPR_ci_lo"]
            hi_dict[name][g]   = row["AUCPR_ci_hi"]
            rows_all.append({"model": name, "model_source": g,
                             "AUCPR": row["AUCPR"],
                             "ci_lo": row["AUCPR_ci_lo"],
                             "ci_hi": row["AUCPR_ci_hi"],
                             "n_residues": row["n_residues_total"],
                             "n_positive": row["n_positive_residues"]})
        print(f"  {name}: AF={vals_dict[name].get('AF', 'n/a'):.3f}  PDB={vals_dict[name].get('PDB', 'n/a'):.3f}")

    pd.DataFrame(rows_all).to_csv(os.path.join(out_dir, "1_pdb_af_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    grouped_barplot(ax, groups, model_names, vals_dict, lo_dict, hi_dict,
                    groups, colors)
    ax.set_title("AUCPR: PDB vs AlphaFold structures", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_pdb_af_comparison.png"), dpi=200)
    plt.close(fig)
    print("  saved 1_pdb_af_comparison.*")

# ── analysis 2: amino acid ────────────────────────────────────────────────────

def analysis_aa(model_dirs, model_names, colors, out_dir):
    print("\n[2] Amino acid type")
    all_data = {}
    for d, name in zip(model_dirs, model_names):
        path = os.path.join(d, "metrics_by_aa.csv")
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_csv(path).set_index("aa")
        all_data[name] = df

    # Heatmap: rows=AA, cols=models
    mat = pd.DataFrame({name: all_data[name]["AUCPR"] for name in model_names
                        if name in all_data}).reindex(AA_ORDER)
    mat.to_csv(os.path.join(out_dir, "2_aa_aucpr_matrix.csv"))

    fig, ax = plt.subplots(figsize=(len(model_names)*1.6 + 1, 8))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="AUCPR")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_yticklabels(AA_ORDER, fontsize=9)
    for i in range(len(AA_ORDER)):
        for j, name in enumerate(model_names):
            v = mat.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.8 else "white")
    ax.set_title("AUCPR per amino acid", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_aa_heatmap.png"), dpi=200)
    plt.close(fig)

    # Barplot for catalytically important AAs
    key_aas = ["C", "D", "E", "H", "K", "R", "S", "T", "Y"]
    vals_dict = {n: {} for n in model_names}
    lo_dict   = {n: {} for n in model_names}
    hi_dict   = {n: {} for n in model_names}
    for name in model_names:
        if name not in all_data:
            continue
        df = all_data[name]
        for aa in key_aas:
            if aa in df.index:
                vals_dict[name][aa] = df.loc[aa, "AUCPR"]
                lo_dict[name][aa]   = df.loc[aa, "AUCPR_ci_lo"]
                hi_dict[name][aa]   = df.loc[aa, "AUCPR_ci_hi"]

    fig, ax = plt.subplots(figsize=(10, 4))
    grouped_barplot(ax, key_aas, model_names, vals_dict, lo_dict, hi_dict,
                    key_aas, colors)
    ax.set_title("AUCPR for catalytically important amino acids", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_aa_key_barplot.png"), dpi=200)
    plt.close(fig)
    print("  saved 2_aa_heatmap.png, 2_aa_key_barplot.png, 2_aa_aucpr_matrix.csv")

# ── analysis 3: EC class + chemotype ─────────────────────────────────────────

def analysis_ec_chemotype(model_dirs, model_names, colors, dataset_csv, out_dir):
    print("\n[3] EC class + chemotype")
    if dataset_csv is None or not os.path.exists(dataset_csv):
        print("  [SKIP] --dataset_csv not provided or not found")
        return

    ds = pd.read_csv(dataset_csv, low_memory=False)
    # normalize Sequence_ID
    if "Sequence_ID" not in ds.columns:
        print("  [SKIP] no Sequence_ID column in dataset_csv")
        return
    ec_map = ds.set_index("Sequence_ID")["EC_number"].to_dict() if "EC_number" in ds.columns else {}

    ec_rows, chem_rows = [], []

    for d, name in zip(model_dirs, model_names):
        rt_path = os.path.join(d, "residue_table.csv")
        if not os.path.exists(rt_path):
            print(f"  [WARN] missing residue_table: {rt_path}")
            continue
        df = load_residue_table(rt_path)

        # --- EC top-level ---
        df["ec_number"] = df["Sequence_ID"].map(ec_map)
        df["ec_top"] = df["ec_number"].apply(
            lambda x: str(int(float(str(x).split(".")[0])))
            if pd.notna(x) and str(x)[0].isdigit() else None
        )
        for ec in sorted(df["ec_top"].dropna().unique()):
            sub = df[df["ec_top"] == ec]
            y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
            a = aucpr(y, p)
            lo, hi = bootstrap_ci(y, p)
            ec_rows.append({"model": name, "ec_top": ec, "AUCPR": a,
                            "ci_lo": lo, "ci_hi": hi,
                            "n_residues": len(sub), "n_positive": int(y.sum())})

        # --- Chemotype (from catalytic residues per chain) ---
        cat_per_chain = (df[df["y_true"] == 1]
                         .groupby("Sequence_ID")["aa"]
                         .apply(list))
        chain_chem = cat_per_chain.apply(get_chemotype).rename("chemotype")
        df = df.join(chain_chem, on="Sequence_ID")
        for chem in sorted(df["chemotype"].dropna().unique()):
            sub = df[df["chemotype"] == chem]
            y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
            a = aucpr(y, p)
            lo, hi = bootstrap_ci(y, p)
            chem_rows.append({"model": name, "chemotype": chem, "AUCPR": a,
                              "ci_lo": lo, "ci_hi": hi,
                              "n_residues": len(sub), "n_positive": int(y.sum())})

    if not ec_rows:
        print("  [WARN] no EC data produced")
    else:
        ec_df = pd.DataFrame(ec_rows)
        ec_df.to_csv(os.path.join(out_dir, "3_ec_comparison.csv"), index=False)
        ec_classes = sorted(ec_df["ec_top"].unique())
        vals_d = {n: {} for n in model_names}
        lo_d   = {n: {} for n in model_names}
        hi_d   = {n: {} for n in model_names}
        for _, row in ec_df.iterrows():
            vals_d[row["model"]][row["ec_top"]] = row["AUCPR"]
            lo_d[row["model"]][row["ec_top"]]   = row["ci_lo"]
            hi_d[row["model"]][row["ec_top"]]   = row["ci_hi"]
        fig, ax = plt.subplots(figsize=(max(8, len(ec_classes)*1.5), 4))
        grouped_barplot(ax, ec_classes, model_names, vals_d, lo_d, hi_d,
                        [f"EC {e}" for e in ec_classes], colors)
        ax.set_title("AUCPR by EC top-level class", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "3_ec_comparison.png"), dpi=200)
        plt.close(fig)
        print("  saved 3_ec_comparison.*")

    if not chem_rows:
        print("  [WARN] no chemotype data produced")
    else:
        chem_df = pd.DataFrame(chem_rows)
        chem_df.to_csv(os.path.join(out_dir, "3_chemotype_comparison.csv"), index=False)
        chems = sorted(chem_df["chemotype"].unique())
        vals_d = {n: {} for n in model_names}
        lo_d   = {n: {} for n in model_names}
        hi_d   = {n: {} for n in model_names}
        for _, row in chem_df.iterrows():
            vals_d[row["model"]][row["chemotype"]] = row["AUCPR"]
            lo_d[row["model"]][row["chemotype"]]   = row["ci_lo"]
            hi_d[row["model"]][row["chemotype"]]   = row["ci_hi"]
        fig, ax = plt.subplots(figsize=(max(9, len(chems)*1.8), 4))
        grouped_barplot(ax, chems, model_names, vals_d, lo_d, hi_d,
                        chems, colors)
        ax.set_title("AUCPR by chemotype (catalytic residue chemistry)", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "3_chemotype_comparison.png"), dpi=200)
        plt.close(fig)
        print("  saved 3_chemotype_comparison.*")

# ── analysis 4: PR curves per RSA bin ────────────────────────────────────────

def analysis_pr_curves(model_dirs, model_names, colors, out_dir):
    print("\n[4] PR curves per RSA bin")
    fig, axes = plt.subplots(1, len(RSA_BIN_ORDER),
                             figsize=(4.5 * len(RSA_BIN_ORDER), 4.5))

    for ax, rsa_bin, short_label in zip(axes, RSA_BIN_ORDER, RSA_BIN_SHORT):
        for (d, name, color) in zip(model_dirs, model_names, colors):
            rt_path = os.path.join(d, "residue_table.csv")
            if not os.path.exists(rt_path):
                continue
            df = load_residue_table(rt_path)
            sub = df[df["rsa_bin"] == rsa_bin]
            if len(sub) == 0 or sub["y_true"].sum() == 0:
                continue
            y = sub["y_true"].to_numpy()
            p = sub["y_pred"].to_numpy()
            mask = ~(np.isnan(p) | np.isinf(p))
            prec, rec, _ = precision_recall_curve(y[mask], p[mask])
            a = auc(rec, prec)
            ax.plot(rec, prec, label=f"{name} ({a:.3f})", color=color,
                    linewidth=1.4, alpha=0.9)

        baseline = sub["y_true"].mean() if len(sub) > 0 else 0
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
    out_png = os.path.join(out_dir, "4_pr_curves_by_rsa_bin.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_dir", required=True,
                    help="Dir containing per-model stratification subdirs")
    ap.add_argument("--dataset_csv", default=None,
                    help="dataset.csv with Sequence_ID and EC_number (for analysis 3)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_names", nargs="+", default=None,
                    help="Override display names (same order as auto-discovered dirs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    entries = sorted([
        e for e in os.scandir(args.search_dir)
        if e.is_dir() and os.path.exists(os.path.join(e.path, "residue_table.csv"))
    ], key=lambda e: e.name)

    model_dirs  = [e.path for e in entries]
    model_names = args.model_names if args.model_names else [e.name for e in entries]

    print(f"Models found: {model_names}")
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))

    analysis_pdb_af(model_dirs, model_names, colors, args.out_dir)
    analysis_aa(model_dirs, model_names, colors, args.out_dir)
    analysis_ec_chemotype(model_dirs, model_names, colors, args.dataset_csv, args.out_dir)
    analysis_pr_curves(model_dirs, model_names, colors, args.out_dir)

    print(f"\n[DONE] all outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
