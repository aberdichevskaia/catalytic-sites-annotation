#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified evaluation for ScanNet catalytic site prediction.

Stratifications:
- EC top-level: first digit of EC_number (1..7)
- Chemotype: from *true catalytic residues* amino-acid identities (derived via split*.txt sequences)

Inputs:
  --pkl         path to test_results.pkl (or merged test pickle with same keys)
  --dataset_csv path to dataset.csv containing Sequence_ID and EC_number
  --split_txts  one or more split*.txt label files (to recover sequences)
  --out_dir     output directory

Outputs (in --out_dir):
  - metrics_by_ec_top.csv
  - metrics_by_chemotype.csv
  - barplot_aucpr_by_ec_top__with_ci_clean.svg
  - barplot_aucpr_by_chemotype__with_ci_clean.svg
  - per_chain_debug.csv
"""

import argparse
import os
import re
import pickle
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc


# ----------------- Plot style -----------------

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
    "svg.fonttype": "path",  # keep as "path" for paper consistency
})

# figures sizes (for the paper)
W_FULL = 6.27
GUTTER = 0.15
W_HALF = (W_FULL - GUTTER) / 2
W_THIRD = (W_FULL - GUTTER) / 3
W_TWO_THIRDS = 2 * (W_FULL - GUTTER) / 3
H_SMALL = 1.9
H_WIDE = 2.7


# ----------------- Chemotype rule -----------------

def get_catalytic_class(residues: List[str]) -> int:
    if any(r in residues for r in "ILMVWF"):
        return 0
    if any(r in residues for r in "AGP"):
        return 1
    if any(r in residues for r in "QN"):
        return 2
    if any(r in residues for r in "KR"):
        return 3
    if any(r == "S" for r in residues):
        return 4
    if any(r == "T" for r in residues):
        return 5
    if any(r in residues for r in "DE"):
        return 6
    return 7


CHEMOTYPE_NAMES = {
    0: "0 (ILMVWF)",
    1: "1 (AGP)",
    2: "2 (QN)",
    3: "3 (KR)",
    4: "4 (S)",
    5: "5 (T)",
    6: "6 (DE)",
    7: "7 (other/none)",
}


# ----------------- Helpers -----------------

REQUIRED_KEYS = ["labels", "predictions", "weights", "ids", "splits"]


def normalize_id(x) -> str:
    # ids can be strings like "A0A1L8G2K9_A" or [pdb_id, chain_id]
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, str) for t in x):
            return f"{x[0]}_{x[1]}" if not str(x[0]).endswith(f"_{x[1]}") else str(x[0])
        return str(x[0])
    return str(x)


def load_results_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for k in REQUIRED_KEYS:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    n = len(data["labels"])
    if not (n == len(data["predictions"]) == len(data["weights"]) == len(data["ids"]) == len(data["splits"])):
        raise ValueError(f"{path}: length mismatch")
    return data


def parse_ec_top(ec_number: Any) -> Optional[int]:
    if ec_number is None or (isinstance(ec_number, float) and np.isnan(ec_number)):
        return None
    s = str(ec_number).strip()
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None


def _ensure_1d_labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1D per chain, got shape={y.shape}")
    return y


def _ensure_1d_preds(p: Any) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim > 1:
        # mimic your PR-curve behavior (take last channel)
        p = p[..., -1]
    if p.ndim != 1:
        raise ValueError(f"predictions must be 1D per chain (after squeeze), got shape={p.shape}")
    return p


def parse_split_txt(path: str) -> Dict[str, str]:
    """
    Parses split*.txt in your format:
      >SEQID
      A 1 M 0
      A 2 G 0
      ...
    Returns: {SEQID: "MG..."} (sequence only)
    """
    seq_map: Dict[str, str] = {}
    cur_id = None
    residues: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if cur_id is not None:
                    seq_map[cur_id] = "".join(residues)
                cur_id = normalize_id(line[1:].strip())
                residues = []
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            aa = parts[2]
            residues.append(aa)

    if cur_id is not None:
        seq_map[cur_id] = "".join(residues)

    return seq_map


def build_sequence_map(split_txts: List[str]) -> Dict[str, str]:
    seq_map: Dict[str, str] = {}
    for p in split_txts:
        part = parse_split_txt(p)
        for k, v in part.items():
            if k not in seq_map:
                seq_map[k] = v
    return seq_map


# ----------------- MP@k (kept for context columns) -----------------

def mp_per_chain(y: np.ndarray, p: np.ndarray, k: int) -> float:
    y = _ensure_1d_labels(y).astype(int)
    p = _ensure_1d_preds(p).astype(float)

    L = min(len(y), len(p))
    if L <= 0:
        return 0.0
    y = y[:L]
    p = p[:L]

    k_eff = min(int(k), len(p))
    if k_eff <= 0:
        return 0.0

    top_idx = np.argpartition(p, -k_eff)[-k_eff:]
    denom = min(int(y.sum()), k_eff)
    return 0.0 if denom == 0 else float(y[top_idx].sum()) / denom


def maxprecision_at_k(labels, preds, weights, k: int) -> float:
    num = 0.0
    den = 0.0

    for y, p, w in zip(labels, preds, weights):
        y = _ensure_1d_labels(y).astype(int)
        p = _ensure_1d_preds(p).astype(float)

        L = min(len(y), len(p))
        if L <= 0:
            continue
        y = y[:L]
        p = p[:L]

        k_eff = min(int(k), len(p))
        if k_eff <= 0:
            continue

        top_idx = np.argpartition(p, -k_eff)[-k_eff:]
        denom = min(int(np.sum(y)), k_eff)
        mp_i = 0.0 if denom == 0 else float(np.sum(y[top_idx])) / denom

        num += mp_i * float(w)
        den += float(w)

    return 0.0 if den == 0.0 else num / den


# ----------------- Weighted stats (for context columns) -----------------

def weighted_frac(mask: np.ndarray, w: np.ndarray) -> float:
    mask = np.asarray(mask, bool)
    w = np.asarray(w, float)
    s = w.sum()
    return 0.0 if s == 0 else float(w[mask].sum() / s)


def n_eff(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    return 0.0 if s2 == 0 else float((s1 ** 2) / s2)


# ----------------- AUCPR: EXACTLY like make_PR_curve -----------------

def aucpr_like_make_PR_curve(labels, predictions, weights) -> float:
    """
    AUCPR in the style of your make_PR_curve:
    - repeat per-chain weights to residues
    - flatten across residues
    - if preds are multi-dim, take last channel (preds[..., -1])
    - ignore NaN/Inf (is_bad mask), fill preds_bad with nanmedian(preds_good)
    - compute precision_recall_curve with sample_weight, then auc(recall, precision)
    """
    y_list = []
    p_list = []
    w_list = []

    for lbl, pred, w in zip(labels, predictions, weights):
        lbl = _ensure_1d_labels(lbl)
        pred = _ensure_1d_preds(pred)

        L = min(len(lbl), len(pred))
        if L <= 0:
            continue

        lbl = lbl[:L]
        pred = pred[:L]

        y_list.append(lbl)
        p_list.append(pred)
        w_list.append(np.ones(L, dtype=float) * float(w))

    if len(y_list) == 0:
        return 0.0

    labels_flat = np.concatenate(y_list)
    preds_flat = np.concatenate(p_list)
    w_flat = np.concatenate(w_list)

    is_bad = (
        np.isnan(preds_flat) | np.isnan(labels_flat) |
        np.isinf(preds_flat) | np.isinf(labels_flat)
    )
    if is_bad.any():
        good = ~is_bad
        fill = np.nanmedian(preds_flat[good]) if good.any() else 0.0
        preds_flat = preds_flat.copy()
        preds_flat[is_bad] = fill

    good = ~is_bad
    if not good.any():
        return 0.0

    # if no positives in this group -> define AUCPR=0 for convenience
    if np.sum(labels_flat[good]) == 0:
        return 0.0

    precision, recall, _ = precision_recall_curve(
        labels_flat[good],
        preds_flat[good],
        sample_weight=w_flat[good]
    )
    return float(auc(recall, precision))


def bootstrap_ci_aucpr_group(labels, predictions, weights,
                            n_boot: int, seed: int,
                            progress_every: int = 0,
                            tag: str = "") -> Tuple[float, float]:
    """
    Bootstrap CI for AUCPR computed on pooled residues (make_PR_curve style),
    by resampling proteins with replacement.
    """
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    vals = np.empty(n_boot, dtype=float)
    weights = np.asarray(weights)

    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        lab_s = [labels[i] for i in samp]
        prd_s = [predictions[i] for i in samp]
        w_s = weights[samp]
        vals[b] = aucpr_like_make_PR_curve(lab_s, prd_s, w_s)

        if progress_every and ((b + 1) % progress_every == 0):
            print(f"[BOOT] {tag} {b+1}/{n_boot}")

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


# ----------------- Plotting -----------------

def barplot_with_ci_clean(values, labels, ci_lo, ci_hi, ylabel, title, out_path, ylim=(0.0, 1.0)):
    fig, ax = plt.subplots(figsize=(W_FULL, H_SMALL))

    x = np.arange(len(values))
    ax.bar(x, values)

    v = np.asarray(values, float)
    lo = np.asarray(ci_lo, float)
    hi = np.asarray(ci_hi, float)

    bad = np.isnan(lo) | np.isnan(hi)
    lo2 = lo.copy()
    hi2 = hi.copy()
    lo2[bad] = v[bad]
    hi2[bad] = v[bad]

    yerr = np.vstack([v - lo2, hi2 - v])
    ax.errorbar(x, v, yerr=yerr, fmt="none", capsize=3, elinewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="svg")
    plt.close(fig)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Path to test_results.pkl (or merged test pickle).")
    ap.add_argument("--dataset_csv", type=str, required=True, help="Path to dataset.csv with Sequence_ID, EC_number.")
    ap.add_argument("--split_txts", type=str, nargs="+", required=True,
                    help="Paths to split*.txt label files (used to recover sequences).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--k", type=int, default=5, help="k for MP@k context columns (default: 5).")
    ap.add_argument("--n_boot", type=int, default=200,
                    help="Bootstrap samples for AUCPR CI (default: 200). WARNING: 1000 can be very slow.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap.")
    ap.add_argument("--progress_every", type=int, default=0,
                    help="Print bootstrap progress every N iterations (0 disables).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load predictions
    data = load_results_pkl(args.pkl)
    ids = [normalize_id(x) for x in data["ids"]]
    labels = [np.asarray(x) for x in data["labels"]]
    preds = [np.asarray(x) for x in data["predictions"]]
    weights = np.asarray(data["weights"], dtype=float)

    # Load EC map
    df = pd.read_csv(args.dataset_csv).drop_duplicates(subset=["Sequence_ID"]).copy()
    df["Sequence_ID"] = df["Sequence_ID"].map(normalize_id)
    ec_map = df.set_index("Sequence_ID")["EC_number"].to_dict()

    # Load sequences from split txts (needed for chemotype)
    seq_map = build_sequence_map(args.split_txts)

    # Assign group labels per chain
    ec_top: List[Optional[int]] = []
    chemotype: List[Optional[int]] = []
    n_pos: List[int] = []
    missing_seq = 0
    missing_ec = 0

    for _id, y in zip(ids, labels):
        y1 = _ensure_1d_labels(y).astype(int)
        npos_i = int(y1.sum())
        n_pos.append(npos_i)

        ec = ec_map.get(_id, None)
        if ec is None or (isinstance(ec, float) and np.isnan(ec)):
            missing_ec += 1
            ec_top.append(None)
        else:
            ec_top.append(parse_ec_top(ec))

        seq = seq_map.get(_id, None)
        if seq is None:
            missing_seq += 1
            chemotype.append(None)
        else:
            L = min(len(seq), len(y1))
            pos_idx = np.where(y1[:L] == 1)[0]
            residues = [seq[i] for i in pos_idx]
            chemotype.append(get_catalytic_class(residues) if len(residues) > 0 else 7)

    # Per-chain MP@k (for debug/context columns)
    mp_i = np.array([mp_per_chain(y, p, k=args.k) for y, p in zip(labels, preds)], dtype=float)

    print(f"[INFO] N={len(ids)}")
    print(f"[INFO] missing EC_number: {missing_ec}")
    print(f"[INFO] missing sequences from split_txts: {missing_seq}")
    print(f"[INFO] n_boot={args.n_boot} (AUCPR CI can be slow if large)")

    # Per-chain df (sanity checks)
    df_chain = pd.DataFrame({
        "Sequence_ID": ids,
        "w": weights,
        "ec_top": ec_top,
        "chemotype": chemotype,
        "npos": n_pos,
        f"mp@{args.k}": mp_i,
    })
    df_chain.to_csv(os.path.join(args.out_dir, "per_chain_debug.csv"), index=False)

    # ---------- Stratify by EC top-level (1..7) ----------
    ec_levels = list(range(1, 8))
    rows_ec = []
    labels_ec = []
    au_ec, au_ec_lo, au_ec_hi = [], [], []

    for lvl in ec_levels:
        mask = (df_chain["ec_top"] == lvl).to_numpy()
        idx = np.where(mask)[0]

        labels_ec.append(f"EC{lvl}")

        if len(idx) == 0:
            rows_ec.append({
                "EC_top": f"EC{lvl}",
                "n_chains": 0,
                "n_pos_res": 0,
                "weight_sum": 0.0,
                "n_eff": 0.0,
                "median_npos": np.nan,
                f"frac_npos_lt_{args.k}": np.nan,
                f"MP@{args.k}": 0.0,
                "AUCPR": 0.0,
                "AUCPR_ci_lo": np.nan,
                "AUCPR_ci_hi": np.nan,
            })
            au_ec.append(0.0)
            au_ec_lo.append(np.nan)
            au_ec_hi.append(np.nan)
            continue

        lab_g = [labels[i] for i in idx]
        prd_g = [preds[i] for i in idx]
        w_g = weights[idx]

        au = aucpr_like_make_PR_curve(lab_g, prd_g, w_g)
        lo, hi = bootstrap_ci_aucpr_group(
            lab_g, prd_g, w_g,
            n_boot=args.n_boot, seed=args.seed,
            progress_every=args.progress_every,
            tag=f"EC{lvl}"
        )

        # context columns
        mp = maxprecision_at_k(lab_g, prd_g, w_g, k=args.k)
        npos_vals = df_chain.iloc[idx]["npos"].to_numpy()
        med_npos = float(np.median(npos_vals))
        neff = n_eff(w_g)
        frac_lt_k = weighted_frac(npos_vals < args.k, w_g)

        rows_ec.append({
            "EC_top": f"EC{lvl}",
            "n_chains": int(len(idx)),
            "n_pos_res": int(np.sum(npos_vals)),
            "weight_sum": float(w_g.sum()),
            "n_eff": float(neff),
            "median_npos": float(med_npos),
            f"frac_npos_lt_{args.k}": float(frac_lt_k),
            f"MP@{args.k}": float(mp),
            "AUCPR": float(au),
            "AUCPR_ci_lo": float(lo),
            "AUCPR_ci_hi": float(hi),
        })

        au_ec.append(float(au))
        au_ec_lo.append(float(lo))
        au_ec_hi.append(float(hi))

    df_ec = pd.DataFrame(rows_ec)
    df_ec.to_csv(os.path.join(args.out_dir, "metrics_by_ec_top.csv"), index=False)

    barplot_with_ci_clean(
        values=au_ec,
        labels=labels_ec,
        ci_lo=au_ec_lo,
        ci_hi=au_ec_hi,
        ylabel="AUCPR",
        title="AUCPR by EC top-level (test)",
        out_path=os.path.join(args.out_dir, "barplot_aucpr_by_ec_top__with_ci_clean.svg"),
    )

    # ---------- Stratify by chemotype (0..7) ----------
    chem_levels = list(range(0, 8))
    rows_ch = []
    labels_ch = []
    au_ch, au_ch_lo, au_ch_hi = [], [], []

    for c in chem_levels:
        mask = (df_chain["chemotype"] == c).to_numpy()
        idx = np.where(mask)[0]

        labels_ch.append(CHEMOTYPE_NAMES[c])

        if len(idx) == 0:
            rows_ch.append({
                "chemotype": int(c),
                "n_chains": 0,
                "n_pos_res": 0,
                "weight_sum": 0.0,
                "n_eff": 0.0,
                "median_npos": np.nan,
                f"frac_npos_lt_{args.k}": np.nan,
                f"MP@{args.k}": 0.0,
                "AUCPR": 0.0,
                "AUCPR_ci_lo": np.nan,
                "AUCPR_ci_hi": np.nan,
            })
            au_ch.append(0.0)
            au_ch_lo.append(np.nan)
            au_ch_hi.append(np.nan)
            continue

        lab_g = [labels[i] for i in idx]
        prd_g = [preds[i] for i in idx]
        w_g = weights[idx]

        au = aucpr_like_make_PR_curve(lab_g, prd_g, w_g)
        lo, hi = bootstrap_ci_aucpr_group(
            lab_g, prd_g, w_g,
            n_boot=args.n_boot, seed=args.seed,
            progress_every=args.progress_every,
            tag=f"chem{c}"
        )

        # context columns
        mp = maxprecision_at_k(lab_g, prd_g, w_g, k=args.k)
        npos_vals = df_chain.iloc[idx]["npos"].to_numpy()
        med_npos = float(np.median(npos_vals))
        neff = n_eff(w_g)
        frac_lt_k = weighted_frac(npos_vals < args.k, w_g)

        rows_ch.append({
            "chemotype": int(c),
            "n_chains": int(len(idx)),
            "n_pos_res": int(np.sum(npos_vals)),
            "weight_sum": float(w_g.sum()),
            "n_eff": float(neff),
            "median_npos": float(med_npos),
            f"frac_npos_lt_{args.k}": float(frac_lt_k),
            f"MP@{args.k}": float(mp),
            "AUCPR": float(au),
            "AUCPR_ci_lo": float(lo),
            "AUCPR_ci_hi": float(hi),
        })

        au_ch.append(float(au))
        au_ch_lo.append(float(lo))
        au_ch_hi.append(float(hi))

    df_ch = pd.DataFrame(rows_ch)
    df_ch.to_csv(os.path.join(args.out_dir, "metrics_by_chemotype.csv"), index=False)

    barplot_with_ci_clean(
        values=au_ch,
        labels=labels_ch,
        ci_lo=au_ch_lo,
        ci_hi=au_ch_hi,
        ylabel="AUCPR",
        title="AUCPR by chemotype (test)",
        out_path=os.path.join(args.out_dir, "barplot_aucpr_by_chemotype__with_ci_clean.svg"),
    )

    print(f"[OK] wrote to: {args.out_dir}")
    print("[OK] metrics_by_ec_top.csv, metrics_by_chemotype.csv")
    print("[OK] barplot_aucpr_by_ec_top__with_ci_clean.svg, barplot_aucpr_by_chemotype__with_ci_clean.svg")
    print("[OK] per_chain_debug.csv (sanity checks)")


if __name__ == "__main__":
    main()
