#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified evaluation for ScanNet catalytic site prediction.

- Stratify by EC top-level (first digit of EC_number)
- Stratify by chemotype, defined from *true catalytic residues* amino-acid identities

Inputs:
  --pkl         path to test_results.pkl (or merged test pickle with same keys)
  --dataset_csv path to dataset.csv containing Sequence_ID and EC_number
  --split_txts  one or more split*.txt label files (to recover sequences)
Outputs:
  - metrics_by_ec_top.csv, metrics_by_chemotype.csv
  - barplot_mp5_by_ec_top.pdf, barplot_mp5_by_chemotype.pdf
"""

import argparse
import os
import re
import pickle
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc


# ----------------- Your chemotype rule -----------------

def get_catalytic_class(residues: List[str]) -> int:
    # add more classes?
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


# ----------------- Helpers -----------------

REQUIRED_KEYS = ["labels", "predictions", "weights", "ids", "splits"]

def normalize_id(x) -> str:
    # In your training: ids are Sequence_ID strings like "A0A1L8G2K9_A"
    # In some other scripts: ids could be [pdb_id, chain_id]
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, str) for t in x):
            # best-effort join
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
    assert n == len(data["predictions"]) == len(data["weights"]) == len(data["ids"]) == len(data["splits"]), \
        f"{path}: length mismatch"
    return data

def parse_ec_top(ec_number: Any) -> Optional[int]:
    if ec_number is None or (isinstance(ec_number, float) and np.isnan(ec_number)):
        return None
    s = str(ec_number).strip()
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None

def parse_split_txt(path: str) -> Dict[str, str]:
    """
    Parses split*.txt in your format:
      >SEQID
      A 1 M 0
      A 2 G 0
      ...
    Returns: {SEQID: "MG..."} (sequence only, labels ignored here)
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
                cur_id = line[1:].strip()
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
        # later files should not override, but in normal CV they shouldn't overlap
        for k, v in part.items():
            if k not in seq_map:
                seq_map[k] = v
    return seq_map

def _ensure_1d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("labels/preds must be 1D per chain")
    return a

def maxprecision_at_k(labels, preds, weights, k: int) -> float:
    """Weighted per-chain MaxPrecision@k (same logic as your f_at_k)."""
    num = 0.0
    den = 0.0
    for y, p, w in zip(labels, preds, weights):
        y = _ensure_1d(y)
        p = _ensure_1d(p)
        if len(y) != len(p):
            L = min(len(y), len(p))
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

def aucpr_weighted(labels, preds, weights) -> float:
    """Weighted AUCPR: flatten residues and repeat chain weight across positions."""
    y_list = []
    p_list = []
    w_list = []

    for y, p, w in zip(labels, preds, weights):
        y = _ensure_1d(y).astype(np.float32)
        p = _ensure_1d(p).astype(np.float32)
        L = min(len(y), len(p))
        y = y[:L]
        p = p[:L]
        y_list.append(y)
        p_list.append(p)
        w_list.append(np.full(L, float(w), dtype=np.float32))

    if len(y_list) == 0:
        return 0.0

    y_flat = np.concatenate(y_list)
    p_flat = np.concatenate(p_list)
    w_flat = np.concatenate(w_list)

    if float(y_flat.sum()) == 0.0:
        # no positives in this group
        return 0.0

    bad = np.isnan(p_flat) | np.isnan(y_flat) | np.isinf(p_flat)
    if bad.any():
        p_flat[bad] = np.nanmedian(p_flat[~bad]) if (~bad).any() else 0.0
        y_flat[bad] = 0.0
        w_flat[bad] = 0.0

    pr, rc, _ = precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)
    return float(auc(rc, pr))

def barplot(values: List[float], labels: List[str], ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(10, 4))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="pdf")
    plt.close()


# ----------------- Main stratified evaluation -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Path to test_results.pkl (or merged test pickle).")
    ap.add_argument("--dataset_csv", type=str, required=True, help="Path to dataset.csv with Sequence_ID, EC_number.")
    ap.add_argument("--split_txts", type=str, nargs="+", required=True,
                    help="Paths to split*.txt label files (used to recover sequences).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--k", type=int, default=5, help="k for MaxPrecision@k barplots (default: 5).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load predictions
    data = load_results_pkl(args.pkl)
    ids = [normalize_id(x) for x in data["ids"]]
    labels = [np.asarray(x) for x in data["labels"]]
    preds = [np.asarray(x) for x in data["predictions"]]
    weights = np.asarray(data["weights"], dtype=float)

    # Load EC map
    df = pd.read_csv(args.dataset_csv).drop_duplicates(subset=["Sequence_ID"])
    ec_map = df.set_index("Sequence_ID")["EC_number"].to_dict()

    # Load sequences from split txts (needed for chemotype)
    seq_map = build_sequence_map(args.split_txts)

    # Assign group labels per chain
    ec_top = []
    chemotype = []
    n_pos = []

    missing_seq = 0
    missing_ec = 0

    for _id, y in zip(ids, labels):
        y = _ensure_1d(y).astype(int)
        n_pos.append(int(y.sum()))

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
            L = min(len(seq), len(y))
            pos_idx = np.where(y[:L] == 1)[0]
            residues = [seq[i] for i in pos_idx]
            chemotype.append(get_catalytic_class(residues) if len(residues) > 0 else 7)

    print(f"[INFO] N={len(ids)}")
    print(f"[INFO] missing EC_number: {missing_ec}")
    print(f"[INFO] missing sequences from split_txts: {missing_seq}")

    # ---------- Stratify by EC top-level (1..7) ----------
    ec_levels = list(range(1, 8))
    rows_ec = []
    mp5_ec = []
    labels_ec = []

    for lvl in ec_levels:
        mask = [e == lvl for e in ec_top]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            rows_ec.append({"EC_top": f"EC{lvl}", "n_chains": 0, "n_pos_res": 0, f"MP@{args.k}": 0.0, "AUCPR": 0.0})
            mp5_ec.append(0.0)
            labels_ec.append(f"EC{lvl}")
            continue

        lab_g = [labels[i] for i in idx]
        prd_g = [preds[i] for i in idx]
        w_g = weights[idx]

        mp = maxprecision_at_k(lab_g, prd_g, w_g, k=args.k)
        au = aucpr_weighted(lab_g, prd_g, w_g)
        rows_ec.append({
            "EC_top": f"EC{lvl}",
            "n_chains": int(len(idx)),
            "n_pos_res": int(np.sum([n_pos[i] for i in idx])),
            f"MP@{args.k}": float(mp),
            "AUCPR": float(au),
        })
        mp5_ec.append(float(mp))
        labels_ec.append(f"EC{lvl}")

    df_ec = pd.DataFrame(rows_ec)
    df_ec.to_csv(os.path.join(args.out_dir, "metrics_by_ec_top.csv"), index=False)

    barplot(
        values=mp5_ec,
        labels=labels_ec,
        ylabel=f"MaxPrecision@{args.k}",
        title=f"MaxPrecision@{args.k} by EC top-level (test)",
        out_path=os.path.join(args.out_dir, "barplot_mp5_by_ec_top.pdf"),
    )

    # ---------- Stratify by chemotype (0..7) ----------
    chem_levels = list(range(0, 8))
    rows_ch = []
    mp5_ch = []
    labels_ch = []

    for c in chem_levels:
        mask = [t == c for t in chemotype]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            rows_ch.append({"chemotype": c, "n_chains": 0, "n_pos_res": 0, f"MP@{args.k}": 0.0, "AUCPR": 0.0})
            mp5_ch.append(0.0)
            labels_ch.append(str(c))
            continue

        lab_g = [labels[i] for i in idx]
        prd_g = [preds[i] for i in idx]
        w_g = weights[idx]

        mp = maxprecision_at_k(lab_g, prd_g, w_g, k=args.k)
        au = aucpr_weighted(lab_g, prd_g, w_g)
        rows_ch.append({
            "chemotype": int(c),
            "n_chains": int(len(idx)),
            "n_pos_res": int(np.sum([n_pos[i] for i in idx])),
            f"MP@{args.k}": float(mp),
            "AUCPR": float(au),
        })
        mp5_ch.append(float(mp))
        labels_ch.append(str(c))

    df_ch = pd.DataFrame(rows_ch)
    df_ch.to_csv(os.path.join(args.out_dir, "metrics_by_chemotype.csv"), index=False)

    barplot(
        values=mp5_ch,
        labels=labels_ch,
        ylabel=f"MaxPrecision@{args.k}",
        title=f"MaxPrecision@{args.k} by chemotype (test)",
        out_path=os.path.join(args.out_dir, "barplot_mp5_by_chemotype.pdf"),
    )

    print(f"[OK] wrote to: {args.out_dir}")
    print(f"[OK] metrics_by_ec_top.csv, metrics_by_chemotype.csv")
    print(f"[OK] barplot_mp5_by_ec_top.pdf, barplot_mp5_by_chemotype.pdf")


if __name__ == "__main__":
    main()
