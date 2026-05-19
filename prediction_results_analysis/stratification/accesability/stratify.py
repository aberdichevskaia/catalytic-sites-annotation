#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Residue-wise accessibility extraction + residue-wise stratification.

Input split format:
>Sequence_ID
A 1 M 0
A 2 G 0
A 3 D 0
...

Optionally also takes test_results.pkl with:
labels, predictions, weights, ids, splits

Outputs:
- residue_table.csv
- metrics_by_accessibility_bin.csv
- metrics_by_aa.csv
- metrics_by_model_source.csv

Fast RSA mode (--pipeline_folder):
  Reads RSA directly from pre-computed HandcraftedFeaturesPipeline cache files
  (catalytic_sites_weight_based_v9_splitN_MSA_pipeline_Handcrafted_features-*).
  These are the .data files built by train_handcrafter_features_catalytic_xgboost.py.
  For the ~2-3% of chains whose length doesn't match, falls back to on-the-fly DSSP.
  This eliminates the ~20k sequential DSSP subprocess calls that make the script slow.
"""

import argparse
import os
import re
import glob
import pickle
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

# # ---------------- ScanNet_Ub in sys.path ----------------
# PROJECT_ROOT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/"
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP


# ----------------------------
# Constants
# ----------------------------

AA_MAX_ACC = {
    # Common Rose/Miller-style max ASA values for RSA normalization
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0
}


# ----------------------------
# General helpers
# ----------------------------

def normalize_id(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, str) for t in x):
            return f"{x[0]}_{x[1]}"
        return str(x[0])
    return str(x)


def is_pdb_identifier(s: str) -> bool:
    return len(s) == 4 and s.isalnum()


def parse_model_source(seq_id: str) -> str:
    base = seq_id.split("_", 1)[0]
    return "PDB" if is_pdb_identifier(base) else "AF"


def ensure_1d_preds(p: Any) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim > 1:
        p = p[..., -1]
    if p.ndim != 1:
        raise ValueError(f"Predictions must be 1D after squeeze, got {p.shape}")
    return p


def ensure_1d_labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {y.shape}")
    return y


# ----------------------------
# Parsing split txt
# ----------------------------

def parse_split_txt(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      {
        "A0A1L8G2K9_A": [
            {"chain":"A", "resnum":"1", "aa":"M", "label":0},
            ...
        ],
        ...
      }
    """
    data = {}
    cur_id = None
    cur_rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if cur_id is not None:
                    data[cur_id] = cur_rows
                cur_id = normalize_id(line[1:].strip())
                cur_rows = []
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            chain, resnum, aa, label = parts[:4]
            cur_rows.append({
                "chain": chain,
                "resnum": str(resnum),
                "aa": aa,
                "label": int(label),
            })

    if cur_id is not None:
        data[cur_id] = cur_rows

    return data


def build_split_map(split_txts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for path in split_txts:
        part = parse_split_txt(path)
        for k, v in part.items():
            if k in out:
                raise ValueError(f"Duplicate Sequence_ID across split txts: {k}")
            out[k] = v
    return out


# ----------------------------
# Load model results pkl
# ----------------------------

def load_results_pkl(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    required = ["labels", "predictions", "weights", "ids"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")

    n = len(data["ids"])
    if not (n == len(data["labels"]) == len(data["predictions"]) == len(data["weights"])):
        raise ValueError("Length mismatch in results pkl")

    result_map = {}
    for seq_id, labels, preds, w in zip(data["ids"], data["labels"], data["predictions"], data["weights"]):
        sid = normalize_id(seq_id)
        y = ensure_1d_labels(labels)
        p = ensure_1d_preds(preds)
        result_map[sid] = {
            "labels": y,
            "preds": p,
            "weight": float(w),
        }
    return result_map


def load_results_pkls(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load and merge multiple pkl files.  When the same seq_id appears in
    more than one file (e.g. all 5 folds share a fixed test set), predictions
    are averaged across files (ensemble); labels and weight are taken from the
    first occurrence."""
    accum: Dict[str, Dict[str, Any]] = {}   # seq_id -> {labels, preds_list, weight}
    for path in paths:
        part = load_results_pkl(path)
        for sid, rec in part.items():
            if sid not in accum:
                accum[sid] = {"labels": rec["labels"], "preds_list": [rec["preds"]], "weight": rec["weight"]}
            else:
                accum[sid]["preds_list"].append(rec["preds"])
        print(f"[INFO] loaded {len(part)} chains from {path}")

    merged: Dict[str, Dict[str, Any]] = {}
    n_ensembled = 0
    for sid, rec in accum.items():
        if len(rec["preds_list"]) > 1:
            n_ensembled += 1
        merged[sid] = {
            "labels": rec["labels"],
            "preds": np.mean(rec["preds_list"], axis=0).astype(np.float32),
            "weight": rec["weight"],
        }

    print(f"[INFO] total predictions loaded: {len(merged)} chains "
          f"({n_ensembled} ensembled across {len(paths)} files)")
    return merged


# ----------------------------
# Structure file resolution
# ----------------------------

def resolve_structure_file(seq_id: str, structure_dir: str) -> Optional[str]:
    base = seq_id.split("_", 1)[0]

    candidates = [
        f"{seq_id}.cif",
        f"{seq_id}.pdb",
        f"{base}.cif",
        f"{base}.pdb",
        f"{base}.mmcif",
        f"{seq_id}.mmcif",
    ]

    for fname in candidates:
        path = os.path.join(structure_dir, fname)
        if os.path.exists(path):
            return path

    return None


# ----------------------------
# Pipeline cache RSA loader
# ----------------------------

# Column index of RSA in the MSA HandcraftedFeaturesPipeline cache (pscparvhc, 58 features total):
#   [0:20]  primary (one-hot AA)
#   [20:28] secondary structure (DSSP 8-class)
#   [28]    conservation score
#   [29]    RSA  <-- this one
#   [30:32] residue depth (backbone, sidechain)
#   [32]    half-sphere exposure
#   [33]    coordination number
#   [34:37] volume index
#   [37:58] PWM (21 AA + gaps)
_PIPELINE_RSA_COL_DEFAULT = 29

_PIPELINE_CACHE_PATTERN = (
    "catalytic_sites_weight_based_v9_{split_name}_MSA_"
    "pipeline_Handcrafted_features-pscparvhc_gaps-True_Beff-500.data"
)


def build_rsa_lookup_from_pipeline_cache(
    split_txt_paths: List[str],
    pipeline_folder: str,
    rsa_col: int = _PIPELINE_RSA_COL_DEFAULT,
) -> Dict[str, np.ndarray]:
    """
    Load per-residue RSA from pre-computed pipeline .data files.

    Returns {seq_id: rsa_array} for chains where the cache residue count
    matches the split file.  Chains with length mismatches (pipeline was built
    with biounit=True and may have extra residues) are silently omitted so
    the caller can fall back to on-the-fly DSSP for them.
    """
    rsa_lookup: Dict[str, np.ndarray] = {}

    for split_txt in split_txt_paths:
        split_name = os.path.splitext(os.path.basename(split_txt))[0]
        cache_filename = _PIPELINE_CACHE_PATTERN.format(split_name=split_name)
        cache_path = os.path.join(pipeline_folder, cache_filename)

        if not os.path.exists(cache_path):
            print(f"[INFO] pipeline cache not found: {cache_path}")
            continue

        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
        except Exception as exc:
            print(f"[WARN] failed to load pipeline cache {cache_path}: {exc}")
            continue

        inputs = cache["inputs"]
        failed_set = {int(x) for x in cache.get("failed_samples", [])}

        # Parse ordered seq_ids and per-chain residue counts from split file
        seq_ids: List[str] = []
        residue_counts: List[int] = []
        cur_id: Optional[str] = None
        cnt = 0
        with open(split_txt, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_id is not None:
                        residue_counts.append(cnt)
                    cur_id = normalize_id(line[1:].strip())
                    seq_ids.append(cur_id)
                    cnt = 0
                else:
                    cnt += 1
        if cur_id is not None:
            residue_counts.append(cnt)

        cache_idx = 0
        n_ok = 0
        n_mismatch = 0

        for orig_idx, (sid, split_n) in enumerate(zip(seq_ids, residue_counts)):
            if orig_idx in failed_set:
                continue

            arr = np.asarray(inputs[cache_idx])
            if arr.ndim == 2 and arr.shape[1] > rsa_col and arr.shape[0] == split_n:
                rsa_lookup[sid] = arr[:, rsa_col].astype(np.float32)
                n_ok += 1
            else:
                n_mismatch += 1

            cache_idx += 1

        print(
            f"[INFO] {split_name}: RSA from cache for {n_ok} chains, "
            f"{n_mismatch} length mismatches → DSSP fallback, "
            f"{len(failed_set)} failed in pipeline"
        )

    total = len(rsa_lookup)
    print(f"[INFO] RSA lookup ready: {total} chains total")
    return rsa_lookup


# ----------------------------
# DSSP / accessibility
# ----------------------------

def load_structure(structure_path: str, structure_id: str):
    if structure_path.endswith((".cif", ".mmcif")):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, structure_path)


def extract_dssp_accessibility(structure_path: str,
                               chain_id: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Returns mapping:
      (chain_id, residue_number_string) -> {"asa": ..., "rsa": ...}

    Important:
    DSSP keys are usually like (chain_id, (' ', resseq, icode))
    """
    structure = load_structure(structure_path, os.path.basename(structure_path))
    model = structure[0]

    dssp = DSSP(model, structure_path)

    out = {}
    for key in dssp.keys():
        dssp_chain = key[0]
        res_id = key[1]   # (' ', resseq, icode)
        hetflag, resseq, icode = res_id

        if dssp_chain != chain_id:
            continue
        if hetflag.strip():
            continue  # skip hetero atoms / ligands

        dssp_record = dssp[key]

        # Biopython DSSP tuple convention:
        # index 1 = amino acid
        # index 2 = secondary structure
        # index 3 = relative ASA (in many versions)
        aa = dssp_record[1]
        rsa = dssp_record[3]

        resnum_str = str(resseq) if str(icode).strip() == "" else f"{resseq}{icode.strip()}"

        # Reconstruct absolute ASA approximately from RSA * maxASA
        # If rsa is already missing/special -> NaN
        if aa in AA_MAX_ACC and rsa not in [None, "NA"]:
            try:
                rsa_val = float(rsa)
                asa_val = rsa_val * AA_MAX_ACC[aa]
            except Exception:
                rsa_val = np.nan
                asa_val = np.nan
        else:
            rsa_val = np.nan
            asa_val = np.nan

        out[(dssp_chain, resnum_str)] = {
            "asa": asa_val,
            "rsa": rsa_val,
        }

        # Also keep plain integer form without insertion code if useful
        out[(dssp_chain, str(resseq))] = {
            "asa": asa_val,
            "rsa": rsa_val,
        }

    return out


# ----------------------------
# Residue table builder
# ----------------------------

def build_residue_table(split_map: Dict[str, List[Dict[str, Any]]],
                        structure_dir: str,
                        results_map: Optional[Dict[str, Dict[str, Any]]] = None,
                        rsa_lookup: Optional[Dict[str, np.ndarray]] = None,
                        report_every: int = 50,
                        verbose_success: bool = False) -> pd.DataFrame:
    rows = []

    stats = {
        "total_proteins": 0,
        "structures_found": 0,
        "structures_missing": 0,
        "dssp_ok": 0,
        "dssp_failed": 0,
        "proteins_with_predictions": 0,
        "proteins_without_predictions": 0,
        "length_mismatch": 0,
        "residues_total": 0,
        "residues_with_pred": 0,
        "residues_without_pred": 0,
        "residues_with_asa": 0,
        "residues_without_asa": 0,
    }

    seq_ids = list(split_map.keys())

    for prot_idx, seq_id in enumerate(seq_ids, start=1):
        residue_rows = split_map[seq_id]
        stats["total_proteins"] += 1
        stats["residues_total"] += len(residue_rows)

        source = parse_model_source(seq_id)

        structure_found = False
        dssp_ok = False
        pred_found = False
        mismatch_here = False

        # --- RSA: try pre-computed cache first, fall back to on-the-fly DSSP ---
        cached_rsa: Optional[np.ndarray] = None
        dssp_map: Dict = {}
        structure_path: Optional[str] = None

        if rsa_lookup is not None and seq_id in rsa_lookup:
            cached_rsa = rsa_lookup[seq_id]
            structure_found = True  # structure was processed when cache was built
            dssp_ok = True
            stats["structures_found"] += 1
            stats["dssp_ok"] += 1
        else:
            structure_path = resolve_structure_file(seq_id, structure_dir)
            structure_found = structure_path is not None

            if structure_found:
                stats["structures_found"] += 1
            else:
                stats["structures_missing"] += 1
                print(f"[WARN] structure not found for {seq_id}")

            if structure_path is not None:
                chain_id = seq_id.split("_", 1)[1] if "_" in seq_id else residue_rows[0]["chain"]
                try:
                    dssp_map = extract_dssp_accessibility(structure_path, chain_id=chain_id)
                    dssp_ok = True
                    stats["dssp_ok"] += 1
                except Exception as e:
                    print(f"[WARN] DSSP failed for {seq_id}: {e}")
                    stats["dssp_failed"] += 1

        y_true = None
        y_pred = None
        chain_weight = np.nan

        if results_map is not None and seq_id in results_map:
            pred_found = True
            stats["proteins_with_predictions"] += 1
            y_true = results_map[seq_id]["labels"]
            y_pred = results_map[seq_id]["preds"]
            chain_weight = results_map[seq_id]["weight"]

            if len(y_true) != len(residue_rows):
                mismatch_here = True
                stats["length_mismatch"] += 1
                print(f"[WARN] length mismatch for {seq_id}: split={len(residue_rows)} vs labels={len(y_true)}")
        elif results_map is not None:
            stats["proteins_without_predictions"] += 1
            print(f"[WARN] no predictions for {seq_id} in pkl")

        protein_res_with_pred = 0
        protein_res_with_asa = 0

        for i, r in enumerate(residue_rows):
            if cached_rsa is not None and i < len(cached_rsa):
                rsa_val = float(cached_rsa[i])
                asa_val = rsa_val * AA_MAX_ACC.get(r["aa"], np.nan) if not np.isnan(rsa_val) else np.nan
                asa = asa_val
                rsa = rsa_val
                protein_res_with_asa += 1
                stats["residues_with_asa"] += 1
            else:
                key = (r["chain"], r["resnum"])
                acc = dssp_map.get(key, None)

                if acc is None:
                    asa = np.nan
                    rsa = np.nan
                    stats["residues_without_asa"] += 1
                else:
                    asa = acc["asa"]
                    rsa = acc["rsa"]
                    protein_res_with_asa += 1
                    stats["residues_with_asa"] += 1

            pred_label = np.nan
            pred_score = np.nan

            if y_true is not None and i < len(y_true):
                pred_label = int(y_true[i])

            if y_pred is not None and i < len(y_pred):
                pred_score = float(y_pred[i])
                protein_res_with_pred += 1
                stats["residues_with_pred"] += 1
            else:
                stats["residues_without_pred"] += 1

            rows.append({
                "Sequence_ID": seq_id,
                "model_source": source,
                "chain": r["chain"],
                "resnum": r["resnum"],
                "aa": r["aa"],
                "label_from_txt": int(r["label"]),
                "y_true": pred_label,
                "y_pred": pred_score,
                "chain_weight": chain_weight,
                "asa": asa,
                "rsa": rsa,
                "structure_path": structure_path if structure_path is not None else "",
            })

        if verbose_success:
            status_bits = []
            status_bits.append("structure=OK" if structure_found else "structure=MISSING")
            status_bits.append("DSSP=OK" if dssp_ok else "DSSP=FAIL")
            status_bits.append("pred=OK" if pred_found else "pred=MISSING")
            if mismatch_here:
                status_bits.append("len=MISMATCH")
            print(
                f"[OK] {prot_idx}/{len(seq_ids)} {seq_id} | "
                + ", ".join(status_bits)
                + f" | residues={len(residue_rows)}, pred_res={protein_res_with_pred}, asa_res={protein_res_with_asa}"
            )

        if report_every and (prot_idx % report_every == 0):
            print("\n[INFO] intermediate summary")
            print(f"  processed proteins:      {prot_idx}/{len(seq_ids)}")
            print(f"  structures found:        {stats['structures_found']}")
            print(f"  structures missing:      {stats['structures_missing']}")
            print(f"  DSSP ok:                 {stats['dssp_ok']}")
            print(f"  DSSP failed:             {stats['dssp_failed']}")
            print(f"  proteins with preds:     {stats['proteins_with_predictions']}")
            print(f"  proteins without preds:  {stats['proteins_without_predictions']}")
            print(f"  length mismatches:       {stats['length_mismatch']}")
            print(f"  residues total:          {stats['residues_total']}")
            print(f"  residues with pred:      {stats['residues_with_pred']}")
            print(f"  residues with ASA:       {stats['residues_with_asa']}")
            if stats["residues_total"] > 0:
                print(f"  pred coverage:           {stats['residues_with_pred'] / stats['residues_total']:.3f}")
                print(f"  ASA coverage:            {stats['residues_with_asa'] / stats['residues_total']:.3f}")
            print("")

    df = pd.DataFrame(rows)

    print("\n[INFO] final summary")
    print(f"  total proteins:          {stats['total_proteins']}")
    print(f"  structures found:        {stats['structures_found']}")
    print(f"  structures missing:      {stats['structures_missing']}")
    print(f"  DSSP ok:                 {stats['dssp_ok']}")
    print(f"  DSSP failed:             {stats['dssp_failed']}")
    print(f"  proteins with preds:     {stats['proteins_with_predictions']}")
    print(f"  proteins without preds:  {stats['proteins_without_predictions']}")
    print(f"  length mismatches:       {stats['length_mismatch']}")
    print(f"  residues total:          {stats['residues_total']}")
    print(f"  residues with pred:      {stats['residues_with_pred']}")
    print(f"  residues without pred:   {stats['residues_without_pred']}")
    print(f"  residues with ASA:       {stats['residues_with_asa']}")
    print(f"  residues without ASA:    {stats['residues_without_asa']}")

    if stats["residues_total"] > 0:
        print(f"  pred coverage:           {stats['residues_with_pred'] / stats['residues_total']:.3f}")
        print(f"  ASA coverage:            {stats['residues_with_asa'] / stats['residues_total']:.3f}")

    return df


# ----------------------------
# Stratification helpers
# ----------------------------

def aucpr_residuewise(df: pd.DataFrame,
                      label_col: str = "y_true",
                      pred_col: str = "y_pred",
                      sample_weight_col: Optional[str] = None) -> float:
    sub = df[[label_col, pred_col] + ([sample_weight_col] if sample_weight_col else [])].copy()
    sub = sub.dropna(subset=[label_col, pred_col])

    if len(sub) == 0:
        return np.nan

    y = sub[label_col].to_numpy().astype(int)
    p = sub[pred_col].to_numpy().astype(float)

    bad = np.isnan(y) | np.isnan(p) | np.isinf(y) | np.isinf(p)
    sub = sub.loc[~bad]
    if len(sub) == 0:
        return np.nan

    y = sub[label_col].to_numpy().astype(int)
    p = sub[pred_col].to_numpy().astype(float)

    if y.sum() == 0:
        return 0.0

    if sample_weight_col:
        w = sub[sample_weight_col].to_numpy().astype(float)
        precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
    else:
        precision, recall, _ = precision_recall_curve(y, p)

    return float(auc(recall, precision))


def bootstrap_ci_residuewise(df: pd.DataFrame,
                             n_boot: int = 200,
                             seed: int = 0,
                             label_col: str = "y_true",
                             pred_col: str = "y_pred",
                             sample_weight_col: Optional[str] = None) -> Tuple[float, float]:
    if len(df) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    vals = []

    for _ in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        boot_df = df.iloc[samp]
        vals.append(aucpr_residuewise(
            boot_df,
            label_col=label_col,
            pred_col=pred_col,
            sample_weight_col=sample_weight_col,
        ))

    vals = np.asarray(vals, dtype=float)
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def add_accessibility_bins(df: pd.DataFrame,
                           rsa_col: str = "rsa") -> pd.DataFrame:
    out = df.copy()
    bins = [-np.inf, 0.05, 0.20, 0.50, np.inf]
    labels = ["buried(<=0.05)", "partly_buried(0.05-0.2)", "intermediate(0.2-0.5)", "exposed(>0.5)"]
    out["rsa_bin"] = pd.cut(out[rsa_col], bins=bins, labels=labels)
    return out


def summarize_by_group(df: pd.DataFrame,
                       group_col: str,
                       out_csv: str,
                       out_png: str,
                       title: str,
                       use_chain_weights: bool = False) -> pd.DataFrame:
    rows = []
    order = [x for x in df[group_col].dropna().unique()]
    order = sorted(order, key=str)

    vals, lo_list, hi_list, labels = [], [], [], []

    for g in order:
        sub = df[df[group_col] == g].copy()
        sub_eval = sub.dropna(subset=["y_true", "y_pred"])

        if use_chain_weights:
            weight_col = "chain_weight"
        else:
            weight_col = None

        aucpr = aucpr_residuewise(sub_eval, sample_weight_col=weight_col)
        ci_lo, ci_hi = bootstrap_ci_residuewise(sub_eval, sample_weight_col=weight_col)

        n_res = len(sub)
        n_eval = len(sub_eval)
        n_pos = int(sub_eval["y_true"].sum()) if n_eval > 0 else 0
        prevalence = (n_pos / n_eval) if n_eval > 0 else np.nan

        rows.append({
            group_col: g,
            "n_residues_total": n_res,
            "n_residues_eval": n_eval,
            "n_positive_residues": n_pos,
            "positive_rate": prevalence,
            "AUCPR": aucpr,
            "AUCPR_ci_lo": ci_lo,
            "AUCPR_ci_hi": ci_hi,
        })

        labels.append(str(g))
        vals.append(0.0 if pd.isna(aucpr) else aucpr)
        lo_list.append(ci_lo)
        hi_list.append(ci_hi)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_csv, index=False)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, vals, edgecolor="black")

    v = np.asarray(vals, dtype=float)
    lo = np.asarray(lo_list, dtype=float)
    hi = np.asarray(hi_list, dtype=float)

    bad = np.isnan(lo) | np.isnan(hi)
    lo2 = lo.copy()
    hi2 = hi.copy()
    lo2[bad] = v[bad]
    hi2[bad] = v[bad]
    yerr = np.vstack([v - lo2, hi2 - v])

    ax.errorbar(x, v, yerr=yerr, fmt="none", capsize=3, elinewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AUCPR")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return metrics


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split_txts", nargs="+", required=True,
                        help="Paths to split*.txt files")
    parser.add_argument("--structure_dirs", nargs="+", required=True,
                        help="Directories with PDB/mmCIF structures")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory")

    parser.add_argument("--results_pkl", nargs="+", default=None,
                        help="One or more test_results.pkl files (one per fold)")
    parser.add_argument("--use_chain_weights", action="store_true",
                        help="Use chain_weight as residue weight in AUCPR")

    parser.add_argument("--pipeline_folder", default=None,
                        help="Path to ScanNet_Ub/pipelines/ containing pre-computed "
                             "HandcraftedFeaturesPipeline .data files. "
                             "When provided, RSA is read from the cache instead of "
                             "running DSSP for each protein (much faster).")
    parser.add_argument("--rsa_col", type=int, default=_PIPELINE_RSA_COL_DEFAULT,
                        help="Column index of RSA in the pipeline cache feature matrix "
                             f"(default={_PIPELINE_RSA_COL_DEFAULT} for the pscparvhc MSA pipeline).")
    
    parser.add_argument("--report_every", type=int, default=50,
                        help="Print intermediate summary every N proteins")
    parser.add_argument("--verbose_success", action="store_true",
                        help="Print [OK] line for every successfully processed protein")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    split_map = build_split_map(args.split_txts)

    results_map = None
    if args.results_pkl:
        results_map = load_results_pkls(args.results_pkl)

    rsa_lookup = None
    if args.pipeline_folder:
        rsa_lookup = build_rsa_lookup_from_pipeline_cache(
            args.split_txts,
            args.pipeline_folder,
            rsa_col=args.rsa_col,
        )

    df = build_residue_table(
        split_map=split_map,
        structure_dir=args.structure_dirs[0],
        results_map=results_map,
        rsa_lookup=rsa_lookup,
        report_every=args.report_every,
        verbose_success=args.verbose_success,
    )

    df = add_accessibility_bins(df)

    residue_table_csv = os.path.join(args.out_dir, "residue_table.csv")
    df.to_csv(residue_table_csv, index=False)
    print(f"[OK] wrote {residue_table_csv}")

    # Accessibility bins
    summarize_by_group(
        df=df,
        group_col="rsa_bin",
        out_csv=os.path.join(args.out_dir, "metrics_by_accessibility_bin.csv"),
        out_png=os.path.join(args.out_dir, "barplot_aucpr_by_accessibility_bin.png"),
        title="Residue-wise AUCPR by accessibility bin",
        use_chain_weights=args.use_chain_weights,
    )

    # Amino acid
    summarize_by_group(
        df=df,
        group_col="aa",
        out_csv=os.path.join(args.out_dir, "metrics_by_aa.csv"),
        out_png=os.path.join(args.out_dir, "barplot_aucpr_by_aa.png"),
        title="Residue-wise AUCPR by amino acid",
        use_chain_weights=args.use_chain_weights,
    )

    # Model source
    summarize_by_group(
        df=df,
        group_col="model_source",
        out_csv=os.path.join(args.out_dir, "metrics_by_model_source.csv"),
        out_png=os.path.join(args.out_dir, "barplot_aucpr_by_model_source.png"),
        title="Residue-wise AUCPR by model source",
        use_chain_weights=args.use_chain_weights,
    )

    print(f"[OK] all outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()