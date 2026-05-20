#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratify.py — Build residue_table.csv for a single model and produce
stratified AUCPR and AUC-ROC metrics (RSA bin, amino acid, EC class, chemotype, model source).

Stage 1 (slow, once per model):
  Load pkl predictions, split*.txt labels, RSA from pipeline cache / DSSP fallback,
  EC from dataset.csv, chemotype from per-chain positive residues.
  → residue_table.csv

Stage 2 (fast):
  For each stratification group, compute chain-weighted AUCPR + AUC-ROC + 95% CI
  by resampling proteins.
  → metrics_by_{rsa_bin,aa,ec_top,chemotype,model_source}.csv
  → aucpr_by_{tag}.png + aucroc_by_{tag}.png per stratification

Usage:
  python stratify.py \\
    --split_txts split1.txt split2.txt ... \\
    --results_pkl fold1/test_results.pkl ... \\
    --out_dir /path/to/output \\
    [--dataset_csv dataset.csv] \\
    [--structure_dirs /path/to/pdbs] \\
    [--pipeline_folder /path/to/pipelines] \\
    [--rsa_col 29] \\
    [--n_boot 200] [--seed 0] [--report_every 50]
"""

import argparse
import logging
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP


# ─── figure style ──────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 14, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "axes.titlesize": 16,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "lines.linewidth": 0.9,
    "svg.fonttype": "path",
})

W_FULL  = 6.27
GUTTER  = 0.15
W_HALF  = (W_FULL - GUTTER) / 2
W_THIRD = (W_FULL - GUTTER) / 3
H_SMALL = 1.9
H_WIDE  = 2.7


# ─── constants ─────────────────────────────────────────────────────────────────

AA_MAX_ACC = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

RSA_BINS   = [-np.inf, 0.05, 0.20, 0.50, np.inf]
RSA_LABELS = [
    "buried(<=0.05)", "partly_buried(0.05-0.2)",
    "intermediate(0.2-0.5)", "exposed(>0.5)",
]

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

_PIPELINE_RSA_COL_DEFAULT = 29
_PIPELINE_CACHE_PATTERN = (
    "catalytic_sites_weight_based_v9_{split_name}_MSA_"
    "pipeline_Handcrafted_features-pscparvhc_gaps-True_Beff-500.data"
)


# ─── chemotype ─────────────────────────────────────────────────────────────────

CHEMOTYPE_NAMES = {
    0: "0 (ILMVWF)", 1: "1 (AGP)", 2: "2 (QN)", 3: "3 (KR)",
    4: "4 (S)",      5: "5 (T)",   6: "6 (DE)", 7: "7 (other/none)",
}


def get_catalytic_class(residues: List[str]) -> int:
    if any(r in residues for r in "ILMVWF"): return 0
    if any(r in residues for r in "AGP"):    return 1
    if any(r in residues for r in "QN"):     return 2
    if any(r in residues for r in "KR"):     return 3
    if any(r == "S"      for r in residues): return 4
    if any(r == "T"      for r in residues): return 5
    if any(r in residues for r in "DE"):     return 6
    return 7


# ─── id helpers ────────────────────────────────────────────────────────────────

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


def parse_ec_top(ec_number: Any) -> Optional[str]:
    if ec_number is None or (isinstance(ec_number, float) and np.isnan(ec_number)):
        return None
    m = re.match(r"^(\d+)", str(ec_number).strip())
    return str(int(m.group(1))) if m else None


def ensure_1d_preds(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.ndim > 1:
        p = p[..., -1]
    return p.reshape(-1)


def ensure_1d_labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {y.shape}")
    return y


# ─── split txt ─────────────────────────────────────────────────────────────────

def parse_split_txt(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse split*.txt → {seq_id: [{chain, resnum, aa, label}, ...]}."""
    data: Dict[str, List[Dict[str, Any]]] = {}
    cur_id: Optional[str] = None
    cur_rows: List[Dict[str, Any]] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    data[cur_id] = cur_rows
                cur_id  = normalize_id(line[1:].strip())
                cur_rows = []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    chain, resnum, aa, label = parts[:4]
                    cur_rows.append({"chain": chain, "resnum": str(resnum),
                                     "aa": aa, "label": int(label)})

    if cur_id is not None:
        data[cur_id] = cur_rows
    return data


def build_split_map(split_txts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in split_txts:
        for k, v in parse_split_txt(path).items():
            if k in out:
                raise ValueError(f"Duplicate Sequence_ID across split txts: {k}")
            out[k] = v
    return out


# ─── pkl loading ───────────────────────────────────────────────────────────────

def load_results_pkl(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    for k in ("labels", "predictions", "weights", "ids"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    result: Dict[str, Dict[str, Any]] = {}
    for sid, labels, preds, w in zip(data["ids"], data["labels"], data["predictions"], data["weights"]):
        sid = normalize_id(sid)
        result[sid] = {
            "labels": ensure_1d_labels(labels),
            "preds":  ensure_1d_preds(preds),
            "weight": float(w),
        }
    return result


def load_results_pkls(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load and average predictions across multiple pkl files (one per fold)."""
    accum: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for sid, rec in load_results_pkl(path).items():
            if sid not in accum:
                accum[sid] = {"labels": rec["labels"], "preds_list": [rec["preds"]], "weight": rec["weight"]}
            else:
                accum[sid]["preds_list"].append(rec["preds"])
        log.info("loaded %s", path)

    merged = {
        sid: {
            "labels": rec["labels"],
            "preds":  np.mean(rec["preds_list"], axis=0).astype(np.float32),
            "weight": rec["weight"],
        }
        for sid, rec in accum.items()
    }
    log.info("total chains: %d", len(merged))
    return merged


# ─── structure resolution ──────────────────────────────────────────────────────

def resolve_structure_file(seq_id: str, structure_dirs: List[str]) -> Optional[str]:
    base = seq_id.split("_", 1)[0]
    candidates = (
        f"{seq_id}.cif", f"{seq_id}.pdb", f"{seq_id}.mmcif",
        f"{base}.cif",   f"{base}.pdb",   f"{base}.mmcif",
    )
    for sd in structure_dirs:
        for fname in candidates:
            p = os.path.join(sd, fname)
            if os.path.exists(p):
                return p
    return None


# ─── pipeline cache RSA ────────────────────────────────────────────────────────

def build_rsa_lookup_from_pipeline_cache(
    split_txt_paths: List[str],
    pipeline_folder: str,
    rsa_col: int = _PIPELINE_RSA_COL_DEFAULT,
) -> Dict[str, np.ndarray]:
    """
    Load per-residue RSA from pre-computed HandcraftedFeaturesPipeline .data files.
    Returns {seq_id: rsa_array}; chains with length mismatches are omitted (→ DSSP fallback).
    """
    rsa_lookup: Dict[str, np.ndarray] = {}

    for split_txt in split_txt_paths:
        split_name = os.path.splitext(os.path.basename(split_txt))[0]
        cache_path = os.path.join(
            pipeline_folder, _PIPELINE_CACHE_PATTERN.format(split_name=split_name)
        )
        if not os.path.exists(cache_path):
            log.info("pipeline cache not found: %s", cache_path)
            continue
        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
        except Exception as exc:
            log.warning("failed to load pipeline cache %s: %s", cache_path, exc)
            continue

        inputs     = cache["inputs"]
        failed_set = {int(x) for x in cache.get("failed_samples", [])}

        # Read ordered seq_ids and residue counts from the split file
        seq_ids: List[str] = []
        res_counts: List[int] = []
        cur_id: Optional[str] = None
        cnt = 0
        with open(split_txt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_id is not None:
                        res_counts.append(cnt)
                    cur_id = normalize_id(line[1:].strip())
                    seq_ids.append(cur_id)
                    cnt = 0
                else:
                    cnt += 1
        if cur_id is not None:
            res_counts.append(cnt)

        cache_idx = 0
        n_ok = n_mismatch = 0
        for orig_idx, (sid, split_n) in enumerate(zip(seq_ids, res_counts)):
            if orig_idx in failed_set:
                continue
            arr = np.asarray(inputs[cache_idx])
            if arr.ndim == 2 and arr.shape[1] > rsa_col and arr.shape[0] == split_n:
                rsa_lookup[sid] = arr[:, rsa_col].astype(np.float32)
                n_ok += 1
            else:
                n_mismatch += 1
            cache_idx += 1

        log.info("%s: RSA from cache %d chains, %d mismatches → DSSP fallback",
                 split_name, n_ok, n_mismatch)

    log.info("RSA lookup total: %d chains", len(rsa_lookup))
    return rsa_lookup


# ─── DSSP RSA ──────────────────────────────────────────────────────────────────

def extract_dssp_rsa(structure_path: str, chain_id: str) -> Dict[Tuple[str, str], float]:
    """Returns {(chain, resnum_str): rsa_float} for the given chain."""
    if structure_path.endswith((".cif", ".mmcif")):
        structure = MMCIFParser(QUIET=True).get_structure("s", structure_path)
    else:
        structure = PDBParser(QUIET=True).get_structure("s", structure_path)

    dssp = DSSP(structure[0], structure_path)
    out: Dict[Tuple[str, str], float] = {}

    for key in dssp.keys():
        if key[0] != chain_id:
            continue
        hetflag, resseq, icode = key[1]
        if hetflag.strip():
            continue
        try:
            rsa_val = float(dssp[key][3])
        except Exception:
            rsa_val = np.nan
        resnum_str = str(resseq) if not str(icode).strip() else f"{resseq}{icode.strip()}"
        out[(chain_id, resnum_str)] = rsa_val
        out[(chain_id, str(resseq))] = rsa_val  # also store without insertion code

    return out


# ─── residue table ─────────────────────────────────────────────────────────────

def build_residue_table(
    split_map:    Dict[str, List[Dict[str, Any]]],
    structure_dirs: List[str],
    results_map:  Optional[Dict[str, Dict[str, Any]]] = None,
    rsa_lookup:   Optional[Dict[str, np.ndarray]] = None,
    report_every: int = 50,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seq_ids = list(split_map.keys())
    stats = {k: 0 for k in ("structs_ok", "structs_miss", "dssp_ok", "dssp_fail",
                              "preds_ok", "preds_miss", "len_mismatch")}

    for idx, seq_id in enumerate(seq_ids, 1):
        residue_rows = split_map[seq_id]
        source = parse_model_source(seq_id)

        # ── RSA: pipeline cache → DSSP fallback ────────────────────────────────
        cached_rsa: Optional[np.ndarray] = None
        dssp_map:   Dict                 = {}
        structure_path: Optional[str]    = None

        if rsa_lookup is not None and seq_id in rsa_lookup:
            cached_rsa = rsa_lookup[seq_id]
            stats["structs_ok"] += 1
            stats["dssp_ok"]    += 1
        else:
            structure_path = resolve_structure_file(seq_id, structure_dirs)
            if structure_path:
                stats["structs_ok"] += 1
                chain_id = seq_id.split("_", 1)[1] if "_" in seq_id else residue_rows[0]["chain"]
                try:
                    dssp_map = extract_dssp_rsa(structure_path, chain_id)
                    stats["dssp_ok"] += 1
                except Exception as exc:
                    log.warning("DSSP failed %s: %s", seq_id, exc)
                    stats["dssp_fail"] += 1
            else:
                stats["structs_miss"] += 1

        # ── predictions ────────────────────────────────────────────────────────
        y_true: Optional[np.ndarray] = None
        y_pred: Optional[np.ndarray] = None
        chain_weight = np.nan

        if results_map is not None:
            if seq_id in results_map:
                rec          = results_map[seq_id]
                chain_weight = rec["weight"]
                if len(rec["labels"]) == len(residue_rows):
                    y_true = rec["labels"]
                    y_pred = rec["preds"]
                    stats["preds_ok"] += 1
                else:
                    stats["len_mismatch"] += 1
                    log.warning("length mismatch %s: split=%d labels=%d",
                                seq_id, len(residue_rows), len(rec["labels"]))
            else:
                stats["preds_miss"] += 1

        # ── emit rows ──────────────────────────────────────────────────────────
        for i, r in enumerate(residue_rows):
            if cached_rsa is not None and i < len(cached_rsa):
                rsa = float(cached_rsa[i])
            else:
                rsa = dssp_map.get((r["chain"], r["resnum"]), np.nan)
                if not isinstance(rsa, float):
                    rsa = np.nan

            rows.append({
                "Sequence_ID":  seq_id,
                "model_source": source,
                "chain":        r["chain"],
                "resnum":       r["resnum"],
                "aa":           r["aa"],
                "label":        r["label"],
                "y_true":       float(y_true[i]) if y_true is not None else np.nan,
                "y_pred":       float(y_pred[i]) if y_pred is not None else np.nan,
                "chain_weight": chain_weight,
                "rsa":          rsa,
            })

        if report_every and idx % report_every == 0:
            log.info("%d/%d | structs=%d miss=%d dssp=%d fail=%d preds=%d no_pred=%d",
                     idx, len(seq_ids),
                     stats["structs_ok"], stats["structs_miss"],
                     stats["dssp_ok"],   stats["dssp_fail"],
                     stats["preds_ok"],  stats["preds_miss"])

    df = pd.DataFrame(rows)
    log.info("residue table: %d rows, %d chains", len(df), df["Sequence_ID"].nunique())
    for k, v in stats.items():
        log.info("  %s: %d", k, v)
    return df


def add_rsa_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsa_bin"] = pd.cut(df["rsa"], bins=RSA_BINS, labels=RSA_LABELS).astype(str)
    df.loc[df["rsa"].isna(), "rsa_bin"] = np.nan
    return df


def add_ec_top(df: pd.DataFrame, dataset_csv: str) -> pd.DataFrame:
    ds = pd.read_csv(dataset_csv, low_memory=False)
    if "Sequence_ID" not in ds.columns or "EC_number" not in ds.columns:
        log.warning("dataset_csv missing Sequence_ID or EC_number; skipping EC enrichment")
        df = df.copy()
        df["ec_top"] = np.nan
        return df
    ds = ds.drop_duplicates(subset=["Sequence_ID"])
    ds["Sequence_ID"] = ds["Sequence_ID"].apply(normalize_id)
    ec_map = ds.set_index("Sequence_ID")["EC_number"].to_dict()
    df = df.copy()
    df["ec_top"] = df["Sequence_ID"].map(ec_map).apply(parse_ec_top)
    return df


def add_chemotype(df: pd.DataFrame) -> pd.DataFrame:
    """Assign chemotype per chain from positive-label residues (y_true == 1)."""
    df = df.copy()
    chem_map: Dict[str, Optional[str]] = {}
    for sid, grp in df.groupby("Sequence_ID"):
        pos_aas = grp.loc[grp["y_true"] == 1.0, "aa"].tolist()
        chem_map[sid] = str(get_catalytic_class(pos_aas)) if pos_aas else None
    df["chemotype"] = df["Sequence_ID"].map(chem_map)
    return df


# ─── AUCPR (chain-weighted, matches ScanNet make_PR_curve) ─────────────────────

def aucpr_like_make_PR_curve(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
) -> float:
    y_list, p_list, w_list = [], [], []
    for lbl, pred, w in zip(labels, predictions, weights):
        lbl  = ensure_1d_labels(lbl)
        pred = ensure_1d_preds(pred)
        L = min(len(lbl), len(pred))
        if L == 0:
            continue
        y_list.append(lbl[:L])
        p_list.append(pred[:L])
        w_list.append(np.full(L, float(w)))

    if not y_list:
        return 0.0

    y_all = np.concatenate(y_list)
    p_all = np.concatenate(p_list)
    w_all = np.concatenate(w_list)

    bad = np.isnan(p_all) | np.isnan(y_all) | np.isinf(p_all) | np.isinf(y_all)
    if bad.any():
        good_mask = ~bad
        fill = float(np.nanmedian(p_all[good_mask])) if good_mask.any() else 0.0
        p_all = p_all.copy()
        p_all[bad] = fill

    good = ~bad
    if not good.any() or y_all[good].sum() == 0:
        return 0.0

    prec, rec, _ = precision_recall_curve(y_all[good], p_all[good], sample_weight=w_all[good])
    return float(auc(rec, prec))


def bootstrap_ci_aucpr_group(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
    n_boot: int = 200,
    seed:   int = 0,
) -> Tuple[float, float]:
    """Bootstrap 95% CI by resampling proteins (not residues)."""
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return np.nan, np.nan

    rng   = np.random.default_rng(seed)
    w_arr = np.asarray(weights, dtype=float)
    vals  = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        s      = rng.choice(n, size=n, replace=True)
        vals[b] = aucpr_like_make_PR_curve(
            [labels[i]      for i in s],
            [predictions[i] for i in s],
            w_arr[s].tolist(),
        )

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def aucroc_chain_weighted(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
) -> float:
    """Chain-weighted AUC-ROC (same flattening logic as aucpr_like_make_PR_curve)."""
    y_list, p_list, w_list = [], [], []
    for lbl, pred, w in zip(labels, predictions, weights):
        lbl  = ensure_1d_labels(lbl)
        pred = ensure_1d_preds(pred)
        L = min(len(lbl), len(pred))
        if L == 0:
            continue
        y_list.append(lbl[:L])
        p_list.append(pred[:L])
        w_list.append(np.full(L, float(w)))

    if not y_list:
        return np.nan

    y_all = np.concatenate(y_list)
    p_all = np.concatenate(p_list)
    w_all = np.concatenate(w_list)

    good = ~(np.isnan(p_all) | np.isnan(y_all) | np.isinf(p_all) | np.isinf(y_all))
    if not good.any():
        return np.nan

    y_g = y_all[good].astype(int)
    # need both classes present
    if y_g.sum() == 0 or y_g.sum() == len(y_g):
        return np.nan

    return float(roc_auc_score(y_g, p_all[good], sample_weight=w_all[good]))


def bootstrap_ci_aucroc_group(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
    n_boot: int = 200,
    seed:   int = 0,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for AUC-ROC by resampling proteins."""
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return np.nan, np.nan

    rng   = np.random.default_rng(seed)
    w_arr = np.asarray(weights, dtype=float)
    vals  = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        s      = rng.choice(n, size=n, replace=True)
        vals[b] = aucroc_chain_weighted(
            [labels[i]      for i in s],
            [predictions[i] for i in s],
            w_arr[s].tolist(),
        )

    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def compute_group_aucpr(
    df:        pd.DataFrame,
    group_col: str,
    groups:    Optional[List] = None,
    n_boot:    int = 200,
    seed:      int = 0,
) -> pd.DataFrame:
    """
    Chain-weighted AUCPR and AUC-ROC + bootstrap CI for each group in group_col.
    Works for both residue-level groups (rsa_bin) and chain-level groups (ec_top, chemotype).
    """
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    valid = valid[valid[group_col].notna()]
    valid["y_true"] = valid["y_true"].astype(int)

    if groups is None:
        groups = sorted(valid[group_col].dropna().unique(), key=str)

    rows = []
    for g in groups:
        sub = valid[valid[group_col].astype(str) == str(g)]
        if len(sub) == 0:
            rows.append({group_col: g,
                         "AUCPR": np.nan, "AUCPR_ci_lo": np.nan, "AUCPR_ci_hi": np.nan,
                         "AUCROC": np.nan, "AUCROC_ci_lo": np.nan, "AUCROC_ci_hi": np.nan,
                         "n_chains": 0, "n_residues": 0, "n_positive": 0})
            continue

        labels_list, preds_list, weights_list = [], [], []
        for sid, grp in sub.groupby("Sequence_ID"):
            w = float(grp["chain_weight"].iloc[0])
            labels_list.append(grp["y_true"].to_numpy(dtype=int))
            preds_list.append(grp["y_pred"].to_numpy(dtype=float))
            weights_list.append(1.0 if np.isnan(w) else w)

        au_pr  = aucpr_like_make_PR_curve(labels_list, preds_list, weights_list)
        lo_pr, hi_pr = bootstrap_ci_aucpr_group(labels_list, preds_list, weights_list, n_boot=n_boot, seed=seed)

        au_roc = aucroc_chain_weighted(labels_list, preds_list, weights_list)
        lo_roc, hi_roc = bootstrap_ci_aucroc_group(labels_list, preds_list, weights_list, n_boot=n_boot, seed=seed)

        prevalence = int(sub["y_true"].sum()) / len(sub)
        rows.append({
            group_col:      g,
            "AUCPR":        au_pr,
            "AUCPR_ci_lo":  lo_pr,
            "AUCPR_ci_hi":  hi_pr,
            "AUCROC":       au_roc,
            "AUCROC_ci_lo": lo_roc,
            "AUCROC_ci_hi": hi_roc,
            "prevalence":   prevalence,
            "n_chains":     len(labels_list),
            "n_residues":   len(sub),
            "n_positive":   int(sub["y_true"].sum()),
        })

    return pd.DataFrame(rows)


# ─── plotting ──────────────────────────────────────────────────────────────────

def barplot_with_ci(
    values:   List[float],
    labels:   List[str],
    ci_lo:    List[float],
    ci_hi:    List[float],
    ylabel:   str,
    title:    str,
    out_path: str,
    ylim:     Optional[Tuple[float, float]] = (0.0, 1.0),
    color:    str = "#7BC8F6",
) -> None:
    fig, ax = plt.subplots(figsize=(W_FULL, H_WIDE))
    x   = np.arange(len(values))
    v   = np.asarray(values, float)
    lo  = np.asarray(ci_lo,  float)
    hi  = np.asarray(ci_hi,  float)

    ax.bar(x, v, color=color, edgecolor="black")

    bad = np.isnan(lo) | np.isnan(hi)
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[bad], hi2[bad] = v[bad], v[bad]
    ax.errorbar(x, v, yerr=np.vstack([v - lo2, hi2 - v]), fmt="none", capsize=3, elinewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def stratify_and_plot(
    df:        pd.DataFrame,
    group_col: str,
    out_dir:   str,
    tag:       str,
    title:     str,
    groups:    Optional[List] = None,
    xlabels:   Optional[List[str]] = None,
    n_boot:    int = 200,
    seed:      int = 0,
) -> pd.DataFrame:
    if group_col not in df.columns:
        log.warning("column '%s' not in residue_table — skipping %s", group_col, tag)
        return pd.DataFrame()

    metrics = compute_group_aucpr(df, group_col, groups=groups, n_boot=n_boot, seed=seed)
    metrics.to_csv(os.path.join(out_dir, f"metrics_by_{tag}.csv"), index=False)

    lbl = xlabels if xlabels else [str(g) for g in metrics[group_col]]
    barplot_with_ci(
        values=metrics["AUCPR"].fillna(0).tolist(),
        labels=lbl,
        ci_lo=metrics["AUCPR_ci_lo"].tolist(),
        ci_hi=metrics["AUCPR_ci_hi"].tolist(),
        ylabel="AUCPR",
        title=f"{title} — AUCPR",
        out_path=os.path.join(out_dir, f"aucpr_by_{tag}.png"),
    )
    barplot_with_ci(
        values=metrics["AUCROC"].fillna(0.5).tolist(),
        labels=lbl,
        ci_lo=metrics["AUCROC_ci_lo"].tolist(),
        ci_hi=metrics["AUCROC_ci_hi"].tolist(),
        ylabel="AUC-ROC",
        title=f"{title} — AUC-ROC",
        out_path=os.path.join(out_dir, f"aucroc_by_{tag}.png"),
    )
    log.info("%s: %d groups → %s", tag, len(metrics), out_dir)
    return metrics


# ─── main ──────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split_txts",      nargs="+", required=True,
                    help="Paths to split*.txt files")
    ap.add_argument("--results_pkl",     nargs="+", default=None,
                    help="One or more test_results.pkl (one per fold; predictions are averaged)")
    ap.add_argument("--out_dir",         required=True)
    ap.add_argument("--dataset_csv",     default=None,
                    help="CSV with Sequence_ID and EC_number columns (for EC stratification)")
    ap.add_argument("--structure_dirs",  nargs="+", default=None,
                    help="Directories with PDB/mmCIF files (DSSP fallback when pipeline cache misses)")
    ap.add_argument("--pipeline_folder", default=None,
                    help="ScanNet_Ub/pipelines/ dir with pre-computed HandcraftedFeaturesPipeline caches (fast RSA)")
    ap.add_argument("--rsa_col",      type=int, default=_PIPELINE_RSA_COL_DEFAULT,
                    help=f"Column index of RSA in pipeline cache (default={_PIPELINE_RSA_COL_DEFAULT})")
    ap.add_argument("--n_boot",       type=int, default=200,
                    help="Bootstrap samples for AUCPR CI (0 disables CI)")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--report_every", type=int, default=50,
                    help="Print progress every N proteins")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    split_map = build_split_map(args.split_txts)

    results_map = None
    if args.results_pkl:
        results_map = load_results_pkls(args.results_pkl)

    rsa_lookup = None
    if args.pipeline_folder:
        rsa_lookup = build_rsa_lookup_from_pipeline_cache(
            args.split_txts, args.pipeline_folder, rsa_col=args.rsa_col
        )

    df = build_residue_table(
        split_map=split_map,
        structure_dirs=args.structure_dirs or [],
        results_map=results_map,
        rsa_lookup=rsa_lookup,
        report_every=args.report_every,
    )
    df = add_rsa_bins(df)

    if args.dataset_csv:
        df = add_ec_top(df, args.dataset_csv)
    else:
        df["ec_top"] = np.nan

    df = add_chemotype(df)

    residue_table_csv = os.path.join(args.out_dir, "residue_table.csv")
    df.to_csv(residue_table_csv, index=False)
    log.info("wrote %s (%d rows)", residue_table_csv, len(df))

    kw = dict(df=df, out_dir=args.out_dir, n_boot=args.n_boot, seed=args.seed)

    stratify_and_plot(group_col="rsa_bin",      tag="rsa_bin",
                      title="AUCPR by RSA bin",          groups=RSA_LABELS, **kw)
    stratify_and_plot(group_col="aa",            tag="aa",
                      title="AUCPR by amino acid",       groups=AA_ORDER, **kw)
    stratify_and_plot(group_col="model_source",  tag="model_source",
                      title="AUCPR by model source (PDB vs AF)", **kw)

    if args.dataset_csv:
        stratify_and_plot(group_col="ec_top", tag="ec_top",
                          title="AUCPR by EC top-level class", **kw)

    chem_groups = [str(c) for c in range(8)]
    chem_labels = [CHEMOTYPE_NAMES[c] for c in range(8)]
    stratify_and_plot(group_col="chemotype", tag="chemotype",
                      title="AUCPR by chemotype",
                      groups=chem_groups, xlabels=chem_labels, **kw)

    log.info("all outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
