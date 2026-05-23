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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP

from stratify_utils import (
    normalize_id, is_pdb_identifier, parse_model_source, parse_ec_top,
    ensure_1d_preds, ensure_1d_labels,
    load_results_pkl, load_results_pkls,
    aucpr_like_make_PR_curve, bootstrap_ci_aucpr_group,
    aucroc_chain_weighted, bootstrap_ci_aucroc_group,
    compute_group_aucpr, barplot_with_ci, stratify_and_plot,
    parse_split_txt, build_split_map, load_dataset_csv, resolve_structure_file,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


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
        out[(chain_id, str(resseq))] = rsa_val

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
    df = df.copy()
    chem_map: Dict[str, Optional[str]] = {}
    for sid, grp in df.groupby("Sequence_ID"):
        pos_aas = grp.loc[grp["y_true"] == 1.0, "aa"].tolist()
        chem_map[sid] = str(get_catalytic_class(pos_aas)) if pos_aas else None
    df["chemotype"] = df["Sequence_ID"].map(chem_map)
    return df


# ─── main ──────────────────────────────────────────────────────────────────────

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
