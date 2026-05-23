#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratify_plddt.py — Stratification #2: pLDDT-bin stratification for AF models.

Tests whether ESM2 embeddings help more on low-confidence (disordered) regions,
which are often multi-subunit enzyme interfaces.

Protocol:
  1. Keep only AF models (seq_id base is NOT a 4-char PDB identifier).
  2. Read pLDDT per residue from the B-factor column of the AF structure file.
  3. Stratify residues into three bins:
       low:    pLDDT in [0, 70)
       medium: pLDDT in [70, 85)
       high:   pLDDT in [85, 100]
  4. Compute chain-weighted AUCPR and AUC-ROC per bin.

Usage:
  python stratify_plddt.py \\
    --split_txts split1.txt ... split5.txt \\
    --results_pkl fold1/test_results.pkl ... fold5/test_results.pkl \\
    --structure_dirs /path/to/af_models \\
    --out_dir /path/to/output \\
    [--n_boot 200] [--seed 0] [--report_every 200]
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, MMCIFParser

from stratify_utils import (
    is_pdb_identifier,
    ensure_1d_labels, ensure_1d_preds,
    load_results_pkls,
    compute_group_aucpr, barplot_with_ci, stratify_and_plot,
    build_split_map, resolve_all_structure_files,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


PLDDT_BINS   = [0, 70, 85, 100]
PLDDT_LABELS = ["low(<70)", "medium(70-85)", "high(>85)"]


def extract_plddt(structure_path: str, chain_id: str) -> Dict[str, float]:
    """
    Returns {resnum_str → pLDDT} by reading Cα B-factors from an AF structure file.
    pLDDT is stored in the B-factor column for AlphaFold models.
    """
    if structure_path.endswith((".cif", ".mmcif")):
        structure = MMCIFParser(QUIET=True).get_structure("s", structure_path)
    else:
        structure = PDBParser(QUIET=True).get_structure("s", structure_path)

    out: Dict[str, float] = {}
    for chain in structure[0]:
        if chain.id != chain_id:
            continue
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag.strip():
                continue
            if not residue.has_id("CA"):
                continue
            ca = residue["CA"]
            resnum_str = str(resseq) if not str(icode).strip() else f"{resseq}{icode.strip()}"
            out[resnum_str] = ca.get_bfactor()
            out[str(resseq)] = ca.get_bfactor()
    return out


def _plddt_bin(val: float) -> str:
    if val < 70:
        return PLDDT_LABELS[0]
    elif val < 85:
        return PLDDT_LABELS[1]
    else:
        return PLDDT_LABELS[2]


def build_plddt_table(
    split_map:     Dict[str, List[Dict[str, Any]]],
    results_map:   Dict[str, Dict[str, Any]],
    structure_dirs: List[str],
    report_every:  int = 200,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    af_ids = [sid for sid in split_map if not is_pdb_identifier(sid.split("_")[0])]
    n_found = n_miss = n_pred_ok = n_pred_miss = 0

    for idx, seq_id in enumerate(af_ids, 1):
        split_rows = split_map[seq_id]
        chain_id   = seq_id.split("_", 1)[1] if "_" in seq_id else split_rows[0]["chain"]

        struct_paths = resolve_all_structure_files(seq_id, structure_dirs)
        plddt_map: Dict[str, float] = {}
        if struct_paths:
            for sp in struct_paths:
                try:
                    plddt_map.update(extract_plddt(sp, chain_id))
                except Exception as exc:
                    log.warning("pLDDT read failed %s (%s): %s", seq_id, sp, exc)
            if plddt_map:
                n_found += 1
            else:
                n_miss += 1
        else:
            n_miss += 1

        if seq_id in results_map:
            rec = results_map[seq_id]
            if len(rec["labels"]) == len(split_rows):
                y_true = rec["labels"]
                y_pred = rec["preds"]
                chain_weight = rec["weight"]
                n_pred_ok += 1
            else:
                log.warning("length mismatch %s: pred=%d rows=%d",
                            seq_id, len(rec["labels"]), len(split_rows))
                y_true = y_pred = None
                chain_weight = np.nan
                n_pred_miss += 1
        else:
            y_true = y_pred = None
            chain_weight = np.nan
            n_pred_miss += 1

        for i, r in enumerate(split_rows):
            plddt_val = float(plddt_map.get(r["resnum"], np.nan))
            plddt_bin = _plddt_bin(plddt_val) if not np.isnan(plddt_val) else np.nan

            rows.append({
                "Sequence_ID":  seq_id,
                "chain":        r["chain"],
                "resnum":       r["resnum"],
                "aa":           r["aa"],
                "y_true":       float(y_true[i]) if y_true is not None else np.nan,
                "y_pred":       float(y_pred[i]) if y_pred is not None else np.nan,
                "chain_weight": chain_weight,
                "plddt":        float(plddt_val) if plddt_val is not None else np.nan,
                "plddt_bin":    plddt_bin,
            })

        if report_every and idx % report_every == 0:
            log.info("%d/%d AF chains | struct_found=%d miss=%d preds_ok=%d miss=%d",
                     idx, len(af_ids), n_found, n_miss, n_pred_ok, n_pred_miss)

    df = pd.DataFrame(rows)
    log.info("pLDDT table: %d rows, %d AF chains", len(df), df["Sequence_ID"].nunique())
    log.info("  struct found=%d miss=%d  preds ok=%d miss=%d",
             n_found, n_miss, n_pred_ok, n_pred_miss)
    return df


def run(
    split_txts:    List[str],
    results_pkls:  List[str],
    structure_dirs: List[str],
    out_dir:       str,
    n_boot:        int = 200,
    seed:          int = 0,
    report_every:  int = 200,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    split_map   = build_split_map(split_txts)
    results_map = load_results_pkls(results_pkls)

    df = build_plddt_table(split_map, results_map, structure_dirs,
                           report_every=report_every)

    table_csv = os.path.join(out_dir, "plddt_table.csv")
    df.to_csv(table_csv, index=False)
    log.info("wrote %s (%d rows)", table_csv, len(df))

    kw = dict(df=df, out_dir=out_dir, n_boot=n_boot, seed=seed)
    stratify_and_plot(
        group_col="plddt_bin",
        tag="plddt_bin",
        title="AUCPR by pLDDT bin (AF only)",
        groups=PLDDT_LABELS,
        **kw,
    )

    log.info("all outputs in %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split_txts",      nargs="+", required=True)
    ap.add_argument("--results_pkl",     nargs="+", required=True)
    ap.add_argument("--structure_dirs",  nargs="+", required=True,
                    help="Directories containing AF PDB/CIF files (pLDDT in B-factor column)")
    ap.add_argument("--out_dir",         required=True)
    ap.add_argument("--n_boot",       type=int, default=200)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--report_every", type=int, default=200)
    args = ap.parse_args()

    run(
        split_txts=args.split_txts,
        results_pkls=args.results_pkl,
        structure_dirs=args.structure_dirs,
        out_dir=args.out_dir,
        n_boot=args.n_boot,
        seed=args.seed,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
