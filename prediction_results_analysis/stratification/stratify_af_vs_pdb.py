#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratify_af_vs_pdb.py — Stratification #1: AF model vs PDB structure comparison.

Tests whether the model works differently on AlphaFold models (single conformation)
vs experimentally determined PDB structures (potentially inactive conformations).

Protocol:
  1. Keep only proteins that have BOTH an AF model AND at least one PDB chain
     in the test set.
  2. Per-residue coverage mask: for each AF chain, mark a position as covered if
     it appears in ANY PDB chain from the same protein (via pairwise alignment of
     PDB chain sequence to AF chain sequence).
  3. AUCPR of AF: mask out uncovered residues, weight each chain by W_sequence.
  4. AUCPR of PDB: all residues, weight each chain by
        W_structure_PDB = W_sequence / (W_sequence/W_structure - 1)
     (= W_sequence / N_PDB_total, so each protein contributes W_sequence in total)

Usage:
  python stratify_af_vs_pdb.py \\
    --split_txts split1.txt ... split5.txt \\
    --results_pkl fold1/test_results.pkl ... fold5/test_results.pkl \\
    --dataset_csv dataset.csv \\
    --protein_json all_protein_table_modified.json \\
    --out_dir /path/to/output \\
    [--n_boot 200] [--seed 0]
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from stratify_utils import (
    is_pdb_identifier,
    ensure_1d_labels, ensure_1d_preds,
    load_results_pkls,
    aucpr_like_make_PR_curve, bootstrap_ci_aucpr_group,
    aucroc_chain_weighted, bootstrap_ci_aucroc_group,
    max_f1_recall_chain_weighted, bootstrap_ci_f1_group,
    barplot_with_ci,
    build_split_map, load_dataset_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _make_aligner() -> PairwiseAligner:
    aln = PairwiseAligner()
    aln.mode = "global"
    aln.match_score = 1
    aln.mismatch_score = -1
    aln.open_gap_score = -0.75
    aln.extend_gap_score = -0.5
    return aln


def _load_pdb_to_uniprot(json_path: str) -> Dict[str, str]:
    """Build {pdb_id_upper → uniprot_id} from all_protein_table_modified.json."""
    with open(json_path) as fh:
        data = json.load(fh)
    pdb_to_uid: Dict[str, str] = {}
    for uid, entry in data.items():
        for pdb_id in entry.get("pdb_ids", []):
            pdb_to_uid.setdefault(pdb_id.upper(), uid)
    log.info("loaded JSON: %d uniprot entries, %d pdb→uniprot mappings",
             len(data), len(pdb_to_uid))
    return pdb_to_uid


def _coverage_mask(
    af_rows: List[Dict[str, Any]],
    pdb_chains: Dict[str, List[Dict[str, Any]]],
    aligner: PairwiseAligner,
) -> Set[int]:
    """
    Align each PDB chain sequence against the AF chain sequence.
    Returns set of 0-based indices into af_rows that are covered by ≥1 PDB chain.
    """
    af_seq = "".join(r["aa"] for r in af_rows)
    covered: Set[int] = set()
    for pdb_sid, pdb_rows in pdb_chains.items():
        pdb_seq = "".join(r["aa"] for r in pdb_rows)
        if not pdb_seq:
            continue
        try:
            aln = next(iter(aligner.align(af_seq, pdb_seq)))
        except (StopIteration, Exception):
            log.debug("alignment failed for PDB chain %s", pdb_sid)
            continue
        for (u_start, u_end) in aln.aligned[0]:
            covered.update(range(u_start, u_end))
    return covered


def run(
    split_txts:    List[str],
    results_pkls:  List[str],
    dataset_csv:   str,
    protein_json:  str,
    out_dir:       str,
    n_boot:        int = 200,
    seed:          int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    split_map   = build_split_map(split_txts)
    results_map = load_results_pkls(results_pkls)
    dataset_df  = load_dataset_csv(dataset_csv)
    pdb_to_uid  = _load_pdb_to_uniprot(protein_json)

    # Index dataset_df by Sequence_ID for fast lookup
    ds_idx = dataset_df.drop_duplicates("Sequence_ID").set_index("Sequence_ID")

    # ── group chains by UniProt protein ───────────────────────────────────────
    groups: Dict[str, Dict[str, Dict]] = defaultdict(lambda: {"af": {}, "pdb": {}})

    for seq_id, rows in split_map.items():
        base = seq_id.split("_")[0].upper()
        if not is_pdb_identifier(base):
            groups[base]["af"][seq_id] = rows
        else:
            uid = pdb_to_uid.get(base)
            if uid:
                groups[uid]["pdb"][seq_id] = rows
            else:
                log.debug("no UniProt mapping for PDB %s (chain %s)", base, seq_id)

    proteins_both = {uid: g for uid, g in groups.items() if g["af"] and g["pdb"]}
    log.info("proteins with both AF and PDB in test set: %d", len(proteins_both))

    aligner = _make_aligner()

    af_labels:  List[np.ndarray] = []
    af_preds:   List[np.ndarray] = []
    af_weights: List[float]      = []

    pdb_labels:  List[np.ndarray] = []
    pdb_preds:   List[np.ndarray] = []
    pdb_weights: List[float]      = []

    n_af_chains_kept = n_pdb_chains_kept = 0

    for uid, group in proteins_both.items():
        # Use first AF chain as the reference sequence for coverage alignment
        first_af_sid = next(iter(group["af"]))
        first_af_rows = group["af"][first_af_sid]
        coverage = _coverage_mask(first_af_rows, group["pdb"], aligner)

        if not coverage:
            log.debug("%s: empty coverage mask, skipping", uid)
            continue

        # ── AF chains ─────────────────────────────────────────────────────────
        for af_sid, af_rows in group["af"].items():
            if af_sid not in results_map:
                continue
            rec = results_map[af_sid]
            if len(rec["labels"]) != len(af_rows):
                log.warning("AF length mismatch %s: pred=%d rows=%d",
                            af_sid, len(rec["labels"]), len(af_rows))
                continue

            mask_idx = [i for i in range(len(af_rows)) if i in coverage]
            if not mask_idx:
                continue

            mi = np.array(mask_idx, dtype=int)
            y  = rec["labels"][mi]
            p  = rec["preds"][mi]

            if af_sid in ds_idx.index:
                w_seq = float(ds_idx.loc[af_sid, "W_Sequence"])
            else:
                w_seq = float(rec["weight"])
                log.debug("AF %s not in dataset_csv, using W_Structure as weight", af_sid)

            af_labels.append(y)
            af_preds.append(p)
            af_weights.append(w_seq)
            n_af_chains_kept += 1

        # ── PDB chains ────────────────────────────────────────────────────────
        for pdb_sid, pdb_rows in group["pdb"].items():
            if pdb_sid not in results_map:
                continue
            rec = results_map[pdb_sid]

            if pdb_sid in ds_idx.index:
                w_seq = float(ds_idx.loc[pdb_sid, "W_Sequence"])
                w_str = float(ds_idx.loc[pdb_sid, "W_Structure"])
                n_total = w_seq / w_str if w_str > 0 else 1.0
                n_pdb   = n_total - 1.0
                w_pdb   = w_seq / n_pdb if n_pdb > 0 else w_seq
            else:
                w_pdb = float(rec["weight"])
                log.debug("PDB %s not in dataset_csv, using W_Structure as weight", pdb_sid)

            pdb_labels.append(rec["labels"])
            pdb_preds.append(rec["preds"])
            pdb_weights.append(w_pdb)
            n_pdb_chains_kept += 1

    log.info("AF chains used: %d (masked to covered residues)", n_af_chains_kept)
    log.info("PDB chains used: %d", n_pdb_chains_kept)

    # ── metrics ───────────────────────────────────────────────────────────────
    rows_out = []
    for grp_name, lbls, prds, wts in [
        ("AF (masked)", af_labels, af_preds, af_weights),
        ("PDB", pdb_labels, pdb_preds, pdb_weights),
    ]:
        if not lbls:
            log.warning("%s: no data", grp_name)
            rows_out.append({"group": grp_name,
                              "AUCPR": np.nan, "AUCPR_ci_lo": np.nan, "AUCPR_ci_hi": np.nan,
                              "AUCROC": np.nan, "AUCROC_ci_lo": np.nan, "AUCROC_ci_hi": np.nan,
                              "n_chains": 0, "n_residues": 0})
            continue

        au_pr = aucpr_like_make_PR_curve(lbls, prds, wts)
        lo_pr, hi_pr = bootstrap_ci_aucpr_group(lbls, prds, wts, n_boot=n_boot, seed=seed)
        au_roc = aucroc_chain_weighted(lbls, prds, wts)
        lo_roc, hi_roc = bootstrap_ci_aucroc_group(lbls, prds, wts, n_boot=n_boot, seed=seed)
        max_f1, recall_f1 = max_f1_recall_chain_weighted(lbls, prds, wts)
        lo_f1, hi_f1 = bootstrap_ci_f1_group(lbls, prds, wts, n_boot=n_boot, seed=seed)
        n_res = sum(len(l) for l in lbls)
        n_pos = sum(int(l.sum()) for l in lbls)
        prev  = n_pos / n_res if n_res > 0 else np.nan
        aucpr_norm = (au_pr - prev) / (1 - prev) if (prev is not np.nan and prev < 1) else np.nan

        rows_out.append({
            "group": grp_name,
            "AUCPR": au_pr, "AUCPR_ci_lo": lo_pr, "AUCPR_ci_hi": hi_pr,
            "AUCPR_norm": aucpr_norm,
            "AUCROC": au_roc, "AUCROC_ci_lo": lo_roc, "AUCROC_ci_hi": hi_roc,
            "max_F1": max_f1, "F1_ci_lo": lo_f1, "F1_ci_hi": hi_f1,
            "Recall_at_F1": recall_f1,
            "n_chains": len(lbls), "n_residues": n_res, "prevalence": prev,
        })
        log.info("%s: AUCPR=%.4f  AUCPR_norm=%.4f  AUCROC=%.4f  F1=%.4f  Recall=%.4f",
                 grp_name, au_pr, aucpr_norm, au_roc, max_f1, recall_f1)

    metrics = pd.DataFrame(rows_out)
    metrics_csv = os.path.join(out_dir, "metrics_af_vs_pdb.csv")
    metrics.to_csv(metrics_csv, index=False)
    log.info("wrote %s", metrics_csv)

    # ── plots ─────────────────────────────────────────────────────────────────
    groups_list  = metrics["group"].tolist()
    colors = ["#7BC8F6", "#FF9A5C"]

    barplot_with_ci(
        values=metrics["AUCPR"].fillna(0).tolist(),
        labels=groups_list,
        ci_lo=metrics["AUCPR_ci_lo"].tolist(),
        ci_hi=metrics["AUCPR_ci_hi"].tolist(),
        ylabel="AUCPR",
        title="AF vs PDB — AUCPR",
        out_path=os.path.join(out_dir, "aucpr_af_vs_pdb.png"),
        color=colors[0],
    )
    barplot_with_ci(
        values=metrics["AUCROC"].fillna(0.5).tolist(),
        labels=groups_list,
        ci_lo=metrics["AUCROC_ci_lo"].tolist(),
        ci_hi=metrics["AUCROC_ci_hi"].tolist(),
        ylabel="AUC-ROC",
        title="AF vs PDB — AUC-ROC",
        out_path=os.path.join(out_dir, "aucroc_af_vs_pdb.png"),
        color=colors[0],
    )
    log.info("all outputs in %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split_txts",   nargs="+", required=True)
    ap.add_argument("--results_pkl",  nargs="+", required=True)
    ap.add_argument("--dataset_csv",  required=True,
                    help="CSV with Sequence_ID, W_Sequence, W_Structure columns")
    ap.add_argument("--protein_json", required=True,
                    help="all_protein_table_modified.json (for pdb_id→uniprot mapping)")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    run(
        split_txts=args.split_txts,
        results_pkls=args.results_pkl,
        dataset_csv=args.dataset_csv,
        protein_json=args.protein_json,
        out_dir=args.out_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
