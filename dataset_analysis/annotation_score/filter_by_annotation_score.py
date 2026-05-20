#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter dataset entries by UniProt annotation score.

Subcommands:
  uniprot        Filter a raw UniProt JSON dump to entries with annotationScore
                 above threshold and at least one Active site feature.
  protein-table  Filter an internal protein_table JSON by annotation score,
                 write filtered table + FASTA + score histogram.
"""

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── subcommand: uniprot ───────────────────────────────────────────────────────

def cmd_uniprot(args: argparse.Namespace) -> None:
    with open(args.uniprot_data) as f:
        uniprot_data = json.load(f)

    entries = uniprot_data["results"]
    filtered = [
        e for e in entries
        if e["annotationScore"] > args.min_score
        and any(feat.get("type") == "Active site" for feat in e.get("features", []))
    ]
    log.info(
        "uniprot: %s / %s entries pass (score > %s, has Active site)",
        len(filtered), len(entries), args.min_score,
    )

    with open(args.filtered_json, "w") as f:
        json.dump(filtered, f, indent=4)
    log.info("wrote -> %s", args.filtered_json)


# ── subcommand: protein-table ─────────────────────────────────────────────────

def cmd_protein_table(args: argparse.Namespace) -> None:
    with open(args.protein_table) as f:
        protein_table = json.load(f)

    with open(args.uniprot_data) as f:
        uniprot_data = json.load(f)

    score_by_acc = {
        e["primaryAccession"]: e["annotationScore"]
        for e in uniprot_data["results"]
    }

    if args.score_hist:
        scores = list(score_by_acc.values())
        unique, counts = np.unique(scores, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(unique, counts, width=0.6, edgecolor="black", alpha=0.7)
        for s, c in zip(unique, counts):
            ax.text(s, c + max(counts) * 0.04, str(c), ha="center", fontsize=11)
        ax.set_xlabel("Annotation Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Annotation Scores")
        ax.set_xticks(range(1, 6))
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.savefig(args.score_hist, dpi=500, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote histogram -> %s", args.score_hist)

    filtered = {
        uid: data
        for uid, data in protein_table.items()
        if score_by_acc.get(uid, 0) > args.min_score
    }
    log.info(
        "protein-table: %s / %s proteins pass (score > %s)",
        len(filtered), len(protein_table), args.min_score,
    )

    with open(args.filtered_json, "w") as f:
        json.dump(filtered, f, indent=4)
    log.info("wrote filtered table -> %s", args.filtered_json)

    if args.filtered_fasta:
        with open(args.filtered_fasta, "w") as f:
            for uid, data in filtered.items():
                f.write(f">{uid}\n{data['uniprot_sequence']}\n")
        log.info("wrote filtered FASTA -> %s", args.filtered_fasta)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- uniprot ---
    p_uni = sub.add_parser("uniprot", help="Filter raw UniProt JSON dump.")
    p_uni.add_argument("--uniprot_data", required=True,
                       help="Input UniProt JSON (with 'results' list).")
    p_uni.add_argument("--filtered_json", required=True,
                       help="Output path for filtered UniProt JSON.")
    p_uni.add_argument("--min_score", type=float, default=2,
                       help="Keep entries with annotationScore > this (default: 2).")

    # --- protein-table ---
    p_pt = sub.add_parser("protein-table", help="Filter internal protein_table JSON.")
    p_pt.add_argument("--protein_table", required=True,
                      help="Input protein_table JSON.")
    p_pt.add_argument("--uniprot_data", required=True,
                      help="UniProt JSON (with 'results' list) for score lookup.")
    p_pt.add_argument("--filtered_json", required=True,
                      help="Output path for filtered protein_table JSON.")
    p_pt.add_argument("--filtered_fasta", default=None,
                      help="Output FASTA path (optional).")
    p_pt.add_argument("--score_hist", default=None,
                      help="Output score histogram image path (optional).")
    p_pt.add_argument("--min_score", type=float, default=3,
                      help="Keep entries with annotationScore > this (default: 3).")

    args = ap.parse_args()
    {"uniprot": cmd_uniprot, "protein-table": cmd_protein_table}[args.cmd](args)


if __name__ == "__main__":
    main()
