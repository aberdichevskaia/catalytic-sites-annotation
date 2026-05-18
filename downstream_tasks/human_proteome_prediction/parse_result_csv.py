#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter a prediction table where residue entries are formatted like:
  ['0', 'A', '716'],['0','A','836']
— taking ONLY the third element of each triplet (the residue index).

Expected columns (default names):
- "protein id (uniprot / PDB)"
- "predicted with 35% threshold"
- "predicted with 65% threshold"
- "predicted with 85% threshold"
- "known catalytic sites"
- "EC number (if exists)"
- "protein name"
- "gene name"

Output:
- CSV containing only rows with at least one novel position,
  plus columns novel_sites_35/65/85, n_novel_35/65/85, any_novel.
"""

import argparse
import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Match ONLY the third element inside each triplet ['..','..','123']
TRIPLET_IDX_RE = re.compile(r"\[\s*'[^']*'\s*,\s*'[^']*'\s*,\s*'(\d+)'\s*\]")
# Generic number extractor — for simple lists like "12,34"
ANY_NUM_RE = re.compile(r"\d+")

def parse_pred_triplets(cell) -> set[int]:
    """
    For prediction columns: extract the THIRD element from each triplet.
    Examples:
      "['0','A','125']"                       -> {125}
      "['0','A','216'],['0','A','236']"       -> {216,236}
      "" / NaN                                 -> {}
    Falls back to extracting any numbers if triplets are not recognised.
    """
    if cell is None:
        return set()
    if isinstance(cell, float) and pd.isna(cell):
        return set()
    s = str(cell)
    hits = TRIPLET_IDX_RE.findall(s)
    if hits:
        return {int(x) for x in hits}
    # fallback: extract any numbers (in case the format differs)
    return {int(x) for x in ANY_NUM_RE.findall(s)}

def parse_known_any(cell) -> set[int]:
    """
    For the 'known catalytic sites' column: extract all numbers.
    Examples:
      "125, 236"   -> {125,236}
      "" / NaN     -> {}
    """
    if cell is None:
        return set()
    if isinstance(cell, float) and pd.isna(cell):
        return set()
    return {int(x) for x in ANY_NUM_RE.findall(str(cell))}

def main():
    ap = argparse.ArgumentParser(description="Filter rows with novel (non-UniProt) catalytic sites (triplet format).")
    ap.add_argument("--in_csv", required=True, help="Input CSV from predict.py")
    ap.add_argument("--out_csv", required=True, help="Output CSV (novel sites only)")
    ap.add_argument("--pred_cols", nargs="+",
                    default=["predicted with 35% threshold",
                             "predicted with 65% threshold",
                             "predicted with 85% threshold"],
                    help="List of prediction columns (as produced by predict.py)")
    ap.add_argument("--col_known", default="known catalytic sites",
                    help="Column with UniProt-known positions")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    known_sets = df[args.col_known].apply(parse_known_any) if args.col_known in df.columns else pd.Series([set()]*len(df))

    novel_cols = []
    count_cols = []
    any_novel_mask = pd.Series(False, index=df.index)

    for col in args.pred_cols:
        if col not in df.columns:
            continue
        preds = df[col].apply(parse_pred_triplets)
        novel = [sorted(p - k) for p, k in zip(preds, known_sets)]
        tag = re.search(r"(\d+)%", col)
        suffix = tag.group(1) if tag else col.replace("predicted with ", "").replace(" threshold", "").strip()
        col_sites = f"novel_sites_{suffix}"
        col_count = f"n_novel_{suffix}"
        df[col_sites] = [",".join(map(str, x)) if x else "" for x in novel]
        df[col_count] = [len(x) for x in novel]
        novel_cols.append(col_sites)
        count_cols.append(col_count)
        any_novel_mask = any_novel_mask | (df[col_count] > 0)

    df["any_novel"] = any_novel_mask

    out = df[df["any_novel"]].copy()

    sort_by = [c for c in ["n_novel_85", "n_novel_65", "n_novel_35"] if c in out.columns]
    if sort_by:
        out = out.sort_values(sort_by, ascending=False)

    out.to_csv(args.out_csv, index=False)
    logging.info("%s rows with novel catalytic sites -> %s", len(out), args.out_csv)

if __name__ == "__main__":
    main()
