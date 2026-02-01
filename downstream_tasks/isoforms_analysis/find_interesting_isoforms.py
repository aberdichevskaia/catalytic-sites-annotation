#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find base_ids where a catalytic-site prediction disappears in one isoform,
even though the aligned residue is present in another isoform.

Input:
  - CSV file (result of previous filtering), with at least columns:
      - "protein id (uniprot / PDB)"      : isoform-level ID (e.g. O14733-2)
      - "base uniprot id"                 : base UniProt ID (e.g. O14733)
      - "predicted with XX% threshold"    : per-residue predictions for a given threshold

Output:
  - CSV file with the same columns, but only for those base IDs
    where there exists at least one residue that:
       - is predicted in one isoform at the given threshold,
       - aligns to the same amino acid in another isoform,
       - is NOT predicted in that other isoform.

Usage example:
  python3 find_isoform_losses.py \
      --in_csv isoforms_cv_catalytic_thr65_var.csv \
      --out_csv isoforms_cv_catalytic_thr65_loss.csv \
      --thr 65 \
      --pdb_dir /path/to/isoform_pdbs
"""

import os
import argparse
import numpy as np
import pandas as pd

import ast
import re

import biotite.structure as struc
from biotite.structure.info import one_letter_code
from biotite.structure.io.pdb import PDBFile

from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal


# ---------- Alignment parameters (identity matrix, strong gap penalty) ----------

_ALPH = ProteinSequence.alphabet
_N = len(_ALPH)
_SUB_RAW = np.full((_N, _N), -100, dtype=int)
for _i in range(_N):
    _SUB_RAW[_i, _i] = 1
SUB_MAT = SubstitutionMatrix(_ALPH, _ALPH, _SUB_RAW)
GAP_PEN = (-100, -10)


# ---------- Helpers ----------

def parse_predicted_positions(cell) -> set[int]:
    """
    Parse predictions of the form:
      "['0', 'A', '306']"
      "['0', 'A', '186'],['0', 'A', '244']"
    We keep only the last number from each triplet (306, 186, 244, ...).

    Returns a set of positions (int).
    """
    if cell is None:
        return set()

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return set()

    # Try parsing as a list of triplets using ast.literal_eval:
    #  - "['0','A','306']"                 -> [['0','A','306']]
    #  - "['0','A','186'],['0','A','244']" -> [['0','A','186'], ['0','A','244']]
    try:
        text = s
        if not (text.startswith("[") and text.endswith("]")):
            text = "[" + text + "]"
        v = ast.literal_eval(text)
    except Exception:
        # Fallback: if the format changes, extract all positive numbers via regex.
        nums = {int(m.group(0)) for m in re.finditer(r"\d+", s) if int(m.group(0)) > 0}
        return nums

    positions: set[int] = set()

    if isinstance(v, list):
        for elem in v:
            # Expect a list/tuple like ['0','A','306'].
            if isinstance(elem, (list, tuple)) and len(elem) >= 3:
                try:
                    pos = int(elem[2])
                    if pos > 0:
                        positions.add(pos)
                except Exception:
                    pass
            else:
                # If not a triplet, attempt to extract numbers defensively.
                try:
                    # If elem is already a number or a numeric string.
                    pos = int(elem)
                    if pos > 0:
                        positions.add(pos)
                except Exception:
                    # If it's a nested string like "['0','A','306']", extract numbers.
                    elem_s = str(elem)
                    for m in re.finditer(r"\d+", elem_s):
                        val = int(m.group(0))
                        if val > 0:
                            positions.add(val)

    return positions



def load_isoform_sequence(isoform_id: str, pdb_dir: str) -> str:
    """
    Load AA sequence for isoform from PDB file.

    Strategy:
      - read {pdb_dir}/{isoform_id}.pdb
      - choose the chain with the largest number of amino-acid residues
      - return 1-letter AA sequence for that chain
    """
    path = os.path.join(pdb_dir, f"{isoform_id}.pdb")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDB file not found for {isoform_id}: {path}")

    pdb_file = PDBFile.read(path)
    atoms = pdb_file.get_structure(model=1)
    if atoms is None or atoms.array_length() == 0:
        raise ValueError(f"Empty structure for {isoform_id}")

    chain_ids = np.unique(atoms.chain_id)
    best_seq = None
    best_len = 0

    for chain_id in chain_ids:
        chain_atoms = atoms[atoms.chain_id == chain_id]
        chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
        if chain_atoms.array_length() == 0:
            continue

        # Get 3-letter residue names
        _, res_names = struc.get_residues(chain_atoms)
        seq_chars = []
        for res3 in res_names:
            try:
                aa1 = one_letter_code(res3)
            except KeyError:
                aa1 = "X"
            seq_chars.append(aa1)

        seq = "".join(seq_chars)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    if best_seq is None:
        raise ValueError(f"No amino-acid sequence found for {isoform_id}")

    return best_seq


def compute_mapping(seq1: str, seq2: str) -> dict[int, int]:
    """
    Align seq1 to seq2 and return mapping:
       1-based index in seq1 -> 1-based index in seq2
    for positions that are aligned (non-gap in both sequences).
    """
    s1 = ProteinSequence(seq1)
    s2 = ProteinSequence(seq2)
    aln = align_optimal(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]
    g1, g2 = aln.get_gapped_sequences()

    mapping: dict[int, int] = {}
    i1 = 0
    i2 = 0
    for a, b in zip(g1, g2):
        idx1 = None
        idx2 = None

        if a != "-":
            i1 += 1
            idx1 = i1
        if b != "-":
            i2 += 1
            idx2 = i2

        if idx1 is not None and idx2 is not None:
            mapping[idx1] = idx2

    return mapping


def group_has_loss_event(
    base_id: str,
    group_df: pd.DataFrame,
    thr_col: str,
    pdb_dir: str,
    id_col: str,
    seq_cache: dict[str, str],
) -> bool:
    """
    For a given base_id, check if there exists a residue that:
      - is predicted in one isoform,
      - aligns to the same amino acid in another isoform,
      - but is not predicted there.
    Returns True if such a "loss" is found.
    """
    isoform_ids = list(group_df[id_col].unique())
    if len(isoform_ids) < 2:
        return False

    # Predictions per isoform
    preds: dict[str, set[int]] = {}
    for iso in isoform_ids:
        # assume one row per isoform in this CSV; take the first if duplicates
        row_mask = group_df[id_col] == iso
        cell = group_df.loc[row_mask, thr_col].iloc[0]
        preds[iso] = parse_predicted_positions(cell)

    # Load sequences (with caching)
    for iso in isoform_ids:
        if iso not in seq_cache:
            try:
                seq_cache[iso] = load_isoform_sequence(iso, pdb_dir)
            except Exception as exc:
                print(f"[WARN] base_id={base_id}, isoform={iso}: failed to load sequence ({exc})")
                return False  # cannot reliably analyse this group

    # Alignment cache to avoid recomputing
    align_cache: dict[tuple[str, str], dict[int, int]] = {}

    # Check for losses
    for i, iso_from in enumerate(isoform_ids):
        seq_from = seq_cache[iso_from]
        pred_from = preds[iso_from]
        if not pred_from:
            # nothing to lose from this isoform
            continue

        len_from = len(seq_from)

        for j, iso_to in enumerate(isoform_ids):
            if i == j:
                continue

            seq_to = seq_cache[iso_to]
            pred_to = preds[iso_to]
            len_to = len(seq_to)

            key = (iso_from, iso_to)
            if key not in align_cache:
                try:
                    align_cache[key] = compute_mapping(seq_from, seq_to)
                except Exception as exc:
                    print(
                        f"[WARN] base_id={base_id}, "
                        f"align {iso_from} -> {iso_to} failed ({exc})"
                    )
                    return False

            mapping = align_cache[key]

            # For each predicted site in iso_from, check its aligned position in iso_to
            for pos_from in pred_from:
                if pos_from < 1 or pos_from > len_from:
                    continue

                pos_to = mapping.get(pos_from)
                if pos_to is None or pos_to < 1 or pos_to > len_to:
                    # not aligned to a residue (gap or outside)
                    continue

                aa_from = seq_from[pos_from - 1]
                aa_to = seq_to[pos_to - 1]

                # Require the same amino acid (site still "present")
                if aa_from != aa_to:
                    continue

                # If the site is not predicted in iso_to -> we found a loss
                if pos_to not in pred_to:
                    return True

    return False


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter isoform prediction table to base IDs where a catalytic-site "
            "prediction disappears in one isoform, while the aligned residue "
            "is still present in another isoform."
        )
    )
    parser.add_argument("--in_csv", required=True, help="Input CSV (filtered isoform table)")
    parser.add_argument("--out_csv", required=True, help="Output CSV with base_ids having loss events")
    parser.add_argument(
        "--thr",
        type=int,
        choices=[35, 58, 65, 85],
        required=True,
        help="Threshold in percent: 35, 58, 65, or 85 (matching column 'predicted with XX%% threshold')",
    )
    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="Directory with isoform PDB files.",
    )
    parser.add_argument(
        "--id_col",
        default="protein id (uniprot / PDB)",
        help="Column name for isoform IDs (default: 'protein id (uniprot / PDB)')",
    )
    parser.add_argument(
        "--base_col",
        default="base uniprot id",
        help="Column name for base UniProt ID (default: 'base uniprot id')",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    thr_col = f"predicted with {args.thr}% threshold"
    if thr_col not in df.columns:
        raise SystemExit(f"Column '{thr_col}' not found in CSV.")

    if args.id_col not in df.columns:
        raise SystemExit(f"Column '{args.id_col}' (isoform ID) not found in CSV.")

    if args.base_col not in df.columns:
        raise SystemExit(f"Column '{args.base_col}' (base ID) not found in CSV.")

    seq_cache: dict[str, str] = {}
    base_ids_keep: set[str] = set()

    grouped = df.groupby(args.base_col)

    for base_id, group in grouped:
        if group.shape[0] < 2:
            continue

        has_loss = group_has_loss_event(
            base_id=base_id,
            group_df=group,
            thr_col=thr_col,
            pdb_dir=args.pdb_dir,
            id_col=args.id_col,
            seq_cache=seq_cache,
        )
        if has_loss:
            base_ids_keep.add(base_id)

    df_out = df[df[args.base_col].isin(base_ids_keep)].copy()
    df_out.to_csv(args.out_csv, index=False)

    print(
        f"[OK] kept {len(df_out)} rows "
        f"for {len(base_ids_keep)} base IDs -> {args.out_csv}"
    )


if __name__ == "__main__":
    main()
