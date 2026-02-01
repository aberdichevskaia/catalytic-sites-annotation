#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU stage: compute spread score per base_id from ONE dumped NPZ file.

Input: dump_npz produced by GPU stage (one file):
  - isoform_ids, base_ids, pdb_paths, offsets, prob_concat, ...

We load sequences from PDB files on CPU (choose the longest AA chain).
Then for each base_id:
  score = max_i (max_over_isoforms(p_i) - min_over_isoforms(p_i)) over aligned ref positions,
          considering only aligned positions where aa_ref == aa_iso.

Parallelization: base_id groups are processed in multiprocessing Pool.

Output CSV: columns [base id, score]
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from multiprocessing import Pool, get_context

import biotite.structure as struc
from biotite.structure.info import one_letter_code
from biotite.structure.io.pdb import PDBFile

from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal

try:
    from biotite.sequence.align import align_local  # type: ignore
except Exception:
    align_local = None


# ---------------- Alignment parameters (identity matrix, strong penalties) ----------------
_ALPH = ProteinSequence.alphabet
_N = len(_ALPH)
_SUB_RAW = np.full((_N, _N), -100, dtype=int)
for _i in range(_N):
    _SUB_RAW[_i, _i] = 1
SUB_MAT = SubstitutionMatrix(_ALPH, _ALPH, _SUB_RAW)
GAP_PEN = (-100, -10)


# ---------------- Global data for forked workers ----------------
ISOFORM_IDS: np.ndarray
BASE_IDS: np.ndarray
PDB_PATHS: np.ndarray
OFFSETS: np.ndarray
PROB_CONCAT: np.ndarray

ISO_TO_IDX: Dict[str, int] = {}
SEQ_MAP: Dict[str, str] = {}


def choose_reference_isoform(iso_ids: List[str]) -> str:
    iso_ids = sorted(set(iso_ids))
    for x in iso_ids:
        if x.endswith("-1"):
            return x

    def iso_num(x: str) -> Tuple[int, str]:
        if "-" in x:
            suf = x.split("-", 1)[1]
            try:
                return (int(suf), x)
            except Exception:
                pass
        return (10**9, x)

    return sorted(iso_ids, key=iso_num)[0]


def load_best_chain_sequence(pdb_path: str) -> str:
    """
    Extract AA sequence from a PDB:
    - choose the chain with the largest number of AA residues
    - convert 3-letter codes to 1-letter (unknown -> 'X')
    """
    pdb_file = PDBFile.read(pdb_path)
    atoms = pdb_file.get_structure(model=1)
    if atoms is None or atoms.array_length() == 0:
        raise ValueError(f"Empty structure: {pdb_path}")

    best_seq = ""
    best_len = 0

    for chain_id in np.unique(atoms.chain_id):
        chain_atoms = atoms[atoms.chain_id == chain_id]
        chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
        if chain_atoms.array_length() == 0:
            continue

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

    if not best_seq:
        raise ValueError(f"No amino-acid residues found: {pdb_path}")

    return best_seq


def compute_mapping_ref_to_iso(seq_ref: str, seq_iso: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mapping arrays (0-based):
      ref_idx: positions in ref sequence
      iso_idx: corresponding positions in iso sequence
    for aligned (non-gap in both) positions.
    """
    s1 = ProteinSequence(seq_ref)
    s2 = ProteinSequence(seq_iso)

    if align_local is not None:
        aln = align_local(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]
    else:
        aln = align_optimal(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]

    g1, g2 = aln.get_gapped_sequences()

    ref_pos = 0
    iso_pos = 0
    ref_idx_list: List[int] = []
    iso_idx_list: List[int] = []

    for a, b in zip(g1, g2):
        a_is_gap = (a == "-")
        b_is_gap = (b == "-")

        if not a_is_gap:
            ref_pos += 1
        if not b_is_gap:
            iso_pos += 1

        if (not a_is_gap) and (not b_is_gap):
            # store 0-based indices
            ref_idx_list.append(ref_pos - 1)
            iso_idx_list.append(iso_pos - 1)

    if not ref_idx_list:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    return np.asarray(ref_idx_list, dtype=np.int32), np.asarray(iso_idx_list, dtype=np.int32)


def get_prob_for_isoform(iso: str) -> np.ndarray:
    """Return float32 probability vector for isoform (view slice -> cast)."""
    i = ISO_TO_IDX.get(iso)
    if i is None:
        return np.zeros(0, dtype=np.float32)
    start = int(OFFSETS[i])
    end = int(OFFSETS[i + 1])
    if end <= start:
        return np.zeros(0, dtype=np.float32)
    # prob_concat stored as float16, cast to float32 for numeric stability
    return PROB_CONCAT[start:end].astype(np.float32, copy=False)


def _seq_to_s1_array(seq: str) -> np.ndarray:
    """Convert sequence string to numpy array of dtype 'S1' for fast comparisons."""
    # ASCII uppercase letters + 'X' are fine
    return np.frombuffer(seq.encode("ascii", errors="ignore"), dtype="S1")


def score_one_base(task):
    """
    Worker: compute score for one base_id group.
    task = (base_id, iso_ids)
    Uses global ISO_TO_IDX/SEQ_MAP/PROB_CONCAT slices.
    """
    base_id, iso_ids = task

    if len(iso_ids) < 2:
        return base_id, 0.0

    ref = choose_reference_isoform(iso_ids)
    seq_ref = SEQ_MAP.get(ref, "")
    if not seq_ref:
        return base_id, 0.0

    p_ref = get_prob_for_isoform(ref)
    L_ref = min(len(seq_ref), int(p_ref.shape[0]))
    if L_ref < 1:
        return base_id, 0.0

    seq_ref = seq_ref[:L_ref]
    p_ref = p_ref[:L_ref]

    maxv = p_ref.copy()
    minv = p_ref.copy()
    counts = np.ones(L_ref, dtype=np.int32)

    ref_s1 = _seq_to_s1_array(seq_ref)

    for iso in iso_ids:
        if iso == ref:
            continue

        seq_iso = SEQ_MAP.get(iso, "")
        if not seq_iso:
            continue

        p_iso = get_prob_for_isoform(iso)
        L_iso = min(len(seq_iso), int(p_iso.shape[0]))
        if L_iso < 1:
            continue

        seq_iso = seq_iso[:L_iso]
        p_iso = p_iso[:L_iso]

        ref_idx, iso_idx = compute_mapping_ref_to_iso(seq_ref, seq_iso)
        if ref_idx.size == 0:
            continue

        iso_s1 = _seq_to_s1_array(seq_iso)

        # filter by AA match at aligned positions
        # (require residue "present" and identical letter)
        aa_match = (ref_s1[ref_idx] == iso_s1[iso_idx])
        if not np.any(aa_match):
            continue

        rr = ref_idx[aa_match]
        ii = iso_idx[aa_match]
        vals = p_iso[ii]

        # Update max/min/counts at selected ref positions
        maxv[rr] = np.maximum(maxv[rr], vals)
        minv[rr] = np.minimum(minv[rr], vals)
        counts[rr] += 1

    spread = maxv - minv
    spread[counts < 2] = 0.0

    return base_id, float(np.max(spread)) if spread.size else 0.0


def main():
    ap = argparse.ArgumentParser(description="CPU stage: compute base_id spread scores from one dumped NPZ.")
    ap.add_argument("--dump_npz", required=True, help="NPZ produced by GPU stage (one file).")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--min_isoforms", type=int, default=2)
    ap.add_argument("--maxtasksperchild", type=int, default=50)
    ap.add_argument("--seq_cache_csv", default=None, help="Optional: save loaded sequences to CSV for reuse/debug.")
    args = ap.parse_args()

    global ISOFORM_IDS, BASE_IDS, PDB_PATHS, OFFSETS, PROB_CONCAT, ISO_TO_IDX, SEQ_MAP

    d = np.load(args.dump_npz, allow_pickle=False)
    ISOFORM_IDS = d["isoform_ids"].astype(str)
    BASE_IDS = d["base_ids"].astype(str)
    PDB_PATHS = d["pdb_paths"].astype(str)
    OFFSETS = d["offsets"].astype(np.int64)
    PROB_CONCAT = d["prob_concat"]  # keep float16 to save RAM; slice casts to float32

    if ISOFORM_IDS.shape[0] + 1 != OFFSETS.shape[0]:
        raise SystemExit("Bad dump_npz: offsets length must be N+1.")

    ISO_TO_IDX = {iso: i for i, iso in enumerate(ISOFORM_IDS)}

    # Load sequences once in parent (fast enough) -> fork shares memory (copy-on-write)
    print(f"[INFO] loading sequences from PDBs for {len(ISOFORM_IDS)} isoforms...", flush=True)
    SEQ_MAP = {}
    bad = 0
    for iso, pdb_path in zip(ISOFORM_IDS, PDB_PATHS):
        try:
            SEQ_MAP[iso] = load_best_chain_sequence(pdb_path)
        except Exception as exc:
            bad += 1
            if bad <= 10:
                print(f"[WARN] seq load failed for {iso} ({pdb_path}): {exc}", flush=True)

    if bad:
        print(f"[WARN] failed to load sequences for {bad} isoforms (they will be skipped).", flush=True)

    # Build base_id groups using only isoforms with sequences
    df = pd.DataFrame({"isoform_id": ISOFORM_IDS, "base_id": BASE_IDS})
    df = df[df["isoform_id"].isin(list(SEQ_MAP.keys()))]

    base_groups: List[Tuple[str, List[str]]] = []
    for base_id, g in df.groupby("base_id"):
        iso_ids = sorted(g["isoform_id"].astype(str).unique().tolist())
        if len(iso_ids) >= args.min_isoforms:
            base_groups.append((str(base_id), iso_ids))

    print(f"[INFO] base groups to process: {len(base_groups)}", flush=True)

    # Optional: save seq cache
    if args.seq_cache_csv:
        outp = args.seq_cache_csv
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        pd.DataFrame(
            {"isoform_id": list(SEQ_MAP.keys()), "seq": list(SEQ_MAP.values())}
        ).to_csv(outp, index=False)
        print(f"[OK] wrote seq cache -> {outp}", flush=True)

    # Use fork context to avoid duplicating big arrays in workers
    try:
        ctx = get_context("fork")
    except Exception:
        ctx = None

    rows = []
    PoolCls = (ctx.Pool if ctx is not None else Pool)

    with PoolCls(processes=args.nproc, maxtasksperchild=args.maxtasksperchild) as pool:
        for base_id, score in pool.imap_unordered(score_one_base, base_groups, chunksize=1):
            rows.append({"base id": base_id, "score": score})

    out_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {len(out_df)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
