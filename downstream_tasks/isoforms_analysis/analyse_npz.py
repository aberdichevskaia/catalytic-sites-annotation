#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU stage: analyse isoform predictions from a single NPZ file.

For each base_id group (ref isoform + others) this script:
  1. Aligns each non-reference isoform to the reference (biotite, AA-match filter).
  2. Computes diff_score — how much the isoform diverges from the reference:

       diff_score = Σ_{r: aligned, AA-match} |p_ref(r) − p_iso(r)| / Σ_r p_ref(r)

     The numerator sums only over positions that are aligned with identical AA,
     so deletions and mismatches do not contribute (those are captured separately
     by lost_missing). The denominator is the total reference probability mass.
     Score is 0 for the reference isoform itself; NaN when no aligned positions.

  3. Categorises each reference site (p_ref ≥ tau) as:
       retained       — aligned with AA-match and p_iso ≥ tau
       lost_aligned   — aligned with AA-match but p_iso < tau
       lost_missing   — not aligned / AA mismatch

     Isoform category:
       no_loss              — lost_aligned == 0 and lost_missing == 0
       loss_missing_only    — lost_missing > 0  and lost_aligned == 0
       loss_aligned_present — lost_aligned > 0

Output (one row per isoform, reference isoform included):
  base_id, ref_isoform, isoform, diff_score,
  n_ref_sites, retained, lost_aligned, lost_missing, category,
  retained_pos_ref1, lost_missing_pos_ref1, lost_aligned_pos_ref1,
  n_aligned_aa_match, note

Parallelisation: base_id groups are processed in a multiprocessing Pool (fork).
"""

import argparse
import logging
import os
from multiprocessing import Pool, get_context
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.info import one_letter_code
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal

try:
    from biotite.sequence.align import align_local  # type: ignore
except Exception:
    align_local = None


# ── Alignment constants ────────────────────────────────────────────────────────
_ALPH = ProteinSequence.alphabet
_N = len(_ALPH)
_SUB_RAW = np.full((_N, _N), -100, dtype=int)
for _i in range(_N):
    _SUB_RAW[_i, _i] = 1
SUB_MAT = SubstitutionMatrix(_ALPH, _ALPH, _SUB_RAW)
GAP_PEN = (-100, -10)


# ── Globals shared across forked workers ──────────────────────────────────────
ISOFORM_IDS: np.ndarray
BASE_IDS: np.ndarray
PDB_PATHS: np.ndarray
OFFSETS: np.ndarray
PROB_CONCAT: np.ndarray
ISO_TO_IDX: Dict[str, int] = {}
SEQ_MAP: Dict[str, str] = {}
TAU: float = 0.35


# ── Helpers ───────────────────────────────────────────────────────────────────

def choose_reference_isoform(iso_ids: List[str]) -> str:
    iso_ids = sorted(set(iso_ids))
    for x in iso_ids:
        if x.endswith("-1"):
            return x

    def _key(x: str) -> Tuple[int, str]:
        if "-" in x:
            suf = x.split("-", 1)[1]
            try:
                return (int(suf), x)
            except Exception:
                pass
        return (10 ** 9, x)

    return sorted(iso_ids, key=_key)[0]


def load_best_chain_sequence(pdb_path: str) -> str:
    pdb_file = PDBFile.read(pdb_path)
    atoms = pdb_file.get_structure(model=1)
    if atoms is None or atoms.array_length() == 0:
        raise ValueError(f"empty structure: {pdb_path}")

    best_seq = ""
    best_len = 0
    for chain_id in np.unique(atoms.chain_id):
        chain = atoms[atoms.chain_id == chain_id]
        chain = chain[struc.filter_amino_acids(chain)]
        if chain.array_length() == 0:
            continue
        _, res_names = struc.get_residues(chain)
        seq = "".join(
            (one_letter_code(r) if r in one_letter_code.__self__ else "X")
            for r in res_names
        )
        if len(seq) > best_len:
            best_len, best_seq = len(seq), seq

    if not best_seq:
        raise ValueError(f"no amino-acid residues: {pdb_path}")
    return best_seq


def _seq_to_s1(seq: str) -> np.ndarray:
    return np.frombuffer(seq.encode("ascii", errors="ignore"), dtype="S1")


def compute_alignment(seq_ref: str, seq_iso: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ref_idx, iso_idx) 0-based arrays for aligned non-gap positions."""
    s1 = ProteinSequence(seq_ref)
    s2 = ProteinSequence(seq_iso)
    aln_fn = align_local if align_local is not None else align_optimal
    aln = aln_fn(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]
    g1, g2 = aln.get_gapped_sequences()

    ref_pos = iso_pos = 0
    ri: List[int] = []
    ii: List[int] = []
    for a, b in zip(g1, g2):
        if a != "-":
            ref_pos += 1
        if b != "-":
            iso_pos += 1
        if a != "-" and b != "-":
            ri.append(ref_pos - 1)
            ii.append(iso_pos - 1)

    if not ri:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.array(ri, dtype=np.int32), np.array(ii, dtype=np.int32)


def get_prob(iso: str) -> np.ndarray:
    i = ISO_TO_IDX.get(iso)
    if i is None:
        return np.zeros(0, dtype=np.float32)
    s, e = int(OFFSETS[i]), int(OFFSETS[i + 1])
    return PROB_CONCAT[s:e].astype(np.float32, copy=False)


def _fmt(pos0: List[int]) -> str:
    return ", ".join(str(p + 1) for p in sorted(pos0))


# ── Per-isoform row builder ────────────────────────────────────────────────────

def _make_missing_row(
    base_id: str, ref: str, iso: str, n_ref_sites: int, ref_sites: List[int], note: str
) -> Dict[str, Any]:
    return {
        "base_id": base_id, "ref_isoform": ref, "isoform": iso,
        "diff_score": float("nan"),
        "n_ref_sites": n_ref_sites,
        "retained": 0, "lost_aligned": 0, "lost_missing": n_ref_sites,
        "category": "loss_missing_only",
        "retained_pos_ref1": "",
        "lost_aligned_pos_ref1": "",
        "lost_missing_pos_ref1": _fmt(ref_sites),
        "n_aligned_aa_match": 0,
        "note": note,
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def analyse_one_base(task: Tuple[str, List[str]]) -> List[Dict[str, Any]]:
    base_id, iso_ids = task
    iso_ids = sorted(set(iso_ids))
    ref = choose_reference_isoform(iso_ids)

    seq_ref = SEQ_MAP.get(ref, "")
    p_ref_full = get_prob(ref)
    if not seq_ref or p_ref_full.size == 0:
        return []

    L = min(len(seq_ref), p_ref_full.size)
    seq_ref = seq_ref[:L]
    p_ref = p_ref_full[:L]

    ref_s1 = _seq_to_s1(seq_ref)
    ref_prob_sum = float(p_ref.sum())

    # Reference sites for categorisation
    ref_sites_mask = p_ref >= TAU
    ref_sites = np.where(ref_sites_mask)[0].tolist()
    n_ref_sites = len(ref_sites)

    rows: List[Dict[str, Any]] = []

    # Row for the reference isoform itself
    rows.append({
        "base_id": base_id, "ref_isoform": ref, "isoform": ref,
        "diff_score": 0.0,
        "n_ref_sites": n_ref_sites,
        "retained": n_ref_sites, "lost_aligned": 0, "lost_missing": 0,
        "category": "reference",
        "retained_pos_ref1": _fmt(ref_sites),
        "lost_aligned_pos_ref1": "",
        "lost_missing_pos_ref1": "",
        "n_aligned_aa_match": L,
        "note": "",
    })

    for iso in iso_ids:
        if iso == ref:
            continue

        seq_iso = SEQ_MAP.get(iso, "")
        if not seq_iso:
            rows.append(_make_missing_row(base_id, ref, iso, n_ref_sites, ref_sites, "no_seq"))
            continue

        p_iso_full = get_prob(iso)
        if p_iso_full.size == 0:
            rows.append(_make_missing_row(base_id, ref, iso, n_ref_sites, ref_sites, "no_probs"))
            continue

        L_iso = min(len(seq_iso), p_iso_full.size)
        seq_iso = seq_iso[:L_iso]
        p_iso = p_iso_full[:L_iso]

        ref_idx, iso_idx = compute_alignment(seq_ref, seq_iso)
        if ref_idx.size == 0:
            rows.append(_make_missing_row(base_id, ref, iso, n_ref_sites, ref_sites, "no_alignment"))
            continue

        iso_s1 = _seq_to_s1(seq_iso)
        aa_match = ref_s1[ref_idx] == iso_s1[iso_idx]
        rr = ref_idx[aa_match]
        ii = iso_idx[aa_match]

        # diff_score: only over AA-match aligned positions (deletions excluded)
        if rr.size > 0 and ref_prob_sum > 0:
            diff_score = float(np.abs(p_ref[rr] - p_iso[ii]).sum() / ref_prob_sum)
        else:
            diff_score = float("nan")

        # Loss categorisation at ref sites (threshold-based)
        map_ref_to_iso = np.full(L, -1, dtype=np.int32)
        if rr.size > 0:
            map_ref_to_iso[rr] = ii

        retained_pos: List[int] = []
        lost_aligned_pos: List[int] = []
        lost_missing_pos: List[int] = []
        for rpos in ref_sites:
            ipos = int(map_ref_to_iso[rpos])
            if ipos < 0:
                lost_missing_pos.append(rpos)
            elif float(p_iso[ipos]) >= TAU:
                retained_pos.append(rpos)
            else:
                lost_aligned_pos.append(rpos)

        if not lost_aligned_pos and not lost_missing_pos:
            category = "no_loss"
        elif not lost_aligned_pos:
            category = "loss_missing_only"
        else:
            category = "loss_aligned_present"

        rows.append({
            "base_id": base_id, "ref_isoform": ref, "isoform": iso,
            "diff_score": diff_score,
            "n_ref_sites": n_ref_sites,
            "retained": len(retained_pos),
            "lost_aligned": len(lost_aligned_pos),
            "lost_missing": len(lost_missing_pos),
            "category": category,
            "retained_pos_ref1": _fmt(retained_pos),
            "lost_aligned_pos_ref1": _fmt(lost_aligned_pos),
            "lost_missing_pos_ref1": _fmt(lost_missing_pos),
            "n_aligned_aa_match": int(rr.size),
            "note": "",
        })

    return rows


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    dump_npz: str,
    out_csv: str,
    tau: float = 0.35,
    nproc: int = 16,
    min_isoforms: int = 2,
    maxtasksperchild: int = 50,
) -> None:
    global ISOFORM_IDS, BASE_IDS, PDB_PATHS, OFFSETS, PROB_CONCAT, ISO_TO_IDX, SEQ_MAP, TAU
    TAU = float(tau)

    d = np.load(dump_npz, allow_pickle=False)
    ISOFORM_IDS = d["isoform_ids"].astype(str)
    BASE_IDS    = d["base_ids"].astype(str)
    PDB_PATHS   = d["pdb_paths"].astype(str)
    OFFSETS     = d["offsets"].astype(np.int64)
    PROB_CONCAT = d["prob_concat"]  # float16; cast to float32 in get_prob()

    if ISOFORM_IDS.shape[0] + 1 != OFFSETS.shape[0]:
        raise SystemExit("Bad NPZ: offsets length must be N+1.")

    ISO_TO_IDX = {iso: i for i, iso in enumerate(ISOFORM_IDS)}

    log.info("loading sequences from %s PDB files...", len(ISOFORM_IDS))
    SEQ_MAP = {}
    bad = 0
    for iso, pdb_path in zip(ISOFORM_IDS, PDB_PATHS):
        try:
            SEQ_MAP[iso] = load_best_chain_sequence(pdb_path)
        except Exception as exc:
            bad += 1
            if bad <= 10:
                log.warning("seq load failed %s (%s): %s", iso, pdb_path, exc)
    if bad:
        log.warning("%s isoforms had sequence load failures (treated as missing)", bad)

    df = pd.DataFrame({"isoform_id": ISOFORM_IDS, "base_id": BASE_IDS})
    base_groups: List[Tuple[str, List[str]]] = [
        (str(bid), sorted(g["isoform_id"].astype(str).unique().tolist()))
        for bid, g in df.groupby("base_id")
        if len(g) >= min_isoforms
    ]
    log.info("base groups: %s (min_isoforms=%s, tau=%.2f)", len(base_groups), min_isoforms, TAU)

    try:
        ctx = get_context("fork")
        PoolCls = ctx.Pool
    except Exception:
        PoolCls = Pool

    all_rows: List[Dict[str, Any]] = []
    with PoolCls(processes=nproc, maxtasksperchild=maxtasksperchild) as pool:
        for rows in pool.imap_unordered(analyse_one_base, base_groups, chunksize=1):
            all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log.info("wrote %s rows -> %s", len(out_df), out_csv)

    # Summary
    compared = out_df[out_df["category"] != "reference"]
    total = len(compared)
    if total:
        counts = compared["category"].value_counts()
        log.info(
            "summary (tau=%.2f): no_loss=%s  loss_missing_only=%s  loss_aligned_present=%s  (n=%s)",
            TAU,
            counts.get("no_loss", 0),
            counts.get("loss_missing_only", 0),
            counts.get("loss_aligned_present", 0),
            total,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPU stage: compute diff_score + loss categories from isoform NPZ."
    )
    ap.add_argument("--dump_npz",        required=True, help="NPZ from run_inference.py.")
    ap.add_argument("--out_csv",         required=True, help="Output CSV path.")
    ap.add_argument("--tau",             type=float, default=0.35,
                    help="Threshold for retained/lost_aligned/lost_missing (default 0.35).")
    ap.add_argument("--nproc",           type=int, default=16)
    ap.add_argument("--min_isoforms",    type=int, default=2)
    ap.add_argument("--maxtasksperchild",type=int, default=50)
    args = ap.parse_args()

    run(
        dump_npz=args.dump_npz,
        out_csv=args.out_csv,
        tau=args.tau,
        nproc=args.nproc,
        min_isoforms=args.min_isoforms,
        maxtasksperchild=args.maxtasksperchild,
    )


if __name__ == "__main__":
    main()
