#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isoform loss categorization at tau=0.35 from dump_npz (same style as your spread-score script),
WITH saving ref-position lists for later plots.

Definitions (for each base_id group):
- Choose reference isoform (prefer '-1', else smallest numeric suffix).
- Extract sequences from PDB (longest AA chain).
- Reference "sites" = ref positions with p_ref >= tau (0-based internally, saved as 1-based).
- For each other isoform:
    Align seq_ref vs seq_iso (biotite), take aligned positions with identical AA (aa_ref == aa_iso).
    For each reference site position r:
        - retained: mapped with AA-match AND p_iso(mapped) >= tau
        - lost_aligned: mapped with AA-match BUT p_iso(mapped) < tau
        - lost_missing: NOT mapped with AA-match (gap / no mapping / AA mismatch)

Isoform-level categories:
(a) no_loss:
    lost_missing == 0 and lost_aligned == 0
(b) loss_missing_only:
    lost_missing > 0 and lost_aligned == 0
(c) loss_aligned_present:
    lost_aligned > 0 (may also have lost_missing)

Outputs:
- per_isoform_loss_categories.csv (per isoform vs ref):
    base_id, ref_isoform, isoform, n_ref_sites,
    retained, lost_missing, lost_aligned, category,
    retained_pos_ref1, lost_missing_pos_ref1, lost_aligned_pos_ref1,
    n_aligned_aa_match, note
- per_base_meta.csv:
    base_id, skipped, reason, ref_isoform, n_ref_sites, n_isoforms,
    ref_sites_pos_ref1
- summary_loss_categories.csv:
    counts + fractions of isoforms in categories (among compared isoforms)
"""

import os
import argparse
from typing import Dict, List, Tuple, Any

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
TAU: float = 0.35


def _fmt_pos_1based(pos0: List[int]) -> str:
    """Format 0-based positions as '1, 5, 10' (1-based). Empty -> ''."""
    if not pos0:
        return ""
    pos1 = [p + 1 for p in pos0]
    pos1.sort()
    return ", ".join(str(int(x)) for x in pos1)


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
    return PROB_CONCAT[start:end].astype(np.float32, copy=False)


def _seq_to_s1_array(seq: str) -> np.ndarray:
    """Convert sequence string to numpy array of dtype 'S1' for fast comparisons."""
    return np.frombuffer(seq.encode("ascii", errors="ignore"), dtype="S1")


def analyze_one_base(task: Tuple[str, List[str]]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Worker: analyze one base_id group.
    Returns:
      base_id,
      base_meta dict,
      per_isoform rows list
    """
    base_id, iso_ids = task
    iso_ids = sorted(set(iso_ids))

    ref = choose_reference_isoform(iso_ids)
    seq_ref = SEQ_MAP.get(ref, "")
    if not seq_ref:
        return base_id, {"skipped": True, "reason": "no_ref_seq", "ref_isoform": ref}, []

    p_ref = get_prob_for_isoform(ref)
    L_ref = min(len(seq_ref), int(p_ref.shape[0]))
    if L_ref < 1:
        return base_id, {"skipped": True, "reason": "no_ref_probs", "ref_isoform": ref}, []

    seq_ref = seq_ref[:L_ref]
    p_ref = p_ref[:L_ref]

    # Reference "sites": positions where p_ref >= TAU (0-based)
    ref_sites = np.where(p_ref >= TAU)[0].astype(np.int32)
    n_ref_sites = int(ref_sites.size)

    if n_ref_sites == 0:
        # Nothing to "lose" => skip group to avoid trivial "no-loss"
        meta = {
            "skipped": True,
            "reason": "no_ref_sites",
            "ref_isoform": ref,
            "n_ref_sites": 0,
            "n_isoforms": len(iso_ids),
            "ref_sites_pos_ref1": "",
        }
        return base_id, meta, []

    ref_sites_list0 = ref_sites.tolist()
    ref_s1 = _seq_to_s1_array(seq_ref)

    rows: List[Dict[str, Any]] = []
    for iso in iso_ids:
        if iso == ref:
            continue

        # Defaults (in case missing seq / probs)
        retained_pos0: List[int] = []
        lost_missing_pos0: List[int] = ref_sites_list0.copy()
        lost_aligned_pos0: List[int] = []
        n_aligned_aa_match = 0

        seq_iso = SEQ_MAP.get(iso, "")
        if not seq_iso:
            rows.append({
                "base_id": base_id,
                "ref_isoform": ref,
                "isoform": iso,
                "n_ref_sites": n_ref_sites,
                "retained": 0,
                "lost_missing": n_ref_sites,
                "lost_aligned": 0,
                "category": "loss_missing_only",
                "retained_pos_ref1": _fmt_pos_1based(retained_pos0),
                "lost_missing_pos_ref1": _fmt_pos_1based(lost_missing_pos0),
                "lost_aligned_pos_ref1": _fmt_pos_1based(lost_aligned_pos0),
                "n_aligned_aa_match": 0,
                "note": "no_iso_seq",
            })
            continue

        p_iso = get_prob_for_isoform(iso)
        L_iso = min(len(seq_iso), int(p_iso.shape[0]))
        if L_iso < 1:
            rows.append({
                "base_id": base_id,
                "ref_isoform": ref,
                "isoform": iso,
                "n_ref_sites": n_ref_sites,
                "retained": 0,
                "lost_missing": n_ref_sites,
                "lost_aligned": 0,
                "category": "loss_missing_only",
                "retained_pos_ref1": _fmt_pos_1based(retained_pos0),
                "lost_missing_pos_ref1": _fmt_pos_1based(lost_missing_pos0),
                "lost_aligned_pos_ref1": _fmt_pos_1based(lost_aligned_pos0),
                "n_aligned_aa_match": 0,
                "note": "no_iso_probs",
            })
            continue

        seq_iso = seq_iso[:L_iso]
        p_iso = p_iso[:L_iso]

        ref_idx, iso_idx = compute_mapping_ref_to_iso(seq_ref, seq_iso)
        if ref_idx.size == 0:
            rows.append({
                "base_id": base_id,
                "ref_isoform": ref,
                "isoform": iso,
                "n_ref_sites": n_ref_sites,
                "retained": 0,
                "lost_missing": n_ref_sites,
                "lost_aligned": 0,
                "category": "loss_missing_only",
                "retained_pos_ref1": _fmt_pos_1based(retained_pos0),
                "lost_missing_pos_ref1": _fmt_pos_1based(lost_missing_pos0),
                "lost_aligned_pos_ref1": _fmt_pos_1based(lost_aligned_pos0),
                "n_aligned_aa_match": 0,
                "note": "no_alignment",
            })
            continue

        iso_s1 = _seq_to_s1_array(seq_iso)

        # Keep only aligned positions with identical AA
        aa_match = (ref_s1[ref_idx] == iso_s1[iso_idx])
        rr = ref_idx[aa_match]
        ii = iso_idx[aa_match]
        n_aligned_aa_match = int(rr.size)

        # Map ref position -> iso position for AA-match aligned positions
        map_ref_to_iso = np.full(L_ref, -1, dtype=np.int32)
        if rr.size > 0:
            map_ref_to_iso[rr] = ii

        retained_pos0 = []
        lost_missing_pos0 = []
        lost_aligned_pos0 = []

        for rpos in ref_sites_list0:
            ipos = int(map_ref_to_iso[int(rpos)])
            if ipos < 0:
                lost_missing_pos0.append(int(rpos))
            else:
                if float(p_iso[ipos]) >= TAU:
                    retained_pos0.append(int(rpos))
                else:
                    lost_aligned_pos0.append(int(rpos))

        retained = len(retained_pos0)
        lost_missing = len(lost_missing_pos0)
        lost_aligned = len(lost_aligned_pos0)

        if (lost_missing == 0) and (lost_aligned == 0):
            category = "no_loss"
        elif (lost_aligned == 0) and (lost_missing > 0):
            category = "loss_missing_only"
        else:
            category = "loss_aligned_present"

        rows.append({
            "base_id": base_id,
            "ref_isoform": ref,
            "isoform": iso,
            "n_ref_sites": n_ref_sites,
            "retained": retained,
            "lost_missing": lost_missing,
            "lost_aligned": lost_aligned,
            "category": category,
            "retained_pos_ref1": _fmt_pos_1based(retained_pos0),
            "lost_missing_pos_ref1": _fmt_pos_1based(lost_missing_pos0),
            "lost_aligned_pos_ref1": _fmt_pos_1based(lost_aligned_pos0),
            "n_aligned_aa_match": n_aligned_aa_match,
            "note": "",
        })

    meta = {
        "skipped": False,
        "reason": "",
        "ref_isoform": ref,
        "n_ref_sites": n_ref_sites,
        "n_isoforms": len(iso_ids),
        "ref_sites_pos_ref1": _fmt_pos_1based(ref_sites_list0),
    }
    return base_id, meta, rows


def run(
    dump_npz: str,
    out_dir: str,
    tau: float = 0.35,
    nproc: int = 16,
    min_isoforms: int = 2,
    maxtasksperchild: int = 50,
) -> None:
    global ISOFORM_IDS, BASE_IDS, PDB_PATHS, OFFSETS, PROB_CONCAT, ISO_TO_IDX, SEQ_MAP, TAU
    TAU = float(tau)

    out_dir_p = os.path.abspath(out_dir)
    os.makedirs(out_dir_p, exist_ok=True)

    d = np.load(dump_npz, allow_pickle=False)
    ISOFORM_IDS = d["isoform_ids"].astype(str)
    BASE_IDS = d["base_ids"].astype(str)
    PDB_PATHS = d["pdb_paths"].astype(str)
    OFFSETS = d["offsets"].astype(np.int64)
    PROB_CONCAT = d["prob_concat"]  # keep float16; slice casts to float32

    if ISOFORM_IDS.shape[0] + 1 != OFFSETS.shape[0]:
        raise SystemExit("Bad dump_npz: offsets length must be N+1.")

    ISO_TO_IDX = {iso: i for i, iso in enumerate(ISOFORM_IDS)}

    # Load sequences once in parent
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
        print(f"[WARN] failed to load sequences for {bad} isoforms (they will be treated as missing).", flush=True)

    # Build base groups (all isoforms, even if some sequences missing; handled downstream)
    df = pd.DataFrame({"isoform_id": ISOFORM_IDS, "base_id": BASE_IDS})
    base_groups: List[Tuple[str, List[str]]] = []
    for base_id, g in df.groupby("base_id"):
        iso_ids = sorted(g["isoform_id"].astype(str).unique().tolist())
        if len(iso_ids) >= min_isoforms:
            base_groups.append((str(base_id), iso_ids))

    print(f"[INFO] base groups to process: {len(base_groups)} (min_isoforms={min_isoforms})", flush=True)
    print(f"[INFO] tau = {TAU}", flush=True)

    # Fork context to share large arrays
    try:
        ctx = get_context("fork")
        PoolCls = ctx.Pool
    except Exception:
        PoolCls = Pool

    all_rows: List[Dict[str, Any]] = []
    base_meta_rows: List[Dict[str, Any]] = []

    with PoolCls(processes=nproc, maxtasksperchild=maxtasksperchild) as pool:
        for base_id, meta, rows in pool.imap_unordered(analyze_one_base, base_groups, chunksize=1):
            base_meta_rows.append({"base_id": base_id, **meta})
            all_rows.extend(rows)

    per_iso = pd.DataFrame(all_rows)
    base_meta = pd.DataFrame(base_meta_rows)

    per_iso_path = os.path.join(out_dir_p, "per_isoform_loss_categories.csv")
    base_meta_path = os.path.join(out_dir_p, "per_base_meta.csv")

    per_iso.to_csv(per_iso_path, index=False)
    base_meta.to_csv(base_meta_path, index=False)

    # Summary counts by category
    total = int(len(per_iso))
    if total == 0:
        summary = pd.DataFrame([{
            "tau": TAU,
            "total_isoforms_compared": 0,
            "no_loss": 0,
            "loss_missing_only": 0,
            "loss_aligned_present": 0,
            "frac_no_loss": np.nan,
            "frac_loss_missing_only": np.nan,
            "frac_loss_aligned_present": np.nan,
            "n_bases_skipped_no_ref_sites": int(((base_meta["skipped"] == True) & (base_meta["reason"] == "no_ref_sites")).sum()) if "reason" in base_meta.columns else 0,
        }])
    else:
        counts = per_iso["category"].value_counts().to_dict()
        a = int(counts.get("no_loss", 0))
        b = int(counts.get("loss_missing_only", 0))
        c = int(counts.get("loss_aligned_present", 0))

        n_skip_no_ref_sites = int(((base_meta["skipped"] == True) & (base_meta["reason"] == "no_ref_sites")).sum()) if "reason" in base_meta.columns else 0

        summary = pd.DataFrame([{
            "tau": TAU,
            "total_isoforms_compared": total,
            "no_loss": a,
            "loss_missing_only": b,
            "loss_aligned_present": c,
            "frac_no_loss": a / total,
            "frac_loss_missing_only": b / total,
            "frac_loss_aligned_present": c / total,
            "n_bases_skipped_no_ref_sites": n_skip_no_ref_sites,
        }])

    summary_path = os.path.join(out_dir_p, "summary_loss_categories.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[OK] wrote per-isoform table -> {per_iso_path}", flush=True)
    print(f"[OK] wrote per-base meta -> {base_meta_path}", flush=True)
    print(f"[OK] wrote summary -> {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isoform loss categorization at tau=0.35 from dump_npz (with positions).")
    ap.add_argument("--dump_npz", required=True, help="NPZ produced by GPU stage (one file).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--min_isoforms", type=int, default=2)
    ap.add_argument("--maxtasksperchild", type=int, default=50)
    args = ap.parse_args()

    run(
        dump_npz=args.dump_npz,
        out_dir=args.out_dir,
        tau=args.tau,
        nproc=args.nproc,
        min_isoforms=args.min_isoforms,
        maxtasksperchild=args.maxtasksperchild,
    )


if __name__ == "__main__":
    main()
