#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid MSA/noMSA inference for isoforms + isoform variability score.

For each isoform:
  - if MSA files exist in --msa_dir: run MSA model
  - else: run noMSA model

Then for each base_id compute:
  score = max_over_ref_positions( max_over_isoforms(p) - min_over_isoforms(p) )
where p is catalytic probability at aligned residue positions.

We align each isoform to a chosen reference isoform (prefer '-1' if exists),
using local alignment if available (biotite.align_local), else fallback to global.
We only compare positions where aa_ref == aa_iso (same residue present).
Probabilities < --min_prob are treated as 0.
"""

# ---------- Thread limits: must be set BEFORE importing numpy/keras/tf ----------
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_DEFAULT_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
# -------------------------------------------------------------------------------

import sys
import argparse
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------- ScanNet_Ub in sys.path ----------------
PROJECT_ROOT = os.environ.get("SCANNET_PROJECT_ROOT")
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import predict_bindingsites  # noqa: E402

# ---------------- biotite (alignment + PDB sequence) ----------------
import biotite.structure as struc
from biotite.structure.info import one_letter_code
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal

try:
    from biotite.sequence.align import align_local  # type: ignore
except Exception:
    align_local = None


def ensure_trailing_slash(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if p.endswith(os.sep) else p + os.sep


def isoform_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].upper()


def base_id_from_isoform(iso_id: str) -> str:
    return iso_id.split("-", 1)[0].upper()


def msa_exists_for_stem(msa_root: Optional[str], stem: str) -> bool:
    if not msa_root:
        return False
    pattern = os.path.join(msa_root, f"MSA_{stem}_*_*.fasta")
    return bool(glob(pattern))


def catalytic_channel(arr: np.ndarray) -> np.ndarray:
    """Extract catalytic probability channel from model output."""
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return (arr[:, 0] if arr.shape[1] == 1 else arr[:, 1]).astype(np.float32)
    return np.asarray(arr).reshape(-1).astype(np.float32)


def run_cv_catalytic_inference(
    struct_paths: List[str],
    entry_names: List[str],
    use_msa: bool,
    ncores: int,
    msa_dir: Optional[str],
) -> Dict[str, np.ndarray]:
    """Run cv_catalytic inference and return name->raw_pred_array."""
    pipeline = predict_bindingsites.pipeline_MSA if use_msa else predict_bindingsites.pipeline_noMSA
    if use_msa:
        model = predict_bindingsites.cv_catalytic_model_MSA
        model_name = predict_bindingsites.cv_catalytic_model_name_MSA
    else:
        model = predict_bindingsites.cv_catalytic_model_noMSA
        model_name = predict_bindingsites.cv_catalytic_model_name_noMSA

    _, _, preds, _, _ = predict_bindingsites.predict_interface_residues(
        query_pdbs=struct_paths,
        query_names=entry_names,
        query_chain_ids=None,
        query_sequences=None,
        pipeline=pipeline,
        model=model,
        model_name=model_name,
        model_folder=predict_bindingsites.model_folder,
        structures_folder=predict_bindingsites.structures_folder,
        biounit=False,
        assembly=True,
        layer=None,
        use_MSA=use_msa,
        overwrite_MSA=False,
        MSA_folder=(msa_dir or predict_bindingsites.MSA_folder),
        Lmin=1,
        output_predictions=False,
        aggregate_models=True,
        output_chimera=None,
        permissive=True,
        output_format="numpy",
        ncores=ncores,
    )
    return dict(zip(entry_names, preds))


def infer_in_chunks(
    paths: List[str],
    names: List[str],
    use_msa: bool,
    ncores: int,
    msa_dir: Optional[str],
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """Chunked inference to avoid huge single calls."""
    out: Dict[str, np.ndarray] = {}
    for start in range(0, len(paths), chunk_size):
        p = paths[start:start + chunk_size]
        n = names[start:start + chunk_size]
        pr = run_cv_catalytic_inference(p, n, use_msa=use_msa, ncores=ncores, msa_dir=msa_dir)
        out.update(pr)
        print(f"[INFO] {'MSA' if use_msa else 'noMSA'} inferred {min(start+chunk_size, len(paths))}/{len(paths)}", flush=True)
    return out


def load_isoform_sequence_from_pdb_path(pdb_path: str) -> str:
    """
    Load AA sequence from a PDB file:
      - choose the chain with the largest number of amino-acid residues
      - return 1-letter AA sequence for that chain
    """
    pdb_file = PDBFile.read(pdb_path)
    atoms = pdb_file.get_structure(model=1)
    if atoms is None or atoms.array_length() == 0:
        raise ValueError(f"Empty structure: {pdb_path}")

    chain_ids = np.unique(atoms.chain_id)
    best_seq = None
    best_len = 0

    for chain_id in chain_ids:
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

    if best_seq is None:
        raise ValueError(f"No amino-acid sequence found: {pdb_path}")

    return best_seq


# ---------- Alignment parameters (identity matrix, strong penalties) ----------
_ALPH = ProteinSequence.alphabet
_N = len(_ALPH)
_SUB_RAW = np.full((_N, _N), -100, dtype=int)
for _i in range(_N):
    _SUB_RAW[_i, _i] = 1
SUB_MAT = SubstitutionMatrix(_ALPH, _ALPH, _SUB_RAW)
GAP_PEN = (-100, -10)


def compute_mapping_ref_to_iso(seq_ref: str, seq_iso: str) -> Dict[int, int]:
    """
    Alignment mapping:
      ref position (1-based) -> iso position (1-based)
    for aligned (non-gap in both) positions.
    """
    s1 = ProteinSequence(seq_ref)
    s2 = ProteinSequence(seq_iso)

    if align_local is not None:
        aln = align_local(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]
    else:
        aln = align_optimal(s1, s2, SUB_MAT, gap_penalty=GAP_PEN)[0]

    g1, g2 = aln.get_gapped_sequences()

    mapping: Dict[int, int] = {}
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


def compute_group_score(
    iso_ids: List[str],
    seqs: Dict[str, str],
    probs_thr: Dict[str, np.ndarray],
) -> float:
    """
    Compute max over reference positions of (max - min) across isoforms
    that align to that reference position and have the same AA.
    """
    if len(iso_ids) < 2:
        return 0.0

    ref = choose_reference_isoform(iso_ids)
    if ref not in seqs or ref not in probs_thr:
        return 0.0

    seq_ref = seqs[ref]
    pr_ref = probs_thr[ref]

    L_ref = min(len(seq_ref), len(pr_ref))
    seq_ref = seq_ref[:L_ref]
    pr_ref = pr_ref[:L_ref]

    # Precompute mappings ref->iso (within group)
    mappings: Dict[str, Dict[int, int]] = {}
    for iso in iso_ids:
        if iso == ref:
            continue
        if iso not in seqs or iso not in probs_thr:
            continue
        seq_iso = seqs[iso]
        pr_iso = probs_thr[iso]
        L_iso = min(len(seq_iso), len(pr_iso))
        seqs[iso] = seq_iso[:L_iso]
        probs_thr[iso] = pr_iso[:L_iso]
        mappings[iso] = compute_mapping_ref_to_iso(seq_ref, seqs[iso])

    best = 0.0

    for i in range(1, L_ref + 1):
        aa_ref = seq_ref[i - 1]
        vals = [float(pr_ref[i - 1])]  # ref contributes

        for iso in iso_ids:
            if iso == ref:
                continue
            if iso not in mappings:
                continue
            j = mappings[iso].get(i)
            if j is None:
                continue
            if j < 1 or j > len(seqs[iso]):
                continue
            aa_iso = seqs[iso][j - 1]
            if aa_iso != aa_ref:
                continue
            vals.append(float(probs_thr[iso][j - 1]))

        if len(vals) < 2:
            continue

        spread = max(vals) - min(vals)
        if spread > best:
            best = spread

    return float(best)


def main():
    ap = argparse.ArgumentParser(description="Hybrid MSA/noMSA isoform inference + spread score per base_id.")
    ap.add_argument("--structures_dir", required=True, help="Directory with isoform PDBs (*.pdb)")
    ap.add_argument("--msa_dir", default=None, help="Directory with existing MSAs (MSA_<iso>_*_*.fasta). If absent -> all noMSA.")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--ncores", type=int, default=8, help="ncores for predict_bindingsites")
    ap.add_argument("--chunk_size", type=int, default=64, help="Inference chunk size")
    ap.add_argument("--min_prob", type=float, default=0.1, help="Keep probabilities >= min_prob, set lower to 0")
    args = ap.parse_args()

    structures_dir = ensure_trailing_slash(args.structures_dir)
    msa_dir = ensure_trailing_slash(args.msa_dir) if args.msa_dir else None

    pdb_paths = sorted(glob(os.path.join(structures_dir, "*.pdb")))
    if not pdb_paths:
        raise SystemExit(f"No PDBs found in {structures_dir}")

    iso_ids = [isoform_id_from_path(p) for p in pdb_paths]
    iso_to_path = dict(zip(iso_ids, pdb_paths))

    # Split isoforms by MSA presence (hybrid mode)
    if msa_dir:
        with_msa_ids = [iso for iso in iso_ids if msa_exists_for_stem(msa_dir, iso)]
    else:
        with_msa_ids = []
    with_msa_set = set(with_msa_ids)
    without_msa_ids = [iso for iso in iso_ids if iso not in with_msa_set]

    print(f"[INFO] total isoforms: {len(iso_ids)}", flush=True)
    print(f"[INFO] hybrid: MSA for {len(with_msa_ids)}, noMSA for {len(without_msa_ids)}", flush=True)
    if msa_dir and without_msa_ids:
        print("[DEBUG] examples noMSA (first 10): " + ", ".join(without_msa_ids[:10]), flush=True)

    # Inference
    probs_raw: Dict[str, np.ndarray] = {}

    if with_msa_ids:
        paths = [iso_to_path[iso] for iso in with_msa_ids]
        pr = infer_in_chunks(
            paths=paths,
            names=with_msa_ids,
            use_msa=True,
            ncores=args.ncores,
            msa_dir=msa_dir,
            chunk_size=args.chunk_size,
        )
        probs_raw.update(pr)

    if without_msa_ids:
        paths = [iso_to_path[iso] for iso in without_msa_ids]
        pr = infer_in_chunks(
            paths=paths,
            names=without_msa_ids,
            use_msa=False,
            ncores=args.ncores,
            msa_dir=None,
            chunk_size=args.chunk_size,
        )
        probs_raw.update(pr)

    # Convert to catalytic probs and apply min_prob thresholding once
    probs_thr: Dict[str, np.ndarray] = {}
    for iso, arr in probs_raw.items():
        if arr is None:
            continue
        p = catalytic_channel(np.asarray(arr))
        p = np.where(p >= args.min_prob, p, 0.0).astype(np.float32)
        probs_thr[iso] = p

    # Load sequences
    seqs: Dict[str, str] = {}
    for iso, path in iso_to_path.items():
        try:
            seqs[iso] = load_isoform_sequence_from_pdb_path(path)
        except Exception as exc:
            print(f"[WARN] failed to load sequence for {iso}: {exc}", flush=True)

    # Group by base_id (only isoforms with both seq and probs)
    base_to_isos: Dict[str, List[str]] = {}
    for iso in iso_ids:
        if iso not in probs_thr or iso not in seqs:
            continue
        base = base_id_from_isoform(iso)
        base_to_isos.setdefault(base, []).append(iso)

    # Compute scores
    out_rows = []
    for base_id, isos in sorted(base_to_isos.items()):
        if len(isos) < 2:
            continue
        g_seqs = {iso: seqs[iso] for iso in isos}
        g_probs = {iso: probs_thr[iso] for iso in isos}
        try:
            score = compute_group_score(isos, g_seqs, g_probs)
        except Exception as exc:
            print(f"[WARN] base_id={base_id}: failed to compute score ({exc})", flush=True)
            continue
        out_rows.append({"base id": base_id, "score": score})

    df = pd.DataFrame(out_rows, columns=["base id", "score"])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {len(df)} base IDs -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
