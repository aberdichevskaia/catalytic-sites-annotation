#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU stage: hybrid (MSA/noMSA) inference for isoforms and dump predictions into ONE file.

Key design choices:
- Fast MSA detection: index MSA stems once (no per-isoform glob()).
- Fast inference: call predict_interface_residues only twice (MSA group + noMSA group), no chunking.
- No PDB parsing here (biotite removed): GPU stage only does inference + serialization.

Outputs:
  - out_npz: a single .npz with concatenated predictions and metadata
  - (optional) manifest_csv: a small CSV with isoform_id/base_id/path/mode/length

NPZ fields:
  - isoform_ids: (N,) unicode
  - base_ids:    (N,) unicode
  - pdb_paths:   (N,) unicode
  - mode:        (N,) uint8  (1=MSA, 0=noMSA)
  - lengths:     (N,) int32
  - offsets:     (N+1,) int64, offsets[i]:offsets[i+1] slice into prob_concat
  - prob_concat: (sum(lengths),) float16 (values < min_prob set to 0)
  - min_prob:    float32
  - msa_dir:     unicode (saved for provenance)
  - structures_dir: unicode (saved for provenance)
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
# ------------------------------------------------------------------------------

import sys
import re
import argparse
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- ScanNet_Ub in sys.path ----------------
PROJECT_ROOT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import predict_bindingsites  # noqa: E402


def ensure_trailing_slash(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if p.endswith(os.sep) else p + os.sep


def stem_from_path(path: str) -> str:
    """Return file stem as-is (case-sensitive, important for MSA file matching)."""
    return os.path.splitext(os.path.basename(path))[0]


def base_id_from_isoform(iso_id: str) -> str:
    """Base UniProt ID (uppercase for consistency in grouping)."""
    return iso_id.split("-", 1)[0].upper()


def catalytic_channel(arr: np.ndarray) -> np.ndarray:
    """Extract catalytic probability channel from model output."""
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return (arr[:, 0] if arr.shape[1] == 1 else arr[:, 1]).astype(np.float32)
    return np.asarray(arr).reshape(-1).astype(np.float32)


_MSA_RE = re.compile(r"^MSA_(.+)_\d+_[A-Za-z0-9]+\.fasta$")


def collect_msa_stems(msa_dir: str, recursive: bool = False) -> set[str]:
    """
    Collect all stems that have at least one MSA file named like:
      MSA_<stem>_0_A.fasta, MSA_<stem>_0_B.fasta, ...

    This is O(#MSA files) once, and makes membership checks O(1).
    """
    stems: set[str] = set()

    if not msa_dir or not os.path.isdir(msa_dir):
        return stems

    if recursive:
        for root, _, files in os.walk(msa_dir):
            for fn in files:
                m = _MSA_RE.match(fn)
                if m:
                    stems.add(m.group(1))
    else:
        # non-recursive: fastest for flat dirs
        for fn in os.listdir(msa_dir):
            m = _MSA_RE.match(fn)
            if m:
                stems.add(m.group(1))

    return stems


def run_cv_catalytic_inference(
    struct_paths: List[str],
    entry_names: List[str],
    use_msa: bool,
    ncores: int,
    msa_dir: Optional[str],
) -> Dict[str, np.ndarray]:
    """
    Run cv_catalytic inference and return name->raw_pred_array.
    IMPORTANT: we call predict_interface_residues only a few times (twice total in this script).
    """
    pipeline = predict_bindingsites.pipeline_MSA if use_msa else predict_bindingsites.pipeline_noMSA
    if use_msa:
        model = predict_bindingsites.cv_catalytic_model_MSA
        model_name = predict_bindingsites.cv_catalytic_model_name_MSA
    else:
        model = predict_bindingsites.cv_catalytic_model_noMSA
        model_name = predict_bindingsites.cv_catalytic_model_name_noMSA

    _, _, preds, _, _ = predict_bindingsites.predict_interface_residues(
        query_pdbs=struct_paths,
        query_names=entry_names,  # MUST match stem used in MSA file names
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
        overwrite_MSA=False,  # strict: do not build missing MSAs
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


def main():
    ap = argparse.ArgumentParser(
        description="GPU stage: hybrid MSA/noMSA isoform inference, save ONE npz with concatenated predictions."
    )
    ap.add_argument("--structures_dir", required=True, help="Directory with isoform PDBs (*.pdb)")
    ap.add_argument("--msa_dir", default=None, help="Directory with MSAs (MSA_<stem>_*_*.fasta).")
    ap.add_argument("--msa_recursive", action="store_true", help="Scan MSA dir recursively (if files are in subdirs).")
    ap.add_argument("--out_npz", required=True, help="Output .npz file path (single file).")
    ap.add_argument("--manifest_csv", default=None, help="Optional manifest CSV path.")
    ap.add_argument("--ncores", type=int, default=8, help="ncores passed to predict_bindingsites")
    ap.add_argument("--min_prob", type=float, default=0.1, help="Set probs < min_prob to 0 before saving")
    args = ap.parse_args()

    structures_dir = ensure_trailing_slash(args.structures_dir)
    msa_dir = ensure_trailing_slash(args.msa_dir) if args.msa_dir else None

    pdb_paths = sorted(glob(os.path.join(structures_dir, "*.pdb")))
    if not pdb_paths:
        raise SystemExit(f"No PDBs found in {structures_dir}")

    stems = [stem_from_path(p) for p in pdb_paths]  # IMPORTANT: keep case as in filename
    stem_to_path: Dict[str, str] = dict(zip(stems, pdb_paths))

    # Index MSA stems once
    msa_stems: set[str] = set()
    if msa_dir:
        msa_stems = collect_msa_stems(msa_dir, recursive=args.msa_recursive)

    with_msa = [s for s in stems if s in msa_stems]
    without_msa = [s for s in stems if s not in msa_stems]

    print(f"[INFO] total structures: {len(stems)}", flush=True)
    print(f"[INFO] msa_dir: {msa_dir if msa_dir else '(none)'}", flush=True)
    if msa_dir:
        print(f"[INFO] indexed MSA stems: {len(msa_stems)}", flush=True)
    print(f"[INFO] hybrid split: MSA={len(with_msa)}, noMSA={len(without_msa)}", flush=True)

    if msa_dir and len(with_msa) == 0:
        # Helpful debug in case of naming mismatch
        ex = stems[0]
        pat = os.path.join(msa_dir, f"MSA_{ex}_*_*.fasta")
        print(f"[WARN] No MSAs matched. Example stem='{ex}'. Example pattern='{pat}'", flush=True)
        sample = glob(pat)[:3]
        print(f"[WARN] Example matches (first 3): {sample}", flush=True)

    preds_raw: Dict[str, np.ndarray] = {}

    # Run inference only twice (fast)
    if with_msa:
        paths = [stem_to_path[s] for s in with_msa]
        preds_raw.update(run_cv_catalytic_inference(paths, with_msa, use_msa=True, ncores=args.ncores, msa_dir=msa_dir))
        print(f"[OK] inferred MSA group: {len(with_msa)}", flush=True)

    if without_msa:
        paths = [stem_to_path[s] for s in without_msa]
        preds_raw.update(run_cv_catalytic_inference(paths, without_msa, use_msa=False, ncores=args.ncores, msa_dir=None))
        print(f"[OK] inferred noMSA group: {len(without_msa)}", flush=True)

    # Build concatenated storage in the same order as `stems`
    isoform_ids: List[str] = []
    base_ids: List[str] = []
    pdbs: List[str] = []
    modes: List[int] = []
    lengths: List[int] = []
    offsets: List[int] = [0]
    prob_chunks: List[np.ndarray] = []

    kept = 0
    for s in stems:
        arr = preds_raw.get(s)
        if arr is None:
            # keep as empty (or skip). Here: skip to avoid inconsistencies.
            continue

        p = catalytic_channel(np.asarray(arr))
        if p.size == 0:
            continue

        # Threshold -> 0, cast to float16 for compactness
        p = np.where(p >= args.min_prob, p, 0.0).astype(np.float16)

        isoform_ids.append(s)  # stem = isoform id for isoform PDBs
        base_ids.append(base_id_from_isoform(s))
        pdbs.append(stem_to_path[s])
        modes.append(1 if s in msa_stems else 0)

        L = int(p.shape[0])
        lengths.append(L)
        prob_chunks.append(p)
        offsets.append(offsets[-1] + L)
        kept += 1

    if kept == 0:
        raise SystemExit("No predictions were produced (kept=0). Check model run and inputs.")

    prob_concat = np.concatenate(prob_chunks, axis=0).astype(np.float16)

    # Save one NPZ
    out_npz = args.out_npz
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    np.savez_compressed(
        out_npz,
        isoform_ids=np.array(isoform_ids, dtype="U"),
        base_ids=np.array(base_ids, dtype="U"),
        pdb_paths=np.array(pdbs, dtype="U"),
        mode=np.array(modes, dtype=np.uint8),
        lengths=np.array(lengths, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
        prob_concat=prob_concat,
        min_prob=np.float32(args.min_prob),
        msa_dir=np.array(msa_dir or "", dtype="U"),
        structures_dir=np.array(structures_dir, dtype="U"),
    )

    print(f"[OK] wrote {kept} isoforms into one file -> {out_npz}", flush=True)
    print(f"[INFO] total prob elements: {prob_concat.size:,}", flush=True)

    # Optional manifest
    if args.manifest_csv:
        manifest = args.manifest_csv
        os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
        df = pd.DataFrame({
            "isoform_id": isoform_ids,
            "base_id": base_ids,
            "pdb_path": pdbs,
            "mode": ["MSA" if m == 1 else "noMSA" for m in modes],
            "L": lengths,
            "offset_start": offsets[:-1],
            "offset_end": offsets[1:],
        })
        df.to_csv(manifest, index=False)
        print(f"[OK] wrote manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
