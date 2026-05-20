#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU stage: ESM2-first inference for isoforms, with noMSA fallback.

Key design choices:
- Prefer cached ESM2 embeddings when available.
- Fast inference: call predict_interface_residues only twice (ESM2 group + noMSA group).
- No PDB parsing here: GPU stage only does inference + serialization.

Outputs:
  - out_npz: a single .npz with concatenated predictions and metadata
  - (optional) manifest_csv: a small CSV with isoform_id/base_id/path/mode/length

NPZ fields:
  - isoform_ids:   (N,) unicode
  - base_ids:      (N,) unicode
  - pdb_paths:     (N,) unicode
  - mode:          (N,) uint8  (2=ESM2, 0=noMSA)
  - inference_type:(N,) unicode ("esm2" or "nomsa")
  - lengths:       (N,) int32
  - offsets:       (N+1,) int64, offsets[i]:offsets[i+1] slice into prob_concat / resids_concat
  - prob_concat:   (sum(lengths),) float16 (values < min_prob set to 0)
  - resids_concat: (sum(lengths),) int32  (1-based PDB residue numbers)
  - min_prob:      float32
  - esm2_dir:      unicode (saved for provenance)
  - structures_dir:unicode (saved for provenance)
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

import logging
import sys
import argparse
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import numpy as np
import pandas as pd

from isoform_pipeline_utils import (
    catalytic_channel,
    collect_structure_paths,
    ensure_trailing_slash,
    esm_exists_for_origin,
)

# ---------------- ScanNet_Ub in sys.path ----------------
SCANNET_ROOT = os.environ.get("SCANNET_ROOT", "")
if SCANNET_ROOT and SCANNET_ROOT not in sys.path:
    sys.path.insert(0, SCANNET_ROOT)

import predict_bindingsites  # noqa: E402


def base_id_from_isoform(iso_id: str) -> str:
    """Base UniProt ID (uppercase for consistency in grouping)."""
    return iso_id.split("-", 1)[0].upper()


def _make_esm2_pipeline(esm2_dir: str):
    homology_search = getattr(predict_bindingsites, "homology_search", "mmseqs")
    return predict_bindingsites.pipelines.ScanNetPipeline(
        with_aa=True,
        with_atom=True,
        aa_features="esm2",
        atom_features="valency",
        aa_frames="triplet_sidechain",
        Beff=500,
        homology_search=homology_search,
        esm2_dir=esm2_dir,
    )


def run_cv_catalytic_inference(
    struct_paths: List[str],
    entry_names: List[str],
    mode: str,
    ncores: int,
    esm2_dir: Optional[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]]]:
    """
    Run cv_catalytic inference.
    Returns (name->prob_array, name->resid_array).
    """
    if mode == "esm2":
        if not esm2_dir:
            raise ValueError("esm2_dir is required for mode='esm2'")
        pipeline = _make_esm2_pipeline(esm2_dir=esm2_dir)
        model = predict_bindingsites.cv_catalytic_model_ESM2
        model_name = predict_bindingsites.cv_catalytic_model_name_ESM2
        use_msa = False
        msa_folder = predict_bindingsites.MSA_folder
    elif mode == "nomsa":
        pipeline = predict_bindingsites.pipeline_noMSA
        model = predict_bindingsites.cv_catalytic_model_noMSA
        model_name = predict_bindingsites.cv_catalytic_model_name_noMSA
        use_msa = False
        msa_folder = predict_bindingsites.MSA_folder
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _, _, preds, resids_list, _ = predict_bindingsites.predict_interface_residues(
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
        MSA_folder=msa_folder,
        Lmin=1,
        output_predictions=False,
        aggregate_models=True,
        output_chimera=None,
        permissive=True,
        output_format="numpy",
        ncores=ncores,
    )
    return dict(zip(entry_names, preds)), dict(zip(entry_names, resids_list))


def run_stage(
    structures_dir: str,
    esm2_dir: Optional[str],
    out_npz: str,
    manifest_csv: Optional[str],
    ncores: int,
    min_prob: float,
) -> None:
    structures_dir = ensure_trailing_slash(structures_dir)
    esm2_dir = ensure_trailing_slash(esm2_dir) if esm2_dir else None

    by_stem = collect_structure_paths(structures_dir)
    stems = sorted(by_stem.keys())
    if not stems:
        raise SystemExit(f"No structures found in {structures_dir}")

    with_esm2 = [s for s in stems if esm_exists_for_origin(esm2_dir, s)]
    without_esm2 = [s for s in stems if s not in set(with_esm2)]

    logging.info("total structures: %s", len(stems))
    logging.info("ESM2 split: esm2=%s, nomsa=%s", len(with_esm2), len(without_esm2))

    preds_raw: Dict[str, np.ndarray] = {}
    resids_raw: Dict[str, Optional[np.ndarray]] = {}

    if with_esm2:
        paths = [by_stem[s] for s in with_esm2]
        pr, rs = run_cv_catalytic_inference(paths, with_esm2, mode="esm2", ncores=ncores, esm2_dir=esm2_dir)
        preds_raw.update(pr)
        resids_raw.update(rs)
        logging.info("inferred ESM2 group: %s", len(with_esm2))

    if without_esm2:
        paths = [by_stem[s] for s in without_esm2]
        pr, rs = run_cv_catalytic_inference(paths, without_esm2, mode="nomsa", ncores=ncores, esm2_dir=None)
        preds_raw.update(pr)
        resids_raw.update(rs)
        logging.info("inferred noMSA group: %s", len(without_esm2))

    isoform_ids: List[str] = []
    base_ids: List[str] = []
    pdbs: List[str] = []
    modes: List[int] = []
    inference_types: List[str] = []
    lengths: List[int] = []
    offsets: List[int] = [0]
    prob_chunks: List[np.ndarray] = []
    resids_chunks: List[np.ndarray] = []

    kept = 0
    esm2_set = set(with_esm2)
    for s in stems:
        arr = preds_raw.get(s)
        if arr is None:
            continue

        p = catalytic_channel(np.asarray(arr))
        if p.size == 0:
            continue

        p_stored = np.where(p >= min_prob, p, 0.0).astype(np.float16)

        r = resids_raw.get(s)
        if r is not None and len(r) == len(p):
            resids = np.asarray(r, dtype=np.int32)
        else:
            resids = np.arange(1, len(p) + 1, dtype=np.int32)

        isoform_ids.append(s)
        base_ids.append(base_id_from_isoform(s))
        pdbs.append(by_stem[s])

        if s in esm2_set:
            modes.append(2)
            inference_types.append("esm2")
        else:
            modes.append(0)
            inference_types.append("nomsa")

        length = int(p_stored.shape[0])
        lengths.append(length)
        prob_chunks.append(p_stored)
        resids_chunks.append(resids)
        offsets.append(offsets[-1] + length)
        kept += 1

    if kept == 0:
        raise SystemExit("No predictions were produced (kept=0). Check model run and inputs.")

    prob_concat = np.concatenate(prob_chunks, axis=0).astype(np.float16)
    resids_concat = np.concatenate(resids_chunks, axis=0).astype(np.int32)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        isoform_ids=np.array(isoform_ids, dtype="U"),
        base_ids=np.array(base_ids, dtype="U"),
        pdb_paths=np.array(pdbs, dtype="U"),
        mode=np.array(modes, dtype=np.uint8),
        inference_type=np.array(inference_types, dtype="U"),
        lengths=np.array(lengths, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
        prob_concat=prob_concat,
        resids_concat=resids_concat,
        min_prob=np.float32(min_prob),
        esm2_dir=np.array(esm2_dir or "", dtype="U"),
        structures_dir=np.array(structures_dir, dtype="U"),
    )

    logging.info("wrote %s isoforms -> %s", kept, out_npz)
    logging.info("total prob elements: %s", f"{prob_concat.size:,}")

    if manifest_csv:
        os.makedirs(os.path.dirname(manifest_csv) or ".", exist_ok=True)
        mdf = pd.DataFrame({
            "isoform_id": isoform_ids,
            "base_id": base_ids,
            "pdb_path": pdbs,
            "inference_type": inference_types,
            "L": lengths,
            "offset_start": offsets[:-1],
            "offset_end": offsets[1:],
        })
        mdf.to_csv(manifest_csv, index=False)
        logging.info("wrote manifest -> %s", manifest_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU stage: ESM2-first isoform inference → NPZ.")
    ap.add_argument("--structures_dir", required=True, help="Directory with isoform PDBs/CIFs.")
    ap.add_argument("--esm2_dir", default=None,
                    help="Root of cached ESM2 embeddings (<root>/<stem[:2]>/<stem>.npy).")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path.")
    ap.add_argument("--manifest_csv", default=None, help="Optional per-isoform manifest CSV.")
    ap.add_argument("--ncores", type=int, default=8, help="Cores for predict_interface_residues.")
    ap.add_argument("--min_prob", type=float, default=0.1,
                    help="Zero out probabilities below this value in the NPZ.")
    args = ap.parse_args()

    run_stage(
        structures_dir=args.structures_dir,
        esm2_dir=args.esm2_dir,
        out_npz=args.out_npz,
        manifest_csv=args.manifest_csv,
        ncores=args.ncores,
        min_prob=args.min_prob,
    )


if __name__ == "__main__":
    main()
