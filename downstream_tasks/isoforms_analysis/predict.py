#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Catalytic-site screening over AlphaFold Human proteome (cv_catalytic)
and human isoforms.

Key points:
- For AF Human:
  * We pass query_names = AF-stem (e.g. 'AF-P81877-F1-model_v6'),
    so predict_bindingsites looks for MSA files like
    'MSA_AF-P81877-F1-model_v6_0_A.fasta', etc.
  * Strict rule: if no MSA exists for a given stem, we do NOT build it,
    but run the model without MSA.

- For isoforms:
  * Structures are named like 'A0A0K2S4Q6-1.pdb'.
  * meta JSON (isoform_meta.json) uses exactly the same keys:
    "A0A0K2S4Q6-1": [ { ... } ].
  * We use the file stem ('A0A0K2S4Q6-1') as display id and metadata key.

ESM2 mode:
- If --use_esm2 and --esm2_dir are provided:
  * for each stem we check cached embedding:
      <esm2_dir>/<origin[:2]>/<origin>.npy
    where origin == query_name (stem).
  * if exists -> run cv_catalytic_ESM2
  * else -> fallback to cv_catalytic_noMSA (default "loser" model)

Output CSV:
- protein id (uniprot / PDB)   # 'ACC_F#' for AF or 'ACC-iso' for isoforms
- base uniprot id
- inference_type               # msa / esm2 / nomsa
- predicted with 35% threshold
- predicted with 65% threshold
- predicted with 85% threshold
- known catalytic sites
- EC number (if exists)
- protein name
- gene name
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

import re
import sys
import json
import time
import argparse
from glob import glob
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from isoform_pipeline_utils import (
    catalytic_channel,
    collect_structure_paths,
    ensure_trailing_slash,
    esm_exists_for_origin,
)

# ---------------- ScanNet_Ub in sys.path ----------------
PROJECT_ROOT = os.environ.get("SCANNET_PROJECT_ROOT")
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import predict_bindingsites  # noqa: E402

# ---------------- helpers -------------------------
UNIPROT_RE = re.compile(r"[A-Z0-9]{6,10}", re.I)


def entry_from_path(path: str) -> str:
    """
    Convert structure file name into a "nice" key for CSV/metadata.

    Examples:
    - AF-P81877-F1-model_v4.cif -> 'P81877_F1'  (AF Human)
    - A0A0K2S4Q6-1.pdb          -> 'A0A0K2S4Q6-1' (isoform, same as JSON key)
    - fallback: use stem in UPPERCASE.
    """
    base = os.path.basename(path)

    # AF Human case: AF-ACC-F#-model_... -> ACC_F#
    m = re.match(r"AF-([A-Z0-9]{6,10})-F(\d+)-model_", base, re.I)
    if m:
        return f"{m.group(1).upper()}_F{m.group(2)}"

    # Generic / isoform case: use file stem directly
    stem = os.path.splitext(base)[0]
    return stem.upper()


def acc_only(name: str) -> str:
    """
    Extract base ACC-like id from display name.

    Examples:
    - 'P81877_F1'      -> 'P81877'
    - 'A0A0K2S4Q6-1'   -> 'A0A0K2S4Q6-1' (isoform id is already the key)
    """
    return name.split("_", 1)[0].upper()


# ---- Check if MSA exists for given AF/generic stem ----
def msa_exists_for_stem(msa_root: Optional[str], stem: str) -> bool:
    """
    Check if there is at least one file like:
      MSA_<stem>_0_A.fasta / MSA_<stem>_0_B.fasta / ...
    """
    if not msa_root:
        return False
    pattern = os.path.join(msa_root, f"MSA_{stem}_*_*.fasta")
    return bool(glob(pattern))


def _make_esm2_pipeline(esm2_dir: str):
    """
    Build ScanNetPipeline that uses cached ESM2 embeddings (aa_features='esm2').
    """
    # prefer to reuse homology_search if it exists, but it is irrelevant because use_MSA=False here
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


# --------------- unified MetaDB -------------------
class MetaDB:
    """
    Unified metadata wrapper.

    Supports:

    1) Old ACC-format:
       {
         "ACC": {
           "full_name": ...,
           "gene_name": ...,
           "ec_numbers": [...],
           "active_sites": [{"pos": int, ...}, ...]
         },
         ...
       }

    2) Isoform-format (isoform_meta.json):
       {
         "O14733-2": [
           {
             "full_name": ...,
             "gene_name": ...,
             "ec_numbers": [...],
             "active_sites": [243, ...] or [{"pos": 243}, ...],
             "base_id": "O14733"
           }
         ],
         ...
       }
    """

    def __init__(self, path: Optional[str]):
        self.db: Dict[str, Dict[str, Any]] = {}
        if not path:
            return

        with open(path, "r") as f:
            raw = json.load(f)

        norm: Dict[str, Dict[str, Any]] = {}

        for k, v in raw.items():
            key = str(k).upper()

            # isoform-style: value is list with one dict
            if isinstance(v, list):
                if v and isinstance(v[0], dict):
                    rec = v[0] or {}
                else:
                    rec = {}
            # old-style: value is dict
            elif isinstance(v, dict):
                rec = v or {}
            else:
                rec = {}

            norm[key] = rec

        self.db = norm

    def _get_with_base(self, acc_like: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return (record_for_id, record_for_base_id).

        AF:
          acc_like = 'P81877_F1' -> key 'P81877'

        Isoform:
          acc_like = 'O14733-2'  -> key 'O14733-2'
          base read from 'base_id' if present.
        """
        key = acc_only(acc_like)
        rec = self.db.get(key, {})
        base_rec: Dict[str, Any] = {}

        base_id = rec.get("base_id")
        if isinstance(base_id, str):
            base_rec = self.db.get(base_id.upper(), {})

        return rec, base_rec

    def base_uniprot(self, acc_like: str) -> str:
        """
        Return base UniProt ID (without isoform suffix):
        - AF: 'P81877_F1' -> 'P81877'
        - isoform: if base_id exists -> base_id,
                   else split by '-' (Q92782-2 -> Q92782),
                   else return key.
        """
        key = acc_only(acc_like)
        rec = self.db.get(key, {})

        base_id = rec.get("base_id")
        if isinstance(base_id, str) and base_id.strip():
            return base_id.upper()

        if "-" in key:
            return key.split("-", 1)[0]

        return key

    def names(self, acc_like: str) -> Tuple[str, str]:
        rec, base_rec = self._get_with_base(acc_like)
        full_name = rec.get("full_name") or base_rec.get("full_name") or ""
        gene_name = rec.get("gene_name") or base_rec.get("gene_name") or ""
        return full_name, gene_name

    def ecs(self, acc_like: str) -> List[str]:
        rec, base_rec = self._get_with_base(acc_like)
        ecs = rec.get("ec_numbers") or base_rec.get("ec_numbers") or []
        return list(ecs) if isinstance(ecs, list) else []

    def known_positions(self, acc_like: str) -> List[int]:
        rec, base_rec = self._get_with_base(acc_like)

        sites = rec.get("active_sites")
        if not sites:
            sites = base_rec.get("active_sites") or []

        pos: List[int] = []
        if isinstance(sites, list):
            for it in sites:
                if isinstance(it, dict) and "pos" in it:
                    try:
                        p = int(it["pos"])
                        if p >= 1:
                            pos.append(p)
                    except Exception:
                        pass
                else:
                    try:
                        p = int(it)
                        if p >= 1:
                            pos.append(p)
                    except Exception:
                        pass

        return sorted(set(pos))


# --------------- model wrappers -------------------
def run_cv_catalytic_inference(
    struct_paths: List[str],
    entry_names: List[str],   # stems
    mode: str,                # 'msa' | 'nomsa' | 'esm2'
    ncores: int,
    msa_dir: Optional[str] = None,
    esm2_dir: Optional[str] = None,
):
    if mode == "msa":
        pipeline = predict_bindingsites.pipeline_MSA
        model = predict_bindingsites.cv_catalytic_model_MSA
        model_name = predict_bindingsites.cv_catalytic_model_name_MSA
        use_msa_flag = True
        msa_folder = (msa_dir or predict_bindingsites.MSA_folder)

    elif mode == "esm2":
        if not esm2_dir:
            raise ValueError("esm2_dir is required for mode='esm2'")
        pipeline = _make_esm2_pipeline(esm2_dir=esm2_dir)
        model = predict_bindingsites.cv_catalytic_model_ESM2
        model_name = predict_bindingsites.cv_catalytic_model_name_ESM2
        use_msa_flag = False
        msa_folder = predict_bindingsites.MSA_folder  # unused but required

    elif mode == "nomsa":
        pipeline = predict_bindingsites.pipeline_noMSA
        model = predict_bindingsites.cv_catalytic_model_noMSA
        model_name = predict_bindingsites.cv_catalytic_model_name_noMSA
        use_msa_flag = False
        msa_folder = predict_bindingsites.MSA_folder

    else:
        raise ValueError(f"Unknown mode: {mode}")

    _, _, preds, resids, _ = predict_bindingsites.predict_interface_residues(
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
        use_MSA=use_msa_flag,
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
    return dict(zip(entry_names, preds)), dict(zip(entry_names, resids))


def hits_to_str(probs: np.ndarray, resids: Optional[np.ndarray], thr: float) -> str:
    """Convert predictions above threshold into comma-separated list of residue positions."""
    if resids is None or len(resids) != len(probs):
        resids = np.arange(1, len(probs) + 1, dtype=int)
    return ",".join(map(str, resids[probs >= thr].tolist()))


# ----------------------- CLI ---------------------
def main():
    t0 = time.time()
    print("[BOOT] starting predict.py", flush=True)

    ap = argparse.ArgumentParser(
        description=(
            "Catalytic-site inference over AF Human or Human Isoforms "
            "(cv_catalytic, one meta JSON)."
        )
    )
    ap.add_argument("--structures_dir", required=True, help="Directory with *.cif or *.pdb structures.")
    ap.add_argument("--msa_dir", default=None, help="Directory with MSA files (MSA_<stem>_*_*.fasta).")
    ap.add_argument("--meta_json", required=True, help="Unified JSON with names/genes/EC/catalytic sites.")

    ap.add_argument("--use_msa", action="store_true")
    ap.add_argument("--use_esm2", action="store_true", help="Use ESM2 cached embeddings when present; fallback to noMSA")
    ap.add_argument("--esm2_dir", default=None, help="Root with cached ESM2 .npy: <root>/<origin[:2]>/<origin>.npy")

    ap.add_argument("--thr_extra", type=float, default=0.35)
    ap.add_argument("--thr_lo", type=float, default=0.65)
    ap.add_argument("--thr_hi", type=float, default=0.85)
    ap.add_argument("--ncores", type=int, default=8)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    structures_dir = ensure_trailing_slash(args.structures_dir)
    msa_dir = ensure_trailing_slash(args.msa_dir) if args.use_msa and args.msa_dir else None
    esm2_dir = ensure_trailing_slash(args.esm2_dir) if args.use_esm2 and args.esm2_dir else None

    if args.use_esm2 and not esm2_dir:
        raise SystemExit("[ERROR] --use_esm2 requires --esm2_dir")

    # 1) Collect structures.
    print(f"[STEP] scanning structures under {structures_dir}", flush=True)
    by_runname = collect_structure_paths(structures_dir)
    display_for: Dict[str, str] = {}
    for rn, p in by_runname.items():
        display_for[rn] = entry_from_path(p)

    entries_all = sorted(by_runname.keys())
    if not entries_all:
        raise SystemExit(f"No structures in {structures_dir} (*.cif|*.pdb)")
    print(
        f"[INFO] using {len(entries_all)} structures (prefer .pdb when both exist) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    # 2) Metadata
    meta = MetaDB(args.meta_json)

    preds_raw: Dict[str, np.ndarray] = {}
    resids_map: Dict[str, np.ndarray] = {}

    # Track per-entry inference type
    inference_type: Dict[str, str] = {}

    # 3) Decide inference branches
    if args.use_esm2:
        # ESM2 split: with cached embedding -> esm2 model; else -> nomsa fallback
        with_esm2 = [rn for rn in entries_all if esm_exists_for_origin(esm2_dir, rn)]
        without_esm2 = [rn for rn in entries_all if rn not in set(with_esm2)]

        for rn in with_esm2:
            inference_type[rn] = "esm2"
        for rn in without_esm2:
            inference_type[rn] = "nomsa"

        print(
            f"[INFO] ESM2 MODE: esm2 for {len(with_esm2)}, "
            f"fallback noMSA for {len(without_esm2)}.",
            flush=True,
        )
        if without_esm2:
            print(
                "[DEBUG] examples without detected ESM2 cache (first 10): "
                + ", ".join(without_esm2[:10]),
                flush=True,
            )

        if with_esm2:
            paths = [by_runname[rn] for rn in with_esm2]
            pr, rs = run_cv_catalytic_inference(
                paths, with_esm2, mode="esm2", ncores=args.ncores, esm2_dir=esm2_dir
            )
            preds_raw.update(pr)
            resids_map.update(rs)

        if without_esm2:
            paths = [by_runname[rn] for rn in without_esm2]
            pr, rs = run_cv_catalytic_inference(
                paths, without_esm2, mode="nomsa", ncores=args.ncores
            )
            preds_raw.update(pr)
            resids_map.update(rs)

    else:
        # Original strict MSA split
        if args.use_msa and msa_dir:
            print("use MSA mode: checking for existing MSAs...", flush=True)
            with_msa = [rn for rn in entries_all if msa_exists_for_stem(msa_dir, rn)]
        else:
            with_msa = []
        without_msa = [rn for rn in entries_all if rn not in set(with_msa)]

        for rn in with_msa:
            inference_type[rn] = "msa"
        for rn in without_msa:
            inference_type[rn] = "nomsa"

        print(
            f"[INFO] STRICT MSA MODE: will NOT build missing MSAs; "
            f"use_MSA for {len(with_msa)}, noMSA for {len(without_msa)}.",
            flush=True,
        )
        if args.use_msa and msa_dir and without_msa:
            print(
                "[DEBUG] examples without detected MSA (first 10): "
                + ", ".join(without_msa[:10]),
                flush=True,
            )

        if with_msa:
            paths = [by_runname[rn] for rn in with_msa]
            pr, rs = run_cv_catalytic_inference(
                paths, with_msa, mode="msa", ncores=args.ncores, msa_dir=msa_dir
            )
            preds_raw.update(pr)
            resids_map.update(rs)

        if without_msa:
            paths = [by_runname[rn] for rn in without_msa]
            pr, rs = run_cv_catalytic_inference(
                paths, without_msa, mode="nomsa", ncores=args.ncores
            )
            preds_raw.update(pr)
            resids_map.update(rs)

    # 4) Build table
    rows = []
    for rn in entries_all:
        disp = display_for[rn]
        arr = preds_raw.get(rn)

        if arr is None:
            probs = np.zeros(0, dtype=float)
            resids = None
        else:
            probs = catalytic_channel(np.asarray(arr))
            resids = resids_map.get(rn)

        base_acc = meta.base_uniprot(disp)
        known = meta.known_positions(disp)
        ecs = meta.ecs(disp)
        full_name, gene_name = meta.names(disp)

        rows.append({
            "protein id (uniprot / PDB)": disp,
            "base uniprot id": base_acc,
            "inference_type": inference_type.get(rn, "nomsa"),
            f"predicted with {int(args.thr_extra * 100)}% threshold":
                hits_to_str(probs, resids, args.thr_extra) if probs.size else "",
            f"predicted with {int(args.thr_lo * 100)}% threshold":
                hits_to_str(probs, resids, args.thr_lo) if probs.size else "",
            f"predicted with {int(args.thr_hi * 100)}% threshold":
                hits_to_str(probs, resids, args.thr_hi) if probs.size else "",
            "known catalytic sites": ",".join(map(str, known)) if known else "",
            "EC number (if exists)": ";".join(ecs) if ecs else "",
            "protein name": full_name,
            "gene name": gene_name,
        })

    cols = [
        "protein id (uniprot / PDB)",
        "base uniprot id",
        "inference_type",
        f"predicted with {int(args.thr_extra * 100)}% threshold",
        f"predicted with {int(args.thr_lo * 100)}% threshold",
        f"predicted with {int(args.thr_hi * 100)}% threshold",
        "known catalytic sites",
        "EC number (if exists)",
        "protein name",
        "gene name",
    ]

    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {len(df)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
