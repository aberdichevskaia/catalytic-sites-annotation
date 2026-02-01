#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Catalytic-site screening over AlphaFold Human proteome (cv_catalytic).

Ключевые моменты:
- В модель передаём query_names = AF-stem (например, 'AF-P81877-F1-model_v6'),
  чтобы predict_bindingsites искал MSA ровно как у тебя в папке:
  'MSA_AF-P81877-F1-model_v6_0_A.fasta' и т.п.
- Строгое правило: если MSA под таким stem не найден, НЕ строим его, а запускаем без MSA.
- В CSV показываем "красивый" ID вида 'ACC_F#', метаданные (EC/active sites/names) берём по ACC.

ESM2 режим:
- Если задано --use_esm2 --esm2_dir, то:
  * при наличии кеша <esm2_dir>/<origin[:2]>/<origin>.npy используем cv_catalytic_ESM2
  * иначе используем дефолтную cv_catalytic_noMSA (fallback для "неудачников")

Выходной CSV:
- protein id (uniprot / PDB)  # 'ACC_F#'
- predicted with 35% threshold
- predicted with 65% threshold
- predicted with 85% threshold
- known catalytic sites
- EC number (if exists)
- protein name
- gene name
"""

# ---------- Thread limits: must be set BEFORE numpy/keras/tf imports ----------
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
# ---------------------------------------------------------------------------

import re
import sys
import json
import time
import argparse
from glob import glob
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# ---------------- ScanNet_Ub in sys.path ----------------
PROJECT_ROOT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import predict_bindingsites  # noqa: E402

# ---------------- helpers -------------------------
UNIPROT_RE = re.compile(r"[A-Z0-9]{6,10}", re.I)


def ensure_trailing_slash(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if p.endswith(os.sep) else p + os.sep


def entry_from_path(path: str) -> str:
    """
    Convert AF structure filename into "pretty" key for CSV/meta.
    AF-P81877-F1-model_v4.cif -> 'P81877_F1'
    If F# is not found -> fallback to ACC if possible.
    """
    base = os.path.basename(path)
    m = re.match(r"AF-([A-Z0-9]{6,10})-F(\d+)-model_", base, re.I)
    if m:
        return f"{m.group(1).upper()}_F{m.group(2)}"
    stem = os.path.splitext(base)[0]
    m2 = UNIPROT_RE.search(stem.upper())
    return m2.group(0) if m2 else stem


def af_stem(path: str) -> str:
    """
    Return AF-stem to pass into predict_bindingsites:
    '/.../AF-P81877-F1-model_v6.cif' -> 'AF-P81877-F1-model_v6'
    """
    return os.path.splitext(os.path.basename(path))[0]


def acc_only(name: str) -> str:
    """'P81877_F1' -> 'P81877'"""
    return name.split("_", 1)[0].upper()


# ---- MSA existence check for AF-stem ----
def msa_exists_for_stem(msa_root: str, stem: str) -> bool:
    """
    Check if at least one file exists:
      MSA_<stem>_0_A.fasta / MSA_<stem>_0_B.fasta / ...
    """
    if not msa_root:
        return False
    pattern = os.path.join(msa_root, f"MSA_{stem}_*_*.fasta")
    return bool(glob(pattern))


# ---- ESM2 cache existence check (origin-based) ----
def esm_cache_path(out_dir: str, origin: str) -> str:
    sub = origin[:2] if len(origin) >= 2 else "__"
    return os.path.join(out_dir, sub, f"{origin}.npy")


def esm_exists_for_origin(esm2_root: str, origin: str) -> bool:
    """
    ESM2 cache path convention:
      <esm2_root>/<origin[:2]>/<origin>.npy
    We try both exact origin and lowercased origin (just in case).
    """
    if not esm2_root:
        return False
    p1 = esm_cache_path(esm2_root, origin)
    if os.path.exists(p1):
        return True
    p2 = esm_cache_path(esm2_root, origin.lower())
    return os.path.exists(p2)


# --------------- unified MetaDB -------------------
class MetaDB:
    """
    Unified JSON: {
      "ACC": {
        "full_name": str | null,
        "gene_name": str | null,
        "ec_numbers": [str, ...] | null,
        "active_sites": [{"pos": int, "aa": str?, "description": str?}, ...] | null
      }, ...
    }
    """
    def __init__(self, path: Optional[str]):
        self.db: Dict[str, Dict[str, Any]] = {}
        if path:
            with open(path, "r") as f:
                raw = json.load(f)
            # normalize keys to upper
            self.db = {str(k).upper(): (v or {}) for k, v in raw.items()}

    def names(self, acc_like: str) -> Tuple[str, str]:
        r = self.db.get(acc_only(acc_like), {})
        return (r.get("full_name") or ""), (r.get("gene_name") or "")

    def ecs(self, acc_like: str) -> List[str]:
        r = self.db.get(acc_only(acc_like), {})
        ecs = r.get("ec_numbers") or []
        return list(ecs) if isinstance(ecs, list) else []

    def known_positions(self, acc_like: str) -> List[int]:
        r = self.db.get(acc_only(acc_like), {})
        sites = r.get("active_sites") or []
        pos = []
        if isinstance(sites, list):
            for it in sites:
                if isinstance(it, dict) and "pos" in it:
                    try:
                        p = int(it["pos"])
                        if p >= 1:
                            pos.append(p)
                    except Exception:
                        pass
        return sorted(set(pos))


# --------------- model wrappers -------------------
def _make_esm2_pipeline(esm2_dir: str):
    """
    Build ScanNetPipeline that uses cached ESM2 embeddings (aa_features='esm2').
    """
    return predict_bindingsites.pipelines.ScanNetPipeline(
        with_aa=True,
        with_atom=True,
        aa_features="esm2",
        atom_features="valency",
        aa_frames="triplet_sidechain",
        Beff=500,
        homology_search=getattr(predict_bindingsites, "homology_search", "mmseqs"),
        esm2_dir=esm2_dir,
    )


def run_cv_catalytic_inference(
    struct_paths: List[str],
    entry_names: List[str],   # query_names passed into predict_bindingsites (AF-stems)
    mode: str,                # 'msa' | 'nomsa' | 'esm2'
    ncores: int,
    msa_dir: Optional[str] = None,
    esm2_dir: Optional[str] = None,
):
    # Select pipeline/model depending on mode
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
        msa_folder = predict_bindingsites.MSA_folder  # unused, but required by interface

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


def catalytic_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # If there are 2 columns, use the second as "catalyticness"
        return (arr[:, 0] if arr.shape[1] == 1 else arr[:, 1]).astype(float)
    return np.asarray(arr).reshape(-1).astype(float)


def hits_to_str(probs: np.ndarray, resids: Optional[np.ndarray], thr: float) -> str:
    if resids is None or len(resids) != len(probs):
        resids = np.arange(1, len(probs) + 1, dtype=int)
    return ",".join(map(str, resids[probs >= thr].tolist()))


# ----------------------- CLI ---------------------
def main():
    t0 = time.time()
    print("[BOOT] starting predict.py", flush=True)

    ap = argparse.ArgumentParser(description="Catalytic-site inference over AF Human (cv_catalytic, one meta JSON).")
    ap.add_argument("--structures_dir", default="/home/iscb/wolfson/jeromet/AFDB/Human_v2")
    ap.add_argument("--msa_dir", default="/home/iscb/wolfson/jeromet/Data/MSA_v2")
    ap.add_argument("--meta_json", required=True, help="Unified JSON with names/genes/EC/catalytic sites")

    ap.add_argument("--use_msa", action="store_true", help="Use MSA when present (strict: do NOT build missing MSAs)")
    ap.add_argument("--use_esm2", action="store_true", help="Use ESM2 cached embeddings when present; fallback to noMSA if missing")
    ap.add_argument("--esm2_dir", default=None, help="Root folder with cached ESM2 .npy: <root>/<origin[:2]>/<origin>.npy")

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

    # 1) Collect structures:
    #    run_name = AF-stem (for predict_bindingsites + MSA/ESM2 lookup)
    #    disp_key = 'ACC_F#' (for CSV/MetaDB)
    print(f"[STEP] scanning structures under {structures_dir}", flush=True)
    cif_paths = glob(os.path.join(structures_dir, "*.cif"))
    pdb_paths = glob(os.path.join(structures_dir, "*.pdb"))
    by_runname: Dict[str, str] = {}     # AF-stem -> path
    display_for: Dict[str, str] = {}    # AF-stem -> 'ACC_F#'

    for p in cif_paths + pdb_paths:
        rn = af_stem(p)               # 'AF-P81877-F1-model_v6'
        disp = entry_from_path(p)     # 'P81877_F1'
        prev = by_runname.get(rn)
        if prev is None or (prev.lower().endswith(".cif") and p.lower().endswith(".pdb")):
            by_runname[rn] = p
            display_for[rn] = disp

    entries_all = sorted(by_runname.keys())
    if not entries_all:
        raise SystemExit(f"No structures in {structures_dir} (*.cif|*.pdb)")

    print(f"[INFO] using {len(entries_all)} structures (prefer .pdb when both exist) "
          f"in {time.time()-t0:.1f}s", flush=True)

    # 2) Split by mode
    preds_raw: Dict[str, np.ndarray] = {}
    resids_map: Dict[str, np.ndarray] = {}

    # 3) Meta
    meta = MetaDB(args.meta_json)

    if args.use_esm2:
        # ESM2 mode: embeddings present -> ESM2 model, else fallback to default noMSA model
        with_esm2 = [rn for rn in entries_all if esm_exists_for_origin(esm2_dir, rn)]
        without_esm2 = [rn for rn in entries_all if rn not in set(with_esm2)]

        print(f"[INFO] ESM2 MODE: will use ESM2 for {len(with_esm2)}, "
              f"fallback noMSA for {len(without_esm2)}.", flush=True)
        if without_esm2:
            print("[DEBUG] examples without detected ESM2 cache (first 10):",
                  ", ".join(without_esm2[:10]), flush=True)

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
        # Original MSA strict split (if requested), else everything is noMSA
        if args.use_msa and msa_dir:
            with_msa = [rn for rn in entries_all if msa_exists_for_stem(msa_dir, rn)]
        else:
            with_msa = []
        without_msa = [rn for rn in entries_all if rn not in set(with_msa)]

        print(f"[INFO] STRICT MSA MODE: will NOT build missing MSAs; "
              f"use_MSA for {len(with_msa)}, noMSA for {len(without_msa)}.", flush=True)
        if args.use_msa and msa_dir and without_msa:
            print("[DEBUG] examples without detected MSA (first 10):",
                  ", ".join(without_msa[:10]), flush=True)

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

    # 4) Build output table (iterate by AF-stem, but show 'ACC_F#')
    rows = []
    for rn in entries_all:
        disp = display_for[rn]  # 'ACC_F#'
        arr = preds_raw.get(rn)

        if arr is None:
            probs = np.zeros(0, dtype=float)
            resids = None
        else:
            probs = catalytic_channel(np.asarray(arr))
            resids = resids_map.get(rn)

        known = meta.known_positions(disp)
        ecs = meta.ecs(disp)
        full_name, gene_name = meta.names(disp)

        rows.append({
            "protein id (uniprot / PDB)": disp,
            f"predicted with {int(args.thr_extra*100)}% threshold": hits_to_str(probs, resids, args.thr_extra) if probs.size else "",
            f"predicted with {int(args.thr_lo*100)}% threshold": hits_to_str(probs, resids, args.thr_lo) if probs.size else "",
            f"predicted with {int(args.thr_hi*100)}% threshold": hits_to_str(probs, resids, args.thr_hi) if probs.size else "",
            "known catalytic sites": ",".join(map(str, known)) if known else "",
            "EC number (if exists)": ";".join(ecs) if ecs else "",
            "protein name": full_name,
            "gene name": gene_name,
        })

    cols = [
        "protein id (uniprot / PDB)",
        f"predicted with {int(args.thr_extra*100)}% threshold",
        f"predicted with {int(args.thr_lo*100)}% threshold",
        f"predicted with {int(args.thr_hi*100)}% threshold",
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
