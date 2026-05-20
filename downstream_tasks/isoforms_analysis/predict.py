#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Postprocessing: annotate NPZ predictions with metadata and apply thresholds.

Reads the NPZ produced by rank_isoforms_part1.py (GPU stage) and writes a
human-readable CSV.  No inference is performed here.

Input:
  --dump_npz    NPZ produced by rank_isoforms_part1.py
  --meta_json   Optional JSON with names/genes/EC/active sites
  --thresholds  One or more probability thresholds (default: 0.35 0.65 0.85)
  --out_csv     Output CSV path

Output columns:
  protein id (uniprot / PDB), base uniprot id, inference_type,
  predicted with X% threshold  (one column per threshold, comma-separated residue numbers),
  known catalytic sites, EC number (if exists), protein name, gene name
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd


# --------------- metadata helpers -------------------

def _acc_only(name: str) -> str:
    """'P81877_F1' -> 'P81877';  'A0A0K2S4Q6-1' -> 'A0A0K2S4Q6-1'."""
    return name.split("_", 1)[0].upper()


class MetaDB:
    """
    Unified metadata wrapper.

    Supports two JSON formats:

    1) Old ACC-format:
       { "ACC": { "full_name": ..., "gene_name": ...,
                  "ec_numbers": [...], "active_sites": [{"pos": int}, ...] } }

    2) Isoform-format (isoform_meta.json):
       { "O14733-2": [{ "full_name": ..., "gene_name": ...,
                        "ec_numbers": [...], "active_sites": [243, ...],
                        "base_id": "O14733" }] }
    """

    def __init__(self, path: Optional[str]):
        self.db: Dict[str, Dict[str, Any]] = {}
        if not path:
            return

        with open(path, "r") as fh:
            raw = json.load(fh)

        norm: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            key = str(k).upper()
            if isinstance(v, list):
                rec = v[0] if (v and isinstance(v[0], dict)) else {}
            elif isinstance(v, dict):
                rec = v
            else:
                rec = {}
            norm[key] = rec
        self.db = norm

    def _get_with_base(self, acc_like: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key = _acc_only(acc_like)
        rec = self.db.get(key, {})
        base_rec: Dict[str, Any] = {}
        base_id = rec.get("base_id")
        if isinstance(base_id, str):
            base_rec = self.db.get(base_id.upper(), {})
        return rec, base_rec

    def base_uniprot(self, acc_like: str) -> str:
        key = _acc_only(acc_like)
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
        sites = rec.get("active_sites") or base_rec.get("active_sites") or []
        pos: List[int] = []
        for it in (sites if isinstance(sites, list) else []):
            try:
                p = int(it["pos"] if isinstance(it, dict) else it)
                if p >= 1:
                    pos.append(p)
            except Exception:
                pass
        return sorted(set(pos))


# --------------- output helpers -------------------

def hits_to_str(probs: np.ndarray, resids: np.ndarray, thr: float) -> str:
    """Residue numbers (1-based) with probability >= thr, comma-separated."""
    mask = probs >= thr
    return ",".join(map(str, resids[mask].tolist())) if mask.any() else ""


# ----------------------- CLI ---------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate isoform NPZ predictions with metadata → CSV."
    )
    ap.add_argument("--dump_npz", required=True,
                    help="NPZ produced by rank_isoforms_part1.py.")
    ap.add_argument("--meta_json", default=None,
                    help="JSON with names/genes/EC/active sites (optional).")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.35, 0.65, 0.85],
                    help="Probability thresholds for hit columns (default: 0.35 0.65 0.85).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    args = ap.parse_args()

    thresholds = sorted(args.thresholds)

    log.info("loading %s", args.dump_npz)
    data = np.load(args.dump_npz, allow_pickle=False)

    isoform_ids   = data["isoform_ids"]
    inference_types = data["inference_type"]
    offsets       = data["offsets"].astype(np.int64)
    prob_concat   = data["prob_concat"].astype(np.float32)

    if "resids_concat" in data:
        resids_concat: Optional[np.ndarray] = data["resids_concat"].astype(np.int32)
    else:
        log.warning("NPZ has no resids_concat; falling back to sequential 1-based residue indices")
        resids_concat = None

    meta = MetaDB(args.meta_json)

    rows: List[Dict[str, Any]] = []
    for i, iso_id in enumerate(isoform_ids):
        probs = prob_concat[offsets[i]:offsets[i + 1]]
        if resids_concat is not None:
            resids = resids_concat[offsets[i]:offsets[i + 1]]
        else:
            resids = np.arange(1, len(probs) + 1, dtype=np.int32)

        disp = str(iso_id)
        base_acc = meta.base_uniprot(disp)
        known    = meta.known_positions(disp)
        ecs      = meta.ecs(disp)
        full_name, gene_name = meta.names(disp)

        row: Dict[str, Any] = {
            "protein id (uniprot / PDB)": disp,
            "base uniprot id":            base_acc,
            "inference_type":             str(inference_types[i]),
        }
        for thr in thresholds:
            row[f"predicted with {int(round(thr * 100))}% threshold"] = hits_to_str(probs, resids, thr)
        row["known catalytic sites"]  = ",".join(map(str, known)) if known else ""
        row["EC number (if exists)"]  = ";".join(ecs) if ecs else ""
        row["protein name"]           = full_name
        row["gene name"]              = gene_name
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info("wrote %s rows -> %s", len(df), args.out_csv)


if __name__ == "__main__":
    main()
