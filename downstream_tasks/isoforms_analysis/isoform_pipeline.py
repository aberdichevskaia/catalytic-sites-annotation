#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end isoform pipeline:
1) Scan structures in a directory.
2) Run inference with ESM2 preferred and noMSA fallback (GPU stage).
3) Analyse predictions: diff_score + loss categories (CPU stage).
4) Emit a summary table joining manifest and per-isoform analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

from run_inference import run_stage as run_gpu_stage
from analyse_npz import run as run_analysis


def _default_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


def run_pipeline(
    structures_dir: str,
    esm2_dir: Optional[str],
    out_dir: str,
    ncores: int,
    nproc: int,
    min_prob: float,
    min_isoforms: int,
    maxtasksperchild: int,
    tau: float,
    out_npz: Optional[str],
    manifest_csv: Optional[str],
    analysis_csv: Optional[str],
    summary_csv: Optional[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    out_npz      = out_npz      or _default_path(out_dir, "isoform_preds_esm2.npz")
    manifest_csv = manifest_csv or _default_path(out_dir, "isoform_preds_manifest.csv")
    analysis_csv = analysis_csv or _default_path(out_dir, "isoforms_analysis.csv")
    summary_csv  = summary_csv  or _default_path(out_dir, "isoform_summary.csv")

    run_gpu_stage(
        structures_dir=structures_dir,
        esm2_dir=esm2_dir,
        out_npz=out_npz,
        manifest_csv=manifest_csv,
        ncores=ncores,
        min_prob=min_prob,
    )

    run_analysis(
        dump_npz=out_npz,
        out_csv=analysis_csv,
        tau=tau,
        nproc=nproc,
        min_isoforms=min_isoforms,
        maxtasksperchild=maxtasksperchild,
    )

    analysis_df = pd.read_csv(analysis_csv)

    if os.path.exists(manifest_csv):
        manifest_df = pd.read_csv(manifest_csv)
        merged = manifest_df.merge(
            analysis_df,
            how="left",
            left_on="isoform_id",
            right_on="isoform",
        )
        merged.drop(columns=["isoform"], inplace=True, errors="ignore")
        merged.to_csv(summary_csv, index=False)
        log.info("wrote summary -> %s", summary_csv)
    else:
        analysis_df.to_csv(summary_csv, index=False)
        log.info("wrote summary (analysis only) -> %s", summary_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ESM2-first isoform pipeline end-to-end.")
    ap.add_argument("--structures_dir", required=True, help="Directory with isoform structures.")
    ap.add_argument("--esm2_dir",       default=None,  help="Root of cached ESM2 embeddings.")
    ap.add_argument("--out_dir",        required=True, help="Output directory for pipeline artifacts.")
    ap.add_argument("--out_npz",        default=None,  help="Override for stage-1 NPZ output.")
    ap.add_argument("--manifest_csv",   default=None,  help="Override for manifest CSV output.")
    ap.add_argument("--analysis_csv",   default=None,  help="Override for per-isoform analysis CSV.")
    ap.add_argument("--summary_csv",    default=None,  help="Override for merged summary CSV.")
    ap.add_argument("--ncores",         type=int,   default=8,    help="Cores for GPU stage.")
    ap.add_argument("--nproc",          type=int,   default=16,   help="Processes for CPU stage.")
    ap.add_argument("--min_prob",       type=float, default=0.1,  help="Zero out probs below this.")
    ap.add_argument("--min_isoforms",   type=int,   default=2,    help="Minimum isoforms per base-id.")
    ap.add_argument("--tau",            type=float, default=0.35, help="Threshold for loss categories.")
    ap.add_argument("--maxtasksperchild", type=int, default=50)
    args = ap.parse_args()

    run_pipeline(
        structures_dir=args.structures_dir,
        esm2_dir=args.esm2_dir,
        out_dir=args.out_dir,
        ncores=args.ncores,
        nproc=args.nproc,
        min_prob=args.min_prob,
        min_isoforms=args.min_isoforms,
        maxtasksperchild=args.maxtasksperchild,
        tau=args.tau,
        out_npz=args.out_npz,
        manifest_csv=args.manifest_csv,
        analysis_csv=args.analysis_csv,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()
