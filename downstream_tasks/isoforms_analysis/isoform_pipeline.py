#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end isoform pipeline:
1) Scan structures in a directory.
2) Run inference with ESM2 preferred and noMSA fallback (GPU stage).
3) Rank isoforms by base-id spread score (CPU stage).
4) Emit a summary table with per-isoform metadata and base-id rank.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

from rank_isoforms_part1 import run_stage as run_gpu_stage
from rank_isoforms_part2 import run_stage as run_cpu_stage


def _default_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


def _add_rank(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        scores["interest_rank"] = []
        return scores
    scores = scores.copy()
    scores["interest_rank"] = scores["score"].rank(method="dense", ascending=False).astype(int)
    return scores


def run_pipeline(
    structures_dir: str,
    esm2_dir: Optional[str],
    out_dir: str,
    ncores: int,
    nproc: int,
    min_prob: float,
    min_isoforms: int,
    maxtasksperchild: int,
    out_npz: Optional[str],
    manifest_csv: Optional[str],
    scores_csv: Optional[str],
    summary_csv: Optional[str],
    seq_cache_csv: Optional[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    out_npz = out_npz or _default_path(out_dir, "isoform_preds_esm2.npz")
    manifest_csv = manifest_csv or _default_path(out_dir, "isoform_preds_manifest.csv")
    scores_csv = scores_csv or _default_path(out_dir, "isoform_spread_scores.csv")
    summary_csv = summary_csv or _default_path(out_dir, "isoform_summary.csv")

    run_gpu_stage(
        structures_dir=structures_dir,
        esm2_dir=esm2_dir,
        out_npz=out_npz,
        manifest_csv=manifest_csv,
        ncores=ncores,
        min_prob=min_prob,
    )

    scores_df = run_cpu_stage(
        dump_npz=out_npz,
        out_csv=scores_csv,
        nproc=nproc,
        min_isoforms=min_isoforms,
        maxtasksperchild=maxtasksperchild,
        seq_cache_csv=seq_cache_csv,
    )

    scores_rank = _add_rank(scores_df)

    if os.path.exists(manifest_csv):
        manifest_df = pd.read_csv(manifest_csv)
    else:
        manifest_df = pd.DataFrame()

    if not manifest_df.empty:
        merged = manifest_df.merge(
            scores_rank,
            how="left",
            left_on="base_id",
            right_on="base id",
        )
        merged.drop(columns=["base id"], inplace=True, errors="ignore")
        merged.to_csv(summary_csv, index=False)
        print(f"[OK] wrote summary -> {summary_csv}", flush=True)
    else:
        scores_rank.to_csv(summary_csv, index=False)
        print(f"[OK] wrote summary (scores only) -> {summary_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ESM2-first isoform pipeline end-to-end.")
    ap.add_argument("--structures_dir", required=True, help="Directory with isoform structures.")
    ap.add_argument("--esm2_dir", default=None, help="Root of cached ESM2 embeddings.")
    ap.add_argument("--out_dir", required=True, help="Output directory for pipeline artifacts.")
    ap.add_argument("--out_npz", default=None, help="Optional override for stage-1 NPZ output.")
    ap.add_argument("--manifest_csv", default=None, help="Optional override for manifest CSV output.")
    ap.add_argument("--scores_csv", default=None, help="Optional override for base-id score output.")
    ap.add_argument("--summary_csv", default=None, help="Optional override for merged summary output.")
    ap.add_argument("--seq_cache_csv", default=None, help="Optional sequence cache CSV path.")
    ap.add_argument("--ncores", type=int, default=8, help="Cores for GPU stage inference.")
    ap.add_argument("--nproc", type=int, default=16, help="Processes for CPU stage.")
    ap.add_argument("--min_prob", type=float, default=0.1, help="Zero out probabilities below this value.")
    ap.add_argument("--min_isoforms", type=int, default=2, help="Minimum isoforms per base-id.")
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
        out_npz=args.out_npz,
        manifest_csv=args.manifest_csv,
        scores_csv=args.scores_csv,
        summary_csv=args.summary_csv,
        seq_cache_csv=args.seq_cache_csv,
    )


if __name__ == "__main__":
    main()
