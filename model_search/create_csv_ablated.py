#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ablated_runs.csv for train_ablated2.py --ablate_config 1 (MLP aa_encoder only).

Sweep: ESM2 3B only (layer36 and layer24), learning_rate=[1e-4, 3e-4],
       dropout=[0.1, 0.3], hidden_dims=[256-64, 512-128],
       folds=[1..5], seeds=[1..6].  No 150M.

Total rows: 2 esm2 × 2 lr × 2 dropout × 2 hidden_dims × 5 folds × 6 seeds = 480.
"""

import csv
import itertools
import os

OUT_CSV = "ablated_runs.csv"

MODEL_NAME_BASE = "ScanNet_ablated"
ABLATE_CONFIG = 1
BATCH_SIZE = 1

EXPERIMENTS = [
    {
        "esm2_version": "esm2_t36_3B_UR50D",
        "esm2_layer": 36,
        "esm2_dir": "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/ESM2_3B_layer36",
    },
    {
        "esm2_version": "esm2_t36_3B_UR50D",
        "esm2_layer": 24,
        "esm2_dir": "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/ESM2_3B_layer24",
    },
]

LEARNING_RATES = [1e-4, 3e-4]
DROPOUTS = [0.1, 0.3]
HIDDEN_DIMS = [(256, 64), (512, 128)]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [1, 2, 3, 4, 5, 6]

USE_SKIP_EXISTING = 1
USE_FRESH = 0
USE_CHECK = 0

RESULTS_DIR = "/home/iscb/wolfson/annab4/ablate_grid"
MODELS_DIR  = "/home/iscb/wolfson/annab4/ablate_grid_models"

FIELDNAMES = [
    "model_name",
    "esm2_dir",
    "esm2_version",
    "esm2_layer",
    "learning_rate",
    "dropout",
    "hidden_dims",
    "batch_size",
    "cv_fold",
    "seed",
    "ablate_config",
    "results_dir",
    "models_dir",
    "skip_existing",
    "fresh",
    "check",
]


def build_model_name(exp, lr, dropout, hidden_dims, fold, seed):
    """
    Encode all config params in the model_name so parse_run_dir_name can extract them.
    train_ablated2.py will prepend 'ablate01_', giving the final output dir name.
    """
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    hd_str = "-".join(str(h) for h in hidden_dims)
    return (
        f"{MODEL_NAME_BASE}"
        f"__{exp['esm2_version']}"
        f"__layer{exp['esm2_layer']}"
        f"__lr={lr_str}"
        f"__drop={dropout}"
        f"__hd={hd_str}"
        f"__bs={BATCH_SIZE}"
        f"__fold={fold}"
        f"__seed={seed}"
    )


def main():
    rows = []

    for exp, lr, dropout, hidden_dims, fold, seed in itertools.product(
        EXPERIMENTS, LEARNING_RATES, DROPOUTS, HIDDEN_DIMS, FOLDS, SEEDS
    ):
        hd_str = "-".join(str(h) for h in hidden_dims)
        row = {
            "model_name": build_model_name(exp, lr, dropout, hidden_dims, fold, seed),
            "esm2_dir": exp["esm2_dir"],
            "esm2_version": exp["esm2_version"],
            "esm2_layer": exp["esm2_layer"],
            "learning_rate": lr,
            "dropout": dropout,
            "hidden_dims": hd_str,
            "batch_size": BATCH_SIZE,
            "cv_fold": fold,
            "seed": seed,
            "ablate_config": ABLATE_CONFIG,
            "results_dir": RESULTS_DIR,
            "models_dir": MODELS_DIR,
            "skip_existing": USE_SKIP_EXISTING,
            "fresh": USE_FRESH,
            "check": USE_CHECK,
        }
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} runs to {os.path.abspath(OUT_CSV)} "
          f"(2 esm2 × 2 lr × 2 dropout × 2 hd × 5 folds × 6 seeds = {len(rows)})")
    print(f"[INFO] pickles/plots -> {RESULTS_DIR}/ablate01_*/")
    print(f"[INFO] model weights -> {MODELS_DIR}/ablate01_*")


if __name__ == "__main__":
    main()
