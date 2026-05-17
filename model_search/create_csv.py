#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import itertools
import os

OUT_CSV = "runs2.csv"

TRAIN_SCRIPT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/baselines/esm2_no_geometry.py"

MODEL_NAME = "Catalytic_Sites_retrain"
ARCHITECTURE = "MLP"
BATCH_SIZE = 1

# Что именно перебираем
EXPERIMENTS = [
    {
        "esm2_version": "esm2_t30_150M_UR50D",
        "esm2_layer": 30,
        "esm2_dir": "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/ESM2_150M_layer30",
    },
    {
        "esm2_version": "esm2_t30_150M_UR50D",
        "esm2_layer": 20,
        "esm2_dir": "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/ESM2_150M_layer20",
    },
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

HIDDEN_DIMS = [
    "512 128",
    "1024 256",
    #"1024 512 128",
]

DROPOUTS = [0.1, 0.3]
HEAD_NORMS = ["layernorm"]
ACTIVATIONS = ["relu"]

# Убрали 1e-3, оставили более спокойные learning rates
LEARNING_RATES = [1e-4, 3e-4]
WEIGHT_DECAYS = [0.0, 1e-4]

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [4, 5, 6]

# Можно включить/выключить опции сразу на весь sweep
USE_SKIP_EXISTING = 1
USE_FRESH = 0
USE_TENSORBOARD = 0
USE_CHECK = 0

FIELDNAMES = [
    "model_name",
    "run_name",
    "esm2_dir",
    "esm2_version",
    "esm2_layer",
    "architecture",
    "hidden_dims",
    "dropout",
    "head_norm",
    "activation",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "cv_fold",
    "seed",
    "skip_existing",
    "fresh",
    "tensorboard",
    "check",
]


def build_run_name(row):
    hidden_tag = row["hidden_dims"].replace(" ", "-")
    return (
        f'{row["model_name"]}'
        f'__{row["esm2_version"]}'
        f'__layer{row["esm2_layer"]}'
        f'__arch={row["architecture"]}'
        f'__hd={hidden_tag}'
        f'__drop={row["dropout"]}'
        f'__norm={row["head_norm"]}'
        f'__act={row["activation"]}'
        f'__lr={row["learning_rate"]}'
        f'__wd={row["weight_decay"]}'
        f'__bs={row["batch_size"]}'
        f'__fold={row["cv_fold"]}'
        f'__seed={row["seed"]}'
    )


def main():
    rows = []

    for exp, hidden_dims, dropout, head_norm, activation, lr, wd, fold, seed in itertools.product(
        EXPERIMENTS,
        HIDDEN_DIMS,
        DROPOUTS,
        HEAD_NORMS,
        ACTIVATIONS,
        LEARNING_RATES,
        WEIGHT_DECAYS,
        FOLDS,
        SEEDS,
    ):
        row = {
            "model_name": MODEL_NAME,
            "run_name": "",
            "esm2_dir": exp["esm2_dir"],
            "esm2_version": exp["esm2_version"],
            "esm2_layer": exp["esm2_layer"],
            "architecture": ARCHITECTURE,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "head_norm": head_norm,
            "activation": activation,
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": BATCH_SIZE,
            "cv_fold": fold,
            "seed": seed,
            "skip_existing": USE_SKIP_EXISTING,
            "fresh": USE_FRESH,
            "tensorboard": USE_TENSORBOARD,
            "check": USE_CHECK,
        }
        row["run_name"] = build_run_name(row)
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} runs to {os.path.abspath(OUT_CSV)}")
    print(f"[INFO] training script expected at: {TRAIN_SCRIPT}")


if __name__ == "__main__":
    main()