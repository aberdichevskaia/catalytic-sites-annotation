#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

OUT_CSV = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/model_search/runs_winner.csv"

MODEL_NAME = "Catalytic_Sites_retrain_final"
ARCHITECTURE = "MLP"
BATCH_SIZE = 1

WINNER = {
    "esm2_dir": "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/ESM2_3B_layer24",
    "esm2_version": "esm2_t36_3B_UR50D",
    "esm2_layer": "24",
    "architecture": "MLP",
    "hidden_dims": "1024 256",
    "dropout": "0.3",
    "head_norm": "layernorm",
    "activation": "relu",
    "learning_rate": "0.0001",
    "weight_decay": "0.0001",
    "batch_size": "1",
    "skip_existing": "1",
    "fresh": "0",
    "tensorboard": "0",
    "check": "0",
}

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [1, 2, 3, 4, 5, 6]

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
        f'__winner'
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

    for fold in FOLDS:
        for seed in SEEDS:
            row = {
                "model_name": MODEL_NAME,
                "run_name": "",
                "esm2_dir": WINNER["esm2_dir"],
                "esm2_version": WINNER["esm2_version"],
                "esm2_layer": WINNER["esm2_layer"],
                "architecture": WINNER["architecture"],
                "hidden_dims": WINNER["hidden_dims"],
                "dropout": WINNER["dropout"],
                "head_norm": WINNER["head_norm"],
                "activation": WINNER["activation"],
                "learning_rate": WINNER["learning_rate"],
                "weight_decay": WINNER["weight_decay"],
                "batch_size": WINNER["batch_size"],
                "cv_fold": str(fold),
                "seed": str(seed),
                "skip_existing": WINNER["skip_existing"],
                "fresh": WINNER["fresh"],
                "tensorboard": WINNER["tensorboard"],
                "check": WINNER["check"],
            }
            row["run_name"] = build_run_name(row)
            rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} runs to {os.path.abspath(OUT_CSV)}")

if __name__ == "__main__":
    main()