#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner for ablated_runs.csv → train_ablated2.py.
Usage: python run_ablated_from_csv.py --csv ablated_runs.csv --index $SLURM_ARRAY_TASK_ID
"""

import argparse
import csv
import subprocess
import sys


def parse_bool(value):
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_row(csv_path, index):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Row index {index} is out of range for {len(rows)} rows")
    return rows[index], len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to ablated_runs.csv")
    parser.add_argument("--index", type=int, required=True, help="0-based row index")
    parser.add_argument("--script", required=True, help="Path to train_ablated2.py")
    args = parser.parse_args()

    row, total = load_row(args.csv, args.index)

    cmd = [sys.executable, args.script]

    cmd += ["--model_name", row["model_name"]]
    cmd += ["--aa_features", "esm2"]
    cmd += ["--esm2_dir", row["esm2_dir"]]
    cmd += ["--esm2_version", row["esm2_version"]]

    esm2_layer = row.get("esm2_layer", "").strip()
    if esm2_layer:
        cmd += ["--esm2_layer", esm2_layer]

    cmd += ["--learning_rate", str(row["learning_rate"])]

    dropout = row.get("dropout", "").strip()
    if dropout:
        cmd += ["--aa_encoder_dropout", dropout]

    hidden_dims = row.get("hidden_dims", "").strip()
    if hidden_dims:
        cmd += ["--aa_encoder_mlp_hidden"] + hidden_dims.split("-")

    cmd += ["--batch_size", str(row["batch_size"])]
    cmd += ["--cv_fold", str(row["cv_fold"])]
    cmd += ["--seed", str(row["seed"])]
    cmd += ["--ablate_config", str(row["ablate_config"])]

    results_dir = row.get("results_dir", "").strip()
    if results_dir:
        cmd += ["--results_dir", results_dir]

    models_dir = row.get("models_dir", "").strip()
    if models_dir:
        cmd += ["--models_dir", models_dir]

    if parse_bool(row.get("skip_existing", 0)):
        cmd += ["--skip_existing"]
    if parse_bool(row.get("fresh", 0)):
        cmd += ["--fresh"]
    if parse_bool(row.get("check", 0)):
        cmd += ["--check"]

    print(f"[INFO] Running row {args.index + 1}/{total}")
    print("[CMD]", " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
