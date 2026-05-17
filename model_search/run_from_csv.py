#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import subprocess
import sys


def parse_bool(value):
    if value is None:
        return False
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y"}


def load_row(csv_path, index):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if index < 0 or index >= len(rows):
        raise IndexError(f"Row index {index} is out of range for {len(rows)} rows")

    return rows[index], len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to runs.csv")
    parser.add_argument("--index", type=int, required=True, help="0-based row index, excluding header")
    parser.add_argument("--script", required=True, help="Path to esm2_no_geometry.py")
    args = parser.parse_args()

    row, total = load_row(args.csv, args.index)

    cmd = [sys.executable, args.script]

    # Required / common args
    cmd += ["--model_name", row["model_name"]]
    cmd += ["--run_name", row["run_name"]]
    cmd += ["--esm2_dir", row["esm2_dir"]]
    cmd += ["--esm2_version", row["esm2_version"]]
    cmd += ["--architecture", row["architecture"]]
    cmd += ["--learning_rate", str(row["learning_rate"])]
    cmd += ["--weight_decay", str(row["weight_decay"])]
    cmd += ["--batch_size", str(row["batch_size"])]
    cmd += ["--cv_fold", str(row["cv_fold"])]
    cmd += ["--seed", str(row["seed"])]

    if row.get("esm2_layer", "").strip() != "":
        cmd += ["--esm2_layer", str(row["esm2_layer"])]

    hidden_dims = row.get("hidden_dims", "").strip()
    if hidden_dims:
        cmd += ["--hidden_dims"] + hidden_dims.replace(",", " ").split()

    dropout = row.get("dropout", "").strip()
    if dropout:
        cmd += ["--dropout", dropout]

    head_norm = row.get("head_norm", "").strip()
    if head_norm:
        cmd += ["--head_norm", head_norm]

    activation = row.get("activation", "").strip()
    if activation:
        cmd += ["--activation", activation]

    if parse_bool(row.get("skip_existing", 0)):
        cmd += ["--skip_existing"]
    if parse_bool(row.get("fresh", 0)):
        cmd += ["--fresh"]
    if parse_bool(row.get("tensorboard", 0)):
        cmd += ["--tensorboard"]
    if parse_bool(row.get("check", 0)):
        cmd += ["--check"]

    print(f"[INFO] Running row {args.index + 1}/{total}")
    print("[CMD]", " ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()