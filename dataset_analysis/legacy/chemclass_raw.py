#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import defaultdict

import pandas as pd

BASE_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9"
DATASET_CSV = os.path.join(BASE_DIR, "dataset.csv")
SPLIT_TEMPLATE = os.path.join(BASE_DIR, "split{}.txt")

WEIGHT_COL = "W_Structure"   # change if you want W_Sequence etc.


# ---------- Chemotype definition ----------

def get_catalytic_class(residues):
    """Return chemotype id (0..7) given list of catalytic residues."""
    if any(r in residues for r in "ILMVWF"):
        return 0
    if any(r in residues for r in "AGP"):
        return 1
    if any(r in residues for r in "QN"):
        return 2
    if any(r in residues for r in "KR"):
        return 3
    if any(r == "S" for r in residues):
        return 4
    if any(r == "T" for r in residues):
        return 5
    if any(r in residues for r in "DE"):
        return 6
    return 7  # other / none


CLASS_LABELS = {
    0: "Class 0 (ILMVWF)",
    1: "Class 1 (AGP)",
    2: "Class 2 (QN)",
    3: "Class 3 (KR)",
    4: "Class 4 (S)",
    5: "Class 5 (T)",
    6: "Class 6 (DE)",
    7: "Class 7 (other/none)",
}


# ---------- Parsing splits ----------

def parse_splits():
    """
    Parse split*.txt files and return:
      - seq_to_split: Sequence_ID -> split name (e.g. 'split1')
      - residues_by_seq: Sequence_ID -> list of catalytic residues
    """
    seq_to_split = {}
    residues_by_seq = defaultdict(list)

    for i in range(1, 6):
        split_name = f"split{i}"
        path = SPLIT_TEMPLATE.format(i)
        with open(path) as f:
            current_seq = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    current_seq = line[1:].strip()
                    seq_to_split[current_seq] = split_name
                    continue

                if current_seq is None:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                aa = parts[2]
                try:
                    label = int(parts[3])
                except ValueError:
                    continue

                if label == 1:
                    residues_by_seq[current_seq].append(aa)

    return seq_to_split, residues_by_seq


def main():
    # Load dataset with weights
    df = pd.read_csv(DATASET_CSV)
    df = df.drop_duplicates(subset=["Sequence_ID"])
    df = df.set_index("Sequence_ID")

    seq_to_split, residues_by_seq = parse_splits()

    # Only keep Sequence_IDs present in both dataset.csv and splits
    common_ids = sorted(set(df.index) & set(seq_to_split.keys()))
    print(f"[INFO] Sequence_ID in dataset: {len(df)}")
    print(f"[INFO] IDs from splits:        {len(seq_to_split)}")
    print(f"[INFO] Intersection size:       {len(common_ids)}")

    records = []
    for seq_id in common_ids:
        residues = residues_by_seq.get(seq_id, [])
        chem_id = get_catalytic_class(residues)
        split = seq_to_split[seq_id]
        weight = df.at[seq_id, WEIGHT_COL]

        records.append(
            {
                "Sequence_ID": seq_id,
                "Split": split,
                "ChemClassLabel": CLASS_LABELS[chem_id],
                "Weight": weight,
            }
        )

    df_long = pd.DataFrame(records)

    # Pivot: rows = split, columns = chemotype, values = raw weight sums
    pivot = df_long.pivot_table(
        index="Split",
        columns="ChemClassLabel",
        values="Weight",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Add total weight per split
    pivot["Total weight"] = pivot.sum(axis=1)

    # Move "Total weight" to the first column
    cols = ["Total weight"] + [c for c in pivot.columns if c != "Total weight"]
    pivot = pivot[cols]

    out_dir = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/data_plots/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "chemclass_split_weights_raw.csv")

    pivot.to_csv(out_csv, float_format="%.6f")
    print(f"[OK] Saved raw chemotype weights per split to {out_csv}")


if __name__ == "__main__":
    main()
