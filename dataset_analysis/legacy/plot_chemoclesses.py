#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# --- пути ---

DATASET_CSV = (
    "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/"
    "weight_based_v9/dataset.csv"
)
SPLIT_PATTERN = (
    "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/"
    "weight_based_v9/split{}.txt"
)

OUTPUT_COUNTS_CSV = "chemclass_split_counts.csv"
OUTPUT_HEATMAP_PNG = "chemclass_split_heatmap_rownorm.png"
OUTPUT_HEATMAP2_PNG = "chemclass_split_heatmap_colnorm.png"


# --- хемоклассы, как у тебя ---

def get_catalytic_class(residues):
    """residues: список аминокислот в каталитических позициях."""
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
    return 7


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


# --- парсинг split*.txt ---

def parse_splits(split_pattern: str, n_splits: int = 5):
    """
    Читает split{i}.txt и возвращает:
    - seq_to_split: Sequence_ID (полный, типа A0A1L8G2K9_A) -> 'split1'..'split5'
    - residues_by_seq: Sequence_ID -> список каталитических аминокислот
    """
    seq_to_split = {}
    residues_by_seq = defaultdict(list)

    for i in range(1, n_splits + 1):
        path = split_pattern.format(i)
        split_name = f"split{i}"

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path, "r") as f:
            current_chain = None
            aas = []
            labs = []

            def flush_chain():
                nonlocal current_chain, aas, labs
                if current_chain is None:
                    return
                seq_id = current_chain  # без обрезания префикса
                cat_res = [aa for aa, lab in zip(aas, labs) if lab == 1]
                residues_by_seq[seq_id].extend(cat_res)
                # запоминаем сплит
                if seq_id in seq_to_split:
                    if seq_to_split[seq_id] != split_name:
                        raise ValueError(
                            f"{seq_id} appears in multiple splits: "
                            f"{seq_to_split[seq_id]} and {split_name}"
                        )
                else:
                    seq_to_split[seq_id] = split_name
                current_chain = None
                aas = []
                labs = []

            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush_chain()
                    current_chain = line[1:].split()[0]
                    aas = []
                    labs = []
                else:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    aa = parts[2]
                    lab = int(parts[3])
                    aas.append(aa)
                    labs.append(lab)

            flush_chain()

    return seq_to_split, residues_by_seq


def build_chemclass_split_table(dataset_csv: str,
                                seq_to_split: dict,
                                residues_by_seq: dict) -> pd.DataFrame:
    """
    Собирает counts-таблицу: chem_class_label × split.
    """
    df = pd.read_csv(dataset_csv).drop_duplicates(subset=["Sequence_ID"])

    # пересечение ID
    known_ids = set(df["Sequence_ID"])
    ids_in_splits = set(seq_to_split.keys())
    common = known_ids & ids_in_splits
    print(f"[INFO] Sequence_ID in dataset: {len(known_ids)}")
    print(f"[INFO] IDs from splits:        {len(ids_in_splits)}")
    print(f"[INFO] Intersection size:       {len(common)}")

    df = df[df["Sequence_ID"].isin(common)].copy()
    if df.empty:
        print("[WARN] After filtering by split IDs, dataset is empty!")

    # добавим сплит
    df["Split"] = df["Sequence_ID"].map(seq_to_split)

    # хемокласс
    def seq_to_class(seq_id: str) -> int:
        residues = residues_by_seq.get(seq_id, [])
        return get_catalytic_class(residues)

    df["ChemClass"] = df["Sequence_ID"].apply(seq_to_class)
    df["ChemClassLabel"] = df["ChemClass"].map(CLASS_LABELS)

    table_counts = pd.crosstab(df["ChemClassLabel"], df["Split"])

    # отсортируем сплиты
    if not table_counts.empty:
        split_cols = sorted(
            table_counts.columns,
            key=lambda x: int(x.replace("split", "")),
        )
        table_counts = table_counts[split_cols]

    print("[INFO] chemclass×split counts:")
    print(table_counts)

    return table_counts

def plot_heatmap_col_normalized(table_counts: pd.DataFrame, output_png: str):
    """
    Heatmap: fractions of chemotypes within each split (columns normalized to 1).
    """
    if table_counts.empty:
        print("[WARN] Empty table, skip heatmap.")
        return

    # normalize columns: each split sums to 1
    table_frac = table_counts.div(table_counts.sum(axis=0), axis=1)
    print("[DEBUG] col-normalized fractions:")
    print(table_frac)
    print("[DEBUG] min =", table_frac.to_numpy().min(),
          "max =", table_frac.to_numpy().max())

    import numpy as np
    import matplotlib.pyplot as plt

    data = table_frac.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xticklabels(table_frac.columns, rotation=45, ha="right")
    ax.set_yticklabels(table_frac.index)

    ax.set_xlabel("Cross-validation split")
    ax.set_ylabel("Catalytic chemotype")
    ax.set_title("Composition of chemotypes within each split")

    # annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of chemotype within split")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_row_normalized(table_counts: pd.DataFrame, output_png: str):
    """
    Хитмап по долям внутри каждого хемокласса (строки нормированы).
    """
    if table_counts.empty:
        print("[WARN] Empty table, skip heatmap.")
        return

    table_frac = table_counts.div(table_counts.sum(axis=1), axis=0)
    print("[DEBUG] row-normalized fractions:")
    print(table_frac)
    print("[DEBUG] min =", table_frac.to_numpy().min(),
          "max =", table_frac.to_numpy().max())

    import numpy as np
    import matplotlib.pyplot as plt

    data = table_frac.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    # жёстко задаём cmap и диапазон значений
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xticklabels(table_frac.columns, rotation=45, ha="right")
    ax.set_yticklabels(table_frac.index)

    ax.set_xlabel("Cross-validation split")
    ax.set_ylabel("Catalytic chemotype")
    ax.set_title("Relative distribution of catalytic chemotypes across splits")

    # подписи в ячейках
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction within chemotype")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    seq_to_split, residues_by_seq = parse_splits(SPLIT_PATTERN, n_splits=5)
    table_counts = build_chemclass_split_table(
        DATASET_CSV,
        seq_to_split,
        residues_by_seq,
    )

    table_counts.to_csv(OUTPUT_COUNTS_CSV)
    print(f"[OK] Saved chemclass×split counts to {OUTPUT_COUNTS_CSV}")

    plot_heatmap_row_normalized(table_counts, OUTPUT_HEATMAP_PNG)
    plot_heatmap_col_normalized(table_counts, OUTPUT_HEATMAP2_PNG)
    print(f"[OK] Saved heatmap to {OUTPUT_HEATMAP_PNG}")
