#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# ---------------------------
# Plotting & saving (твой стиль)
# ---------------------------

def make_PR_curve(labels, predictions, weights, subset_name,
                  title="", figsize=(10, 10), margin=0.05, grid=0.1, fs=16):
    """Plot PR curve with per-chain weights repeated to residues."""
    weights_repeated = np.array(
        [np.ones(len(lbl)) * w for lbl, w in zip(labels, weights)],
        dtype=object
    )
    labels_flat = np.concatenate(labels)
    preds_flat = np.concatenate(predictions)

    # If predictions are multi-class or contain last-channel targets
    if preds_flat.ndim > 1:
        preds_flat = preds_flat[..., -1]
        labels_flat = (labels_flat == labels_flat.max())

    is_bad = (
        np.isnan(preds_flat) | np.isnan(labels_flat) |
        np.isinf(preds_flat) | np.isinf(labels_flat)
    )
    if is_bad.any():
        preds_flat[is_bad] = np.nanmedian(preds_flat)

    precision, recall, _ = precision_recall_curve(
        labels_flat[~is_bad],
        preds_flat[~is_bad],
        sample_weight=np.concatenate(weights_repeated)[~is_bad]
    )
    aucpr = auc(recall, precision)
    print(f"{title} | {subset_name} | AUCPR={aucpr:.4f}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, linewidth=2.0,
            label=f"{subset_name} (AUCPR={aucpr:.3f})")
    plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.xlim([0 - margin, 1 + margin])
    plt.ylim([0 - margin, 1 + margin])
    plt.grid()
    plt.legend(fontsize=fs)
    plt.xlabel("Recall", fontsize=fs)
    plt.ylabel("Precision", fontsize=fs)
    plt.title(title, fontsize=fs)
    plt.tight_layout()
    return fig, ax


def save_subset_results(output_dir, subset_key, title,
                        labels, predictions, weights, ids, splits,
                        model_name, batch_size):
    """Save pickle and PR-curve PNG for a subset (train/validation/test)."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(labels)
    assert n == len(predictions) == len(weights) == len(ids) == len(splits), \
        f"Length mismatch: labels={len(labels)}, preds={len(predictions)}, weights={len(weights)}, ids={len(ids)}, splits={len(splits)}"

    payload = {
        "subset": subset_key,
        "model_name": model_name,
        "labels": labels,
        "predictions": predictions,
        "weights": weights,
        "ids": ids,
        "splits": splits,
        "batch_size": batch_size,
    }
    pkl_path = os.path.join(output_dir, f"{subset_key}_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] {subset_key} pickle -> {pkl_path}")

    fig, _ = make_PR_curve(
        labels=labels,
        predictions=predictions,
        weights=weights,
        subset_name=subset_key,
        title=title,
        figsize=(10, 10),
        margin=0.05,
        grid=0.1,
        fs=16
    )
    png_path = os.path.join(output_dir, f"{subset_key}_plot.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] {subset_key} PR-curve -> {png_path}")


# ---------------------------
# BLAST → per-residue tensors
# ---------------------------

def _parse_split_file(path: str) -> pd.DataFrame:
    """Read splitX.txt → DataFrame(id, seq, true_set (0-based))"""
    entries = []
    cur_id = None
    rows: List[Tuple[int, str, int]] = []  # (pos1, aa, label)

    def flush():
        nonlocal cur_id, rows
        if cur_id is None:
            return
        rows.sort(key=lambda x: x[0])
        seq = "".join(aa for (pos, aa, lab) in rows)
        true_set = {pos - 1 for (pos, aa, lab) in rows if int(lab) == 1}
        entries.append({"id": cur_id, "seq": seq, "true_set": true_set})
        cur_id, rows = None, []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                cur_id = s[1:].strip()
                rows = []
            else:
                parts = s.split()
                if len(parts) < 4:
                    continue
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                aa = parts[2]
                try:
                    lab = int(parts[3])
                except ValueError:
                    lab = 0
                rows.append((pos, aa, lab))
    flush()
    return pd.DataFrame(entries)


def _parse_pipe_set(s: str) -> Set[int]:
    if pd.isna(s) or s == "":
        return set()
    return set(int(x) for x in str(s).split("|") if str(x).strip() != "")


def build_labels_preds_for_split(split_txt: str, blast_csv: str):
    """
    From splitX.txt and blast_splitX.csv build:
      labels: list[np.ndarray] of shape [L_i] with 0/1 ints
      preds : list[np.ndarray] of shape [L_i] with float scores in [0,1]
      weights: list[float] (here all ones)
      ids: list[str]
      splits: list[str] (all 'splitX')
    """
    split_name = Path(split_txt).stem  # "split3"
    qdf = _parse_split_file(split_txt)   # id, seq, true_set
    bdf = pd.read_csv(blast_csv) if os.path.exists(blast_csv) else pd.DataFrame(columns=["From","BLAST_residues"])

    # Map predictions by query id
    pred_map = {r["From"]: _parse_pipe_set(r["BLAST_residues"]) for _, r in bdf.iterrows()}

    labels, preds, weights, ids, splits = [], [], [], [], []
    for _, row in qdf.iterrows():
        qid = row["id"]
        seq = row["seq"]
        true_set: Set[int] = row["true_set"]
        L = len(seq)

        # hard baseline scores: 1.0 на предсказанных позициях, 0.0 — иначе
        pred_set = pred_map.get(qid, set())
        y_true = np.zeros(L, dtype=np.int8)
        if true_set:
            idx = [i for i in true_set if 0 <= i < L]
            y_true[idx] = 1

        y_pred = np.zeros(L, dtype=np.float32)
        if pred_set:
            idxp = [i for i in pred_set if 0 <= i < L]
            y_pred[idxp] = 1.0

        labels.append(y_true)
        preds.append(y_pred)
        weights.append(1.0)     # <- при желании заменим на веса из dataset.csv
        ids.append(qid)
        splits.append(split_name)

    return labels, preds, weights, ids, splits


# ---------------------------
# Orchestrator
# ---------------------------

def run_eval(splits_glob: str, blast_dir: str, out_dir: str,
             model_name="BLAST-baseline", title_prefix="BLAST PR"):
    """
    Для каждого splitX.txt ищет соответствующий blast_splitX.csv в blast_dir,
    строит labels/preds и рисует PR-кривую.
    Также считает «overall» (всё вместе).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    split_txts = sorted(glob.glob(splits_glob))
    if not split_txts:
        raise SystemError(f"No files match: {splits_glob}")

    all_labels, all_preds, all_weights, all_ids, all_splits = [], [], [], [], []

    for split_txt in split_txts:
        split_name = Path(split_txt).stem
        blast_csv = os.path.join(blast_dir, f"blast_{split_name}.csv")

        labels, preds, weights, ids, splits = build_labels_preds_for_split(split_txt, blast_csv)

        # Сохраняем per-split
        save_subset_results(
            output_dir=out_dir,
            subset_key=split_name,
            title=f"{title_prefix} — {split_name}",
            labels=labels,
            predictions=preds,
            weights=weights,
            ids=ids,
            splits=splits,
            model_name=model_name,
            batch_size=None
        )

        # Копим для overall
        all_labels.extend(labels)
        all_preds .extend(preds)
        all_weights.extend(weights)
        all_ids.extend(ids)
        all_splits.extend(splits)

    # Overall
    save_subset_results(
        output_dir=out_dir,
        subset_key="overall",
        title=f"{title_prefix} — overall",
        labels=all_labels,
        predictions=all_preds,
        weights=all_weights,
        ids=all_ids,
        splits=all_splits,
        model_name=model_name,
        batch_size=None
    )


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_glob", required=True, help='e.g. "/path/split*.txt"')
    ap.add_argument("--blast_dir", required=True, help="dir with blast_splitX.csv files")
    ap.add_argument("--out_dir", required=True, help="where to save pkl/png")
    ap.add_argument("--model_name", default="BLAST-baseline")
    ap.add_argument("--title_prefix", default="BLAST PR")
    args = ap.parse_args()

    run_eval(
        splits_glob=args.splits_glob,
        blast_dir=args.blast_dir,
        out_dir=args.out_dir,
        model_name=args.model_name,
        title_prefix=args.title_prefix
    )
