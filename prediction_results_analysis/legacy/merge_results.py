#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge CV result pickles and compute metrics/plots.

Input per-file schema (as saved by your training script):
{
  "subset": str,
  "model_name": str,
  "labels": list[list[int]],
  "predictions": list[list[float]],
  "weights": list[float],
  "ids": list[[pdb_id, chain_id]],
  "splits": list[str],
  #"batch_size": int
}
"""

import os
import json
import argparse
import pickle
from typing import Iterable, Tuple, Dict, Any, List

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# ---------- I/O ----------

REQUIRED_KEYS = ["labels", "predictions", "weights", "ids", "splits"]

def load_subset_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    for k in REQUIRED_KEYS:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    # quick consistency check
    n = len(data["labels"])
    assert n == len(data["predictions"]) == len(data["weights"]) == len(data["ids"]) == len(data["splits"]), \
        f"{path}: length mismatch"
    return data

def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------- Merge helpers ----------

def _append_lists(dst, labels, preds, weights, ids, splits):
    dst["labels"].extend(labels)
    dst["predictions"].extend(preds)
    dst["weights"].extend(weights)
    dst["ids"].extend(ids)
    dst["splits"].extend(splits)

def _ensure_np_1d(x):
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError("labels/preds must be 1D per chain.")
    return a

def merge_across_folds(
    base_dir: str,
    run_dir_template: str,  # e.g. 'results_adaptive_cutoff_cv{fold}'
    subset: str,            # 'train' | 'validation' | 'test' | 'train_noskip' | 'train_upperbound'
    folds: Iterable[int] = (1, 2, 3, 4, 5),
    dedupe_train: str = "mean"   # 'mean' | 'median' | 'first' | None
) -> Dict[str, Any]:
    """
    Concatenate results from base_dir/<run_dir_template.format(fold=...)>/ for the given subset.
    If subset == 'train' and dedupe_train is not None, deduplicate same ids across folds.
    Returns dict with labels, predictions, weights, ids, splits + meta.
    """
    merged = dict(labels=[], predictions=[], weights=[], ids=[], splits=[])
    raw_parts: List[Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Any], List[str]]] = []
    file_meta = []
    per_sample_cv = []
    per_sample_src = []
    per_sample_model = []

    for fold in folds:
        run_dir = os.path.join(base_dir, run_dir_template.format(fold=fold))
        pkl_path = os.path.join(run_dir, f"{subset}_results.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Not found: {pkl_path}")

        data = load_subset_pkl(pkl_path)
        labels = [np.asarray(x) for x in data["labels"]]
        preds  = [np.asarray(x) for x in data["predictions"]]
        weights = list(map(float, data["weights"]))
        ids    = list(data["ids"])
        splits = list(data["splits"])
        model_name = data.get("model_name", "unknown")
        #batch_size = int(data.get("batch_size", -1))

        raw_parts.append((labels, preds, weights, ids, splits))
        file_meta.append({"cv_fold": fold, "path": pkl_path, "n": len(labels),
                          #"batch_size": batch_size, 
                          "model_name": model_name})

        # if we end up doing plain concat, we’ll need these:
        per_sample_cv.extend([fold] * len(labels))
        per_sample_src.extend([pkl_path] * len(labels))
        per_sample_model.extend([model_name] * len(labels))

        print(f"[OK] {subset}: fold {fold} -> +{len(labels)} chains")

    # Non-train or dedupe disabled: plain concatenation
    if subset != "train" or dedupe_train is None:
        for labels, preds, w, ids, splits in raw_parts:
            _append_lists(merged, labels, preds, w, ids, splits)
        merged["cv_folds"] = per_sample_cv
        merged["source_files"] = per_sample_src
        merged["model_names"] = per_sample_model
        merged["file_meta"] = file_meta
        return merged

    # ---- TRAIN: deduplicate by protein id across folds ----
    bucket: Dict[Any, List[Tuple[np.ndarray, np.ndarray, float, str, int, str, str]]] = {}
    # entries: (label, pred, weight, split, cv_fold, source_path, model_name)
    for idx_fold, (labels, preds, w, ids, splits) in enumerate(raw_parts):
        for i, _id in enumerate(ids):
            bucket.setdefault(_id, []).append(
                (labels[i], preds[i], float(w[i]), splits[i],
                 list(folds)[idx_fold],  # cv fold for this entry
                 file_meta[idx_fold]["path"], file_meta[idx_fold]["model_name"])
            )

    def combine_entries(entries, how="mean"):
        lbl0 = entries[0][0]
        L = len(lbl0)
        for (lbl, pr, *_ ) in entries:
            if len(lbl) != L or len(pr) != L:
                raise ValueError("Sequence length mismatch while merging duplicates.")
        label = lbl0 if all(np.array_equal(lbl0, e[0]) for e in entries) else entries[0][0]
        P = np.stack([e[1] for e in entries], axis=0)  # (n, L)
        if how == "mean":
            pred = P.mean(axis=0)
        elif how == "median":
            pred = np.median(P, axis=0)
        elif how == "first":
            pred = entries[0][1]
        else:
            raise ValueError(f"Unknown combine strategy: {how}")
        weight = float(np.mean([e[2] for e in entries]))
        split_str = "|".join([e[3] for e in entries])
        cv_list = [e[4] for e in entries]
        src_list = [e[5] for e in entries]
        mdl_list = [e[6] for e in entries]
        return label, pred, weight, split_str, cv_list, src_list, mdl_list

    cv_folds_dedup, srcs_dedup, models_dedup = [], [], []
    for _id, entries in bucket.items():
        lbl, pr, w, split_str, cv_list, src_list, mdl_list = combine_entries(entries, how=dedupe_train)
        merged["labels"].append(lbl)
        merged["predictions"].append(pr)
        merged["weights"].append(w)
        merged["ids"].append(_id)
        merged["splits"].append(f"train_agg:{split_str}")
        cv_folds_dedup.append(cv_list)
        srcs_dedup.append(src_list)
        models_dedup.append(mdl_list)

    # For transparency we keep lists of contributing folds/sources per deduped sample
    merged["cv_folds"] = cv_folds_dedup
    merged["source_files"] = srcs_dedup
    merged["model_names"] = models_dedup
    merged["file_meta"] = file_meta
    return merged


# ---------- Metrics ----------

def f_at_k(labels, preds, weights, k: int) -> float:
    """Weighted per-chain F@k."""
    num = 0.0
    den = 0.0
    for y, p, w in zip(labels, preds, weights):
        y = _ensure_np_1d(y)
        p = _ensure_np_1d(p)
        if len(y) != len(p):
            raise ValueError("labels and preds length mismatch.")
        k_eff = min(k, len(p))
        if k_eff <= 0:
            continue
        top_idx = np.argpartition(p, -k_eff)[-k_eff:]
        denom = min(int(np.sum(y)), k_eff)
        f_i = 0.0 if denom == 0 else float(np.sum(y[top_idx])) / denom
        num += f_i * float(w)
        den += float(w)
    return 0.0 if den == 0.0 else num / den

def fk_curve(labels, preds, weights, max_k=20) -> np.ndarray:
    return np.array([f_at_k(labels, preds, weights, k) for k in range(1, max_k + 1)])

def aucpr_weighted(labels, preds, weights) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Weighted PR-AUC: flatten to 1D and repeat per-chain weight across positions."""
    weights_repeated = [np.full(len(y), float(w), dtype=np.float32)
                        for y, w in zip(labels, weights)]
    y_flat = np.concatenate(labels).astype(np.float32)
    p_flat = np.concatenate(preds).astype(np.float32)
    w_flat = np.concatenate(weights_repeated).astype(np.float32)

    bad = np.isnan(p_flat) | np.isnan(y_flat) | np.isinf(p_flat)
    if bad.any():
        p_flat[bad] = np.nanmedian(p_flat[~bad]) if (~bad).any() else 0.0
        y_flat[bad] = 0.0
        w_flat[bad] = 0.0

    pr, rc, _ = precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)
    return auc(rc, pr), (rc, pr)


# ---------- Plotting ----------

def save_pr_plot(rc, pr, aucpr, title, out_path):
    plt.figure(figsize=(8, 8))
    plt.plot(rc, pr, linewidth=2.0, label=f"AUCPR={aucpr:.3f}")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="pdf")
    plt.close()

def save_fk_plot(Fk, out_path):
    x = np.arange(1, len(Fk) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, Fk, marker="o")
    plt.xlabel("k")
    plt.ylabel("MaxPrecision@k")
    plt.title("MaxPrecision@k curve")
    plt.xticks(x)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="pdf")
    plt.close()


# ---------- High-level ----------

def evaluate_merged(merged_dict, subset_name, out_dir=None, max_k=20, save_plots=False):
    labels = [np.asarray(x) for x in merged_dict["labels"]]
    preds  = [np.asarray(x) for x in merged_dict["predictions"]]
    weights = np.asarray(merged_dict["weights"], dtype=float)

    Fk = fk_curve(labels, preds, weights, max_k=max_k)
    aucpr, (rc, pr) = aucpr_weighted(labels, preds, weights)

    print(f"{subset_name}:")
    print(f"  F@1   = {Fk[0]:.3f}")
    if max_k >= 5:
        print(f"  F@5   = {Fk[4]:.3f}")
    if max_k >= 10:
        print(f"  F@10  = {Fk[9]:.3f}")
    print(f"  AUCPR = {aucpr:.3f}\n")

    if save_plots and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_pr_plot(rc, pr, aucpr, f"{subset_name}", os.path.join(out_dir, f"{subset_name}_pr.pdf"))
        save_fk_plot(Fk, os.path.join(out_dir, f"{subset_name}_fk.pdf"))

    return {"Fk": Fk.tolist(), "AUCPR": float(aucpr)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str,
                    default="/home/iscb/wolfson/annab4/DB/all_proteins/sequence_homology_baseline",
                    help="Directory containing per-fold subdirs")
    ap.add_argument("--save_dir", type=str,
                    default="/home/iscb/wolfson/annab4/catalytic-sites-annotation/cross_validation/sequence_homology_baseline_merged",
                    help="Where to write merged pickles and plots")
    ap.add_argument("--run_tpl", type=str,
                    default="results_adaptive_cutoff_cv{fold}",
                    help="Per-fold subdir name pattern under base_dir")
    ap.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--subsets", type=str, nargs="+",
                    default=["validation", "test", "train"],
                    help="Which subsets to merge (files: <subset>.pkl)")
    ap.add_argument("--dedupe_train", type=str, default="mean",
                    choices=["mean", "median", "first", "None"],
                    help="How to combine duplicate train ids across folds")
    ap.add_argument("--max_k", type=int, default=20)
    ap.add_argument("--save_plots", action="store_true")
    args = ap.parse_args()

    dedupe = None if args.dedupe_train == "None" else args.dedupe_train

    os.makedirs(args.save_dir, exist_ok=True)
    all_metrics = {}

    for subset in args.subsets:
        merged = merge_across_folds(args.base_dir, args.run_tpl, subset,
                                    folds=args.folds, dedupe_train=dedupe if subset=="train" else None)
        out_pkl = os.path.join(args.save_dir, f"{subset}.pkl" if subset!="train" else "train_dedup.pkl")
        # convert np arrays to lists for portability
        merged_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in merged.items()}
        save_pickle(merged_serializable, out_pkl)
        print(f"[OK] saved merged {subset} -> {out_pkl}")

        metrics = evaluate_merged(merged, subset_name=f"{subset}_merged",
                                  out_dir=args.save_dir, max_k=args.max_k, save_plots=args.save_plots)
        all_metrics[subset] = metrics

    save_json(all_metrics, os.path.join(args.save_dir, "metrics_summary.json"))
    print(f"[OK] metrics JSON -> {os.path.join(args.save_dir, 'metrics_summary.json')}")

if __name__ == "__main__":
    main()
