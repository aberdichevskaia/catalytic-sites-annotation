#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_results.py

One-script evaluation toolbox for ScanNet catalytic-site prediction.

Subcommands:
  1) merge  : merge k-fold CV pickles (per subset) + compute metrics + plots + per-chain table
  2) seeds  : aggregate multiple versions/seeds (whiskers for MP@k, PR overlays + mean) + optional SNR vs baseline
  3) cases  : mine best/worst predictions by FP/FN (threshold or topk), with pretty names from dataset.csv

Expected per-fold pickle schema (saved by training):
{
  "subset": str,
  "model_name": str,
  "labels": list[list[int]] or list[np.ndarray],
  "predictions": list[list[float]] or list[np.ndarray],
  "weights": list[float],
  "ids": list[str] or list[[pdb_id, chain_id]] or list[tuple],
  "splits": list[str],
}

Notes:
- "MaxPrecision@k" == TP@k / min(#pos, k), computed per-chain, then averaged with weights.
- AUCPR is computed by flattening residues and repeating chain weight across residues.
"""

import os
import json
import csv
import argparse
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, average_precision_score


# =========================
# IO + sanity
# =========================

REQUIRED_KEYS = ["labels", "predictions", "weights", "ids", "splits"]


def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_subset_pkl(path: str) -> Dict[str, Any]:
    data = load_pickle(path)
    for k in REQUIRED_KEYS:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")

    n = len(data["labels"])
    if not (n == len(data["predictions"]) == len(data["weights"]) == len(data["ids"]) == len(data["splits"])):
        raise ValueError(
            f"{path}: length mismatch "
            f"labels={len(data['labels'])}, preds={len(data['predictions'])}, "
            f"weights={len(data['weights'])}, ids={len(data['ids'])}, splits={len(data['splits'])}"
        )
    return data


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_id(x: Any) -> str:
    """
    Normalize ids to a stable string key.

    Common cases:
      - "A0A1..._A" -> keep
      - ["1ABC", "A"] -> "1ABC_A"
      - ("A0A1.._A", "A") -> handle best effort
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, str) for t in x):
            a, b = x[0], x[1]
            if a.endswith(f"_{b}"):
                return a
            return f"{a}_{b}"
        if len(x) >= 1:
            return str(x[0])
    return str(x)


def ensure_1d(arr: Any, name: str = "vector") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D per chain, got shape {a.shape}")
    return a


# =========================
# Metrics
# =========================


def _sanitize_flatten(labels, preds, weights):
    """
    Flatten residues, repeating per-chain weights per residue (consistent with AUCPR implementation).
    Returns y_flat (0/1), p_flat (float), w_flat (float).
    """
    y_list, p_list, w_list = [], [], []
    for y, p, w in zip(labels, preds, weights):
        y = np.asarray(y).astype(np.float32)
        p = np.asarray(p).astype(np.float32)
        L = min(len(y), len(p))
        if L <= 0:
            continue
        y = y[:L]
        p = p[:L]
        w_rep = np.full(L, float(w), dtype=np.float32)

        y_list.append(y)
        p_list.append(p)
        w_list.append(w_rep)

    if len(y_list) == 0:
        return np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    y_flat = np.concatenate(y_list)
    p_flat = np.concatenate(p_list)
    w_flat = np.concatenate(w_list)

    bad = ~np.isfinite(p_flat) | ~np.isfinite(y_flat)
    if bad.any():
        fill = np.nanmedian(p_flat[~bad]) if (~bad).any() else 0.0
        p_flat[bad] = fill
        y_flat[bad] = 0.0
        w_flat[bad] = 0.0

    # labels should be 0/1
    y_flat = (y_flat > 0.5).astype(np.int32)
    return y_flat, p_flat, w_flat

def prf_at_tau_weighted(labels, preds, weights, tau: float):
    """
    Precision/Recall/F1 at threshold tau (residue-level), weighted like AUCPR (repeat chain weights per residue).
    """
    y, p, w = _sanitize_flatten(labels, preds, weights)
    if y.size == 0:
        return {"tau": float(tau), "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0.0, "fp": 0.0, "fn": 0.0}

    yhat = (p >= float(tau)).astype(np.int32)

    tp = float(w[(y == 1) & (yhat == 1)].sum())
    fp = float(w[(y == 0) & (yhat == 1)].sum())
    fn = float(w[(y == 1) & (yhat == 0)].sum())

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    return {"tau": float(tau), "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}

def best_f1_threshold_weighted(labels, preds, weights):
    """
    Find tau that maximizes F1 on the PR curve (weighted residue-level).
    Returns dict with tau, precision, recall, f1.
    """
    y, p, w = _sanitize_flatten(labels, preds, weights)
    if y.size == 0 or y.sum() == 0:
        return {"tau": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    # thr has length len(prec)-1; align by dropping the first point
    if thr.size == 0:
        return {"tau": float("nan"), "precision": float(prec[-1]), "recall": float(rec[-1]), "f1": 0.0}

    prec2 = prec[1:]
    rec2  = rec[1:]
    f1 = 2.0 * prec2 * rec2 / np.maximum(prec2 + rec2, 1e-12)
    j = int(np.argmax(f1))
    return {"tau": float(thr[j]), "precision": float(prec2[j]), "recall": float(rec2[j]), "f1": float(f1[j])}

def topk_metrics_curves(labels, preds, weights, max_k=10):
    """
    Per-protein metrics aggregated with chain weights:
      Precision@k, Recall@k, Hit@k for k=1..max_k
    Recall@k and Hit@k are computed over proteins with npos>0 (skip npos==0).
    Precision@k is computed over all proteins with L>0.
    """
    K = int(max_k)
    prec_k = np.zeros(K, dtype=float)
    rec_k  = np.zeros(K, dtype=float)
    hit_k  = np.zeros(K, dtype=float)

    # accumulators
    p_num = np.zeros(K, dtype=float); p_den = np.zeros(K, dtype=float)
    r_num = np.zeros(K, dtype=float); r_den = np.zeros(K, dtype=float)
    h_num = np.zeros(K, dtype=float); h_den = np.zeros(K, dtype=float)

    for y, p, w in zip(labels, preds, weights):
        y = np.asarray(y).astype(int)
        p = np.asarray(p).astype(float)
        L = min(len(y), len(p))
        if L <= 0:
            continue
        y = y[:L]
        p = p[:L]
        w = float(w)

        npos = int(y.sum())
        # precompute ranking once
        # argsort ok for L<=800; if huge, replace with argpartition per k
        order = np.argsort(p)

        for k in range(1, K+1):
            k_eff = min(k, L)
            top_idx = order[-k_eff:]
            tp = int(y[top_idx].sum())

            # Precision@k: tp/k
            p_num[k-1] += w * (tp / k_eff)
            p_den[k-1] += w

            if npos > 0:
                # Recall@k: tp/npos
                r_num[k-1] += w * (tp / npos)
                r_den[k-1] += w

                # Hit@k: 1 if any TP in top-k
                h_num[k-1] += w * (1.0 if tp > 0 else 0.0)
                h_den[k-1] += w

    prec_k = np.divide(p_num, np.maximum(p_den, 1e-12))
    rec_k  = np.divide(r_num, np.maximum(r_den, 1e-12))
    hit_k  = np.divide(h_num, np.maximum(h_den, 1e-12))
    return prec_k, rec_k, hit_k


def mp_per_chain(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """
    MaxPrecision@k for one chain: TP@k / min(#pos, k).
    """
    y = ensure_1d(y, "labels").astype(int)
    p = ensure_1d(p, "preds").astype(float)

    L = min(len(y), len(p))
    if L <= 0:
        return 0.0
    y = y[:L]
    p = p[:L]

    kk = min(int(k), L)
    if kk <= 0:
        return 0.0

    top_idx = np.argpartition(p, -kk)[-kk:]
    denom = min(int(y.sum()), kk)
    return 0.0 if denom == 0 else float(y[top_idx].sum()) / denom


def maxprecision_at_k(labels: List[np.ndarray],
                      preds: List[np.ndarray],
                      weights: Union[List[float], np.ndarray],
                      k: int) -> float:
    """
    Weighted mean of per-chain MaxPrecision@k.
    """
    num = 0.0
    den = 0.0
    for y, p, w in zip(labels, preds, weights):
        y = ensure_1d(y, "labels")
        p = ensure_1d(p, "preds")
        L = min(len(y), len(p))
        if L <= 0:
            continue
        y = y[:L]
        p = p[:L]
        kk = min(int(k), L)
        if kk <= 0:
            continue
        top_idx = np.argpartition(p, -kk)[-kk:]
        denom = min(int(y.sum()), kk)
        mp = 0.0 if denom == 0 else float(y[top_idx].sum()) / denom
        num += mp * float(w)
        den += float(w)
    return 0.0 if den == 0.0 else float(num / den)


def mp_curve(labels: List[np.ndarray],
             preds: List[np.ndarray],
             weights: Union[List[float], np.ndarray],
             max_k: int) -> np.ndarray:
    return np.array([maxprecision_at_k(labels, preds, weights, k) for k in range(1, max_k + 1)], dtype=float)


def pr_curve_weighted(labels: List[np.ndarray],
                      preds: List[np.ndarray],
                      weights: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted PR curve: flatten residues and repeat chain weight across residues.
    Returns: (recall, precision, aucpr).
    """
    y_list = []
    p_list = []
    w_list = []

    for y, p, w in zip(labels, preds, weights):
        y = ensure_1d(y, "labels").astype(np.float32)
        p = ensure_1d(p, "preds").astype(np.float32)
        L = min(len(y), len(p))
        if L <= 0:
            continue
        y = y[:L]
        p = p[:L]

        bad = np.isnan(p) | np.isinf(p) | np.isnan(y) | np.isinf(y)
        if bad.any():
            fill = np.nanmedian(p[~bad]) if (~bad).any() else 0.0
            p = p.copy()
            p[bad] = fill
            y = y.copy()
            y[bad] = 0.0
            w_eff = 0.0
        else:
            w_eff = float(w)

        y_list.append(y)
        p_list.append(p)
        w_list.append(np.full(L, w_eff, dtype=np.float32))

    if len(y_list) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    y_flat = np.concatenate(y_list)
    p_flat = np.concatenate(p_list)
    w_flat = np.concatenate(w_list)

    # If a group has no positives, AUCPR is 0 by convention here
    if float(y_flat.sum()) == 0.0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0

    precision, recall, _ = precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)
    aucpr = float(auc(recall, precision))
    return recall, precision, aucpr


def auprc_per_chain(y: np.ndarray, p: np.ndarray) -> float:
    """
    Unweighted per-chain AUCPR (average_precision_score).
    Used in per-chain table / cases ranking.
    """
    y = ensure_1d(y, "labels").astype(int)
    p = ensure_1d(p, "preds").astype(float)
    L = min(len(y), len(p))
    if L <= 0:
        return 0.0
    y = y[:L]
    p = p[:L]
    valid = np.isfinite(p) & np.isfinite(y)
    y = y[valid]
    p = p[valid]
    if y.size == 0 or int(y.sum()) == 0:
        return 0.0
    return float(average_precision_score(y, p))


# =========================
# Plot styling (fixed ticks)
# =========================

def apply_fixed_axes_pr(ax: plt.Axes, grid: float = 0.1, margin: float = 0.05) -> None:
    ticks = np.arange(0.0, 1.0 + 1e-9, grid)
    ax.set_xlim(-margin, 1.0 + margin)
    ax.set_ylim(-margin, 1.0 + margin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, alpha=0.35)


def apply_fixed_axes_metric(ax: plt.Axes, max_k: int, grid_y: float = 0.1) -> None:
    ax.set_xlim(1, max_k)
    ax.set_xticks(np.arange(1, max_k + 1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, grid_y))
    ax.grid(True, alpha=0.35)


def save_prec_rec_at_k_plot(prec_k, rec_k, out_path, title, grid_y=0.1):
    K = len(prec_k)
    x = np.arange(1, K+1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, prec_k, marker="o", label="Precision@k")
    plt.plot(x, rec_k,  marker="o", label="Recall@k")

    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(title)

    plt.xlim(1, K)
    plt.ylim(-0.05, 1.05)

    # fixed ticks
    yt = np.arange(0.0, 1.0 + 1e-9, float(grid_y))
    plt.yticks(yt)
    plt.xticks(x)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="svg")
    plt.close()

def save_hit_at_k_plot(hit_k, out_path, title, grid_y=0.1):
    K = len(hit_k)
    x = np.arange(1, K+1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, hit_k, marker="o", label="Hit@k")

    plt.xlabel("k")
    plt.ylabel("Hit rate")
    plt.title(title)

    plt.xlim(1, K)
    plt.ylim(-0.05, 1.05)

    yt = np.arange(0.0, 1.0 + 1e-9, float(grid_y))
    plt.yticks(yt)
    plt.xticks(x)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format="svg")
    plt.close()


def save_pr_plot(recall: np.ndarray,
                 precision: np.ndarray,
                 aucpr: float,
                 title: str,
                 out_path: str,
                 grid: float = 0.1,
                 margin: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(recall, precision, linewidth=2.0, label=f"AUCPR={aucpr:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    apply_fixed_axes_pr(ax, grid=grid, margin=margin)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mp_plot(mp_k: np.ndarray,
                 title: str,
                 out_path: str,
                 grid_y: float = 0.1) -> None:
    max_k = int(len(mp_k))
    x = np.arange(1, max_k + 1)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, mp_k, marker="o", linewidth=2.0)
    ax.set_xlabel("k")
    ax.set_ylabel("MaxPrecision@k")
    ax.set_title(title)
    apply_fixed_axes_metric(ax, max_k=max_k, grid_y=grid_y)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mp_whisker(mp_arr: np.ndarray,
                    out_path: str,
                    title: str,
                    grid_y: float = 0.1) -> None:
    """
    mp_arr: (V, K)
    """
    V, K = mp_arr.shape
    mu = mp_arr.mean(axis=0)
    sd = mp_arr.std(axis=0, ddof=1) if V > 1 else np.zeros(K, dtype=float)
    x = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(x, mu, yerr=sd, fmt="o-", capsize=3, linewidth=1.8, markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("MaxPrecision@k")
    ax.set_title(title)
    apply_fixed_axes_metric(ax, max_k=K, grid_y=grid_y)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pr_overlays(curves: List[Tuple[np.ndarray, np.ndarray, float]],
                     out_path: str,
                     title: str,
                     overlay_alpha: float = 0.25,
                     grid: float = 0.1,
                     margin: float = 0.05,
                     n_grid: int = 501) -> Dict[str, Any]:
    """
    Draw each PR curve with transparency + mean PR curve on a fixed recall grid.
    Returns summary dict.
    """
    recall_grid = np.linspace(0.0, 1.0, n_grid)
    P = []

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    aucs = []

    for rc, pr, aucpr in curves:
        rc = np.asarray(rc, float)
        pr = np.asarray(pr, float)

        # Ensure increasing recall for interpolation
        order = np.argsort(rc)
        rc = rc[order]
        pr = pr[order]

        # Interpolate precision on recall_grid
        pr_i = np.interp(recall_grid, rc, pr, left=pr[0], right=pr[-1])
        P.append(pr_i)
        aucs.append(float(aucpr))

        ax.plot(rc, pr, linewidth=1.2, alpha=overlay_alpha)

    P = np.stack(P, axis=0)  # (V, n_grid)
    pr_mean = P.mean(axis=0)
    auc_mean_curve = float(auc(recall_grid, pr_mean))

    ax.plot(recall_grid, pr_mean, linewidth=2.5,
            label=f"Mean curve AUCPR={auc_mean_curve:.3f} | mean(AUCPR)={np.mean(aucs):.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    apply_fixed_axes_pr(ax, grid=grid, margin=margin)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_versions": len(curves),
        "mean_aucpr_over_seeds": float(np.mean(aucs)),
        "std_aucpr_over_seeds": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "aucpr_of_mean_curve": auc_mean_curve,
        "n_grid": int(n_grid),
    }


# =========================
# Merge across folds
# =========================

def merge_across_folds(base_dir: str,
                       run_dir_template: str,
                       subset: str,
                       folds: Iterable[int],
                       dedupe_train: Optional[str] = None,
                       version: int | None = None) -> Dict[str, Any]:
    """
    Concatenate results from:
      base_dir / run_dir_template.format(fold=fold) / f"{subset}_results.pkl"

    If subset == "train" and dedupe_train in {"mean","median","first"}:
      deduplicate same ids across folds by combining predictions.
    """
    if ("{version}" in run_dir_template) and (version is None):
        raise ValueError("run_dir_template contains '{version}', but version=None was passed")

    parts = []
    file_meta = []

    per_sample_cv = []
    per_sample_src = []
    per_sample_model = []

    for fold in folds:
        if version is None:
            run_dir_name = run_dir_template.format(fold=fold)
        else:
            run_dir_name = run_dir_template.format(fold=fold, version=version)
        
        run_dir = os.path.join(base_dir, run_dir_name)
        pkl_path = os.path.join(run_dir, f"{subset}_results.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Not found: {pkl_path}")

        data = load_subset_pkl(pkl_path)

        labels = [ensure_1d(x, "labels") for x in data["labels"]]
        preds = [ensure_1d(x, "preds") for x in data["predictions"]]
        weights = list(map(float, data["weights"]))
        ids = [normalize_id(x) for x in data["ids"]]
        splits = list(map(str, data["splits"]))
        model_name = str(data.get("model_name", "unknown"))

        parts.append((labels, preds, weights, ids, splits))
        file_meta.append({"cv_fold": int(fold), "path": pkl_path, "n": len(ids), "model_name": model_name})

        per_sample_cv.extend([int(fold)] * len(ids))
        per_sample_src.extend([pkl_path] * len(ids))
        per_sample_model.extend([model_name] * len(ids))

        print(f"[OK] {subset}: fold {fold} -> +{len(ids)} chains")

    merged = {
        "labels": [],
        "predictions": [],
        "weights": [],
        "ids": [],
        "splits": [],
        "cv_folds": [],
        "source_files": [],
        "model_names": [],
        "file_meta": file_meta,
    }

    # Plain concat if not train or dedupe disabled
    if subset != "train" or dedupe_train is None:
        for labels, preds, w, ids, splits in parts:
            merged["labels"].extend(labels)
            merged["predictions"].extend(preds)
            merged["weights"].extend(w)
            merged["ids"].extend(ids)
            merged["splits"].extend(splits)
        merged["cv_folds"] = per_sample_cv
        merged["source_files"] = per_sample_src
        merged["model_names"] = per_sample_model
        return merged

    # Train dedupe
    if dedupe_train not in ("mean", "median", "first"):
        raise ValueError("dedupe_train must be one of: mean, median, first, or None")

    bucket: Dict[str, List[Tuple[np.ndarray, np.ndarray, float, str, int, str, str]]] = {}
    for (labels, preds, w, ids, splits), meta in zip(parts, file_meta):
        fold = int(meta["cv_fold"])
        src = str(meta["path"])
        mdl = str(meta["model_name"])
        for i, sid in enumerate(ids):
            bucket.setdefault(sid, []).append((labels[i], preds[i], float(w[i]), splits[i], fold, src, mdl))

    def combine(entries: List[Tuple[np.ndarray, np.ndarray, float, str, int, str, str]]) -> Tuple:
        lbl0 = entries[0][0]
        L = len(lbl0)
        for lbl, pr, *_ in entries:
            if len(lbl) != L or len(pr) != L:
                raise ValueError("Length mismatch while deduping train across folds.")

        P = np.stack([e[1] for e in entries], axis=0)
        if dedupe_train == "mean":
            pred = P.mean(axis=0)
        elif dedupe_train == "median":
            pred = np.median(P, axis=0)
        else:
            pred = entries[0][1]

        weight = float(np.mean([e[2] for e in entries]))
        split_str = "|".join([e[3] for e in entries])
        cv_list = [e[4] for e in entries]
        src_list = [e[5] for e in entries]
        mdl_list = [e[6] for e in entries]
        return lbl0, pred, weight, split_str, cv_list, src_list, mdl_list

    for sid, entries in bucket.items():
        lbl, pr, w, split_str, cv_list, src_list, mdl_list = combine(entries)
        merged["labels"].append(lbl)
        merged["predictions"].append(pr)
        merged["weights"].append(w)
        merged["ids"].append(sid)
        merged["splits"].append(f"train_agg:{split_str}")
        merged["cv_folds"].append(cv_list)
        merged["source_files"].append(src_list)
        merged["model_names"].append(mdl_list)

    return merged


# =========================
# Per-chain table (saved in merge)
# =========================

def build_per_chain_table(merged: Dict[str, Any],
                          subset: str,
                          max_k: int,
                          dataset_csv: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Create per-chain rows for debugging / mining.
    Includes: id, uniprot_id, split, weight, length, n_pos, mp@k (per chain), auprc_per_chain.
    Optionally enrich with dataset.csv fields by Sequence_ID.
    """
    labels = [ensure_1d(x, "labels") for x in merged["labels"]]
    preds = [ensure_1d(x, "preds") for x in merged["predictions"]]
    weights = np.asarray(merged["weights"], dtype=float)
    ids = [normalize_id(x) for x in merged["ids"]]
    splits = list(map(str, merged["splits"]))

    enrich = {}
    if dataset_csv is not None and os.path.exists(dataset_csv):
        try:
            import pandas as pd
            df = pd.read_csv(dataset_csv)
            if "Sequence_ID" in df.columns:
                df = df.drop_duplicates(subset=["Sequence_ID"])
                for _, r in df.iterrows():
                    sid = str(r["Sequence_ID"])
                    enrich[sid] = {
                        "full_name": str(r.get("full_name", "")),
                        "EC_number": str(r.get("EC_number", "")),
                        "Component_ID": str(r.get("Component_ID", "")),
                        "Set_Type": str(r.get("Set_Type", "")),
                        "Cluster_1": str(r.get("Cluster_1", "")),
                        "Cluster_2": str(r.get("Cluster_2", "")),
                    }
        except Exception:
            enrich = {}

    rows = []
    for i, (sid, y, p, w, sp) in enumerate(zip(ids, labels, preds, weights, splits)):
        L = min(len(y), len(p))
        y2 = y[:L].astype(int)
        p2 = p[:L].astype(float)

        npos = int(y2.sum())
        uniprot = sid.split("_")[0] if "_" in sid else sid

        row = {
            "subset": subset,
            "idx": int(i),
            "Sequence_ID": sid,
            "UniProt_ID": uniprot,
            "split": sp,
            "weight": float(w),
            "length": int(L),
            "n_pos": int(npos),
            "auprc_chain": auprc_per_chain(y2, p2),
        }
        # Per-chain MP@k values
        for k in range(1, max_k + 1):
            row[f"mp@{k}"] = mp_per_chain(y2, p2, k=k)

        # Optional enrichment
        if sid in enrich:
            row.update(enrich[sid])

        rows.append(row)

    return rows


# =========================
# SNR helpers
# =========================

def snr_signal_over_noise(signal: float, noise_std: float, eps: float = 1e-12) -> float:
    """
    SNR = signal / noise.
    In your wording: noise = std(model over seeds), signal = (mean(model) - mean(baseline)).
    """
    return float(signal) / max(float(noise_std), eps)


def save_snr_curve(snr_k: np.ndarray, out_path: str, title: str) -> None:
    K = int(len(snr_k))
    x = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(x, snr_k, marker="o", linewidth=2.0)
    ax.set_xlabel("k")
    ax.set_ylabel("SNR(k)")
    ax.set_title(title)
    ax.set_xlim(1, K)
    ax.set_xticks(np.arange(1, K + 1, 1))
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# cases mining
# =========================

def load_dataset_map(dataset_csv: Optional[str]) -> Dict[str, Dict[str, str]]:
    """
    Map Sequence_ID -> metadata strings (full_name, EC_number, etc.)
    """
    out: Dict[str, Dict[str, str]] = {}
    if dataset_csv is None:
        return out
    if not os.path.exists(dataset_csv):
        return out

    try:
        import pandas as pd
        df = pd.read_csv(dataset_csv)
        if "Sequence_ID" not in df.columns:
            return out
        df = df.drop_duplicates(subset=["Sequence_ID"])
        for _, r in df.iterrows():
            sid = str(r["Sequence_ID"])
            out[sid] = {
                "full_name": str(r.get("full_name", "")),
                "EC_number": str(r.get("EC_number", "")),
                "Component_ID": str(r.get("Component_ID", "")),
                "Set_Type": str(r.get("Set_Type", "")),
            }
    except Exception:
        return out
    return out


def cases_threshold(y: np.ndarray, p: np.ndarray, tau: float) -> Dict[str, Any]:
    """
    Count FP/FN at threshold tau over all residues.
    """
    y = ensure_1d(y, "labels").astype(int)
    p = ensure_1d(p, "preds").astype(float)
    L = min(len(y), len(p))
    y = y[:L]
    p = p[:L]

    valid = np.isfinite(p) & np.isfinite(y)
    y = y[valid]
    p = p[valid]

    pred = (p >= float(tau)).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    max_fp_score = float(p[(y == 0) & (pred == 1)].max()) if fp > 0 else 0.0
    max_fn_score = float(p[(y == 1) & (pred == 0)].max()) if fn > 0 else 0.0

    return {
        "L": int(len(y)),
        "npos": int(y.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "max_fp_score": max_fp_score,
        "max_fn_score": max_fn_score,
    }


def cases_topk(y: np.ndarray, p: np.ndarray, k: int, fp_th: float) -> Dict[str, Any]:
    """
    Count FP/FN in top-k by score.
    FP@k: negatives inside top-k
    TP@k: positives inside top-k
    FN@k: positives missed by top-k = max(0, #pos - TP@k)
    CFP@k: FP inside top-k with score >= fp_th
    """
    y = ensure_1d(y, "labels").astype(int)
    p = ensure_1d(p, "preds").astype(float)
    L = min(len(y), len(p))
    y = y[:L]
    p = p[:L]

    valid = np.isfinite(p) & np.isfinite(y)
    y = y[valid]
    p = p[valid]

    L2 = int(len(y))
    if L2 == 0:
        return {"L": 0, "npos": 0, "tp": 0, "fp": 0, "fn": 0, "cfp": 0, "max_fp_score": 0.0}

    kk = min(int(k), L2)
    top_idx = np.argpartition(p, -kk)[-kk:]
    top_scores = p[top_idx]
    top_labels = y[top_idx]

    tp = int((top_labels == 1).sum())
    fp_mask = (top_labels == 0)
    fp = int(fp_mask.sum())
    cfp = int((fp_mask & (top_scores >= float(fp_th))).sum())

    npos = int(y.sum())
    fn = int(max(0, npos - tp))

    max_fp_score = float(top_scores[fp_mask].max()) if fp > 0 else 0.0

    return {
        "L": L2,
        "npos": npos,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cfp": cfp,
        "max_fp_score": max_fp_score,
    }


def resolve_results_paths(path: str) -> List[str]:
    """
    If 'path' is a file -> [path]
    If dir -> try to find typical merged outputs and analyze each.
    """
    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not found: {path}")

    candidates = [
        "test.pkl", "validation.pkl", "train.pkl", "train_dedup.pkl",
        "test_results.pkl", "validation_results.pkl", "train_results.pkl",
    ]
    out = []
    for c in candidates:
        p = os.path.join(path, c)
        if os.path.exists(p):
            out.append(p)
    if not out:
        raise FileNotFoundError(f"No result pickles found in dir: {path}")
    return out


def run_cases_one(results_pkl: str,
                  dataset_csv: Optional[str],
                  topn: int,
                  mode: str,
                  tau: float,
                  k: int,
                  fp_th: float,
                  fp_weight: float,
                  fn_weight: float,
                  out_csv: Optional[str]) -> None:
    d = load_subset_pkl(results_pkl)
    ids = [normalize_id(x) for x in d["ids"]]
    labels = [ensure_1d(x, "labels") for x in d["labels"]]
    preds = [ensure_1d(x, "preds") for x in d["predictions"]]
    weights = np.asarray(d["weights"], dtype=float)
    splits = list(map(str, d["splits"]))

    n = min(len(ids), len(labels), len(preds), len(weights), len(splits))
    ids, labels, preds, weights, splits = ids[:n], labels[:n], preds[:n], weights[:n], splits[:n]

    meta_map = load_dataset_map(dataset_csv)

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        sid = ids[i]
        y = labels[i]
        p = preds[i]
        uniprot = sid.split("_")[0] if "_" in sid else sid

        if mode == "threshold":
            m = cases_threshold(y, p, tau=tau)
            key_fp = m["fp"]
        elif mode == "topk":
            m = cases_topk(y, p, k=k, fp_th=fp_th)
            key_fp = m["cfp"]  # confident FP is usually the most informative
        else:
            raise ValueError("mode must be threshold or topk")

        # per-chain auprc (unweighted)
        au_chain = auprc_per_chain(y, p)

        row = {
            "idx": int(i),
            "Sequence_ID": sid,
            "UniProt_ID": uniprot,
            "split": splits[i],
            "weight": float(weights[i]),
            "auprc_chain": float(au_chain),
            "mode": mode,
        }
        row.update(m)

        # error for overall ranking
        row["error"] = float(fp_weight * row.get("fp", 0) + fn_weight * row.get("fn", 0))
        row["key_fp"] = int(key_fp)

        # enrich from dataset.csv if possible
        if sid in meta_map:
            row.update(meta_map[sid])

        rows.append(row)

    # Rankings
    worst_fp = sorted(rows, key=lambda r: (r["key_fp"], r.get("max_fp_score", 0.0), r["error"]), reverse=True)[:topn]
    worst_fn = sorted(rows, key=lambda r: (r.get("fn", 0), r["error"]), reverse=True)[:topn]
    best_overall = sorted(rows, key=lambda r: (r["error"], -r["auprc_chain"]))[:topn]

    title = os.path.basename(results_pkl)
    print(f"\n=== CASES: {title} ===")
    print(f"mode={mode} | topn={topn}")
    if mode == "threshold":
        print(f"tau={tau} | error = {fp_weight}*FP + {fn_weight}*FN")
    else:
        print(f"k={k} | fp_th={fp_th} | error = {fp_weight}*FP + {fn_weight}*FN  (FP uses top-k counts)")

    def fmt_row(r: Dict[str, Any]) -> str:
        name = r.get("full_name", "")
        ec = r.get("EC_number", "")
        return (f"{r['Sequence_ID']:20s} | UniProt={r['UniProt_ID']:12s} | "
                f"FP={r.get('fp', 0):4d} | FN={r.get('fn', 0):4d} | "
                f"keyFP={r.get('key_fp', 0):4d} | err={r['error']:.1f} | "
                f"AUPRC(chain)={r['auprc_chain']:.4f} | EC={ec} | {name[:60]}")

    print(f"\nTop-{topn} WORST by FP (key_fp):")
    for r in worst_fp:
        print("  " + fmt_row(r))

    print(f"\nTop-{topn} WORST by FN:")
    for r in worst_fn:
        print("  " + fmt_row(r))

    print(f"\nTop-{topn} BEST overall (min error):")
    for r in best_overall:
        print("  " + fmt_row(r))

    # CSV dump (full rows)
    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        # stable field order
        fieldnames = sorted({k for rr in rows for k in rr.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[OK] saved CSV -> {out_csv}")


# =========================
# CLI commands
# =========================

def cmd_merge(args: argparse.Namespace) -> None:
    dedupe = None if args.dedupe_train == "None" else args.dedupe_train
    os.makedirs(args.save_dir, exist_ok=True)

    all_metrics = {}

    for subset in args.subsets:
        merged = merge_across_folds(
            base_dir=args.base_dir,
            run_dir_template=args.run_tpl,
            subset=subset,
            folds=args.folds,
            dedupe_train=(dedupe if subset == "train" else None),
        )

        # Save merged pickle (portable)
        out_pkl = os.path.join(args.save_dir, f"{subset}.pkl" if subset != "train" else "train_dedup.pkl")
        merged_serializable = dict(merged)
        save_pickle(merged_serializable, out_pkl)
        print(f"[OK] saved merged -> {out_pkl}")

        labels = [ensure_1d(x, "labels") for x in merged["labels"]]
        preds = [ensure_1d(x, "preds") for x in merged["predictions"]]
        weights = np.asarray(merged["weights"], dtype=float)

        mp_k = mp_curve(labels, preds, weights, max_k=args.max_k)
        rc, pr, aucpr = pr_curve_weighted(labels, preds, weights)

        best_f1 = best_f1_threshold_weighted(labels, preds, weights)
        prf_best = prf_at_tau_weighted(labels, preds, weights, tau=best_f1["tau"])

        # @k metrics for k=1..10 (weighted per-chain aggregation)
        prec_k10, rec_k10, hit_k10 = topk_metrics_curves(labels, preds, weights, max_k=10)

        # handy scalars for k in [1,3,5,8]
        def pick(arr, k):  # arr is 0-indexed, k is 1-indexed
            return float(arr[k-1]) if len(arr) >= k else float(arr[-1])

        pr_at = {k: pick(prec_k10, k) for k in [1, 3, 5, 8]}
        rc_at = {k: pick(rec_k10,  k) for k in [1, 3, 5, 8]}
        hit_at = {k: pick(hit_k10, k) for k in [1, 3, 5, 8]}

        metrics = {
            "subset": subset,
            "MaxPrecision@k": mp_k.tolist(),
            "AUCPR": float(aucpr),
            "MP@1": float(mp_k[0]) if len(mp_k) >= 1 else 0.0,
            "MP@5": float(mp_k[4]) if len(mp_k) >= 5 else 0.0,

            "BestF1_threshold": best_f1,          # tau + precision/recall/f1
            "PRF_at_bestF1_tau": prf_best,        # tp/fp/fn + precision/recall/f1 at that tau

            "Precision@k_1to10": prec_k10.tolist(),
            "Recall@k_1to10": rec_k10.tolist(),
            "Hit@k_1to10": hit_k10.tolist(),

            "Precision@{1,3,5,8}": pr_at,
            "Recall@{1,3,5,8}": rc_at,
            "Hit@{1,3,5,8}": hit_at,
        }

        all_metrics[subset] = metrics

        # Plots
        if args.save_plots:
            save_pr_plot(
                rc, pr, aucpr,
                title=f"{subset} (merged)",
                out_path=os.path.join(args.save_dir, f"{subset}_pr.svg"),
                grid=args.grid, margin=args.margin
            )
            save_mp_plot(
                mp_k,
                title=f"{subset} (merged): MaxPrecision@k",
                out_path=os.path.join(args.save_dir, f"{subset}_mpk.svg"),
                grid_y=args.grid_y
            )

            save_prec_rec_at_k_plot(
                prec_k10, rec_k10,
                out_path=os.path.join(args.save_dir, f"{subset}_prec_rec_at_k.svg"),
                title=f"{subset} (merged): Precision@k / Recall@k",
                grid_y=args.grid_y,
            )

            save_hit_at_k_plot(
                hit_k10,
                out_path=os.path.join(args.save_dir, f"{subset}_hit_at_k.svg"),
                title=f"{subset} (merged): Hit@k",
                grid_y=args.grid_y,
            )


        # Per-chain table
        if args.save_per_chain:
            rows = build_per_chain_table(
                merged=merged,
                subset=subset,
                max_k=args.per_chain_max_k,
                dataset_csv=args.dataset_csv,
            )
            out_csv = os.path.join(args.save_dir, f"{subset}_per_chain.csv")
            os.makedirs(args.save_dir, exist_ok=True)
            fieldnames = list(rows[0].keys()) if rows else []
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"[OK] per-chain CSV -> {out_csv}")

    save_json(all_metrics, os.path.join(args.save_dir, "metrics_summary.json"))
    print(f"[OK] metrics JSON -> {os.path.join(args.save_dir, 'metrics_summary.json')}")


def cmd_seeds(args: argparse.Namespace) -> None:
    versions = list(args.versions)
    os.makedirs(args.save_dir, exist_ok=True)

    use_baseline = (args.baseline_run_tpl is not None)
    base_dir_model = args.base_dir
    base_dir_base = args.baseline_base_dir if args.baseline_base_dir is not None else base_dir_model

    baseline_versions = list(args.baseline_versions) if args.baseline_versions is not None else versions
    if use_baseline and len(baseline_versions) != len(versions):
        raise ValueError("--baseline_versions must have same length as --versions (or omit it)")

    best_tau_for_version = {}
    
    for subset in args.subsets:
        mp_list = []
        pr_list = []
        auc_list = []

        prec_list = []
        rec_list = []
        hit_list = []
        f1_list = []
        tau_list = []


        mp_base_list = []
        auc_base_list = []

        for j, v in enumerate(versions):
            merged = merge_across_folds(
                base_dir=base_dir_model,
                run_dir_template=args.run_tpl,
                subset=subset,
                folds=args.folds,
                dedupe_train=None,
                version=v,
            )

            labels = [ensure_1d(x, "labels") for x in merged["labels"]]
            preds = [ensure_1d(x, "preds") for x in merged["predictions"]]
            weights = np.asarray(merged["weights"], dtype=float)

            mp_k = mp_curve(labels, preds, weights, max_k=args.max_k)
            rc, pr, aucpr = pr_curve_weighted(labels, preds, weights)

            if subset == "validation":
                best_f1 = best_f1_threshold_weighted(labels, preds, weights)
                prf_best = prf_at_tau_weighted(labels, preds, weights, tau=best_f1["tau"])
                prec_k10, rec_k10, hit_k10 = topk_metrics_curves(labels, preds, weights, max_k=10)
                best_tau_for_version[v] = best_f1["tau"]
            elif subset == "test":
                prf_best = prf_at_tau_weighted(labels, preds, weights, tau=best_tau_for_version[v])
                prec_k10, rec_k10, hit_k10 = topk_metrics_curves(labels, preds, weights, max_k=10)

            mp_list.append(mp_k)
            pr_list.append((rc, pr, aucpr))
            auc_list.append(float(aucpr))

            prec_list.append(prec_k10)
            rec_list.append(rec_k10)
            hit_list.append(hit_k10)
            f1_list.append(float(prf_best["f1"]))
            tau_list.append(float(best_tau_for_version[v]))

            # baseline
            if use_baseline:
                vb = baseline_versions[j]

                merged_b = merge_across_folds(
                    base_dir=base_dir_base,
                    run_dir_template=args.baseline_run_tpl,
                    subset=subset,
                    folds=args.folds,
                    dedupe_train=None,
                    version=vb
                )
                labels_b = [ensure_1d(x, "labels") for x in merged_b["labels"]]
                preds_b = [ensure_1d(x, "preds") for x in merged_b["predictions"]]
                weights_b = np.asarray(merged_b["weights"], dtype=float)

                mp_b = mp_curve(labels_b, preds_b, weights_b, max_k=args.max_k)
                _, _, auc_b = pr_curve_weighted(labels_b, preds_b, weights_b)

                mp_base_list.append(mp_b)
                auc_base_list.append(float(auc_b))

            print(f"[OK] subset={subset} version={v}: AUCPR={aucpr:.4f} MP@5={mp_k[4] if len(mp_k)>4 else mp_k[-1]:.4f}")

        mp_arr = np.stack(mp_list, axis=0)  # (V, K)
        auc_arr = np.asarray(auc_list, dtype=float)

        prec_arr = np.stack(prec_list, axis=0)  # (V, 10)
        rec_arr  = np.stack(rec_list, axis=0)   # (V, 10)
        hit_arr  = np.stack(hit_list, axis=0)   # (V, 10)
        f1_arr   = np.asarray(f1_list, dtype=float)
        tau_arr  = np.asarray(tau_list, dtype=float)


        # plots
        mp_out = os.path.join(args.save_dir, f"{subset}_mp_whisker.svg")
        save_mp_whisker(mp_arr, mp_out, title=f"{subset}: MaxPrecision@k (mean ± std over seeds)", grid_y=args.grid_y)

        pr_out = os.path.join(args.save_dir, f"{subset}_pr_overlays.svg")
        pr_summary = save_pr_overlays(
            curves=pr_list,
            out_path=pr_out,
            title=f"{subset}: PR overlays + mean",
            overlay_alpha=args.overlay_alpha,
            grid=args.grid,
            margin=args.margin,
        )

        save_prec_rec_at_k_plot(
            prec_arr.mean(axis=0), rec_arr.mean(axis=0),
            out_path=os.path.join(args.save_dir, f"{subset}_prec_rec_at_k_mean.svg"),
            title=f"{subset}: Precision@k / Recall@k (mean over seeds)",
            grid_y=args.grid_y
        )
        
        save_hit_at_k_plot(
            hit_arr.mean(axis=0),
            out_path=os.path.join(args.save_dir, f"{subset}_hit_at_k_mean.svg"),
            title=f"{subset}: Hit@k (mean over seeds)",
            grid_y=args.grid_y
        )


        summary = {
            "subset": subset,
            "versions": versions,
            "MaxPrecision@k_mean": mp_arr.mean(axis=0).tolist(),
            "MaxPrecision@k_std": (mp_arr.std(axis=0, ddof=1) if mp_arr.shape[0] > 1 else np.zeros(mp_arr.shape[1])).tolist(),
            "AUCPR_mean": float(np.mean(auc_arr)),
            "AUCPR_std": float(np.std(auc_arr, ddof=1)) if auc_arr.size > 1 else 0.0,
            "PR_summary": pr_summary,

            "F1_mean": float(np.mean(f1_arr)),
            "F1_std": float(np.std(f1_arr, ddof=1)) if f1_arr.size > 1 else 0.0,
            "tau_bestF1_mean": float(np.mean(tau_arr)),
            "tau_bestF1_std": float(np.std(tau_arr, ddof=1)) if tau_arr.size > 1 else 0.0,

            "Precision@k_mean_1to10": prec_arr.mean(axis=0).tolist(),
            "Precision@k_std_1to10": (prec_arr.std(axis=0, ddof=1) if prec_arr.shape[0] > 1 else np.zeros(prec_arr.shape[1])).tolist(),

            "Recall@k_mean_1to10": rec_arr.mean(axis=0).tolist(),
            "Recall@k_std_1to10": (rec_arr.std(axis=0, ddof=1) if rec_arr.shape[0] > 1 else np.zeros(rec_arr.shape[1])).tolist(),

            "Hit@k_mean_1to10": hit_arr.mean(axis=0).tolist(),
            "Hit@k_std_1to10": (hit_arr.std(axis=0, ddof=1) if hit_arr.shape[0] > 1 else np.zeros(hit_arr.shape[1])).tolist(),
        }

        # SNR vs baseline (optional)
        if use_baseline:
            mpb_arr = np.stack(mp_base_list, axis=0)
            aucb_arr = np.asarray(auc_base_list, dtype=float)

            # Baseline means
            base_auc_mean = float(np.mean(aucb_arr))
            base_mp_mean = mpb_arr.mean(axis=0)

            # Model stats
            model_auc_mean = float(np.mean(auc_arr))
            model_auc_std = float(np.std(auc_arr, ddof=1)) if len(auc_arr) > 1 else 0.0

            model_mp_mean = mp_arr.mean(axis=0)
            model_mp_std = mp_arr.std(axis=0, ddof=1) if mp_arr.shape[0] > 1 else np.zeros(mp_arr.shape[1])

            # Signal (delta of means)
            signal_auc = model_auc_mean - base_auc_mean
            signal_mp = model_mp_mean - base_mp_mean

            snr_auc = snr_signal_over_noise(signal_auc, model_auc_std)
            snr_k = np.array([snr_signal_over_noise(float(signal_mp[i]), float(model_mp_std[i]))
                              for i in range(len(signal_mp))], dtype=float)

            snr_out = os.path.join(args.save_dir, f"{subset}_snr_mpk.svg")
            save_snr_curve(snr_k, snr_out, title=f"{subset}: SNR(k) for MaxPrecision@k vs baseline")

            # "one number per seed" (z-score style vs baseline mean, using model std as noise)
            eps = 1e-12
            per_seed_snr_auc = ((auc_arr - base_auc_mean) / max(model_auc_std, eps)).tolist() if len(auc_arr) > 1 else [float("nan")]

            summary.update({
                "baseline_versions": baseline_versions,
                "baseline_AUCPR_mean": base_auc_mean,
                "baseline_AUCPR_std": float(np.std(aucb_arr, ddof=1)) if len(aucb_arr) > 1 else 0.0,
                "baseline_MaxPrecision@k_mean": base_mp_mean.tolist(),
                "baseline_MaxPrecision@k_std": (mpb_arr.std(axis=0, ddof=1) if mpb_arr.shape[0] > 1 else np.zeros(mpb_arr.shape[1])).tolist(),

                "signal_AUCPR": float(signal_auc),
                "signal_MaxPrecision@k": signal_mp.tolist(),

                "SNR_AUCPR": float(snr_auc),
                "SNR_AUCPR_per_seed": per_seed_snr_auc,
                "SNR_MaxPrecision@k": snr_k.tolist(),
                "snr_plot": os.path.basename(snr_out),
            })

        save_json(summary, os.path.join(args.save_dir, f"{subset}_seed_summary.json"))
        print(f"[OK] wrote -> {os.path.join(args.save_dir, f'{subset}_seed_summary.json')}")


def cmd_cases(args: argparse.Namespace) -> None:
    paths = resolve_results_paths(args.results_path)
    for p in paths:
        out_csv = args.out_csv
        if out_csv and len(paths) > 1:
            base, ext = os.path.splitext(out_csv)
            out_csv = f"{base}__{os.path.splitext(os.path.basename(p))[0]}{ext}"

        run_cases_one(
            results_pkl=p,
            dataset_csv=args.dataset_csv,
            topn=args.topn,
            mode=args.mode,
            tau=args.tau,
            k=args.k,
            fp_th=args.fp_th,
            fp_weight=args.fp_weight,
            fn_weight=args.fn_weight,
            out_csv=out_csv,
        )


# =========================
# CLI
# =========================

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evaluate_results.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # merge
    ap_m = sub.add_parser("merge", help="Merge folds and compute metrics/plots/per-chain table.")
    ap_m.add_argument("--base_dir", type=str, required=True, help="Directory containing per-fold subdirs.")
    ap_m.add_argument("--save_dir", type=str, required=True, help="Output directory.")
    ap_m.add_argument("--run_tpl", type=str, required=True,
                      help="Per-fold dir template under base_dir, must include {fold}.")
    ap_m.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap_m.add_argument("--subsets", type=str, nargs="+", default=["validation", "test", "train"])
    ap_m.add_argument("--dedupe_train", type=str, default="mean",
                      choices=["mean", "median", "first", "None"])
    ap_m.add_argument("--max_k", type=int, default=20)
    ap_m.add_argument("--save_plots", action="store_true")
    ap_m.add_argument("--grid", type=float, default=0.1, help="Tick step for PR plots.")
    ap_m.add_argument("--margin", type=float, default=0.05, help="Axis margin for PR plots.")
    ap_m.add_argument("--grid_y", type=float, default=0.1, help="Y tick step for MP plots.")

    ap_m.add_argument("--save_per_chain", action="store_true", help="Save per-chain CSV table.")
    ap_m.add_argument("--per_chain_max_k", type=int, default=20, help="How many mp@k columns in per-chain CSV.")
    ap_m.add_argument("--dataset_csv", type=str, default=None, help="Optional dataset.csv for enrichment in per-chain CSV.")
    ap_m.set_defaults(func=cmd_merge)

    # seeds
    ap_s = sub.add_parser("seeds", help="Aggregate multiple versions/seeds. Optional SNR vs baseline.")
    ap_s.add_argument("--base_dir", type=str, required=True)
    ap_s.add_argument("--save_dir", type=str, required=True)
    ap_s.add_argument("--run_tpl", type=str, required=True,
                      help="Template for model fold dirs. Must include {version} and {fold}.")
    ap_s.add_argument("--versions", type=int, nargs="+", required=True)
    ap_s.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap_s.add_argument("--subsets", type=str, nargs="+", default=["validation", "test"])
    ap_s.add_argument("--max_k", type=int, default=20)
    ap_s.add_argument("--overlay_alpha", type=float, default=0.25)
    ap_s.add_argument("--grid", type=float, default=0.1)
    ap_s.add_argument("--margin", type=float, default=0.05)
    ap_s.add_argument("--grid_y", type=float, default=0.1)

    ap_s.add_argument("--baseline_base_dir", type=str, default=None,
                      help="Base dir for baseline runs. If omitted, uses --base_dir.")
    ap_s.add_argument("--baseline_run_tpl", type=str, default=None,
                      help=("Template for baseline fold dirs. May include {version} and {fold}. "
                            "If no {version}, same baseline used for all seeds."))
    ap_s.add_argument("--baseline_versions", type=int, nargs="+", default=None,
                      help="Optional list of baseline versions (same length as --versions).")
    ap_s.set_defaults(func=cmd_seeds)

    # cases
    ap_c = sub.add_parser("cases", help="Mine best/worst predictions by FP/FN from a merged pickle.")
    ap_c.add_argument("--results-path", dest="results_path", type=str, required=True,
                      help="Path to merged pickle file OR dir containing merged pickles.")
    ap_c.add_argument("--dataset-csv", dest="dataset_csv", type=str, default=None,
                      help="dataset.csv for full_name / EC_number etc (joined by Sequence_ID).")
    ap_c.add_argument("--topn", type=int, default=20)
    ap_c.add_argument("--mode", type=str, default="threshold", choices=["threshold", "topk"])
    ap_c.add_argument("--tau", type=float, default=0.5, help="Threshold for mode=threshold.")
    ap_c.add_argument("--k", type=int, default=5, help="k for mode=topk.")
    ap_c.add_argument("--fp-th", dest="fp_th", type=float, default=0.3,
                      help="Confident FP threshold for mode=topk (score>=fp_th).")
    ap_c.add_argument("--fp-weight", type=float, default=1.0)
    ap_c.add_argument("--fn-weight", type=float, default=2.0)
    ap_c.add_argument("--out-csv", dest="out_csv", type=str, default=None,
                      help="Where to save per-chain cases table (CSV).")
    ap_c.set_defaults(func=cmd_cases)

    return ap


def main() -> None:
    ap = build_cli()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
