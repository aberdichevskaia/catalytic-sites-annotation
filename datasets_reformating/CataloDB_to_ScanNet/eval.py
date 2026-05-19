#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold selection + F1 evaluation for catalytic site prediction.

Workflow:
1) Find best F1 threshold on merged OOF validation pickles (5 folds):
   python threshold_and_f1.py find-threshold --val_pickles fold1/validation_results.pkl ... fold5/validation_results.pkl --out_json best_thr.json

2) Evaluate F1 on an ensemble test pickle (already averaged predictions):
   python threshold_and_f1.py eval-f1 --pickle ensemble/test_results.pkl --thr_json best_thr.json

Pickle schema expected (robust):
- labels: list/obj-array of per-chain arrays (shape (L,) or (L,C))
- predictions: list/obj-array of per-chain arrays (shape (L,) or (L,C))
- weights (optional): per-chain weights (len = n_chains)

By default:
- Positive class = 1 (for C=2)
- If labels are one-hot (L,2), we use argmax -> {0,1}
- If labels are binary floats/ints (L,), positive is y>0
- If predictions are (L,2), score is pred[..., 1]
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score


# ----------------------------- Utilities ----------------------------- #

def _as_list(x: Any) -> List[Any]:
    """Convert list / object-array / numpy array into a Python list."""
    if isinstance(x, list):
        return x
    arr = np.asarray(x, dtype=object)
    return list(arr.tolist())


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: pickle is not a dict (got {type(obj)})")
    if "labels" not in obj or "predictions" not in obj:
        raise ValueError(f"{path}: expected keys 'labels' and 'predictions', got keys={list(obj.keys())}")
    return obj


def _sanitize_scores(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in scores with median (stable)."""
    x = x.astype(np.float32, copy=False)
    bad = ~np.isfinite(x)
    if bad.any():
        med = np.nanmedian(x[~bad]) if (~bad).any() else 0.0
        x = x.copy()
        x[bad] = med
    return x


def _labels_to_binary(y: np.ndarray, positive_class: int) -> np.ndarray:
    """
    Convert labels to 0/1.
    - If y is (L, C) with C>1: argmax -> class id, then (==positive_class)
    - Else: (y > 0)
    """
    y = np.asarray(y)
    if y.ndim >= 2 and y.shape[-1] > 1:
        cls = np.argmax(y, axis=-1).astype(np.int32)
        return (cls == int(positive_class)).astype(np.int32).reshape(-1)
    return (y.reshape(-1) > 0).astype(np.int32)


def _preds_to_score(p: np.ndarray, positive_class: int) -> np.ndarray:
    """
    Convert predictions to 1D score.
    - If p is (L, C) with C>1: score = p[..., positive_class]
    - Else: p as-is
    """
    p = np.asarray(p)
    if p.ndim >= 2 and p.shape[-1] > 1:
        pc = int(positive_class)
        if pc < 0:
            pc = p.shape[-1] + pc
        if pc < 0 or pc >= p.shape[-1]:
            raise ValueError(f"positive_class={positive_class} out of range for pred last-dim={p.shape[-1]}")
        return p[..., pc].reshape(-1).astype(np.float32)
    return p.reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class FlatData:
    y_true: np.ndarray     # (N,)
    y_score: np.ndarray    # (N,)
    w: np.ndarray          # (N,)


def flatten_pickle_payload(
    payload: dict,
    positive_class: int = 1,
    use_weights: bool = False,
) -> FlatData:
    """
    Flatten a pickle payload into residue-level arrays.
    weights:
      - if use_weights and payload has 'weights': per-chain weight repeated to residues
      - else: ones
    """
    labels_list = _as_list(payload["labels"])
    preds_list = _as_list(payload["predictions"])

    if len(labels_list) != len(preds_list):
        raise ValueError(f"labels/predictions length mismatch: {len(labels_list)} vs {len(preds_list)}")

    if use_weights and "weights" in payload:
        chain_w = np.asarray(payload["weights"], dtype=np.float32)
        if chain_w.shape[0] != len(labels_list):
            raise ValueError(f"weights length mismatch: {chain_w.shape[0]} vs {len(labels_list)}")
    else:
        chain_w = np.ones(len(labels_list), dtype=np.float32)

    y_true_parts: List[np.ndarray] = []
    y_score_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []

    for y, p, w0 in zip(labels_list, preds_list, chain_w):
        y_bin = _labels_to_binary(np.asarray(y), positive_class=positive_class)
        score = _preds_to_score(np.asarray(p), positive_class=positive_class)
        if y_bin.shape[0] != score.shape[0]:
            raise ValueError(f"Per-chain length mismatch: labels={y_bin.shape[0]} preds={score.shape[0]}")

        score = _sanitize_scores(score)

        y_true_parts.append(y_bin)
        y_score_parts.append(score)
        w_parts.append(np.full_like(y_bin, float(w0), dtype=np.float32))

    y_true = np.concatenate(y_true_parts, axis=0)
    y_score = np.concatenate(y_score_parts, axis=0)
    w = np.concatenate(w_parts, axis=0)

    return FlatData(y_true=y_true, y_score=y_score, w=w)


# ----------------------------- Threshold selection ----------------------------- #

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    n_residues: int
    n_positives: int


def find_best_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> ThresholdResult:
    """
    Find threshold maximizing F1 using PR curve thresholds.
    Uses sklearn precision_recall_curve (supports sample_weight).
    """
    if w is None:
        w = np.ones_like(y_true, dtype=np.float32)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, sample_weight=w)

    # precision/recall have length = len(thresholds)+1
    # For each threshold t_j, the corresponding point is precision[j+1], recall[j+1]
    p = precision[1:]
    r = recall[1:]
    t = thresholds

    denom = p + r
    f1 = np.where(denom > 0, 2.0 * p * r / denom, 0.0)

    j = int(np.argmax(f1))
    thr = float(t[j])
    f1v = float(f1[j])
    pv = float(p[j])
    rv = float(r[j])

    return ThresholdResult(
        threshold=thr,
        f1=f1v,
        precision=pv,
        recall=rv,
        n_residues=int(y_true.shape[0]),
        n_positives=int(y_true.sum()),
    )


# ----------------------------- F1 evaluation ----------------------------- #

@dataclass(frozen=True)
class F1Report:
    threshold: float
    f1: float
    precision: float
    recall: float
    n_residues: int
    n_positives: int


def eval_f1_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> F1Report:
    y_hat = (y_score >= float(threshold)).astype(np.int32)

    p = float(precision_score(y_true, y_hat, zero_division=0))
    r = float(recall_score(y_true, y_hat, zero_division=0))
    f1v = float(f1_score(y_true, y_hat, zero_division=0))

    return F1Report(
        threshold=float(threshold),
        f1=f1v,
        precision=p,
        recall=r,
        n_residues=int(y_true.shape[0]),
        n_positives=int(y_true.sum()),
    )


# ----------------------------- CLI ----------------------------- #

def cmd_find_threshold(args: argparse.Namespace) -> None:
    all_true: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    all_w: List[np.ndarray] = []

    for path in args.val_pickles:
        payload = _load_pickle(path)
        flat = flatten_pickle_payload(payload, positive_class=args.positive_class, use_weights=args.use_weights)
        all_true.append(flat.y_true)
        all_score.append(flat.y_score)
        all_w.append(flat.w)

    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    w = np.concatenate(all_w, axis=0)

    res = find_best_threshold_f1(y_true, y_score, w=w if args.use_weights else None)

    pos_rate = res.n_positives / max(res.n_residues, 1)
    print(f"[INFO] residues={res.n_residues:,} positives={res.n_positives:,} pos_rate={pos_rate:.6f}")
    print(f"[BEST] thr={res.threshold:.6f}  F1={res.f1:.6f}  P={res.precision:.6f}  R={res.recall:.6f}")

    if args.out_json:
        out = {
            "threshold": res.threshold,
            "f1": res.f1,
            "precision": res.precision,
            "recall": res.recall,
            "n_residues": res.n_residues,
            "n_positives": res.n_positives,
            "positive_class": args.positive_class,
            "use_weights": bool(args.use_weights),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved -> {args.out_json}")


def cmd_eval_f1(args: argparse.Namespace) -> None:
    payload = _load_pickle(args.pickle)
    flat = flatten_pickle_payload(payload, positive_class=args.positive_class, use_weights=False)

    if args.thr_json:
        with open(args.thr_json, "r", encoding="utf-8") as f:
            thr_obj = json.load(f)
        threshold = float(thr_obj["threshold"])
    else:
        threshold = float(args.threshold)

    rep = eval_f1_at_threshold(flat.y_true, flat.y_score, threshold=threshold)
    pos_rate = rep.n_positives / max(rep.n_residues, 1)
    print(f"[INFO] residues={rep.n_residues:,} positives={rep.n_positives:,} pos_rate={pos_rate:.6f}")
    print(f"[TEST] thr={rep.threshold:.6f}  F1={rep.f1:.6f}  P={rep.precision:.6f}  R={rep.recall:.6f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("threshold_and_f1")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_thr = sub.add_parser("find-threshold", help="Find best F1 threshold on merged validation pickles (OOF).")
    p_thr.add_argument("--val_pickles", nargs="+", required=True,
                       help="Paths to validation_results.pkl files (typically 5).")
    p_thr.add_argument("--positive_class", type=int, default=1,
                       help="Positive class index if labels/preds are one-hot. Default=1.")
    p_thr.add_argument("--use_weights", action="store_true",
                       help="Use per-chain weights from pickles (repeated to residues).")
    p_thr.add_argument("--out_json", type=str, default=None,
                       help="Optional: save best threshold + stats to JSON.")
    p_thr.set_defaults(func=cmd_find_threshold)

    p_eval = sub.add_parser("eval-f1", help="Evaluate F1 on a (ensemble) pickle at a given threshold.")
    p_eval.add_argument("--pickle", required=True, help="Path to test_results.pkl (e.g. ensemble).")
    p_eval.add_argument("--positive_class", type=int, default=1,
                        help="Positive class index if labels/preds are one-hot. Default=1.")
    g = p_eval.add_mutually_exclusive_group(required=True)
    g.add_argument("--threshold", type=float, help="Threshold value.")
    g.add_argument("--thr_json", type=str, help="JSON produced by find-threshold.")
    p_eval.set_defaults(func=cmd_eval_f1)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()