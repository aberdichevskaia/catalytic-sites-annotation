#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensemble evaluation by averaging existing per-fold pickle outputs.

Input:  5 pickle files (e.g. fold1/test_results.pkl ... fold5/test_results.pkl)
Output: <out_dir>/test_results.pkl and <out_dir>/test_plot.png

The script:
- loads 5 pickles
- aligns by 'ids' (intersection)
- averages predictions per chain (arithmetic mean)
- saves pickle in the same schema as the input pickles
- saves PR curve png
"""

import argparse
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pickle_paths",
        nargs=5,
        required=True,
        help="Five per-fold pickle paths (typically each is .../test_results.pkl).",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ensemble_name", default="ensemble_mean")
    p.add_argument("--subset_key", default="test", help="Usually 'test'.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ---------------- Plotting (same logic as your main script style) ---------------- #

def make_PR_curve(labels, predictions, weights, subset_name, title=""):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc

    # Repeat per-chain weights over residues
    weights_repeated = np.array(
        [np.ones(len(lbl)) * float(w) for lbl, w in zip(labels, weights)],
        dtype=object,
    )

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)

    # If multi-dim (e.g. (L,2)), use last channel as positive
    if y_pred.ndim > 1:
        y_pred = y_pred[..., -1]
        # make y_true boolean if it is categorical
        if y_true.ndim > 1:
            y_true = y_true[..., -1]
        else:
            y_true = (y_true == np.max(y_true))

    bad = np.isnan(y_pred) | np.isnan(y_true) | np.isinf(y_pred) | np.isinf(y_true)
    if bad.any():
        y_pred = y_pred.copy()
        y_pred[bad] = np.nanmedian(y_pred)

    w_flat = np.concatenate(weights_repeated)
    precision, recall, _ = precision_recall_curve(y_true[~bad], y_pred[~bad], sample_weight=w_flat[~bad])
    aucpr = auc(recall, precision)
    print(f"[OK] {subset_name} AUCPR={aucpr:.4f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(recall, precision, linewidth=2.0, label=f"{subset_name} (AUCPR={aucpr:.3f})")
    ax.grid(True)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


# ---------------- Core helpers ---------------- #

def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    # object array -> list
    arr = np.asarray(x, dtype=object)
    return list(arr.tolist())


def _to_np_list(preds: Any) -> List[np.ndarray]:
    return [np.asarray(p) for p in _as_list(preds)]


def _to_np_int_list(labels: Any) -> List[np.ndarray]:
    out = []
    for y in _as_list(labels):
        y = np.asarray(y)
        # keep as-is, but ensure 1D if it's plain 0/1
        out.append(y)
    return out


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Pickle {path} is not a dict payload.")
    required = ["labels", "predictions", "weights", "ids", "splits"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Pickle {path} missing key '{k}'. Keys={list(obj.keys())}")
    return obj


def _index_by_id(payload: Dict[str, Any]) -> Dict[str, int]:
    ids = payload["ids"]
    return {str(i): idx for idx, i in enumerate(ids)}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    payloads = [_load_pickle(p) for p in args.pickle_paths]

    # Build ID intersections
    id_sets = [set(map(str, pl["ids"])) for pl in payloads]
    common_ids = set.intersection(*id_sets)
    if len(common_ids) == 0:
        raise RuntimeError("No common ids across the 5 pickles.")

    # Keep original order from the 1st pickle
    base_ids = [str(i) for i in payloads[0]["ids"]]
    kept_ids = [i for i in base_ids if i in common_ids]

    # Reindex for each payload
    idx_maps = [_index_by_id(pl) for pl in payloads]

    # Convert base labels/weights/splits
    base_labels = _to_np_int_list(payloads[0]["labels"])
    base_weights = np.asarray(payloads[0]["weights"], dtype=float)
    base_splits = list(payloads[0]["splits"])

    # Some pickles might include extra entries not in kept_ids; we will align everything.
    out_labels: List[np.ndarray] = []
    out_weights: List[float] = []
    out_splits: List[Any] = []
    out_preds_sum: List[np.ndarray] = []

    dropped_shape = 0
    dropped_label_mismatch = 0

    # For faster checks: pre-convert predictions for each payload
    preds_all = [_to_np_list(pl["predictions"]) for pl in payloads]
    labels_all = [_to_np_int_list(pl["labels"]) for pl in payloads]
    weights_all = [np.asarray(pl["weights"], dtype=float) for pl in payloads]

    for id_ in kept_ids:
        # Gather per-model indices
        indices = [m[id_] for m in idx_maps]

        # Check shapes match across models
        pred_list = [preds_all[k][indices[k]] for k in range(5)]
        shapes = [p.shape for p in pred_list]
        if any(s != shapes[0] for s in shapes[1:]):
            dropped_shape += 1
            continue

        # Check labels match (strict) across models (optional but good)
        lab_list = [labels_all[k][indices[k]] for k in range(5)]
        if any(l.shape != lab_list[0].shape for l in lab_list[1:]) or any(
            not np.array_equal(lab_list[0], l) for l in lab_list[1:]
        ):
            dropped_label_mismatch += 1
            continue

        # Average weights: ideally identical; we'll take mean and warn later if needed
        w_list = [float(weights_all[k][indices[k]]) for k in range(5)]
        w_mean = float(np.mean(w_list))

        # Average predictions
        pred_mean = (pred_list[0].astype(np.float32) +
                     pred_list[1].astype(np.float32) +
                     pred_list[2].astype(np.float32) +
                     pred_list[3].astype(np.float32) +
                     pred_list[4].astype(np.float32)) / 5.0

        out_preds_sum.append(pred_mean)
        out_labels.append(lab_list[0])
        out_weights.append(w_mean)
        out_splits.append(base_splits[idx_maps[0][id_]])

    if len(out_preds_sum) == 0:
        raise RuntimeError("All common ids were dropped (shape/label mismatches).")

    if dropped_shape or dropped_label_mismatch:
        print(f"[WARN] Dropped {dropped_shape} ids due to prediction shape mismatch.")
        print(f"[WARN] Dropped {dropped_label_mismatch} ids due to label mismatch.")

    # Save pickle (same schema as input pickles)
    out_payload = dict(payloads[0])  # keep extra keys if present
    out_payload["subset"] = args.subset_key
    out_payload["model_name"] = args.ensemble_name
    out_payload["ids"] = [i for i in kept_ids if i in set(out_payload["ids"])]  # not used; replaced below
    out_payload["ids"] = [str_id for str_id in kept_ids if str_id in common_ids]  # raw, but might include dropped
    # Actually use the final kept list in order:
    out_payload["ids"] = [str_id for str_id in kept_ids if str_id in common_ids][:len(out_preds_sum)]
    # More robust: reconstruct from aligned loop
    out_payload["ids"] = [id_ for id_ in kept_ids if id_ in common_ids][:len(out_preds_sum)]

    # But we *must* keep exact same order as outputs:
    # We'll rebuild ids from out_splits length (same as out_preds_sum).
    # Easiest: keep ids in the same order as we appended:
    # We didn't store them in that loop — so let's do it now properly:
    # (Small fix: store ids during loop)
    # For correctness, we will instead re-run ids accumulation in a clean pass:

    # --- Rebuild ids in output order ---
    out_ids = []
    out_labels2 = []
    out_weights2 = []
    out_splits2 = []
    out_preds2 = []

    # Re-run to capture exact ids in the same order
    for id_ in kept_ids:
        indices = [m[id_] for m in idx_maps]
        pred_list = [preds_all[k][indices[k]] for k in range(5)]
        shapes = [p.shape for p in pred_list]
        if any(s != shapes[0] for s in shapes[1:]):
            continue
        lab_list = [labels_all[k][indices[k]] for k in range(5)]
        if any(l.shape != lab_list[0].shape for l in lab_list[1:]) or any(
            not np.array_equal(lab_list[0], l) for l in lab_list[1:]
        ):
            continue

        w_list = [float(weights_all[k][indices[k]]) for k in range(5)]
        w_mean = float(np.mean(w_list))
        pred_mean = sum(p.astype(np.float32) for p in pred_list) / 5.0

        out_ids.append(id_)
        out_labels2.append(lab_list[0])
        out_weights2.append(w_mean)
        out_splits2.append(payloads[0]["splits"][idx_maps[0][id_]])
        out_preds2.append(pred_mean)

    out_payload["ids"] = out_ids
    out_payload["labels"] = out_labels2
    out_payload["predictions"] = out_preds2
    out_payload["weights"] = out_weights2
    out_payload["splits"] = out_splits2

    pkl_path = os.path.join(args.out_dir, f"{args.subset_key}_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(out_payload, f)
    print(f"[OK] Saved ensemble pickle -> {pkl_path}")

    # Save PR curve
    fig = make_PR_curve(
        labels=out_labels2,
        predictions=out_preds2,
        weights=out_weights2,
        subset_name=args.subset_key,
        title=f"Enzyme active site prediction: {args.ensemble_name} ({args.subset_key})",
    )
    png_path = os.path.join(args.out_dir, f"{args.subset_key}_plot.png")
    fig.savefig(png_path, dpi=args.dpi)
    print(f"[OK] Saved PR curve -> {png_path}")

    # Quick summary
    print(f"[INFO] Ensemble kept {len(out_ids)} chains out of common {len(common_ids)}.")


if __name__ == "__main__":
    main()