#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


import numpy as np
from sklearn.metrics import recall_score, precision_score

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize": 16,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 0.9,
    "svg.fonttype": "path",
})


# figures sizes (for the paper)
W_FULL = 6.27
GUTTER = 0.15
W_HALF = (W_FULL - GUTTER) / 2
W_THIRD = (W_FULL - GUTTER) / 3
W_TWO_THIRDS = 2 * (W_FULL - GUTTER) / 3

H_SMALL = 1.9
H_WIDE = 2.7

# def plot_recall_vs_fp_load(thresholds, recall_val, fp_stats, out_path, mark_thrs=(0.65, 0.50, 0.35, 0.26, 0.0612), zoom=False):
#     x = fp_stats["avg_fp_raw"]   # FP-load: average FP per protein (RAW)
#     y = recall_val               # recall (weighted)

#     if zoom:
#         plt.figure(figsize=(W_HALF, H_SMALL))
#     else:
#         plt.figure(figsize=(W_FULL, H_WIDE))
#     plt.scatter(x, y, s=10, alpha=0.4, label="All thresholds")

#     # Highlight chosen point tau=0.35
#     # (we'll annotate marked thresholds too)
#     for t0 in mark_thrs:
#         idx = int(np.argmin(np.abs(thresholds - t0)))
#         plt.scatter([x[idx]], [y[idx]], s=80)
#         plt.annotate(f"τ≈{thresholds[idx]:.3f}", (x[idx], y[idx]),
#                      textcoords="offset points", xytext=(6, 6), fontsize=10)

#     # Optional: emphasize tau=0.35
#     idx35 = int(np.argmin(np.abs(thresholds - 0.35)))
#     plt.scatter([x[idx35]], [y[idx35]], s=140, marker="*", label=f"Chosen τ≈{thresholds[idx35]:.3f}")

#     if zoom:
#         plt.xlim(0, 10)

#     plt.xlabel("Avg FP per protein (validation, RAW)")
#     plt.ylabel("Recall (validation, weighted)")
#     plt.title("Threshold trade-off: Recall vs FP-load")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(out_path, dpi=1000, format="png")
#     plt.close()


def plot_val_precision_recall_vs_tau(thresholds, precision_val, recall_val, out_path, chosen_tau=0.35):
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.asarray(thresholds, dtype=np.float64)
    p = np.asarray(precision_val, dtype=np.float64)
    r = np.asarray(recall_val, dtype=np.float64)

    plt.figure(figsize=(2*W_HALF, 2*W_HALF))
    plt.plot(t, r, label="Recall (validation)")
    plt.plot(t, p, label="Precision (validation)")

    plt.axvline(chosen_tau, linestyle=":", label=f"Chosen τ={chosen_tau:.2f}")

    plt.xlabel("Threshold τ")
    plt.ylabel("Value")
    plt.title("Validation precision and recall vs threshold")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=1000, format="png")
    plt.close()


def weighted_quantile(values, q, sample_weight):
    """Weighted quantile, q in [0,1]."""
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    w = np.clip(w, 0.0, None)

    if values.size == 0:
        return float("nan")

    idx = np.argsort(values)
    v = values[idx]
    w = w[idx]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(np.quantile(values, q))
    cutoff = q * total
    j = int(np.searchsorted(cw, cutoff, side="left"))
    j = min(max(j, 0), v.size - 1)
    return float(v[j])


def confusion_per_chain_at_threshold(labels_per_chain, preds_per_chain, thr):
    """
    labels_per_chain: list of (L,) {0,1}
    preds_per_chain : list of (L,) float in [0,1]
    Returns arrays of shape (n_chains,) for TP, FP, FN, n_pred.
    """
    n = len(labels_per_chain)
    tp = np.zeros(n, dtype=np.int32)
    fp = np.zeros(n, dtype=np.int32)
    fn = np.zeros(n, dtype=np.int32)
    n_pred = np.zeros(n, dtype=np.int32)

    for i, (y, p) in enumerate(zip(labels_per_chain, preds_per_chain)):
        pred = (p >= thr)
        y = (y.astype(np.int32) == 1)
        tp[i] = int(np.sum(pred & y))
        fp[i] = int(np.sum(pred & (~y)))
        fn[i] = int(np.sum((~pred) & y))
        n_pred[i] = int(np.sum(pred))

    return tp, fp, fn, n_pred


def eval_fp_load_vs_thresholds(
    labels_per_chain, preds_per_chain, w_chain, thresholds
):
    """
    Computes FP load metrics per threshold (raw + weighted).
    Returns dict of arrays.
    """
    w = np.asarray(w_chain, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    w_sum = float(w.sum()) if w.sum() > 0 else 1.0

    out = {
        "avg_fp_raw": np.zeros_like(thresholds, dtype=np.float32),
        "p95_fp_raw": np.zeros_like(thresholds, dtype=np.float32),
        "frac_fp_ge1_raw": np.zeros_like(thresholds, dtype=np.float32),
        "avg_pred_raw": np.zeros_like(thresholds, dtype=np.float32),
        "p95_pred_raw": np.zeros_like(thresholds, dtype=np.float32),

        "avg_fp_w": np.zeros_like(thresholds, dtype=np.float32),
        "p95_fp_w": np.zeros_like(thresholds, dtype=np.float32),
        "frac_fp_ge1_w": np.zeros_like(thresholds, dtype=np.float32),
        "avg_pred_w": np.zeros_like(thresholds, dtype=np.float32),
        "p95_pred_w": np.zeros_like(thresholds, dtype=np.float32),
    }

    for k, thr in enumerate(thresholds):
        tp, fp, fn, n_pred = confusion_per_chain_at_threshold(labels_per_chain, preds_per_chain, float(thr))

        fp_f = fp.astype(np.float64)
        pred_f = n_pred.astype(np.float64)

        # RAW (unweighted)
        out["avg_fp_raw"][k] = float(fp_f.mean())
        out["p95_fp_raw"][k] = float(np.quantile(fp_f, 0.95))
        out["frac_fp_ge1_raw"][k] = float(np.mean(fp_f >= 1))
        out["avg_pred_raw"][k] = float(pred_f.mean())
        out["p95_pred_raw"][k] = float(np.quantile(pred_f, 0.95))

        # Weighted by chain weights
        out["avg_fp_w"][k] = float(np.sum(fp_f * w) / w_sum)
        out["p95_fp_w"][k] = float(weighted_quantile(fp_f, 0.95, w))
        out["frac_fp_ge1_w"][k] = float(np.sum((fp_f >= 1).astype(np.float64) * w) / w_sum)
        out["avg_pred_w"][k] = float(np.sum(pred_f * w) / w_sum)
        out["p95_pred_w"][k] = float(weighted_quantile(pred_f, 0.95, w))

    return out


def pick_tau_max_recall_under_fp_budget(
    thresholds, recall_val, fp_stats,  # fp_stats from eval_fp_load_vs_thresholds
    avg_fp_max=1.0, p95_fp_max=5.0, use_weighted=False
):
    """
    Choose threshold maximizing recall subject to FP load constraints.
    If use_weighted=True, uses weighted FP metrics; else raw.
    """
    if use_weighted:
        avg_fp = fp_stats["avg_fp_w"]
        p95_fp = fp_stats["p95_fp_w"]
    else:
        avg_fp = fp_stats["avg_fp_raw"]
        p95_fp = fp_stats["p95_fp_raw"]

    mask = (avg_fp <= avg_fp_max) & (p95_fp <= p95_fp_max)
    if not np.any(mask):
        return None

    idxs = np.where(mask)[0]
    # max recall, tie-break: smaller avg_fp, then larger threshold (stricter)
    best = idxs[np.lexsort((-thresholds[idxs], avg_fp[idxs], -recall_val[idxs]))][0]
    return int(best)


from sklearn.metrics import precision_score, recall_score
import numpy as np

import numpy as np
from sklearn.metrics import precision_score, recall_score

def eval_one_threshold(thr, y_flat, p_flat, w_res, y_chain=None, p_chain=None, w_chain=None, show_chain_stats=True):
    """
    Prints residue-level (weighted) precision/recall/F1/FPR at threshold thr.
    Optionally prints per-chain RAW candidate/FP summaries (if chains provided).
    """

    # --- sanitize ---
    y = np.asarray(y_flat, dtype=np.int32)
    p = np.asarray(p_flat, dtype=np.float64)
    w = np.asarray(w_res, dtype=np.float64)
    w = np.clip(w, 0.0, None)

    pred = (p >= thr).astype(np.int32)

    # --- residue-level weighted precision/recall (sklearn) ---
    prec = precision_score(y, pred, sample_weight=w, zero_division=0)
    rec  = recall_score(y, pred, sample_weight=w)

    # --- weighted confusion counts (manual) ---
    y_pos = (y == 1)
    y_neg = ~y_pos
    p_pos = (pred == 1)
    p_neg = ~p_pos

    TPw = float(np.sum(w[y_pos & p_pos]))
    FPw = float(np.sum(w[y_neg & p_pos]))
    TNw = float(np.sum(w[y_neg & p_neg]))
    FNw = float(np.sum(w[y_pos & p_neg]))

    # --- F1 (weighted, derived from prec/rec to be consistent) ---
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # --- FPR (weighted) ---
    denom = (FPw + TNw)
    fpr = (FPw / denom) if denom > 0 else 0.0

    n_pred_total = int(pred.sum())
    frac_res_ge = float(np.mean(p >= thr))

    print(f"thr={thr:.3f}")
    print(f"  residue-level (weighted):  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  fpr={fpr:.6f}")
    print(f"  weighted confusion: TP={TPw:.2f} FP={FPw:.2f} TN={TNw:.2f} FN={FNw:.2f}")
    print(f"  residues >= thr: {n_pred_total}  (fraction {frac_res_ge:.6f})")

    # --- optional per-chain RAW summaries (your previous block) ---
    if show_chain_stats and (y_chain is not None) and (p_chain is not None):
        fp = []
        tp = []
        n_pred = []
        for yc, pc in zip(y_chain, p_chain):
            pc = np.asarray(pc)
            yc = np.asarray(yc, dtype=np.int32)

            pr = (pc >= thr)
            y1 = (yc == 1)

            tp_i = int(np.sum(pr & y1))
            fp_i = int(np.sum(pr & (~y1)))

            tp.append(tp_i)
            fp.append(fp_i)
            n_pred.append(int(np.sum(pr)))

        fp = np.asarray(fp)
        tp = np.asarray(tp)
        n_pred = np.asarray(n_pred)

        print(f"  per-chain RAW: avgPred={n_pred.mean():.2f} p95Pred={np.quantile(n_pred,0.95):.1f}")
        print(f"               avgTP={tp.mean():.2f}  avgFP={fp.mean():.2f}  p95FP={np.quantile(fp,0.95):.1f}  frac(FP>=1)={(fp>=1).mean():.3f}")




# ---------------- IO ----------------

def find_existing(path_dir: str, candidates: List[str]) -> str:
    """Return first existing path in candidates, joined with path_dir."""
    for name in candidates:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found under {path_dir}")


def load_results_per_chain(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load merged pickles and return:
      labels_per_chain: list of (L_i,) int arrays
      preds_per_chain : list of (L_i,) float arrays
      w_chain         : (n_chains,) float array
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    labels_per_chain = [np.asarray(x, dtype=np.int32) for x in data["labels"]]
    preds_per_chain = [np.asarray(x, dtype=np.float32) for x in data["predictions"]]
    w_chain = np.asarray([float(w) for w in data["weights"]], dtype=np.float32)

    if len(labels_per_chain) != len(preds_per_chain) or len(labels_per_chain) != len(w_chain):
        raise ValueError("Mismatch between number of chains in labels/predictions/weights.")

    for i, (y, p) in enumerate(zip(labels_per_chain, preds_per_chain)):
        if len(y) != len(p):
            raise ValueError(f"Length mismatch in chain {i}: y={len(y)} p={len(p)}")

    # Clean NaN/Inf in predictions (per chain)
    for i in range(len(preds_per_chain)):
        p = preds_per_chain[i]
        bad = np.isnan(p) | np.isinf(p)
        if bad.any():
            med = np.nanmedian(p[~bad]) if (~bad).any() else 0.0
            p = p.copy()
            p[bad] = med
            preds_per_chain[i] = p

    return labels_per_chain, preds_per_chain, w_chain


def flatten_with_residue_weights(
    labels_per_chain: List[np.ndarray],
    preds_per_chain: List[np.ndarray],
    w_chain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten residues and create per-residue weights by repeating chain weight.
    """
    y_flat = np.concatenate(labels_per_chain).astype(np.int32)
    p_flat = np.concatenate(preds_per_chain).astype(np.float32)
    w_res = np.concatenate([
        np.full(len(y), w, dtype=np.float32)
        for y, w in zip(labels_per_chain, w_chain)
    ])
    return y_flat, p_flat, w_res


# ---------------- Stats helpers ----------------

def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
    """
    Weighted quantile in [0,1]. Assumes quantile in [0,1].
    """
    values = np.asarray(values, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if values.size == 0:
        return float("nan")

    sorter = np.argsort(values)
    v = values[sorter]
    w = sample_weight[sorter]
    w = np.clip(w, 0.0, None)

    cum_w = np.cumsum(w)
    total = cum_w[-1]
    if total <= 0:
        # Fallback to обычный квантиль
        return float(np.quantile(values, quantile))

    cutoff = quantile * total
    idx = int(np.searchsorted(cum_w, cutoff, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def make_threshold_grid_from_quantiles(p_flat: np.ndarray, n: int = 400) -> np.ndarray:
    """
    Build a threshold grid using quantiles of the probability distribution.
    This avoids iterating over millions of unique scores.
    """
    p_flat = np.asarray(p_flat, dtype=np.float64)
    p_flat = p_flat[np.isfinite(p_flat)]
    if p_flat.size == 0:
        return np.linspace(0.0, 1.0, n)

    qs = np.linspace(0.0, 1.0, n)
    thr = np.quantile(p_flat, qs)
    thr = np.unique(np.clip(thr, 0.0, 1.0))
    # Ensure endpoints exist
    thr = np.unique(np.concatenate([np.array([0.0], dtype=np.float64), thr, np.array([1.0], dtype=np.float64)]))
    return thr.astype(np.float32)


def candidates_per_chain(preds_per_chain: List[np.ndarray], threshold: float) -> np.ndarray:
    """Return counts of residues with p>=threshold for each chain."""
    return np.asarray([int(np.sum(p >= threshold)) for p in preds_per_chain], dtype=np.int32)

def plot_val_precision_recall_fdr_vs_tau(thresholds, precision_val, recall_val, out_path, chosen_tau=0.35):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    precision_val = np.asarray(precision_val, dtype=np.float64)
    recall_val = np.asarray(recall_val, dtype=np.float64)
    fdr_val = 1.0 - precision_val

    plt.figure(figsize=(2*W_HALF, 2*W_HALF))
    plt.plot(thresholds, recall_val, label="Recall (validation)")
    plt.plot(thresholds, precision_val, label="Precision (validation)")
    plt.plot(thresholds, fdr_val, label="FDR = 1 - Precision (validation)")

    # chosen tau (exact)
    plt.axvline(chosen_tau, linestyle=":", label=f"Chosen τ={chosen_tau:.2f}")

    plt.xlabel("Threshold τ")
    plt.ylabel("Value")
    plt.title("Validation precision/recall and apparent FDR vs threshold")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=1000, format="png")
    plt.close()



# ---------------- Core evaluation ----------------

@dataclass
class ThresholdStats:
    threshold: np.ndarray
    precision_val: np.ndarray
    recall_val: np.ndarray
    avg_cand_val: np.ndarray
    p95_cand_val: np.ndarray
    frac_over_b_val: Dict[int, np.ndarray]  # key=B -> array frac_over_B(τ)


def eval_thresholds_on_validation(
    y_val_flat: np.ndarray,
    p_val_flat: np.ndarray,
    w_val_res: np.ndarray,
    preds_val_per_chain: List[np.ndarray],
    w_val_chain: np.ndarray,
    thresholds: np.ndarray,
    b_list: List[int],
) -> ThresholdStats:
    """
    Compute weighted precision/recall (residue-level) and chain-level candidate budgets for each threshold.
    """
    precision_vals = np.zeros_like(thresholds, dtype=np.float32)
    recall_vals = np.zeros_like(thresholds, dtype=np.float32)
    avg_cands = np.zeros_like(thresholds, dtype=np.float32)
    p95_cands = np.zeros_like(thresholds, dtype=np.float32)

    frac_over_b = {b: np.zeros_like(thresholds, dtype=np.float32) for b in b_list}

    w_chain = np.asarray(w_val_chain, dtype=np.float64)
    w_chain = np.clip(w_chain, 0.0, None)
    w_chain_sum = float(np.sum(w_chain)) if np.sum(w_chain) > 0 else 1.0

    for i, t in enumerate(thresholds):
        # Residue-level metrics on validation
        y_pred = (p_val_flat >= t).astype(np.int32)
        precision_vals[i] = precision_score(
            y_val_flat, y_pred, sample_weight=w_val_res, zero_division=0
        )
        recall_vals[i] = recall_score(
            y_val_flat, y_pred, sample_weight=w_val_res
        )

        # Chain-level candidate counts
        c = candidates_per_chain(preds_val_per_chain, float(t)).astype(np.float64)

        # Weighted avg candidates per chain
        avg_cands[i] = float(np.sum(c * w_chain) / w_chain_sum)

        # Weighted p95 candidates per chain
        p95_cands[i] = weighted_quantile(c, 0.95, w_chain)

        # Weighted fraction of chains over budget
        for b in b_list:
            over = (c > b).astype(np.float64)
            frac_over_b[b][i] = float(np.sum(over * w_chain) / w_chain_sum)

    return ThresholdStats(
        threshold=thresholds,
        precision_val=precision_vals,
        recall_val=recall_vals,
        avg_cand_val=avg_cands,
        p95_cand_val=p95_cands,
        frac_over_b_val=frac_over_b,
    )


# ---------------- Pareto + operating points ----------------

def pareto_frontier(avg_cand: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """
    Return boolean mask for points on the Pareto frontier:
      maximize recall, minimize avg_cand.
    Simple scan after sorting by avg_cand.
    """
    order = np.argsort(avg_cand)
    best_recall = -1.0
    is_front = np.zeros_like(avg_cand, dtype=bool)
    for idx in order:
        r = float(recall[idx])
        if r > best_recall + 1e-12:
            is_front[idx] = True
            best_recall = r
    return is_front


@dataclass
class OperatingPoints:
    conservative_idx: int
    balanced_idx: int
    discovery_idx: int


def choose_operating_points(
    stats: ThresholdStats,
    # Heuristics (can be overridden via CLI)
    alpha_conservative: float = 0.60,   # keep at least alpha * max_recall
    avg_limit_discovery: float = 15.0,  # avoid too many candidates on average
    p95_limit_discovery: float = 30.0,  # avoid extreme chains
) -> OperatingPoints:
    thr = stats.threshold
    recall = stats.recall_val
    avg_c = stats.avg_cand_val
    p95 = stats.p95_cand_val

    max_r = float(np.max(recall))
    target_r = alpha_conservative * max_r

    # Conservative: minimal avg_c among those with recall >= alpha * max_recall
    mask_cons = recall >= target_r
    if np.any(mask_cons):
        idxs = np.where(mask_cons)[0]
        # choose smallest avg, tie-breaker: higher recall, then higher threshold (more strict)
        idx = idxs[np.lexsort((-thr[idxs], -recall[idxs], avg_c[idxs]))][0]
        conservative_idx = int(idx)
    else:
        # fallback: smallest avg overall
        conservative_idx = int(np.argmin(avg_c))

    # Balanced: maximize recall / log1p(avg_c) with mild penalty on p95
    score = recall / np.log1p(avg_c + 1e-6) - 0.005 * p95
    balanced_idx = int(np.argmax(score))

    # Discovery: maximize recall under reasonable avg & p95 constraints
    mask_disc = (avg_c <= avg_limit_discovery) & (p95 <= p95_limit_discovery)
    if np.any(mask_disc):
        idxs = np.where(mask_disc)[0]
        discovery_idx = int(idxs[np.argmax(recall[idxs])])
    else:
        # fallback: just maximize recall
        discovery_idx = int(np.argmax(recall))

    return OperatingPoints(
        conservative_idx=conservative_idx,
        balanced_idx=balanced_idx,
        discovery_idx=discovery_idx,
    )


# ---------------- Plotting ----------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_pareto(stats: ThresholdStats, ops: OperatingPoints, out_path: str) -> None:
    avg_c = stats.avg_cand_val
    recall = stats.recall_val
    front = pareto_frontier(avg_c, recall)

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_c, recall, s=10, alpha=0.4, label="All thresholds")
    plt.scatter(avg_c[front], recall[front], s=18, alpha=0.9, label="Pareto frontier")

    for name, idx in [
        ("conservative", ops.conservative_idx),
        ("balanced", ops.balanced_idx),
        ("discovery", ops.discovery_idx),
    ]:
        plt.scatter([avg_c[idx]], [recall[idx]], s=80, label=f"{name}: τ={stats.threshold[idx]:.3f}")

    plt.xlabel("Avg candidates per chain (validation)")
    plt.ylabel("Recall (validation, weighted)")
    plt.title("Recall vs Candidate Load (validation) with Pareto frontier")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=1000, format="png")
    plt.close()


def plot_curves(stats: ThresholdStats, ops: OperatingPoints, b_to_show: int, out_path: str) -> None:
    t = stats.threshold
    plt.figure(figsize=(10, 6))

    plt.plot(t, stats.recall_val, label="Recall_val (weighted)")
    plt.plot(t, stats.precision_val, label="Precision_val (weighted)")
    plt.plot(t, stats.avg_cand_val, label="AvgCand_val (per-chain)")
    plt.plot(t, stats.p95_cand_val, label="P95Cand_val (per-chain)")
    if b_to_show in stats.frac_over_b_val:
        plt.plot(t, stats.frac_over_b_val[b_to_show], label=f"Frac(chains > {b_to_show})")

    for name, idx in [
        ("conservative", ops.conservative_idx),
        ("balanced", ops.balanced_idx),
        ("discovery", ops.discovery_idx),
    ]:
        plt.axvline(float(t[idx]), linestyle=":", label=f"{name} τ={t[idx]:.3f}")

    plt.xlabel("Threshold τ")
    plt.ylabel("Value")
    plt.title("Validation curves vs threshold")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=1000, format="png")
    plt.close()


# ---------------- Main ----------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--n_thresholds", type=int, default=400,
                        help="Number of quantile thresholds to evaluate.")
    parser.add_argument("--b_list", type=str, default="3,8,15,30",
                        help="Budgets B for frac(chains > B). Comma-separated.")

    # Operating-point heuristics
    parser.add_argument("--alpha_conservative", type=float, default=0.60)
    parser.add_argument("--avg_limit_discovery", type=float, default=15.0)
    parser.add_argument("--p95_limit_discovery", type=float, default=30.0)
    parser.add_argument("--b_show", type=int, default=8,
                        help="Which B to show on curve plot (Frac(chains>B)).")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    val_path = find_existing(args.input_dir, ["validation.pkl", "validation_results.pkl"])
    print(f"[use] validation: {val_path}")

    y_val_chain, p_val_chain, w_val_chain = load_results_per_chain(val_path)
    y_val_flat, p_val_flat, w_val_res = flatten_with_residue_weights(y_val_chain, p_val_chain, w_val_chain)

    p = p_val_flat  # как в скрипте
    print("min/max:", float(p.min()), float(p.max()))
    print("q50/q90/q99:", np.quantile(p, [0.5, 0.9, 0.99]))
    print("frac<0:", float(np.mean(p < 0)))
    print("frac>1:", float(np.mean(p > 1)))
    


    thresholds = make_threshold_grid_from_quantiles(p_val_flat, n=args.n_thresholds)

    anchors = np.array([0.35, 0.50, 0.65, 0.26, 0.0612], dtype=np.float32)
    thresholds = np.unique(np.concatenate([thresholds, anchors]))
    thresholds = np.sort(thresholds).astype(np.float32)
    print("closest_to_0.35:", thresholds[np.argmin(np.abs(thresholds - 0.35))])


    b_list = [int(x.strip()) for x in args.b_list.split(",") if x.strip()]
    print(f"[info] thresholds: {len(thresholds)} | B list: {b_list}")

    stats = eval_thresholds_on_validation(
        y_val_flat=y_val_flat,
        p_val_flat=p_val_flat,
        w_val_res=w_val_res,
        preds_val_per_chain=p_val_chain,
        w_val_chain=w_val_chain,
        thresholds=thresholds,
        b_list=b_list,
    )

    ops = choose_operating_points(
        stats,
        alpha_conservative=args.alpha_conservative,
        avg_limit_discovery=args.avg_limit_discovery,
        p95_limit_discovery=args.p95_limit_discovery,
    )

    def pick_tau_by_budget(stats, avg_max=None, p95_max=None, frac_over=None):
        mask = np.ones_like(stats.threshold, dtype=bool)
        if avg_max is not None:
            mask &= stats.avg_cand_val <= avg_max
        if p95_max is not None:
            mask &= stats.p95_cand_val <= p95_max
        if frac_over is not None:
            # frac_over = (B, max_frac)
            B, max_frac = frac_over
            mask &= stats.frac_over_b_val[B] <= max_frac

        if not mask.any():
            return None

        idxs = np.where(mask)[0]
        # max recall, tie-break: smaller avg candidates, then larger threshold (stricter)
        best = idxs[np.lexsort((-stats.threshold[idxs], stats.avg_cand_val[idxs], -stats.recall_val[idxs]))][0]
        return int(best)

    budgets = [
        ("strict",   dict(avg_max=3,  p95_max=12, frac_over=(8, 0.10))),
        ("discover", dict(avg_max=8,  p95_max=25, frac_over=(8, 0.30))),
        ("wide",     dict(avg_max=15, p95_max=35, frac_over=(15, 0.30))),
    ]

    for name, kw in budgets:
        idx = pick_tau_by_budget(stats, **kw)
        if idx is None:
            print(f"[{name}] no threshold satisfies {kw}")
            continue
        print(
            f"[{name}] τ={stats.threshold[idx]:.4f} | "
            f"rec={stats.recall_val[idx]:.3f} prec={stats.precision_val[idx]:.3f} | "
            f"avg={stats.avg_cand_val[idx]:.2f} p95={stats.p95_cand_val[idx]:.1f} | "
            f"frac(>8)={stats.frac_over_b_val[8][idx]:.3f}"
        )


    def report(name: str, idx: int) -> None:
        print(
            f"[{name}] τ={stats.threshold[idx]:.4f} | "
            f"rec={stats.recall_val[idx]:.3f} prec={stats.precision_val[idx]:.3f} | "
            f"avgCand={stats.avg_cand_val[idx]:.2f} p95Cand={stats.p95_cand_val[idx]:.1f}"
        )
        for b in b_list:
            print(f"    frac(chains > {b}) = {stats.frac_over_b_val[b][idx]:.3f}")

    report("conservative", ops.conservative_idx)
    report("balanced", ops.balanced_idx)
    report("discovery", ops.discovery_idx)

    out_pareto = os.path.join(args.output_dir, "pareto_recall_vs_candidates.png")
    out_curves = os.path.join(args.output_dir, "validation_curves_vs_threshold.png")

    #plot_pareto(stats, ops, out_pareto)
    #plot_curves(stats, ops, b_to_show=args.b_show, out_path=out_curves)

    # 1) Recall/precision по residues (как у тебя)
    recall_val = []
    precision_val = []
    for t in thresholds:
        y_pred = (p_val_flat >= t).astype(np.int32)
        precision_val.append(precision_score(y_val_flat, y_pred, sample_weight=w_val_res, zero_division=0))
        recall_val.append(recall_score(y_val_flat, y_pred, sample_weight=w_val_res))
    recall_val = np.asarray(recall_val, dtype=np.float32)
    precision_val = np.asarray(precision_val, dtype=np.float32)

    out_val = os.path.join(args.output_dir, "validation_precision_recall_fdr_vs_tau.png")
    plot_val_precision_recall_fdr_vs_tau(thresholds, precision_val, recall_val, out_val, chosen_tau=0.35)
    print(f"[OK] saved -> {out_val}")

    out_pr_tau = os.path.join(args.output_dir, "validation_precision_recall_vs_tau.png")
    plot_val_precision_recall_vs_tau(thresholds, precision_val, recall_val, out_pr_tau, chosen_tau=0.35)
    print(f"[OK] saved -> {out_pr_tau}")



    # 2) FP-нагрузка по белкам (RAW и weighted)
    fp_stats = eval_fp_load_vs_thresholds(y_val_chain, p_val_chain, w_val_chain, thresholds)

    out_fp = os.path.join(args.output_dir, "recall_vs_fp_load.png")
    #plot_recall_vs_fp_load(thresholds, recall_val, fp_stats, out_fp, zoom=False)

    out_fp_zoomed = os.path.join(args.output_dir, "recall_vs_fp_load_zoom10.png")
    #plot_recall_vs_fp_load(thresholds, recall_val, fp_stats, out_fp_zoomed, zoom=True)

    print(f"[OK] saved -> {out_fp}")


    # 3) Выбираем τ: max recall при ограничении FP-нагрузки
    # (подкрути бюджеты под себя)
    budgets = [
        ("strict_fp",   dict(avg_fp_max=1.0, p95_fp_max=5.0)),
        ("discover_fp", dict(avg_fp_max=2.0, p95_fp_max=10.0)),
    ]

    for name, cfg in budgets:
        idx = pick_tau_max_recall_under_fp_budget(
            thresholds, recall_val, fp_stats,
            avg_fp_max=cfg["avg_fp_max"], p95_fp_max=cfg["p95_fp_max"],
            use_weighted=False  # я бы для “мусора” смотрел RAW
        )
        if idx is None:
            print(f"[{name}] no threshold satisfies avg_fp<={cfg['avg_fp_max']} and p95_fp<={cfg['p95_fp_max']}")
            continue

        t = float(thresholds[idx])
        print(
            f"[{name}] τ={t:.4f} | rec={recall_val[idx]:.3f} prec={precision_val[idx]:.3f} | "
            f"avgFP(raw)={fp_stats['avg_fp_raw'][idx]:.2f} p95FP(raw)={fp_stats['p95_fp_raw'][idx]:.1f} "
            f"fracFP>=1(raw)={fp_stats['frac_fp_ge1_raw'][idx]:.3f} | "
            f"avgPred(raw)={fp_stats['avg_pred_raw'][idx]:.2f} p95Pred(raw)={fp_stats['p95_pred_raw'][idx]:.1f}"
        )


    # пример: проверяем 0.65 и рядом
    for thr in [0.85, 0.65, 0.50, 0.35, 0.261, 0.2, 0.1, 0.05]:
        eval_one_threshold(thr, y_val_flat, p_val_flat, w_val_res, y_val_chain, p_val_chain, w_val_chain)


    print(f"[OK] saved -> {out_pareto}")
    print(f"[OK] saved -> {out_curves}")


if __name__ == "__main__":
    main()
