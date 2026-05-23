#!/usr/bin/env python3
"""
stratify_utils.py — shared utilities for all stratification scripts.
"""
import glob
import logging
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

log = logging.getLogger(__name__)

# ── figure style ──────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 14, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "axes.titlesize": 16,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "lines.linewidth": 0.9,
    "svg.fonttype": "path",
})

W_FULL  = 6.27
GUTTER  = 0.15
W_HALF  = (W_FULL - GUTTER) / 2
W_THIRD = (W_FULL - GUTTER) / 3
H_SMALL = 1.9
H_WIDE  = 2.7


# ── id helpers ────────────────────────────────────────────────────────────────

def normalize_id(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(t, str) for t in x):
            return f"{x[0]}_{x[1]}"
        return str(x[0])
    return str(x)


def is_pdb_identifier(s: str) -> bool:
    return len(s) == 4 and s.isalnum()


def parse_model_source(seq_id: str) -> str:
    base = seq_id.split("_", 1)[0]
    return "PDB" if is_pdb_identifier(base) else "AF"


def parse_ec_top(ec_number: Any) -> Optional[str]:
    if ec_number is None or (isinstance(ec_number, float) and np.isnan(ec_number)):
        return None
    m = re.match(r"^(\d+)", str(ec_number).strip())
    return str(int(m.group(1))) if m else None


def ensure_1d_preds(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.ndim > 1:
        p = p[..., -1]
    return p.reshape(-1)


def ensure_1d_labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {y.shape}")
    return y


# ── pkl loading ───────────────────────────────────────────────────────────────

def load_results_pkl(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    for k in ("labels", "predictions", "weights", "ids"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    result: Dict[str, Dict[str, Any]] = {}
    for sid, labels, preds, w in zip(
        data["ids"], data["labels"], data["predictions"], data["weights"]
    ):
        sid = normalize_id(sid)
        result[sid] = {
            "labels": ensure_1d_labels(labels),
            "preds":  ensure_1d_preds(preds),
            "weight": float(w),
        }
    return result


def load_results_pkls(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load and average predictions across multiple pkl files (one per fold)."""
    accum: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for sid, rec in load_results_pkl(path).items():
            if sid not in accum:
                accum[sid] = {
                    "labels": rec["labels"],
                    "preds_list": [rec["preds"]],
                    "weight": rec["weight"],
                }
            else:
                accum[sid]["preds_list"].append(rec["preds"])
        log.info("loaded %s", path)

    merged = {
        sid: {
            "labels": rec["labels"],
            "preds":  np.mean(rec["preds_list"], axis=0).astype(np.float32),
            "weight": rec["weight"],
        }
        for sid, rec in accum.items()
    }
    log.info("total chains: %d", len(merged))
    return merged


# ── metrics ───────────────────────────────────────────────────────────────────

def aucpr_like_make_PR_curve(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
) -> float:
    y_list, p_list, w_list = [], [], []
    for lbl, pred, w in zip(labels, predictions, weights):
        lbl  = ensure_1d_labels(lbl)
        pred = ensure_1d_preds(pred)
        L = min(len(lbl), len(pred))
        if L == 0:
            continue
        y_list.append(lbl[:L])
        p_list.append(pred[:L])
        w_list.append(np.full(L, float(w)))

    if not y_list:
        return 0.0

    y_all = np.concatenate(y_list)
    p_all = np.concatenate(p_list)
    w_all = np.concatenate(w_list)

    bad = np.isnan(p_all) | np.isnan(y_all) | np.isinf(p_all) | np.isinf(y_all)
    if bad.any():
        good_mask = ~bad
        fill = float(np.nanmedian(p_all[good_mask])) if good_mask.any() else 0.0
        p_all = p_all.copy()
        p_all[bad] = fill

    good = ~bad
    if not good.any() or y_all[good].sum() == 0:
        return 0.0

    prec, rec, _ = precision_recall_curve(y_all[good], p_all[good], sample_weight=w_all[good])
    return float(auc(rec, prec))


def bootstrap_ci_aucpr_group(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
    n_boot: int = 200,
    seed:   int = 0,
) -> Tuple[float, float]:
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return np.nan, np.nan
    rng  = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rng.choice(n, size=n, replace=True)
        vals[b] = aucpr_like_make_PR_curve(
            [labels[i] for i in s], [predictions[i] for i in s],
            np.asarray(weights)[s].tolist(),
        )
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def aucroc_chain_weighted(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
) -> float:
    y_list, p_list, w_list = [], [], []
    for lbl, pred, w in zip(labels, predictions, weights):
        lbl  = ensure_1d_labels(lbl)
        pred = ensure_1d_preds(pred)
        L = min(len(lbl), len(pred))
        if L == 0:
            continue
        y_list.append(lbl[:L])
        p_list.append(pred[:L])
        w_list.append(np.full(L, float(w)))

    if not y_list:
        return np.nan

    y_all = np.concatenate(y_list)
    p_all = np.concatenate(p_list)
    w_all = np.concatenate(w_list)

    good = ~(np.isnan(p_all) | np.isnan(y_all) | np.isinf(p_all) | np.isinf(y_all))
    if not good.any():
        return np.nan

    y_g = y_all[good].astype(int)
    if y_g.sum() == 0 or y_g.sum() == len(y_g):
        return np.nan

    return float(roc_auc_score(y_g, p_all[good], sample_weight=w_all[good]))


def bootstrap_ci_aucroc_group(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
    n_boot: int = 200,
    seed:   int = 0,
) -> Tuple[float, float]:
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return np.nan, np.nan
    rng  = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rng.choice(n, size=n, replace=True)
        vals[b] = aucroc_chain_weighted(
            [labels[i] for i in s], [predictions[i] for i in s],
            np.asarray(weights)[s].tolist(),
        )
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def max_f1_recall_chain_weighted(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
) -> Tuple[float, float]:
    """
    Chain-weighted max F1 and the corresponding Recall.
    Uses the same flattening as aucpr_like_make_PR_curve.
    Returns (max_F1, Recall_at_max_F1).
    """
    y_list, p_list, w_list = [], [], []
    for lbl, pred, w in zip(labels, predictions, weights):
        lbl  = ensure_1d_labels(lbl)
        pred = ensure_1d_preds(pred)
        L = min(len(lbl), len(pred))
        if L == 0:
            continue
        y_list.append(lbl[:L])
        p_list.append(pred[:L])
        w_list.append(np.full(L, float(w)))

    if not y_list:
        return np.nan, np.nan

    y_all = np.concatenate(y_list)
    p_all = np.concatenate(p_list)
    w_all = np.concatenate(w_list)

    bad = np.isnan(p_all) | np.isnan(y_all) | np.isinf(p_all) | np.isinf(y_all)
    if bad.any():
        good_mask = ~bad
        fill = float(np.nanmedian(p_all[good_mask])) if good_mask.any() else 0.0
        p_all = p_all.copy()
        p_all[bad] = fill

    good = ~bad
    if not good.any() or y_all[good].sum() == 0:
        return np.nan, np.nan

    prec, rec, _ = precision_recall_curve(
        y_all[good].astype(int), p_all[good], sample_weight=w_all[good]
    )
    denom = prec + rec
    f1 = np.zeros_like(prec)
    mask = denom > 0
    f1[mask] = 2 * prec[mask] * rec[mask] / denom[mask]
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(rec[idx])


def bootstrap_ci_f1_group(
    labels:      List[np.ndarray],
    predictions: List[np.ndarray],
    weights:     List[float],
    n_boot: int = 200,
    seed:   int = 0,
) -> Tuple[float, float]:
    n = len(labels)
    if n == 0 or n_boot <= 0:
        return np.nan, np.nan
    rng  = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rng.choice(n, size=n, replace=True)
        f1, _ = max_f1_recall_chain_weighted(
            [labels[i] for i in s], [predictions[i] for i in s],
            np.asarray(weights)[s].tolist(),
        )
        vals[b] = f1
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def compute_group_aucpr(
    df:        pd.DataFrame,
    group_col: str,
    groups:    Optional[List] = None,
    n_boot:    int = 200,
    seed:      int = 0,
) -> pd.DataFrame:
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    valid = valid[valid[group_col].notna()]
    valid["y_true"] = valid["y_true"].astype(int)

    if groups is None:
        groups = sorted(valid[group_col].dropna().unique(), key=str)

    rows = []
    for g in groups:
        sub = valid[valid[group_col].astype(str) == str(g)]
        if len(sub) == 0:
            rows.append({
                group_col: g,
                "AUCPR": np.nan, "AUCPR_ci_lo": np.nan, "AUCPR_ci_hi": np.nan,
                "AUCPR_norm": np.nan,
                "AUCROC": np.nan, "AUCROC_ci_lo": np.nan, "AUCROC_ci_hi": np.nan,
                "max_F1": np.nan, "F1_ci_lo": np.nan, "F1_ci_hi": np.nan,
                "Recall_at_F1": np.nan,
                "prevalence": np.nan, "n_chains": 0, "n_residues": 0, "n_positive": 0,
            })
            continue

        labels_list, preds_list, weights_list = [], [], []
        for sid, grp in sub.groupby("Sequence_ID"):
            w = float(grp["chain_weight"].iloc[0])
            labels_list.append(grp["y_true"].to_numpy(dtype=int))
            preds_list.append(grp["y_pred"].to_numpy(dtype=float))
            weights_list.append(1.0 if np.isnan(w) else w)

        au_pr  = aucpr_like_make_PR_curve(labels_list, preds_list, weights_list)
        lo_pr, hi_pr = bootstrap_ci_aucpr_group(
            labels_list, preds_list, weights_list, n_boot=n_boot, seed=seed)
        au_roc = aucroc_chain_weighted(labels_list, preds_list, weights_list)
        lo_roc, hi_roc = bootstrap_ci_aucroc_group(
            labels_list, preds_list, weights_list, n_boot=n_boot, seed=seed)
        max_f1, recall_f1 = max_f1_recall_chain_weighted(labels_list, preds_list, weights_list)
        lo_f1, hi_f1 = bootstrap_ci_f1_group(
            labels_list, preds_list, weights_list, n_boot=n_boot, seed=seed)

        prevalence = int(sub["y_true"].sum()) / len(sub)
        aucpr_norm = (au_pr - prevalence) / (1 - prevalence) if prevalence < 1 else np.nan
        rows.append({
            group_col:       g,
            "AUCPR":         au_pr,
            "AUCPR_ci_lo":   lo_pr,
            "AUCPR_ci_hi":   hi_pr,
            "AUCPR_norm":    aucpr_norm,
            "AUCROC":        au_roc,
            "AUCROC_ci_lo":  lo_roc,
            "AUCROC_ci_hi":  hi_roc,
            "max_F1":        max_f1,
            "F1_ci_lo":      lo_f1,
            "F1_ci_hi":      hi_f1,
            "Recall_at_F1":  recall_f1,
            "prevalence":    prevalence,
            "n_chains":      len(labels_list),
            "n_residues":    len(sub),
            "n_positive":    int(sub["y_true"].sum()),
        })

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def barplot_with_ci(
    values:   List[float],
    labels:   List[str],
    ci_lo:    List[float],
    ci_hi:    List[float],
    ylabel:   str,
    title:    str,
    out_path: str,
    ylim:     Optional[Tuple[float, float]] = (0.0, 1.0),
    color:    str = "#7BC8F6",
) -> None:
    fig, ax = plt.subplots(figsize=(W_FULL, H_WIDE))
    x  = np.arange(len(values))
    v  = np.asarray(values, float)
    lo = np.asarray(ci_lo,  float)
    hi = np.asarray(ci_hi,  float)

    ax.bar(x, v, color=color, edgecolor="black")

    bad = np.isnan(lo) | np.isnan(hi)
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[bad], hi2[bad] = v[bad], v[bad]
    ax.errorbar(x, v, yerr=np.vstack([v - lo2, hi2 - v]),
                fmt="none", capsize=3, elinewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def stratify_and_plot(
    df:        pd.DataFrame,
    group_col: str,
    out_dir:   str,
    tag:       str,
    title:     str,
    groups:    Optional[List] = None,
    xlabels:   Optional[List[str]] = None,
    n_boot:    int = 200,
    seed:      int = 0,
) -> pd.DataFrame:
    if group_col not in df.columns:
        log.warning("column '%s' not in table — skipping %s", group_col, tag)
        return pd.DataFrame()

    metrics = compute_group_aucpr(df, group_col, groups=groups, n_boot=n_boot, seed=seed)
    metrics.to_csv(os.path.join(out_dir, f"metrics_by_{tag}.csv"), index=False)

    lbl = xlabels if xlabels else [str(g) for g in metrics[group_col]]
    barplot_with_ci(
        values=metrics["AUCPR"].fillna(0).tolist(), labels=lbl,
        ci_lo=metrics["AUCPR_ci_lo"].tolist(), ci_hi=metrics["AUCPR_ci_hi"].tolist(),
        ylabel="AUCPR", title=f"{title} — AUCPR",
        out_path=os.path.join(out_dir, f"aucpr_by_{tag}.png"),
    )
    barplot_with_ci(
        values=metrics["AUCPR_norm"].fillna(0).tolist(), labels=lbl,
        ci_lo=[np.nan] * len(lbl), ci_hi=[np.nan] * len(lbl),
        ylabel="AUCPR (normalised)", title=f"{title} — AUCPR norm",
        out_path=os.path.join(out_dir, f"aucpr_norm_by_{tag}.png"),
        ylim=(-0.1, 1.0),
    )
    barplot_with_ci(
        values=metrics["AUCROC"].fillna(0.5).tolist(), labels=lbl,
        ci_lo=metrics["AUCROC_ci_lo"].tolist(), ci_hi=metrics["AUCROC_ci_hi"].tolist(),
        ylabel="AUC-ROC", title=f"{title} — AUC-ROC",
        out_path=os.path.join(out_dir, f"aucroc_by_{tag}.png"),
    )
    barplot_with_ci(
        values=metrics["max_F1"].fillna(0).tolist(), labels=lbl,
        ci_lo=metrics["F1_ci_lo"].tolist(), ci_hi=metrics["F1_ci_hi"].tolist(),
        ylabel="max F1", title=f"{title} — max F1",
        out_path=os.path.join(out_dir, f"f1_by_{tag}.png"),
    )
    barplot_with_ci(
        values=metrics["Recall_at_F1"].fillna(0).tolist(), labels=lbl,
        ci_lo=[np.nan] * len(lbl), ci_hi=[np.nan] * len(lbl),
        ylabel="Recall @ max F1", title=f"{title} — Recall @ max F1",
        out_path=os.path.join(out_dir, f"recall_by_{tag}.png"),
    )
    log.info("%s: %d groups → %s", tag, len(metrics), out_dir)
    return metrics


# ── split txt ─────────────────────────────────────────────────────────────────

def parse_split_txt(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse split*.txt → {seq_id: [{chain, resnum, aa, label}, ...]}."""
    data: Dict[str, List[Dict[str, Any]]] = {}
    cur_id: Optional[str] = None
    cur_rows: List[Dict[str, Any]] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    data[cur_id] = cur_rows
                cur_id   = normalize_id(line[1:].strip())
                cur_rows = []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    chain, resnum, aa, label = parts[:4]
                    cur_rows.append({"chain": chain, "resnum": str(resnum),
                                     "aa": aa, "label": int(label)})

    if cur_id is not None:
        data[cur_id] = cur_rows
    return data


def build_split_map(split_txts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in split_txts:
        for k, v in parse_split_txt(path).items():
            if k in out:
                raise ValueError(f"Duplicate Sequence_ID across split txts: {k}")
            out[k] = v
    return out


# ── dataset csv ───────────────────────────────────────────────────────────────

def load_dataset_csv(path: str) -> pd.DataFrame:
    """Load dataset.csv, normalise Sequence_ID, return full frame."""
    ds = pd.read_csv(path, low_memory=False)
    ds["Sequence_ID"] = ds["Sequence_ID"].apply(normalize_id)
    return ds


# ── structure file resolution ─────────────────────────────────────────────────

def resolve_structure_file(seq_id: str, structure_dirs: List[str]) -> Optional[str]:
    """Return the first structure file found for seq_id (direct names or AFDB pattern)."""
    base = seq_id.split("_", 1)[0]
    candidates = (
        f"{seq_id}.cif", f"{seq_id}.pdb", f"{seq_id}.mmcif",
        f"{base}.cif",   f"{base}.pdb",   f"{base}.mmcif",
    )
    for sd in structure_dirs:
        for fname in candidates:
            p = os.path.join(sd, fname)
            if os.path.exists(p):
                return p
        # AFDB naming: AF-{UniProtID}-F*-model_v*.cif/.pdb
        for ext in ("cif", "pdb"):
            hits = sorted(glob.glob(os.path.join(sd, f"AF-{base}-F*-model_v*.{ext}")))
            if hits:
                return hits[0]
    return None


def resolve_all_structure_files(seq_id: str, structure_dirs: List[str]) -> List[str]:
    """Like resolve_structure_file but returns ALL fragment files (needed for AFDB F1/F2/…)."""
    base = seq_id.split("_", 1)[0]
    candidates = (
        f"{seq_id}.cif", f"{seq_id}.pdb", f"{seq_id}.mmcif",
        f"{base}.cif",   f"{base}.pdb",   f"{base}.mmcif",
    )
    for sd in structure_dirs:
        for fname in candidates:
            p = os.path.join(sd, fname)
            if os.path.exists(p):
                return [p]
        for ext in ("cif", "pdb"):
            hits = sorted(glob.glob(os.path.join(sd, f"AF-{base}-F*-model_v*.{ext}")))
            if hits:
                return hits
    return []
