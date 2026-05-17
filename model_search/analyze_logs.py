#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust SLURM log summary for ESM2 sweeps.

Metric computation follows the 'seeds' approach in evaluate_results.py:
  config + seed -> require exactly N folds -> concatenate per-fold labels/predictions/weights
  from result pickles -> compute AUCPR once -> average AUCPR across seeds.

When --results_base_dir is provided, run directories are discovered by scanning that
directory for completed runs (dirs with subset pickles) — no SLURM logs needed for metrics.
SLURM logs are still parsed for status/error reporting in --out_csv.
"""

import argparse
import glob
import os
import pickle
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


SUMMARY_RE = re.compile(r"\[SUMMARY\]\s+(\w+)\s+AUCPR=([0-9]*\.?[0-9]+)")
DONE_RE = re.compile(r"--- DONE in ([0-9]*\.?[0-9]+) h ---")
RUN_ROW_RE = re.compile(r"\[INFO\]\s+Running row\s+(\d+)/(\d+)")
CSV_PATH_RE = re.compile(r"\[INFO\]\s+CSV_PATH=(\S+)")
SKIP_RE = re.compile(r"\[SKIP\]")
SKIP_PATH_RE = re.compile(r"\[SKIP\]\s+Existing run found:\s*(\S+)")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
CMD_RE = re.compile(r"^\[CMD\]\s+(.*)$", re.MULTILINE)
OUT_DIR_RE = re.compile(r"\[INFO\]\s+OUT_DIR=(\S+)")

REAL_ERROR_RE = re.compile(
    r"(?i)\b("
    r"traceback|runtimeerror|valueerror|assertionerror|keyerror|indexerror|typeerror|"
    r"exception|slurmstepd: error|segmentation fault|killed|out of memory|oom|"
    r"cuda error|failed"
    r")\b"
)

REQUIRED_KEYS = ["labels", "predictions", "weights", "ids", "splits"]

# Hyperparameter columns defining one config. Do not include fold or seed here.
GROUP_COLS = [
    "esm2_version",
    "esm2_layer",
    "architecture",
    "hidden_dims",
    "dropout",
    "head_norm",
    "activation",
    "learning_rate",
    "weight_decay",
    "batch_size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SLURM logs and compute merged-fold AUCPR from result pickles "
            "(seeds approach: merge folds per seed, then average across seeds)."
        )
    )
    parser.add_argument("--logs_dir", default=None,
                        help="Directory with *.out and *.err logs (optional; used for status/error CSV only).")
    parser.add_argument("--results_base_dir", default=None,
                        help="Base dir containing run result folders (scanned directly for metrics).")
    parser.add_argument("--runs_csv", default=None, help="Optional runs.csv used to fill params from row_idx.")
    parser.add_argument("--out_csv", default="log_summary.csv")
    parser.add_argument("--version_csv", default="log_version_merged_summary.csv")
    parser.add_argument("--grouped_csv", default="log_grouped_summary.csv")
    parser.add_argument("--version_col", default="seed", help="Usually 'seed' or 'version'.")
    parser.add_argument("--subsets", nargs="+", default=["validation", "test"])
    parser.add_argument("--expected_n_folds", type=int, default=5)
    parser.add_argument(
        "--ignore_skips_for_metrics",
        action="store_true",
        help="Do not use [SKIP] rows even to locate pickles (log-based fallback only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_job_and_task(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1]
    return stem, ""


def normalize_hidden_dims(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    parts = re.split(r"[\s,\-]+", text)
    parts = [p for p in parts if p]
    return "-".join(parts)


def normalize_scalar(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def parse_cmd_args(out_text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    match = CMD_RE.search(out_text)
    if not match:
        return result

    cmd = match.group(1).strip()
    result["cmd"] = cmd

    try:
        tokens = shlex.split(cmd)
    except Exception:
        return result

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue

        key = tok[2:]
        if key == "hidden_dims":
            vals = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                vals.append(tokens[j])
                j += 1
            result[key] = normalize_hidden_dims(" ".join(vals))
            i = j
            continue

        if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
            result[key] = True
            i += 1
            continue

        result[key] = tokens[i + 1]
        i += 2

    return result


def parse_run_dir_name(path: str) -> Dict[str, Any]:
    """Best-effort parser for names like ...__esm2_t36_3B_UR50D__layer36__lr=0.0001__..."""
    out: Dict[str, Any] = {}
    base = os.path.basename(str(path).rstrip("/"))
    if not base:
        return out

    parts = base.split("__")
    if parts:
        out["model_name"] = parts[0]

    for part in parts[1:]:
        if part.startswith("esm2_"):
            out["esm2_version"] = part
        elif part.startswith("layer") and part[len("layer"):].isdigit():
            out["esm2_layer"] = part[len("layer"):]
        elif part.startswith("arch="):
            out["architecture"] = part.split("=", 1)[1]
        elif part.startswith("hd="):
            out["hidden_dims"] = normalize_hidden_dims(part.split("=", 1)[1])
        elif part.startswith("drop="):
            out["dropout"] = part.split("=", 1)[1]
        elif part.startswith("norm="):
            out["head_norm"] = part.split("=", 1)[1]
        elif part.startswith("act="):
            out["activation"] = part.split("=", 1)[1]
        elif part.startswith("lr="):
            out["learning_rate"] = part.split("=", 1)[1]
        elif part.startswith("wd="):
            out["weight_decay"] = part.split("=", 1)[1]
        elif part.startswith("bs="):
            out["batch_size"] = part.split("=", 1)[1]
        elif part.startswith("fold="):
            out["cv_fold"] = part.split("=", 1)[1]
        elif part.startswith("seed="):
            out["seed"] = part.split("=", 1)[1]
        elif part.startswith("version="):
            out["version"] = part.split("=", 1)[1]

    return out


def has_real_error_text(text: str) -> bool:
    if not text.strip():
        return False
    return any(REAL_ERROR_RE.search(line.strip()) for line in text.splitlines() if line.strip())


def extract_error_snippet(err_text: str, out_text: str, max_lines: int = 12) -> str:
    if has_real_error_text(err_text):
        lines = [line for line in err_text.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    lines = out_text.splitlines()
    hits = [i for i, line in enumerate(lines) if TRACEBACK_RE.search(line)]
    if hits:
        i = hits[-1]
        return "\n".join(lines[i:i + max_lines])
    return ""


# ---------------------------------------------------------------------------
# Log parsing (for status/error tracking, not metrics)
# ---------------------------------------------------------------------------

def empty_row(stem: str, out_path: str, err_path: str) -> Dict[str, Any]:
    job_id, task_id = parse_job_and_task(stem)
    return {
        "stem": stem,
        "job_id": job_id,
        "task_id": task_id,
        "out_path": out_path or "",
        "err_path": err_path or "",
        "status": "incomplete",
        "csv_path": "",
        "row_idx_1based": np.nan,
        "row_total": np.nan,
        "run_dir_from_log": "",
        "run_dir_resolved": "",
        "cmd": "",
        "run_name": "",
        "model_name": "",
        "esm2_dir": "",
        "esm2_version": "",
        "esm2_layer": "",
        "architecture": "",
        "hidden_dims": "",
        "dropout": "",
        "head_norm": "",
        "activation": "",
        "learning_rate": "",
        "weight_decay": "",
        "batch_size": "",
        "cv_fold": np.nan,
        "seed": np.nan,
        "version": np.nan,
        "train_aucpr": np.nan,
        "validation_aucpr": np.nan,
        "test_aucpr": np.nan,
        "done_hours": np.nan,
        "error_snippet": "",
    }


def fill_row_from_dict(row: Dict[str, Any], data: Dict[str, Any], overwrite: bool = False) -> None:
    mapping = {
        "run_name": "run_name",
        "model_name": "model_name",
        "esm2_dir": "esm2_dir",
        "esm2_version": "esm2_version",
        "esm2_layer": "esm2_layer",
        "architecture": "architecture",
        "hidden_dims": "hidden_dims",
        "dropout": "dropout",
        "head_norm": "head_norm",
        "activation": "activation",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "cv_fold": "cv_fold",
        "seed": "seed",
        "version": "version",
        "cmd": "cmd",
    }
    for dst, src in mapping.items():
        if src not in data:
            continue
        old = row.get(dst, "")
        if overwrite or old == "" or (isinstance(old, float) and np.isnan(old)):
            value = data[src]
            if dst == "hidden_dims":
                value = normalize_hidden_dims(value)
            row[dst] = value


def parse_log_pair(out_path: str, err_path: str) -> Dict[str, Any]:
    stem = Path(out_path).stem if out_path else Path(err_path).stem
    out_text = read_text(out_path)
    err_text = read_text(err_path)
    row = empty_row(stem, out_path, err_path)

    match = CSV_PATH_RE.search(out_text)
    if match:
        row["csv_path"] = match.group(1)

    match = RUN_ROW_RE.search(out_text)
    if match:
        row["row_idx_1based"] = int(match.group(1))
        row["row_total"] = int(match.group(2))

    match = SKIP_PATH_RE.search(out_text)
    if match:
        row["run_dir_from_log"] = match.group(1)
        fill_row_from_dict(row, parse_run_dir_name(row["run_dir_from_log"]), overwrite=False)

    # Also pick up OUT_DIR from train_ablated2.py style
    match = OUT_DIR_RE.search(out_text)
    if match and not row["run_dir_from_log"]:
        row["run_dir_from_log"] = match.group(1)
        fill_row_from_dict(row, parse_run_dir_name(row["run_dir_from_log"]), overwrite=False)

    cmd_args = parse_cmd_args(out_text)
    fill_row_from_dict(row, cmd_args, overwrite=True)

    for subset, value in SUMMARY_RE.findall(out_text):
        key = f"{subset.lower()}_aucpr"
        if key in row:
            row[key] = float(value)

    match = DONE_RE.search(out_text)
    if match:
        row["done_hours"] = float(match.group(1))

    has_skip = bool(SKIP_RE.search(out_text))
    has_done = bool(DONE_RE.search(out_text))
    has_summary = not pd.isna(row["validation_aucpr"]) or not pd.isna(row["test_aucpr"])
    has_traceback = bool(TRACEBACK_RE.search(out_text)) or bool(TRACEBACK_RE.search(err_text))
    has_real_err = has_real_error_text(err_text)

    if has_skip:
        row["status"] = "skipped"
    elif has_done or has_summary:
        row["status"] = "failed" if has_traceback else "success"
    elif has_traceback or has_real_err:
        row["status"] = "failed"
    else:
        row["status"] = "incomplete"

    row["error_snippet"] = extract_error_snippet(err_text, out_text)
    return row


def load_runs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["row_idx_1based"] = np.arange(1, len(df) + 1)
    return df


def enrich_from_runs_csv(df: pd.DataFrame, runs_csv_arg: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df

    csv_paths: List[str] = []
    if runs_csv_arg:
        csv_paths.append(runs_csv_arg)
    csv_paths.extend([p for p in df.get("csv_path", pd.Series(dtype=str)).dropna().unique() if str(p)])

    seen = set()
    csv_paths = [p for p in csv_paths if not (p in seen or seen.add(p)) and os.path.exists(p)]
    if not csv_paths:
        return df

    out = df.copy()
    for csv_path in csv_paths:
        try:
            runs = load_runs_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] cannot read runs_csv {csv_path}: {exc}")
            continue

        if len(csv_paths) == 1:
            mask = out["row_idx_1based"].notna()
        else:
            mask = (out["csv_path"] == csv_path) & out["row_idx_1based"].notna()

        if not mask.any():
            continue

        right = runs.copy()
        merged = out.loc[mask].merge(right, on="row_idx_1based", how="left", suffixes=("", "__csv"))

        for col in [
            "run_name", "model_name", "esm2_dir", "esm2_version", "esm2_layer", "architecture",
            "hidden_dims", "dropout", "head_norm", "activation", "learning_rate", "weight_decay",
            "batch_size", "cv_fold", "seed", "version",
        ]:
            csv_col = f"{col}__csv"
            if csv_col not in merged.columns:
                continue
            original = merged[col] if col in merged.columns else pd.Series([np.nan] * len(merged))
            fill = original.isna() | (original.astype(str).str.strip() == "")
            merged[col] = original.where(~fill, merged[csv_col])

        out.loc[mask, merged.columns.intersection(out.columns)] = merged[merged.columns.intersection(out.columns)].values

    return out


def normalize_df_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in GROUP_COLS + ["run_name", "model_name", "esm2_dir"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].apply(normalize_scalar)
    out["hidden_dims"] = out["hidden_dims"].apply(normalize_hidden_dims)

    for col in ["row_idx_1based", "row_total", "esm2_layer", "cv_fold", "seed", "version"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["train_aucpr", "validation_aucpr", "test_aucpr", "done_hours"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def subset_pickle_path(run_dir: str, subset: str) -> Optional[str]:
    for filename in [f"{subset}_results.pkl", f"{subset}.pkl"]:
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            return path
    return None


def ensure_1d(arr: Any, name: str = "vector") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        # Take last column (positive-class probability for 2-class softmax/sigmoid)
        a = a[:, -1]
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D per chain, got shape {a.shape}")
    return a


def load_subset_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"Missing key {key!r} in {path}")
    n = len(data["labels"])
    if not (n == len(data["predictions"]) == len(data["weights"]) == len(data["ids"]) == len(data["splits"])):
        raise ValueError(f"{path}: length mismatch in result pickle")
    return data


def pr_curve_weighted(labels: List[np.ndarray], preds: List[np.ndarray], weights: Iterable[float]) -> float:
    """
    Compute AUCPR: flatten residues across all chains, repeating per-chain weight per residue.
    Identical to the approach in evaluate_results.py cmd_seeds.
    """
    y_list, p_list, w_list = [], [], []
    for y, p, w in zip(labels, preds, weights):
        y = ensure_1d(y, "labels").astype(np.float32)
        p = ensure_1d(p, "preds").astype(np.float32)
        length = min(len(y), len(p))
        if length <= 0:
            continue
        y = y[:length]
        p = p[:length]
        bad = ~np.isfinite(p) | ~np.isfinite(y)
        if bad.any():
            fill = np.nanmedian(p[~bad]) if (~bad).any() else 0.0
            p = p.copy()
            y = y.copy()
            p[bad] = fill
            y[bad] = 0.0
            w_eff = 0.0
        else:
            w_eff = float(w)
        y_list.append(y)
        p_list.append(p)
        w_list.append(np.full(length, w_eff, dtype=np.float32))

    if not y_list:
        return np.nan

    y_flat = (np.concatenate(y_list) > 0.5).astype(np.int32)
    p_flat = np.concatenate(p_list)
    w_flat = np.concatenate(w_list)

    if float(y_flat.sum()) == 0.0:
        return 0.0

    precision, recall, _ = precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)
    return float(auc(recall, precision))


def load_and_merge_pickles(rows: pd.DataFrame, subset: str) -> Tuple[float, int, int, str]:
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    weights: List[float] = []
    paths: List[str] = []

    for _, row in rows.iterrows():
        run_dir = str(row["run_dir_resolved"])
        pkl_path = subset_pickle_path(run_dir, subset)
        if pkl_path is None:
            continue
        data = load_subset_pkl(pkl_path)
        labels.extend([ensure_1d(x, "labels") for x in data["labels"]])
        preds.extend([ensure_1d(x, "preds") for x in data["predictions"]])
        weights.extend(list(map(float, data["weights"])))
        paths.append(pkl_path)

    aucpr = pr_curve_weighted(labels, preds, weights)
    return aucpr, len(paths), len(labels), "|".join(paths)


# ---------------------------------------------------------------------------
# Metric computation: scan results_base_dir directly (primary approach)
# ---------------------------------------------------------------------------

def build_version_merged_summary_from_results(args: argparse.Namespace) -> pd.DataFrame:
    """
    Scan results_base_dir for all dirs that contain completed result pickles.
    Parse dir names to extract config params. Group by config + seed.
    For each group with the required number of folds, merge pickles and compute
    one AUCPR (same algorithm as cmd_seeds in evaluate_results.py).
    Then build a DataFrame with one row per (config, seed).
    """
    results_base_dir = args.results_base_dir
    if not results_base_dir or not os.path.isdir(results_base_dir):
        print(f"[WARN] results_base_dir not found or not set: {results_base_dir}")
        return pd.DataFrame()

    # Discover completed run dirs
    dir_rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(results_base_dir)):
        d = os.path.join(results_base_dir, name)
        if not os.path.isdir(d):
            continue
        if not any(subset_pickle_path(d, s) is not None for s in args.subsets):
            continue
        info = parse_run_dir_name(d)
        info["run_dir_resolved"] = d
        dir_rows.append(info)

    if not dir_rows:
        print(f"[WARN] No completed run dirs found in {results_base_dir}")
        return pd.DataFrame()

    dirs_df = normalize_df_types(pd.DataFrame(dir_rows))
    print(f"[INFO] Found {len(dirs_df)} completed run dirs in {results_base_dir}")

    # Group by config + seed (like cmd_seeds)
    by_cols = [c for c in GROUP_COLS + [args.version_col] if c in dirs_df.columns]

    rows: List[Dict[str, Any]] = []
    for keys, part in dirs_df.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(by_cols, keys))
        out["n_rows_used_for_metrics"] = int(len(part))
        out["n_unique_run_dirs"] = int(part["run_dir_resolved"].nunique())

        n_folds = int(part["cv_fold"].nunique(dropna=True)) if "cv_fold" in part.columns else 0
        out["n_folds_for_version"] = n_folds
        out["folds_for_version"] = ",".join(
            map(str, sorted(part["cv_fold"].dropna().astype(int).unique()))
        ) if "cv_fold" in part.columns else ""
        out["run_dirs_resolved"] = "|".join(sorted(part["run_dir_resolved"].unique()))

        if n_folds != int(args.expected_n_folds):
            out["merge_status"] = f"missing_folds: expected {args.expected_n_folds}, found {n_folds}"
            rows.append(out)
            continue

        # Reject if the same fold maps to multiple run dirs (ambiguous merge)
        fold_dir_counts = part.groupby("cv_fold", dropna=False)["run_dir_resolved"].nunique()
        if (fold_dir_counts > 1).any():
            out["merge_status"] = "duplicate_run_dirs_for_same_fold"
            rows.append(out)
            continue

        out["merge_status"] = "ok"
        for subset in args.subsets:
            try:
                aucpr, n_pickles, n_chains, paths = load_and_merge_pickles(part, subset)
                out[f"merged_{subset}_aucpr"] = aucpr
                out[f"n_{subset}_pickles"] = n_pickles
                out[f"n_{subset}_chains_merged"] = n_chains
                out[f"{subset}_pickle_paths"] = paths
                if n_pickles != int(args.expected_n_folds):
                    out["merge_status"] = (
                        f"missing_{subset}_pickles: expected {args.expected_n_folds}, found {n_pickles}"
                    )
            except Exception as exc:
                out[f"merged_{subset}_aucpr"] = np.nan
                out[f"n_{subset}_pickles"] = 0
                out[f"n_{subset}_chains_merged"] = 0
                out[f"{subset}_pickle_paths"] = ""
                out["merge_status"] = f"error: {type(exc).__name__}: {exc}"

        rows.append(out)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Fallback: log-based metric path (used when results_base_dir is not given)
# ---------------------------------------------------------------------------

def resolve_run_dir(row: pd.Series, results_base_dir: Optional[str], subsets: Iterable[str]) -> str:
    candidates: List[str] = []
    for key in ["run_dir_from_log", "run_name", "model_name"]:
        val = str(row.get(key, "") or "").strip()
        if not val or val.lower() == "nan":
            continue
        candidates.append(val)
        if results_base_dir:
            candidates.append(os.path.join(results_base_dir, val))
            candidates.append(os.path.join(results_base_dir, os.path.basename(val.rstrip("/"))))

    seen = set()
    for cand in candidates:
        cand = os.path.expanduser(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if not os.path.isdir(cand):
            continue
        if any(subset_pickle_path(cand, subset) is not None for subset in subsets):
            return cand
    return ""


def choose_metric_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.ignore_skips_for_metrics:
        metric = df[df["status"] == "success"].copy()
    else:
        metric = df[df["status"].isin(["success", "skipped"])].copy()

    if metric.empty:
        return metric

    metric["run_dir_resolved"] = metric.apply(
        lambda row: resolve_run_dir(row, args.results_base_dir, args.subsets), axis=1
    )
    metric = metric[metric["run_dir_resolved"] != ""].copy()

    dedup_cols = GROUP_COLS + [args.version_col, "cv_fold", "run_dir_resolved"]
    existing = [col for col in dedup_cols if col in metric.columns]
    metric = metric.sort_values(by=["status", "stem"]).drop_duplicates(subset=existing, keep="first")
    return metric


def build_version_merged_summary_from_logs(metric_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Log-based fallback for build_version_merged_summary (same algorithm)."""
    if metric_df.empty:
        return pd.DataFrame()
    if args.version_col not in metric_df.columns:
        raise ValueError(f"version_col={args.version_col!r} is not present in parsed table")

    rows: List[Dict[str, Any]] = []
    by_cols = GROUP_COLS + [args.version_col]

    for keys, part in metric_df.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(by_cols, keys))
        out["n_rows_used_for_metrics"] = int(len(part))
        out["n_unique_run_dirs"] = int(part["run_dir_resolved"].nunique())
        out["n_folds_for_version"] = int(part["cv_fold"].nunique(dropna=True))
        out["folds_for_version"] = ",".join(
            map(str, sorted(part["cv_fold"].dropna().astype(int).unique()))
        )
        out["run_dirs_resolved"] = "|".join(sorted(part["run_dir_resolved"].unique()))

        fold_dir_counts = part.groupby("cv_fold", dropna=False)["run_dir_resolved"].nunique()
        if (fold_dir_counts > 1).any():
            out["merge_status"] = "duplicate_run_dirs_for_same_fold"
            rows.append(out)
            continue

        if out["n_folds_for_version"] != int(args.expected_n_folds):
            out["merge_status"] = (
                f"missing_folds: expected {args.expected_n_folds}, found {out['n_folds_for_version']}"
            )
            rows.append(out)
            continue

        out["merge_status"] = "ok"
        for subset in args.subsets:
            try:
                aucpr, n_pickles, n_chains, paths = load_and_merge_pickles(part, subset)
                out[f"merged_{subset}_aucpr"] = aucpr
                out[f"n_{subset}_pickles"] = n_pickles
                out[f"n_{subset}_chains_merged"] = n_chains
                out[f"{subset}_pickle_paths"] = paths
                if n_pickles != int(args.expected_n_folds):
                    out["merge_status"] = (
                        f"missing_{subset}_pickles: expected {args.expected_n_folds}, found {n_pickles}"
                    )
            except Exception as exc:
                out[f"merged_{subset}_aucpr"] = np.nan
                out[f"n_{subset}_pickles"] = 0
                out[f"n_{subset}_chains_merged"] = 0
                out[f"{subset}_pickle_paths"] = ""
                out["merge_status"] = f"error: {type(exc).__name__}: {exc}"
        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grouped summary (mean ± std across seeds)
# ---------------------------------------------------------------------------

def build_grouped_summary(version_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average version-level merged AUCPRs across seeds.
    Returns one row per config with mean_*_aucpr and std_*_aucpr columns only
    (no fold-level averages — those would be incorrect).
    """
    if version_df.empty:
        return pd.DataFrame()

    ok = version_df[version_df["merge_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    agg: Dict[str, Tuple[str, Any]] = {
        "n_versions": ("merge_status", "count"),
    }
    for col in ok.columns:
        if col.startswith("merged_") and col.endswith("_aucpr"):
            subset = col.removeprefix("merged_").removesuffix("_aucpr")
            agg[f"mean_{subset}_aucpr"] = (col, "mean")
            agg[f"std_{subset}_aucpr"] = (col, "std")

    grouped = ok.groupby(GROUP_COLS, dropna=False).agg(**agg).reset_index()

    sort_cols = [col for col in ["mean_validation_aucpr", "mean_test_aucpr"] if col in grouped.columns]
    if sort_cols:
        grouped = grouped.sort_values(by=sort_cols, ascending=False, na_position="last")

    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Log analysis (status / error tracking) ----
    log_df = pd.DataFrame()
    if args.logs_dir and os.path.isdir(args.logs_dir):
        out_files = sorted(glob.glob(os.path.join(args.logs_dir, "*.out")))
        err_files = sorted(glob.glob(os.path.join(args.logs_dir, "*.err")))
        out_map = {Path(path).stem: path for path in out_files}
        err_map = {Path(path).stem: path for path in err_files}

        all_rows = [
            parse_log_pair(out_map.get(stem, ""), err_map.get(stem, ""))
            for stem in sorted(set(out_map) | set(err_map))
        ]
        df_all = normalize_df_types(enrich_from_runs_csv(pd.DataFrame(all_rows), args.runs_csv))
        log_df = df_all[df_all["status"] != "skipped"].copy()
    elif args.logs_dir:
        print(f"[WARN] --logs_dir not found: {args.logs_dir}")

    log_df.to_csv(args.out_csv, index=False)
    print(f"[OK] per-task log summary saved to: {args.out_csv}")

    if not log_df.empty:
        counts = log_df["status"].value_counts(dropna=False)
        print(f"[INFO] log status counts: {counts.to_dict()}")

    # ---- Metric computation (seeds approach) ----
    # Primary: scan results_base_dir directly — no dependency on SLURM logs.
    # Fallback: use log-resolved run dirs (only when results_base_dir not provided).
    if args.results_base_dir:
        version_df = build_version_merged_summary_from_results(args)
    else:
        if log_df.empty:
            print("[WARN] Neither --results_base_dir nor --logs_dir provided; no metrics computed.")
            version_df = pd.DataFrame()
        else:
            df_all_for_metrics = normalize_df_types(
                enrich_from_runs_csv(
                    pd.DataFrame(
                        [parse_log_pair(out_map.get(stem, ""), err_map.get(stem, ""))  # type: ignore[name-defined]
                         for stem in sorted(set(out_map) | set(err_map))]  # type: ignore[name-defined]
                    ),
                    args.runs_csv,
                )
            )
            metric_df = choose_metric_rows(df_all_for_metrics, args)
            version_df = build_version_merged_summary_from_logs(metric_df, args)

    version_df.to_csv(args.version_csv, index=False)
    print(f"[OK] per-version merged summary saved to: {args.version_csv}")

    grouped = build_grouped_summary(version_df)
    grouped.to_csv(args.grouped_csv, index=False)
    print(f"[OK] grouped summary saved to: {args.grouped_csv}")

    # ---- Status summary ----
    if not version_df.empty and "merge_status" in version_df.columns:
        print("\n[MERGE STATUS]")
        print(version_df["merge_status"].value_counts(dropna=False).to_string())

    # ---- Top groups: only val/test AUCPR mean ± std ----
    if not grouped.empty:
        print("\n[TOP GROUPS — val/test AUCPR mean ± std across seeds]")
        display_cols = []
        for col in [
            "esm2_version", "esm2_layer", "hidden_dims", "dropout",
            "learning_rate", "weight_decay",
            "mean_validation_aucpr", "std_validation_aucpr",
            "mean_test_aucpr", "std_test_aucpr",
            "n_versions",
        ]:
            if col in grouped.columns:
                display_cols.append(col)
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(grouped[display_cols].head(20).to_string(index=False))
    else:
        print("\n[WARN] No grouped summary available (no configs with all folds completed).")
        if not version_df.empty:
            print("Merge status breakdown:")
            print(version_df["merge_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
