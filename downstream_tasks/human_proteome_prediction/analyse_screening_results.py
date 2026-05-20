#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

THRESHOLDS = (35, 65, 85)

# --- Your exact column names ---
KNOWN_COL = "known catalytic sites"
EC_COL = "EC number (if exists)"
PRED_COL_TMPL = "predicted with {thr}% threshold"

# Output novel columns (computed by this script if --write_cleaned)
NOVEL_SITES_COL_TMPL = "novel_sites_{thr}"
N_NOVEL_COL_TMPL = "n_novel_{thr}"
ANY_NOVEL_COL_TMPL = "any_novel_{thr}"

NO_EC_LABEL = "No EC"

_INT_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")
_EC_TOP_RE = re.compile(r"^\s*(\d)\.")


# ---------------------------
# Reading
# ---------------------------

def robust_read_table(path: Path) -> pd.DataFrame:
    """
    Try TSV then CSV (keeps dtype=str). Pick the parse with most columns.
    """
    tried = []
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, engine="python")
            tried.append((sep, df))
        except Exception:
            continue
    if not tried:
        raise ValueError(f"Could not read file: {path}")
    tried.sort(key=lambda x: x[1].shape[1], reverse=True)
    return tried[0][1]


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------
# Parsing residue indices
# ---------------------------

def _to_int(x: Any) -> Optional[int]:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
    return None


def _is_chainlike(x: Any) -> bool:
    return isinstance(x, str) and x.strip().isalpha() and 1 <= len(x.strip()) <= 3


def _extract_ints_from_text(s: str) -> List[int]:
    ints = [int(m.group(1)) for m in _INT_RE.finditer(s)]
    # Drop 0: it's metadata in your triplets
    return [v for v in ints if v != 0]


def _flatten(obj: Any) -> List[Any]:
    out: List[Any] = []

    def rec(x: Any) -> None:
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        else:
            out.append(x)

    rec(obj)
    return out


def parse_residue_positions(cell: Any) -> List[int]:
    """
    Returns sorted unique residue indices (1-based, as in your table).
    Accepts:
      - "174,345" / "174;345" / "174 345" / "174, 345"
      - "['0','A','66']"
      - "[['0','A','66'], ['0','A','120']]"
      - "['0','A','66','0','A','120']"
      - "[]" / "" / NaN
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    # If looks like Python literal, try literal_eval
    if s.startswith("[") or s.startswith("("):
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return sorted(set(_extract_ints_from_text(s)))

        v = _to_int(obj)
        if v is not None and v != 0:
            return [v]

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return []

            # list of triplets
            if all(isinstance(x, (list, tuple)) for x in obj):
                vals: List[int] = []
                for entry in obj:
                    if not entry:
                        continue
                    v_last = _to_int(entry[-1])
                    if v_last is not None and v_last != 0:
                        vals.append(v_last)
                return sorted(set(vals))

            # single triplet ['0','A','66']
            if len(obj) == 3 and _is_chainlike(obj[1]) and _to_int(obj[2]) is not None:
                v_last = _to_int(obj[2])
                return [v_last] if v_last is not None and v_last != 0 else []

            # flattened triplets ['0','A','66','0','A','120']
            if len(obj) % 3 == 0 and len(obj) >= 3:
                ok = True
                vals2: List[int] = []
                for i in range(0, len(obj), 3):
                    chain = obj[i + 1]
                    res = obj[i + 2]
                    if not (_is_chainlike(chain) and _to_int(res) is not None):
                        ok = False
                        break
                    v_res = _to_int(res)
                    if v_res is not None and v_res != 0:
                        vals2.append(v_res)
                if ok:
                    return sorted(set(vals2))

            # generic: collect all ints anywhere (excluding 0)
            flat = _flatten(obj)
            vals3: List[int] = []
            for x in flat:
                v_any = _to_int(x)
                if v_any is not None and v_any != 0:
                    vals3.append(v_any)
            return sorted(set(vals3))

        return sorted(set(_extract_ints_from_text(s)))

    # Plain text list like "174;345" / "174,345" / "174 345"
    return sorted(set(_extract_ints_from_text(s)))


def format_positions(xs: Sequence[int]) -> str:
    """
    Format as '66, 120'. Empty -> '' (NOT '[]').
    """
    if not xs:
        return ""
    return ", ".join(str(int(v)) for v in xs)


# ---------------------------
# EC parsing
# ---------------------------

def ec_top_level(ec: Any) -> Optional[str]:
    """
    Returns 'EC1'..'EC7' if possible, otherwise None.
    """
    if ec is None or (isinstance(ec, float) and np.isnan(ec)):
        return None
    s = str(ec).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    m = _EC_TOP_RE.match(s)
    return f"EC{m.group(1)}" if m else None


def order_ec_buckets(buckets: Sequence[str]) -> List[str]:
    """
    Order EC1..EC7 numerically, and put 'No EC' last (if present).
    """
    ecs = []
    other = []
    has_no_ec = False
    for b in buckets:
        bb = str(b)
        if bb == NO_EC_LABEL:
            has_no_ec = True
        elif bb.startswith("EC") and bb[2:].isdigit():
            ecs.append(bb)
        else:
            other.append(bb)

    ecs_sorted = sorted(set(ecs), key=lambda x: int(x[2:]))
    other_sorted = sorted(set(other))
    out = ecs_sorted + other_sorted
    if has_no_ec:
        out.append(NO_EC_LABEL)
    return out


# ---------------------------
# Plot style + sizes
# ---------------------------

matplotlib.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.titlesize": 7,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 0.9,
    "svg.fonttype": "path",
})

W_FULL = 6.27
GUTTER = 0.15
W_HALF = (W_FULL - GUTTER) / 2
H_SMALL = 1.9


def _save_both(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=1000, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


# ---------------------------
# Plot helpers (all W_HALF x H_SMALL)
# ---------------------------

def save_barplot_fraction(labels: List[str], values: List[float], out_base: Path,
                          ylabel: str = "Fraction", title: str = "") -> None:
    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()
    ax.bar(labels, values, color=plt.colormaps["BuGn"](0.55), edgecolor='black')
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_both(fig, out_base)



def save_barplot_counts(labels: List[str], values: List[int], out_base: Path,
                        ylabel: str = "Number of proteins") -> None:
    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_both(fig, out_base)

def save_stacked_bar_novel_counts_2lvl(
    thr_labels: List[str],
    n_with_novel: List[int],
    n_no_known: List[int],
    out_base: Path,
) -> None:
    """
    Two-level stacked bar (absolute counts) among proteins with ≥1 novel residue:
      - No known sites (n_known == 0)
      - Annotation completion (n_known > 0)
    """
    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()

    x = np.arange(len(thr_labels))

    cmap = matplotlib.colormaps.get("Paired")
    c_no_known = cmap(2)
    c_completion = cmap(0)

    total = np.asarray(n_with_novel, dtype=int)
    no_known = np.asarray(n_no_known, dtype=int)
    completion = np.maximum(total - no_known, 0)

    ax.bar(x, no_known, label="No known sites", color=c_no_known, edgecolor='black')
    ax.bar(x, completion, bottom=no_known, label="Annotation completion", color=c_completion,edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(thr_labels)
    ax.set_ylabel("Number of proteins")
    ax.set_title("Novel predictions (proteins)")
    ax.grid(axis="y", alpha=0.25)

    # legend inside, compact
    ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=6,
        borderpad=0.2,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.4,
    ).get_frame().set_alpha(0.85)

    fig.tight_layout()
    _save_both(fig, out_base)


def save_stacked_bar_novel_counts(
    thr_labels: List[str],
    total_novel: List[int],
    n_new_enz_no_known: List[int],
    n_new_noec_no_known: List[int],
    out_base: Path,
) -> None:
    """
    One bar per threshold (absolute counts):
      total height = #proteins with >=1 novel residue (pred \ known)
      highlighted segments:
        - enzymes with EC but no curated catalytic-site annotations (known empty)
        - proteins without EC assignment and without curated annotations (known empty)
      remainder = site completion / other novel predictions
    """
    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()

    x = np.arange(len(thr_labels))

    cmap = matplotlib.colormaps.get("Paired")
    c_other = cmap(0)
    c_noec = cmap(2)
    c_enz = cmap(1)

    total = np.asarray(total_novel, dtype=int)
    a = np.asarray(n_new_enz_no_known, dtype=int)
    b = np.asarray(n_new_noec_no_known, dtype=int)
    plt.title("Novel predictions (proteins)")
    other = np.maximum(total - a - b, 0)

    ax.bar(x, other, label="Site completion /\nother novel predictions", color=c_other)
    ax.bar(x, b, bottom=other, label="No EC + no curated annotations", color=c_noec)
    ax.bar(x, a, bottom=other + b, label="EC-assigned +\nno curated annotations", color=c_enz)

    ax.set_xticks(x)
    ax.set_xticklabels(thr_labels)
    ax.set_ylabel("Number of proteins")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")


    fig.tight_layout()
    _save_both(fig, out_base)


def save_pie(values: List[float], labels: List[str], out_base: Path, title=None) -> None:
    """
    Pie chart + legend (Paired palette).
    Show percentage only for the largest wedge.
    """
    fig = plt.figure(figsize=(W_HALF, H_SMALL))
    ax = plt.gca()

    vals = np.asarray(values, dtype=float)
    total = float(vals.sum())
    if total <= 0:
        # nothing to plot
        fig.tight_layout()
        _save_both(fig, out_base)
        return

    max_pct = 100.0 * float(vals.max()) / total
    eps = 1e-6

    def autopct_only_max(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= (max_pct - eps) else ""

    cmap = matplotlib.colormaps.get("Paired")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]

    if title is not None:
        plt.title(title)

    wedges, _, _ = ax.pie(
        vals,
        colors=colors,
        startangle=90,
        autopct=autopct_only_max,   # only largest slice gets text
        pctdistance=0.7,            # position inside wedge
        textprops={"fontsize": 6},
    )
    ax.axis("equal")

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    _save_both(fig, out_base)



# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="Input CSV/TSV file")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--write_cleaned", action="store_true",
                    help="Write cleaned CSV with normalized lists + computed novel columns")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = robust_read_table(in_path)

    required = [KNOWN_COL, EC_COL] + [PRED_COL_TMPL.format(thr=t) for t in THRESHOLDS]
    require_columns(df, required)

    # Parse known
    df["_known_list"] = df[KNOWN_COL].apply(parse_residue_positions)
    df["_known_set"] = df["_known_list"].apply(set)
    df["_n_known"] = df["_known_list"].apply(len)

    # Parse EC top-level (EC1..EC7 or None)
    df["_ec_top"] = df[EC_COL].apply(ec_top_level)
    df["_has_ec"] = df["_ec_top"].notna()

    # Parse predicted + novel
    for thr in THRESHOLDS:
        pred_col = PRED_COL_TMPL.format(thr=thr)

        df[f"_pred_list_{thr}"] = df[pred_col].apply(parse_residue_positions)
        df[f"_pred_set_{thr}"] = df[f"_pred_list_{thr}"].apply(set)
        df[f"_any_pred_{thr}"] = df[f"_pred_list_{thr}"].apply(len) > 0

        # novel = predicted \ known
        df[f"_novel_list_{thr}"] = df.apply(
            lambda r, t=thr: sorted(r[f"_pred_set_{t}"] - r["_known_set"]),
            axis=1,
        )
        df[f"_n_novel_{thr}"] = df[f"_novel_list_{thr}"].apply(len)
        df[f"_any_novel_{thr}"] = df[f"_n_novel_{thr}"] > 0

        # Highlights for the stacked bar:
        # - EC-assigned proteins with no curated catalytic-site annotations (no known sites)
        # - No-EC proteins with no curated annotations (no known sites)
        df[f"_new_ec_no_known_{thr}"] = (df["_has_ec"]) & (df["_n_known"] == 0) & (df[f"_any_pred_{thr}"])
        df[f"_new_noec_no_known_{thr}"] = (~df["_has_ec"]) & (df["_n_known"] == 0) & (df[f"_any_pred_{thr}"])

    # -------------------
    # Summaries (CSV)
    # -------------------
    n_total = len(df)

    # 1) Novel counts (absolute) + fraction (useful for text)
    novel_rows = []
    for thr in THRESHOLDS:
        any_novel = df[f"_any_novel_{thr}"]
        n_with_novel = int(any_novel.sum())

        n_no_known = int((any_novel & (df["_n_known"] == 0)).sum())
        n_completion = int(max(n_with_novel - n_no_known, 0))

        novel_rows.append({
            "threshold": thr,
            "n_proteins": n_total,
            "n_with_novel": n_with_novel,
            "n_no_known_sites": n_no_known,
            "n_annotation_completion": n_completion,
            "frac_with_novel": safe_div(n_with_novel, n_total),
        })
    novel_summary = pd.DataFrame(novel_rows)
    novel_summary.to_csv(out_dir / "summary__novel_counts.csv", index=False)

    novel_summary = pd.DataFrame(novel_rows)
    novel_summary.to_csv(out_dir / "summary__novel_counts.csv", index=False)

    # 2) Global recall of known catalytic residues at each threshold:
    #    (micro-averaged) = sum_i |K_i ∩ P_i| / sum_i |K_i|
    known_df = df.loc[df["_n_known"] > 0].copy()
    recall_rows = []
    if len(known_df) > 0:
        total_known = int(known_df["_n_known"].sum())
        for thr in THRESHOLDS:
            total_hit = int(known_df.apply(
                lambda r, t=thr: len(r[f"_pred_set_{t}"].intersection(r["_known_set"])),
                axis=1
            ).sum())
            recall_rows.append({
                "threshold": thr,
                "n_proteins_with_known": len(known_df),
                "total_known_residues": total_known,
                "total_known_hit": total_hit,
                "global_known_recall": safe_div(total_hit, total_known),
            })
    known_recall = pd.DataFrame(recall_rows)
    known_recall.to_csv(out_dir / "summary__global_known_recall.csv", index=False)

    # 3) EC top-level distribution for all proteins at 35%
    ec_all = df.copy()
    ec_all["ec_bucket"] = ec_all["_ec_top"].fillna(NO_EC_LABEL)
    ec_dist_all = ec_all.groupby("ec_bucket").size().rename("count").reset_index()
    ec_dist_all["fraction"] = ec_dist_all["count"] / float(n_total)
    order_all = order_ec_buckets(ec_dist_all["ec_bucket"].astype(str).tolist())
    ec_dist_all["ec_bucket"] = pd.Categorical(ec_dist_all["ec_bucket"], categories=order_all, ordered=True)
    ec_dist_all = ec_dist_all.sort_values("ec_bucket")
    ec_dist_all.to_csv(out_dir / "summary__ec_dist_all_35.csv", index=False)

    # 4) EC top-level distribution for proteins with novel@35
    novel35 = df.loc[df["_any_novel_35"]].copy()
    if len(novel35) > 0:
        novel35["ec_bucket"] = novel35["_ec_top"].fillna(NO_EC_LABEL)
        ec_dist_novel = novel35.groupby("ec_bucket").size().rename("count").reset_index()
        ec_dist_novel["fraction"] = ec_dist_novel["count"] / float(len(novel35))
        ec_dist_novel["n_in_group"] = len(novel35)
        order_novel = order_ec_buckets(ec_dist_novel["ec_bucket"].astype(str).tolist())
        ec_dist_novel["ec_bucket"] = pd.Categorical(ec_dist_novel["ec_bucket"], categories=order_novel, ordered=True)
        ec_dist_novel = ec_dist_novel.sort_values("ec_bucket")
    else:
        ec_dist_novel = pd.DataFrame(columns=["ec_bucket", "count", "fraction", "n_in_group"])
    ec_dist_novel.to_csv(out_dir / "summary__ec_dist_novel_35.csv", index=False)

    # -------------------
    # Plots (PNG + SVG), all W_HALF x H_SMALL
    # -------------------
    thr_labels = [f"{(t/100):.2f}" for t in THRESHOLDS]

    # (c) Novel counts (stacked, absolute)
    save_stacked_bar_novel_counts_2lvl(
        thr_labels=thr_labels,
        n_with_novel=novel_summary["n_with_novel"].astype(int).tolist(),
        n_no_known=novel_summary["n_no_known_sites"].astype(int).tolist(),
        out_base=out_dir / "bar_novel_counts_stacked",
    )


    # (d) Known recall (fraction)
    if not known_recall.empty:
        save_barplot_fraction(
            labels=thr_labels,
            values=known_recall["global_known_recall"].astype(float).tolist(),
            out_base=out_dir / "bar_global_known_recall",
            ylabel="Recall",
            title="Curated annotations recall",
        )

    else:
        (out_dir / "NOTE__known_recall_missing.txt").write_text(
            "No proteins with known catalytic sites detected (or known column is empty).",
            encoding="utf-8",
        )

    # (a) EC distributions (all @35): bar + pie
    save_barplot_fraction(
        labels=ec_dist_all["ec_bucket"].astype(str).tolist(),
        values=ec_dist_all["fraction"].astype(float).tolist(),
        out_base=out_dir / "bar_ec_dist_all_35",
        ylabel="Fraction",
    )
    save_pie(
        values=ec_dist_all["fraction"].astype(float).tolist(),
        labels=ec_dist_all["ec_bucket"].astype(str).tolist(),
        out_base=out_dir / "pie_ec_dist_all_35",
        title="EC composition (all proteins)"
    )

    # (b) EC distributions (novel@35): bar + pie
    if not ec_dist_novel.empty:
        save_barplot_fraction(
            labels=ec_dist_novel["ec_bucket"].astype(str).tolist(),
            values=ec_dist_novel["fraction"].astype(float).tolist(),
            out_base=out_dir / "bar_ec_dist_novel_35",
            ylabel="Fraction",
        )
        save_pie(
            values=ec_dist_novel["fraction"].astype(float).tolist(),
            labels=ec_dist_novel["ec_bucket"].astype(str).tolist(),
            out_base=out_dir / "pie_ec_dist_novel_35",
            title="EC composition (novel)"
        )
    else:
        (out_dir / "NOTE__ec_dist_novel_35_missing.txt").write_text(
            "No proteins with novel@35 found; cannot plot EC distribution for novel@35 subset.",
            encoding="utf-8",
        )

    # -------------------
    # Cleaned CSV (optional)
    # -------------------
    if args.write_cleaned:
        cleaned = df.copy()

        # normalize predicted columns + add computed novel columns
        for thr in THRESHOLDS:
            pred_col = PRED_COL_TMPL.format(thr=thr)
            cleaned[pred_col] = cleaned[f"_pred_list_{thr}"].apply(format_positions)

            cleaned[NOVEL_SITES_COL_TMPL.format(thr=thr)] = cleaned[f"_novel_list_{thr}"].apply(format_positions)
            cleaned[N_NOVEL_COL_TMPL.format(thr=thr)] = cleaned[f"_n_novel_{thr}"].astype(int)
            cleaned[ANY_NOVEL_COL_TMPL.format(thr=thr)] = cleaned[f"_any_novel_{thr}"].astype(bool)

        cleaned[KNOWN_COL] = cleaned["_known_list"].apply(format_positions)

        # drop internal helper columns
        drop_cols = [c for c in cleaned.columns if c.startswith("_")]
        cleaned = cleaned.drop(columns=drop_cols, errors="ignore")

        cleaned_path = out_dir / f"{in_path.stem}.cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)

        # Filtered + sorted: only rows with at least one novel site at any threshold
        any_novel_mask = cleaned[[ANY_NOVEL_COL_TMPL.format(thr=t) for t in THRESHOLDS]].any(axis=1)
        novel_only = cleaned[any_novel_mask].copy()
        sort_cols = [N_NOVEL_COL_TMPL.format(thr=t) for t in sorted(THRESHOLDS, reverse=True)
                     if N_NOVEL_COL_TMPL.format(thr=t) in novel_only.columns]
        if sort_cols:
            novel_only = novel_only.sort_values(sort_cols, ascending=False)
        novel_only_path = out_dir / f"{in_path.stem}.novel_only.csv"
        novel_only.to_csv(novel_only_path, index=False)

    print("Saved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
