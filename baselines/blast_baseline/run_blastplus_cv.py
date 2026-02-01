#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling-window BLAST baseline ON-THE-FLY from split*.txt with PR curves
and chain weights from dataset.csv (W_Structure), using Bio.Align.PairwiseAligner
(with OverflowError-safe handling and strict type guards).

Per fold (idx in 0..4):
  train       = [idx, idx+1, idx+2] mod 5
  validation  =  idx+3              mod 5
  test        =  idx+4              mod 5

For each fold, produce four subsets:
  - train         (self-hits allowed)
  - train_noself  (self-hits forbidden)
  - validation
  - test

Outputs (per fold):
  fold{idx}/
    blast_fold{idx}_{subset}.csv            # From,target,sequence identity,BLAST_residues
    metrics_fold{idx}_{subset}.csv          # macro P,R,F1
    {subset}_results.pkl, {subset}_plot.png # PR-curve assets
  metrics_overall.csv (summary across folds/subsets)

Requirements:
  - pandas, numpy, matplotlib, scikit-learn, biopython>=1.78
  - NCBI BLAST+ (makeblastdb, blastp) in PATH if --engine blastp (recommended)
"""

import os
import glob
import argparse
import shutil
import subprocess
import pickle
from pathlib import Path
from typing import List, Tuple, Set, Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from Bio.Align import PairwiseAligner


# -------------------- PR & saving helpers --------------------

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
    """Save pickle and PR-curve PNG for a subset."""
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


# -------------------- parsing & DB --------------------

def parse_split_file(path: str) -> pd.DataFrame:
    """splitX.txt -> DataFrame(id, seq, true_set)."""
    rows, cur_id, buf = [], None, []
    def flush():
        nonlocal cur_id, buf
        if cur_id is None:
            return
        buf.sort(key=lambda x: x[0])
        seq = "".join(aa for (pos, aa, lab) in buf)
        true = {pos - 1 for (pos, aa, lab) in buf if int(lab) == 1}
        rows.append({"id": cur_id, "seq": seq, "true_set": true})
        cur_id, buf = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                cur_id = s[1:].strip()
                buf = []
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
                buf.append((pos, aa, lab))
    flush()
    return pd.DataFrame(rows)


def build_db_from_splits(split_paths: List[str]) -> pd.DataFrame:
    """Build DB (Entry,Sequence,Residue) from split files."""
    parts = [parse_split_file(p) for p in split_paths]
    df = pd.concat(parts, ignore_index=True)
    def to_pipe(st: Set[int]) -> str:
        return "|".join(map(str, sorted(st))) if st else ""
    db = pd.DataFrame({
        "Entry": df["id"],
        "Sequence": df["seq"],
        "Residue": df["true_set"].apply(to_pipe)
    })
    return db.drop_duplicates(subset=["Entry"], keep="first").reset_index(drop=True)


def write_fasta(df: pd.DataFrame, id_col: str, seq_col: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, s in zip(df[id_col], df[seq_col]):
            f.write(f">{i}\n")
            for k in range(0, len(s), 60):
                f.write(s[k:k+60] + "\n")


# -------------------- alignment (PairwiseAligner, safe) --------------------

def _make_aligner(match=2.0, mismatch=-1.001, open_gap=-4.001, extend_gap=-1.001) -> PairwiseAligner:
    """
    Global aligner with tiny epsilons in penalties to break ties and
    max_number_of_alignments=1 to avoid counting astronomical numbers of optimal paths.
    """
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score = match
    al.mismatch_score = mismatch
    al.open_gap_score = open_gap
    al.extend_gap_score = extend_gap
    try:
        al.max_number_of_alignments = 1  # avoid OverflowError in __len__
    except AttributeError:
        pass
    return al


def transfer_by_global(qseq: str, tseq: str, t_idx: Set[int], aligner: PairwiseAligner) -> Set[int]:
    """Map target catalytic indices (ungapped 0-based) -> query indices via global alignment blocks."""
    if not qseq or not tseq or not t_idx:
        return set()
    aln = aligner.align(qseq, tseq)
    try:
        aln0 = aln[0]  # DO NOT check "if not aln": it triggers __len__ and may overflow
    except (IndexError, OverflowError):
        return set()
    q_blocks, t_blocks = aln0.aligned
    mapped: Set[int] = set()
    for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
        L = min(qe - qs, te - ts)
        if L <= 0:
            continue
        for i in range(L):
            t_pos = ts + i
            if t_pos in t_idx:
                q_pos = qs + i
                mapped.add(q_pos)
    return mapped


# -------------------- search engines --------------------

def run_blastp(db_fasta: Path, q_fasta: Path, out_tsv: Path, tmpdir: Path, threads: int, k: int = 50):
    """makeblastdb + blastp (top-k per query), outfmt6 TSV."""
    db_prefix = tmpdir / "blastdb"
    subprocess.run(
        ["makeblastdb", "-in", str(db_fasta), "-dbtype", "prot", "-out", str(db_prefix)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    cmd = [
        "blastp",
        "-query", str(q_fasta),
        "-db", str(db_prefix),
        "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
        "-max_target_seqs", str(k),
        "-max_hsps", "1",
        "-evalue", "10",
        "-seg", "yes",
        "-comp_based_stats", "2",
        "-num_threads", str(max(1, threads)),
        "-out", str(out_tsv),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def rank_hits_python(qdf: pd.DataFrame, db_df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """
    Slow fallback: compute global identity vs every DB seq with PairwiseAligner;
    keep top-k per query. Safe against OverflowError.
    """
    # identity aligner: matches=1, mismatches=0, mild gaps; epsilons to break ties
    aln_id = _make_aligner(match=1.0, mismatch=0.0, open_gap=-2.001, extend_gap=-1.001)

    rows = []
    for _, qr in qdf.iterrows():
        qseq = qr["seq"]
        bucket: List[Tuple[str, float]] = []
        for _, dr in db_df.iterrows():
            tseq = dr["Sequence"]
            aln = aln_id.align(qseq, tseq)
            try:
                aln0 = aln[0]
            except (IndexError, OverflowError):
                continue
            q_blocks, t_blocks = aln0.aligned
            matches, aligned_len = 0, 0
            for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
                L = min(qe - qs, te - ts)
                if L <= 0:
                    continue
                qs_str = qseq[qs:qe]
                ts_str = tseq[ts:te]
                for i in range(L):
                    if qs_str[i] == ts_str[i]:
                        matches += 1
                aligned_len += L
            ident = (matches / aligned_len) if aligned_len else 0.0
            bucket.append((dr["Entry"], ident * 100.0))
        bucket.sort(key=lambda x: x[1], reverse=True)
        for sseqid, pident in bucket[:k]:
            rows.append({"qseqid": qr["id"], "sseqid": sseqid, "pident": pident,
                         "length": None, "evalue": None, "bitscore": None})
    return pd.DataFrame(rows)


# -------------------- metrics & tensors --------------------

def prf1(true_set, pred_set) -> Tuple[float, float, float]:
    """Robust macro P/R/F1 for sets; tolerate wrong types/None."""
    if not isinstance(true_set, set):
        true_set = set(true_set) if true_set is not None else set()
    if not isinstance(pred_set, set):
        pred_set = set(pred_set) if pred_set is not None else set()
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def sets_to_arrays(qdf: pd.DataFrame, pred_map: Dict[str, Set[int]], id2w: Dict[str, float]):
    """Build per-residue labels/preds/weights/ids/splits for PR plotting."""
    labels, preds, weights, ids, splits = [], [], [], [], []
    for _, row in qdf.iterrows():
        qid, seq, true_set = row["id"], row["seq"], row["true_set"]
        L = len(seq)
        y_true = np.zeros(L, dtype=np.int8)
        if isinstance(true_set, set) and true_set:
            idx = [i for i in true_set if 0 <= i < L]
            if idx:
                y_true[idx] = 1
        y_pred = np.zeros(L, dtype=np.float32)
        ps = pred_map.get(qid, set())
        if isinstance(ps, set) and ps:
            idxp = [i for i in ps if 0 <= i < L]
            if idxp:
                y_pred[idxp] = 1.0
        labels.append(y_true)
        preds.append(y_pred)
        w = id2w.get(qid, 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        weights.append(w)
        ids.append(qid)
        splits.append("")  # не используем
    return labels, preds, weights, ids, splits


# -------------------- core: choosing hit & mapping --------------------

def pick_top_hit_grouped(
    grouped_hits: Iterable[Tuple[str, pd.DataFrame]],
    qdf: pd.DataFrame,
    db_seq: Dict[str, str],
    db_sites: Dict[str, Set[int]],
    aligner: PairwiseAligner,
    allow_self: bool,
    min_identity: float | None
) -> Dict[str, Set[int]]:
    """
    From grouped ranked hits (desc by pident), build pred_map: qid -> mapped indices set.
    Enforces self-hit policy and min_identity threshold.
    """
    pred_map: Dict[str, Set[int]] = {}
    for qid, g in grouped_hits:
        chosen = None
        if g is not None and not g.empty:
            for _, r in g.iterrows():
                tid = r["sseqid"]
                pid = float(r.get("pident", 0)) / 100.0 if pd.notna(r.get("pident", None)) else None
                if (min_identity is not None) and (pid is None or pid < min_identity):
                    continue
                if (not allow_self) and (tid == qid):
                    continue
                chosen = (tid, pid)
                break

        if chosen is None:
            pred_map[qid] = set()
            continue

        tid, _ = chosen
        qs_series = qdf.loc[qdf["id"] == qid, "seq"]
        qs = qs_series.iloc[0] if not qs_series.empty else ""
        ts = db_seq.get(tid, "")
        tset = db_sites.get(tid, set())
        mapped = transfer_by_global(qs, ts, tset, aligner) if (qs and ts and tset) else set()
        pred_map[qid] = mapped
    return pred_map


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_glob", required=True, help='"/path/split*.txt" (expects 5 files split1..split5)')
    ap.add_argument("--dataset_csv", required=True, help="CSV with Sequence_ID and W_Structure")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--engine", choices=["blastp", "python"], default="blastp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min_identity", type=float, default=None, help="keep hit only if pident >= this (0..1)")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect 5 split files and order by numeric suffix
    split_paths = sorted(glob.glob(args.splits_glob))
    if len(split_paths) != 5:
        raise SystemExit(f"Expected 5 split files, got {len(split_paths)}")
    def split_idx(p: str) -> int:
        return int("".join(ch for ch in Path(p).stem if ch.isdigit()))
    split_paths = sorted(split_paths, key=split_idx)

    # Load weights
    ds = pd.read_csv(args.dataset_csv)
    if not {"Sequence_ID", "W_Structure"}.issubset(ds.columns):
        raise SystemExit("dataset.csv must contain columns: Sequence_ID, W_Structure")
    id2w = ds.set_index("Sequence_ID")["W_Structure"].to_dict()

    # Tool check
    if args.engine == "blastp":
        for tool in ["makeblastdb", "blastp"]:
            if shutil.which(tool) is None:
                raise SystemExit(f"{tool} not found in PATH")

    # One aligner instance for mapping catalytic indices (scoring for structural conservation)
    mapping_aligner = _make_aligner(match=2.0, mismatch=-1.001, open_gap=-4.001, extend_gap=-1.001)

    all_macro = []

    for idx in range(5):
        train_ids = [(idx + j) % 5 for j in range(3)]
        val_id    = (idx + 3) % 5
        test_id   = (idx + 4) % 5

        fold_dir = out_root / f"fold{idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # DB from TRAIN splits
        db_df = build_db_from_splits([split_paths[t] for t in train_ids])

        # DB maps
        def parse_res(s: str) -> Set[int]:
            if pd.isna(s) or str(s) == "":
                return set()
            return set(int(x) for x in str(s).split("|") if str(x).strip() != "")
        db_seq  = dict(zip(db_df["Entry"], db_df["Sequence"]))
        db_sites = {row["Entry"]: parse_res(row["Residue"]) for _, row in db_df.iterrows()}

        # Prepare DB FASTA once per fold
        tmp_db = fold_dir / "tmp_db"
        tmp_db.mkdir(exist_ok=True, parents=True)
        db_fa = tmp_db / "db.fasta"
        write_fasta(db_df.rename(columns={"Entry": "id", "Sequence": "seq"}), "id", "seq", db_fa)

        # subsets to run
        plan = {
            "train":      [split_paths[t] for t in train_ids],
            "validation": [split_paths[val_id]],
            "test":       [split_paths[test_id]],
        }

        for subset, files in plan.items():
            # Build queries DF
            qdf = pd.concat([parse_split_file(p) for p in files], ignore_index=True)

            # Prepare query FASTA
            tmp = fold_dir / f"tmp_{subset}"
            tmp.mkdir(exist_ok=True, parents=True)
            q_fa = tmp / "queries.fasta"
            write_fasta(qdf, "id", "seq", q_fa)

            # Search (collect ranked hits, top-k) to enable "noself"
            if args.engine == "blastp":
                hits_tsv = tmp / "hits.tsv"
                run_blastp(db_fa, q_fa, hits_tsv, tmp, threads=args.threads, k=50)
                hits = pd.read_csv(
                    hits_tsv, sep="\t", header=None,
                    names=["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
                )
            else:
                hits = rank_hits_python(qdf, db_df, k=50)

            if hits.empty:
                grouped = {qid: pd.DataFrame(columns=["qseqid","sseqid","pident","length","evalue","bitscore"])
                           for qid in qdf["id"].tolist()}
            else:
                hits = hits.sort_values(["qseqid", "pident"], ascending=[True, False]).reset_index(drop=True)
                grouped = dict(tuple(hits.groupby("qseqid", sort=False)))

            # Build both variants for TRAIN; for val/test — single variant
            subset_keys = [subset] if subset != "train" else ["train", "train_noself"]
            allow_map   = {"train": True, "train_noself": False, "validation": True, "test": True}

            for key in subset_keys:
                allow_self = allow_map[key]

                # Build pred_map
                pred_map = pick_top_hit_grouped(
                    grouped_hits=grouped.items(),
                    qdf=qdf,
                    db_seq=db_seq,
                    db_sites=db_sites,
                    aligner=mapping_aligner,
                    allow_self=allow_self,
                    min_identity=args.min_identity
                )

                # ----- raw CSV (From, target, sequence identity, BLAST_residues)
                rows = []
                for qid, g in grouped.items():
                    chosen_t = None
                    if g is not None and not g.empty:
                        for _, r in g.iterrows():
                            tid = r["sseqid"]
                            pid = float(r.get("pident", 0)) / 100.0 if pd.notna(r.get("pident", None)) else None
                            if (args.min_identity is not None) and (pid is None or pid < args.min_identity):
                                continue
                            if (not allow_self) and (tid == qid):
                                continue
                            chosen_t = (tid, pid)
                            break
                    row = {
                        "From": qid,
                        "BLAST_residues": "|".join(map(str, sorted(pred_map.get(qid, set())))) if pred_map.get(qid) else ""
                    }
                    if chosen_t is not None:
                        row.update({"target": chosen_t[0], "sequence identity": chosen_t[1]})
                    else:
                        row.update({"target": None, "sequence identity": None})
                    rows.append(row)
                pd.DataFrame(rows).to_csv(fold_dir / f"blast_fold{idx}_{key}.csv", index=False)

                # ----- metrics (macro P/R/F1)
                merged = qdf.copy()
                merged["pred_set"] = merged["id"].map(lambda x: pred_map.get(x, set()))
                prs = merged.apply(lambda rr: prf1(rr["true_set"], rr["pred_set"]), axis=1)
                merged[["P", "R", "F1"]] = pd.DataFrame(prs.tolist(), index=merged.index)
                macro = merged[["P", "R", "F1"]].mean().to_frame().T
                macro.insert(0, "fold", idx)
                macro.insert(1, "subset", key)
                macro.to_csv(fold_dir / f"metrics_fold{idx}_{key}.csv", index=False)

                # ----- PR curves + pickle with weights
                labels, predictions, weights, ids, splits = sets_to_arrays(qdf, pred_map, id2w)
                save_subset_results(
                    output_dir=str(fold_dir),
                    subset_key=key,
                    title=f"BLAST ({key}) — fold{idx}",
                    labels=labels,
                    predictions=predictions,
                    weights=weights,
                    ids=ids,
                    splits=[f"fold{idx}"] * len(ids),
                    model_name="BLAST-baseline",
                    batch_size=None
                )

            if not args.keep_tmp:
                shutil.rmtree(tmp, ignore_errors=True)

        if not args.keep_tmp:
            shutil.rmtree(tmp_db, ignore_errors=True)

    # Overall summary
    # Собираем из файлов, чтобы не держать всё в памяти и быть устойчивыми к падениям
    frames = []
    for p in sorted(out_root.glob("fold*/metrics_fold*_*.*")):
        if p.name.endswith(".csv") and "metrics_fold" in p.name:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    if frames:
        overall = pd.concat(frames, ignore_index=True)
        overall.to_csv(out_root / "metrics_overall.csv", index=False)
        print("Overall macro by subset:\n", overall.groupby("subset")[["P", "R", "F1"]].mean())
    else:
        print("No metrics produced (no hits?).")


if __name__ == "__main__":
    main()
