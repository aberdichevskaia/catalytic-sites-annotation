#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AA sequence-homology baseline with 5-fold CV (Smith–Waterman + BLOSUM62).
- Template reweighting via exponent (template_weight_exp).
- Adaptive per-target thresholding and TOP-K hit capping.
- Weighted label propagation from templates to target AAs.

Inputs: JSON with keys for each split 'split1'..'split5' containing:
  db_ids:                list[ [pdb_id, chain_id] ]
  db_annotated_aa_sequences: list[str]  (AA sequences)
  db_labels:             list[list[int]] (0/1 per AA position)
  structure_weights:     optional list[float] (per template weight)

Outputs: per-subset pickles + PR-curve PNGs.
"""

import os, json, argparse, pickle
from multiprocessing import Pool, cpu_count
import numpy as np

import biotite.sequence.align as align
from biotite.sequence import ProteinSequence

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# --------------------- Config (can be overridden by CLI) ---------------------
SPLITS_JSON = "/home/iscb/wolfson/annab4/DB/all_proteins/structural_homology/3Di_DB_splits.json"
OUTPUT_DIR_BASE = "/home/iscb/wolfson/annab4/DB/all_proteins/sequence_homology_baseline/results_adaptive_cutoff"

DENOM = 2                # threshold = min(max_score/DENOM, pctl_score)
PCTL = 95.0              # hybrid threshold percentile of raw best scores
CUTOFF = 60              # minimal alignment score (BLOSUM62-ish)
MIN_ALIGN_LEN = 9        # minimal aligned AA positions to accept hit
TOPK_HITS = 32           # cap number of hits per target
WEIGHT_GAMMA = 1.0       # non-linear emphasis on stronger alignments
TEMPLATE_WEIGHT_EXP = 0.0
GAP_PENALTY = (-8, -1)   # (open, extend) for Smith–Waterman
THREADS = min(128, cpu_count())

# --------------------- Utils ---------------------
def to_prot_list(seq_str_list):
    return [ProteinSequence(s) for s in seq_str_list]

def to_int_labels(labels_list):
    return [list(map(int, labs)) for labs in labels_list]

def load_splits(path):
    with open(path, "r") as f:
        return json.load(f)

def collect_from_splits(all_data, names):
    ids, aa, labels, weights, split_tags = [], [], [], [], []
    for name in names:
        d = all_data[name]
        ids.extend([tuple(x) for x in d["db_ids"]])
        aa.extend(to_prot_list(d["db_annotated_aa_sequences"]))
        labels.extend(to_int_labels(d["db_labels"]))
        weights.extend(d.get("structure_weights", [1.0]*len(d["db_ids"])))
        split_tags.extend([name]*len(d["db_ids"]))
    return {
        "ids": ids,
        "aa": aa,
        "labels": labels,
        "weights": np.asarray(weights, float),
        "split_tags": split_tags,
    }

def compute_index_mapping(gapped_a, gapped_b):
    """Map indices from ungapped A to ungapped B using two gapped strings."""
    m, i, j = {}, 0, 0
    for a, b in zip(gapped_a, gapped_b):
        if a != "-" and b != "-":
            m[i] = j; i += 1; j += 1
        elif a != "-" and b == "-":
            i += 1
        elif a == "-" and b != "-":
            j += 1
    return m

def flatten_for_pr(labels, predictions, chain_weights):
    # Repeat chain weights to per-position
    sw = np.concatenate([np.ones(len(l))*w for l, w in zip(labels, chain_weights)])
    y_true = np.concatenate(labels).astype(float)
    y_pred = np.concatenate(predictions).astype(float)
    bad = ~np.isfinite(y_pred) | ~np.isfinite(y_true)
    if bad.any():
        y_pred[bad] = np.nanmedian(y_pred[~bad]) if (~bad).any() else 0.0
        y_true[bad] = 0.0
        sw[bad] = 0.0
    return y_true, y_pred, sw

def pr_and_plot(labels, preds, weights, subset, title, out_png):
    y, p, w = flatten_for_pr(labels, preds, weights)
    prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
    score = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(rec, prec, lw=2, label=f"{subset} (AUCPR={score:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(-0.02,1.02), ylim=(-0.02,1.02), title=title)
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    return score

def save_subset(out_dir, subset_key, title, labels, predictions, weights, ids, splits, model_name):
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(
        subset=subset_key, model_name=model_name, labels=labels,
        predictions=predictions, weights=weights, ids=ids, splits=splits
    )
    pkl = os.path.join(out_dir, f"{subset_key}.pkl")
    with open(pkl, "wb") as f: pickle.dump(payload, f)
    png = os.path.join(out_dir, f"{subset_key}.png")
    aupr = pr_and_plot(labels, predictions, weights, subset_key, title, png)
    print(f"[{subset_key}] AUCPR={aupr:.4f}  -> {pkl} ; {png}")

# --------------------- Worker globals ---------------------
G_TEMPL_AA = None
G_TEMPL_LABELS = None
G_TEMPL_IDS = None
G_TEMPL_W = None
G_SUBMAT = None
G_CUTOFF = None
G_GAP = None
G_SKIP_SELF = False
G_DENOM = None
G_MINLEN = None
G_TOPK = None
G_GAMMA = None
G_TW_EXP = None
G_PCTL = None
G_SELF_AA = None
G_SELF_LABELS = None

def init_worker(db_aa, db_labels, db_ids, db_w, submat, cutoff, gap, skip_self,
                denom, minlen, topk, gamma, tw_exp, pctl):
    global G_TEMPL_AA, G_TEMPL_LABELS, G_TEMPL_IDS, G_TEMPL_W, G_SUBMAT
    global G_CUTOFF, G_GAP, G_SKIP_SELF, G_DENOM, G_MINLEN, G_TOPK, G_GAMMA, G_TW_EXP, G_PCTL
    G_TEMPL_AA = db_aa
    G_TEMPL_LABELS = [np.asarray(x, int) for x in db_labels]
    G_TEMPL_IDS = db_ids
    G_TEMPL_W = np.asarray(db_w, float)
    G_SUBMAT = submat
    G_CUTOFF = cutoff
    G_GAP = gap
    G_SKIP_SELF = skip_self
    G_DENOM = denom
    G_MINLEN = minlen
    G_TOPK = topk
    G_GAMMA = gamma
    G_TW_EXP = tw_exp
    G_PCTL = pctl

def predict_one(job):
    """Propagate labels from templates to one target AA sequence."""
    idx, tid, taa = job

    # 1) collect best local alignments (score, aln, j)
    hits, best_scores = [], []
    for j, templ in enumerate(G_TEMPL_AA):
        if G_SKIP_SELF and tid == G_TEMPL_IDS[j]:
            continue
        alns = align.align_optimal(taa, templ, G_SUBMAT, gap_penalty=G_GAP, local=True)
        if not alns: continue
        best = alns[0]
        best_scores.append(best.score)
        if best.score >= G_CUTOFF:
            hits.append((best.score, best, j))

    pred = np.zeros(len(taa), float)
    cnt = np.zeros(len(taa), float)

    if hits:
        max_s = max(s for s,_,_ in hits)
        pctl_s = np.percentile(best_scores, G_PCTL) if best_scores else G_CUTOFF
        thr = min(max_s / G_DENOM, pctl_s)
        thr = min(thr, max_s - 1e-6)
        kept = [(s,a,j) for (s,a,j) in hits if s > thr]
        kept.sort(key=lambda x: x[0], reverse=True)
        if G_TOPK and len(kept) > G_TOPK: kept = kept[:G_TOPK]
        if not kept:  # fallback to single best
            kept = [max(hits, key=lambda x: x[0])]

        denom = max(np.percentile(best_scores, 95) - thr, 1.0)
        for score, aln, j in kept:
            g_f, g_o = aln.get_gapped_sequences()  # both AA strings with gaps
            mAA = compute_index_mapping(g_f, g_o)
            if len(mAA) < G_MINLEN: continue
            w_align = max(0.0, (score - thr) / denom) ** G_GAMMA
            w_templ = (G_TEMPL_W[j] ** G_TW_EXP)
            w = w_align * w_templ
            labs = G_TEMPL_LABELS[j]
            for it, iu in mAA.items():
                if 0 <= iu < len(labs) and 0 <= it < len(taa):
                    lab = float(labs[iu])
                    pred[it] += w * lab
                    cnt[it]  += w

    covered = cnt > 0
    if covered.any():
        pred[covered] /= cnt[covered]
        pred[~covered] = 0.0
    return idx, pred.tolist()

def run_subset(templates, targets, subset_name, skip_self=False):
    submat = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
    init_args = (
        templates["aa"], templates["labels"], templates["ids"], templates["weights"],
        submat, CUTOFF, GAP_PENALTY, skip_self,
        DENOM, MIN_ALIGN_LEN, TOPK_HITS, WEIGHT_GAMMA, TEMPLATE_WEIGHT_EXP, PCTL
    )
    jobs = [(i, targets["ids"][i], targets["aa"][i]) for i in range(len(targets["ids"]))]
    preds = [None]*len(jobs)
    with Pool(THREADS, initializer=init_worker, initargs=init_args) as pool:
        for i, p in pool.imap_unordered(predict_one, jobs, chunksize=8):
            preds[i] = p
    return dict(
        labels=targets["labels"],
        predictions=preds,
        weights=targets["weights"],
        ids=targets["ids"],
        splits=targets["split_tags"],
    )
    
def init_self_worker(db_aa, db_labels):
    global G_SELF_AA, G_SELF_LABELS
    G_SELF_AA = db_aa
    G_SELF_LABELS = [np.asarray(x, int) for x in db_labels]

def self_reconstruct_worker(i):
    # Reconstruct labels by aligning a chain to itself
    submat = align.SubstitutionMatrix.std_protein_matrix()
    taa = G_SELF_AA[i]
    labs = G_SELF_LABELS[i]
    # локальное выравнивание себя к себе
    aln_list = align.align_optimal(taa, taa, submat, gap_penalty=GAP_PENALTY, local=True)
    aln = aln_list[0]
    g_f, g_o = aln.get_gapped_sequences()
    mAA = compute_index_mapping(g_f, g_o)
    pred = np.zeros(len(taa), float)
    for it, iu in mAA.items():
        if 0 <= iu < len(labs) and 0 <= it < len(taa):
            pred[it] = float(labs[iu])
    return i, pred.tolist()


def run_upperbound_train(templates):
    jobs = list(range(len(templates["ids"])))
    preds = [None] * len(jobs)
    with Pool(THREADS, initializer=init_self_worker,
              initargs=(templates["aa"], templates["labels"])) as pool:
        for i, p in pool.imap_unordered(self_reconstruct_worker, jobs, chunksize=16):
            preds[i] = p
    return dict(
        labels=templates["labels"],
        predictions=preds,
        weights=templates["weights"],
        ids=templates["ids"],
        splits=templates["split_tags"],
    )


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    global CUTOFF, DENOM, MIN_ALIGN_LEN, TOPK_HITS, TEMPLATE_WEIGHT_EXP, THREADS
    parser.add_argument("--cv_fold", type=int, required=True, help="fold in [1..5]")
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON)
    parser.add_argument("--out_base", type=str, default=OUTPUT_DIR_BASE)
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    parser.add_argument("--denom", type=float, default=DENOM)
    parser.add_argument("--min_align_len", type=int, default=MIN_ALIGN_LEN)
    parser.add_argument("--topk_hits", type=int, default=TOPK_HITS)
    parser.add_argument("--template_weight_exp", type=float, default=TEMPLATE_WEIGHT_EXP)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()

    CUTOFF = args.cutoff
    DENOM = args.denom
    MIN_ALIGN_LEN = args.min_align_len
    TOPK_HITS = args.topk_hits
    TEMPLATE_WEIGHT_EXP = args.template_weight_exp
    THREADS = min(args.threads, cpu_count())

    cv = args.cv_fold
    assert 1 <= cv <= 5, "cv_fold must be in [1..5]"
    idx = cv - 1
    split_names = [f"split{i}" for i in range(1, 6)]
    train_splits = [split_names[(idx + j) % 5] for j in range(3)]
    val_split = [split_names[(idx + 3) % 5]]
    test_split = [split_names[(idx + 4) % 5]]

    print(f"[CV] fold={cv} | train={train_splits} | val={val_split} | test={test_split}")
    print(f"[CFG] cutoff={CUTOFF} denom={DENOM} minlen={MIN_ALIGN_LEN} topk={TOPK_HITS} tw_exp={TEMPLATE_WEIGHT_EXP}")

    data = load_splits(args.splits_json)
    templ = collect_from_splits(data, train_splits)
    targ_val = collect_from_splits(data, val_split)
    targ_test = collect_from_splits(data, test_split)

    # Train self-only upper bound
    upper = run_upperbound_train(templ)

    # Predictions
    outs_train_skip = run_subset(templ, templ, "train", skip_self=True)
    outs_train_noskip = run_subset(templ, templ, "train_noskip", skip_self=False)
    outs_val = run_subset(templ, targ_val, "validation", skip_self=False)
    outs_test = run_subset(templ, targ_test, "test", skip_self=False)

    model_name = (f"AA-seq-homology (cv={cv}, cutoff={CUTOFF}, denom={DENOM}, "
                  f"minlen={MIN_ALIGN_LEN}, topk={TOPK_HITS}, tw_exp={TEMPLATE_WEIGHT_EXP})")

    out_dir = f"{args.out_base}_cv{cv}"
    save_subset(out_dir, "train_upperbound",
                f"Upper bound (self-only): {model_name}",
                upper["labels"], upper["predictions"], upper["weights"],
                upper["ids"], upper["splits"], "upperbound")

    save_subset(out_dir, "validation",
                f"Enzyme active-site prediction [validation]: {model_name}",
                outs_val["labels"], outs_val["predictions"], outs_val["weights"],
                outs_val["ids"], outs_val["splits"], model_name)

    save_subset(out_dir, "test",
                f"Enzyme active-site prediction [test]: {model_name}",
                outs_test["labels"], outs_test["predictions"], outs_test["weights"],
                outs_test["ids"], outs_test["splits"], model_name)

    save_subset(out_dir, "train",
                f"Enzyme active-site prediction [train]: {model_name}",
                outs_train_skip["labels"], outs_train_skip["predictions"], outs_train_skip["weights"],
                outs_train_skip["ids"], outs_train_skip["splits"], model_name)

    save_subset(out_dir, "train_noskip",
                f"Enzyme active-site prediction [train no skip-self]: {model_name}",
                outs_train_noskip["labels"], outs_train_noskip["predictions"], outs_train_noskip["weights"],
                outs_train_noskip["ids"], outs_train_noskip["splits"], model_name)

    print("[DONE] All results saved.")

if __name__ == "__main__":
    main()
