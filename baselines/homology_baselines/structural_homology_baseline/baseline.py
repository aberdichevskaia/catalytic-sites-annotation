#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3Di-based structural-homology baseline with 5-fold CV
"""

import os
import json
import pickle
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import biotite.sequence.align as align
from biotite.sequence import ProteinSequence
from biotite.structure.alphabet import I3DSequence

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# -------------------- Hardcoded paths & params --------------------
SPLITS_JSON = "/home/iscb/wolfson/annab4/DB/all_proteins/structural_homology/3Di_DB_splits.json"

OUTPUT_DIR_BASE = "/home/iscb/wolfson/annab4/DB/all_proteins/structural_homology_baseline/results_adaptive_cutoff"

# Scoring/propagation knobs:
DENOMS = [2]            # adaptive thresholds: max_score/denom (>= hybrid threshold)
CUTOFF = 110               # hard minimal alignment score (3Di score scale)
PCTL = 95.0                # percentile of raw best-scores per target for hybrid threshold
MIN_ALIGN_LEN = 9          # minimal number of matched 3Di positions to accept a hit
TOPK_HITS = 32             # cap number of strongest hits per target (after threshold)
WEIGHT_GAMMA = 1.0         # non-linear emphasis on stronger alignments
EPS = 1e-6                 # numerical epsilon to avoid division by zero / ties
TEMPLATE_WEIGHT_EXP = 0.0  # exponent for per-template structure weight

# Local alignment gap penalties (Smith–Waterman in biotite)
GAP_PENALTY = (-8, -1)

# Parallelism
THREADS = min(256, cpu_count())

# Diagnostics
DIAG_EVERY = 50            # print score quantiles every ~N targets


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
    """Save pickle and PR-curve PNG for a subset (train/validation/test)."""
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
    print(f"[OK] {subset_key} PR-curve -> {png_path}")


# -------------------- Data preparation --------------------
def _to_i3d_list(seq_str_list):
    return [I3DSequence(s) for s in seq_str_list]


def _to_prot_list(seq_str_list):
    return [ProteinSequence(s) for s in seq_str_list]


def _to_int_labels(labels_list):
    return [list(map(int, labs)) for labs in labels_list]


def _to_int_mapping_list(maps):
    out = []
    for m in maps:
        out.append({int(k): int(v) for k, v in m.items()})
    return out


def load_splits(path):
    """Load JSON with keys 'split1'..'split5'."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def collect_from_splits(all_data, split_names):
    """
    Flatten fields from multiple split dicts.
    Returns dict with:
      ids, seq3di (I3D), aa_annot (ProteinSequence), labels (list[int]),
      mapping (dict[int->int]), weights (np.array), split_tags (list[str])
    """
    ids, seq3di, aa_annot, labels, mapping, weights, split_tags = \
        [], [], [], [], [], [], []

    for name in split_names:
        d = all_data[name]
        ids.extend([tuple(x) for x in d["db_ids"]])
        seq3di.extend(_to_i3d_list(d["db_3di_sequences"]))
        aa_annot.extend(_to_prot_list(d["db_annotated_aa_sequences"]))
        labels.extend(_to_int_labels(d["db_labels"]))
        mapping.extend(_to_int_mapping_list(d["db_index_mappings"]))
        weights.extend(d.get("structure_weights", [1.0] * len(d["db_ids"])))
        split_tags.extend([name] * len(d["db_ids"]))

    return {
        "ids": ids,
        "seq3di": seq3di,
        "aa_annot": aa_annot,
        "labels": labels,
        "mapping": mapping,
        "weights": np.array(weights, dtype=float),
        "split_tags": split_tags,
    }


# -------------------- Alignment & propagation --------------------
def compute_index_mapping(aligned_filtered, aligned_original):
    """Map indices from filtered AA seq to original AA seq using gapped strings."""
    mapping = {}
    i_f = 0
    i_o = 0
    for a, b in zip(aligned_filtered, aligned_original):
        if a != "-" and b != "-":
            mapping[i_f] = i_o
            i_f += 1
            i_o += 1
        elif a != "-" and b == "-":
            i_f += 1
        elif a == "-" and b != "-":
            i_o += 1
    return mapping


# Coarse AA classes for relaxed compatibility (partial credit)
AA_CLASS = {
    "A": "hyd", "V": "hyd", "L": "hyd", "I": "hyd", "M": "hyd",
    "F": "hyd", "W": "hyd", "Y": "hyd",
    "G": "pol", "S": "pol", "T": "pol", "C": "pol", "N": "pol",
    "Q": "pol", "P": "pol",
    "D": "neg", "E": "neg",
    "K": "pos", "R": "pos", "H": "pos",
    "X": "unk"
}
CLASS_PARTIAL_WEIGHT = 0.7  # weight when classes match but letters differ


def aa_compat_weight(a: str, b: str) -> float:
    """Return weight multiplier for AA pair similarity."""
    if a == b:
        return 1.0
    if a == "X" or b == "X":
        return 0.6  # unknown residue relaxed credit
    ca = AA_CLASS.get(a, "unk")
    cb = AA_CLASS.get(b, "unk")
    return CLASS_PARTIAL_WEIGHT if (ca != "unk" and ca == cb) else 0.0


# -------------------- Main predictor worker globals --------------------
G_TEMPL_3DI = None
G_TEMPL_AA = None
G_TEMPL_LABELS = None
G_TEMPL_MAP = None
G_TEMPL_IDS = None
G_TEMPL_STRUCT_W = None
G_SUBMAT = None
G_DENOMS = None
G_CUTOFF = None
G_GAP_PENALTY = None
G_SKIP_SELF = False


def init_worker(db_3di, db_aa, db_labels, db_map, db_ids, db_struct_w,
                submat, denoms, cutoff, gap_penalty, skip_self):
    """Initializer to attach templates & params to worker globals."""
    global G_TEMPL_3DI, G_TEMPL_AA, G_TEMPL_LABELS, G_TEMPL_MAP, G_TEMPL_IDS, G_TEMPL_STRUCT_W
    global G_SUBMAT, G_DENOMS, G_CUTOFF, G_GAP_PENALTY, G_SKIP_SELF
    G_TEMPL_3DI = db_3di
    G_TEMPL_AA = db_aa
    G_TEMPL_LABELS = [np.array(l) for l in db_labels]
    G_TEMPL_MAP = db_map
    G_TEMPL_IDS = db_ids
    G_TEMPL_STRUCT_W = np.asarray(db_struct_w, dtype=float)
    G_SUBMAT = submat
    G_DENOMS = denoms
    G_CUTOFF = cutoff
    G_GAP_PENALTY = gap_penalty
    G_SKIP_SELF = skip_self


def predict_one_target(args):
    """
    Predict labels for a single target by propagating from template DB.
    args = (target_idx, target_id, t3di, taa, tmap)
    Returns (target_idx, {denom: list_of_floats})
    """
    t_idx, t_id, t_3di, t_aa, t_map = args

    # 1) Scan all templates: collect best local alignments and their scores
    all_hits = []         # list of (score, alignment, templ_idx)
    raw_best_scores = []  # raw best-score per template for diagnostics

    for j, templ_3di in enumerate(G_TEMPL_3DI):
        # Optional leave-one-out skip for training subset
        if G_SKIP_SELF and t_id == G_TEMPL_IDS[j]:
            continue

        alns = align.align_optimal(
            t_3di, templ_3di, G_SUBMAT,
            gap_penalty=G_GAP_PENALTY,
            local=True
        )
        if not alns:
            continue
        best = alns[0]
        raw_best_scores.append(best.score)
        all_hits.append((best.score, best, j))

    # Sparse diagnostic print to understand score scale
    if (t_idx % DIAG_EVERY) == 0:
        if raw_best_scores:
            arr = np.array(raw_best_scores, float)
            print(
                f"[diag] target={t_id} n_aln={len(arr)} "
                f"max={arr.max():.3f} q50={np.percentile(arr,50):.3f} "
                f"q90={np.percentile(arr,90):.3f} q95={np.percentile(arr,95):.3f}",
                flush=True
            )
        else:
            print(f"[diag] target={t_id} n_aln=0", flush=True)

    # 2) Containers for position-wise aggregation and coverage
    pred_per_denom = {d: np.zeros(len(t_aa), float) for d in G_DENOMS}
    cnt_per_denom = {d: np.zeros(len(t_aa), float) for d in G_DENOMS}

    if all_hits:
        max_score = max(s for s, _, _ in all_hits)
        pctl_score = np.percentile(raw_best_scores, PCTL) if raw_best_scores else CUTOFF

        for d in G_DENOMS:
            # Soft hybrid threshold -> keep denom > 0
            thr = min(max(max_score / d, 0.0), pctl_score)
            thr = min(thr, max_score - 1.0)

            denom = max(np.percentile(raw_best_scores, 95) - thr, 1.0)
            kept = [(s, aln, j) for (s, aln, j) in all_hits if s > thr]
            kept.sort(key=lambda x: x[0], reverse=True)
            if TOPK_HITS is not None and len(kept) > TOPK_HITS:
                kept = kept[:TOPK_HITS]
            if not kept:
                # Fallback: use single best hit to avoid all-zero predictions
                best_s, best_aln, best_j = max(all_hits, key=lambda x: x[0])
                kept = [(best_s, best_aln, best_j)]
                thr = min(thr, best_s - EPS)
                denom = max(max_score - thr, EPS)

            if (t_idx % DIAG_EVERY) == 0:
                print(f"[diag] target={t_id} denom={d} thr={thr:.3f} max={max_score:.3f} kept={len(kept)}", flush=True)

            for score, aln, j in kept:
                # Map gapped-3Di indices: target3Di_idx -> templ3Di_idx
                g_f, g_o = aln.get_gapped_sequences()
                m3d = compute_index_mapping(g_f, g_o)
                if len(m3d) < MIN_ALIGN_LEN:
                    continue  # discard too-short local matches

                # Nonlinear score weight (strictly positive because score > thr)
                w_align = max(0.0, (score - thr) / denom) ** WEIGHT_GAMMA

                templ_aa = G_TEMPL_AA[j]           # ProteinSequence
                templ_labels = G_TEMPL_LABELS[j]   # np.array of {0,1}
                templ_map = G_TEMPL_MAP[j]         # 3Di->AA mapping for template
                templ_struct_w = G_TEMPL_STRUCT_W[j]  # scalar

                for idx_t3d, idx_u3d in m3d.items():
                    # Map 3Di positions back to AA indices
                    idx_t_aa = t_map.get(idx_t3d)
                    idx_u_aa = templ_map.get(idx_u3d)
                    if idx_t_aa is None or idx_u_aa is None:
                        continue

                    # AA compatibility weight (exact match=1, class match partial)
                    aw = aa_compat_weight(t_aa[idx_t_aa], templ_aa[idx_u_aa])
                    if aw <= 0.0:
                        continue

                    lab = float(templ_labels[idx_u_aa])

                    # Reweight by template rarity (structure weight)
                    w = w_align * aw * (templ_struct_w ** TEMPLATE_WEIGHT_EXP)

                    # Central position contribution (NO smoothing to neighbors)
                    pred_per_denom[d][idx_t_aa] += w * lab
                    cnt_per_denom[d][idx_t_aa] += w

    # 3) Finalize: average ONLY over covered positions
    for d in G_DENOMS:
        covered_mask = cnt_per_denom[d] > 0
        cov_ratio = covered_mask.mean() if len(covered_mask) > 0 else 0.0
        if cov_ratio < 0.05:
            print(f"[warn] low coverage {cov_ratio:.3f} for {t_id} (denom={d})", flush=True)

        pred = pred_per_denom[d]
        cnt = cnt_per_denom[d]
        if covered_mask.any():
            pred[covered_mask] = pred[covered_mask] / cnt[covered_mask]
            pred[~covered_mask] = 0.0
        pred_per_denom[d] = pred.tolist()

    return t_idx, pred_per_denom


def run_subset_predictions(templates, targets, subset_name, do_skip_self=False):
    """
    templates: dict from collect_from_splits() for training DB
    targets: dict from collect_from_splits() for evaluation subset
    subset_name: 'train'|'validation'|'test'
    """
    # Prepare worker globals
    submat = align.SubstitutionMatrix.std_3di_matrix()

    init_args = (
        templates["seq3di"],
        templates["aa_annot"],
        templates["labels"],
        templates["mapping"],
        templates["ids"],
        templates["weights"],          # pass per-template structure weights
        submat,
        DENOMS,
        CUTOFF,
        GAP_PENALTY,
        do_skip_self,
    )

    # Build target job list
    jobs = []
    for i in range(len(targets["ids"])):  # one job per target chain
        jobs.append((
            i,
            targets["ids"][i],
            targets["seq3di"][i],
            targets["aa_annot"][i],
            targets["mapping"][i],
        ))

    # Parallel predict
    preds_by_denom = {d: [None] * len(jobs) for d in DENOMS}
    with Pool(THREADS, initializer=init_worker, initargs=init_args) as pool:
        for t_idx, pred_per_denom in pool.imap_unordered(predict_one_target, jobs, chunksize=8):
            for d in DENOMS:
                preds_by_denom[d][t_idx] = pred_per_denom[d]

    # Build outputs per denom
    outputs = {}
    for d in DENOMS:
        outputs[d] = {
            "labels": targets["labels"],
            "predictions": preds_by_denom[d],
            "weights": targets["weights"],
            "ids": targets["ids"],
            "splits": targets["split_tags"],
        }
    return outputs


# -------------------- Upperbound worker globals & helpers --------------------
UB_SEQ3DI = None
UB_LABELS = None
UB_MAP = None
UB_IDS = None
UB_SUBMAT = None
UB_ID2IDX = None  # dict: id(tuple) -> index


def init_upper_worker(seq3di, labels, mapping, ids, submat):
    """Initializer for train upperbound (self-only) worker."""
    global UB_SEQ3DI, UB_LABELS, UB_MAP, UB_IDS, UB_SUBMAT, UB_ID2IDX
    UB_SEQ3DI = seq3di
    UB_LABELS = [np.array(l) for l in labels]
    UB_MAP = mapping
    UB_IDS = ids
    UB_SUBMAT = submat
    UB_ID2IDX = {ids[i]: i for i in range(len(ids))}


def upper_self_only(job):
    """
    Self-only reconstruction for upper bound.
    job = (i, tid, t3, taa, tmap)
    Returns (i, {denom: list_of_floats})  # same preds for all DENOMS
    """
    i, tid, t3, taa, tmap = job
    j = UB_ID2IDX.get(tid)
    if j is None:
        # No exact self found (should not happen for train); return zeros
        return i, {d: [0.0] * len(taa) for d in DENOMS}

    # Align 3Di to itself
    aln = align.align_optimal(
        t3, UB_SEQ3DI[j], UB_SUBMAT, gap_penalty=GAP_PENALTY, local=True
    )[0]
    g_f, g_o = aln.get_gapped_sequences()
    m3d = compute_index_mapping(g_f, g_o)

    pred = np.zeros(len(taa), float)
    tlabels = UB_LABELS[j]      # np.array of {0,1}
    tmap_u = UB_MAP[j]          # 3Di->AA mapping for this template (self)

    for idx_t3d, idx_u3d in m3d.items():
        it = tmap.get(idx_t3d)
        iu = tmap_u.get(idx_u3d)
        if it is None or iu is None:
            continue
        pred[it] = float(tlabels[iu])

    # Same vector for all DENOMS (doesn't matter for upperbound)
    return i, {d: pred.tolist() for d in DENOMS}


def run_train_upperbound(templates):
    """
    Upper bound sanity: for each train target, use ONLY its self-alignment to
    reconstruct labels. AUCPR should be ~1.0 if mapping & pipeline are OK.
    """
    submat = align.SubstitutionMatrix.std_3di_matrix()

    jobs = [
        (
            i,
            templates["ids"][i],
            templates["seq3di"][i],
            templates["aa_annot"][i],
            templates["mapping"][i],
        )
        for i in range(len(templates["ids"]))
    ]

    preds = {d: [None] * len(jobs) for d in DENOMS}

    with Pool(
        THREADS,
        initializer=init_upper_worker,
        initargs=(
            templates["seq3di"],
            templates["labels"],
            templates["mapping"],
            templates["ids"],
            submat,
        ),
    ) as pool:
        for i, pred_d in pool.imap_unordered(upper_self_only, jobs, chunksize=16):
            for d in DENOMS:
                preds[d][i] = pred_d[d]

    return {
        d: {
            "labels": templates["labels"],
            "predictions": preds[d],
            "weights": templates["weights"],
            "ids": templates["ids"],
            "splits": templates["split_tags"],
        }
        for d in DENOMS
    }


def slice_dict(info):
    res = {}
    for key, val in info.items():
        res[key] = val[:10]
    return res


# -------------------- CV fold logic & main --------------------
def main():
    global TEMPLATE_WEIGHT_EXP
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_fold", type=int, required=True,
                        help="CV fold index in [1..5]")
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--template_weight_exp", type=float, default=TEMPLATE_WEIGHT_EXP,
                        help="Exponent for template structure weight reweighting")
    args = parser.parse_args()

    # allow overriding from CLI
    TEMPLATE_WEIGHT_EXP = float(args.template_weight_exp)

    assert args.cv_fold is not None, "For CV you must pass --cv_fold (1..5)."
    cv_fold = int(args.cv_fold)
    assert 1 <= cv_fold <= 5, "--cv_fold must be in [1..5]"
    idx_fold = cv_fold - 1

    # train: 3 folds rolling window; validation: +3; test: +4
    train_indices = [(idx_fold + j) % 5 for j in range(3)]
    validate_index = (idx_fold + 3) % 5
    test_index = (idx_fold + 4) % 5

    split_names = [f"split{i}" for i in range(1, 6)]
    train_splits = [split_names[i] for i in train_indices]
    val_split = [split_names[validate_index]]
    test_split = [split_names[test_index]]

    print(f"[CV] fold={cv_fold} | train={train_splits} | val={val_split} | test={test_split}")
    print(f"[CFG] TEMPLATE_WEIGHT_EXP={TEMPLATE_WEIGHT_EXP}")

    # Load split JSON
    all_data = load_splits(SPLITS_JSON)

    # Collect template DB and targets
    templates = collect_from_splits(all_data, train_splits)
    targets_val = collect_from_splits(all_data, val_split)
    targets_test = collect_from_splits(all_data, test_split)

    if args.check:
        templates = slice_dict(templates)
        targets_val = slice_dict(targets_val)
        targets_test = slice_dict(targets_test)

    # Run upper bound sanity on train (should be ~1.0 AUCPR)
    upper = run_train_upperbound(templates)

    # Predict on validation/test; NOTE: here do_skip_self=False (as in your last version)
    outs_train = run_subset_predictions(templates, templates, "train", do_skip_self=True)
    outs_train_noskip = run_subset_predictions(templates, templates, "train", do_skip_self=False)
    outs_val = run_subset_predictions(templates, targets_val, "validation", do_skip_self=False)
    outs_test = run_subset_predictions(templates, targets_test, "test", do_skip_self=False)

    # Save results per denom and subset
    for d in DENOMS:
        out_dir = f"{OUTPUT_DIR_BASE}_den{d}_cv{cv_fold}"
        model_name = (f"3Di-homology-baseline (cutoff={CUTOFF}, denom={d}, pct={PCTL}, "
                      f"minlen={MIN_ALIGN_LEN}, cv={cv_fold}, tw_exp={TEMPLATE_WEIGHT_EXP})")

        # Sanity: train upper bound (self only)
        save_subset_results(
            output_dir=out_dir,
            subset_key="train_upperbound",
            title=f"Upper bound (self only): cv={cv_fold}, denom={d}",
            labels=upper[d]["labels"],
            predictions=upper[d]["predictions"],
            weights=upper[d]["weights"],
            ids=upper[d]["ids"],
            splits=upper[d]["splits"],
            model_name="upperbound",
            batch_size=0,
        )

        # Validation
        save_subset_results(
            output_dir=out_dir,
            subset_key="validation",
            title=f"Enzyme active site prediction: {model_name} [validation]",
            labels=outs_val[d]["labels"],
            predictions=outs_val[d]["predictions"],
            weights=outs_val[d]["weights"],
            ids=outs_val[d]["ids"],
            splits=outs_val[d]["splits"],
            model_name=model_name,
            batch_size=0,
        )

        # Test
        save_subset_results(
            output_dir=out_dir,
            subset_key="test",
            title=f"Enzyme active site prediction: {model_name} [test]",
            labels=outs_test[d]["labels"],
            predictions=outs_test[d]["predictions"],
            weights=outs_test[d]["weights"],
            ids=outs_test[d]["ids"],
            splits=outs_test[d]["splits"],
            model_name=model_name,
            batch_size=0,
        )

        # Train 
        save_subset_results(
            output_dir=out_dir,
            subset_key="train",
            title=f"Enzyme active site prediction: {model_name} [train]",
            labels=outs_train[d]["labels"],
            predictions=outs_train[d]["predictions"],
            weights=outs_train[d]["weights"],
            ids=outs_train[d]["ids"],
            splits=outs_train[d]["splits"],
            model_name=model_name,
            batch_size=0,
        )
        
        # Train (no skip-self)
        save_subset_results(
            output_dir=out_dir,
            subset_key="train no skip-self",
            title=f"Enzyme active site prediction: {model_name} [train]",
            labels=outs_train_noskip[d]["labels"],
            predictions=outs_train_noskip[d]["predictions"],
            weights=outs_train_noskip[d]["weights"],
            ids=outs_train_noskip[d]["ids"],
            splits=outs_train_noskip[d]["splits"],
            model_name=model_name,
            batch_size=0,
        )

    print("[DONE] All results saved.")


if __name__ == "__main__":
    main()
