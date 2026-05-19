#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5-fold split "as before" (minimax objective), AF-only, all weights = 1.

Inputs:
- protein_table.json
- train_raw.txt
- test.txt
- cluster_level_1_cluster.tsv
- cluster_level_2_cluster.tsv

Outputs:
- split1.txt ... split5.txt
- test.txt (copied)
- dataset.csv
- split_summary.json
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protein_table_json", required=True)
    p.add_argument("--train_raw_txt", required=True)
    p.add_argument("--test_txt", required=True)
    p.add_argument("--cluster1_tsv", required=True)
    p.add_argument("--cluster2_tsv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_comp_size", type=int, default=3000)
    p.add_argument("--refine_1opt", action="store_true", default=False)
    p.add_argument("--refine_2opt", action="store_true", default=False)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_annotation_txt(path: str) -> Dict[str, List[str]]:
    """
    Returns chain_id -> list of raw annotation lines.
    chain_id is like 'A0A0H2UNG0_A'.
    """
    out: Dict[str, List[str]] = {}
    curr = None
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if curr is not None:
                    out[curr] = buf
                curr = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if curr is not None:
            out[curr] = buf
    return out


def save_annotation_txt(blocks: Dict[str, List[str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for cid in sorted(blocks.keys()):
            f.write(f">{cid}\n")
            for line in blocks[cid]:
                f.write(f"{line}\n")


def parse_chain_seq_and_labels(ann_lines: List[str]) -> Tuple[str, List[int]]:
    seq = []
    labs = []
    for line in ann_lines:
        parts = line.strip().split()
        if len(parts) != 4:
            raise ValueError(f"Bad annotation line: {line}")
        seq.append(parts[2])
        labs.append(int(parts[3]))
    return "".join(seq), labs


def valid_ec_number(ec: str) -> bool:
    if ec == "not found":
        return False
    parts = ec.split(".")
    if len(parts) < 3:
        return False
    return all(p.isdigit() for p in parts[:3])


def trunc_ec3(ec: str) -> str:
    return ".".join(ec.split(".")[:3]) if valid_ec_number(ec) else "not found"


# Chemotype definition (same logic)
N_CLASSES = 8


def get_catalytic_class(residues: List[str]) -> int:
    if any(r in residues for r in "ILMVWF"):
        return 0
    if any(r in residues for r in "AGP"):
        return 1
    if any(r in residues for r in "QN"):
        return 2
    if any(r in residues for r in "KR"):
        return 3
    if any(r == "S" for r in residues):
        return 4
    if any(r == "T" for r in residues):
        return 5
    if any(r in residues for r in "DE"):
        return 6
    return 7


def discrepancy_metric(fold_w: np.ndarray, ideal_w: np.ndarray) -> float:
    denom = np.maximum(ideal_w, 1e-12)
    diff = np.abs(fold_w - ideal_w) / denom
    mul = np.array([N_CLASSES * 3] + [1] * N_CLASSES, dtype=float)
    return float(np.dot(mul, diff))


def global_score(fw_list: List[np.ndarray], ideal_fold_weights: np.ndarray) -> float:
    return max(discrepancy_metric(fw, ideal_fold_weights) for fw in fw_list)


def local_refine_full_1opt(components_weights, fold_weights, component_fold, ideal_fold_weights, max_iters=50):
    eps = 1e-12
    n_comp = components_weights.shape[0]
    for _ in range(max_iters):
        base = global_score(fold_weights, ideal_fold_weights)
        best_gain, best_move = 0.0, None
        for i in range(n_comp):
            k = component_fold[i]
            for kk in range(len(fold_weights)):
                if kk == k:
                    continue
                fold_weights[k] -= components_weights[i]
                fold_weights[kk] += components_weights[i]
                sc = global_score(fold_weights, ideal_fold_weights)
                fold_weights[kk] -= components_weights[i]
                fold_weights[k] += components_weights[i]
                gain = base - sc
                if gain > best_gain + eps:
                    best_gain, best_move = gain, (i, k, kk)
        if best_move is None:
            break
        i, k, kk = best_move
        fold_weights[k] -= components_weights[i]
        fold_weights[kk] += components_weights[i]
        component_fold[i] = kk


def local_refine_swaps_2opt(components_weights, fold_weights, component_fold, ideal_fold_weights, max_pairs_per_fold=60):
    eps = 1e-12
    base = global_score(fold_weights, ideal_fold_weights)
    best_gain, best_swap = 0.0, None
    comp_in_fold = [np.where(np.array(component_fold) == f)[0] for f in range(len(fold_weights))]

    small_in_fold = []
    for f in range(len(fold_weights)):
        idx = comp_in_fold[f]
        if idx.size == 0:
            small_in_fold.append(idx)
            continue
        order = idx[np.argsort(components_weights[idx, 0])]
        small_in_fold.append(order[:max_pairs_per_fold])

    for a in range(len(fold_weights)):
        for b in range(a + 1, len(fold_weights)):
            A, B = small_in_fold[a], small_in_fold[b]
            if A.size == 0 or B.size == 0:
                continue
            for i in A:
                for j in B:
                    fold_weights[a] -= components_weights[i]
                    fold_weights[b] += components_weights[i]
                    fold_weights[b] -= components_weights[j]
                    fold_weights[a] += components_weights[j]

                    sc = global_score(fold_weights, ideal_fold_weights)

                    fold_weights[a] += components_weights[i]
                    fold_weights[b] -= components_weights[i]
                    fold_weights[b] += components_weights[j]
                    fold_weights[a] -= components_weights[j]

                    gain = base - sc
                    if gain > best_gain + eps:
                        best_gain, best_swap = gain, (a, b, i, j)

    if best_swap is None:
        return
    a, b, i, j = best_swap
    fold_weights[a] -= components_weights[i]
    fold_weights[b] += components_weights[i]
    fold_weights[b] -= components_weights[j]
    fold_weights[a] += components_weights[j]
    component_fold[i] = b
    component_fold[j] = a


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.protein_table_json, "r", encoding="utf-8") as f:
        protein_table = json.load(f)

    train_blocks = parse_annotation_txt(args.train_raw_txt)
    test_blocks = parse_annotation_txt(args.test_txt)

    # Train UIDs are prefixes of train_blocks (UID_A)
    train_uids = sorted({cid.rsplit("_", 1)[0] for cid in train_blocks.keys()})

    # Load clustering
    cl1 = pd.read_csv(args.cluster1_tsv, sep="\t", header=None, names=["Cluster_1", "Sequence_ID"], dtype=str)
    cl2 = pd.read_csv(args.cluster2_tsv, sep="\t", header=None, names=["Cluster_2", "Centroid_ID"], dtype=str)

    uid_to_c1: Dict[str, str] = {}
    train_uid_set = set(train_uids)
    for _, r in cl1.iterrows():
        if r["Sequence_ID"] in train_uid_set:
            uid_to_c1[r["Sequence_ID"]] = r["Cluster_1"]

    centroid_to_c2 = {r["Centroid_ID"]: r["Cluster_2"] for _, r in cl2.iterrows()}

    uid_to_c2: Dict[str, str] = {}
    for uid in train_uids:
        if uid not in uid_to_c1:
            raise RuntimeError(f"Train UID missing from cluster_level_1_cluster.tsv: {uid}")
        c1 = uid_to_c1[uid]
        if c1 not in centroid_to_c2:
            raise RuntimeError(f"Cluster_1 centroid missing from cluster_level_2_cluster.tsv: {c1}")
        uid_to_c2[uid] = centroid_to_c2[c1]

    # --- "Graph" connected components via union-by-group (equivalent to cliques) ---
    parent = {u: u for u in train_uids}
    size = {u: 1 for u in train_uids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    by_c2 = defaultdict(list)
    by_ec = defaultdict(list)

    for uid in train_uids:
        by_c2[uid_to_c2[uid]].append(uid)
        ec3 = trunc_ec3(protein_table[uid].get("EC_number", "not found"))
        if ec3 != "not found":
            by_ec[ec3].append(uid)

    for group in by_c2.values():
        if len(group) > 1:
            a = group[0]
            for u in group[1:]:
                union(a, u)

    for group in by_ec.values():
        if len(group) > 1:
            a = group[0]
            for u in group[1:]:
                union(a, u)

    comps_map = defaultdict(list)
    for uid in train_uids:
        comps_map[find(uid)].append(uid)

    components = [sorted(m) for m in comps_map.values()]
    components.sort(key=len, reverse=True)

    # --- Round-robin subsampling for huge components (same idea as your code) ---
    subsampled_components: List[List[str]] = []
    for comp in components:
        if len(comp) <= args.max_comp_size:
            subsampled_components.append(comp)
            continue

        by_c2_local = defaultdict(list)
        for uid in comp:
            by_c2_local[uid_to_c2[uid]].append(uid)

        # Keep deterministic behavior: shuffle inside each c2 bucket with a fixed seed
        for lst in by_c2_local.values():
            random.shuffle(lst)

        keys = list(by_c2_local.keys())
        idxs = {k: 0 for k in keys}
        sel: List[str] = []

        while len(sel) < args.max_comp_size:
            progressed = False
            for k in keys:
                if idxs[k] < len(by_c2_local[k]):
                    sel.append(by_c2_local[k][idxs[k]])
                    idxs[k] += 1
                    progressed = True
                if len(sel) >= args.max_comp_size:
                    break
            if not progressed:
                break
        subsampled_components.append(sel)

    # --- Catalytic classes from annotations ---
    catalytic_class: Dict[str, int] = {}
    for uid in train_uids:
        cid = f"{uid}_A"
        seq, labs = parse_chain_seq_and_labels(train_blocks[cid])
        residues = [aa for aa, lab in zip(seq, labs) if lab == 1]
        catalytic_class[uid] = get_catalytic_class(residues)

    # --- Components weights (all weights=1) ---
    n_comp = len(subsampled_components)
    components_weights = np.zeros((n_comp, N_CLASSES + 1), dtype=float)
    for i, comp in enumerate(subsampled_components):
        components_weights[i, 0] = float(len(comp))
        for uid in comp:
            cl = catalytic_class[uid]
            components_weights[i, cl + 1] += 1.0

    dataset_weights = components_weights.sum(axis=0)
    ideal_fold_weights = dataset_weights / float(args.n_splits)

    # --- Greedy minimax fold assignment (as in your code) ---
    fold_weights = [np.zeros(N_CLASSES + 1, dtype=float) for _ in range(args.n_splits)]
    component_fold = [-1] * n_comp

    order = np.argsort(-components_weights[:, 0])
    for j in range(min(args.n_splits, len(order))):
        i = int(order[j])
        component_fold[i] = j
        fold_weights[j] += components_weights[i]

    for pos in range(args.n_splits, len(order)):
        i = int(order[pos])
        candidates = []
        for k in range(args.n_splits):
            fold_weights[k] += components_weights[i]
            sc = global_score(fold_weights, ideal_fold_weights)
            candidates.append((sc, k))
            fold_weights[k] -= components_weights[i]

        best_sc, best_k = min(candidates, key=lambda x: x[0])

        ties = [k for sc, k in candidates if np.isclose(sc, best_sc)]
        if len(ties) > 1:
            best_k = min(ties, key=lambda k: fold_weights[k][0])

        component_fold[i] = best_k
        fold_weights[best_k] += components_weights[i]

    if args.refine_1opt:
        local_refine_full_1opt(components_weights, fold_weights, component_fold, ideal_fold_weights, max_iters=80)
    if args.refine_2opt:
        local_refine_swaps_2opt(components_weights, fold_weights, component_fold, ideal_fold_weights, max_pairs_per_fold=80)

    assert np.allclose(np.sum(fold_weights, axis=0), dataset_weights), "Fold weights don't sum to dataset!"

    # --- Build mapping uid -> split{i} and uid -> component_id ---
    split_sets = [set() for _ in range(args.n_splits)]
    for comp_idx, comp in enumerate(subsampled_components):
        split_sets[component_fold[comp_idx]].update(comp)

    set_mapping: Dict[str, str] = {}
    component_id_map: Dict[str, int] = {}
    for comp_idx, comp in enumerate(subsampled_components, start=1):
        for uid in comp:
            component_id_map[uid] = comp_idx

    for i, s in enumerate(split_sets, start=1):
        for uid in s:
            set_mapping[uid] = f"split{i}"

    # --- Write split{i}.txt ---
    split_data = [dict() for _ in range(args.n_splits)]
    for cid, ann in train_blocks.items():
        uid = cid.rsplit("_", 1)[0]
        st = set_mapping.get(uid)
        if st is None:
            continue
        idx = int(st.replace("split", "")) - 1
        split_data[idx][cid] = ann

    for i, d in enumerate(split_data, start=1):
        save_annotation_txt(d, os.path.join(args.out_dir, f"split{i}.txt"))

    save_annotation_txt(test_blocks, os.path.join(args.out_dir, "test.txt"))

    # --- dataset.csv ---
    rows = []
    for cid in sorted(train_blocks.keys()):
        uid = cid.rsplit("_", 1)[0]
        rows.append({
            "Sequence_ID": cid,
            "Cluster_1": uid_to_c1[uid],
            "Cluster_2": uid_to_c2[uid],
            "Set_Type": set_mapping[uid],
            "EC_number": protein_table[uid].get("EC_number", "not found"),
            "Component_ID": component_id_map.get(uid, "N/A"),
            "full_name": protein_table[uid].get("full_name", ""),
            "W_Cluster_2": 1.0,
            "W_Cluster_1": 1.0,
            "W_Sequence": 1.0,
            "W_Structure": 1.0,
        })

    for cid in sorted(test_blocks.keys()):
        uid = cid.rsplit("_", 1)[0]
        rows.append({
            "Sequence_ID": cid,
            "Cluster_1": "N/A",
            "Cluster_2": "N/A",
            "Set_Type": "test",
            "EC_number": protein_table.get(uid, {}).get("EC_number", "not found"),
            "Component_ID": "N/A",
            "full_name": protein_table.get(uid, {}).get("full_name", ""),
            "W_Cluster_2": 1.0,
            "W_Cluster_1": 1.0,
            "W_Sequence": 1.0,
            "W_Structure": 1.0,
        })

    df = pd.DataFrame(rows, columns=[
        "Sequence_ID", "Cluster_1", "Cluster_2", "Set_Type",
        "EC_number", "Component_ID", "full_name",
        "W_Cluster_2", "W_Cluster_1", "W_Sequence", "W_Structure"
    ])
    df.to_csv(os.path.join(args.out_dir, "dataset.csv"), index=False)

    summary = {
        "seed": args.seed,
        "n_train_uids": len(train_uids),
        "n_test_chains": len(test_blocks),
        "n_components_before_subsample": len(components),
        "n_components_after_subsample": len(subsampled_components),
        "max_comp_size": args.max_comp_size,
        "ideal_fold_weights": ideal_fold_weights.tolist(),
        "fold_weights": [fw.tolist() for fw in fold_weights],
        "fold_sizes_uids": [len(s) for s in split_sets],
        "refine_1opt": args.refine_1opt,
        "refine_2opt": args.refine_2opt,
    }
    with open(os.path.join(args.out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] 5-fold split created")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()