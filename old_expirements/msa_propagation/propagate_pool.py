#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import glob
from collections import defaultdict
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix


# -----------------------------------------------------------------------------
# Constants and globals (for Pool initializers)
# -----------------------------------------------------------------------------
MIN_QUORUM = 2
NEARBY_WINDOW = 10

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY-")
AMBIGUOUS_MAP = {
    "U": "C",  # Selenocysteine → Cys
    "O": "K",  # Pyrrolysine → Lys
    "B": "D",  # Asx (D or N) → Asp
    "Z": "E",  # Glx (E or Q) → Glu
    "J": "L",  # Xle (I or L) → Leu
    "X": "A",  # unknown → Ala (placeholder)
    "*": ""    # stop codon → drop
}

# Globals for workers
G_UNIPROT_SEQS: Dict[str, str] = {}
G_PROPAGATED_BASE: Dict[str, List[int]] = {}
G_THRESHOLD: float = 0.5
G_ALIGNMENT_TYPE: str = "TMAlign"  # or "3di"

G_PDB_TO_UNIPROT: Dict[str, str] = {}
G_PROPAGATED_FINAL: Dict[str, List[int]] = {}


# -----------------------------------------------------------------------------
# Parsers and basic IO
# -----------------------------------------------------------------------------
def parse_annotation_file(path: str) -> Dict[str, Dict[int, Tuple[str, int]]]:
    """Parse annotation flat file into {chain_id: {pos: (aa, label)}}."""
    ann: Dict[str, Dict[int, Tuple[str, int]]] = {}
    with open(path) as f:
        current = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:]
                ann[current] = {}
            else:
                _, pos, aa, lab = line.split()
                ann[current][int(pos)] = (aa, int(lab))
    return ann


def read_a3m(fn: str):
    """Read A3M; return (headers, ids, seqs)."""
    headers, ids, seqs = [], [], []
    with open(fn) as f:
        cur = []
        for l in f:
            l = l.rstrip()
            if not l:
                continue
            if l.startswith(">"):
                headers.append(l)
                ids.append(l[1:].split()[0])
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(l)
        if cur:
            seqs.append("".join(cur))
    return headers, ids, seqs


def parse_a3m_header(header: str) -> int:
    """Extract start index from A3M header; fall back to 0 for centroids."""
    parts = header[1:].split()
    if len(parts) < 8:
        return 0
    try:
        return int(parts[7])
    except ValueError:
        return 0


def a3m_to_aligned_and_col2pos(a3mseq: str, original_seq: str, start_index: int = 0):
    """
    Convert raw A3M sequence to aligned (only uppercase + '-') and build col->pos map.
    Lowercase letters advance original sequence but do not occupy an alignment column.
    """
    aligned_seq, col2pos = [], {}
    col_counter = -1
    seq_counter = -1
    for aa in a3mseq:
        if aa != "-":
            seq_counter += 1
        if aa.isupper() or aa == "-":
            col_counter += 1
            aligned_seq.append(aa)
            if aa != "-":
                col2pos[col_counter] = seq_counter + start_index
    return "".join(aligned_seq), col2pos


# -----------------------------------------------------------------------------
# Sequence utilities
# -----------------------------------------------------------------------------
def sanitize_sequence(seq: str) -> str:
    out = []
    for aa in seq:
        if aa in STANDARD_AAS:
            out.append(aa)
        elif aa in AMBIGUOUS_MAP and AMBIGUOUS_MAP[aa]:
            out.append(AMBIGUOUS_MAP[aa])
    return "".join(out)


def map_catalytic_sites(uniprot_seq: str, pdb_seq: str, uniprot_labels: List[int]) -> List[int]:
    """
    Local alignment PDB→UniProt, labels propagation.
    """
    u = sanitize_sequence(uniprot_seq)
    p = sanitize_sequence(pdb_seq)

    alpha = ProteinSequence.alphabet
    mat = np.full((len(alpha), len(alpha)), -1, int)
    np.fill_diagonal(mat, 1)
    sm = SubstitutionMatrix(alpha, alpha, mat)

    a_u, a_p = ProteinSequence(u), ProteinSequence(p)
    aligns = align_optimal(a_u, a_p, sm, gap_penalty=(-0.75, -0.5), local=True)
    if not aligns:
        return [0] * len(p)
    gu, gp = aligns[0].get_gapped_sequences()

    res = [0] * len(p)
    iu = ip = 0
    for au, ap in zip(gu, gp):
        if au != "-" and ap != "-":
            if au == ap and iu < len(uniprot_labels):
                res[ip] = uniprot_labels[iu]
            iu += 1
            ip += 1
        elif au != "-" and ap == "-":
            iu += 1
        elif au == "-" and ap != "-":
            ip += 1
    return res


# -----------------------------------------------------------------------------
# Pool initializers
# -----------------------------------------------------------------------------
def init_pool_msa(uniprot_seqs, propagated_base, threshold: float, alignment_type: str):
    """Initializer for MSA processing workers."""
    global G_UNIPROT_SEQS, G_PROPAGATED_BASE, G_THRESHOLD, G_ALIGNMENT_TYPE
    G_UNIPROT_SEQS = uniprot_seqs
    G_PROPAGATED_BASE = propagated_base
    G_THRESHOLD = threshold
    G_ALIGNMENT_TYPE = alignment_type


def init_pool_map(uniprot_seqs, propagated_final, pdb_to_uniprot):
    """Initializer for mapping-to-PDB workers."""
    global G_UNIPROT_SEQS, G_PROPAGATED_FINAL, G_PDB_TO_UNIPROT
    G_UNIPROT_SEQS = uniprot_seqs
    G_PROPAGATED_FINAL = propagated_final
    G_PDB_TO_UNIPROT = pdb_to_uniprot


# -----------------------------------------------------------------------------
# Worker functions
# -----------------------------------------------------------------------------
def process_msa_file(fn: str) -> Tuple[Dict[str, Dict[int, int]], bool]:
    """
    Process single MSA file:
    - build aligned sequences and col->pos maps for all members
    - compute per-column consensus and propose label=1 updates
    Returns (proposals, skipped_flag).
    proposals: {uniprot_id: {pos: 1, ...}, ...}
    """
    headers, ids, raw = read_a3m(fn)

    aligned: Dict[str, str] = {}
    col2pos: Dict[str, Dict[int, int]] = {}

    for header, hid, seq in zip(headers, ids, raw):
        full = G_UNIPROT_SEQS.get(hid)
        if full is None:
            return {}, True
        start = parse_a3m_header(header)
        a_seq, mapping = a3m_to_aligned_and_col2pos(seq, full, start)
        aligned[hid] = a_seq
        col2pos[hid] = mapping

    members = [h for h in aligned if h in G_PROPAGATED_BASE]
    if len(members) < 2:
        return {}, True

    proposals: Dict[str, Dict[int, int]] = defaultdict(dict)
    L = max(len(aligned[h]) for h in members)

    for c in range(L):
        aa_labels: Dict[str, List[int]] = defaultdict(list)
        for hid in members:
            seq_c = aligned[hid]
            if c < len(seq_c) and seq_c[c] != "-":
                pos = col2pos[hid].get(c)
                if pos is not None and pos < len(G_PROPAGATED_BASE[hid]):
                    aa_labels[seq_c[c]].append(G_PROPAGATED_BASE[hid][pos])

        if not aa_labels:
            continue

        most, labs = max(aa_labels.items(), key=lambda kv: len(kv[1]))
        if len(labs) < MIN_QUORUM:
            continue
        flag = int(sum(labs) / len(labs) >= G_THRESHOLD)
        if flag != 1:
            continue

        # Propose propagation (with nearby same-AA check against base labels)
        for hid in members:
            if c >= len(aligned[hid]) or aligned[hid][c] != most:
                continue
            pos = col2pos[hid].get(c)
            if pos is None or pos >= len(G_PROPAGATED_BASE[hid]):
                continue

            # Conservative skip if a same-AA catalytic residue is nearby in base labels
            seq = G_UNIPROT_SEQS[hid]
            labels_list = G_PROPAGATED_BASE[hid]
            start = max(0, pos - NEARBY_WINDOW)
            end = min(len(seq) - 1, pos + NEARBY_WINDOW)
            found_nearby = any(
                (q != pos) and labels_list[q] == 1 and seq[q] == most
                for q in range(start, end + 1)
            )
            if found_nearby:
                continue

            proposals[hid][pos] = 1

    return proposals, False


def map_chain_worker(item: Tuple[str, Dict[int, Tuple[str, int]]]) -> Optional[Tuple[str, List[int]]]:
    """
    Map UniProt-level propagated labels to a single PDB chain.
    item: (chain_id, {pos: (aa, label)})
    Returns (chain_id, labels_list) or None if mapping not possible.
    """
    cid, pm = item
    pid = cid.split("_")[0]
    if pid in G_UNIPROT_SEQS:
        uid = pid
    else:
        uid = G_PDB_TO_UNIPROT.get(pid)
    if not uid:
        return None
    seq_u = G_UNIPROT_SEQS.get(uid)
    if not seq_u:
        return None
    labels = G_PROPAGATED_FINAL.get(uid)
    if labels is None:
        return None

    pdb_seq = "".join(aa for _, (aa, _) in sorted(pm.items()))
    mapped = map_catalytic_sites(seq_u, pdb_seq, labels)
    return cid, mapped


# -----------------------------------------------------------------------------
# Ann saving and comparison/report helpers
# -----------------------------------------------------------------------------
def save_ann(data: Dict[str, list], path: str, ann_chain: Dict[str, Dict[int, Tuple[str, int]]]):
    """Save annotations in the standard flat format used elsewhere."""
    with open(path, "w") as f:
        for cid, labs in data.items():
            f.write(f">{cid}\n")
            for i, res in enumerate(sorted(ann_chain[cid].keys())):
                aa, _ = ann_chain[cid][res]
                f.write(f"{cid.split('_')[1]} {res} {aa} {labs[i]}\n")


def build_before_after_lists(
    ann_chain: Dict[str, Dict[int, Tuple[str, int]]],
    prop_chain: Dict[str, List[int]],
) -> Tuple[Dict[str, List[tuple]], Dict[str, List[tuple]]]:
    """
    Build dicts {cid: [(chain, pos, aa, lab), ...]} for before/after,
    keeping a consistent residue order.
    """
    before: Dict[str, List[tuple]] = {}
    after: Dict[str, List[tuple]] = {}

    for cid, pm in ann_chain.items():
        if cid not in prop_chain:
            continue
        chain_letter = cid.split("_")[1]
        sorted_pos = sorted(pm.keys())
        labs_after = prop_chain[cid]
        if len(labs_after) != len(sorted_pos):
            # Safety: skip inconsistent chains
            continue

        b_list, a_list = [], []
        for i, pos in enumerate(sorted_pos):
            aa, lab_before = pm[pos]
            lab_after = labs_after[i]
            b_list.append((chain_letter, pos, aa, lab_before))
            a_list.append((chain_letter, pos, aa, lab_after))
        before[cid] = b_list
        after[cid] = a_list

    return before, after


def catalytic_label_sum(arr: List[tuple]) -> int:
    return sum(lab for _, _, _, lab in arr)


def save_annotations_ordered(ids: List[str], data: Dict[str, List[tuple]], output_file: str):
    with open(output_file, "w") as f:
        for id_chain in ids:
            f.write(f">{id_chain}\n")
            for chain, pos, aa, lab in data[id_chain]:
                f.write(f"{chain} {pos} {aa} {lab}\n")


def write_diff_info(ids: List[str],
                    before: Dict[str, List[tuple]],
                    after: Dict[str, List[tuple]],
                    output_file: str):
    with open(output_file, "w") as f:
        for id_chain in ids:
            before_list = before[id_chain]
            after_list = after[id_chain]
            before_ones = [pos for (chain, pos, aa, lab) in before_list if lab == 1]
            propagated_ones = [
                pos for i, (chain, pos, aa, lab) in enumerate(after_list)
                if lab == 1 and before_list[i][3] == 0
            ]
            f.write(f"{id_chain}:\n")
            f.write(f"- before: {before_ones}\n")
            f.write(f"- propagated: {propagated_ones}\n\n")


def write_diff_info_with_aa(ids: List[str],
                            before: Dict[str, List[tuple]],
                            after: Dict[str, List[tuple]],
                            output_file: str):
    with open(output_file, "w") as f:
        for id_chain in ids:
            before_list = before[id_chain]
            after_list = after[id_chain]
            before_entries = [
                f"{chain} {pos} {aa}"
                for chain, pos, aa, lab in before_list
                if lab == 1
            ]
            propagated_entries = [
                f"{chain} {pos} {aa}"
                for (chain, pos, aa, lab), (_, _, _, lab0) in zip(after_list, before_list)
                if lab == 1 and lab0 == 0
            ]
            f.write(f"{id_chain}:\n")
            f.write(f"- before: {before_entries}\n")
            f.write(f"- propagated: {propagated_entries}\n\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ----------------------------------------
    # Args
    # ----------------------------------------
    parser = argparse.ArgumentParser(description="Propagate labels through MSAs (parallel).")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Propagate when fraction of positive labels >= threshold")
    parser.add_argument("--alignment_type", type=int, required=True,
                        help="Alignment used in clustering. 1: TM-align (global), 2: 3Di+AA (local)")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    if args.alignment_type not in (1, 2):
        raise ValueError("Wrong alignment type, must be 1 or 2")

    threshold = args.threshold
    alignment_type = "TMAlign" if args.alignment_type == 1 else "3di"
    n_workers = max(1, int(args.workers))

    # ----------------------------------------
    # Paths
    # ----------------------------------------
    protein_table_path = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
    annot_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9"
    pdb_to_uniprot_path = "/home/iscb/wolfson/annab4/DB/all_proteins/pdb_to_uniprot.json"

    if alignment_type == "TMAlign":
        msa_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_TMAlign"
    else:
        msa_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_3di"

    dataset_table = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9/dataset.csv"
    output_dir = f"/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated_v9_pools_{alignment_type}_{threshold}"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------
    # Load core data
    # ----------------------------------------
    with open(protein_table_path) as f:
        protein_table = json.load(f)
    with open(pdb_to_uniprot_path) as f:
        pdb_to_uniprot = json.load(f)

    uniprot_seqs = {uid: d["uniprot_sequence"] for uid, d in protein_table.items()}

    ann_chain: Dict[str, Dict[int, Tuple[str, int]]] = {}
    for fn in glob.glob(os.path.join(annot_dir, "*.txt")):
        ann_chain.update(parse_annotation_file(fn))

    # ----------------------------------------
    # Build initial UniProt-level labels (base)
    # ----------------------------------------
    propagated_uniprot: Dict[str, List[int]] = {}
    for cid, pm in ann_chain.items():
        uid, chain = cid.rsplit("_", 1)
        if uid not in uniprot_seqs or chain != "A":
            continue
        labs = [0] * len(uniprot_seqs[uid])
        for pos, (_, label) in pm.items():
            labs[pos - 1] = label
        propagated_uniprot[uid] = labs  # base labels at UniProt level

    # ----------------------------------------
    # Parallel MSA-based propagation
    # ----------------------------------------
    msa_files = sorted(glob.glob(os.path.join(msa_dir, "*.a3m")))
    processed = 0
    skipped = 0

    if not msa_files:
        print("No MSA files found — skipping MSA propagation step.")
    else:
        print(f"Found {len(msa_files)} MSA files. Using {n_workers} workers.")
        chunksize = max(1, len(msa_files) // (n_workers * 8))

        with Pool(processes=n_workers,
                  initializer=init_pool_msa,
                  initargs=(uniprot_seqs, propagated_uniprot, threshold, alignment_type)) as pool:

            for proposals, was_skipped in pool.imap_unordered(process_msa_file, msa_files, chunksize=chunksize):
                if was_skipped:
                    skipped += 1
                else:
                    # Apply proposals to main dict (only sets to 1)
                    for hid, pos_map in proposals.items():
                        labels = propagated_uniprot.setdefault(hid, [0] * len(uniprot_seqs[hid]))
                        for pos, val in pos_map.items():
                            if 0 <= pos < len(labels):
                                labels[pos] = max(labels[pos], val)
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Processed {processed} / {len(msa_files)} MSAs ...")

        print(f"Done MSA phase: processed {processed}, skipped {skipped} MSAs.")

    # ----------------------------------------
    # Map propagated labels back to PDB chains (parallel)
    # ----------------------------------------
    print("Mapping propagated UniProt labels back to PDB chains...")
    items = list(ann_chain.items())
    prop_chain: Dict[str, List[int]] = {}

    chunksize_map = max(1, len(items) // (n_workers * 8)) if items else 1
    with Pool(processes=n_workers,
              initializer=init_pool_map,
              initargs=(uniprot_seqs, propagated_uniprot, pdb_to_uniprot)) as pool:

        done = 0
        for res in pool.imap_unordered(map_chain_worker, items, chunksize=chunksize_map):
            if res is None:
                done += 1
                continue
            cid, mapped = res
            prop_chain[cid] = mapped
            done += 1
            if done % 200 == 0:
                print(f"Mapped {done} / {len(items)} chains ...")

    print(f"Mapped total {len(prop_chain)} chains.")

    # ----------------------------------------
    # Save propagated_all.txt + per split
    # ----------------------------------------
    save_ann(prop_chain, os.path.join(output_dir, "propagated_all.txt"), ann_chain)

    df = pd.read_csv(dataset_table, dtype=str)
    assert "Sequence_ID" in df.columns and "Set_Type" in df.columns
    for split in df["Set_Type"].unique():
        seqs = df[df["Set_Type"] == split]["Sequence_ID"]
        subset = {cid: prop_chain[cid] for cid in seqs if cid in prop_chain}
        save_ann(subset, os.path.join(output_dir, f"{split}.txt"), ann_chain)
        print(f"Wrote {len(subset)} chains to {split}.txt")

    # ----------------------------------------
    # EXTRA: Build and save comparison reports (before/after/diffs)
    # ----------------------------------------
    before_lists, after_lists = build_before_after_lists(ann_chain, prop_chain)

    # Totals
    total_before = sum(catalytic_label_sum(v) for v in before_lists.values())
    total_after = sum(catalytic_label_sum(v) for v in after_lists.values())
    print(f"Number of positive labels before propagation: {total_before}")
    print(f"Number of positive labels after propagation:  {total_after}")

    # Improved chains (sum after > sum before)
    improved_ids = [
        cid for cid in before_lists
        if (cid in after_lists) and (catalytic_label_sum(after_lists[cid]) > catalytic_label_sum(before_lists[cid]))
    ]
    print(f"Number of improved chains: {len(improved_ids)}")

    # Save ordered before/after only for improved chains
    save_annotations_ordered(improved_ids, before_lists, os.path.join(output_dir, "before.txt"))
    save_annotations_ordered(improved_ids, after_lists, os.path.join(output_dir, "after.txt"))
    print("Saved before.txt and after.txt")

    # Save diffs
    write_diff_info_with_aa(improved_ids, before_lists, after_lists, os.path.join(output_dir, "diff_with_aa.txt"))
    print("Saved diff_with_aa.txt")
    write_diff_info(improved_ids, before_lists, after_lists, os.path.join(output_dir, "diff.txt"))
    print("Saved diff.txt")

    print(f"Done propagation with threshold={threshold}, alignment={alignment_type}")


if __name__ == "__main__":
    main()
