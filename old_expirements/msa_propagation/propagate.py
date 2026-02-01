#!/usr/bin/env python3

#TODO: добавить коствль, что не надо пропагарировать, если рядом есть такая же каталитическая аминокислота 

import os
import argparse
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

# ----------------------------------------
# Argument parsing
# ----------------------------------------
parser = argparse.ArgumentParser(
    description="Propagate labels through MSAs."
)
parser.add_argument(
    "--threshold", type=float, required=True,
    help="Propagate when fraction of positive labels >= threshold"
)
parser.add_argument(
    "--alignment_type", type=int, required=True,
    help="Aligment used in clustering. 1: TMalign (global), 2: 3Di+AA Gotoh-Smith-Waterman (local)"
)
args = parser.parse_args()
THRESHOLD = args.threshold

if not (args.alignment_type == 1 or args.alignment_type == 2):
    raise ValueError("Wrong alignment type, must be 1 or 2")

ALIGNMENT_TYPE = "TMAlign" if args.alignment_type==1 else "3di"
MIN_QUORUM = 2
NEARBY_WINDOW  = 10

# ----------------------------------------
# Paths
# ----------------------------------------
PROTEIN_TABLE = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
ANNOT_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v8"
PDB_TO_UNIPROT = "/home/iscb/wolfson/annab4/DB/all_proteins/pdb_to_uniprot.json"

if ALIGNMENT_TYPE == "TMAlign":
    MSA_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_TMAlign"
else:
    MSA_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_3di"

DATASET_TABLE = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v8/dataset.csv"
OUTPUT_DIR = f"/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated_v8_{ALIGNMENT_TYPE}_{THRESHOLD}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Load UniProt table
# ----------------------------------------
with open(PROTEIN_TABLE) as f:
    protein_table = json.load(f)
    
with open(PDB_TO_UNIPROT) as f:
    pdb_to_uniprot = json.load(f)
    
# ----------------------------------------
# Precompute UniProt sequences
# ----------------------------------------
uniprot_seqs = {
    uid: d["uniprot_sequence"]
    for uid, d in protein_table.items()
}

# ----------------------------------------
# Parse annotation files
# ----------------------------------------
def parse_annotation_file(path: str) -> Dict[str, Dict[int, Tuple[str,int]]]:
    ann = {}
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

ann_chain = {}
for fn in glob.glob(os.path.join(ANNOT_DIR, "*.txt")):
    ann_chain.update(parse_annotation_file(fn))


# ----------------------------------------
# A3M parsing and col2pos helpers
# ----------------------------------------
def parse_a3m_header(header: str) -> int:
    parts = header[1:].split()
    if len(parts) < 8:
        # probably a centroid sequence
        return 0
    try:
        return int(parts[7])
    except ValueError:
        return 0
    
def a3m_to_aligned_and_col2pos(a3mseq, original_seq, start_index=0):
    aligned_seq, col2pos = [],{}
    seq_len = len(original_seq)
    col_counter = -1
    seq_counter = -1
    for i,aa in enumerate(a3mseq):
        if aa != '-':
            seq_counter +=1
        if aa.isupper() or aa == '-':
            col_counter += 1
            aligned_seq.append(aa)
            if aa != '-':
                col2pos[col_counter] = seq_counter + start_index      
    return "".join(aligned_seq), col2pos

def compute_aligned_and_mapping(
    msa_seq: str,
    orig_seq: str,
    orig_start: int
) -> Tuple[str, Dict[int,int]]:
    aligned_chars = []
    col2pos = {}
    orig_i = orig_start
    aligned_i = 0
    L = len(orig_seq)

    for c in msa_seq:
        if c == "-":
            # gap
            aligned_chars.append("-")
            aligned_i += 1
        elif c.isalpha():
            if c.islower():
                orig_i += 1
            else:
                # match
                if 0 <= orig_i < L:
                    aligned_chars.append(c)
                    col2pos[aligned_i] = orig_i
                orig_i += 1
                aligned_i += 1

    return "".join(aligned_chars), col2pos

def read_a3m(fn: str):
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

# ----------------------------------------
# Map catalytic sites via PDB→UniProt alignment
# ----------------------------------------
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY-")
AMBIGUOUS_MAP = {
    "U": "C",  # Selenocysteine → Cys
    "O": "K",  # Pyrrolysine → Lys
    "B": "D",  # Asx (D or N) → Asp
    "Z": "E",  # Glx (E or Q) → Glu
    "J": "L",  # Xle (I or L) → Leu
    "X": "A",  # unknown → Ala (placeholder)
    "*": ""    # stop codon → удаляем
}

def sanitize_sequence(seq: str) -> str:
    out = []
    for aa in seq:
        if aa in STANDARD_AAS:
            out.append(aa)
        elif aa in AMBIGUOUS_MAP and AMBIGUOUS_MAP[aa]:
            out.append(AMBIGUOUS_MAP[aa])
    return "".join(out)


def map_catalytic_sites(uniprot_seq, pdb_seq, uniprot_labels):
    """
    Local alignment PDB→UniProt, labels propagation.
    """
    # sanitize sequences
    u = sanitize_sequence(uniprot_seq)
    p = sanitize_sequence(pdb_seq)
    # build simple identity matrix
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
        if au != '-' and ap != '-':
            if au == ap and iu < len(uniprot_labels):
                res[ip] = uniprot_labels[iu]
            iu += 1
            ip += 1
        elif au != '-' and ap == '-':
            iu += 1
        elif au == '-' and ap != '-':
            ip += 1
    return res

# ----------------------------------------
# Build initial propagated_uniprot directly
# ----------------------------------------
propagated_uniprot = {}
for cid, pm in ann_chain.items():
    uid, chain = cid.rsplit("_", 1)
    if uid not in uniprot_seqs or chain != "A":
        continue
    labs = [0] * len(uniprot_seqs[uid])
    for pos, (_, label) in pm.items():
        labs[pos-1] = label
    propagated_uniprot[uid] = labs


# ----------------------------------------
# MSA-based propagation
# ----------------------------------------
msa_files = sorted(glob.glob(os.path.join(MSA_DIR, "*.a3m")))
processed = skipped = 0

for fn in msa_files:
    headers, ids, raw = read_a3m(fn)
    aligned, col2pos = {}, {}

    # build per-sequence aligned + mapping
    for header, hid, seq in zip(headers, ids, raw):
        full = uniprot_seqs.get(hid)
        if full is None:
            raise KeyError(f"UniProt sequence for {hid} not found")
        start = parse_a3m_header(header)
        #a_seq, mapping = compute_aligned_and_mapping(seq, full, start)
        a_seq, mapping = a3m_to_aligned_and_col2pos(seq, full, start)
        aligned[hid]  = a_seq
        col2pos[hid] = mapping

    # filter members that have annotations
    members = [h for h in aligned if h in propagated_uniprot]
    if len(members) < 2:
        skipped += 1
        continue

    L = max(len(aligned[h]) for h in members)
    for c in range(L):
        aa_labels = defaultdict(list)
        for hid in members:
            seq_c = aligned[hid]
            if c < len(seq_c) and seq_c[c] != '-':
                pos = col2pos[hid].get(c)
                if pos is not None:
                    #print(hid, pos)
                    aa_labels[seq_c[c]].append(propagated_uniprot[hid][pos])
        if not aa_labels:
            continue

        # pick consensus amino acid with most labels
        most, labs = max(aa_labels.items(), key=lambda kv: len(kv[1]))
        if len(labs) < MIN_QUORUM:
            continue
        flag = int(sum(labs) / len(labs) >= THRESHOLD)

        # propagate с проверкой на «соседние» каталитические остатки
        for hid in members:
            if c >= len(aligned[hid]) or aligned[hid][c] != most:
                continue
            pos = col2pos[hid].get(c)
            if pos is None:
                continue

            # если мы хотим поставить 1 — проверяем окно ±NEARBY_WINDOW
            if flag == 1:
                seq         = uniprot_seqs[hid]
                labels_list = propagated_uniprot[hid]
                start       = max(0, pos - NEARBY_WINDOW)
                end         = min(len(seq)-1, pos + NEARBY_WINDOW)
                # если найдётся хотя бы одна другая 1 того же AA — пропускаем
                if any(
                    labels_list[q] == 1 and seq[q] == most
                    for q in range(start, end+1)
                    if q != pos
                ):
                    print(f"Possible artifact propagation due to mis-alignment, protein {hid}, position {pos}")
                    continue

            # собственно, само пропагирование
            old = propagated_uniprot[hid][pos]
            propagated_uniprot[hid][pos] = max(old, flag)


    processed += 1
    if processed % 50 == 0:
        print(f"Processed {processed} MSAs")

print(f"Done: processed {processed}, skipped {skipped} MSAs")

# ----------------------------------------
# Map propagated labels to PDB chains
# ----------------------------------------
prop_chain = {}
for cid, pm in ann_chain.items():
    pid = cid.split('_')[0]
    uid = None
    if pid in uniprot_seqs:
        uid = pid
    else:
        uid = pdb_to_uniprot.get(pid)
    if not uid:
        continue
    seq_u = uniprot_seqs.get(uid)
    if not seq_u:
        continue
    pdb_seq = ''.join(aa for _, (aa, _) in sorted(pm.items()))
    labels  = propagated_uniprot.get(uid)
    if labels is None:
        continue
    prop_chain[cid] = map_catalytic_sites(seq_u, pdb_seq, labels)

# ----------------------------------------
# Save helpers
# ----------------------------------------
def save_ann(data: Dict[str, list], path: str):
    with open(path, 'w') as f:
        for cid, labs in data.items():
            f.write(f">{cid}\n")
            for i, res in enumerate(sorted(ann_chain[cid].keys())):
                aa, _ = ann_chain[cid][res]
                f.write(f"{cid.split('_')[1]} {res} {aa} {labs[i]}\n")

# Save full propagated
save_ann(prop_chain, os.path.join(OUTPUT_DIR, "propagated_all.txt"))

# Split by dataset
df = pd.read_csv(DATASET_TABLE, dtype=str)
assert 'Sequence_ID' in df.columns and 'Set_Type' in df.columns
for split in df['Set_Type'].unique():
    seqs = df[df['Set_Type']==split]['Sequence_ID']
    subset = {
        cid: prop_chain[cid]
        for cid in seqs
        if cid in prop_chain
    }
    save_ann(subset, os.path.join(OUTPUT_DIR, f"{split}.txt"))
    print(f"Wrote {len(subset)} chains to {split}.txt")

print(f"Done propagation with threshold={THRESHOLD}")
