#!/usr/bin/env python3
"""
Propagate catalytic‐site labels through pairwise structural alignments
within precomputed clusters, for all sequences regardless of previous train/test splits.
Comments are in English only.
"""
import os
import json
import glob
import argparse
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
import pandas as pd

from biotite.structure.alphabet import I3DSequence
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from biotite.structure.io.pdbx import CIFFile, get_sequence
import biotite.database.rcsb as rcsb

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Smith–Waterman gap penalties for 3Di alignment
GAP_PENALTY = (-10, -1)
# Minimum alignment score to accept a hit
CUTOFF = 150 # maybe don't need it at all: we are already working inside clusters. or check percentiles to be sure
# Minimum number of sequences supporting an MSA‐based call 
MIN_QUORUM = 2

STD_3DI = SubstitutionMatrix.std_3di_matrix()
PDB_DIR        = "/home/iscb/wolfson/annab4/Data/PDB_files"
PDB_TO_UNIPROT = "/home/iscb/wolfson/annab4/DB/all_proteins/pdb_to_uniprot.json"
with open(PDB_TO_UNIPROT) as f:
    pdb_to_uniprot = json.load(f)

_DEBUG_HITS = []
_DEBUG_TMPL_COUNTS = []

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def compute_index_mapping(aligned_filtered: str, aligned_original: str) -> dict:
    """
    Build a map from positions in the filtered (3Di) alignment
    to positions in the original sequence.
    """
    mapping = {}
    i_f = i_o = 0
    for a, b in zip(aligned_filtered, aligned_original):
        if a != '-' and b != '-':
            mapping[i_f] = i_o
            i_f += 1
            i_o += 1
        elif a != '-' and b == '-':
            i_f += 1
        elif a == '-' and b != '-':
            i_o += 1
    return mapping

def parse_annotation_file(path: str) -> dict:
    """
    Read per‐chain annotation file into:
      { "UniProtID_chain": { residue_number: (AA, label), ... }, ... }
    """
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
                chain, resnum, aa, lab = line.split()
                ann[current][int(resnum)] = (aa, int(lab))
    return ann

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


def map_catalytic_sites(uniprot_seq: str, pdb_seq: str, uniprot_labels: list) -> list:
    """
    Align PDB‐derived sequence to UniProt sequence and propagate labels.
    Returns a list of propagated labels for the PDB sequence.
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

# -----------------------------------------------------------------------------
# Per‐target propagation
# -----------------------------------------------------------------------------
def process_target(item):
    """
    Align one sequence (target) against its cluster members (templates)
    and propagate labels by simple averaging of hits above CUTOFF.
    Returns (index, numpy array of propagated labels).
    """
    idx, (uid, chain) = item
    has_cluster = uid in membership_map
    print(f"[TGT] idx={idx:<5} uid={uid:<10} chain={chain}  has_cluster={has_cluster}")
    
    annotated = test_annotated_aa_sequences[idx]
    pred      = np.zeros(len(annotated), float)
    hits      = 0
    # find this sequence’s cluster representative
    rep = membership_map.get(uid)
    if rep is not None:
        members = cluster_members.get(rep, [])
        print(f"      rep={rep}  members={len(members)}")
        # map each member to its index in the DB
        tmpls = [train_id2idx[m] for m in members if m in train_id2idx]
        _DEBUG_TMPL_COUNTS.append(len(tmpls))
        print(f"      mapped to {len(tmpls)} templates")
        if not tmpls:
            return idx, pred.astype(int)
        # do pairwise 3Di‐alignments
        for j in tmpls:
            if j == idx:
                continue
            alns = align_optimal(
                test_3di_sequences[idx],
                db_3di_sequences[j],
                STD_3DI,
                gap_penalty=GAP_PENALTY,
                local=True
            )
            # if not alns or alns[0].score <= CUTOFF:
            #     continue
            ### for debug pupposes
            if not alns:
                print(f"DEBUG: {uid} vs {db_ids[j][0]} → no alignment")
                continue
            if alns[0].score <= CUTOFF:
                print(f"DEBUG: {uid} vs {db_ids[j][0]} → score = {alns[0].score}")
                continue
            #####
            g_t, g_s = alns[0].get_gapped_sequences()
            mapping_3di = compute_index_mapping(g_t, g_s)
            tm_annot    = db_annotated_aa_sequences[j]
            tm_labels   = np.array(db_labels[j])
            tm_map      = db_index_mappings[j]
            # propagate
            for tgt3d, tmp3d in mapping_3di.items():
                tgt_idx = test_index_mappings[idx].get(tgt3d)
                tmp_idx = tm_map.get(tmp3d)
                if tgt_idx is not None and tmp_idx is not None:
                    if annotated[tgt_idx] == tm_annot[tmp_idx]:
                        pred[tgt_idx] += tm_labels[tmp_idx]
            hits += 1
    # if hits < MIN_QUORUM:
    #     return idx, np.zeros(len(annotated), int)
    if hits > 0:
        pred /= hits
    binary = (pred >= THRESHOLD).astype(int)
    _DEBUG_HITS.append(hits)

    return idx, binary

def map_chain(item):
    """
    Map propagated UniProt labels back onto one PDB chain.
    Returns (cid, mapped_labels) or None if skipped.
    """
    cid, pm = item
    pid, chain = cid.rsplit("_", 1)
    uid = None
    if pid in uniprot_seqs:
        uid = pid
    else:
        uid = pdb_to_uniprot.get(pid)
    if not uid:
        return None
    seq_u = uniprot_seqs.get(uid)
    if seq_u is None:
        return None
    labs  = propagated_uniprot.get(uid)
    if labs is None:
        return None
    # reconstruct the PDB sequence in chain-A order
    pdb_seq = "".join(aa for _, (aa, _) in sorted(pm.items()))
    # do the alignment‐based mapping
    labels = map_catalytic_sites(seq_u, pdb_seq, labs)
    return cid, labels

# --- Helper to write .txt annotation ---
def save_ann(data: dict, path: str):
    with open(path, 'w') as f:
        for cid, labs in data.items():
            chain = cid.rsplit("_", 1)[1]
            f.write(f">{cid}\n")
            if cid in ann_chain:
                # UniProt-цепи: порядок и номера из исходных аннотаций
                orig = ann_chain[cid]
                # orig — dict: resnum → (AA, old_label)
                for res,(aa,_) in sorted(orig.items()):
                    # labs хранит метки для _всего_ UniProt-цепи, поэтому берём labs[res-1]
                    f.write(f"{chain} {res} {aa} {labs[res-1]}\n")

            else:
                # PDB-цепи: нет ann_chain[cid], поэтому берем просто последовательность
                seq = pdb_seqs[cid]
                # нумеруем 1…len, AA из seq, метки из labs
                for pos, (aa, lab) in enumerate(zip(seq, labs), start=1):
                    f.write(f"{chain} {pos} {aa} {lab}\n")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster‐based structural propagation over all sequences"
    )
    parser.add_argument(
        "--threads", type=int, default=32,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Propagate when fraction of positive labels >= threshold"
    )
    args = parser.parse_args()
    THRESHOLD = args.threshold
    
    # --- Paths and settings ---
    DB_3DI              = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/3Di_DB_all.json"
    MSA_DIR             = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_3di"
    PROTEIN_TABLE       = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
    ANNOT_DIR           = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v7"
    DATASET_TABLE       = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v7/dataset.csv"
    OUTPUT_DIR          = f"/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated_clusters_{THRESHOLD}"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Load UniProt sequences ---
    with open(PROTEIN_TABLE) as f:
        protein_table = json.load(f)
    uniprot_seqs = { uid: d["uniprot_sequence"] for uid,d in protein_table.items() }

    # --- Load and merge all 3Di JSONs ---
    with open(DB_3DI) as f:
        data = json.load(f)
    db_ids                    = data["db_ids"]
    db_3di_sequences          =  [ I3DSequence(seq) for seq in data["db_3di_sequences"] ]
    db_annotated_aa_sequences = data["db_annotated_aa_sequences"]
    db_labels                 = data["db_labels"]
    db_index_mappings         = data["db_index_mappings"]

    # --- Load test set same as full db (we process all) ---
    test_ids                    = db_ids
    test_3di_sequences          = db_3di_sequences
    test_annotated_aa_sequences = db_annotated_aa_sequences
    test_index_mappings         = db_index_mappings

    # --- Build lookup from ID → index ---
    train_id2idx = { uid: i for i,(uid, ch) in enumerate(db_ids) }

    # --- Load cluster membership ---
    #   membership_map:  query_key -> cluster_id
    #   cluster_members: cluster_id -> [ query_key, ... ]
    membership_map  = {}
    cluster_members = defaultdict(list)

    for path in glob.glob(os.path.join(MSA_DIR, "*.a3m")):
        cluster_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                # >A0A0H2ZMF9 или >Q8DNB6   2158  1.00  ... 
                header = line[1:].strip()
                seq_id = header.split()[0]
                membership_map[seq_id] = cluster_id
                cluster_members[cluster_id].append(seq_id)


    # --- Parse original per‐chain annotation files ---
    ann_chain = {}
    for fn in glob.glob(os.path.join(ANNOT_DIR, "*.txt")):
        ann_chain.update(parse_annotation_file(fn))

    # --- Initialize propagated_uniprot with existing chain-A labels ---
    propagated_uniprot = {}
    for cid, pm in ann_chain.items():
        uid, chain = cid.rsplit("_", 1)
        if chain != "A" or uid not in uniprot_seqs:
            continue
        labs = [0]*len(uniprot_seqs[uid])
        for pos,(aa,lab) in pm.items():
            labs[pos-1] = lab
        propagated_uniprot[uid] = labs

    # После загрузки JSON и .a3m
    print(f"▶ Total DB entries:      {len(db_ids)}")
    print(f"▶ membership_map size:   {len(membership_map)}")
    print(f"▶ train_id2idx size:     {len(train_id2idx)}")
    common = set(membership_map.keys()) & set(train_id2idx.keys())
    print(f"▶ Common IDs:            {len(common)} / {len(db_ids)}")
    print("▶ Examples in common:", list(common)[:10])
    print("▶ Examples in membership_map:", list(membership_map.keys())[:10])
    print("▶ Examples in db_ids:", [u for u,_ in db_ids[:10]])
    
    # --- Run propagation in parallel ---
    pool    = Pool(args.threads)
    items = [
        (i, (uid, chain))
        for i, (uid, chain) in enumerate(db_ids)
        if uid in uniprot_seqs
    ]
    results = pool.map(process_target, items)
    pool.close(); pool.join()
    # … после pool.join() …
    if _DEBUG_HITS:
        import numpy as _np
        print("▶ hits per target:",
              f"min={min(_DEBUG_HITS)}, "
              f"median={_np.median(_DEBUG_HITS)}, "
              f"max={max(_DEBUG_HITS)}")
    if _DEBUG_TMPL_COUNTS:
        import numpy as _np
        print("▶ tmpls per target:",
              f"min={min(_DEBUG_TMPL_COUNTS)}, "
              f"median={_np.median(_DEBUG_TMPL_COUNTS)}, "
              f"max={max(_DEBUG_TMPL_COUNTS)}")



    # --- Collect propagated scores back into propagated_uniprot ---
    for idx, pred in results:
        uid, chain = test_ids[idx]
        if chain == "A" and uid in propagated_uniprot:
            orig = propagated_uniprot[uid] 
            merged = [ int(o or p) for o,p in zip(orig, pred)]
            propagated_uniprot[uid] = merged
            
    prop_chain = {}
    pdb_seqs   = {}
    
    with Pool(args.threads) as pool:
        # map_chain(item) возвращает (cid, labs) или None
        for item in pool.map(map_chain, ann_chain.items()):
            if item is None:
                continue
            cid, labs = item
            prop_chain[cid] = labs

    # --- Map propagated_uniprot back to each PDB chain ---
    # Заново прочитаем protein_data, чтобы получить pdb_ids
    with open(PROTEIN_TABLE) as f:
        protein_data = json.load(f)  # uid -> {…, "pdb_ids": […], …}
        
    df = pd.read_csv(DATASET_TABLE, dtype=str)
    wanted = set(df['Sequence_ID'])  # e.g. {"1ABC_A", "1ABC_B", …}
    wanted_pdbs = { cid.split("_",1)[0] for cid in wanted }

    # 2) готовим список задач — только по PDB, которые реально нужны
    tasks = [
        (uid, pdb_id, uni_labels)
        for uid, uni_labels in propagated_uniprot.items()
        for pdb_id in protein_data.get(uid, {}).get("pdb_ids", [])
        if pdb_id in wanted_pdbs
    ]

    def process_pdb_chain(args):
        uid, pdb_id, uni_labels = args
        # а) пропускаем всe не-PDB  
        if not (len(pdb_id)==4 and pdb_id.isalnum()):
            print(f"[WARN] not a PDB {pdb_id}")
            return []
        # б) скачиваем CIF
        try:
            cif_path = rcsb.fetch(pdb_id, "cif", PDB_DIR)
            cif      = CIFFile.read(cif_path)
        except Exception as e:
            print(f"[WARN] fetch/parse failed for {pdb_id}: {e}")
            return []
        # в) достаём цепи
        try:
            chains = get_sequence(cif)   # dict: chain_id → Sequence
        except Exception as e:
            print(f"[WARN] bad CIF for {pdb_id}: {e}")
            return []
        out = []
        # г) для каждой цепи проверяем wanted и мапим метки
        for chain_id, seq_obj in chains.items():
            cid = f"{pdb_id}_{chain_id}"
            if cid not in wanted:
                continue
            pdb_seq = str(seq_obj)    # one-letter
            labs    = map_catalytic_sites(
                        uniprot_seqs[uid],
                        pdb_seq,
                        uni_labels
                    )
            out.append((cid, labs, pdb_seq))
        return out

    # 3) запускаем Pool, собираем prop_chain и pdb_seqs
            
    with Pool(args.threads) as pool:
        for result_list in pool.imap_unordered(process_pdb_chain, tasks):
            for cid, labs, seq in result_list:
                prop_chain[cid] = labs
                pdb_seqs[cid]   = seq
                
    for uid, uni_labels in propagated_uniprot.items():
        cid = f"{uid}_A"
        if cid in ann_chain:
            prop_chain[cid] = uni_labels
            pdb_seqs[cid] = ""

    # --- Save full propagated file ---
    save_ann(prop_chain, os.path.join(OUTPUT_DIR, "propagated_all.txt"))

    # --- Split output by DATASET_TABLE Set_Type ---
    df = pd.read_csv(DATASET_TABLE, dtype=str)
    for split in df['Set_Type'].unique():
        seqs  = df[df['Set_Type']==split]['Sequence_ID']
        subset = { cid: prop_chain[cid]
                   for cid in seqs if cid in prop_chain }
        outp = os.path.join(OUTPUT_DIR, f"{split}.txt")
        save_ann(subset, outp)
        print(f"Wrote {len(subset)} chains to {split}.txt")

    print("Done structural propagation within clusters.")
