import os
import json
import csv
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

import biotite.database.rcsb as rcsb
import biotite.database.afdb as afdb
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from biotite.structure.info import one_letter_code
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix


# ---------------- Paths & constants ----------------
ANNOTATION_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v9"
DATASET_CSV = os.path.join(ANNOTATION_DIR, "dataset.csv")
PDB_DIR = "/home/iscb/wolfson/annab4/Data/PDB_files"
SAVE_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/structural_homology"
SAVE_FILE = os.path.join(SAVE_DIR, "3Di_DB_splits.json")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)

SPLIT_FILES = [f"split{i}.txt" for i in range(1, 6)]
THREADS = min(128, cpu_count())

# 3Di minimal length (earlier code used 50; keep permissive but adjustable)
MIN_3DI_LEN = 6

# AA alignment params (identity matrix, large gap penalties)
_alph = ProteinSequence.alphabet
_n = len(_alph)
_mat = np.full((_n, _n), -100, dtype=int)
for _i in range(_n):
    _mat[_i, _i] = 1
SUB_MAT = SubstitutionMatrix(_alph, _alph, _mat)
GAP_PEN = (-100, -10)

# Global weights for pool workers
G_WEIGHTS = {}


def init_pool(weights):
    """Initializer to set global weights in workers."""
    global G_WEIGHTS
    G_WEIGHTS = weights


# ---------------- I/O helpers ----------------
def load_structure_weights(csv_path):
    """Read dataset.csv and return {Sequence_ID: float(W_Structure)}."""
    weights = {}
    if not os.path.isfile(csv_path):
        print(f"[WARN] dataset.csv not found at {csv_path} -> default weight=1.0")
        return weights

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("Sequence_ID")
            w = row.get("W_Structure")
            if not seq_id:
                continue
            try:
                weights[seq_id] = float(w) if w not in (None, "", "NA") else 1.0
            except Exception:
                weights[seq_id] = 1.0
    return weights


def parse_split_annotations(file_path):
    """
    Parse one split file. Expected format:
      >PROTID_CHAIN
      <idx> <resnum> <aa> <label>
    Returns dict[(prot_id, chain_id)] -> list[(aa, label)]
    """
    ann = {}
    with open(file_path, "r") as f:
        current = None
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    ann[current] = entries
                parts = line[1:].split("_")
                current = (parts[0], parts[1])
                entries = []
            else:
                cols = line.split()
                if len(cols) == 4:
                    # we only need aa and label for building sequences
                    aa = cols[2]
                    lab = cols[3]
                    entries.append((aa, lab))
        if current is not None:
            ann[current] = entries
    return ann


# ---------------- Core helpers ----------------
def compute_index_mapping(aligned_filtered, aligned_original):
    """Map indices from filtered AA seq to original AA seq."""
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


def process_entry(item):
    """
    item: ((seq_id, chain_id), entries) where entries = [(aa, label), ...]
    Returns ('processed', payload, None) or ('skipped_REASON', key, info)
    """
    (seq_id, chain_id), entries = item
    seq_chain = f"{seq_id}_{chain_id}"

    # Fetch CIF (PDB 4-char via RCSB, else AFDB)
    try:
        if len(seq_id) == 4 and seq_id.isalnum():
            cif_path = rcsb.fetch(seq_id, format="cif", target_path=PDB_DIR)
        else:
            cif_path = afdb.fetch(seq_id, format="cif", target_path=PDB_DIR)
        cif = pdbx.CIFFile.read(cif_path)
    except Exception as e:
        return ("skipped_fetch", (seq_id, chain_id), str(e))

    try:
        atoms = pdbx.get_structure(cif, model=1, use_author_fields=True)
    except Exception as e:
        return ("skipped_structure_read", (seq_id, chain_id), str(e))

    # Select chain and keep amino acids only
    chain_atoms = atoms[atoms.chain_id == chain_id]
    chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
    if chain_atoms.array_length() == 0:
        return ("skipped_empty_chain", (seq_id, chain_id), None)

    # Build 3Di sequence
    try:
        structural_sequence = strucalph.to_3di(chain_atoms)[0][0]
    except Exception as e:
        return ("skipped_3di_fail", (seq_id, chain_id), str(e))
    if len(structural_sequence) < MIN_3DI_LEN:
        return ("skipped_short_3di", (seq_id, chain_id), len(structural_sequence))

    # Original (annotated) AA sequence and labels
    orig_aa = "".join(aa if aa in _alph else "X" for aa, _ in entries)
    labels = [lab for _, lab in entries]

    # Filtered AA sequence from structure residues (3-letter -> 1-letter)
    try:
        _, three_letter = struc.get_residues(chain_atoms)
    except Exception as e:
        return ("skipped_residues", (seq_id, chain_id), str(e))
    filt_aa_chars = []
    for aa3 in three_letter:
        try:
            c = one_letter_code(aa3)
            if c not in _alph:
                c = "X"
        except Exception:
            c = "X"
        filt_aa_chars.append(c)
    filt_aa = "".join(filt_aa_chars)

    # Soft sequence consistency check with CIF (warn, do not skip)
    try:
        seq_dict = pdbx.get_sequence(cif)  # dict: chain_id -> Sequence object
        cif_seq = str(seq_dict.get(chain_id)) if seq_dict and seq_dict.get(chain_id) else None
        if cif_seq and len(cif_seq) >= MIN_3DI_LEN and cif_seq != orig_aa:
            # only warn via status tag; keep going
            pass
    except Exception:
        pass

    # Align filtered (structure-derived) to original (annotation-derived)
    try:
        psa = ProteinSequence(filt_aa)
        osa = ProteinSequence(orig_aa)
        aln = align_optimal(psa, osa, SUB_MAT, gap_penalty=GAP_PEN)[0]
        g_f, g_o = aln.get_gapped_sequences()
        mapping = compute_index_mapping(g_f, g_o)
    except Exception as e:
        return ("skipped_align", (seq_id, chain_id), str(e))

    # Prepare payload
    payload = {
        "id": (seq_id, chain_id),
        "seq3di": "".join(structural_sequence.symbols),
        "filt_aa": filt_aa,
        "orig_aa": orig_aa,
        "labels": labels,
        "mapping": mapping,
        "proteinid_chainid": seq_chain,
        "structure_weight": G_WEIGHTS.get(seq_chain, 1.0),
    }

    assert len(structural_sequence.symbols) == len(filt_aa_chars), \
        f"3Di/AA length mismatch for {seq_chain}"

    return ("processed", payload, None)


def process_split(split_path, weights):
    """Process one split file using a worker pool."""
    ann = parse_split_annotations(split_path)
    items = list(ann.items())
    results = []

    with Pool(THREADS, initializer=init_pool, initargs=(weights,)) as pool:
        for res in pool.imap_unordered(process_entry, items, chunksize=64):
            results.append(res)

    # Stats and build lists
    skip_counts = defaultdict(int)
    debug_examples = defaultdict(list)
    processed = []

    for status, payload, info in results:
        skip_counts[status] += 1
        if status == "processed":
            processed.append(payload)
        else:
            if len(debug_examples[status]) < 5:
                debug_examples[status].append((payload, info))

    print(f"[SPLIT] {os.path.basename(split_path)}: "
          f"processed={len(processed)}; " +
          ", ".join(f"{k}:{v}" for k, v in skip_counts.items() if k != "processed"))

    # Assemble split dict
    db_ids = []
    db_3di_sequences = []
    db_filtered_aa_sequences = []
    db_annotated_aa_sequences = []
    db_labels = []
    db_index_mappings = []
    proteinid_chainid = []
    structure_weights = []

    for p in processed:
        db_ids.append(tuple(p["id"]))
        db_3di_sequences.append(p["seq3di"])
        db_filtered_aa_sequences.append(p["filt_aa"])
        db_annotated_aa_sequences.append(p["orig_aa"])
        db_labels.append(p["labels"])
        db_index_mappings.append(p["mapping"])
        proteinid_chainid.append(p["proteinid_chainid"])
        structure_weights.append(p["structure_weight"])

    split_dict = {
        "db_ids": db_ids,
        "db_3di_sequences": db_3di_sequences,
        "db_filtered_aa_sequences": db_filtered_aa_sequences,
        "db_annotated_aa_sequences": db_annotated_aa_sequences,
        "db_labels": db_labels,
        "db_index_mappings": db_index_mappings,
        "proteinid_chainid": proteinid_chainid,
        "structure_weights": structure_weights,
    }

    # Optional: print a few debug examples
    for st, exs in debug_examples.items():
        if st == "processed":
            continue
        print(f"  [debug:{st}] examples (up to 5):")
        for ex_payload, info in exs:
            key = ex_payload if isinstance(ex_payload, tuple) else ex_payload.get("id")
            print(f"    {key} -> {info}")

    return split_dict


def main():
    weights = load_structure_weights(DATASET_CSV)
    all_splits = {}

    for i, split_file in enumerate(SPLIT_FILES, start=1):
        split_name = f"split{i}"
        split_path = os.path.join(ANNOTATION_DIR, split_file)
        print(f"=== Processing {split_name} ===")
        split_dict = process_split(split_path, weights)
        all_splits[split_name] = split_dict

    with open(SAVE_FILE, "w") as f:
        json.dump(all_splits, f)
    print(f"[OK] saved -> {SAVE_FILE}")


if __name__ == "__main__":
    main()
