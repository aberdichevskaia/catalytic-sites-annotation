import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.sequence.io.fasta import FastaFile

PROTEIN_TABLE = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
ANNOT_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/splitted5"
MSA_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/msa_dir"
DATASET_TABLE = "/home/iscb/wolfson/annab4/DB/all_proteins/rebalanced_reweighted/dataset_rebalanced.csv"

THRESHOLD = 0.2

OUTPUT_DIR = f"/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation/propagated{THRESHOLD}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_annotations(input_file):
    data = {}
    current_id = None
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:]
                data[current_id] = []
            else:
                data[current_id].append(line)
    return data

all_annotations = load_annotations(os.path.join(OUTPUT_DIR, "propagated_all.txt"))
df = pd.read_csv(DATASET_TABLE)
if 'Sequence_ID' not in df.columns or 'Set_Type' not in df.columns:
    raise RuntimeError("CSV must have 'Sequence_ID' and 'Set_Type' columns")
for split in df['Set_Type'].unique():
    ids_split = df.loc[df['Set_Type'] == split, 'Sequence_ID'].astype(str)
    split_ann = {id_chain: all_annotations[id_chain]
                 for id_chain in ids_split if id_chain in all_annotations}
    out_fn = os.path.join(OUTPUT_DIR, f"{split}.txt")
    with open(out_fn, 'w') as f:
        for id_chain, lines in split_ann.items():
            f.write(f">{id_chain}\n")
            for l in lines:
                f.write(f"{l}\n")
    print(f"Wrote {len(split_ann)} entries to {out_fn}")
