import biotite.database.afdb as afdb
import os
import json
import pandas as pd

from biotite.structure.io.pdbx import CIFFile, get_sequence

# Paths to input files
BASE_DIR = "/home/iscb/wolfson/annab4/DB/all_proteins/MSA_propagation"
STRUCTURES_DIR = os.path.join(BASE_DIR, "AF_structures")
PATHES_JSON = os.path.join(BASE_DIR, "pathes.json")

json_file = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"
csv_file = "/home/iscb/wolfson/annab4/DB/all_proteins/dataset_tables5.csv"

# Load the JSON data
with open(json_file, 'r') as f:
    protein_data = json.load(f)

# Load the CSV data
df = pd.read_csv(csv_file, sep=",")

# Extract unique protein IDs from the CSV (before the underscore)
sequence_ids = set(df["Sequence_ID"].str.split("_").str[0].unique())

# Write the matching sequences to the FASTA file
id_to_path = dict()
skipped = 0
deleted = 0
success = 0

for uniprot_id in sequence_ids:
    info = protein_data.get(uniprot_id)
    if info is None:
        continue
    expected_seq = info.get("uniprot_sequence", "").strip().upper()
    if not expected_seq:
        continue
    
    # Download or fetch the path to the already downloaded AF structure
    try:
        cif_path = afdb.fetch(
            uniprot_id,
            format='cif',
            target_path=STRUCTURES_DIR,
            overwrite=False
        )
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {uniprot_id}: {e}")
        skipped += 1
        continue
    
    # Get aminoacid sequence from the structure file 
    try:
        cif = CIFFile.read(cif_path)
        chains_sequences = get_sequence(cif)
        struct_seq = str(chains_sequences.get("A"))
    except Exception as e:
        print(f"[PARSE ERROR] {uniprot_id}: {e}")
        # Remove demaged file
        os.remove(cif_path)
        print(f"  → Deleted {cif_path}")
        deleted += 1
        continue
    
    # Compare sequences, remove if doesn't match 
    if struct_seq != expected_seq:
        print(f"[MISMATCH] {uniprot_id}: expected {len(expected_seq)} aa, got {len(struct_seq)} aa")
        os.remove(cif_path)
        print(f"  → Deleted {cif_path}")
        deleted += 1
    else:
        id_to_path[uniprot_id] = cif_path
        success += 1

print()
print(f"Failed to download {skipped} structures")
print(f"Deleted {deleted} mismatched files")
print(f"Successfuly downloaded {success} structures")
with open(PATHES_JSON, 'w') as f:
    json.dump(id_to_path, f)
            
