import numpy as np
from scipy.spatial.distance import cdist
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
import argparse
import os
import time

ANNOTATION_DIR = "/home/iscb/wolfson/annab4/DB/splitted_by_EC_number_fixed"
PDB_DIR = "/home/iscb/wolfson/annab4/Data/PDB_files"
ALPHAFOLD_SERVER = "https://alphafold.ebi.ac.uk/files"

def extract_calpha_and_sequence(file_path, chain_id=None):
    if file_path.endswith(".pdb"):
        structure = get_structure(file_path)
    elif file_path.endswith(".cif"):
        structure = get_structure(file_path, format='mmcif')
    else:
        raise ValueError(f"Invalid structure format")
    
    chain = next((c for c in structure.get_chains() if c.id == chain_id), next(structure.get_chains()))
    calpha_coords, sequence = get_residue_data(chain)
    return calpha_coords, sequence


def propagate_labels(pdb_file1, pdb_file2, labels1, labels2, tm_score_cutoff=0.7):
    calpha1, sequence1 = extract_calpha_and_sequence(pdb_file1)
    calpha2, sequence2 = extract_calpha_and_sequence(pdb_file2)
    t_before = time.time()
    res = tm_align(calpha1, calpha2, sequence1, sequence2)
    translation, rotation, tm_score = res.t, res.u, res.tm_norm_chain1
    t_after = time.time()
    print(f"tm_alighn time {t_after - t_before}")
    
    if tm_score < tm_score_cutoff:
        print(f"Low score ({tm_score}), return")
        return labels1, labels2

    calpha2_new = np.dot(calpha2, rotation.T) + translation

    d12 = cdist(calpha2_new, calpha1) 
    same_aa = np.equal.outer(list(sequence2), list(sequence1))

    correspondences_2, correspondences_1 = np.nonzero((d12 < 2.0) & same_aa)  

    labels1 = np.asarray(labels1, dtype=np.int8)
    labels2 = np.asarray(labels2, dtype=np.int8)

    labels1_propagated = np.maximum(labels1, np.bincount(correspondences_1, weights=labels2[correspondences_2], minlength=len(labels1)))
    labels2_propagated = np.maximum(labels2, np.bincount(correspondences_2, weights=labels1[correspondences_1], minlength=len(labels2)))

    return labels1_propagated, labels2_propagated



def parse_annotation(file_path):
    annotations = {}
    with open(file_path, "r") as f:
        current_id, current_chain = None, None
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split("_")
                current_id = parts[0]
                current_chain = parts[1] if len(parts) > 1 else "A"
                annotations[(current_id, current_chain)] = []
            else:
                cols = line.strip().split()
                if len(cols) == 4:
                    _, _, res_id, label = cols
                    annotations[(current_id, current_chain)].append((int(res_id), int(label)))
    return annotations

def find_pdb_file(uniprot_id):
    possible_files = [
        os.path.join(PDB_DIR, f"{uniprot_id.lower()}_bioentry.cif"),
        os.path.join(PDB_DIR, f"AF_{uniprot_id}.cif"),
    ]

    for file in os.listdir(PDB_DIR):
        if file.startswith("pdb") and file.endswith(".bioent") and uniprot_id in file:
            possible_files.append(os.path.join(PDB_DIR, file))

    for file_path in possible_files:
        if os.path.exists(file_path):
            use_mmcif = file_path.endswith(".cif")
            return file_path, use_mmcif

    return download_structure(uniprot_id)

def download_structure(code):
    if len(code) == 4 and code.isalnum():
        return download_pdb(code)
    else:
        return download_alphafold(code)

def download_pdb(pdb_id):
    pdb_list = PDB.PDBList()
    save_path = pdb_list.retrieve_pdb_file(pdb_id, pdir=PDB_DIR, file_format="mmCif", overwrite=True)
    return (save_path, True) if os.path.exists(save_path) else (None, None)

def download_alphafold(uniprot_id):
    af_url = f"{ALPHAFOLD_SERVER}/AF-{uniprot_id}-F1-model_v4.cif"
    save_path = os.path.join(PDB_DIR, f"AF_{uniprot_id}.cif")
    try:
        response = requests.get(af_url, timeout=30)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path, True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading AlphaFold structure for {uniprot_id}: {e}")
    return None, None


t1 = time.time()

protein1, chain1 = "P07259", "A"
protein2, chain2 = "P05990", "A"

annotations = parse_annotation("/home/iscb/wolfson/annab4/DB/splitted_by_EC_number_fixed/validate.txt")

pdb_file1 = find_pdb_file(protein1)[0]

pdb_file2 = find_pdb_file(protein2)[0]

res_ids1, labels1_values = zip(*annotations[(protein1, chain1)])
res_ids2, labels2_values = zip(*annotations[(protein2, chain2)])

labels1 = np.array(labels1_values, dtype=int)
labels2 = np.array(labels2_values, dtype=int)

t2 = time.time()
print(f"loaded all data in {t2 - t1} sec")

labels1_propagated, labels2_propagated = propagate_labels(pdb_file1, pdb_file2, labels1, labels2)

t3 = time.time()
print(f"perfomed propagation in {t3 - t2} sec")

propagated_annotations1 = list(zip(res_ids1, labels1_propagated))
propagated_annotations2 = list(zip(res_ids2, labels2_propagated))

output_file = "propagated_labels.txt"
with open(output_file, "w") as f:
    f.write(f"> {protein1}_{chain1}\n")
    for res_id, label in propagated_annotations1:
        f.write(f"{res_id} {label}\n")
    
    f.write(f"\n> {protein2}_{chain2}\n")
    for res_id, label in propagated_annotations2:
        f.write(f"{res_id} {label}\n")

t4 = time.time()
print(f"saved to {output_file} in {t4 - t3} sec")




# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument('--protein1', type=str)
# parser.add_argument('--protein2', type=str)
# args = parser.parse_args()

# protein1 = args.protein1
# protein2 = args.protein2

# chain_id1, chain_id2 = None, None

# if '_' in protein1:
#     protein1, chain_id1 = protein1.split('_')
    
# if '_' in protein1:
#     protein2, chain_id2 = protein2.split('_')
    
