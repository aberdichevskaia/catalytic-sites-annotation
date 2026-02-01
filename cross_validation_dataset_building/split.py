import json
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import random
import os
import pickle
import numpy as np
from itertools import combinations

import biotite.structure.io.pdbx as pdbx
import biotite.database.afdb as afdb
import biotite.database.rcsb as rcsb

from multiprocessing import Pool, cpu_count

from prettytable import PrettyTable
from glob import glob

random.seed(42)
np.random.seed(42)

n_procs = min(128, cpu_count())

PDB_DIR = "/home/iscb/wolfson/annab4/Data/PDB_files"

def valid_ec_number(ec_number):
    if ec_number == "not found":
        return False
    parts = ec_number.split('.')
    if len(parts) < 3:
        return False
    return all(part.isdigit() for part in parts[:3])


def structure_exists(protein_id, expected_seq=None):
    try:
        if len(protein_id) == 4 and protein_id.isalnum():
            cif_path = rcsb.fetch(protein_id, format='cif', target_path=PDB_DIR)
        else:
            cif_path = afdb.fetch(protein_id, format='cif', target_path=PDB_DIR)
    except Exception as err:
        print(f"[DOWNLOAD ERROR] {protein_id}: {err}")
        return False

    # Get aminoacid sequence from the structure file 
    try:
        cif = pdbx.CIFFile.read(cif_path)
        chains_sequences = pdbx.get_sequence(cif)
    except Exception as err:
        print(f"[READ ERROR] {protein_id}: {err}")
        # Remove demaged file
        os.remove(cif_path)
        print(f"  → Deleted {cif_path}")
        return False

    # Compare sequences, remove if doesn't match 
    if expected_seq != None:
        try:
            struct_seq = str(chains_sequences.get("A"))
        except Exception as err:
            print(f"[PARSE ERROR] {protein_id}: {err}")
            # Remove demaged file
            os.remove(cif_path)
            print(f"  → Deleted {cif_path}")
            return False
        if struct_seq != expected_seq:  
            print(f"[MISMATCH] {protein_id}: expected {len(expected_seq)} aa, got {len(struct_seq)} aa")
            os.remove(cif_path)
            print(f"  → Deleted {cif_path}")
            return False
    
    return True


def _process_seq(args):
    """
    Проверяет одну пару (cl1, uid): 
    1) наличие структуры для UniProt uid 
    2) для каждого его pdb_id — тоже проверяем
    Возвращает (cl1, uid, num_structures, [valid_pdbs]) или None.
    """
    cl1, uid = args
    # проверяем UniProt-структуру
    useq = protein_data[uid]["uniprot_sequence"]
    if not structure_exists(uid, useq):
        return None
    # начальное количество = 1 (сама UniProt)
    cnt = 1
    valid_pdbs = []
    # пробегаем по pdb_ids
    for pdb in protein_data.get(uid, {}).get("pdb_ids", []):
        if pdb_counter[pdb] == 1 and structure_exists(pdb):
            cnt += 1
            valid_pdbs.append(pdb)
    return cl1, uid, cnt, valid_pdbs


cluster_level_1_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_1_cluster.tsv"
cluster_level_2_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_2_cluster.tsv"
protein_table_path  = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"

pkl_folder_path = "/home/iscb/wolfson/annab4/DB/all_proteins/batches/"
output_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v99"
final_dataset_path = os.path.join(output_dir, "dataset.csv")

os.makedirs(output_dir, exist_ok=True)

cluster_level_1 = pd.read_csv(cluster_level_1_path, sep='\t', header=None,
                              names=['Cluster_1', 'Sequence_ID'])
cluster_level_2 = pd.read_csv(cluster_level_2_path, sep='\t', header=None,
                              names=['Cluster_2', 'Centroid_ID'])
with open(protein_table_path) as f:
    protein_data = json.load(f)

pdb_counter = Counter()
for rec in protein_data.values():
    pdb_counter.update(rec.get("pdb_ids", []))
    
for key, value in pdb_counter.items():
    if value > 1:
        print(f"PDB {key} встречается {value} раз")

pdb_to_uniprot = {}
cluster_1_to_seqs = defaultdict(list)
structures_cnt = {}

# 1) Готовим list of tasks
tasks = [
    (cl1, uid)
    for cl1, grp in cluster_level_1.groupby("Cluster_1")
    for uid in grp["Sequence_ID"]
]

# 2) Параллельно обрабатываем
with Pool(n_procs) as pool:
    results = pool.map(_process_seq, tasks)

# 3) Собираем успешные результаты
for item in results:
    if item is None:
        continue
    cl1, uid, cnt, valid_pdbs = item
    # добавляем UniProt-uid
    cluster_1_to_seqs[cl1].append(uid)
    structures_cnt[uid] = cnt
    # и все валидные pdb
    for pdb in valid_pdbs:
        cluster_1_to_seqs[cl1].append(pdb)
        pdb_to_uniprot[pdb] = uid

# with open("/home/iscb/wolfson/annab4/DB/all_proteins/pdb_to_uniprot.json", "w") as file: #TODO: вставить это в какое-то более уместное место. и обратный словарь тоже
#     json.dump(pdb_to_uniprot, file)

    
# 4) Убираем дубликаты
for cl1 in cluster_1_to_seqs:
    cluster_1_to_seqs[cl1] = list(set(cluster_1_to_seqs[cl1]))

    
seq_to_c1 = {}
for cl1, seqs in cluster_1_to_seqs.items():
    for seq in seqs:
        seq_to_c1[seq] = cl1

centroid_to_c2 = {r['Centroid_ID']: r['Cluster_2'] for _, r in cluster_level_2.iterrows()}
cluster_2_to_1 = defaultdict(list)
for cl1, seqs in cluster_1_to_seqs.items():
    c2 = centroid_to_c2.get(cl1)
    if c2:
        cluster_2_to_1[c2].append(cl1)

cluster_1_to_2 = {}
for c2, cls in cluster_2_to_1.items():
    for cl1 in cls:
        cluster_1_to_2[cl1] = c2

for cl1, seqs in cluster_1_to_seqs.items():
    for uid in seqs:
        if uid in protein_data:
            protein_data[uid].update({
                "Cluster_1": cl1,
                "Cluster_2": cluster_1_to_2.get(cl1)
            })
            
# ==== map every node (UniProt и PDB) to its Cluster_2 ====
seq_to_c2 = {}
for cl1, seqs in cluster_1_to_seqs.items():
    c2 = cluster_1_to_2.get(cl1)
    if c2 is None:
        # логируем, если вдруг кто-то остался без C2
        print(f"Warning: cluster1={cl1} has no Cluster_2")
        continue
    for seq in seqs:
        seq_to_c2[seq] = c2

# sanity check
missing = [n for cl in cluster_1_to_seqs.values() for n in cl if n not in seq_to_c2]
if missing:
    print("Nodes without seq_to_c2 mapping:", missing)


# create a graph
valid_nodes = set(seq_to_c2.keys())

G = nx.Graph()
G.add_nodes_from(valid_nodes)

for pdb, uni in pdb_to_uniprot.items():
    if pdb in valid_nodes and uni in valid_nodes:
        G.add_edge(uni, pdb)

# edges by Cluster_2 over ALL seq_to_c2 nodes (UniProt + PDB)
by_c2 = defaultdict(list)
for seq, c2 in seq_to_c2.items():
    by_c2[c2].append(seq)
for seqs in by_c2.values():
    if len(seqs) > 1:
        G.add_edges_from(combinations(seqs, 2))

# edges by EC
uids = list(protein_data.keys())
ecs  = []
for uid in uids:
    ec = protein_data[uid].get("EC_number", "not found")
    if valid_ec_number(ec):
        ec = '.'.join(ec.split('.')[:3])
    else:
        ec = "not found"
    ecs.append(ec)

ec_map = defaultdict(list)
for uid, ec in zip(uids, ecs):
    if ec != "not found":
        ec_map[ec].append(uid)
        
for ec, uids in ec_map.items():
    filt = [u for u in uids if u in valid_nodes]
    if len(filt) > 1:
        G.add_edges_from(combinations(filt, 2))

components = list(nx.connected_components(G))

# print EC numbers for each component 
def trunc(ec): return '.'.join(ec.split('.')[:3]) if ec!="not found" else ec
for i, comp in enumerate(components, 1):
    ecs = {trunc(protein_data[n].get("EC_number","not found")) for n in comp if n in protein_data}
    tag = "single" if len(ecs)==1 else f"multiple {ecs}"
    print(f"Component {i}: {tag}, size={len(comp)}")

# round-robin to subsample big components
max_comp_size = 3000
subsampled_components = []
for comp in components:
    if len(comp) <= max_comp_size:
        subsampled_components.append(list(comp))
    else:
        by_c2 = defaultdict(list)
        for uid in comp:
            by_c2[seq_to_c2.get(uid)].append(uid)
        for lst in by_c2.values():
            random.shuffle(lst)
        keys = list(by_c2)
        idxs = {k:0 for k in keys}
        sel = []
        while len(sel) < max_comp_size:
            for k in keys:
                if idxs[k] < len(by_c2[k]):
                    sel.append(by_c2[k][idxs[k]])
                    idxs[k] += 1
                if len(sel) >= max_comp_size:
                    break
        subsampled_components.append(sel)

# parse annotations to get labels and amino acids
def process_batch_file(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    out = {}
    curr = None
    annots = []
    for line in data:
        if (isinstance(line, bytes) and line.startswith(b'>')) or (isinstance(line,str) and line.startswith('>')):
            if curr is not None:
                out[curr] = annots.copy()
            curr = line.lstrip(b'>').decode() if isinstance(line,bytes) else line[1:].strip()
            annots = []
        else:
            annots.append(line)
    if curr is not None:
        out[curr] = annots.copy()
    return out

batch_files = glob(os.path.join(pkl_folder_path, "batch*_annotations.pkl"))

def _parse_batch_file(fn):
    return process_batch_file(fn)  # возвращает {chain_id: annots}

with Pool(n_procs) as pool:
    all_batches = pool.map(_parse_batch_file, batch_files)

# слить в один словарь
sequences_dict = {}
labels_dict   = {}
no_positive_seqids = set()
for batch in all_batches:
    for chain_id, ann in batch.items():
        seq  = []
        labs = []
        for line in ann:
            aa, lab = line.strip().split()[2], int(line.strip().split()[3])
            seq.append(aa); labs.append(lab)
        if sum(labs)==0:
            no_positive_seqids.add(chain_id)
        sequences_dict[chain_id] = "".join(seq)
        labels_dict  [chain_id] = labs


# collect only catalytic residues
# Собираем mapping chain_id → префикс (UniProt / PDB ID)
all_chains = set(sequences_dict.keys())
chains_by_prefix = defaultdict(list)
for chain in all_chains:
    prefix = chain.rsplit('_', 1)[0]
    chains_by_prefix[prefix].append(chain)

annotated_prefixes = set(chains_by_prefix.keys())
print(f"Annotated sequences: {annotated_prefixes}")

filtered_components = []
for comp in subsampled_components:
    good = [seq_id for seq_id in comp if seq_id in annotated_prefixes]
    missing = set(comp) - set(good)
    if missing:
        print(f"Deleted seq_id without annotations: {missing}")
    if good:
        filtered_components.append(good)

subsampled_components = filtered_components

def _compute_residues_for_seq(seq_id):
    residues = []
    for chain in chains_by_prefix[seq_id]:
        seq  = sequences_dict[chain]
        labs = labels_dict   [chain]
        residues += [aa for aa, lab in zip(seq, labs) if lab==1]
    return seq_id, residues

# main
seq_ids = [seq for comp in subsampled_components for seq in comp]
with Pool(n_procs) as pool:
    items = pool.map(_compute_residues_for_seq, seq_ids)
catalytic_residue_dict = dict(items)

# assign class based on chemical properties of catalutic sites
N_CLASSES = 8
def get_catalytic_class(residues):             # add more classes?
    if any(r in residues for r in "ILMVWF"):
        return 0
    if any(r in residues for r in "AGP"):   
        return 1
    if any(r in residues for r in "QN"):     
        return 2
    if any(r in residues for r in "KR"):     
        return 3
    if any(r == "S"  for r in residues):       
        return 4
    if any(r == "T"  for r in residues):       
        return 5
    if any(r in residues for r in "DE"):       
        return 6
    return 7

c1_counts = {}
for cl1, seqs in cluster_1_to_seqs.items():
    # UniProt-ID — это те seq, которых нет в словаре pdb_to_uniprot
    uni_seqs = [uid for uid in seqs if uid in protein_data]
    c1_counts[cl1] = len(uni_seqs)
    
c2_counts = {c2: len(c1s) for c2, c1s in cluster_2_to_1.items()}

missing = {nid
           for comp in subsampled_components
           for nid in comp
           if nid not in seq_to_c2}
print("Ноды без seq_to_c2:", missing)

for pdb in missing:
    uid = pdb_to_uniprot.get(pdb)
    c1  = seq_to_c1.get(uid)
    c2  = centroid_to_c2.get(c1)
    print(f"PDB {pdb} ← UniProt {uid}, Cluster_1={c1}, Cluster_2={c2}")



# 1) Capped W_Cluster_2
max_W_c2 = 100.0
W_c2_by_c2 = {c2: min(max_W_c2, float(cnt)) for c2, cnt in c2_counts.items()}

# 2) per-seq W_Cluster_2 and W_Cluster_1
W_c2 = {}
W_c1 = {}
for comp in subsampled_components:
    for seq in comp:
        c2 = seq_to_c2.get(seq)
        c2_size = float(c2_counts.get(c2, 1))
        w_c2 = W_c2_by_c2.get(c2, 1.0)           # min(max_W_c2, c2_size)
        W_c2[seq] = w_c2
        # W_Cluster_1 = W_c2 / (#Cluster_1 in this Cluster_2) = min(1, max_W_c2 / c2_size)
        W_c1[seq] = w_c2 / c2_size

# 3) W_Sequence = W_Cluster_1 / (#UniProt in Cluster_1)
W_sequence = {}
for comp in subsampled_components:
    for seq in comp:
        c1 = seq_to_c1[seq]
        n_uniprot = float(c1_counts.get(c1, 1))
        W_sequence[seq] = W_c1[seq] / n_uniprot

# 4) W_Structure = W_Sequence / (#structures for UniProt)
W_structure = {}
for comp in subsampled_components:
    for seq in comp:
        parent = pdb_to_uniprot.get(seq, seq)
        n_struct = float(structures_cnt.get(parent, 1))
        W_structure[seq] = W_sequence[seq] / n_struct

# split into folds
N_SPLITS=5

components_weights = np.zeros((len(subsampled_components), N_CLASSES + 1), dtype=float)
for i, comp in enumerate(subsampled_components):
    # total
    components_weights[i, 0] = sum(W_structure[s] for s in comp)
    # classes: 
    for s in comp:
        cl = get_catalytic_class(catalytic_residue_dict[s])  # 0..N_CLASSES-1
        components_weights[i, cl + 1] += W_structure[s]

dataset_weights = components_weights.sum(axis=0)           # shape: (N_CLASSES+1,)
ideal_fold_weights = dataset_weights / N_SPLITS            # N_SPLITS=5
print(f"Ideal weights: {list(ideal_fold_weights)}")
print()

def discrepancy_metric(fold_w, ideal_w):
    denom = np.maximum(ideal_w, 1e-12)               
    diff = np.abs(fold_w - ideal_w) / denom
    mul = np.array([N_CLASSES*3] + [1]*N_CLASSES)      
    return float(np.dot(mul, diff))

def global_score(fw_list):
    # MINIMAX over folds instead of sum/mean
    return max(discrepancy_metric(fw, ideal_fold_weights) for fw in fw_list)

# init
fold_weights = [np.zeros(N_CLASSES + 1, dtype=float) for _ in range(N_SPLITS)]
component_fold = [-1] * len(subsampled_components)

# optional: seed the heaviest N_SPLITS components one-per-fold
order = np.argsort(-components_weights[:, 0])
for j in range(min(N_SPLITS, len(order))):
    i = int(order[j])
    component_fold[i] = j
    fold_weights[j]  += components_weights[i]

# greedy with MINIMAX objective
for pos in range(N_SPLITS, len(order)):
    i = int(order[pos])

    # current worst-fold score
    base = global_score(fold_weights)

    # try placing component i into each fold
    candidates = []
    for k in range(N_SPLITS):
        fold_weights[k] += components_weights[i]
        sc = global_score(fold_weights)
        candidates.append((sc, k))
        fold_weights[k] -= components_weights[i]

    # pick placement that minimizes the worst-fold score
    best_sc, best_k = min(candidates, key=lambda x: x[0])

    # tie-breaker: among ties, pick the fold with the smallest current total
    ties = [k for sc,k in candidates if np.isclose(sc, best_sc)]
    if len(ties) > 1:
        best_k = min(ties, key=lambda k: fold_weights[k][0])

    component_fold[i] = best_k
    fold_weights[best_k] += components_weights[i]

# ----------------- ONE-PASS LOCAL REFINEMENT (move-1) -----------------
EPS = 1e-12

def local_refine_full_1opt(components_weights, fold_weights, component_fold, n_splits, max_iters=100):
    n_comp = components_weights.shape[0]
    improved_any = False
    for _ in range(max_iters):
        base = global_score(fold_weights)
        best_gain, best_move = 0.0, None
        for i in range(n_comp):
            k = component_fold[i]
            for kk in range(n_splits):
                if kk == k: continue
                fold_weights[k]  -= components_weights[i]
                fold_weights[kk] += components_weights[i]
                sc = global_score(fold_weights)
                fold_weights[kk] -= components_weights[i]
                fold_weights[k]  += components_weights[i]
                gain = base - sc
                if gain > best_gain + EPS:
                    best_gain, best_move = gain, (i, k, kk)
        if best_move is None:
            break
        i, k, kk = best_move
        fold_weights[k]  -= components_weights[i]
        fold_weights[kk] += components_weights[i]
        component_fold[i] = kk
        improved_any = True
    return improved_any

def local_refine_swaps_2opt(components_weights, fold_weights, component_fold, n_splits, max_pairs_per_fold=50):
    base = global_score(fold_weights)
    best_gain, best_swap = 0.0, None
    comp_in_fold = [np.where(np.array(component_fold) == f)[0] for f in range(n_splits)]
    small_in_fold = []
    for f in range(n_splits):
        idx = comp_in_fold[f]
        if idx.size == 0: 
            small_in_fold.append(idx); 
            continue
        order = idx[np.argsort(components_weights[idx, 0])]
        small_in_fold.append(order[:max_pairs_per_fold])
    # пробуем пары фолдов (хватает всех пар; если долго — ограничься топ-3 худшими)
    for a in range(n_splits):
        for b in range(a+1, n_splits):
            A, B = small_in_fold[a], small_in_fold[b]
            if A.size == 0 or B.size == 0: 
                continue
            for i in A:
                for j in B:
                    fold_weights[a] -= components_weights[i]
                    fold_weights[b] += components_weights[i]
                    fold_weights[b] -= components_weights[j]
                    fold_weights[a] += components_weights[j]
                    sc = global_score(fold_weights)
                    fold_weights[a] += components_weights[i]
                    fold_weights[b] -= components_weights[i]
                    fold_weights[b] += components_weights[j]
                    fold_weights[a] -= components_weights[j]
                    gain = base - sc
                    if gain > best_gain + EPS:
                        best_gain, best_swap = gain, (a, b, i, j)
    if best_swap is None:
        return False
    a, b, i, j = best_swap
    fold_weights[a] -= components_weights[i]
    fold_weights[b] += components_weights[i]
    fold_weights[b] -= components_weights[j]
    fold_weights[a] += components_weights[j]
    component_fold[i] = b
    component_fold[j] = a
    return True

# итеративно до «тишины»
# changed = True
# iters = 0
# while changed and iters < 100:
#     changed = False
#     if local_refine_full_1opt(components_weights, fold_weights, component_fold, N_SPLITS, max_iters=3):
#         changed = True
#     if local_refine_swaps_2opt(components_weights, fold_weights, component_fold, N_SPLITS, max_pairs_per_fold=80):
#         changed = True
#     iters += 1
# print(f"[refine] passes: {iters}")

def refine_double_relocate(components_weights, fold_weights, component_fold, n_splits,
                           max_iters=10, max_src_small=80, max_src_large=10, max_targets=3, eps=1e-12):
    """
    На каждом проходе выбираем самый тяжёлый фолд и пытаемся перенести сразу ДВЕ компоненты
    (обычно мелкие) в фолды с наибольшим дефицитом тотала. Оцениваем все пары из ограниченного
    набора кандидатов; применяем лучший улучшающий ход. Повторяем несколько раз.
    """
    def fold_score(fw):
        return discrepancy_metric(fw, ideal_fold_weights)
    def global_score(fw_list):
        return max(fold_score(w) for w in fw_list)

    improved_any = False
    for _ in range(max_iters):
        # выбираем самый "плохой" фолд по minimax-оценке
        scores = [fold_score(w) for w in fold_weights]
        worst = int(np.argmax(scores))

        # дефициты тотала по фолдам относительно идеала
        totals  = np.array([fw[0] for fw in fold_weights])
        target  = ideal_fold_weights[0]
        deficits = target - totals  # >0 значит недовес

        # кандидаты-источники из худшего фолда: в основном мелкие, плюс немного крупных
        in_worst = np.where(np.array(component_fold) == worst)[0]
        if in_worst.size < 2:
            break
        order_asc  = in_worst[np.argsort(components_weights[in_worst, 0])]
        order_desc = order_asc[::-1]
        cand = np.unique(np.concatenate([order_asc[:max_src_small], order_desc[:max_src_large]]))

        # целевые фолды: с наибольшим дефицитом тотала
        tgt = [f for f in np.argsort(-deficits)[:max_targets] if f != worst]
        if not tgt:
            break

        base = global_score(fold_weights)
        best_gain = 0.0
        best_move = None

        # перебираем пары компонент (i,j) и назначения (k1,k2) — k1/k2 могут совпадать
        from itertools import combinations, product
        pairs = list(combinations(cand, 2))
        for i, j in pairs:
            wi = components_weights[i]; wj = components_weights[j]
            for k1, k2 in product(tgt, repeat=2):
                # временно применяем двойной перенос
                fold_weights[worst] -= wi; fold_weights[k1] += wi
                fold_weights[worst] -= wj; fold_weights[k2] += wj
                sc = global_score(fold_weights)
                # откат
                fold_weights[k2] -= wj; fold_weights[worst] += wj
                fold_weights[k1] -= wi; fold_weights[worst] += wi

                gain = base - sc
                if gain > best_gain + eps:
                    best_gain = gain
                    best_move = (i, j, k1, k2)

        if best_move is None:
            break  # улучшений на этом уровне нет
        # применяем лучший двойной перенос
        i, j, k1, k2 = best_move
        fold_weights[worst] -= components_weights[i]; fold_weights[k1] += components_weights[i]
        fold_weights[worst] -= components_weights[j]; fold_weights[k2] += components_weights[j]
        component_fold[i] = k1
        component_fold[j] = k2
        improved_any = True

    return improved_any

# improved = True
# iterations = 0
# while improved and iterations < 30:
#     improved = refine_double_relocate(components_weights, fold_weights, component_fold, N_SPLITS,
#                            max_iters=8, max_src_small=100, max_src_large=12, max_targets=3)
#     iterations += 1
# print(f"Improved {iterations} times")


# sanity check
assert np.allclose(np.sum(fold_weights, axis=0), dataset_weights), "Fold weights don't sum to dataset!"



# === 4) разворачивание на seq_id ===
split_sets = [set() for _ in range(N_SPLITS)]
for comp_idx, comp in enumerate(subsampled_components):
    split_sets[component_fold[comp_idx]].update(comp)

set_mapping = {}
for i, s in enumerate(split_sets,1):
    for cid in s:
        set_mapping[cid] = f"split{i}"


# create PrettyTable here to pring the weights
table = PrettyTable()
table.field_names = ["Split", "Total weight", "Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Other"]
for f_idx, f_weight in enumerate(fold_weights):
    table.add_row([f_idx + 1] + list(f_weight))
print("Folds weights")
print(table)

# split sequences
split_data = [{} for _ in range(N_SPLITS)]
for i in range(1,101):
    fn = os.path.join(pkl_folder_path, f"batch{i}_annotations.pkl")
    if not os.path.isfile(fn): continue
    batch, chains = process_batch_file(fn), None  # повторный парсинг
    for chain_id, ann in batch.items():
        if chain_id in no_positive_seqids:
            continue
        seq_id = chain_id.rsplit('_', 1)[0]
        st = set_mapping.get(seq_id)
        if st is None: continue
        idx = int(st.replace("split",""))-1
        split_data[idx][chain_id] = ann

all_txt = {}
for d in split_data:
    all_txt.update(d)

# build dataframe
component_mapping = {
    cid: comp_idx+1
    for comp_idx, comp in enumerate(subsampled_components)
    for cid in comp
}

final_data = []
for chain_id, ann in all_txt.items():
    prefix = chain_id.split('_')[0]
    uniprot = pdb_to_uniprot.get(prefix, prefix)
    if uniprot not in protein_data:
        print(f"Warning: no record for {chain_id}")
        continue
    rec = protein_data[uniprot]
    final_data.append({
        "Sequence_ID": chain_id,
        "Cluster_1":   rec.get("Cluster_1"),
        "Cluster_2":   rec.get("Cluster_2"),
        "Set_Type":    set_mapping.get(prefix),
        "EC_number":   rec.get("EC_number"),
        "Component_ID": component_mapping.get(prefix),
        "full_name":   rec.get("full_name")
    })

print(f"Found {len(all_txt)} relevant chains")
print(f"DataFrame size is {len(final_data)}")

final_df = pd.DataFrame(final_data)

# --- helpers ---
max_W_c2 = 100.0

def get_parent(cid):
    return pdb_to_uniprot.get(cid.split('_')[0], cid.split('_')[0])

# родитель и число структур у родителя
final_df['Parent_ID'] = final_df['Sequence_ID'].map(get_parent)
struct_counts = final_df.groupby('Parent_ID')['Sequence_ID'].nunique()

# сколько Cluster_1 в каждом Cluster_2 (+cap)
c2_count = final_df.groupby('Cluster_2')['Cluster_1'].nunique()
final_df['c2_n_cl1']  = final_df['Cluster_2'].map(c2_count).astype(float)
final_df['c2_capped'] = final_df['c2_n_cl1'].clip(upper=max_W_c2)

# W_Cluster_2 = min(max_W_c2, #Cluster_1 in Cluster_2)
final_df['W_Cluster_2'] = final_df['c2_capped']

# W_Cluster_1 = W_Cluster_2 / (#Cluster_1 in Cluster_2) = min(1, max_W_c2 / #Cluster_1_in_C2))
final_df['W_Cluster_1'] = (final_df['c2_capped'] / final_df['c2_n_cl1'])

# #UniProt in Cluster_1
c1_parent_count = final_df.groupby('Cluster_1')['Parent_ID'].nunique()
final_df['c1_n_parents'] = final_df['Cluster_1'].map(c1_parent_count).astype(float)

# W_Sequence = W_Cluster_1 / (#UniProt in Cluster_1)
final_df['W_Sequence'] = (final_df['W_Cluster_1'] / final_df['c1_n_parents'])

# W_Structure = W_Sequence / (#сstructures for UniProt)
final_df['structures_for_parent'] = final_df['Parent_ID'].map(struct_counts).astype(float)
final_df['W_Structure'] = final_df['W_Sequence'] / final_df['structures_for_parent']

final_df[['W_Cluster_1','W_Cluster_2','W_Sequence','W_Structure']] = \
    final_df[['W_Cluster_1','W_Cluster_2','W_Sequence','W_Structure']].fillna(1.0)

final_df.drop(columns=['Parent_ID','c2_n_cl1','c2_capped','c1_n_parents','structures_for_parent'],
              inplace=True)
final_df.to_csv(final_dataset_path, index=False)


# save annotations
def save_annotations(data, out_fn):
    with open(out_fn, 'w') as f:
        for cid, ann in data.items():
            f.write(f">{cid}\n")
            for entry in ann:
                f.write(f"{entry}\n")

for i, d in enumerate(split_data,1):
    save_annotations(d, os.path.join(output_dir, f"split{i}.txt"))

print("Datasets have been split into 5 folds and files saved.")
