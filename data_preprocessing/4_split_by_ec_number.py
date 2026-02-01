import json
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import random
import os
import pickle
import numpy as np
from itertools import combinations
from copy import deepcopy

# Check if the EC number has three first numbers
def valid_ec_number(ec_number):
    if ec_number == "not found":
        return False
    parts = ec_number.split('.')
    if len(parts) < 3:
        return False
    return all(part.isdigit() for part in parts[:3]) 

# Paths to files
cluster_level_1_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_1_cluster.tsv"
cluster_level_2_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_2_cluster.tsv"
protein_table_path = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"

pkl_folder_path = "/home/iscb/wolfson/annab4/DB/all_proteins/batches/"
output_dir = "/home/iscb/wolfson/annab4/DB/all_proteins/splitted5"
final_dataset_path = "/home/iscb/wolfson/annab4/DB/all_proteins/dataset_tables5.csv"

# Load clustering and protein data
cluster_level_1 = pd.read_csv(cluster_level_1_path, sep='\t', header=None, names=['Cluster_1', 'Sequence_ID'])
cluster_level_2 = pd.read_csv(cluster_level_2_path, sep='\t', header=None, names=['Cluster_2', 'Centroid_ID'])
with open(protein_table_path) as f:
    protein_data = json.load(f)

# Prepare sequence structures count and pdb_to_uniprot mapping
structure_counts = {seq_id: len(data.get("pdb_ids", [])) + 1 for seq_id, data in protein_data.items()}
pdb_to_uniprot = {}

# Populate cluster_1_to_seqs with uniprot and pdb entries
cluster_1_to_seqs = defaultdict(list)
for cluster_1, row in cluster_level_1.groupby("Cluster_1"):
    seq_ids = row["Sequence_ID"].tolist()
    for uniprot_id in seq_ids:
        cluster_1_to_seqs[cluster_1].append(uniprot_id)
        for pdb_id in protein_data.get(uniprot_id, {}).get("pdb_ids", []):
            cluster_1_to_seqs[cluster_1].append(pdb_id)
            pdb_to_uniprot[pdb_id] = uniprot_id

# Map clusters for Cluster_1 and Cluster_2
cluster_2_to_cluster_1 = defaultdict(list)
centroid_to_cluster_2 = {row['Centroid_ID']: row['Cluster_2'] for _, row in cluster_level_2.iterrows()}
for cluster_1, seq_ids in cluster_1_to_seqs.items():
    cluster_2 = centroid_to_cluster_2.get(cluster_1)
    if cluster_2:
        cluster_2_to_cluster_1[cluster_2].append(cluster_1)

cluster_1_to_cluster_2 = dict()
for cluster_2, clusters_1 in cluster_2_to_cluster_1.items():
    for cluster_1 in clusters_1:
        cluster_1_to_cluster_2[cluster_1] = cluster_2

# Calculate initial weights for W_Cluster_1 and W_Sequence
for cluster_1, seq_ids in cluster_1_to_seqs.items():
    uniprot_count = sum(1 for seq_id in seq_ids if seq_id in protein_data)
    W_Cluster_1 = 1.0
    W_Cluster_2 = len(cluster_2_to_cluster_1[ cluster_1_to_cluster_2[cluster_1] ])
    W_Sequence = 1.0 / uniprot_count if uniprot_count > 0 else 1.0
    
    for seq_id in seq_ids:
        if seq_id in protein_data:
            protein_data[seq_id].update({
                "Cluster_1" : cluster_1,
                "Cluster_2" : cluster_1_to_cluster_2[cluster_1],
                "W_Cluster_1": W_Cluster_1,
                "W_Cluster_2" : W_Cluster_2,
                "W_Sequence": W_Sequence
            })

# Calculate W_Structure for all sequences, including pdb chains
for sequence_id in protein_data:
    num_structures = structure_counts.get(sequence_id, 1)
    W_Structure = protein_data[sequence_id]["W_Sequence"] / num_structures
    protein_data[sequence_id]["W_Structure"] = W_Structure
    
pdb_counter = Counter()
for rec in protein_data.values():
    pdb_counter.update(rec.get("pdb_ids", []))

############## graph clustering ##############

# Map each sequence to its cluster_2 for fast lookup
cluster_2_mapping = cluster_level_2.set_index("Centroid_ID")["Cluster_2"].to_dict()
sequence_to_cluster_2 = cluster_level_1.assign(Cluster_2=cluster_level_1["Cluster_1"].map(cluster_2_mapping))

# Initialize a graph where each uniprot ID will be a node
graph = nx.Graph()

for seq_id in protein_data:
    graph.add_node(seq_id)
    
# Add PDBs to graph
for seq_id, record in protein_data.items():
    for pdb_id in record.get("pdb_ids", []):
        if pdb_counter[pdb_id] == 1:
            graph.add_node(pdb_id)
            graph.add_edge(seq_id, pdb_id)

# Add edges for each pair within the same `Cluster_2`
for cluster_2, group in sequence_to_cluster_2.groupby("Cluster_2"):
    ids = group["Sequence_ID"].values
    if len(ids) > 1:
        graph.add_edges_from(combinations(ids, 2))

# Add edges for each pair within the same `Cluster_1`     
for cl1, members in cluster_1_to_seqs.items():
    if len(members) > 1:
        graph.add_edges_from(combinations(members, 2))

# Process EC numbers and use dictionary to group UniProt IDs by EC groups
uniprot_ids = np.array(list(protein_data.keys()))
ec_numbers = []
for uniprot_id in uniprot_ids:
    ec_number = protein_data[uniprot_id].get("EC_number", "not found")
    if valid_ec_number(ec_number):
        ec_number = '.'.join(ec_number.split('.')[:3])
    else:
        ec_number = "not found"
    ec_numbers.append(ec_number)
    

ec_group_dict = defaultdict(list)

# Group UniProt IDs by the first three fields of their EC number, excluding "not found"
for idx, ec_group in enumerate(ec_numbers):
    if ec_group != "not found":  # Exclude groups with "not found"
        ec_group_dict[ec_group].append(uniprot_ids[idx])

# Add edges for UniProt IDs in the same EC group and log EC differences
for ids in ec_group_dict.values():
    if len(ids) > 1:
        graph.add_edges_from(combinations(ids, 2))

# Identify connected components and label each with a unique Component_ID
components = list(nx.connected_components(graph))


# Analyze components
def truncate_ec_number(ec_number):
    if ec_number != "not found":
        return '.'.join(ec_number.split('.')[:3])
    else:
        return "not found"
    
total_node_count = 0  # Total number of nodes
total_uniprot_count = 0  # Total number of UniProt IDs
for idx, component in enumerate(components):
    total_node_count += len(component)
    total_uniprot_count += len([node for node in component if node in protein_data])
    ec_numbers_in_component = {truncate_ec_number(protein_data[node].get("EC_number", "not found")) 
                               for node in component if node in protein_data}
    if len(ec_numbers_in_component) > 1:
        print(f"Component {idx+1} contains multiple EC groups: {ec_numbers_in_component}, size {len(component)}")
    else:
        print(f"Component {idx+1} contains a single EC group: {ec_numbers_in_component}, size {len(component)}")

# Initialize sets
train_set, validate_set, test_set = set(), set(), set()
current_train, current_validate, current_test = 0, 0, 0

# Random subsampling
# max_component_size = 2000
# subsampled_components = []
# for component in components:
#     if len(component) > max_component_size:
#         subsampled_component = np.random.choice(np.array(list(component)), 
#                                                 size=max_component_size, 
#                                                 replace=False)
#         subsampled_components.append(list(subsampled_component))
#     else:
#         subsampled_components.append(component)

# components = subsampled_components

# Target sizes
total_nodes = sum(len(component) for component in components)
print(f"Total nodes calculated: {total_nodes}")

train_target, validate_target, test_target = 0.6 * total_nodes, 0.2 * total_nodes, 0.2 * total_nodes

# Shuffle components to ensure randomness

random.shuffle(components)

# Distribute components, balancing by the smallest current fill percentage
for component in components:
    component_size = len(component)

    train_fill = current_train / train_target
    validate_fill = current_validate / validate_target
    test_fill = current_test / test_target

    # Choose the set with the smallest fill percentage
    if train_fill <= validate_fill and train_fill <= test_fill:
        train_set.update(component)
        current_train += component_size
    elif validate_fill <= train_fill and validate_fill <= test_fill:
        validate_set.update(component)
        current_validate += component_size
    else:
        test_set.update(component)
        current_test += component_size

# Final confirmation of sizes
print(f"Train size: {current_train} ({current_train / total_nodes * 100:.2f}%)")
print(f"Validate size: {current_validate} ({current_validate / total_nodes * 100:.2f}%)")
print(f"Test size: {current_test} ({current_test / total_nodes * 100:.2f}%)")

# Assign `Set_Type` and `Component_ID` to each sequence in final dataset
set_mapping = {seq_id: "train" for seq_id in train_set}
set_mapping.update({seq_id: "validate" for seq_id in validate_set})
set_mapping.update({seq_id: "test" for seq_id in test_set})

component_mapping = {node: idx + 1 for idx, component in enumerate(components) for node in component}

# Prepare directories and datasets for train/validate/test
train_data, val_data, test_data = {}, {}, {}

def process_batch_file(batch_file):
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)
    data = {}
    chains = []
    prev_id_chain = ""
    annotations = []
    for line in batch_data:
        if line[0] == '>':
            chains.append(line.lstrip(">")) 
            if annotations:
                data[prev_id_chain[1:]] = annotations.copy()
            annotations = []
            prev_id_chain = line
        else:
            annotations.append(line)
    return data, chains

chains = []

# Process each batch file and assign based on the new Set_Type
for i in range(1, 101):
    batch_file = f'/home/iscb/wolfson/annab4/DB/all_proteins/batches/batch{i}_annotations.pkl'
    if os.path.isfile(batch_file):
        batch_data, file_chains = process_batch_file(batch_file)
        chains = chains + file_chains
        for id_chain, annotations in batch_data.items():
            seq_id = id_chain.split('_')[0]
            set_type = set_mapping.get(seq_id)
            if set_type == 'train':
                train_data[id_chain] = annotations
            elif set_type == 'validate':
                val_data[id_chain] = annotations
            elif set_type == 'test':
                test_data[id_chain] = annotations
    else:
        print(f"Batch file {batch_file} not found, skipping.")
        
        
######## правильная таблица ##################
final_data = []

def process_chains(seq_id, relevant_chains, protein_info, set_type):
    for chain in relevant_chains:
        final_data.append({
            "Sequence_ID": chain,
            "Cluster_1": protein_info.get("Cluster_1"),
            "Cluster_2": protein_info.get("Cluster_2"),
            "W_Structure": protein_info.get("W_Structure"),
            "W_Sequence": protein_info.get("W_Sequence"),
            "W_Cluster_1": protein_info.get("W_Cluster_1"),
            "W_Cluster_2": protein_info.get("W_Cluster_2"),
            "Set_Type": set_mapping.get(seq_id),
            "EC_number": protein_info.get("EC_number"),
            "Component_ID" : component_mapping.get(seq_id),
            "full_name": protein_info.get("full_name")
        })
        
        
# Обработка каждого белка (uniprot_id) и его связанных pdb_id
selected_nodes = set(component_mapping) & set(set_mapping)

for seq_id, record in protein_data.items():
    if not seq_id in selected_nodes:
        continue
    set_type = set_mapping.get(seq_id)
    # Преобразование строки с pdb_ids в список
    pdb_ids = record.get("pdb_ids", [])

    # Find chains of this uniprot_id
    relevant_chains = [chain for chain in chains if seq_id in chain]

    # Process chains of uniprot_id
    if relevant_chains:
        process_chains(seq_id, relevant_chains, record, set_type)
    
    # Process PDBs for this uniprot_id
    for pdb_id in pdb_ids:
        if not pdb_id in selected_nodes:
            continue
        relevant_chains_pdb = [chain for chain in chains if pdb_id in chain]
        
        # Process chains for this pdb_id
        if relevant_chains_pdb:
            process_chains(pdb_id, relevant_chains_pdb, record, set_type)

# Шаг 4: Сохранение итоговой таблицы
final_df = pd.DataFrame(final_data)
final_df.to_csv(final_dataset_path, index=False)


# Function to save annotations to text files
def save_annotations(data, output_file):
    with open(output_file, 'w') as f:
        for id_chain, annotations in data.items():
            f.write(f">{id_chain}\n")
            for annotation in annotations:
                f.write(f"{annotation}\n")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save each dataset
save_annotations(train_data, os.path.join(output_dir, 'train.txt'))
save_annotations(val_data, os.path.join(output_dir, 'validate.txt'))
save_annotations(test_data, os.path.join(output_dir, 'test.txt'))

print("Datasets have been split, and data files have been saved successfully.")
