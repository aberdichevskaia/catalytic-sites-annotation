import json
import pandas as pd
from collections import defaultdict
import os

# Paths to files
cluster_level_1_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_1_cluster.tsv"
cluster_level_2_path = "/home/iscb/wolfson/annab4/DB/clustering/cluster_level_2_cluster.tsv"
protein_table_path = "/home/iscb/wolfson/annab4/DB/all_protein_table.json"
output_protein_table_path = "/home/iscb/wolfson/annab4/DB/all_protein_table_modified.json"

# Load clustering and protein data
cluster_level_1 = pd.read_csv(cluster_level_1_path, sep='\t', header=None, names=['Cluster_1', 'Sequence_ID'])
cluster_level_2 = pd.read_csv(cluster_level_2_path, sep='\t', header=None, names=['Cluster_2', 'Centroid_ID'])
with open(protein_table_path) as f:
    protein_data = json.load(f)

# Map each sequence to its cluster_2 for fast lookup
cluster_2_mapping = cluster_level_2.set_index("Centroid_ID")["Cluster_2"].to_dict()
sequence_to_cluster_2 = cluster_level_1.assign(Cluster_2=cluster_level_1["Cluster_1"].map(cluster_2_mapping))

# Calculate most common first digit in the EC number for each Cluster_2
cluster_2_ec_numbers = defaultdict(lambda: defaultdict(int))
for cluster_2, group in sequence_to_cluster_2.groupby("Cluster_2"):
    ids = group["Sequence_ID"].values
    for seq_chain in ids:
        seq_id = seq_chain.split('_')[0]
        ec_number = protein_data.get(seq_id, {}).get("EC_number", "not found")
        if ec_number != "not found":
            first_digit = int(ec_number.split('.')[0])
            cluster_2_ec_numbers[cluster_2][first_digit] += 1

# Find the most common first digit for each Cluster_2
cluster_2_most_common_ec = {}
for cluster_2, ec_counts in cluster_2_ec_numbers.items():
    most_common_ec = max(ec_counts.items(), key=lambda x: x[1])[0]
    cluster_2_most_common_ec[cluster_2] = most_common_ec

print(cluster_2_most_common_ec.keys())

# Modify protein_data to replace EC numbers with "not found" if the first digit does not match
modified_protein_data = protein_data.copy()
for cluster_2, group in sequence_to_cluster_2.groupby("Cluster_2"):
    most_common_ec = cluster_2_most_common_ec.get(cluster_2, "not found")
    ids = group["Sequence_ID"].values
    for seq_chain in ids:
        seq_id = seq_chain.split('_')[0]
        ec_number = protein_data.get(seq_id, {}).get("EC_number", "not found")
        if ec_number != "not found":
            first_digit = int(ec_number.split('.')[0])
            if first_digit != most_common_ec:
                modified_protein_data[seq_id]["EC_number"] = "not found"

# Save the modified protein table to a new file
with open(output_protein_table_path, 'w') as f:
    json.dump(modified_protein_data, f, indent=4)

print(f"Modified protein table saved to {output_protein_table_path}")
