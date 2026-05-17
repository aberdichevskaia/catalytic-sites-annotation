import argparse
import json
import pandas as pd
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(
        description="Filter EC numbers by per-cluster majority vote and write a cleaned protein table."
    )
    ap.add_argument("--cluster_level_1", required=True,
                    help="Path to cluster_level_1_cluster.tsv (see config.example.yaml: cluster_level_1)")
    ap.add_argument("--cluster_level_2", required=True,
                    help="Path to cluster_level_2_cluster.tsv (see config.example.yaml: cluster_level_2)")
    ap.add_argument("--protein_table", required=True,
                    help="Input protein table JSON (see config.example.yaml: protein_table)")
    ap.add_argument("--output_table", required=True,
                    help="Output path for EC-filtered table (see config.example.yaml: protein_table_modified)")
    args = ap.parse_args()

    cluster_level_1 = pd.read_csv(args.cluster_level_1, sep="\t", header=None,
                                  names=["Cluster_1", "Sequence_ID"])
    cluster_level_2 = pd.read_csv(args.cluster_level_2, sep="\t", header=None,
                                  names=["Cluster_2", "Centroid_ID"])
    with open(args.protein_table) as f:
        protein_data = json.load(f)

    cluster_2_mapping = cluster_level_2.set_index("Centroid_ID")["Cluster_2"].to_dict()
    sequence_to_cluster_2 = cluster_level_1.assign(
        Cluster_2=cluster_level_1["Cluster_1"].map(cluster_2_mapping)
    )

    # Count EC first digits per Cluster_2
    cluster_2_ec_numbers = defaultdict(lambda: defaultdict(int))
    for cluster_2, group in sequence_to_cluster_2.groupby("Cluster_2"):
        for seq_chain in group["Sequence_ID"].values:
            seq_id = seq_chain.split("_")[0]
            ec_number = protein_data.get(seq_id, {}).get("EC_number", "not found")
            if ec_number != "not found":
                first_digit = int(ec_number.split(".")[0])
                cluster_2_ec_numbers[cluster_2][first_digit] += 1

    # Most common EC first digit per Cluster_2
    cluster_2_most_common_ec = {
        c2: max(counts.items(), key=lambda x: x[1])[0]
        for c2, counts in cluster_2_ec_numbers.items()
    }

    print(cluster_2_most_common_ec.keys())

    # Zero out EC numbers that disagree with the cluster majority
    modified_protein_data = protein_data.copy()
    for cluster_2, group in sequence_to_cluster_2.groupby("Cluster_2"):
        most_common_ec = cluster_2_most_common_ec.get(cluster_2, "not found")
        for seq_chain in group["Sequence_ID"].values:
            seq_id = seq_chain.split("_")[0]
            ec_number = protein_data.get(seq_id, {}).get("EC_number", "not found")
            if ec_number != "not found":
                first_digit = int(ec_number.split(".")[0])
                if first_digit != most_common_ec:
                    modified_protein_data[seq_id]["EC_number"] = "not found"

    with open(args.output_table, "w") as f:
        json.dump(modified_protein_data, f, indent=4)

    print(f"Modified protein table saved to {args.output_table}")


if __name__ == "__main__":
    main()
