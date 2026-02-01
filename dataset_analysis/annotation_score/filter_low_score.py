import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

protein_table_path = "/home/iscb/wolfson/annab4/DB/protein_table_modified.json"
uniprot_data_path = "/home/iscb/wolfson/annab4/uniprot_files/active_sites_proteins.json"

filtered_json_path = "filtered_protein_table.json"
filtered_fasta_path = "filtered_protein_sequences.fasta"

with open(protein_table_path, "r") as f:
    protein_table = json.load(f)

with open(uniprot_data_path, "r") as f:
    uniprot_data = json.load(f)

annotation_scores = [entry["annotationScore"] for entry in uniprot_data["results"]]
unique_scores, counts = np.unique(annotation_scores, return_counts=True)

plt.figure(figsize=(8, 5))
plt.bar(unique_scores, counts, width=0.6, edgecolor="black", alpha=0.7)

for score, count in zip(unique_scores, counts):
    plt.text(score, count + 300, str(count), ha="center", fontsize=11)

plt.xlabel("Annotation Score")
plt.ylabel("Frequency")
plt.title("Histogram of Annotation Scores")
plt.xticks(range(1, 6))  
plt.ylim(0, 8000) 
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("score_hist.png", dpi=500, bbox_inches="tight")


uniprot_annotation_scores = {
    entry["primaryAccession"]: entry["annotationScore"] for entry in uniprot_data["results"]
}

filtered_proteins = {
    uniprot_id: data
    for uniprot_id, data in protein_table.items()
    if uniprot_annotation_scores.get(uniprot_id, 0) > 3
}

with open(filtered_json_path, "w") as f:
    json.dump(filtered_proteins, f, indent=4)

with open(filtered_fasta_path, "w") as f:
    for uniprot_id, data in filtered_proteins.items():
        f.write(f">{uniprot_id}\n{data['uniprot_sequence']}\n")
