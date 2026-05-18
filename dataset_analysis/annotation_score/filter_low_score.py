import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Filter protein table by annotation score > 3 and plot score histogram."
    )
    ap.add_argument("--protein_table", required=True,
                    help="Path to protein_table JSON file")
    ap.add_argument("--uniprot_data", required=True,
                    help="Path to UniProt active-sites JSON file (with 'results' list)")
    ap.add_argument("--filtered_json", default="filtered_protein_table.json",
                    help="Output path for filtered protein table JSON (default: %(default)s)")
    ap.add_argument("--filtered_fasta", default="filtered_protein_sequences.fasta",
                    help="Output path for filtered sequences FASTA (default: %(default)s)")
    ap.add_argument("--score_hist", default="score_hist.png",
                    help="Output path for annotation score histogram image (default: %(default)s)")
    args = ap.parse_args()

    with open(args.protein_table, "r") as f:
        protein_table = json.load(f)

    with open(args.uniprot_data, "r") as f:
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
    plt.savefig(args.score_hist, dpi=500, bbox_inches="tight")

    uniprot_annotation_scores = {
        entry["primaryAccession"]: entry["annotationScore"] for entry in uniprot_data["results"]
    }

    filtered_proteins = {
        uniprot_id: data
        for uniprot_id, data in protein_table.items()
        if uniprot_annotation_scores.get(uniprot_id, 0) > 3
    }

    with open(args.filtered_json, "w") as f:
        json.dump(filtered_proteins, f, indent=4)

    with open(args.filtered_fasta, "w") as f:
        for uniprot_id, data in filtered_proteins.items():
            f.write(f">{uniprot_id}\n{data['uniprot_sequence']}\n")


if __name__ == "__main__":
    main()
