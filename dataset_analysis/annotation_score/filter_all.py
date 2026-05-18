import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Filter UniProt JSON to entries with annotationScore > 2 and an Active site feature."
    )
    ap.add_argument("--uniprot_data", required=True,
                    help="Path to input UniProt JSON file (with 'results' list)")
    ap.add_argument("--filtered_json", required=True,
                    help="Output path for filtered UniProt JSON file")
    args = ap.parse_args()

    with open(args.uniprot_data, "r") as f:
        uniprot_data = json.load(f)

    print(2)
    uniprot_filtered = [
        entry
        for entry in uniprot_data["results"]
        if entry["annotationScore"] > 2 and
           any(feature.get("type") == "Active site" for feature in entry.get("features", []))
    ]
    print(f"Total size: {len(uniprot_filtered)}")

    with open(args.filtered_json, "w") as f:
        json.dump(uniprot_filtered, f, indent=4)
    print("saved")


if __name__ == "__main__":
    main()
