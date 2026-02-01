import json
import numpy as np

uniprot_data_path = "/home/iscb/wolfson/annab4/uniprot_files/all_proteins.json"

filtered_json_path = "/home/iscb/wolfson/annab4/uniprot_files/filtered_all_protein.json"
print(1)
with open(uniprot_data_path, "r") as f:
    uniprot_data = json.load(f)

print(2)
uniprot_filtered = [
    entry
    for entry in uniprot_data["results"]
    if entry["annotationScore"] > 2 and
       any(feature.get("type") == "Active site" for feature in entry.get("features", []))
]
print(f"Total size: {len(uniprot_filtered)}")

with open(filtered_json_path, "w") as f:
    json.dump(uniprot_filtered, f, indent=4)
print("saved")