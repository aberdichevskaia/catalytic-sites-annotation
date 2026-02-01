import os
import re

folder_path = '/home/iscb/wolfson/annab4/slurm_outputs'

patterns = [
    re.compile(r'Epoch \d+: early stopping'),
    re.compile(r'Training completed! Saving model transfer_msa_alpha2_.*')
]

def extract_matching_lines(file_path, patterns):
    matching_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if any(pattern.search(line) for pattern in patterns):
                matching_lines.append(line.strip())
    return matching_lines

for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.out'):
            file_path = os.path.join(root, file)
            matches = extract_matching_lines(file_path, patterns)
            if matches:
                print(f"\nFile: {file_path}")
                for match in matches:
                    print(match)
