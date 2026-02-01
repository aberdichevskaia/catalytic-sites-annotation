import json
import os
import numpy as np
import pickle
from tqdm import tqdm

# Splits and directories
SPLITS = ["test"]
ANNOTATION_DIR = "/home/iscb/wolfson/annab4/DB/splitted_by_EC_number"

OUTPUT_DIR = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/random_results"

def parse_annotation(file_path):
    annotations = {}
    with open(file_path, "r") as f:
        current_id, current_chain = None, None
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split("_")
                current_id = parts[0]
                current_chain = parts[1]
                annotations[(current_id, current_chain)] = []
            else:
                cols = line.strip().split()
                if len(cols) == 4:
                    res_id, label = cols[2], cols[3]
                    annotations[(current_id, current_chain)].append((res_id, label))
    return annotations


def process_split(annotations):
    labels = []
    predictions = []
    for key, val in tqdm(annotations.items()):
        true_labels = np.array([int(label) for res, label in val])
        labels.append(true_labels)
        prediction = np.random.choice([0, 1], size=true_labels.shape, p=[0.995487, 0.004513]) # training set probabilities
        predictions.append(prediction)
    return labels, predictions

    
for split in SPLITS:
    annotation_file = os.path.join(ANNOTATION_DIR, f"{split}.txt")
    annotations = parse_annotation(annotation_file)
    labels, predictions = process_split(annotations)
    plot_data = {
        "labels": labels,
        "predictions": predictions
    }

    pkl_path = os.path.join(OUTPUT_DIR, f'{split}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(plot_data, f)
        

    