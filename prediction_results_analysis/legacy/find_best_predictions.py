import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

import warnings
warnings.filterwarnings("ignore")

def calculate_best_prediction_error(file_path, threshold, id_to_chain_id):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = np.array(data['labels'], dtype=object)
    predictions = np.array(data['predictions'], dtype=object)
    
    protein_errors = []
    for i in range(len(labels)):
        true_labels = np.array(labels[i])
        predicted_scores = np.array(predictions[i])
        predicted_binary = (predicted_scores >= threshold).astype(int)
        
        tp = np.sum((true_labels == 1) & (predicted_binary == 1))
        fp = np.sum((true_labels == 0) & (predicted_binary == 1))
        fn = np.sum((true_labels == 1) & (predicted_binary == 0))
        
        if tp == 0:
            error = 10000000000000
        else:
            error = fp * 1 + fn * 2
        
        protein_errors.append([error, id_to_chain_id[i]])
    
    return protein_errors

versions = [5]#list(range(1, 7))
datasets = ['test']#, 'validation']
with open(f"/home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/id_to_chain_id.pkl", 'rb') as f:
    id_to_chain_id = pickle.load(f)

for dataset in datasets:
    protein_error = []
    file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/transfer_msa_alpha2_1_gamma_0_v5/{dataset}_results.pkl'
    errors = calculate_best_prediction_error(file_path, 0.5, id_to_chain_id[dataset])
    errors.sort(key=lambda el : el[0])
    
    print(f"10 best proteins for {dataset} (averaged across versions):")
    for el in errors:
        print(el[0], el[1])
    #out = [f'"{el[1]}"' for el in errors[-2000:-800]]
    #print(" ".join(out))
        
