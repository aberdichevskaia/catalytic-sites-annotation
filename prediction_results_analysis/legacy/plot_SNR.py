import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_F(file_path, max_k=20):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = np.array(data['predictions'], dtype=object)
    labels = np.array(data['labels'][:len(predictions)], dtype=object)
    weights = np.array(data['weights'][:len(predictions)])
    F_k = np.zeros(max_k)
    for k in range(1, max_k + 1):
        top_k_true = []
        min_k_values = []
        f_values = []
        for i in range(len(predictions)):
            true_labels = np.array(labels[i])
            predicted_scores = np.array(predictions[i])
            weight_i = weights[i]
            top_k_indices = np.argsort(predicted_scores)[-k:]
            denom = min(np.sum(true_labels), k)
            if denom == 0:
                f_i = 0.0
            else:
                f_i = np.sum(true_labels[top_k_indices]) / denom
            f_values.append(f_i)
        f_values = np.array(f_values)
        weights = np.array(weights)
        F_k[k - 1] = np.sum(f_values * weights) / np.sum(weights)
    return F_k

def compute_SNR(F):
    """Compute Signal-to-Noise Ratio (SNR) based on F tensor."""
    SNR_k = F.mean(-1).std((1,2)) / F.std(-1).mean((1,2))
    return SNR_k

def plot_SNR(SNR_k, max_k=20):
    """Plot SNR(k)."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), SNR_k, marker='o')
    plt.xlabel("k")
    plt.ylabel("SNR(k)")
    plt.title("Signal-to-Noise Ratio (SNR) as a function of k")
    plt.grid(True)
    plt.savefig("/home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/SNR.pdf", dpi=500, bbox_inches="tight")

def plot_SNR_heatmap(F, k=5, alphas=[1, 5, 10, 50, 100, 150], gammas=[0, 1, 2]):
    """Plot SNR as a heatmap for a specific k value."""
    SNR_k = compute_SNR(F)[k - 1]  # Selecting SNR at k=5
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(SNR_k, annot=True, xticklabels=alphas, yticklabels=gammas, cmap="coolwarm", fmt=".2f")
    plt.xlabel("Alpha")
    plt.ylabel("Gamma")
    plt.title(f"SNR Heatmap at k={k}")
    plt.savefig("/home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/SNR_heatmap_k5.pdf", dpi=500, bbox_inches="tight")
    #plt.show()

def main(alphas, gammas, max_k=20):
    versions = list(range(1, 7))
    F_tensor = np.zeros((max_k, len(gammas), len(alphas), len(versions)))
    
    for i, g in enumerate(gammas):
        for j, a in enumerate(alphas):
            for v in versions:
                #model_name = f"transfer_msa_alpha2_{a}_gamma_{g}"
                model_name = f"transfer_msa_batch_size_{a}_lr_{g}"
                print(model_name)
                file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/{model_name}_v{v}/validation_results.pkl'
                if os.path.exists(file_path):
                    F_tensor[:, i, j, v-1] = calculate_F(file_path, max_k=max_k)
    
    SNR_k = compute_SNR(F_tensor)
    plot_SNR(SNR_k, max_k)
    #plot_SNR_heatmap(F_tensor, k=5, alphas=alphas, gammas=gammas)

#alphas = [1, 5, 10, 50, 100, 150]
#gammas = [0, 1, 2]

batch_size = [1, 2, 4, 8, 16]
learning_rate = [0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.002]

#main(alphas, gammas, max_k=20)
main(batch_size, learning_rate)
