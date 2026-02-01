import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_F(file_path, max_k=20):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = np.array(data['labels'], dtype=object)
    predictions = np.array(data['predictions'], dtype=object)
    weights = np.array(data['weights'])
    
    F_k = np.zeros(max_k)
    for k in range(1, max_k + 1):  # k теперь начинается с 1
        top_k_true = []
        min_k_values = []
        for i in range(len(labels)):
            true_labels = labels[i]
            predicted_scores = predictions[i]
            weight_i = weights[i]
            top_k_indices = np.argsort(predicted_scores)[-k:]
            top_k_true.append(np.sum(true_labels[top_k_indices]) * weight_i)
            min_k_values.append(min(np.sum(true_labels), k) * weight_i)
        F_k[k - 1] = np.sum(top_k_true) / np.sum(min_k_values)
    
    # Добавляем нулевое значение в начало
    F_k = np.insert(F_k, 0, 0)
    return F_k

def plot_whisker_multiplot(alphas, gammas, max_k=20):
    fig, axes = plt.subplots(len(gammas), len(alphas), figsize=(len(alphas) * 3.5, len(gammas) * 2.5), sharex=True, sharey=True)
    
    versions = list(range(1, 7))
    datasets = ['test_results', 'validation_results']
    
    for row, g in enumerate(gammas):
        for col, a in enumerate(alphas):
            model_name = f"transfer_msa_alpha2_{a}_gamma_{g}"
            F_all_versions = []
            
            for v in versions:
                file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/{model_name}_v{v}/validation_results.pkl'
                if os.path.exists(file_path):
                    F_all_versions.append(calculate_F(file_path, max_k=max_k))
            
            if F_all_versions:
                F_all_versions = np.array(F_all_versions)
                F_mean = np.mean(F_all_versions, axis=0)
                F_std = np.std(F_all_versions, axis=0)
                
                ax = axes[row, col]
                ax.errorbar(range(0, max_k + 1), F_mean, yerr=F_std, fmt='o', capsize=3, markersize=1, elinewidth=0.5, capthick=0.5)
                ax.set_title(f'α={a}, γ={g}', fontsize=10)
                ax.set_xticks(range(0, max_k + 1, max(1, max_k // 10)))
                ax.set_xticklabels(range(0, max_k + 1, max(1, max_k // 10)), fontsize=5)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, max_k + 1)
                ax.grid(True, linewidth=0.25)
    
    fig.suptitle("F(k) with Whisker Plots", fontsize=14)
    fig.supxlabel("k", fontsize=5)
    fig.supylabel("F(k)", fontsize=5)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig("plot.pdf", dpi=500, bbox_inches="tight")


def plot_heatmap(alphas, gammas, k=5):
    fig, ax = plt.subplots()
    
    versions = list(range(1, 7))
    datasets = ['test_results', 'validation_results']
    
    a_g_map = np.zeros((len(alphas), len(gammas)))
    
    for row, g in enumerate(gammas):
        for col, a in enumerate(alphas):
            model_name = f"transfer_msa_alpha2_{a}_gamma_{g}"
            F_all_versions = []
            
            for v in versions:
                file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/{model_name}_v{v}/validation_results.pkl'
                if os.path.exists(file_path):
                    F_all_versions.append(calculate_F(file_path, max_k=6))
            
            if F_all_versions:
                F_all_versions = np.array(F_all_versions)
                F_mean = np.mean(F_all_versions, axis=0)
                F_std = np.std(F_all_versions, axis=0)
                
                a_g_map[col][row] = F_mean[k]
    
    fig.suptitle("F(k) heatmap", fontsize=14)
    ax = sns.heatmap(a_g_map)
    plt.savefig("heatmap.pdf", dpi=500, bbox_inches="tight")

def std_heatmap(alphas, gammas, k=5):
    fig, ax = plt.subplots()
    
    versions = list(range(1, 7))
    var_map = np.zeros((len(gammas), len(alphas)))
    
    for row, gamma in enumerate(gammas):
        for col, alpha in enumerate(alphas):
            model_name = f"transfer_msa_alpha2_{alpha}_gamma_{gamma}"
            F_all_versions = []
            
            for v in versions:
                file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/{model_name}_v{v}/validation_results.pkl'
                if os.path.exists(file_path):
                    F_all_versions.append(calculate_F(file_path, max_k=k))
            
            if F_all_versions:
                F_all_versions = np.array(F_all_versions)
                F_var = np.std(F_all_versions, axis=0)  
                var_map[row, col] = F_var[k - 1]  
    
    fig.suptitle(f"Standart deviation of F({k}) for different alpha and gamma combinations", fontsize=14)
    ax = sns.heatmap(var_map, annot=True, xticklabels=alphas, yticklabels=gammas, cmap="coolwarm", fmt=".2f")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Gamma")
    plt.savefig(f"alpha_gamma_std_heatmap_k{k}.png", dpi=500, bbox_inches="tight")


alphas = [1, 5, 10, 50, 100, 150]
gammas = [0, 1, 2]
#plot_whisker_multiplot(alphas, gammas, max_k=20)
#plot_heatmap(alphas, gammas, k=5)
std_heatmap(alphas, gammas, 5)