import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


DIR_PATH = "/home/iscb/wolfson/annab4/scannet_experiments_outputs/newDB_transfer_msa_v1"

def calculate_macro_F(file_path, max_k=20, save=False, save_path=None):
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
        
    if save:
        plt.figure()
        x_values = np.arange(1, max_k + 1)
        plt.plot(x_values, F_k, marker='o', linestyle='-')
        plt.xlabel("k")
        plt.ylabel("F(k)")
        plt.title("F(k) Curve")
        if save_path is None:
            save_path = "F_plot.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    #print(file_path, F_k)
    return F_k


def calculate_F(file_path, max_k=20, save=False, save_path=None):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = np.array(data['predictions'], dtype=object)
    labels = np.array(data['labels'], dtype=object)
    #print(labels.shape)
    #print(predictions.shape)
    if 'weights' in data:
        weights = np.array(data['weights'])
    else:
        weights = None
    F_k = np.zeros(max_k)
    for k in range(1, max_k + 1):
        top_k_true = []
        min_k_values = []
        for i in range(len(predictions)):
            true_labels = np.array(labels[i])
            #print(len(true_labels))
            predicted_scores = np.array(predictions[i])
            #print(len(predicted_scores))
            weight_i = weights[i] if type(weights)!=type(None) else 1.0
            top_k_indices = np.argsort(predicted_scores)[-k:]
            top_k_true.append(np.sum(true_labels[top_k_indices]) * weight_i)
            min_k_values.append(min(np.sum(true_labels), k) * weight_i)
        F_k[k - 1] = np.sum(top_k_true) / np.sum(min_k_values)
        
    if save:
        plt.figure()
        x_values = np.arange(1, max_k + 1)
        plt.plot(x_values, F_k, marker='o', linestyle='-')
        plt.xlabel("k")
        plt.ylabel("F(k)")
        plt.title("F(k) Curve")
        if save_path is None:
            save_path = "F_plot.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    #print(file_path, F_k)
    return F_k

def get_aucpr(file_path, save=False):    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = np.array(data['predictions'], dtype=object)
    labels = np.array(data['labels'][:len(predictions)], dtype=object)
    weights = np.array(data['weights'][:len(predictions)])
    weights_repeated = np.array([np.ones(len(label)) * weight for label, weight in zip(labels, weights)], dtype=object)
    labels_flat=np.concatenate(labels)
    predictions_flat=np.concatenate(predictions)

    if predictions_flat.ndim > 1:
        print('Multi-class! Plotting last class versus all others')
        predictions_flat = predictions_flat[...,-1]
        labels_flat = (labels_flat == labels_flat.max() )

    is_nan = np.isnan(predictions_flat) | np.isinf(labels_flat)
    is_missing = np.isnan(labels_flat) | (labels_flat<0)
    count_nan = is_nan.sum()
    if count_nan>0:
        print('Found %s nan predictions in subset %s'%(count_nan) )
        predictions_flat[is_nan] = np.nanmedian(predictions_flat)

    precision, recall, _ = precision_recall_curve(
        labels_flat[~is_missing],
        predictions_flat[~is_missing],
        sample_weight=np.concatenate(weights_repeated)[~is_missing]
    )
    aucpr = auc(recall, precision)
    return aucpr
    #print(f'AUCPR= {aucpr}')
    
    if save:
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='C0',linewidth=2.0,
                    label=f'{subset_name} (AUCPR= {aucpr:.3f})')
        plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.xlim([0 - margin, 1 + margin])
        plt.ylim([0 - margin, 1 + margin])
        plt.grid()

        plt.legend(fontsize=fs)
        plt.xlabel('Recall', fontsize=fs)
        plt.ylabel('Precision', fontsize=fs)
        plt.title(title,fontsize=fs)
        plt.tight_layout()
        return fig, ax


def calculate_mean_Fk(file_path_template, max_k=20):
    datasets = ['test_results', 'validation_results']
    
    for dataset in datasets:
        print(dataset)
        F_all_versions = []
        aucpr_all_versions = []
        for v in range(1, 7):
            file_path = f'{file_path_template}_v{v}/{dataset}.pkl'
            if os.path.exists(file_path):
                F_all_versions.append(calculate_macro_F(file_path, max_k=max_k))
                aucpr_all_versions.append(get_aucpr(file_path))
                
                #print(F_all_versions[-1][0], F_all_versions[-1][4])
        
        if F_all_versions:
            F_all_versions = np.array(F_all_versions)
            F_mean = np.mean(F_all_versions, axis=0)
            F_std = np.std(F_all_versions, axis=0)
            
            aucpr_all_versions = np.array(aucpr_all_versions)
            aucpr_mean = np.mean(aucpr_all_versions)
            aucpr_std = np.std(aucpr_all_versions)
            print(f'F@0, {F_mean[0]}(±{F_std[0]})')
            print(f'F@5: {F_mean[4]}(±{F_std[4]})')
            print(f'AUCPR: {aucpr_mean}(±{aucpr_std})')
            print()
        else:
            print('no F_mean')
            print()
    print()
    

def plot_whisker_multiplot(alphas, gammas, save_path, max_k=20):
    fig, axes = plt.subplots(len(gammas), len(alphas), figsize=(len(alphas) * 3.5, len(gammas) * 2.5), sharex=True, sharey=True)
    
    versions = list(range(1, 3))
    datasets = ['test_results', 'validation_results']
    
    for row, g in enumerate(gammas):
        for col, a in enumerate(alphas):
            model_name = f"transfer_msa_alpha2_{a}_gamma_{g}"
            F_all_versions = []
            
            for v in versions:
                file_path = f'/home/iscb/wolfson/annab4/scannet_experiments_outputs/{model_name}_v{v}/test_results.pkl'
                if os.path.exists(file_path):
                    F_all_versions.append(calculate_F(file_path, max_k=max_k))
            
            if F_all_versions:
                F_all_versions = np.array(F_all_versions)
                F_mean = np.mean(F_all_versions, axis=0)
                F_std = np.std(F_all_versions, axis=0)
                print(a, g)
                print(F_mean)
                print()
                
                ax = axes[row, col]
                ax.errorbar(range(0, max_k), F_mean, yerr=F_std, fmt='o', capsize=3, markersize=1, elinewidth=0.5, capthick=0.5)
                ax.set_title(f'α={a}, γ={g}', fontsize=10)
                ax.set_xticks(range(0, max_k, max(1, max_k // 10)))
                ax.set_xticklabels(range(0, max_k, max(1, max_k // 10)), fontsize=5)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, max_k)
                ax.grid(True, linewidth=0.25)
    
    fig.suptitle("F(k) with Whisker Plots", fontsize=14)
    fig.supxlabel("k", fontsize=5)
    fig.supylabel("F(k)", fontsize=5)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig(save_path, dpi=500, bbox_inches="tight")

alphas = [1, 5, 10, 50, 100, 150]
gammas = [0, 1, 2]

#print('random')
#print(calculate_F('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/random_results/test_results.pkl'))

# print('homology cuttof 300')
# print(calculate_macro_F('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results_adaptive_cutoff_1/test_results.pkl'))
# print()
# print('homology cuttof/2')
# print(calculate_macro_F('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results_adaptive_cutoff_2/test_results.pkl'))
# print()
# print('homology cuttof/3')
# print(calculate_macro_F('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results_adaptive_cutoff_3/test_results.pkl'))
# print()


#print('handcrafted')
#print(calculate_macro_F('/home/iscb/wolfson/annab4/scannet_experiments_outputs/handcrafted_features_features_psahcEC_numbers/test_results.pkl'))


print('new DB')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/rebalanced_transfer_msa_v', max_k=6)

print('propogation 0.2')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/propagated_02_v4', max_k=6)
print()

print('propogation 0.3')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/propagated_03_v4', max_k=6)
print()

print('propogation 0.5')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/propagated_05_v4', max_k=6)
print()

print('propogation 0.6')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/propagated_06_v4', max_k=6)

print('propogation 0.7')
calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/propagated_07_v4', max_k=6)



#print('new DB structure weights')
#calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/newDB_retrain_transfer_msa', max_k=6)

# print()
# print('transfer_msa')
# calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/newDB_transfer_msa', max_k=6)

# print()
# print('no TL')
# calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/msa_alpha2_1_gamma_0', max_k=6)

# print()
# print('no MSA')
# calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/transfer_alpha2_1_gamma_0', max_k=6)


# print()
# print('no anything')
# calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/alpha2_1_gamma_0', max_k=6)


#calculate_mean_Fk('/home/iscb/wolfson/annab4/scannet_experiments_outputs/newDB_transfer_msa', max_k=6)

#plot_whisker_multiplot(alphas, gammas, save_path='/home/iscb/wolfson/annab4/catalytic-sites-annotation/play_with_plot_data/plots/multiplot.pdf', max_k=6)
