import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


DIR_PATH = "/home/iscb/wolfson/annab4/scannet_experiments_outputs/newDB_transfer_msa_v1"

def calculate_F(file_path, max_k=20, save=False, save_path=None, weights=None):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    min_len = min([len(data['predictions']), len(data['labels']), len(weights)])
    predictions = np.array(data['predictions'], dtype=object)
    labels = np.array(data['labels'], dtype=object)
    weights_tmp = weights[:len(predictions)]
    print(len(predictions), len(labels), len(weights_tmp))
    F_k = np.zeros(max_k)
    for k in range(1, max_k + 1):
        top_k_true = []
        min_k_values = []
        for i in range(len(predictions)):
            try:
                true_labels = np.array(labels[i])
                predicted_scores = np.array(predictions[i])
                weight_i = weights_tmp[i]
            except Exception as e:
                print(f'index {i}, exception {type(e)}')
                break
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


def find_auc_roc(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    predictions = data['predictions']
    labels = data['labels'][:len(predictions)]
    weights = data['weights'][:len(predictions)] 
    weights_repeated = np.array([np.ones(len(label)) * weight for label, weight in zip(labels, weights)], dtype=object)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return roc_auc_score(labels, predictions, sample_weight=np.concatenate(weights_repeated))

    
    return parser.parse_args()

def make_PR_curve(
        labels,
        predictions,
        weights,
        title = '',
        subset_name='test',
        figsize=(10, 10),
        margin=0.05,
        grid=0.1,
        fs=25):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc

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
        print('Found %s nan predictions in subset %s'%(count_nan,subset_name) )
        predictions_flat[is_nan] = np.nanmedian(predictions_flat)

    precision, recall, _ = precision_recall_curve(
        labels_flat[~is_missing],
        predictions_flat[~is_missing],
        sample_weight=np.concatenate(weights_repeated)[~is_missing]
    )
    aucpr = auc(recall, precision)
    print(f'Title: {title}, AUCPR= {aucpr}')
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


#print(find_auc_roc('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results300/test_results_1200_with_weights.pkl'))

pickle_file = "/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results300/test_results_1200_with_weights.pkl"
with open(pickle_file, 'rb') as f:
    test_plot_data = pickle.load(f)
predictions = test_plot_data['predictions']
labels = test_plot_data['labels'][:len(predictions)]
weights = test_plot_data['weights']
    
fig, ax = make_PR_curve(labels, predictions, weights, title="homology 1200")
fig.savefig('/home/iscb/wolfson/annab4/catalytic-sites-annotation/structural_homology_baseline/results300/1200.png',dpi=300) 
