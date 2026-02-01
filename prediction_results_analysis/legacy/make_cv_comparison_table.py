import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

# ---------- Config ----------
# Adjust these two to match your run folders:
BASE_DIR = '/home/iscb/wolfson/annab4/scannet_experiments_outputs'
RUN_DIR_TEMPLATE = 'cv_fold{fold}_weigh_based_v9_noTransfer_v2'   # e.g. cv_fold1_weigh_based_v8_prop
FOLDS = (1, 2, 3, 4, 5)
SUBSETS = ('test', 'validation')  # rows per fold
OUT_CSV = '/home/iscb/wolfson/annab4/catalytic-sites-annotation/cross_validation/weigh_based_v9_noTransfer_v2_merged/cv_summary.csv'


# ---------- IO helpers ----------

def load_subset_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Expect the new training format
    for k in ('labels', 'predictions', 'weights', 'ids', 'splits'):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    labels = [np.asarray(x) for x in data['labels']]
    preds  = [np.asarray(x) for x in data['predictions']]
    weights = np.asarray(data['weights'], dtype=float)
    splits = list(data['splits'])
    return labels, preds, weights, splits


# ---------- Metrics ----------

def f_at_k(labels, preds, weights, k):
    """Weighted per-chain F@k."""
    num = 0.0
    den = 0.0
    for y, p, w in zip(labels, preds, weights):
        y = np.asarray(y)
        p = np.asarray(p)
        k_eff = min(k, len(p))
        if k_eff <= 0:
            continue
        top_idx = np.argpartition(p, -k_eff)[-k_eff:]
        denom = min(int(np.sum(y)), k_eff)
        f_i = 0.0 if denom == 0 else float(np.sum(y[top_idx])) / denom
        num += f_i * float(w)
        den += float(w)
    return 0.0 if den == 0.0 else num / den


def aucpr_weighted(labels, preds, weights):
    """Weighted PR-AUC: flatten and repeat per-chain weights across positions."""
    w_rep = [np.full(len(y), float(w), dtype=np.float32) for y, w in zip(labels, weights)]
    y_flat = np.concatenate(labels).astype(np.float32)
    p_flat = np.concatenate(preds).astype(np.float32)
    w_flat = np.concatenate(w_rep).astype(np.float32)

    bad = np.isnan(p_flat) | np.isnan(y_flat) | np.isinf(p_flat)
    if bad.any():
        if (~bad).any():
            p_flat[bad] = np.nanmedian(p_flat[~bad])
        else:
            p_flat[bad] = 0.0
        y_flat[bad] = 0.0
        w_flat[bad] = 0.0

    precision, recall, _ = precision_recall_curve(y_flat, p_flat, sample_weight=w_flat)
    return auc(recall, precision)


# ---------- Aggregation ----------

def compute_metrics_for_subset(labels, preds, weights, k_list=(1, 5)):
    """Return dict with F@k and AUCPR and their losses (1 - metric)."""
    metrics = {}
    for k in k_list:
        f = f_at_k(labels, preds, weights, k)
        metrics[f'F{k}'] = f
        metrics[f'F{k}_loss'] = 1.0 - f
    pr = aucpr_weighted(labels, preds, weights)
    metrics['AUCPR'] = pr
    metrics['AUCPR_loss'] = 1.0 - pr
    return metrics


def merged_over_folds(base_dir, run_tpl, subset, folds):
    """Concatenate labels/preds/weights across folds for the given subset."""
    all_labels, all_preds, all_weights = [], [], []
    for fold in folds:
        p = os.path.join(base_dir, run_tpl.format(fold=fold), f'{subset}_results.pkl')
        labels, preds, weights, _splits = load_subset_pkl(p)
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_weights.extend(list(weights))
    return all_labels, all_preds, np.asarray(all_weights, dtype=float)


def main():
    rows = []

    # --- merged rows (across all folds) ---
    for subset in SUBSETS:
        labels, preds, weights = merged_over_folds(BASE_DIR, RUN_DIR_TEMPLATE, subset, FOLDS)
        m = compute_metrics_for_subset(labels, preds, weights)
        rows.append({
            'Fold': 'merged',
            'Set': subset,
            'Split': '',
            'F1_loss': m['F1_loss'],
            'F5_loss': m['F5_loss'],
            'AUCPR_loss': m['AUCPR_loss'],
        })

    # --- per-fold rows ---
    for fold in FOLDS:
        run_dir = os.path.join(BASE_DIR, RUN_DIR_TEMPLATE.format(fold=fold))
        for subset in SUBSETS:
            p = os.path.join(run_dir, f'{subset}_results.pkl')
            labels, preds, weights, splits = load_subset_pkl(p)
            # identify the split name for this subset (should be unique)
            uniq = {}
            for s in splits:
                uniq[s] = uniq.get(s, 0) + 1
            # pick the most frequent; join if somehow multiple
            split_name = sorted(uniq.items(), key=lambda x: -x[1])[0][0] if uniq else ''
            m = compute_metrics_for_subset(labels, preds, weights)
            rows.append({
                'Fold': f'fold{fold}',
                'Set': subset,
                'Split': split_name,
                'F1_loss': m['F1_loss'],
                'F5_loss': m['F5_loss'],
                'AUCPR_loss': m['AUCPR_loss'],
            })

    df = pd.DataFrame(rows, columns=['Fold', 'Set', 'Split', 'F1_loss', 'F5_loss', 'AUCPR_loss'])
    df = df.round({'F1_loss': 3, 'F5_loss': 3, 'AUCPR_loss': 3})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
