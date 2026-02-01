#!/usr/bin/env python3
import os, re, sys, argparse, pickle, csv
import numpy as np
from sklearn.metrics import average_precision_score

def parse_split_from_dataset_name(dsname: str):
    m = re.search(r"split\d+", dsname or "")
    return m.group(0) if m else None

def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def sanitize_vectors(y_true_raw, y_score_raw):
    y_true = np.asarray(y_true_raw)
    y_score = np.asarray(y_score_raw, dtype=float)
    valid = (y_true >= 0)
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]
    if y_score.size:
        finite = np.isfinite(y_score)
        if not np.all(finite):
            fill = np.nanmedian(y_score[finite]) if finite.any() else 0.0
            y_score[~finite] = fill
    return y_true, y_score

def per_sample_metrics(y_true_raw, y_score_raw, k=5, fp_th=0.3):
    """
    Возвращает:
      AUPRC,
      F@k (TP@k/min(#pos,k)),
      FP@k (все отрицательные в топ-k),
      CFP@k (только FP с score>=fp_th),
      max_FP_score@k (0 если FP нет),
      topk_idx (индексы по валидированному вектору)
    """
    y_true, y_score = sanitize_vectors(y_true_raw, y_score_raw)
    if y_true.size == 0:
        return 0.0, 0.0, 0, 0, 0.0, np.array([], dtype=int)

    auprc = 0.0 if y_true.sum() == 0 else float(average_precision_score(y_true, y_score))
    kk = min(k, y_score.size)
    topk = np.argsort(y_score)[-kk:]
    topk_scores = y_score[topk]
    topk_labels = y_true[topk]

    tp = int((topk_labels == 1).sum())
    fp_mask = (topk_labels == 0)
    fp = int(fp_mask.sum())

    # confident FP: отрицательные в топ-k с score >= fp_th
    cfp_mask = fp_mask & (topk_scores >= fp_th)
    cfp = int(cfp_mask.sum())

    denom = min(int(y_true.sum()), kk)
    f_at_k = (tp / denom) if denom > 0 else 0.0
    max_fp_score = float(topk_scores[fp_mask].max()) if fp > 0 else 0.0
    return auprc, f_at_k, fp, cfp, max_fp_score, topk

def analyze_file(results_path, k=5, topn=10, fp_th=0.3, csv_out=None, dump_conf_fp_positions=0):
    d = load_results(results_path)
    preds = d["predictions"]
    labels = d["labels"]
    pids = d.get("protein_ids")  # предпочтительно использовать уже сохранённые ID
    ds_names = d.get("dataset_names")  # не обязателен, если есть pids

    if pids is None:
        raise RuntimeError("В результате нет 'protein_ids'. Для уверенных FP лучше мерджить с protein_ids.")

    n = min(len(preds), len(labels), len(pids))
    preds, labels, pids = preds[:n], labels[:n], pids[:n]

    rows = []
    conf_fp_positions = []  # список (protein_id, resid_idx, score) только для score>=fp_th и FP в топ-k
    for i in range(n):
        auprc, f_at_k, fp, cfp, max_fp_score, topk = per_sample_metrics(labels[i], preds[i], k=k, fp_th=fp_th)
        rows.append({
            "idx": i,
            "protein_id": pids[i],
            "AUPRC": auprc,
            f"F@{k}": f_at_k,
            f"FP@{k}": fp,
            f"CFP@{k}(≥{fp_th})": cfp,
            "max_FP_score@k": max_fp_score
        })
        if dump_conf_fp_positions:
            y_true, y_score = sanitize_vectors(labels[i], preds[i])
            for j in topk:
                if j < y_true.size and y_true[j] == 0 and y_score[j] >= fp_th:
                    conf_fp_positions.append((pids[i], int(j), float(y_score[j])))

    title = os.path.basename(results_path)
    print(f"\n=== {title} ===  samples={n}  (τ={fp_th})")

    # Ранжируем "самые уверенные FP" по двум критериям:
    worst_by_cfp = sorted(rows, key=lambda r: r[f"CFP@{k}(≥{fp_th})"], reverse=True)[:topn]
    worst_by_maxfp = sorted(rows, key=lambda r: r["max_FP_score@k"], reverse=True)[:topn]

    print(f"\nTop-{topn} by CONFIDENT FP@{k} (score ≥ {fp_th}):")
    for r in worst_by_cfp:
        print(f"  {r['protein_id']:20s} | idx={r['idx']:4d} | CFP@{k}={r[f'CFP@{k}(≥{fp_th})']:2d} | "
              f"FP@{k}={r[f'FP@{k}']:2d} | maxFP={r['max_FP_score@k']:.3f} | AUPRC={r['AUPRC']:.4f} | F@{k}={r[f'F@{k}']:.4f}")

    print(f"\nTop-{topn} by max FP score in top-{k}:")
    for r in worst_by_maxfp:
        print(f"  {r['protein_id']:20s} | idx={r['idx']:4d} | maxFP={r['max_FP_score@k']:.3f} | "
              f"CFP@{k}={r[f'CFP@{k}(≥{fp_th})']:2d} | FP@{k}={r[f'FP@{k}']:2d} | AUPRC={r['AUPRC']:.4f} | F@{k}={r[f'F@{k}']:.4f}")

    if csv_out:
        fields = ["idx", "protein_id", "AUPRC", f"F@{k}", f"FP@{k}", f"CFP@{k}(≥{fp_th})", "max_FP_score@k"]
        with open(csv_out, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved metrics to {csv_out}")

    if dump_conf_fp_positions:
        conf_fp_positions.sort(key=lambda x: x[2], reverse=True)
        m = min(dump_conf_fp_positions, len(conf_fp_positions))
        print(f"\nTop-{m} confident FP residues (score ≥ {fp_th}) across file:")
        for pid, resid, score in conf_fp_positions[:m]:
            print(f"  {pid:20s} | resid={resid:4d} | score={score:.6f}")

def main():
    ap = argparse.ArgumentParser(description="Confident FP analysis for merged results.")
    ap.add_argument("--results-path", required=True, help="File or dir containing test_results.pkl / validation_results.pkl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--fp-th", type=float, default=0.3, help="confidence threshold τ for CFP@k")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--dump-conf-fp-positions", type=int, default=0)
    args = ap.parse_args()

    path = args.results_path
    if os.path.isdir(path):
        for fname in ["test_results.pkl", "validation_results.pkl"]:
            f = os.path.join(path, fname)
            if os.path.exists(f):
                out_csv = None
                if args.csv_out:
                    base = os.path.splitext(args.csv_out)[0]
                    out_csv = f"{base}_{os.path.splitext(fname)[0]}_tau{args.fp_th}.csv"
                analyze_file(f, k=args.k, topn=args.topn, fp_th=args.fp_th,
                             csv_out=out_csv, dump_conf_fp_positions=args.dump_conf_fp_positions)
            else:
                print(f"[INFO] not found: {f}")
    else:
        analyze_file(path, k=args.k, topn=args.topn, fp_th=args.fp_th,
                     csv_out=args.csv_out, dump_conf_fp_positions=args.dump_conf_fp_positions)

if __name__ == "__main__":
    main()
