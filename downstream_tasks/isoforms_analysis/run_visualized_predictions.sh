#!/bin/bash
set -u  # падать на неинициализированных переменных
# set -e  # (опционально) падать на первой ошибке; если хочешь прогнать все белки — НЕ ставь
set -o pipefail

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
export KERAS_BACKEND=jax
conda activate scannet_keras3

python_script="/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/predict_bindingsites.py"

# Папка, где лежат файлы-группы (по одному на base_id):
# O14492_isoforms.txt, O00555_isoforms.txt, ...
groups_dir="${1:-/home/iscb/wolfson/annab4/outputs/isoform_groups_todo}"

logdir="${2:-./logs_predict}"
mkdir -p "$logdir"

echo "=== Launch ==="
mode="cv_catalytic"
echo "Mode: $mode"
echo "Groups dir: $groups_dir"
echo "Log dir: $logdir"
echo

shopt -s nullglob

n_total=0
n_ok=0
n_fail=0

for path_to_proteins in "$groups_dir"/*_isoforms.txt; do
  n_total=$((n_total + 1))

  base_id="$(basename "$path_to_proteins" _isoforms.txt)"
  batch_cxc="${base_id}_isoforms"
  logfile="$logdir/${batch_cxc}.log"

  echo ">>> [$n_total] base_id=$base_id"
  echo "    file=$path_to_proteins"
  echo "    batch_cxc=$batch_cxc"
  echo "    log=$logfile"

  # Запуск + логирование stdout/stderr в файл
  #if python "$python_script" "$path_to_proteins" --assembly --mode "$mode" --batch_cxc "$batch_cxc" \
  if python "$python_script" "$path_to_proteins" --mode "$mode" --batch_cxc "$batch_cxc" --use_ESM2_baseline --esm2_dir /home/iscb/wolfson/annab4/Data/Human_Isoforms_ESM2_150M_layer30 --predictions_folder /home/iscb/wolfson/annab4/isoforms_predictions_baseline_ESM2/ \
      > >(tee -a "$logfile") 2> >(tee -a "$logfile" >&2); then
    n_ok=$((n_ok + 1))
    echo "    status=OK"
  else
    n_fail=$((n_fail + 1))
    echo "    status=FAIL (see $logfile)" >&2
  fi

  echo
done

echo "=== Finished ==="
echo "Total: $n_total  OK: $n_ok  FAIL: $n_fail"
