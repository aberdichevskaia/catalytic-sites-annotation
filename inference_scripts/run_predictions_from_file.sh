#!/bin/bash
set -u  # падать на неинициализированных переменных
# set -e  # (опционально) падать на первой ошибке; если хочешь прогнать все фолды — не ставь

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
export KERAS_BACKEND=jax
conda activate scannet_keras3

python_script="/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/predict_bindingsites.py"
path_to_proteins="/home/iscb/wolfson/annab4/catalytic-sites-annotation/inference_scripts/to_predict.txt"

logdir="./logs_predict"
mkdir -p "$logdir"

echo "=== Launch ==="
mode="cv_catalytic"

python "$python_script" "$path_to_proteins" --assembly --mode "$mode" 
echo "=== Finished ==="
